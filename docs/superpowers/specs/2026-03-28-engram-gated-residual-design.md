# EngramGatedResidual Module Design

**Goal:** Add a gated residual injection module to harmony-inference that allows Engram embeddings (knowledge from the mesh) to be injected into the Qwen3 forward pass at configurable layer indices.

**Motivation:** The "Power of Friendship" architecture enables a $30 router with 512MB RAM to access 100B+ parameters of distributed knowledge. The `EngramGatedResidual` module is the compute core — it takes a transformer hidden state and a pre-resolved Engram embedding, applies learned gating to determine relevance, and adds the result as a residual. This is the building block that harmony-5h0b (forward pass wiring) and harmony-ws11 (training pipeline) depend on.

**Scope:** The `EngramGatedResidual` candle Module and the `forward_with_engram()` callback injection point in `ModelWeights`. No changes to the `InferenceEngine` trait or any harmony-node consumers.

---

## Architecture

The module follows the DeepSeek Engram conditional memory architecture: project → gate → conv1d → activate → residual add. It is a pure tensor math module with no I/O — the caller resolves Engram embeddings from the mesh before calling the forward pass.

Injection into the Qwen3 forward pass uses a callback pattern: `ModelWeights::forward_with_engram()` takes an optional `Fn(layer_idx, &hidden_state) -> Result<Option<Tensor>>` callback invoked after each transformer layer. The existing `forward()` becomes a thin wrapper passing `None`, preserving zero overhead for non-Engram inference.

---

## EngramGatedResidual Module

```rust
pub struct EngramGatedResidual {
    key_proj: Linear,           // [engram_dim → hidden_dim], no bias
    value_proj: Linear,         // [engram_dim → hidden_dim], no bias
    gate_norm: candle_nn::RmsNorm,  // normalize hidden state for gating
    key_norm: candle_nn::RmsNorm,   // normalize projected key for gating
    conv1d: Conv1d,             // depthwise causal conv1d, no bias
    hidden_dim: usize,
}
```

Uses `candle_nn::RmsNorm` directly (not the private tracing-wrapped version in `qwen3_ext.rs`). The Engram module is not on the hot path of every token like the transformer layers, so the tracing span overhead isn't warranted.

### Constructors

**For testing — random initialization:**

```rust
EngramGatedResidual::new(
    engram_dim: usize,      // e.g., 160 (DeepSeek) or model-specific
    hidden_dim: usize,      // e.g., 1536 (Qwen3-0.6B)
    conv_kernel_size: usize, // default 3
    rms_norm_eps: f64,      // e.g., 1e-6
    device: &Device,
) -> Result<Self>
```

Creates the module with random weights (Kaiming uniform for projections, ones for norms, zeros for conv1d). Used for unit tests and shape verification.

**For inference — pre-loaded weight tensors:**

```rust
EngramGatedResidual::from_tensors(
    key_proj_weight: Tensor,     // [hidden_dim, engram_dim]
    value_proj_weight: Tensor,   // [hidden_dim, engram_dim]
    gate_norm_weight: Tensor,    // [hidden_dim]
    key_norm_weight: Tensor,     // [hidden_dim]
    conv1d_weight: Tensor,       // [hidden_dim, 1, kernel_size]
    hidden_dim: usize,
    rms_norm_eps: f64,
) -> Result<Self>
```

Takes pre-loaded weight tensors (from CAS). The CAS loader (harmony-5h0b's scope) deserializes the weights and calls this constructor.

All weights are f32 (not quantized). Total parameter count for Qwen3-0.6B (hidden_dim=1536, engram_dim=160, kernel=3):

| Component | Shape | Params |
|-----------|-------|--------|
| key_proj weight | `[1536, 160]` | 245,760 |
| value_proj weight | `[1536, 160]` | 245,760 |
| gate_norm weight | `[1536]` | 1,536 |
| key_norm weight | `[1536]` | 1,536 |
| conv1d weight | `[1536, 1, 3]` | 4,608 |
| **Total** | | **~499,200 (~2MB f32)** |

Per injection layer. Two layers (2 and 14) = ~4MB total. Trivial on 512MB hardware. All Linear and Conv1d layers have no bias.

### Forward Pass

```rust
pub fn forward(
    &self,
    hidden_state: &Tensor,    // [b, l, hidden_dim]
    engram_embedding: &Tensor, // [b, l, engram_dim]
) -> Result<Tensor>           // [b, l, hidden_dim]
```

Steps (all tensors in NLC layout `[batch, seq_len, channels]` unless noted):

1. `key = key_proj(engram_embedding)` → `[b, l, hidden_dim]`
2. `value = value_proj(engram_embedding)` → `[b, l, hidden_dim]`
3. Compute gating scalar:
   - `h_norm = gate_norm(hidden_state)` → `[b, l, hidden_dim]`
   - `k_norm = key_norm(key)` → `[b, l, hidden_dim]`
   - `dot = (h_norm * k_norm).sum(dim=-1, keepdim=true)` → `[b, l, 1]` (element-wise multiply, then sum over hidden_dim)
   - `gate = (dot / sqrt(hidden_dim)).sigmoid()` → `[b, l, 1]`
4. `gated_value = gate * value` → `[b, l, hidden_dim]` (broadcast multiply)
5. Conv1d (requires NCL layout):
   - `gated_ncl = gated_value.transpose(1, 2)` → `[b, hidden_dim, l]`
   - `gated_ncl = left_pad(gated_ncl, kernel_size - 1)` → `[b, hidden_dim, l + kernel_size - 1]`
   - `conv_ncl = conv1d(gated_ncl)` → `[b, hidden_dim, l]`
   - `conv_out = conv_ncl.transpose(1, 2)` → `[b, l, hidden_dim]`
6. `activated = silu(conv_out)` → `[b, l, hidden_dim]`
7. `return hidden_state + activated` → `[b, l, hidden_dim]`

The gating scalar (step 3) is the key mechanism — it lets the model learn to ignore irrelevant Engram embeddings. When the gate is near 0, the hidden state passes through unchanged.

### Causal Conv1d

Depthwise (groups = hidden_dim, no bias) with left-padding only (causal). For kernel_size=3: pad 2 zeros on the left, no right padding. Candle's `Conv1d` expects NCL input `[batch, channels, length]`, so we transpose NLC → NCL before and NCL → NLC after. The `Conv1dConfig` uses `padding=0` (we handle padding manually via `Tensor::pad_with_zeros`) and `groups=hidden_dim` for depthwise. This ensures output at position `t` depends only on positions `≤ t`, preserving autoregressive causality.

### Sigmoid

The gating step uses `Tensor::sigmoid()` (candle built-in unary op, not from `candle_nn`).

---

## Callback Injection in ModelWeights

### New method

```rust
pub fn forward_with_engram(
    &self,
    input: &Tensor,
    cache: &mut InferenceCache,
    engram_fn: Option<&dyn Fn(usize, &Tensor) -> Result<Option<Tensor>>>,
) -> Result<Tensor>
```

### Existing forward() becomes a wrapper

```rust
pub fn forward(&self, input: &Tensor, cache: &mut InferenceCache) -> Result<Tensor> {
    self.forward_with_engram(input, cache, None)
}
```

### Layer loop with injection

```rust
for (i, layer) in self.layers.iter().enumerate() {
    h = layer.forward(&h, causal_mask.as_ref(), offset, &mut cache.layers[i])?;

    if let Some(f) = &engram_fn {
        if let Some(engram_residual) = f(i, &h)? {
            h = (h + engram_residual)?;
        }
    }
}
```

The callback returns `Ok(None)` for layers without injection, `Ok(Some(tensor))` for configured layers. The `EngramGatedResidual::forward` is called inside the callback by the caller, who owns both the module and the pre-resolved embeddings.

### Zero overhead guarantee

`forward()` (no Engram) calls `forward_with_engram(None)`. When `engram_fn` is `None`, the `if let Some(f)` check is a single branch-not-taken per layer — no allocation, no tensor ops, no overhead.

---

## File Map

**Created:**

| File | Purpose |
|------|---------|
| `crates/harmony-inference/src/engram_residual.rs` | `EngramGatedResidual` struct, forward, constructor, unit tests |

**Modified:**

| File | Change |
|------|--------|
| `crates/harmony-inference/src/qwen3_ext.rs` | Add `forward_with_engram()`, refactor `forward()` as wrapper |
| `crates/harmony-inference/src/lib.rs` | Add `pub mod engram_residual;` and re-export `EngramGatedResidual` |

**No dependency changes.** `candle_core` and `candle_nn` (already in Cargo.toml) provide Linear, Conv1d, RmsNorm, Activation::Silu. Sigmoid is a built-in `Tensor` method from `candle_core`.

**No consumer changes.** `InferenceEngine` trait, harmony-node event loop, DSD, WASM host — all unchanged.

---

## What This Does NOT Include

- Loading Engram weights from CAS (harmony-5h0b's scope)
- Wiring the callback into the event loop (harmony-5h0b's scope)
- Training the Engram tables (harmony-ws11's scope)
- Async shard prefetching (harmony-geef's scope)
- Layer index configuration / discovery (caller's responsibility)

---

## Testing Strategy

**Unit tests in `engram_residual.rs`** (no GGUF model needed):

1. **Shape preservation** — Random weights, random tensors `[1, 5, 64]` hidden + `[1, 5, 16]` engram → verify output is `[1, 5, 64]`
2. **Zero embedding passthrough** — All-zeros engram embedding → output equals hidden_state (key_proj(zeros)=zeros, value_proj(zeros)=zeros → gate * zeros = zeros regardless of gate value)
3. **Gate range** — Verify gating scalar is in `[0, 1]` (sigmoid bounds)
4. **Causal conv1d** — Verify output at position `t` is unaffected by inputs at position `t+1`

**Integration tests** (`#[ignore]`, require GGUF):

5. **forward_with_engram(None) == forward()** — Identical logits for both paths
6. **forward_with_engram(Some) completes** — Dummy callback at layer 2, verify no panic

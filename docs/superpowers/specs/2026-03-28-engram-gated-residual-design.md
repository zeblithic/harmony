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
    key_proj: Linear,       // [engram_dim → hidden_dim]
    value_proj: Linear,     // [engram_dim → hidden_dim]
    gate_norm: RmsNorm,     // normalize hidden state for gating
    key_norm: RmsNorm,      // normalize projected key for gating
    conv1d: Conv1d,         // depthwise causal conv1d [hidden_dim, kernel_size]
    hidden_dim: usize,
}
```

### Constructor

```rust
EngramGatedResidual::new(
    engram_dim: usize,      // e.g., 160 (DeepSeek) or model-specific
    hidden_dim: usize,      // e.g., 1536 (Qwen3-0.6B)
    conv_kernel_size: usize, // default 3
    rms_norm_eps: f64,      // e.g., 1e-6
    device: &Device,
) -> Result<Self>
```

All weights are f32 (not quantized) — these are small projection matrices loaded from CAS, not from the GGUF model file. Total parameter count for Qwen3-0.6B (hidden_dim=1536, engram_dim=160, kernel=3):
- key_proj: 160 × 1536 = 245,760
- value_proj: 160 × 1536 = 245,760
- gate_norm: 1536
- key_norm: 1536
- conv1d (depthwise): 1536 × 3 = 4,608
- **Total: ~499,200 params (~2MB in f32)**

Per injection layer. Two layers (2 and 14) = ~4MB total. Trivial on 512MB hardware.

### Forward Pass

```rust
pub fn forward(
    &self,
    hidden_state: &Tensor,    // [b, l, hidden_dim]
    engram_embedding: &Tensor, // [b, l, engram_dim]
) -> Result<Tensor>           // [b, l, hidden_dim]
```

Steps:

1. `key = key_proj(engram_embedding)` → `[b, l, hidden_dim]`
2. `value = value_proj(engram_embedding)` → `[b, l, hidden_dim]`
3. `gate = sigmoid(dot(gate_norm(hidden_state), key_norm(key)) / sqrt(hidden_dim))` → `[b, l, 1]` scalar per token
4. `gated_value = gate * value` → `[b, l, hidden_dim]`
5. `conv_out = causal_conv1d(gated_value)` → `[b, l, hidden_dim]` (left-padded, depthwise, groups=hidden_dim)
6. `activated = silu(conv_out)` → `[b, l, hidden_dim]`
7. `return hidden_state + activated` → `[b, l, hidden_dim]`

The gating scalar (step 3) is the key mechanism — it lets the model learn to ignore irrelevant Engram embeddings. When the gate is near 0, the hidden state passes through unchanged.

### Causal Conv1d

Depthwise (groups = hidden_dim) with left-padding only (causal). For kernel_size=3: pad 2 zeros on the left, no right padding. This ensures output at position `t` depends only on positions `≤ t`, preserving autoregressive causality.

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

**No dependency changes.** `candle_core` and `candle_nn` (already in Cargo.toml) provide Linear, Conv1d, RmsNorm, Activation::Silu.

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
2. **Zero embedding passthrough** — All-zeros engram embedding → output ≈ hidden_state (gate * zeros = zeros)
3. **Gate range** — Verify gating scalar is in `[0, 1]` (sigmoid bounds)
4. **Causal conv1d** — Verify output at position `t` is unaffected by inputs at position `t+1`

**Integration tests** (`#[ignore]`, require GGUF):

5. **forward_with_engram(None) == forward()** — Identical logits for both paths
6. **forward_with_engram(Some) completes** — Dummy callback at layer 2, verify no panic

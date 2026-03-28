# KV Cache Externalization for harmony-inference

**Goal:** Refactor `harmony-inference` to externalize the KV cache from the engine, making the engine stateless after initialization and enabling downstream consumers to inspect, compress, serialize, and share cache state.

**Motivation:** The current `QwenEngine` couples model weights to KV cache state via candle-transformers' `&mut self` forward pass. This blocks four P3 beads: TurboQuant compression (harmony-p3zv), q8_0 KV quantization (harmony-5c5), PagedAttention (harmony-643), and CAS-based prefill sharing (harmony-hbf0). All require the ability to read, write, or replace the KV cache between forward passes.

**Scope:** harmony-inference crate refactor + harmony-node consumer updates. No serialization, no compression hooks, no multi-conversation wiring — those are downstream beads.

---

## Architecture

The engine becomes a stateless function box (weights + tokenizer) after initialization. All mutable inference state moves to a caller-owned `InferenceCache` value. The caller creates, passes, and drops caches; the engine just runs math.

```
Before:  engine.forward(&mut self, tokens) → logits   [cache hidden inside]
After:   engine.forward(&self, tokens, &mut cache) → logits   [cache owned by caller]
```

This enables:
- `Arc<QwenEngine>` sharing across conversations (weights are `&self`)
- Raw K/V tensor access for TurboQuant compression between forward calls
- Cache serialization for CAS distribution (prefill sharing)
- Clean lifecycle — no `reset()`, just drop the cache

---

## InferenceEngine Trait

```rust
pub trait InferenceEngine {
    // --- Initialization (&mut self) ---
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError>;
    fn load_tokenizer(&mut self, tokenizer_json: &[u8]) -> Result<(), InferenceError>;

    // --- Inference (&self, stateless) ---
    fn forward(&self, tokens: &[u32], cache: &mut InferenceCache) -> Result<Vec<f32>, InferenceError>;
    fn sample(&self, logits: &[f32], params: &SamplingParams, history: &[u32]) -> Result<u32, InferenceError>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError>;
    fn eos_token_id(&self) -> Option<u32>;

    // --- Cache factory ---
    fn new_cache(&self) -> Result<InferenceCache, InferenceError>;
}
```

Changes from current trait:
- `forward`: `&mut self` → `&self` + `&mut InferenceCache`. Returns `ForwardFailed` if tokens slice is empty. On error, the cache may be in an indeterminate state — drop and recreate before reuse.
- `sample`: gains `history: &[u32]` parameter. Pass the **full** token history; the implementation applies `repeat_last_n` windowing from `params` internally.
- `reset()`: removed — caller drops the cache or creates a new one
- `new_cache()`: added — engine creates a correctly-sized empty cache from its architecture params. Returns `ModelNotLoaded` if model not yet loaded.

---

## InferenceCache Type

```rust
/// Externalized KV cache state for transformer inference.
///
/// Created via `InferenceEngine::new_cache()`. Passed to `forward()` on each call.
/// The engine appends new K/V tensors; the caller owns the lifecycle.
pub struct InferenceCache {
    /// Per-layer Key/Value tensor pairs. `None` = layer not yet populated.
    /// Shape per tensor: [1, num_kv_heads, seq_len, head_dim]
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// Number of tokens consumed so far (passed as position offset to candle).
    pub position: usize,
    /// Expected layer count (for validation).
    pub num_layers: usize,
    /// Expected head dimension (for validation).
    pub head_dim: usize,
    /// Expected KV head count (for validation).
    pub num_kv_heads: usize,
}
```

- Lives in `harmony-inference` alongside the trait
- Depends only on `candle_core::Tensor` (not candle-transformers)
- `forward()` validates `cache.num_layers` against model; returns `InferenceError::CacheMismatch` on mismatch
- Head dim / KV head count validated implicitly by candle tensor ops (dimension mismatch → `ForwardFailed`)

Public methods: `new(num_layers, head_dim, num_kv_heads)`, `len()`, `is_empty()`.

`InferenceCache::new()` is public for external construction (e.g., deserialization in harmony-hbf0). Callers must know the architecture params — these are available from the model's GGUF metadata or from `InferenceEngine::new_cache()` which reads them from the loaded model.

---

## QwenEngine Refactor

**State reduction:**

```rust
// Before (5 fields, 2 mutable during inference)
pub struct QwenEngine {
    model: Option<ModelWeights>,
    tokenizer: Option<Tokenizer>,
    device: Device,
    position: usize,          // REMOVED
    token_history: Vec<u32>,  // REMOVED
}

// After (3 fields, none mutable during inference)
pub struct QwenEngine {
    model: Option<ModelWeights>,
    tokenizer: Option<Tokenizer>,
    device: Device,
}
```

**Forked Qwen3 module:** Upstream `candle_transformers::models::quantized_qwen3::ModelWeights::forward` takes `&mut self` (internal cache). We fork into `src/qwen3_ext.rs`, applying the candle-pipelines-models refactor:

- `ModelWeights::forward(&self, input, cache: &mut InferenceCache)` — iterates layers with `enumerate`, passing `&mut cache.layers[i]` to each `DecoderLayer`
- `DecoderLayer::forward(&self, ..., kv_slot: &mut Option<(Tensor, Tensor)>)` — passes slot to attention
- `AttentionWeights::forward(&self, ..., kv_slot: &mut Option<(Tensor, Tensor)>)` — on each call, concatenates new K/V with existing slot via `Tensor::cat` along dim 2 (seq_len), storing the result back. If slot is `None`, stores the new K/V directly.
- `forward()` advances `cache.position` by `tokens.len()` on success.

**KV cache concat logic in AttentionWeights::forward (pseudocode):**
```rust
// After computing k, v from projections and applying RoPE:
let (k, v) = match kv_slot.take() {
    Some((prev_k, prev_v)) => {
        (Tensor::cat(&[&prev_k, &k], 2)?,   // concat along seq_len dim
         Tensor::cat(&[&prev_v, &v], 2)?)
    }
    None => (k, v),
};
*kv_slot = Some((k.clone(), v.clone()));
// Use k, v for attention computation...
```

**Additional forked utilities (~70 lines):**
- `QMatMul`: tracing wrapper around `candle::quantized::QMatMul` (~30 lines)
- `RmsNorm`: dequantized weight + `candle_nn::ops::rms_norm` (~27 lines)
- `repeat_kv`: GQA key/value head expansion via cat + reshape (~10 lines)

Total fork: ~320 lines. All attention math is identical to upstream — only the cache threading changes.

**Dependency change:** Remove `candle-transformers` from harmony-inference. Add `candle-nn` as direct dependency. Net simplification — we depend on lower-level candle crates, not the entire model zoo.

---

## harmony-node Consumer Changes

### Event loop — `run_inference_loop()` (event_loop.rs)

```rust
fn run_inference_loop(
    engine: &QwenEngine,  // was: mut engine: QwenEngine
    tx: mpsc::Sender<InferenceResult>,
    ...
) {
    let mut cache = engine.new_cache()?;
    let mut history: Vec<u32> = Vec::new();

    // Prefill
    let mut logits = engine.forward(&input_tokens, &mut cache)?;
    history.extend_from_slice(&input_tokens);

    // Decode loop
    loop {
        let next = engine.sample(&logits, &params, &history)?;
        history.push(next);
        // ... stream chunk, check eos ...
        logits = engine.forward(&[next], &mut cache)?;
    }
    // cache dropped naturally — no reset() needed
}
```

The `InferenceResult::Complete` and `Failed` variants still return the engine for reuse. Since it's stateless, no `reset()` needed before returning.

### Runtime — engine ownership (runtime.rs)

`take_inference_engine()`/`return_inference_engine()` pattern unchanged. Engine is stateless so handoff between inference and DSD is trivial. `reset_inference_after_panic()` simplifies — no stale cache to worry about.

### DSD verification — `verify_draft_tokens()` (runtime.rs)

```rust
fn verify_draft_tokens(engine: &QwenEngine, request: &VerifyRequest) -> Result<VerifyResponse, String> {
    let mut cache = engine.new_cache()?;
    let mut logits = engine.forward(&context_tokens, &mut cache)?;
    for draft in &drafts {
        // ... accept/reject ...
        logits = engine.forward(&[draft.token_id], &mut cache)?;
    }
    // cache dropped — clean slate
}
```

No `reset()` call needed. Fresh cache per verification.

### DSD drafting — `handle_speculative_query()` (runtime.rs)

Currently calls `engine.reset()`, then `engine.forward(&prompt_tokens)`, then loops `engine.forward(&[token_id])` to generate draft tokens. Converts to the same pattern: create fresh cache, prefill, draft loop. Cache is dropped when the function returns.

### DSD continuation — `handle_verify_response()` (runtime.rs)

Currently calls `engine.reset()`, then `engine.forward(&full_sequence)` to re-prefill the full context (accepted tokens + bonus), then drafts more tokens. Same conversion: fresh cache, prefill full sequence, draft loop.

**Note:** Both DSD functions currently re-prefill from scratch on each call (no cache persistence between rounds). This is correct for the current stateless DSD design. A future optimization could store an `InferenceCache` in `DsdSession` to avoid re-prefilling the accepted prefix — but that's out of scope for this change.

---

## Error Handling

New variant:

```rust
#[error("cache mismatch: expected {expected} layers, got {actual}")]
CacheMismatch { expected: usize, actual: usize },
```

Checked at the top of `forward()`. All other error paths unchanged.

---

## File Map

**Modified:**

| File | Change |
|------|--------|
| `crates/harmony-inference/src/lib.rs` | New trait signature, `InferenceCache` struct, export `InferenceCache` |
| `crates/harmony-inference/src/engine.rs` | Remove `position`/`token_history`, all inference methods `&self` |
| `crates/harmony-inference/src/error.rs` | Add `CacheMismatch` variant |
| `crates/harmony-inference/Cargo.toml` | Remove `candle-transformers`, add `candle-nn` |
| `crates/harmony-node/src/event_loop.rs` | `run_inference_loop` uses external cache and history |
| `crates/harmony-node/src/runtime.rs` | Remove `reset()` calls, simplify lifecycle |

**Created:**

| File | Purpose |
|------|---------|
| `crates/harmony-inference/src/qwen3_ext.rs` | Forked quantized Qwen3 with externalized cache (~320 lines, includes QMatMul/RmsNorm/repeat_kv utilities) |

**Unchanged:**

| File | Reason |
|------|--------|
| `crates/harmony-inference/src/sampling.rs` | Already pure functions |
| `crates/harmony-node/src/inference.rs` | Wire format types unaffected |

---

## What This Unblocks

- **harmony-p3zv** (TurboQuant) — `cache.layers` gives raw K/V tensors for compression
- **harmony-5c5** (q8_0 KV) — same access point for quantization
- **harmony-643** (PagedAttention) — can replace `InferenceCache` internals without trait changes
- **harmony-hbf0** (Prefill sharing) — cache is a serializable value, can be distributed via CAS

## What This Does NOT Include

- Serialization/deserialization of `InferenceCache` (harmony-hbf0's scope)
- TurboQuant/q8_0 compression hooks (harmony-p3zv / harmony-5c5)
- `Arc<QwenEngine>` multi-conversation wiring (trivially enabled, not connected)
- TOPLOC proof generation/verification (harmony-hbf0's scope)

---

## Testing Strategy

- **Unit tests in harmony-inference:** Round-trip `new_cache()` → `forward()` → verify cache population (layers filled, position advanced). Cache mismatch detection. Sampling with explicit history matches previous behavior.
- **Integration tests:** Existing `tests/integration.rs` updated to use external cache pattern. Same model loading, same generation, same outputs — just different API surface.
- **harmony-node tests:** Existing inference query tests updated for new wire format expectations (unchanged) and new engine lifecycle (no reset).

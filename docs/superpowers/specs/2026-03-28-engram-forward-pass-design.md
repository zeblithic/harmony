# Engram Forward Pass Integration Design

**Goal:** Wire Engram conditional memory into the inference forward pass via a two-phase sans-I/O bridge and a public `forward_with_engram` method on `QwenEngine`.

**Motivation:** The EngramGatedResidual module (harmony-vuig, PR #143) and the callback injection point (`forward_with_engram` in qwen3_ext.rs) are built. This bead connects them to the harmony-engram crate's hash-based lookup system, producing a complete inference-side API that takes tokens + shard data and returns Engram-augmented logits. No I/O — the event loop wiring (harmony-d13v) provides shard data separately.

**Scope:** harmony-inference crate only. New `engram_bridge.rs` module, new public method on `QwenEngine`, new `EngramContext` struct. No changes to harmony-node, no event loop changes, no InferenceEngine trait changes.

---

## Architecture

Two-phase sans-I/O bridge between harmony-engram (hash lookups) and EngramGatedResidual (tensor math):

```
Phase 1: prepare_engram_request(client, tokens)
    tokens → extract bigrams + trigrams → client.lookup() per N-gram
    → deduplicate shards → EngramRequest { required_shards, lookups }

    [Caller fetches shards from cache/mesh/disk]

Phase 2: resolve_engram_embeddings(client, request, shard_data, device)
    shard_data → client.resolve() per lookup → aggregate per position
    → candle Tensor [1, seq_len, engram_dim]
```

The resolved tensor feeds into `QwenEngine::forward_with_engram()` via `EngramContext`, which builds the callback closure internally.

---

## engram_bridge.rs — Two-Phase API

### Phase 1: Prepare Lookups

```rust
pub fn prepare_engram_request(
    client: &EngramClient,
    tokens: &[u32],
) -> EngramRequest
```

Extracts bigrams and trigrams from the token sequence. For tokens `[A, B, C, D]`:
- Bigrams: `[A,B]` at pos 1, `[B,C]` at pos 2, `[C,D]` at pos 3
- Trigrams: `[A,B,C]` at pos 2, `[B,C,D]` at pos 3

Each N-gram is attributed to its **last token's position** (the position where the embedding will be injected). Multiple N-grams at the same position are aggregated in Phase 2.

Calls `client.lookup()` for each N-gram, deduplicates required shard indices, and returns:

```rust
pub struct EngramRequest {
    /// Deduplicated shards needed — caller fetches these.
    pub required_shards: Vec<ShardRequest>,
    /// Per-N-gram lookups with position attribution.
    pub lookups: Vec<NgramLookup>,
    /// Input sequence length (for output tensor sizing).
    pub seq_len: usize,
}

pub struct ShardRequest {
    pub shard_index: u64,
    pub cid: [u8; 32],
}

pub struct NgramLookup {
    /// Position in the token sequence this N-gram's embedding covers.
    pub token_position: usize,
    /// Lookup result from harmony-engram (shard indices + byte offsets).
    pub lookup: EngramLookup,
}
```

For seq_len=1 (decode step with no prior context), no N-grams are extracted and the request has empty `required_shards` and `lookups`.

### Phase 2: Resolve to Tensor

```rust
pub fn resolve_engram_embeddings(
    client: &EngramClient,
    request: &EngramRequest,
    shard_data: &HashMap<u64, Vec<u8>>,
    device: &Device,
) -> Result<Tensor>
```

The caller fetches each shard by its CID but stores the result in a `HashMap<u64, Vec<u8>>` keyed by `shard_index` (not CID). Phase 2 looks up shard data by `shard_index`.

**Per-lookup assembly** (the core glue logic): `EngramClient::resolve()` takes `&[&[u8]]` — one byte slice per head, ordered by `lookup.shard_indices`. For each `NgramLookup`:
1. Build `Vec<&[u8]>` of length `num_heads` by indexing into `shard_data` using `lookup.shard_indices[head_idx]` for each head
2. Call `client.resolve(&lookup, &shard_slices)` → `Result<Vec<u8>>` (f32 little-endian, length = `embedding_dim * 4`)
3. Interpret the result as `embedding_dim` little-endian f32 values

**Per-position aggregation:** Sum f32 values element-wise in a CPU buffer (`Vec<f32>`) for positions where multiple N-grams overlap. Positions with no N-gram coverage keep zeros.

**Final conversion:** Construct candle Tensor `[1, seq_len, engram_dim]` from the aggregated f32 buffer via `Tensor::from_vec`.

**Errors:** Missing shard data (key not in HashMap) → `candle_core::bail!`. `EngramError` from `resolve()` → mapped to `candle_core::Error` via `format!`.

### Decode-Step Usage

During autoregressive decoding, `prepare_engram_request` receives only the new token(s), not the full context. With seq_len=1 and no prior context, no bigrams or trigrams can be formed — the request is empty, and `resolve_engram_embeddings` returns a zero tensor.

For Engram coverage during decode, the caller should maintain a sliding window of the last 3+ tokens and pass it to `prepare_engram_request`. The resulting embeddings tensor should have shape `[1, window_len, engram_dim]`, but only the last position (matching the new token) feeds into the forward pass. This windowing strategy is the event loop's responsibility (harmony-d13v), not the bridge's.

---

## EngramContext and QwenEngine API

### EngramContext

```rust
pub struct EngramContext<'a> {
    /// The gated residual module (holds projection weights).
    pub module: &'a EngramGatedResidual,
    /// Pre-resolved embeddings from resolve_engram_embeddings().
    /// Shape: [1, seq_len, engram_dim]
    pub embeddings: Tensor,
    /// Which transformer layers to inject at (e.g., &[2, 14]).
    pub injection_layers: &'a [usize],
}
```

### QwenEngine::forward_with_engram

```rust
impl QwenEngine {
    pub fn forward_with_engram(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
        engram: &EngramContext<'_>,
    ) -> Result<Vec<f32>, InferenceError>
}
```

Internally:
1. Same validation as `forward()` (empty tokens, model loaded, cache mismatch)
2. Build tensor input from tokens
3. Build callback closure from `EngramContext`:
   ```rust
   let engram_fn = |layer_idx: usize, hidden_state: &Tensor| -> Result<Option<Tensor>> {
       if engram.injection_layers.contains(&layer_idx) {
           Ok(Some(engram.module.forward(hidden_state, &engram.embeddings)?))
       } else {
           Ok(None)
       }
   };
   ```
4. Delegate to `model.forward_with_engram(input, cache, Some(&engram_fn))`
5. Extract logits (same as `forward()`)

The existing `forward()` method is completely unchanged. The `InferenceEngine` trait is unchanged. `forward_with_engram` is an inherent method on `QwenEngine`, not a trait method — Engram injection is optional and engine-specific.

Errors from `EngramGatedResidual::forward()` (candle `Result`) are mapped through `InferenceError::ForwardFailed(String)`, consistent with the existing error mapping pattern in `forward()`.

---

## File Map

**Created:**

| File | Purpose |
|------|---------|
| `crates/harmony-inference/src/engram_bridge.rs` | Two-phase API, N-gram extraction, types (`EngramRequest`, `ShardRequest`, `NgramLookup`) |

**Modified:**

| File | Change |
|------|--------|
| `crates/harmony-inference/src/engine.rs` | Add `forward_with_engram()` method, `EngramContext` struct |
| `crates/harmony-inference/src/lib.rs` | Add `pub mod engram_bridge;`, re-export `EngramContext` |
| `crates/harmony-inference/Cargo.toml` | Add `harmony-engram = { workspace = true }` dependency |

**Unchanged:** harmony-node, harmony-compute, InferenceEngine trait, qwen3_ext.rs, engram_residual.rs.

---

## What This Does NOT Include

- Shard fetching from CAS/mesh (harmony-d13v's scope)
- Event loop integration (harmony-d13v's scope)
- Node config for Engram manifest CID (harmony-d13v's scope)
- EngramGatedResidual weight loading from CAS (harmony-d13v's scope)
- Training Engram tables (harmony-ws11's scope)
- Async prefetch pipeline (harmony-geef's scope)

---

## Testing Strategy

**Unit tests in `engram_bridge.rs`** (no GGUF, no real shards):

1. **N-gram extraction** — `[1, 2, 3, 4]` produces bigrams `[1,2]@1`, `[2,3]@2`, `[3,4]@3` and trigrams `[1,2,3]@2`, `[2,3,4]@3`
2. **Prepare request deduplication** — multiple N-grams needing the same shard → only one `ShardRequest`
3. **Resolve with mock shard data** — fake EngramClient, mock shard bytes → verify output tensor shape `[1, seq_len, engram_dim]`
4. **Single token produces empty request** — seq_len=1 → empty required_shards, empty lookups
5. **Position aggregation** — multiple N-grams at position 3 → embeddings summed
6. **Missing shard data returns error** — `shard_data` HashMap missing a required shard index → error

**Unit test in `engine.rs`:**

7. **forward_with_engram without model returns ModelNotLoaded**

**Integration test (`#[ignore]`):**

8. **forward_with_engram produces different logits than forward** — non-zero Engram embeddings at layer 2 → logits differ

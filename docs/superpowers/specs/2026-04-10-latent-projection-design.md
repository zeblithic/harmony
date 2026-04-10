# ZEB-63: Latent Projection for Engram Key Compression

## Overview

Replace the current xxhash64 token-byte hashing with a learned semantic hash for Engram key generation. A 2-layer MLP projects token embeddings into a compact latent space, then binarizes them into locality-sensitive hash codes. Semantically similar inputs produce the same or nearby binary codes, upgrading Engram retrieval from surface-level N-gram matching to semantic retrieval while reducing mesh broadcast key size.

## Architecture

### LatentProjection Module

A 2-layer MLP with tanh output activation that maps token embeddings to compact latent codes:

```
token_embeddings [batch, seq_len, hidden_dim]
  → Linear(hidden_dim, 256) + SiLU
  → Linear(256, 128) + tanh
  → latent_codes [batch, seq_len, 128]
```

- **SiLU** intermediate activation (matches model MLP style)
- **tanh** output activation (pushes values toward {-1, +1} for binarization)
- Intermediate dim (256) and output dim (128) are inferred from GGUF weight shapes — no hardcoded constants
- Operates on token embeddings (layer 0 output), not deeper hidden states — this keeps shard requests parallelizable with the forward pass and avoids sequencing dependencies

### Binary Key Derivation (Learned LSH)

The tanh output is binarized via sign bits:

```
latent_codes [seq_len, 128]
  → sign(x) → {0, 1} per dimension
  → pack into bytes → 16 bytes per position
```

Each position's 16-byte binary key is then hashed with xxhash64 using per-head seeds, producing shard indices via the existing `compute_lookup` infrastructure. The multi-head structure provides multiple lookup candidates per query.

**Why this preserves nearest-neighbor topology:** The MLP is trained (via contrastive loss) to map semantically similar embeddings to nearby latent vectors. The tanh activation concentrates values near {-1, +1}, so the sign-bit binarization has low information loss. Similar inputs get the same binary code → same hash → same shard. The xxhash64 step is just a table index, not a semantic operation.

### N-gram Windowing

Rather than projecting single-position embeddings (which are non-contextual), the module averages embeddings within each N-gram window before projecting:

- Bigram at [i, i+1]: `avg(embed[i], embed[i+1])` → project → binary key at position i+1
- Trigram at [i, i+1, i+2]: `avg(embed[i], embed[i+1], embed[i+2])` → project → binary key at position i+2

This mirrors the existing bi/trigram structure and position attribution scheme (last token of the N-gram). The averaging gives the MLP local context — it sees "what tokens are in this window" rather than just a single token ID.

A convenience method `project_ngrams()` handles windowing + averaging + projection + binarization in one call:

```rust
pub fn project_ngrams(
    &self,
    embeddings: &Tensor,  // [1, seq_len, hidden_dim]
    seq_len: usize,
) -> Result<(Vec<Vec<u8>>, Vec<usize>)>  // (binary_keys, positions)
```

## Bridge Integration

### New function: `prepare_engram_request_latent()`

```rust
pub fn prepare_engram_request_latent(
    client: &EngramClient,
    binary_keys: &[Vec<u8>],
    positions: &[usize],
    seq_len: usize,
) -> Result<EngramRequest>
```

Takes pre-computed binary keys from `LatentProjection` and produces the same `EngramRequest` type used by the existing resolve path. Internally hashes each binary key with xxhash64 per-head seeds.

### New hash helper: `compute_lookup_from_bytes()`

```rust
pub fn compute_lookup_from_bytes(config: &EngramConfig, key_bytes: &[u8]) -> EngramLookup
```

In `harmony-engram/src/hash.rs`. Same logic as `compute_lookup` but takes arbitrary bytes. The existing `compute_lookup` is refactored to call this internally (token-to-bytes is just a special case).

### Resolve path unchanged

`resolve_engram_embeddings()` works with `EngramRequest` regardless of how the lookups were generated. No changes needed.

## GGUF Loading

### Tensor names

```
harmony.latent_projection.layer1.weight  [intermediate_dim, hidden_dim]
harmony.latent_projection.layer1.bias    [intermediate_dim]
harmony.latent_projection.layer2.weight  [latent_dim, intermediate_dim]
harmony.latent_projection.layer2.bias    [latent_dim]
```

### Auto-detection

During `HarmonyEngine::load_gguf()`, check for `harmony.latent_projection.layer1.weight` in the tensor list. If present, construct `LatentProjection` from the four weight tensors and store in `self.latent_projection`. If absent, field stays `None`.

All dimensions (hidden_dim, intermediate_dim, latent_dim) are inferred from weight shapes. No GGUF metadata keys needed.

## HarmonyEngine Integration

```rust
pub struct HarmonyEngine {
    // ... existing fields ...
    latent_projection: Option<LatentProjection>,
}
```

- `set_latent_projection(&mut self, proj: LatentProjection)` — setter (same pattern as `set_uq_head()`)
- `latent_projection(&self) -> Option<&LatentProjection>` — getter for event loop
- `token_embeddings(&self, token_ids: &[u32]) -> Result<Tensor>` — new method that runs `model.embed_tokens.forward()` and returns `[1, seq_len, hidden_dim]`

### HarmonyModel accessor

```rust
impl HarmonyModel {
    pub fn token_embeddings(&self, token_ids: &[u32]) -> Result<Tensor>
}
```

Converts token IDs to a tensor, runs the embedding lookup, returns `[1, seq_len, hidden_dim]`.

## Contrastive Loss for Training

Standalone function in `latent_projection.rs` that KRILE imports:

```rust
pub fn contrastive_loss(
    original: &Tensor,    // [batch, seq_len, hidden_dim]
    projected: &Tensor,   // [batch, seq_len, latent_dim]
    temperature: f32,
    k: usize,             // number of top neighbors to preserve
) -> Result<Tensor>
```

**InfoNCE-style loss** ensuring nearest neighbors in the original embedding space remain nearest neighbors in the projected latent space:

1. Compute cosine similarity matrices in both spaces
2. For each anchor, identify top-k neighbors in original space (k=4 default)
3. Cross-entropy loss where projected similarities / temperature are logits and original top-k neighbors are targets

**Why InfoNCE over triplet:** Uses the full batch as negatives for richer gradient signal and more stable training.

**Defaults:** k=4 (preserve top-4 neighbors), temperature=0.07. Both are function parameters KRILE can tune.

## Event Loop Changes

Conditional path within the existing Engram branch:

```
Has Engram? ──no──→ plain forward (unchanged)
    │
   yes
    │
Has LatentProjection? ──no──→ existing token-hash path (unchanged)
    │
   yes
    │
token_embeddings() → project_ngrams() → prepare_engram_request_latent()
→ fetch shards → resolve → forward_with_engram (unchanged from here)
```

### ChunkedEngramScheduler

During chunked decode, `prepare_request()` accepts an optional `&LatentProjection` and model reference. If present, uses latent keys for the buffered token window. If absent, existing token-hash behavior.

### Prefill vs. decode

- **Prefill**: Full token sequence available, `project_ngrams` runs once on the whole sequence
- **Decode**: Runs on the sliding window at each chunk boundary (same cadence as today's N-gram extraction)

## Feature Gate

GGUF weight presence gates the feature (same pattern as UQ head):
- Projection weights in GGUF → `LatentProjection` loaded → latent key path active
- No projection weights → `latent_projection` is `None` → falls back to xxhash64 token-hash path
- No config flag needed. Zero overhead when not loaded.

## Fallback

If no latent projection weights are present in the GGUF:
- `engine.latent_projection()` returns `None`
- Event loop takes the existing token-hash path
- `prepare_engram_request()` with xxhash64 runs exactly as today
- All existing GGUF models continue to work unchanged

## File Changes

| File | Change |
|------|--------|
| **NEW** `harmony-inference/src/latent_projection.rs` | `LatentProjection` struct, `project()`, `project_ngrams()`, `to_binary_keys()`, `contrastive_loss()` |
| `harmony-inference/src/lib.rs` | Module declaration + re-exports |
| `harmony-inference/src/harmony_model.rs` | `token_embeddings()` accessor on `HarmonyModel` |
| `harmony-inference/src/harmony_engine.rs` | `latent_projection` field, getter/setter, auto-load in `load_gguf()` |
| `harmony-engram/src/hash.rs` | `compute_lookup_from_bytes()` + refactor `compute_lookup` to use it |
| `harmony-inference/src/engram_bridge.rs` | `prepare_engram_request_latent()` function |
| `harmony-inference/src/chunked_engram.rs` | New `prepare_request_latent()` method; existing `prepare_request()` unchanged |
| `harmony-node/src/event_loop.rs` | Conditional path: if projection available, use latent keys |

Not changed: `engram_residual.rs`, `uq_head.rs`, `speculative_decode.rs`, `config.rs`, `continuous_thought.rs`.

## Tests

- **LatentProjection (6):** `project_output_shape`, `project_output_bounded_by_tanh`, `to_binary_keys_length_and_determinism`, `project_ngrams_bigrams_and_trigrams`, `similar_embeddings_produce_same_binary_key`, `different_embeddings_produce_different_keys`
- **Contrastive loss (2):** `contrastive_loss_shape_and_finite`, `contrastive_loss_decreases_with_aligned_projections`
- **Bridge (2):** `prepare_latent_request_produces_valid_lookups`, `prepare_latent_request_matches_token_count`
- **Hash (2):** `compute_lookup_from_bytes_matches_manual`, `compute_lookup_refactor_preserves_existing_hashes` (regression test)
- **Integration (2):** `engine_without_projection_uses_token_hash`, `token_embeddings_accessor_shape`

## Verification

```bash
cargo test -p harmony-engram -- --nocapture
cargo test -p harmony-inference -- --nocapture
cargo test -p harmony-inference latent_projection -- --nocapture
cargo check -p harmony-node
```

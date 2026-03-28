# Engram Forward Pass Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Engram conditional memory into the inference forward pass via a two-phase sans-I/O bridge and a public `forward_with_engram` method on `QwenEngine`.

**Architecture:** New `engram_bridge.rs` module with `prepare_engram_request()` (N-gram extraction + hash lookup) and `resolve_engram_embeddings()` (shard assembly + tensor conversion). `QwenEngine` gains a public `forward_with_engram()` that builds the callback closure from an `EngramContext` struct. No I/O, no event loop changes.

**Tech Stack:** harmony-inference, harmony-engram, candle-core

**Spec:** `docs/superpowers/specs/2026-03-28-engram-forward-pass-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-inference/src/engram_bridge.rs` | Two-phase API: N-gram extraction, prepare lookups, resolve to tensor. Types: `EngramRequest`, `ShardRequest`, `NgramLookup` |
| `crates/harmony-inference/src/engine.rs` | Add `forward_with_engram()` method and `EngramContext` struct |
| `crates/harmony-inference/src/lib.rs` | Add `pub mod engram_bridge;`, re-export `EngramContext` |
| `crates/harmony-inference/Cargo.toml` | Add `harmony-engram` dependency |

---

### Task 1: Engram Bridge — Types and N-gram Extraction

Add the `engram_bridge` module with types and the Phase 1 function `prepare_engram_request`.

**Files:**
- Create: `crates/harmony-inference/src/engram_bridge.rs`
- Modify: `crates/harmony-inference/src/lib.rs`
- Modify: `crates/harmony-inference/Cargo.toml`

- [ ] **Step 1: Add harmony-engram dependency**

In `crates/harmony-inference/Cargo.toml`, add to `[dependencies]` after the `candle-nn` line:

```toml
harmony-engram = { workspace = true }
```

- [ ] **Step 2: Add module declaration in lib.rs**

In `crates/harmony-inference/src/lib.rs`, add after `pub mod engram_residual;` (line 21):

```rust
pub mod engram_bridge;
```

- [ ] **Step 3: Create engram_bridge.rs with types and prepare function**

Create `crates/harmony-inference/src/engram_bridge.rs`:

```rust
//! Two-phase sans-I/O bridge between harmony-engram (hash lookups) and
//! EngramGatedResidual (tensor math).
//!
//! Phase 1: [`prepare_engram_request`] — extract N-grams from tokens, compute
//! hash lookups, return which shards to fetch.
//!
//! Phase 2: [`resolve_engram_embeddings`] — assemble shard data, resolve
//! embeddings, aggregate per position, return candle Tensor.

use std::collections::{HashMap, HashSet};

use candle_core::{DType, Device, Result, Tensor};
use harmony_engram::{EngramClient, EngramLookup};

/// A shard that needs to be fetched by the caller.
#[derive(Debug, Clone)]
pub struct ShardRequest {
    /// Index into the Engram table's shard list.
    pub shard_index: u64,
    /// Content identifier for CAS fetch.
    pub cid: [u8; 32],
}

/// Per-N-gram lookup result with position attribution.
#[derive(Debug, Clone)]
pub struct NgramLookup {
    /// Position in the token sequence this N-gram's embedding covers.
    /// This is the **last token's position** in the N-gram.
    pub token_position: usize,
    /// Lookup result from harmony-engram (shard indices + byte offsets per head).
    pub lookup: EngramLookup,
}

/// Result of Phase 1: which shards to fetch and how to resolve them.
#[derive(Debug, Clone)]
pub struct EngramRequest {
    /// Deduplicated shards needed — caller fetches these by CID.
    pub required_shards: Vec<ShardRequest>,
    /// Per-N-gram lookups with position attribution.
    pub lookups: Vec<NgramLookup>,
    /// Input sequence length (for output tensor sizing).
    pub seq_len: usize,
}

/// Phase 1: Extract N-grams from tokens and compute hash lookups.
///
/// Extracts bigrams and trigrams, attributed to the **last token's position**.
/// For tokens `[A, B, C, D]`:
/// - Bigrams: `[A,B]` at pos 1, `[B,C]` at pos 2, `[C,D]` at pos 3
/// - Trigrams: `[A,B,C]` at pos 2, `[B,C,D]` at pos 3
///
/// Returns an [`EngramRequest`] with deduplicated shard requirements.
/// For seq_len < 2, returns an empty request (no N-grams possible).
pub fn prepare_engram_request(client: &EngramClient, tokens: &[u32]) -> EngramRequest {
    let seq_len = tokens.len();
    let mut lookups = Vec::new();
    let mut seen_shards = HashSet::new();
    let mut required_shards = Vec::new();

    // Extract bigrams (need at least 2 tokens)
    for i in 0..seq_len.saturating_sub(1) {
        let bigram = &tokens[i..i + 2];
        let lookup = client.lookup(bigram);
        collect_shards(client, &lookup, &mut seen_shards, &mut required_shards);
        lookups.push(NgramLookup {
            token_position: i + 1, // last token of bigram
            lookup,
        });
    }

    // Extract trigrams (need at least 3 tokens)
    for i in 0..seq_len.saturating_sub(2) {
        let trigram = &tokens[i..i + 3];
        let lookup = client.lookup(trigram);
        collect_shards(client, &lookup, &mut seen_shards, &mut required_shards);
        lookups.push(NgramLookup {
            token_position: i + 2, // last token of trigram
            lookup,
        });
    }

    EngramRequest {
        required_shards,
        lookups,
        seq_len,
    }
}

/// Helper: collect unique shard requests from a lookup.
fn collect_shards(
    client: &EngramClient,
    lookup: &EngramLookup,
    seen: &mut HashSet<u64>,
    shards: &mut Vec<ShardRequest>,
) {
    for &shard_idx in &lookup.shard_indices {
        if seen.insert(shard_idx) {
            if let Some(&cid) = client.shard_cid(shard_idx) {
                shards.push(ShardRequest {
                    shard_index: shard_idx,
                    cid,
                });
            }
        }
    }
}

/// Phase 2: Resolve shard data into a candle Tensor of embeddings.
///
/// The caller fetches each shard by CID and stores results in a
/// `HashMap<u64, Vec<u8>>` keyed by `shard_index` (not CID).
///
/// For each N-gram lookup, assembles per-head shard slices and calls
/// `client.resolve()` to get f32 bytes. Multiple N-grams at the same
/// position are summed element-wise. Positions with no N-gram coverage
/// get zero embeddings.
///
/// Returns a Tensor `[1, seq_len, embedding_dim]`.
pub fn resolve_engram_embeddings(
    client: &EngramClient,
    request: &EngramRequest,
    shard_data: &HashMap<u64, Vec<u8>>,
    device: &Device,
) -> Result<Tensor> {
    let embedding_dim = client.config().embedding_dim;
    let num_heads = client.config().num_heads as usize;

    // Aggregation buffer: seq_len × embedding_dim, initialized to zero.
    let mut buffer = vec![0.0f32; request.seq_len * embedding_dim];

    for ngram in &request.lookups {
        // Assemble per-head shard slices for client.resolve().
        let mut head_slices: Vec<&[u8]> = Vec::with_capacity(num_heads);
        for &shard_idx in &ngram.lookup.shard_indices {
            let data = shard_data.get(&shard_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing shard data for shard_index {shard_idx}"))
            })?;
            head_slices.push(data.as_slice());
        }

        // Resolve: extract + aggregate f16→f32 across heads.
        let f32_bytes = client
            .resolve(&ngram.lookup, &head_slices)
            .map_err(|e| candle_core::Error::Msg(format!("engram resolve failed: {e}")))?;

        // Interpret as f32 little-endian and sum into position buffer.
        let pos_offset = ngram.token_position * embedding_dim;
        for (i, chunk) in f32_bytes.chunks_exact(4).enumerate() {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            buffer[pos_offset + i] += val;
        }
    }

    // Convert aggregated buffer to Tensor [1, seq_len, embedding_dim].
    Tensor::from_vec(buffer, (1, request.seq_len, embedding_dim), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_engram::EngramConfig;

    /// Create a minimal EngramClient for testing.
    fn test_client() -> EngramClient {
        let config = EngramConfig {
            version: "test".into(),
            embedding_dim: 4,
            dtype_bytes: 2, // f16
            num_heads: 2,
            shard_size: 100,
            num_shards: 10,
            total_entries: 1000,
            hash_seeds: vec![42, 99],
        };
        // 10 shards with dummy CIDs
        let shard_cids: Vec<[u8; 32]> = (0..10)
            .map(|i| {
                let mut cid = [0u8; 32];
                cid[0] = i as u8;
                cid
            })
            .collect();
        EngramClient::from_manifest(config, shard_cids)
    }

    /// Create mock shard data: all zeros (valid f16 zero = 0x0000).
    fn zero_shard_data(num_shards: usize, shard_bytes: usize) -> HashMap<u64, Vec<u8>> {
        (0..num_shards as u64)
            .map(|i| (i, vec![0u8; shard_bytes]))
            .collect()
    }

    #[test]
    fn ngram_extraction_bigrams_and_trigrams() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[1, 2, 3, 4]);

        // 3 bigrams + 2 trigrams = 5 lookups
        assert_eq!(request.lookups.len(), 5);
        assert_eq!(request.seq_len, 4);

        // Bigrams at positions 1, 2, 3
        assert_eq!(request.lookups[0].token_position, 1);
        assert_eq!(request.lookups[1].token_position, 2);
        assert_eq!(request.lookups[2].token_position, 3);

        // Trigrams at positions 2, 3
        assert_eq!(request.lookups[3].token_position, 2);
        assert_eq!(request.lookups[4].token_position, 3);
    }

    #[test]
    fn single_token_produces_empty_request() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[42]);

        assert!(request.lookups.is_empty());
        assert!(request.required_shards.is_empty());
        assert_eq!(request.seq_len, 1);
    }

    #[test]
    fn empty_tokens_produces_empty_request() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[]);

        assert!(request.lookups.is_empty());
        assert!(request.required_shards.is_empty());
        assert_eq!(request.seq_len, 0);
    }

    #[test]
    fn shard_deduplication() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[1, 2, 3, 4]);

        // Multiple lookups may hit the same shard — required_shards should be deduplicated
        let shard_indices: Vec<u64> = request.required_shards.iter().map(|s| s.shard_index).collect();
        let unique: HashSet<u64> = shard_indices.iter().copied().collect();
        assert_eq!(shard_indices.len(), unique.len(), "shards should be deduplicated");
    }

    #[test]
    fn resolve_produces_correct_shape() {
        let client = test_client();
        let tokens = [1u32, 2, 3, 4];
        let request = prepare_engram_request(&client, &tokens);

        // Each shard needs to be large enough: shard_size(100) * vector_bytes(4*2=8) = 800 bytes
        let shard_data = zero_shard_data(10, 800);

        let tensor = resolve_engram_embeddings(
            &client,
            &request,
            &shard_data,
            &Device::Cpu,
        )
        .expect("resolve failed");

        assert_eq!(tensor.dims(), &[1, 4, 4]); // [1, seq_len=4, embedding_dim=4]
    }

    #[test]
    fn resolve_missing_shard_returns_error() {
        let client = test_client();
        let tokens = [1u32, 2, 3];
        let request = prepare_engram_request(&client, &tokens);

        // Provide empty HashMap — all shards missing
        let shard_data = HashMap::new();

        let result = resolve_engram_embeddings(
            &client,
            &request,
            &shard_data,
            &Device::Cpu,
        );

        assert!(result.is_err(), "should fail when shard data is missing");
    }

    #[test]
    fn resolve_zero_shards_produces_zero_tensor() {
        let client = test_client();
        let tokens = [1u32, 2, 3, 4];
        let request = prepare_engram_request(&client, &tokens);

        let shard_data = zero_shard_data(10, 800);

        let tensor = resolve_engram_embeddings(
            &client,
            &request,
            &shard_data,
            &Device::Cpu,
        )
        .expect("resolve failed");

        // All-zero shards → all-zero embeddings
        let max_val: f32 = tensor.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max_val < 1e-6, "zero shards should produce zero embeddings, got {max_val}");
    }
}
```

- [ ] **Step 4: Verify tests pass**

Run: `cargo test -p harmony-inference`

Expected: All existing tests pass plus 7 new engram_bridge tests.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/Cargo.toml crates/harmony-inference/src/lib.rs crates/harmony-inference/src/engram_bridge.rs
git commit -m "feat(inference): add engram_bridge module with two-phase lookup API"
```

---

### Task 2: EngramContext and QwenEngine::forward_with_engram

Add the `EngramContext` struct and the public `forward_with_engram` method to `QwenEngine`.

**Files:**
- Modify: `crates/harmony-inference/src/engine.rs`
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Add EngramContext struct and forward_with_engram method**

In `crates/harmony-inference/src/engine.rs`, add the following after the `use` statements (after line 13) and before the `QwenEngine` struct:

```rust
use crate::engram_residual::EngramGatedResidual;

/// Context for Engram-augmented inference.
///
/// Constructed by the caller with a trained module, pre-resolved embeddings
/// (from [`engram_bridge::resolve_engram_embeddings`](crate::engram_bridge::resolve_engram_embeddings)),
/// and the layer indices to inject at.
pub struct EngramContext<'a> {
    /// The gated residual module (holds projection + gating weights).
    pub module: &'a EngramGatedResidual,
    /// Pre-resolved Engram embeddings. Shape: `[1, seq_len, engram_dim]`.
    pub embeddings: Tensor,
    /// Which transformer layers to inject at (e.g., `&[2, 14]`).
    pub injection_layers: &'a [usize],
}
```

Then, add the `forward_with_engram` method on `QwenEngine` as an inherent method. Add it after the `InferenceEngine` impl block ends (after the closing `}` of the impl block), but before the `#[cfg(test)]` block:

```rust
impl QwenEngine {
    /// Run inference with Engram conditional memory injection.
    ///
    /// Like [`forward`](InferenceEngine::forward), but injects Engram embeddings
    /// at the specified transformer layers via the gated residual module.
    /// This is an inherent method, not a trait method — Engram injection is
    /// optional and engine-specific.
    ///
    /// # Arguments
    ///
    /// - `tokens`: Input token IDs (same as `forward`)
    /// - `cache`: Externalized KV cache (same as `forward`)
    /// - `engram`: Pre-resolved Engram context (module + embeddings + layer indices)
    ///
    /// # Errors
    ///
    /// Same as `forward()` plus `ForwardFailed` if the Engram module errors.
    pub fn forward_with_engram(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
        engram: &EngramContext<'_>,
    ) -> Result<Vec<f32>, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::ForwardFailed(
                "tokens slice must not be empty".into(),
            ));
        }
        #[cfg(feature = "kv-compress")]
        if cache.is_compressed() {
            return Err(InferenceError::CacheCompressed);
        }
        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;

        if cache.num_layers != model.num_layers {
            return Err(InferenceError::CacheMismatch {
                expected: model.num_layers,
                actual: cache.num_layers,
            });
        }

        let seq_len = tokens.len();
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Build the Engram callback from the context.
        let engram_fn = |layer_idx: usize, hidden_state: &Tensor| -> candle_core::Result<Option<Tensor>> {
            if engram.injection_layers.contains(&layer_idx) {
                Ok(Some(engram.module.forward(hidden_state, &engram.embeddings)?))
            } else {
                Ok(None)
            }
        };

        let logits = model
            .forward_with_engram(&input, cache, Some(&engram_fn))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Extract last logit row (identical to forward()).
        let logits = match logits.dims().len() {
            1 => logits,
            2 => {
                let rows = logits
                    .dim(0)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
                if rows == 0 {
                    return Err(InferenceError::ForwardFailed(
                        "model returned empty logits tensor [0, vocab_size]".into(),
                    ));
                }
                logits
                    .get(rows - 1)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?
            }
            n => {
                return Err(InferenceError::ForwardFailed(format!(
                    "unexpected logits dimensionality: {n}D"
                )))
            }
        };

        logits
            .to_vec1::<f32>()
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))
    }
}
```

- [ ] **Step 2: Add re-export in lib.rs**

In `crates/harmony-inference/src/lib.rs`, add after the existing re-exports (after `pub use error::InferenceError;`):

```rust
pub use engine::EngramContext;
```

- [ ] **Step 3: Add unit test for forward_with_engram without model**

In the `#[cfg(test)] mod tests` block of `engine.rs`, add:

```rust
    #[test]
    fn forward_with_engram_without_model_returns_error() {
        let device = Device::Cpu;
        let engine = QwenEngine::new(device.clone());
        let mut cache = InferenceCache::new(28, 128, 8);
        let module = crate::engram_residual::EngramGatedResidual::new(4, 64, 3, &device).unwrap();
        let embeddings = Tensor::zeros((1, 3, 4), candle_core::DType::F32, &device).unwrap();
        let ctx = EngramContext {
            module: &module,
            embeddings,
            injection_layers: &[2, 14],
        };
        let result = engine.forward_with_engram(&[1, 2, 3], &mut cache, &ctx);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }
```

- [ ] **Step 4: Verify all tests pass**

Run: `cargo test -p harmony-inference`

Expected: All tests pass including the new one.

- [ ] **Step 5: Verify workspace compiles**

Run: `cargo check --workspace`

Expected: Clean compilation.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/engine.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add EngramContext and forward_with_engram on QwenEngine"
```

---

### Task 3: Verify and Clean Up

Final verification, clippy, and format check.

**Files:**
- All modified files

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-inference`

Fix any warnings in the new code.

- [ ] **Step 2: Run format check**

Run: `cargo fmt -p harmony-inference -- --check`

Fix any formatting issues.

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace --exclude harmony-tunnel`

All tests must pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add crates/harmony-inference/
git commit -m "chore: clippy and fmt fixes for Engram forward pass integration"
```

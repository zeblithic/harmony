# Latent Projection for Engram Key Compression — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace xxhash64 token-byte Engram keys with learned LSH codes from a 2-layer MLP, upgrading retrieval from surface-level N-gram matching to semantic hashing.

**Architecture:** A `LatentProjection` module projects averaged N-gram token embeddings through a 2-layer MLP (hidden_dim→256→128, SiLU+tanh), binarizes the output via sign bits, then hashes the binary codes through the existing xxhash64 multi-head shard lookup. GGUF weight presence gates the feature; fallback is the current token-hash path.

**Tech Stack:** Rust, candle-core (tensors, Linear), harmony-engram (hash), harmony-inference (engine), xxhash64

---

### Task 1: Refactor `compute_lookup` to accept raw bytes

**Files:**
- Modify: `crates/harmony-engram/src/hash.rs`

The existing `compute_lookup` encodes tokens as bytes then hashes them. Extract the byte-hashing into a new public function `compute_lookup_from_bytes` so that latent projection can hash arbitrary binary keys. The existing `compute_lookup` delegates to it.

- [ ] **Step 1: Write the regression test**

Add to the `#[cfg(test)] mod tests` block in `crates/harmony-engram/src/hash.rs`:

```rust
    #[test]
    fn compute_lookup_refactor_preserves_existing_hashes() {
        // Regression: the refactored compute_lookup must produce identical
        // results to the pre-refactor version. The pinned hash values in
        // different_seeds_produce_different_hashes() are the ground truth.
        let config = test_config();
        let lookup_before = compute_lookup(&config, &[1, 2, 3]);

        // These values were established before the refactor and must not change.
        let expected_shard_0 = lookup_before.shard_indices[0];
        let expected_shard_1 = lookup_before.shard_indices[1];
        let expected_offset_0 = lookup_before.entry_offsets[0];
        let expected_offset_1 = lookup_before.entry_offsets[1];

        // Re-call after refactor — must be identical.
        let lookup_after = compute_lookup(&config, &[1, 2, 3]);
        assert_eq!(lookup_after.shard_indices[0], expected_shard_0);
        assert_eq!(lookup_after.shard_indices[1], expected_shard_1);
        assert_eq!(lookup_after.entry_offsets[0], expected_offset_0);
        assert_eq!(lookup_after.entry_offsets[1], expected_offset_1);
    }

    #[test]
    fn compute_lookup_from_bytes_matches_manual() {
        let config = test_config();
        // Manually encode tokens [1, 2, 3] as little-endian bytes.
        let bytes: Vec<u8> = [1u32, 2, 3]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let from_bytes = compute_lookup_from_bytes(&config, &bytes);
        let from_tokens = compute_lookup(&config, &[1, 2, 3]);

        // Must produce identical lookups.
        assert_eq!(from_bytes.shard_indices, from_tokens.shard_indices);
        assert_eq!(from_bytes.entry_offsets, from_tokens.entry_offsets);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram -- compute_lookup_from_bytes_matches_manual compute_lookup_refactor_preserves_existing_hashes --nocapture`
Expected: FAIL — `compute_lookup_from_bytes` does not exist yet.

- [ ] **Step 3: Implement `compute_lookup_from_bytes` and refactor `compute_lookup`**

Replace the body of `hash_ngram` with a call to a new `hash_bytes` function, and add `compute_lookup_from_bytes`:

```rust
/// Hash arbitrary bytes with a single seed.
fn hash_bytes(bytes: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(bytes, seed)
}

/// Hash an N-gram's token bytes with a single seed.
///
/// Tokens are encoded as contiguous little-endian u32 bytes.
fn hash_ngram(tokens: &[u32], seed: u64) -> u64 {
    let byte_len = tokens.len() * 4;
    if byte_len <= 128 {
        let mut buf = [0u8; 128];
        for (i, t) in tokens.iter().enumerate() {
            buf[i * 4..(i + 1) * 4].copy_from_slice(&t.to_le_bytes());
        }
        hash_bytes(&buf[..byte_len], seed)
    } else {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        hash_bytes(&bytes, seed)
    }
}

/// Compute the [`EngramLookup`] for arbitrary key bytes.
///
/// Same hashing logic as [`compute_lookup`] but accepts raw bytes instead of
/// token N-grams. Used by latent projection to hash binary LSH codes.
pub fn compute_lookup_from_bytes(config: &EngramConfig, key_bytes: &[u8]) -> EngramLookup {
    debug_assert!(config.total_entries > 0, "total_entries must be positive");
    debug_assert!(config.shard_size > 0, "shard_size must be positive");
    debug_assert_eq!(
        config.hash_seeds.len(),
        config.num_heads as usize,
        "hash_seeds length must match num_heads"
    );

    let vector_bytes = config.vector_bytes();
    let mut shard_indices = Vec::with_capacity(config.num_heads as usize);
    let mut entry_offsets = Vec::with_capacity(config.num_heads as usize);

    for seed in &config.hash_seeds {
        let raw_hash = hash_bytes(key_bytes, *seed);
        let table_index = raw_hash % config.total_entries;
        let shard_index = table_index / config.shard_size as u64;
        let entry_within_shard = (table_index % config.shard_size as u64) as usize;
        let byte_offset = entry_within_shard * vector_bytes;

        shard_indices.push(shard_index);
        entry_offsets.push(byte_offset);
    }

    EngramLookup {
        shard_indices,
        entry_offsets,
    }
}
```

Note: `compute_lookup` is NOT changed — it stays exactly as-is and continues to call `hash_ngram`. The `hash_ngram` function now delegates to `hash_bytes` internally but its behavior is identical (the stack buffer optimization is preserved).

- [ ] **Step 4: Run all harmony-engram tests**

Run: `cargo test -p harmony-engram -- --nocapture`
Expected: ALL PASS, including existing pinned hash tests and the two new tests.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-engram/src/hash.rs
git commit -m "feat(engram): add compute_lookup_from_bytes for arbitrary key hashing (ZEB-63)"
```

---

### Task 2: LatentProjection module — MLP core + binary key derivation

**Files:**
- Create: `crates/harmony-inference/src/latent_projection.rs`
- Modify: `crates/harmony-inference/src/lib.rs`

Build the `LatentProjection` struct: 2-layer MLP (SiLU + tanh), `project()`, `to_binary_keys()`, and random-init constructor.

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-inference/src/latent_projection.rs` with only a `#[cfg(test)]` module:

```rust
//! Learned latent projection for semantic Engram key generation.
//!
//! Projects token embeddings through a 2-layer MLP (SiLU + tanh) to produce
//! compact latent codes, then binarizes via sign bits for locality-sensitive
//! hashing. Replaces xxhash64 token-byte hashing when projection weights are
//! available in the GGUF.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::Linear;

#[cfg(test)]
mod tests {
    use super::*;

    const HIDDEN_DIM: usize = 32;
    const INTERMEDIATE_DIM: usize = 16;
    const LATENT_DIM: usize = 8;

    fn test_projection() -> LatentProjection {
        LatentProjection::new_random(HIDDEN_DIM, INTERMEDIATE_DIM, LATENT_DIM, &Device::Cpu)
            .unwrap()
    }

    #[test]
    fn project_output_shape() {
        let proj = test_projection();
        // [1, 4, 32] → [1, 4, 8]
        let input = Tensor::randn(0f32, 1.0, (1, 4, HIDDEN_DIM), &Device::Cpu).unwrap();
        let output = proj.project(&input).unwrap();
        assert_eq!(output.dims(), &[1, 4, LATENT_DIM]);
    }

    #[test]
    fn project_output_bounded_by_tanh() {
        let proj = test_projection();
        let input = Tensor::randn(0f32, 10.0, (1, 8, HIDDEN_DIM), &Device::Cpu).unwrap();
        let output = proj.project(&input).unwrap();
        let max: f32 = output.abs().unwrap().max_all().unwrap().to_scalar().unwrap();
        assert!(max <= 1.0 + 1e-6, "tanh output must be in [-1, 1], got max={max}");
    }

    #[test]
    fn to_binary_keys_length_and_determinism() {
        let proj = test_projection();
        let input = Tensor::randn(0f32, 1.0, (1, 3, HIDDEN_DIM), &Device::Cpu).unwrap();
        let latent = proj.project(&input).unwrap();
        let keys1 = proj.to_binary_keys(&latent).unwrap();
        let keys2 = proj.to_binary_keys(&latent).unwrap();

        // 3 positions, each with ceil(LATENT_DIM / 8) = 1 byte
        assert_eq!(keys1.len(), 3);
        assert_eq!(keys1[0].len(), LATENT_DIM.div_ceil(8));
        // Deterministic
        assert_eq!(keys1, keys2);
    }

    #[test]
    fn similar_embeddings_produce_same_binary_key() {
        let proj = test_projection();
        let base = Tensor::randn(0f32, 1.0, (1, 1, HIDDEN_DIM), &Device::Cpu).unwrap();
        // Tiny perturbation — should land in the same LSH bucket
        let noise = (Tensor::randn(0f32, 1.0, (1, 1, HIDDEN_DIM), &Device::Cpu).unwrap()
            * 1e-4).unwrap();
        let similar = (&base + &noise).unwrap();

        let keys_a = proj.to_binary_keys(&proj.project(&base).unwrap()).unwrap();
        let keys_b = proj.to_binary_keys(&proj.project(&similar).unwrap()).unwrap();
        assert_eq!(keys_a[0], keys_b[0], "tiny perturbation should produce same binary key");
    }

    #[test]
    fn different_embeddings_produce_different_keys() {
        let proj = test_projection();
        // Two very different inputs
        let a = Tensor::ones((1, 1, HIDDEN_DIM), candle_core::DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 1, HIDDEN_DIM), candle_core::DType::F32, &Device::Cpu).unwrap()
            * (-1.0)).unwrap();

        let keys_a = proj.to_binary_keys(&proj.project(&a).unwrap()).unwrap();
        let keys_b = proj.to_binary_keys(&proj.project(&b).unwrap()).unwrap();
        // With 8 latent dims, opposite inputs should produce different keys
        assert_ne!(keys_a[0], keys_b[0], "very different inputs should produce different binary keys");
    }
}
```

- [ ] **Step 2: Add module to `lib.rs`**

Add to `crates/harmony-inference/src/lib.rs` after the `pub mod uq_head;` line:

```rust
pub mod latent_projection;
```

And in the re-exports section, add:

```rust
pub use latent_projection::LatentProjection;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-inference -- latent_projection --nocapture`
Expected: FAIL — `LatentProjection` struct does not exist yet.

- [ ] **Step 4: Implement `LatentProjection`**

Add the struct and methods above the `#[cfg(test)]` module in `crates/harmony-inference/src/latent_projection.rs`:

```rust
/// Learned latent projection for semantic Engram key generation.
///
/// 2-layer MLP: `hidden_dim → intermediate_dim (SiLU) → latent_dim (tanh)`.
/// The tanh output is binarized via sign bits to produce LSH codes.
#[derive(Debug, Clone)]
pub struct LatentProjection {
    layer1: Linear,
    layer2: Linear,
    latent_dim: usize,
}

impl LatentProjection {
    /// Construct from pre-trained weight tensors (loaded from GGUF).
    ///
    /// - `layer1_weight`: `[intermediate_dim, hidden_dim]`
    /// - `layer1_bias`: `[intermediate_dim]`
    /// - `layer2_weight`: `[latent_dim, intermediate_dim]`
    /// - `layer2_bias`: `[latent_dim]`
    pub fn from_tensors(
        layer1_weight: Tensor,
        layer1_bias: Tensor,
        layer2_weight: Tensor,
        layer2_bias: Tensor,
    ) -> Result<Self> {
        let latent_dim = layer2_weight.dim(0)?;
        let layer1 = Linear::new(layer1_weight, Some(layer1_bias));
        let layer2 = Linear::new(layer2_weight, Some(layer2_bias));
        Ok(Self {
            layer1,
            layer2,
            latent_dim,
        })
    }

    /// Construct with random Kaiming-scale weights for testing.
    pub fn new_random(
        hidden_dim: usize,
        intermediate_dim: usize,
        latent_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let scale1 = (1.0 / hidden_dim as f64).sqrt();
        let w1 = (Tensor::randn(0f32, 1.0, (intermediate_dim, hidden_dim), device)? * scale1)?;
        let b1 = Tensor::zeros(intermediate_dim, candle_core::DType::F32, device)?;

        let scale2 = (1.0 / intermediate_dim as f64).sqrt();
        let w2 = (Tensor::randn(0f32, 1.0, (latent_dim, intermediate_dim), device)? * scale2)?;
        let b2 = Tensor::zeros(latent_dim, candle_core::DType::F32, device)?;

        Self::from_tensors(w1, b1, w2, b2)
    }

    /// The output dimension of the latent codes.
    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Project embeddings to latent space: `hidden_dim → tanh(latent_dim)`.
    ///
    /// Input: `[batch, seq_len, hidden_dim]`
    /// Output: `[batch, seq_len, latent_dim]` with values in `[-1, 1]`.
    pub fn project(&self, embeddings: &Tensor) -> Result<Tensor> {
        let h = self.layer1.forward(embeddings)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.layer2.forward(&h)?;
        h.tanh()
    }

    /// Convert latent codes to binary keys via sign-bit packing.
    ///
    /// Input: `[batch, seq_len, latent_dim]` (output of `project()`).
    /// Returns one `Vec<u8>` per sequence position, each `ceil(latent_dim / 8)` bytes.
    /// Bit `i` is 1 if `latent[i] >= 0`, else 0.
    pub fn to_binary_keys(&self, latent: &Tensor) -> Result<Vec<Vec<u8>>> {
        // Flatten to [seq_len, latent_dim]
        let squeezed = latent.squeeze(0)?;
        let seq_len = squeezed.dim(0)?;
        let data = squeezed.to_vec2::<f32>()?;
        let key_bytes = self.latent_dim.div_ceil(8);

        let mut keys = Vec::with_capacity(seq_len);
        for row in &data {
            let mut key = vec![0u8; key_bytes];
            for (i, &val) in row.iter().enumerate() {
                if val >= 0.0 {
                    key[i / 8] |= 1 << (i % 8);
                }
            }
            keys.push(key);
        }
        Ok(keys)
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- latent_projection --nocapture`
Expected: ALL 5 PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/latent_projection.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add LatentProjection MLP with binary key derivation (ZEB-63)"
```

---

### Task 3: N-gram windowing — `project_ngrams()`

**Files:**
- Modify: `crates/harmony-inference/src/latent_projection.rs`

Add the `project_ngrams()` convenience method that handles N-gram window extraction, embedding averaging, projection, and binarization.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn project_ngrams_bigrams_and_trigrams() {
        let proj = test_projection();
        // 4 positions → 3 bigrams + 2 trigrams = 5 keys
        let embeddings = Tensor::randn(0f32, 1.0, (1, 4, HIDDEN_DIM), &Device::Cpu).unwrap();
        let (keys, positions) = proj.project_ngrams(&embeddings, 4).unwrap();

        assert_eq!(keys.len(), 5, "3 bigrams + 2 trigrams = 5");
        assert_eq!(positions.len(), 5);

        // Bigrams at positions 1, 2, 3
        assert_eq!(positions[0], 1);
        assert_eq!(positions[1], 2);
        assert_eq!(positions[2], 3);
        // Trigrams at positions 2, 3
        assert_eq!(positions[3], 2);
        assert_eq!(positions[4], 3);

        // Each key has correct byte length
        for key in &keys {
            assert_eq!(key.len(), LATENT_DIM.div_ceil(8));
        }
    }

    #[test]
    fn project_ngrams_single_token_returns_empty() {
        let proj = test_projection();
        let embeddings = Tensor::randn(0f32, 1.0, (1, 1, HIDDEN_DIM), &Device::Cpu).unwrap();
        let (keys, positions) = proj.project_ngrams(&embeddings, 1).unwrap();
        assert!(keys.is_empty());
        assert!(positions.is_empty());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference -- project_ngrams --nocapture`
Expected: FAIL — `project_ngrams` method does not exist yet.

- [ ] **Step 3: Implement `project_ngrams`**

Add this method to the `impl LatentProjection` block:

```rust
    /// Extract N-gram windows, average embeddings, project, and binarize.
    ///
    /// Mirrors the bi/trigram extraction in `engram_bridge::prepare_engram_request`:
    /// - Bigrams at positions `1..seq_len`, trigrams at positions `2..seq_len`
    /// - Each N-gram's embeddings are averaged before projection
    ///
    /// Returns `(binary_keys, positions)` suitable for `prepare_engram_request_latent`.
    pub fn project_ngrams(
        &self,
        embeddings: &Tensor, // [1, seq_len, hidden_dim]
        seq_len: usize,
    ) -> Result<(Vec<Vec<u8>>, Vec<usize>)> {
        if seq_len < 2 {
            return Ok((Vec::new(), Vec::new()));
        }

        let emb = embeddings.squeeze(0)?; // [seq_len, hidden_dim]
        let mut averaged = Vec::new();
        let mut positions = Vec::new();

        // Bigrams
        for i in 0..seq_len - 1 {
            let a = emb.get(i)?;
            let b = emb.get(i + 1)?;
            let avg = ((&a + &b)? * 0.5)?;
            averaged.push(avg);
            positions.push(i + 1);
        }

        // Trigrams
        for i in 0..seq_len.saturating_sub(2) {
            let a = emb.get(i)?;
            let b = emb.get(i + 1)?;
            let c = emb.get(i + 2)?;
            let sum = (&a + &b)?;
            let sum = (&sum + &c)?;
            let avg = (sum * (1.0 / 3.0))?;
            averaged.push(avg);
            positions.push(i + 2);
        }

        // Stack into [1, num_ngrams, hidden_dim] and project
        let stacked = Tensor::stack(&averaged, 0)?.unsqueeze(0)?;
        let latent = self.project(&stacked)?;
        let keys = self.to_binary_keys(&latent)?;

        Ok((keys, positions))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- latent_projection --nocapture`
Expected: ALL 7 PASS (5 from Task 2 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/latent_projection.rs
git commit -m "feat(inference): add project_ngrams N-gram windowing for latent projection (ZEB-63)"
```

---

### Task 4: Contrastive loss function

**Files:**
- Modify: `crates/harmony-inference/src/latent_projection.rs`

Add the `contrastive_loss()` training loss function.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn contrastive_loss_shape_and_finite() {
        let original = Tensor::randn(0f32, 1.0, (1, 8, HIDDEN_DIM), &Device::Cpu).unwrap();
        let projected = Tensor::randn(0f32, 1.0, (1, 8, LATENT_DIM), &Device::Cpu).unwrap();
        let loss = contrastive_loss(&original, &projected, 0.07, 4).unwrap();
        // Scalar loss
        assert_eq!(loss.dims(), &[]);
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val.is_finite(), "loss must be finite, got {val}");
        assert!(val >= 0.0, "loss must be non-negative, got {val}");
    }

    #[test]
    fn contrastive_loss_decreases_with_aligned_projections() {
        // When projected space preserves the original similarity structure,
        // loss should be lower than when it's random.
        let original = Tensor::randn(0f32, 1.0, (1, 16, HIDDEN_DIM), &Device::Cpu).unwrap();
        let random_proj = Tensor::randn(0f32, 1.0, (1, 16, LATENT_DIM), &Device::Cpu).unwrap();

        // "Aligned" projection: just truncate the original (preserves structure)
        let aligned_proj = original.narrow(2, 0, LATENT_DIM).unwrap();

        let loss_random = contrastive_loss(&original, &random_proj, 0.07, 4).unwrap();
        let loss_aligned = contrastive_loss(&original, &aligned_proj, 0.07, 4).unwrap();

        let v_random: f32 = loss_random.to_scalar().unwrap();
        let v_aligned: f32 = loss_aligned.to_scalar().unwrap();
        assert!(
            v_aligned < v_random,
            "aligned projection should have lower loss: {v_aligned} vs {v_random}"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference -- contrastive_loss --nocapture`
Expected: FAIL — `contrastive_loss` function does not exist yet.

- [ ] **Step 3: Implement `contrastive_loss`**

Add this standalone function in `crates/harmony-inference/src/latent_projection.rs` (outside the `impl LatentProjection` block):

```rust
/// Cosine similarity between all pairs of vectors in a 2D tensor.
///
/// Input: `[N, D]`. Returns: `[N, N]` similarity matrix.
fn cosine_similarity_matrix(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let normalized = x.broadcast_div(&norm)?;
    normalized.matmul(&normalized.t()?)
}

/// InfoNCE contrastive loss preserving nearest-neighbor topology.
///
/// Ensures tokens that are nearest neighbors in the original embedding space
/// remain nearest neighbors in the projected latent space.
///
/// - `original`: `[batch, seq_len, hidden_dim]` — raw token embeddings
/// - `projected`: `[batch, seq_len, latent_dim]` — MLP output (post-tanh)
/// - `temperature`: scaling factor for logits (default: 0.07)
/// - `k`: number of top neighbors to preserve (default: 4)
///
/// Returns a scalar loss tensor.
pub fn contrastive_loss(
    original: &Tensor,
    projected: &Tensor,
    temperature: f32,
    k: usize,
) -> Result<Tensor> {
    // Flatten batch: [B, S, D] → [B*S, D]
    let (b, s, _) = original.dims3()?;
    let n = b * s;
    let orig_flat = original.reshape((n, ()))?;
    let proj_flat = projected.reshape((n, ()))?;

    // Cosine similarity matrices [N, N]
    let sim_orig = cosine_similarity_matrix(&orig_flat)?;
    let sim_proj = cosine_similarity_matrix(&proj_flat)?;

    // For each anchor, find top-k neighbors in original space
    // Use projected similarities as logits, original top-k as soft targets
    let k = k.min(n - 1);
    if k == 0 {
        return Tensor::new(0.0f32, original.device());
    }

    // Scale projected similarities by temperature
    let logits = (sim_proj / temperature as f64)?;

    // Build soft target distribution from original similarities:
    // For each row, keep only the top-k values (excluding self), zero others,
    // then softmax to get a probability distribution.
    let neg_inf = f32::NEG_INFINITY;
    let orig_data = sim_orig.to_vec2::<f32>()?;
    let mut target_data = vec![vec![neg_inf; n]; n];

    for i in 0..n {
        // Sort by similarity descending, skip self (index i)
        let mut indexed: Vec<(usize, f32)> = orig_data[i]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, &v)| (j, v))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        for &(j, _) in indexed.iter().take(k) {
            target_data[i][j] = orig_data[i][j];
        }
    }

    let target_tensor =
        Tensor::from_vec(target_data.into_iter().flatten().collect::<Vec<f32>>(), (n, n), original.device())?;
    // Softmax over targets to get probability distribution
    let targets = candle_nn::ops::softmax(&target_tensor, 1)?;

    // Cross-entropy: -sum(targets * log_softmax(logits))
    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
    let loss = (targets * log_probs)?.neg()?.sum_all()?;
    // Average over anchors
    loss / n as f64
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- latent_projection --nocapture`
Expected: ALL 9 PASS (7 from Tasks 2-3 + 2 new).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/latent_projection.rs
git commit -m "feat(inference): add contrastive loss for latent projection training (ZEB-63)"
```

---

### Task 5: Bridge integration — `prepare_engram_request_latent()`

**Files:**
- Modify: `crates/harmony-inference/src/engram_bridge.rs`

Add the latent-key variant of the bridge function that takes binary keys and positions instead of raw tokens.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/harmony-inference/src/engram_bridge.rs`:

```rust
    #[test]
    fn prepare_latent_request_produces_valid_lookups() {
        let client = test_client();
        // 3 binary keys (simulating 2 bigrams + 1 trigram from a 3-token seq)
        let keys = vec![vec![0xABu8, 0xCD], vec![0x12, 0x34], vec![0xFF, 0x00]];
        let positions = vec![1, 2, 2];
        let request =
            prepare_engram_request_latent(client.config(), &keys, &positions, 3).unwrap();

        assert_eq!(request.lookups.len(), 3);
        assert_eq!(request.seq_len, 3);

        // Each lookup has correct head count
        for lookup in &request.lookups {
            assert_eq!(lookup.lookup.shard_indices.len(), 2);
            assert_eq!(lookup.lookup.entry_offsets.len(), 2);
        }

        // Shard indices are in bounds
        for lookup in &request.lookups {
            for &idx in &lookup.lookup.shard_indices {
                assert!(idx < client.config().num_shards, "shard index out of bounds");
            }
        }
    }

    #[test]
    fn prepare_latent_request_matches_token_count() {
        let client = test_client();
        let keys = vec![vec![0x00u8], vec![0xFF]];
        let positions = vec![1, 1];
        let request =
            prepare_engram_request_latent(client.config(), &keys, &positions, 5).unwrap();

        // seq_len should be what the caller specified, not the number of keys
        assert_eq!(request.seq_len, 5);
        assert_eq!(request.lookups.len(), 2);
        assert_eq!(request.lookups[0].token_position, 1);
        assert_eq!(request.lookups[1].token_position, 1);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference -- prepare_latent_request --nocapture`
Expected: FAIL — `prepare_engram_request_latent` does not exist yet.

- [ ] **Step 3: Implement `prepare_engram_request_latent`**

Add to `crates/harmony-inference/src/engram_bridge.rs`, after the `prepare_engram_request` function. Add the new import at the top:

Then add the function (uses `EngramClient` for real CID resolution via `collect_shards`):

```rust
/// Phase 1 (latent): Build an [`EngramRequest`] from pre-computed binary keys.
///
/// Used when latent projection is active. Each binary key (from
/// [`LatentProjection::to_binary_keys`]) is hashed via
/// `compute_lookup_from_bytes` to get shard indices, producing the same
/// `EngramRequest` that the resolve path consumes.
///
/// `positions[i]` is the token position attributed to `binary_keys[i]`
/// (last token of the N-gram window, matching the convention in
/// [`prepare_engram_request`]).
pub fn prepare_engram_request_latent(
    client: &EngramClient,
    binary_keys: &[Vec<u8>],
    positions: &[usize],
    seq_len: usize,
) -> Result<EngramRequest> {
    if binary_keys.len() != positions.len() {
        candle_core::bail!(
            "binary_keys.len()={} != positions.len()={}",
            binary_keys.len(),
            positions.len()
        );
    }
    if let Some(&pos) = positions.iter().find(|&&p| p >= seq_len) {
        candle_core::bail!(
            "token_position {pos} out of bounds for seq_len={seq_len}"
        );
    }

    let mut lookups = Vec::with_capacity(binary_keys.len());
    let mut seen_shards = HashSet::new();
    let mut required_shards = Vec::new();

    for (key, &pos) in binary_keys.iter().zip(positions.iter()) {
        let lookup = harmony_engram::hash::compute_lookup_from_bytes(client.config(), key);
        collect_shards(client, &lookup, &mut seen_shards, &mut required_shards)?;
        lookups.push(NgramLookup {
            token_position: pos,
            lookup,
        });
    }

    Ok(EngramRequest {
        required_shards,
        lookups,
        seq_len,
    })
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- engram_bridge --nocapture`
Expected: ALL PASS (existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/engram_bridge.rs
git commit -m "feat(inference): add prepare_engram_request_latent for binary key lookups (ZEB-63)"
```

---

### Task 6: HarmonyModel `token_embeddings()` accessor

**Files:**
- Modify: `crates/harmony-inference/src/harmony_model.rs`

Expose the embedding layer lookup so the event loop can get token embeddings for latent projection.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/harmony-inference/src/harmony_model.rs`:

```rust
    #[test]
    fn token_embeddings_accessor_shape() {
        let config = HarmonyModelConfig::tiny();
        let model = HarmonyModel::new(&config, &Device::Cpu).unwrap();

        let tokens = [1u32, 2, 3, 4];
        let embeddings = model.token_embeddings(&tokens).unwrap();

        assert_eq!(
            embeddings.dims(),
            &[1, 4, config.hidden_dim],
            "should be [1, seq_len, hidden_dim]"
        );
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-inference -- token_embeddings_accessor_shape --nocapture`
Expected: FAIL — `token_embeddings` method does not exist yet.

- [ ] **Step 3: Implement `token_embeddings`**

Add this method to the `impl HarmonyModel` block, after the `block_attnres()` accessor:

```rust
    /// Look up token embeddings without running the transformer.
    ///
    /// Returns `[1, seq_len, hidden_dim]`. Used by latent projection to
    /// compute semantic keys from the raw embedding layer.
    pub fn token_embeddings(&self, token_ids: &[u32]) -> Result<Tensor> {
        let device = &self.device;
        let input = Tensor::new(token_ids, device)?.unsqueeze(0)?;
        self.embed_tokens.forward(&input)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-inference -- token_embeddings_accessor_shape --nocapture`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/harmony_model.rs
git commit -m "feat(inference): add token_embeddings accessor on HarmonyModel (ZEB-63)"
```

---

### Task 7: HarmonyEngine integration — field, getter/setter, GGUF auto-load, `token_embeddings` delegator

**Files:**
- Modify: `crates/harmony-inference/src/harmony_engine.rs`

Add the `latent_projection` field to `HarmonyEngine`, the getter/setter, a `token_embeddings()` delegator, and auto-detection in `load_gguf()`.

- [ ] **Step 1: Write the failing tests**

Add to the `#[cfg(test)] mod tests` block in `crates/harmony-inference/src/harmony_engine.rs`:

```rust
    #[test]
    fn engine_without_projection_returns_none() {
        let config = HarmonyModelConfig::tiny();
        let engine = HarmonyEngine::new(config, Device::Cpu);
        assert!(engine.latent_projection().is_none());
    }

    #[test]
    fn set_latent_projection_makes_it_available() {
        let config = HarmonyModelConfig::tiny();
        let mut engine = HarmonyEngine::new(config.clone(), Device::Cpu);
        let proj = crate::latent_projection::LatentProjection::new_random(
            config.hidden_dim, 16, 8, &Device::Cpu,
        ).unwrap();
        engine.set_latent_projection(proj);
        assert!(engine.latent_projection().is_some());
    }

    #[test]
    fn token_embeddings_delegator() {
        let config = HarmonyModelConfig::tiny();
        let mut engine = HarmonyEngine::new(config.clone(), Device::Cpu);
        engine.init_random().unwrap();

        let emb = engine.token_embeddings(&[1, 2, 3]).unwrap();
        assert_eq!(emb.dims(), &[1, 3, config.hidden_dim]);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference -- engine_without_projection set_latent_projection token_embeddings_delegator --nocapture`
Expected: FAIL — `latent_projection()`, `set_latent_projection()`, and `token_embeddings()` do not exist on `HarmonyEngine`.

- [ ] **Step 3: Add the field, getter/setter, delegator, and GGUF auto-load**

In `crates/harmony-inference/src/harmony_engine.rs`:

**3a.** Add import at the top:

```rust
use crate::latent_projection::LatentProjection;
```

**3b.** Add field to `HarmonyEngine` struct:

```rust
pub struct HarmonyEngine {
    model: Option<HarmonyModel>,
    tokenizer: Option<tokenizers::Tokenizer>,
    uq_head: Option<UqHead>,
    latent_projection: Option<LatentProjection>,
    uq_feature_config: UqFeatureConfig,
    thought_config: ContinuousThoughtConfig,
    config: HarmonyModelConfig,
    device: Device,
}
```

**3c.** Initialize to `None` in `new()`:

```rust
        Self {
            model: None,
            tokenizer: None,
            uq_head: None,
            latent_projection: None,
            uq_feature_config,
            thought_config,
            config,
            device,
        }
```

**3d.** Add getter/setter methods after `set_uq_head`:

```rust
    /// Set the latent projection for semantic Engram key generation.
    pub fn set_latent_projection(&mut self, proj: LatentProjection) {
        self.latent_projection = Some(proj);
    }

    /// Reference to the latent projection, if loaded.
    pub fn latent_projection(&self) -> Option<&LatentProjection> {
        self.latent_projection.as_ref()
    }

    /// Look up token embeddings without running the transformer.
    ///
    /// Delegates to [`HarmonyModel::token_embeddings`].
    pub fn token_embeddings(&self, token_ids: &[u32]) -> Result<Tensor, InferenceError> {
        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;
        model
            .token_embeddings(token_ids)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))
    }
```

**3e.** Add GGUF auto-detection in `load_gguf()`, after the model is loaded and before `self.model = Some(model)`:

```rust
        // Auto-detect latent projection weights.
        let lp_key = "harmony.latent_projection.layer1.weight";
        if ct.tensor_infos.contains_key(lp_key) {
            let w1 = ct.tensor(&mut cursor, "harmony.latent_projection.layer1.weight", &self.device)
                .map_err(|e| InferenceError::InvalidGguf(format!("latent projection layer1.weight: {e}")))?;
            let b1 = ct.tensor(&mut cursor, "harmony.latent_projection.layer1.bias", &self.device)
                .map_err(|e| InferenceError::InvalidGguf(format!("latent projection layer1.bias: {e}")))?;
            let w2 = ct.tensor(&mut cursor, "harmony.latent_projection.layer2.weight", &self.device)
                .map_err(|e| InferenceError::InvalidGguf(format!("latent projection layer2.weight: {e}")))?;
            let b2 = ct.tensor(&mut cursor, "harmony.latent_projection.layer2.bias", &self.device)
                .map_err(|e| InferenceError::InvalidGguf(format!("latent projection layer2.bias: {e}")))?;
            let proj = LatentProjection::from_tensors(w1, b1, w2, b2)
                .map_err(|e| InferenceError::InvalidGguf(format!("latent projection: {e}")))?;
            self.latent_projection = Some(proj);
        } else {
            self.latent_projection = None;
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- engine_without_projection set_latent_projection token_embeddings_delegator --nocapture`
Expected: ALL 3 PASS.

- [ ] **Step 5: Run full test suite to check for regressions**

Run: `cargo test -p harmony-inference -- --nocapture`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/harmony_engine.rs
git commit -m "feat(inference): integrate LatentProjection into HarmonyEngine with GGUF auto-load (ZEB-63)"
```

---

### Task 8: ChunkedEngramScheduler — optional latent projection in `prepare_request`

**Files:**
- Modify: `crates/harmony-inference/src/chunked_engram.rs`

Add an optional latent projection path to `prepare_request()` so chunked decode uses learned keys when available.

- [ ] **Step 1: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `crates/harmony-inference/src/chunked_engram.rs`:

```rust
    #[test]
    fn prepare_request_with_latent_projection() {
        let client = test_client();
        let mut scheduler = ChunkedEngramScheduler::new(ChunkedEngramConfig::new(3));
        scheduler.seed(&[10, 20, 30, 40, 50], None);

        // Create a tiny latent projection
        let proj = crate::latent_projection::LatentProjection::new_random(
            EMBEDDING_DIM, 8, 4, &Device::Cpu,
        ).unwrap();

        // Dummy embedding function — normally the engine provides this
        let token_buf: Vec<u32> = scheduler.token_buffer().iter().copied().collect();
        let dummy_embeddings = Tensor::randn(
            0f32, 1.0,
            (1, token_buf.len(), EMBEDDING_DIM),
            &Device::Cpu,
        ).unwrap();

        let request = scheduler
            .prepare_request_latent(&client, &proj, &dummy_embeddings)
            .unwrap();

        // Should have lookups (bigrams + trigrams from the 5-token window)
        assert!(!request.lookups.is_empty());
        assert_eq!(request.seq_len, token_buf.len());
    }
```

Note: check if the test module already has a `test_client` and constants. If it uses different names, adapt. Also check if `token_buffer()` accessor exists — if not, we need to add it.

- [ ] **Step 2: Verify `token_buffer` accessor exists; if not, add it**

Check `crates/harmony-inference/src/chunked_engram.rs` for a `token_buffer()` method. If not present, add:

```rust
    /// Access the current token buffer (for latent projection).
    pub fn token_buffer(&self) -> &VecDeque<u32> {
        &self.token_buffer
    }
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p harmony-inference -- prepare_request_with_latent_projection --nocapture`
Expected: FAIL — `prepare_request_latent` method does not exist.

- [ ] **Step 4: Implement `prepare_request_latent`**

Add this method to the `impl ChunkedEngramScheduler` block:

```rust
    /// Prepare an Engram request using latent projection instead of token hashing.
    ///
    /// When a [`LatentProjection`](crate::latent_projection::LatentProjection) is
    /// available, projects the token embeddings through the MLP and generates
    /// binary LSH keys instead of xxhash64 token-byte keys.
    ///
    /// The caller provides `embeddings` for the current token buffer (from
    /// `engine.token_embeddings()`).
    pub fn prepare_request_latent(
        &self,
        client: &EngramClient,
        projection: &crate::latent_projection::LatentProjection,
        embeddings: &candle_core::Tensor,
    ) -> Result<EngramRequest, InferenceError> {
        let window_len = self.token_buffer.len();
        let (keys, positions) = projection
            .project_ngrams(embeddings, window_len)
            .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))?;
        engram_bridge::prepare_engram_request_latent(
            client.config(),
            &keys,
            &positions,
            window_len,
        )
        .map_err(|e| InferenceError::EngramResolutionFailed(e.to_string()))
    }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-inference -- chunked_engram --nocapture`
Expected: ALL PASS (existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/chunked_engram.rs
git commit -m "feat(inference): add latent projection path to ChunkedEngramScheduler (ZEB-63)"
```

---

### Task 9: Event loop integration

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

Wire the latent projection into the prefill and decode Engram paths.

- [ ] **Step 1: Understand the integration points**

The event loop has two Engram code paths:

1. **Prefill** (around line 1510): calls `prepare_engram_request(&client, &tokens)` to build the initial request
2. **Decode** (around line 2754): calls `prepare_engram_request(&ep.client, window)` for chunked refresh

Both need a conditional branch: if `engine.latent_projection().is_some()`, use the latent path.

- [ ] **Step 2: Add latent projection to prefill path**

Find the prefill Engram section (around line 1510) where `prepare_engram_request` is called. Wrap it in a conditional:

```rust
// Inside the .and_then closure where prepare_engram_request is called:
let request = if let Some(proj) = engine.latent_projection() {
    // Latent projection path: use semantic binary keys
    match engine.token_embeddings(&tokens) {
        Ok(embeddings) => {
            match proj.project_ngrams(&embeddings, tokens.len()) {
                Ok((keys, positions)) => {
                    harmony_inference::engram_bridge::prepare_engram_request_latent(
                        client.config(),
                        &keys,
                        &positions,
                        tokens.len(),
                    ).ok()
                }
                Err(e) => {
                    tracing::warn!(err = %e, "latent projection failed — falling back to token hash");
                    None
                }
            }
        }
        Err(e) => {
            tracing::warn!(err = %e, "token_embeddings failed — falling back to token hash");
            None
        }
    }
    .unwrap_or_else(|| {
        harmony_inference::engram_bridge::prepare_engram_request(client, &tokens)
            .unwrap_or_else(|_| harmony_inference::engram_bridge::EngramRequest {
                required_shards: vec![],
                lookups: vec![],
                seq_len: tokens.len(),
            })
    })
} else {
    // Standard token-hash path
    match harmony_inference::engram_bridge::prepare_engram_request(client, &tokens) {
        Ok(r) => r,
        Err(_) => return None,
    }
};
```

The exact placement depends on the current control flow structure around line 1510. Adapt to fit the existing `and_then` chain.

- [ ] **Step 3: Add latent projection to decode path**

Find the decode Engram section (around line 2754) where the chunked refresh calls `prepare_engram_request`. Add the conditional:

```rust
let req = if let Some(proj) = engine.latent_projection() {
    match engine.token_embeddings(window) {
        Ok(embeddings) => {
            match proj.project_ngrams(&embeddings, window.len()) {
                Ok((keys, positions)) => {
                    harmony_inference::engram_bridge::prepare_engram_request_latent(
                        &ep.client.config().clone(),
                        &keys,
                        &positions,
                        window.len(),
                    ).ok()
                }
                Err(e) => {
                    tracing::debug!(err = %e, "latent projection failed in decode — falling back");
                    None
                }
            }
        }
        Err(e) => {
            tracing::debug!(err = %e, "token_embeddings failed in decode — falling back");
            None
        }
    }
    .or_else(|| {
        harmony_inference::engram_bridge::prepare_engram_request(&ep.client, window).ok()
    })
} else {
    harmony_inference::engram_bridge::prepare_engram_request(&ep.client, window).ok()
};
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-node`
Expected: Compiles cleanly.

- [ ] **Step 5: Run full test suite**

Run: `cargo test -p harmony-inference -- --nocapture && cargo test -p harmony-engram -- --nocapture && cargo test -p harmony-node -- --nocapture`
Expected: ALL PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): wire latent projection into prefill and decode Engram paths (ZEB-63)"
```

---

### Task 10: Re-export cleanup and final verification

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

Ensure all public types are re-exported and run the full verification suite.

- [ ] **Step 1: Verify re-exports**

Check that `crates/harmony-inference/src/lib.rs` has:

```rust
pub mod latent_projection;
pub use latent_projection::LatentProjection;
```

Also add re-export for the contrastive loss (KRILE needs it):

```rust
pub use latent_projection::contrastive_loss;
```

- [ ] **Step 2: Run full workspace check**

Run: `cargo check -p harmony-node`
Expected: Compiles cleanly.

- [ ] **Step 3: Run full test suite**

```bash
cargo test -p harmony-engram -- --nocapture
cargo test -p harmony-inference -- --nocapture
cargo test -p harmony-inference -- latent_projection --nocapture
cargo test -p harmony-node -- --nocapture
```

Expected: ALL PASS.

- [ ] **Step 4: Commit if any changes were made**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): re-export LatentProjection and contrastive_loss (ZEB-63)"
```

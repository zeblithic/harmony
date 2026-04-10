//! Learned latent projection for semantic Engram key generation.
//!
//! Projects token embeddings through a 2-layer MLP (SiLU + tanh) to produce
//! compact latent codes, then binarizes via sign bits for locality-sensitive
//! hashing. Replaces xxhash64 token-byte hashing when projection weights are
//! available in the GGUF.

use candle_core::{Device, Module, Result, Tensor};
use candle_nn::Linear;

#[derive(Debug, Clone)]
pub struct LatentProjection {
    layer1: Linear,
    layer2: Linear,
    latent_dim: usize,
}

impl LatentProjection {
    pub fn from_tensors(
        layer1_weight: Tensor,
        layer1_bias: Tensor,
        layer2_weight: Tensor,
        layer2_bias: Tensor,
    ) -> Result<Self> {
        let latent_dim = layer2_weight.dim(0)?;
        let layer1 = Linear::new(layer1_weight, Some(layer1_bias));
        let layer2 = Linear::new(layer2_weight, Some(layer2_bias));
        Ok(Self { layer1, layer2, latent_dim })
    }

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

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn project(&self, embeddings: &Tensor) -> Result<Tensor> {
        let h = self.layer1.forward(embeddings)?;
        let h = candle_nn::ops::silu(&h)?;
        let h = self.layer2.forward(&h)?;
        h.tanh()
    }

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

        // Bigram averages via bulk slicing: avg(emb[0..n-1], emb[1..n])
        let num_bi = seq_len - 1;
        let bi_avg = ((emb.narrow(0, 0, num_bi)? + emb.narrow(0, 1, num_bi)?)? * 0.5)?;

        // Trigram averages: avg(emb[0..n-2], emb[1..n-1], emb[2..n])
        let num_tri = seq_len.saturating_sub(2);
        let parts = if num_tri > 0 {
            let tri_sum = (emb.narrow(0, 0, num_tri)? + emb.narrow(0, 1, num_tri)?)?;
            let tri_sum = (tri_sum + emb.narrow(0, 2, num_tri)?)?;
            let tri_avg = (tri_sum * (1.0 / 3.0))?;
            // Concatenate: [num_bi + num_tri, hidden_dim]
            Tensor::cat(&[&bi_avg, &tri_avg], 0)?
        } else {
            bi_avg
        };

        // Positions: bigrams at 1..seq_len, trigrams at 2..seq_len
        let mut positions = Vec::with_capacity(num_bi + num_tri);
        for i in 1..=num_bi {
            positions.push(i);
        }
        for i in 2..2 + num_tri {
            positions.push(i);
        }

        // Project all N-grams in one batch: [1, num_ngrams, hidden_dim]
        let latent = self.project(&parts.unsqueeze(0)?)?;
        let keys = self.to_binary_keys(&latent)?;

        Ok((keys, positions))
    }

    pub fn to_binary_keys(&self, latent: &Tensor) -> Result<Vec<Vec<u8>>> {
        let squeezed = latent.squeeze(0)?;
        let (seq_len, row_dim) = squeezed.dims2()?;
        if row_dim != self.latent_dim {
            candle_core::bail!(
                "latent width ({row_dim}) must match projection latent_dim ({})",
                self.latent_dim
            );
        }
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

/// Cosine similarity between all pairs of vectors in a 2D tensor.
///
/// Input: `[N, D]`. Returns: `[N, N]` similarity matrix.
fn cosine_similarity_matrix(x: &Tensor) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
    let eps = Tensor::new(1e-8f32, x.device())?.broadcast_as(norm.shape())?;
    let norm = norm.maximum(&eps)?;
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
    let (pb, ps, _) = projected.dims3()?;
    if (pb, ps) != (b, s) {
        candle_core::bail!(
            "projected batch/seq ({pb}, {ps}) must match original ({b}, {s})"
        );
    }
    let n = b * s;

    // Need at least 2 vectors for contrastive loss (anchor + neighbor).
    if n <= 1 {
        return Tensor::new(0.0f32, original.device());
    }

    let orig_flat = original.reshape((n, ()))?;
    let proj_flat = projected.reshape((n, ()))?;

    // Cosine similarity matrices [N, N]
    let sim_orig = cosine_similarity_matrix(&orig_flat)?;
    let sim_proj = cosine_similarity_matrix(&proj_flat)?;

    // For each anchor, find top-k neighbors in original space
    let k = k.min(n - 1);
    if k == 0 {
        return Tensor::new(0.0f32, original.device());
    }

    // Scale projected similarities by temperature, mask diagonal to -inf
    // so self-similarity doesn't dominate the softmax partition.
    let logits = (sim_proj / temperature as f64)?;
    let mut mask_data = vec![0.0f32; n * n];
    for i in 0..n {
        mask_data[i * n + i] = -1e9;
    }
    let diag_mask = Tensor::from_vec(mask_data, (n, n), original.device())?;
    let logits = (logits + diag_mask)?;

    // Build soft target distribution from original similarities:
    // For each row, keep only the top-k values (excluding self), zero others,
    // then softmax to get a probability distribution.
    let neg_inf = f32::NEG_INFINITY;
    let orig_data = sim_orig.to_vec2::<f32>()?;
    let mut target_data = vec![vec![neg_inf; n]; n];

    for i in 0..n {
        let mut indexed: Vec<(usize, f32)> = orig_data[i]
            .iter()
            .enumerate()
            .filter(|&(j, _)| j != i)
            .map(|(j, &v)| (j, v))
            .collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

        for &(j, _) in indexed.iter().take(k) {
            target_data[i][j] = orig_data[i][j];
        }
    }

    let target_tensor =
        Tensor::from_vec(target_data.into_iter().flatten().collect::<Vec<f32>>(), (n, n), original.device())?;
    let targets = candle_nn::ops::softmax(&target_tensor, 1)?;

    // Cross-entropy: -sum(targets * log_softmax(logits))
    let log_probs = candle_nn::ops::log_softmax(&logits, 1)?;
    let loss = (targets * log_probs)?.neg()?.sum_all()?;
    loss / n as f64
}

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
        assert_eq!(keys1.len(), 3);
        assert_eq!(keys1[0].len(), LATENT_DIM.div_ceil(8));
        assert_eq!(keys1, keys2);
    }

    #[test]
    fn similar_embeddings_produce_same_binary_key() {
        let proj = test_projection();
        let base = Tensor::randn(0f32, 1.0, (1, 1, HIDDEN_DIM), &Device::Cpu).unwrap();
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
        let a = Tensor::ones((1, 1, HIDDEN_DIM), candle_core::DType::F32, &Device::Cpu).unwrap();
        let b = (Tensor::ones((1, 1, HIDDEN_DIM), candle_core::DType::F32, &Device::Cpu).unwrap()
            * (-1.0)).unwrap();
        let keys_a = proj.to_binary_keys(&proj.project(&a).unwrap()).unwrap();
        let keys_b = proj.to_binary_keys(&proj.project(&b).unwrap()).unwrap();
        assert_ne!(keys_a[0], keys_b[0], "very different inputs should produce different binary keys");
    }

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

    #[test]
    fn contrastive_loss_shape_and_finite() {
        let original = Tensor::randn(0f32, 1.0, (1, 8, HIDDEN_DIM), &Device::Cpu).unwrap();
        let projected = Tensor::randn(0f32, 1.0, (1, 8, LATENT_DIM), &Device::Cpu).unwrap();
        let loss = contrastive_loss(&original, &projected, 0.07, 4).unwrap();
        assert_eq!(loss.dims(), &[] as &[usize]);
        let val: f32 = loss.to_scalar().unwrap();
        assert!(val.is_finite(), "loss must be finite, got {val}");
        assert!(val >= 0.0, "loss must be non-negative, got {val}");
    }

    #[test]
    fn contrastive_loss_decreases_with_aligned_projections() {
        // Use larger dimensions for statistical stability
        let hidden = 128;
        let latent = 32;
        let original = Tensor::randn(0f32, 1.0, (1, 16, hidden), &Device::Cpu).unwrap();
        let random_proj = Tensor::randn(0f32, 1.0, (1, 16, latent), &Device::Cpu).unwrap();
        let aligned_proj = original.narrow(2, 0, latent).unwrap();

        let loss_random = contrastive_loss(&original, &random_proj, 0.07, 4).unwrap();
        let loss_aligned = contrastive_loss(&original, &aligned_proj, 0.07, 4).unwrap();

        let v_random: f32 = loss_random.to_scalar().unwrap();
        let v_aligned: f32 = loss_aligned.to_scalar().unwrap();
        assert!(
            v_aligned < v_random,
            "aligned projection should have lower loss: {v_aligned} vs {v_random}"
        );
    }
}

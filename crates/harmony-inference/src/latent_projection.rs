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

    pub fn to_binary_keys(&self, latent: &Tensor) -> Result<Vec<Vec<u8>>> {
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
}

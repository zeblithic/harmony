//! TurboQuant KV cache compression.
//!
//! Three-stage pipeline replacing uniform 3-bit quantization:
//! 1. Random orthogonal rotation (smooths channel-wise outliers)
//! 2. PolarQuant: radius + Cartesian Lloyd-Max on unit vector (3-bit, global codebook)
//! 3. QJL: 1-bit sign projection of quantization residual (error correction)
//!
//! Feature-gated behind `kv-compress`.

pub(crate) mod lloyd_max;
pub(crate) mod orthogonal;
pub(crate) mod packing;
pub(crate) mod qjl;

use crate::error::InferenceError;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use lloyd_max::{LloydMaxCodebook, NMSE_3BIT_GAUSSIAN};
use orthogonal::{generate_orthogonal_matrix, matvec, transpose};
use packing::{pack_3bit, unpack_3bit};
use qjl::JlMatrix;

/// Configuration for TurboQuant compression.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Dimension of each KV head vector.
    pub head_dim: usize,
    /// Deterministic seed for random matrix generation.
    pub seed: u64,
}

/// Precomputed state for TurboQuant compression/decompression.
///
/// Initialized once at model load. Immutable after creation. Multiple
/// caches (different conversations) share the same state via `&TurboQuantState`.
pub struct TurboQuantState {
    config: TurboQuantConfig,
    /// Random orthogonal matrix Q, row-major [head_dim × head_dim].
    ortho_matrix: Vec<f32>,
    /// Q^T (inverse rotation), row-major [head_dim × head_dim].
    ortho_transpose: Vec<f32>,
    /// Lloyd-Max codebook scaled for unit vector coordinates.
    codebook: LloydMaxCodebook,
    /// Rademacher JL matrix for sign-bit error correction.
    jl_matrix: JlMatrix,
    /// Precomputed: sqrt(π/2) * sqrt(NMSE).
    correction_scale: f32,
}

impl TurboQuantState {
    /// Initialize TurboQuant state from config.
    ///
    /// Generates the random orthogonal matrix and JL matrix deterministically
    /// from the seed. Uses different seed offsets for the two matrices to
    /// avoid correlation.
    pub fn new(config: &TurboQuantConfig) -> Result<Self, InferenceError> {
        if config.head_dim == 0 {
            return Err(InferenceError::CompressionFailed(
                "head_dim must be > 0".into(),
            ));
        }

        let dim = config.head_dim;
        let ortho_matrix = generate_orthogonal_matrix(dim, config.seed);
        let ortho_transpose = transpose(&ortho_matrix, dim);

        // Unit vector coordinates ~ N(0, 1/dim), so sigma = 1/sqrt(dim)
        let sigma = 1.0 / (dim as f32).sqrt();
        let codebook = LloydMaxCodebook::gaussian(sigma);

        // Use offset seed for JL matrix to avoid correlation
        let jl_matrix = JlMatrix::new(dim, config.seed.wrapping_add(0x517cc1b727220a95));

        // Precompute correction scale: sqrt(π/2) * sqrt(NMSE)
        let expected_relative_error = NMSE_3BIT_GAUSSIAN.sqrt();
        let correction_scale =
            core::f32::consts::FRAC_PI_2.sqrt() * expected_relative_error;

        Ok(Self {
            config: config.clone(),
            ortho_matrix,
            ortho_transpose,
            codebook,
            jl_matrix,
            correction_scale,
        })
    }

    /// Compress a single f32 vector through the TurboQuant pipeline.
    fn compress_vec(&self, data: &[f32]) -> TurboQuantVec {
        let dim = self.config.head_dim;

        // Stage 1: Rotate
        let rotated = matvec(&self.ortho_matrix, data, dim);

        // Stage 2: PolarQuant — radius + Lloyd-Max on unit vector
        let radius: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        if radius < 1e-30 {
            // Zero vector — no quantization needed
            return TurboQuantVec {
                radius: 0.0,
                angles: vec![0u8; (dim * 3).div_ceil(8)],
                signs: vec![0u8; dim.div_ceil(8)],
            };
        }

        let inv_radius = 1.0 / radius;
        let indices: Vec<u8> = rotated
            .iter()
            .map(|&v| self.codebook.quantize(v * inv_radius))
            .collect();
        let angles = pack_3bit(&indices);

        // Reconstruct for residual computation
        let reconstructed: Vec<f32> = indices
            .iter()
            .map(|&idx| self.codebook.dequantize(idx) * radius)
            .collect();

        // Stage 3: QJL — sign-bit encode of residual
        let residual: Vec<f32> = rotated
            .iter()
            .zip(reconstructed.iter())
            .map(|(r, q)| r - q)
            .collect();
        let signs = self.jl_matrix.encode_signs(&residual);

        TurboQuantVec {
            radius,
            angles,
            signs,
        }
    }

    /// Decompress a TurboQuantVec back to an f32 vector.
    fn decompress_vec(&self, qv: &TurboQuantVec) -> Vec<f32> {
        let dim = self.config.head_dim;

        if qv.radius.abs() < 1e-30 {
            return vec![0.0f32; dim];
        }

        // Stage 2 (reverse): Dequantize unit vector, scale by radius
        let indices = unpack_3bit(&qv.angles, dim);
        let mut reconstructed: Vec<f32> = indices
            .iter()
            .map(|&idx| self.codebook.dequantize(idx) * qv.radius)
            .collect();

        // Stage 3 (reverse): QJL correction
        let scale = self.correction_scale * qv.radius;
        let correction = self.jl_matrix.decode_correction(&qv.signs, scale);
        for i in 0..dim {
            reconstructed[i] += correction[i];
        }

        // Stage 1 (reverse): Inverse rotate
        matvec(&self.ortho_transpose, &reconstructed, dim)
    }

    /// Compress an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
    ///
    /// Returns (compressed_vecs, seq_len). One `TurboQuantVec` per (head, token)
    /// pair, stored row-major.
    pub fn compress_tensor(
        &self,
        tensor: &Tensor,
    ) -> Result<(Vec<TurboQuantVec>, usize), InferenceError> {
        let (batch, num_heads, seq_len, head_dim) = tensor
            .dims4()
            .map_err(|e| InferenceError::CompressionFailed(format!("unexpected shape: {e}")))?;
        if batch != 1 {
            return Err(InferenceError::CompressionFailed(format!(
                "expected batch=1, got {batch}"
            )));
        }
        if head_dim != self.config.head_dim {
            return Err(InferenceError::CompressionFailed(format!(
                "head_dim mismatch: state has {}, tensor has {head_dim}",
                self.config.head_dim
            )));
        }

        let total_vecs = num_heads * seq_len;
        let flat = tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.reshape((total_vecs, head_dim)))
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
        let rows = flat
            .to_vec2::<f32>()
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;

        let vecs: Vec<TurboQuantVec> = rows.iter().map(|row| self.compress_vec(row)).collect();
        Ok((vecs, seq_len))
    }

    /// Decompress to an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
    pub fn decompress_tensor(
        &self,
        vecs: &[TurboQuantVec],
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Tensor, InferenceError> {
        if head_dim != self.config.head_dim {
            return Err(InferenceError::CompressionFailed(format!(
                "head_dim mismatch: state has {}, requested {head_dim}",
                self.config.head_dim
            )));
        }
        let expected = num_kv_heads * seq_len;
        if vecs.len() != expected {
            return Err(InferenceError::CompressionFailed(format!(
                "expected {expected} vecs, got {}",
                vecs.len()
            )));
        }

        let mut flat = Vec::with_capacity(expected * head_dim);
        for qv in vecs {
            flat.extend_from_slice(&self.decompress_vec(qv));
        }

        Tensor::from_vec(flat, (1, num_kv_heads, seq_len, head_dim), device)
            .and_then(|t| t.to_dtype(DType::F16))
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))
    }
}

/// A single TurboQuant-compressed vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurboQuantVec {
    /// Vector magnitude (L2 norm of rotated vector).
    pub radius: f32,
    /// 3-bit packed Lloyd-Max indices for unit vector coordinates.
    pub angles: Vec<u8>,
    /// 1-bit packed QJL sign projections.
    pub signs: Vec<u8>,
}

impl TurboQuantVec {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        4 + self.angles.len() + self.signs.len()
    }
}

/// Compressed representation of one layer's K and V tensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    /// Compressed key vectors, row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
    pub k: Vec<TurboQuantVec>,
    /// Compressed value vectors, same layout as k.
    pub v: Vec<TurboQuantVec>,
    /// Sequence length at compression time.
    pub seq_len: usize,
}

impl CompressedKvLayer {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        self.k.iter().map(|q| q.byte_size()).sum::<usize>()
            + self.v.iter().map(|q| q.byte_size()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TurboQuantConfig {
        TurboQuantConfig {
            head_dim: 80,
            seed: 42,
        }
    }

    fn test_state() -> TurboQuantState {
        TurboQuantState::new(&test_config()).unwrap()
    }

    // ── State construction ──

    #[test]
    fn new_state_succeeds() {
        let state = test_state();
        assert_eq!(state.config.head_dim, 80);
    }

    #[test]
    fn new_state_rejects_zero_dim() {
        let cfg = TurboQuantConfig {
            head_dim: 0,
            seed: 42,
        };
        assert!(TurboQuantState::new(&cfg).is_err());
    }

    // ── Per-vector compress/decompress ──

    #[test]
    fn compress_vec_produces_correct_sizes() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        assert_eq!(qv.angles.len(), 30); // ceil(80 * 3 / 8)
        assert_eq!(qv.signs.len(), 10);  // ceil(80 / 8)
        assert!(qv.radius > 0.0);
    }

    #[test]
    fn compress_decompress_vec_bounded_error() {
        let state = test_state();
        let data: Vec<f32> = (0..80).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let data_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        let qv = state.compress_vec(&data);
        let recovered = state.decompress_vec(&qv);

        let err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        let relative_err = err / data_norm;
        assert!(
            relative_err < 0.3,
            "relative error {relative_err:.4} too large (norm={data_norm:.4}, err={err:.4})"
        );
    }

    #[test]
    fn zero_vector_roundtrips() {
        let state = test_state();
        let data = vec![0.0f32; 80];
        let qv = state.compress_vec(&data);
        assert!(qv.radius.abs() < 1e-20);
        let recovered = state.decompress_vec(&qv);
        assert!(recovered.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn byte_size_matches_budget() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        // 4 (f32 radius) + 30 (angles) + 10 (signs) = 44
        assert_eq!(qv.byte_size(), 44);
    }

    // ── Tensor bridge ──

    #[test]
    fn compress_tensor_roundtrip() {
        let state = test_state();
        let shape = (1, 8, 4, 80);
        let tensor = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let (vecs, seq_len) = state.compress_tensor(&tensor).unwrap();
        assert_eq!(seq_len, 4);
        assert_eq!(vecs.len(), 32); // 8 heads * 4 tokens

        let restored = state
            .decompress_tensor(&vecs, 8, 4, 80, &Device::Cpu)
            .unwrap();
        assert_eq!(restored.dims4().unwrap(), (1, 8, 4, 80));
        assert_eq!(restored.dtype(), DType::F16);
    }

    #[test]
    fn compress_tensor_validates_head_dim() {
        let state = test_state();
        let tensor = Tensor::rand(0f32, 1f32, (1, 8, 4, 128), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert!(state.compress_tensor(&tensor).is_err());
    }

    #[test]
    fn compress_tensor_validates_batch_size() {
        let state = test_state();
        let tensor = Tensor::rand(0f32, 1f32, (2, 8, 4, 80), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert!(state.compress_tensor(&tensor).is_err());
    }

    // ── Serialization ──

    #[test]
    fn turboquant_vec_serde_roundtrip() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);

        let bytes = postcard::to_allocvec(&qv).unwrap();
        let restored: TurboQuantVec = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(qv.radius, restored.radius);
        assert_eq!(qv.angles, restored.angles);
        assert_eq!(qv.signs, restored.signs);
    }

    #[test]
    fn compressed_kv_layer_serde_roundtrip() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 4],
            v: vec![qv; 4],
            seq_len: 2,
        };

        let bytes = postcard::to_allocvec(&layer).unwrap();
        let restored: CompressedKvLayer = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(layer.seq_len, restored.seq_len);
        assert_eq!(layer.k.len(), restored.k.len());
        assert_eq!(layer.v.len(), restored.v.len());
    }

    #[test]
    fn compressed_kv_layer_byte_size() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 8],
            v: vec![qv; 8],
            seq_len: 1,
        };
        // 2 tensors * 8 vecs * 44 bytes = 704
        assert_eq!(layer.byte_size(), 704);
    }
}

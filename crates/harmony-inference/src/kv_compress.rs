//! 3-bit uniform KV cache compression.
//!
//! Compresses f16 KV cache tensors to ~56 bytes per 128-dim vector (vs 256
//! bytes at f16), a ~4.6x reduction. Each vector is independently quantized:
//! `min` and `scale` header (8 bytes f32) + 3-bit packed indices.
//!
//! Feature-gated behind `kv-compress`.

use crate::error::InferenceError;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

/// Maximum quantization index (2^3 - 1 = 7, i.e., 8 levels 0..=7).
const MAX_INDEX: u8 = 7;

/// A single compressed vector: per-vector min/scale + 3-bit packed indices.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedVec {
    /// Minimum value of the original vector (dequant base).
    pub min: f32,
    /// Scale factor: `(max - min) / 7.0`. Zero if vector is constant.
    pub scale: f32,
    /// 3-bit packed quantization indices. `ceil(dim * 3 / 8)` bytes.
    pub packed: Vec<u8>,
}

impl QuantizedVec {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        // 4 (min) + 4 (scale) + packed data
        8 + self.packed.len()
    }
}

/// Compressed representation of one layer's K and V tensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    /// Compressed key vectors, row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
    pub k: Vec<QuantizedVec>,
    /// Compressed value vectors, same layout as k.
    pub v: Vec<QuantizedVec>,
    /// Sequence length at compression time (for tensor shape reconstruction).
    pub seq_len: usize,
}

impl CompressedKvLayer {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        self.k.iter().map(|q| q.byte_size()).sum::<usize>()
            + self.v.iter().map(|q| q.byte_size()).sum::<usize>()
    }
}

// ── 3-bit packing ────────────────────────────────────────────────────

/// Pack 3-bit values (each 0..7) into bytes, 8 values per 3 bytes.
///
/// Bit layout within each 3-byte group (little-endian):
/// ```text
/// byte 0: [v0₂ v0₁ v0₀ | v1₂ v1₁ v1₀ | v2₁ v2₀]
/// byte 1: [v2₂ | v3₂ v3₁ v3₀ | v4₂ v4₁ v4₀ | v5₀]
/// byte 2: [v5₂ v5₁ | v6₂ v6₁ v6₀ | v7₂ v7₁ v7₀]
/// ```
fn pack_3bit(values: &[u8]) -> Vec<u8> {
    let num_bytes = (values.len() * 3).div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        packed[byte_idx] |= (v & 0x7) << bit_idx;
        // Handle straddling a byte boundary (bit_idx > 5)
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            packed[byte_idx + 1] |= (v & 0x7) >> (8 - bit_idx);
        }
    }
    packed
}

/// Unpack 3-bit values from packed bytes.
fn unpack_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut v = (packed[byte_idx] >> bit_idx) & 0x7;
        // Handle straddling
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            v |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x7;
        }
        values.push(v);
    }
    values
}

// ── Per-vector quantize/dequantize ───────────────────────────────────

/// Quantize an f32 slice to a QuantizedVec (3-bit uniform).
fn quantize_vec(data: &[f32]) -> QuantizedVec {
    if data.is_empty() {
        return QuantizedVec {
            min: 0.0,
            scale: 0.0,
            packed: vec![],
        };
    }

    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = max - min;
    let scale = if range > 0.0 {
        range / MAX_INDEX as f32
    } else {
        0.0
    };

    let indices: Vec<u8> = data
        .iter()
        .map(|&v| {
            if scale == 0.0 {
                0
            } else {
                ((v - min) / scale).round().clamp(0.0, MAX_INDEX as f32) as u8
            }
        })
        .collect();

    QuantizedVec {
        min,
        scale,
        packed: pack_3bit(&indices),
    }
}

/// Dequantize a QuantizedVec back to f32 values.
fn dequantize_vec(qv: &QuantizedVec, dim: usize) -> Vec<f32> {
    let indices = unpack_3bit(&qv.packed, dim);
    indices
        .iter()
        .map(|&idx| qv.min + idx as f32 * qv.scale)
        .collect()
}

// ── Tensor bridge ────────────────────────────────────────────────────

/// Compress an f16 tensor `[1, num_kv_heads, seq_len, head_dim]` to quantized vecs.
///
/// Returns (quantized_vecs, seq_len). One QuantizedVec per (head, token) pair,
/// stored row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
pub(crate) fn compress_tensor(
    tensor: &Tensor,
) -> Result<(Vec<QuantizedVec>, usize), InferenceError> {
    let (batch, num_heads, seq_len, head_dim) = tensor
        .dims4()
        .map_err(|e| InferenceError::CompressionFailed(format!("unexpected tensor shape: {e}")))?;
    if batch != 1 {
        return Err(InferenceError::CompressionFailed(format!(
            "expected batch size 1, got {batch}"
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

    let vecs: Vec<QuantizedVec> = rows.iter().map(|row| quantize_vec(row)).collect();
    Ok((vecs, seq_len))
}

/// Decompress quantized vecs back to an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
pub(crate) fn decompress_tensor(
    vecs: &[QuantizedVec],
    num_kv_heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &Device,
) -> Result<Tensor, InferenceError> {
    let expected = num_kv_heads * seq_len;
    if vecs.len() != expected {
        return Err(InferenceError::CompressionFailed(format!(
            "expected {expected} quantized vecs, got {}",
            vecs.len()
        )));
    }

    let mut flat = Vec::with_capacity(expected * head_dim);
    for qv in vecs {
        flat.extend_from_slice(&dequantize_vec(qv, head_dim));
    }

    Tensor::from_vec(flat, (1, num_kv_heads, seq_len, head_dim), device)
        .and_then(|t| t.to_dtype(DType::F16))
        .map_err(|e| InferenceError::CompressionFailed(e.to_string()))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_roundtrip() {
        let values: Vec<u8> = (0..16).map(|i| i % 8).collect();
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_sevens() {
        let values = vec![7u8; 24];
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, 24);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_unpack_all_zeros() {
        let values = vec![0u8; 128];
        let packed = pack_3bit(&values);
        assert_eq!(packed.len(), 48); // 128 * 3 / 8 = 48
        let unpacked = unpack_3bit(&packed, 128);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let data: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
        let qv = quantize_vec(&data);
        let restored = dequantize_vec(&qv, 128);

        assert_eq!(restored.len(), 128);
        let max_err: f32 = data
            .iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 3-bit quantization of [0, 1] range: step = 1/7 ≈ 0.143
        // Max error should be < half step ≈ 0.072
        assert!(max_err < 0.08, "max error {max_err} too large");
    }

    #[test]
    fn quantize_constant_vector() {
        let data = vec![42.0_f32; 128];
        let qv = quantize_vec(&data);
        let restored = dequantize_vec(&qv, 128);
        // Constant vector: all values should be exactly 42.0
        assert!(restored.iter().all(|&v| (v - 42.0).abs() < 1e-6));
    }

    #[test]
    fn quantize_empty_vector() {
        let qv = quantize_vec(&[]);
        assert_eq!(qv.byte_size(), 8); // just min + scale
        let restored = dequantize_vec(&qv, 0);
        assert!(restored.is_empty());
    }

    #[test]
    fn quantized_vec_byte_size() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        // 8 (header) + 48 (128 * 3 / 8) = 56 bytes
        assert_eq!(qv.byte_size(), 56);
    }

    #[test]
    fn compressed_kv_layer_byte_size() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 8], // 8 heads × 1 token
            v: vec![qv; 8],
            seq_len: 1,
        };
        // 2 tensors * 8 * 56 = 896 bytes
        assert_eq!(layer.byte_size(), 896);
    }

    #[test]
    fn compress_tensor_roundtrip() {
        let shape = (1, 8, 4, 128);
        let tensor = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let (vecs, seq_len) = compress_tensor(&tensor).unwrap();
        assert_eq!(seq_len, 4);
        assert_eq!(vecs.len(), 32); // 8 heads * 4 tokens

        let restored = decompress_tensor(&vecs, 8, 4, 128, &Device::Cpu).unwrap();
        assert_eq!(restored.dims4().unwrap(), (1, 8, 4, 128));
        assert_eq!(restored.dtype(), DType::F16);
    }

    #[test]
    fn quantized_vec_serde_roundtrip() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        let bytes = postcard::to_allocvec(&qv).unwrap();
        let restored: QuantizedVec = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(qv.min, restored.min);
        assert_eq!(qv.scale, restored.scale);
        assert_eq!(qv.packed, restored.packed);
    }

    #[test]
    fn compressed_kv_layer_serde_roundtrip() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
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
}

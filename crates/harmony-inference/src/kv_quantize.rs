//! Q8_0 quantization for KV cache entries.
//!
//! Each key/value vector of `head_dim` elements is independently quantized
//! using symmetric absmax INT8:
//!
//!   scale = max(|x|) / 127
//!   quantized[i] = round(x[i] / scale)
//!   dequantized[i] = quantized[i] * scale
//!
//! This halves KV cache memory (F16 → INT8 + f32 scale) while maintaining
//! acceptable attention quality for autoregressive decode. During a forward
//! pass only one layer is dequantized to F16 at a time; the remaining layers
//! stay in q8 storage, yielding ~45% memory savings for the full cache.

use candle_core::{DType, Device, Result, Tensor};

/// Quantized KV storage for a single transformer layer.
///
/// Stores keys and values as per-vector absmax INT8. The flat layout matches
/// the tensor convention `[1, num_kv_heads, seq_len, head_dim]` so that
/// head-then-token ordering is contiguous in memory.
pub(crate) struct Q8KvLayer {
    k_scales: Vec<f32>,
    k_data: Vec<i8>,
    v_scales: Vec<f32>,
    v_data: Vec<i8>,
    head_dim: usize,
    num_kv_heads: usize,
    seq_len: usize,
    /// Original dtype of the tensors (restored on dequantize).
    dtype: DType,
}

impl Q8KvLayer {
    /// Quantize KV tensors into q8_0 storage.
    ///
    /// `k` and `v` must have shape `[1, num_kv_heads, seq_len, head_dim]`.
    /// The original dtype is stored and restored on [`Self::dequantize`].
    pub(crate) fn quantize(k: &Tensor, v: &Tensor) -> Result<Self> {
        let (_, num_kv_heads, seq_len, head_dim) = k.dims4()?;
        let dtype = k.dtype();
        let num_vecs = num_kv_heads * seq_len;

        let k_f32 = k
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let v_f32 = v
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        let mut k_scales = Vec::with_capacity(num_vecs);
        let mut k_data = Vec::with_capacity(num_vecs * head_dim);
        let mut v_scales = Vec::with_capacity(num_vecs);
        let mut v_data = Vec::with_capacity(num_vecs * head_dim);

        for i in 0..num_vecs {
            let off = i * head_dim;
            let (scale, qvec) = quantize_vec(&k_f32[off..off + head_dim]);
            k_scales.push(scale);
            k_data.extend_from_slice(&qvec);
        }
        for i in 0..num_vecs {
            let off = i * head_dim;
            let (scale, qvec) = quantize_vec(&v_f32[off..off + head_dim]);
            v_scales.push(scale);
            v_data.extend_from_slice(&qvec);
        }

        Ok(Self {
            k_scales,
            k_data,
            v_scales,
            v_data,
            head_dim,
            num_kv_heads,
            seq_len,
            dtype,
        })
    }

    /// Dequantize to tensors in the original dtype.
    ///
    /// Returns `(K, V)` with shape `[1, num_kv_heads, seq_len, head_dim]`
    /// in the same dtype that was passed to [`Self::quantize`].
    pub(crate) fn dequantize(&self, device: &Device) -> Result<(Tensor, Tensor)> {
        let num_vecs = self.num_kv_heads * self.seq_len;
        let total = num_vecs * self.head_dim;

        let mut k_f32 = Vec::with_capacity(total);
        let mut v_f32 = Vec::with_capacity(total);

        for i in 0..num_vecs {
            let off = i * self.head_dim;
            let scale = self.k_scales[i];
            for j in 0..self.head_dim {
                k_f32.push(self.k_data[off + j] as f32 * scale);
            }
        }

        for i in 0..num_vecs {
            let off = i * self.head_dim;
            let scale = self.v_scales[i];
            for j in 0..self.head_dim {
                v_f32.push(self.v_data[off + j] as f32 * scale);
            }
        }

        let shape = (1, self.num_kv_heads, self.seq_len, self.head_dim);
        let k = Tensor::from_vec(k_f32, shape, device)?.to_dtype(self.dtype)?;
        let v = Tensor::from_vec(v_f32, shape, device)?.to_dtype(self.dtype)?;

        Ok((k, v))
    }

    /// Total memory in bytes (both K and V).
    pub(crate) fn memory_bytes(&self) -> usize {
        let num_vecs = self.num_kv_heads * self.seq_len;
        // Each side: f32 scale per vector + i8 per element
        2 * (num_vecs * 4 + num_vecs * self.head_dim)
    }
}

/// Quantize a single vector to INT8 using absmax scaling.
///
/// Returns `(scale, quantized)` where `scale = max(|x|) / 127`.
fn quantize_vec(values: &[f32]) -> (f32, Vec<i8>) {
    let absmax = values.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if absmax == 0.0 {
        return (0.0, vec![0i8; values.len()]);
    }
    let scale = absmax / 127.0;
    let inv_scale = 127.0 / absmax;
    let quantized: Vec<i8> = values
        .iter()
        .map(|&v| (v * inv_scale).round().clamp(-128.0, 127.0) as i8)
        .collect();
    (scale, quantized)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    /// Helper: create a random F16 tensor with shape [1, heads, seq, dim].
    fn rand_f16(heads: usize, seq: usize, dim: usize) -> Tensor {
        Tensor::rand(-1.0f32, 1.0f32, (1, heads, seq, dim), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap()
    }

    #[test]
    fn quantize_dequantize_roundtrip_low_error() {
        let k = rand_f16(4, 8, 64);
        let v = rand_f16(4, 8, 64);

        let q8 = Q8KvLayer::quantize(&k, &v).unwrap();
        let (k_hat, v_hat) = q8.dequantize(&Device::Cpu).unwrap();

        // Compare in F32 to avoid F16 precision noise.
        let k_f32 = k.to_dtype(DType::F32).unwrap();
        let v_f32 = v.to_dtype(DType::F32).unwrap();
        let k_hat_f32 = k_hat.to_dtype(DType::F32).unwrap();
        let v_hat_f32 = v_hat.to_dtype(DType::F32).unwrap();

        // Max absolute error: flatten to scalar via max of abs diff.
        let k_max: f32 = (&k_f32 - &k_hat_f32)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        let v_max: f32 = (&v_f32 - &v_hat_f32)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            k_max < 0.02,
            "key roundtrip max error too high: {k_max}"
        );
        assert!(
            v_max < 0.02,
            "value roundtrip max error too high: {v_max}"
        );
    }

    #[test]
    fn quantize_zeros() {
        let z = Tensor::zeros((1, 2, 4, 8), DType::F16, &Device::Cpu).unwrap();
        let q8 = Q8KvLayer::quantize(&z, &z).unwrap();
        let (k, v) = q8.dequantize(&Device::Cpu).unwrap();

        let k_sum: f32 = k
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(k_sum, 0.0);
        let v_sum: f32 = v
            .to_dtype(DType::F32)
            .unwrap()
            .abs()
            .unwrap()
            .sum_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert_eq!(v_sum, 0.0);
    }

    #[test]
    fn shapes_and_dtype_preserved() {
        let k = rand_f16(8, 16, 80);
        let v = rand_f16(8, 16, 80);
        let q8 = Q8KvLayer::quantize(&k, &v).unwrap();
        let (k_hat, v_hat) = q8.dequantize(&Device::Cpu).unwrap();

        assert_eq!(k_hat.dims(), &[1, 8, 16, 80]);
        assert_eq!(v_hat.dims(), &[1, 8, 16, 80]);
        assert_eq!(k_hat.dtype(), DType::F16);
        assert_eq!(v_hat.dtype(), DType::F16);

        // F32 input should dequantize back to F32.
        let k32 = Tensor::rand(-1.0f32, 1.0, (1, 4, 8, 32), &Device::Cpu).unwrap();
        let v32 = Tensor::rand(-1.0f32, 1.0, (1, 4, 8, 32), &Device::Cpu).unwrap();
        let q8_32 = Q8KvLayer::quantize(&k32, &v32).unwrap();
        let (k_hat32, v_hat32) = q8_32.dequantize(&Device::Cpu).unwrap();
        assert_eq!(k_hat32.dtype(), DType::F32);
        assert_eq!(v_hat32.dtype(), DType::F32);
    }

    #[test]
    fn memory_savings() {
        let heads = 8;
        let seq = 32;
        let dim = 80;
        let k = rand_f16(heads, seq, dim);
        let v = rand_f16(heads, seq, dim);

        let f16_bytes = (k.elem_count() + v.elem_count()) * 2; // F16 = 2 bytes
        let q8 = Q8KvLayer::quantize(&k, &v).unwrap();
        let q8_bytes = q8.memory_bytes();

        // q8 should be roughly half of F16.
        assert!(
            q8_bytes < f16_bytes,
            "q8 ({q8_bytes}) should be less than f16 ({f16_bytes})"
        );
        let ratio = q8_bytes as f64 / f16_bytes as f64;
        assert!(
            ratio < 0.6,
            "compression ratio {ratio:.2} should be < 0.6"
        );
    }

    #[test]
    fn single_token_single_head() {
        let k = rand_f16(1, 1, 16);
        let v = rand_f16(1, 1, 16);
        let q8 = Q8KvLayer::quantize(&k, &v).unwrap();
        assert_eq!(q8.seq_len, 1);
        assert_eq!(q8.num_kv_heads, 1);
        assert_eq!(q8.k_scales.len(), 1);
        assert_eq!(q8.k_data.len(), 16);
        let (k_hat, _) = q8.dequantize(&Device::Cpu).unwrap();
        assert_eq!(k_hat.dims(), &[1, 1, 1, 16]);
    }

    #[test]
    fn quantize_vec_absmax_scaling() {
        let vals = vec![0.0, 0.5, -1.0, 0.25];
        let (scale, qvec) = quantize_vec(&vals);
        // absmax = 1.0, scale = 1.0/127
        assert!((scale - 1.0 / 127.0).abs() < 1e-6);
        // -1.0 should map to -127
        assert_eq!(qvec[2], -127);
        // 0.5 should map to ~64
        assert!((qvec[1] as f32 - 63.5).abs() < 1.0);
    }

    #[test]
    fn quantize_vec_all_zeros() {
        let vals = vec![0.0; 8];
        let (scale, qvec) = quantize_vec(&vals);
        assert_eq!(scale, 0.0);
        assert!(qvec.iter().all(|&q| q == 0));
    }
}

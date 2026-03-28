//! Gated residual injection module for Engram embeddings.
//!
//! Implements the DeepSeek-inspired conditional memory injection pattern:
//! project -> gate -> depthwise conv1d -> SiLU -> residual.
//!
//! The caller adds the returned residual to the hidden state:
//! `h = h + module.forward(h, engram_embedding)?`

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::{Conv1d, Conv1dConfig, Linear};

/// Gated residual module that injects Engram embeddings into transformer
/// hidden states.
///
/// Given a hidden state `h` and an Engram embedding `e`, computes:
/// 1. Project engram into key/value: `k = key_proj(e)`, `v = value_proj(e)`
/// 2. Gate via normalized dot product: `gate = sigmoid(dot(norm(h), norm(k)) / sqrt(d))`
/// 3. Apply gate: `gated = gate * v`
/// 4. Causal depthwise conv1d with left-padding
/// 5. Return `silu(conv_out)` (the residual, NOT added to h)
#[derive(Debug, Clone)]
pub struct EngramGatedResidual {
    key_proj: Linear,
    value_proj: Linear,
    gate_norm: candle_nn::RmsNorm,
    key_norm: candle_nn::RmsNorm,
    conv1d: Conv1d,
    hidden_dim: usize,
    conv_kernel_size: usize,
}

impl EngramGatedResidual {
    /// Create with random Kaiming-scale weights for testing.
    ///
    /// `engram_dim` is the dimension of the Engram embedding input.
    /// `hidden_dim` is the model's hidden dimension (output dimension).
    /// `conv_kernel_size` is the kernel size for the causal depthwise conv1d.
    pub fn new(
        engram_dim: usize,
        hidden_dim: usize,
        conv_kernel_size: usize,
        device: &Device,
    ) -> Result<Self> {
        // Kaiming uniform scale for fan_in = engram_dim
        let scale = (1.0 / engram_dim as f64).sqrt();

        let key_proj_weight =
            (Tensor::randn(0f32, 1f32, (hidden_dim, engram_dim), device)? * scale)?;
        let value_proj_weight =
            (Tensor::randn(0f32, 1f32, (hidden_dim, engram_dim), device)? * scale)?;

        let key_proj = Linear::new(key_proj_weight, None);
        let value_proj = Linear::new(value_proj_weight, None);

        let gate_norm_weight = Tensor::ones(hidden_dim, DType::F32, device)?;
        let key_norm_weight = Tensor::ones(hidden_dim, DType::F32, device)?;
        let gate_norm = candle_nn::RmsNorm::new(gate_norm_weight, 1e-6);
        let key_norm = candle_nn::RmsNorm::new(key_norm_weight, 1e-6);

        // Depthwise conv1d: groups = hidden_dim, so weight is [hidden_dim, 1, kernel_size]
        let conv1d_weight = Tensor::zeros((hidden_dim, 1, conv_kernel_size), DType::F32, device)?;
        let conv1d_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: hidden_dim,
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv1d_weight, None, conv1d_config);

        Ok(Self {
            key_proj,
            value_proj,
            gate_norm,
            key_norm,
            conv1d,
            hidden_dim,
            conv_kernel_size,
        })
    }

    /// Construct from pre-loaded weight tensors.
    ///
    /// Expected shapes:
    /// - `key_proj_weight`: `[hidden_dim, engram_dim]`
    /// - `value_proj_weight`: `[hidden_dim, engram_dim]`
    /// - `gate_norm_weight`: `[hidden_dim]`
    /// - `key_norm_weight`: `[hidden_dim]`
    /// - `conv1d_weight`: `[hidden_dim, 1, kernel_size]` (kernel_size inferred from shape)
    /// - `rms_norm_eps`: epsilon for RmsNorm layers (e.g., from model GGUF metadata)
    pub fn from_tensors(
        key_proj_weight: Tensor,
        value_proj_weight: Tensor,
        gate_norm_weight: Tensor,
        key_norm_weight: Tensor,
        conv1d_weight: Tensor,
        hidden_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        // Validate tensor shapes upfront for clear error messages.
        let kw = key_proj_weight.shape().dims();
        if kw.len() != 2 || kw[0] != hidden_dim {
            candle_core::bail!(
                "key_proj_weight shape {kw:?} does not match hidden_dim={hidden_dim}"
            );
        }
        let vw = value_proj_weight.shape().dims();
        if vw.len() != 2 || vw[0] != hidden_dim {
            candle_core::bail!(
                "value_proj_weight shape {vw:?} does not match hidden_dim={hidden_dim}"
            );
        }
        let cw = conv1d_weight.shape().dims();
        if cw.len() != 3 || cw[0] != hidden_dim || cw[1] != 1 {
            candle_core::bail!(
                "conv1d_weight shape {cw:?} expected [{hidden_dim}, 1, kernel_size]"
            );
        }
        let conv_kernel_size = cw[2];

        let key_proj = Linear::new(key_proj_weight, None);
        let value_proj = Linear::new(value_proj_weight, None);

        let gate_norm = candle_nn::RmsNorm::new(gate_norm_weight, rms_norm_eps);
        let key_norm = candle_nn::RmsNorm::new(key_norm_weight, rms_norm_eps);

        let conv1d_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: hidden_dim,
            cudnn_fwd_algo: None,
        };
        let conv1d = Conv1d::new(conv1d_weight, None, conv1d_config);

        Ok(Self {
            key_proj,
            value_proj,
            gate_norm,
            key_norm,
            conv1d,
            hidden_dim,
            conv_kernel_size,
        })
    }

    /// Compute the gated residual for Engram injection.
    ///
    /// Returns ONLY the residual tensor. The caller performs the addition:
    /// `hidden_state = hidden_state + self.forward(hidden_state, engram_embedding)?`
    ///
    /// # Arguments
    /// - `hidden_state`: `[batch, seq_len, hidden_dim]`
    /// - `engram_embedding`: `[batch, seq_len, engram_dim]`
    ///
    /// # Returns
    /// Residual tensor: `[batch, seq_len, hidden_dim]`
    pub fn forward(&self, hidden_state: &Tensor, engram_embedding: &Tensor) -> Result<Tensor> {
        // Project engram into key/value space: [b, l, hidden_dim]
        let key = self.key_proj.forward(engram_embedding)?;
        let value = self.value_proj.forward(engram_embedding)?;

        // Normalize hidden state and key for stable gating
        let h_norm = self.gate_norm.forward(hidden_state)?;
        let k_norm = self.key_norm.forward(&key)?;

        // Dot product gate: [b, l, 1]
        let dot = (&h_norm * &k_norm)?.sum_keepdim(D::Minus1)?;

        // Scale by 1/sqrt(hidden_dim) and sigmoid
        let scale = (self.hidden_dim as f64).sqrt();
        let gate = candle_nn::ops::sigmoid(&(dot / scale)?)?;

        // Apply gate to value: [b, l, hidden_dim]
        let gated_value = gate.broadcast_mul(&value)?;

        // Causal depthwise conv1d:
        // NLC -> NCL for conv1d
        let ncl = gated_value.transpose(1, 2)?;

        // Left-pad with (kernel_size - 1) zeros for causal masking
        let padded = ncl.pad_with_zeros(2, self.conv_kernel_size - 1, 0)?;

        // Apply depthwise conv1d
        let conv_out = self.conv1d.forward(&padded)?;

        // NCL -> NLC
        let nlc = conv_out.transpose(1, 2)?;

        // SiLU activation
        candle_nn::Activation::Silu.forward(&nlc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_preservation() -> Result<()> {
        let device = Device::Cpu;
        let module = EngramGatedResidual::new(16, 64, 3, &device)?;

        let hidden = Tensor::randn(0f32, 1f32, (1, 5, 64), &device)?;
        let engram = Tensor::randn(0f32, 1f32, (1, 5, 16), &device)?;

        let output = module.forward(&hidden, &engram)?;
        assert_eq!(output.dims(), &[1, 5, 64]);
        Ok(())
    }

    #[test]
    fn shape_preservation_batch() -> Result<()> {
        let device = Device::Cpu;
        let module = EngramGatedResidual::new(16, 64, 3, &device)?;

        let hidden = Tensor::randn(0f32, 1f32, (2, 10, 64), &device)?;
        let engram = Tensor::randn(0f32, 1f32, (2, 10, 16), &device)?;

        let output = module.forward(&hidden, &engram)?;
        assert_eq!(output.dims(), &[2, 10, 64]);
        Ok(())
    }

    #[test]
    fn zero_embedding_returns_zero_residual() -> Result<()> {
        let device = Device::Cpu;
        let module = EngramGatedResidual::new(16, 64, 3, &device)?;

        let hidden = Tensor::randn(0f32, 1f32, (1, 5, 64), &device)?;
        let engram = Tensor::zeros((1, 5, 16), DType::F32, &device)?;

        let output = module.forward(&hidden, &engram)?;

        // With zero engram, key_proj(zeros)=zeros, value_proj(zeros)=zeros (no bias).
        // gate = sigmoid(0) = 0.5, but gated_value = 0.5 * zeros = zeros.
        // conv1d(zeros) = zeros, silu(zeros) = 0.
        let max_val: f32 = output.abs()?.max_all()?.to_scalar()?;
        assert!(
            max_val < 1e-6,
            "expected near-zero residual, got max abs {max_val}"
        );
        Ok(())
    }

    #[test]
    fn gate_range_is_zero_to_one() -> Result<()> {
        let device = Device::Cpu;
        let engram_dim = 16;
        let hidden_dim = 64;

        let module = EngramGatedResidual::new(engram_dim, hidden_dim, 3, &device)?;

        let hidden = Tensor::randn(0f32, 1f32, (1, 5, hidden_dim), &device)?;
        let engram = Tensor::randn(0f32, 1f32, (1, 5, engram_dim), &device)?;

        // Manually compute the gate to verify its range
        let key = module.key_proj.forward(&engram)?;
        let h_norm = module.gate_norm.forward(&hidden)?;
        let k_norm = module.key_norm.forward(&key)?;
        let dot = (&h_norm * &k_norm)?.sum_keepdim(D::Minus1)?;
        let scale = (hidden_dim as f64).sqrt();
        let gate = candle_nn::ops::sigmoid(&(dot / scale)?)?;

        let gate_data: Vec<f32> = gate.flatten_all()?.to_vec1()?;
        for (i, &g) in gate_data.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&g),
                "gate[{i}] = {g} is outside [0, 1]"
            );
        }
        Ok(())
    }

    #[test]
    fn causal_conv1d_no_future_leakage() -> Result<()> {
        let device = Device::Cpu;
        let hidden_dim = 8;
        let seq_len = 5;
        let kernel_size = 3;

        // Use from_tensors with known weights so conv actually does something
        let engram_dim = 4;
        let key_w = Tensor::ones((hidden_dim, engram_dim), DType::F32, &device)?;
        let val_w = Tensor::ones((hidden_dim, engram_dim), DType::F32, &device)?;
        let gate_norm_w = Tensor::ones(hidden_dim, DType::F32, &device)?;
        let key_norm_w = Tensor::ones(hidden_dim, DType::F32, &device)?;
        // Conv weight of ones so it sums over the kernel window
        let conv_w = Tensor::ones((hidden_dim, 1, kernel_size), DType::F32, &device)?;

        let module = EngramGatedResidual::from_tensors(
            key_w,
            val_w,
            gate_norm_w,
            key_norm_w,
            conv_w,
            hidden_dim,
            1e-6,
        )?;

        // Create engram with non-zero value only at the LAST position
        let mut engram_data = vec![0f32; seq_len * engram_dim];
        // Set last position (index 4) to non-zero
        for d in 0..engram_dim {
            engram_data[(seq_len - 1) * engram_dim + d] = 1.0;
        }
        let engram = Tensor::from_vec(engram_data, (1, seq_len, engram_dim), &device)?;
        let hidden = Tensor::ones((1, seq_len, hidden_dim), DType::F32, &device)?;

        let output = module.forward(&hidden, &engram)?;

        // Positions 0..3 should see zero engram input.
        // With causal conv (left-padding), earlier positions cannot see future data.
        // Positions 0, 1, 2 should have zero output because:
        // - gated_value is zero at positions 0..3
        // - conv kernel_size=3, so position 2's window covers positions 0,1,2 (all zero)
        // - position 3's window covers 1,2,3 (all zero) => also zero
        // Only position 4 (last) should be non-zero.
        let output_data: Vec<Vec<Vec<f32>>> = output.to_vec3()?;
        for pos in 0..(seq_len - 1) {
            let max_at_pos: f32 = output_data[0][pos]
                .iter()
                .map(|v| v.abs())
                .fold(0f32, f32::max);
            assert!(
                max_at_pos < 1e-6,
                "position {pos} should be unaffected by future data, got max abs {max_at_pos}"
            );
        }

        // Last position should be non-zero (engram signal passed through)
        let max_at_last: f32 = output_data[0][seq_len - 1]
            .iter()
            .map(|v| v.abs())
            .fold(0f32, f32::max);
        assert!(
            max_at_last > 1e-6,
            "last position should have non-zero output, got max abs {max_at_last}"
        );
        Ok(())
    }

    #[test]
    fn single_token_works() -> Result<()> {
        let device = Device::Cpu;
        let module = EngramGatedResidual::new(16, 64, 3, &device)?;

        // seq_len = 1, simulating a decode step
        let hidden = Tensor::randn(0f32, 1f32, (1, 1, 64), &device)?;
        let engram = Tensor::randn(0f32, 1f32, (1, 1, 16), &device)?;

        let output = module.forward(&hidden, &engram)?;
        assert_eq!(output.dims(), &[1, 1, 64]);
        Ok(())
    }

    #[test]
    fn from_tensors_matches_shapes() -> Result<()> {
        let device = Device::Cpu;
        let engram_dim = 32;
        let hidden_dim = 128;
        let kernel_size = 5;

        let key_w = Tensor::randn(0f32, 1f32, (hidden_dim, engram_dim), &device)?;
        let val_w = Tensor::randn(0f32, 1f32, (hidden_dim, engram_dim), &device)?;
        let gate_norm_w = Tensor::ones(hidden_dim, DType::F32, &device)?;
        let key_norm_w = Tensor::ones(hidden_dim, DType::F32, &device)?;
        let conv_w = Tensor::randn(0f32, 1f32, (hidden_dim, 1, kernel_size), &device)?;

        let module = EngramGatedResidual::from_tensors(
            key_w,
            val_w,
            gate_norm_w,
            key_norm_w,
            conv_w,
            hidden_dim,
            1e-6,
        )?;

        let hidden = Tensor::randn(0f32, 1f32, (1, 7, hidden_dim), &device)?;
        let engram = Tensor::randn(0f32, 1f32, (1, 7, engram_dim), &device)?;

        let output = module.forward(&hidden, &engram)?;
        assert_eq!(output.dims(), &[1, 7, 128]);
        Ok(())
    }
}

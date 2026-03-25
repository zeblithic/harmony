//! QwenEngine: candle-based inference engine for quantized Qwen3 models.

use candle_core::Device;
use crate::{InferenceEngine, SamplingParams};
use crate::error::InferenceError;

/// Inference engine wrapping candle-transformers' quantized Qwen3 implementation.
pub struct QwenEngine {
    device: Device,
}

impl QwenEngine {
    /// Create a new engine targeting the given device.
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

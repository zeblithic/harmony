//! QwenEngine: candle-based inference engine for quantized Qwen3 models.

#[allow(unused_imports)]
use crate::error::InferenceError;
#[allow(unused_imports)]
use crate::{InferenceEngine, SamplingParams};
use candle_core::Device;

/// Inference engine wrapping candle-transformers' quantized Qwen3 implementation.
pub struct QwenEngine {
    #[allow(dead_code)]
    device: Device,
}

impl QwenEngine {
    /// Create a new engine targeting the given device.
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}

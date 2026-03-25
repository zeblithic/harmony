//! QwenEngine: candle-based inference engine for quantized Qwen3 models.

use std::io::Cursor;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3;
use rand::thread_rng;

use crate::error::InferenceError;
use crate::{InferenceEngine, SamplingParams};

/// Inference engine wrapping candle-transformers' quantized Qwen3 implementation.
///
/// # Usage
///
/// ```ignore
/// let mut engine = QwenEngine::new(Device::Cpu);
/// engine.load_gguf(&gguf_bytes)?;
/// engine.load_tokenizer(&tokenizer_json)?;
///
/// let tokens = engine.tokenize("Hello")?;
/// let logits = engine.forward(&tokens)?;
/// let next = engine.sample(&logits, &SamplingParams::greedy())?;
/// ```
pub struct QwenEngine {
    model: Option<quantized_qwen3::ModelWeights>,
    tokenizer: Option<tokenizers::Tokenizer>,
    device: Device,
    /// Current position in the KV cache (advances with each forward call).
    position: usize,
    /// Tokens seen so far in this conversation (for repeat penalty).
    token_history: Vec<u32>,
}

impl QwenEngine {
    /// Create a new engine targeting the given device.
    ///
    /// Use `Device::Cpu` for CPU inference. GPU support requires the
    /// `cuda` or `metal` feature flags.
    pub fn new(device: Device) -> Self {
        Self {
            model: None,
            tokenizer: None,
            device,
            position: 0,
            token_history: Vec::new(),
        }
    }
}

impl InferenceEngine for QwenEngine {
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError> {
        let mut cursor = Cursor::new(gguf_data);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        let model = quantized_qwen3::ModelWeights::from_gguf(content, &mut cursor, &self.device)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        self.model = Some(model);
        self.position = 0;
        self.token_history.clear();
        Ok(())
    }

    fn load_tokenizer(&mut self, tokenizer_json: &[u8]) -> Result<(), InferenceError> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
        self.tokenizer = Some(tokenizer);
        Ok(())
    }

    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(InferenceError::TokenizerNotLoaded)?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or(InferenceError::TokenizerNotLoaded)?;
        tokenizer
            .decode(tokens, true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))
    }

    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let model = self.model.as_mut().ok_or(InferenceError::ModelNotLoaded)?;
        // ModelWeights::forward expects a 2D tensor (batch, seq_len).
        let seq_len = tokens.len();
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        let logits = model
            .forward(&input, self.position)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Candle's quantized_qwen3 outputs [batch, vocab_size] with batch=1.
        // Handle both 1D (defensive fallback) and 2D shapes.
        let logits = match logits.dims().len() {
            1 => logits, // Defensive: [vocab_size]
            2 => {
                // [batch, vocab_size] — select the last row
                let batch = logits
                    .dim(0)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
                logits
                    .get(batch - 1)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?
            }
            n => {
                return Err(InferenceError::ForwardFailed(format!(
                    "unexpected logits dimensionality: {n}D"
                )))
            }
        };

        let result = logits
            .to_vec1::<f32>()
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Only advance state after successful logits extraction, so a failure
        // doesn't leave position/token_history out of sync with the KV cache.
        self.token_history.extend_from_slice(tokens);
        self.position += tokens.len();

        Ok(result)
    }

    fn sample(&self, logits: &[f32], params: &SamplingParams) -> Result<u32, InferenceError> {
        let mut rng = thread_rng();
        crate::sampling::sample(logits, params, &self.token_history, &mut rng)
    }

    fn reset(&mut self) {
        if let Some(model) = &mut self.model {
            model.clear_kv_cache();
        }
        self.position = 0;
        self.token_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_without_model_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.forward(&[1, 2, 3]);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn test_tokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.tokenize("hello");
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn test_detokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.detokenize(&[1, 2, 3]);
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn test_invalid_gguf_bytes_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_gguf(b"not a valid gguf file");
        assert!(matches!(result, Err(InferenceError::InvalidGguf(_))));
    }

    #[test]
    fn test_invalid_tokenizer_json_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_tokenizer(b"not valid json");
        assert!(matches!(result, Err(InferenceError::TokenizerError(_))));
    }

    #[test]
    fn test_sample_without_model_still_works() {
        // sample() doesn't need a model -- only logits + params
        let engine = QwenEngine::new(Device::Cpu);
        let logits = [1.0_f32, 5.0, 3.0];
        let result = engine.sample(&logits, &SamplingParams::greedy());
        assert_eq!(result.unwrap(), 1); // greedy argmax
    }

    #[test]
    fn test_reset_clears_state() {
        let mut engine = QwenEngine::new(Device::Cpu);
        // Simulate some state (without a real model, just verify fields reset)
        engine.position = 42;
        engine.token_history = vec![1, 2, 3];
        engine.reset();
        assert_eq!(engine.position, 0);
        assert!(engine.token_history.is_empty());
    }
}

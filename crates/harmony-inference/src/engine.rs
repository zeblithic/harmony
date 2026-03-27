//! QwenEngine: stateless inference engine for quantized Qwen3 models.
//!
//! After initialization (`load_gguf`, `load_tokenizer`), all inference methods
//! take `&self`. Mutable state lives in the caller-owned [`InferenceCache`].

use std::io::Cursor;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use rand::thread_rng;

use crate::error::InferenceError;
use crate::{InferenceCache, InferenceEngine, SamplingParams};

/// Inference engine wrapping a forked quantized Qwen3 implementation.
///
/// Stateless after initialization — all inference methods take `&self`.
/// KV cache and token history are caller-managed via [`InferenceCache`].
///
/// # Usage
///
/// ```ignore
/// let mut engine = QwenEngine::new(Device::Cpu);
/// engine.load_gguf(&gguf_bytes)?;
/// engine.load_tokenizer(&tokenizer_json)?;
///
/// let mut cache = engine.new_cache()?;
/// let tokens = engine.tokenize("Hello")?;
/// let logits = engine.forward(&tokens, &mut cache)?;
/// let next = engine.sample(&logits, &SamplingParams::greedy(), &tokens)?;
/// ```
pub struct QwenEngine {
    model: Option<crate::qwen3_ext::ModelWeights>,
    tokenizer: Option<tokenizers::Tokenizer>,
    device: Device,
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
        }
    }
}

impl InferenceEngine for QwenEngine {
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError> {
        let mut cursor = Cursor::new(gguf_data);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        let model =
            crate::qwen3_ext::ModelWeights::from_gguf(content, &mut cursor, &self.device)
                .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        self.model = Some(model);
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

    fn forward(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
    ) -> Result<Vec<f32>, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::ForwardFailed(
                "tokens slice must not be empty".into(),
            ));
        }
        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;

        // Validate cache matches model architecture.
        if cache.num_layers != model.num_layers {
            return Err(InferenceError::CacheMismatch {
                expected: model.num_layers,
                actual: cache.num_layers,
            });
        }

        let seq_len = tokens.len();
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // model.forward advances cache.position internally.
        let logits = model
            .forward(&input, cache)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Extract last logit row.
        let logits = match logits.dims().len() {
            1 => logits,
            2 => {
                let rows = logits
                    .dim(0)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
                if rows == 0 {
                    return Err(InferenceError::ForwardFailed(
                        "model returned empty logits tensor [0, vocab_size]".into(),
                    ));
                }
                logits
                    .get(rows - 1)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?
            }
            n => {
                return Err(InferenceError::ForwardFailed(format!(
                    "unexpected logits dimensionality: {n}D"
                )))
            }
        };

        logits
            .to_vec1::<f32>()
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))
    }

    fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        history: &[u32],
    ) -> Result<u32, InferenceError> {
        let mut rng = thread_rng();
        // Apply repeat_last_n windowing to the caller-provided full history.
        let context =
            if params.repeat_last_n > 0 && params.repeat_last_n < history.len() {
                &history[history.len() - params.repeat_last_n..]
            } else {
                history
            };
        crate::sampling::sample(logits, params, context, &mut rng)
    }

    fn eos_token_id(&self) -> Option<u32> {
        let tokenizer = self.tokenizer.as_ref()?;
        for eos_str in &[
            "<|endoftext|>",
            "<|im_end|>",
            "</s>",
            "<|end|>",
            "<eos>",
        ] {
            if let Some(id) = tokenizer.token_to_id(eos_str) {
                return Some(id);
            }
        }
        None
    }

    fn new_cache(&self) -> Result<InferenceCache, InferenceError> {
        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;
        Ok(InferenceCache::new(
            model.num_layers,
            model.head_dim,
            model.num_kv_heads,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_without_model_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let mut cache = InferenceCache::new(28, 128, 8);
        let result = engine.forward(&[1, 2, 3], &mut cache);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn forward_empty_tokens_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let mut cache = InferenceCache::new(28, 128, 8);
        let result = engine.forward(&[], &mut cache);
        assert!(matches!(result, Err(InferenceError::ForwardFailed(_))));
    }

    #[test]
    fn tokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.tokenize("hello");
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn detokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.detokenize(&[1, 2, 3]);
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn invalid_gguf_bytes_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_gguf(b"not a valid gguf file");
        assert!(matches!(result, Err(InferenceError::InvalidGguf(_))));
    }

    #[test]
    fn invalid_tokenizer_json_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_tokenizer(b"not valid json");
        assert!(matches!(result, Err(InferenceError::TokenizerError(_))));
    }

    #[test]
    fn sample_greedy_works_without_model() {
        let engine = QwenEngine::new(Device::Cpu);
        let logits = [1.0_f32, 5.0, 3.0];
        let result = engine.sample(&logits, &SamplingParams::greedy(), &[]);
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn sample_applies_repeat_penalty_from_history() {
        let engine = QwenEngine::new(Device::Cpu);
        let logits = [1.0_f32, 5.0, 3.0];
        let params = SamplingParams {
            temperature: 0.0,
            repeat_penalty: 10.0,
            ..SamplingParams::greedy()
        };
        // Token 1 in history → logit 5.0 / 10.0 = 0.5, so token 2 (3.0) wins
        let result = engine.sample(&logits, &params, &[1]);
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn new_cache_requires_model() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.new_cache();
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn cache_mismatch_detected() {
        let engine = QwenEngine::new(Device::Cpu);
        // No model loaded — ModelNotLoaded fires before CacheMismatch check.
        let mut cache = InferenceCache::new(99, 128, 8);
        let result = engine.forward(&[1], &mut cache);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }
}

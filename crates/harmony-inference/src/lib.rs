//! GGUF model loader and inference engine for quantized Qwen3 models.
//!
//! Wraps candle-transformers behind an [`InferenceEngine`] trait for composable
//! integration with the Harmony compute stack. Takes bytes in, returns
//! tokens/logits out.
//!
//! # Usage
//!
//! ```ignore
//! let mut engine = QwenEngine::new(candle_core::Device::Cpu);
//! engine.load_gguf(&gguf_bytes)?;
//! engine.load_tokenizer(&tokenizer_json)?;
//!
//! let tokens = engine.tokenize("Hello")?;
//! let logits = engine.forward(&tokens)?;
//! let next = engine.sample(&logits, &SamplingParams::greedy())?;
//! ```

pub mod engine;
pub mod error;
pub mod sampling;

pub use engine::QwenEngine;
pub use error::InferenceError;

/// Sampling parameters for token generation.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    /// Temperature for softmax. 0.0 = greedy (argmax).
    pub temperature: f32,
    /// Nucleus sampling threshold. 1.0 = disabled.
    pub top_p: f32,
    /// Top-k filtering. 0 = disabled.
    pub top_k: u32,
    /// Penalty for repeated tokens. 1.0 = no penalty.
    pub repeat_penalty: f32,
}

impl SamplingParams {
    /// Greedy decoding (deterministic argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
        }
    }
}

/// Trait for running inference on quantized language models.
///
/// The caller drives the autoregressive loop, enabling streaming output,
/// custom stopping criteria, and future Engram embedding injection.
pub trait InferenceEngine {
    /// Load a GGUF model from raw bytes.
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError>;

    /// Load tokenizer from a `tokenizer.json` file's bytes.
    fn load_tokenizer(&mut self, tokenizer_json: &[u8]) -> Result<(), InferenceError>;

    /// Tokenize text into token IDs.
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError>;

    /// Detokenize token IDs back to text.
    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError>;

    /// Run a single forward pass: token IDs → logits.
    ///
    /// Manages KV cache internally. First call = prefill, subsequent = decode.
    /// Advances the internal position counter by `tokens.len()`.
    ///
    /// # Error recovery
    ///
    /// If this method returns `ForwardFailed`, the KV cache may be in an
    /// indeterminate state. Call [`reset()`](Self::reset) before reusing
    /// the engine.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError>;

    /// Sample the next token from logits.
    ///
    /// Uses internal token history (populated by `forward()`) for repeat penalty.
    fn sample(&self, logits: &[f32], params: &SamplingParams) -> Result<u32, InferenceError>;

    /// Reset the KV cache and position (start a new conversation).
    fn reset(&mut self);
}

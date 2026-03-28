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
//! let mut cache = engine.new_cache()?;
//! let tokens = engine.tokenize("Hello")?;
//! let logits = engine.forward(&tokens, &mut cache)?;
//! let next = engine.sample(&logits, &SamplingParams::greedy(), &tokens)?;
//! ```

pub mod engine;
pub mod error;
pub(crate) mod qwen3_ext;
#[cfg(feature = "kv-compress")]
pub(crate) mod kv_compress;
pub mod sampling;

pub use engine::QwenEngine;
pub use error::InferenceError;

use candle_core::Tensor;

/// Externalized KV cache state for transformer inference.
///
/// Created via [`InferenceEngine::new_cache()`]. Passed to `forward()` on each
/// call. The engine appends new K/V tensors; the caller owns the lifecycle.
///
/// # Lifecycle
///
/// ```ignore
/// let cache = engine.new_cache()?;
/// let logits = engine.forward(&tokens, &mut cache)?;
/// // cache.layers now contains populated K/V tensors
/// // Drop cache to start a new conversation
/// ```
pub struct InferenceCache {
    /// Per-layer Key/Value tensor pairs. `None` = layer not yet populated.
    /// Shape per tensor: `[1, num_kv_heads, seq_len, head_dim]`
    pub layers: Vec<Option<(Tensor, Tensor)>>,
    /// Number of tokens consumed so far (position offset for RoPE).
    pub position: usize,
    /// Expected layer count (validated by `forward()`).
    pub num_layers: usize,
    /// Expected head dimension (for downstream consumers).
    pub head_dim: usize,
    /// Expected KV head count (for downstream consumers).
    pub num_kv_heads: usize,
}

impl InferenceCache {
    /// Create an empty cache for the given architecture parameters.
    pub fn new(num_layers: usize, head_dim: usize, num_kv_heads: usize) -> Self {
        Self {
            layers: (0..num_layers).map(|_| None).collect(),
            position: 0,
            num_layers,
            head_dim,
            num_kv_heads,
        }
    }

    /// Number of tokens in the cache.
    pub fn len(&self) -> usize {
        self.position
    }

    /// Whether the cache is empty (no tokens consumed).
    pub fn is_empty(&self) -> bool {
        self.position == 0
    }
}

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
    /// Must be > 0.0; values <= 0.0 are treated as no penalty.
    pub repeat_penalty: f32,
    /// Number of recent tokens to consider for repeat penalty.
    /// 0 = use entire history (unbounded). Default: 64.
    pub repeat_last_n: usize,
}

impl SamplingParams {
    /// Greedy decoding (deterministic argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
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
            repeat_last_n: 64,
        }
    }
}

/// Trait for running inference on quantized language models.
///
/// After initialization (`load_gguf`, `load_tokenizer`), all inference methods
/// take `&self` — the engine is stateless. Mutable state lives in the
/// caller-owned [`InferenceCache`].
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
    /// Appends new K/V tensors to `cache` and advances `cache.position`.
    /// First call = prefill, subsequent = decode.
    ///
    /// # Errors
    ///
    /// - `ModelNotLoaded` if no model is loaded.
    /// - `CacheMismatch` if `cache.num_layers` doesn't match the model.
    /// - `ForwardFailed` on empty tokens or tensor errors. After this error,
    ///   the cache may be in an indeterminate state — drop and recreate it.
    fn forward(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
    ) -> Result<Vec<f32>, InferenceError>;

    /// Sample the next token from logits.
    ///
    /// Pass the full token history; the implementation applies the
    /// `repeat_last_n` window from `params` internally.
    fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        history: &[u32],
    ) -> Result<u32, InferenceError>;

    /// Return the EOS (end-of-sequence) token ID, if the tokenizer defines one.
    fn eos_token_id(&self) -> Option<u32>;

    /// Create a new empty cache matching the loaded model's architecture.
    ///
    /// Returns `ModelNotLoaded` if no model has been loaded yet.
    fn new_cache(&self) -> Result<InferenceCache, InferenceError>;
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    #[test]
    fn cache_new_creates_empty_layers() {
        let cache = InferenceCache::new(28, 128, 8);
        assert_eq!(cache.layers.len(), 28);
        assert!(cache.layers.iter().all(|l| l.is_none()));
        assert_eq!(cache.position, 0);
        assert_eq!(cache.num_layers, 28);
        assert_eq!(cache.head_dim, 128);
        assert_eq!(cache.num_kv_heads, 8);
    }

    #[test]
    fn cache_len_tracks_position() {
        let mut cache = InferenceCache::new(28, 128, 8);
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
        cache.position = 42;
        assert!(!cache.is_empty());
        assert_eq!(cache.len(), 42);
    }
}

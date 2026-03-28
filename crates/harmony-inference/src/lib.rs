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
    /// Per-layer compressed K/V state. Populated by compress(), consumed by decompress().
    #[cfg(feature = "kv-compress")]
    pub(crate) compressed: Vec<Option<kv_compress::CompressedKvLayer>>,
    /// Whether the cache is currently in compressed form.
    #[cfg(feature = "kv-compress")]
    pub(crate) is_compressed: bool,
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
            #[cfg(feature = "kv-compress")]
            compressed: (0..num_layers).map(|_| None).collect(),
            #[cfg(feature = "kv-compress")]
            is_compressed: false,
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

#[cfg(feature = "kv-compress")]
impl InferenceCache {
    /// Whether the cache is currently in compressed form.
    pub fn is_compressed(&self) -> bool {
        self.is_compressed
    }

    /// Compress all populated layers to 3-bit quantized format.
    /// Frees full-precision tensors to reclaim memory.
    /// No-op if already compressed.
    /// Atomic: on error, cache remains in uncompressed state.
    pub fn compress(&mut self) -> Result<(), InferenceError> {
        if self.is_compressed {
            return Ok(());
        }

        // Phase 1: compress all populated layers into a temporary vec.
        // If any layer fails, self is untouched.
        let mut new_compressed = Vec::with_capacity(self.num_layers);
        for layer in &self.layers {
            match layer {
                Some((k, v)) => {
                    let (k_vecs, seq_len) = kv_compress::compress_tensor(k)?;
                    let (v_vecs, _) = kv_compress::compress_tensor(v)?;
                    new_compressed.push(Some(kv_compress::CompressedKvLayer {
                        k: k_vecs,
                        v: v_vecs,
                        seq_len,
                    }));
                }
                None => new_compressed.push(None),
            }
        }

        // Phase 2: commit (cannot fail — just moving data).
        self.compressed = new_compressed;
        for layer in &mut self.layers {
            *layer = None;
        }
        self.is_compressed = true;
        Ok(())
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

    #[test]
    #[cfg(feature = "kv-compress")]
    fn new_cache_is_not_compressed() {
        let cache = InferenceCache::new(28, 128, 8);
        assert!(!cache.is_compressed());
    }

    #[test]
    #[cfg(feature = "kv-compress")]
    fn new_cache_compressed_vec_matches_layers() {
        let cache = InferenceCache::new(28, 128, 8);
        assert_eq!(cache.compressed.len(), 28);
        assert!(cache.compressed.iter().all(|c| c.is_none()));
    }
}

#[cfg(test)]
#[cfg(feature = "kv-compress")]
mod kv_compress_cache_tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};

    /// Create a cache with `n_tokens` of random f16 KV tensors in layer 0.
    fn cache_with_data(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            cache.layers[0] = Some((k, v));
            cache.position = n_tokens;
        }
        cache
    }

    #[test]
    fn compress_empty_cache_succeeds() {
        let mut cache = InferenceCache::new(2, 128, 8);
        assert!(cache.compress().is_ok());
        assert!(cache.is_compressed());
    }

    #[test]
    fn compress_noop_when_already_compressed() {
        let mut cache = InferenceCache::new(2, 128, 8);
        cache.compress().unwrap();
        assert!(cache.compress().is_ok());
        assert!(cache.is_compressed());
    }

    #[test]
    fn compress_frees_full_precision_tensors() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        assert!(cache.layers[0].is_some());

        cache.compress().unwrap();

        assert!(cache.is_compressed());
        assert!(cache.layers[0].is_none());
        assert!(cache.layers[1].is_none());
        assert!(cache.compressed[0].is_some());
        assert!(cache.compressed[1].is_none()); // was unpopulated
    }
}

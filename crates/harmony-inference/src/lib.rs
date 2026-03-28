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
pub mod engram_residual;
pub mod error;
#[cfg(feature = "kv-compress")]
pub(crate) mod kv_compress;
pub(crate) mod qwen3_ext;
pub mod sampling;

pub use engine::QwenEngine;
pub use engram_residual::EngramGatedResidual;
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

/// Internal type for postcard serialization of compressed cache state.
#[cfg(feature = "kv-compress")]
#[derive(serde::Serialize, serde::Deserialize)]
struct CompressedCachePayload {
    position: usize,
    layers: Vec<Option<kv_compress::CompressedKvLayer>>,
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

    /// Approximate memory usage in bytes (accounts for compression state).
    /// Unpopulated layers contribute 0 bytes.
    pub fn memory_bytes(&self) -> usize {
        let mut total = 0;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some((k, v)) = layer {
                total += (k.elem_count() + v.elem_count()) * k.dtype().size_in_bytes();
            }
            if let Some(comp) = &self.compressed[i] {
                total += comp.byte_size();
            }
        }
        total
    }

    /// Decompress all layers back to full-precision f16 tensors.
    /// No-op if not compressed.
    /// Atomic: on error, cache remains in compressed state.
    ///
    /// # Device limitation
    /// Decompressed tensors are always placed on `Device::Cpu`. Do not use
    /// with CUDA/Metal engines until harmony-51dy resolves device threading.
    pub fn decompress(&mut self) -> Result<(), InferenceError> {
        if !self.is_compressed {
            return Ok(());
        }

        // TODO(harmony-51dy): thread device from engine instead of hardcoding Cpu
        let device = candle_core::Device::Cpu;

        // Phase 1: decompress into temporary vec.
        let mut new_layers: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.num_layers);
        for comp in &self.compressed {
            match comp {
                Some(c) => {
                    let k = kv_compress::decompress_tensor(
                        &c.k,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
                    let v = kv_compress::decompress_tensor(
                        &c.v,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
                    new_layers.push(Some((k, v)));
                }
                None => new_layers.push(None),
            }
        }

        // Phase 2: commit.
        self.layers = new_layers;
        for comp in &mut self.compressed {
            *comp = None;
        }
        self.is_compressed = false;
        Ok(())
    }

    /// Serialize the compressed cache to bytes.
    /// Returns Err if the cache is not compressed.
    pub fn serialize_compressed(&self) -> Result<Vec<u8>, InferenceError> {
        if !self.is_compressed {
            return Err(InferenceError::SerializationFailed(
                "cache is not compressed — call compress() first".into(),
            ));
        }
        let payload = CompressedCachePayload {
            position: self.position,
            layers: self.compressed.clone(),
        };
        postcard::to_allocvec(&payload)
            .map_err(|e| InferenceError::SerializationFailed(e.to_string()))
    }

    /// Deserialize a compressed cache from bytes.
    /// Validates that the serialized layer count matches `num_layers`.
    /// `head_dim` and `num_kv_heads` are trusted inputs (from PrefillCacheHeader).
    pub fn deserialize_compressed(
        data: &[u8],
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Result<InferenceCache, InferenceError> {
        let payload: CompressedCachePayload = postcard::from_bytes(data)
            .map_err(|e| InferenceError::SerializationFailed(e.to_string()))?;

        if payload.layers.len() != num_layers {
            return Err(InferenceError::SerializationFailed(format!(
                "layer count mismatch: expected {num_layers}, got {}",
                payload.layers.len()
            )));
        }

        // Cross-validate num_kv_heads against the first populated layer's
        // vector count. Each layer stores num_kv_heads * seq_len vectors.
        if let Some(Some(layer)) = payload.layers.iter().find(|l| l.is_some()) {
            if layer.seq_len > 0 {
                let expected_vecs = num_kv_heads * layer.seq_len;
                if layer.k.len() != expected_vecs {
                    return Err(InferenceError::SerializationFailed(format!(
                        "vector count mismatch: header claims {} kv_heads × {} tokens = {}, \
                         but layer has {} k vectors",
                        num_kv_heads, layer.seq_len, expected_vecs, layer.k.len()
                    )));
                }
            }
        }

        Ok(InferenceCache {
            layers: (0..num_layers).map(|_| None).collect(),
            position: payload.position,
            num_layers,
            head_dim,
            num_kv_heads,
            compressed: payload.layers,
            is_compressed: true,
        })
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
    fn cache_with_data(
        num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        n_tokens: usize,
    ) -> InferenceCache {
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

    #[test]
    fn decompress_noop_when_uncompressed() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        assert!(cache.decompress().is_ok());
        assert!(!cache.is_compressed());
        assert!(cache.layers[0].is_some());
    }

    #[test]
    fn compress_decompress_roundtrip() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let orig_k = cache.layers[0]
            .as_ref()
            .unwrap()
            .0
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();

        cache.compress().unwrap();
        cache.decompress().unwrap();
        assert!(!cache.is_compressed());

        let (k, v) = cache.layers[0]
            .as_ref()
            .expect("layer 0 should be restored");
        assert_eq!(k.dims4().unwrap(), (1, 8, 16, 128));
        assert_eq!(v.dims4().unwrap(), (1, 8, 16, 128));
        assert_eq!(k.dtype(), DType::F16);

        let restored_k = k
            .to_dtype(DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(orig_k.len(), restored_k.len());
        let max_err: f32 = orig_k
            .iter()
            .zip(restored_k.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        // 3-bit on [0,1]: step=1/7≈0.143, max error < half step + f16 rounding
        assert!(
            max_err < 0.15,
            "max reconstruction error {max_err} too large"
        );
    }

    #[test]
    fn is_compressed_state_tracking() {
        let mut cache = cache_with_data(2, 8, 128, 4);
        assert!(!cache.is_compressed());
        cache.compress().unwrap();
        assert!(cache.is_compressed());
        cache.decompress().unwrap();
        assert!(!cache.is_compressed());
        cache.compress().unwrap();
        assert!(cache.is_compressed());
    }

    #[test]
    fn memory_bytes_empty_cache() {
        let cache = InferenceCache::new(2, 128, 8);
        assert_eq!(cache.memory_bytes(), 0);
    }

    #[test]
    fn memory_bytes_uncompressed() {
        let cache = cache_with_data(2, 8, 128, 16);
        let bytes = cache.memory_bytes();
        // Layer 0: K + V = 2 * 1 * 8 * 16 * 128 * 2 bytes = 65536
        assert_eq!(bytes, 65536);
    }

    #[test]
    fn compress_reduces_memory() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let before = cache.memory_bytes();
        cache.compress().unwrap();
        let after = cache.memory_bytes();
        assert!(
            after < before,
            "compressed {after} should be < uncompressed {before}"
        );
        let ratio = before as f64 / after as f64;
        // 3-bit uniform: 56 bytes/vec vs 256 bytes/vec ≈ 4.6x
        assert!(ratio > 3.0, "compression ratio {ratio:.1}x too low");
    }

    #[test]
    fn memory_bytes_restored_matches_original() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        let original = cache.memory_bytes();
        cache.compress().unwrap();
        cache.decompress().unwrap();
        assert_eq!(cache.memory_bytes(), original);
    }

    #[test]
    fn compress_atomic_on_error() {
        let mut cache = InferenceCache::new(2, 128, 8);

        // Layer 0: correct shape [1, 8, 4, 128] f16
        let shape_ok = (1, 8, 4, 128);
        let k0 = Tensor::rand(0f32, 1f32, shape_ok, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v0 = Tensor::rand(0f32, 1f32, shape_ok, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        cache.layers[0] = Some((k0, v0));

        // Layer 1: 3D tensor (wrong rank) — will fail dims4()
        let k1 = Tensor::rand(0f32, 1f32, (8, 4, 128), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v1 = Tensor::rand(0f32, 1f32, (8, 4, 128), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        cache.layers[1] = Some((k1, v1));

        let result = cache.compress();
        assert!(result.is_err(), "should fail on malformed tensor");

        // Atomic: cache must remain fully uncompressed
        assert!(!cache.is_compressed());
        assert!(cache.layers[0].is_some());
        assert!(cache.layers[1].is_some());
        assert!(cache.compressed.iter().all(|c| c.is_none()));
    }

    #[test]
    fn double_roundtrip_preserves_shape() {
        let mut cache = cache_with_data(2, 8, 128, 8);

        cache.compress().unwrap();
        cache.decompress().unwrap();
        cache.compress().unwrap();
        cache.decompress().unwrap();

        assert!(!cache.is_compressed());
        let (k, v) = cache.layers[0].as_ref().unwrap();
        assert_eq!(k.dims4().unwrap(), (1, 8, 8, 128));
        assert_eq!(v.dims4().unwrap(), (1, 8, 8, 128));
        assert_eq!(k.dtype(), DType::F16);
    }

    #[test]
    fn compress_with_partial_layers() {
        let mut cache = InferenceCache::new(4, 128, 8);
        let shape = (1, 8, 4, 128);
        let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        cache.layers[0] = Some((k.clone(), v.clone()));
        cache.layers[2] = Some((k, v));

        cache.compress().unwrap();
        assert!(cache.compressed[0].is_some());
        assert!(cache.compressed[1].is_none());
        assert!(cache.compressed[2].is_some());
        assert!(cache.compressed[3].is_none());

        cache.decompress().unwrap();
        assert!(cache.layers[0].is_some());
        assert!(cache.layers[1].is_none());
        assert!(cache.layers[2].is_some());
        assert!(cache.layers[3].is_none());
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();

        assert!(restored.is_compressed());
        assert_eq!(restored.position, 16);
        assert_eq!(restored.num_layers, 2);
        assert!(restored.compressed[0].is_some());
        assert!(restored.compressed[1].is_none());
    }

    #[test]
    fn serialize_uncompressed_errors() {
        let cache = cache_with_data(2, 8, 128, 16);
        let result = cache.serialize_compressed();
        assert!(matches!(result, Err(InferenceError::SerializationFailed(_))));
    }

    #[test]
    fn deserialize_validates_num_layers() {
        let mut cache = cache_with_data(2, 8, 128, 4);
        cache.compress().unwrap();
        let bytes = cache.serialize_compressed().unwrap();

        let result = InferenceCache::deserialize_compressed(&bytes, 99, 128, 8);
        assert!(matches!(result, Err(InferenceError::SerializationFailed(_))));
    }

    #[test]
    fn serialize_empty_compressed_cache() {
        let mut cache = InferenceCache::new(2, 128, 8);
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();

        assert!(restored.is_compressed());
        assert_eq!(restored.position, 0);
        assert!(restored.compressed.iter().all(|c| c.is_none()));
    }

    #[test]
    fn serialize_preserves_position() {
        let mut cache = cache_with_data(2, 8, 128, 42);
        cache.position = 42;
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();
        assert_eq!(restored.position, 42);
    }
}

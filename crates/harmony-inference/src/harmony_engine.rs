//! HarmonyEngine: inference engine wrapping HarmonyModel with UQ classification.
//!
//! Wraps [`HarmonyModel`] and implements [`InferenceEngine`]. Adds UQ
//! classification via [`HarmonyEngine::forward_full`] + [`HarmonyEngine::classify_uncertainty`].
//!
//! After initialization (`init_random` or future `load_gguf`), all inference
//! methods take `&self`. Mutable state lives in the caller-owned
//! [`InferenceCache`].

use candle_core::{Device, Tensor};
use rand::thread_rng;

use crate::engine::EngramContext;
use crate::error::InferenceError;
use crate::harmony_model::{EngramFn, HarmonyForwardOutput, HarmonyModel, HarmonyModelConfig};
use crate::uq_features::{extract_uq_features, UqFeatureConfig};
use crate::uq_head::{UqClass, UqHead};
use crate::{logits_to_vec, InferenceCache, InferenceEngine, SamplingParams};

/// Inference engine wrapping the ct87 HarmonyModel.
///
/// Stateless after initialization — all inference methods take `&self`.
/// KV cache and token history are caller-managed via [`InferenceCache`].
///
/// # Usage
///
/// ```ignore
/// let config = HarmonyModelConfig::tiny();
/// let mut engine = HarmonyEngine::new(config, Device::Cpu);
/// engine.init_random()?;
///
/// let mut cache = engine.new_cache()?;
/// let logits = engine.forward(&[1, 2, 3], &mut cache)?;
/// let next = engine.sample(&logits, &SamplingParams::greedy(), &[1, 2, 3])?;
/// ```
pub struct HarmonyEngine {
    model: Option<HarmonyModel>,
    tokenizer: Option<tokenizers::Tokenizer>,
    uq_head: Option<UqHead>,
    uq_feature_config: UqFeatureConfig,
    config: HarmonyModelConfig,
    device: Device,
}

impl HarmonyEngine {
    /// Create a new engine with the given model config and device.
    ///
    /// No model is loaded until `init_random()` (or future `load_gguf()`) is
    /// called. The UQ feature config is derived from `config.num_layers`.
    pub fn new(config: HarmonyModelConfig, device: Device) -> Self {
        let uq_feature_config = UqFeatureConfig::for_num_layers(config.num_layers);
        Self {
            model: None,
            tokenizer: None,
            uq_head: None,
            uq_feature_config,
            config,
            device,
        }
    }

    /// Initialize the model with randomly-generated weights.
    ///
    /// Useful for testing and architecture verification without trained weights.
    pub fn init_random(&mut self) -> Result<(), InferenceError> {
        let model = HarmonyModel::new(&self.config, &self.device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Set the UQ head for uncertainty classification.
    ///
    /// After setting, calls to [`Self::classify_uncertainty`] will return a
    /// classification result instead of `None`.
    pub fn set_uq_head(&mut self, uq_head: UqHead) {
        self.uq_head = Some(uq_head);
    }

    /// Reference to the device this engine targets.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Run a forward pass with Engram injection, returning logits.
    ///
    /// Behaves identically to [`InferenceEngine::forward`] but injects Engram
    /// embeddings at the specified transformer layers. This is an inherent
    /// method — Engram injection is optional and engine-specific.
    ///
    /// # Errors
    ///
    /// Returns the same errors as `forward()`, plus any tensor errors from
    /// the Engram module itself (wrapped as `ForwardFailed`).
    pub fn forward_with_engram(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
        engram: &EngramContext<'_>,
    ) -> Result<Vec<f32>, InferenceError> {
        let output = self.forward_full(tokens, cache, Some(engram))?;
        logits_to_vec(&output.logits)
    }

    /// Full forward pass returning the complete [`HarmonyForwardOutput`].
    ///
    /// Includes both logits and per-layer L2 norms needed for UQ classification.
    /// The `engram` parameter is optional — pass `None` for standard inference.
    pub fn forward_full(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
        engram: Option<&EngramContext<'_>>,
    ) -> Result<HarmonyForwardOutput, InferenceError> {
        if tokens.is_empty() {
            return Err(InferenceError::ForwardFailed(
                "tokens slice must not be empty".into(),
            ));
        }
        #[cfg(feature = "kv-compress")]
        if cache.is_compressed() {
            return Err(InferenceError::CacheCompressed);
        }
        let model = self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;

        if cache.num_layers != self.config.num_layers {
            return Err(InferenceError::CacheMismatch {
                expected: self.config.num_layers,
                actual: cache.num_layers,
            });
        }

        let seq_len = tokens.len();

        // Validate engram context.
        if let Some(ctx) = engram {
            let emb_len = ctx
                .embeddings
                .dim(1)
                .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
            if emb_len != seq_len {
                return Err(InferenceError::ForwardFailed(format!(
                    "engram embeddings have {emb_len} positions but tokens.len()={seq_len}; \
                     caller must provide embeddings matching the current token batch"
                )));
            }
            // Unlike QwenEngine (which invokes the callback at every layer and
            // relies on injection_layers for filtering), HarmonyModel only
            // invokes the callback at config.engram_injection_layer. Validate
            // that the caller's injection_layers includes that layer so the
            // intent is consistent.
            let inj = self.config.engram_injection_layer;
            if !ctx.injection_layers.contains(&inj) {
                return Err(InferenceError::ForwardFailed(format!(
                    "EngramContext.injection_layers {:?} does not contain the model's \
                     engram_injection_layer ({inj}); HarmonyEngine injects at the \
                     model-configured layer, not at injection_layers",
                    ctx.injection_layers
                )));
            }
        }

        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        let output = if let Some(ctx) = engram {
            // The model only invokes the callback at config.engram_injection_layer,
            // so no layer filtering needed here — always inject when called.
            let engram_fn =
                |_layer_idx: usize, hidden_state: &Tensor| -> candle_core::Result<Option<Tensor>> {
                    Ok(Some(ctx.module.forward(hidden_state, &ctx.embeddings)?))
                };
            model.forward(&input, cache, Some(&engram_fn as EngramFn<'_>))
        } else {
            model.forward(&input, cache, None)
        }
        .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        Ok(output)
    }

    /// Classify the uncertainty of a forward pass output.
    ///
    /// Returns `None` if no UQ head has been set (via [`set_uq_head`]).
    /// Returns `Ok(Some((class, confidence)))` otherwise.
    pub fn classify_uncertainty(
        &self,
        output: &HarmonyForwardOutput,
    ) -> Result<Option<(UqClass, f32)>, InferenceError> {
        let uq_head = match self.uq_head.as_ref() {
            Some(h) => h,
            None => return Ok(None),
        };

        let logits_vec = logits_to_vec(&output.logits)?;

        let features =
            extract_uq_features(&output.layer_norms, &logits_vec, &self.uq_feature_config)?;

        let features_tensor = Tensor::from_slice(&features, (1, 8), &self.device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        let (class, conf) = uq_head
            .classify(&features_tensor)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        Ok(Some((class, conf)))
    }
}

impl InferenceEngine for HarmonyEngine {
    fn load_gguf(&mut self, _gguf_data: &[u8]) -> Result<(), InferenceError> {
        Err(InferenceError::InvalidGguf(
            "harmony GGUF loading not yet implemented (Phase 0g)".into(),
        ))
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
        let output = self.forward_full(tokens, cache, None)?;
        logits_to_vec(&output.logits)
    }

    fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        history: &[u32],
    ) -> Result<u32, InferenceError> {
        let mut rng = thread_rng();
        // Apply repeat_last_n windowing to the caller-provided full history.
        let context = if params.repeat_last_n > 0 && params.repeat_last_n < history.len() {
            &history[history.len() - params.repeat_last_n..]
        } else {
            history
        };
        crate::sampling::sample(logits, params, context, &mut rng)
    }

    fn eos_token_id(&self) -> Option<u32> {
        let tokenizer = self.tokenizer.as_ref()?;
        for eos_str in &["<|endoftext|>", "<|im_end|>", "</s>", "<|end|>", "<eos>"] {
            if let Some(id) = tokenizer.token_to_id(eos_str) {
                return Some(id);
            }
        }
        None
    }

    fn new_cache(&self) -> Result<InferenceCache, InferenceError> {
        // Validate that a model is loaded before creating a cache.
        self.model.as_ref().ok_or(InferenceError::ModelNotLoaded)?;
        Ok(InferenceCache::new(
            self.config.num_layers,
            self.config.head_dim,
            self.config.num_kv_heads,
        ))
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Extract a `Vec<f32>` from a logits tensor.
// logits_to_vec is now shared: crate::logits_to_vec

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::uq_head::UqHeadConfig;

    fn test_config() -> HarmonyModelConfig {
        HarmonyModelConfig {
            num_layers: 4,
            hidden_dim: 32,
            num_query_heads: 4,
            num_kv_heads: 2,
            head_dim: 8,
            ffn_dim: 64,
            vocab_size: 128,
            max_seq_len: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            layers_per_block: 2,
            engram_injection_layer: 1,
            engram_dim: 16,
            tie_embeddings: true,
        }
    }

    // 1. new_engine_has_no_model
    #[test]
    fn new_engine_has_no_model() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        assert!(engine.model.is_none());
        assert!(engine.tokenizer.is_none());
        assert!(engine.uq_head.is_none());
    }

    // 2. init_random_creates_model
    #[test]
    fn init_random_creates_model() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        engine.init_random().expect("init_random should succeed");
        assert!(engine.model.is_some());
    }

    // 3. forward_without_model_errors — ModelNotLoaded
    #[test]
    fn forward_without_model_errors() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let mut cache = InferenceCache::new(4, 8, 2);
        let result = engine.forward(&[1, 2, 3], &mut cache);
        assert!(
            matches!(result, Err(InferenceError::ModelNotLoaded)),
            "expected ModelNotLoaded, got {result:?}"
        );
    }

    // 4. forward_empty_tokens_errors — ForwardFailed
    #[test]
    fn forward_empty_tokens_errors() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = engine.new_cache().unwrap();
        let result = engine.forward(&[], &mut cache);
        assert!(
            matches!(result, Err(InferenceError::ForwardFailed(_))),
            "expected ForwardFailed, got {result:?}"
        );
    }

    // 5. forward_returns_logits — length == vocab_size
    #[test]
    fn forward_returns_logits() {
        let cfg = test_config();
        let vocab_size = cfg.vocab_size;
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = engine.new_cache().unwrap();
        let logits = engine.forward(&[1, 2, 3], &mut cache).unwrap();
        assert_eq!(
            logits.len(),
            vocab_size,
            "logits length should equal vocab_size"
        );
    }

    // 6. forward_full_returns_layer_norms — length == num_layers
    #[test]
    fn forward_full_returns_layer_norms() {
        let cfg = test_config();
        let num_layers = cfg.num_layers;
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = engine.new_cache().unwrap();
        let output = engine.forward_full(&[1, 2, 3], &mut cache, None).unwrap();
        assert_eq!(
            output.layer_norms.len(),
            num_layers,
            "layer_norms length should equal num_layers"
        );
    }

    // 7. classify_uncertainty_none_without_uq_head
    #[test]
    fn classify_uncertainty_none_without_uq_head() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = engine.new_cache().unwrap();
        let output = engine.forward_full(&[1, 2, 3], &mut cache, None).unwrap();
        let result = engine.classify_uncertainty(&output).unwrap();
        assert!(
            result.is_none(),
            "classify_uncertainty should return None without UQ head"
        );
    }

    // 8. classify_uncertainty_returns_class — class in 0-3, confidence in [0,1]
    #[test]
    fn classify_uncertainty_returns_class() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();

        let uq_config = UqHeadConfig::default();
        let uq_head = UqHead::new(&uq_config, &Device::Cpu).unwrap();
        engine.set_uq_head(uq_head);

        let mut cache = engine.new_cache().unwrap();
        let output = engine.forward_full(&[1, 2, 3], &mut cache, None).unwrap();
        let result = engine.classify_uncertainty(&output).unwrap();

        let (class, conf) = result.expect("should return Some with UQ head set");
        let class_u8 = class as u8;
        assert!(class_u8 <= 3, "class discriminant {class_u8} out of 0-3");
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence {conf} out of [0, 1]"
        );
    }

    // 9. load_gguf_returns_not_implemented — InvalidGguf
    #[test]
    fn load_gguf_returns_not_implemented() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let result = engine.load_gguf(b"some bytes");
        assert!(
            matches!(result, Err(InferenceError::InvalidGguf(_))),
            "expected InvalidGguf, got {result:?}"
        );
    }

    // 10. new_cache_matches_config — num_layers, head_dim, num_kv_heads
    #[test]
    fn new_cache_matches_config() {
        let cfg = test_config();
        let num_layers = cfg.num_layers;
        let head_dim = cfg.head_dim;
        let num_kv_heads = cfg.num_kv_heads;
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        let cache = engine.new_cache().unwrap();
        assert_eq!(cache.num_layers, num_layers);
        assert_eq!(cache.head_dim, head_dim);
        assert_eq!(cache.num_kv_heads, num_kv_heads);
    }

    // 11. new_cache_without_model_errors — ModelNotLoaded
    #[test]
    fn new_cache_without_model_errors() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let result = engine.new_cache();
        assert!(
            matches!(result, Err(InferenceError::ModelNotLoaded)),
            "expected ModelNotLoaded"
        );
    }

    // 12. cache_mismatch_detected — CacheMismatch
    #[test]
    fn cache_mismatch_detected() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        // Create a cache with wrong layer count
        let mut cache = InferenceCache::new(99, 8, 2);
        let result = engine.forward(&[1, 2, 3], &mut cache);
        assert!(
            matches!(result, Err(InferenceError::CacheMismatch { expected: 4, actual: 99 })),
            "expected CacheMismatch, got {result:?}"
        );
    }

    // 13. sample_greedy_works — argmax of [1.0, 5.0, 3.0] == 1
    #[test]
    fn sample_greedy_works() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let logits = [1.0_f32, 5.0, 3.0];
        let result = engine.sample(&logits, &SamplingParams::greedy(), &[]);
        assert_eq!(result.unwrap(), 1, "greedy argmax of [1,5,3] should be 1");
    }

    // 14. full_pipeline_init_forward_classify — integration test
    #[test]
    fn full_pipeline_init_forward_classify() {
        let cfg = test_config();
        let vocab_size = cfg.vocab_size;
        let num_layers = cfg.num_layers;

        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        // init_random
        engine.init_random().expect("init_random should succeed");

        // set UQ head
        let uq_config = UqHeadConfig::default();
        let uq_head = UqHead::new(&uq_config, &Device::Cpu).unwrap();
        engine.set_uq_head(uq_head);

        // forward_full
        let mut cache = engine.new_cache().unwrap();
        let output = engine
            .forward_full(&[1, 2, 3], &mut cache, None)
            .expect("forward_full should succeed");

        // Validate output shapes
        assert_eq!(output.layer_norms.len(), num_layers);
        assert_eq!(output.logits.dims(), &[1, vocab_size]);

        // classify_uncertainty
        let uq_result = engine
            .classify_uncertainty(&output)
            .expect("classify_uncertainty should succeed");
        let (class, conf) = uq_result.expect("should return Some with UQ head set");
        let class_u8 = class as u8;
        assert!(class_u8 <= 3, "class discriminant out of range");
        assert!((0.0..=1.0).contains(&conf), "confidence out of [0, 1]");

        // Sample from logits
        let logits_vec = engine.forward(&[4u32], &mut cache).unwrap();
        assert_eq!(logits_vec.len(), vocab_size);
        let next_token = engine
            .sample(&logits_vec, &SamplingParams::greedy(), &[1, 2, 3, 4])
            .unwrap();
        // Decode (tokenizer not loaded — just verify token is in valid range)
        assert!(
            (next_token as usize) < vocab_size,
            "sampled token {next_token} out of vocab range"
        );
    }
}

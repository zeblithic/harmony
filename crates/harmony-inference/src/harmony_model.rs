//! ct87 custom model: a Qwen3-derived transformer with BlockAttnRes, Engram
//! injection, and UQ feature collection.
//!
//! This module defines the model configuration, internal layer types,
//! random-weight construction, and the full forward pass.

use crate::block_attnres::{BlockAttnRes, BlockAttnResConfig, BlockAttnResState};
use crate::InferenceCache;
use candle_core::{Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear, RmsNorm};
use std::sync::Arc;

/// Callback type for Engram injection in the HarmonyModel forward pass.
///
/// Called after each transformer layer with `(layer_index, &hidden_state)`.
/// Return `Ok(Some(residual))` to inject, or `Ok(None)` to skip this layer.
pub(crate) type EngramFn<'a> = &'a dyn Fn(usize, &Tensor) -> Result<Option<Tensor>>;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Full configuration for the ct87 HarmonyModel.
#[derive(Debug, Clone)]
pub struct HarmonyModelConfig {
    /// Total number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension of the model.
    pub hidden_dim: usize,
    /// Number of query attention heads.
    pub num_query_heads: usize,
    /// Number of key/value attention heads (GQA).
    pub num_kv_heads: usize,
    /// Per-head dimension for Q, K, V.
    pub head_dim: usize,
    /// Intermediate (gate/up/down) dimension of the MLP.
    pub ffn_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length for RoPE precomputation.
    pub max_seq_len: usize,
    /// RoPE theta frequency base.
    pub rope_theta: f64,
    /// RmsNorm epsilon.
    pub rms_norm_eps: f64,
    /// Number of transformer layers per BlockAttnRes block.
    pub layers_per_block: usize,
    /// Layer index at which Engram injection is applied.
    pub engram_injection_layer: usize,
    /// Dimension of Engram embeddings.
    pub engram_dim: usize,
    /// Whether lm_head shares the embedding weight (tied embeddings).
    pub tie_embeddings: bool,
}

impl HarmonyModelConfig {
    /// Target (production) configuration — 24-layer, 1280-hidden ct87 model.
    pub fn target() -> Self {
        Self {
            num_layers: 24,
            hidden_dim: 1280,
            num_query_heads: 16,
            num_kv_heads: 8,
            head_dim: 80,
            ffn_dim: 3413,
            vocab_size: 32000,
            max_seq_len: 32768,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            layers_per_block: 3,
            engram_injection_layer: 2,
            engram_dim: 256,
            tie_embeddings: true,
        }
    }

    /// Tiny configuration — 8-layer, 512-hidden model for fast iteration.
    pub fn tiny() -> Self {
        Self {
            num_layers: 8,
            hidden_dim: 512,
            num_query_heads: 8,
            num_kv_heads: 4,
            head_dim: 64,
            ffn_dim: 1365,
            vocab_size: 32000,
            max_seq_len: 4096,
            rope_theta: 1e6,
            rms_norm_eps: 1e-6,
            layers_per_block: 2,
            engram_injection_layer: 2,
            engram_dim: 128,
            tie_embeddings: true,
        }
    }

    /// Derive the number of BlockAttnRes blocks from the layer config.
    pub fn num_blocks(&self) -> usize {
        self.num_layers / self.layers_per_block
    }

    /// Build a [`BlockAttnResConfig`] from this model config.
    pub fn block_attnres_config(&self) -> BlockAttnResConfig {
        BlockAttnResConfig {
            num_blocks: self.num_blocks(),
            layers_per_block: self.layers_per_block,
            hidden_dim: self.hidden_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Forward output
// ---------------------------------------------------------------------------

/// Output of a single HarmonyModel forward pass.
pub struct HarmonyForwardOutput {
    /// Output logits `[batch, vocab_size]`.
    pub logits: Tensor,
    /// Per-layer L2 norms collected during the forward pass.
    pub layer_norms: Vec<f32>,
}

// ---------------------------------------------------------------------------
// Rotary embedding (f32, no dtype parameter)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        head_dim: usize,
        max_seq_len: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / head_dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(candle_core::DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Apply RoPE. q and k shape: `[batch, heads, seq_len, head_dim]`.
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    fn new(hidden_dim: usize, ffn_dim: usize, device: &Device) -> Result<Self> {
        let gate_proj = random_linear(ffn_dim, hidden_dim, device)?;
        let up_proj = random_linear(ffn_dim, hidden_dim, device)?;
        let down_proj = random_linear(hidden_dim, ffn_dim, device)?;
        Ok(Self { gate_proj, up_proj, down_proj })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head query norm (weight size = head_dim).
    q_norm: RmsNorm,
    /// Per-head key norm (weight size = head_dim).
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl Attention {
    fn new(
        hidden_dim: usize,
        num_query_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = random_linear(num_query_heads * head_dim, hidden_dim, device)?;
        let k_proj = random_linear(num_kv_heads * head_dim, hidden_dim, device)?;
        let v_proj = random_linear(num_kv_heads * head_dim, hidden_dim, device)?;
        let o_proj = random_linear(hidden_dim, num_query_heads * head_dim, device)?;

        let q_norm = random_rms_norm(head_dim, rms_norm_eps, device)?;
        let k_norm = random_rms_norm(head_dim, rms_norm_eps, device)?;

        let num_kv_groups = num_query_heads / num_kv_heads;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: num_query_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        kv_slot: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norm: flatten heads into batch dim, normalize, reshape back.
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Externalized KV cache: append to existing tensors or start fresh.
        let (k, v) = match kv_slot.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[&prev_k, &k], 2)?,
                Tensor::cat(&[&prev_v, &v], 2)?,
            ),
            None => (k, v),
        };
        *kv_slot = Some((k.clone(), v.clone()));

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask_cast = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask_cast)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

// ---------------------------------------------------------------------------
// Transformer layer
// ---------------------------------------------------------------------------

struct TransformerLayer {
    attn_norm: RmsNorm,
    attn: Attention,
    ffn_norm: RmsNorm,
    mlp: Mlp,
}

impl TransformerLayer {
    fn new(
        config: &HarmonyModelConfig,
        rotary_emb: Arc<RotaryEmbedding>,
        device: &Device,
    ) -> Result<Self> {
        let attn_norm = random_rms_norm(config.hidden_dim, config.rms_norm_eps, device)?;
        let attn = Attention::new(
            config.hidden_dim,
            config.num_query_heads,
            config.num_kv_heads,
            config.head_dim,
            config.rms_norm_eps,
            rotary_emb,
            device,
        )?;
        let ffn_norm = random_rms_norm(config.hidden_dim, config.rms_norm_eps, device)?;
        let mlp = Mlp::new(config.hidden_dim, config.ffn_dim, device)?;
        Ok(Self { attn_norm, attn, ffn_norm, mlp })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        kv_slot: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let h = self.attn_norm.forward(x)?;
        let h = self.attn.forward(&h, mask, offset, kv_slot)?;
        let x = (x + h)?;
        let h2 = self.ffn_norm.forward(&x)?;
        let h2 = self.mlp.forward(&h2)?;
        x + h2
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

/// The ct87 HarmonyModel.
pub struct HarmonyModel {
    config: HarmonyModelConfig,
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: RmsNorm,
    /// lm_head shares embedding weight when `config.tie_embeddings` is true.
    lm_head: Linear,
    block_attnres: BlockAttnRes,
    #[allow(dead_code)]
    device: Device,
}

impl HarmonyModel {
    /// Construct a HarmonyModel with randomly-initialized weights (Kaiming init).
    pub fn new(config: &HarmonyModelConfig, device: &Device) -> Result<Self> {
        let hidden_dim = config.hidden_dim;
        let vocab_size = config.vocab_size;

        // Embedding: shape [vocab_size, hidden_dim], scale = 1/sqrt(hidden_dim)
        let embed_weight = scaled_randn(&[vocab_size, hidden_dim], 1.0 / (hidden_dim as f64).sqrt(), device)?;
        let embed_tokens = Embedding::new(embed_weight.clone(), hidden_dim);

        // Shared rotary embedding
        let rotary_emb = Arc::new(RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        )?);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config, rotary_emb.clone(), device)?);
        }

        // Final layer norm
        let final_norm = random_rms_norm(hidden_dim, config.rms_norm_eps, device)?;

        // lm_head: tied or independent
        let lm_head = if config.tie_embeddings {
            // Tied: reuse the embedding weight tensor.
            // Linear stores weight as [out_features, in_features].
            // Embedding weight is [vocab_size, hidden_dim] = [out, in] for a projection
            // from hidden_dim → vocab_size. This matches the lm_head shape.
            Linear::new(embed_weight, None)
        } else {
            random_linear(vocab_size, hidden_dim, device)?
        };

        // BlockAttnRes
        let block_attnres = BlockAttnRes::new(&config.block_attnres_config(), device)?;

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            block_attnres,
            device: device.clone(),
        })
    }

    /// Reference to the model configuration.
    pub fn config(&self) -> &HarmonyModelConfig {
        &self.config
    }

    /// Reference to the BlockAttnRes module.
    pub fn block_attnres(&self) -> &BlockAttnRes {
        &self.block_attnres
    }

    /// Run one forward pass of the model.
    ///
    /// `input` is a `[1, seq_len]` u32 token-ID tensor.
    /// `cache` holds per-layer KV tensors and position offset.
    /// `engram_fn` is an optional callback called after the configured injection layer.
    pub fn forward(
        &self,
        input: &Tensor,
        cache: &mut InferenceCache,
        engram_fn: Option<EngramFn<'_>>,
    ) -> Result<HarmonyForwardOutput> {
        let (batch, seq_len) = input.dims2()?;
        let offset = cache.position;

        // Token embedding: [batch, seq_len, hidden_dim]
        let mut h = self.embed_tokens.forward(input)?;

        // Causal attention mask (only needed for prefill — decode has seq_len=1).
        let mask = if seq_len > 1 {
            Some(causal_mask(batch, seq_len, offset, input.device())?)
        } else {
            None
        };

        let mut state: BlockAttnResState = self.block_attnres.new_state();
        let layers_per_block = self.config.layers_per_block;
        let mut layer_norms: Vec<f32> = Vec::with_capacity(self.config.num_layers);

        for i in 0..self.layers.len() {
            // At block boundaries (except the very first layer) apply BlockAttnRes mixing.
            if i > 0 && i % layers_per_block == 0 {
                h = self.block_attnres.block_input(i / layers_per_block, &h, &state)?;
            }

            h = self.layers[i].forward(&h, mask.as_ref(), offset, &mut cache.layers[i])?;

            // Engram injection at the configured layer.
            if let Some(f) = engram_fn {
                if i == self.config.engram_injection_layer {
                    if let Some(residual) = f(i, &h)? {
                        h = (h + residual)?;
                    }
                }
            }

            // Record L2 norm at last sequence position.
            layer_norms.push(l2_norm_last_position(&h)?);

            // Notify BlockAttnRes of this layer's output (stores summary at block ends).
            if (i + 1) % layers_per_block == 0 {
                self.block_attnres.notify_layer_output(i, &h, &mut state)?;
            }
        }

        // Final norm → narrow to last token → lm_head logits.
        h = self.final_norm.forward(&h)?;
        // [batch, 1, hidden_dim]
        let last = h.narrow(1, seq_len - 1, 1)?;
        // [batch, hidden_dim]
        let last = last.squeeze(1)?;
        // [batch, vocab_size]
        let logits = self.lm_head.forward(&last)?;

        cache.position += seq_len;

        Ok(HarmonyForwardOutput { logits, layer_norms })
    }
}

// ---------------------------------------------------------------------------
// Forward-pass helpers
// ---------------------------------------------------------------------------

/// Causal attention mask for prefill (seq_len > 1).
///
/// Returns a `[batch, 1, tgt, tgt + offset]` f32 tensor with 0.0 for
/// attending positions and -inf for masked-out positions.
fn causal_mask(b: usize, tgt: usize, offset: usize, device: &Device) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                if j <= i + offset { 0.0 } else { minf }
            })
        })
        .collect();
    Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), device)
}

/// L2 norm of the hidden state at the last sequence position.
fn l2_norm_last_position(h: &Tensor) -> Result<f32> {
    let (_, seq_len, _) = h.dims3()?;
    let last = h.narrow(1, seq_len - 1, 1)?;
    last.sqr()?.sum_all()?.sqrt()?.to_scalar()
}

// ---------------------------------------------------------------------------
// Utility: repeat_kv (copied from qwen3_ext for GQA)
// ---------------------------------------------------------------------------

/// Repeats key/value tensors for grouped query attention.
/// Input shape: `(batch, num_kv_heads, seq_len, head_dim)`.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// Weight construction helpers
// ---------------------------------------------------------------------------

/// Create a tensor of random normal values scaled by `scale`.
fn scaled_randn(shape: &[usize], scale: f64, device: &Device) -> Result<Tensor> {
    let t = Tensor::randn(0f32, 1f32, shape, device)?;
    t * scale
}

/// Create a [`Linear`] layer with random Kaiming init (no bias).
///
/// Weight shape: `[out_features, in_features]`.
/// Scale: `1 / sqrt(in_features)`.
fn random_linear(out_features: usize, in_features: usize, device: &Device) -> Result<Linear> {
    let scale = 1.0 / (in_features as f64).sqrt();
    let weight = scaled_randn(&[out_features, in_features], scale, device)?;
    Ok(Linear::new(weight, None))
}

/// Create an [`RmsNorm`] with ones weight (standard init).
fn random_rms_norm(size: usize, eps: f64, device: &Device) -> Result<RmsNorm> {
    let weight = Tensor::ones(&[size], candle_core::DType::F32, device)?;
    Ok(RmsNorm::new(weight, eps))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InferenceCache;
    use candle_core::Device;

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

    #[test]
    fn target_config_values() {
        let c = HarmonyModelConfig::target();
        assert_eq!(c.num_layers, 24);
        assert_eq!(c.hidden_dim, 1280);
        assert_eq!(c.num_query_heads, 16);
        assert_eq!(c.num_kv_heads, 8);
        assert_eq!(c.head_dim, 80);
        assert_eq!(c.ffn_dim, 3413);
        assert_eq!(c.vocab_size, 32000);
        assert_eq!(c.max_seq_len, 32768);
        assert!((c.rope_theta - 1e6).abs() < 1.0);
        assert!((c.rms_norm_eps - 1e-6).abs() < 1e-12);
        assert_eq!(c.layers_per_block, 3);
        assert_eq!(c.engram_injection_layer, 2);
        assert_eq!(c.engram_dim, 256);
        assert!(c.tie_embeddings);
    }

    #[test]
    fn tiny_config_values() {
        let c = HarmonyModelConfig::tiny();
        assert_eq!(c.num_layers, 8);
        assert_eq!(c.hidden_dim, 512);
        assert_eq!(c.num_query_heads, 8);
        assert_eq!(c.num_kv_heads, 4);
        assert_eq!(c.head_dim, 64);
        assert_eq!(c.ffn_dim, 1365);
        assert_eq!(c.engram_dim, 128);
        assert_eq!(c.layers_per_block, 2);
        assert!(c.tie_embeddings);
    }

    #[test]
    fn num_blocks_correct() {
        // target: 24 layers / 3 per block = 8 blocks
        assert_eq!(HarmonyModelConfig::target().num_blocks(), 8);
        // tiny: 8 layers / 2 per block = 4 blocks
        assert_eq!(HarmonyModelConfig::tiny().num_blocks(), 4);
        // test_config: 4 layers / 2 per block = 2 blocks
        assert_eq!(test_config().num_blocks(), 2);
    }

    #[test]
    fn block_attnres_config_derived() {
        let c = HarmonyModelConfig::target();
        let bac = c.block_attnres_config();
        assert_eq!(bac.num_blocks, 8);
        assert_eq!(bac.layers_per_block, 3);
        assert_eq!(bac.hidden_dim, 1280);
        assert_eq!(bac.total_layers(), 24);
    }

    #[test]
    fn model_constructs_with_test_config() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).expect("model construction failed");
        assert_eq!(model.layers.len(), cfg.num_layers);
    }

    #[test]
    fn model_block_attnres_query_count() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).expect("model construction failed");
        // test_config: 2 blocks → 1 boundary query
        let state = model.block_attnres().new_state();
        assert!(state.summaries.is_empty());
    }

    // -----------------------------------------------------------------------
    // Forward pass tests (Task 3)
    // -----------------------------------------------------------------------

    fn make_input_3tokens() -> candle_core::Result<Tensor> {
        Tensor::new(&[1u32, 2, 3], &Device::Cpu)?.reshape((1, 3))
    }

    fn make_input_1token() -> candle_core::Result<Tensor> {
        Tensor::new(&[4u32], &Device::Cpu)?.reshape((1, 1))
    }

    #[test]
    fn forward_output_shape() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = make_input_3tokens().unwrap();
        let out = model.forward(&input, &mut cache, None).unwrap();
        // logits must be [1, vocab_size]
        assert_eq!(out.logits.dims(), &[1, cfg.vocab_size]);
    }

    #[test]
    fn forward_layer_norms_length() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = make_input_3tokens().unwrap();
        let out = model.forward(&input, &mut cache, None).unwrap();
        assert_eq!(out.layer_norms.len(), cfg.num_layers);
    }

    #[test]
    fn forward_advances_cache_position() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);

        // Prefill 3 tokens → position should be 3
        let input3 = make_input_3tokens().unwrap();
        model.forward(&input3, &mut cache, None).unwrap();
        assert_eq!(cache.position, 3);

        // Decode 1 token → position should be 4
        let input1 = make_input_1token().unwrap();
        model.forward(&input1, &mut cache, None).unwrap();
        assert_eq!(cache.position, 4);
    }

    #[test]
    fn forward_with_engram_modifies_output() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let input = make_input_3tokens().unwrap();

        // Baseline: no engram
        let mut cache_base = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let out_base = model.forward(&input, &mut cache_base, None).unwrap();

        // With engram injection: inject a ones residual at the injection layer
        let hidden_dim = cfg.hidden_dim;
        let engram_fn: EngramFn<'_> = &|_layer_idx: usize, h: &Tensor| {
            let shape = h.dims().to_vec();
            Ok(Some(Tensor::ones(shape.as_slice(), candle_core::DType::F32, h.device())?))
        };
        let mut cache_engram = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let out_engram = model
            .forward(&input, &mut cache_engram, Some(engram_fn))
            .unwrap();

        // logits should differ because we injected a residual
        let base_vals: Vec<f32> = out_base.logits.flatten_all().unwrap().to_vec1().unwrap();
        let eng_vals: Vec<f32> = out_engram.logits.flatten_all().unwrap().to_vec1().unwrap();
        let max_diff = base_vals
            .iter()
            .zip(eng_vals.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff > 1e-6,
            "logits should differ with engram injection; max_diff={max_diff}"
        );
        let _ = hidden_dim; // silence unused warning
    }

    #[test]
    fn forward_layer_norms_are_positive() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = make_input_3tokens().unwrap();
        let out = model.forward(&input, &mut cache, None).unwrap();
        for (i, &norm) in out.layer_norms.iter().enumerate() {
            assert!(norm >= 0.0, "layer {i} norm should be non-negative, got {norm}");
        }
    }

    #[test]
    fn forward_populates_kv_cache_slots() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = make_input_3tokens().unwrap(); // seq_len = 3

        model.forward(&input, &mut cache, None).unwrap();

        // All layers should have a populated KV slot with shape [1, num_kv_heads, 3, head_dim]
        for (i, slot) in cache.layers.iter().enumerate() {
            let (k, v) = slot
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} KV slot should be populated"));
            assert_eq!(
                k.dims(),
                &[1, cfg.num_kv_heads, 3, cfg.head_dim],
                "layer {i} K shape mismatch"
            );
            assert_eq!(
                v.dims(),
                &[1, cfg.num_kv_heads, 3, cfg.head_dim],
                "layer {i} V shape mismatch"
            );
        }
    }

    #[test]
    fn forward_decode_extends_kv_cache() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);

        // Prefill 3 tokens
        let input3 = make_input_3tokens().unwrap();
        model.forward(&input3, &mut cache, None).unwrap();

        // Decode 1 more token
        let input1 = make_input_1token().unwrap();
        model.forward(&input1, &mut cache, None).unwrap();

        // KV cache seq_len should now be 4
        for (i, slot) in cache.layers.iter().enumerate() {
            let (k, _v) = slot
                .as_ref()
                .unwrap_or_else(|| panic!("layer {i} KV slot should be populated"));
            assert_eq!(
                k.dims(),
                &[1, cfg.num_kv_heads, 4, cfg.head_dim],
                "layer {i} K should have seq_len=4 after prefill+decode"
            );
        }
    }
}

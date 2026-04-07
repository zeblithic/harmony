# Phase 0e: Custom Model Forward Pass — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define the ct87 custom model architecture in candle and implement a forward pass that natively orchestrates Block AttnRes, Engram injection, and UQ feature collection.

**Architecture:** Three new files — `uq_features.rs` (pure-math feature extraction), `harmony_model.rs` (model definition with transformer layers + BlockAttnRes + Engram callback), `harmony_engine.rs` (engine wrapping model + UQ head). The model uses non-quantized `candle_nn::Linear` with random weight init for testing before GGUF exists (Phase 0g).

**Tech Stack:** Rust, candle-core, candle-nn (Linear, Embedding, RmsNorm, rotary_emb)

**Spec:** `docs/superpowers/specs/2026-04-06-phase0e-custom-model-design.md`

---

## File Structure

| File | Responsibility |
|------|----------------|
| Create: `crates/harmony-inference/src/uq_features.rs` | `UqFeatureConfig`, `extract_uq_features()`, entropy/slope/top-k math |
| Create: `crates/harmony-inference/src/harmony_model.rs` | `HarmonyModelConfig`, `HarmonyModel`, `HarmonyForwardOutput`, internal layer types |
| Create: `crates/harmony-inference/src/harmony_engine.rs` | `HarmonyEngine` implementing `InferenceEngine`, `forward_full`, `classify_uncertainty` |
| Modify: `crates/harmony-inference/src/lib.rs` | Add `pub mod` + re-exports for new modules |

---

### Task 1: UQ Feature Extraction

**Files:**
- Create: `crates/harmony-inference/src/uq_features.rs`

- [ ] **Step 1: Write the failing tests**

Add the complete `uq_features.rs` file with types, a stub `extract_uq_features` that returns an error, and all tests:

```rust
//! UQ feature extraction — pure math on floats.
//!
//! Takes raw data from a forward pass (per-layer L2 norms + output logits)
//! and produces the 8-feature vector the UQ Head expects.
//! No tensors, no device — deterministic and easy to test.

use crate::error::InferenceError;

/// Configuration for UQ feature extraction.
#[derive(Debug, Clone)]
pub struct UqFeatureConfig {
    /// Which layer indices to sample L2 norms from (4 layers).
    pub norm_layers: [usize; 4],
    /// Number of top tokens for probability mass feature (f7).
    pub top_k_for_mass: usize,
}

impl Default for UqFeatureConfig {
    fn default() -> Self {
        Self {
            norm_layers: [0, 8, 16, 23],
            top_k_for_mass: 10,
        }
    }
}

impl UqFeatureConfig {
    /// Config for the tiny model (8 layers).
    pub fn tiny() -> Self {
        Self {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 10,
        }
    }

    /// Derive config for an arbitrary number of layers.
    /// Picks 4 evenly-spaced layer indices including first and last.
    pub fn for_num_layers(num_layers: usize) -> Self {
        match num_layers {
            24 => Self::default(),
            8 => Self::tiny(),
            n => {
                let last = n.saturating_sub(1);
                let step = if n >= 4 { last / 3 } else { 1 };
                Self {
                    norm_layers: [0, step.min(last), (2 * step).min(last), last],
                    top_k_for_mass: 10,
                }
            }
        }
    }
}

/// Extract 8 UQ features from forward pass outputs.
///
/// Features:
/// - f1-f4: L2 norms at `config.norm_layers`
/// - f5: Linear regression slope over f1-f4
/// - f6: Shannon entropy of logits (after softmax)
/// - f7: Top-k probability mass (after softmax)
/// - f8: Stubbed as 0.0 (attention lookback ratio, deferred)
pub fn extract_uq_features(
    layer_norms: &[f32],
    logits: &[f32],
    config: &UqFeatureConfig,
) -> Result<[f32; 8], InferenceError> {
    // TODO: implement
    Err(InferenceError::ForwardFailed("not implemented".into()))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    vec![]
}

fn shannon_entropy(logits: &[f32]) -> f32 {
    0.0
}

fn top_k_mass(logits: &[f32], k: usize) -> f32 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> UqFeatureConfig {
        UqFeatureConfig {
            norm_layers: [0, 2, 5, 7],
            top_k_for_mass: 3,
        }
    }

    #[test]
    fn entropy_uniform_is_maximal() {
        let logits = vec![0.0f32; 100];
        let entropy = shannon_entropy(&logits);
        let expected = (100.0f32).ln();
        assert!(
            (entropy - expected).abs() < 1e-4,
            "entropy={entropy}, expected={expected}"
        );
    }

    #[test]
    fn entropy_one_hot_is_zero() {
        let mut logits = vec![-100.0f32; 100];
        logits[0] = 100.0;
        let entropy = shannon_entropy(&logits);
        assert!(entropy < 1e-4, "one-hot entropy should be ~0, got {entropy}");
    }

    #[test]
    fn top_k_mass_peaked() {
        let logits = vec![10.0f32, 0.0, 0.0, 0.0];
        let mass = top_k_mass(&logits, 1);
        assert!(mass > 0.99, "top-1 of peaked dist should be ~1.0, got {mass}");
    }

    #[test]
    fn top_k_mass_uniform() {
        let logits = vec![0.0f32; 10];
        let mass = top_k_mass(&logits, 3);
        let expected = 3.0 / 10.0;
        assert!(
            (mass - expected).abs() < 1e-4,
            "mass={mass}, expected={expected}"
        );
    }

    #[test]
    fn slope_positive_for_increasing_norms() {
        let config = test_config();
        let layer_norms = vec![1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 0.0, 4.0];
        let logits = vec![0.0; 10];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert!(
            features[4] > 0.0,
            "slope should be positive, got {}",
            features[4]
        );
    }

    #[test]
    fn slope_negative_for_decreasing_norms() {
        let config = test_config();
        let layer_norms = vec![4.0, 0.0, 3.0, 0.0, 0.0, 2.0, 0.0, 1.0];
        let logits = vec![0.0; 10];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert!(
            features[4] < 0.0,
            "slope should be negative, got {}",
            features[4]
        );
    }

    #[test]
    fn slope_zero_for_constant_norms() {
        let config = test_config();
        let layer_norms = vec![5.0, 0.0, 5.0, 0.0, 0.0, 5.0, 0.0, 5.0];
        let logits = vec![0.0; 10];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert!(
            features[4].abs() < 1e-6,
            "slope should be zero, got {}",
            features[4]
        );
    }

    #[test]
    fn f8_always_zero() {
        let config = test_config();
        let layer_norms = vec![1.0; 8];
        let logits = vec![1.0, 2.0, 3.0];
        let features = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert_eq!(features[7], 0.0, "f8 should be stubbed as 0.0");
    }

    #[test]
    fn norm_layers_out_of_bounds_returns_error() {
        let config = UqFeatureConfig {
            norm_layers: [0, 1, 2, 99],
            top_k_for_mass: 3,
        };
        let layer_norms = vec![1.0; 8];
        let logits = vec![1.0];
        let result = extract_uq_features(&layer_norms, &logits, &config);
        assert!(result.is_err());
    }

    #[test]
    fn features_are_deterministic() {
        let config = test_config();
        let layer_norms = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let logits = vec![1.0, 2.0, 3.0, 4.0];
        let f1 = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        let f2 = extract_uq_features(&layer_norms, &logits, &config).unwrap();
        assert_eq!(f1, f2);
    }

    #[test]
    fn for_num_layers_target() {
        let cfg = UqFeatureConfig::for_num_layers(24);
        assert_eq!(cfg.norm_layers, [0, 8, 16, 23]);
    }

    #[test]
    fn for_num_layers_tiny() {
        let cfg = UqFeatureConfig::for_num_layers(8);
        assert_eq!(cfg.norm_layers, [0, 2, 5, 7]);
    }

    #[test]
    fn for_num_layers_4() {
        let cfg = UqFeatureConfig::for_num_layers(4);
        assert_eq!(cfg.norm_layers, [0, 1, 2, 3]);
    }
}
```

- [ ] **Step 2: Add module to lib.rs and run tests to verify they fail**

Add to `crates/harmony-inference/src/lib.rs` after the existing module declarations (after line 29):

```rust
pub mod uq_features;
```

And add the re-export after the existing re-exports (after line 36):

```rust
pub use uq_features::{extract_uq_features, UqFeatureConfig};
```

Run:
```bash
cargo test -p harmony-inference uq_features
```

Expected: 13 FAILED (stub returns error, helpers return wrong values).

- [ ] **Step 3: Implement the feature extraction**

Replace the stub functions in `crates/harmony-inference/src/uq_features.rs` with the real implementations. Replace `extract_uq_features`, `softmax`, `shannon_entropy`, and `top_k_mass`:

```rust
pub fn extract_uq_features(
    layer_norms: &[f32],
    logits: &[f32],
    config: &UqFeatureConfig,
) -> Result<[f32; 8], InferenceError> {
    for &idx in &config.norm_layers {
        if idx >= layer_norms.len() {
            return Err(InferenceError::ForwardFailed(format!(
                "norm_layers index {idx} out of bounds (layer_norms.len()={})",
                layer_norms.len()
            )));
        }
    }

    let f1 = layer_norms[config.norm_layers[0]];
    let f2 = layer_norms[config.norm_layers[1]];
    let f3 = layer_norms[config.norm_layers[2]];
    let f4 = layer_norms[config.norm_layers[3]];

    // f5: least-squares slope over 4 equally-spaced points at indices 0,1,2,3
    let f5 = (3.0 * (f4 - f1) + (f3 - f2)) / 10.0;

    let f6 = shannon_entropy(logits);
    let f7 = top_k_mass(logits, config.top_k_for_mass);
    let f8 = 0.0; // attention lookback ratio — stubbed

    Ok([f1, f2, f3, f4, f5, f6, f7, f8])
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        let uniform = 1.0 / logits.len() as f32;
        return vec![uniform; logits.len()];
    }
    exps.iter().map(|&e| e / sum).collect()
}

fn shannon_entropy(logits: &[f32]) -> f32 {
    let probs = softmax(logits);
    let mut entropy = 0.0f32;
    for &p in &probs {
        if p > 1e-10 {
            entropy -= p * p.ln();
        }
    }
    entropy
}

fn top_k_mass(logits: &[f32], k: usize) -> f32 {
    let probs = softmax(logits);
    let mut sorted = probs;
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    sorted.iter().take(k).sum()
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-inference uq_features
```

Expected: 13 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/uq_features.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add UQ feature extraction (Phase 0e, Task 1)"
```

---

### Task 2: HarmonyModel Config + Construction

**Files:**
- Create: `crates/harmony-inference/src/harmony_model.rs`

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-inference/src/harmony_model.rs` with the full type definitions, a stub `HarmonyModel::new()` that always errors, and all config + construction tests. The forward method is NOT included yet (Task 3).

```rust
//! Custom ct87 model definition — transformer backbone with BlockAttnRes,
//! Engram injection point, and UQ feature collection.
//!
//! Uses non-quantized candle_nn types (Linear, Embedding, RmsNorm) with
//! random weight init for testing. GGUF loading deferred to Phase 0g.

use std::sync::Arc;

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Embedding, Linear};

use crate::block_attnres::{BlockAttnRes, BlockAttnResConfig, BlockAttnResState};
use crate::InferenceCache;

// ── Callback type (same signature as qwen3_ext::EngramFn) ──────────────

/// Engram injection callback. Called after each layer with (layer_idx, &hidden_state).
/// Return `Ok(Some(residual))` to inject, `Ok(None)` to skip.
pub(crate) type EngramFn<'a> = &'a dyn Fn(usize, &Tensor) -> Result<Option<Tensor>>;

// ── Configuration ──────────────────────────────────────────────────────

/// Configuration for the ct87 custom model.
#[derive(Debug, Clone)]
pub struct HarmonyModelConfig {
    pub num_layers: usize,
    pub hidden_dim: usize,
    pub num_query_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub ffn_dim: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub rope_theta: f64,
    pub rms_norm_eps: f32,
    pub layers_per_block: usize,
    pub engram_injection_layer: usize,
    pub engram_dim: usize,
    pub tie_embeddings: bool,
}

impl HarmonyModelConfig {
    /// Full 471M target configuration.
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
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            layers_per_block: 3,
            engram_injection_layer: 2,
            engram_dim: 256,
            tie_embeddings: true,
        }
    }

    /// ~50M tiny configuration for local experiments.
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
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            layers_per_block: 2,
            engram_injection_layer: 2,
            engram_dim: 128,
            tie_embeddings: true,
        }
    }

    /// Number of blocks (for Block AttnRes).
    pub fn num_blocks(&self) -> usize {
        self.num_layers / self.layers_per_block
    }

    /// Derive BlockAttnResConfig from this model config.
    pub fn block_attnres_config(&self) -> BlockAttnResConfig {
        BlockAttnResConfig {
            num_blocks: self.num_blocks(),
            layers_per_block: self.layers_per_block,
            hidden_dim: self.hidden_dim,
        }
    }
}

// ── Forward output ─────────────────────────────────────────────────────

/// Output of `HarmonyModel::forward()`.
pub struct HarmonyForwardOutput {
    /// Logits for the last position. Shape: `[1, vocab_size]`.
    pub logits: Tensor,
    /// L2 norm of hidden state at each layer (length = num_layers).
    /// Computed at the last sequence position.
    pub layer_norms: Vec<f32>,
}

// ── Internal layer types ───────────────────────────────────────────────

struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: candle_nn::RmsNorm,
    k_norm: candle_nn::RmsNorm,
    rotary_emb: Arc<RotaryEmbedding>,
    num_query_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
}

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

struct TransformerLayer {
    attn_norm: candle_nn::RmsNorm,
    attn: Attention,
    ffn_norm: candle_nn::RmsNorm,
    mlp: Mlp,
}

// ── Model ──────────────────────────────────────────────────────────────

/// The ct87 custom model.
pub struct HarmonyModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: candle_nn::RmsNorm,
    lm_head: Linear,
    block_attnres: BlockAttnRes,
    config: HarmonyModelConfig,
}

impl HarmonyModel {
    /// Create with random weights (for testing before GGUF exists).
    pub fn new(config: &HarmonyModelConfig, device: &Device) -> Result<Self> {
        candle_core::bail!("not implemented yet")
    }

    /// Reference to the model config.
    pub fn config(&self) -> &HarmonyModelConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let cfg = HarmonyModelConfig::target();
        assert_eq!(cfg.num_layers, 24);
        assert_eq!(cfg.hidden_dim, 1280);
        assert_eq!(cfg.num_query_heads, 16);
        assert_eq!(cfg.num_kv_heads, 8);
        assert_eq!(cfg.head_dim, 80);
        assert_eq!(cfg.ffn_dim, 3413);
        assert_eq!(cfg.layers_per_block, 3);
        assert_eq!(cfg.engram_injection_layer, 2);
        assert_eq!(cfg.engram_dim, 256);
        assert!(cfg.tie_embeddings);
    }

    #[test]
    fn tiny_config_values() {
        let cfg = HarmonyModelConfig::tiny();
        assert_eq!(cfg.num_layers, 8);
        assert_eq!(cfg.hidden_dim, 512);
        assert_eq!(cfg.num_query_heads, 8);
        assert_eq!(cfg.num_kv_heads, 4);
        assert_eq!(cfg.head_dim, 64);
        assert_eq!(cfg.ffn_dim, 1365);
        assert_eq!(cfg.layers_per_block, 2);
    }

    #[test]
    fn num_blocks_correct() {
        assert_eq!(HarmonyModelConfig::target().num_blocks(), 8);
        assert_eq!(HarmonyModelConfig::tiny().num_blocks(), 4);
        assert_eq!(test_config().num_blocks(), 2);
    }

    #[test]
    fn block_attnres_config_derived() {
        let cfg = test_config();
        let attnres_cfg = cfg.block_attnres_config();
        assert_eq!(attnres_cfg.num_blocks, 2);
        assert_eq!(attnres_cfg.layers_per_block, 2);
        assert_eq!(attnres_cfg.hidden_dim, 32);
    }

    #[test]
    fn model_constructs_with_test_config() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        assert_eq!(model.layers.len(), 4);
        assert_eq!(model.config().num_layers, 4);
    }

    #[test]
    fn model_block_attnres_query_count() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        // 2 blocks → 1 boundary → 1 query vector
        // Verified via block_attnres internals (it was constructed with correct config)
        let state = model.block_attnres.new_state();
        assert!(state.summaries.is_empty());
    }
}
```

- [ ] **Step 2: Add module to lib.rs and run tests to verify they fail**

Add to `crates/harmony-inference/src/lib.rs` after existing module declarations:

```rust
pub mod harmony_model;
```

And add re-exports:

```rust
pub use harmony_model::{HarmonyModel, HarmonyModelConfig, HarmonyForwardOutput};
```

Run:
```bash
cargo test -p harmony-inference harmony_model
```

Expected: `model_constructs_with_test_config` and `model_block_attnres_query_count` FAIL (stub returns error). Config tests pass.

- [ ] **Step 3: Implement model construction**

Replace `HarmonyModel::new()` and add all the internal `new()` methods for `RotaryEmbedding`, `Attention`, `Mlp`, `TransformerLayer`, and the `repeat_kv` helper. Replace the stub `new()` and add the implementations above the `impl HarmonyModel` block:

```rust
impl RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, rope_theta: f64, device: &Device) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0f32 / (rope_theta as f32).powf(i as f32 / head_dim as f32))
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(DType::F32)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

impl Attention {
    fn new(config: &HarmonyModelConfig, rotary_emb: Arc<RotaryEmbedding>, device: &Device) -> Result<Self> {
        let h = config.hidden_dim;
        let qh = config.num_query_heads;
        let kvh = config.num_kv_heads;
        let d = config.head_dim;
        let scale_h = (1.0 / (h as f64).sqrt());

        let q_proj = Linear::new(
            (Tensor::randn(0f32, 1f32, (qh * d, h), device)? * scale_h)?,
            None,
        );
        let k_proj = Linear::new(
            (Tensor::randn(0f32, 1f32, (kvh * d, h), device)? * scale_h)?,
            None,
        );
        let v_proj = Linear::new(
            (Tensor::randn(0f32, 1f32, (kvh * d, h), device)? * scale_h)?,
            None,
        );
        let o_proj = Linear::new(
            (Tensor::randn(0f32, 1f32, (h, qh * d), device)? * (1.0 / ((qh * d) as f64).sqrt()))?,
            None,
        );

        let q_norm = candle_nn::RmsNorm::new(
            Tensor::ones(d, DType::F32, device)?,
            config.rms_norm_eps as f64,
        );
        let k_norm = candle_nn::RmsNorm::new(
            Tensor::ones(d, DType::F32, device)?,
            config.rms_norm_eps as f64,
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            rotary_emb,
            num_query_heads: qh,
            num_kv_heads: kvh,
            num_kv_groups: qh / kvh,
            head_dim: d,
        })
    }
}

impl Mlp {
    fn new(config: &HarmonyModelConfig, device: &Device) -> Result<Self> {
        let h = config.hidden_dim;
        let f = config.ffn_dim;
        let scale = (1.0 / (h as f64).sqrt());

        Ok(Self {
            gate_proj: Linear::new(
                (Tensor::randn(0f32, 1f32, (f, h), device)? * scale)?,
                None,
            ),
            up_proj: Linear::new(
                (Tensor::randn(0f32, 1f32, (f, h), device)? * scale)?,
                None,
            ),
            down_proj: Linear::new(
                (Tensor::randn(0f32, 1f32, (h, f), device)? * (1.0 / (f as f64).sqrt()))?,
                None,
            ),
        })
    }
}

impl TransformerLayer {
    fn new(config: &HarmonyModelConfig, rotary_emb: Arc<RotaryEmbedding>, device: &Device) -> Result<Self> {
        Ok(Self {
            attn_norm: candle_nn::RmsNorm::new(
                Tensor::ones(config.hidden_dim, DType::F32, device)?,
                config.rms_norm_eps as f64,
            ),
            attn: Attention::new(config, rotary_emb, device)?,
            ffn_norm: candle_nn::RmsNorm::new(
                Tensor::ones(config.hidden_dim, DType::F32, device)?,
                config.rms_norm_eps as f64,
            ),
            mlp: Mlp::new(config, device)?,
        })
    }
}

/// Repeat KV heads for grouped query attention.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?
            .reshape((b, n_kv_head * n_rep, seq_len, head_dim))
    }
}
```

And replace the `HarmonyModel::new()` stub:

```rust
impl HarmonyModel {
    pub fn new(config: &HarmonyModelConfig, device: &Device) -> Result<Self> {
        let embed_scale = 1.0 / (config.hidden_dim as f64).sqrt();
        let embed_weight = (Tensor::randn(
            0f32,
            1f32,
            (config.vocab_size, config.hidden_dim),
            device,
        )? * embed_scale)?;
        let embed_tokens = Embedding::new(embed_weight.clone(), config.hidden_dim);

        let rotary_emb = Arc::new(RotaryEmbedding::new(
            config.head_dim,
            config.max_seq_len,
            config.rope_theta,
            device,
        )?);

        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(TransformerLayer::new(config, rotary_emb.clone(), device)?);
        }

        let final_norm = candle_nn::RmsNorm::new(
            Tensor::ones(config.hidden_dim, DType::F32, device)?,
            config.rms_norm_eps as f64,
        );

        let lm_head = if config.tie_embeddings {
            Linear::new(embed_weight, None)
        } else {
            let lm_weight = (Tensor::randn(
                0f32,
                1f32,
                (config.vocab_size, config.hidden_dim),
                device,
            )? * embed_scale)?;
            Linear::new(lm_weight, None)
        };

        let block_attnres = BlockAttnRes::new(&config.block_attnres_config(), device)?;

        Ok(Self {
            embed_tokens,
            layers,
            final_norm,
            lm_head,
            block_attnres,
            config: config.clone(),
        })
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-inference harmony_model
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/harmony_model.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add HarmonyModel config and construction (Phase 0e, Task 2)"
```

---

### Task 3: HarmonyModel Forward Pass

**Files:**
- Modify: `crates/harmony-inference/src/harmony_model.rs`

- [ ] **Step 1: Write the failing tests**

Add forward-pass tests to the existing `tests` module in `harmony_model.rs`:

```rust
    #[test]
    fn forward_output_shape() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        let output = model.forward(&input, &mut cache, None).unwrap();
        // Logits shape: [1, vocab_size]
        assert_eq!(output.logits.dims(), &[1, cfg.vocab_size]);
    }

    #[test]
    fn forward_layer_norms_length() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let output = model.forward(&input, &mut cache, None).unwrap();
        assert_eq!(output.layer_norms.len(), cfg.num_layers);
    }

    #[test]
    fn forward_advances_cache_position() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        assert_eq!(cache.position, 0);

        let input = Tensor::new(&[1u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        model.forward(&input, &mut cache, None).unwrap();
        assert_eq!(cache.position, 3);

        // Decode step
        let input2 = Tensor::new(&[4u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        model.forward(&input2, &mut cache, None).unwrap();
        assert_eq!(cache.position, 4);
    }

    #[test]
    fn forward_with_engram_modifies_output() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();

        let input = Tensor::new(&[1u32, 2], &Device::Cpu)
            .unwrap()
            .reshape((1, 2))
            .unwrap();

        // Without engram
        let mut cache1 = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let out1 = model.forward(&input, &mut cache1, None).unwrap();

        // With engram: inject a large residual at the injection layer
        let mut cache2 = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let engram_fn = |layer_idx: usize, _h: &Tensor| -> Result<Option<Tensor>> {
            if layer_idx == cfg.engram_injection_layer {
                Ok(Some(Tensor::ones(
                    (1, 2, cfg.hidden_dim),
                    DType::F32,
                    &Device::Cpu,
                )?))
            } else {
                Ok(None)
            }
        };
        let out2 = model.forward(&input, &mut cache2, Some(&engram_fn)).unwrap();

        // Logits should differ
        let diff: f32 = (&out1.logits - &out2.logits)
            .unwrap()
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(diff > 1e-6, "engram injection should change logits, diff={diff}");
    }

    #[test]
    fn forward_layer_norms_are_positive() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        let output = model.forward(&input, &mut cache, None).unwrap();
        for (i, &norm) in output.layer_norms.iter().enumerate() {
            assert!(norm >= 0.0, "layer_norm[{i}] should be non-negative, got {norm}");
        }
    }

    #[test]
    fn forward_populates_kv_cache_slots() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let input = Tensor::new(&[1u32, 2], &Device::Cpu)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        model.forward(&input, &mut cache, None).unwrap();

        // All layers should now have KV tensors
        for (i, slot) in cache.layers.iter().enumerate() {
            assert!(slot.is_some(), "layer {i} KV slot should be populated");
            let (k, v) = slot.as_ref().unwrap();
            assert_eq!(k.dims4().unwrap(), (1, cfg.num_kv_heads, 2, cfg.head_dim));
            assert_eq!(v.dims4().unwrap(), (1, cfg.num_kv_heads, 2, cfg.head_dim));
        }
    }

    #[test]
    fn forward_decode_extends_kv_cache() {
        let cfg = test_config();
        let model = HarmonyModel::new(&cfg, &Device::Cpu).unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);

        // Prefill 3 tokens
        let input1 = Tensor::new(&[1u32, 2, 3], &Device::Cpu)
            .unwrap()
            .reshape((1, 3))
            .unwrap();
        model.forward(&input1, &mut cache, None).unwrap();

        // Decode 1 token
        let input2 = Tensor::new(&[4u32], &Device::Cpu)
            .unwrap()
            .reshape((1, 1))
            .unwrap();
        model.forward(&input2, &mut cache, None).unwrap();

        // KV cache should now have seq_len=4
        let (k, _) = cache.layers[0].as_ref().unwrap();
        assert_eq!(k.dim(2).unwrap(), 4);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cargo test -p harmony-inference harmony_model
```

Expected: The new forward tests fail (no `forward` method exists yet). Config/construction tests still pass.

- [ ] **Step 3: Implement the forward pass**

Add the `forward` method to `HarmonyModel`, plus layer forward methods and helpers. Add the `Attention::forward`, `TransformerLayer::forward`, causal mask, and L2 norm helpers.

Add the `Attention::forward` method inside the existing `impl Attention` block (or add a new impl block):

```rust
impl Attention {
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
            .reshape((b, l, self.num_query_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Per-head QK norm (Qwen3 pattern)
        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_query_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // KV cache: append to existing or start fresh
        let (k, v) = match kv_slot.take() {
            Some((prev_k, prev_v)) => (
                Tensor::cat(&[&prev_k, &k], 2)?,
                Tensor::cat(&[&prev_v, &v], 2)?,
            ),
            None => (k, v),
        };
        *kv_slot = Some((k.clone(), v.clone()));

        // GQA: repeat KV heads
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = mask {
            let mask = if m.dtype() != scores.dtype() {
                m.to_dtype(scores.dtype())?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;

        let out = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_query_heads * self.head_dim))?;
        self.o_proj.forward(&out)
    }
}
```

Add the `Mlp::forward` method:

```rust
impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = candle_nn::Activation::Silu.forward(&self.gate_proj.forward(x)?)?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}
```

Add the `TransformerLayer::forward` method:

```rust
impl TransformerLayer {
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
```

Add helper functions (before the `impl HarmonyModel` block):

```rust
/// Causal attention mask for prefill (seq_len > 1).
fn causal_mask(b: usize, tgt: usize, offset: usize, device: &Device) -> Result<Tensor> {
    let minf = f32::NEG_INFINITY;
    let mask: Vec<f32> = (0..tgt)
        .flat_map(|i| {
            (0..(tgt + offset)).map(move |j| {
                if j <= i + offset {
                    0.0
                } else {
                    minf
                }
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
```

Add the `forward` method to `impl HarmonyModel`:

```rust
    /// Forward pass with optional Engram injection callback.
    ///
    /// Returns logits for the last position + per-layer L2 norms.
    pub fn forward(
        &self,
        input: &Tensor,
        cache: &mut InferenceCache,
        engram_fn: Option<EngramFn<'_>>,
    ) -> Result<HarmonyForwardOutput> {
        let (b, l) = input.dims2()?;
        let offset = cache.position;
        let device = input.device();
        let mut h = self.embed_tokens.forward(input)?;

        let mask = if l == 1 {
            None
        } else {
            Some(causal_mask(b, l, offset, device)?)
        };

        let mut attnres_state = self.block_attnres.new_state();
        let mut layer_norms = Vec::with_capacity(self.config.num_layers);
        let lpb = self.config.layers_per_block;

        for (i, layer) in self.layers.iter().enumerate() {
            // Block AttnRes boundary (block > 0)
            if i > 0 && i % lpb == 0 {
                let block_idx = i / lpb;
                h = self.block_attnres.block_input(block_idx, &h, &attnres_state)?;
            }

            // Standard transformer layer
            h = layer.forward(&h, mask.as_ref(), offset, &mut cache.layers[i])?;

            // Engram injection
            if let Some(f) = &engram_fn {
                if i == self.config.engram_injection_layer {
                    if let Some(residual) = f(i, &h)? {
                        h = (h + residual)?;
                    }
                }
            }

            // Collect L2 norm for UQ features
            layer_norms.push(l2_norm_last_position(&h)?);

            // Record block summary at block end
            if (i + 1) % lpb == 0 {
                self.block_attnres
                    .notify_layer_output(i, &h, &mut attnres_state)?;
            }
        }

        let h = self.final_norm.forward(&h)?;
        let last_hidden = h.narrow(1, l - 1, 1)?;
        cache.position += l;
        let logits = self.lm_head.forward(&last_hidden)?.squeeze(1)?;

        Ok(HarmonyForwardOutput {
            logits,
            layer_norms,
        })
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cargo test -p harmony-inference harmony_model
```

Expected: All 13 tests pass (6 config/construction + 7 forward).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/harmony_model.rs
git commit -m "feat(inference): add HarmonyModel forward pass with BlockAttnRes + Engram (Phase 0e, Task 3)"
```

---

### Task 4: HarmonyEngine + Module Registration

**Files:**
- Create: `crates/harmony-inference/src/harmony_engine.rs`
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write the complete harmony_engine.rs with tests**

```rust
//! HarmonyEngine: inference engine for the ct87 custom model.
//!
//! Wraps [`HarmonyModel`] behind [`InferenceEngine`] and adds UQ
//! classification via [`forward_full`] + [`classify_uncertainty`].

use candle_core::{Device, Tensor};
use rand::thread_rng;

use crate::engine::EngramContext;
use crate::error::InferenceError;
use crate::harmony_model::{HarmonyForwardOutput, HarmonyModel, HarmonyModelConfig};
use crate::uq_features::{extract_uq_features, UqFeatureConfig};
use crate::uq_head::{UqClass, UqHead};
use crate::{InferenceCache, InferenceEngine, SamplingParams};

/// Inference engine for the ct87 custom model.
///
/// After `init_random()` (or eventually `load_gguf()`), all inference
/// methods take `&self`. Mutable state lives in caller-owned
/// [`InferenceCache`].
pub struct HarmonyEngine {
    model: Option<HarmonyModel>,
    tokenizer: Option<tokenizers::Tokenizer>,
    uq_head: Option<UqHead>,
    uq_feature_config: UqFeatureConfig,
    config: HarmonyModelConfig,
    device: Device,
}

impl HarmonyEngine {
    /// Create an engine with the given config. No model initialized yet.
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

    /// Initialize the model with random weights (for testing before GGUF).
    pub fn init_random(&mut self) -> Result<(), InferenceError> {
        let model = HarmonyModel::new(&self.config, &self.device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        self.model = Some(model);
        Ok(())
    }

    /// Set a UQ head for uncertainty classification.
    pub fn set_uq_head(&mut self, uq_head: UqHead) {
        self.uq_head = Some(uq_head);
    }

    /// Reference to the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Forward with Engram injection. Returns just logits.
    pub fn forward_with_engram(
        &self,
        tokens: &[u32],
        cache: &mut InferenceCache,
        engram: &EngramContext<'_>,
    ) -> Result<Vec<f32>, InferenceError> {
        let output = self.forward_full(tokens, cache, Some(engram))?;
        self.logits_to_vec(&output.logits)
    }

    /// Forward returning full output (logits + layer norms for UQ).
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
        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Build engram callback and run forward pass
        let output = if let Some(ctx) = engram {
            let engram_fn = |layer_idx: usize, hidden_state: &Tensor| -> candle_core::Result<Option<Tensor>> {
                if ctx.injection_layers.contains(&layer_idx) {
                    Ok(Some(ctx.module.forward(hidden_state, &ctx.embeddings)?))
                } else {
                    Ok(None)
                }
            };
            model.forward(&input, cache, Some(&engram_fn))
        } else {
            model.forward(&input, cache, None)
        };
        let output = output.map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        Ok(output)
    }

    /// Run UQ feature extraction + classification on forward output.
    /// Returns `None` if no UQ head is loaded.
    pub fn classify_uncertainty(
        &self,
        output: &HarmonyForwardOutput,
    ) -> Result<Option<(UqClass, f32)>, InferenceError> {
        let uq_head = match &self.uq_head {
            Some(h) => h,
            None => return Ok(None),
        };

        let logits_vec = self.logits_to_vec(&output.logits)?;
        let features = extract_uq_features(
            &output.layer_norms,
            &logits_vec,
            &self.uq_feature_config,
        )?;

        let features_tensor = Tensor::from_slice(&features, (1, 8), &self.device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        let (class, conf) = uq_head
            .classify(&features_tensor)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        Ok(Some((class, conf)))
    }

    fn logits_to_vec(&self, logits: &Tensor) -> Result<Vec<f32>, InferenceError> {
        let logits = match logits.dims().len() {
            1 => logits.clone(),
            2 => {
                let rows = logits
                    .dim(0)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
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
        self.logits_to_vec(&output.logits)
    }

    fn sample(
        &self,
        logits: &[f32],
        params: &SamplingParams,
        history: &[u32],
    ) -> Result<u32, InferenceError> {
        let mut rng = thread_rng();
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
        if self.model.is_none() {
            return Err(InferenceError::ModelNotLoaded);
        }
        Ok(InferenceCache::new(
            self.config.num_layers,
            self.config.head_dim,
            self.config.num_kv_heads,
        ))
    }
}

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

    #[test]
    fn new_engine_has_no_model() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        assert!(engine.model.is_none());
    }

    #[test]
    fn init_random_creates_model() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        engine.init_random().unwrap();
        assert!(engine.model.is_some());
    }

    #[test]
    fn forward_without_model_errors() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let mut cache = InferenceCache::new(4, 8, 2);
        let result = engine.forward(&[1, 2, 3], &mut cache);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn forward_empty_tokens_errors() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = InferenceCache::new(4, 8, 2);
        let result = engine.forward(&[], &mut cache);
        assert!(matches!(result, Err(InferenceError::ForwardFailed(_))));
    }

    #[test]
    fn forward_returns_logits() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let logits = engine.forward(&[1, 2, 3], &mut cache).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
    }

    #[test]
    fn forward_full_returns_layer_norms() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let output = engine.forward_full(&[1, 2], &mut cache, None).unwrap();
        assert_eq!(output.layer_norms.len(), cfg.num_layers);
    }

    #[test]
    fn classify_uncertainty_none_without_uq_head() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();
        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let output = engine.forward_full(&[1], &mut cache, None).unwrap();
        let result = engine.classify_uncertainty(&output).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn classify_uncertainty_returns_class() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();

        let uq_head = UqHead::new(&UqHeadConfig::default(), &Device::Cpu).unwrap();
        engine.set_uq_head(uq_head);

        let mut cache = InferenceCache::new(cfg.num_layers, cfg.head_dim, cfg.num_kv_heads);
        let output = engine.forward_full(&[1], &mut cache, None).unwrap();
        let result = engine.classify_uncertainty(&output).unwrap();
        assert!(result.is_some());
        let (class, conf) = result.unwrap();
        // Class should be one of the 4 valid values
        assert!((class as u8) < 4);
        // Confidence should be in [0, 1]
        assert!(conf >= 0.0 && conf <= 1.0, "confidence {conf} out of range");
    }

    #[test]
    fn load_gguf_returns_not_implemented() {
        let mut engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let result = engine.load_gguf(b"fake gguf data");
        assert!(matches!(result, Err(InferenceError::InvalidGguf(_))));
    }

    #[test]
    fn new_cache_matches_config() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();
        let cache = engine.new_cache().unwrap();
        assert_eq!(cache.num_layers, cfg.num_layers);
        assert_eq!(cache.head_dim, cfg.head_dim);
        assert_eq!(cache.num_kv_heads, cfg.num_kv_heads);
    }

    #[test]
    fn new_cache_without_model_errors() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let result = engine.new_cache();
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn cache_mismatch_detected() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg, Device::Cpu);
        engine.init_random().unwrap();
        // Wrong number of layers
        let mut cache = InferenceCache::new(99, 8, 2);
        let result = engine.forward(&[1], &mut cache);
        assert!(matches!(result, Err(InferenceError::CacheMismatch { .. })));
    }

    #[test]
    fn sample_greedy_works() {
        let engine = HarmonyEngine::new(test_config(), Device::Cpu);
        let logits = [1.0f32, 5.0, 3.0];
        let result = engine.sample(&logits, &SamplingParams::greedy(), &[]);
        assert_eq!(result.unwrap(), 1);
    }

    #[test]
    fn full_pipeline_init_forward_classify() {
        let cfg = test_config();
        let mut engine = HarmonyEngine::new(cfg.clone(), Device::Cpu);
        engine.init_random().unwrap();

        let uq_head = UqHead::new(&UqHeadConfig::default(), &Device::Cpu).unwrap();
        engine.set_uq_head(uq_head);

        let mut cache = engine.new_cache().unwrap();

        // Prefill
        let output = engine.forward_full(&[1, 2, 3], &mut cache, None).unwrap();
        assert_eq!(output.layer_norms.len(), cfg.num_layers);

        let classification = engine.classify_uncertainty(&output).unwrap();
        assert!(classification.is_some());

        // Decode
        let logits = engine.forward(&[4], &mut cache).unwrap();
        assert_eq!(logits.len(), cfg.vocab_size);
        assert_eq!(cache.position, 4);
    }
}
```

- [ ] **Step 2: Register module in lib.rs**

Add to module declarations in `crates/harmony-inference/src/lib.rs`:

```rust
pub mod harmony_engine;
```

Add re-exports:

```rust
pub use harmony_engine::HarmonyEngine;
```

- [ ] **Step 3: Run tests to verify they pass**

```bash
cargo test -p harmony-inference harmony_engine
```

Expected: 13 passed.

- [ ] **Step 4: Run full test suite**

```bash
cargo test -p harmony-inference
```

Expected: All tests pass (existing tests unchanged + ~39 new tests).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/harmony_engine.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add HarmonyEngine with UQ classification (Phase 0e, Task 4)"
```

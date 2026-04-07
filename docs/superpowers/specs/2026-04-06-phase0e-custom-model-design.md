# Phase 0e: Custom Model Forward Pass — Design Spec

**Epic:** harmony-ct87 / harmony-5nwj
**Phase:** 0e (blocked on 0a, 0b, 0c, 0d — all complete)
**Status:** Design approved
**Date:** 2026-04-06

## Goal

Define the ct87 custom model architecture in candle and implement a forward pass
that natively orchestrates all four Phase 0 innovations: Block Attention Residuals,
Engram gated injection, TurboQuant KV compression (via externalized cache), and
UQ feature extraction for uncertainty classification.

This is the integration point where the four independently-built modules meet for
the first time.

## Architecture

### Relationship to Existing Code

The custom model (`HarmonyModel`) lives alongside the existing `qwen3_ext.rs`
(forked Qwen3). The two coexist:

- `QwenEngine` + `qwen3_ext::ModelWeights` — loads pre-trained Qwen3 GGUFs
- `HarmonyEngine` + `HarmonyModel` — the ct87 custom architecture

They share `InferenceEngine` trait and `InferenceCache`. No generic
`ModelArchitecture` trait — YAGNI. If duplication becomes a problem later, we
extract a trait then.

### Layer Building Blocks

The custom model uses **non-quantized** `candle_nn` types (`Linear`, `Embedding`,
`RmsNorm`) since there is no GGUF to load yet (that's Phase 0g). Weight
initialization uses Kaiming uniform (1/sqrt(fan_in)) for random init.

```
TransformerLayer:
  attn_norm: RmsNorm
  attn: Attention {
    q_proj: Linear    [hidden_dim, num_query_heads * head_dim]
    k_proj: Linear    [hidden_dim, num_kv_heads * head_dim]
    v_proj: Linear    [hidden_dim, num_kv_heads * head_dim]
    o_proj: Linear    [num_query_heads * head_dim, hidden_dim]
    q_norm: RmsNorm
    k_norm: RmsNorm
    rotary_emb: Arc<RotaryEmbedding>
  }
  ffn_norm: RmsNorm
  mlp: Mlp {
    gate_proj: Linear  [hidden_dim, ffn_dim]
    up_proj: Linear    [hidden_dim, ffn_dim]
    down_proj: Linear  [ffn_dim, hidden_dim]
  }
```

Per-layer forward: PreNorm -> Attention -> Residual -> PreNorm -> SwiGLU MLP -> Residual.

RoPE implementation duplicated from qwen3_ext (~35 lines). Small enough that
sharing isn't worth the coupling.

### Forward Pass Orchestration

```
fn forward(input_ids, cache, engram_fn?) -> HarmonyForwardOutput:
    h = embed_tokens(input_ids)          // [1, seq_len, hidden_dim]
    attnres_state = block_attnres.new_state()
    layer_norms = Vec::new()

    for (i, layer) in layers:
        // -- Block AttnRes boundary --
        if i > 0 && i % layers_per_block == 0:
            block_idx = i / layers_per_block
            h = block_attnres.block_input(block_idx, &h, &attnres_state)?

        // -- Standard transformer layer --
        h = layer.forward(&h, mask, offset, &mut cache.layers[i])?

        // -- Engram injection (Layer 2 only) --
        if let Some(engram_fn) = &engram_fn:
            if i == config.engram_injection_layer:
                if let Some(residual) = engram_fn(i, &h)?:
                    h = h + residual

        // -- Collect L2 norm for UQ features --
        layer_norms.push(l2_norm_last_position(&h))

        // -- Record block summary at block end --
        if (i + 1) % layers_per_block == 0:
            block_attnres.notify_layer_output(i, &h, &mut attnres_state)?

    logits = lm_head(final_norm(h))
    return HarmonyForwardOutput { logits, layer_norms }
```

Key design points:

- **BlockAttnRes is native** — `block_input()` at block starts,
  `notify_layer_output()` at block ends. Integrated in the layer loop, not a
  callback.
- **Engram injection uses callback pattern** — same as qwen3_ext. The engine
  constructs the closure from `EngramContext`. Keeps the model decoupled from
  `EngramGatedResidual`.
- **Layer norms collected at every layer** — cheap (one L2 norm per layer). The
  UQ feature extractor picks the 4 it needs from this vec.
- **L2 norm of last position** — during decode (seq_len=1) there's only one
  position. During prefill, takes the last token's norm.

### UQ Feature Extraction

Pure function, no tensors, no device — just floats in, floats out.

```rust
pub fn extract_uq_features(
    layer_norms: &[f32],
    logits: &[f32],
    config: &UqFeatureConfig,
) -> Result<[f32; 8]>
```

| Feature | Source | Computation |
|---------|--------|-------------|
| f1-f4 | layer_norms | L2 norms at config.norm_layers (default [0, 8, 16, 23]) |
| f5 | f1-f4 | Linear regression slope: `(3(f4-f1) + (f3-f2)) / 10` |
| f6 | logits | Shannon entropy after softmax: `-sum(p * ln(p))` |
| f7 | logits | Sum of top-k probabilities after softmax |
| f8 | — | Stubbed as 0.0 (attention lookback ratio, deferred) |

Properties:
- Deterministic
- Testable in isolation (no tensor/device dependencies)
- f6 clamps probabilities to avoid ln(0)

### Ownership Model

```
Model owns:     layers, embeddings, lm_head, final_norm, BlockAttnRes
Engine owns:    model, tokenizer, UqHead (optional), UqFeatureConfig
Caller owns:    InferenceCache, EngramGatedResidual + embeddings (via EngramContext)
```

## Types

### HarmonyModelConfig

```rust
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
```

Convenience constructors:
- `HarmonyModelConfig::target()` — full 471M config (24 layers, 1280 hidden, etc.)
- `HarmonyModelConfig::tiny()` — ~50M config (8 layers, 512 hidden, etc.)
- `num_blocks()` — `num_layers / layers_per_block`
- `block_attnres_config()` — derives `BlockAttnResConfig`

### HarmonyForwardOutput

```rust
pub struct HarmonyForwardOutput {
    pub logits: Tensor,
    pub layer_norms: Vec<f32>,
}
```

### UqFeatureConfig

```rust
pub struct UqFeatureConfig {
    pub norm_layers: [usize; 4],
    pub top_k_for_mass: usize,
}
```

### HarmonyModel

```rust
pub struct HarmonyModel {
    embed_tokens: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: RmsNorm,
    lm_head: Linear,
    rotary_emb: Arc<RotaryEmbedding>,
    block_attnres: BlockAttnRes,
    config: HarmonyModelConfig,
}
```

### HarmonyEngine

```rust
pub struct HarmonyEngine {
    model: Option<HarmonyModel>,
    tokenizer: Option<Tokenizer>,
    uq_head: Option<UqHead>,
    uq_feature_config: UqFeatureConfig,
    device: Device,
}
```

## API

### HarmonyModel

```rust
impl HarmonyModel {
    /// Random weight init for testing.
    pub fn new(config: &HarmonyModelConfig, device: &Device) -> Result<Self>;

    /// Forward pass — returns logits + collected layer norms.
    pub fn forward(
        &self,
        input: &Tensor,
        cache: &mut InferenceCache,
        engram_fn: Option<EngramFn<'_>>,
    ) -> Result<HarmonyForwardOutput>;
}
```

### HarmonyEngine

```rust
impl InferenceEngine for HarmonyEngine {
    fn load_gguf(&mut self, data: &[u8]) -> Result<(), InferenceError>;
    fn load_tokenizer(&mut self, json: &[u8]) -> Result<(), InferenceError>;
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError>;
    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError>;
    fn forward(&self, tokens: &[u32], cache: &mut InferenceCache) -> Result<Vec<f32>, InferenceError>;
    fn sample(&self, logits: &[f32], params: &SamplingParams, history: &[u32]) -> Result<u32, InferenceError>;
    fn eos_token_id(&self) -> Option<u32>;
    fn new_cache(&self) -> Result<InferenceCache, InferenceError>;
}

impl HarmonyEngine {
    /// Create engine with config. Model not yet initialized.
    pub fn new(config: HarmonyModelConfig, device: Device) -> Self;

    /// Initialize model with random weights (for testing before GGUF exists).
    pub fn init_random(&mut self) -> Result<(), InferenceError>;

    /// Forward with Engram injection.
    pub fn forward_with_engram(
        &self, tokens: &[u32], cache: &mut InferenceCache, engram: &EngramContext<'_>,
    ) -> Result<Vec<f32>, InferenceError>;

    /// Forward returning full output (logits + layer norms for UQ).
    pub fn forward_full(
        &self, tokens: &[u32], cache: &mut InferenceCache, engram: Option<&EngramContext<'_>>,
    ) -> Result<HarmonyForwardOutput, InferenceError>;

    /// Run UQ feature extraction + classification on a forward output.
    /// Returns None if no UqHead is loaded.
    pub fn classify_uncertainty(
        &self, output: &HarmonyForwardOutput,
    ) -> Result<Option<(UqClass, f32)>, InferenceError>;
}
```

### UQ Features

```rust
pub fn extract_uq_features(
    layer_norms: &[f32],
    logits: &[f32],
    config: &UqFeatureConfig,
) -> Result<[f32; 8], InferenceError>;
```

## File Structure

| File | Responsibility | Approximate size |
|------|----------------|-----------------|
| `harmony_model.rs` | `HarmonyModelConfig`, `HarmonyModel`, `HarmonyForwardOutput`, `TransformerLayer`, `Attention`, `Mlp`, `RotaryEmbedding` | ~400 lines |
| `uq_features.rs` | `UqFeatureConfig`, `extract_uq_features()`, entropy/slope math | ~120 lines |
| `harmony_engine.rs` | `HarmonyEngine`, `InferenceEngine` impl, `forward_full`, `classify_uncertainty`, `init_random` | ~250 lines |
| `lib.rs` | Add `pub mod` + re-exports for new modules | ~10 lines changed |

Total: ~770 lines across 3 new files.

## Scope Boundary

**In scope (0e):**
- `HarmonyModelConfig` with `target()` and `tiny()` constructors
- `HarmonyModel` with `new()` (random init) and `forward()`
- Full transformer layer stack: Attention (GQA, RoPE, externalized KV), SwiGLU MLP, RmsNorm
- BlockAttnRes integration: `block_input()` at boundaries, `notify_layer_output()` at block ends
- Engram injection via callback at configurable layer
- L2 norm collection during forward pass
- `HarmonyForwardOutput` with logits + layer norms
- `UqFeatureConfig` and `extract_uq_features()` (7 features + f8 stubbed)
- `HarmonyEngine` implementing `InferenceEngine`
- `init_random()` for testing without GGUF
- `forward_full()` and `classify_uncertainty()` orchestration
- Unit tests for all components
- Module registration in lib.rs

**Out of scope (deferred):**
- GGUF loading for HarmonyModel (Phase 0g)
- PyTorch training scaffold (Phase 0f)
- UQ routing decisions (post-0e integration)
- Attention lookback ratio / f8 (deferred, stubbed as 0.0)
- TurboQuant compress/decompress calls (already in InferenceCache, caller manages)
- `from_tensors()` constructors for layers (Phase 0g, when GGUF provides weights)

## Testing Strategy

### Per-file tests

| File | Tests |
|------|-------|
| `harmony_model` | Config target/tiny values correct, config derives BlockAttnResConfig, model construction (tiny config), forward output shape correct, forward advances cache position, layer_norms length equals num_layers, BlockAttnRes state populated after forward, causal mask shape, tied embeddings share weights |
| `uq_features` | Entropy of uniform distribution is maximal, entropy of one-hot is zero, top-k mass sums correctly, slope positive for increasing norms, slope negative for decreasing norms, slope zero for constant norms, f8 is always 0.0, norm_layers out of bounds returns error |
| `harmony_engine` | init_random creates model, forward without model returns ModelNotLoaded, forward_full returns layer_norms, classify_uncertainty returns None without UqHead, classify_uncertainty returns class with UqHead, forward_with_engram injects at correct layer, load_gguf returns not-implemented error, new_cache matches model config |

### Integration tests

- Full pipeline: init_random -> forward_full -> extract_uq_features -> classify
- Engram injection: forward_with_engram produces different logits than forward
- BlockAttnRes effect: verify block boundaries produce different hidden states than passthrough

# KV Cache Externalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor harmony-inference to externalize the KV cache, making the engine stateless and enabling downstream compression/serialization of cache state.

**Architecture:** Fork ~320 lines of candle's quantized Qwen3 into `qwen3_ext.rs` with cache passed as parameter instead of stored internally. Change `InferenceEngine` trait from `forward(&mut self)` to `forward(&self, cache)`. Update all harmony-node consumers (inference loop, 3 DSD functions).

**Tech Stack:** harmony-inference, harmony-node, candle-core, candle-nn (new dep)

**Spec:** `docs/superpowers/specs/2026-03-27-kv-cache-externalization-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-inference/Cargo.toml` | Add `candle-nn`, `tracing`; remove `candle-transformers` |
| `crates/harmony-inference/src/lib.rs` | New trait signature, `InferenceCache` struct |
| `crates/harmony-inference/src/error.rs` | Add `CacheMismatch` variant |
| `crates/harmony-inference/src/qwen3_ext.rs` | Forked quantized Qwen3 with externalized cache |
| `crates/harmony-inference/src/engine.rs` | Stateless QwenEngine using qwen3_ext |
| `crates/harmony-node/src/event_loop.rs` | Inference loop uses external cache |
| `crates/harmony-node/src/runtime.rs` | DSD functions use external cache, remove reset() calls |
| `Cargo.toml` (workspace root) | Add `candle-nn` to workspace deps |

---

### Task 1: InferenceCache Type and Error Variant

Add the `InferenceCache` struct and `CacheMismatch` error variant. These are the foundation types everything else depends on.

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`
- Modify: `crates/harmony-inference/src/error.rs`

- [ ] **Step 1: Add CacheMismatch error variant**

In `crates/harmony-inference/src/error.rs`, add after the `SamplingFailed` variant (line 26):

```rust
    /// Cache does not match the loaded model architecture.
    #[error("cache mismatch: expected {expected} layers, got {actual}")]
    CacheMismatch { expected: usize, actual: usize },
```

- [ ] **Step 2: Add InferenceCache struct to lib.rs**

In `crates/harmony-inference/src/lib.rs`, add after the `use` statements and before `SamplingParams` (line 25). Also add the `candle_core` import at the top:

```rust
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
```

- [ ] **Step 3: Add unit tests for InferenceCache**

Add at the bottom of `crates/harmony-inference/src/lib.rs`:

```rust
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
```

- [ ] **Step 5: Verify tests pass**

Run: `cargo test -p harmony-inference`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/lib.rs crates/harmony-inference/src/error.rs
git commit -m "feat(inference): add InferenceCache type and CacheMismatch error"
```

---

### Task 2: Updated InferenceEngine Trait

Change the trait signature to externalize cache and history. This will temporarily break the QwenEngine impl — that's expected.

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Update the InferenceEngine trait**

Replace the entire trait definition (lines 68-107 of `lib.rs`) with:

```rust
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
```

- [ ] **Step 2: Update the module doc example**

Replace the doc example at the top of `lib.rs` (lines 9-17):

```rust
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
```

- [ ] **Step 3: Verify it compiles (expect engine.rs failures)**

Run: `cargo check -p harmony-inference`

Expected: compilation errors in `engine.rs` because `QwenEngine` still implements the old trait. This is correct — Task 3 fixes it.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): update InferenceEngine trait for external cache"
```

---

### Task 3: Forked Qwen3 Module (qwen3_ext.rs)

Fork the upstream `quantized_qwen3.rs` (~320 lines) with externalized cache. This is the core mechanical refactor.

**Files:**
- Create: `crates/harmony-inference/src/qwen3_ext.rs`
- Modify: `crates/harmony-inference/src/lib.rs` (add `mod qwen3_ext`)
- Modify: `crates/harmony-inference/Cargo.toml` (add candle-nn, tracing; remove candle-transformers)
- Modify: `Cargo.toml` (workspace root — add candle-nn)

**Reference:** Read the upstream source at `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/candle-transformers-0.9.2/src/models/quantized_qwen3.rs` (434 lines). The forked module applies the same refactor as candle-pipelines-models: `&mut self` → `&self` + cache parameter.

- [ ] **Step 1: Add candle-nn to workspace deps**

In the workspace root `Cargo.toml`, find the `[workspace.dependencies]` section (near lines 100-102 where `candle-core` and `candle-transformers` are defined) and add:

```toml
candle-nn = "0.9"
```

- [ ] **Step 2: Update harmony-inference Cargo.toml**

Replace the `[dependencies]` section of `crates/harmony-inference/Cargo.toml`:

```toml
[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
tokenizers = { workspace = true }
rand = { workspace = true }
thiserror = { workspace = true, features = ["std"] }
tracing = { workspace = true }
```

Note: `candle-transformers` is removed. `candle-nn` and `tracing` are added.

Also update the `[features]` section to forward GPU features to candle-nn:

```toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
```

- [ ] **Step 3: Add the mod declaration**

In `crates/harmony-inference/src/lib.rs`, add after `pub mod sampling;`:

```rust
pub(crate) mod qwen3_ext;
```

- [ ] **Step 4: Create qwen3_ext.rs**

Create `crates/harmony-inference/src/qwen3_ext.rs` with the forked implementation. This is the largest single piece of code. The key changes from upstream are:

1. `AttentionWeights::forward(&self, ..., kv_slot: &mut Option<(Tensor, Tensor)>)` — cache passed in, not stored
2. `LayerWeights::forward(&self, ..., kv_slot: &mut Option<(Tensor, Tensor)>)` — propagates
3. `ModelWeights::forward(&self, input, cache: &mut InferenceCache)` — iterates layers with `enumerate`, passes `&mut cache.layers[i]`
4. `ModelWeights::clear_kv_cache()` removed (caller drops cache)
5. Utility types (`QMatMul`, `RmsNorm`, `repeat_kv`) inlined

```rust
//! Forked quantized Qwen3 with externalized KV cache.
//!
//! Based on candle-transformers 0.9.2 `models/quantized_qwen3.rs`.
//! Key change: KV cache is passed as `&mut InferenceCache` parameter
//! instead of being stored inside `AttentionWeights`. This makes
//! `ModelWeights` fully `&self` after construction, enabling
//! `Arc<ModelWeights>` sharing and external cache inspection.

use candle_core::quantized::{gguf_file, QMatMul as CoreQMatMul, QTensor};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::Embedding;
use std::io::{Read, Seek};
use std::sync::Arc;

use crate::InferenceCache;

// ---------------------------------------------------------------------------
// Utility types (forked from candle-transformers internals)
// ---------------------------------------------------------------------------

/// Tracing wrapper around candle's quantized matrix multiply.
#[derive(Clone)]
struct QMatMul {
    inner: CoreQMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_weights(ws: Arc<QTensor>) -> Result<Self> {
        let inner = CoreQMatMul::from_arc(ws)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }
}

impl Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

impl std::fmt::Debug for QMatMul {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "QMatMul")
    }
}

/// Quantized RmsNorm: dequantizes weight at load time, applies candle_nn's rms_norm op.
#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
    span: tracing::Span,
}

impl RmsNorm {
    fn from_qtensor(weight: QTensor, eps: f64) -> Result<Self> {
        let span = tracing::span!(tracing::Level::TRACE, "rms-norm");
        let weight = weight.dequantize(&weight.device())?;
        Ok(Self { weight, eps, span })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        candle_nn::ops::rms_norm(x, &self.weight, self.eps as f32)
    }
}

/// Expand KV heads for Grouped-Query Attention.
fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

// ---------------------------------------------------------------------------
// GGUF loader helper
// ---------------------------------------------------------------------------

pub(crate) struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    pub fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_weights(ws.into())
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    pub fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

// ---------------------------------------------------------------------------
// Rotary embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
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

// ---------------------------------------------------------------------------
// Model components
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: candle_nn::Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: candle_nn::Activation::Silu,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;
        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;
        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;
        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            span_attn,
        })
    }

    /// Run attention with externalized KV cache slot.
    fn forward(
        &self,
        x: &Tensor,
        attn_mask: Option<&Tensor>,
        offset: usize,
        kv_slot: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
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

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;
        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Append to externalized KV cache slot.
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
        if let Some(m) = attn_mask {
            let mask = if m.dtype() != scores.dtype() {
                m.to_dtype(scores.dtype())?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?;
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");
        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        offset: usize,
        kv_slot: &mut Option<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset, kv_slot)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }
}

// ---------------------------------------------------------------------------
// Top-level model
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub(crate) struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    pub num_layers: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    pub fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_weights(lm_head_tensor.into())?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            num_layers,
            num_kv_heads,
            head_dim,
            span,
            span_output,
        })
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }

    /// Run the full model forward pass with externalized KV cache.
    pub fn forward(&self, input: &Tensor, cache: &mut InferenceCache) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, cache.position, None)?)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, causal_mask.as_ref(), cache.position, &mut cache.layers[i])?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        let logits = self.lm_head.forward(&last_hidden)?.squeeze(1)?;
        cache.position += l;
        Ok(logits)
    }
}
```

- [ ] **Step 5: Verify the module compiles**

Run: `cargo check -p harmony-inference`

Expected: `qwen3_ext.rs` compiles. `engine.rs` still has errors (implements old trait). That's Task 4.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/harmony-inference/Cargo.toml crates/harmony-inference/src/lib.rs crates/harmony-inference/src/qwen3_ext.rs
git commit -m "feat(inference): add forked Qwen3 module with externalized KV cache"
```

---

### Task 4: Refactor QwenEngine to Stateless

Update `engine.rs` to use `qwen3_ext::ModelWeights` and implement the new trait. Remove `position` and `token_history` fields.

**Files:**
- Modify: `crates/harmony-inference/src/engine.rs`

- [ ] **Step 1: Rewrite engine.rs**

Replace the entire contents of `crates/harmony-inference/src/engine.rs` with:

```rust
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
        // Wrong layer count — but no model loaded, so we get ModelNotLoaded first.
        // This test just verifies the check path exists; full test needs a loaded model.
        let mut cache = InferenceCache::new(99, 128, 8);
        let result = engine.forward(&[1], &mut cache);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }
}
```

- [ ] **Step 2: Verify harmony-inference compiles and tests pass**

Run: `cargo test -p harmony-inference`

Expected: all tests pass. harmony-node may have compile errors (expects old API) — that's Task 5.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-inference/src/engine.rs
git commit -m "feat(inference): make QwenEngine stateless with external cache"
```

---

### Task 5: Update harmony-node Event Loop

Update `run_inference_loop` and `InferenceResult` to use the new API.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

- [ ] **Step 1: Update InferenceResult enum**

In `event_loop.rs`, the `InferenceResult::Complete` and `Failed` variants (lines 60-72) return the engine. The engine is now `&self`-safe, but `spawn_blocking` still needs ownership for the `'static` bound. Keep the engine field — it's returned for reuse. No structural change needed to the enum.

- [ ] **Step 2: Update run_inference_loop**

Replace the `run_inference_loop` function (lines 1999-2124) with the new version that creates a local cache and passes history to `sample()`. The key changes are:
- `mut engine` → `engine` (no longer needs mut for inference methods)
- Create `cache` via `engine.new_cache()`
- Track `history: Vec<u32>` locally
- Pass `&mut cache` to `forward()`, `&history` to `sample()`
- Remove all `engine.reset()` calls — cache is dropped naturally
- Error paths: no longer call `engine.reset()` before returning

```rust
#[cfg(feature = "inference")]
fn run_inference_loop(
    engine: harmony_inference::QwenEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    input: crate::inference::InferenceInput,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
) {
    use harmony_inference::InferenceEngine;

    let mut cache = match engine.new_cache() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.blocking_send(InferenceResult::Failed {
                query_id,
                task_id,
                error: format!("failed to create cache: {e}"),
                engine,
            });
            return;
        }
    };

    let (tokens, is_token_mode) = match input {
        crate::inference::InferenceInput::Text(prompt) => match engine.tokenize(&prompt) {
            Ok(t) => (t, false),
            Err(e) => {
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("tokenize failed: {e}"),
                    engine,
                });
                return;
            }
        },
        crate::inference::InferenceInput::TokenIds(ids) => (ids, true),
    };

    let mut history: Vec<u32> = Vec::new();

    let mut logits = match engine.forward(&tokens, &mut cache) {
        Ok(l) => l,
        Err(e) => {
            let _ = tx.blocking_send(InferenceResult::Failed {
                query_id,
                task_id,
                error: format!("forward failed: {e}"),
                engine,
            });
            return;
        }
    };
    history.extend_from_slice(&tokens);

    let mut full_text = String::new();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut sequence = 0u32;
    let eos = engine.eos_token_id();

    loop {
        let next_token = match engine.sample(&logits, &sampling_params, &history) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("sample failed: {e}"),
                    engine,
                });
                return;
            }
        };

        if eos == Some(next_token) || sequence >= max_tokens {
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: String::new(),
                token_id: if is_token_mode { Some(0) } else { None },
                final_chunk: true,
            });
            break;
        }

        if is_token_mode {
            generated_ids.push(next_token);
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: String::new(),
                token_id: Some(next_token),
                final_chunk: false,
            });
        } else {
            let text = engine.detokenize(&[next_token]).unwrap_or_default();
            full_text.push_str(&text);
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: text,
                token_id: None,
                final_chunk: false,
            });
        }

        history.push(next_token);
        sequence += 1;

        logits = match engine.forward(&[next_token], &mut cache) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("forward failed: {e}"),
                    engine,
                });
                return;
            }
        };
    }

    let output = if is_token_mode {
        crate::inference::InferenceOutput::TokenIds(generated_ids)
    } else {
        crate::inference::InferenceOutput::Text(full_text)
    };
    let _ = tx.blocking_send(InferenceResult::Complete {
        query_id,
        task_id,
        output,
        engine,
    });
}
```

- [ ] **Step 3: Verify harmony-node compiles (expect runtime.rs errors)**

Run: `cargo check -p harmony-node --features inference`

Expected: `event_loop.rs` compiles. `runtime.rs` will have errors from old `engine.forward()` and `engine.reset()` calls. That's Task 6.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): update inference loop for external cache API"
```

---

### Task 6: Update harmony-node Runtime (DSD + Lifecycle)

Update all DSD functions and engine lifecycle methods in runtime.rs for the new API. There are three DSD functions that call `engine.forward()` and `engine.reset()`:
- `run_verification` (line ~2688) — DSD sequential verification
- `handle_speculative_query` (line ~2785) — DSD draft generation
- `handle_verify_response` (line ~2969) — DSD continuation after verification

**Important:** These DSD functions do NOT call `engine.sample()`. They use `harmony_speculative::verify::sample_greedy_with_logprob()` which takes raw logits — not the engine's sample method. The changes are strictly: remove `engine.reset()`, add `&mut cache` to `forward()`, and change `&mut` borrows to `&`.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Update run_verification (DSD verification, line ~2688)**

Change the signature from `engine: &mut harmony_inference::QwenEngine` to `engine: &harmony_inference::QwenEngine`. Replace the body to use external cache:

```rust
#[cfg(feature = "inference")]
fn run_verification(
    engine: &harmony_inference::QwenEngine,
    request: &harmony_speculative::VerifyRequest,
) -> Result<harmony_speculative::VerifyResponse, String> {
    use harmony_inference::InferenceEngine;
    use harmony_speculative::verify::{sample_greedy_with_logprob, should_accept_draft};

    if request.drafts.is_empty() {
        return Err("empty drafts".into());
    }
    debug_assert!(
        request.drafts.len() <= u8::MAX as usize,
        "draft count {} exceeds u8::MAX",
        request.drafts.len()
    );

    // Create fresh cache (replaces engine.reset())
    let mut cache = engine.new_cache().map_err(|e| e.to_string())?;

    // Prefill with context tokens
    let mut current_logits = if request.context_tokens.is_empty() {
        return Err("empty context".into());
    } else {
        let logits = engine
            .forward(&request.context_tokens, &mut cache)
            .map_err(|e| format!("prefill failed: {e}"))?;
        if logits.is_empty() {
            return Err("prefill returned empty logits".into());
        }
        logits
    };

    // Verify each draft token sequentially
    for (i, draft) in request.drafts.iter().enumerate() {
        if !should_accept_draft(&current_logits, draft.token_id, draft.logprob) {
            let (bonus_token, bonus_logprob) = sample_greedy_with_logprob(&current_logits);
            return Ok(harmony_speculative::VerifyResponse {
                accepted_count: i as u8,
                bonus_token,
                bonus_logprob,
            });
        }
        current_logits = engine
            .forward(&[draft.token_id], &mut cache)
            .map_err(|e| format!("forward failed at draft {i}: {e}"))?;
        if current_logits.is_empty() {
            return Err(format!("forward returned empty logits at draft {i}"));
        }
    }

    let (bonus_token, bonus_logprob) = sample_greedy_with_logprob(&current_logits);
    Ok(harmony_speculative::VerifyResponse {
        accepted_count: request.drafts.len() as u8,
        bonus_token,
        bonus_logprob,
    })
}
```

Also update the doc comment above the function to remove the `engine.reset()` step.

- [ ] **Step 2: Update handle_speculative_query (DSD drafting, line ~2785)**

This function borrows `self.verification_engine` as `&mut`. Change to `&`:

```rust
let engine = match &self.verification_engine {
    Some(e) => e,
    None => { /* existing error handling */ return; }
};
```

Then create a fresh cache and thread it through:

1. Remove `engine.reset();` (line ~2873)
2. After the engine borrow, add: `let mut cache = match engine.new_cache() { Ok(c) => c, Err(e) => { /* error reply */ return; } };`
3. Change `engine.forward(&prompt_tokens)` → `engine.forward(&prompt_tokens, &mut cache)` (line ~2874)
4. Change `engine.forward(&[token_id])` → `engine.forward(&[token_id], &mut cache)` in the draft loop (line ~2912)

- [ ] **Step 3: Update handle_verify_response (DSD continuation, line ~2969)**

Same pattern as handle_speculative_query:

1. Change `&mut self.verification_engine` borrow to `&self.verification_engine` (line ~3080)
2. Remove `engine.reset();` (line ~3094)
3. Add cache creation: `let mut cache = match engine.new_cache() { ... };`
4. Change `engine.forward(&full_sequence)` → `engine.forward(&full_sequence, &mut cache)` (line ~3097)
5. Change `engine.forward(&[token_id])` → `engine.forward(&[token_id], &mut cache)` in the draft loop (line ~3140)

- [ ] **Step 4: Update callers of run_verification**

Search for all call sites of `run_verification` in runtime.rs. They pass `engine` as `&mut` — change to `&`:

```rust
// Before:
let result = Self::run_verification(engine, &request);
// After (no change needed if engine is already a shared ref):
let result = Self::run_verification(engine, &request);
```

The key is that the call site's borrow of `self.verification_engine` changes from `&mut` to `&`.

- [ ] **Step 5: Verify check_inference_model_ready and reset_inference_after_panic**

These functions don't need structural changes:
- `check_inference_model_ready`: `engine.load_gguf()` still takes `&mut self` — that's initialization, not inference. No position/history to clear.
- `reset_inference_after_panic`: just clears `inference_running` and `inference_queryable_id`. No cache state to worry about.

Verify neither function references `engine.reset()` or `engine.forward()`.

- [ ] **Step 6: Verify full workspace compiles**

Run: `cargo check --workspace`

Expected: clean compilation across all crates.

- [ ] **Step 7: Run all tests**

Run: `cargo test --workspace`

Expected: all tests pass (365+).

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): update DSD and engine lifecycle for external cache API"
```

---

### Task 7: Verify and Clean Up

Final verification, clippy, and format check.

**Files:**
- All modified files

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --workspace`

Fix any warnings.

- [ ] **Step 2: Run format check**

Run: `cargo fmt --all -- --check`

Fix any formatting issues.

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace`

All 365+ tests must pass.

- [ ] **Step 4: Verify the inference feature specifically**

Run: `cargo test -p harmony-node --features inference`

All harmony-node inference tests must pass.

- [ ] **Step 5: Commit any fixes**

```bash
git add crates/harmony-inference/ crates/harmony-node/src/event_loop.rs crates/harmony-node/src/runtime.rs
git commit -m "chore: clippy and fmt fixes for cache externalization"
```

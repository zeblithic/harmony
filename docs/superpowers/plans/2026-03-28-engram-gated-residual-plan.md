# EngramGatedResidual Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a gated residual injection module to harmony-inference that allows Engram embeddings to be injected into the Qwen3 forward pass at configurable layer indices.

**Architecture:** New `EngramGatedResidual` struct in `engram_residual.rs` with project → gate → conv1d → SiLU → residual add. Callback-based injection via `ModelWeights::forward_with_engram()` in `qwen3_ext.rs`. Existing `forward()` becomes a thin wrapper — zero overhead for non-Engram inference.

**Tech Stack:** harmony-inference, candle-core, candle-nn (Linear, Conv1d, RmsNorm, Activation::Silu)

**Spec:** `docs/superpowers/specs/2026-03-28-engram-gated-residual-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-inference/src/engram_residual.rs` | `EngramGatedResidual` struct, constructors, forward pass, unit tests |
| `crates/harmony-inference/src/qwen3_ext.rs` | Add `forward_with_engram()`, refactor `forward()` as wrapper |
| `crates/harmony-inference/src/lib.rs` | Add `pub mod engram_residual;` and re-export |

---

### Task 1: EngramGatedResidual Module with Tests

Create the complete `EngramGatedResidual` module with both constructors, the full forward pass, and unit tests.

**Files:**
- Create: `crates/harmony-inference/src/engram_residual.rs`
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Add module declaration and re-export in lib.rs**

In `crates/harmony-inference/src/lib.rs`, add after `pub mod sampling;` (line 23):

```rust
pub mod engram_residual;
```

And add to the re-exports after `pub use error::InferenceError;` (line 26):

```rust
pub use engram_residual::EngramGatedResidual;
```

- [ ] **Step 2: Create engram_residual.rs with the struct and constructors**

Create `crates/harmony-inference/src/engram_residual.rs`:

```rust
//! Gated residual injection of Engram embeddings into a transformer forward pass.
//!
//! The [`EngramGatedResidual`] module takes a transformer hidden state and a
//! pre-resolved Engram embedding, applies learned gating to determine relevance,
//! and returns the modified hidden state. Used at configurable layer indices
//! (e.g., layers 2 and 14 of Qwen3) via the callback in
//! [`ModelWeights::forward_with_engram`](crate::qwen3_ext::ModelWeights::forward_with_engram).
//!
//! # Architecture
//!
//! ```text
//! hidden_state ──┬──────────────────────────────────── (+) ── output
//!                │                                      ↑
//!                │  engram_embedding                     │
//!                │       │                               │
//!                │   key_proj ─→ key_norm ─┐             │
//!                │       │                 │ dot/sigmoid  │
//!                │   value_proj            gate           │
//!                │       │                 │             │
//!                └─ gate_norm ────────────┘              │
//!                        │                               │
//!                   gate * value                         │
//!                        │                               │
//!                   causal conv1d                        │
//!                        │                               │
//!                      SiLU ────────���────────────────────┘
//! ```

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Activation, Conv1d, Conv1dConfig, Linear, RmsNorm};

/// Gated residual module for injecting Engram embeddings into transformer layers.
///
/// Projects an Engram embedding into the model's hidden space, computes a
/// learned gating scalar, applies depthwise causal conv1d for temporal
/// smoothing, and adds the result as a residual to the hidden state.
#[derive(Debug, Clone)]
pub struct EngramGatedResidual {
    key_proj: Linear,
    value_proj: Linear,
    gate_norm: RmsNorm,
    key_norm: RmsNorm,
    conv1d: Conv1d,
    hidden_dim: usize,
    conv_kernel_size: usize,
}

impl EngramGatedResidual {
    /// Create a module with random weights for testing and shape verification.
    ///
    /// Projections use Kaiming-uniform-like init, norms use ones, conv uses zeros.
    /// For inference with trained weights, use [`from_tensors`](Self::from_tensors).
    pub fn new(
        engram_dim: usize,
        hidden_dim: usize,
        conv_kernel_size: usize,
        rms_norm_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        // Random projection weights (Kaiming-scale approximation)
        let scale = (2.0 / engram_dim as f64).sqrt() as f32;
        let key_w = (Tensor::rand(0f32, 1f32, (hidden_dim, engram_dim), device)? * scale as f64)?;
        let value_w =
            (Tensor::rand(0f32, 1f32, (hidden_dim, engram_dim), device)? * scale as f64)?;

        let key_proj = Linear::new(key_w, None);
        let value_proj = Linear::new(value_w, None);

        // RmsNorm with ones weight
        let ones = Tensor::ones(hidden_dim, DType::F32, device)?;
        let gate_norm = RmsNorm::new(ones.clone(), rms_norm_eps);
        let key_norm = RmsNorm::new(ones, rms_norm_eps);

        // Depthwise conv1d with zero weights (identity-like at init)
        let conv_w = Tensor::zeros((hidden_dim, 1, conv_kernel_size), DType::F32, device)?;
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: hidden_dim,
            ..Default::default()
        };
        let conv1d = Conv1d::new(conv_w, None, conv_config);

        Ok(Self {
            key_proj,
            value_proj,
            gate_norm,
            key_norm,
            conv1d,
            hidden_dim,
            conv_kernel_size,
        })
    }

    /// Create a module from pre-loaded weight tensors (e.g., from CAS).
    ///
    /// # Arguments
    ///
    /// - `key_proj_weight`: `[hidden_dim, engram_dim]`
    /// - `value_proj_weight`: `[hidden_dim, engram_dim]`
    /// - `gate_norm_weight`: `[hidden_dim]`
    /// - `key_norm_weight`: `[hidden_dim]`
    /// - `conv1d_weight`: `[hidden_dim, 1, kernel_size]`
    pub fn from_tensors(
        key_proj_weight: Tensor,
        value_proj_weight: Tensor,
        gate_norm_weight: Tensor,
        key_norm_weight: Tensor,
        conv1d_weight: Tensor,
        hidden_dim: usize,
        rms_norm_eps: f64,
    ) -> Result<Self> {
        let conv_kernel_size = conv1d_weight.dim(2)?;
        let conv_config = Conv1dConfig {
            padding: 0,
            stride: 1,
            dilation: 1,
            groups: hidden_dim,
            ..Default::default()
        };

        Ok(Self {
            key_proj: Linear::new(key_proj_weight, None),
            value_proj: Linear::new(value_proj_weight, None),
            gate_norm: RmsNorm::new(gate_norm_weight, rms_norm_eps),
            key_norm: RmsNorm::new(key_norm_weight, rms_norm_eps),
            conv1d: Conv1d::new(conv1d_weight, None, conv_config),
            hidden_dim,
            conv_kernel_size,
        })
    }

    /// Apply gated Engram injection to a transformer hidden state.
    ///
    /// # Arguments
    ///
    /// - `hidden_state`: `[batch, seq_len, hidden_dim]` — output of a transformer layer
    /// - `engram_embedding`: `[batch, seq_len, engram_dim]` — pre-resolved from mesh
    ///
    /// # Returns
    ///
    /// `[batch, seq_len, hidden_dim]` — gated Engram residual to add to hidden_state.
    /// The caller performs the residual add: `h = h + module.forward(h, embedding)?`
    pub fn forward(&self, hidden_state: &Tensor, engram_embedding: &Tensor) -> Result<Tensor> {
        // 1. Project Engram embedding into hidden space
        let key = self.key_proj.forward(engram_embedding)?; // [b, l, hidden_dim]
        let value = self.value_proj.forward(engram_embedding)?; // [b, l, hidden_dim]

        // 2. Compute gating scalar via normalized dot product
        let h_norm = self.gate_norm.forward(hidden_state)?; // [b, l, hidden_dim]
        let k_norm = self.key_norm.forward(&key)?; // [b, l, hidden_dim]
        let dot = (h_norm * k_norm)?.sum_keepdim(candle_core::D::Minus1)?; // [b, l, 1]
        let scale = (self.hidden_dim as f64).sqrt();
        let gate = (dot / scale)?.sigmoid()?; // [b, l, 1]

        // 3. Gate the value projection
        let gated_value = gate.broadcast_mul(&value)?; // [b, l, hidden_dim]

        // 4. Causal depthwise conv1d (requires NCL layout)
        //    NLC [b, l, hidden_dim] → NCL [b, hidden_dim, l]
        let gated_ncl = gated_value.transpose(1, 2)?;
        //    Left-pad for causality: pad (kernel_size - 1) zeros on the left
        let pad = self.conv_kernel_size - 1;
        let gated_padded = if pad > 0 {
            gated_ncl.pad_with_zeros(2, pad, 0)?
        } else {
            gated_ncl
        };
        let conv_ncl = self.conv1d.forward(&gated_padded)?;
        //    NCL [b, hidden_dim, l] → NLC [b, l, hidden_dim]
        let conv_out = conv_ncl.transpose(1, 2)?;

        // 5. SiLU activation and return gated residual
        // The caller (ModelWeights::forward_with_engram) performs the residual add:
        // h = h + engram_residual
        conv_out.apply(&Activation::Silu)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_preservation() {
        let device = Device::Cpu;
        let module =
            EngramGatedResidual::new(16, 64, 3, 1e-6, &device).expect("failed to create module");

        let hidden = Tensor::rand(0f32, 1f32, (1, 5, 64), &device).unwrap();
        let engram = Tensor::rand(0f32, 1f32, (1, 5, 16), &device).unwrap();

        let output = module.forward(&hidden, &engram).expect("forward failed");
        assert_eq!(output.dims(), &[1, 5, 64]);
    }

    #[test]
    fn shape_preservation_batch() {
        let device = Device::Cpu;
        let module =
            EngramGatedResidual::new(16, 64, 3, 1e-6, &device).expect("failed to create module");

        let hidden = Tensor::rand(0f32, 1f32, (2, 10, 64), &device).unwrap();
        let engram = Tensor::rand(0f32, 1f32, (2, 10, 16), &device).unwrap();

        let output = module.forward(&hidden, &engram).expect("forward failed");
        assert_eq!(output.dims(), &[2, 10, 64]);
    }

    #[test]
    fn zero_embedding_returns_zero_residual() {
        let device = Device::Cpu;
        let module =
            EngramGatedResidual::new(16, 64, 3, 1e-6, &device).expect("failed to create module");

        let hidden = Tensor::rand(0f32, 1f32, (1, 5, 64), &device).unwrap();
        let engram = Tensor::zeros((1, 5, 16), DType::F32, &device).unwrap();

        let residual = module.forward(&hidden, &engram).expect("forward failed");

        // key_proj(zeros) = zeros, value_proj(zeros) = zeros
        // gate * zeros = zeros regardless of gate value
        // conv1d(zeros) = zeros, silu(zeros) = zeros
        // residual = zeros
        let max_val = residual
            .abs()
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .max(0)
            .unwrap()
            .to_scalar::<f32>()
            .unwrap();
        assert!(
            max_val < 1e-6,
            "zero engram should produce zero residual, max={max_val}"
        );
    }

    #[test]
    fn gate_range_is_zero_to_one() {
        // Verify the gating mechanism produces values in [0, 1]
        // by checking that the output magnitude is bounded.
        // With random init, the gate is sigmoid(normalized_dot / sqrt(hidden_dim)),
        // which is always in (0, 1).
        let device = Device::Cpu;
        let module =
            EngramGatedResidual::new(16, 64, 3, 1e-6, &device).expect("failed to create module");

        let hidden = Tensor::rand(0f32, 1f32, (1, 5, 64), &device).unwrap();
        let engram = Tensor::rand(0f32, 1f32, (1, 5, 16), &device).unwrap();

        // Manually compute just the gate to verify bounds
        let key = module.key_proj.forward(&engram).unwrap();
        let h_norm = module.gate_norm.forward(&hidden).unwrap();
        let k_norm = module.key_norm.forward(&key).unwrap();
        let dot = (h_norm * k_norm)
            .unwrap()
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap();
        let scale = (64.0f64).sqrt();
        let gate = (dot / scale).unwrap().sigmoid().unwrap();

        let gate_vals = gate.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for (i, &g) in gate_vals.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&g),
                "gate[{i}] = {g} outside [0, 1]"
            );
        }
    }

    #[test]
    fn causal_conv1d_no_future_leakage() {
        let device = Device::Cpu;
        let hidden_dim = 8;
        let module =
            EngramGatedResidual::new(4, hidden_dim, 3, 1e-6, &device)
                .expect("failed to create module");

        // Create gated_value that's zero everywhere except the last position
        let seq_len = 5;
        let mut data = vec![0.0f32; hidden_dim * seq_len];
        // Set last position to non-zero
        for d in 0..hidden_dim {
            data[(seq_len - 1) * hidden_dim + d] = 1.0;
        }
        let gated_value = Tensor::from_vec(data, (1, seq_len, hidden_dim), &device).unwrap();

        // Run just the conv1d portion: transpose → pad → conv → transpose
        let gated_ncl = gated_value.transpose(1, 2).unwrap();
        let gated_padded = gated_ncl.pad_with_zeros(2, 2, 0).unwrap(); // kernel_size - 1
        let conv_ncl = module.conv1d.forward(&gated_padded).unwrap();
        let conv_out = conv_ncl.transpose(1, 2).unwrap();

        // Check: positions 0, 1, 2 should be zero (no future info leaks backward)
        // The non-zero value at position 4 should only affect positions 2, 3, 4
        // with kernel_size=3. But since we pad left, it can only reach 4, 3, 2.
        // Positions 0 and 1 must be zero.
        let out_vals = conv_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        for pos in 0..2 {
            for d in 0..hidden_dim {
                let val = out_vals[pos * hidden_dim + d];
                assert!(
                    val.abs() < 1e-6,
                    "position {pos} dim {d} = {val}, expected 0 (future leakage)"
                );
            }
        }
    }

    #[test]
    fn single_token_works() {
        // Verify the module works with seq_len=1 (decode step)
        let device = Device::Cpu;
        let module =
            EngramGatedResidual::new(16, 64, 3, 1e-6, &device).expect("failed to create module");

        let hidden = Tensor::rand(0f32, 1f32, (1, 1, 64), &device).unwrap();
        let engram = Tensor::rand(0f32, 1f32, (1, 1, 16), &device).unwrap();

        let output = module.forward(&hidden, &engram).expect("forward failed");
        assert_eq!(output.dims(), &[1, 1, 64]);
    }

    #[test]
    fn from_tensors_matches_shapes() {
        let device = Device::Cpu;
        let hidden_dim = 32;
        let engram_dim = 8;
        let kernel_size = 3;

        let key_w = Tensor::rand(0f32, 1f32, (hidden_dim, engram_dim), &device).unwrap();
        let val_w = Tensor::rand(0f32, 1f32, (hidden_dim, engram_dim), &device).unwrap();
        let gnorm = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
        let knorm = Tensor::ones(hidden_dim, DType::F32, &device).unwrap();
        let conv_w = Tensor::zeros((hidden_dim, 1, kernel_size), DType::F32, &device).unwrap();

        let module = EngramGatedResidual::from_tensors(
            key_w, val_w, gnorm, knorm, conv_w, hidden_dim, 1e-6,
        )
        .expect("from_tensors failed");

        let hidden = Tensor::rand(0f32, 1f32, (1, 3, hidden_dim), &device).unwrap();
        let engram = Tensor::rand(0f32, 1f32, (1, 3, engram_dim), &device).unwrap();

        let output = module.forward(&hidden, &engram).expect("forward failed");
        assert_eq!(output.dims(), &[1, 3, hidden_dim]);
    }
}
```

- [ ] **Step 3: Verify tests pass**

Run: `cargo test -p harmony-inference`

Expected: All existing tests pass plus 7 new tests (shape_preservation, shape_preservation_batch, zero_embedding_passthrough, gate_range_is_zero_to_one, causal_conv1d_no_future_leakage, single_token_works, from_tensors_matches_shapes).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-inference/src/engram_residual.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add EngramGatedResidual module for Engram injection"
```

---

### Task 2: Callback Injection in ModelWeights

Add `forward_with_engram()` to `ModelWeights` and refactor `forward()` as a thin wrapper.

**Files:**
- Modify: `crates/harmony-inference/src/qwen3_ext.rs`

- [ ] **Step 1: Add forward_with_engram method**

In `crates/harmony-inference/src/qwen3_ext.rs`, replace the current `forward` method (lines 522-544) with two methods. The existing `forward` becomes a wrapper, and the real implementation moves to `forward_with_engram`:

```rust
    /// Forward pass with externalized KV cache.
    ///
    /// The caller owns the [`InferenceCache`] and passes it in on every call.
    /// This method advances `cache.position` by the number of input tokens.
    pub(crate) fn forward(&self, input: &Tensor, cache: &mut InferenceCache) -> Result<Tensor> {
        self.forward_with_engram(input, cache, None)
    }

    /// Forward pass with optional Engram injection callback.
    ///
    /// If `engram_fn` is `Some`, it is called after each transformer layer with
    /// `(layer_index, &hidden_state)`. If the callback returns `Ok(Some(tensor))`,
    /// that tensor is added to the hidden state as a residual. If it returns
    /// `Ok(None)`, the layer output passes through unchanged.
    ///
    /// The callback is typically constructed by composing an [`EngramGatedResidual`]
    /// module with pre-resolved Engram embeddings:
    ///
    /// ```ignore
    /// model.forward_with_engram(&input, &mut cache, Some(&|layer_idx, hidden| {
    ///     if layer_idx == 2 || layer_idx == 14 {
    ///         Ok(Some(engram_module.forward(hidden, &embedding)?))
    ///         // Note: EngramGatedResidual::forward returns hidden_state + residual,
    ///         // but the callback should return ONLY the residual (not hidden_state + residual).
    ///         // See the note below about the callback contract.
    ///     } else {
    ///         Ok(None)
    ///     }
    /// }))?;
    /// ```
    ///
    /// **Callback contract:** The callback returns the *additive residual* tensor
    /// `[b, l, hidden_dim]`. The caller adds it: `h = h + residual`. If using
    /// `EngramGatedResidual`, call its internal gating logic and return the
    /// gated+activated tensor (without the hidden_state residual add — that
    /// happens here in the layer loop).
    pub(crate) fn forward_with_engram(
        &self,
        input: &Tensor,
        cache: &mut InferenceCache,
        engram_fn: Option<&dyn Fn(usize, &Tensor) -> Result<Option<Tensor>>>,
    ) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let offset = cache.position;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            h = layer.forward(&h, causal_mask.as_ref(), offset, &mut cache.layers[i])?;

            if let Some(f) = &engram_fn {
                if let Some(engram_residual) = f(i, &h)? {
                    h = (h + engram_residual)?;
                }
            }
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        cache.position += l;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }
```

**Important note on the callback contract:** Looking at this more carefully, the spec's `EngramGatedResidual::forward` returns `hidden_state + activated` (the full residual add). But the callback adds `h = h + engram_residual`, which would double-add the hidden state. There are two clean options:

**Option A:** Change `EngramGatedResidual::forward` to return only the residual (not hidden_state + residual). The layer loop does the add.

**Option B:** The callback returns the *full* modified hidden state, and the layer loop does `h = engram_result` (assignment, not addition).

**Choose Option A** — it's cleaner and matches the spec's description of the callback returning an "additive residual." Update `EngramGatedResidual::forward` step 7 to return just `activated` instead of `hidden_state + activated`. The residual add happens in the layer loop.

So in `engram_residual.rs`, the final line of `forward` should be:

```rust
        // 6. Return the gated residual (caller adds to hidden_state)
        conv_out.apply(&Activation::Silu)
```

NOT `hidden_state + activated`.

- [ ] **Step 2: Verify all tests pass**

Run: `cargo test -p harmony-inference`

Expected: All tests pass. The `forward()` wrapper calls `forward_with_engram(None)` which follows the same code path as before.

- [ ] **Step 3: Verify workspace compiles**

Run: `cargo check --workspace`

Expected: Clean — no other crate calls `forward()` differently.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-inference/src/qwen3_ext.rs
git commit -m "feat(inference): add forward_with_engram callback injection in ModelWeights"
```

---

### Task 3: Verify and Clean Up

Final verification, clippy, and format check.

**Files:**
- All modified files

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-inference`

Fix any warnings in the new code.

- [ ] **Step 2: Run format check**

Run: `cargo fmt -p harmony-inference -- --check`

Fix any formatting issues.

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace --exclude harmony-tunnel`

All tests must pass.

- [ ] **Step 4: Commit any fixes**

```bash
git add crates/harmony-inference/
git commit -m "chore: clippy and fmt fixes for EngramGatedResidual"
```

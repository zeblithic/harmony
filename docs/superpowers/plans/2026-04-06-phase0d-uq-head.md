# Phase 0d: UQ Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Uncertainty Quantification Head as a candle inference module — a parallel MLP that takes 8 pre-extracted features and produces a 4-class uncertainty classification plus scalar confidence score.

**Architecture:** Two-path MLP (classifier + confidence) operating on a pre-extracted feature vector. The classifier path is Linear(8→32) → ReLU → Linear(32→4) → softmax. The confidence path is Linear(8→1) → sigmoid. Feature extraction from model internals is deferred to Phase 0e.

**Tech Stack:** Rust, candle-core, candle-nn (`softmax_last_dim`, `sigmoid`)

**Spec:** `docs/superpowers/specs/2026-04-06-phase0d-uq-head-design.md`

**Pattern reference:** `crates/harmony-inference/src/block_attnres.rs` — same crate, same Config/Module/State pattern.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/harmony-inference/src/uq_head.rs` | Create | UqClass enum, UqHeadConfig, UqHead module, UqOutput, forward/classify, all tests |
| `crates/harmony-inference/src/lib.rs` | Modify (lines 20, 30) | Add `pub mod uq_head;` and `pub use` exports |

---

### Task 1: UqClass enum and UqHeadConfig

**Files:**
- Create: `crates/harmony-inference/src/uq_head.rs`

This task creates the foundational types: the 4-class enum with parsing, and the
configuration struct. No candle dependency yet — pure Rust types.

- [ ] **Step 1: Write failing tests for UqClass and UqHeadConfig**

Add the test module and tests at the bottom of the new file. The types don't exist
yet, so these will fail to compile.

```rust
//! Uncertainty Quantification Head — parallel metacognitive MLP monitor.
//!
//! Takes 8 pre-extracted features (hidden state norms, entropy, etc.) and
//! produces a 4-class uncertainty classification plus scalar confidence.
//! The caller uses the classification for routing decisions (emit, retrieve,
//! or abort). Feature extraction is external — this module is a pure classifier.

use candle_core::{Device, Result, Tensor};

#[cfg(test)]
mod tests {
    use super::*;

    // ── UqClass parsing ──

    #[test]
    fn from_u8_valid_classes() {
        assert_eq!(UqClass::from_u8(0), Some(UqClass::Confident));
        assert_eq!(UqClass::from_u8(1), Some(UqClass::HighVolume));
        assert_eq!(UqClass::from_u8(2), Some(UqClass::SpectralCollapse));
        assert_eq!(UqClass::from_u8(3), Some(UqClass::Uncertain));
    }

    #[test]
    fn from_u8_invalid_returns_none() {
        assert_eq!(UqClass::from_u8(4), None);
        assert_eq!(UqClass::from_u8(255), None);
    }

    #[test]
    fn try_from_u8_valid() {
        assert_eq!(UqClass::try_from(0u8), Ok(UqClass::Confident));
        assert_eq!(UqClass::try_from(3u8), Ok(UqClass::Uncertain));
    }

    #[test]
    fn try_from_u8_invalid() {
        assert_eq!(UqClass::try_from(4u8), Err(4u8));
    }

    // ── UqHeadConfig ──

    #[test]
    fn default_config_values() {
        let cfg = UqHeadConfig::default();
        assert_eq!(cfg.num_features, 8);
        assert_eq!(cfg.hidden_dim, 32);
        assert_eq!(cfg.num_classes, 4);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- 2>&1 | head -20
```

Expected: compilation error — `UqClass` and `UqHeadConfig` not defined.

- [ ] **Step 3: Implement UqClass enum**

Add above the `#[cfg(test)]` block:

```rust
/// Uncertainty classification output.
///
/// Discriminant values (0–3) match training label indices and are stored
/// as `u8` in serialized models. Use [`from_u8`](Self::from_u8) to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum UqClass {
    /// Model is confident in its output. Action: emit token, continue.
    Confident = 0,
    /// High-volume uncertainty — many plausible candidates.
    /// Action: trigger Engram lookup with hidden state as semantic query.
    HighVolume = 1,
    /// Spectral collapse — hidden state norms collapsing toward zero.
    /// Action: abort generation, flag as unknowable, escalate.
    SpectralCollapse = 2,
    /// Ambiguous uncertainty. Conservative action: treat as HighVolume.
    Uncertain = 3,
}

impl UqClass {
    /// Parse a class from its `u8` discriminant (0–3).
    ///
    /// Returns `None` for values outside the valid range.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Confident),
            1 => Some(Self::HighVolume),
            2 => Some(Self::SpectralCollapse),
            3 => Some(Self::Uncertain),
            _ => None,
        }
    }
}

impl TryFrom<u8> for UqClass {
    type Error = u8;

    fn try_from(value: u8) -> core::result::Result<Self, Self::Error> {
        Self::from_u8(value).ok_or(value)
    }
}
```

- [ ] **Step 4: Implement UqHeadConfig**

Add below `UqClass`:

```rust
/// Configuration for the Uncertainty Quantification Head.
#[derive(Debug, Clone)]
pub struct UqHeadConfig {
    /// Number of input features. Default: 8.
    pub num_features: usize,
    /// Hidden dimension of the classifier MLP. Default: 32.
    pub hidden_dim: usize,
    /// Number of output classes. Default: 4.
    pub num_classes: usize,
}

impl Default for UqHeadConfig {
    fn default() -> Self {
        Self {
            num_features: 8,
            hidden_dim: 32,
            num_classes: 4,
        }
    }
}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- --nocapture
```

Expected: 5 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/uq_head.rs
git commit -m "feat(inference): UqClass enum and UqHeadConfig for UQ Head"
```

---

### Task 2: UqHead struct, constructors, and forward pass

**Files:**
- Modify: `crates/harmony-inference/src/uq_head.rs`

This task adds the core module: UqHead struct holding 6 weight tensors, UqOutput
for forward results, `new()` for random init, `from_tensors()` with shape
validation, and `forward()` implementing the two-path MLP.

**Context:** The crate already uses `candle_nn::ops::softmax_last_dim` (in
`block_attnres.rs:220`) and `candle_nn::ops::sigmoid` (in
`engram_residual.rs:172`). Both are available without additional dependencies.

- [ ] **Step 1: Write failing tests for UqHead construction and forward pass**

Add these tests to the existing `mod tests` block:

```rust
    // ── UqHead construction ──

    fn test_config() -> UqHeadConfig {
        UqHeadConfig {
            num_features: 8,
            hidden_dim: 32,
            num_classes: 4,
        }
    }

    #[test]
    fn new_creates_module() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        assert_eq!(head.config().num_features, 8);
        assert_eq!(head.config().hidden_dim, 32);
        assert_eq!(head.config().num_classes, 4);
    }

    #[test]
    fn from_tensors_accepts_correct_shapes() {
        let cfg = test_config();
        let d = &Device::Cpu;
        let fc1 = Tensor::zeros((8, 32), candle_core::DType::F32, d).unwrap();
        let b1 = Tensor::zeros(32, candle_core::DType::F32, d).unwrap();
        let fc2 = Tensor::zeros((32, 4), candle_core::DType::F32, d).unwrap();
        let b2 = Tensor::zeros(4, candle_core::DType::F32, d).unwrap();
        let cw = Tensor::zeros((8, 1), candle_core::DType::F32, d).unwrap();
        let cb = Tensor::zeros(1, candle_core::DType::F32, d).unwrap();
        let result = UqHead::from_tensors(&cfg, fc1, b1, fc2, b2, cw, cb);
        assert!(result.is_ok());
    }

    #[test]
    fn from_tensors_rejects_wrong_fc1_shape() {
        let cfg = test_config();
        let d = &Device::Cpu;
        // Wrong: [4, 32] instead of [8, 32]
        let fc1 = Tensor::zeros((4, 32), candle_core::DType::F32, d).unwrap();
        let b1 = Tensor::zeros(32, candle_core::DType::F32, d).unwrap();
        let fc2 = Tensor::zeros((32, 4), candle_core::DType::F32, d).unwrap();
        let b2 = Tensor::zeros(4, candle_core::DType::F32, d).unwrap();
        let cw = Tensor::zeros((8, 1), candle_core::DType::F32, d).unwrap();
        let cb = Tensor::zeros(1, candle_core::DType::F32, d).unwrap();
        let result = UqHead::from_tensors(&cfg, fc1, b1, fc2, b2, cw, cb);
        assert!(result.is_err());
    }

    // ── Forward pass ──

    #[test]
    fn forward_output_shapes() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        let features = Tensor::zeros((1, 8), candle_core::DType::F32, &Device::Cpu).unwrap();
        let output = head.forward(&features).unwrap();
        assert_eq!(output.class_probs.dims(), &[1, 4]);
        assert_eq!(output.confidence.dims(), &[1, 1]);
    }

    #[test]
    fn forward_batch_output_shapes() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        let features = Tensor::zeros((5, 8), candle_core::DType::F32, &Device::Cpu).unwrap();
        let output = head.forward(&features).unwrap();
        assert_eq!(output.class_probs.dims(), &[5, 4]);
        assert_eq!(output.confidence.dims(), &[5, 1]);
    }

    #[test]
    fn forward_class_probs_are_valid_distribution() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        let features = Tensor::randn(0f32, 1f32, (3, 8), &Device::Cpu).unwrap();
        let output = head.forward(&features).unwrap();

        // Each row should sum to ~1.0 (softmax output)
        let sums: Vec<f32> = output
            .class_probs
            .sum_keepdim(candle_core::D::Minus1)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &s) in sums.iter().enumerate() {
            assert!(
                (s - 1.0).abs() < 1e-5,
                "row {i} class_probs sum = {s}, expected ~1.0"
            );
        }

        // All probabilities should be in [0, 1]
        let all_probs: Vec<f32> = output
            .class_probs
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &p) in all_probs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&p),
                "class_probs[{i}] = {p}, expected [0, 1]"
            );
        }
    }

    #[test]
    fn forward_confidence_in_zero_one() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        let features = Tensor::randn(0f32, 1f32, (3, 8), &Device::Cpu).unwrap();
        let output = head.forward(&features).unwrap();

        let confs: Vec<f32> = output
            .confidence
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &c) in confs.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&c),
                "confidence[{i}] = {c}, expected [0, 1]"
            );
        }
    }

    #[test]
    fn forward_zero_features_produce_uniform_probs() {
        // With zero-initialized weights and zero input, matmul produces zeros.
        // softmax(zeros) = uniform = [0.25, 0.25, 0.25, 0.25]
        // sigmoid(0) = 0.5
        let cfg = test_config();
        let d = &Device::Cpu;
        let fc1 = Tensor::zeros((8, 32), candle_core::DType::F32, d).unwrap();
        let b1 = Tensor::zeros(32, candle_core::DType::F32, d).unwrap();
        let fc2 = Tensor::zeros((32, 4), candle_core::DType::F32, d).unwrap();
        let b2 = Tensor::zeros(4, candle_core::DType::F32, d).unwrap();
        let cw = Tensor::zeros((8, 1), candle_core::DType::F32, d).unwrap();
        let cb = Tensor::zeros(1, candle_core::DType::F32, d).unwrap();
        let head = UqHead::from_tensors(&cfg, fc1, b1, fc2, b2, cw, cb).unwrap();

        let features = Tensor::zeros((1, 8), candle_core::DType::F32, d).unwrap();
        let output = head.forward(&features).unwrap();

        let probs: Vec<f32> = output
            .class_probs
            .flatten_all()
            .unwrap()
            .to_vec1()
            .unwrap();
        for (i, &p) in probs.iter().enumerate() {
            assert!(
                (p - 0.25).abs() < 1e-5,
                "probs[{i}] = {p}, expected 0.25 (uniform)"
            );
        }

        let conf: f32 = output
            .confidence
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()[0];
        assert!(
            (conf - 0.5).abs() < 1e-5,
            "confidence = {conf}, expected 0.5 (sigmoid(0))"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- 2>&1 | head -20
```

Expected: compilation error — `UqHead`, `UqOutput` not defined.

- [ ] **Step 3: Implement UqOutput and UqHead struct**

Add below `UqHeadConfig` (above `#[cfg(test)]`):

```rust
/// Output of the UQ Head forward pass.
pub struct UqOutput {
    /// Class probabilities after softmax. Shape: `[batch, num_classes]`.
    pub class_probs: Tensor,
    /// Confidence scalar after sigmoid. Shape: `[batch, 1]`.
    pub confidence: Tensor,
}

/// Uncertainty Quantification Head — parallel metacognitive MLP.
///
/// Two-path classifier operating on pre-extracted features:
/// - **Classifier:** Linear(F→H) → ReLU → Linear(H→C) → softmax
/// - **Confidence:** Linear(F→1) → sigmoid
///
/// Does not modify the model's forward pass. The caller extracts
/// features and uses the classification for routing decisions.
pub struct UqHead {
    classifier_fc1: Tensor, // [num_features, hidden_dim]
    classifier_b1: Tensor,  // [hidden_dim]
    classifier_fc2: Tensor, // [hidden_dim, num_classes]
    classifier_b2: Tensor,  // [num_classes]
    confidence_w: Tensor,   // [num_features, 1]
    confidence_b: Tensor,   // [1]
    config: UqHeadConfig,
}

impl UqHead {
    /// Create with small random weights (for testing / fresh init).
    pub fn new(config: &UqHeadConfig, device: &Device) -> Result<Self> {
        let f = config.num_features;
        let h = config.hidden_dim;
        let c = config.num_classes;
        let scale_f = 1.0 / (f as f64).sqrt();
        let scale_h = 1.0 / (h as f64).sqrt();

        Ok(Self {
            classifier_fc1: (Tensor::randn(0f32, 1f32, (f, h), device)? * scale_f)?,
            classifier_b1: Tensor::zeros(h, candle_core::DType::F32, device)?,
            classifier_fc2: (Tensor::randn(0f32, 1f32, (h, c), device)? * scale_h)?,
            classifier_b2: Tensor::zeros(c, candle_core::DType::F32, device)?,
            confidence_w: (Tensor::randn(0f32, 1f32, (f, 1), device)? * scale_f)?,
            confidence_b: Tensor::zeros(1, candle_core::DType::F32, device)?,
            config: config.clone(),
        })
    }

    /// Create from pre-loaded weight tensors (for loading trained weights).
    ///
    /// Validates that all tensor dimensions match the config. Returns an error
    /// if any shape is wrong — catching mismatches early prevents confusing
    /// matmul errors during forward().
    pub fn from_tensors(
        config: &UqHeadConfig,
        classifier_fc1: Tensor,
        classifier_b1: Tensor,
        classifier_fc2: Tensor,
        classifier_b2: Tensor,
        confidence_w: Tensor,
        confidence_b: Tensor,
    ) -> Result<Self> {
        let f = config.num_features;
        let h = config.hidden_dim;
        let c = config.num_classes;

        if classifier_fc1.dims() != [f, h] {
            candle_core::bail!(
                "classifier_fc1: expected [{f}, {h}], got {:?}",
                classifier_fc1.dims()
            );
        }
        if classifier_b1.dims() != [h] {
            candle_core::bail!(
                "classifier_b1: expected [{h}], got {:?}",
                classifier_b1.dims()
            );
        }
        if classifier_fc2.dims() != [h, c] {
            candle_core::bail!(
                "classifier_fc2: expected [{h}, {c}], got {:?}",
                classifier_fc2.dims()
            );
        }
        if classifier_b2.dims() != [c] {
            candle_core::bail!(
                "classifier_b2: expected [{c}], got {:?}",
                classifier_b2.dims()
            );
        }
        if confidence_w.dims() != [f, 1] {
            candle_core::bail!(
                "confidence_w: expected [{f}, 1], got {:?}",
                confidence_w.dims()
            );
        }
        if confidence_b.dims() != [1] {
            candle_core::bail!(
                "confidence_b: expected [1], got {:?}",
                confidence_b.dims()
            );
        }

        Ok(Self {
            classifier_fc1,
            classifier_b1,
            classifier_fc2,
            classifier_b2,
            confidence_w,
            confidence_b,
            config: config.clone(),
        })
    }

    /// Forward pass — returns class probabilities and confidence score.
    ///
    /// `features` shape: `[batch, num_features]`.
    /// Returns `UqOutput` with `class_probs [batch, num_classes]` and
    /// `confidence [batch, 1]`.
    pub fn forward(&self, features: &Tensor) -> Result<UqOutput> {
        // Classifier: Linear(F→H) → ReLU → Linear(H→C) → softmax
        let h = features.matmul(&self.classifier_fc1)?;
        let h = h.broadcast_add(&self.classifier_b1)?;
        let h = h.relu()?;
        let h = h.matmul(&self.classifier_fc2)?;
        let h = h.broadcast_add(&self.classifier_b2)?;
        let class_probs = candle_nn::ops::softmax_last_dim(&h)?;

        // Confidence: Linear(F→1) → sigmoid
        let c = features.matmul(&self.confidence_w)?;
        let c = c.broadcast_add(&self.confidence_b)?;
        let confidence = candle_nn::ops::sigmoid(&c)?;

        Ok(UqOutput {
            class_probs,
            confidence,
        })
    }

    /// Reference to the config.
    pub fn config(&self) -> &UqHeadConfig {
        &self.config
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- --nocapture
```

Expected: all 12 tests pass (5 from Task 1 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/uq_head.rs
git commit -m "feat(inference): UqHead struct with forward pass — two-path MLP classifier"
```

---

### Task 3: classify() convenience method and module export

**Files:**
- Modify: `crates/harmony-inference/src/uq_head.rs`
- Modify: `crates/harmony-inference/src/lib.rs:20,30`

This task adds the single-sample `classify()` method that extracts a `UqClass`
and confidence `f32` from the forward output, then wires the module into the
crate's public API.

- [ ] **Step 1: Write failing tests for classify()**

Add these tests to the existing `mod tests` block:

```rust
    // ── classify() ──

    #[test]
    fn classify_returns_valid_class_and_confidence() {
        let cfg = test_config();
        let head = UqHead::new(&cfg, &Device::Cpu).unwrap();
        let features = Tensor::randn(0f32, 1f32, (1, 8), &Device::Cpu).unwrap();
        let (class, conf) = head.classify(&features).unwrap();

        // Class must be one of the 4 variants
        let class_u8 = class as u8;
        assert!(class_u8 <= 3, "class discriminant {class_u8} out of range");

        // Confidence must be in [0, 1] (sigmoid output)
        assert!(
            (0.0..=1.0).contains(&conf),
            "confidence {conf} out of [0, 1]"
        );
    }

    #[test]
    fn classify_with_zero_weights_returns_confident() {
        // Zero weights → softmax(zeros) = uniform → argmax picks index 0 = Confident
        let cfg = test_config();
        let d = &Device::Cpu;
        let fc1 = Tensor::zeros((8, 32), candle_core::DType::F32, d).unwrap();
        let b1 = Tensor::zeros(32, candle_core::DType::F32, d).unwrap();
        let fc2 = Tensor::zeros((32, 4), candle_core::DType::F32, d).unwrap();
        let b2 = Tensor::zeros(4, candle_core::DType::F32, d).unwrap();
        let cw = Tensor::zeros((8, 1), candle_core::DType::F32, d).unwrap();
        let cb = Tensor::zeros(1, candle_core::DType::F32, d).unwrap();
        let head = UqHead::from_tensors(&cfg, fc1, b1, fc2, b2, cw, cb).unwrap();

        let features = Tensor::zeros((1, 8), candle_core::DType::F32, d).unwrap();
        let (class, conf) = head.classify(&features).unwrap();

        // argmax of uniform distribution picks first index (0 = Confident)
        assert_eq!(class, UqClass::Confident);
        assert!((conf - 0.5).abs() < 1e-5, "confidence = {conf}, expected 0.5");
    }

    #[test]
    fn classify_biased_toward_class_2() {
        // Bias classifier_b2 heavily toward class 2 (SpectralCollapse)
        let cfg = test_config();
        let d = &Device::Cpu;
        let fc1 = Tensor::zeros((8, 32), candle_core::DType::F32, d).unwrap();
        let b1 = Tensor::zeros(32, candle_core::DType::F32, d).unwrap();
        let fc2 = Tensor::zeros((32, 4), candle_core::DType::F32, d).unwrap();
        // b2 = [0, 0, 100, 0] — heavily biased toward class 2
        let b2 = Tensor::new(&[0.0f32, 0.0, 100.0, 0.0], d).unwrap();
        let cw = Tensor::zeros((8, 1), candle_core::DType::F32, d).unwrap();
        let cb = Tensor::zeros(1, candle_core::DType::F32, d).unwrap();
        let head = UqHead::from_tensors(&cfg, fc1, b1, fc2, b2, cw, cb).unwrap();

        let features = Tensor::zeros((1, 8), candle_core::DType::F32, d).unwrap();
        let (class, _) = head.classify(&features).unwrap();

        assert_eq!(class, UqClass::SpectralCollapse);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- 2>&1 | head -20
```

Expected: compilation error — `classify` method not defined.

- [ ] **Step 3: Implement classify()**

Add this method to the `impl UqHead` block, after `forward()`:

```rust
    /// Single-sample classification for inference routing.
    ///
    /// Takes features of shape `[1, num_features]`. Returns the predicted
    /// `UqClass` (argmax of class probabilities) and the confidence scalar.
    pub fn classify(&self, features: &Tensor) -> Result<(UqClass, f32)> {
        let output = self.forward(features)?;

        let class_idx: u32 = output
            .class_probs
            .argmax(candle_core::D::Minus1)?
            .squeeze(0)?
            .to_scalar()?;

        let class = UqClass::from_u8(class_idx as u8).ok_or_else(|| {
            candle_core::Error::Msg(format!("argmax returned invalid class index: {class_idx}"))
        })?;

        let conf: f32 = output.confidence.squeeze(0)?.squeeze(0)?.to_scalar()?;

        Ok((class, conf))
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib uq_head -- --nocapture
```

Expected: all 15 tests pass.

- [ ] **Step 5: Register module in lib.rs**

In `crates/harmony-inference/src/lib.rs`:

After line 20 (`pub mod block_attnres;`), add:
```rust
pub mod uq_head;
```

After line 30 (`pub use block_attnres::{BlockAttnRes, BlockAttnResConfig, BlockAttnResState};`), add:
```rust
pub use uq_head::{UqClass, UqHead, UqHeadConfig, UqOutput};
```

- [ ] **Step 6: Verify full crate compiles and all tests pass**

```bash
cd crates/harmony-inference && cargo test --lib
```

Expected: all existing tests + 15 new UQ Head tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-inference/src/uq_head.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): classify() method and module export for UQ Head"
```

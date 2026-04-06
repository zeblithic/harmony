# Phase 0d: Uncertainty Quantification Head — Design Spec

**Epic:** harmony-ct87 / harmony-iyot
**Phase:** 0d (parallel with 0a, 0b, 0c)
**Status:** Design approved
**Date:** 2026-04-06

## Goal

Implement the UQ Head classification MLP as a candle inference module. Takes 8
pre-extracted features and produces a 4-class uncertainty classification plus a
scalar confidence score. ~300 parameters, negligible inference overhead.

Feature extraction from model internals (hidden state norms, logit entropy, etc.)
is deferred to Phase 0e (custom model forward pass).

## Architecture

The UQ Head is a **parallel metacognitive monitor** — it observes model internals
after the forward pass and classifies the model's confidence state. It does not
modify the forward pass or make routing decisions; the caller uses the returned
classification to decide whether to emit, retrieve, or abort.

### Feature Vector (8 floats, extracted by caller)

| Index | Feature | Description |
|-------|---------|-------------|
| f1–f4 | Hidden state L2 norms | At layers 0, 8, 16, 23 (one per network quadrant) |
| f5 | Norm trajectory slope | Growing, stable, or collapsing? |
| f6 | Logit entropy (Shannon) | Distribution flatness |
| f7 | Top-k probability mass | Distribution concentration |
| f8 | Attention lookback ratio | Prompt attention vs generated-token attention |

### Classification MLP

```
Input: [batch, 8]

Classifier path:
  Linear(8 → 32) → ReLU → Linear(32 → 4) → softmax → class_probs [batch, 4]

Confidence path:
  Linear(8 → 1) → sigmoid → confidence [batch, 1]
```

Parameter count:
- Classifier: (8×32 + 32) + (32×4 + 4) = 420 params
- Confidence: (8×1 + 1) = 9 params
- Total: 429 params

### UQ Classes

```rust
#[repr(u8)]
pub enum UqClass {
    Confident = 0,        // emit token, continue
    HighVolume = 1,       // trigger Engram lookup
    SpectralCollapse = 2, // abort, escalate
    Uncertain = 3,        // conservative: treat as HighVolume
}
```

### Dual-Trigger Geometry

**High-volume uncertainty:**
- Norms: stable, normal magnitude
- Entropy: HIGH (flat distribution)
- Top-k mass: low concentration
- Meaning: many candidates, model knows the category but not the answer
- Action: trigger Engram lookup with hidden state as semantic query

**Spectral collapse (gray vector):**
- Norms: collapsing toward zero across depth
- Entropy: may be LOW (false confidence on random token)
- Norm slope: steep negative
- Meaning: no signal, model has nothing
- Action: abort generation, flag as unknowable, escalate

**Critical insight:** Spectral collapse can produce LOW entropy (confident wrong
answer). Hidden state norm monitoring catches this; logit entropy alone would miss it.

## Module Design

### File: `crates/harmony-inference/src/uq_head.rs`

Follows the same pattern as `block_attnres.rs`:

- `UqHeadConfig` — configuration struct
- `UqHead` — module holding weight tensors
- `UqOutput` — forward pass output (class probs + confidence)
- `UqClass` — classification enum with `from_u8` / `TryFrom<u8>`

### Types

```rust
#[derive(Debug, Clone)]
pub struct UqHeadConfig {
    pub num_features: usize,  // default: 8
    pub hidden_dim: usize,    // default: 32
    pub num_classes: usize,   // default: 4
}

pub struct UqOutput {
    pub class_probs: Tensor,  // [batch, num_classes]
    pub confidence: Tensor,   // [batch, 1]
}

pub struct UqHead {
    classifier_fc1: Tensor,   // [num_features, hidden_dim]
    classifier_b1: Tensor,    // [hidden_dim]
    classifier_fc2: Tensor,   // [hidden_dim, num_classes]
    classifier_b2: Tensor,    // [num_classes]
    confidence_w: Tensor,     // [num_features, 1]
    confidence_b: Tensor,     // [1]
    config: UqHeadConfig,
}
```

### Interface

```rust
impl UqHead {
    /// Create with random weights (for testing / fresh init).
    pub fn new(config: &UqHeadConfig, device: &Device) -> Result<Self>;

    /// Create from pre-loaded weight tensors (for loading trained weights).
    pub fn from_tensors(
        config: &UqHeadConfig,
        classifier_fc1: Tensor,
        classifier_b1: Tensor,
        classifier_fc2: Tensor,
        classifier_b2: Tensor,
        confidence_w: Tensor,
        confidence_b: Tensor,
    ) -> Result<Self>;

    /// Forward pass — returns raw tensors for batched use.
    /// features: [batch, num_features]
    pub fn forward(&self, features: &Tensor) -> Result<UqOutput>;

    /// Single-sample classification for inference routing.
    /// features: [1, num_features]
    /// Returns (predicted class, confidence scalar).
    pub fn classify(&self, features: &Tensor) -> Result<(UqClass, f32)>;
}
```

### Routing Contract

The UQ Head only classifies. It returns `(UqClass, f32)` — the caller decides
what to do. The routing logic from the spec:

```
confident (score > tau_high)    → emit token, continue
high_volume                     → Engram lookup, re-run with injection
spectral_collapse               → abort, escalate
uncertain (ambiguous)           → conservative: treat as high_volume
```

This routing is implemented in Phase 0e, not here.

## Scope Boundary

**In scope (0d):**
- `UqHeadConfig`, `UqHead`, `UqOutput`, `UqClass` types
- `new()` and `from_tensors()` constructors
- `forward()` and `classify()` methods
- `UqClass::from_u8()` and `TryFrom<u8>` for UqClass
- Unit tests (config, construction, forward pass shapes, classify routing,
  softmax/sigmoid properties, from_tensors validation)
- Module registration in `lib.rs`

**Out of scope (deferred to 0e):**
- Feature extraction from model internals (L2 norms, entropy, etc.)
- Integration into the custom model forward pass
- Routing decision logic
- VarBuilder / GGUF weight loading (0e/0g handle this)

## Future: SConU Integration (v1.1)

The UQ Head's interface (features → class + confidence) is designed to be
SConU-ready. Selective Conformal Uncertainty wraps the output in conformal
prediction p-values for mathematically guaranteed abstention. Requires a
calibration dataset. Not in scope for Phase 0.

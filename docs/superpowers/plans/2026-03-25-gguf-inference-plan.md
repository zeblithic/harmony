# GGUF Inference Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Load quantized GGUF models (Qwen 3.5) and run token-by-token inference on CPU, wrapping candle-transformers behind a composable `InferenceEngine` trait.

**Architecture:** New `harmony-inference` library crate in `crates/harmony-inference/`. Wraps `candle_transformers::models::quantized_qwen3::ModelWeights` with a clean `InferenceEngine` trait. Sampling logic is isolated in its own module for TDD. The engine manages KV cache position and token history internally; callers drive the autoregressive loop.

**Tech Stack:** candle-core 0.9, candle-transformers 0.9, tokenizers 0.22, rand 0.8, thiserror 2

**Spec:** `docs/superpowers/specs/2026-03-25-gguf-inference-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `Cargo.toml` (workspace root) | Add harmony-inference member + candle/tokenizers workspace deps |
| `crates/harmony-inference/Cargo.toml` | Crate manifest with candle, tokenizers, rand, thiserror |
| `crates/harmony-inference/src/lib.rs` | `InferenceEngine` trait, `SamplingParams` struct, module re-exports |
| `crates/harmony-inference/src/error.rs` | `InferenceError` enum |
| `crates/harmony-inference/src/sampling.rs` | Pure sampling functions (temperature, top-k, top-p, repeat penalty) + unit tests |
| `crates/harmony-inference/src/engine.rs` | `QwenEngine` struct + `InferenceEngine` impl + error path unit tests |
| `crates/harmony-inference/tests/integration.rs` | `#[ignore]` tests requiring real GGUF model files |

---

### Task 1: Crate Scaffold + Types + Error Types

Create the crate structure, define all public types, and verify compilation. No logic — just the skeleton.

**Files:**
- Modify: `Cargo.toml` (workspace root, lines 3-36 for members, lines 45-100+ for deps)
- Create: `crates/harmony-inference/Cargo.toml`
- Create: `crates/harmony-inference/src/error.rs`
- Create: `crates/harmony-inference/src/lib.rs`
- Create: `crates/harmony-inference/src/sampling.rs` (stub)
- Create: `crates/harmony-inference/src/engine.rs` (stub)

- [ ] **Step 1: Add workspace dependencies and member**

In `Cargo.toml` (workspace root), add `"crates/harmony-inference"` to the `[workspace] members` list (after `harmony-ingest`).

Add these to `[workspace.dependencies]` in the ML inference section (after the `# WASM runtime` block):

```toml
# ML inference
candle-core = "0.9"
candle-transformers = "0.9"
tokenizers = "0.22"
```

Also add the internal crate reference in the internal crates section:

```toml
harmony-inference = { path = "crates/harmony-inference" }
```

- [ ] **Step 2: Create crate Cargo.toml**

Create `crates/harmony-inference/Cargo.toml`:

```toml
[package]
name = "harmony-inference"
version.workspace = true
edition.workspace = true
license.workspace = true
rust-version.workspace = true
repository.workspace = true
description = "GGUF model loader and inference engine wrapping candle-transformers"

[features]
default = []
cuda = ["candle-core/cuda"]
metal = ["candle-core/metal"]

[dependencies]
candle-core = { workspace = true }
candle-transformers = { workspace = true }
tokenizers = { workspace = true }
rand = { workspace = true }
thiserror = { workspace = true, features = ["std"] }
```

- [ ] **Step 3: Create error.rs**

Create `crates/harmony-inference/src/error.rs`:

```rust
/// Errors from inference operations.
#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    /// No model loaded — call `load_gguf()` first.
    #[error("no model loaded — call load_gguf() first")]
    ModelNotLoaded,

    /// No tokenizer loaded — call `load_tokenizer()` first.
    #[error("no tokenizer loaded — call load_tokenizer() first")]
    TokenizerNotLoaded,

    /// GGUF file is invalid or unsupported.
    #[error("invalid GGUF: {0}")]
    InvalidGguf(String),

    /// Tokenizer JSON is invalid.
    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    /// Forward pass failed (tensor operation error).
    #[error("forward pass failed: {0}")]
    ForwardFailed(String),

    /// Sampling failed (empty logits, invalid params).
    #[error("sampling failed: {0}")]
    SamplingFailed(String),
}
```

- [ ] **Step 4: Create lib.rs with trait and types**

Create `crates/harmony-inference/src/lib.rs`:

```rust
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
//! let tokens = engine.tokenize("Hello")?;
//! let logits = engine.forward(&tokens)?;
//! let next = engine.sample(&logits, &SamplingParams::greedy())?;
//! ```

pub mod engine;
pub mod error;
pub mod sampling;

pub use engine::QwenEngine;
pub use error::InferenceError;

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
    pub repeat_penalty: f32,
}

impl SamplingParams {
    /// Greedy decoding (deterministic argmax).
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_p: 1.0,
            top_k: 0,
            repeat_penalty: 1.0,
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
        }
    }
}

/// Trait for running inference on quantized language models.
///
/// The caller drives the autoregressive loop, enabling streaming output,
/// custom stopping criteria, and future Engram embedding injection.
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
    /// Manages KV cache internally. First call = prefill, subsequent = decode.
    /// Advances the internal position counter by `tokens.len()`.
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError>;

    /// Sample the next token from logits.
    ///
    /// Uses internal token history (populated by `forward()`) for repeat penalty.
    fn sample(&self, logits: &[f32], params: &SamplingParams) -> Result<u32, InferenceError>;

    /// Reset the KV cache and position (start a new conversation).
    fn reset(&mut self);
}
```

- [ ] **Step 5: Create module stubs**

Create `crates/harmony-inference/src/sampling.rs`:

```rust
//! Pure sampling functions — no model state needed.

use crate::error::InferenceError;
```

Create `crates/harmony-inference/src/engine.rs`:

```rust
//! QwenEngine: candle-based inference engine for quantized Qwen3 models.

use candle_core::Device;
use crate::{InferenceEngine, SamplingParams};
use crate::error::InferenceError;

/// Inference engine wrapping candle-transformers' quantized Qwen3 implementation.
pub struct QwenEngine {
    device: Device,
}

impl QwenEngine {
    /// Create a new engine targeting the given device.
    pub fn new(device: Device) -> Self {
        Self { device }
    }
}
```

- [ ] **Step 6: Verify compilation**

Run: `cargo check -p harmony-inference`
Expected: Compiles with warnings (unused fields, unimplemented trait)

Run: `cargo check --workspace`
Expected: Entire workspace still compiles (new workspace deps don't conflict)

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml crates/harmony-inference/
git commit -m "feat(inference): scaffold harmony-inference crate with types and errors"
```

---

### Task 2: Sampling Module (TDD)

Implement all pure sampling functions with test-first development. These functions operate on logit slices — no model or candle dependency.

**Files:**
- Modify: `crates/harmony-inference/src/sampling.rs`

**Reference:** Spec section "Sampling" (lines 129-139)

- [ ] **Step 1: Write test for greedy sampling, then implement**

Add to `crates/harmony-inference/src/sampling.rs`:

```rust
use crate::error::InferenceError;
use rand::Rng;

/// Return the index of the maximum logit (argmax).
pub fn sample_greedy(logits: &[f32]) -> Result<u32, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::SamplingFailed("empty logits".into()));
    }
    let (idx, _) = logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap(); // safe: non-empty checked above
    Ok(idx as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_returns_argmax() {
        let logits = [1.0_f32, 3.0, 2.0, 0.5];
        assert_eq!(sample_greedy(&logits).unwrap(), 1);
    }

    #[test]
    fn test_greedy_first_max_on_tie() {
        let logits = [5.0_f32, 5.0, 1.0];
        // On tie, max_by returns last max; verify it returns a valid max index
        let result = sample_greedy(&logits).unwrap();
        assert!(result == 0 || result == 1);
        assert_eq!(logits[result as usize], 5.0);
    }

    #[test]
    fn test_greedy_empty_logits() {
        let result = sample_greedy(&[]);
        assert!(matches!(result, Err(InferenceError::SamplingFailed(_))));
    }
}
```

Run: `cargo test -p harmony-inference sampling::tests::test_greedy -- --nocapture`
Expected: 3 tests PASS

- [ ] **Step 2: Write test for temperature scaling, then implement**

Add to `sampling.rs`:

```rust
/// Apply temperature scaling in place: `logits[i] /= temperature`.
pub fn apply_temperature(logits: &mut [f32], temperature: f32) {
    for l in logits.iter_mut() {
        *l /= temperature;
    }
}
```

Add tests:

```rust
    #[test]
    fn test_temperature_scales_logits() {
        let mut logits = [2.0_f32, 4.0, 6.0];
        apply_temperature(&mut logits, 2.0);
        assert_eq!(logits, [1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_temperature_one_is_identity() {
        let mut logits = [1.5_f32, 2.5, 3.5];
        let original = logits;
        apply_temperature(&mut logits, 1.0);
        assert_eq!(logits, original);
    }
```

Run: `cargo test -p harmony-inference sampling::tests::test_temperature`
Expected: 2 tests PASS

- [ ] **Step 3: Write test for top-k filtering, then implement**

Add to `sampling.rs`:

```rust
/// Keep only the `k` highest logits; set the rest to `-inf`.
///
/// If `k == 0` or `k >= logits.len()`, this is a no-op.
pub fn apply_top_k(logits: &mut [f32], k: u32) {
    if k == 0 || k as usize >= logits.len() {
        return;
    }
    // Sort indices by logit value descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    // Set everything after position k to -inf
    for &(idx, _) in &indexed[k as usize..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}
```

Add tests:

```rust
    #[test]
    fn test_top_k_keeps_highest() {
        let mut logits = [1.0_f32, 4.0, 2.0, 3.0];
        apply_top_k(&mut logits, 2);
        // Indices 1 (4.0) and 3 (3.0) kept, rest -inf
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 4.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
        assert_eq!(logits[3], 3.0);
    }

    #[test]
    fn test_top_k_zero_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_top_k(&mut logits, 0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_k_one_keeps_max_only() {
        let mut logits = [1.0_f32, 5.0, 3.0];
        apply_top_k(&mut logits, 1);
        assert_eq!(logits[0], f32::NEG_INFINITY);
        assert_eq!(logits[1], 5.0);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_ge_len_is_noop() {
        let mut logits = [1.0_f32, 2.0];
        let original = logits;
        apply_top_k(&mut logits, 5);
        assert_eq!(logits, original);
    }
```

Run: `cargo test -p harmony-inference sampling::tests::test_top_k`
Expected: 4 tests PASS

- [ ] **Step 4: Write test for top-p (nucleus) filtering, then implement**

Add to `sampling.rs`:

```rust
/// Apply nucleus (top-p) filtering: sort by probability, keep tokens whose
/// cumulative probability mass ≤ `top_p`, set the rest to `-inf`.
///
/// If `top_p >= 1.0`, this is a no-op.
pub fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if top_p >= 1.0 {
        return;
    }
    // Compute softmax probabilities (numerically stable)
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    // Sort indices by probability descending
    let mut indexed: Vec<(usize, f32)> = exps.iter().map(|&e| e / sum).enumerate().collect();
    indexed.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Find cutoff: keep tokens until cumulative mass exceeds top_p
    let mut cumsum = 0.0;
    let mut keep = Vec::new();
    for (idx, prob) in &indexed {
        cumsum += prob;
        keep.push(*idx);
        if cumsum >= top_p {
            break;
        }
    }

    // Set everything not in `keep` to -inf
    let keep_set: std::collections::HashSet<usize> = keep.into_iter().collect();
    for (i, l) in logits.iter_mut().enumerate() {
        if !keep_set.contains(&i) {
            *l = f32::NEG_INFINITY;
        }
    }
}
```

Add tests:

```rust
    #[test]
    fn test_top_p_filters_low_probability() {
        // logits [10.0, 1.0, 0.0] → probabilities ~[0.9999, 0.0001, 0.00004]
        // top_p=0.9 should keep only index 0
        let mut logits = [10.0_f32, 1.0, 0.0];
        apply_top_p(&mut logits, 0.9);
        assert_eq!(logits[0], 10.0);
        assert_eq!(logits[1], f32::NEG_INFINITY);
        assert_eq!(logits[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_p_one_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_top_p(&mut logits, 1.0);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_top_p_keeps_enough_mass() {
        // Equal logits → equal probabilities (0.25 each)
        // top_p=0.6 → need at least 3 tokens (0.25+0.25+0.25 = 0.75 ≥ 0.6)
        let mut logits = [1.0_f32, 1.0, 1.0, 1.0];
        apply_top_p(&mut logits, 0.6);
        let kept = logits.iter().filter(|&&l| l != f32::NEG_INFINITY).count();
        assert!(kept >= 3, "should keep at least 3 tokens, kept {kept}");
    }
```

Run: `cargo test -p harmony-inference sampling::tests::test_top_p`
Expected: 3 tests PASS

- [ ] **Step 5: Write test for repeat penalty, then implement**

Add to `sampling.rs`:

```rust
/// Apply repeat penalty for tokens seen in context.
///
/// For each token in `context_tokens`:
/// - If its logit is positive, divide by `penalty`
/// - If its logit is negative, multiply by `penalty`
///
/// This reduces the probability of recently seen tokens.
pub fn apply_repeat_penalty(logits: &mut [f32], penalty: f32, context_tokens: &[u32]) {
    if penalty == 1.0 {
        return;
    }
    for &token in context_tokens {
        if let Some(logit) = logits.get_mut(token as usize) {
            if *logit > 0.0 {
                *logit /= penalty;
            } else {
                *logit *= penalty;
            }
        }
    }
}
```

Add tests:

```rust
    #[test]
    fn test_repeat_penalty_reduces_positive_logits() {
        let mut logits = [1.0_f32, 4.0, 2.0];
        apply_repeat_penalty(&mut logits, 2.0, &[1]); // penalize token 1
        assert_eq!(logits[0], 1.0); // unchanged
        assert_eq!(logits[1], 2.0); // 4.0 / 2.0
        assert_eq!(logits[2], 2.0); // unchanged
    }

    #[test]
    fn test_repeat_penalty_amplifies_negative_logits() {
        let mut logits = [1.0_f32, -2.0, 3.0];
        apply_repeat_penalty(&mut logits, 2.0, &[1]);
        assert_eq!(logits[1], -4.0); // -2.0 * 2.0
    }

    #[test]
    fn test_repeat_penalty_one_is_noop() {
        let mut logits = [1.0_f32, 2.0, 3.0];
        let original = logits;
        apply_repeat_penalty(&mut logits, 1.0, &[0, 1, 2]);
        assert_eq!(logits, original);
    }

    #[test]
    fn test_repeat_penalty_out_of_bounds_ignored() {
        let mut logits = [1.0_f32, 2.0];
        let original = logits;
        apply_repeat_penalty(&mut logits, 2.0, &[99]); // index 99 doesn't exist
        assert_eq!(logits, original);
    }
```

Run: `cargo test -p harmony-inference sampling::tests::test_repeat_penalty`
Expected: 4 tests PASS

- [ ] **Step 6: Write test for full sample pipeline, then implement**

Add to `sampling.rs`:

```rust
/// Sample a token from logits using the full pipeline.
///
/// Order: temperature → repeat penalty → top-k → top-p → weighted random.
/// If `temperature == 0.0`, returns greedy (argmax) immediately.
pub fn sample(
    logits: &[f32],
    params: &crate::SamplingParams,
    context_tokens: &[u32],
    rng: &mut impl Rng,
) -> Result<u32, InferenceError> {
    if logits.is_empty() {
        return Err(InferenceError::SamplingFailed("empty logits".into()));
    }
    if params.temperature == 0.0 {
        return sample_greedy(logits);
    }

    let mut logits = logits.to_vec();

    apply_temperature(&mut logits, params.temperature);
    apply_repeat_penalty(&mut logits, params.repeat_penalty, context_tokens);
    apply_top_k(&mut logits, params.top_k);
    apply_top_p(&mut logits, params.top_p);

    sample_from_distribution(&logits, rng)
}

/// Sample a token from a logit distribution using weighted random selection.
///
/// Applies softmax internally, then draws from the resulting distribution.
fn sample_from_distribution(logits: &[f32], rng: &mut impl Rng) -> Result<u32, InferenceError> {
    let max_logit = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    if max_logit == f32::NEG_INFINITY {
        return Err(InferenceError::SamplingFailed(
            "all logits are -inf after filtering".into(),
        ));
    }

    let exps: Vec<f32> = logits.iter().map(|&l| (l - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    let threshold = rng.gen::<f32>() * sum;
    let mut cumsum = 0.0;
    for (i, &exp) in exps.iter().enumerate() {
        cumsum += exp;
        if cumsum >= threshold {
            return Ok(i as u32);
        }
    }
    // Floating-point edge case — return last valid token
    Ok((logits.len() - 1) as u32)
}
```

Add tests:

```rust
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_sample_temperature_zero_is_greedy() {
        let logits = [1.0_f32, 5.0, 3.0];
        let params = crate::SamplingParams::greedy();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&logits, &params, &[], &mut rng).unwrap();
        assert_eq!(result, 1); // argmax
    }

    #[test]
    fn test_sample_returns_valid_token_id() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0];
        let params = crate::SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&logits, &params, &[], &mut rng).unwrap();
        assert!(result < 4, "token ID {result} out of range");
    }

    #[test]
    fn test_sample_empty_logits_error() {
        let params = crate::SamplingParams::default();
        let mut rng = StdRng::seed_from_u64(42);
        let result = sample(&[], &params, &[], &mut rng);
        assert!(matches!(result, Err(InferenceError::SamplingFailed(_))));
    }

    #[test]
    fn test_sample_deterministic_with_same_seed() {
        let logits = [1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let params = crate::SamplingParams {
            temperature: 0.8,
            top_p: 0.9,
            top_k: 3,
            repeat_penalty: 1.0,
        };
        let r1 = sample(&logits, &params, &[], &mut StdRng::seed_from_u64(123)).unwrap();
        let r2 = sample(&logits, &params, &[], &mut StdRng::seed_from_u64(123)).unwrap();
        assert_eq!(r1, r2, "same seed must produce same result");
    }
```

Run: `cargo test -p harmony-inference sampling::tests`
Expected: All 18 tests PASS

- [ ] **Step 7: Run full test suite and clippy**

Run: `cargo test -p harmony-inference`
Expected: All tests PASS

Run: `cargo fmt -p harmony-inference -- --check`
Expected: No formatting issues

Run: `cargo clippy -p harmony-inference -- -D warnings`
Expected: No warnings

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-inference/src/sampling.rs
git commit -m "feat(inference): implement sampling module with temperature, top-k, top-p, repeat penalty"
```

---

### Task 3: QwenEngine + InferenceEngine Implementation

Wire candle APIs into the `QwenEngine` struct and implement all `InferenceEngine` trait methods. Includes error path unit tests (no real model needed).

**Files:**
- Modify: `crates/harmony-inference/src/engine.rs`

**Key candle APIs used:**
- `candle_core::quantized::gguf_file::Content::read(&mut reader)` — parse GGUF header
- `candle_transformers::models::quantized_qwen3::ModelWeights::from_gguf(content, &mut reader, &device)` — load model
- `ModelWeights::forward(&mut self, input: &Tensor, offset: usize)` — forward pass, returns logits tensor
- `ModelWeights::clear_kv_cache(&mut self)` — reset KV cache
- `tokenizers::Tokenizer::from_bytes(bytes)` — load tokenizer from JSON
- `tokenizer.encode(text, add_special_tokens)` → `encoding.get_ids()` — tokenize
- `tokenizer.decode(ids, skip_special_tokens)` — detokenize

- [ ] **Step 1: Write the full QwenEngine struct with all fields**

Replace the stub in `crates/harmony-inference/src/engine.rs`:

```rust
//! QwenEngine: candle-based inference engine for quantized Qwen3 models.

use std::io::Cursor;

use candle_core::quantized::gguf_file;
use candle_core::{Device, Tensor};
use candle_transformers::models::quantized_qwen3;
use rand::thread_rng;

use crate::error::InferenceError;
use crate::{InferenceEngine, SamplingParams};

/// Inference engine wrapping candle-transformers' quantized Qwen3 implementation.
///
/// # Usage
///
/// ```ignore
/// let mut engine = QwenEngine::new(Device::Cpu);
/// engine.load_gguf(&gguf_bytes)?;
/// engine.load_tokenizer(&tokenizer_json)?;
///
/// let tokens = engine.tokenize("Hello")?;
/// let logits = engine.forward(&tokens)?;
/// let next = engine.sample(&logits, &SamplingParams::greedy())?;
/// ```
pub struct QwenEngine {
    model: Option<quantized_qwen3::ModelWeights>,
    tokenizer: Option<tokenizers::Tokenizer>,
    device: Device,
    /// Current position in the KV cache (advances with each forward call).
    position: usize,
    /// Tokens seen so far in this conversation (for repeat penalty).
    token_history: Vec<u32>,
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
            position: 0,
            token_history: Vec::new(),
        }
    }
}
```

- [ ] **Step 2: Implement load_gguf**

Add the `InferenceEngine` impl block:

```rust
impl InferenceEngine for QwenEngine {
    fn load_gguf(&mut self, gguf_data: &[u8]) -> Result<(), InferenceError> {
        let mut cursor = Cursor::new(gguf_data);
        let content = gguf_file::Content::read(&mut cursor)
            .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        let model =
            quantized_qwen3::ModelWeights::from_gguf(content, &mut cursor, &self.device)
                .map_err(|e| InferenceError::InvalidGguf(e.to_string()))?;
        self.model = Some(model);
        self.position = 0;
        self.token_history.clear();
        Ok(())
    }
```

- [ ] **Step 3: Implement load_tokenizer**

```rust
    fn load_tokenizer(&mut self, tokenizer_json: &[u8]) -> Result<(), InferenceError> {
        let tokenizer = tokenizers::Tokenizer::from_bytes(tokenizer_json)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
        self.tokenizer = Some(tokenizer);
        Ok(())
    }
```

- [ ] **Step 4: Implement tokenize and detokenize**

```rust
    fn tokenize(&self, text: &str) -> Result<Vec<u32>, InferenceError> {
        let tokenizer = self.tokenizer.as_ref().ok_or(InferenceError::TokenizerNotLoaded)?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }

    fn detokenize(&self, tokens: &[u32]) -> Result<String, InferenceError> {
        let tokenizer = self.tokenizer.as_ref().ok_or(InferenceError::TokenizerNotLoaded)?;
        tokenizer
            .decode(tokens, true)
            .map_err(|e| InferenceError::TokenizerError(e.to_string()))
    }
```

- [ ] **Step 5: Implement forward**

```rust
    fn forward(&mut self, tokens: &[u32]) -> Result<Vec<f32>, InferenceError> {
        let model = self.model.as_mut().ok_or(InferenceError::ModelNotLoaded)?;
        let input = Tensor::new(tokens, &self.device)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
        let logits = model
            .forward(&input, self.position)
            .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;

        // Track tokens for repeat penalty
        self.token_history.extend_from_slice(tokens);
        self.position += tokens.len();

        // Candle's quantized_qwen3 projects only the last hidden state through
        // lm_head, so the output is typically [1, vocab_size]. Handle both
        // [vocab_size] and [seq_len, vocab_size] shapes for robustness.
        let logits = match logits.dims().len() {
            1 => logits, // Already [vocab_size]
            2 => {
                // [seq_len, vocab_size] — select the last position
                let seq_len = logits
                    .dim(0)
                    .map_err(|e| InferenceError::ForwardFailed(e.to_string()))?;
                logits
                    .get(seq_len - 1)
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
```

- [ ] **Step 6: Implement sample**

```rust
    fn sample(&self, logits: &[f32], params: &SamplingParams) -> Result<u32, InferenceError> {
        let mut rng = thread_rng();
        crate::sampling::sample(logits, params, &self.token_history, &mut rng)
    }
```

- [ ] **Step 7: Implement reset**

```rust
    fn reset(&mut self) {
        if let Some(model) = &mut self.model {
            model.clear_kv_cache();
        }
        self.position = 0;
        self.token_history.clear();
    }
}
```

- [ ] **Step 8: Write error path unit tests**

Add at the bottom of `engine.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_without_model_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.forward(&[1, 2, 3]);
        assert!(matches!(result, Err(InferenceError::ModelNotLoaded)));
    }

    #[test]
    fn test_tokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.tokenize("hello");
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn test_detokenize_without_tokenizer_returns_error() {
        let engine = QwenEngine::new(Device::Cpu);
        let result = engine.detokenize(&[1, 2, 3]);
        assert!(matches!(result, Err(InferenceError::TokenizerNotLoaded)));
    }

    #[test]
    fn test_invalid_gguf_bytes_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_gguf(b"not a valid gguf file");
        assert!(matches!(result, Err(InferenceError::InvalidGguf(_))));
    }

    #[test]
    fn test_invalid_tokenizer_json_returns_error() {
        let mut engine = QwenEngine::new(Device::Cpu);
        let result = engine.load_tokenizer(b"not valid json");
        assert!(matches!(result, Err(InferenceError::TokenizerError(_))));
    }

    #[test]
    fn test_sample_without_model_still_works() {
        // sample() doesn't need a model — only logits + params
        let engine = QwenEngine::new(Device::Cpu);
        let logits = [1.0_f32, 5.0, 3.0];
        let result = engine.sample(&logits, &SamplingParams::greedy());
        assert_eq!(result.unwrap(), 1); // greedy argmax
    }

    #[test]
    fn test_reset_clears_state() {
        let mut engine = QwenEngine::new(Device::Cpu);
        // Simulate some state (without a real model, just verify fields reset)
        engine.position = 42;
        engine.token_history = vec![1, 2, 3];
        engine.reset();
        assert_eq!(engine.position, 0);
        assert!(engine.token_history.is_empty());
    }
}
```

- [ ] **Step 9: Run full test suite and clippy**

Run: `cargo test -p harmony-inference`
Expected: All tests PASS (sampling + engine tests)

Run: `cargo fmt -p harmony-inference -- --check`
Expected: No formatting issues

Run: `cargo clippy -p harmony-inference -- -D warnings`
Expected: No warnings

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-inference/src/engine.rs
git commit -m "feat(inference): implement QwenEngine with InferenceEngine trait wrapping candle quantized Qwen3"
```

---

### Task 4: Integration Tests

Write `#[ignore]` integration tests that require real GGUF model files. These validate the full pipeline but are not run in CI.

**Files:**
- Create: `crates/harmony-inference/tests/integration.rs`

**Environment variables:**
- `HARMONY_TEST_GGUF` — path to a Qwen3 GGUF file (e.g., `qwen3-0.8b-q4_k_m.gguf`)
- `HARMONY_TEST_TOKENIZER` — path to the matching `tokenizer.json`

**How to run:** `cargo test -p harmony-inference --test integration -- --ignored`

- [ ] **Step 1: Write test helper to load model from env vars**

Create `crates/harmony-inference/tests/integration.rs`:

```rust
//! Integration tests for harmony-inference.
//!
//! These tests require real GGUF model files and are marked `#[ignore]`.
//! To run: `cargo test -p harmony-inference --test integration -- --ignored`
//!
//! Required environment variables:
//! - `HARMONY_TEST_GGUF`: path to a Qwen3 GGUF file (e.g. qwen3-0.8b-q4_k_m.gguf)
//! - `HARMONY_TEST_TOKENIZER`: path to the matching tokenizer.json

use candle_core::Device;
use harmony_inference::{InferenceEngine, QwenEngine, SamplingParams};

fn load_test_engine() -> QwenEngine {
    let gguf_path =
        std::env::var("HARMONY_TEST_GGUF").expect("set HARMONY_TEST_GGUF to a Qwen3 GGUF path");
    let tokenizer_path = std::env::var("HARMONY_TEST_TOKENIZER")
        .expect("set HARMONY_TEST_TOKENIZER to a tokenizer.json path");

    let gguf_bytes = std::fs::read(&gguf_path)
        .unwrap_or_else(|e| panic!("failed to read {gguf_path}: {e}"));
    let tokenizer_bytes = std::fs::read(&tokenizer_path)
        .unwrap_or_else(|e| panic!("failed to read {tokenizer_path}: {e}"));

    let mut engine = QwenEngine::new(Device::Cpu);
    engine
        .load_gguf(&gguf_bytes)
        .unwrap_or_else(|e| panic!("failed to load GGUF: {e}"));
    engine
        .load_tokenizer(&tokenizer_bytes)
        .unwrap_or_else(|e| panic!("failed to load tokenizer: {e}"));
    engine
}
```

- [ ] **Step 2: Write test for GGUF loading**

```rust
#[test]
#[ignore]
fn test_load_gguf_model() {
    let _engine = load_test_engine();
    // If we get here without panicking, the model loaded successfully
}
```

- [ ] **Step 3: Write test for tokenization roundtrip**

```rust
#[test]
#[ignore]
fn test_tokenize_and_detokenize() {
    let engine = load_test_engine();
    let text = "What is Harmony?";
    let tokens = engine.tokenize(text).expect("tokenize failed");
    assert!(!tokens.is_empty(), "tokenization produced no tokens");
    assert!(
        tokens.len() < 100,
        "unexpectedly many tokens: {}",
        tokens.len()
    );

    let decoded = engine.detokenize(&tokens).expect("detokenize failed");
    assert!(
        decoded.contains("Harmony"),
        "roundtrip lost content: {decoded:?}"
    );
}
```

- [ ] **Step 4: Write test for forward pass**

```rust
#[test]
#[ignore]
fn test_forward_pass_returns_logits() {
    let mut engine = load_test_engine();
    let tokens = engine.tokenize("Hello").expect("tokenize failed");
    let logits = engine.forward(&tokens).expect("forward failed");

    // Logits should have vocab_size entries (Qwen3 vocab is ~151k)
    assert!(
        logits.len() > 1000,
        "logits too short: {} (expected vocab_size)",
        logits.len()
    );
    // All values should be finite
    assert!(
        logits.iter().all(|l| l.is_finite()),
        "logits contain non-finite values"
    );
}
```

- [ ] **Step 5: Write test for sampling**

```rust
#[test]
#[ignore]
fn test_sample_produces_valid_token() {
    let mut engine = load_test_engine();
    let tokens = engine.tokenize("The capital of France is").expect("tokenize");
    let logits = engine.forward(&tokens).expect("forward");

    let token = engine
        .sample(&logits, &SamplingParams::greedy())
        .expect("sample");
    assert!(
        (token as usize) < logits.len(),
        "sampled token {token} >= vocab size {}",
        logits.len()
    );

    let text = engine.detokenize(&[token]).expect("detokenize");
    assert!(!text.is_empty(), "sampled token decoded to empty string");
}
```

- [ ] **Step 6: Write test for multi-token generation**

```rust
#[test]
#[ignore]
fn test_generate_ten_tokens() {
    let mut engine = load_test_engine();
    let prompt_tokens = engine.tokenize("Once upon a time").expect("tokenize");
    let logits = engine.forward(&prompt_tokens).expect("prefill");
    let mut next = engine.sample(&logits, &SamplingParams::greedy()).expect("sample");

    let mut generated = vec![next];
    for _ in 0..9 {
        let logits = engine.forward(&[next]).expect("decode step");
        next = engine.sample(&logits, &SamplingParams::greedy()).expect("sample");
        generated.push(next);
    }

    let text = engine.detokenize(&generated).expect("detokenize");
    assert!(!text.is_empty(), "generated text is empty");
    eprintln!("Generated: {text}");
}
```

- [ ] **Step 7: Write test for reset**

```rust
#[test]
#[ignore]
fn test_reset_allows_new_conversation() {
    let mut engine = load_test_engine();

    // First conversation
    let tokens = engine.tokenize("Hello").expect("tokenize");
    let _ = engine.forward(&tokens).expect("forward");

    // Reset
    engine.reset();

    // Second conversation should work from position 0
    let tokens = engine.tokenize("Goodbye").expect("tokenize");
    let logits = engine.forward(&tokens).expect("forward after reset");
    assert!(!logits.is_empty());
}
```

- [ ] **Step 8: Verify compilation**

Run: `cargo check -p harmony-inference --tests`
Expected: Compiles (integration tests won't run without model files)

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-inference/tests/
git commit -m "test(inference): add integration tests for GGUF model loading and generation"
```

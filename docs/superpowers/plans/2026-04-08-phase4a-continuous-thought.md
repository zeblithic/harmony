# Phase 4a: Continuous Thought Infrastructure — Implementation Plan

**Spec:** `docs/superpowers/specs/2026-04-08-phase4a-continuous-thought-design.md`
**Date:** 2026-04-08

## Overview

Add pause token (`<think>`) infrastructure and UQ-driven routing to the ct87
model, establishing the control flow for Phase 4b (full COCONUT continuous thought).

**This can be built now** — the Rust/candle types and routing logic don't require
a trained model. Training data preparation and UQ threshold tuning happen later
(Phase 3+), but the infrastructure should be in place early so training can target it.

## Task Breakdown

### Task 1: Add `ContinuousThoughtConfig` and `ThoughtAction` to harmony-inference

**Files:**
- NEW: `crates/harmony-inference/src/continuous_thought.rs`
- EDIT: `crates/harmony-inference/src/lib.rs` (add module, re-exports)

**Changes:**

Create `continuous_thought.rs` with:

```rust
/// Configuration for continuous thought routing.
#[derive(Debug, Clone, PartialEq)]
pub struct ContinuousThoughtConfig {
    pub think_token_id: Option<u32>,
    pub max_think_steps: u32,
    pub confidence_threshold: f32,
}

impl Default for ContinuousThoughtConfig { ... }

/// Routing decision for the generation loop.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThoughtAction {
    Emit,
    Think,
    Abort,
}
```

Add `pub mod continuous_thought;` to `lib.rs` and re-export types.

**Tests:**
- Default config values are correct
- `ThoughtAction` variants are distinct

### Task 2: Add `think_token_id` to `HarmonyModelConfig`

**Files:**
- EDIT: `crates/harmony-inference/src/harmony_model.rs`

**Changes:**

Add `pub think_token_id: Option<u32>` to `HarmonyModelConfig`.

Update `target()` and `tiny()` constructors: `think_token_id: None`.

Update `from_gguf()`: read optional metadata key
`harmony.continuous_thought.think_token_id`. If absent, set to `None`.
If present, validate `< vocab_size`.

**Tests:**
- `target()` has `think_token_id: None`
- `tiny()` has `think_token_id: None`
- GGUF roundtrip with and without think_token_id

### Task 3: Add `route_thought()` to `HarmonyEngine`

**Files:**
- EDIT: `crates/harmony-inference/src/harmony_engine.rs`

**Changes:**

Add field `thought_config: ContinuousThoughtConfig` to `HarmonyEngine`.

Add constructor methods:
- `set_thought_config(&mut self, config: ContinuousThoughtConfig)`
- `think_token_id(&self) -> Option<u32>`

Add routing method:

```rust
pub fn route_thought(
    &self,
    output: &HarmonyForwardOutput,
    consecutive_thinks: u32,
) -> Result<ThoughtAction, InferenceError> {
    // 1. If no UQ head or think_token_id is None → Emit
    // 2. classify_uncertainty(output)
    // 3. Route based on class + confidence + consecutive_thinks cap
}
```

Load `ContinuousThoughtConfig` from GGUF metadata in `load_gguf()`.

**Tests:**
- Returns `Emit` when no UQ head set
- Returns `Emit` when `think_token_id` is `None`
- Returns `Think` for `Uncertain` class under max steps
- Returns `Think` for `HighVolume` class under max steps
- Returns `Emit` at max consecutive think steps (safety cap)
- Returns `Abort` for `SpectralCollapse`
- Returns `Emit` for `Confident` above threshold
- Returns `Emit` for `Confident` below threshold (conservative: still emit)
- Consecutive thinks counter resets between calls correctly

### Task 4: Add continuous thought GGUF metadata to PyTorch export

**Files:**
- EDIT: `training/ct87/model.py` — add `think_token_id` to `HarmonyModelConfig`
- EDIT: `training/ct87/export_gguf.py` — write continuous thought metadata
- EDIT: `training/tests/test_model.py` — test config change
- EDIT: `training/tests/test_export_gguf.py` — test metadata export

**Changes in `model.py`:**

Add `think_token_id: int | None = None` to `HarmonyModelConfig`.

**Changes in `export_gguf.py`:**

In `write_metadata()`, conditionally write:
```python
if config.think_token_id is not None:
    writer.add_bool("harmony.continuous_thought.enabled", True)
    writer.add_uint32("harmony.continuous_thought.think_token_id",
                      config.think_token_id)
    writer.add_uint32("harmony.continuous_thought.max_steps", 4)
    writer.add_float32("harmony.continuous_thought.confidence_threshold", 0.85)
```

**Tests:**
- Config with `think_token_id=None` produces no continuous thought metadata
- Config with `think_token_id=32000` produces all 4 metadata keys
- GGUF roundtrip preserves continuous thought metadata

### Task 5: Candle GGUF loader reads continuous thought metadata

**Files:**
- EDIT: `crates/harmony-inference/src/harmony_model.rs` (`from_gguf`)
- EDIT: `crates/harmony-inference/src/harmony_engine.rs` (`load_gguf`)

**Changes:**

In `HarmonyModel::from_gguf()`: read `harmony.continuous_thought.think_token_id`
as optional u32. Set in config.

In `HarmonyEngine::load_gguf()`: after loading model, read continuous thought
metadata and populate `self.thought_config`:
```rust
let enabled = ct.metadata.get("harmony.continuous_thought.enabled")
    .and_then(|v| v.to_bool().ok())
    .unwrap_or(false);

if enabled {
    let think_id = /* read think_token_id */;
    let max_steps = /* read max_steps, default 4 */;
    let threshold = /* read confidence_threshold, default 0.85 */;
    self.thought_config = ContinuousThoughtConfig {
        think_token_id: Some(think_id),
        max_think_steps: max_steps,
        confidence_threshold: threshold,
    };
}
```

**Tests:**
- GGUF without continuous thought keys → config defaults to disabled
- GGUF with continuous thought keys → config populated correctly
- `think_token_id >= vocab_size` → error on load

### Task 6: Integration test — generation loop with forced thinking

**Files:**
- NEW or EDIT: `crates/harmony-inference/tests/` (integration test)

**Changes:**

Write an integration test that:
1. Creates a `HarmonyEngine` with random weights and a UQ Head biased toward
   `Uncertain` (b2 biased toward class 3)
2. Sets `think_token_id = Some(32000)`
3. Runs a generation loop that calls `route_thought()` on each step
4. Verifies that `Think` actions produce the correct token in history
5. Verifies the safety cap triggers after `max_think_steps`
6. Verifies `<think>` tokens are filterable from the output

**Tests:**
- UQ-biased engine produces Think actions
- Safety cap forces Emit after max_think_steps
- Output stream filters `<think>` tokens correctly

## Dependency Graph

```
Task 1 (types)
   |
   +--> Task 2 (config) --+--> Task 5 (GGUF loader)
   |                       |
   +--> Task 3 (routing) --+--> Task 6 (integration test)
   |
   +--> Task 4 (PyTorch export)
```

Tasks 1-4 can proceed in parallel after Task 1. Task 5 depends on Tasks 2+4.
Task 6 depends on Tasks 3+5.

## Test Strategy

**Unit tests (per task):** Each task includes specific unit tests as described.

**Integration test (Task 6):** End-to-end generation loop verifying the routing
logic with a biased UQ Head.

**Validation test:** After Phase 3 training, run the trained model with continuous
thought enabled and verify:
- Thinking triggers on hard tokens (low confidence)
- Thinking does not trigger on easy tokens (high confidence)
- Output quality improves with thinking enabled vs disabled

**Commands:**
```bash
cd crates/harmony-inference && cargo test
cd training && python -m pytest tests/
```

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| `<think>` token pollutes training data | Use progressive curriculum, start late in training |
| UQ Head not accurate enough for routing | Tune thresholds; default to Emit on ambiguity |
| KV cache bloat from think tokens | Cap at 4 thinks max; Phase 4b introduces ephemeral cache |
| Latency increase from extra forward passes | Adaptive: only think when uncertain (<10% of tokens expected) |
| Infinite think loop | Hard cap via `max_think_steps` |

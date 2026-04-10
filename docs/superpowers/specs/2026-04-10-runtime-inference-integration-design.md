# ZEB-64: Runtime Inference Integration — Wire Speculative Decode + UQ into Event Loop

## Overview

Wire the recently completed sans-I/O inference modules into harmony-node's event loop for end-to-end usability. The core change is swapping `QwenEngine` for `HarmonyEngine` in the runtime, enabling UQ-gated speculative decoding during autoregressive inference.

## Engine Swap: QwenEngine → HarmonyEngine

`HarmonyEngine` wraps `QwenEngine` + UQ head + continuous thought routing. It implements the same `InferenceEngine` trait (`forward()`, `sample()`, `tokenize()`, `new_cache()`, etc.), so all existing call sites compile unchanged.

The swap enables `forward_full()` (returns `HarmonyForwardOutput` with layer norms) and `classify_uncertainty()` (runs the UQ head), which feed the speculative decode scheduler.

### harmony-runtime/src/runtime.rs

- `verification_engine: Option<QwenEngine>` → `Option<HarmonyEngine>`
- `check_inference_model_ready()`: create `HarmonyEngine::new(HarmonyModelConfig::tiny(), Device::Cpu)`, then `load_gguf()` (overwrites config from GGUF metadata) + `load_tokenizer()`
- Type signature changes on: `inference_engine_ref()`, `take_inference_engine()`, `return_inference_engine()`, `run_verification()`
- No functional changes to DSD verification — it only uses `InferenceEngine` trait methods

### harmony-node/src/event_loop.rs

- `InferenceResult` variants: `engine: QwenEngine` → `engine: HarmonyEngine`
- `run_inference_loop()` parameter: `engine: QwenEngine` → `engine: HarmonyEngine`
- Existing `engine.forward()` / `engine.forward_with_engram()` calls work unchanged

## Speculative Decode Wiring in run_inference_loop()

### Current decode loop

```
prefill → sample → [forward → sample → stream chunk]* → complete
```

### New decode loop

```
prefill → sample → [forward_full → classify → scheduler_track → stream chunk]* → complete
```

The Engram integration path changes from `engine.forward()` / `engine.forward_with_engram()` to their `forward_full` equivalents. Since `HarmonyEngine::forward()` already delegates to `forward_full()` internally, the non-Engram path just stores the `HarmonyForwardOutput` before extracting logits. The Engram path uses `engine.forward_full(tokens, cache, Some(engram))` directly (the method already accepts optional Engram context).

### Detailed flow

1. After sampling `next_token`, call `engine.forward_full(&[next_token], &mut cache, engram)` instead of `engine.forward()` — returns `HarmonyForwardOutput` with layer norms
2. Call `engine.classify_uncertainty(&output)` → `Option<(UqClass, f32)>`
3. If UQ head is loaded, scheduler is present, and classify returns confidence:
   - Build `SpecContext { is_thinking: false, engram_steps_remaining: None }` (placeholders — COCONUT and ChunkedEngramScheduler not wired yet)
   - Call `scheduler.begin_draft(confidence, &context)` — returns true/false
   - Call `scheduler.complete(0)` to record the decision in metrics (no actual drafting in v1)
4. Normal single-token decode continues unchanged

The scheduler tracks **what would have happened** based on UQ confidence:
- How often `begin_draft` would have returned true (speculation readiness)
- Confidence distribution across decode steps
- Bypass eligibility (tokens above `tau_accept`)

This telemetry is valuable for calibrating thresholds before MTP or batch verification is available.

### Why no draft→verify loop in v1

Self-speculation (same model as both draft and target) provides **no throughput benefit** without either:
- **MTP head**: predict K tokens in one forward pass (1 forward instead of K)
- **Batch verification kernels**: verify K draft tokens in one forward pass (needs per-position logits + custom attention masks)

Without these, drafting K tokens sequentially through the same model is identical cost to normal decode. The scheduler framework is wired and ready — the draft loop activates when a draft model becomes available.

### Fallback

If no UQ head is loaded, `classify_uncertainty()` returns `None` and the entire classification + scheduling path is skipped. The loop behaves identically to today. UQ head presence is the feature gate; scheduler presence is the second gate.

### Not wired (explicit placeholders)

- Draft→verify loop: needs MTP head or lightweight draft model (self-speculation has no throughput benefit)
- COCONUT coordination: `is_thinking` is always `false`
- Engram steps tracking: `engram_steps_remaining` is always `None`
- PagedKvCache: needs attention kernel changes to replace `InferenceCache`
- DraftTree verification masks: needs model to return per-position logits

## Config Surface

### harmony-node/src/config.rs

Add an optional `[speculation]` section to `ConfigFile`:

```toml
[speculation]
tau_generate = 0.3
tau_chain = 0.2
tau_accept = 0.95
max_draft_len = 8
```

All fields optional with defaults from `SpecDecConfig::default()`. Omitting `[speculation]` entirely disables speculation (no scheduler created, zero overhead).

New struct:

```rust
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct SpeculationConfig {
    pub tau_generate: Option<f32>,
    pub tau_chain: Option<f32>,
    pub tau_accept: Option<f32>,
    pub max_draft_len: Option<usize>,
}
```

`ConfigFile` gains `pub speculation: Option<SpeculationConfig>`.

### Wiring

- Event loop: if `config.speculation.is_some()`, create `SpeculativeDecodeScheduler` with merged config values and pass into `run_inference_loop()`
- `run_inference_loop()` takes `spec_scheduler: Option<SpeculativeDecodeScheduler>` — `None` skips all speculation logic

### UQ head

No config needed. The UQ head is loaded from GGUF metadata automatically by `HarmonyEngine::load_gguf()`. If GGUF has UQ weights, they're available. If not, `classify_uncertainty()` returns `None` and speculation is gated off regardless of config.

## Metrics

After each `scheduler.complete()` or `scheduler.cancel()`, emit `tracing::debug!` with running metrics:

- `acceptance_rate` (f64)
- `avg_draft_length` (f64)
- `verification_bypass_rate` (f64)
- `total_speculations`, `skipped_speculations` (u64)

No new telemetry infrastructure — structured tracing fields that existing subscribers capture.

## File Changes

| File | Change |
|------|--------|
| `harmony-runtime/src/runtime.rs` | Swap `QwenEngine` → `HarmonyEngine` in field + 5 method signatures. Update engine construction. |
| `harmony-node/src/event_loop.rs` | Swap engine type in `InferenceResult` + `run_inference_loop()`. Add spec decode logic. Pass scheduler. |
| `harmony-node/src/config.rs` | Add `SpeculationConfig` struct + `speculation` field. |

Not changed: `harmony-inference/` (already complete), `harmony-node/src/inference.rs` (payload parsing), `chunked_engram.rs` (`steps_until_boundary()` already exists).

## Tests

- Config: `speculation_config_parses`, `speculation_config_defaults_to_none`
- Runtime: existing tests recompile with `HarmonyEngine` type change
- Compilation: `cargo check -p harmony-node`

## Verification

```bash
cargo test -p harmony-inference -- --nocapture
cargo test -p harmony-runtime -- --nocapture
cargo test -p harmony-node -- --nocapture
cargo check -p harmony-node
```

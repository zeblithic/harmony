# ZEB-64: Runtime Inference Integration

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Swap QwenEngine for HarmonyEngine in the runtime and wire UQ classification + speculative decode scheduler into the inference decode loop.

**Architecture:** Replace `QwenEngine` with `HarmonyEngine` in `harmony-runtime` (same `InferenceEngine` trait, adds `forward_full()` + `classify_uncertainty()`). Wire `SpeculativeDecodeScheduler` into the decode loop for UQ-gated speculation telemetry. No draft→verify loop in v1 (needs MTP). Config surface for speculation thresholds.

**Tech Stack:** Rust, candle, harmony-inference, harmony-runtime, harmony-node, serde/toml

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-inference/src/harmony_model.rs` | Add `logits_vec()` helper on `HarmonyForwardOutput` |
| `crates/harmony-node/src/config.rs` | Add `SpeculationConfig` struct + field |
| `crates/harmony-runtime/src/runtime.rs` | Swap engine type QwenEngine → HarmonyEngine |
| `crates/harmony-node/src/event_loop.rs` | Swap engine type + add UQ classification + scheduler wiring |

---

### Task 1: Add `logits_vec()` to HarmonyForwardOutput

The event loop needs to extract `Vec<f32>` logits from `HarmonyForwardOutput` for sampling. The existing `logits_to_vec` is `pub(crate)`, so add a public method.

**Files:**
- Modify: `crates/harmony-inference/src/harmony_model.rs:126-131`

- [ ] **Step 1: Add the method and test**

In `crates/harmony-inference/src/harmony_model.rs`, after the `HarmonyForwardOutput` struct definition (line 131), add:

```rust
impl HarmonyForwardOutput {
    /// Extract logits as a flat `Vec<f32>` for sampling.
    ///
    /// Handles both 1D `[vocab_size]` and 2D `[batch, vocab_size]` tensors
    /// (taking the last row for batched output).
    pub fn logits_vec(&self) -> Result<Vec<f32>, crate::error::InferenceError> {
        crate::logits_to_vec(&self.logits)
    }
}
```

Then in the `#[cfg(test)]` module at the bottom of the same file, add:

```rust
#[test]
fn forward_output_logits_vec() {
    let logits = Tensor::new(&[0.1f32, 0.2, 0.3], &Device::Cpu).unwrap();
    let output = HarmonyForwardOutput {
        logits,
        layer_norms: vec![1.0],
    };
    let v = output.logits_vec().unwrap();
    assert_eq!(v.len(), 3);
    assert!((v[0] - 0.1).abs() < 1e-6);
}
```

- [ ] **Step 2: Run test**

```bash
cargo test -p harmony-inference forward_output_logits_vec -- --nocapture
```

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-inference/src/harmony_model.rs
git commit -m "feat(inference): add logits_vec() to HarmonyForwardOutput"
```

---

### Task 2: Add SpeculationConfig to node config

**Files:**
- Modify: `crates/harmony-node/src/config.rs`

- [ ] **Step 1: Add the struct and field**

In `crates/harmony-node/src/config.rs`, after the `ArchivistConfig` struct (around line 118), add:

```rust
/// Speculation thresholds for UQ-gated speculative decoding.
/// All fields are optional — defaults come from `SpecDecConfig::default()`.
#[derive(Deserialize, Debug)]
#[serde(deny_unknown_fields)]
pub struct SpeculationConfig {
    /// Confidence below this skips speculation entirely.
    pub tau_generate: Option<f32>,
    /// Cumulative product below this stops drafting.
    pub tau_chain: Option<f32>,
    /// Confidence above this bypasses verification (ConfSpec).
    pub tau_accept: Option<f32>,
    /// Maximum draft length.
    pub max_draft_len: Option<usize>,
}
```

Then add the field to `ConfigFile` (after the `archivist` field, around line 96):

```rust
    /// Speculation config for UQ-gated speculative decoding.
    pub speculation: Option<SpeculationConfig>,
```

- [ ] **Step 2: Add tests**

In the `#[cfg(test)] mod tests` block, add:

```rust
    #[test]
    fn speculation_config_parses() {
        let toml = r#"
[speculation]
tau_generate = 0.4
tau_chain = 0.25
tau_accept = 0.9
max_draft_len = 6
"#;
        let (_f, path) = write_temp(toml);
        let cfg = load(&path, false).expect("should parse speculation config");
        let spec = cfg.speculation.expect("speculation section present");
        assert!((spec.tau_generate.unwrap() - 0.4).abs() < 1e-6);
        assert!((spec.tau_chain.unwrap() - 0.25).abs() < 1e-6);
        assert!((spec.tau_accept.unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(spec.max_draft_len.unwrap(), 6);
    }

    #[test]
    fn speculation_config_defaults_to_none() {
        let toml = r#"listen_address = "127.0.0.1:4242""#;
        let (_f, path) = write_temp(toml);
        let cfg = load(&path, false).expect("should parse without speculation");
        assert!(cfg.speculation.is_none());
    }

    #[test]
    fn speculation_config_partial_fields() {
        let toml = r#"
[speculation]
tau_generate = 0.5
"#;
        let (_f, path) = write_temp(toml);
        let cfg = load(&path, false).expect("should parse partial speculation");
        let spec = cfg.speculation.expect("speculation section present");
        assert!((spec.tau_generate.unwrap() - 0.5).abs() < 1e-6);
        assert!(spec.tau_chain.is_none());
        assert!(spec.tau_accept.is_none());
        assert!(spec.max_draft_len.is_none());
    }
```

- [ ] **Step 3: Run tests**

```bash
cargo test -p harmony-node speculation_config -- --nocapture
```

Expected: all 3 PASS

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/config.rs
git commit -m "feat(node): add SpeculationConfig to node config"
```

---

### Task 3: Engine swap in harmony-runtime

Replace `QwenEngine` with `HarmonyEngine` in the runtime struct, field initializer, accessor methods, and DSD verification. All callsites use `InferenceEngine` trait methods, so this is a type-level change with no behavioral difference.

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs`

- [ ] **Step 1: Change the struct field**

At line 785-788, change:

```rust
    /// QwenEngine for native verification (DSD target side).
    /// Separate from the WasmiRuntime's persisted engine used by WASM workflows.
    #[cfg(feature = "inference")]
    verification_engine: Option<harmony_inference::QwenEngine>,
```

to:

```rust
    /// HarmonyEngine for native inference, UQ classification, and DSD verification.
    /// Wraps QwenEngine + UQ head. Separate from the WasmiRuntime's persisted engine.
    #[cfg(feature = "inference")]
    verification_engine: Option<harmony_inference::HarmonyEngine>,
```

- [ ] **Step 2: Change the engine construction in `check_inference_model_ready()`**

At line 3100, change:

```rust
                    let mut engine = harmony_inference::QwenEngine::new(candle_core::Device::Cpu);
```

to:

```rust
                    let mut engine = harmony_inference::HarmonyEngine::new(
                        harmony_inference::HarmonyModelConfig::tiny(),
                        candle_core::Device::Cpu,
                    );
```

`load_gguf()` overwrites the config from GGUF metadata, so the initial `tiny()` config is a placeholder.

- [ ] **Step 3: Change accessor method signatures**

At line 3031, change:

```rust
    pub fn inference_engine_ref(&self) -> Option<&harmony_inference::QwenEngine> {
```

to:

```rust
    pub fn inference_engine_ref(&self) -> Option<&harmony_inference::HarmonyEngine> {
```

At line 3056, change:

```rust
    pub fn take_inference_engine(&mut self) -> Option<harmony_inference::QwenEngine> {
```

to:

```rust
    pub fn take_inference_engine(&mut self) -> Option<harmony_inference::HarmonyEngine> {
```

At line 3066, change:

```rust
    pub fn return_inference_engine(&mut self, engine: harmony_inference::QwenEngine) {
```

to:

```rust
    pub fn return_inference_engine(&mut self, engine: harmony_inference::HarmonyEngine) {
```

- [ ] **Step 4: Change `run_verification()` parameter**

At line 3283-3284, change:

```rust
    fn run_verification(
        engine: &harmony_inference::QwenEngine,
```

to:

```rust
    fn run_verification(
        engine: &harmony_inference::HarmonyEngine,
```

The body only uses `InferenceEngine` trait methods (`new_cache`, `forward`, `sample`) which `HarmonyEngine` implements identically.

- [ ] **Step 5: Verify compilation**

```bash
cargo test -p harmony-runtime -- --nocapture 2>&1 | tail -20
```

Expected: existing tests pass. The type change is transparent because `HarmonyEngine` implements `InferenceEngine`.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/src/runtime.rs
git commit -m "refactor(runtime): swap QwenEngine for HarmonyEngine

HarmonyEngine wraps QwenEngine + UQ head + continuous thought routing.
All InferenceEngine trait methods are unchanged. Enables forward_full()
and classify_uncertainty() for speculative decode integration."
```

---

### Task 4: Engine swap in event_loop.rs

Change the engine type in `InferenceResult` enum and `run_inference_loop()` function signature. No behavioral changes.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

- [ ] **Step 1: Change `InferenceResult` enum**

At line 81, change:

```rust
        engine: harmony_inference::QwenEngine,
```

to:

```rust
        engine: harmony_inference::HarmonyEngine,
```

At line 88, change:

```rust
        engine: harmony_inference::QwenEngine,
```

to:

```rust
        engine: harmony_inference::HarmonyEngine,
```

- [ ] **Step 2: Change `run_inference_loop()` parameter**

At line 2534, change:

```rust
    engine: harmony_inference::QwenEngine,
```

to:

```rust
    engine: harmony_inference::HarmonyEngine,
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check -p harmony-node 2>&1 | tail -20
```

Expected: compiles clean. All `engine.forward()`, `engine.sample()`, `engine.forward_with_engram()`, `engine.tokenize()`, `engine.detokenize()`, `engine.new_cache()`, `engine.eos_token_id()`, `engine.device()` calls exist on `HarmonyEngine`.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "refactor(node): swap QwenEngine for HarmonyEngine in event loop"
```

---

### Task 5: Wire UQ classification + scheduler into decode loop

This is the core new logic. After the non-Engram forward in the decode loop, use `forward_full()` to get `HarmonyForwardOutput`, classify uncertainty, and feed the scheduler for metrics tracking.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

- [ ] **Step 1: Add scheduler parameter to `run_inference_loop()`**

At line 2533 (the function signature), add `spec_scheduler` parameter. Change from:

```rust
fn run_inference_loop(
    engine: harmony_inference::HarmonyEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    tokens: Vec<u32>,
    is_token_mode: bool,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
    engram: Option<EngramPrefill>,
) {
```

to:

```rust
fn run_inference_loop(
    engine: harmony_inference::HarmonyEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    tokens: Vec<u32>,
    is_token_mode: bool,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
    engram: Option<EngramPrefill>,
    mut spec_scheduler: Option<harmony_inference::SpeculativeDecodeScheduler>,
) {
```

- [ ] **Step 2: Switch the non-Engram decode forward to `forward_full()`**

In the decode loop's `'decode_fwd:` block, the plain (non-Engram) forward path is at the end (around line 2755). Currently:

```rust
            match engine.forward(&[next_token], &mut cache) {
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
            }
```

Change the `'decode_fwd:` block's return type from `Vec<f32>` to `(Vec<f32>, Option<harmony_inference::HarmonyForwardOutput>)`. Wrap the entire block to return tuples:

For the **Engram success paths** (every `break 'decode_fwd l` inside the Engram block), change each to:
```rust
break 'decode_fwd (l, None)
```

For all **plain forward** paths — both the Engram fallback-to-plain paths (inside the Engram `if` block, when Engram resolve/forward fails) and the **final non-Engram path** at the bottom — change each `engine.forward(...)` to `engine.forward_full(...)` and return the output tuple. There are 3 such paths in the block. Each should become:
```rust
            match engine.forward_full(&[next_token], &mut cache, None) {
                Ok(output) => {
                    match output.logits_vec() {
                        Ok(l) => (l, Some(output)),
                        Err(e) => {
                            let _ = tx.blocking_send(InferenceResult::Failed {
                                query_id,
                                task_id,
                                error: format!("logits extraction failed: {e}"),
                                engine,
                            });
                            return;
                        }
                    }
                }
                Err(e) => {
                    let _ = tx.blocking_send(InferenceResult::Failed {
                        query_id,
                        task_id,
                        error: format!("forward failed: {e}"),
                        engine,
                    });
                    return;
                }
            }
```

Update the assignment at the top of the block from `logits = 'decode_fwd: {` to:
```rust
        let (new_logits, decode_output) = 'decode_fwd: {
```

And after the block, assign logits:
```rust
        logits = new_logits;
```

- [ ] **Step 3: Add UQ classification + scheduler tracking after the decode block**

After `logits = new_logits;`, add the UQ classification and scheduler tracking:

```rust
        // UQ classification + speculation tracking (non-Engram path only).
        if let (Some(ref mut scheduler), Some(ref output)) = (&mut spec_scheduler, &decode_output) {
            if let Ok(Some((_class, confidence))) = engine.classify_uncertainty(output) {
                let context = harmony_inference::SpecContext {
                    is_thinking: false,
                    engram_steps_remaining: None,
                };
                if scheduler.begin_draft(confidence, &context) {
                    // Draft loop deferred until MTP or lightweight draft model is available.
                    // Cancel to record the would-have-speculated metric and return to Idle.
                    scheduler.cancel();
                }
                // If begin_draft returned false, scheduler recorded a skip internally.
            }
        }
```

- [ ] **Step 4: Add metrics logging before the Complete result**

Just before the final `tx.blocking_send(InferenceResult::Complete { ... })` at the end of `run_inference_loop()` (around line 2775), add:

```rust
    // Log speculation metrics at end of generation.
    if let Some(ref scheduler) = spec_scheduler {
        let m = scheduler.metrics();
        tracing::debug!(
            total_speculations = m.total_speculations,
            skipped = m.skipped_speculations,
            canceled = m.canceled_speculations,
            acceptance_rate = %format!("{:.2}", m.acceptance_rate()),
            avg_draft_length = %format!("{:.1}", m.avg_draft_length()),
            bypass_rate = %format!("{:.2}", m.verification_bypass_rate()),
            "speculation metrics"
        );
    }
```

- [ ] **Step 5: Verify compilation**

```bash
cargo check -p harmony-node 2>&1 | tail -20
```

Expected: compile error — `run_inference_loop` callers don't pass the new `spec_scheduler` parameter yet. This is expected; Task 6 fixes it.

- [ ] **Step 6: Commit (WIP — callers updated in Task 6)**

Do NOT commit yet — proceed to Task 6.

---

### Task 6: Wire scheduler creation + passing from event loop setup

Create the scheduler from config and pass it through to `run_inference_loop()` at both callsites (Engram branch and non-Engram branch).

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

- [ ] **Step 1: Create the scheduler from config in the event loop setup**

Near the top of `run_event_loop()`, after the inference channel creation (around line 638), add:

```rust
    // Create speculation scheduler from config (if enabled).
    #[cfg(feature = "inference")]
    let spec_dec_config: Option<harmony_inference::SpecDecConfig> = config
        .speculation
        .as_ref()
        .map(|spec| {
            harmony_inference::SpecDecConfig::new(
                spec.tau_generate.unwrap_or(0.3),
                spec.tau_chain.unwrap_or(0.2),
                spec.tau_accept.unwrap_or(0.95),
                spec.max_draft_len.unwrap_or(8),
            )
        });
```

Note: `config` is the `ConfigFile` parameter to `run_event_loop()`. Verify the parameter name by checking the function signature.

- [ ] **Step 2: Pass scheduler to `run_inference_loop()` at both callsites**

There are two calls to `run_inference_loop()` in the event loop:

**Callsite 1 — Engram branch** (around line 1552):

```rust
                                    run_inference_loop(
                                        engine,
                                        tx,
                                        query_id,
                                        task_id,
                                        tokens,
                                        is_token_mode,
                                        params,
                                        max_tokens,
                                        engram_prefill,
                                    );
```

Add the new parameter:

```rust
                                    run_inference_loop(
                                        engine,
                                        tx,
                                        query_id,
                                        task_id,
                                        tokens,
                                        is_token_mode,
                                        params,
                                        max_tokens,
                                        engram_prefill,
                                        spec_dec_config.as_ref().map(|c| {
                                            harmony_inference::SpeculativeDecodeScheduler::new(c.clone())
                                        }),
                                    );
```

**Callsite 2 — Non-Engram branch** (around line 1601):

```rust
                                run_inference_loop(
                                    engine,
                                    tx,
                                    query_id,
                                    task_id,
                                    tokens,
                                    is_token_mode,
                                    params,
                                    max_tokens,
                                    None,
                                );
```

Add the new parameter:

```rust
                                run_inference_loop(
                                    engine,
                                    tx,
                                    query_id,
                                    task_id,
                                    tokens,
                                    is_token_mode,
                                    params,
                                    max_tokens,
                                    None,
                                    spec_dec_config.as_ref().map(|c| {
                                        harmony_inference::SpeculativeDecodeScheduler::new(c.clone())
                                    }),
                                );
```

- [ ] **Step 3: Verify compilation**

```bash
cargo check -p harmony-node 2>&1 | tail -20
```

Expected: compiles clean.

- [ ] **Step 4: Commit Tasks 5 + 6 together**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): wire UQ classification + speculation scheduler into decode loop

After each non-Engram decode step, calls forward_full() to get layer norms,
runs classify_uncertainty() for UQ confidence, and feeds the speculation
scheduler for metrics tracking. Draft loop deferred until MTP is available.

Scheduler is created from [speculation] config section. When absent,
no scheduler is created and the decode loop behaves identically to before."
```

---

### Task 7: Full verification

- [ ] **Step 1: Run harmony-inference tests**

```bash
cargo test -p harmony-inference -- --nocapture 2>&1 | tail -5
```

Expected: all tests pass (backward compat + new `forward_output_logits_vec` test).

- [ ] **Step 2: Run harmony-runtime tests**

```bash
cargo test -p harmony-runtime -- --nocapture 2>&1 | tail -5
```

Expected: all tests pass (engine type swap is transparent via `InferenceEngine` trait).

- [ ] **Step 3: Run harmony-node tests**

```bash
cargo test -p harmony-node -- --nocapture 2>&1 | tail -5
```

Expected: all tests pass (including new config tests).

- [ ] **Step 4: Full workspace compilation check**

```bash
cargo check -p harmony-node 2>&1 | tail -5
```

Expected: compiles clean.

- [ ] **Step 5: Check for warnings**

```bash
cargo check -p harmony-node 2>&1 | grep "^warning" | grep -v "harmony_engine\|qwen3_ext" | head -10
```

Expected: no new warnings (pre-existing warnings in `harmony_engine.rs` and `qwen3_ext.rs` are acceptable).

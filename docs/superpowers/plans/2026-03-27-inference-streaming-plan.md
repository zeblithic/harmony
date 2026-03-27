# Inference Streaming Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the StreamChunk protocol into the NodeRuntime inference path so tokens are published to Zenoh as they're generated.

**Architecture:** On inference request, the event loop takes the QwenEngine from the runtime, spawns a `spawn_blocking` task that runs the token loop, sends StreamChunk payloads back via an mpsc channel, and returns the engine on completion. The event loop publishes chunks to Zenoh and sends the final AgentResult via query reply.

**Tech Stack:** harmony-agent (StreamChunk), harmony-inference (QwenEngine), harmony-zenoh (namespace), tokio (spawn_blocking, mpsc)

**Spec:** `docs/superpowers/specs/2026-03-27-inference-streaming-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/Cargo.toml` | Add `harmony-agent` dependency behind `inference` feature |
| `crates/harmony-node/src/inference.rs` | Add `SamplingParams` conversion from raw bytes, `InferenceStreamConfig` |
| `crates/harmony-node/src/event_loop.rs` | Add `InferenceResult` enum, mpsc channel, `select!` arm, spawn inference task |
| `crates/harmony-node/src/runtime.rs` | Add `RuntimeAction::RunInference`, modify inference dispatch to emit it |

**Out of scope for this plan:** `runtime.rs` is ~6000 lines. This plan touches it minimally — adding one RuntimeAction variant and changing ~20 lines in the inference dispatch match arm.

---

### Task 1: Add harmony-agent Dependency and SamplingParams Conversion

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`
- Modify: `crates/harmony-node/src/inference.rs`

**Context:** `InferenceRequest::sampling_params` is `[u8; 20]` — 5 packed values (temp:f32, top_p:f32, top_k:u32, repeat_penalty:f32, repeat_last_n:u32). The inference engine needs `harmony_inference::SamplingParams`. We need a conversion function. Also, add `harmony-agent` to the node's dependencies.

- [ ] **Step 1: Add harmony-agent to Cargo.toml**

In `crates/harmony-node/Cargo.toml`, add `harmony-agent` as an optional dependency and include it in the `inference` feature:

Under `[dependencies]`, add:
```toml
harmony-agent = { workspace = true, optional = true }
serde_json.workspace = true
```

In the `inference` feature, add `"dep:harmony-agent"`:
```toml
inference = ["harmony-compute/inference", "harmony-workflow/inference", "dep:harmony-inference", "dep:candle-core", "dep:harmony-agent"]
```

Note: `serde_json` is unconditional (not feature-gated) because it's a lightweight dependency used for StreamChunk payload construction.

- [ ] **Step 2: Add SamplingParams conversion to inference.rs**

Add this function to `crates/harmony-node/src/inference.rs`:

```rust
/// Convert raw 20-byte sampling parameters to `SamplingParams`.
///
/// Layout: `[temperature:f32 LE] [top_p:f32 LE] [top_k:u32 LE] [repeat_penalty:f32 LE] [repeat_last_n:u32 LE]`
#[cfg(feature = "inference")]
pub fn decode_sampling_params(raw: &[u8; 20]) -> harmony_inference::SamplingParams {
    let temperature = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let top_p = f32::from_le_bytes([raw[4], raw[5], raw[6], raw[7]]);
    let top_k = u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]);
    let repeat_penalty = f32::from_le_bytes([raw[12], raw[13], raw[14], raw[15]]);
    let repeat_last_n = u32::from_le_bytes([raw[16], raw[17], raw[18], raw[19]]) as usize;
    harmony_inference::SamplingParams {
        temperature,
        top_p,
        top_k,
        repeat_penalty,
        repeat_last_n,
    }
}
```

- [ ] **Step 3: Add test for SamplingParams conversion**

Add to the existing `#[cfg(test)] mod tests` block in `inference.rs`:

```rust
#[test]
fn decode_sampling_params_greedy() {
    // Greedy defaults: temp=0.0, top_p=1.0, top_k=0, repeat_penalty=1.0, repeat_last_n=64
    let mut raw = [0u8; 20];
    raw[4..8].copy_from_slice(&1.0f32.to_le_bytes()); // top_p
    raw[12..16].copy_from_slice(&1.0f32.to_le_bytes()); // repeat_penalty
    raw[16..20].copy_from_slice(&64u32.to_le_bytes()); // repeat_last_n

    let params = super::decode_sampling_params(&raw);
    assert_eq!(params.temperature, 0.0);
    assert_eq!(params.top_p, 1.0);
    assert_eq!(params.top_k, 0);
    assert_eq!(params.repeat_penalty, 1.0);
    assert_eq!(params.repeat_last_n, 64);
}

#[test]
fn decode_sampling_params_custom() {
    let mut raw = [0u8; 20];
    raw[0..4].copy_from_slice(&0.7f32.to_le_bytes()); // temperature
    raw[4..8].copy_from_slice(&0.9f32.to_le_bytes()); // top_p
    raw[8..12].copy_from_slice(&40u32.to_le_bytes()); // top_k
    raw[12..16].copy_from_slice(&1.1f32.to_le_bytes()); // repeat_penalty
    raw[16..20].copy_from_slice(&128u32.to_le_bytes()); // repeat_last_n

    let params = super::decode_sampling_params(&raw);
    assert!((params.temperature - 0.7).abs() < 1e-6);
    assert!((params.top_p - 0.9).abs() < 1e-6);
    assert_eq!(params.top_k, 40);
    assert!((params.repeat_penalty - 1.1).abs() < 1e-6);
    assert_eq!(params.repeat_last_n, 128);
}
```

Note: These tests need the `inference` feature. Run with: `cargo test -p harmony-node --features inference -- decode_sampling_params`

If the `inference` feature requires heavy dependencies (candle, etc.) that aren't available in the test environment, wrap the test in `#[cfg(feature = "inference")]` and verify compilation with `cargo check -p harmony-node --features inference` instead.

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles (may warn about unused imports — OK)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/inference.rs Cargo.lock
git commit -m "feat(node): add harmony-agent dependency and SamplingParams conversion"
```

---

### Task 2: InferenceResult Enum and Channel Setup

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

**Context:** The event loop already has `DiskIoResult`/`S3IoResult` enums and mpsc channels at lines 25-47 and 488-493. Follow the exact same pattern. The `InferenceResult` enum needs to carry the engine back on completion/failure.

- [ ] **Step 1: Add InferenceResult enum**

Add after the `S3IoResult` enum definition (around line 47) in `event_loop.rs`:

```rust
/// Results from a streaming inference task.
#[cfg(feature = "inference")]
enum InferenceResult {
    /// A streaming token to publish to Zenoh.
    Chunk {
        task_id: String,
        sequence: u32,
        token_text: String,
        final_chunk: bool,
    },
    /// Inference completed — send query reply and return engine.
    Complete {
        query_id: u64,
        task_id: String,
        full_text: String,
        engine: harmony_inference::QwenEngine,
    },
    /// Inference failed — send error reply and return engine.
    Failed {
        query_id: u64,
        task_id: String,
        error: String,
        engine: harmony_inference::QwenEngine,
    },
}
```

- [ ] **Step 2: Add channel creation**

Find where `disk_tx`/`disk_rx` and `s3_tx`/`s3_rx` are created (around line 488-493). Add the inference channel nearby:

```rust
#[cfg(feature = "inference")]
let (inference_tx, mut inference_rx) = mpsc::channel::<InferenceResult>(64);
```

- [ ] **Step 3: Add inference_engine field**

The engine currently lives in the runtime as `verification_engine`. For streaming, we need to take it out via `Option::take()`. The simplest approach: the event loop accesses the runtime's engine directly via `runtime.take_inference_engine()` / `runtime.return_inference_engine(engine)` methods. Add these methods to the runtime (they just wrap `self.verification_engine.take()` and `self.verification_engine = Some(engine)`).

In `runtime.rs`, add to the `impl NodeRuntime` block:

```rust
/// Take the inference engine for an async streaming task.
/// Returns None if the engine is already in use or not loaded.
#[cfg(feature = "inference")]
pub fn take_inference_engine(&mut self) -> Option<harmony_inference::QwenEngine> {
    self.verification_engine.take()
}

/// Return the inference engine after an async streaming task completes.
#[cfg(feature = "inference")]
pub fn return_inference_engine(&mut self, engine: harmony_inference::QwenEngine) {
    self.verification_engine = Some(engine);
}
```

- [ ] **Step 4: Add inference_tx to dispatch_action parameters**

The `dispatch_action` function (line 1249) needs the inference channel sender. Add `inference_tx: &mpsc::Sender<InferenceResult>` parameter (feature-gated). Update all call sites of `dispatch_action` to pass the sender.

Note: There's only one call site in the select loop. Add the parameter and pass `&inference_tx` at the call site.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles (the channel and enum exist but aren't used yet — may warn)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/runtime.rs
git commit -m "feat(node): add InferenceResult channel and engine take/return methods"
```

---

### Task 3: RuntimeAction::RunInference and Dispatch

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs` — add action variant, modify inference dispatch
- Modify: `crates/harmony-node/src/event_loop.rs` — handle action in dispatch_action

**Context:** Currently, inference requests go through WASM via WorkflowEngine (lines 2260-2289 of runtime.rs). We replace that path with a `RuntimeAction::RunInference` that the event loop handles by spawning a `spawn_blocking` task.

- [ ] **Step 1: Add RuntimeAction::RunInference variant**

Add to the `RuntimeAction` enum in `runtime.rs` (after `S3Lookup` or at end):

```rust
/// Spawn a streaming inference task.
#[cfg(feature = "inference")]
RunInference {
    query_id: u64,
    task_id: String,
    prompt: String,
    sampling_params: harmony_inference::SamplingParams,
},
```

- [ ] **Step 2: Modify inference dispatch in handle_compute_query**

In `runtime.rs`, find the `ParsedCompute::Inference { request }` match arm (around line 2260). Replace the WorkflowEngine submission with:

```rust
ParsedCompute::Inference { request } => {
    #[cfg(feature = "inference")]
    {
        // Generate a unique task ID for this inference request.
        self.inference_request_nonce += 1;
        let task_id = format!(
            "{:016x}-{:016x}",
            u128::from_le_bytes(
                self.node_addr_bytes()[..16]
                    .try_into()
                    .unwrap_or([0u8; 16])
            ),
            self.inference_request_nonce
        );
        let params = crate::inference::decode_sampling_params(&request.sampling_params);
        return vec![RuntimeAction::RunInference {
            query_id,
            task_id,
            prompt: request.prompt,
            sampling_params: params,
        }];
    }
    #[cfg(not(feature = "inference"))]
    {
        let _ = &request;
        // Return empty reply for non-inference builds.
        self.pending_direct_actions.push(RuntimeAction::SendReply {
            query_id,
            payload: vec![],
        });
    }
}
```

Note: The exact structure of this match arm depends on the existing code. The implementer should read the current arm at lines 2260-2289 and replace the WorkflowEngine submission, keeping the CID checks and error handling for missing model/tokenizer.

- [ ] **Step 3: Intercept RunInference in the action dispatch loop**

**Critical design note:** `dispatch_action` is a free `async fn` that does NOT have `&mut runtime`. The engine take/return must happen at the call site in the event loop's action dispatch loop — NOT inside `dispatch_action`.

Find the action dispatch loop in the event loop (around lines 640-660). It iterates over `runtime.tick()` actions and calls `dispatch_action(action, ...)` for each. Add a pre-dispatch intercept:

```rust
for action in actions {
    // Intercept RunInference before dispatch_action (needs &mut runtime).
    #[cfg(feature = "inference")]
    if let RuntimeAction::RunInference { query_id, task_id, prompt, sampling_params } = action {
        // Take engine from runtime. If None, engine is busy or not loaded.
        let engine = match runtime.take_inference_engine() {
            Some(e) => e,
            None => {
                // Send busy rejection via query reply.
                // Use the pending_queries map or direct SendReply pattern.
                // Look at how other SendReply actions reach the query responder.
                continue;
            }
        };

        let tx = inference_tx.clone();
        let max_tokens = crate::inference::DEFAULT_MAX_INFERENCE_TOKENS;

        tokio::task::spawn_blocking(move || {
            run_inference_loop(engine, tx, query_id, task_id, prompt, sampling_params, max_tokens);
        });
        continue;
    }

    dispatch_action(action, /* existing params */).await;
}
```

Then add a helper function `run_inference_loop` (either in `event_loop.rs` or a new `inference_task.rs` file) that contains the token generation loop:

```rust
#[cfg(feature = "inference")]
fn run_inference_loop(
    mut engine: harmony_inference::QwenEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    prompt: String,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
) {
    use harmony_inference::InferenceEngine;

    // Tokenize.
    let tokens = match engine.tokenize(&prompt) {
        Ok(t) => t,
        Err(e) => {
            engine.reset();
            let _ = tx.blocking_send(InferenceResult::Failed {
                query_id, task_id, error: format!("tokenize failed: {e}"), engine,
            });
            return;
        }
    };

    // Initial forward pass with prompt tokens.
    let mut logits = match engine.forward(&tokens) {
        Ok(l) => l,
        Err(e) => {
            engine.reset();
            let _ = tx.blocking_send(InferenceResult::Failed {
                query_id, task_id, error: format!("forward failed: {e}"), engine,
            });
            return;
        }
    };

    let mut full_text = String::new();
    let mut sequence = 0u32;
    let eos = engine.eos_token_id();

    // Token generation loop.
    loop {
        let next_token = match engine.sample(&logits, &sampling_params) {
            Ok(t) => t,
            Err(e) => {
                engine.reset();
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id, task_id, error: format!("sample failed: {e}"), engine,
                });
                return;
            }
        };

        // Check for EOS or max tokens.
        if eos == Some(next_token) || sequence >= max_tokens {
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(), sequence, token_text: String::new(), final_chunk: true,
            });
            break;
        }

        // Detokenize.
        let text = engine.detokenize(&[next_token]).unwrap_or_default();
        full_text.push_str(&text);

        // Send chunk.
        let _ = tx.blocking_send(InferenceResult::Chunk {
            task_id: task_id.clone(), sequence, token_text: text, final_chunk: false,
        });

        sequence += 1;

        // Forward next token.
        logits = match engine.forward(&[next_token]) {
            Ok(l) => l,
            Err(e) => {
                engine.reset();
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id, task_id, error: format!("forward failed: {e}"), engine,
                });
                return;
            }
        };
    }

    engine.reset();
    let _ = tx.blocking_send(InferenceResult::Complete {
        query_id, task_id, full_text, engine,
    });
}
```

**Key point:** The implementer must read the existing action dispatch loop to find the exact insertion point and understand how `runtime` is accessed there. The loop has `&mut runtime` in scope (it's inside the `select!` body that owns runtime).

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles (may have warnings about unused inference_rx — OK, handled in Task 4)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): RuntimeAction::RunInference with spawn_blocking token loop"
```

---

### Task 4: Select Arm for Inference Results

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

**Context:** The event loop needs a `select!` arm that receives `InferenceResult` from the mpsc channel, publishes StreamChunks to Zenoh, and handles completion/failure. Follow the existing disk_rx/s3_rx pattern.

- [ ] **Step 1: Add select! arm for inference_rx**

Find the existing `select!` arms for `disk_rx` and `s3_rx` (around lines 666-687). Add a new arm:

```rust
#[cfg(feature = "inference")]
Some(inference_result) = inference_rx.recv() => {
    match inference_result {
        InferenceResult::Chunk { task_id, sequence, token_text, final_chunk } => {
            // Build StreamChunk and publish to Zenoh.
            let chunk = harmony_agent::StreamChunk {
                task_id: task_id.clone(),
                sequence,
                payload: serde_json::json!({"token": token_text}),
                final_chunk,
            };
            if let Ok(payload) = harmony_agent::encode_chunk(&chunk) {
                let key_expr = harmony_zenoh::namespace::agent::stream_key(
                    &node_addr_hex,
                    &task_id,
                );
                let session = session.clone();
                tokio::spawn(async move {
                    if let Err(e) = session.put(&key_expr, payload).await {
                        tracing::warn!(%key_expr, err = %e, "stream chunk publish error");
                    }
                });
            }
        }
        InferenceResult::Complete { query_id, task_id, full_text, engine } => {
            // Return engine to runtime.
            runtime.return_inference_engine(engine);
            // Send AgentResult via query reply.
            let result = harmony_agent::AgentResult {
                task_id,
                status: harmony_agent::TaskStatus::Success,
                output: Some(serde_json::json!({"text": full_text})),
                error: None,
            };
            if let Ok(payload) = harmony_agent::encode_result(&result) {
                // Send query reply via runtime action.
                runtime.push_event(/* query reply mechanism */);
                // OR directly dispatch:
                let session = session.clone();
                tokio::spawn(async move {
                    // Reply to the original query — the implementer needs to
                    // determine how query replies work in this event loop.
                    // Look at how other SendReply actions are dispatched.
                });
            }
        }
        InferenceResult::Failed { query_id, task_id, error, engine } => {
            // Return engine to runtime.
            runtime.return_inference_engine(engine);
            // Send error AgentResult.
            let result = harmony_agent::AgentResult {
                task_id,
                status: harmony_agent::TaskStatus::Failed,
                output: None,
                error: Some(error),
            };
            if let Ok(payload) = harmony_agent::encode_result(&result) {
                // Same query reply mechanism as Complete.
            }
        }
    }
}
```

**Important notes for the implementer:**
- `node_addr_hex` needs to be available — likely `hex::encode(runtime.node_addr_bytes())` or similar. Check how other parts of the event loop access the node address.
- Query replies: look at how `RuntimeAction::SendReply` is dispatched in `dispatch_action` (around line 1290). The `query_id` maps to a pending query. The event loop likely has a `pending_queries` map or similar. The implementer needs to find and follow this pattern.
- `serde_json` is likely already a dependency of harmony-node (via harmony-agent). Verify.

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles with no errors

- [ ] **Step 3: Run existing tests**

Run: `cargo test -p harmony-node`
Expected: all existing tests pass (no regressions)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): select arm for inference streaming results"
```

---

### Task 5: Integration Verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Verify full workspace compilation**

Run: `cargo check --workspace`
Expected: compiles with no new errors

- [ ] **Step 2: Run format check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues in changed files (pre-existing issues in other files are OK)

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -p harmony-node --features inference`
Expected: no new warnings

- [ ] **Step 4: Run workspace tests**

Run: `cargo test --workspace --exclude harmony-tunnel`
Expected: all tests pass

- [ ] **Step 5: Commit any fixups**

If Steps 1-4 revealed issues, fix and commit:

```bash
git add -u
git commit -m "fix(node): address integration issues from workspace verification"
```

If no issues, skip this step.

# Wire Agent Streaming into NodeRuntime Inference Path

## Overview

Wire the StreamChunk protocol (harmony-agent) into the NodeRuntime inference path so tokens are published to Zenoh as they're generated. The event loop spawns a `spawn_blocking` inference task that runs the tokenize → forward → sample loop, sending StreamChunk payloads back via an mpsc channel. The event loop publishes chunks to Zenoh and sends the final AgentResult via query reply on completion.

**Goal:** Inference requests produce a real-time stream of tokens via Zenoh pub/sub, in addition to the existing query-reply result.

**Scope:** Producer-side only — the inference node publishes StreamChunk messages. Consumer-side subscription is out of scope.

## What Already Exists

- **StreamChunk type:** `harmony-agent::StreamChunk` with `task_id`, `sequence`, `payload`, `final_chunk` fields. Wire encode/decode via `encode_chunk`/`decode_chunk`.
- **Namespace builders:** `harmony-zenoh::namespace::agent::stream_key(agent_id, task_id)` and `stream_sub(agent_id)`.
- **InferenceEngine trait:** `QwenEngine` with `load_gguf`, `tokenize`, `forward`, `sample`, `detokenize`, `reset`, `eos_token_id`.
- **Inference queryable:** Already registered in NodeRuntime when model + tokenizer are loaded. Currently submits to WorkflowEngine as a batch.
- **Event loop async patterns:** Disk I/O and S3 fetches already use mpsc channels + `select!` arms.

## Design Decisions

### Async task with mpsc channel (not sans-I/O state machine)

Inference `forward()` takes milliseconds per token — too slow for the sans-I/O tick loop. The token-generation loop runs in a `spawn_blocking` task, sending results back via an mpsc channel. This matches the existing disk I/O and S3 channel patterns in the event loop.

### Engine ownership via move

The `QwenEngine` is moved out of `Option<QwenEngine>` into the spawned task for the duration of inference. The engine is returned via the `Complete`/`Failed` channel variants. This avoids shared-state issues — the event loop doesn't touch the engine while inference is running. While `self.engine` is `None`, new inference requests get an immediate "busy" rejection.

### Detokenized text payload

StreamChunk payloads contain `{"token": "text"}` — human-readable output. Token IDs are not included (YAGNI). DSD verification uses its own dedicated queryable path, not the streaming path.

### Single inference at a time

The node has one QwenEngine. While it's in use, new requests are rejected with `AgentResult { status: Rejected, error: "inference busy" }`. No queuing.

## Channel Protocol

```rust
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
        engine: QwenEngine,
    },
    /// Inference failed — send error reply and return engine.
    Failed {
        query_id: u64,
        task_id: String,
        error: String,
        engine: QwenEngine,
    },
}
```

The engine is always returned — even on failure — so the event loop can put it back into the `Option`.

## Data Flow

```
ComputeQuery { query_id, payload } arrives (inference request)
  → runtime: parse payload, extract prompt + sampling_params + task_id
  → runtime: emit RuntimeAction::RunInference { query_id, task_id, prompt, sampling_params }
  → event loop: take engine out of Option (if None → busy reply)
  → event loop: spawn_blocking with engine + inference_tx:
      1. engine.tokenize(prompt)
      2. loop:
           logits = engine.forward(&[token])
           next_token = engine.sample(&logits, &params)
           text = engine.detokenize(&[next_token])
           send Chunk { task_id, sequence++, token_text: text, final_chunk: false }
           if next_token == eos || sequence >= MAX_TOKENS → break
      3. send Chunk { ..., final_chunk: true }
      4. engine.reset()
      5. send Complete { query_id, task_id, full_text, engine }
      (on error at any step: engine.reset(), send Failed { ..., engine })
  → event loop select! arm receives InferenceResult:
      Chunk → encode StreamChunk, publish to harmony/agent/{agent_id}/stream/{task_id}
      Complete → encode AgentResult { status: Success, output: full_text }, send query reply, put engine back
      Failed → encode AgentResult { status: Failed, error }, send query reply, put engine back
```

## RuntimeAction Changes

```rust
/// Spawn an inference streaming task.
#[cfg(feature = "inference")]
RunInference {
    query_id: u64,
    task_id: String,
    prompt: String,
    sampling_params: SamplingParams,
},
```

This is the only new RuntimeAction variant. The runtime emits it instead of submitting to the WorkflowEngine for streaming-capable inference requests.

## Event Loop Changes

- Add `mpsc::channel::<InferenceResult>()` (same pattern as disk I/O channel)
- Add `inference_engine: Option<QwenEngine>` field (moved from wherever it currently lives)
- Handle `RuntimeAction::RunInference`: take engine, spawn_blocking the token loop
- Add `select!` arm for `inference_rx`: publish chunks, handle completion/failure
- Busy guard: if engine is `None` when `RunInference` arrives, send immediate rejection reply

## Constants

```rust
/// Maximum tokens to generate before stopping (safety limit).
const MAX_INFERENCE_TOKENS: u32 = 512;
```

Hardcoded default. Configurable limits are future work.

## Testing

### Event loop integration tests

1. **Inference produces stream chunks** — submit inference request, verify StreamChunk messages are published with incrementing sequence numbers.
2. **Final chunk has final_chunk: true** — verify the last chunk before completion has the flag set.
3. **AgentResult sent on completion** — verify query reply contains full concatenated text.
4. **Busy rejection** — submit two concurrent requests, verify second gets Rejected status.
5. **Engine returned on failure** — simulate inference error, verify engine is returned and subsequent requests work.
6. **Engine reset after completion** — verify engine.reset() called so KV cache is cleared for next request.

Note: These are integration tests in harmony-node, not unit tests — the token loop runs in a spawned task. They require a loaded QwenEngine (or a mock implementing InferenceEngine).

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-node/Cargo.toml` | Add `harmony-agent` dependency (feature-gated behind `inference`) |
| `crates/harmony-node/src/event_loop.rs` | Add `InferenceResult` enum, mpsc channel, `select!` arm, spawn inference task, busy guard |
| `crates/harmony-node/src/runtime.rs` | Add `RuntimeAction::RunInference`, modify `handle_compute_query` to emit it for inference requests |

## Dependencies

`harmony-agent` added to `harmony-node` (behind `inference` feature gate). No new external crates.

## What This Does NOT Include

- **Consumer-side subscription** — consumers subscribe to stream keys themselves.
- **DSD streaming** — DSD uses its own verify queryable, not the streaming path.
- **Configurable max_tokens** — hardcoded at 512.
- **Cancellation** — inference runs to completion once started.
- **Concurrent inference** — single engine, busy rejection for overlapping requests.
- **Token IDs in payload** — text only, no `token_id` field.

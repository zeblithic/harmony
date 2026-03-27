# Token-Level Inference Queryable API

## Overview

Extend the inference queryable to accept pre-tokenized token IDs (tag 0x03) and return raw token IDs in stream chunks and the final result. The existing text-in/text-out path (tag 0x02) is unchanged. This enables clients that manage their own tokenization for advanced use cases like constrained decoding.

**Goal:** Clients can send raw token IDs and receive raw token IDs back, bypassing tokenization/detokenization entirely.

**Scope:** Single-shot token-level inference (input tokens → stream of output token IDs → final result). No interactive multi-step sessions, no logprobs.

## What Already Exists

- **Tag 0x02 (text inference):** `InferenceRequest` with `prompt: String`, parsed in `inference.rs`, dispatched via `RuntimeAction::RunInference`, streamed via `run_inference_loop` with `StreamChunk` payloads `{"token": "text"}`.
- **Tag 0x03:** Unassigned and available in `parse_compute_payload`.
- **InferenceEngine trait:** `tokenize(&str)`, `forward(&[u32])`, `sample(&[f32], &SamplingParams)`, `detokenize(&[u32])` — tokenization is cleanly separated from forward/sample.
- **Streaming infrastructure:** `InferenceResult` channel, `select!` arm, `handle_inference_result` — all from PR #139.

## Design Decisions

### Token IDs only in stream (not text + IDs)

Callers using the 0x03 path are doing advanced work (constrained decoding) and need IDs. Callers who want text use the existing 0x02 path. Mixing both adds payload size for no benefit — the two APIs serve different audiences.

### Single-shot, not interactive

The 0x03 path follows the same fire-and-forget flow as 0x02: send input, receive stream of outputs, get final result. Interactive token-by-token sessions (send one token, get one back, repeat) require a fundamentally different stateful protocol and should be a separate bead.

### Shared RuntimeAction with input enum

Both 0x02 and 0x03 use the same `RuntimeAction::RunInference` variant. An `InferenceInput` enum distinguishes text from token input. This avoids duplicating the dispatch path and the busy/DSD guards.

## New Types

### inference.rs

```rust
/// Request payload tag for token-level inference queries.
pub const TOKEN_INFERENCE_TAG: u8 = 0x03;

/// Parsed token-level inference request.
pub struct TokenInferenceRequest {
    /// Pre-tokenized input token IDs.
    pub token_ids: Vec<u32>,
    /// 20-byte sampling parameters (same layout as InferenceRequest).
    pub sampling_params: [u8; 20],
}
```

Wire format:
```
[0x03] [token_count: u32 LE] [token_ids: u32 LE × count] [sampling_params: 20 bytes (optional)]
```

Same greedy defaults as 0x02 when sampling params are absent.

### InferenceInput / InferenceOutput

```rust
/// Input to the inference loop — either text (0x02) or pre-tokenized (0x03).
pub enum InferenceInput {
    Text(String),
    TokenIds(Vec<u32>),
}

/// Output from the inference loop — matches the input type.
pub enum InferenceOutput {
    Text(String),
    TokenIds(Vec<u32>),
}
```

## Modified Types

### RuntimeAction::RunInference

```rust
RunInference {
    query_id: u64,
    task_id: String,
    input: InferenceInput,
    sampling_params_raw: [u8; 20],
},
```

Replaces the current `prompt: String` field with `input: InferenceInput`.

### InferenceResult::Complete

```rust
Complete {
    query_id: u64,
    task_id: String,
    output: InferenceOutput,
    engine: QwenEngine,
},
```

Replaces `full_text: String` with `output: InferenceOutput`.

## Data Flow

### 0x02 path (unchanged behavior)

```
ComputeQuery [0x02] → InferenceRequest::parse → RunInference { input: Text(prompt) }
→ run_inference_loop: tokenize(prompt) → forward/sample loop
→ Chunk { payload: {"token": "text"} }
→ Complete { output: Text(full_text) }
→ AgentResult { output: {"text": "full text"} }
```

### 0x03 path (new)

```
ComputeQuery [0x03] → TokenInferenceRequest::parse → RunInference { input: TokenIds(ids) }
→ run_inference_loop: skip tokenize, use ids directly → forward/sample loop
→ Chunk { payload: {"token_id": 42} }
→ Complete { output: TokenIds(all_ids) }
→ AgentResult { output: {"token_ids": [42, 17, ...]} }
```

## run_inference_loop Changes

The function signature changes from `prompt: String` to `input: InferenceInput`.

At the top of the loop:

```rust
let (tokens, is_token_mode) = match input {
    InferenceInput::Text(prompt) => (engine.tokenize(&prompt)?, false),
    InferenceInput::TokenIds(ids) => (ids, true),
};
```

In the token generation loop:
- `is_token_mode == false`: chunk payload `{"token": text}`, accumulate `full_text: String`
- `is_token_mode == true`: chunk payload `{"token_id": id}`, accumulate `generated_ids: Vec<u32>`

On completion:
- `is_token_mode == false`: `InferenceOutput::Text(full_text)`
- `is_token_mode == true`: `InferenceOutput::TokenIds(generated_ids)`

## handle_inference_result Changes

The `Complete` arm checks the output type:
- `InferenceOutput::Text(text)` → `AgentResult { output: {"text": text} }` (unchanged)
- `InferenceOutput::TokenIds(ids)` → `AgentResult { output: {"token_ids": [id1, id2, ...]} }`

## ParsedCompute Changes

New variant:
```rust
TokenInference { request: crate::inference::TokenInferenceRequest },
```

New match arm in `parse_compute_payload`:
```rust
0x03 => match crate::inference::TokenInferenceRequest::parse(&payload) {
    Ok(request) => Some(ParsedCompute::TokenInference { request }),
    Err(_) => None,
},
```

The `TokenInference` handler follows the same path as `Inference` — same model-loaded guard, same DSD session guard, same `RunInference` emission — just with `InferenceInput::TokenIds` instead of `InferenceInput::Text`.

## Testing

### inference.rs unit tests

1. **TokenInferenceRequest parse valid** — valid 0x03 payload with 3 token IDs + sampling params → correct parse.
2. **TokenInferenceRequest parse no params** — valid payload without sampling params → greedy defaults.
3. **TokenInferenceRequest parse wrong tag** — 0x02 tag → error.
4. **TokenInferenceRequest parse truncated** — payload too short for declared token count → error.
5. **TokenInferenceRequest parse empty tokens** — zero token count → valid (empty vec).

### runtime.rs unit tests

6. **Token inference query no model** — 0x03 payload without model loaded → Rejected AgentResult.
7. **Token inference query emits RunInference** — 0x03 payload with model loaded → RunInference with TokenIds input.

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-node/src/inference.rs` | Add `TOKEN_INFERENCE_TAG`, `TokenInferenceRequest`, `InferenceInput`, `InferenceOutput` |
| `crates/harmony-node/src/runtime.rs` | Add `ParsedCompute::TokenInference`, `0x03` parse arm, modify `RunInference` to use `InferenceInput`, add handler for `TokenInference` |
| `crates/harmony-node/src/event_loop.rs` | Modify `run_inference_loop` to accept `InferenceInput`, modify `InferenceResult::Complete` to use `InferenceOutput`, modify chunk/completion payload logic |

## Dependencies

No new crates. No new features.

## What This Does NOT Include

- **Interactive multi-step sessions** — different protocol, separate bead.
- **Logprobs in stream chunks** — future enhancement.
- **Changes to 0x02 behavior** — fully backwards compatible.
- **Client-side constrained decoding** — this provides the primitive; decoding logic is caller-side.

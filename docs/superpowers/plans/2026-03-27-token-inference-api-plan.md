# Token-Level Inference Queryable API Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tag 0x03 token-level inference API — clients send pre-tokenized token IDs, receive raw token IDs back in stream chunks and final result.

**Architecture:** New `TokenInferenceRequest` type + `InferenceInput`/`InferenceOutput` enums. The existing `RunInference` action and `run_inference_loop` are modified to accept either text or token input. The 0x02 text path is unchanged.

**Tech Stack:** Rust, harmony-node (inference feature), harmony-inference (InferenceEngine trait)

**Spec:** `docs/superpowers/specs/2026-03-27-token-inference-api-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/src/inference.rs` | Add `TOKEN_INFERENCE_TAG`, `TokenInferenceRequest`, `InferenceInput`, `InferenceOutput`, greedy defaults helper |
| `crates/harmony-node/src/runtime.rs` | Add `ParsedCompute::TokenInference`, `0x03` parse arm, modify `RunInference` to use `InferenceInput`, add handler |
| `crates/harmony-node/src/event_loop.rs` | Modify `run_inference_loop` to accept `InferenceInput`, modify `InferenceResult::Complete` to use `InferenceOutput`, modify chunk/completion payload logic |

---

### Task 1: TokenInferenceRequest Type and InferenceInput/Output Enums

**Files:**
- Modify: `crates/harmony-node/src/inference.rs`

**Context:** Follow the existing `InferenceRequest` pattern exactly. The new type parses tag 0x03 with token IDs instead of a text prompt. `InferenceInput`/`InferenceOutput` enums will be used by the runtime and event loop to distinguish text vs token mode.

- [ ] **Step 1: Add constants and types**

After `INFERENCE_TAG` (line 7), add:

```rust
/// Request payload tag for token-level inference queries.
pub const TOKEN_INFERENCE_TAG: u8 = 0x03;

/// Maximum token count to prevent allocation bombs from untrusted input.
const MAX_INPUT_TOKENS: u32 = 131_072;
```

After the `InferenceRequest` impl block (after line 73), add:

```rust
/// Parsed token-level inference request from a Zenoh query payload.
pub struct TokenInferenceRequest {
    /// Pre-tokenized input token IDs.
    pub token_ids: Vec<u32>,
    /// 20-byte sampling parameters (same layout as InferenceRequest).
    pub sampling_params: [u8; 20],
}

impl TokenInferenceRequest {
    /// Parse a token-level inference request from a query payload tagged with 0x03.
    ///
    /// Format: `[0x03] [token_count: u32 LE] [token_ids: u32 LE × count] [sampling_params: 20 bytes (optional)]`
    ///
    /// If sampling params are absent, greedy defaults are used (temperature=0.0).
    pub fn parse(payload: &[u8]) -> Result<Self, String> {
        if payload.is_empty() || payload[0] != TOKEN_INFERENCE_TAG {
            return Err(format!(
                "expected token inference tag 0x{:02x}, got 0x{:02x}",
                TOKEN_INFERENCE_TAG,
                payload.first().copied().unwrap_or(0)
            ));
        }
        if payload.len() < 5 {
            return Err("payload too short for token count".into());
        }
        let token_count =
            u32::from_le_bytes([payload[1], payload[2], payload[3], payload[4]]);
        if token_count > MAX_INPUT_TOKENS {
            return Err(format!(
                "token count {} exceeds maximum {}",
                token_count, MAX_INPUT_TOKENS
            ));
        }
        let token_count = token_count as usize;
        let tokens_end = 5 + token_count * 4;
        if payload.len() < tokens_end {
            return Err(format!(
                "payload too short: need {} bytes for {} tokens, have {}",
                token_count * 4,
                token_count,
                payload.len() - 5
            ));
        }
        let mut token_ids = Vec::with_capacity(token_count);
        for i in 0..token_count {
            let offset = 5 + i * 4;
            token_ids.push(u32::from_le_bytes([
                payload[offset],
                payload[offset + 1],
                payload[offset + 2],
                payload[offset + 3],
            ]));
        }

        let sampling_params = if payload.len() >= tokens_end + 20 {
            let mut params = [0u8; 20];
            params.copy_from_slice(&payload[tokens_end..tokens_end + 20]);
            params
        } else {
            greedy_defaults()
        };

        Ok(TokenInferenceRequest {
            token_ids,
            sampling_params,
        })
    }
}

/// Greedy sampling defaults: temperature=0.0, top_p=1.0, top_k=0, repeat_penalty=1.0, repeat_last_n=64.
fn greedy_defaults() -> [u8; 20] {
    let mut params = [0u8; 20];
    params[4..8].copy_from_slice(&1.0f32.to_le_bytes()); // top_p
    params[12..16].copy_from_slice(&1.0f32.to_le_bytes()); // repeat_penalty
    params[16..20].copy_from_slice(&64u32.to_le_bytes()); // repeat_last_n
    params
}

/// Input to the inference loop — either text (0x02) or pre-tokenized (0x03).
#[derive(Debug, Clone)]
pub enum InferenceInput {
    /// Text prompt — will be tokenized by the engine.
    Text(String),
    /// Pre-tokenized token IDs — skip tokenization.
    TokenIds(Vec<u32>),
}

/// Output from the inference loop — matches the input type.
#[derive(Debug, Clone)]
pub enum InferenceOutput {
    /// Concatenated text from detokenized tokens.
    Text(String),
    /// Raw generated token IDs.
    TokenIds(Vec<u32>),
}
```

Also refactor `InferenceRequest::parse` to use the shared `greedy_defaults()` helper. Replace lines 60-65 in the existing `parse` method:

```rust
        } else {
            greedy_defaults()
        };
```

- [ ] **Step 2: Add tests**

Add to the existing `#[cfg(test)] mod tests` block:

```rust
    #[test]
    fn parse_token_request_with_params() {
        let tokens: Vec<u32> = vec![100, 200, 300];
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in &tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }
        let mut params = [0u8; 20];
        params[0..4].copy_from_slice(&0.7f32.to_le_bytes());
        params[4..8].copy_from_slice(&1.0f32.to_le_bytes());
        params[12..16].copy_from_slice(&1.0f32.to_le_bytes());
        params[16..20].copy_from_slice(&64u32.to_le_bytes());
        payload.extend_from_slice(&params);

        let req = TokenInferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.token_ids, vec![100, 200, 300]);
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.7
        );
    }

    #[test]
    fn parse_token_request_greedy_defaults() {
        let tokens: Vec<u32> = vec![42];
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for &t in &tokens {
            payload.extend_from_slice(&t.to_le_bytes());
        }

        let req = TokenInferenceRequest::parse(&payload).unwrap();
        assert_eq!(req.token_ids, vec![42]);
        // temperature should be 0.0 (greedy)
        assert_eq!(
            f32::from_le_bytes([
                req.sampling_params[0],
                req.sampling_params[1],
                req.sampling_params[2],
                req.sampling_params[3]
            ]),
            0.0
        );
    }

    #[test]
    fn parse_token_request_wrong_tag() {
        let payload = vec![0x02, 0, 0, 0, 0];
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_token_request_truncated() {
        // Declare 100 tokens but only provide bytes for 0
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&100u32.to_le_bytes());
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }

    #[test]
    fn parse_token_request_empty_tokens() {
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&0u32.to_le_bytes());
        let req = TokenInferenceRequest::parse(&payload).unwrap();
        assert!(req.token_ids.is_empty());
    }

    #[test]
    fn parse_token_request_too_many_tokens() {
        let mut payload = vec![TOKEN_INFERENCE_TAG];
        payload.extend_from_slice(&(MAX_INPUT_TOKENS + 1).to_le_bytes());
        assert!(TokenInferenceRequest::parse(&payload).is_err());
    }
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-node -- inference`
Expected: all existing + 6 new tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/inference.rs
git commit -m "feat(node): add TokenInferenceRequest, InferenceInput/Output types for 0x03 API"
```

---

### Task 2: Runtime Dispatch for Token Inference

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Context:** Add `ParsedCompute::TokenInference` variant, a `0x03` match arm in `parse_compute_payload`, modify `RuntimeAction::RunInference` to use `InferenceInput`, and add a handler for `TokenInference` that follows the same path as text inference (same guards, same action).

- [ ] **Step 1: Modify RuntimeAction::RunInference**

Find the `RunInference` variant (line 356). Change `prompt: String` to `input: crate::inference::InferenceInput`:

```rust
#[cfg(feature = "inference")]
RunInference {
    query_id: u64,
    task_id: String,
    input: crate::inference::InferenceInput,
    sampling_params_raw: [u8; 20],
},
```

This will cause compile errors in the two places that construct `RunInference` — fix them in Steps 2 and 3.

- [ ] **Step 2: Update existing text inference dispatch**

Find where `RunInference` is constructed for text inference (around line 2332). Change `prompt: request.prompt` to `input: crate::inference::InferenceInput::Text(request.prompt)`.

- [ ] **Step 3: Add ParsedCompute::TokenInference and 0x03 parse arm**

Find the `ParsedCompute` enum (around line 3788). Add:

```rust
TokenInference { request: crate::inference::TokenInferenceRequest },
```

Find `parse_compute_payload` (around line 3130). Add a `0x03` match arm after the `0x02` arm:

```rust
0x03 => match crate::inference::TokenInferenceRequest::parse(&payload) {
    Ok(request) => Some(ParsedCompute::TokenInference { request }),
    Err(_) => None,
},
```

- [ ] **Step 4: Add TokenInference handler**

Find the `ParsedCompute::Inference { request }` handler (around line 2315). After it, add a handler for `TokenInference` that follows the same structure — same model-loaded guard, same DSD session guard, same `inference_running` guard, but constructs `InferenceInput::TokenIds`:

```rust
ParsedCompute::TokenInference { request } => {
    // Same guards as text inference.
    match (self.inference_model_cid, self.inference_tokenizer_cid) {
        (Some(_), Some(_)) => {}
        _ => {
            #[cfg(feature = "inference")]
            {
                let result = harmony_agent::AgentResult {
                    task_id: String::new(),
                    status: harmony_agent::TaskStatus::Rejected,
                    output: None,
                    error: Some("no inference model loaded".into()),
                };
                let payload = harmony_agent::encode_result(&result)
                    .unwrap_or_else(|_| b"no inference model loaded".to_vec());
                self.pending_direct_actions
                    .push(RuntimeAction::SendReply { query_id, payload });
            }
            #[cfg(not(feature = "inference"))]
            {
                self.pending_direct_actions
                    .push(RuntimeAction::SendReply { query_id, payload: vec![] });
            }
            return Vec::new();
        }
    };

    #[cfg(feature = "inference")]
    {
        if self.dsd_session.is_some() {
            let result = harmony_agent::AgentResult {
                task_id: String::new(),
                status: harmony_agent::TaskStatus::Rejected,
                output: None,
                error: Some("engine busy with DSD session".into()),
            };
            let payload = harmony_agent::encode_result(&result)
                .unwrap_or_else(|_| b"engine busy".to_vec());
            self.pending_direct_actions
                .push(RuntimeAction::SendReply { query_id, payload });
            return Vec::new();
        }

        self.inference_request_nonce += 1;
        let task_id = format!(
            "{:016x}-{}",
            self.inference_request_nonce,
            hex::encode(self.local_identity_hash)
        );
        self.pending_direct_actions
            .push(RuntimeAction::RunInference {
                query_id,
                task_id,
                input: crate::inference::InferenceInput::TokenIds(request.token_ids),
                sampling_params_raw: request.sampling_params,
            });
        return Vec::new();
    }
    #[cfg(not(feature = "inference"))]
    {
        let _ = &request;
        self.pending_direct_actions
            .push(RuntimeAction::SendReply { query_id, payload: vec![] });
    }
    Vec::new()
}
```

Note: The implementer should DRY the model-loaded and DSD guards between the two handlers if possible (e.g., extract a helper). But getting it working first is more important.

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles (event_loop.rs will have errors since `run_inference_loop` signature changed — that's Task 3)

Actually — `RunInference` field change will break the event loop intercept too. The implementer should also update the event loop's destructuring of `RunInference` in this task to avoid a broken intermediate state. Find the `if let RuntimeAction::RunInference { ... }` intercept (around line 1157) and change `ref prompt` to `ref input`, then pass `input.clone()` to `run_inference_loop`. But `run_inference_loop` still takes `prompt: String` — so temporarily change the call to extract the text:

```rust
let prompt = match input {
    crate::inference::InferenceInput::Text(ref t) => t.clone(),
    crate::inference::InferenceInput::TokenIds(_) => {
        // Token mode not yet supported in the loop — Task 3 will add it.
        // For now, reject with error.
        todo!("token inference loop support added in Task 3")
    }
};
```

This is a temporary bridge. Task 3 will change `run_inference_loop` to accept `InferenceInput`.

Run: `cargo check -p harmony-node --features inference`
Expected: compiles

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): add ParsedCompute::TokenInference and 0x03 dispatch"
```

---

### Task 3: Token Mode in Inference Loop and Completion Handling

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

**Context:** Modify `run_inference_loop` to accept `InferenceInput` (skip tokenization for TokenIds), modify `InferenceResult::Complete` to use `InferenceOutput`, and update chunk/completion payload generation based on the mode.

- [ ] **Step 1: Modify InferenceResult::Complete**

Find the `InferenceResult` enum (around line 51). Change `Complete`:

```rust
Complete {
    query_id: u64,
    task_id: String,
    output: crate::inference::InferenceOutput,
    engine: harmony_inference::QwenEngine,
},
```

- [ ] **Step 2: Modify run_inference_loop signature and tokenization**

Change the function signature from `prompt: String` to `input: crate::inference::InferenceInput`.

At the start, replace the tokenization with:

```rust
let (tokens, is_token_mode) = match input {
    crate::inference::InferenceInput::Text(prompt) => {
        match engine.tokenize(&prompt) {
            Ok(t) => (t, false),
            Err(e) => {
                engine.reset();
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("tokenize failed: {e}"),
                    engine,
                });
                return;
            }
        }
    }
    crate::inference::InferenceInput::TokenIds(ids) => (ids, true),
};
```

- [ ] **Step 3: Modify token generation loop for dual mode**

In the token loop, change chunk construction:

```rust
// Replace:
let text = engine.detokenize(&[next_token]).unwrap_or_default();
full_text.push_str(&text);
let _ = tx.blocking_send(InferenceResult::Chunk {
    task_id: task_id.clone(),
    sequence,
    token_text: text,
    final_chunk: false,
});

// With:
if is_token_mode {
    generated_ids.push(next_token);
    let _ = tx.blocking_send(InferenceResult::Chunk {
        task_id: task_id.clone(),
        sequence,
        token_text: next_token.to_string(), // token_id as string for the JSON payload
        final_chunk: false,
    });
} else {
    let text = engine.detokenize(&[next_token]).unwrap_or_default();
    full_text.push_str(&text);
    let _ = tx.blocking_send(InferenceResult::Chunk {
        task_id: task_id.clone(),
        sequence,
        token_text: text,
        final_chunk: false,
    });
}
```

Add `let mut generated_ids: Vec<u32> = Vec::new();` alongside the existing `let mut full_text = String::new();`.

Wait — the `Chunk` variant uses `token_text: String` which works for text mode but is awkward for token mode. Rather than changing the `Chunk` variant, use the existing `token_text` field: for text mode it's the detokenized text, for token mode it's the stringified token ID. The `handle_inference_result` select arm will construct the right JSON payload.

Actually simpler: add a `token_id: Option<u32>` field to `Chunk`:

```rust
Chunk {
    task_id: String,
    sequence: u32,
    token_text: String,
    token_id: Option<u32>,
    final_chunk: bool,
},
```

Text mode: `token_id: None`, `token_text: "Hello"` → payload `{"token": "Hello"}`
Token mode: `token_id: Some(42)`, `token_text: ""` → payload `{"token_id": 42}`

- [ ] **Step 4: Modify completion**

Change the completion send:

```rust
let output = if is_token_mode {
    crate::inference::InferenceOutput::TokenIds(generated_ids)
} else {
    crate::inference::InferenceOutput::Text(full_text)
};
engine.reset();
let _ = tx.blocking_send(InferenceResult::Complete {
    query_id,
    task_id,
    output,
    engine,
});
```

- [ ] **Step 5: Update handle_inference_result**

In `handle_inference_result`, update the `Chunk` arm to check `token_id`:

```rust
InferenceResult::Chunk { task_id, sequence, token_text, token_id, final_chunk } => {
    let payload_value = if let Some(id) = token_id {
        serde_json::json!({"token_id": id})
    } else {
        serde_json::json!({"token": token_text})
    };
    let chunk = harmony_agent::StreamChunk {
        task_id: task_id.clone(),
        sequence,
        payload: payload_value,
        final_chunk,
    };
    // ... publish to Zenoh (existing code)
}
```

Update the `Complete` arm:

```rust
InferenceResult::Complete { query_id, task_id, output, engine } => {
    runtime.return_inference_engine(engine);
    let output_json = match &output {
        crate::inference::InferenceOutput::Text(text) => serde_json::json!({"text": text}),
        crate::inference::InferenceOutput::TokenIds(ids) => serde_json::json!({"token_ids": ids}),
    };
    let result = harmony_agent::AgentResult {
        task_id,
        status: harmony_agent::TaskStatus::Success,
        output: Some(output_json),
        error: None,
    };
    // ... encode and dispatch SendReply (existing code)
}
```

- [ ] **Step 6: Remove the temporary bridge from Task 2**

Replace the `todo!()` / temporary text extraction in the event loop's `RunInference` intercept with passing `input.clone()` directly to `run_inference_loop`.

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p harmony-node --features inference`
Expected: compiles with no errors

- [ ] **Step 8: Run all tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

- [ ] **Step 9: Run fmt**

Run: `cargo fmt -p harmony-node`

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): token mode in inference loop and completion handling"
```

---

### Task 4: Integration Verification

**Files:**
- No new files — verification only

- [ ] **Step 1: Run format check**

Run: `cargo fmt -p harmony-node -- --check`
Expected: clean

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-node --features inference`
Expected: no new warnings

- [ ] **Step 3: Run workspace check**

Run: `cargo check --workspace`
Expected: compiles

- [ ] **Step 4: Run workspace tests**

Run: `cargo test --workspace --exclude harmony-tunnel`
Expected: all tests pass

- [ ] **Step 5: Commit any fixups**

If Steps 1-4 revealed issues, fix and commit:

```bash
git add -u
git commit -m "fix(node): address integration issues from workspace verification"
```

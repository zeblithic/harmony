# Decentralized Speculative Decoding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable edge nodes to draft tokens locally and send them to powerful mesh nodes for verification, achieving ~2-3x inference speedup via speculative decoding over Zenoh.

**Architecture:** New `harmony-speculative` crate for DSD protocol logic (sans-I/O). Integration in `harmony-node` for the target verifier queryable and edge orchestrator. Minor additions to `harmony-inference` for logprob computation and `harmony-zenoh` for namespace constants.

**Tech Stack:** harmony-speculative (new), harmony-inference, harmony-zenoh, harmony-node

**Spec:** `docs/superpowers/specs/2026-03-25-decentralized-speculative-decoding-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-speculative/Cargo.toml` | New crate manifest |
| `crates/harmony-speculative/src/lib.rs` | DraftEntry, VerifyRequest, VerifyResponse, constants |
| `crates/harmony-speculative/src/verify.rs` | should_accept_draft, sample_with_logprob, softmax |
| `crates/harmony-speculative/src/protocol.rs` | Payload serialization/deserialization (0x04 tag) |
| `crates/harmony-zenoh/src/namespace.rs` | VERIFY_ACTIVITY, SPECULATIVE_ACTIVITY constants |
| `crates/harmony-node/src/runtime.rs` | Target verifier + edge orchestrator integration |

---

### Task 1: New Crate — harmony-speculative Types + Verification Logic

Create the `harmony-speculative` crate with types, verification functions, and comprehensive unit tests.

**Files:**
- Create: `crates/harmony-speculative/Cargo.toml`
- Create: `crates/harmony-speculative/src/lib.rs`
- Create: `crates/harmony-speculative/src/verify.rs`
- Modify: `Cargo.toml` (workspace — add member + dep)

- [ ] **Step 1: Create crate manifest**

Create `crates/harmony-speculative/Cargo.toml`:
```toml
[package]
name = "harmony-speculative"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Decentralized Speculative Decoding protocol for Harmony"

[dependencies]
```

Add to workspace `Cargo.toml`:
- Add `"crates/harmony-speculative"` to `[workspace] members`
- Add `harmony-speculative = { path = "crates/harmony-speculative" }` to `[workspace.dependencies]`

- [ ] **Step 2: Create lib.rs with types and constants**

```rust
pub mod verify;
pub mod protocol;

/// Default number of draft tokens per verification round.
pub const DEFAULT_DRAFT_GAMMA: u8 = 5;

/// Payload tag for verify requests.
pub const VERIFY_TAG: u8 = 0x04;

/// A single draft token paired with its log-probability from the draft model.
#[derive(Debug, Clone, PartialEq)]
pub struct DraftEntry {
    pub token_id: u32,
    pub logprob: f32,
}

/// Verify request — sent from edge to target.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyRequest {
    pub context_tokens: Vec<u32>,
    pub drafts: Vec<DraftEntry>,
}

/// Verify response — sent from target to edge.
#[derive(Debug, Clone, PartialEq)]
pub struct VerifyResponse {
    pub accepted_count: u8,
    pub bonus_token: u32,
    pub bonus_logprob: f32,
}
```

- [ ] **Step 3: Create verify.rs with softmax, acceptance check, and sample_with_logprob**

```rust
/// Compute softmax probabilities from logits (in-place numerically stable).
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// Check whether a draft token should be accepted (greedy criterion).
///
/// Accept if P_target(token) >= P_draft(token).
pub fn should_accept_draft(
    target_logits: &[f32],
    draft_token_id: u32,
    draft_logprob: f32,
) -> bool {
    let probs = softmax(target_logits);
    let p_target = probs.get(draft_token_id as usize).copied().unwrap_or(0.0);
    let p_draft = draft_logprob.exp();
    p_target >= p_draft
}

/// Sample the token with the highest probability (greedy) and return
/// its ID and log-probability.
pub fn sample_greedy_with_logprob(logits: &[f32]) -> (u32, f32) {
    let probs = softmax(logits);
    let (token_id, &p) = probs
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    (token_id as u32, p.ln())
}

/// Compute the log-probability of a specific token given logits.
pub fn logprob_of(logits: &[f32], token_id: u32) -> f32 {
    let probs = softmax(logits);
    probs.get(token_id as usize).copied().unwrap_or(f32::MIN).ln()
}
```

- [ ] **Step 4: Write unit tests for verify.rs**

Tests in `verify.rs`:
- `softmax` produces valid probability distribution (sums to ~1.0)
- `should_accept_draft` returns true when draft token has high target probability
- `should_accept_draft` returns false when draft token has low target probability
- `sample_greedy_with_logprob` returns argmax token with correct logprob
- `logprob_of` returns correct logprob for a token
- Edge: single-element logits, all-zero logits (uniform distribution)

- [ ] **Step 5: Verify compilation and tests**

Run: `cargo test -p harmony-speculative`
Run: `cargo check -p harmony-speculative`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-speculative/ Cargo.toml Cargo.lock
git commit -m "feat: add harmony-speculative crate with DSD types and verification logic"
```

---

### Task 2: Protocol Serialization + Namespace Constants

Add payload serialization for verify requests/responses and namespace key expression constants.

**Files:**
- Create: `crates/harmony-speculative/src/protocol.rs`
- Modify: `crates/harmony-zenoh/src/namespace.rs`

- [ ] **Step 1: Create protocol.rs with serialization**

Implement `VerifyRequest::serialize()`, `VerifyRequest::parse()`, `VerifyResponse::serialize()`, `VerifyResponse::parse()`.

Verify Request format:
```
[0x04] [context_len: u32 LE] [context_tokens: u32[] LE] [draft_count: u8] [draft_entries...]
```
Each draft entry: `[token_id: u32 LE] [logprob: f32 LE]`

Verify Response success: `[0x00] [accepted_count: u8] [bonus_token: u32 LE] [bonus_logprob: f32 LE]`
Verify Response error: `[0x01] [error_message_utf8]`

- [ ] **Step 2: Write serialization unit tests**

- Roundtrip: serialize → parse → equals original (request and response)
- Parse wrong tag returns error
- Parse truncated payload returns error
- Parse empty payload returns error
- Response error variant roundtrip

- [ ] **Step 3: Add namespace constants**

In `crates/harmony-zenoh/src/namespace.rs` compute module:
```rust
/// Key expression for DSD verification queryable.
pub const VERIFY_ACTIVITY: &str = "harmony/compute/activity/verify";
/// Key expression for DSD speculative inference queryable.
pub const SPECULATIVE_ACTIVITY: &str = "harmony/compute/activity/speculative";
```

- [ ] **Step 4: Verify and commit**

Run: `cargo test -p harmony-speculative`
Run: `cargo check -p harmony-zenoh`

```bash
git add crates/harmony-speculative/src/protocol.rs crates/harmony-zenoh/src/namespace.rs
git commit -m "feat: add DSD protocol serialization and namespace constants"
```

---

### Task 3: Target Verifier in NodeRuntime

Add the target-side verification queryable to NodeRuntime. When a verify request arrives, the target runs sequential verification using its loaded model and replies with accepted count + bonus token.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml` (add harmony-speculative dep)

**Key context:**
- The target's `QwenEngine` is already loaded (via the inference queryable's startup flow)
- The verify queryable should be declared alongside the inference queryable when the model is loaded
- Verification uses `engine.reset()` + `engine.forward(&context)` for prefill, then `engine.forward(&[token])` per draft token
- The `inference_engine` is persisted in `WasmiRuntime::shared_inference_engine` — but verification should use the engine directly, NOT through WASM. The target verifier calls `QwenEngine` methods natively.

**Important design note:** The target verifier does NOT use the WASM workflow path. It calls `QwenEngine` directly from NodeRuntime for maximum efficiency. This means NodeRuntime needs direct access to a `QwenEngine` instance — separate from the one persisted inside WasmiRuntime.

The simplest approach: when the model data arrives at startup (in `check_inference_model_ready`), construct TWO engines — one goes to the WasmiRuntime (for WASM inference workflows) and one is stored directly in NodeRuntime (for native verification). However, this doubles memory. Better approach: construct one engine in NodeRuntime for both verification AND to seed the WasmiRuntime. On first inference workflow, the WasmiRuntime builds its own from the model data (which we cache until consumed). The NodeRuntime's engine is used exclusively for verification.

Actually, the simplest correct approach: store a `QwenEngine` in NodeRuntime for verification. The WASM inference path already has its own engine lifecycle (persisted in WasmiRuntime). The verification engine is separate.

- [ ] **Step 1: Add harmony-speculative dependency to harmony-node**

In `crates/harmony-node/Cargo.toml`:
```toml
harmony-speculative = { workspace = true }
```

- [ ] **Step 2: Add verification engine and queryable ID fields to NodeRuntime**

Add to `NodeRuntime`:
```rust
/// QwenEngine for native verification (DSD target side).
/// Separate from the WasmiRuntime's persisted engine.
#[cfg(feature = "inference")]
verification_engine: Option<harmony_inference::QwenEngine>,
/// Queryable ID for the verify endpoint.
verify_queryable_id: Option<QueryableId>,
```

Initialize to `None` in `new()`.

- [ ] **Step 3: Build verification engine during model loading**

In `check_inference_model_ready()`, after loading the GGUF + tokenizer, construct a second `QwenEngine` for verification. This requires the model data to still be available — since we now `take()` it in the LoadModel handler, we need to clone it for the verification engine BEFORE taking.

Alternative: build the verification engine directly in `check_inference_model_ready()` from the cached `inference_gguf_data` / `inference_tokenizer_data` before they're consumed. Add this before the existing queryable declaration.

Also declare the `verify` queryable alongside the existing `inference` queryable.

- [ ] **Step 4: Handle verify queries**

Add a new match arm in the query routing logic. When a query arrives on the verify queryable:

1. Parse `VerifyRequest` from payload
2. Call verification engine: `reset()`, `forward(&context_tokens)` for prefill
3. For each draft token: `forward(&[token])`, check `should_accept_draft`
4. On rejection or all-accepted: sample bonus token with `sample_greedy_with_logprob`
5. Serialize `VerifyResponse`, reply

This is direct native code in `push_event` or a helper method — no WASM, no WorkflowEngine.

- [ ] **Step 5: Verify compilation and existing tests**

Run: `cargo check -p harmony-node --features inference`
Run: `cargo test -p harmony-node`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): add DSD target verifier queryable with native QwenEngine verification"
```

---

### Task 4: Edge Orchestrator in NodeRuntime

Add the edge-side DSD orchestrator. When a speculative query arrives, the edge drafts tokens with its local model, sends verify requests to the target, and assembles the final output.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Key challenge:** The DSD loop is multi-round (draft → send verify → wait for reply → repeat). The NodeRuntime is sans-I/O, so "sending a verify query and waiting for a reply" must be modeled as:
1. Edge receives speculative query → starts DSD, drafts γ tokens
2. Edge emits a `RuntimeAction` to send a verify query to the target
3. The caller (event loop) sends the Zenoh query and delivers the reply back as a `RuntimeEvent`
4. Edge processes the reply, appends tokens, drafts again or finishes

This requires new `RuntimeAction` and `RuntimeEvent` variants for verify queries.

- [ ] **Step 1: Add DSD state to NodeRuntime**

```rust
/// Active DSD session state — tracks an in-progress speculative decoding loop.
#[cfg(feature = "inference")]
dsd_session: Option<DsdSession>,
```

Where `DsdSession` holds: the original `query_id` (to reply to the requester), accepted token sequence so far, draft model engine reference, target node address, sampling params, max_tokens, EOS token ID.

- [ ] **Step 2: Add RuntimeAction/RuntimeEvent for verify queries**

```rust
// RuntimeAction:
SendVerifyQuery { target_addr: String, payload: Vec<u8> },

// RuntimeEvent:
VerifyResponse { payload: Vec<u8> },
```

- [ ] **Step 3: Handle speculative query arrival**

When a query arrives on the `speculative` queryable:
1. Parse the text prompt (reuse `InferenceRequest::parse` with tag 0x02, or define a new tag)
2. Tokenize the prompt using the draft model's tokenizer
3. Prefill the draft model
4. Draft γ tokens, collecting logprobs
5. Build `VerifyRequest`, serialize, emit `SendVerifyQuery` action
6. Store DSD session state (awaiting verify response)

- [ ] **Step 4: Handle verify response**

When `RuntimeEvent::VerifyResponse` arrives:
1. Parse `VerifyResponse` from payload
2. Append accepted draft tokens + bonus token to session's token sequence
3. Check for EOS or max_tokens → if done, detokenize and reply to requester
4. Otherwise: reset draft model, prefill with full sequence, draft γ more tokens, emit another `SendVerifyQuery`

- [ ] **Step 5: Target discovery from capacity advertisements**

When capacity advertisements arrive (existing path), check if the model CID differs from own. Store as `target_node_addr`. Declare `speculative` queryable once a target is found.

- [ ] **Step 6: Verify compilation and tests**

Run: `cargo check -p harmony-node --features inference`
Run: `cargo test -p harmony-node`

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): add DSD edge orchestrator with draft-verify-append loop"
```

---

### Task 5: Integration Tests

Write tests for the full DSD flow.

**Files:**
- Modify: `crates/harmony-speculative/src/verify.rs` (add tests if needed)
- Modify: `crates/harmony-speculative/src/protocol.rs` (add tests if needed)
- Modify: `crates/harmony-node/src/runtime.rs` (add integration tests)

- [ ] **Step 1: End-to-end verification test with synthetic logits**

Create synthetic logits where specific tokens have known probabilities. Build a `VerifyRequest`, run the verification algorithm step by step, verify the response.

- [ ] **Step 2: NodeRuntime verify query routing test**

Create a NodeRuntime with inference configured. Send a verify query payload (tag 0x04). Verify it produces a reply (even if the model isn't real — test the routing, not the model).

- [ ] **Step 3: Protocol roundtrip with realistic sizes**

Test serialization with a 1000-token context and γ=5 drafts. Verify parse produces identical data.

- [ ] **Step 4: Run all tests**

Run: `cargo test -p harmony-speculative`
Run: `cargo test -p harmony-node`
Run: `cargo clippy -p harmony-speculative -- -D warnings`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-speculative/ crates/harmony-node/
git commit -m "test: add DSD verification and integration tests"
```

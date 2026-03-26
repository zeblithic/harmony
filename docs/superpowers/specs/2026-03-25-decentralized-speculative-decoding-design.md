# Decentralized Speculative Decoding

## Goal

Enable an edge node with a small draft model (e.g., Qwen3 0.8B) to generate candidate tokens locally, send them to a powerful mesh node (e.g., Qwen3 9B/35B) for parallel verification, and receive back which tokens were accepted — achieving ~2-3x speedup when network latency is lower than per-token target compute time.

Based on [Decentralized Speculative Decoding](https://arxiv.org/abs/2511.11733).

## Architecture

The edge node orchestrates the DSD loop. It drafts γ tokens with its small model, sends them to a target node for verification, and assembles the final output from accepted tokens plus a bonus/correction token.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-speculative` (new) | DSD protocol logic — sans-I/O state machine |
| `harmony-inference` | `InferenceEngine` trait, `QwenEngine`, `forward()` returns logits |
| `harmony-zenoh` | QueryableRouter, namespace key expressions |
| `harmony-node` | Integration — edge orchestrator + target verifier |

### Node Roles

- **Edge node:** Has a small draft model loaded. Discovers a target node via capacity advertisements. Declares `harmony/compute/activity/speculative` queryable for incoming DSD requests from other Harmony nodes or client applications.
- **Target node:** Has a large model loaded. Declares `harmony/compute/activity/verify` queryable. Receives draft tokens + logprobs, runs parallel verification, returns accepted count + bonus token.

### Key Expressions

| Key Expression | Purpose |
|---------------|---------|
| `harmony/compute/activity/verify` | Target node verification queryable |
| `harmony/compute/activity/speculative` | Edge node DSD queryable (text prompt in, text out) |
| `harmony/compute/capacity/{node_addr}` | Existing — model CID + status advertisement |

## Verification Protocol

### Verify Request

```
[0x04] [context_len: u32 LE] [context_tokens: u32[] LE] [draft_count: u8] [draft_entries...]
```

Each draft entry (8 bytes):
```
[token_id: u32 LE] [logprob: f32 LE]
```

- Tag `0x04` distinguishes verify requests from other compute payloads
- `context_tokens` is the full token sequence (prompt + previously accepted tokens) — the target needs them for its forward pass
- `draft_count` is γ (number of draft tokens, max 255)
- Each draft entry pairs the token ID with its log-probability from the draft model

Example: γ=5 with 512-token context ≈ 2KB context + 40 bytes drafts = ~2KB total. This payload is carried over Zenoh (not Reticulum packets), so the 500-byte Reticulum MTU does not apply.

### Verify Response

Success:
```
[0x00] [accepted_count: u8] [bonus_token: u32 LE] [bonus_logprob: f32 LE]
```

Error:
```
[0x01] [error_message_utf8]
```

- `accepted_count` — how many draft tokens were accepted (0 to γ)
- `bonus_token` — correction token at the first rejection point, or extra token if all accepted
- `bonus_logprob` — log-probability of the bonus token from the target model

## Verification Algorithm (Target Side)

### Sequential Verification

The existing `InferenceEngine::forward()` returns logits for the last position only (not all positions). The target verifies draft tokens sequentially, leveraging KV cache for efficiency:

1. Call `engine.reset()` to clear prior state
2. Call `engine.forward(&context_tokens)` — prefill, builds KV cache, returns logits at last context position (not used for verification, but primes the cache)
3. For each draft position `i` (0 to `draft_count - 1`):
   - Call `engine.forward(&[draft_token[i]])` — single-token decode, returns `target_logits`
   - Apply softmax to `target_logits` → `P_target`
   - Compute `p_target = P_target(draft_token[i])`
   - Compute `p_draft = exp(draft_logprob[i])`
   - **Greedy accept:** accept if `p_target >= p_draft`
   - If rejected: sample bonus token from target distribution at this position. Break.
4. If all γ tokens accepted: call `engine.forward(&[last_draft_token])` one more time, sample bonus token from those logits (the "free" extra token)
5. Compute bonus token's logprob, return response

This is γ+1 sequential forward calls (1 prefill + γ single-token decodes), not a single parallel pass. However, the KV cache from the prefill is reused across all decode steps, so each single-token decode is fast. The total compute is comparable to the target generating γ tokens normally — the speedup comes from the network round trips saved (1 round instead of γ).

**Note:** A future `forward_all()` API that returns logits at every position from a single pass would enable true parallel verification. This is tracked as future work.

### Greedy Verification

This first pass uses greedy verification (`accept if p_target >= p_draft`). Stochastic rejection sampling (`accept with probability min(1, p_target/p_draft)`) adds complexity and is a future enhancement.

### KV Cache Management

The target calls `engine.reset()` at the start of each verification request. Each request brings its own full context, so there's no KV cache reuse across rounds. This is less efficient but simpler and correct — KV cache persistence across DSD rounds is a future optimization.

## Edge Orchestrator

### DSD Loop

1. Receive text prompt from requester (via `speculative` queryable)
2. Tokenize prompt, prefill draft model
3. Loop until EOS or max_tokens:
   a. Draft γ tokens with draft model, collecting each token's logprob
   b. Send verify request to target node's `verify` queryable
   c. Receive response: `accepted_count` + `bonus_token`
   d. Append `accepted_count` draft tokens + `bonus_token` to output sequence
   e. Reset draft model, prefill with full accepted sequence
4. Detokenize output sequence, reply to requester

### Draft Token Generation

For each draft token, the edge:
1. Calls `engine.forward(&[last_token])` → logits
2. Applies softmax to get probabilities
3. Calls `engine.sample(logits, params)` → token_id
4. Records `logprob = ln(P_draft(token_id))`
5. Repeats γ times

### EOS Detection

The DSD loop terminates when the bonus token is the model's EOS token. The EOS token ID is obtained from the tokenizer via `tokenizer.token_to_id("</s>")` (Qwen3 convention) at model load time and stored alongside the engine. If the tokenizer has no EOS token, the loop relies on `max_tokens` only.

### Target Discovery

The edge discovers target nodes by observing `harmony/compute/capacity/{*}` advertisements. The capacity payload format is `[model_gguf_cid: 32 bytes] [status: u8]` (defined in `harmony-node/src/inference.rs`). A node qualifies as a target if its advertised model CID differs from the edge's draft model CID (implying a different, presumably larger model). For the first pass, the edge uses the first target it discovers — smart selection is future work.

### Draft Model State Reset

After each verification round, the draft model must be in the correct state for the next round. The simplest approach: `engine.reset()` followed by `engine.forward(&full_accepted_sequence)` to rebuild KV cache. This costs one extra prefill per round but is correct.

## New Crate: `harmony-speculative`

Sans-I/O state machine for DSD protocol logic.

### Public API

```rust
/// Default number of draft tokens per round.
pub const DEFAULT_DRAFT_GAMMA: u8 = 5;

/// A single draft token with its log-probability.
pub struct DraftEntry {
    pub token_id: u32,
    pub logprob: f32,
}

/// Verify request — sent from edge to target.
pub struct VerifyRequest {
    pub context_tokens: Vec<u32>,
    pub drafts: Vec<DraftEntry>,
}

/// Verify response — sent from target to edge.
pub struct VerifyResponse {
    pub accepted_count: u8,
    pub bonus_token: u32,
    pub bonus_logprob: f32,
}

/// Check whether a single draft token should be accepted.
///
/// Greedy criterion: accept if P_target(token) >= P_draft(token).
/// Returns true if accepted.
pub fn should_accept_draft(
    target_logits: &[f32],
    draft_token_id: u32,
    draft_logprob: f32,
) -> bool;

/// Sample a token from logits (greedy argmax or with sampling params).
/// Returns (token_id, logprob).
pub fn sample_with_logprob(
    logits: &[f32],
    params: &SamplingParams,
) -> (u32, f32);
```

### Payload Serialization

- `VerifyRequest::serialize() -> Vec<u8>` / `VerifyRequest::parse(payload: &[u8]) -> Result<Self, String>`
- `VerifyResponse::serialize() -> Vec<u8>` / `VerifyResponse::parse(payload: &[u8]) -> Result<Self, String>`

## Integration with NodeRuntime

### Target Node

- Declares `verify` queryable when model is loaded (alongside existing `inference` queryable)
- When a verify query arrives: parse request, run forward pass on context + drafts, call `verify_draft_tokens`, serialize response, reply
- Reuses the same model/engine as the inference queryable — no additional model loading

### Edge Node

- Declares `speculative` queryable when draft model is loaded AND a target node is discovered
- When a speculative query arrives: run the DSD loop (draft → verify → append → repeat)
- The verify step sends a Zenoh query to the target's `verify` queryable and waits for the reply

### Capacity Discovery

The edge subscribes to `harmony/compute/capacity/*`. When a capacity advertisement arrives with a model CID different from the edge's own, it records this as a potential target. The edge declares the `speculative` queryable only after discovering at least one target.

## Testing Strategy

### Unit tests (no real model, in `harmony-speculative`)

- Verify request/response serialization roundtrip
- `should_accept_draft` with synthetic logits: all accepted case, bonus sampled after last draft
- `verify_draft_tokens` with synthetic logits: first token rejected (accepted=0, bonus from position 0)
- `verify_draft_tokens` with synthetic logits: partial acceptance (accepted=3 out of 5)
- Edge cases: empty drafts, single draft token
- Logprob computation: softmax + ln roundtrip

### Integration tests (`#[ignore]`, real models)

- Load Qwen3 0.8B (draft) and Qwen3 9B (target), run DSD loop on a prompt
- Verify output is non-empty, valid UTF-8, and coherent
- Compare accepted token rate across rounds

## Scope Exclusions

- **Adaptive gamma** — fixed γ=5 (tracked as harmony-zee)
- **Stochastic rejection sampling** — greedy verification only
- **KV cache reuse across DSD rounds** — reset + re-prefill each round
- **Adaptive threshold by semantic importance** — paper's τ mechanism
- **Multi-target load balancing** — single target node
- **GPU/Metal acceleration** — CPU only
- **Streaming output** — single reply when DSD loop completes
- **Smart target selection** — uses first discovered target

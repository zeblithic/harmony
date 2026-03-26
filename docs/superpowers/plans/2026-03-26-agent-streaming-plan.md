# Agent Streaming Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pub/sub streaming to the agent protocol so long-running tasks can publish partial results (e.g. token-by-token inference) via `StreamChunk` messages.

**Architecture:** New `StreamChunk` type and wire encode/decode in `harmony-agent`. New `stream_key()` and `stream_sub()` namespace builders in `harmony-zenoh`. No changes to existing types or NodeRuntime.

**Tech Stack:** harmony-agent, harmony-zenoh

**Spec:** `docs/superpowers/specs/2026-03-26-agent-streaming-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-agent/src/types.rs` | Add `StreamChunk` struct |
| `crates/harmony-agent/src/wire.rs` | Add `encode_chunk`, `decode_chunk` |
| `crates/harmony-agent/src/lib.rs` | Re-export new type and functions |
| `crates/harmony-zenoh/src/namespace.rs` | Add `stream_key()`, `stream_sub()` builders |

---

### Task 1: StreamChunk Type

Add the `StreamChunk` struct to `harmony-agent` with serde support and unit tests.

**Files:**
- Modify: `crates/harmony-agent/src/types.rs`
- Modify: `crates/harmony-agent/src/lib.rs`

- [ ] **Step 1: Add StreamChunk struct to types.rs**

Add after the `AgentStatus` enum (around line 67), before the `#[cfg(test)]` block:

```rust
/// A single chunk of streaming output from a long-running task.
///
/// Published to `harmony/agent/{agent_id}/stream/{task_id}` via Zenoh pub/sub.
/// The `final_chunk` payload is advisory — the query-reply `AgentResult` is
/// the authoritative source of truth for task completion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    /// ID of the task this chunk belongs to.
    pub task_id: String,
    /// Monotonically increasing sequence number (0-indexed).
    pub sequence: u32,
    /// Task-type-specific payload (e.g. {"token": "hello"} for inference).
    pub payload: serde_json::Value,
    /// True if this is the last chunk in the stream.
    pub final_chunk: bool,
}
```

- [ ] **Step 2: Add re-export in lib.rs**

Update the `pub use types::` line to include `StreamChunk`:

```rust
pub use types::{
    AgentCapacity, AgentResult, AgentStatus, AgentTask, StreamChunk, TaskContext, TaskStatus,
};
```

- [ ] **Step 3: Add unit tests in types.rs**

Add these tests to the existing `mod tests` block in `types.rs`:

```rust
#[test]
fn stream_chunk_round_trip() {
    let chunk = StreamChunk {
        task_id: "task-001".to_string(),
        sequence: 0,
        payload: json!({"token": "hello"}),
        final_chunk: false,
    };
    let json = serde_json::to_string(&chunk).unwrap();
    let decoded: StreamChunk = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.task_id, "task-001");
    assert_eq!(decoded.sequence, 0);
    assert_eq!(decoded.payload, json!({"token": "hello"}));
    assert!(!decoded.final_chunk);
}

#[test]
fn stream_chunk_final_round_trip() {
    let chunk = StreamChunk {
        task_id: "task-002".to_string(),
        sequence: 42,
        payload: json!({"done": true}),
        final_chunk: true,
    };
    let json = serde_json::to_string(&chunk).unwrap();
    let decoded: StreamChunk = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded.task_id, "task-002");
    assert_eq!(decoded.sequence, 42);
    assert!(decoded.final_chunk);
}

#[test]
fn stream_chunk_snake_case_keys() {
    let chunk = StreamChunk {
        task_id: "task-sc".to_string(),
        sequence: 0,
        payload: json!({}),
        final_chunk: true,
    };
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(json.contains("\"task_id\""), "expected snake_case key task_id, got: {json}");
    assert!(json.contains("\"final_chunk\""), "expected snake_case key final_chunk, got: {json}");
    assert!(!json.contains("\"taskId\""), "unexpected camelCase in: {json}");
    assert!(!json.contains("\"finalChunk\""), "unexpected camelCase in: {json}");
}
```

- [ ] **Step 4: Verify tests pass**

Run: `cargo test -p harmony-agent`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-agent/src/types.rs crates/harmony-agent/src/lib.rs
git commit -m "feat(agent): add StreamChunk type for pub/sub streaming"
```

---

### Task 2: Wire Encode/Decode for StreamChunk

Add `encode_chunk` and `decode_chunk` functions following the existing wire format pattern.

**Files:**
- Modify: `crates/harmony-agent/src/wire.rs`
- Modify: `crates/harmony-agent/src/lib.rs`

- [ ] **Step 1: Add encode_chunk and decode_chunk to wire.rs**

Add after `decode_capacity` (around line 91), before the `#[cfg(test)]` block:

```rust
/// Encode a StreamChunk to wire format: [JSON_TAG][json_bytes].
pub fn encode_chunk(chunk: &StreamChunk) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(chunk)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode a StreamChunk from wire format.
///
/// Returns `serde_json::Error` wrapped in `AgentError::Json` for deserialization
/// failures, and `AgentError::EmptyPayload` or `AgentError::UnsupportedFormat`
/// for wire-level issues.
pub fn decode_chunk(payload: &[u8]) -> Result<StreamChunk, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}
```

Also add the import at the top of wire.rs — update the existing `use` statement:

```rust
use crate::types::{AgentCapacity, AgentResult, AgentTask, StreamChunk};
```

- [ ] **Step 2: Add re-exports in lib.rs**

Update the `pub use wire::` line to include the new functions:

```rust
pub use wire::{
    decode_capacity, decode_chunk, decode_result, decode_task, encode_capacity, encode_chunk,
    encode_result, encode_task, AgentError,
};
```

- [ ] **Step 3: Add wire tests**

Add these tests to the existing `mod tests` block in `wire.rs`:

```rust
#[test]
fn chunk_wire_round_trip() {
    let chunk = StreamChunk {
        task_id: "task-wire".to_string(),
        sequence: 7,
        payload: serde_json::json!({"token": "world"}),
        final_chunk: false,
    };
    let encoded = encode_chunk(&chunk).unwrap();
    assert_eq!(encoded[0], JSON_TAG);
    let decoded = decode_chunk(&encoded).unwrap();
    assert_eq!(decoded.task_id, "task-wire");
    assert_eq!(decoded.sequence, 7);
    assert_eq!(decoded.payload, serde_json::json!({"token": "world"}));
    assert!(!decoded.final_chunk);
}

#[test]
fn chunk_wire_final_round_trip() {
    let chunk = StreamChunk {
        task_id: "task-final".to_string(),
        sequence: 99,
        payload: serde_json::json!({"summary": "done"}),
        final_chunk: true,
    };
    let encoded = encode_chunk(&chunk).unwrap();
    let decoded = decode_chunk(&encoded).unwrap();
    assert_eq!(decoded.task_id, "task-final");
    assert_eq!(decoded.sequence, 99);
    assert!(decoded.final_chunk);
}

#[test]
fn chunk_decode_empty_payload() {
    assert!(matches!(decode_chunk(&[]), Err(AgentError::EmptyPayload)));
}

#[test]
fn chunk_decode_unknown_tag() {
    assert!(matches!(
        decode_chunk(&[0xFF, b'{', b'}']),
        Err(AgentError::UnsupportedFormat(0xFF))
    ));
}

#[test]
fn chunk_decode_invalid_json() {
    assert!(matches!(
        decode_chunk(&[JSON_TAG, b'n', b'o', b't']),
        Err(AgentError::Json(_))
    ));
}
```

- [ ] **Step 4: Verify tests pass**

Run: `cargo test -p harmony-agent`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-agent/src/wire.rs crates/harmony-agent/src/lib.rs
git commit -m "feat(agent): add encode_chunk/decode_chunk wire format for streaming"
```

---

### Task 3: Namespace Builders + Follow-Up Bead

Add `stream_key()` and `stream_sub()` to the agent namespace module, with tests. Also file the follow-on bead for NodeRuntime integration.

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

- [ ] **Step 1: Add stream_key and stream_sub builders**

In `crates/harmony-zenoh/src/namespace.rs`, in the `agent` module (around line 707, after `task_key`), add:

```rust
/// Per-task stream key: `harmony/agent/{agent_id}/stream/{task_id}`
///
/// Agents publish `StreamChunk` messages here for long-running tasks.
/// Requesters subscribe before sending the task query.
pub fn stream_key(agent_id: &str, task_id: &str) -> String {
    format!("{PREFIX}/{agent_id}/stream/{task_id}")
}

/// Subscribe to all streams from one agent: `harmony/agent/{agent_id}/stream/*`
///
/// Agent-scoped subscription — takes `agent_id` as parameter (unlike
/// `capacity_sub_all()` which is cross-agent). Stream subscriptions are
/// always scoped to a specific agent.
pub fn stream_sub(agent_id: &str) -> String {
    format!("{PREFIX}/{agent_id}/stream/*")
}
```

- [ ] **Step 2: Add namespace tests**

In the existing `agent_namespace_keys` test (around line 1329), add after the `capacity_sub_all` assertion:

```rust
let stream = agent::stream_key("deadbeef01020304", "task-abc");
assert_eq!(stream, "harmony/agent/deadbeef01020304/stream/task-abc");

let stream_sub = agent::stream_sub("deadbeef01020304");
assert_eq!(stream_sub, "harmony/agent/deadbeef01020304/stream/*");
```

- [ ] **Step 3: Verify tests pass**

Run: `cargo test -p harmony-zenoh`

- [ ] **Step 4: File follow-on bead for NodeRuntime integration**

```bash
bd create --title="Wire agent streaming into NodeRuntime inference path" --description="Integrate the StreamChunk protocol (harmony-q73) into the inference queryable and DSD edge orchestrator. Agents publish token-by-token StreamChunk messages to harmony/agent/{agent_id}/stream/{task_id} as inference runs. Depends on harmony-q73 (streaming protocol types)." --type=feature --priority=3
```

Then add dependency:
```bash
bd dep add <new_bead_id> harmony-q73
```

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add stream_key and stream_sub namespace builders for agent streaming"
```

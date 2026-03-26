# harmony-agent Protocol Crate

## Overview

A new `harmony-agent` crate defining structured agent-to-agent messaging. Agents are identified by their Harmony address hash, advertise capabilities via Zenoh liveliness + capacity publications, accept tasks via Zenoh query-reply, and return typed JSON results.

**Goal:** Formalize agent-to-agent communication with typed message envelopes, a Zenoh namespace for discovery and task routing, and capacity advertisement — the protocol foundation that delegation strategies and orchestration build on.

**Scope:** Protocol types and namespace only. No NodeRuntime integration, no delegation logic, no streaming results.

## Message Types

### AgentTask

Submitted by a requestor to an agent's task endpoint via Zenoh query.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct AgentTask {
    /// Unique task identifier. BLAKE3 hex of (submitter_addr + params + nonce).
    pub task_id: String,
    /// Semantic task type. Agents match this against their declared capabilities.
    /// Examples: "inference", "summarize", "code_exec", "translate".
    pub task_type: String,
    /// Task-type-specific parameters. Structure depends on task_type.
    pub params: serde_json::Value,
    /// Optional chaining context for multi-step workflows.
    pub context: Option<TaskContext>,
}
```

### TaskContext

Optional metadata for task chaining and caller context.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct TaskContext {
    /// Links to the parent task in a multi-step chain.
    pub parent_task_id: Option<String>,
    /// Opaque caller metadata forwarded through the chain.
    pub metadata: Option<serde_json::Value>,
}
```

### AgentResult

Returned by the agent as the Zenoh query reply.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct AgentResult {
    /// Echoes the task_id from the request.
    pub task_id: String,
    /// Outcome status.
    pub status: TaskStatus,
    /// Present on Success — task-type-specific output.
    pub output: Option<serde_json::Value>,
    /// Present on Failed — human-readable error description.
    pub error: Option<String>,
}
```

### TaskStatus

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]

pub enum TaskStatus {
    /// Task completed successfully. `output` is present.
    Success,
    /// Task execution failed. `error` is present.
    Failed,
    /// Agent cannot handle this task type. Requestor should try another agent.
    Rejected,
}
```

### Wire Format Example

```json
{
  "task_id": "a1b2c3d4e5f6...",
  "task_type": "inference",
  "params": {
    "prompt": "Explain quantum computing in simple terms",
    "max_tokens": 256,
    "temperature": 0.7
  },
  "context": {
    "parent_task_id": "f6e5d4c3b2a1...",
    "metadata": { "chain": "research-pipeline" }
  }
}
```

Result:

```json
{
  "task_id": "a1b2c3d4e5f6...",
  "status": "success",
  "output": {
    "text": "Quantum computing uses quantum bits...",
    "tokens_generated": 142
  },
  "error": null
}
```

## Zenoh Namespace

All agent communication uses key expressions under `harmony/agent/`:

```
harmony/agent/{agent_id}/capacity   — Capacity advertisement (publish, periodic)
harmony/agent/{agent_id}/task       — Task submission endpoint (queryable)
harmony/agent/*/capacity            — Discovery subscription (wildcard)
```

Where `agent_id` is the hex-encoded 16-byte address hash of the agent's Harmony identity.

### Namespace Constants

```rust
pub mod agent {
    pub const PREFIX: &str = "harmony/agent";

    /// Capacity advertisement for a specific agent.
    pub fn capacity_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/capacity")
    }

    /// Task submission endpoint for a specific agent.
    pub fn task_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/task")
    }

    /// Subscribe to all agent capacity advertisements.
    pub fn capacity_sub_all() -> String {
        format!("{PREFIX}/*/capacity")
    }
}
```

## Capacity Advertisement

Agents publish their capabilities so requestors can discover them.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]

pub struct AgentCapacity {
    /// Hex-encoded address hash identifying this agent.
    pub agent_id: String,
    /// Task types this agent can handle.
    pub task_types: Vec<String>,
    /// Current availability status.
    pub status: AgentStatus,
    /// Maximum concurrent tasks this agent will accept.
    pub max_concurrent: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]

pub enum AgentStatus {
    /// Accepting new tasks.
    Ready,
    /// At capacity — existing tasks running, new tasks may be queued or rejected.
    Busy,
    /// Shutting down — no new tasks, finishing current work.
    Draining,
}
```

Published periodically (e.g. every 30s) and on status change to `harmony/agent/{agent_id}/capacity`. Requestors subscribe to `harmony/agent/*/capacity` to build a local view of available agents.

## Wire Envelope

Task and result payloads are prefixed with a single tag byte for forward compatibility:

| Tag | Format |
|-----|--------|
| `0x00` | JSON (this version) |
| `0x01`–`0xFF` | Reserved for future binary formats |

Serialization: `[tag: u8][json_bytes]`. The tag allows future versions to introduce binary encodings (e.g. MessagePack, Protobuf) without breaking existing agents.

Helper functions:

```rust
/// Serialize an AgentTask to wire format.
pub fn encode_task(task: &AgentTask) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(task)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(0x00); // JSON tag
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Deserialize an AgentTask from wire format.
pub fn decode_task(payload: &[u8]) -> Result<AgentTask, AgentError> {
    if payload.is_empty() {
        return Err(AgentError::EmptyPayload);
    }
    if payload[0] != 0x00 {
        return Err(AgentError::UnsupportedFormat(payload[0]));
    }
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

// Same pattern for encode_result/decode_result and encode_capacity/decode_capacity.
```

## Error Types

```rust
#[derive(Debug)]
pub enum AgentError {
    /// Payload was empty (no tag byte).
    EmptyPayload,
    /// Unknown wire format tag.
    UnsupportedFormat(u8),
    /// JSON deserialization failed.
    Json(serde_json::Error),
}
```

## Task ID Generation

```rust
/// Generate a deterministic task ID from submitter identity, params, and nonce.
pub fn generate_task_id(
    submitter_addr: &[u8; 16],
    params: &serde_json::Value,
    nonce: u64,
) -> String {
    let params_bytes = serde_json::to_vec(params).unwrap_or_default();
    let mut input = Vec::with_capacity(16 + params_bytes.len() + 8);
    input.extend_from_slice(submitter_addr);
    input.extend_from_slice(&params_bytes);
    input.extend_from_slice(&nonce.to_le_bytes());
    hex::encode(harmony_crypto::hash::blake3_hash(&input))
}
```

The nonce ensures unique task IDs even for identical params (same prompt submitted twice gets different IDs, unlike WorkflowEngine dedup which is intentional for WASM modules).

## Testing Strategy

### Unit Tests

1. **AgentTask round-trip:** Serialize → deserialize, verify all fields preserved.
2. **AgentResult round-trip:** Success, Failed, and Rejected variants.
3. **AgentCapacity round-trip:** With multiple task types, all status variants.
4. **snake_case verification:** JSON keys use snake_case (taskId, taskType, etc.).
5. **Wire encode/decode:** Tag byte present, JSON body correct.
6. **Empty payload error:** decode_task on empty slice returns EmptyPayload.
7. **Unknown tag error:** decode_task with tag 0xFF returns UnsupportedFormat.
8. **TaskContext optional:** Task with and without context both serialize correctly.
9. **Task ID generation:** Same inputs + same nonce = same ID. Different nonce = different ID.
10. **Namespace constants:** Verify key expressions are well-formed strings.

## Files

| File | Responsibility |
|------|---------------|
| `crates/harmony-agent/Cargo.toml` | Crate manifest — depends on serde, serde_json, harmony-crypto, hex |
| `crates/harmony-agent/src/lib.rs` | Re-exports |
| `crates/harmony-agent/src/types.rs` | AgentTask, AgentResult, TaskStatus, TaskContext, AgentCapacity, AgentStatus |
| `crates/harmony-agent/src/wire.rs` | encode/decode functions, AgentError |
| `crates/harmony-agent/src/task_id.rs` | generate_task_id |
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod agent` with namespace constants |

## Dependencies

- `serde` + `serde_json` — serialization (already in workspace)
- `harmony-crypto` — BLAKE3 for task ID generation (workspace member)
- `hex` — hex encoding for task IDs and agent IDs (already in workspace)

No new external dependencies.

## What This Does NOT Include

- **NodeRuntime integration** — wiring agents into the event loop (future bead)
- **Task delegation** — strategies for choosing which agent handles a task (future bead)
- **Streaming results** — pub/sub for partial/token-level output (future bead)
- **UCAN authorization** — capability-based access control on tasks (future bead)
- **Global Workspace Theory** — orchestration patterns (future bead)

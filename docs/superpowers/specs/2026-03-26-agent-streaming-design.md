# Agent Streaming Protocol

## Goal

Add pub/sub streaming to the agent protocol so long-running tasks (primarily inference) can publish partial results as they're produced. The query-reply path continues to deliver the final result — streaming is additive and optional.

## Architecture

The protocol adds one new type (`StreamChunk`) and corresponding wire encode/decode functions to `harmony-agent`, plus namespace builders in `harmony-zenoh`. No existing types or behavior change.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-agent` | `StreamChunk` type, wire encode/decode |
| `harmony-zenoh` | `stream_key()`, `stream_sub()` namespace builders |

### Protocol Flow

1. Requester generates a `task_id` (via existing `generate_task_id()`)
2. Requester subscribes to `harmony/agent/{agent_id}/stream/{task_id}`
3. Requester sends `AgentTask` via Zenoh query to `harmony/agent/{agent_id}/task`
4. Agent publishes `StreamChunk` messages to that stream key as work progresses
5. Agent sends final `AgentResult` via query reply (unchanged from current protocol)

Non-streaming consumers skip steps 2 and 4 — the query-reply path works exactly as before.

### Key Expressions

| Key Expression | Purpose |
|---------------|---------|
| `harmony/agent/{agent_id}/stream/{task_id}` | Per-task stream for chunk publication |
| `harmony/agent/{agent_id}/stream/*` | Subscribe to all streams from one agent |
| `harmony/agent/{agent_id}/task` | Existing — task submission queryable |
| `harmony/agent/{agent_id}/capacity` | Existing — capacity advertisement |

No cross-agent stream subscription (`harmony/agent/*/stream/**`) is provided. Stream subscriptions are always agent-scoped — the requester knows which agent it submitted a task to. Cross-agent stream discovery is not a use case for this protocol.

## StreamChunk Type

```rust
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

- `task_id` — matches the `AgentTask.task_id` that initiated the work. Allows a subscriber watching `stream/*` to demux multiple concurrent tasks.
- `sequence` — starts at 0, increments by 1 per chunk. Subscribers can detect gaps (missed chunks) by checking continuity. No retry or reorder mechanism — gaps are informational only.
- `payload` — opaque to the protocol. For inference streaming, the convention is `{"token": "text"}` or `{"token_id": 42}`. Other task types define their own payload shapes.
- `final_chunk` — signals stream completion. The `final_chunk` payload is advisory — consumers MUST NOT treat it as the authoritative result. The query-reply `AgentResult` is the source of truth for task completion and output. The `final_chunk` flag exists solely so stream-only subscribers know when to stop listening.

## Wire Format

Follows the existing agent wire convention:

```
[0x00 (JSON_TAG)] [json_bytes]
```

New functions:
- `encode_chunk(chunk: &StreamChunk) -> Result<Vec<u8>, serde_json::Error>` — encode returns `serde_json::Error` directly
- `decode_chunk(payload: &[u8]) -> Result<StreamChunk, AgentError>` — decode returns `AgentError` (tag check + JSON)

This asymmetry (different error types for encode vs. decode) matches the existing pattern in `encode_task`/`decode_task`.

## Namespace Builders

Two new functions in the `agent` module of `harmony-zenoh/src/namespace.rs`:

```rust
/// Per-task stream key: `harmony/agent/{agent_id}/stream/{task_id}`
pub fn stream_key(agent_id: &str, task_id: &str) -> String {
    format!("{PREFIX}/{agent_id}/stream/{task_id}")
}

/// Subscribe to all streams from one agent: `harmony/agent/{agent_id}/stream/*`
pub fn stream_sub(agent_id: &str) -> String {
    format!("{PREFIX}/{agent_id}/stream/*")
}
```

These are agent-scoped builders (take `agent_id` as parameter), following the pattern of `task_key()` and `capacity_key()`. They differ from `capacity_sub_all()` which is a cross-agent parameter-free glob — that pattern is intentionally not used here since stream subscriptions are always agent-scoped.

No separate `STREAM` prefix constant is needed — the builders construct the key inline from `PREFIX`, matching how `task_key()` and `capacity_key()` already work.

## Testing Strategy

### Unit tests in `harmony-agent`

- `StreamChunk` serde roundtrip (basic fields, verify `task_id` preserved)
- `StreamChunk` with `final_chunk: true` roundtrip
- Wire encode/decode roundtrip (assert `task_id` field matches after decode)
- Wire decode error cases: empty payload, wrong tag, invalid JSON
- Sequence number verified 0-indexed in roundtrip

### Namespace tests in `harmony-zenoh`

- `stream_key("agent_id", "task_id")` produces `harmony/agent/agent_id/stream/task_id`
- `stream_sub("agent_id")` produces `harmony/agent/agent_id/stream/*`

## Scope Exclusions

- **No NodeRuntime integration** — streaming is protocol-layer only. Wiring into the inference path is a follow-on bead.
- **No cross-agent stream subscription** — stream subscriptions are agent-scoped. An orchestrator watching all agents uses capacity advertisements to discover agents, then subscribes per-agent.
- **No backpressure** — Zenoh pub/sub is fire-and-forget. Subscribers that fall behind miss chunks; sequence numbers let them detect this.
- **No chunk aggregation/reassembly** — consumers process chunks individually.
- **No binary wire format** — JSON only (tag 0x00), matching the existing agent protocol. Binary optimization is future work if profiling shows JSON overhead matters.
- **No retry/reorder** — sequence gaps are detected but not recovered. The query-reply final result is the source of truth.

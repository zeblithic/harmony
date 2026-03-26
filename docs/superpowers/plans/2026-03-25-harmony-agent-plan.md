# harmony-agent Protocol Crate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the `harmony-agent` crate with typed message envelopes (AgentTask, AgentResult, AgentCapacity), wire encode/decode, task ID generation, and Zenoh namespace constants for agent discovery and task routing.

**Architecture:** New crate with three focused modules: `types.rs` (message structs + serde), `wire.rs` (tag-prefixed encode/decode + error type), `task_id.rs` (BLAKE3-based ID generation). Zenoh namespace constants added to the existing `harmony-zenoh` crate's `namespace.rs`.

**Tech Stack:** Rust, serde + serde_json, harmony-crypto (BLAKE3), hex

**Spec:** `docs/superpowers/specs/2026-03-25-harmony-agent-design.md`

**Test command:** `cargo test -p harmony-agent`
**Lint command:** `cargo clippy -p harmony-agent`
**Workspace test:** `cargo test --workspace`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `Cargo.toml` (workspace root) | Register new crate | Modify |
| `crates/harmony-agent/Cargo.toml` | Crate manifest | Create |
| `crates/harmony-agent/src/lib.rs` | Module declarations + re-exports | Create |
| `crates/harmony-agent/src/types.rs` | AgentTask, AgentResult, TaskStatus, TaskContext, AgentCapacity, AgentStatus | Create |
| `crates/harmony-agent/src/wire.rs` | encode/decode functions, AgentError | Create |
| `crates/harmony-agent/src/task_id.rs` | generate_task_id | Create |
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod agent` namespace constants | Modify |

---

### Task 1: Crate scaffold and message types

Create the new crate with Cargo.toml, lib.rs, and types.rs containing all message structs with serde.

**Files:**
- Create: `crates/harmony-agent/Cargo.toml`
- Create: `crates/harmony-agent/src/lib.rs`
- Create: `crates/harmony-agent/src/types.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-agent/src/types.rs` with test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn agent_task_round_trip() {
        let task = AgentTask {
            task_id: "abc123".to_string(),
            task_type: "inference".to_string(),
            params: serde_json::json!({
                "prompt": "Hello",
                "max_tokens": 256
            }),
            context: Some(TaskContext {
                parent_task_id: Some("parent_001".to_string()),
                metadata: Some(serde_json::json!({"chain": "research"})),
            }),
        };
        let json = serde_json::to_string(&task).unwrap();
        let loaded: AgentTask = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.task_id, "abc123");
        assert_eq!(loaded.task_type, "inference");
        assert_eq!(loaded.params["prompt"], "Hello");
        assert_eq!(
            loaded.context.as_ref().unwrap().parent_task_id.as_deref(),
            Some("parent_001")
        );
    }

    #[test]
    fn agent_task_without_context() {
        let task = AgentTask {
            task_id: "abc123".to_string(),
            task_type: "summarize".to_string(),
            params: serde_json::json!({"text": "Long document..."}),
            context: None,
        };
        let json = serde_json::to_string(&task).unwrap();
        let loaded: AgentTask = serde_json::from_str(&json).unwrap();
        assert!(loaded.context.is_none());
    }

    #[test]
    fn agent_result_success_round_trip() {
        let result = AgentResult {
            task_id: "abc123".to_string(),
            status: TaskStatus::Success,
            output: Some(serde_json::json!({"text": "Generated output"})),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        let loaded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.status, TaskStatus::Success);
        assert!(loaded.output.is_some());
        assert!(loaded.error.is_none());
    }

    #[test]
    fn agent_result_failed_round_trip() {
        let result = AgentResult {
            task_id: "abc123".to_string(),
            status: TaskStatus::Failed,
            output: None,
            error: Some("out of memory".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let loaded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.status, TaskStatus::Failed);
        assert_eq!(loaded.error.as_deref(), Some("out of memory"));
    }

    #[test]
    fn agent_result_rejected() {
        let result = AgentResult {
            task_id: "abc123".to_string(),
            status: TaskStatus::Rejected,
            output: None,
            error: Some("unsupported task type".to_string()),
        };
        let json = serde_json::to_string(&result).unwrap();
        let loaded: AgentResult = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.status, TaskStatus::Rejected);
    }

    #[test]
    fn agent_capacity_round_trip() {
        let cap = AgentCapacity {
            agent_id: "deadbeef".to_string(),
            task_types: vec!["inference".to_string(), "summarize".to_string()],
            status: AgentStatus::Ready,
            max_concurrent: 4,
        };
        let json = serde_json::to_string(&cap).unwrap();
        let loaded: AgentCapacity = serde_json::from_str(&json).unwrap();
        assert_eq!(loaded.task_types.len(), 2);
        assert_eq!(loaded.status, AgentStatus::Ready);
        assert_eq!(loaded.max_concurrent, 4);
    }

    #[test]
    fn agent_capacity_all_statuses() {
        for status in [AgentStatus::Ready, AgentStatus::Busy, AgentStatus::Draining] {
            let cap = AgentCapacity {
                agent_id: "test".to_string(),
                task_types: vec![],
                status,
                max_concurrent: 1,
            };
            let json = serde_json::to_string(&cap).unwrap();
            let loaded: AgentCapacity = serde_json::from_str(&json).unwrap();
            assert_eq!(loaded.status, status);
        }
    }

    #[test]
    fn snake_case_json_keys() {
        let task = AgentTask {
            task_id: "t1".to_string(),
            task_type: "test".to_string(),
            params: serde_json::json!({}),
            context: None,
        };
        let json = serde_json::to_string(&task).unwrap();
        assert!(json.contains("\"task_id\""), "Expected snake_case: {json}");
        assert!(json.contains("\"task_type\""), "Expected snake_case: {json}");
        assert!(!json.contains("\"taskId\""), "Should not be camelCase: {json}");
    }
}
```

- [ ] **Step 2: Create Cargo.toml for the new crate**

Create `crates/harmony-agent/Cargo.toml`:

```toml
[package]
name = "harmony-agent"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Structured agent-to-agent messaging protocol for the Harmony decentralized stack"

[dependencies]
serde = { workspace = true, features = ["derive"] }
serde_json.workspace = true
harmony-crypto = { workspace = true, features = ["std"] }
hex.workspace = true
```

- [ ] **Step 3: Register in workspace Cargo.toml**

Add to the `[workspace] members` list:

```toml
"crates/harmony-agent",
```

Add to `[workspace.dependencies]`:

```toml
harmony-agent = { path = "crates/harmony-agent" }
```

- [ ] **Step 4: Create lib.rs**

Create `crates/harmony-agent/src/lib.rs`:

```rust
//! Structured agent-to-agent messaging protocol.
//!
//! Defines typed message envelopes for task submission, result delivery,
//! and capacity advertisement. Agents are identified by Harmony address
//! hash and communicate via Zenoh query-reply.

pub mod task_id;
pub mod types;
pub mod wire;

pub use types::{
    AgentCapacity, AgentResult, AgentStatus, AgentTask, TaskContext, TaskStatus,
};
pub use wire::{
    decode_capacity, decode_result, decode_task, encode_capacity, encode_result, encode_task,
    AgentError,
};
pub use task_id::generate_task_id;
```

Note: `wire.rs` and `task_id.rs` will be created in later tasks. For now, comment out the `pub mod wire;`, `pub mod task_id;`, and their `pub use` lines so the crate compiles with just types.rs.

- [ ] **Step 5: Implement types.rs**

Create `crates/harmony-agent/src/types.rs` with the structs (before the test module):

```rust
//! Agent protocol message types.

use serde::{Deserialize, Serialize};

/// A task submitted to an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentTask {
    /// Unique task identifier.
    pub task_id: String,
    /// Semantic task type (e.g. "inference", "summarize").
    pub task_type: String,
    /// Task-type-specific parameters.
    pub params: serde_json::Value,
    /// Optional chaining context for multi-step workflows.
    pub context: Option<TaskContext>,
}

/// Optional context for task chaining.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskContext {
    /// Links to the parent task in a multi-step chain.
    pub parent_task_id: Option<String>,
    /// Opaque caller metadata forwarded through the chain.
    pub metadata: Option<serde_json::Value>,
}

/// Result returned by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// Echoes the task_id from the request.
    pub task_id: String,
    /// Outcome status.
    pub status: TaskStatus,
    /// Present on Success — task-type-specific output.
    pub output: Option<serde_json::Value>,
    /// Present on Failed/Rejected — human-readable error.
    pub error: Option<String>,
}

/// Task outcome status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TaskStatus {
    /// Completed successfully.
    Success,
    /// Execution failed.
    Failed,
    /// Agent cannot handle this task type.
    Rejected,
}

/// Capacity advertisement published by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentCapacity {
    /// Hex-encoded address hash identifying this agent.
    pub agent_id: String,
    /// Task types this agent can handle.
    pub task_types: Vec<String>,
    /// Current availability.
    pub status: AgentStatus,
    /// Maximum concurrent tasks.
    pub max_concurrent: u32,
}

/// Agent availability status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AgentStatus {
    /// Accepting new tasks.
    Ready,
    /// At capacity.
    Busy,
    /// No new tasks, finishing current work.
    Draining,
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-agent -- --nocapture`
Expected: ALL PASS (8 tests)

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p harmony-agent 2>&1`
Expected: No errors

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-agent/ Cargo.toml Cargo.lock
git commit -m "feat(agent): add harmony-agent crate with typed message envelopes"
```

---

### Task 2: Wire encode/decode with tag byte

Add `wire.rs` with tag-prefixed serialization for all three message types, plus `AgentError`.

**Files:**
- Create: `crates/harmony-agent/src/wire.rs`
- Modify: `crates/harmony-agent/src/lib.rs` (uncomment wire module)

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-agent/src/wire.rs` with test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;

    #[test]
    fn task_wire_round_trip() {
        let task = AgentTask {
            task_id: "t1".to_string(),
            task_type: "inference".to_string(),
            params: serde_json::json!({"prompt": "Hello"}),
            context: None,
        };
        let encoded = encode_task(&task).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_task(&encoded).unwrap();
        assert_eq!(decoded.task_id, "t1");
    }

    #[test]
    fn result_wire_round_trip() {
        let result = AgentResult {
            task_id: "t1".to_string(),
            status: TaskStatus::Success,
            output: Some(serde_json::json!({"text": "output"})),
            error: None,
        };
        let encoded = encode_result(&result).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_result(&encoded).unwrap();
        assert_eq!(decoded.status, TaskStatus::Success);
    }

    #[test]
    fn capacity_wire_round_trip() {
        let cap = AgentCapacity {
            agent_id: "deadbeef".to_string(),
            task_types: vec!["inference".to_string()],
            status: AgentStatus::Ready,
            max_concurrent: 2,
        };
        let encoded = encode_capacity(&cap).unwrap();
        assert_eq!(encoded[0], JSON_TAG);
        let decoded = decode_capacity(&encoded).unwrap();
        assert_eq!(decoded.agent_id, "deadbeef");
    }

    #[test]
    fn decode_empty_payload_returns_error() {
        let result = decode_task(&[]);
        assert!(matches!(result, Err(AgentError::EmptyPayload)));
    }

    #[test]
    fn decode_unknown_tag_returns_error() {
        let result = decode_task(&[0xFF, b'{', b'}']);
        assert!(matches!(result, Err(AgentError::UnsupportedFormat(0xFF))));
    }

    #[test]
    fn decode_invalid_json_returns_error() {
        let result = decode_task(&[JSON_TAG, b'n', b'o', b't', b'j', b's', b'o', b'n']);
        assert!(matches!(result, Err(AgentError::Json(_))));
    }
}
```

- [ ] **Step 2: Implement wire.rs**

Add above the test module:

```rust
//! Wire format encode/decode with tag-byte prefix.

use crate::types::{AgentCapacity, AgentResult, AgentTask};

/// Tag byte for JSON-encoded payloads.
pub const JSON_TAG: u8 = 0x00;

/// Errors from wire decode operations.
#[derive(Debug)]
pub enum AgentError {
    /// Payload was empty (no tag byte).
    EmptyPayload,
    /// Unknown wire format tag.
    UnsupportedFormat(u8),
    /// JSON deserialization failed.
    Json(serde_json::Error),
}

impl core::fmt::Display for AgentError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyPayload => write!(f, "empty payload"),
            Self::UnsupportedFormat(tag) => write!(f, "unsupported format tag: 0x{tag:02x}"),
            Self::Json(e) => write!(f, "json error: {e}"),
        }
    }
}

impl std::error::Error for AgentError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

/// Encode an AgentTask to wire format: [JSON_TAG][json_bytes].
pub fn encode_task(task: &AgentTask) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(task)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentTask from wire format.
pub fn decode_task(payload: &[u8]) -> Result<AgentTask, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

/// Encode an AgentResult to wire format.
pub fn encode_result(result: &AgentResult) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(result)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentResult from wire format.
pub fn decode_result(payload: &[u8]) -> Result<AgentResult, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

/// Encode an AgentCapacity to wire format.
pub fn encode_capacity(cap: &AgentCapacity) -> Result<Vec<u8>, serde_json::Error> {
    let json = serde_json::to_vec(cap)?;
    let mut payload = Vec::with_capacity(1 + json.len());
    payload.push(JSON_TAG);
    payload.extend_from_slice(&json);
    Ok(payload)
}

/// Decode an AgentCapacity from wire format.
pub fn decode_capacity(payload: &[u8]) -> Result<AgentCapacity, AgentError> {
    check_tag(payload)?;
    serde_json::from_slice(&payload[1..]).map_err(AgentError::Json)
}

fn check_tag(payload: &[u8]) -> Result<(), AgentError> {
    if payload.is_empty() {
        return Err(AgentError::EmptyPayload);
    }
    if payload[0] != JSON_TAG {
        return Err(AgentError::UnsupportedFormat(payload[0]));
    }
    Ok(())
}
```

- [ ] **Step 3: Uncomment wire module in lib.rs**

Uncomment `pub mod wire;` and the wire `pub use` lines in `crates/harmony-agent/src/lib.rs`.

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-agent -- --nocapture`
Expected: ALL PASS (14 tests)

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p harmony-agent 2>&1`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-agent/src/wire.rs crates/harmony-agent/src/lib.rs
git commit -m "feat(agent): add wire encode/decode with tag-byte prefix"
```

---

### Task 3: Task ID generation

Add `task_id.rs` with BLAKE3-based deterministic task ID generation.

**Files:**
- Create: `crates/harmony-agent/src/task_id.rs`
- Modify: `crates/harmony-agent/src/lib.rs` (uncomment task_id module)

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-agent/src/task_id.rs` with test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_inputs() {
        let addr = [0xAA; 16];
        let params = serde_json::json!({"prompt": "Hello"});
        let id1 = generate_task_id(&addr, &params, 42);
        let id2 = generate_task_id(&addr, &params, 42);
        assert_eq!(id1, id2);
    }

    #[test]
    fn different_nonce_different_id() {
        let addr = [0xAA; 16];
        let params = serde_json::json!({"prompt": "Hello"});
        let id1 = generate_task_id(&addr, &params, 1);
        let id2 = generate_task_id(&addr, &params, 2);
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_params_different_id() {
        let addr = [0xAA; 16];
        let p1 = serde_json::json!({"prompt": "Hello"});
        let p2 = serde_json::json!({"prompt": "World"});
        let id1 = generate_task_id(&addr, &p1, 0);
        let id2 = generate_task_id(&addr, &p2, 0);
        assert_ne!(id1, id2);
    }

    #[test]
    fn different_submitter_different_id() {
        let params = serde_json::json!({"prompt": "Hello"});
        let id1 = generate_task_id(&[0xAA; 16], &params, 0);
        let id2 = generate_task_id(&[0xBB; 16], &params, 0);
        assert_ne!(id1, id2);
    }

    #[test]
    fn id_is_hex_encoded() {
        let id = generate_task_id(&[0; 16], &serde_json::json!({}), 0);
        // BLAKE3 output is 32 bytes → 64 hex chars
        assert_eq!(id.len(), 64);
        assert!(id.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
```

- [ ] **Step 2: Implement task_id.rs**

Add above the test module:

```rust
//! Deterministic task ID generation via BLAKE3.

/// Generate a task ID from submitter address, params, and nonce.
///
/// `id = hex(BLAKE3(submitter_addr || json(params) || nonce_le))`
///
/// The nonce ensures unique IDs even for identical params, unlike
/// WorkflowEngine dedup which is intentional for WASM modules.
pub fn generate_task_id(
    submitter_addr: &[u8; 16],
    params: &serde_json::Value,
    nonce: u64,
) -> String {
    // serde_json::to_vec on a Value always succeeds (all Value
    // variants are serializable), so unwrap_or_default is safe.
    let params_bytes = serde_json::to_vec(params).unwrap_or_default();
    let mut input = Vec::with_capacity(16 + params_bytes.len() + 8);
    input.extend_from_slice(submitter_addr);
    input.extend_from_slice(&params_bytes);
    input.extend_from_slice(&nonce.to_le_bytes());
    hex::encode(harmony_crypto::hash::blake3_hash(&input))
}
```

- [ ] **Step 3: Uncomment task_id module in lib.rs**

Uncomment `pub mod task_id;` and `pub use task_id::generate_task_id;` in lib.rs.

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-agent -- --nocapture`
Expected: ALL PASS (19 tests)

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p harmony-agent 2>&1`
Expected: No errors

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-agent/src/task_id.rs crates/harmony-agent/src/lib.rs
git commit -m "feat(agent): add BLAKE3-based task ID generation"
```

---

### Task 4: Zenoh namespace constants

Add `pub mod agent` to `harmony-zenoh/src/namespace.rs` with key expression constants and builders.

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

- [ ] **Step 1: Write the failing test**

Add a test to the existing test module in `namespace.rs` (or create one if none exists). Check the bottom of the file for `#[cfg(test)]`:

```rust
#[test]
fn agent_namespace_keys() {
    let cap = agent::capacity_key("deadbeef01020304");
    assert_eq!(cap, "harmony/agent/deadbeef01020304/capacity");

    let task = agent::task_key("deadbeef01020304");
    assert_eq!(task, "harmony/agent/deadbeef01020304/task");

    let sub = agent::capacity_sub_all();
    assert_eq!(sub, "harmony/agent/*/capacity");
}
```

- [ ] **Step 2: Add the agent module**

Add to `crates/harmony-zenoh/src/namespace.rs`, after the `workflow` module (before any test module):

```rust
// ── Agent Protocol ──────────────────────────────────────────────────

/// Agent-to-agent messaging key expressions.
pub mod agent {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/agent`
    pub const PREFIX: &str = "harmony/agent";

    // ── Subscription patterns ───────────────────────────────────

    /// Subscribe to all agent capacity advertisements: `harmony/agent/*/capacity`
    pub fn capacity_sub_all() -> String {
        format!("{PREFIX}/*/capacity")
    }

    // ── Builders ────────────────────────────────────────────────

    /// Capacity key: `harmony/agent/{agent_id}/capacity`
    pub fn capacity_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/capacity")
    }

    /// Task submission endpoint: `harmony/agent/{agent_id}/task`
    pub fn task_key(agent_id: &str) -> String {
        format!("{PREFIX}/{agent_id}/task")
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-zenoh -- --nocapture`
Expected: ALL PASS

Also run the agent crate tests:
Run: `cargo test -p harmony-agent -- --nocapture`
Expected: ALL PASS

- [ ] **Step 4: Run clippy for both crates**

Run: `cargo clippy -p harmony-agent -p harmony-zenoh 2>&1`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add agent namespace constants for discovery and task routing"
```

---

### Task 5: Final integration — workspace test and cleanup

Run the full workspace test suite to ensure no regressions, verify the crate is properly integrated.

**Files:**
- No new files

- [ ] **Step 1: Run workspace tests**

Run: `cargo test --workspace 2>&1 | tail -20`
Expected: ALL PASS (may take a few minutes)

- [ ] **Step 2: Run workspace clippy**

Run: `cargo clippy --workspace 2>&1`
Expected: No errors from harmony-agent or harmony-zenoh

- [ ] **Step 3: Verify crate is importable**

Run: `cargo check -p harmony-agent 2>&1`
Expected: Clean

- [ ] **Step 4: Final commit (if any cleanup needed)**

```bash
git add -A
git commit -m "chore(agent): final cleanup after harmony-agent integration"
```

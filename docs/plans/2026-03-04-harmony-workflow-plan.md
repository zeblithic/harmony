# harmony-workflow Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `harmony-workflow` crate providing durable execution orchestration over the existing compute tier, with event-sourced history for crash recovery via deterministic WASM replay.

**Architecture:** `WorkflowEngine` wraps `ComputeTier` as a thin orchestration layer. It intercepts compute-tier IO actions, records them in an event log (`WorkflowHistory`), and re-emits them as workflow-level actions. On recovery, the engine replays the WASM module from scratch, feeding cached IO responses from the persisted history. The node runtime swaps `ComputeTier` for `WorkflowEngine`.

**Tech Stack:** Rust, harmony-compute (ComputeTier, ComputeRuntime), harmony-crypto (BLAKE3), wasmi

---

### Task 1: Scaffold harmony-workflow crate

**Files:**
- Create: `crates/harmony-workflow/Cargo.toml`
- Create: `crates/harmony-workflow/src/lib.rs`
- Create: `crates/harmony-workflow/src/error.rs`
- Create: `crates/harmony-workflow/src/types.rs`
- Create: `crates/harmony-workflow/src/offload.rs`
- Create: `crates/harmony-workflow/src/engine.rs`
- Modify: `Cargo.toml:1-54` (workspace members + dependency)

**Step 1: Add crate to workspace Cargo.toml**

In `Cargo.toml`, add `"crates/harmony-workflow"` to the `[workspace] members` list (after `harmony-compute`), and add `harmony-workflow = { path = "crates/harmony-workflow" }` to `[workspace.dependencies]`.

```toml
# In [workspace] members, add:
    "crates/harmony-workflow",

# In [workspace.dependencies], add:
harmony-workflow = { path = "crates/harmony-workflow" }
```

**Step 2: Create Cargo.toml for harmony-workflow**

```toml
[package]
name = "harmony-workflow"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Durable execution orchestration with event-sourced history for the Harmony decentralized stack"

[dependencies]
harmony-compute.workspace = true
harmony-crypto.workspace = true
thiserror.workspace = true
```

**Step 3: Create empty module stubs**

`crates/harmony-workflow/src/lib.rs`:
```rust
pub mod engine;
pub mod error;
pub mod offload;
pub mod types;
```

`crates/harmony-workflow/src/error.rs`:
```rust
// Workflow error types — implemented in Task 2.
```

`crates/harmony-workflow/src/types.rs`:
```rust
// Core workflow types — implemented in Task 3.
```

`crates/harmony-workflow/src/offload.rs`:
```rust
// Compute offload types — implemented in Task 3.
```

`crates/harmony-workflow/src/engine.rs`:
```rust
// WorkflowEngine state machine — implemented in Task 4.
```

**Step 4: Verify the crate compiles**

Run: `cargo check -p harmony-workflow`
Expected: success (no errors)

**Step 5: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-workflow/
git commit -m "feat(workflow): scaffold harmony-workflow crate with empty modules"
```

---

### Task 2: Error and core types with tests

**Files:**
- Modify: `crates/harmony-workflow/src/error.rs`
- Modify: `crates/harmony-workflow/src/types.rs`
- Modify: `crates/harmony-workflow/src/offload.rs`
- Modify: `crates/harmony-workflow/src/lib.rs`

**Context:** This task defines all the data types the rest of the crate uses. Pattern follows `crates/harmony-compute/src/error.rs` and `crates/harmony-compute/src/types.rs`.

**Step 1: Write WorkflowError**

`crates/harmony-workflow/src/error.rs`:
```rust
/// Errors from workflow operations.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("workflow not found: {id}")]
    NotFound { id: String },

    #[error("workflow already exists: {id}")]
    AlreadyExists { id: String },

    #[error("workflow not in expected state: expected {expected}, got {actual}")]
    InvalidState { expected: String, actual: String },

    #[error("compute error: {0}")]
    Compute(#[from] harmony_compute::ComputeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_not_found() {
        let err = WorkflowError::NotFound {
            id: "abc123".into(),
        };
        assert_eq!(err.to_string(), "workflow not found: abc123");
    }

    #[test]
    fn error_display_already_exists() {
        let err = WorkflowError::AlreadyExists {
            id: "abc123".into(),
        };
        assert_eq!(err.to_string(), "workflow already exists: abc123");
    }

    #[test]
    fn error_display_invalid_state() {
        let err = WorkflowError::InvalidState {
            expected: "Executing".into(),
            actual: "Complete".into(),
        };
        assert_eq!(
            err.to_string(),
            "workflow not in expected state: expected Executing, got Complete"
        );
    }
}
```

**Step 2: Write core types with WorkflowId, HistoryEvent, WorkflowHistory, WorkflowStatus**

`crates/harmony-workflow/src/types.rs`:
```rust
/// Content-addressed workflow identifier.
///
/// Computed as `BLAKE3(module_hash || input)`. Deterministic: same module +
/// same input always produces the same workflow ID, enabling deduplication
/// across nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkflowId([u8; 32]);

impl WorkflowId {
    /// Compute a workflow ID from the module hash and input bytes.
    pub fn new(module_hash: &[u8; 32], input: &[u8]) -> Self {
        let mut hasher = harmony_crypto::blake3::Hasher::new();
        hasher.update(module_hash);
        hasher.update(input);
        let hash = hasher.finalize();
        Self(*hash.as_bytes())
    }

    /// Create a WorkflowId from raw bytes (for deserialization).
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// The raw 32 bytes of this ID.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl std::fmt::Display for WorkflowId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Display first 8 hex chars for readability.
        for byte in &self.0[..4] {
            write!(f, "{byte:02x}")?;
        }
        write!(f, "...")
    }
}

/// A single recorded IO event in the workflow's execution history.
#[derive(Debug, Clone)]
pub enum HistoryEvent {
    /// WASM module requested content by CID (fetch_content call).
    IoRequested { cid: [u8; 32] },
    /// Content was resolved — data is `Some` if found, `None` if not found.
    IoResolved { cid: [u8; 32], data: Option<Vec<u8>> },
}

/// The complete event log for a workflow — this IS the durable checkpoint.
///
/// On crash recovery, the workflow engine replays the WASM module from
/// scratch, feeding cached IO responses from this history. WASM determinism
/// guarantees identical execution paths: same module + same input + same IO
/// responses = same `fetch_content` calls in the same order.
#[derive(Debug, Clone)]
pub struct WorkflowHistory {
    pub workflow_id: WorkflowId,
    pub module_hash: [u8; 32],
    /// The original input bytes passed to the WASM module.
    pub input: Vec<u8>,
    /// Ordered sequence of IO events recorded during execution.
    pub events: Vec<HistoryEvent>,
    /// Total fuel consumed across all execution rounds.
    pub total_fuel_consumed: u64,
}

/// Current lifecycle state of a workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    /// Queued, waiting for execution.
    Pending,
    /// Currently executing in the compute tier.
    Executing,
    /// Suspended waiting for external IO resolution.
    WaitingForIo { cid: [u8; 32] },
    /// Completed successfully.
    Complete,
    /// Failed with an error.
    Failed,
}

/// Inbound events for the workflow engine.
#[derive(Debug, Clone)]
pub enum WorkflowEvent {
    /// Submit a new workflow with inline module bytes.
    Submit {
        module: Vec<u8>,
        input: Vec<u8>,
        hint: crate::offload::ComputeHint,
    },
    /// Submit a workflow referencing a module by CID (fetched from storage).
    SubmitByCid {
        module_cid: [u8; 32],
        input: Vec<u8>,
        hint: crate::offload::ComputeHint,
    },
    /// Module bytes fetched from storage (for SubmitByCid).
    ModuleFetched { cid: [u8; 32], module: Vec<u8> },
    /// Module fetch failed.
    ModuleFetchFailed { cid: [u8; 32] },
    /// External IO resolved — content data is available.
    ContentFetched { cid: [u8; 32], data: Vec<u8> },
    /// External IO failed — content not found.
    ContentFetchFailed { cid: [u8; 32] },
}

/// Outbound actions returned by the workflow engine.
#[derive(Debug, Clone)]
pub enum WorkflowAction {
    /// Fetch a WASM module by CID from storage.
    FetchModule { cid: [u8; 32] },
    /// Fetch content by CID (IO request from WASM during execution).
    FetchContent { workflow_id: WorkflowId, cid: [u8; 32] },
    /// Workflow completed successfully — here is the output.
    WorkflowComplete { workflow_id: WorkflowId, output: Vec<u8> },
    /// Workflow failed.
    WorkflowFailed { workflow_id: WorkflowId, error: String },
    /// The workflow's history changed — caller should persist it.
    PersistHistory { workflow_id: WorkflowId },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workflow_id_deterministic() {
        let hash = [0xAB; 32];
        let input = b"hello";
        let id1 = WorkflowId::new(&hash, input);
        let id2 = WorkflowId::new(&hash, input);
        assert_eq!(id1, id2);
    }

    #[test]
    fn workflow_id_different_inputs_differ() {
        let hash = [0xAB; 32];
        let id1 = WorkflowId::new(&hash, b"hello");
        let id2 = WorkflowId::new(&hash, b"world");
        assert_ne!(id1, id2);
    }

    #[test]
    fn workflow_id_display() {
        let id = WorkflowId::from_bytes([0xDE; 32]);
        let display = format!("{id}");
        assert_eq!(display, "dededede...");
    }

    #[test]
    fn workflow_id_round_trip() {
        let hash = [0x42; 32];
        let id = WorkflowId::new(&hash, b"test");
        let bytes = *id.as_bytes();
        let id2 = WorkflowId::from_bytes(bytes);
        assert_eq!(id, id2);
    }

    #[test]
    fn history_event_construction() {
        let cid = [0xFF; 32];
        let req = HistoryEvent::IoRequested { cid };
        assert!(matches!(req, HistoryEvent::IoRequested { cid: c } if c == [0xFF; 32]));

        let resolved = HistoryEvent::IoResolved {
            cid,
            data: Some(vec![1, 2, 3]),
        };
        assert!(matches!(resolved, HistoryEvent::IoResolved { data: Some(d), .. } if d == vec![1, 2, 3]));
    }

    #[test]
    fn workflow_status_variants() {
        assert_eq!(WorkflowStatus::Pending, WorkflowStatus::Pending);
        assert_eq!(WorkflowStatus::Complete, WorkflowStatus::Complete);
        assert_ne!(WorkflowStatus::Pending, WorkflowStatus::Executing);

        let cid = [0x11; 32];
        assert_eq!(
            WorkflowStatus::WaitingForIo { cid },
            WorkflowStatus::WaitingForIo { cid }
        );
    }
}
```

**Step 3: Write offload types**

`crates/harmony-workflow/src/offload.rs`:
```rust
/// Hint from the caller about execution preferences.
///
/// The workflow engine uses this to decide whether to execute locally
/// or offload to a more capable peer. Currently always executes locally.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeHint {
    /// No preference — execute wherever is convenient.
    PreferLocal,
    /// Prefer a peer with more CPU/memory (e.g., desktop over mobile).
    PreferPowerful,
    /// Minimize end-to-end latency (prefer local even if slower).
    LatencySensitive,
    /// Ensure the workflow survives crashes (persist history aggressively).
    DurabilityRequired,
}

/// Decision from the offload evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDecision {
    /// Execute the workflow on this node.
    ExecuteLocally,
    /// Offload to a remote peer (identified by Harmony address).
    OffloadTo { peer: [u8; 16] },
}

/// Evaluate whether a workflow should execute locally or be offloaded.
///
/// Stub implementation: always returns `ExecuteLocally`. Real implementation
/// needs platform signals (battery, CPU load, power source) and Zenoh peer
/// discovery (latency, advertised capacity).
pub fn evaluate_offload(_hint: ComputeHint) -> OffloadDecision {
    OffloadDecision::ExecuteLocally
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_hint_variants_exist() {
        let _a = ComputeHint::PreferLocal;
        let _b = ComputeHint::PreferPowerful;
        let _c = ComputeHint::LatencySensitive;
        let _d = ComputeHint::DurabilityRequired;
    }

    #[test]
    fn offload_decision_always_local() {
        assert_eq!(
            evaluate_offload(ComputeHint::PreferLocal),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::PreferPowerful),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::LatencySensitive),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::DurabilityRequired),
            OffloadDecision::ExecuteLocally
        );
    }
}
```

**Step 4: Update lib.rs with re-exports**

`crates/harmony-workflow/src/lib.rs`:
```rust
pub mod engine;
pub mod error;
pub mod offload;
pub mod types;

pub use engine::WorkflowEngine;
pub use error::WorkflowError;
pub use offload::{ComputeHint, OffloadDecision};
pub use types::{
    HistoryEvent, WorkflowAction, WorkflowEvent, WorkflowHistory, WorkflowId, WorkflowStatus,
};
```

**Step 5: Run tests**

Run: `cargo test -p harmony-workflow`
Expected: All tests pass (types + offload + error tests)

**Step 6: Commit**

```bash
git add crates/harmony-workflow/
git commit -m "feat(workflow): add core types, error types, and offload stub"
```

---

### Task 3: WorkflowEngine — basic submit and tick (non-IO workflows)

**Files:**
- Modify: `crates/harmony-workflow/src/engine.rs`

**Context:** The engine wraps `ComputeTier` from `crates/harmony-node/src/compute.rs`. It delegates execution and intercepts results. For this task, we only handle the happy path: submit a workflow, tick until complete, emit `WorkflowComplete`. No IO handling yet.

Reference: `ComputeTier` API at `crates/harmony-node/src/compute.rs:94-354`. The engine creates `ComputeTierEvent::ExecuteInline` and reads `ComputeTierAction::SendReply`.

**Important:** `ComputeTier` is defined in `harmony-node`, not `harmony-compute`. The workflow engine needs to either depend on `harmony-node` (circular!) or we need to move `ComputeTier` or duplicate the logic. The cleanest approach: **WorkflowEngine owns a `Box<dyn ComputeRuntime>` directly and manages its own task queue**, rather than wrapping `ComputeTier`. This avoids the circular dependency.

**Revised architecture:** WorkflowEngine owns the runtime directly, with a simpler task model (one active workflow at a time for now — YAGNI).

**Step 1: Write the failing tests**

Add to `crates/harmony-workflow/src/engine.rs`:
```rust
use std::collections::HashMap;

use harmony_compute::{ComputeRuntime, InstructionBudget};

use crate::error::WorkflowError;
use crate::offload::ComputeHint;
use crate::types::{
    HistoryEvent, WorkflowAction, WorkflowEvent, WorkflowHistory, WorkflowId, WorkflowStatus,
};

/// Sans-I/O workflow engine providing durable execution over a WASM runtime.
///
/// Owns a [`ComputeRuntime`] and manages workflow lifecycle: submission,
/// execution, IO event recording, and recovery via deterministic replay.
pub struct WorkflowEngine {
    runtime: Box<dyn ComputeRuntime>,
    budget: InstructionBudget,
    /// All tracked workflows, indexed by WorkflowId.
    workflows: HashMap<WorkflowId, WorkflowState>,
    /// The currently executing workflow (if any).
    active: Option<WorkflowId>,
}

/// Internal state for a single workflow.
struct WorkflowState {
    status: WorkflowStatus,
    history: WorkflowHistory,
    hint: ComputeHint,
    /// Output bytes from a completed execution.
    output: Option<Vec<u8>>,
}

impl WorkflowEngine {
    /// Create a new workflow engine with the given runtime and per-tick fuel budget.
    pub fn new(runtime: Box<dyn ComputeRuntime>, budget: InstructionBudget) -> Self {
        Self {
            runtime,
            budget,
            workflows: HashMap::new(),
            active: None,
        }
    }

    /// Process an inbound event, returning any immediate actions.
    pub fn handle(&mut self, event: WorkflowEvent) -> Vec<WorkflowAction> {
        match event {
            WorkflowEvent::Submit {
                module,
                input,
                hint,
            } => self.handle_submit(module, input, hint),
            WorkflowEvent::SubmitByCid {
                module_cid,
                input,
                hint,
            } => self.handle_submit_by_cid(module_cid, input, hint),
            WorkflowEvent::ModuleFetched { cid, module } => {
                self.handle_module_fetched(cid, module)
            }
            WorkflowEvent::ModuleFetchFailed { cid } => self.handle_module_fetch_failed(cid),
            WorkflowEvent::ContentFetched { cid, data } => {
                self.handle_content_fetched(cid, data)
            }
            WorkflowEvent::ContentFetchFailed { cid } => self.handle_content_fetch_failed(cid),
        }
    }

    /// Run one execution slice, returning any resulting actions.
    ///
    /// If there is an active workflow with a yielded execution, resumes it.
    /// Otherwise, picks the next pending workflow and starts execution.
    pub fn tick(&mut self) -> Vec<WorkflowAction> {
        // If there's an active yielded workflow, resume it.
        if let Some(wf_id) = self.active {
            if self.runtime.has_pending() {
                let result = self.runtime.resume(self.budget);
                return self.handle_compute_result(wf_id, result);
            }
        }

        // Find the next pending workflow to start.
        let next = self
            .workflows
            .iter()
            .find(|(_, state)| state.status == WorkflowStatus::Pending)
            .map(|(id, _)| *id);

        let Some(wf_id) = next else {
            return Vec::new();
        };

        let state = self.workflows.get(&wf_id).unwrap();
        let module_hash = state.history.module_hash;
        let input = state.history.input.clone();

        // We need the module bytes. For inline submit, we stored them
        // temporarily. For CID-based, the caller provides them via
        // ModuleFetched. For now, inline stores module bytes in a
        // separate field — see handle_submit.
        //
        // Actually: for inline submit we need to store the module bytes.
        // Let's add a `module_bytes` field to WorkflowState.
        // This is set on Submit and cleared after first execution starts.
        // For recovery, the caller must provide module bytes via recover().
        todo!("start execution — see full implementation below")
    }

    /// Query the status of a workflow.
    pub fn workflow_status(&self, id: &WorkflowId) -> Option<&WorkflowStatus> {
        self.workflows.get(id).map(|s| &s.status)
    }

    /// Extract a workflow's history for external persistence.
    pub fn take_history(&self, id: &WorkflowId) -> Option<WorkflowHistory> {
        self.workflows.get(id).map(|s| s.history.clone())
    }

    /// Number of tracked workflows.
    pub fn workflow_count(&self) -> usize {
        self.workflows.len()
    }

    fn handle_submit(
        &mut self,
        module: Vec<u8>,
        input: Vec<u8>,
        hint: ComputeHint,
    ) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_submit_by_cid(
        &mut self,
        _module_cid: [u8; 32],
        _input: Vec<u8>,
        _hint: ComputeHint,
    ) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_module_fetched(
        &mut self,
        _cid: [u8; 32],
        _module: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_module_fetch_failed(&mut self, _cid: [u8; 32]) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_content_fetched(
        &mut self,
        _cid: [u8; 32],
        _data: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_content_fetch_failed(&mut self, _cid: [u8; 32]) -> Vec<WorkflowAction> {
        todo!()
    }

    fn handle_compute_result(
        &mut self,
        _wf_id: WorkflowId,
        _result: harmony_compute::ComputeResult,
    ) -> Vec<WorkflowAction> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_compute::WasmiRuntime;

    /// WAT module: reads two i32s from input, writes their sum as output.
    const ADD_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (i32.store
          (i32.add (local.get $input_ptr) (local.get $input_len))
          (i32.add
            (i32.load (local.get $input_ptr))
            (i32.load (i32.add (local.get $input_ptr) (i32.const 4)))))
        (i32.const 4)))
    "#;

    fn make_engine() -> WorkflowEngine {
        WorkflowEngine::new(
            Box::new(WasmiRuntime::new()),
            InstructionBudget { fuel: 100_000 },
        )
    }

    fn add_input(a: i32, b: i32) -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&a.to_le_bytes());
        v.extend_from_slice(&b.to_le_bytes());
        v
    }

    #[test]
    fn submit_creates_workflow_with_deterministic_id() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        let actions = engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });

        // Submit should not produce external actions (module is inline).
        assert!(
            actions.is_empty(),
            "inline submit should produce no actions, got: {actions:?}"
        );
        assert_eq!(engine.workflow_count(), 1);

        // Verify deterministic ID.
        let module_hash = *harmony_crypto::blake3::hash(&module).as_bytes();
        let expected_id = WorkflowId::new(&module_hash, &input);
        assert_eq!(
            engine.workflow_status(&expected_id),
            Some(&WorkflowStatus::Pending)
        );
    }

    #[test]
    fn tick_completes_simple_workflow() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(3, 7);

        engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Tick until WorkflowComplete appears (should be one or a few ticks).
        let mut all_actions = Vec::new();
        for _ in 0..100 {
            let actions = engine.tick();
            let done = actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
            all_actions.extend(actions);
            if done {
                break;
            }
        }

        // Should have WorkflowComplete with sum = 10.
        let complete = all_actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
        assert!(complete.is_some(), "expected WorkflowComplete, got: {all_actions:?}");
        if let Some(WorkflowAction::WorkflowComplete { output, .. }) = complete {
            let sum = i32::from_le_bytes(output[..4].try_into().unwrap());
            assert_eq!(sum, 10);
        }
    }

    #[test]
    fn duplicate_submit_returns_existing_id() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(1, 2);

        engine.handle(WorkflowEvent::Submit {
            module: module.clone(),
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });
        engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });

        // Should still be one workflow, not two.
        assert_eq!(engine.workflow_count(), 1);
    }

    #[test]
    fn workflow_status_transitions() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(5, 5);
        let module_hash = *harmony_crypto::blake3::hash(&module).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &input);

        engine.handle(WorkflowEvent::Submit {
            module,
            input,
            hint: ComputeHint::PreferLocal,
        });
        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::Pending)
        );

        // Tick to completion.
        loop {
            let actions = engine.tick();
            if actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }))
            {
                break;
            }
        }

        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::Complete)
        );
    }

    #[test]
    fn take_history_returns_workflow_history() {
        let mut engine = make_engine();
        let module = ADD_WAT.as_bytes().to_vec();
        let input = add_input(1, 1);
        let module_hash = *harmony_crypto::blake3::hash(&module).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &input);

        engine.handle(WorkflowEvent::Submit {
            module,
            input: input.clone(),
            hint: ComputeHint::PreferLocal,
        });

        let history = engine.take_history(&wf_id).expect("should have history");
        assert_eq!(history.workflow_id, wf_id);
        assert_eq!(history.module_hash, module_hash);
        assert_eq!(history.input, input);
        assert!(history.events.is_empty()); // no IO yet
    }
}
```

**Step 2: Implement the engine methods to make all tests pass**

Replace the `todo!()` stubs in the engine with real implementations. Key logic:

- `handle_submit`: compute `module_hash` via `harmony_crypto::blake3::hash()`, compute `WorkflowId::new(&module_hash, &input)`, create `WorkflowState` with `Pending` status, store module bytes in state.
- `tick`: find next `Pending` workflow, call `self.runtime.execute(module, input, budget)`, process result.
- `handle_compute_result`: match on `ComputeResult` variants:
  - `Complete { output }` → set status to `Complete`, emit `WorkflowComplete` + `PersistHistory`
  - `Yielded { .. }` → set status to `Executing`, keep `active` set (resume on next tick)
  - `Failed { error }` → set status to `Failed`, emit `WorkflowFailed`
  - `NeedsIO { .. }` → handled in Task 4

The `WorkflowState` struct needs an additional field:
```rust
struct WorkflowState {
    status: WorkflowStatus,
    history: WorkflowHistory,
    hint: ComputeHint,
    /// Module bytes for execution. Set on Submit, used on first tick.
    module_bytes: Option<Vec<u8>>,
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-workflow`
Expected: All tests pass

**Step 4: Commit**

```bash
git add crates/harmony-workflow/src/engine.rs
git commit -m "feat(workflow): implement WorkflowEngine with basic submit and tick"
```

---

### Task 4: WorkflowEngine — IO event recording and content resolution

**Files:**
- Modify: `crates/harmony-workflow/src/engine.rs`

**Context:** When a WASM module calls `harmony.fetch_content`, the compute runtime returns `NeedsIO`. The workflow engine must:
1. Record `HistoryEvent::IoRequested { cid }` in the workflow's history
2. Extract the runtime session (so the engine can handle other workflows)
3. Set workflow status to `WaitingForIo { cid }`
4. Emit `WorkflowAction::FetchContent` and `WorkflowAction::PersistHistory`

When content arrives (`ContentFetched`):
1. Record `HistoryEvent::IoResolved { cid, data }`
2. Restore the runtime session
3. Call `runtime.resume_with_io(ContentReady { data }, budget)`
4. Handle the compute result (may complete, yield, or need more IO)

Reference: The WASM module that calls `fetch_content` is defined at `crates/harmony-node/src/compute.rs:756-773` (FETCH_WAT). Copy it into the test module.

**Step 1: Write the failing tests**

Add to the `tests` module in `engine.rs`:
```rust
    /// WAT module that calls harmony.fetch_content.
    /// Input: [cid: 32 bytes]
    /// Output: [result_code: i32 LE] [fetched_data if result > 0]
    const FETCH_WAT: &str = r#"
    (module
      (import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $result i32)
        (local $out_ptr i32)
        (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
        (local.set $result
          (call $fetch
            (local.get $input_ptr)
            (i32.add (local.get $out_ptr) (i32.const 4))
            (i32.const 1024)))
        (i32.store (local.get $out_ptr) (local.get $result))
        (if (result i32) (i32.gt_s (local.get $result) (i32.const 0))
          (then (i32.add (i32.const 4) (local.get $result)))
          (else (i32.const 4)))))
    "#;

    #[test]
    fn io_request_recorded_in_history() {
        let mut engine = make_engine();
        let cid = [0xAB; 32];

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        let actions = engine.tick();

        // Should emit FetchContent action.
        let fetch = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::FetchContent { .. }));
        assert!(fetch.is_some(), "expected FetchContent, got: {actions:?}");

        // History should contain IoRequested.
        let module_hash = *harmony_crypto::blake3::hash(FETCH_WAT.as_bytes()).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &cid);
        let history = engine.take_history(&wf_id).unwrap();
        assert_eq!(history.events.len(), 1);
        assert!(matches!(
            &history.events[0],
            HistoryEvent::IoRequested { cid: c } if *c == cid
        ));
    }

    #[test]
    fn content_fetched_resumes_and_completes() {
        let mut engine = make_engine();
        let cid = [0xCD; 32];
        let content = b"fetched data";

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });

        // Tick → NeedsIO → FetchContent.
        let actions = engine.tick();
        assert!(actions
            .iter()
            .any(|a| matches!(a, WorkflowAction::FetchContent { .. })));

        // Deliver content.
        let actions = engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        // Should complete.
        let complete = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
        assert!(
            complete.is_some(),
            "expected WorkflowComplete after content delivery, got: {actions:?}"
        );

        // Verify output contains the fetched data.
        if let Some(WorkflowAction::WorkflowComplete { output, .. }) = complete {
            let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
            assert_eq!(&output[4..], content);
        }
    }

    #[test]
    fn io_response_recorded_in_history() {
        let mut engine = make_engine();
        let cid = [0xEF; 32];
        let content = b"history test";

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // → NeedsIO

        engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        let module_hash = *harmony_crypto::blake3::hash(FETCH_WAT.as_bytes()).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &cid);
        let history = engine.take_history(&wf_id).unwrap();

        // Should have IoRequested + IoResolved.
        assert_eq!(history.events.len(), 2);
        assert!(matches!(&history.events[0], HistoryEvent::IoRequested { .. }));
        assert!(matches!(
            &history.events[1],
            HistoryEvent::IoResolved { data: Some(d), .. } if d == content
        ));
    }

    #[test]
    fn content_fetch_failed_resolves_with_not_found() {
        let mut engine = make_engine();
        let cid = [0x11; 32];

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // → NeedsIO

        let actions = engine.handle(WorkflowEvent::ContentFetchFailed { cid });

        // Module should still complete (with -1 result code).
        let complete = actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
        assert!(complete.is_some(), "should complete even on content not found");

        if let Some(WorkflowAction::WorkflowComplete { output, .. }) = complete {
            let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
            assert_eq!(result_code, -1, "should be -1 for not found");
        }
    }

    #[test]
    fn waiting_for_io_status() {
        let mut engine = make_engine();
        let cid = [0x22; 32];

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // → NeedsIO

        let module_hash = *harmony_crypto::blake3::hash(FETCH_WAT.as_bytes()).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &cid);
        assert_eq!(
            engine.workflow_status(&wf_id),
            Some(&WorkflowStatus::WaitingForIo { cid })
        );
    }
```

**Step 2: Implement IO handling in the engine**

Key additions to `handle_compute_result`:
- `NeedsIO { request: IORequest::FetchContent { cid } }`:
  - Record `HistoryEvent::IoRequested { cid }` in workflow history
  - Extract runtime session with `self.runtime.take_session()`
  - Store saved session in `WorkflowState`
  - Set status to `WaitingForIo { cid }`
  - Clear `self.active`
  - Emit `WorkflowAction::FetchContent` + `WorkflowAction::PersistHistory`

`WorkflowState` needs a new field:
```rust
    /// Saved runtime session when workflow is suspended on IO.
    saved_session: Option<Box<dyn std::any::Any>>,
```

Implement `handle_content_fetched`:
- Find workflow in `WaitingForIo` state matching the CID
- Record `HistoryEvent::IoResolved { cid, data: Some(data.clone()) }`
- Restore runtime session
- Call `runtime.resume_with_io(ContentReady { data }, budget)`
- Process result via `handle_compute_result`

Implement `handle_content_fetch_failed`:
- Same as above but with `IoResolved { data: None }` and `ContentNotFound`

**Step 3: Run tests**

Run: `cargo test -p harmony-workflow`
Expected: All tests pass (previous + new IO tests)

**Step 4: Commit**

```bash
git add crates/harmony-workflow/src/engine.rs
git commit -m "feat(workflow): add IO event recording and content resolution to WorkflowEngine"
```

---

### Task 5: WorkflowEngine — recovery via deterministic replay

**Files:**
- Modify: `crates/harmony-workflow/src/engine.rs`

**Context:** Recovery replays the WASM module from scratch using the persisted event log. When the module hits `fetch_content` during replay, the engine checks the history — if the CID matches a recorded `IoResolved` event, it feeds the cached data immediately (no external fetch). If execution reaches beyond the recorded history, it becomes a new IO request.

**Step 1: Write the failing tests**

```rust
    #[test]
    fn recover_replays_from_history() {
        // First: run a workflow to completion with IO.
        let mut engine = make_engine();
        let cid = [0x33; 32];
        let content = b"recovery test data";

        engine.handle(WorkflowEvent::Submit {
            module: FETCH_WAT.as_bytes().to_vec(),
            input: cid.to_vec(),
            hint: ComputeHint::PreferLocal,
        });
        engine.tick(); // → NeedsIO
        engine.handle(WorkflowEvent::ContentFetched {
            cid,
            data: content.to_vec(),
        });

        let module_hash = *harmony_crypto::blake3::hash(FETCH_WAT.as_bytes()).as_bytes();
        let wf_id = WorkflowId::new(&module_hash, &cid);
        let history = engine.take_history(&wf_id).unwrap();

        // Now: create a fresh engine and recover from the history.
        let mut engine2 = make_engine();
        let actions = engine2.recover(history, FETCH_WAT.as_bytes().to_vec());

        // Recovery should immediately start execution. Since all IO is
        // cached in the history, it should complete without external fetches.
        let mut all_actions = actions;
        for _ in 0..100 {
            let actions = engine2.tick();
            let done = actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
            all_actions.extend(actions);
            if done {
                break;
            }
        }

        let complete = all_actions
            .iter()
            .find(|a| matches!(a, WorkflowAction::WorkflowComplete { .. }));
        assert!(
            complete.is_some(),
            "recovery should complete, got: {all_actions:?}"
        );

        if let Some(WorkflowAction::WorkflowComplete { output, .. }) = complete {
            let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
            assert_eq!(result_code, content.len() as i32);
            assert_eq!(&output[4..], content);
        }
    }

    #[test]
    fn recover_with_new_io_beyond_history() {
        // History has NO events (module was at NeedsIO but never got content).
        let module_hash = *harmony_crypto::blake3::hash(FETCH_WAT.as_bytes()).as_bytes();
        let cid = [0x44; 32];
        let wf_id = WorkflowId::new(&module_hash, &cid);

        let history = WorkflowHistory {
            workflow_id: wf_id,
            module_hash,
            input: cid.to_vec(),
            events: vec![], // No recorded IO
            total_fuel_consumed: 0,
        };

        let mut engine = make_engine();
        let _actions = engine.recover(history, FETCH_WAT.as_bytes().to_vec());

        // Tick: module will call fetch_content → NeedsIO → FetchContent.
        let mut all_actions = Vec::new();
        for _ in 0..100 {
            let actions = engine.tick();
            all_actions.extend(actions.clone());
            if actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::FetchContent { .. }))
            {
                break;
            }
        }

        assert!(
            all_actions
                .iter()
                .any(|a| matches!(a, WorkflowAction::FetchContent { .. })),
            "should emit FetchContent for IO beyond recorded history"
        );
    }
```

**Step 2: Implement recover()**

```rust
    /// Recover a workflow from a persisted history.
    ///
    /// Re-submits the module for execution. During replay, if `fetch_content`
    /// is called for a CID present in the history as `IoResolved`, the cached
    /// data is fed immediately (no external fetch). If execution reaches
    /// beyond the recorded history, new IO requests are emitted normally.
    pub fn recover(
        &mut self,
        history: WorkflowHistory,
        module_bytes: Vec<u8>,
    ) -> Vec<WorkflowAction> {
        // ... implementation details ...
    }
```

The recovery mechanism works by:
1. Creating a `WorkflowState` with the existing history events as a "replay cache"
2. Adding a `replay_cache: HashMap<[u8; 32], Option<Vec<u8>>>` field to `WorkflowState` that is populated from the history's `IoResolved` events
3. When `NeedsIO` occurs during replay, checking the replay cache before emitting `FetchContent`
4. If cache hit: immediately feed content via `runtime.resume_with_io()` (loops until cache miss or completion)
5. If cache miss: emit `FetchContent` normally (new IO beyond recorded history)

Add `replay_cache` field to `WorkflowState`:
```rust
    /// Cached IO responses from a prior execution (used during recovery replay).
    /// Populated from WorkflowHistory.events on recover(). Consumed as IO
    /// requests are replayed.
    replay_cache: HashMap<[u8; 32], Option<Vec<u8>>>,
```

Modify `handle_compute_result` for the `NeedsIO` case:
- Before emitting `FetchContent`, check `state.replay_cache` for the CID
- If found: immediately feed the cached data, recurse on the result
- If not found: normal path (emit `FetchContent`)

**Step 3: Run tests**

Run: `cargo test -p harmony-workflow`
Expected: All tests pass (all previous + recovery tests)

**Step 4: Commit**

```bash
git add crates/harmony-workflow/src/engine.rs
git commit -m "feat(workflow): implement recovery via deterministic replay from event history"
```

---

### Task 6: Node integration — swap ComputeTier for WorkflowEngine

**Files:**
- Modify: `crates/harmony-node/Cargo.toml:17-26`
- Modify: `crates/harmony-node/src/runtime.rs:1-125`
- Modify: `crates/harmony-node/src/runtime.rs` (test section)

**Context:** `NodeRuntime` currently owns `ComputeTier` directly. We replace it with `WorkflowEngine` and translate between `WorkflowEvent`/`WorkflowAction` and `RuntimeEvent`/`RuntimeAction`. The node's existing tests should continue to pass with minimal changes.

**Step 1: Add harmony-workflow dependency to harmony-node**

In `crates/harmony-node/Cargo.toml`, add:
```toml
harmony-workflow.workspace = true
```

**Step 2: Update NodeRuntime to use WorkflowEngine**

In `crates/harmony-node/src/runtime.rs`:

1. Replace `use crate::compute::{ComputeTier, ComputeTierAction, ComputeTierEvent};` with `use harmony_workflow::{WorkflowEngine, WorkflowEvent, WorkflowAction};`
2. Change `compute: ComputeTier` field to `workflow: WorkflowEngine`
3. Change `pending_compute_actions: Vec<ComputeTierAction>` to `pending_workflow_actions: Vec<WorkflowAction>`
4. Update `new()`:
   - Create `WorkflowEngine::new(runtime, budget)` instead of `ComputeTier::new(runtime, budget)`
5. Update `push_event()`:
   - `ComputeQuery` → `WorkflowEvent::Submit` / `SubmitByCid` (parse payload same as before)
   - `ContentFetchResponse` → `WorkflowEvent::ContentFetched` / `ContentFetchFailed`
   - `ModuleFetchResponse` → `WorkflowEvent::ModuleFetched` / `ModuleFetchFailed`
6. Update `tick()`:
   - Dispatch pending workflow actions, then `workflow.tick()`
7. Update `dispatch_compute_actions()` → `dispatch_workflow_actions()`:
   - `WorkflowAction::WorkflowComplete { workflow_id, output }` → `RuntimeAction::SendReply` with `[0x00, output...]`
   - `WorkflowAction::WorkflowFailed { workflow_id, error }` → `RuntimeAction::SendReply` with `[0x01, error...]`
   - `WorkflowAction::FetchContent { cid, .. }` → `RuntimeAction::FetchContent { cid }`
   - `WorkflowAction::FetchModule { cid }` → `RuntimeAction::FetchContent { cid }`
   - `WorkflowAction::PersistHistory { .. }` → no-op (future: Zenoh persistence)

**Important:** The wire format encoding (0x00 success / 0x01 error tags) moves from `ComputeTier::handle_compute_result` to `NodeRuntime::dispatch_workflow_actions`. This is correct — the workflow engine emits semantic actions (`WorkflowComplete` with raw output), and the node encodes them for the wire.

**Step 3: Update routing**

The `route_compute_query` method stays mostly the same, but instead of creating `ComputeTierEvent`, it creates `WorkflowEvent`:

```rust
fn route_compute_query(&mut self, query_id: u64, _key_expr: String, payload: Vec<u8>) -> Vec<WorkflowAction> {
    // Parse payload (same 0x00 inline / 0x01 CID format).
    // Map to WorkflowEvent::Submit or SubmitByCid.
    // Call self.workflow.handle(event)
}
```

**Note:** The workflow engine doesn't use query_ids (it uses WorkflowIds). The node runtime needs to maintain a `HashMap<WorkflowId, u64>` mapping workflow IDs to query IDs for reply routing.

**Step 4: Run ALL existing tests**

Run: `cargo test -p harmony-node`
Expected: All existing tests pass. The behavior is identical — WorkflowEngine wraps the same WasmiRuntime, produces the same outputs.

Run: `cargo test --workspace`
Expected: All workspace tests pass.

Run: `cargo clippy --workspace`
Expected: Zero warnings.

**Step 5: Commit**

```bash
git add crates/harmony-node/
git commit -m "refactor(node): swap ComputeTier for WorkflowEngine in NodeRuntime"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Scaffold crate | compile check |
| 2 | Error + core types + offload | ~12 unit tests |
| 3 | Engine: submit + tick (non-IO) | ~4 tests |
| 4 | Engine: IO event recording | ~5 tests |
| 5 | Engine: recovery via replay | ~2 tests |
| 6 | Node integration | existing ~20 tests adapted |

Total: ~43 tests across harmony-workflow + harmony-node.

CI: `cargo test -p harmony-workflow` + `cargo test -p harmony-node` + `cargo clippy --workspace`.

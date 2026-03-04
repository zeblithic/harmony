# Tier 3 Compute Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire harmony-compute into harmony-node's event loop as Tier 3, completing the Router/Storage/Compute priority trinity.

**Architecture:** A new `ComputeTier` sans-I/O state machine in harmony-node wraps `WasmiRuntime` with a FIFO task queue and auto-resume logic. `NodeRuntime` extends its `tick()` method: drain all router → one storage → one compute slice. Compute tasks arrive as Zenoh queries on `harmony/compute/activity/*` with either inline WASM module bytes or a CID reference to a module stored in Tier 2.

**Tech Stack:** Rust, harmony-compute (wasmi), harmony-zenoh (namespace constants), harmony-node (NodeRuntime)

**Design doc:** `docs/plans/2026-03-04-tier3-compute-integration-design.md`

---

## Task 1: Add harmony-compute dependency to harmony-node

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`

**Step 1: Add the dependency**

In `crates/harmony-node/Cargo.toml`, add `harmony-compute` to `[dependencies]`:

```toml
[dependencies]
harmony-compute.workspace = true
harmony-content.workspace = true
harmony-identity = { path = "../harmony-identity" }
harmony-reticulum.workspace = true
harmony-zenoh.workspace = true
clap = { version = "4", features = ["derive"] }
hex.workspace = true
rand.workspace = true
```

Also add `harmony-compute` to the workspace `Cargo.toml` root's `[workspace.dependencies]` if it's not already there. Check `Cargo.toml` at the repo root — it should already have `harmony-compute = { path = "crates/harmony-compute" }` from when the crate was created. If not, add it.

**Step 2: Verify it compiles**

Run: `cargo check -p harmony-node`
Expected: success (no new code yet, just the dependency)

**Step 3: Commit**

```bash
git add crates/harmony-node/Cargo.toml Cargo.toml Cargo.lock
git commit -m "chore(node): add harmony-compute dependency"
```

---

## Task 2: Create ComputeTier with event/action types and empty tick()

**Files:**
- Create: `crates/harmony-node/src/compute.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod compute;`)

This task creates the `ComputeTier` struct, its event/action enums, and an empty `tick()` + `handle()`. No business logic yet — just types and scaffolding.

**Step 1: Write tests for ComputeTier construction and empty tick**

At the bottom of `crates/harmony-node/src/compute.rs`, add tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_compute::InstructionBudget;

    fn make_compute_tier() -> ComputeTier {
        ComputeTier::new(InstructionBudget { fuel: 100_000 })
    }

    #[test]
    fn tick_with_no_tasks_returns_nothing() {
        let mut ct = make_compute_tier();
        let actions = ct.tick();
        assert!(actions.is_empty());
    }

    #[test]
    fn queue_starts_empty() {
        let ct = make_compute_tier();
        assert_eq!(ct.queue_len(), 0);
        assert!(!ct.has_active());
    }
}
```

**Step 2: Implement the scaffolding**

Create `crates/harmony-node/src/compute.rs`:

```rust
//! Tier 3 compute: WASM execution with task queue and auto-resume.

use std::collections::VecDeque;

use harmony_compute::{ComputeRuntime, InstructionBudget, WasmiRuntime};

/// Events fed into the compute tier.
#[derive(Debug)]
pub enum ComputeTierEvent {
    /// New compute request with inline WASM module bytes.
    ExecuteInline {
        query_id: u64,
        module: Vec<u8>,
        input: Vec<u8>,
    },
    /// New compute request referencing a module by CID.
    ExecuteByCid {
        query_id: u64,
        module_cid: [u8; 32],
        input: Vec<u8>,
    },
    /// Module bytes arrived for a previously-requested CID fetch.
    ModuleFetched {
        cid: [u8; 32],
        module: Vec<u8>,
    },
    /// Module CID fetch failed.
    ModuleFetchFailed {
        cid: [u8; 32],
    },
}

/// Actions emitted by the compute tier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComputeTierAction {
    /// Reply to the originating query with compute output.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Request module bytes from Tier 2 / network.
    FetchModule { cid: [u8; 32] },
    /// Report compute failure back to the query.
    SendError { query_id: u64, error: String },
}

/// A queued compute task.
#[derive(Debug)]
enum ComputeTask {
    /// Module bytes available, ready to execute.
    Ready {
        query_id: u64,
        module: Vec<u8>,
        input: Vec<u8>,
    },
    /// Waiting for module bytes from a CID fetch.
    WaitingForModule {
        query_id: u64,
        cid: [u8; 32],
        input: Vec<u8>,
    },
}

/// Tracks the currently executing WASM task.
#[derive(Debug)]
struct ActiveExecution {
    query_id: u64,
}

/// Sans-I/O state machine for Tier 3 compute.
///
/// Manages a FIFO queue of compute tasks with one active execution at a time.
/// Each `tick()` runs one fuel-bounded slice of WASM instructions. Yielded
/// tasks auto-resume on the next tick.
pub struct ComputeTier {
    runtime: WasmiRuntime,
    queue: VecDeque<ComputeTask>,
    active: Option<ActiveExecution>,
    budget: InstructionBudget,
}

impl ComputeTier {
    pub fn new(budget: InstructionBudget) -> Self {
        Self {
            runtime: WasmiRuntime::new(),
            queue: VecDeque::new(),
            active: None,
            budget,
        }
    }

    /// Process one compute event, returning any immediate actions.
    pub fn handle(&mut self, event: ComputeTierEvent) -> Vec<ComputeTierAction> {
        let _ = event;
        Vec::new()
    }

    /// Run one fuel-bounded slice of the active compute task.
    /// Returns actions if the task completes, fails, or needs a module fetch.
    pub fn tick(&mut self) -> Vec<ComputeTierAction> {
        Vec::new()
    }

    /// Number of tasks in the queue (not counting active execution).
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Whether there is a currently executing task.
    pub fn has_active(&self) -> bool {
        self.active.is_some()
    }
}
```

Add `mod compute;` to `crates/harmony-node/src/main.rs` after the existing `mod runtime;` line.

**Step 3: Run tests to verify**

Run: `cargo test -p harmony-node`
Expected: all existing tests pass + 2 new tests pass

**Step 4: Commit**

```bash
git add crates/harmony-node/src/compute.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): scaffold ComputeTier with event/action types"
```

---

## Task 3: Implement handle() for event processing

**Files:**
- Modify: `crates/harmony-node/src/compute.rs`

This task implements `handle()` for all four event variants: `ExecuteInline`, `ExecuteByCid`, `ModuleFetched`, `ModuleFetchFailed`.

**Step 1: Write tests**

Add these tests to the existing `mod tests` block in `compute.rs`:

```rust
#[test]
fn handle_execute_inline_queues_ready_task() {
    let mut ct = make_compute_tier();
    let actions = ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 1,
        module: ADD_WAT.as_bytes().to_vec(),
        input: vec![0; 8],
    });
    assert!(actions.is_empty()); // no immediate actions, just queued
    assert_eq!(ct.queue_len(), 1);
}

#[test]
fn handle_execute_by_cid_emits_fetch() {
    let mut ct = make_compute_tier();
    let cid = [0xAB; 32];
    let actions = ct.handle(ComputeTierEvent::ExecuteByCid {
        query_id: 1,
        module_cid: cid,
        input: vec![1, 2, 3],
    });
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0],
        ComputeTierAction::FetchModule { cid: c } if *c == [0xAB; 32]
    ));
    assert_eq!(ct.queue_len(), 1);
}

#[test]
fn handle_module_fetched_promotes_to_ready() {
    let mut ct = make_compute_tier();
    let cid = [0xAB; 32];
    ct.handle(ComputeTierEvent::ExecuteByCid {
        query_id: 1,
        module_cid: cid,
        input: vec![1, 2, 3],
    });
    let actions = ct.handle(ComputeTierEvent::ModuleFetched {
        cid,
        module: ADD_WAT.as_bytes().to_vec(),
    });
    assert!(actions.is_empty());
    // Task should now be Ready (verified by executing on next tick)
    assert_eq!(ct.queue_len(), 1);
}

#[test]
fn handle_module_fetch_failed_returns_error() {
    let mut ct = make_compute_tier();
    let cid = [0xAB; 32];
    ct.handle(ComputeTierEvent::ExecuteByCid {
        query_id: 1,
        module_cid: cid,
        input: vec![],
    });
    let actions = ct.handle(ComputeTierEvent::ModuleFetchFailed { cid });
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0],
        ComputeTierAction::SendError { query_id: 1, .. }
    ));
    assert_eq!(ct.queue_len(), 0);
}
```

You'll also need to add the ADD_WAT constant to the test module. Copy it from `crates/harmony-compute/src/wasmi_runtime.rs`:

```rust
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
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node`
Expected: 4 new tests FAIL (handle() is a stub returning empty Vec)

**Step 3: Implement handle()**

Replace the stub `handle()` method:

```rust
pub fn handle(&mut self, event: ComputeTierEvent) -> Vec<ComputeTierAction> {
    match event {
        ComputeTierEvent::ExecuteInline {
            query_id,
            module,
            input,
        } => {
            self.queue
                .push_back(ComputeTask::Ready { query_id, module, input });
            Vec::new()
        }
        ComputeTierEvent::ExecuteByCid {
            query_id,
            module_cid,
            input,
        } => {
            self.queue
                .push_back(ComputeTask::WaitingForModule { query_id, cid: module_cid, input });
            vec![ComputeTierAction::FetchModule { cid: module_cid }]
        }
        ComputeTierEvent::ModuleFetched { cid, module } => {
            // Find the WaitingForModule task with this CID and promote to Ready.
            if let Some(pos) = self.queue.iter().position(|t| matches!(
                t,
                ComputeTask::WaitingForModule { cid: c, .. } if *c == cid
            )) {
                if let Some(ComputeTask::WaitingForModule { query_id, input, .. }) =
                    self.queue.remove(pos)
                {
                    self.queue.push_back(ComputeTask::Ready { query_id, module, input });
                }
            }
            Vec::new()
        }
        ComputeTierEvent::ModuleFetchFailed { cid } => {
            let mut actions = Vec::new();
            if let Some(pos) = self.queue.iter().position(|t| matches!(
                t,
                ComputeTask::WaitingForModule { cid: c, .. } if *c == cid
            )) {
                if let Some(ComputeTask::WaitingForModule { query_id, .. }) =
                    self.queue.remove(pos)
                {
                    actions.push(ComputeTierAction::SendError {
                        query_id,
                        error: format!("module not found: {}", hex::encode(cid)),
                    });
                }
            }
            actions
        }
    }
}
```

Note: this uses `hex::encode` for the error message. The `hex` crate is already a dependency of `harmony-node`.

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node`
Expected: all tests pass

**Step 5: Commit**

```bash
git add crates/harmony-node/src/compute.rs
git commit -m "feat(node): implement ComputeTier handle() for all event variants"
```

---

## Task 4: Implement tick() with execute, auto-resume, and completion

**Files:**
- Modify: `crates/harmony-node/src/compute.rs`

This task implements the core `tick()` logic: dequeue a Ready task, execute it, auto-resume yielded tasks, and emit replies/errors on completion.

**Step 1: Write tests**

Add these tests to `mod tests` in `compute.rs`. You'll also need the LOOP_WAT constant — copy from `crates/harmony-compute/src/wasmi_runtime.rs`:

```rust
/// WAT module: loop `count` times, write count as output. Burns fuel proportional to count.
const LOOP_WAT: &str = r#"
    (module
      (memory (export "memory") 1)
      (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
        (local $count i32)
        (local $i i32)
        (local.set $count (i32.load (local.get $input_ptr)))
        (local.set $i (i32.const 0))
        (block $break
          (loop $loop
            (br_if $break (i32.ge_u (local.get $i) (local.get $count)))
            (local.set $i (i32.add (local.get $i) (i32.const 1)))
            (br $loop)))
        (i32.store
          (i32.add (local.get $input_ptr) (local.get $input_len))
          (local.get $i))
        (i32.const 4)))
"#;

#[test]
fn execute_inline_completes() {
    let mut ct = make_compute_tier();
    let mut input = Vec::new();
    input.extend_from_slice(&3i32.to_le_bytes());
    input.extend_from_slice(&7i32.to_le_bytes());

    ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 42,
        module: ADD_WAT.as_bytes().to_vec(),
        input,
    });

    let actions = ct.tick();
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        ComputeTierAction::SendReply { query_id, payload } => {
            assert_eq!(*query_id, 42);
            // Reply format: [0x00] [output_bytes]
            assert_eq!(payload[0], 0x00);
            let sum = i32::from_le_bytes(payload[1..5].try_into().unwrap());
            assert_eq!(sum, 10);
        }
        other => panic!("expected SendReply, got {other:?}"),
    }
    assert!(!ct.has_active());
}

#[test]
fn execute_invalid_module_returns_error() {
    let mut ct = make_compute_tier();
    ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 1,
        module: b"not wasm".to_vec(),
        input: vec![],
    });

    let actions = ct.tick();
    assert_eq!(actions.len(), 1);
    assert!(matches!(
        &actions[0],
        ComputeTierAction::SendError { query_id: 1, .. }
    ));
    assert!(!ct.has_active());
}

#[test]
fn yielded_task_auto_resumes() {
    let mut ct = ComputeTier::new(InstructionBudget { fuel: 10_000 });
    let input = 50_000_i32.to_le_bytes().to_vec();

    ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 7,
        module: LOOP_WAT.as_bytes().to_vec(),
        input,
    });

    // First tick: should start execution and yield (not enough fuel for 50K iterations)
    let actions = ct.tick();
    assert!(actions.is_empty(), "yielded tasks should produce no actions");
    assert!(ct.has_active());

    // Keep ticking until it completes
    let mut total_ticks = 1;
    loop {
        let actions = ct.tick();
        total_ticks += 1;
        if !actions.is_empty() {
            assert_eq!(actions.len(), 1);
            match &actions[0] {
                ComputeTierAction::SendReply { query_id, payload } => {
                    assert_eq!(*query_id, 7);
                    assert_eq!(payload[0], 0x00);
                    let count = i32::from_le_bytes(payload[1..5].try_into().unwrap());
                    assert_eq!(count, 50_000);
                }
                other => panic!("expected SendReply, got {other:?}"),
            }
            break;
        }
        assert!(total_ticks < 100, "should complete within 100 ticks");
    }
    assert!(total_ticks > 1, "should have needed multiple ticks");
    assert!(!ct.has_active());
}

#[test]
fn multiple_tasks_queued_fifo() {
    let mut ct = make_compute_tier();

    // Queue two tasks
    let mut input1 = Vec::new();
    input1.extend_from_slice(&10i32.to_le_bytes());
    input1.extend_from_slice(&20i32.to_le_bytes());

    let mut input2 = Vec::new();
    input2.extend_from_slice(&100i32.to_le_bytes());
    input2.extend_from_slice(&200i32.to_le_bytes());

    ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 1,
        module: ADD_WAT.as_bytes().to_vec(),
        input: input1,
    });
    ct.handle(ComputeTierEvent::ExecuteInline {
        query_id: 2,
        module: ADD_WAT.as_bytes().to_vec(),
        input: input2,
    });

    // First tick: task 1 completes
    let actions = ct.tick();
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], ComputeTierAction::SendReply { query_id: 1, .. }));

    // Second tick: task 2 completes
    let actions = ct.tick();
    assert_eq!(actions.len(), 1);
    assert!(matches!(&actions[0], ComputeTierAction::SendReply { query_id: 2, .. }));

    // Third tick: nothing
    let actions = ct.tick();
    assert!(actions.is_empty());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node`
Expected: 4 new tests FAIL (tick() is a stub)

**Step 3: Implement tick()**

Replace the stub `tick()` method:

```rust
pub fn tick(&mut self) -> Vec<ComputeTierAction> {
    // If there's an active execution, resume it.
    if self.active.is_some() {
        let result = self.runtime.resume(self.budget);
        return self.handle_compute_result(result);
    }

    // Otherwise, try to dequeue the next Ready task.
    let ready_pos = self.queue.iter().position(|t| matches!(t, ComputeTask::Ready { .. }));
    let Some(pos) = ready_pos else {
        return Vec::new();
    };
    let Some(ComputeTask::Ready { query_id, module, input }) = self.queue.remove(pos) else {
        return Vec::new();
    };

    self.active = Some(ActiveExecution { query_id });
    let result = self.runtime.execute(&module, &input, self.budget);
    self.handle_compute_result(result)
}
```

Add this helper method to `ComputeTier`:

```rust
fn handle_compute_result(&mut self, result: harmony_compute::ComputeResult) -> Vec<ComputeTierAction> {
    use harmony_compute::ComputeResult;

    let query_id = self.active.as_ref().expect("must have active execution").query_id;

    match result {
        ComputeResult::Complete { output } => {
            self.active = None;
            let mut payload = vec![0x00];
            payload.extend_from_slice(&output);
            vec![ComputeTierAction::SendReply { query_id, payload }]
        }
        ComputeResult::Yielded { .. } => {
            // Auto-resume on next tick — no action needed.
            Vec::new()
        }
        ComputeResult::Failed { error } => {
            self.active = None;
            vec![ComputeTierAction::SendError {
                query_id,
                error: format!("{error}"),
            }]
        }
        ComputeResult::NeedsIO { .. } => {
            self.active = None;
            vec![ComputeTierAction::SendError {
                query_id,
                error: "NeedsIO not yet supported".into(),
            }]
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node`
Expected: all tests pass

**Step 5: Commit**

```bash
git add crates/harmony-node/src/compute.rs
git commit -m "feat(node): implement ComputeTier tick() with auto-resume"
```

---

## Task 5: Wire ComputeTier into NodeRuntime

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

This task extends `NodeRuntime` with Tier 3: adds the `ComputeTier` field, new `RuntimeEvent`/`RuntimeAction` variants, compute queryable registration at startup, query routing, and the third priority level in `tick()`.

**Step 1: Extend NodeConfig**

Add `compute_budget` to `NodeConfig`:

```rust
use harmony_compute::InstructionBudget;

pub struct NodeConfig {
    pub storage_budget: StorageBudget,
    pub compute_budget: InstructionBudget,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            storage_budget: StorageBudget {
                cache_capacity: 1024,
                max_pinned_bytes: 100_000_000,
            },
            compute_budget: InstructionBudget { fuel: 100_000 },
        }
    }
}
```

**Step 2: Add RuntimeEvent and RuntimeAction variants**

Add to `RuntimeEvent`:

```rust
/// Tier 3: Compute query received (activity request).
ComputeQuery {
    query_id: u64,
    key_expr: String,
    payload: Vec<u8>,
},
/// Tier 3: Module fetch response for a CID-referenced compute request.
ModuleFetchResponse {
    cid: [u8; 32],
    result: Result<Vec<u8>, String>,
},
```

Add to `RuntimeAction`:

```rust
/// Tier 3: Fetch a WASM module by CID from Tier 2 / network.
FetchContent { cid: [u8; 32] },
```

**Step 3: Add ComputeTier to NodeRuntime struct and constructor**

Add to the struct:

```rust
use crate::compute::{ComputeTier, ComputeTierAction, ComputeTierEvent};

pub struct NodeRuntime<B: BlobStore> {
    // ... existing fields ...
    // Tier 3: Compute
    compute: ComputeTier,
    compute_queryable_ids: HashSet<QueryableId>,
}
```

In `new()`, after the storage setup, add compute setup:

```rust
// Tier 3: Compute — register activity queryable
let compute = ComputeTier::new(config.compute_budget);
let mut compute_queryable_ids = HashSet::new();

let (qid, _) = queryable_router
    .declare(harmony_zenoh::namespace::compute::ACTIVITY_SUB)
    .expect("static key expression must be valid");
compute_queryable_ids.insert(qid);
actions.push(RuntimeAction::DeclareQueryable {
    key_expr: harmony_zenoh::namespace::compute::ACTIVITY_SUB.to_string(),
});
```

Add `compute` and `compute_queryable_ids` to the `Self { ... }` construction.

Add a `compute_queue_len()` method:

```rust
pub fn compute_queue_len(&self) -> usize {
    self.compute.queue_len()
}
```

**Step 4: Extend push_event() for compute events**

Add these match arms to `push_event()`:

```rust
RuntimeEvent::ComputeQuery {
    query_id,
    key_expr,
    payload,
} => {
    self.route_compute_query(query_id, key_expr, payload);
}
RuntimeEvent::ModuleFetchResponse { cid, result } => {
    let event = match result {
        Ok(module) => ComputeTierEvent::ModuleFetched { cid, module },
        Err(_) => ComputeTierEvent::ModuleFetchFailed { cid },
    };
    let actions = self.compute.handle(event);
    // Store actions for next tick (or we could dispatch immediately).
    // For simplicity, handle inline with a small buffer.
    // Actually, we'll handle compute actions in dispatch_compute_actions
    // during the next tick. For ModuleFetched/Failed, the handle() may
    // produce SendError actions that need dispatching now.
    // We'll accumulate them — see implementation.
}
```

Actually, `push_event` doesn't return actions in the current design. The `ModuleFetchResponse` case should call `self.compute.handle()` and we can dispatch any resulting actions through a stored buffer. However, a simpler approach: have `push_event` for `ModuleFetchResponse` just call `self.compute.handle()` and store any resulting actions in a small buffer that `tick()` drains first.

**Better approach:** Add a `pending_compute_actions: Vec<ComputeTierAction>` field to `NodeRuntime`. `push_event(ModuleFetchResponse)` calls `self.compute.handle()` and appends results to `pending_compute_actions`. `tick()` drains `pending_compute_actions` first before calling `self.compute.tick()`.

Alternatively, since `route_compute_query` already handles this problem for storage (it queues events, doesn't produce actions), we can make compute work the same way: `push_event(ComputeQuery)` parses and calls `self.compute.handle()`, storing fetch actions. But `handle()` returns `Vec<ComputeTierAction>` not `RuntimeAction`.

**Simplest approach:** Add `dispatch_compute_actions()` and call it inline:

```rust
RuntimeEvent::ComputeQuery {
    query_id,
    key_expr,
    payload,
} => {
    let compute_actions = self.route_compute_query(query_id, key_expr, payload);
    // We'll need to buffer these for tick() to return.
    self.pending_compute_actions.extend(compute_actions);
}
RuntimeEvent::ModuleFetchResponse { cid, result } => {
    let event = match result {
        Ok(module) => ComputeTierEvent::ModuleFetched { cid, module },
        Err(_) => ComputeTierEvent::ModuleFetchFailed { cid },
    };
    let actions = self.compute.handle(event);
    self.pending_compute_actions.extend(actions);
}
```

Add `pending_compute_actions: Vec<ComputeTierAction>` to the struct, initialized to `Vec::new()` in the constructor.

**Step 5: Extend tick() with Tier 3**

After the Tier 2 block:

```rust
// Tier 3: dispatch any pending compute actions from push_event(),
// then run one compute slice (lowest priority)
self.dispatch_compute_actions(
    std::mem::take(&mut self.pending_compute_actions),
    &mut actions,
);
let compute_actions = self.compute.tick();
self.dispatch_compute_actions(compute_actions, &mut actions);
```

**Step 6: Add helper methods**

```rust
fn route_compute_query(
    &mut self,
    query_id: u64,
    key_expr: String,
    payload: Vec<u8>,
) -> Vec<ComputeTierAction> {
    let event = match self.parse_compute_payload(query_id, &key_expr, payload) {
        Some(evt) => evt,
        None => {
            return vec![ComputeTierAction::SendError {
                query_id,
                error: "malformed compute payload".into(),
            }];
        }
    };
    self.compute.handle(event)
}

fn parse_compute_payload(
    &self,
    query_id: u64,
    _key_expr: &str,
    payload: Vec<u8>,
) -> Option<ComputeTierEvent> {
    if payload.is_empty() {
        return None;
    }
    match payload[0] {
        // Inline: [0x00] [module_len: u32 LE] [module_bytes] [input_bytes]
        0x00 => {
            if payload.len() < 5 {
                return None;
            }
            let module_len =
                u32::from_le_bytes(payload[1..5].try_into().ok()?) as usize;
            if payload.len() < 5 + module_len {
                return None;
            }
            let module = payload[5..5 + module_len].to_vec();
            let input = payload[5 + module_len..].to_vec();
            Some(ComputeTierEvent::ExecuteInline {
                query_id,
                module,
                input,
            })
        }
        // CID reference: [0x01] [cid: 32 bytes] [input_bytes]
        0x01 => {
            if payload.len() < 33 {
                return None;
            }
            let cid: [u8; 32] = payload[1..33].try_into().ok()?;
            let input = payload[33..].to_vec();
            Some(ComputeTierEvent::ExecuteByCid {
                query_id,
                module_cid: cid,
                input,
            })
        }
        _ => None,
    }
}

fn dispatch_compute_actions(
    &self,
    compute_actions: Vec<ComputeTierAction>,
    out: &mut Vec<RuntimeAction>,
) {
    for action in compute_actions {
        match action {
            ComputeTierAction::SendReply { query_id, payload } => {
                out.push(RuntimeAction::SendReply { query_id, payload });
            }
            ComputeTierAction::FetchModule { cid } => {
                out.push(RuntimeAction::FetchContent { cid });
            }
            ComputeTierAction::SendError { query_id, error } => {
                let mut payload = vec![0x01];
                payload.extend_from_slice(error.as_bytes());
                out.push(RuntimeAction::SendReply { query_id, payload });
            }
        }
    }
}
```

Also update `route_query()` to check compute queryable IDs. In the existing `for action in actions` loop, after the storage check:

```rust
if self.storage_queryable_ids.contains(&queryable_id) {
    // ... existing storage routing ...
} else if self.compute_queryable_ids.contains(&queryable_id) {
    let compute_actions = self.route_compute_query(query_id, key_expr, payload);
    self.pending_compute_actions.extend(compute_actions);
}
```

**Step 7: Write integration tests**

Add these to the existing `mod tests` in `runtime.rs`:

```rust
#[test]
fn startup_declares_compute_queryable() {
    let (_, actions) = make_runtime();
    let compute_queryables: Vec<_> = actions
        .iter()
        .filter(|a| matches!(
            a,
            RuntimeAction::DeclareQueryable { key_expr } if key_expr.starts_with("harmony/compute")
        ))
        .collect();
    assert_eq!(compute_queryables.len(), 1);
}

#[test]
fn compute_query_routes_to_compute_tier() {
    let (mut rt, _) = make_runtime();

    // Build inline compute payload: [0x00] [module_len] [module] [input]
    let module = crate::compute::tests::ADD_WAT.as_bytes();
    let mut input = Vec::new();
    input.extend_from_slice(&5i32.to_le_bytes());
    input.extend_from_slice(&3i32.to_le_bytes());

    let mut payload = vec![0x00];
    payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
    payload.extend_from_slice(module);
    payload.extend_from_slice(&input);

    rt.push_event(RuntimeEvent::ComputeQuery {
        query_id: 99,
        key_expr: "harmony/compute/activity/test".into(),
        payload,
    });

    let actions = rt.tick();
    let reply = actions
        .iter()
        .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. }));
    assert!(reply.is_some(), "compute query should produce a reply");
    if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
        assert_eq!(payload[0], 0x00); // success tag
        let sum = i32::from_le_bytes(payload[1..5].try_into().unwrap());
        assert_eq!(sum, 8);
    }
}

#[test]
fn tick_priority_order_with_compute() {
    let (mut rt, _) = make_runtime();

    // Queue a router event, a storage event, and a compute event
    rt.push_event(RuntimeEvent::TimerTick { now: 1000 });
    rt.push_event(RuntimeEvent::QueryReceived {
        query_id: 10,
        key_expr: "harmony/content/stats".into(),
        payload: vec![],
    });

    // Inline compute payload
    let module = crate::compute::tests::ADD_WAT.as_bytes();
    let mut input = Vec::new();
    input.extend_from_slice(&1i32.to_le_bytes());
    input.extend_from_slice(&2i32.to_le_bytes());
    let mut compute_payload = vec![0x00];
    compute_payload.extend_from_slice(&(module.len() as u32).to_le_bytes());
    compute_payload.extend_from_slice(module);
    compute_payload.extend_from_slice(&input);

    rt.push_event(RuntimeEvent::ComputeQuery {
        query_id: 20,
        key_expr: "harmony/compute/activity/test".into(),
        payload: compute_payload,
    });

    // One tick should process: all router events + one storage + one compute
    let actions = rt.tick();

    // Should have both storage reply and compute reply
    let storage_reply = actions.iter().any(|a| matches!(
        a,
        RuntimeAction::SendReply { query_id: 10, .. }
    ));
    let compute_reply = actions.iter().any(|a| matches!(
        a,
        RuntimeAction::SendReply { query_id: 20, .. }
    ));
    assert!(storage_reply, "should have storage reply");
    assert!(compute_reply, "should have compute reply");
}
```

Note: for `compute_query_routes_to_compute_tier`, the test references `crate::compute::tests::ADD_WAT`. To make that work, change `ADD_WAT` in `compute.rs` tests from `const` to `pub(crate) const`.

**Step 8: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

**Step 9: Commit**

```bash
git add crates/harmony-node/src/runtime.rs crates/harmony-node/src/compute.rs
git commit -m "feat(node): wire ComputeTier into NodeRuntime event loop"
```

---

## Task 6: Update CLI and main.rs for Tier 3

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

This task adds the `--compute-budget` CLI flag and prints compute info on startup.

**Step 1: Add CLI arg**

In the `Run` variant of `Commands`, add:

```rust
/// WASM compute fuel budget per tick
#[arg(long, default_value_t = 100_000)]
compute_budget: u64,
```

**Step 2: Update the run() handler**

In the `Commands::Run` match arm, update `NodeConfig` construction to include `compute_budget`:

```rust
Commands::Run { cache_capacity, compute_budget } => {
    use crate::runtime::{NodeConfig, NodeRuntime, RuntimeAction};
    use harmony_compute::InstructionBudget;
    use harmony_content::blob::MemoryBlobStore;
    use harmony_content::storage_tier::StorageBudget;

    let config = NodeConfig {
        storage_budget: StorageBudget {
            cache_capacity,
            max_pinned_bytes: 100_000_000,
        },
        compute_budget: InstructionBudget { fuel: compute_budget },
    };
    let (rt, startup_actions) = NodeRuntime::new(config, MemoryBlobStore::new());

    println!("Harmony node runtime initialized");
    println!("  Cache capacity:   {cache_capacity} items");
    println!("  Compute budget:   {compute_budget} fuel/tick");
    println!("  Router queue:     {} pending", rt.router_queue_len());
    println!("  Storage queue:    {} pending", rt.storage_queue_len());
    println!("  Compute queue:    {} pending", rt.compute_queue_len());
    // ... rest of startup output unchanged ...
}
```

**Step 3: Update CLI test**

Add a test for parsing `--compute-budget`:

```rust
#[test]
fn cli_parses_run_with_compute_budget() {
    let cli = Cli::try_parse_from(["harmony", "run", "--compute-budget", "50000"]).unwrap();
    if let Commands::Run { compute_budget, .. } = cli.command {
        assert_eq!(compute_budget, 50000);
    } else {
        panic!("expected Run command");
    }
}
```

**Step 4: Run all tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

Run: `cargo clippy -p harmony-node`
Expected: zero warnings

Run: `cargo fmt --all -- --check`
Expected: no formatting issues

**Step 5: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(node): add --compute-budget CLI flag for Tier 3"
```

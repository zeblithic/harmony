# Tier 3 Compute Integration Design

## Goal

Integrate harmony-compute into harmony-node's event loop, completing the Router/Storage/Compute trinity with three-tier priority scheduling.

## Architecture

A new `ComputeTier` sans-I/O state machine in harmony-node wraps `WasmiRuntime` with a task queue and auto-resume logic. `NodeRuntime` holds a `ComputeTier` alongside the existing router and storage, extending `tick()` with a third priority level: drain all router, then one storage event, then one compute slice.

## Key Decisions

- **Separate ComputeTier state machine** — independently testable, follows StorageTier pattern
- **Fixed configurable fuel budget** — `NodeConfig` gets a `compute_budget: InstructionBudget` field; no adaptive budgeting (future bead)
- **Both inline and CID-referenced modules** — tag-prefixed payload format distinguishes them
- **CID resolution via emit-fetch-and-retry** — if module CID isn't local, emit `FetchModule` action; when response arrives, promote task to Ready
- **One active execution + FIFO queue** — matches WasmiRuntime's single-session design
- **Auto-resume on yield** — yielded tasks resume on the next tick automatically; caller sees only the final result
- **NeedsIO → SendError** — not implemented yet (future bead)

## ComputeTier State Machine

### Events (inputs)

```rust
pub enum ComputeTierEvent {
    ExecuteInline { query_id: u64, activity_type: String, module: Vec<u8>, input: Vec<u8> },
    ExecuteByCid { query_id: u64, activity_type: String, module_cid: [u8; 32], input: Vec<u8> },
    ModuleFetched { cid: [u8; 32], module: Vec<u8> },
    ModuleFetchFailed { cid: [u8; 32] },
}
```

### Actions (outputs)

```rust
pub enum ComputeTierAction {
    SendReply { query_id: u64, payload: Vec<u8> },
    FetchModule { cid: [u8; 32] },
    SendError { query_id: u64, error: String },
}
```

### Internal State

```rust
enum ComputeTask {
    Ready { query_id: u64, module: Vec<u8>, input: Vec<u8> },
    WaitingForModule { query_id: u64, cid: [u8; 32], input: Vec<u8> },
}

struct ActiveExecution {
    query_id: u64,
}

pub struct ComputeTier {
    runtime: WasmiRuntime,
    queue: VecDeque<ComputeTask>,
    active: Option<ActiveExecution>,
    budget: InstructionBudget,
}
```

### tick() Lifecycle

1. If no active execution and queue has a Ready task → dequeue, call `runtime.execute()`, store ActiveExecution
2. If active execution exists → call `runtime.resume(budget)`
3. Match ComputeResult:
   - Complete → emit SendReply, clear active
   - Yielded → do nothing (auto-resume next tick)
   - Failed → emit SendError, clear active
   - NeedsIO → emit SendError (not yet supported)

### handle() Event Processing

- ExecuteInline → push Ready task to queue
- ExecuteByCid → push WaitingForModule task, emit FetchModule
- ModuleFetched → find matching WaitingForModule, promote to Ready
- ModuleFetchFailed → find matching task, emit SendError, remove

## NodeRuntime Integration

### Config

```rust
pub struct NodeConfig {
    pub storage_budget: StorageBudget,
    pub compute_budget: InstructionBudget,
}
```

### New RuntimeEvent Variants

```rust
ComputeQuery { query_id: u64, key_expr: String, payload: Vec<u8> },
ModuleFetchResponse { cid: [u8; 32], result: Result<Vec<u8>, String> },
```

### New RuntimeAction Variant

```rust
FetchContent { cid: [u8; 32] },
```

### Extended tick()

```rust
// Tier 1: drain ALL router events
// Tier 2: process ONE storage event
// Tier 3: one compute slice
let compute_actions = self.compute.tick();
self.dispatch_compute_actions(compute_actions, &mut actions);
```

### Query Routing

`route_query()` checks if the queryable ID belongs to `compute_queryable_ids`. If so, parse the activity type from the key expression and the payload tag byte, then push the appropriate `ComputeTierEvent`.

### Startup

Register one queryable on `harmony/compute/activity/*`.

## Payload Wire Format

```
Inline:  [0x00] [module_len: u32 LE] [module_bytes] [input_bytes]
CID ref: [0x01] [cid: 32 bytes] [input_bytes]

Reply success: [0x00] [output_bytes]
Reply error:   [0x01] [error_message_utf8]
```

## Out of Scope (YAGNI)

- Adaptive budgeting (scale fuel with router load / battery)
- NeedsIO handling (emit SendError for now)
- Capacity advertisement on `harmony/compute/capacity/{node_addr}`
- Result publishing on `harmony/compute/result/{id}`
- Workflow/checkpoint storage via Zenoh
- Multiple WasmiRuntime pool
- wasmtime JIT runtime

## Files

- Create: `crates/harmony-node/src/compute.rs`
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml`
- Modify: `crates/harmony-node/src/main.rs`

## Testing

ComputeTier (9 tests):
1. execute_inline_completes
2. execute_invalid_module_returns_error
3. execute_by_cid_emits_fetch
4. module_fetched_promotes_to_ready
5. module_fetch_failed_returns_error
6. yielded_task_auto_resumes
7. multiple_tasks_queued_fifo
8. tick_with_no_tasks_returns_nothing
9. malformed_payload_returns_error

NodeRuntime integration (4 tests):
10. compute_query_routes_to_compute_tier
11. startup_declares_compute_queryable
12. tick_priority_order
13. compute_inline_round_trip

# harmony-workflow: Durable Execution Orchestration

## Goal

Add `harmony-workflow` crate providing Temporal-inspired durable execution over the existing `ComputeTier`. Event-sourced history enables crash recovery via deterministic WASM replay. Compute offload types are defined but execute locally for now.

## Architecture

`WorkflowEngine` wraps `ComputeTier` as a thin orchestration layer. It intercepts compute-tier IO actions, records them in an event log, and re-emits them as workflow-level actions. On recovery, the engine replays the WASM module from scratch, feeding cached IO responses from the persisted history. WASM determinism guarantees identical execution paths.

Dependency graph:
```
harmony-compute (existing)
  └── harmony-workflow (new — depends on harmony-compute + harmony-crypto)
      └── harmony-node extends to use WorkflowEngine instead of raw ComputeTier
```

## Key Decisions

- **Event-sourced history** (not WASM memory snapshots) — the checkpoint IS the event log. Works uniformly across wasmi and wasmtime runtimes. Same pattern as wasmtime's NeedsIO replay, lifted to persist across process boundaries.
- **Single-activity workflows initially** — types designed for future multi-step DAG orchestration, but only single WASM execution implemented now.
- **Content-addressed workflow IDs** — `BLAKE3(module_hash || input)`. Deterministic, enables deduplication across nodes.
- **Local-only offload** — `ComputeHint` and `OffloadDecision` types defined. Evaluator always returns `ExecuteLocally`. Real offload requires Zenoh peer discovery.
- **In-memory with serializable state** — engine holds event logs in memory. Provides `take_history()`/`restore_history()` for external persistence. Sans-I/O preserved.

## Core Types

```rust
/// Content-addressed workflow identifier.
/// BLAKE3(module_hash || input) — deterministic, enables deduplication.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkflowId([u8; 32]);

/// A single recorded IO event in the workflow's history.
#[derive(Debug, Clone)]
pub enum HistoryEvent {
    /// WASM requested content by CID.
    IoRequested { cid: [u8; 32] },
    /// Content was resolved (found or not found).
    IoResolved { cid: [u8; 32], data: Option<Vec<u8>> },
}

/// The complete event log for a workflow — this IS the checkpoint.
#[derive(Debug, Clone)]
pub struct WorkflowHistory {
    pub workflow_id: WorkflowId,
    pub module_hash: [u8; 32],
    pub input: Vec<u8>,
    pub events: Vec<HistoryEvent>,
    pub total_fuel_consumed: u64,
}

/// Current state of a workflow in the engine.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkflowStatus {
    Pending,
    Executing,
    WaitingForIo { cid: [u8; 32] },
    Complete,
    Failed,
}

/// Hint from the caller about execution preferences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeHint {
    PreferLocal,
    PreferPowerful,
    LatencySensitive,
    DurabilityRequired,
}

/// Decision from the offload evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDecision {
    ExecuteLocally,
    OffloadTo { peer: [u8; 16] },
}
```

## WorkflowEngine State Machine

```rust
pub struct WorkflowEngine {
    compute: ComputeTier,
    workflows: HashMap<WorkflowId, WorkflowState>,
    query_to_workflow: HashMap<u64, WorkflowId>,
    next_query_id: u64,
}

struct WorkflowState {
    status: WorkflowStatus,
    history: WorkflowHistory,
    hint: ComputeHint,
}
```

### Events (inbound)

```rust
pub enum WorkflowEvent {
    Submit { module: Vec<u8>, input: Vec<u8>, hint: ComputeHint },
    SubmitByCid { module_cid: [u8; 32], input: Vec<u8>, hint: ComputeHint },
    ModuleFetched { cid: [u8; 32], module: Vec<u8> },
    ModuleFetchFailed { cid: [u8; 32] },
    ContentFetched { cid: [u8; 32], data: Vec<u8> },
    ContentFetchFailed { cid: [u8; 32] },
}
```

### Actions (outbound)

```rust
pub enum WorkflowAction {
    FetchModule { cid: [u8; 32] },
    FetchContent { workflow_id: WorkflowId, cid: [u8; 32] },
    WorkflowComplete { workflow_id: WorkflowId, output: Vec<u8> },
    WorkflowFailed { workflow_id: WorkflowId, error: String },
    PersistHistory { workflow_id: WorkflowId },
}
```

### Key methods

- `submit()` — compute WorkflowId, create WorkflowState, delegate to ComputeTier
- `handle(event)` — translate workflow events to compute tier events, record history
- `tick()` — call compute.tick(), intercept actions, record IO events, re-emit as workflow actions
- `take_history(id)` / `restore_history(history)` — extract/restore for persistence
- `workflow_status(id)` — query current status
- `recover(history, module)` — replay from persisted event log

### Interaction flow

```
WorkflowEngine.tick()
  → ComputeTier.tick()
    → ComputeResult::NeedsIO { FetchContent { cid } }
  → ComputeTier returns ComputeTierAction::FetchContent { cid }
  → WorkflowEngine records HistoryEvent::IoRequested { cid }
  → WorkflowEngine emits WorkflowAction::FetchContent { workflow_id, cid }
  → WorkflowEngine emits WorkflowAction::PersistHistory { workflow_id }

[External IO resolves]

WorkflowEngine.handle(ContentFetched { cid, data })
  → WorkflowEngine records HistoryEvent::IoResolved { cid, data }
  → Delegates to ComputeTier.handle(ContentFetched { cid, data })
  → ComputeTier resumes WASM → Complete
  → WorkflowEngine emits WorkflowAction::WorkflowComplete { ... }
  → WorkflowEngine emits WorkflowAction::PersistHistory { ... }
```

## Recovery via Replay

When a node restarts and loads a persisted `WorkflowHistory`:

1. `recover(history, module_bytes)` creates WorkflowState with existing event log
2. Re-submits module+input to ComputeTier
3. As WASM re-executes and hits `fetch_content`:
   - ComputeTier returns `FetchContent { cid }`
   - Engine checks: is this CID in `history.events` as `IoResolved`?
   - **Hit**: immediately feed `ContentFetched { cid, data }` back (no external fetch)
   - **Miss**: new IO beyond recorded history — emit `FetchContent` normally
4. Module completes → `WorkflowComplete`

Recovery requires the module bytes (fetched by CID = module_hash). The history stores `module_hash` but not the full module to avoid bloat.

## Compute Offload

```rust
pub fn evaluate_offload(hint: ComputeHint) -> OffloadDecision {
    // Stub: always execute locally. Real implementation needs:
    // - Battery level, CPU load, power source (platform signals)
    // - Network latency to peers (Zenoh RTT)
    // - Peer advertised capacity (Zenoh queryable)
    OffloadDecision::ExecuteLocally
}
```

Types defined now. Implementation deferred until Zenoh peer discovery is wired.

## Node Integration

`NodeRuntime` replaces `ComputeTier` with `WorkflowEngine`:

```rust
pub struct NodeRuntime<B: BlobStore> {
    workflow: WorkflowEngine,  // was: compute: ComputeTier
    // ...
}
```

Event/action translation:
- `ComputeQuery` → `WorkflowEvent::Submit` / `SubmitByCid`
- `ContentFetchResponse` → `WorkflowEvent::ContentFetched` / `ContentFetchFailed`
- `WorkflowAction::FetchContent` → `RuntimeAction::FetchContent`
- `WorkflowAction::WorkflowComplete` → `RuntimeAction::SendReply`
- `WorkflowAction::PersistHistory` → no-op (future: Zenoh persistence)

## Crate Structure

```
crates/harmony-workflow/
├── Cargo.toml
└── src/
    ├── lib.rs
    ├── types.rs        # WorkflowId, HistoryEvent, WorkflowHistory, etc.
    ├── engine.rs       # WorkflowEngine state machine
    ├── offload.rs      # ComputeHint, OffloadDecision, evaluate()
    └── error.rs        # WorkflowError
```

## Testing

harmony-workflow unit tests (~12 tests):
1. `submit_creates_workflow_with_id` — deterministic ID generation
2. `submit_delegates_to_compute_tier` — verify compute tier receives the task
3. `tick_completes_simple_workflow` — end-to-end for a non-IO module
4. `io_request_recorded_in_history` — verify HistoryEvent::IoRequested logged
5. `io_response_recorded_in_history` — verify HistoryEvent::IoResolved logged
6. `content_fetched_resumes_workflow` — full IO round trip
7. `recover_replays_from_history` — recovery feeds cached IO, completes
8. `recover_with_new_io_beyond_history` — replay hits uncached CID, emits FetchContent
9. `workflow_id_deterministic` — same module+input = same ID
10. `duplicate_submit_deduplicates` — same ID doesn't create second workflow
11. `compute_hint_stored_on_workflow` — verify hint is preserved
12. `offload_decision_always_local` — verify stub returns ExecuteLocally

harmony-node tests: existing compute integration tests adapted to use WorkflowEngine.

CI: `cargo test -p harmony-workflow` + `cargo test -p harmony-node`.

## Out of Scope (YAGNI)

- Multi-step workflow DAG orchestration
- Real offload to remote peers
- Zenoh-backed history persistence
- Workflow cancellation / timeout
- Module pre-compilation caching
- Workflow versioning / migration

## Files

- Create: `crates/harmony-workflow/Cargo.toml`
- Create: `crates/harmony-workflow/src/lib.rs`
- Create: `crates/harmony-workflow/src/types.rs`
- Create: `crates/harmony-workflow/src/engine.rs`
- Create: `crates/harmony-workflow/src/offload.rs`
- Create: `crates/harmony-workflow/src/error.rs`
- Modify: `Cargo.toml` (workspace members + dependency)
- Modify: `crates/harmony-node/Cargo.toml` (add harmony-workflow dep)
- Modify: `crates/harmony-node/src/runtime.rs` (use WorkflowEngine)

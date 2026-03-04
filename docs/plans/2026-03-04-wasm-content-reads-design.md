# WASM Content Read Access via NeedsIO

## Goal

Allow WASM compute tasks to read immutable content from the content-addressed storage system by CID, turning the NeedsIO path from an error into a working content resolution pipeline.

## Architecture

A WASM module calls the `harmony.fetch_content` host import to request content by CID. The wasmi runtime traps (suspends execution), the ComputeTier emits a FetchContent action, external I/O resolves the CID, and the runtime resumes by writing the fetched data into the module's pre-allocated buffer. The task is non-blocking — other compute tasks can execute while IO is pending.

## Key Decisions

- **Trap-and-resume** — host function traps via wasmi's resumable HostTrap API; module code is synchronous (call returns with data). No polling or yield-and-retry.
- **Caller-provides-buffer ABI** — `hm_fetch_content(cid_ptr, out_ptr, out_cap) -> i32`. Module allocates output buffer, host writes into it. Returns bytes written or negative error code.
- **Non-blocking IO** — suspended task moves to `WaitingForContent` state, clearing `active` so other Ready tasks can execute. Matches `WaitingForModule` pattern.
- **One IO request at a time** — a task can only have one outstanding fetch. The module blocks until it resolves, then can issue another.

## Host Function ABI

WASM modules import from the `harmony` namespace:

```wat
(import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
```

**Parameters:**
- `cid_ptr: i32` — pointer to 32-byte CID in WASM linear memory
- `out_ptr: i32` — pointer to output buffer (module-allocated)
- `out_cap: i32` — capacity of output buffer in bytes

**Return value:**
- `>= 0` — bytes written to output buffer (success)
- `-1` — content not found
- `-2` — buffer too small (content larger than `out_cap`)

## WasmiRuntime Changes (harmony-compute)

### HostState

```rust
struct HostState {
    io_request: Option<IORequest>,
    io_write_target: Option<(u32, u32)>,  // (out_ptr, out_cap)
}
```

### Host function registration

The Linker registers `harmony::fetch_content`. When called:
1. Read 32 bytes from WASM memory at `cid_ptr`
2. Store `IORequest::FetchContent { cid }` and write target `(out_ptr, out_cap)` in HostState
3. Return `Err(wasmi::Error)` (trap) to suspend execution

### PendingResumable

Replaces `Option<TypedResumableCallOutOfFuel<i32>>`:

```rust
enum PendingResumable {
    OutOfFuel(TypedResumableCallOutOfFuel<i32>),
    HostTrap(TypedResumableCallHostTrap<i32>),
}
```

### handle_call_result HostTrap arm

Reads `io_request` from HostState. If present, returns `ComputeResult::NeedsIO`. If absent (unexpected trap), returns `Failed`.

### New ComputeRuntime method

```rust
fn resume_with_io(&mut self, response: IOResponse, budget: InstructionBudget) -> ComputeResult;
```

Takes stored HostTrap pending invocation and `(out_ptr, out_cap)` from HostState:
- `ContentReady { data }` where `data.len() <= out_cap`: write data to WASM memory at `out_ptr`, resume with `&[Val::I32(data.len())]`
- `ContentReady { data }` where `data.len() > out_cap`: resume with `&[Val::I32(-2)]` (no write)
- `ContentNotFound`: resume with `&[Val::I32(-1)]`

### New types

```rust
pub enum IOResponse {
    ContentReady { data: Vec<u8> },
    ContentNotFound,
}
```

## ComputeTier Changes (harmony-node)

### New task state

```rust
WaitingForContent { query_id: u64, cid: [u8; 32] }
```

Execution state lives in `WasmiRuntime.session` (HostTrap pending invocation stored there). Task tracks `query_id` and `cid` for matching.

### New events

```rust
ContentFetched { cid: [u8; 32], data: Vec<u8> },
ContentFetchFailed { cid: [u8; 32] },
```

### New action

```rust
FetchContent { query_id: u64, cid: [u8; 32] },
```

### handle_compute_result NeedsIO arm

1. Extract CID from `IORequest::FetchContent`
2. Push `WaitingForContent { query_id, cid }` to queue
3. Clear `active`
4. Emit `FetchContent { query_id, cid }`

### handle() for ContentFetched

1. Find `WaitingForContent` task matching `cid`
2. Call `runtime.resume_with_io(ContentReady { data }, budget)`
3. Set as `active`, handle the compute result

### handle() for ContentFetchFailed

1. Find matching `WaitingForContent` task
2. Call `runtime.resume_with_io(ContentNotFound, budget)`
3. Handle compute result (module sees `-1`)

### tick() behavior

Unchanged. `WaitingForContent` tasks skipped during dequeue (same as `WaitingForModule`).

## NodeRuntime Changes

### New RuntimeEvent variant

```rust
ContentFetchResponse { cid: [u8; 32], result: Result<Vec<u8>, String> },
```

### push_event routing

`ContentFetchResponse` maps to `ComputeTierEvent::ContentFetched` or `ContentFetchFailed`.

### dispatch_compute_actions

`FetchContent { query_id, cid }` maps to `RuntimeAction::FetchContent { cid }` (reuses existing variant).

## Data Flow

```
WASM calls: harmony.fetch_content(cid_ptr, out_ptr, out_cap)
  → Host function traps with IORequest::FetchContent { cid }
  → WasmiRuntime returns ComputeResult::NeedsIO { request }
  → ComputeTier: clear active, push WaitingForContent, emit FetchContent action
  → NodeRuntime: dispatch → RuntimeAction::FetchContent { cid }
  [External I/O resolves CID via storage/network]
  → RuntimeEvent::ContentFetchResponse { cid, Ok(data) }
  → ComputeTier: handle ContentFetched → runtime.resume_with_io(ContentReady { data })
  → Host writes data into WASM memory at out_ptr
  → wasmi resumes, host function "returns" data.len() to WASM
  → WASM module continues with the content in its buffer
```

## Out of Scope (YAGNI)

- Multiple concurrent IO requests from one task
- Local ContentStore shortcut (check cache before emitting FetchContent)
- Write access to content storage from WASM
- Bundle traversal (fetching children automatically)
- Size-prefixed responses

## Files

- Modify: `crates/harmony-compute/src/types.rs`
- Modify: `crates/harmony-compute/src/runtime.rs`
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs`
- Modify: `crates/harmony-node/src/compute.rs`
- Modify: `crates/harmony-node/src/runtime.rs`

## Testing

harmony-compute (5 tests):
1. host_function_triggers_needs_io
2. resume_with_content_ready
3. resume_with_content_not_found
4. resume_with_buffer_too_small
5. fetch_content_then_complete

harmony-node ComputeTier (4 tests):
6. needs_io_emits_fetch_content
7. content_fetched_resumes_task
8. content_fetch_failed_returns_error
9. waiting_for_content_does_not_block_queue

harmony-node NodeRuntime (2 tests):
10. content_fetch_response_routes_to_compute
11. compute_content_read_round_trip

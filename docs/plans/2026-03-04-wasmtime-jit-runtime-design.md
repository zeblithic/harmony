# Wasmtime JIT Runtime for Desktop/Server Compute

## Goal

Add `WasmtimeRuntime` as an alternative `ComputeRuntime` implementation using JIT compilation, feature-gated behind `wasmtime`. Desktop and server nodes get faster WASM execution; embedded/mobile nodes continue using the `WasmiRuntime` interpreter.

## Architecture

`WasmtimeRuntime` implements the existing `ComputeRuntime` trait with two key differences from `WasmiRuntime`:

1. **No cooperative yielding** — wasmtime has no resumable-call API. Modules run to completion or fail on fuel exhaustion. Desktop/server nodes have abundant CPU, so we give a large fuel budget.

2. **NeedsIO via deterministic replay** — When `harmony.fetch_content` traps, the runtime returns `NeedsIO`. On `resume_with_io()`, the module is re-executed from scratch with the IO result cached in `HostState`. The host function checks the cache — if the CID matches, it writes cached data and returns immediately (no trap). New CIDs trap again for another NeedsIO round. WASM determinism guarantees identical execution paths.

## Key Decisions

- **Box<dyn ComputeRuntime>** — `ComputeTier` uses trait object dispatch instead of generics. Vtable overhead is negligible vs WASM execution cost. Avoids generic infection across the crate hierarchy.
- **Fuel metering** (not epoch interruption) — deterministic, same API as wasmi, simpler to reason about.
- **Compile-time feature selection** — `wasmtime` cargo feature, not runtime hardware detection. Simpler, sufficient for the two-tier model (desktop vs embedded).
- **Deterministic replay for IO** — Re-execute from scratch with cached IO results. JIT makes this fast. Avoids needing async or resumable call support.

## WasmtimeRuntime Internal State

```rust
struct WasmtimeSession {
    engine: wasmtime::Engine,
    module_bytes: Vec<u8>,
    input: Vec<u8>,
    io_cache: HashMap<[u8; 32], Vec<u8>>,  // CID → fetched data
    total_fuel_consumed: u64,
}
```

The `engine` is created once with `consume_fuel(true)`. On each `execute()` or replay, a fresh `Store` and `Instance` are created from the compiled module.

## Host Function ABI

Same as WasmiRuntime — modules import `harmony::fetch_content`:

```wat
(import "harmony" "fetch_content" (func $fetch (param i32 i32 i32) (result i32)))
```

Host function behavior:
1. Read 32-byte CID from WASM memory at `cid_ptr`
2. Check `HostState.io_cache` for the CID
3. **Cache hit**: write data to `out_ptr`, return `data.len() as i32` (or `-2` if buffer too small)
4. **Cache miss**: store `IORequest::FetchContent { cid }` and write target in HostState, return `Err` (trap)

## ComputeRuntime Implementation

### execute()

1. Compile module with `wasmtime::Module::new()`
2. Create `Store` with fuel budget, register `harmony::fetch_content` via `Linker`
3. Write input to WASM memory, call `compute`
4. **Success**: read output, return `Complete`
5. **Out of fuel**: return `Failed` (no yielding — runs to completion or fails)
6. **Host trap with IO request**: store module bytes + input + IO state in session, return `NeedsIO`

### resume_with_io()

1. Cache the IO response data in `session.io_cache` (keyed by CID)
2. Re-execute the module from scratch (fresh Store + Instance)
3. Host function finds CID in cache → returns data immediately (no trap)
4. If a new `fetch_content` call hits a cache miss → NeedsIO again
5. If module completes → return `Complete`

### resume()

Returns `ComputeResult::Failed { error: NoPendingExecution }`. Wasmtime cannot resume from fuel exhaustion.

### has_pending()

Returns `true` when a session exists with a pending IO request (after NeedsIO, before resume_with_io completes).

### take_session() / restore_session()

Extract/restore `WasmtimeSession` as `Box<dyn Any>`, same pattern as WasmiRuntime.

### snapshot()

Captures module hash and total fuel consumed from the session.

## Feature Gating

```toml
# harmony-compute/Cargo.toml
[features]
default = []
wasmtime = ["dep:wasmtime"]

[dependencies]
wasmtime = { version = "38", optional = true }
```

```toml
# harmony-node/Cargo.toml
[features]
default = []
wasmtime = ["harmony-compute/wasmtime"]
```

`wasmtime_runtime` module is `#[cfg(feature = "wasmtime")]`.

## ComputeTier Changes

```rust
pub struct ComputeTier {
    runtime: Box<dyn ComputeRuntime>,  // was: WasmiRuntime
    queue: VecDeque<ComputeTask>,
    active: Option<ActiveExecution>,
    budget: InstructionBudget,
}

impl ComputeTier {
    pub fn new(runtime: Box<dyn ComputeRuntime>, budget: InstructionBudget) -> Self
}
```

harmony-node selects runtime at compile time:

```rust
#[cfg(feature = "wasmtime")]
let runtime: Box<dyn ComputeRuntime> = Box::new(WasmtimeRuntime::new());

#[cfg(not(feature = "wasmtime"))]
let runtime: Box<dyn ComputeRuntime> = Box::new(WasmiRuntime::new());
```

## Data Flow (NeedsIO with Replay)

```
WASM calls: harmony.fetch_content(cid_ptr, out_ptr, out_cap)
  → Host checks io_cache: miss
  → Host stores IORequest, traps
  → WasmtimeRuntime: store module_bytes + input in session, return NeedsIO
  → ComputeTier: WaitingForContent, emit FetchContent action
  [External IO resolves CID]
  → resume_with_io(ContentReady { data })
  → Cache data in session.io_cache[cid]
  → Re-execute module from scratch (JIT — fast)
  → WASM calls fetch_content again (deterministic replay)
  → Host checks io_cache: HIT → writes data, returns len
  → WASM continues with content in buffer
  → Module completes → return Complete
```

## Out of Scope (YAGNI)

- Runtime hardware detection for auto-selection
- Epoch-based interruption
- Module pre-compilation / caching
- Async wasmtime support
- Cooperative yielding (resume from fuel exhaustion)

## Files

- Create: `crates/harmony-compute/src/wasmtime_runtime.rs`
- Modify: `crates/harmony-compute/src/lib.rs`
- Modify: `crates/harmony-compute/Cargo.toml`
- Modify: `crates/harmony-node/src/compute.rs`
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml`

## Testing

harmony-compute wasmtime tests (7 tests, `#[cfg(feature = "wasmtime")]`):
1. execute_add_module — basic execution
2. execute_completes_without_yielding — confirms no Yielded result
3. resume_returns_no_pending — resume() always fails
4. host_function_triggers_needs_io — fetch_content traps on cache miss
5. resume_with_io_replays_with_content — replay delivers cached data
6. resume_with_io_content_not_found — replay with -1 return
7. fetch_content_replay_then_complete — end-to-end IO round trip

harmony-node ComputeTier tests (update existing):
- All existing tests continue to work with Box<dyn ComputeRuntime>
- No new node-level tests needed (runtime is transparent behind trait)

CI: `cargo test -p harmony-compute` (wasmi) + `cargo test -p harmony-compute --features wasmtime` (both).

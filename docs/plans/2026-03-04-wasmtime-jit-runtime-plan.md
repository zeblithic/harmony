# Wasmtime JIT Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `WasmtimeRuntime` as an alternative `ComputeRuntime` using JIT compilation, feature-gated behind `wasmtime`.

**Architecture:** WasmtimeRuntime implements ComputeRuntime with fuel metering but no cooperative yielding (runs to completion). NeedsIO is handled via deterministic replay — re-execute from scratch with cached IO results. ComputeTier changes from concrete `WasmiRuntime` to `Box<dyn ComputeRuntime>`.

**Tech Stack:** Rust, wasmtime 38, wasmi 1 (existing), cargo feature flags

**Design doc:** `docs/plans/2026-03-04-wasmtime-jit-runtime-design.md`

---

### Task 1: Make ComputeTier use Box<dyn ComputeRuntime>

Refactor ComputeTier from concrete `WasmiRuntime` to trait object dispatch. This is a prerequisite for plugging in WasmtimeRuntime later.

**Files:**
- Modify: `crates/harmony-node/src/compute.rs`
- Modify: `crates/harmony-node/src/runtime.rs`

**Context:** Currently `ComputeTier` has `runtime: WasmiRuntime` (concrete). After this task, it takes `Box<dyn ComputeRuntime>`. All existing tests must continue to pass — they'll construct with `Box::new(WasmiRuntime::new())`.

**Step 1: Update ComputeTier struct and constructor**

In `crates/harmony-node/src/compute.rs`, change:

```rust
// Before:
use harmony_compute::{ComputeRuntime, InstructionBudget, WasmiRuntime};

pub struct ComputeTier {
    runtime: WasmiRuntime,
    ...
}

impl ComputeTier {
    pub fn new(budget: InstructionBudget) -> Self {
        Self {
            runtime: WasmiRuntime::new(),
            ...
        }
    }
}

// After:
use harmony_compute::{ComputeRuntime, InstructionBudget};

pub struct ComputeTier {
    runtime: Box<dyn ComputeRuntime>,
    ...
}

impl ComputeTier {
    pub fn new(runtime: Box<dyn ComputeRuntime>, budget: InstructionBudget) -> Self {
        Self {
            runtime,
            ...
        }
    }
}
```

**Step 2: Update all ComputeTier::new() call sites**

In `crates/harmony-node/src/compute.rs` tests, change `make_tier()`:

```rust
fn make_tier() -> ComputeTier {
    ComputeTier::new(
        Box::new(harmony_compute::WasmiRuntime::new()),
        InstructionBudget { fuel: 100_000 },
    )
}
```

And the `yielded_task_auto_resumes` test:

```rust
let mut tier = ComputeTier::new(
    Box::new(harmony_compute::WasmiRuntime::new()),
    InstructionBudget { fuel: 10_000 },
);
```

In `crates/harmony-node/src/runtime.rs`, change the NodeRuntime constructor:

```rust
// Before:
let compute = ComputeTier::new(config.compute_budget);

// After:
let compute = ComputeTier::new(
    Box::new(harmony_compute::WasmiRuntime::new()),
    config.compute_budget,
);
```

**Step 3: Run tests**

Run: `cargo test -p harmony-node`
Expected: All 42 tests pass (no behavior change, only constructor signature).

**Step 4: Commit**

```bash
git add crates/harmony-node/src/compute.rs crates/harmony-node/src/runtime.rs
git commit -m "refactor(node): make ComputeTier accept Box<dyn ComputeRuntime>"
```

---

### Task 2: Add wasmtime dependency and feature gate

Set up the cargo feature plumbing so `wasmtime_runtime` module compiles conditionally.

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/harmony-compute/Cargo.toml`
- Modify: `crates/harmony-compute/src/lib.rs`
- Modify: `crates/harmony-node/Cargo.toml`

**Step 1: Add wasmtime to workspace dependencies**

In `Cargo.toml` (workspace root), add to `[workspace.dependencies]`:

```toml
wasmtime = "38"
```

**Step 2: Add feature gate to harmony-compute**

In `crates/harmony-compute/Cargo.toml`, add:

```toml
[features]
default = []
wasmtime = ["dep:wasmtime"]

[dependencies]
wasmtime = { workspace = true, optional = true }
```

(Keep the existing `harmony-crypto`, `thiserror`, `wasmi` deps unchanged.)

**Step 3: Add conditional module to lib.rs**

In `crates/harmony-compute/src/lib.rs`, add:

```rust
#[cfg(feature = "wasmtime")]
pub mod wasmtime_runtime;
#[cfg(feature = "wasmtime")]
pub use wasmtime_runtime::WasmtimeRuntime;
```

**Step 4: Create empty module stub**

Create `crates/harmony-compute/src/wasmtime_runtime.rs`:

```rust
//! WASM execution engine backed by wasmtime with JIT compilation.
//!
//! Alternative to `WasmiRuntime` for desktop/server nodes. Uses JIT compilation
//! for faster execution. Does not support cooperative yielding — modules run to
//! completion or fail on fuel exhaustion. NeedsIO is handled via deterministic
//! replay with a cached IO oracle.
```

**Step 5: Add feature propagation to harmony-node**

In `crates/harmony-node/Cargo.toml`, add:

```toml
[features]
default = []
wasmtime = ["harmony-compute/wasmtime"]
```

**Step 6: Verify compilation**

Run: `cargo check -p harmony-compute --features wasmtime`
Expected: Compiles (empty module).

Run: `cargo test -p harmony-compute`
Expected: All 33 tests pass (no wasmtime feature, no changes to existing code).

**Step 7: Commit**

```bash
git add Cargo.toml crates/harmony-compute/Cargo.toml crates/harmony-compute/src/lib.rs \
  crates/harmony-compute/src/wasmtime_runtime.rs crates/harmony-node/Cargo.toml
git commit -m "feat(compute): add wasmtime feature gate and empty module stub"
```

---

### Task 3: WasmtimeRuntime basic execute()

Implement `WasmtimeRuntime` with `execute()`, `resume()`, `has_pending()`, `take_session()`, `restore_session()`, and `snapshot()`. No NeedsIO yet — host function is not registered.

**Files:**
- Modify: `crates/harmony-compute/src/wasmtime_runtime.rs`

**Context:** Follow the same WASM ABI as WasmiRuntime:
- Module exports `memory` (linear memory) and `compute(input_ptr: i32, input_len: i32) -> i32`
- Input bytes written to memory at offset 0
- Output starts at `input_len`, function returns output byte count

The wasmtime API uses `Config::consume_fuel(true)`, `Store::set_fuel()`, `Store::get_fuel()`, `Linker::func_wrap()`, `Module::new()`, `Linker::instantiate()`, `instance.get_typed_func()`, `func.call()`, `Memory::read()/write()`.

Key difference from wasmi: when fuel runs out, `func.call()` returns `Err(anyhow::Error)` with a "fuel" trap. There's no resumable call — `resume()` always returns `NoPendingExecution`.

**Step 1: Write test for basic execution**

At the bottom of `wasmtime_runtime.rs`, add:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ComputeRuntime;
    use crate::types::InstructionBudget;

    /// Same ADD_WAT as wasmi_runtime tests and compute tier tests.
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

    #[test]
    fn execute_add_module() {
        let mut rt = WasmtimeRuntime::new();
        let mut input = Vec::new();
        input.extend_from_slice(&3i32.to_le_bytes());
        input.extend_from_slice(&7i32.to_le_bytes());

        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::Complete { output } => {
                let sum = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(sum, 10);
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
    }

    #[test]
    fn execute_completes_without_yielding() {
        let mut rt = WasmtimeRuntime::new();
        let input = 100_i32.to_le_bytes().to_vec();

        // LOOP_WAT: loops `count` times, writes count as output.
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

        // Give enough fuel to complete — should NOT yield.
        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );

        assert!(
            matches!(result, crate::types::ComputeResult::Complete { .. }),
            "expected Complete (no yielding in wasmtime), got: {result:?}"
        );
    }

    #[test]
    fn fuel_exhaustion_returns_failed() {
        let mut rt = WasmtimeRuntime::new();
        let input = 1_000_000_i32.to_le_bytes().to_vec();

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

        // Tiny fuel budget — must exhaust before loop completes.
        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 100 },
        );

        assert!(
            matches!(result, crate::types::ComputeResult::Failed { .. }),
            "expected Failed on fuel exhaustion, got: {result:?}"
        );
    }

    #[test]
    fn resume_returns_no_pending() {
        let mut rt = WasmtimeRuntime::new();
        let result = rt.resume(InstructionBudget { fuel: 1000 });
        assert!(matches!(
            result,
            crate::types::ComputeResult::Failed {
                error: crate::error::ComputeError::NoPendingExecution
            }
        ));
    }

    #[test]
    fn has_pending_is_false_after_complete() {
        let mut rt = WasmtimeRuntime::new();
        let mut input = Vec::new();
        input.extend_from_slice(&1i32.to_le_bytes());
        input.extend_from_slice(&2i32.to_le_bytes());
        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::Complete { .. }));
        assert!(!rt.has_pending());
    }
}
```

**Step 2: Run tests to see them fail**

Run: `cargo test -p harmony-compute --features wasmtime`
Expected: FAIL — `WasmtimeRuntime` not defined.

**Step 3: Implement WasmtimeRuntime**

In `crates/harmony-compute/src/wasmtime_runtime.rs`, write the full implementation:

```rust
//! WASM execution engine backed by wasmtime with JIT compilation.
//!
//! Alternative to `WasmiRuntime` for desktop/server nodes. Uses JIT compilation
//! for faster execution. Does not support cooperative yielding — modules run to
//! completion or fail on fuel exhaustion. NeedsIO is handled via deterministic
//! replay with a cached IO oracle.

use std::collections::HashMap;

use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state stored in the wasmtime `Store`.
///
/// When a WASM module calls `harmony.fetch_content`, the host function
/// checks `io_cache` first. On a cache miss, it stores the IO request
/// and traps. On a cache hit, it writes data and returns immediately.
struct HostState {
    /// Cached IO results from previous executions (CID → data).
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    /// IO request captured from a cache-miss host function call.
    io_request: Option<crate::types::IORequest>,
    /// Where to write the IO response data: (out_ptr, out_cap).
    io_write_target: Option<(u32, u32)>,
}

impl HostState {
    fn new(io_cache: HashMap<[u8; 32], Vec<u8>>) -> Self {
        Self {
            io_cache,
            io_request: None,
            io_write_target: None,
        }
    }
}

/// Holds state needed for deterministic replay after NeedsIO.
struct WasmtimeSession {
    module_bytes: Vec<u8>,
    module_hash: [u8; 32],
    input: Vec<u8>,
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    total_fuel_consumed: u64,
}

/// WASM execution engine backed by wasmtime with JIT compilation.
///
/// Uses fuel metering for bounded execution. Does not support cooperative
/// yielding — `resume()` always returns `NoPendingExecution`. For NeedsIO,
/// the module is re-executed from scratch with cached IO results
/// (deterministic replay).
pub struct WasmtimeRuntime {
    engine: wasmtime::Engine,
    session: Option<WasmtimeSession>,
}

impl WasmtimeRuntime {
    /// Create a new `WasmtimeRuntime` with fuel metering enabled.
    pub fn new() -> Self {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        let engine = wasmtime::Engine::new(&config).expect("failed to create wasmtime engine");
        Self {
            engine,
            session: None,
        }
    }

    /// Internal: run a WASM module with the given IO cache.
    /// Returns the ComputeResult and the fuel consumed during this execution.
    fn run_module(
        &self,
        module_bytes: &[u8],
        input: &[u8],
        budget: InstructionBudget,
        io_cache: HashMap<[u8; 32], Vec<u8>>,
    ) -> (ComputeResult, u64, HashMap<[u8; 32], Vec<u8>>, Option<crate::types::IORequest>) {
        // Compile the WASM module.
        let module = match wasmtime::Module::new(&self.engine, module_bytes) {
            Ok(m) => m,
            Err(e) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::InvalidModule {
                            reason: e.to_string(),
                        },
                    },
                    0,
                    io_cache,
                    None,
                );
            }
        };

        // Create a store with host state and fuel budget.
        let mut store = wasmtime::Store::new(&self.engine, HostState::new(io_cache));
        store.set_fuel(budget.fuel).expect("fuel metering enabled");

        // Set up Linker (no host functions in Task 3 — added in Task 4).
        let linker = wasmtime::Linker::new(&self.engine);

        // Instantiate the module.
        let instance = match linker.instantiate(&mut store, &module) {
            Ok(i) => i,
            Err(e) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::InvalidModule {
                            reason: e.to_string(),
                        },
                    },
                    0,
                    store.into_data().io_cache,
                    None,
                );
            }
        };

        // Get exported memory.
        let memory = match instance.get_memory(&mut store, "memory") {
            Some(m) => m,
            None => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::ExportNotFound {
                            name: "memory".into(),
                        },
                    },
                    0,
                    store.into_data().io_cache,
                    None,
                );
            }
        };

        // Bounds-check input fits in memory.
        let mem_size = memory.data_size(&store);
        if input.len() > mem_size {
            return (
                ComputeResult::Failed {
                    error: ComputeError::MemoryTooSmall {
                        need: input.len(),
                        have: mem_size,
                    },
                },
                0,
                store.into_data().io_cache,
                None,
            );
        }

        // Write input into memory at offset 0.
        if let Err(_) = memory.write(&mut store, 0, input) {
            return (
                ComputeResult::Failed {
                    error: ComputeError::MemoryTooSmall {
                        need: input.len(),
                        have: memory.data_size(&store),
                    },
                },
                0,
                store.into_data().io_cache,
                None,
            );
        }

        // Get the compute function.
        let compute_func = match instance
            .get_typed_func::<(i32, i32), i32>(&mut store, "compute")
        {
            Ok(f) => f,
            Err(_) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::ExportNotFound {
                            name: "compute".into(),
                        },
                    },
                    0,
                    store.into_data().io_cache,
                    None,
                );
            }
        };

        let input_len = input.len();
        let input_len_i32: i32 = match input_len.try_into() {
            Ok(v) => v,
            Err(_) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::MemoryTooSmall {
                            need: input_len,
                            have: i32::MAX as usize,
                        },
                    },
                    0,
                    store.into_data().io_cache,
                    None,
                );
            }
        };

        // Call compute.
        match compute_func.call(&mut store, (0i32, input_len_i32)) {
            Ok(output_len) => {
                if output_len < 0 {
                    let fuel_consumed = budget.fuel.saturating_sub(
                        store.get_fuel().unwrap_or(0),
                    );
                    return (
                        ComputeResult::Failed {
                            error: ComputeError::Trap {
                                reason: format!(
                                    "compute returned negative length: {output_len}"
                                ),
                            },
                        },
                        fuel_consumed,
                        store.into_data().io_cache,
                        None,
                    );
                }
                let output_len = output_len as usize;

                // Re-fetch memory (may have grown during execution).
                let memory = instance
                    .get_memory(&mut store, "memory")
                    .expect("memory existed before call");

                let output_end = input_len + output_len;
                let mem_size = memory.data_size(&store);
                if output_end > mem_size {
                    let fuel_consumed = budget.fuel.saturating_sub(
                        store.get_fuel().unwrap_or(0),
                    );
                    return (
                        ComputeResult::Failed {
                            error: ComputeError::MemoryTooSmall {
                                need: output_end,
                                have: mem_size,
                            },
                        },
                        fuel_consumed,
                        store.into_data().io_cache,
                        None,
                    );
                }

                let mut output = vec![0u8; output_len];
                if !output.is_empty() {
                    memory
                        .read(&store, input_len, &mut output)
                        .expect("bounds already checked");
                }

                let fuel_consumed = budget.fuel.saturating_sub(
                    store.get_fuel().unwrap_or(0),
                );

                (
                    ComputeResult::Complete { output },
                    fuel_consumed,
                    store.into_data().io_cache,
                    None,
                )
            }
            Err(e) => {
                let fuel_consumed = budget.fuel.saturating_sub(
                    store.get_fuel().unwrap_or(0),
                );

                // Check if this is a host trap with an IO request.
                let host_state = store.data();
                let io_request = host_state.io_request.clone();

                if io_request.is_some() {
                    (
                        ComputeResult::NeedsIO {
                            request: io_request.clone().unwrap(),
                        },
                        fuel_consumed,
                        store.into_data().io_cache,
                        io_request,
                    )
                } else {
                    (
                        ComputeResult::Failed {
                            error: ComputeError::Trap {
                                reason: e.to_string(),
                            },
                        },
                        fuel_consumed,
                        store.into_data().io_cache,
                        None,
                    )
                }
            }
        }
    }
}

impl Default for WasmtimeRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeRuntime for WasmtimeRuntime {
    fn execute(
        &mut self,
        module_bytes: &[u8],
        input: &[u8],
        budget: InstructionBudget,
    ) -> ComputeResult {
        self.session = None;

        let module_hash = harmony_crypto::hash::blake3_hash(module_bytes);
        let io_cache = HashMap::new();

        let (result, fuel_consumed, returned_cache, _io_req) =
            self.run_module(module_bytes, input, budget, io_cache);

        // Store session for snapshot() and potential NeedsIO replay.
        match &result {
            ComputeResult::NeedsIO { .. } => {
                self.session = Some(WasmtimeSession {
                    module_bytes: module_bytes.to_vec(),
                    module_hash,
                    input: input.to_vec(),
                    io_cache: returned_cache,
                    total_fuel_consumed: fuel_consumed,
                });
            }
            ComputeResult::Complete { .. } => {
                self.session = Some(WasmtimeSession {
                    module_bytes: module_bytes.to_vec(),
                    module_hash,
                    input: input.to_vec(),
                    io_cache: returned_cache,
                    total_fuel_consumed: fuel_consumed,
                });
            }
            _ => {}
        }

        result
    }

    fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
        // Wasmtime does not support cooperative yielding — no resumption.
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn resume_with_io(
        &mut self,
        response: crate::types::IOResponse,
        budget: InstructionBudget,
    ) -> ComputeResult {
        let mut session = match self.session.take() {
            Some(s) => s,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        // Cache the IO response.
        match response {
            crate::types::IOResponse::ContentReady { data } => {
                // Find the CID from the last IO request — it's the most
                // recently requested CID not yet in the cache. We need to
                // extract it from the run. Since we stored the session after
                // NeedsIO, we re-run and the host function will request the
                // same CID. But we need to know WHICH CID to cache.
                // Solution: find the first CID key NOT in io_cache by running
                // again... but that's circular. Instead, we track the pending
                // CID in the session.
                //
                // Actually: the IO request is returned in the ComputeResult::NeedsIO.
                // The caller (ComputeTier) knows the CID. We need to accept it
                // as a parameter or derive it. The simplest approach: the caller
                // must tell us which CID this response is for.
                //
                // But the ComputeRuntime trait takes IOResponse, not a CID.
                // We need to store the pending CID in the session.
                //
                // This is handled below after we know the pending CID.
                //
                // For now, use a placeholder — Task 4 will add proper CID tracking.
                let _ = data;
            }
            crate::types::IOResponse::ContentNotFound => {}
        }

        // Placeholder — Task 4 implements the full replay logic.
        self.session = Some(session);
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn has_pending(&self) -> bool {
        false // Wasmtime never has a pending resumable execution.
    }

    fn take_session(&mut self) -> Option<Box<dyn std::any::Any>> {
        self.session
            .take()
            .map(|s| Box::new(s) as Box<dyn std::any::Any>)
    }

    fn restore_session(&mut self, session: Box<dyn std::any::Any>) {
        self.session = Some(
            *session
                .downcast::<WasmtimeSession>()
                .expect("restore_session: type mismatch — expected WasmtimeSession"),
        );
    }

    fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
        let session = self
            .session
            .as_ref()
            .ok_or(ComputeError::NoPendingExecution)?;

        Ok(Checkpoint {
            module_hash: session.module_hash,
            memory: Vec::new(), // No memory snapshot — wasmtime doesn't retain Store after call.
            fuel_consumed: session.total_fuel_consumed,
        })
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-compute --features wasmtime`
Expected: All tests pass (33 existing + 5 new wasmtime tests).

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/wasmtime_runtime.rs
git commit -m "feat(compute): add WasmtimeRuntime with basic JIT execution"
```

---

### Task 4: Add harmony.fetch_content host function with IO cache

Register the `harmony::fetch_content` host import in WasmtimeRuntime's Linker. On cache miss, the host function stores an IORequest and traps. On cache hit, it writes data and returns the byte count.

**Files:**
- Modify: `crates/harmony-compute/src/wasmtime_runtime.rs`

**Context:** The `run_module()` helper currently creates a bare `Linker` with no imports. This task adds `harmony::fetch_content` and a `pending_cid` field to `WasmtimeSession` for tracking which CID to cache on resume.

**Step 1: Write test for NeedsIO**

Add to the `tests` module in `wasmtime_runtime.rs`:

```rust
    /// WAT module that calls harmony.fetch_content — same as compute tier FETCH_WAT.
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
    fn host_function_triggers_needs_io() {
        let mut rt = WasmtimeRuntime::new();
        let cid = [0xAB; 32];

        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::NeedsIO { request } => {
                match request {
                    crate::types::IORequest::FetchContent { cid: req_cid } => {
                        assert_eq!(req_cid, cid);
                    }
                }
            }
            other => panic!("expected NeedsIO, got: {other:?}"),
        }
    }
```

**Step 2: Run test to see it fail**

Run: `cargo test -p harmony-compute --features wasmtime -- host_function_triggers_needs_io`
Expected: FAIL — module instantiation fails because `harmony::fetch_content` import is missing.

**Step 3: Register the host function in run_module()**

In `run_module()`, replace the bare `Linker::new()` with:

```rust
        let mut linker = wasmtime::Linker::new(&self.engine);
        linker
            .func_wrap(
                "harmony",
                "fetch_content",
                |mut caller: wasmtime::Caller<'_, HostState>,
                 cid_ptr: i32,
                 out_ptr: i32,
                 out_cap: i32|
                 -> Result<i32, wasmtime::Error> {
                    // Read 32-byte CID from WASM memory.
                    let memory = caller
                        .get_export("memory")
                        .and_then(|e| e.into_memory())
                        .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                    let mut cid = [0u8; 32];
                    memory
                        .read(&caller, cid_ptr as usize, &mut cid)
                        .map_err(|e| wasmtime::Error::msg(format!("failed to read CID: {e}")))?;

                    // Check the IO cache for this CID.
                    if let Some(data) = caller.data().io_cache.get(&cid).cloned() {
                        // Cache hit — write data to output buffer.
                        if data.len() > out_cap as usize {
                            return Ok(-2); // Buffer too small.
                        }
                        memory
                            .write(&mut caller, out_ptr as usize, &data)
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("failed to write data: {e}"))
                            })?;
                        return Ok(data.len() as i32);
                    }

                    // Cache miss — store IO request and trap.
                    let host = caller.data_mut();
                    host.io_request =
                        Some(crate::types::IORequest::FetchContent { cid });
                    host.io_write_target = Some((out_ptr as u32, out_cap as u32));

                    Err(wasmtime::Error::msg("harmony_io_trap"))
                },
            )
            .expect("failed to register harmony.fetch_content");
```

Also add `pending_cid: Option<[u8; 32]>` to `WasmtimeSession`:

```rust
struct WasmtimeSession {
    module_bytes: Vec<u8>,
    module_hash: [u8; 32],
    input: Vec<u8>,
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    total_fuel_consumed: u64,
    pending_cid: Option<[u8; 32]>,
}
```

In the NeedsIO branch of `execute()`, store the pending CID:

```rust
ComputeResult::NeedsIO { request } => {
    let pending_cid = match &request {
        crate::types::IORequest::FetchContent { cid } => Some(*cid),
    };
    self.session = Some(WasmtimeSession {
        module_bytes: module_bytes.to_vec(),
        module_hash,
        input: input.to_vec(),
        io_cache: returned_cache,
        total_fuel_consumed: fuel_consumed,
        pending_cid,
    });
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-compute --features wasmtime`
Expected: All tests pass including `host_function_triggers_needs_io`.

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/wasmtime_runtime.rs
git commit -m "feat(compute): register harmony.fetch_content host import with IO cache in WasmtimeRuntime"
```

---

### Task 5: Implement resume_with_io() with deterministic replay

Complete the NeedsIO flow: `resume_with_io()` caches the IO response, then re-executes the module from scratch. The host function finds the CID in cache and returns data immediately.

**Files:**
- Modify: `crates/harmony-compute/src/wasmtime_runtime.rs`

**Step 1: Write tests for resume_with_io**

Add to the `tests` module:

```rust
    #[test]
    fn resume_with_io_replays_with_content() {
        let mut rt = WasmtimeRuntime::new();
        let cid = [0xCD; 32];
        let content = b"hello from storage";

        // First: execute triggers NeedsIO.
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::NeedsIO { .. }));

        // Resume with content — should replay and complete.
        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::Complete { output } => {
                // FETCH_WAT output: [result_code: i32 LE] [data if result > 0]
                let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(result_code, content.len() as i32);
                assert_eq!(&output[4..], content);
            }
            other => panic!("expected Complete after replay, got: {other:?}"),
        }
    }

    #[test]
    fn resume_with_io_content_not_found() {
        let mut rt = WasmtimeRuntime::new();
        let cid = [0xEF; 32];

        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::NeedsIO { .. }));

        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentNotFound,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::Complete { output } => {
                let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(result_code, -1, "not-found result code");
            }
            other => panic!("expected Complete with -1, got: {other:?}"),
        }
    }

    #[test]
    fn fetch_content_replay_then_complete() {
        let mut rt = WasmtimeRuntime::new();
        let cid = [0x42; 32];
        let content = b"deterministic replay works";

        // Execute → NeedsIO.
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::NeedsIO { .. }));

        // Resume → Complete.
        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::Complete { .. }));

        // After completion, no pending execution.
        assert!(!rt.has_pending());
    }
```

**Step 2: Run tests to see them fail**

Run: `cargo test -p harmony-compute --features wasmtime -- resume_with_io`
Expected: FAIL — `resume_with_io` currently returns `NoPendingExecution`.

**Step 3: Implement resume_with_io()**

Replace the placeholder `resume_with_io()` in the `ComputeRuntime` impl:

```rust
    fn resume_with_io(
        &mut self,
        response: crate::types::IOResponse,
        budget: InstructionBudget,
    ) -> ComputeResult {
        let mut session = match self.session.take() {
            Some(s) => s,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        let pending_cid = match session.pending_cid.take() {
            Some(cid) => cid,
            None => {
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        // Cache the IO response keyed by the pending CID.
        match response {
            crate::types::IOResponse::ContentReady { data } => {
                session.io_cache.insert(pending_cid, data);
            }
            crate::types::IOResponse::ContentNotFound => {
                // Insert a sentinel — the host function will check for this.
                // Actually, for ContentNotFound we DON'T insert into cache.
                // Instead, the host function needs to know this CID was not found.
                // Use a separate "not found" set, or encode as empty + flag.
                //
                // Simplest: insert an empty vec with a separate not_found set.
                // But that requires changing HostState. Simpler: use a
                // not_found_cids set in the session.
                //
                // Actually simplest: just don't insert anything. The host
                // function will trap again on cache miss. But then resume_with_io
                // will be called again in an infinite loop!
                //
                // Correct approach: use a not_found set.
            }
        }

        // Re-execute the module from scratch with the updated IO cache.
        let (result, fuel_consumed, returned_cache, _io_req) = self.run_module(
            &session.module_bytes,
            &session.input,
            budget,
            session.io_cache,
        );

        let total_fuel = session.total_fuel_consumed + fuel_consumed;

        // Update session for potential further NeedsIO rounds.
        match &result {
            ComputeResult::NeedsIO { request } => {
                let new_pending_cid = match request {
                    crate::types::IORequest::FetchContent { cid } => Some(*cid),
                };
                self.session = Some(WasmtimeSession {
                    module_bytes: session.module_bytes,
                    module_hash: session.module_hash,
                    input: session.input,
                    io_cache: returned_cache,
                    total_fuel_consumed: total_fuel,
                    pending_cid: new_pending_cid,
                });
            }
            ComputeResult::Complete { .. } => {
                self.session = Some(WasmtimeSession {
                    module_bytes: session.module_bytes,
                    module_hash: session.module_hash,
                    input: session.input,
                    io_cache: returned_cache,
                    total_fuel_consumed: total_fuel,
                    pending_cid: None,
                });
            }
            _ => {}
        }

        result
    }
```

Wait — ContentNotFound needs special handling. The host function needs to return `-1` for not-found CIDs without trapping. Add a `not_found_cids: HashSet<[u8; 32]>` to `HostState`:

```rust
struct HostState {
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    not_found_cids: std::collections::HashSet<[u8; 32]>,
    io_request: Option<crate::types::IORequest>,
    io_write_target: Option<(u32, u32)>,
}
```

Update the host function to check `not_found_cids`:

```rust
                    // Check not-found set.
                    if caller.data().not_found_cids.contains(&cid) {
                        return Ok(-1); // Content not found.
                    }

                    // Check the IO cache for this CID.
                    if let Some(data) = caller.data().io_cache.get(&cid).cloned() {
                        // ... existing cache hit logic
                    }
```

And in `resume_with_io()`, handle ContentNotFound:

```rust
            crate::types::IOResponse::ContentNotFound => {
                session.not_found_cids.insert(pending_cid);
            }
```

Add `not_found_cids` to `WasmtimeSession` and thread it through `HostState::new()` and `run_module()`.

**Step 4: Run tests**

Run: `cargo test -p harmony-compute --features wasmtime`
Expected: All tests pass (33 existing wasmi + 8 wasmtime tests).

Also run full workspace to make sure nothing broke:
Run: `cargo test --workspace`
Expected: All ~780 tests pass.

Run: `cargo clippy --workspace --features wasmtime`
Expected: Zero warnings.

**Step 5: Commit**

```bash
git add crates/harmony-compute/src/wasmtime_runtime.rs
git commit -m "feat(compute): implement resume_with_io via deterministic replay in WasmtimeRuntime"
```

---

### Summary

| Task | What | Tests |
|------|------|-------|
| 1 | ComputeTier → Box<dyn ComputeRuntime> | 42 existing (no new) |
| 2 | wasmtime feature gate + empty stub | 33 existing (compile check) |
| 3 | WasmtimeRuntime basic execute() | +5 new (38 total) |
| 4 | fetch_content host function + IO cache | +1 new (39 total) |
| 5 | resume_with_io() with replay | +3 new (42 total compute, ~790 workspace) |

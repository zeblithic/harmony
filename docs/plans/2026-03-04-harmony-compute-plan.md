# Harmony-Compute Crate Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create the `harmony-compute` crate with a `ComputeRuntime` trait, `WasmiRuntime` interpreter implementation, fuel-based cooperative yielding, and memory checkpoint snapshots for durable execution.

**Architecture:** Sans-I/O state machine pattern matching the rest of the Harmony stack. The `ComputeRuntime` trait abstracts WASM execution behind a synchronous interface: `execute()` runs a bounded instruction slice (fuel-metered), returning `Complete`/`Yielded`/`Failed`. The `WasmiRuntime` uses wasmi's `call_resumable()` API for cooperative scheduling — fuel exhaustion pauses execution mid-function, `resume()` continues it. Checkpointing captures linear memory for cross-session recovery.

**Tech Stack:** Rust, wasmi (WASM interpreter with fuel metering), harmony-crypto (BLAKE3 hashing for checkpoint module IDs), thiserror

**Design doc:** `docs/plans/2026-03-03-node-trinity-design.md` section 5

**WASM ABI Convention:**
- Module must export `memory` (linear memory, at least 1 page = 64 KiB)
- Module must export `compute(input_ptr: i32, input_len: i32) -> i32`
- Host writes input bytes to `memory[0..input_len]` before calling `compute`
- `compute` processes input, writes output bytes to `memory[input_len..]`
- Return value = number of output bytes written (negative = error)
- Host reads `memory[input_len..input_len + return_value]` as output

**MSRV note:** wasmi's latest stable may require a newer Rust than the workspace MSRV (1.75). If `cargo build` fails with version errors, either bump `rust-version` in the workspace `Cargo.toml` or pin wasmi to an older compatible version. Check `cargo msrv` or wasmi's `Cargo.toml` for its `rust-version`.

---

### Task 1: Crate Scaffolding and Error Types

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `crates/harmony-compute/Cargo.toml`
- Create: `crates/harmony-compute/src/lib.rs`
- Create: `crates/harmony-compute/src/error.rs`

**Step 1: Add crate to workspace**

In the workspace root `Cargo.toml`, add `"crates/harmony-compute"` to the `members` array, and add `wasmi` + `harmony-compute` to `[workspace.dependencies]`:

```toml
# In [workspace] members — add after harmony-content:
"crates/harmony-compute",

# In [workspace.dependencies] — add:
wasmi = "1"
harmony-compute = { path = "crates/harmony-compute" }
```

**Step 2: Create crate Cargo.toml**

Create `crates/harmony-compute/Cargo.toml`:

```toml
[package]
name = "harmony-compute"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "WASM compute engine with cooperative yielding for the Harmony decentralized stack"

[dependencies]
harmony-crypto.workspace = true
thiserror.workspace = true
wasmi = { workspace = true }
```

**Step 3: Create lib.rs and error.rs**

Create `crates/harmony-compute/src/lib.rs`:

```rust
pub mod error;

pub use error::ComputeError;
```

Create `crates/harmony-compute/src/error.rs`:

```rust
/// Errors from compute operations.
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("invalid WASM module: {reason}")]
    InvalidModule { reason: String },

    #[error("execution trapped: {reason}")]
    Trap { reason: String },

    #[error("no pending execution to resume")]
    NoPendingExecution,

    #[error("export not found: {name}")]
    ExportNotFound { name: String },

    #[error("memory too small: need {need} bytes, have {have}")]
    MemoryTooSmall { need: usize, have: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_invalid_module() {
        let err = ComputeError::InvalidModule {
            reason: "bad magic".into(),
        };
        assert_eq!(err.to_string(), "invalid WASM module: bad magic");
    }

    #[test]
    fn error_display_trap() {
        let err = ComputeError::Trap {
            reason: "unreachable".into(),
        };
        assert_eq!(err.to_string(), "execution trapped: unreachable");
    }

    #[test]
    fn error_display_no_pending() {
        let err = ComputeError::NoPendingExecution;
        assert_eq!(err.to_string(), "no pending execution to resume");
    }

    #[test]
    fn error_display_export_not_found() {
        let err = ComputeError::ExportNotFound {
            name: "compute".into(),
        };
        assert_eq!(err.to_string(), "export not found: compute");
    }

    #[test]
    fn error_display_memory_too_small() {
        let err = ComputeError::MemoryTooSmall {
            need: 1024,
            have: 512,
        };
        assert_eq!(
            err.to_string(),
            "memory too small: need 1024 bytes, have 512"
        );
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-compute`
Expected: 5 tests pass

**Step 5: Commit**

```bash
git add Cargo.toml crates/harmony-compute/
git commit -m "feat(compute): scaffold harmony-compute crate with error types (Task 1/5)"
```

---

### Task 2: Core Types and ComputeRuntime Trait

**Files:**
- Create: `crates/harmony-compute/src/types.rs`
- Create: `crates/harmony-compute/src/runtime.rs`
- Modify: `crates/harmony-compute/src/lib.rs`

**Step 1: Create types.rs with tests**

Create `crates/harmony-compute/src/types.rs`:

```rust
use crate::error::ComputeError;

/// How many fuel units (approximately instructions) to allow before yielding.
#[derive(Debug, Clone, Copy)]
pub struct InstructionBudget {
    pub fuel: u64,
}

/// Serializable snapshot of WASM execution state for durable recovery.
///
/// Captures linear memory so a computation can be resumed on a different node.
/// The WASM module itself is identified by its BLAKE3 hash — the caller must
/// supply the module bytes when restoring from a checkpoint.
#[derive(Debug, Clone)]
pub struct Checkpoint {
    /// BLAKE3 hash of the original WASM module bytes.
    pub module_hash: [u8; 32],
    /// Snapshot of WASM linear memory at time of checkpoint.
    pub memory: Vec<u8>,
    /// Total fuel consumed before this checkpoint was taken.
    pub fuel_consumed: u64,
}

/// An I/O request from WASM code (for future NeedsIO support).
#[derive(Debug, Clone)]
pub enum IORequest {
    /// Request to fetch content by CID.
    FetchContent { cid: [u8; 32] },
}

/// Result of executing a WASM computation slice.
#[derive(Debug)]
pub enum ComputeResult {
    /// Execution completed successfully.
    Complete { output: Vec<u8> },
    /// Execution yielded due to fuel exhaustion (can resume).
    Yielded { fuel_consumed: u64 },
    /// Execution needs external I/O before continuing.
    NeedsIO { request: IORequest },
    /// Execution failed with an error.
    Failed { error: ComputeError },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_budget_construction() {
        let budget = InstructionBudget { fuel: 10_000 };
        assert_eq!(budget.fuel, 10_000);
    }

    #[test]
    fn checkpoint_construction() {
        let cp = Checkpoint {
            module_hash: [0xAB; 32],
            memory: vec![1, 2, 3, 4],
            fuel_consumed: 500,
        };
        assert_eq!(cp.module_hash, [0xAB; 32]);
        assert_eq!(cp.memory, vec![1, 2, 3, 4]);
        assert_eq!(cp.fuel_consumed, 500);
    }

    #[test]
    fn io_request_construction() {
        let req = IORequest::FetchContent { cid: [0xFF; 32] };
        assert!(matches!(
            req,
            IORequest::FetchContent { cid } if cid == [0xFF; 32]
        ));
    }

    #[test]
    fn compute_result_complete() {
        let result = ComputeResult::Complete {
            output: vec![42, 43],
        };
        assert!(
            matches!(result, ComputeResult::Complete { output } if output == vec![42, 43])
        );
    }

    #[test]
    fn compute_result_yielded() {
        let result = ComputeResult::Yielded {
            fuel_consumed: 5000,
        };
        assert!(matches!(
            result,
            ComputeResult::Yielded { fuel_consumed: 5000 }
        ));
    }

    #[test]
    fn compute_result_needs_io() {
        let result = ComputeResult::NeedsIO {
            request: IORequest::FetchContent { cid: [0; 32] },
        };
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));
    }

    #[test]
    fn compute_result_failed() {
        let result = ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        };
        assert!(matches!(result, ComputeResult::Failed { .. }));
    }
}
```

**Step 2: Create runtime.rs with trait and mock test**

Create `crates/harmony-compute/src/runtime.rs`:

```rust
use crate::error::ComputeError;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Abstract WASM execution engine with cooperative yielding.
///
/// Implementations execute a bounded slice of WASM instructions and return
/// without blocking. The caller controls scheduling by choosing fuel budgets
/// and deciding when to resume yielded executions.
///
/// **WASM ABI:** Modules must export `memory` (linear memory) and
/// `compute(input_ptr: i32, input_len: i32) -> i32`. Input bytes are written
/// to memory at offset 0. The function returns the number of output bytes
/// written starting at offset `input_len`.
pub trait ComputeRuntime {
    /// Execute a WASM module's `compute` export with the given input and fuel budget.
    ///
    /// Starts a new execution, dropping any pending session from a prior call.
    fn execute(
        &mut self,
        module: &[u8],
        input: &[u8],
        budget: InstructionBudget,
    ) -> ComputeResult;

    /// Resume a previously yielded execution with additional fuel.
    ///
    /// Returns `Failed { NoPendingExecution }` if there is nothing to resume.
    fn resume(&mut self, budget: InstructionBudget) -> ComputeResult;

    /// Whether there is a suspended execution that can be resumed.
    fn has_pending(&self) -> bool;

    /// Take a serializable snapshot of the current execution state.
    ///
    /// Captures WASM linear memory and module hash for cross-session recovery.
    /// Returns error if no session is active.
    fn snapshot(&self) -> Result<Checkpoint, ComputeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockRuntime;

    impl ComputeRuntime for MockRuntime {
        fn execute(
            &mut self,
            _module: &[u8],
            _input: &[u8],
            _budget: InstructionBudget,
        ) -> ComputeResult {
            ComputeResult::Complete {
                output: vec![1, 2, 3],
            }
        }

        fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
            ComputeResult::Failed {
                error: ComputeError::NoPendingExecution,
            }
        }

        fn has_pending(&self) -> bool {
            false
        }

        fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
            Err(ComputeError::NoPendingExecution)
        }
    }

    #[test]
    fn mock_runtime_execute() {
        let mut rt = MockRuntime;
        let result = rt.execute(b"", b"input", InstructionBudget { fuel: 1000 });
        assert!(
            matches!(result, ComputeResult::Complete { output } if output == vec![1, 2, 3])
        );
    }

    #[test]
    fn mock_runtime_has_no_pending() {
        let rt = MockRuntime;
        assert!(!rt.has_pending());
    }
}
```

**Step 3: Update lib.rs**

Replace `crates/harmony-compute/src/lib.rs` with:

```rust
pub mod error;
pub mod runtime;
pub mod types;

pub use error::ComputeError;
pub use runtime::ComputeRuntime;
pub use types::{Checkpoint, ComputeResult, IORequest, InstructionBudget};
```

**Step 4: Run tests**

Run: `cargo test -p harmony-compute`
Expected: 14 tests pass (5 error + 7 types + 2 runtime)

**Step 5: Commit**

```bash
git add crates/harmony-compute/
git commit -m "feat(compute): add core types and ComputeRuntime trait (Task 2/5)"
```

---

### Task 3: WasmiRuntime Basic Execution

**Files:**
- Create: `crates/harmony-compute/src/wasmi_runtime.rs`
- Modify: `crates/harmony-compute/src/lib.rs`

**Context:** This task implements the `WasmiRuntime` struct with a working `execute()` method. The implementation uses wasmi's fuel metering from the start (required for cooperative yielding in Task 4). The `resume()`, `has_pending()`, and `snapshot()` methods return stub/error values — they'll be completed in Tasks 4 and 5.

**Key wasmi concepts:**
- `Engine` — compilation/execution manager, configured with fuel metering
- `Module` — compiled WASM, created from bytes (accepts both WASM binary and WAT text if `wat` feature is enabled)
- `Store<T>` — runtime state container with user-defined host state T
- `Linker` — resolves module imports, instantiates modules
- `Instance` — instantiated module with exports (functions, memory, globals)
- `TypedFunc<Params, Results>` — typed function reference for calling exports
- `call_resumable()` — fuel-aware execution that can pause and resume
- `TypedResumableCall::Finished(result)` — function completed normally
- `TypedResumableCall::Resumable(invocation)` — fuel exhausted, can resume

**Step 1: Write the failing tests**

Add tests to `crates/harmony-compute/src/wasmi_runtime.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ComputeRuntime;
    use crate::types::{ComputeResult, InstructionBudget};

    /// WAT module: reads two i32s from input, writes their sum as output.
    /// Input: 8 bytes (two little-endian i32s at memory[0..8])
    /// Output: 4 bytes (one little-endian i32 at memory[8..12])
    /// Returns: 4 (output byte count)
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
        let mut rt = WasmiRuntime::new();
        let mut input = Vec::new();
        input.extend_from_slice(&3_i32.to_le_bytes());
        input.extend_from_slice(&7_i32.to_le_bytes());

        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 100_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                assert_eq!(output.len(), 4);
                let sum = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(sum, 10);
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    fn execute_invalid_module() {
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(
            b"not wasm",
            b"",
            InstructionBudget { fuel: 1000 },
        );
        assert!(matches!(result, ComputeResult::Failed { .. }));
    }

    #[test]
    fn execute_missing_compute_export() {
        let wat = r#"
            (module
              (memory (export "memory") 1)
              (func (export "other") (result i32) (i32.const 0)))
        "#;
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(
            wat.as_bytes(),
            b"",
            InstructionBudget { fuel: 1000 },
        );
        assert!(
            matches!(result, ComputeResult::Failed { error: ComputeError::ExportNotFound { .. } })
        );
    }

    #[test]
    fn execute_missing_memory_export() {
        let wat = r#"
            (module
              (func (export "compute") (param i32 i32) (result i32) (i32.const 0)))
        "#;
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(
            wat.as_bytes(),
            b"",
            InstructionBudget { fuel: 1000 },
        );
        assert!(
            matches!(result, ComputeResult::Failed { error: ComputeError::ExportNotFound { .. } })
        );
    }
}
```

**Step 2: Implement WasmiRuntime**

Create `crates/harmony-compute/src/wasmi_runtime.rs`:

```rust
use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state carried inside the wasmi Store.
struct HostState;

/// In-progress WASM execution session, held between execute() and resume().
struct WasmiSession {
    store: wasmi::Store<HostState>,
    instance: wasmi::Instance,
    module_hash: [u8; 32],
    input_len: usize,
    total_fuel_consumed: u64,
    pending: Option<wasmi::TypedResumableCallOutOfFuel<i32>>,
}

/// WASM interpreter runtime using wasmi with fuel-based cooperative yielding.
///
/// Executes WASM modules with instruction budgets (fuel metering). When fuel
/// runs out, execution pauses and can be resumed with `resume()`. Memory can
/// be captured at any point via `snapshot()` for durable checkpointing.
pub struct WasmiRuntime {
    engine: wasmi::Engine,
    session: Option<WasmiSession>,
}

impl WasmiRuntime {
    pub fn new() -> Self {
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        Self {
            engine,
            session: None,
        }
    }

    /// Compile WASM bytes into a module and instantiate it.
    fn compile_and_instantiate(
        &self,
        module_bytes: &[u8],
        store: &mut wasmi::Store<HostState>,
    ) -> Result<wasmi::Instance, ComputeError> {
        let module = wasmi::Module::new(&self.engine, module_bytes).map_err(|e| {
            ComputeError::InvalidModule {
                reason: e.to_string(),
            }
        })?;
        let linker = wasmi::Linker::new(&self.engine);
        linker
            .instantiate(store, &module)
            .and_then(|pre| pre.start(store))
            .map_err(|e| ComputeError::Trap {
                reason: e.to_string(),
            })
    }

    /// Write input bytes to WASM memory at offset 0. Returns the Memory handle.
    fn write_input(
        store: &mut wasmi::Store<HostState>,
        instance: &wasmi::Instance,
        input: &[u8],
    ) -> Result<wasmi::Memory, ComputeError> {
        let memory = instance
            .get_memory(store, "memory")
            .ok_or_else(|| ComputeError::ExportNotFound {
                name: "memory".into(),
            })?;
        if !input.is_empty() {
            let mem_size = memory.data_size(store);
            if input.len() > mem_size {
                return Err(ComputeError::MemoryTooSmall {
                    need: input.len(),
                    have: mem_size,
                });
            }
            memory
                .write(store, 0, input)
                .map_err(|_| ComputeError::MemoryTooSmall {
                    need: input.len(),
                    have: mem_size,
                })?;
        }
        Ok(memory)
    }

    /// Read output bytes from WASM memory after compute returns.
    fn read_output(
        store: &wasmi::Store<HostState>,
        memory: &wasmi::Memory,
        input_len: usize,
        output_len: i32,
    ) -> Result<Vec<u8>, ComputeError> {
        if output_len < 0 {
            return Err(ComputeError::Trap {
                reason: format!("compute returned negative length: {output_len}"),
            });
        }
        if output_len == 0 {
            return Ok(vec![]);
        }
        let output_len = output_len as usize;
        let mem_size = memory.data_size(store);
        let end = input_len + output_len;
        if end > mem_size {
            return Err(ComputeError::MemoryTooSmall {
                need: end,
                have: mem_size,
            });
        }
        let mut output = vec![0u8; output_len];
        memory
            .read(store, input_len, &mut output)
            .map_err(|_| ComputeError::MemoryTooSmall {
                need: end,
                have: mem_size,
            })?;
        Ok(output)
    }

    /// Handle the result of call_resumable, building a ComputeResult.
    fn handle_resumable_result(
        &mut self,
        result: Result<wasmi::TypedResumableCall<i32>, wasmi::Error>,
        store: wasmi::Store<HostState>,
        instance: wasmi::Instance,
        module_hash: [u8; 32],
        input_len: usize,
        budget_fuel: u64,
        fuel_before: u64,
    ) -> ComputeResult {
        match result {
            Ok(wasmi::TypedResumableCall::Finished(output_len)) => {
                let memory = instance.get_memory(&store, "memory").unwrap();
                match Self::read_output(&store, &memory, input_len, output_len) {
                    Ok(output) => {
                        let remaining = store.get_fuel().unwrap_or(0);
                        let fuel_consumed = fuel_before + budget_fuel.saturating_sub(remaining);
                        self.session = Some(WasmiSession {
                            store,
                            instance,
                            module_hash,
                            input_len,
                            total_fuel_consumed: fuel_consumed,
                            pending: None,
                        });
                        ComputeResult::Complete { output }
                    }
                    Err(e) => ComputeResult::Failed { error: e },
                }
            }
            Ok(wasmi::TypedResumableCall::Resumable(invocation)) => {
                let remaining = store.get_fuel().unwrap_or(0);
                let fuel_consumed = fuel_before + budget_fuel.saturating_sub(remaining);
                self.session = Some(WasmiSession {
                    store,
                    instance,
                    module_hash,
                    input_len,
                    total_fuel_consumed: fuel_consumed,
                    pending: Some(invocation),
                });
                ComputeResult::Yielded { fuel_consumed }
            }
            Err(e) => ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: e.to_string(),
                },
            },
        }
    }
}

impl Default for WasmiRuntime {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeRuntime for WasmiRuntime {
    fn execute(
        &mut self,
        module_bytes: &[u8],
        input: &[u8],
        budget: InstructionBudget,
    ) -> ComputeResult {
        // Drop any existing session
        self.session = None;

        let module_hash = harmony_crypto::hash::blake3_hash(module_bytes);
        let mut store = wasmi::Store::new(&self.engine, HostState);

        let instance = match self.compile_and_instantiate(module_bytes, &mut store) {
            Ok(i) => i,
            Err(e) => return ComputeResult::Failed { error: e },
        };

        // Write input to WASM memory
        if let Err(e) = Self::write_input(&mut store, &instance, input) {
            return ComputeResult::Failed { error: e };
        }

        // Get the compute function export
        let compute = match instance.get_typed_func::<(i32, i32), i32>(&store, "compute") {
            Ok(f) => f,
            Err(_) => {
                return ComputeResult::Failed {
                    error: ComputeError::ExportNotFound {
                        name: "compute".into(),
                    },
                }
            }
        };

        // Set fuel and execute
        store
            .set_fuel(budget.fuel)
            .expect("fuel metering is enabled");
        let input_len = input.len();
        let result = compute.call_resumable(&mut store, (0i32, input_len as i32));
        self.handle_resumable_result(result, store, instance, module_hash, input_len, budget.fuel, 0)
    }

    fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
        // Stub — implemented in Task 4
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn has_pending(&self) -> bool {
        self.session
            .as_ref()
            .is_some_and(|s| s.pending.is_some())
    }

    fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
        // Stub — implemented in Task 5
        Err(ComputeError::NoPendingExecution)
    }
}
```

**Important API notes for the implementer:**
- The `TypedResumableCall` enum variant for fuel exhaustion may be named `Resumable` or `OutOfFuel` depending on the wasmi version. Check `wasmi::TypedResumableCall` docs if the compiler complains.
- Similarly, the stored invocation type may be `TypedResumableCallOutOfFuel<i32>` or `TypedResumableInvocation<i32>`. Follow the compiler's guidance.
- `wasmi::Linker::instantiate` may be called `instantiate_and_start` in some versions. If `instantiate` returns a `InstancePre` without a `start` method, use `ensure_no_start` instead.
- `Memory::data_size` may be called `data_len` or similar. Check the Memory docs.

**Step 3: Update lib.rs**

Add `pub mod wasmi_runtime;` and re-export:

```rust
pub mod error;
pub mod runtime;
pub mod types;
pub mod wasmi_runtime;

pub use error::ComputeError;
pub use runtime::ComputeRuntime;
pub use types::{Checkpoint, ComputeResult, IORequest, InstructionBudget};
pub use wasmi_runtime::WasmiRuntime;
```

**Step 4: Run tests**

Run: `cargo test -p harmony-compute`
Expected: 18 tests pass (5 error + 7 types + 2 runtime mock + 4 wasmi)

**Step 5: Commit**

```bash
git add crates/harmony-compute/
git commit -m "feat(compute): WasmiRuntime execute() with fuel metering (Task 3/5)"
```

---

### Task 4: Cooperative Yielding with resume()

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs`

**Context:** This task completes the `resume()` method. When `execute()` returns `Yielded`, the runtime holds a `WasmiSession` with a pending `TypedResumableCallOutOfFuel` invocation. Calling `resume(budget)` sets new fuel on the store and continues the suspended execution.

**Step 1: Write the failing tests**

Add to the `tests` module in `wasmi_runtime.rs`:

```rust
    /// WAT module: loops N times (fuel-hungry). Reads iteration count from input,
    /// writes final counter to output. With small fuel budgets, this will yield.
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
    fn execute_yields_then_resumes_to_completion() {
        let mut rt = WasmiRuntime::new();
        let input = 10_000_i32.to_le_bytes();

        // Very small fuel budget — should yield
        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 100 },
        );
        assert!(
            matches!(result, ComputeResult::Yielded { fuel_consumed } if fuel_consumed > 0),
            "expected Yielded, got {result:?}"
        );
        assert!(rt.has_pending());

        // Resume with plenty of fuel — should complete
        let result = rt.resume(InstructionBudget { fuel: 1_000_000 });
        match result {
            ComputeResult::Complete { output } => {
                let counter = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(counter, 10_000);
            }
            other => panic!("expected Complete after resume, got {other:?}"),
        }
        assert!(!rt.has_pending());
    }

    #[test]
    fn resume_without_pending_returns_error() {
        let mut rt = WasmiRuntime::new();
        let result = rt.resume(InstructionBudget { fuel: 1000 });
        assert!(matches!(
            result,
            ComputeResult::Failed {
                error: ComputeError::NoPendingExecution
            }
        ));
    }

    #[test]
    fn multiple_yields_then_complete() {
        let mut rt = WasmiRuntime::new();
        let input = 50_000_i32.to_le_bytes();

        // Start with tiny fuel
        let mut result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 50 },
        );

        // Keep resuming with small fuel until complete
        let mut resume_count = 0;
        loop {
            match result {
                ComputeResult::Yielded { .. } => {
                    resume_count += 1;
                    result = rt.resume(InstructionBudget { fuel: 10_000 });
                }
                ComputeResult::Complete { output } => {
                    let counter = i32::from_le_bytes(output[..4].try_into().unwrap());
                    assert_eq!(counter, 50_000);
                    break;
                }
                other => panic!("unexpected result: {other:?}"),
            }
        }
        assert!(resume_count > 0, "should have yielded at least once");
    }
```

**Step 2: Implement resume()**

Replace the `resume()` stub in the `ComputeRuntime` impl block with:

```rust
    fn resume(&mut self, budget: InstructionBudget) -> ComputeResult {
        let mut session = match self.session.take() {
            Some(s) => s,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                }
            }
        };

        let invocation = match session.pending.take() {
            Some(inv) => inv,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                }
            }
        };

        session
            .store
            .set_fuel(budget.fuel)
            .expect("fuel metering is enabled");

        let fuel_before = session.total_fuel_consumed;
        let result = invocation.resume(&mut session.store);
        self.handle_resumable_result(
            result,
            session.store,
            session.instance,
            session.module_hash,
            session.input_len,
            budget.fuel,
            fuel_before,
        )
    }
```

**Step 3: Run tests**

Run: `cargo test -p harmony-compute`
Expected: 21 tests pass (18 prior + 3 new yielding tests)

**Step 4: Commit**

```bash
git add crates/harmony-compute/
git commit -m "feat(compute): cooperative yielding with resume() (Task 4/5)"
```

---

### Task 5: Checkpoint Memory Snapshot

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs`

**Context:** This task completes `snapshot()`. A `Checkpoint` captures the WASM linear memory contents and module hash, allowing a computation's memory state to be serialized for cross-session or cross-node recovery. Note: wasmi does not support serializing the execution stack or call frames — full durable replay will be handled by `harmony-workflow` (separate crate). This checkpoint captures memory-level state, which is sufficient for WASM modules that use memory as their primary state store.

**Step 1: Write the failing tests**

Add to the `tests` module in `wasmi_runtime.rs`:

```rust
    #[test]
    fn snapshot_after_execute_captures_memory() {
        let mut rt = WasmiRuntime::new();
        let mut input = Vec::new();
        input.extend_from_slice(&100_i32.to_le_bytes());
        input.extend_from_slice(&200_i32.to_le_bytes());

        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 100_000 },
        );
        assert!(matches!(result, ComputeResult::Complete { .. }));

        let checkpoint = rt.snapshot().expect("should have active session");

        // Module hash should be deterministic
        let expected_hash = harmony_crypto::hash::blake3_hash(ADD_WAT.as_bytes());
        assert_eq!(checkpoint.module_hash, expected_hash);

        // Memory should contain input and output data
        assert!(!checkpoint.memory.is_empty());
        // Input at offset 0: 100 (le bytes)
        let a = i32::from_le_bytes(checkpoint.memory[0..4].try_into().unwrap());
        assert_eq!(a, 100);
        // Input at offset 4: 200 (le bytes)
        let b = i32::from_le_bytes(checkpoint.memory[4..8].try_into().unwrap());
        assert_eq!(b, 200);
        // Output at offset 8: 300 (le bytes)
        let sum = i32::from_le_bytes(checkpoint.memory[8..12].try_into().unwrap());
        assert_eq!(sum, 300);

        assert!(checkpoint.fuel_consumed > 0);
    }

    #[test]
    fn snapshot_without_session_returns_error() {
        let rt = WasmiRuntime::new();
        let result = rt.snapshot();
        assert!(matches!(result, Err(ComputeError::NoPendingExecution)));
    }
```

**Step 2: Implement snapshot()**

Replace the `snapshot()` stub in the `ComputeRuntime` impl block with:

```rust
    fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
        let session = self
            .session
            .as_ref()
            .ok_or(ComputeError::NoPendingExecution)?;

        let memory = session
            .instance
            .get_memory(&session.store, "memory")
            .ok_or_else(|| ComputeError::ExportNotFound {
                name: "memory".into(),
            })?;

        let memory_data = memory.data(&session.store).to_vec();

        Ok(Checkpoint {
            module_hash: session.module_hash,
            memory: memory_data,
            fuel_consumed: session.total_fuel_consumed,
        })
    }
```

**Step 3: Run all tests**

Run: `cargo test -p harmony-compute`
Expected: 23 tests pass (21 prior + 2 snapshot tests)

Run: `cargo clippy -p harmony-compute`
Expected: zero warnings

Run: `cargo test --workspace`
Expected: all workspace tests pass

**Step 4: Commit**

```bash
git add crates/harmony-compute/
git commit -m "feat(compute): checkpoint memory snapshot (Task 5/5)"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Crate scaffolding + error types | 5 |
| 2 | Core types + ComputeRuntime trait | 9 |
| 3 | WasmiRuntime execute() with fuel | 4 |
| 4 | Cooperative yielding — resume() | 3 |
| 5 | Checkpoint memory snapshot | 2 |
| **Total** | | **23** |

### What this delivers
- `ComputeRuntime` trait: abstract WASM execution with cooperative yielding
- `WasmiRuntime`: interpreter implementation with fuel metering
- `InstructionBudget`: configurable fuel per execution slice
- `Checkpoint`: serializable memory snapshot for durable recovery
- `ComputeResult`: Complete / Yielded / NeedsIO / Failed
- Full test coverage of execution, yielding, multi-resume, and checkpointing

### What this does NOT deliver (separate beads)
- `WasmtimeRuntime` JIT implementation (harmony-hmf7)
- `harmony-workflow` durable execution orchestration (harmony-awg6)
- Tier 3 integration into harmony-node event loop (harmony-nrqp)
- Host function I/O (NeedsIO path — defined but not implemented)

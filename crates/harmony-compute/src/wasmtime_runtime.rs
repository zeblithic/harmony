//! WASM execution engine backed by wasmtime with JIT compilation.
//!
//! Alternative to `WasmiRuntime` for desktop/server nodes. Uses JIT compilation
//! for faster execution. Does not support cooperative yielding — modules run to
//! completion or fail on fuel exhaustion. NeedsIO is handled via deterministic
//! replay with a cached IO oracle.

use std::collections::{HashMap, HashSet};

use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state stored in the wasmtime `Store`.
///
/// When a WASM module calls `harmony.fetch_content`, the host function
/// stores the IO request and write target here before trapping.
/// Fields used by Tasks 4-5 when host functions are registered.
#[allow(dead_code)]
struct HostState {
    /// Cached IO responses keyed by CID (for deterministic replay).
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    /// CIDs that were not found during prior IO rounds.
    not_found_cids: HashSet<[u8; 32]>,
    /// IO request captured from the host function call.
    io_request: Option<crate::types::IORequest>,
    /// Where to write the IO response data: (out_ptr, out_cap).
    io_write_target: Option<(u32, u32)>,
}

impl HostState {
    fn new() -> Self {
        Self {
            io_cache: HashMap::new(),
            not_found_cids: HashSet::new(),
            io_request: None,
            io_write_target: None,
        }
    }
}

/// Saved session state for snapshot, take/restore, and future NeedsIO replay.
///
/// Unlike WasmiRuntime, wasmtime does not support resumable calls, so we
/// store enough information to re-execute the module from scratch with
/// accumulated IO cache state.
/// Fields used by Tasks 4-5 for deterministic replay with IO cache.
#[allow(dead_code)]
struct WasmtimeSession {
    /// Original module bytes (needed for re-execution on replay).
    module_bytes: Vec<u8>,
    /// BLAKE3 hash of the module bytes.
    module_hash: [u8; 32],
    /// Original input bytes (needed for re-execution on replay).
    input: Vec<u8>,
    /// Cached IO responses from prior rounds.
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    /// CIDs that returned not-found in prior rounds.
    not_found_cids: HashSet<[u8; 32]>,
    /// Total fuel consumed across all execution slices.
    total_fuel_consumed: u64,
    /// CID of a pending IO request (if execution suspended for IO).
    pending_cid: Option<[u8; 32]>,
    /// Memory snapshot at the end of the last execution.
    memory_snapshot: Vec<u8>,
}

/// Wasmtime-backed WASM execution engine with JIT compilation.
///
/// Unlike `WasmiRuntime`, wasmtime does not support cooperative yielding via
/// resumable calls. When fuel runs out, the call returns an error and the
/// execution cannot be resumed — `resume()` always returns `NoPendingExecution`.
pub struct WasmtimeRuntime {
    engine: wasmtime::Engine,
    session: Option<WasmtimeSession>,
}

impl WasmtimeRuntime {
    /// Create a new `WasmtimeRuntime` with fuel metering enabled.
    pub fn new() -> Result<Self, ComputeError> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        let engine = wasmtime::Engine::new(&config).map_err(|e| ComputeError::Trap {
            reason: format!("failed to create wasmtime engine: {e}"),
        })?;
        Ok(Self {
            engine,
            session: None,
        })
    }

    /// Core execution helper: compiles module, runs `compute`, returns result.
    ///
    /// This is called by `execute()` and (in future tasks) by `resume_with_io()`
    /// which replays execution with cached IO responses.
    fn run_module(
        &self,
        module_bytes: &[u8],
        input: &[u8],
        budget: InstructionBudget,
        host_state: HostState,
    ) -> (
        ComputeResult,
        Option<(HostState, u64, Vec<u8>)>, // (host_state, fuel_consumed, memory_snapshot)
    ) {
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
                    None,
                );
            }
        };

        // Create a store with host state and set fuel budget.
        let mut store = wasmtime::Store::new(&self.engine, host_state);
        if let Err(e) = store.set_fuel(budget.fuel) {
            return (
                ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: format!("failed to set fuel: {e}"),
                    },
                },
                None,
            );
        }

        // Set up Linker — no host functions registered in Task 3.
        let linker = wasmtime::Linker::new(&self.engine);

        // Instantiate the module.
        let instance = match linker.instantiate(&mut store, &module) {
            Ok(inst) => inst,
            Err(e) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: format!("instantiation failed: {e}"),
                        },
                    },
                    None,
                );
            }
        };

        // Get the exported memory.
        let memory = match instance.get_memory(&mut store, "memory") {
            Some(mem) => mem,
            None => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::ExportNotFound {
                            name: "memory".into(),
                        },
                    },
                    None,
                );
            }
        };

        // Check that memory is large enough for the input.
        let mem_size = memory.data_size(&store);
        if input.len() > mem_size {
            return (
                ComputeResult::Failed {
                    error: ComputeError::MemoryTooSmall {
                        need: input.len(),
                        have: mem_size,
                    },
                },
                None,
            );
        }

        // Write input bytes into memory at offset 0.
        if memory.write(&mut store, 0, input).is_err() {
            return (
                ComputeResult::Failed {
                    error: ComputeError::MemoryTooSmall {
                        need: input.len(),
                        have: memory.data_size(&store),
                    },
                },
                None,
            );
        }

        // Get the exported `compute` function with the expected signature.
        let compute_func = match instance.get_typed_func::<(i32, i32), i32>(&mut store, "compute") {
            Ok(f) => f,
            Err(_) => {
                return (
                    ComputeResult::Failed {
                        error: ComputeError::ExportNotFound {
                            name: "compute".into(),
                        },
                    },
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
                    None,
                );
            }
        };

        // Call compute — wasmtime runs to completion or fails (no resumable calls).
        match compute_func.call(&mut store, (0i32, input_len_i32)) {
            Ok(output_len) => {
                // Validate output length (negative = WASM-side error).
                if output_len < 0 {
                    return (
                        ComputeResult::Failed {
                            error: ComputeError::Trap {
                                reason: format!("compute returned negative length: {output_len}"),
                            },
                        },
                        None,
                    );
                }
                let output_len_usize = output_len as usize;

                // Bounds-check before reading output from memory.
                let output_end = input_len + output_len_usize;
                let current_mem_size = memory.data_size(&store);
                if output_end > current_mem_size {
                    return (
                        ComputeResult::Failed {
                            error: ComputeError::MemoryTooSmall {
                                need: output_end,
                                have: current_mem_size,
                            },
                        },
                        None,
                    );
                }

                let mut output = vec![0u8; output_len_usize];
                if !output.is_empty() {
                    memory
                        .read(&store, input_len, &mut output)
                        .expect("bounds already checked");
                }

                // Compute fuel consumed.
                let remaining_fuel = store.get_fuel().unwrap_or(0);
                let fuel_consumed = budget.fuel.saturating_sub(remaining_fuel);

                // Capture memory snapshot.
                let mem_snapshot = memory.data(&store).to_vec();

                let host_data = store.into_data();

                (
                    ComputeResult::Complete { output },
                    Some((host_data, fuel_consumed, mem_snapshot)),
                )
            }
            Err(e) => {
                // When fuel runs out, wasmtime returns an error (not a resumable state).
                // We report this as a Failed result.
                let remaining_fuel = store.get_fuel().unwrap_or(0);
                let fuel_consumed = budget.fuel.saturating_sub(remaining_fuel);

                // Capture memory snapshot even on failure (for diagnostics).
                let mem_snapshot = memory.data(&store).to_vec();
                let host_data = store.into_data();

                (
                    ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: e.to_string(),
                        },
                    },
                    Some((host_data, fuel_consumed, mem_snapshot)),
                )
            }
        }
    }
}

impl Default for WasmtimeRuntime {
    fn default() -> Self {
        Self::new().expect("failed to create default WasmtimeRuntime")
    }
}

impl ComputeRuntime for WasmtimeRuntime {
    fn execute(
        &mut self,
        module_bytes: &[u8],
        input: &[u8],
        budget: InstructionBudget,
    ) -> ComputeResult {
        // Drop any existing session.
        self.session = None;

        // Hash the module bytes for identity tracking.
        let module_hash = harmony_crypto::hash::blake3_hash(module_bytes);

        let host_state = HostState::new();

        let (result, session_data) = self.run_module(module_bytes, input, budget, host_state);

        // Store session for snapshot/take_session regardless of outcome.
        if let Some((host_data, fuel_consumed, mem_snapshot)) = session_data {
            self.session = Some(WasmtimeSession {
                module_bytes: module_bytes.to_vec(),
                module_hash,
                input: input.to_vec(),
                io_cache: host_data.io_cache,
                not_found_cids: host_data.not_found_cids,
                total_fuel_consumed: fuel_consumed,
                pending_cid: None,
                memory_snapshot: mem_snapshot,
            });
        }

        result
    }

    fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
        // Wasmtime does not support cooperative yielding — there is never a
        // pending execution to resume. Preserve session so snapshot() still works.
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn resume_with_io(
        &mut self,
        _response: crate::types::IOResponse,
        _budget: InstructionBudget,
    ) -> ComputeResult {
        // Placeholder — Task 5 implements NeedsIO support via deterministic replay.
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn has_pending(&self) -> bool {
        // Wasmtime does not support cooperative yielding, so there is never
        // a pending execution.
        false
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
            memory: session.memory_snapshot.clone(),
            fuel_consumed: session.total_fuel_consumed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::ComputeRuntime;
    use crate::types::InstructionBudget;

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

    /// WAT module: loops N times (fuel-hungry). Reads iteration count from input,
    /// writes final counter to output. With small fuel budgets, this exhausts fuel.
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
    fn execute_add_module() {
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        // Input: two i32s in little-endian: 3 and 7
        let mut input = Vec::new();
        input.extend_from_slice(&3i32.to_le_bytes());
        input.extend_from_slice(&7i32.to_le_bytes());

        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                assert_eq!(output, 10i32.to_le_bytes().to_vec());
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
    }

    #[test]
    fn execute_completes_without_yielding() {
        // With enough fuel, the loop module completes — no Yielded variant
        // because wasmtime does not support cooperative yielding.
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let input = 1_000_i32.to_le_bytes();

        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                let counter = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(counter, 1_000);
            }
            other => panic!("expected Complete (no yielding in wasmtime), got: {other:?}"),
        }
    }

    #[test]
    fn fuel_exhaustion_returns_failed() {
        // With very small fuel and a big loop, wasmtime runs out of fuel
        // and returns Failed (not Yielded, since there is no resumable call).
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let input = 1_000_000_i32.to_le_bytes();

        let result = rt.execute(LOOP_WAT.as_bytes(), &input, InstructionBudget { fuel: 100 });

        assert!(
            matches!(
                result,
                ComputeResult::Failed {
                    error: ComputeError::Trap { .. }
                }
            ),
            "expected Failed with Trap (fuel exhaustion), got: {result:?}"
        );
    }

    #[test]
    fn resume_returns_no_pending() {
        // resume() always returns NoPendingExecution for wasmtime.
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let result = rt.resume(InstructionBudget { fuel: 1_000 });

        assert!(matches!(
            result,
            ComputeResult::Failed {
                error: ComputeError::NoPendingExecution
            }
        ));
    }

    #[test]
    fn has_pending_is_false_after_complete() {
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let mut input = Vec::new();
        input.extend_from_slice(&3i32.to_le_bytes());
        input.extend_from_slice(&7i32.to_le_bytes());

        let result = rt.execute(
            ADD_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::Complete { .. }));

        assert!(
            !rt.has_pending(),
            "has_pending() should always be false for wasmtime"
        );
    }
}

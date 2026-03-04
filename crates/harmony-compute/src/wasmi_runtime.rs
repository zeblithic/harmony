use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state stored in the wasmi `Store`. Currently empty.
struct HostState;

/// Holds the active WASM execution session state for resumption.
///
/// Fields are populated during `execute()` and consumed by `resume()` (Task 4)
/// and `snapshot()` (Task 5).
#[allow(dead_code)]
struct WasmiSession {
    store: wasmi::Store<HostState>,
    instance: wasmi::Instance,
    module_hash: [u8; 32],
    input_len: usize,
    total_fuel_consumed: u64,
    pending: Option<wasmi::TypedResumableCallOutOfFuel<i32>>,
}

/// WASM execution engine backed by wasmi with fuel metering.
///
/// Uses wasmi's fuel-based metering and `call_resumable()` API to support
/// cooperative yielding. When fuel runs out mid-execution, the pending
/// invocation is stored for later resumption.
pub struct WasmiRuntime {
    engine: wasmi::Engine,
    session: Option<WasmiSession>,
}

impl WasmiRuntime {
    /// Create a new `WasmiRuntime` with fuel metering enabled.
    pub fn new() -> Self {
        let mut config = wasmi::Config::default();
        config.consume_fuel(true);
        let engine = wasmi::Engine::new(&config);
        Self {
            engine,
            session: None,
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
        // Drop any existing session.
        self.session = None;

        // Hash the module bytes for identity tracking.
        let module_hash = harmony_crypto::hash::blake3_hash(module_bytes);

        // Compile the WASM module.
        let module = match wasmi::Module::new(&self.engine, module_bytes) {
            Ok(m) => m,
            Err(e) => {
                return ComputeResult::Failed {
                    error: ComputeError::InvalidModule {
                        reason: e.to_string(),
                    },
                };
            }
        };

        // Create a store with host state and set fuel budget.
        let mut store = wasmi::Store::new(&self.engine, HostState);
        if let Err(e) = store.set_fuel(budget.fuel) {
            return ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: format!("failed to set fuel: {e}"),
                },
            };
        }

        // Instantiate via Linker (no imports needed for our ABI).
        let linker = <wasmi::Linker<HostState>>::new(&self.engine);
        let instance = match linker.instantiate_and_start(&mut store, &module) {
            Ok(inst) => inst,
            Err(e) => {
                return ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: format!("instantiation failed: {e}"),
                    },
                };
            }
        };

        // Get the exported memory.
        let memory = match instance.get_memory(&store, "memory") {
            Some(mem) => mem,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::ExportNotFound {
                        name: "memory".into(),
                    },
                };
            }
        };

        // Check that memory is large enough for the input.
        let mem_size = memory.data_size(&store);
        if input.len() > mem_size {
            return ComputeResult::Failed {
                error: ComputeError::MemoryTooSmall {
                    need: input.len(),
                    have: mem_size,
                },
            };
        }

        // Write input bytes into memory at offset 0.
        if memory.write(&mut store, 0, input).is_err() {
            return ComputeResult::Failed {
                error: ComputeError::MemoryTooSmall {
                    need: input.len(),
                    have: memory.data_size(&store),
                },
            };
        }

        // Get the exported `compute` function with the expected signature.
        let compute_func: wasmi::TypedFunc<(i32, i32), i32> =
            match instance.get_typed_func::<(i32, i32), i32>(&store, "compute") {
                Ok(f) => f,
                Err(_) => {
                    return ComputeResult::Failed {
                        error: ComputeError::ExportNotFound {
                            name: "compute".into(),
                        },
                    };
                }
            };

        let input_len = input.len();

        // Call compute with resumable API to support fuel exhaustion.
        let call_result = compute_func.call_resumable(&mut store, (0i32, input_len as i32));

        match call_result {
            Ok(wasmi::TypedResumableCall::Finished(output_len)) => {
                // Validate output length (negative = WASM-side error).
                if output_len < 0 {
                    return ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: format!(
                                "compute returned negative length: {output_len}"
                            ),
                        },
                    };
                }
                let output_len = output_len as usize;

                // Bounds-check before reading output from memory.
                let output_end = input_len + output_len;
                let mem_size = memory.data_size(&store);
                if output_end > mem_size {
                    return ComputeResult::Failed {
                        error: ComputeError::MemoryTooSmall {
                            need: output_end,
                            have: mem_size,
                        },
                    };
                }

                let mut output = vec![0u8; output_len];
                if !output.is_empty() {
                    memory
                        .read(&store, input_len, &mut output)
                        .expect("bounds already checked");
                }

                ComputeResult::Complete { output }
            }
            Ok(wasmi::TypedResumableCall::OutOfFuel(pending)) => {
                // Execution ran out of fuel — store session for resumption.
                let remaining_fuel = store.get_fuel().unwrap_or(0);
                let fuel_consumed = budget.fuel.saturating_sub(remaining_fuel);

                self.session = Some(WasmiSession {
                    store,
                    instance,
                    module_hash,
                    input_len,
                    total_fuel_consumed: fuel_consumed,
                    pending: Some(pending),
                });

                ComputeResult::Yielded { fuel_consumed }
            }
            Ok(wasmi::TypedResumableCall::HostTrap(_)) => {
                // Unexpected host trap (we have no host functions).
                ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: "unexpected host trap".into(),
                    },
                }
            }
            Err(e) => ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: e.to_string(),
                },
            },
        }
    }

    fn resume(&mut self, _budget: InstructionBudget) -> ComputeResult {
        // Stub: full implementation in Task 4.
        ComputeResult::Failed {
            error: ComputeError::NoPendingExecution,
        }
    }

    fn has_pending(&self) -> bool {
        self.session.as_ref().is_some_and(|s| s.pending.is_some())
    }

    fn snapshot(&self) -> Result<Checkpoint, ComputeError> {
        // Stub: full implementation in Task 5.
        Err(ComputeError::NoPendingExecution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn execute_add_module() {
        let mut rt = WasmiRuntime::new();
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
    fn execute_invalid_module() {
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(b"not wasm", b"", InstructionBudget { fuel: 1_000_000 });

        assert!(
            matches!(
                result,
                ComputeResult::Failed {
                    error: ComputeError::InvalidModule { .. }
                }
            ),
            "expected Failed with InvalidModule, got: {result:?}"
        );
    }

    #[test]
    fn execute_missing_compute_export() {
        // WAT with memory but no compute export.
        let wat = r#"
            (module
              (memory (export "memory") 1))
        "#;
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(wat.as_bytes(), b"", InstructionBudget { fuel: 1_000_000 });

        match &result {
            ComputeResult::Failed {
                error: ComputeError::ExportNotFound { name },
            } => {
                assert_eq!(name, "compute");
            }
            other => panic!("expected Failed with ExportNotFound(compute), got: {other:?}"),
        }
    }

    #[test]
    fn execute_missing_memory_export() {
        // WAT with compute but no memory export.
        let wat = r#"
            (module
              (memory 1)
              (func (export "compute") (param i32) (param i32) (result i32)
                (i32.const 0)))
        "#;
        let mut rt = WasmiRuntime::new();
        let result = rt.execute(wat.as_bytes(), b"", InstructionBudget { fuel: 1_000_000 });

        match &result {
            ComputeResult::Failed {
                error: ComputeError::ExportNotFound { name },
            } => {
                assert_eq!(name, "memory");
            }
            other => panic!("expected Failed with ExportNotFound(memory), got: {other:?}"),
        }
    }
}

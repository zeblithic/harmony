use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state stored in the wasmi `Store`. Currently empty.
struct HostState;

/// Holds the active WASM execution session state for resumption.
///
/// Fields are populated during `execute()` and consumed by `resume()`
/// and `snapshot()`.
struct WasmiSession {
    store: wasmi::Store<HostState>,
    instance: wasmi::Instance,
    module_hash: [u8; 32],
    input_len: usize,
    total_fuel_consumed: u64,
    pending: Option<wasmi::TypedResumableCallOutOfFuel<i32>>,
}

/// Execution context passed to `handle_call_result` to avoid too many arguments.
struct ExecContext {
    store: wasmi::Store<HostState>,
    instance: wasmi::Instance,
    module_hash: [u8; 32],
    input_len: usize,
    /// Total fuel consumed in prior execution slices.
    fuel_before: u64,
    /// Fuel budget for the current slice.
    fuel_budget: u64,
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

    /// Handle the result of a resumable call (shared between `execute()` and `resume()`).
    ///
    /// Processes the four possible outcomes from `call_resumable()` / `invocation.resume()`:
    /// - `Finished(output_len)` -- read output from memory, return `Complete`
    /// - `OutOfFuel(pending)` -- store session for later resumption, return `Yielded`
    /// - `HostTrap` -- return `Failed` (unexpected, we have no host functions)
    /// - `Err` -- return `Failed` with the trap reason
    fn handle_call_result(
        &mut self,
        call_result: Result<wasmi::TypedResumableCall<i32>, wasmi::Error>,
        ctx: ExecContext,
    ) -> ComputeResult {
        match call_result {
            Ok(wasmi::TypedResumableCall::Finished(output_len)) => {
                // Validate output length (negative = WASM-side error).
                if output_len < 0 {
                    return ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: format!("compute returned negative length: {output_len}"),
                        },
                    };
                }
                let output_len = output_len as usize;

                // Get the exported memory from the instance.
                let memory = match ctx.instance.get_memory(&ctx.store, "memory") {
                    Some(mem) => mem,
                    None => {
                        return ComputeResult::Failed {
                            error: ComputeError::ExportNotFound {
                                name: "memory".into(),
                            },
                        };
                    }
                };

                // Bounds-check before reading output from memory.
                let output_end = ctx.input_len + output_len;
                let mem_size = memory.data_size(&ctx.store);
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
                        .read(&ctx.store, ctx.input_len, &mut output)
                        .expect("bounds already checked");
                }

                // Compute fuel consumed and store session for snapshot().
                // get_fuel() only fails when fuel metering is disabled;
                // WasmiRuntime::new() always enables it.
                let remaining_fuel = ctx.store.get_fuel().unwrap_or(0);
                let fuel_this_slice = ctx.fuel_budget.saturating_sub(remaining_fuel);
                let total_fuel = ctx.fuel_before + fuel_this_slice;

                self.session = Some(WasmiSession {
                    store: ctx.store,
                    instance: ctx.instance,
                    module_hash: ctx.module_hash,
                    input_len: ctx.input_len,
                    total_fuel_consumed: total_fuel,
                    pending: None,
                });

                ComputeResult::Complete { output }
            }
            Ok(wasmi::TypedResumableCall::OutOfFuel(pending)) => {
                // Execution ran out of fuel -- store session for resumption.
                // get_fuel() only fails when fuel metering is disabled;
                // WasmiRuntime::new() always enables it.
                let remaining_fuel = ctx.store.get_fuel().unwrap_or(0);
                let fuel_this_slice = ctx.fuel_budget.saturating_sub(remaining_fuel);
                let total_fuel = ctx.fuel_before + fuel_this_slice;

                self.session = Some(WasmiSession {
                    store: ctx.store,
                    instance: ctx.instance,
                    module_hash: ctx.module_hash,
                    input_len: ctx.input_len,
                    total_fuel_consumed: total_fuel,
                    pending: Some(pending),
                });

                ComputeResult::Yielded {
                    fuel_consumed: total_fuel,
                }
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

        self.handle_call_result(
            call_result,
            ExecContext {
                store,
                instance,
                module_hash,
                input_len,
                fuel_before: 0,
                fuel_budget: budget.fuel,
            },
        )
    }

    fn resume(&mut self, budget: InstructionBudget) -> ComputeResult {
        let mut session = match self.session.take() {
            Some(s) => s,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        let invocation = match session.pending.take() {
            Some(inv) => inv,
            None => {
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        session
            .store
            .set_fuel(budget.fuel)
            .expect("fuel metering is enabled");

        let fuel_before = session.total_fuel_consumed;
        let input_len = session.input_len;
        let module_hash = session.module_hash;
        let instance = session.instance;

        let call_result = invocation.resume(&mut session.store);

        self.handle_call_result(
            call_result,
            ExecContext {
                store: session.store,
                instance,
                module_hash,
                input_len,
                fuel_before,
                fuel_budget: budget.fuel,
            },
        )
    }

    fn has_pending(&self) -> bool {
        self.session.as_ref().is_some_and(|s| s.pending.is_some())
    }

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

        // Small fuel budget — should yield before completing 10k iterations.
        // Must be large enough for wasmi to execute at least one basic block
        // (fuel is consumed at block granularity).
        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 10_000 },
        );
        assert!(
            matches!(result, ComputeResult::Yielded { .. }),
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

        // Use a budget large enough for some iterations but not all 50k.
        let mut result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 10_000 },
        );

        let mut resume_count = 0;
        loop {
            match result {
                ComputeResult::Yielded { .. } => {
                    resume_count += 1;
                    result = rt.resume(InstructionBudget { fuel: 50_000 });
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
    fn snapshot_after_yield_captures_memory() {
        let mut rt = WasmiRuntime::new();
        let input = 10_000_i32.to_le_bytes();

        let result = rt.execute(
            LOOP_WAT.as_bytes(),
            &input,
            InstructionBudget { fuel: 10_000 },
        );

        let yielded_fuel = match result {
            ComputeResult::Yielded { fuel_consumed } => fuel_consumed,
            other => panic!("expected Yielded, got {other:?}"),
        };

        let checkpoint = rt.snapshot().expect("should have active session after yield");

        let expected_hash = harmony_crypto::hash::blake3_hash(LOOP_WAT.as_bytes());
        assert_eq!(checkpoint.module_hash, expected_hash);
        assert!(!checkpoint.memory.is_empty());
        assert_eq!(checkpoint.fuel_consumed, yielded_fuel);
    }

    #[test]
    fn snapshot_without_session_returns_error() {
        let rt = WasmiRuntime::new();
        let result = rt.snapshot();
        assert!(matches!(result, Err(ComputeError::NoPendingExecution)));
    }
}

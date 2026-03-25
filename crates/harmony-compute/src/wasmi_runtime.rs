use crate::error::ComputeError;
use crate::runtime::ComputeRuntime;
use crate::types::{Checkpoint, ComputeResult, InstructionBudget};

/// Host state stored in the wasmi `Store`.
///
/// When a WASM module calls `harmony.fetch_content`, the host function
/// stores the IO request and write target here before trapping.
struct HostState {
    /// IO request captured from the host function call.
    io_request: Option<crate::types::IORequest>,
    /// Where to write the IO response data: (out_ptr, out_cap).
    io_write_target: Option<(u32, u32)>,
    /// Loaded inference engine (only available with `inference` feature).
    #[cfg(feature = "inference")]
    inference_engine: Option<harmony_inference::QwenEngine>,
    /// Logits from the most recent `forward()` call, consumed by `sample()`.
    #[cfg(feature = "inference")]
    last_logits: Option<Vec<f32>>,
}

impl HostState {
    fn new() -> Self {
        Self {
            io_request: None,
            io_write_target: None,
            #[cfg(feature = "inference")]
            inference_engine: None,
            #[cfg(feature = "inference")]
            last_logits: None,
        }
    }
}

/// A suspended WASM execution that can be resumed.
enum PendingResumable {
    /// Execution ran out of fuel (cooperative yield).
    OutOfFuel(wasmi::TypedResumableCallOutOfFuel<i32>),
    /// Execution trapped on a host function (IO request).
    HostTrap(wasmi::TypedResumableCallHostTrap<i32>),
}

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
    pending: Option<PendingResumable>,
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
    /// - `HostTrap` -- check HostState for IO request; return `NeedsIO` or `Failed`
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
                    pending: Some(PendingResumable::OutOfFuel(pending)),
                });

                ComputeResult::Yielded {
                    fuel_consumed: total_fuel,
                }
            }
            Ok(wasmi::TypedResumableCall::HostTrap(pending)) => {
                // Check HostState for an IO request placed by a host function.
                if let Some(request) = ctx.store.data().io_request.clone() {
                    let remaining_fuel = ctx.store.get_fuel().unwrap_or(0);
                    let fuel_this_slice = ctx.fuel_budget.saturating_sub(remaining_fuel);
                    let total_fuel = ctx.fuel_before + fuel_this_slice;

                    self.session = Some(WasmiSession {
                        store: ctx.store,
                        instance: ctx.instance,
                        module_hash: ctx.module_hash,
                        input_len: ctx.input_len,
                        total_fuel_consumed: total_fuel,
                        pending: Some(PendingResumable::HostTrap(pending)),
                    });
                    ComputeResult::NeedsIO { request }
                } else {
                    ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: "unexpected host trap with no IO request".into(),
                        },
                    }
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
        let mut store = wasmi::Store::new(&self.engine, HostState::new());
        if let Err(e) = store.set_fuel(budget.fuel) {
            return ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: format!("failed to set fuel: {e}"),
                },
            };
        }

        // Set up Linker with the harmony.fetch_content host import.
        let mut linker = <wasmi::Linker<HostState>>::new(&self.engine);
        linker
            .func_wrap(
                "harmony",
                "fetch_content",
                |mut caller: wasmi::Caller<'_, HostState>,
                 cid_ptr: i32,
                 out_ptr: i32,
                 out_cap: i32|
                 -> Result<i32, wasmi::Error> {
                    // Read 32 bytes from WASM memory at cid_ptr.
                    let memory = caller
                        .get_export("memory")
                        .and_then(wasmi::Extern::into_memory)
                        .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                    let mut cid = [0u8; 32];
                    memory
                        .read(&caller, cid_ptr as usize, &mut cid)
                        .map_err(|e| wasmi::Error::new(format!("failed to read CID: {e}")))?;

                    // Store the IO request and write target in HostState.
                    let data = caller.data_mut();
                    data.io_request = Some(crate::types::IORequest::FetchContent { cid });
                    data.io_write_target = Some((out_ptr as u32, out_cap as u32));

                    // Trap to suspend execution — the caller will provide
                    // the IO response and resume later.
                    Err(wasmi::Error::new("harmony_io_trap"))
                },
            )
            .expect("failed to register harmony.fetch_content");

        // ---- Inference host functions (feature-gated) ----
        #[cfg(feature = "inference")]
        {
            use harmony_inference::InferenceEngine as _;

            linker
                .func_wrap(
                    "harmony",
                    "model_load",
                    |mut caller: wasmi::Caller<'_, HostState>,
                     gguf_cid_ptr: i32,
                     tokenizer_cid_ptr: i32|
                     -> Result<i32, wasmi::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(wasmi::Extern::into_memory)
                            .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                        let mut gguf_cid = [0u8; 32];
                        memory
                            .read(&caller, gguf_cid_ptr as usize, &mut gguf_cid)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to read GGUF CID: {e}"))
                            })?;

                        let mut tokenizer_cid = [0u8; 32];
                        memory
                            .read(&caller, tokenizer_cid_ptr as usize, &mut tokenizer_cid)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to read tokenizer CID: {e}"))
                            })?;

                        let data = caller.data_mut();
                        data.io_request = Some(crate::types::IORequest::LoadModel {
                            gguf_cid,
                            tokenizer_cid,
                        });
                        // No io_write_target — model_load returns via host function return value

                        Err(wasmi::Error::new("harmony_io_trap"))
                    },
                )
                .expect("failed to register harmony.model_load");

            linker
                .func_wrap(
                    "harmony",
                    "tokenize",
                    |mut caller: wasmi::Caller<'_, HostState>,
                     text_ptr: i32,
                     text_len: i32,
                     out_ptr: i32,
                     out_cap: i32|
                     -> Result<i32, wasmi::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(wasmi::Extern::into_memory)
                            .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                        let engine = match &caller.data().inference_engine {
                            Some(e) => e,
                            None => return Ok(-1),
                        };

                        let mut text_buf = vec![0u8; text_len as usize];
                        memory
                            .read(&caller, text_ptr as usize, &mut text_buf)
                            .map_err(|e| wasmi::Error::new(format!("failed to read text: {e}")))?;

                        let text = std::str::from_utf8(&text_buf)
                            .map_err(|e| wasmi::Error::new(format!("invalid UTF-8: {e}")))?;

                        let tokens = match engine.tokenize(text) {
                            Ok(t) => t,
                            Err(e) => {
                                return Err(wasmi::Error::new(format!("tokenize failed: {e}")))
                            }
                        };

                        let token_bytes: Vec<u8> =
                            tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

                        if token_bytes.len() > out_cap as usize {
                            return Ok(-2);
                        }

                        memory
                            .write(&mut caller, out_ptr as usize, &token_bytes)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to write tokens: {e}"))
                            })?;

                        Ok(token_bytes.len() as i32)
                    },
                )
                .expect("failed to register harmony.tokenize");

            linker
                .func_wrap(
                    "harmony",
                    "detokenize",
                    |mut caller: wasmi::Caller<'_, HostState>,
                     tokens_ptr: i32,
                     tokens_len: i32,
                     out_ptr: i32,
                     out_cap: i32|
                     -> Result<i32, wasmi::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(wasmi::Extern::into_memory)
                            .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                        if tokens_len % 4 != 0 {
                            return Ok(-3);
                        }

                        let engine = match &caller.data().inference_engine {
                            Some(e) => e,
                            None => return Ok(-1),
                        };

                        let mut token_bytes = vec![0u8; tokens_len as usize];
                        memory
                            .read(&caller, tokens_ptr as usize, &mut token_bytes)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to read tokens: {e}"))
                            })?;

                        let tokens: Vec<u32> = token_bytes
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();

                        let text = match engine.detokenize(&tokens) {
                            Ok(t) => t,
                            Err(e) => {
                                return Err(wasmi::Error::new(format!("detokenize failed: {e}")))
                            }
                        };

                        if text.len() > out_cap as usize {
                            return Ok(-2);
                        }

                        memory
                            .write(&mut caller, out_ptr as usize, text.as_bytes())
                            .map_err(|e| wasmi::Error::new(format!("failed to write text: {e}")))?;

                        Ok(text.len() as i32)
                    },
                )
                .expect("failed to register harmony.detokenize");

            linker
                .func_wrap(
                    "harmony",
                    "forward",
                    |mut caller: wasmi::Caller<'_, HostState>,
                     tokens_ptr: i32,
                     tokens_len: i32|
                     -> Result<i32, wasmi::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(wasmi::Extern::into_memory)
                            .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                        if tokens_len % 4 != 0 || tokens_len == 0 {
                            return Ok(-3);
                        }

                        let mut token_bytes = vec![0u8; tokens_len as usize];
                        memory
                            .read(&caller, tokens_ptr as usize, &mut token_bytes)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to read tokens: {e}"))
                            })?;

                        let tokens: Vec<u32> = token_bytes
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();

                        let engine = match caller.data_mut().inference_engine.as_mut() {
                            Some(e) => e,
                            None => return Ok(-1),
                        };

                        let logits = match engine.forward(&tokens) {
                            Ok(l) => l,
                            Err(_) => return Ok(-2),
                        };

                        caller.data_mut().last_logits = Some(logits);
                        Ok(0)
                    },
                )
                .expect("failed to register harmony.forward");

            linker
                .func_wrap(
                    "harmony",
                    "sample",
                    |caller: wasmi::Caller<'_, HostState>,
                     params_ptr: i32,
                     params_len: i32|
                     -> Result<i32, wasmi::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(wasmi::Extern::into_memory)
                            .ok_or_else(|| wasmi::Error::new("missing exported memory"))?;

                        if params_len != 20 {
                            return Ok(-2);
                        }

                        if caller.data().inference_engine.is_none() {
                            return Ok(-1);
                        }

                        let mut param_bytes = [0u8; 20];
                        memory
                            .read(&caller, params_ptr as usize, &mut param_bytes)
                            .map_err(|e| {
                                wasmi::Error::new(format!("failed to read params: {e}"))
                            })?;

                        let params = harmony_inference::SamplingParams {
                            temperature: f32::from_le_bytes([
                                param_bytes[0],
                                param_bytes[1],
                                param_bytes[2],
                                param_bytes[3],
                            ]),
                            top_p: f32::from_le_bytes([
                                param_bytes[4],
                                param_bytes[5],
                                param_bytes[6],
                                param_bytes[7],
                            ]),
                            top_k: u32::from_le_bytes([
                                param_bytes[8],
                                param_bytes[9],
                                param_bytes[10],
                                param_bytes[11],
                            ]),
                            repeat_penalty: f32::from_le_bytes([
                                param_bytes[12],
                                param_bytes[13],
                                param_bytes[14],
                                param_bytes[15],
                            ]),
                            repeat_last_n: u32::from_le_bytes([
                                param_bytes[16],
                                param_bytes[17],
                                param_bytes[18],
                                param_bytes[19],
                            ]) as usize,
                        };

                        // Clone logits to avoid simultaneous borrows of caller.data()
                        let logits = match caller.data().last_logits.as_deref() {
                            Some(l) => l.to_vec(),
                            None => return Ok(-2),
                        };

                        let engine = caller.data().inference_engine.as_ref().unwrap();
                        match engine.sample(&logits, &params) {
                            Ok(token_id) => Ok(token_id as i32),
                            Err(_) => Ok(-2),
                        }
                    },
                )
                .expect("failed to register harmony.sample");

            linker
                .func_wrap(
                    "harmony",
                    "model_reset",
                    |mut caller: wasmi::Caller<'_, HostState>| -> Result<i32, wasmi::Error> {
                        let data = caller.data_mut();
                        if let Some(engine) = &mut data.inference_engine {
                            engine.reset();
                        }
                        data.last_logits = None;
                        Ok(0)
                    },
                )
                .expect("failed to register harmony.model_reset");
        }

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
        let input_len_i32: i32 = match input_len.try_into() {
            Ok(v) => v,
            Err(_) => {
                return ComputeResult::Failed {
                    error: ComputeError::MemoryTooSmall {
                        need: input_len,
                        have: i32::MAX as usize,
                    },
                };
            }
        };

        // Call compute with resumable API to support fuel exhaustion.
        let call_result = compute_func.call_resumable(&mut store, (0i32, input_len_i32));

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
            Some(PendingResumable::OutOfFuel(inv)) => inv,
            Some(host_trap @ PendingResumable::HostTrap(_)) => {
                // HostTrap requires resume_with_io(), not resume().
                // Restore the pending state so resume_with_io() can use it.
                session.pending = Some(host_trap);
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: "pending execution requires IO response; use resume_with_io()"
                            .into(),
                    },
                };
            }
            None => {
                // Restore session so snapshot() still works.
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        if let Err(e) = session.store.set_fuel(budget.fuel) {
            self.session = Some(session);
            return ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: format!("failed to set fuel: {e}"),
                },
            };
        }

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

        let invocation = match session.pending.take() {
            Some(PendingResumable::HostTrap(inv)) => inv,
            Some(out_of_fuel @ PendingResumable::OutOfFuel(_)) => {
                // OutOfFuel requires resume(), not resume_with_io().
                // Restore the pending state so resume() can use it.
                session.pending = Some(out_of_fuel);
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
            None => {
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::NoPendingExecution,
                };
            }
        };

        // Branch on the type of pending IO request.
        let is_model_load = matches!(
            session.store.data().io_request,
            Some(crate::types::IORequest::LoadModel { .. })
        );

        let return_val: i32 = if is_model_load {
            #[cfg(feature = "inference")]
            {
                match response {
                    crate::types::IOResponse::ModelReady {
                        gguf_data,
                        tokenizer_data,
                    } => {
                        use harmony_inference::InferenceEngine;
                        let mut engine =
                            harmony_inference::QwenEngine::new(candle_core::Device::Cpu);
                        if engine.load_gguf(&gguf_data).is_err() {
                            -3
                        } else if engine.load_tokenizer(&tokenizer_data).is_err() {
                            -4
                        } else {
                            let data = session.store.data_mut();
                            data.inference_engine = Some(engine);
                            data.last_logits = None;
                            0
                        }
                    }
                    crate::types::IOResponse::ModelGgufNotFound => -1,
                    crate::types::IOResponse::ModelTokenizerNotFound => -2,
                    _ => -3,
                }
            }
            #[cfg(not(feature = "inference"))]
            {
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: "inference feature not enabled".into(),
                    },
                };
            }
        } else {
            // Existing FetchContent logic — read io_write_target and write data.
            let (out_ptr, out_cap) = match session.store.data().io_write_target {
                Some(target) => target,
                None => {
                    self.session = Some(session);
                    return ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: "resume_with_io: no write target stored".into(),
                        },
                    };
                }
            };

            match &response {
                crate::types::IOResponse::ContentReady { data } => {
                    if data.len() > out_cap as usize {
                        -2
                    } else {
                        let memory = match session.instance.get_memory(&session.store, "memory") {
                            Some(mem) => mem,
                            None => {
                                self.session = Some(session);
                                return ComputeResult::Failed {
                                    error: ComputeError::ExportNotFound {
                                        name: "memory".into(),
                                    },
                                };
                            }
                        };
                        if memory
                            .write(&mut session.store, out_ptr as usize, data)
                            .is_err()
                        {
                            let have = memory.data_size(&session.store);
                            self.session = Some(session);
                            return ComputeResult::Failed {
                                error: ComputeError::MemoryTooSmall {
                                    need: out_ptr as usize + data.len(),
                                    have,
                                },
                            };
                        }
                        data.len() as i32
                    }
                }
                crate::types::IOResponse::ContentNotFound => -1,
                _ => {
                    self.session = Some(session);
                    return ComputeResult::Failed {
                        error: ComputeError::Trap {
                            reason: "unexpected IOResponse for FetchContent".into(),
                        },
                    };
                }
            }
        };

        // Clear the IO state from HostState.
        let host = session.store.data_mut();
        host.io_request = None;
        host.io_write_target = None;

        // Set fuel budget for the resumed execution.
        if let Err(e) = session.store.set_fuel(budget.fuel) {
            self.session = Some(session);
            return ComputeResult::Failed {
                error: ComputeError::Trap {
                    reason: format!("failed to set fuel: {e}"),
                },
            };
        }

        let fuel_before = session.total_fuel_consumed;
        let input_len = session.input_len;
        let module_hash = session.module_hash;
        let instance = session.instance;

        // Resume the host trap with the return value.
        let call_result = invocation.resume(&mut session.store, &[wasmi::Val::I32(return_val)]);

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

    fn take_session(&mut self) -> Option<Box<dyn std::any::Any>> {
        self.session
            .take()
            .map(|s| Box::new(s) as Box<dyn std::any::Any>)
    }

    fn restore_session(&mut self, session: Box<dyn std::any::Any>) {
        self.session = Some(
            *session
                .downcast::<WasmiSession>()
                .expect("restore_session: type mismatch — expected WasmiSession"),
        );
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

        let checkpoint = rt
            .snapshot()
            .expect("should have active session after yield");

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

    /// WAT module that imports harmony.fetch_content and calls it.
    /// Input: [cid: 32 bytes]
    /// Output: [result_code: i32 LE] [fetched_data if result > 0]
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
        use crate::types::IORequest;

        let mut rt = WasmiRuntime::new();
        let cid = [0xAB; 32];
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::NeedsIO {
                request: IORequest::FetchContent { cid: got_cid },
            } => {
                assert_eq!(got_cid, cid);
            }
            other => panic!("expected NeedsIO with FetchContent, got: {other:?}"),
        }
        assert!(
            rt.has_pending(),
            "should have pending execution after NeedsIO"
        );
    }

    #[test]
    fn resume_with_content_ready() {
        use crate::types::IOResponse;

        let mut rt = WasmiRuntime::new();
        let cid = [0xAB; 32];
        let content = b"hello world";

        // Execute — should get NeedsIO
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        // Resume with content
        let result = rt.resume_with_io(
            IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                // Output: [result_code: i32 LE] [data]
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, content.len() as i32);
                assert_eq!(&output[4..], content);
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
    }

    #[test]
    fn resume_with_content_not_found() {
        use crate::types::IOResponse;

        let mut rt = WasmiRuntime::new();
        let cid = [0xCD; 32];

        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        let result = rt.resume_with_io(
            IOResponse::ContentNotFound,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, -1, "content not found should return -1");
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
    }

    #[test]
    fn resume_with_buffer_too_small() {
        use crate::types::IOResponse;

        // WAT with tiny buffer capacity (2 bytes)
        let small_buf_wat = r#"
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
                    (i32.const 2)))
                (i32.store (local.get $out_ptr) (local.get $result))
                (i32.const 4)))
        "#;

        let mut rt = WasmiRuntime::new();
        let cid = [0xEF; 32];

        let result = rt.execute(
            small_buf_wat.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        // Content is 10 bytes, but buffer is only 2
        let result = rt.resume_with_io(
            IOResponse::ContentReady {
                data: vec![0u8; 10],
            },
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, -2, "buffer too small should return -2");
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
    }

    #[test]
    fn fetch_content_then_complete() {
        use crate::types::{IORequest, IOResponse};

        let mut rt = WasmiRuntime::new();
        let cid = [0x42; 32];
        let content = vec![10, 20, 30, 40, 50];

        // Full cycle: execute → NeedsIO → resume_with_io → Complete
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(
            matches!(result, ComputeResult::NeedsIO { request: IORequest::FetchContent { cid: c } } if c == cid)
        );
        assert!(rt.has_pending());

        let result = rt.resume_with_io(
            IOResponse::ContentReady {
                data: content.clone(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            ComputeResult::Complete { output } => {
                let result_code = i32::from_le_bytes(output[0..4].try_into().unwrap());
                assert_eq!(result_code, 5);
                assert_eq!(&output[4..], &content);
            }
            other => panic!("expected Complete, got: {other:?}"),
        }
        assert!(!rt.has_pending());
    }

    #[test]
    fn resume_without_pending_preserves_session_for_snapshot() {
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

        // resume() on a completed execution should fail but NOT destroy the session.
        let resume_result = rt.resume(InstructionBudget { fuel: 100_000 });
        assert!(matches!(
            resume_result,
            ComputeResult::Failed {
                error: ComputeError::NoPendingExecution
            }
        ));

        // snapshot() should still work because the session was preserved.
        let checkpoint = rt
            .snapshot()
            .expect("session should be preserved after resume");
        assert!(checkpoint.fuel_consumed > 0);
        assert!(!checkpoint.memory.is_empty());
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_model_load_triggers_needs_io() {
        let module_wat = r#"
            (module
              (import "harmony" "model_load" (func $load (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $result i32)
                (local.set $result
                  (call $load
                    (local.get $input_ptr)
                    (i32.add (local.get $input_ptr) (i32.const 32))))
                (i32.store
                  (i32.add (local.get $input_ptr) (local.get $input_len))
                  (local.get $result))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let input = [0u8; 64];
        let result = runtime.execute(
            module_wat.as_bytes(),
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::NeedsIO { request } => match request {
                crate::types::IORequest::LoadModel {
                    gguf_cid,
                    tokenizer_cid,
                } => {
                    assert_eq!(gguf_cid, [0u8; 32]);
                    assert_eq!(tokenizer_cid, [0u8; 32]);
                }
                _ => panic!("expected LoadModel, got {request:?}"),
            },
            other => panic!("expected NeedsIO, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_model_load_invalid_gguf_returns_neg3() {
        let module_wat = r#"
            (module
              (import "harmony" "model_load" (func $load (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (local.set $result
                  (call $load (local.get $input_ptr) (i32.add (local.get $input_ptr) (i32.const 32))))
                (i32.store (local.get $out_ptr) (local.get $result))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let input = [0u8; 64];
        let result = runtime.execute(
            module_wat.as_bytes(),
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(
            result,
            crate::types::ComputeResult::NeedsIO { .. }
        ));
        let result = runtime.resume_with_io(
            crate::types::IOResponse::ModelReady {
                gguf_data: b"not a valid gguf".to_vec(),
                tokenizer_data: b"{}".to_vec(),
            },
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -3, "expected -3 for invalid GGUF, got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_model_load_gguf_not_found() {
        let module_wat = r#"
            (module
              (import "harmony" "model_load" (func $load (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (i32.store (local.get $out_ptr)
                  (call $load (local.get $input_ptr) (i32.add (local.get $input_ptr) (i32.const 32))))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let input = [0u8; 64];
        let _ = runtime.execute(
            module_wat.as_bytes(),
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        let result = runtime.resume_with_io(
            crate::types::IOResponse::ModelGgufNotFound,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -1);
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_forward_before_model_load() {
        let module_wat = r#"
            (module
              (import "harmony" "forward" (func $fwd (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (i32.store (local.get $out_ptr)
                  (call $fwd (local.get $input_ptr) (i32.const 4)))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let input = 42u32.to_le_bytes();
        let result = runtime.execute(
            module_wat.as_bytes(),
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -1, "expected -1 (no model), got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_tokenize_before_model_load() {
        let module_wat = r#"
            (module
              (import "harmony" "tokenize" (func $tok (param i32 i32 i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (i32.store (local.get $out_ptr)
                  (call $tok
                    (local.get $input_ptr) (local.get $input_len)
                    (i32.add (local.get $out_ptr) (i32.const 4)) (i32.const 1024)))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let input = b"hello world";
        let result = runtime.execute(
            module_wat.as_bytes(),
            input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -1, "expected -1 (no tokenizer), got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_model_reset_empty_engine() {
        let module_wat = r#"
            (module
              (import "harmony" "model_reset" (func $reset (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (i32.store (local.get $out_ptr) (call $reset))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let result = runtime.execute(
            module_wat.as_bytes(),
            &[],
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, 0);
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }

    #[test]
    #[cfg(feature = "inference")]
    fn inference_sample_before_forward() {
        let module_wat = r#"
            (module
              (import "harmony" "sample" (func $samp (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $out_ptr i32)
                (local.set $out_ptr (i32.add (local.get $input_ptr) (local.get $input_len)))
                (f32.store (local.get $input_ptr) (f32.const 0.0))
                (f32.store (i32.add (local.get $input_ptr) (i32.const 4)) (f32.const 1.0))
                (i32.store (i32.add (local.get $input_ptr) (i32.const 8)) (i32.const 0))
                (f32.store (i32.add (local.get $input_ptr) (i32.const 12)) (f32.const 1.0))
                (i32.store (i32.add (local.get $input_ptr) (i32.const 16)) (i32.const 64))
                (i32.store (local.get $out_ptr)
                  (call $samp (local.get $input_ptr) (i32.const 20)))
                (i32.const 4)))
        "#;
        let mut runtime = super::WasmiRuntime::new();
        let result = runtime.execute(
            module_wat.as_bytes(),
            &[0u8; 20],
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -1, "expected -1 (no model loaded), got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }
}

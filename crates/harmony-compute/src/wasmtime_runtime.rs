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

/// Cached model data for wasmtime replay: (gguf_cid, tokenizer_cid, gguf_bytes, tokenizer_bytes).
/// Keyed by CID pair so a second `model_load` with different CIDs traps correctly.
#[cfg(feature = "inference")]
type ModelCache = ([u8; 32], [u8; 32], Vec<u8>, Vec<u8>);

/// Host state stored in the wasmtime `Store`.
///
/// When a WASM module calls `harmony.fetch_content`, the host function
/// stores the IO request and write target here before trapping.
struct HostState {
    /// Cached IO responses keyed by CID (for deterministic replay).
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    /// CIDs that were not found during prior IO rounds.
    not_found_cids: HashSet<[u8; 32]>,
    /// IO request captured from the host function call.
    io_request: Option<crate::types::IORequest>,
    /// Where to write the IO response data: (out_ptr, out_cap).
    io_write_target: Option<(u32, u32)>,
    /// Loaded inference engine (only available with `inference` feature).
    #[cfg(feature = "inference")]
    inference_engine: Option<harmony_inference::QwenEngine>,
    /// Logits from the most recent `forward()` call, read by `sample()`.
    /// Persists until the next `forward()` or `model_reset()` — multiple
    /// `sample()` calls from the same logits are allowed (e.g. beam search).
    #[cfg(feature = "inference")]
    last_logits: Option<Vec<f32>>,
    /// Cached model data for replay: (gguf_cid, tokenizer_cid, gguf_bytes, tokenizer_bytes).
    /// Keyed by CID pair so a second `model_load` with different CIDs traps correctly.
    #[cfg(feature = "inference")]
    model_cache: Option<ModelCache>,
    /// Cached model-not-found: (gguf_cid, tokenizer_cid, error_code).
    /// Keyed by CID pair so a fallback model_load with different CIDs traps correctly.
    #[cfg(feature = "inference")]
    model_not_found: Option<([u8; 32], [u8; 32], i32)>,
}

impl HostState {
    fn new() -> Self {
        Self {
            io_cache: HashMap::new(),
            not_found_cids: HashSet::new(),
            io_request: None,
            io_write_target: None,
            #[cfg(feature = "inference")]
            inference_engine: None,
            #[cfg(feature = "inference")]
            last_logits: None,
            #[cfg(feature = "inference")]
            model_cache: None,
            #[cfg(feature = "inference")]
            model_not_found: None,
        }
    }

    #[cfg_attr(feature = "inference", allow(dead_code))]
    fn with_cache(io_cache: HashMap<[u8; 32], Vec<u8>>, not_found_cids: HashSet<[u8; 32]>) -> Self {
        Self {
            io_cache,
            not_found_cids,
            io_request: None,
            io_write_target: None,
            #[cfg(feature = "inference")]
            inference_engine: None,
            #[cfg(feature = "inference")]
            last_logits: None,
            #[cfg(feature = "inference")]
            model_cache: None,
            #[cfg(feature = "inference")]
            model_not_found: None,
        }
    }

    #[cfg(feature = "inference")]
    fn with_inference_cache(
        io_cache: HashMap<[u8; 32], Vec<u8>>,
        not_found_cids: HashSet<[u8; 32]>,
        model_cache: Option<ModelCache>,
        model_not_found: Option<([u8; 32], [u8; 32], i32)>,
    ) -> Self {
        Self {
            io_cache,
            not_found_cids,
            io_request: None,
            io_write_target: None,
            inference_engine: None,
            last_logits: None,
            model_cache,
            model_not_found,
        }
    }
}

/// Saved session state for snapshot, take/restore, and future NeedsIO replay.
///
/// Unlike WasmiRuntime, wasmtime does not support resumable calls, so we
/// store enough information to re-execute the module from scratch with
/// accumulated IO cache state.
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
    /// Cumulative fuel consumed across all execution rounds (including replay
    /// overhead). Each `resume_with_io()` re-executes the module from scratch,
    /// so replay rounds re-consume fuel for the cached-IO portion. This
    /// intentionally tracks total CPU cost, not deduplicated logical work.
    total_fuel_consumed: u64,
    /// CID of a pending IO request (if execution suspended for IO).
    pending_cid: Option<[u8; 32]>,
    /// Memory snapshot at the end of the last execution.
    memory_snapshot: Vec<u8>,
    /// Cached model data for replay: (gguf_cid, tokenizer_cid, gguf_bytes, tokenizer_bytes).
    #[cfg(feature = "inference")]
    model_cache: Option<ModelCache>,
    /// Cached model-not-found: (gguf_cid, tokenizer_cid, error_code).
    /// Keyed by CID pair so a fallback model_load with different CIDs traps correctly.
    #[cfg(feature = "inference")]
    model_not_found: Option<([u8; 32], [u8; 32], i32)>,
    /// CIDs of a pending model load request, if any. Replaces a simple bool
    /// so we can key the model cache on the exact CID pair.
    #[cfg(feature = "inference")]
    pending_model_cids: Option<([u8; 32], [u8; 32])>,
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
    /// This is called by `execute()` and by `resume_with_io()` which replays
    /// execution with cached IO responses.
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

        // Set up Linker with host function imports.
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

                    // Check not_found_cids first — known-missing CIDs return -1 immediately.
                    if caller.data().not_found_cids.contains(&cid) {
                        return Ok(-1);
                    }

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
                    host.io_request = Some(crate::types::IORequest::FetchContent { cid });
                    host.io_write_target = Some((out_ptr as u32, out_cap as u32));

                    Err(wasmtime::Error::msg("harmony_io_trap"))
                },
            )
            .expect("failed to register harmony.fetch_content");

        // Register inference host functions (model_load, tokenize, detokenize,
        // forward, sample, model_reset) with replay/cache support.
        #[cfg(feature = "inference")]
        {
            use harmony_inference::InferenceEngine as _;

            linker
                .func_wrap(
                    "harmony",
                    "model_load",
                    |mut caller: wasmtime::Caller<'_, HostState>,
                     gguf_cid_ptr: i32,
                     tokenizer_cid_ptr: i32|
                     -> Result<i32, wasmtime::Error> {
                        // Read CIDs first — needed for both cache check and trap path.
                        let memory = caller
                            .get_export("memory")
                            .and_then(|e| e.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                        let mut gguf_cid = [0u8; 32];
                        memory
                            .read(&caller, gguf_cid_ptr as usize, &mut gguf_cid)
                            .map_err(|e| wasmtime::Error::msg(format!("read GGUF CID: {e}")))?;

                        let mut tokenizer_cid = [0u8; 32];
                        memory
                            .read(&caller, tokenizer_cid_ptr as usize, &mut tokenizer_cid)
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("read tokenizer CID: {e}"))
                            })?;

                        // Check cached model_not_found first (replay path) — only
                        // if CIDs match, so a fallback model_load traps correctly.
                        if let Some((nf_gguf, nf_tok, code)) = &caller.data().model_not_found {
                            if *nf_gguf == gguf_cid && *nf_tok == tokenizer_cid {
                                return Ok(*code);
                            }
                        }

                        // Check cached model data (replay path) — only use if CIDs match.
                        if let Some((cached_gguf_cid, cached_tok_cid, _, _)) =
                            &caller.data().model_cache
                        {
                            if *cached_gguf_cid == gguf_cid && *cached_tok_cid == tokenizer_cid {
                                use harmony_inference::InferenceEngine;
                                let (_, _, gguf_data, tokenizer_data) =
                                    caller.data_mut().model_cache.take().unwrap();
                                let mut engine =
                                    harmony_inference::QwenEngine::new(candle_core::Device::Cpu);
                                if engine.load_gguf(&gguf_data).is_err() {
                                    return Ok(-3);
                                }
                                if engine.load_tokenizer(&tokenizer_data).is_err() {
                                    return Ok(-4);
                                }
                                caller.data_mut().inference_engine = Some(engine);
                                return Ok(0);
                            }
                        }

                        // Cache miss or CID mismatch — trap for IO.
                        let host = caller.data_mut();
                        host.io_request = Some(crate::types::IORequest::LoadModel {
                            gguf_cid,
                            tokenizer_cid,
                        });

                        Err(wasmtime::Error::msg("harmony_io_trap"))
                    },
                )
                .expect("failed to register harmony.model_load");

            linker
                .func_wrap(
                    "harmony",
                    "tokenize",
                    |mut caller: wasmtime::Caller<'_, HostState>,
                     text_ptr: i32,
                     text_len: i32,
                     out_ptr: i32,
                     out_cap: i32|
                     -> Result<i32, wasmtime::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(|e| e.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                        if text_len < 0 {
                            return Ok(-3);
                        }

                        let engine = match &caller.data().inference_engine {
                            Some(e) => e,
                            None => return Ok(-1),
                        };

                        let mut text_buf = vec![0u8; text_len as usize];
                        memory
                            .read(&caller, text_ptr as usize, &mut text_buf)
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("failed to read text: {e}"))
                            })?;

                        let text = std::str::from_utf8(&text_buf)
                            .map_err(|e| wasmtime::Error::msg(format!("invalid UTF-8: {e}")))?;

                        let tokens = match engine.tokenize(text) {
                            Ok(t) => t,
                            Err(e) => {
                                return Err(wasmtime::Error::msg(format!("tokenize failed: {e}")))
                            }
                        };

                        let token_bytes: Vec<u8> =
                            tokens.iter().flat_map(|t| t.to_le_bytes()).collect();

                        if out_cap < 0 || token_bytes.len() > out_cap as usize {
                            return Ok(-2);
                        }

                        memory
                            .write(&mut caller, out_ptr as usize, &token_bytes)
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("failed to write tokens: {e}"))
                            })?;

                        Ok(token_bytes.len() as i32)
                    },
                )
                .expect("failed to register harmony.tokenize");

            linker
                .func_wrap(
                    "harmony",
                    "detokenize",
                    |mut caller: wasmtime::Caller<'_, HostState>,
                     tokens_ptr: i32,
                     tokens_len: i32,
                     out_ptr: i32,
                     out_cap: i32|
                     -> Result<i32, wasmtime::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(|e| e.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                        if tokens_len < 0 || tokens_len % 4 != 0 {
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
                                wasmtime::Error::msg(format!("failed to read tokens: {e}"))
                            })?;

                        let tokens: Vec<u32> = token_bytes
                            .chunks_exact(4)
                            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                            .collect();

                        let text = match engine.detokenize(&tokens) {
                            Ok(t) => t,
                            Err(e) => {
                                return Err(wasmtime::Error::msg(format!("detokenize failed: {e}")))
                            }
                        };

                        if out_cap < 0 || text.len() > out_cap as usize {
                            return Ok(-2);
                        }

                        memory
                            .write(&mut caller, out_ptr as usize, text.as_bytes())
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("failed to write text: {e}"))
                            })?;

                        Ok(text.len() as i32)
                    },
                )
                .expect("failed to register harmony.detokenize");

            linker
                .func_wrap(
                    "harmony",
                    "forward",
                    |mut caller: wasmtime::Caller<'_, HostState>,
                     tokens_ptr: i32,
                     tokens_len: i32|
                     -> Result<i32, wasmtime::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(|e| e.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                        if tokens_len <= 0 || tokens_len % 4 != 0 {
                            return Ok(-3);
                        }

                        let mut token_bytes = vec![0u8; tokens_len as usize];
                        memory
                            .read(&caller, tokens_ptr as usize, &mut token_bytes)
                            .map_err(|e| {
                                wasmtime::Error::msg(format!("failed to read tokens: {e}"))
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
                    |mut caller: wasmtime::Caller<'_, HostState>,
                     params_ptr: i32,
                     params_len: i32|
                     -> Result<i32, wasmtime::Error> {
                        let memory = caller
                            .get_export("memory")
                            .and_then(|e| e.into_memory())
                            .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

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
                                wasmtime::Error::msg(format!("failed to read params: {e}"))
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
                            None => return Ok(-3), // no logits from prior forward()
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
                    |mut caller: wasmtime::Caller<'_, HostState>| -> Result<i32, wasmtime::Error> {
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
                let remaining_fuel = store.get_fuel().unwrap_or(0);
                let fuel_consumed = budget.fuel.saturating_sub(remaining_fuel);
                let mem_snapshot = memory.data(&store).to_vec();
                let host_data = store.into_data();

                if let Some(io_request) = host_data.io_request.clone() {
                    // Host function trapped with an IO request — NeedsIO.
                    (
                        ComputeResult::NeedsIO {
                            request: io_request,
                        },
                        Some((host_data, fuel_consumed, mem_snapshot)),
                    )
                } else {
                    // Genuine error (fuel exhaustion, trap, etc.)
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
            let pending_cid = match &result {
                ComputeResult::NeedsIO { request } => match request {
                    crate::types::IORequest::FetchContent { cid } => Some(*cid),
                    crate::types::IORequest::LoadModel { .. } => None,
                },
                _ => None,
            };
            #[cfg(feature = "inference")]
            let pending_model_cids = match &result {
                ComputeResult::NeedsIO {
                    request:
                        crate::types::IORequest::LoadModel {
                            gguf_cid,
                            tokenizer_cid,
                        },
                } => Some((*gguf_cid, *tokenizer_cid)),
                _ => None,
            };
            self.session = Some(WasmtimeSession {
                module_bytes: module_bytes.to_vec(),
                module_hash,
                input: input.to_vec(),
                io_cache: host_data.io_cache,
                not_found_cids: host_data.not_found_cids,
                total_fuel_consumed: fuel_consumed,
                pending_cid,
                memory_snapshot: mem_snapshot,
                #[cfg(feature = "inference")]
                model_cache: None,
                #[cfg(feature = "inference")]
                model_not_found: None,
                #[cfg(feature = "inference")]
                pending_model_cids,
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

        // Check if this is a model load response (inference feature) or
        // a content fetch response.
        #[cfg(feature = "inference")]
        let is_model_load = session.pending_model_cids.is_some();
        #[cfg(not(feature = "inference"))]
        let is_model_load = false;

        if is_model_load {
            #[cfg(feature = "inference")]
            {
                let (gguf_cid, tokenizer_cid) = session.pending_model_cids.take().unwrap();
                match response {
                    crate::types::IOResponse::ModelReady {
                        gguf_data,
                        tokenizer_data,
                    } => {
                        session.model_cache =
                            Some((gguf_cid, tokenizer_cid, gguf_data, tokenizer_data));
                    }
                    crate::types::IOResponse::ModelGgufNotFound => {
                        session.model_not_found = Some((gguf_cid, tokenizer_cid, -1));
                    }
                    crate::types::IOResponse::ModelTokenizerNotFound => {
                        session.model_not_found = Some((gguf_cid, tokenizer_cid, -2));
                    }
                    _ => {
                        self.session = Some(session);
                        return ComputeResult::Failed {
                            error: ComputeError::Trap {
                                reason: "unexpected IOResponse for LoadModel".into(),
                            },
                        };
                    }
                }

                // Re-execute with updated model cache. Clone is needed because
                // session retains the cache for potential future FetchContent replays.
                let host_state = HostState::with_inference_cache(
                    session.io_cache.clone(),
                    session.not_found_cids.clone(),
                    session.model_cache.clone(),
                    session.model_not_found,
                );
                let (result, session_data) =
                    self.run_module(&session.module_bytes, &session.input, budget, host_state);

                if let Some((host_data, fuel_consumed, mem_snapshot)) = session_data {
                    let new_pending_cid = match &result {
                        ComputeResult::NeedsIO {
                            request: crate::types::IORequest::FetchContent { cid },
                        } => Some(*cid),
                        _ => None,
                    };
                    let new_pending_model_cids = match &result {
                        ComputeResult::NeedsIO {
                            request:
                                crate::types::IORequest::LoadModel {
                                    gguf_cid,
                                    tokenizer_cid,
                                },
                        } => Some((*gguf_cid, *tokenizer_cid)),
                        _ => None,
                    };
                    let total_fuel = session.total_fuel_consumed + fuel_consumed;
                    self.session = Some(WasmtimeSession {
                        module_bytes: session.module_bytes,
                        module_hash: session.module_hash,
                        input: session.input,
                        io_cache: host_data.io_cache,
                        not_found_cids: host_data.not_found_cids,
                        total_fuel_consumed: total_fuel,
                        pending_cid: new_pending_cid,
                        memory_snapshot: mem_snapshot,
                        model_cache: session.model_cache,
                        model_not_found: session.model_not_found,
                        pending_model_cids: new_pending_model_cids,
                    });
                }

                return result;
            }
            // Unreachable when inference is disabled, but needed for
            // compilation when cfg(not(feature = "inference")).
            #[cfg(not(feature = "inference"))]
            unreachable!()
        }

        // --- FetchContent path ---
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
                session.not_found_cids.insert(pending_cid);
            }
            _ => {
                self.session = Some(session);
                return ComputeResult::Failed {
                    error: ComputeError::Trap {
                        reason: "unexpected IOResponse for FetchContent".into(),
                    },
                };
            }
        }

        // Re-execute the module from scratch with the updated IO cache.
        // The host function will find the CID in cache and return data
        // immediately (deterministic replay). When inference is enabled,
        // preserve model cache from prior rounds so model_load replays
        // correctly if this module also uses inference.
        #[cfg(feature = "inference")]
        let host_state = HostState::with_inference_cache(
            session.io_cache,
            session.not_found_cids,
            session.model_cache.clone(),
            session.model_not_found,
        );
        #[cfg(not(feature = "inference"))]
        let host_state = HostState::with_cache(session.io_cache, session.not_found_cids);
        let (result, session_data) =
            self.run_module(&session.module_bytes, &session.input, budget, host_state);

        // Update session for potential further NeedsIO rounds or snapshot.
        if let Some((host_data, fuel_consumed, mem_snapshot)) = session_data {
            let new_pending_cid = match &result {
                ComputeResult::NeedsIO { request } => match request {
                    crate::types::IORequest::FetchContent { cid } => Some(*cid),
                    crate::types::IORequest::LoadModel { .. } => None,
                },
                _ => None,
            };
            #[cfg(feature = "inference")]
            let new_pending_model_cids = match &result {
                ComputeResult::NeedsIO {
                    request:
                        crate::types::IORequest::LoadModel {
                            gguf_cid,
                            tokenizer_cid,
                        },
                } => Some((*gguf_cid, *tokenizer_cid)),
                _ => None,
            };
            // Accumulate fuel across replay rounds. Each replay re-executes
            // the module from scratch, so `fuel_consumed` includes re-running
            // the cached-IO path. We sum rather than replace because the node
            // pays the real CPU cost for every round (see WasmtimeSession doc).
            let total_fuel = session.total_fuel_consumed + fuel_consumed;
            self.session = Some(WasmtimeSession {
                module_bytes: session.module_bytes,
                module_hash: session.module_hash,
                input: session.input,
                io_cache: host_data.io_cache,
                not_found_cids: host_data.not_found_cids,
                total_fuel_consumed: total_fuel,
                pending_cid: new_pending_cid,
                memory_snapshot: mem_snapshot,
                #[cfg(feature = "inference")]
                model_cache: session.model_cache,
                #[cfg(feature = "inference")]
                model_not_found: session.model_not_found,
                #[cfg(feature = "inference")]
                pending_model_cids: new_pending_model_cids,
            });
        }

        result
    }

    fn has_pending(&self) -> bool {
        self.session.as_ref().is_some_and(|s| {
            if s.pending_cid.is_some() {
                return true;
            }
            #[cfg(feature = "inference")]
            if s.pending_model_cids.is_some() {
                return true;
            }
            false
        })
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
            "has_pending() should be false after complete"
        );
    }

    /// WAT module that calls harmony.fetch_content — same ABI as wasmi tests.
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
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let cid = [0xAB; 32];

        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::NeedsIO { request } => match request {
                crate::types::IORequest::FetchContent { cid: req_cid } => {
                    assert_eq!(req_cid, cid, "requested CID should match input");
                }
                other => panic!("expected FetchContent, got: {other:?}"),
            },
            other => panic!("expected NeedsIO, got: {other:?}"),
        }

        // After NeedsIO, has_pending should be true.
        assert!(rt.has_pending(), "should have pending after NeedsIO");
    }

    #[test]
    fn resume_with_io_replays_with_content() {
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let cid = [0xCD; 32];
        let content = b"hello from storage";

        // First: execute triggers NeedsIO.
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        // Resume with content — should replay and complete.
        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
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
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let cid = [0xEF; 32];

        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentNotFound,
            InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            ComputeResult::Complete { output } => {
                let result_code = i32::from_le_bytes(output[..4].try_into().unwrap());
                assert_eq!(result_code, -1, "not-found result code");
            }
            other => panic!("expected Complete with -1, got: {other:?}"),
        }
    }

    #[test]
    fn fetch_content_replay_then_complete() {
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let cid = [0x42; 32];
        let content = b"deterministic replay works";

        // Execute -> NeedsIO.
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));

        // Resume -> Complete.
        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::Complete { .. }));

        // After completion, no pending execution.
        assert!(!rt.has_pending());
    }

    #[test]
    fn fuel_accounting_accumulates_across_replay_rounds() {
        // Verify that total_fuel_consumed includes replay overhead:
        // each resume_with_io re-executes from scratch, so the cached-IO
        // portion is re-consumed. Total = sum of all rounds.
        let mut rt = WasmtimeRuntime::new().expect("engine creation");
        let cid = [0x99; 32];
        let content = b"fuel test data";

        // Round 1: execute -> NeedsIO. Records fuel for partial run.
        let result = rt.execute(
            FETCH_WAT.as_bytes(),
            &cid,
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::NeedsIO { .. }));
        let fuel_after_execute = rt.snapshot().expect("snapshot after execute").fuel_consumed;
        assert!(fuel_after_execute > 0, "should have consumed some fuel");

        // Round 2: resume_with_io -> Complete. Re-executes from scratch.
        let result = rt.resume_with_io(
            crate::types::IOResponse::ContentReady {
                data: content.to_vec(),
            },
            InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, ComputeResult::Complete { .. }));
        let fuel_after_replay = rt.snapshot().expect("snapshot after replay").fuel_consumed;

        // Total must be strictly greater than either individual round,
        // confirming accumulation (not replacement).
        assert!(
            fuel_after_replay > fuel_after_execute,
            "total fuel ({fuel_after_replay}) should exceed first round ({fuel_after_execute}) \
             due to replay accumulation"
        );
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
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
        let mut runtime = super::WasmtimeRuntime::new().expect("failed to create runtime");
        let result = runtime.execute(
            module_wat.as_bytes(),
            &[0u8; 20],
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -1, "expected -1 (no engine), got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }
}

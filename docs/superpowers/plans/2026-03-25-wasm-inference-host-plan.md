# WASM Inference Host Functions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose inference operations (model_load, tokenize, detokenize, forward, sample, model_reset) as WASM host imports in harmony-compute so WASM guests can orchestrate LLM inference while matrix ops run natively.

**Architecture:** Six new host functions gated behind an `inference` feature flag on harmony-compute. `model_load` uses the existing I/O suspension mechanism (trap → NeedsIO → resume). All other functions are synchronous — the host holds the QwenEngine, logits, and token history in native memory. Both wasmi and wasmtime runtimes are updated.

**Tech Stack:** wasmi 1, wasmtime 38 (optional), harmony-inference (optional), candle-core (transitive)

**Spec:** `docs/superpowers/specs/2026-03-25-wasm-inference-host-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-compute/Cargo.toml` | Add `inference` feature flag + harmony-inference dep |
| `crates/harmony-compute/src/types.rs` | Add `LoadModel` to `IORequest`, add `ModelReady`/`ModelGgufNotFound`/`ModelTokenizerNotFound` to `IOResponse` |
| `crates/harmony-compute/src/wasmi_runtime.rs` | Add inference fields to `HostState`, register 6 host functions, update `resume_with_io` for `LoadModel` |
| `crates/harmony-compute/src/wasmtime_runtime.rs` | Same as wasmi but with replay/cache adaptation |

---

### Task 1: Types + Cargo.toml Changes

Add IORequest/IOResponse variants and the `inference` feature flag. No logic — just type definitions and build config.

**Files:**
- Modify: `crates/harmony-compute/Cargo.toml`
- Modify: `crates/harmony-compute/src/types.rs`

- [ ] **Step 1: Add feature flag and dependency to Cargo.toml**

In `crates/harmony-compute/Cargo.toml`, add to the `[features]` section:

```toml
inference = ["harmony-inference", "candle-core"]
```

Add to `[dependencies]`:

```toml
harmony-inference = { workspace = true, optional = true }
candle-core = { workspace = true, optional = true }
```

- [ ] **Step 2: Extend IORequest enum**

In `crates/harmony-compute/src/types.rs`, add the `LoadModel` variant to `IORequest`:

```rust
pub enum IORequest {
    FetchContent { cid: [u8; 32] },
    LoadModel { gguf_cid: [u8; 32], tokenizer_cid: [u8; 32] },
}
```

- [ ] **Step 3: Extend IOResponse enum**

Add three new variants to `IOResponse`:

```rust
pub enum IOResponse {
    ContentReady { data: Vec<u8> },
    ContentNotFound,
    ModelReady { gguf_data: Vec<u8>, tokenizer_data: Vec<u8> },
    ModelGgufNotFound,
    ModelTokenizerNotFound,
}
```

- [ ] **Step 4: Fix exhaustive match arms**

The new enum variants will break existing `match` statements in both runtimes. Add temporary catch-all arms to make it compile:

In `wasmi_runtime.rs`, find the `match &response` block in `resume_with_io` (the block that handles `ContentReady` and `ContentNotFound`) and add a catch-all:

```rust
_ => {
    self.session = Some(session);
    return ComputeResult::Failed {
        error: ComputeError::Trap {
            reason: "unexpected IOResponse for FetchContent".into(),
        },
    };
}
```

Do the same in `wasmtime_runtime.rs`'s `resume_with_io` — find the `match response` block that inserts into `io_cache` / `not_found_cids` and add:

```rust
_ => {
    self.session = Some(session);
    return ComputeResult::Failed {
        error: ComputeError::Trap {
            reason: "unexpected IOResponse for FetchContent".into(),
        },
    };
}
```

Also in `wasmtime_runtime.rs`'s `resume_with_io`, find the match that extracts `pending_cid` from `NeedsIO` results:

```rust
ComputeResult::NeedsIO { request } => match request {
    crate::types::IORequest::FetchContent { cid } => Some(*cid),
},
```

Add the new variant:

```rust
ComputeResult::NeedsIO { request } => match request {
    crate::types::IORequest::FetchContent { cid } => Some(*cid),
    crate::types::IORequest::LoadModel { .. } => None, // handled in Task 4
},
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-compute`
Expected: Compiles (without inference feature, no new deps pulled)

Run: `cargo check -p harmony-compute --features inference`
Expected: Compiles (with inference feature, pulls harmony-inference + candle-core)

Run: `cargo test -p harmony-compute`
Expected: All existing tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-compute/
git commit -m "feat(compute): add inference feature flag and IORequest/IOResponse variants for model loading"
```

---

### Task 2: WasmiRuntime — Host Functions + Resume Update

Add inference fields to `HostState`, register 6 host functions in the linker, and update `resume_with_io` to handle `LoadModel` responses.

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs`

**Key patterns to follow:**
- Existing `fetch_content` registration at ~line 249-280
- Existing `resume_with_io` at ~line 429-546
- Host functions read/write WASM memory via `memory.read()` / `memory.write()`
- I/O suspension: store request in `HostState`, return `Err(wasmi::Error::new("harmony_io_trap"))`

- [ ] **Step 1: Add inference fields to HostState**

Add `#[cfg(feature = "inference")]` fields to the `HostState` struct:

```rust
struct HostState {
    io_request: Option<crate::types::IORequest>,
    io_write_target: Option<(u32, u32)>,
    #[cfg(feature = "inference")]
    inference_engine: Option<harmony_inference::QwenEngine>,
    #[cfg(feature = "inference")]
    last_logits: Option<Vec<f32>>,
}
```

Update the `HostState` construction in `execute()` (where `HostState` is created for the store) to initialize the new fields:

```rust
let host_state = HostState {
    io_request: None,
    io_write_target: None,
    #[cfg(feature = "inference")]
    inference_engine: None,
    #[cfg(feature = "inference")]
    last_logits: None,
};
```

- [ ] **Step 2: Register model_load host function**

After the existing `fetch_content` registration block, add a `#[cfg(feature = "inference")]` block:

```rust
#[cfg(feature = "inference")]
{
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
                    .map_err(|e| wasmi::Error::new(format!("failed to read GGUF CID: {e}")))?;

                let mut tokenizer_cid = [0u8; 32];
                memory
                    .read(&caller, tokenizer_cid_ptr as usize, &mut tokenizer_cid)
                    .map_err(|e| {
                        wasmi::Error::new(format!("failed to read tokenizer CID: {e}"))
                    })?;

                let data = caller.data_mut();
                data.io_request =
                    Some(crate::types::IORequest::LoadModel { gguf_cid, tokenizer_cid });
                // No io_write_target — model_load returns via host function return value

                Err(wasmi::Error::new("harmony_io_trap"))
            },
        )
        .expect("failed to register harmony.model_load");
}
```

- [ ] **Step 3: Register tokenize host function**

```rust
#[cfg(feature = "inference")]
{
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
                    .map_err(|e| wasmi::Error::new(format!("failed to write tokens: {e}")))?;

                Ok(token_bytes.len() as i32)
            },
        )
        .expect("failed to register harmony.tokenize");
}
```

- [ ] **Step 4: Register detokenize host function**

```rust
#[cfg(feature = "inference")]
{
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
                    .map_err(|e| wasmi::Error::new(format!("failed to read tokens: {e}")))?;

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
}
```

- [ ] **Step 5: Register forward host function**

```rust
#[cfg(feature = "inference")]
{
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
                    .map_err(|e| wasmi::Error::new(format!("failed to read tokens: {e}")))?;

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
}
```

- [ ] **Step 6: Register sample host function**

```rust
#[cfg(feature = "inference")]
{
    linker
        .func_wrap(
            "harmony",
            "sample",
            |mut caller: wasmi::Caller<'_, HostState>,
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
                    .map_err(|e| wasmi::Error::new(format!("failed to read params: {e}")))?;

                let params = harmony_inference::SamplingParams {
                    temperature: f32::from_le_bytes([
                        param_bytes[0], param_bytes[1], param_bytes[2], param_bytes[3],
                    ]),
                    top_p: f32::from_le_bytes([
                        param_bytes[4], param_bytes[5], param_bytes[6], param_bytes[7],
                    ]),
                    top_k: u32::from_le_bytes([
                        param_bytes[8], param_bytes[9], param_bytes[10], param_bytes[11],
                    ]),
                    repeat_penalty: f32::from_le_bytes([
                        param_bytes[12], param_bytes[13], param_bytes[14], param_bytes[15],
                    ]),
                    repeat_last_n: u32::from_le_bytes([
                        param_bytes[16], param_bytes[17], param_bytes[18], param_bytes[19],
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
}
```

- [ ] **Step 7: Register model_reset host function**

```rust
#[cfg(feature = "inference")]
{
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
```

- [ ] **Step 8: Update resume_with_io for LoadModel**

The existing `resume_with_io` assumes all I/O is `FetchContent` and reads `io_write_target`. Restructure to branch on the IORequest type.

Find the block in `resume_with_io` that reads `io_write_target` (starts with `let (out_ptr, out_cap) = match session.store.data().io_write_target`). Replace the section from that point through the `match &response` block (up to where `host.io_request = None`) with:

```rust
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
                    if let Err(_) = engine.load_gguf(&gguf_data) {
                        -3
                    } else if let Err(_) = engine.load_tokenizer(&tokenizer_data) {
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
                    let memory =
                        match session.instance.get_memory(&session.store, "memory") {
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
```

- [ ] **Step 9: Verify compilation and existing tests**

Run: `cargo check -p harmony-compute --features inference`
Expected: Compiles

Run: `cargo test -p harmony-compute`
Expected: All existing tests pass (no inference feature in default tests)

Run: `cargo test -p harmony-compute --features inference`
Expected: All existing tests pass with inference feature enabled

Run: `cargo clippy -p harmony-compute --features inference -- -D warnings`
Expected: No warnings

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-compute/src/wasmi_runtime.rs
git commit -m "feat(compute): add inference host functions to WasmiRuntime (model_load, tokenize, forward, sample, detokenize, reset)"
```

---

### Task 3: WasmiRuntime — Unit Tests

Write WAT-based unit tests for the inference host functions, following the existing `fetch_content` test pattern.

**Files:**
- Modify: `crates/harmony-compute/src/wasmi_runtime.rs` (add tests to existing `#[cfg(test)] mod tests`)

**Pattern:** Each test creates a small WAT module that imports and calls a host function, then checks the `ComputeResult`. Tests are `#[cfg(feature = "inference")]` so they only run when the feature is enabled.

- [ ] **Step 1: Write test — model_load triggers NeedsIO**

Add to the `mod tests` block in `wasmi_runtime.rs`:

```rust
    #[test]
    #[cfg(feature = "inference")]
    fn inference_model_load_triggers_needs_io() {
        let module_wat = r#"
            (module
              (import "harmony" "model_load" (func $load (param i32 i32) (result i32)))
              (memory (export "memory") 1)
              (func (export "compute") (param $input_ptr i32) (param $input_len i32) (result i32)
                (local $result i32)
                ;; CID bytes are at input[0..32] (gguf) and input[32..64] (tokenizer)
                (local.set $result
                  (call $load
                    (local.get $input_ptr)           ;; gguf_cid_ptr
                    (i32.add (local.get $input_ptr) (i32.const 32))))  ;; tokenizer_cid_ptr
                ;; Write result to output area
                (i32.store
                  (i32.add (local.get $input_ptr) (local.get $input_len))
                  (local.get $result))
                (i32.const 4)))
        "#;
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();

        // Input: two 32-byte CIDs (all zeros for test)
        let input = [0u8; 64];
        let result = runtime.execute(
            &module_bytes,
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::NeedsIO { request } => {
                match request {
                    crate::types::IORequest::LoadModel { gguf_cid, tokenizer_cid } => {
                        assert_eq!(gguf_cid, [0u8; 32]);
                        assert_eq!(tokenizer_cid, [0u8; 32]);
                    }
                    _ => panic!("expected LoadModel, got {request:?}"),
                }
            }
            other => panic!("expected NeedsIO, got {other:?}"),
        }
    }
```

- [ ] **Step 2: Write test — model_load resume with invalid GGUF returns -3**

```rust
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
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();
        let input = [0u8; 64];

        let result = runtime.execute(
            &module_bytes,
            &input,
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );
        assert!(matches!(result, crate::types::ComputeResult::NeedsIO { .. }));

        // Resume with garbage GGUF data
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
```

- [ ] **Step 3: Write test — model_load resume with GgufNotFound returns -1**

```rust
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
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();
        let input = [0u8; 64];

        let _ = runtime.execute(
            &module_bytes, &input,
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
```

- [ ] **Step 4: Write test — forward before model_load returns -1**

```rust
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
                ;; Call forward with a dummy token (4 bytes)
                (i32.store (local.get $out_ptr)
                  (call $fwd (local.get $input_ptr) (i32.const 4)))
                (i32.const 4)))
        "#;
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();
        // Input: a single token ID (doesn't matter, engine not loaded)
        let input = 42u32.to_le_bytes();

        let result = runtime.execute(
            &module_bytes, &input,
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
```

- [ ] **Step 5: Write test — tokenize before model_load returns -1**

```rust
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
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();
        let input = b"hello world";

        let result = runtime.execute(
            &module_bytes, input,
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
```

- [ ] **Step 6: Write test — model_reset on empty engine returns 0**

```rust
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
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();

        let result = runtime.execute(
            &module_bytes, &[],
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
```

- [ ] **Step 7: Write test — sample before forward returns -2**

```rust
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
                ;; Write greedy SamplingParams (20 bytes) at input_ptr
                ;; temperature=0.0, top_p=1.0, top_k=0, repeat_penalty=1.0, repeat_last_n=64
                (f32.store (local.get $input_ptr) (f32.const 0.0))
                (f32.store (i32.add (local.get $input_ptr) (i32.const 4)) (f32.const 1.0))
                (i32.store (i32.add (local.get $input_ptr) (i32.const 8)) (i32.const 0))
                (f32.store (i32.add (local.get $input_ptr) (i32.const 12)) (f32.const 1.0))
                (i32.store (i32.add (local.get $input_ptr) (i32.const 16)) (i32.const 64))
                (i32.store (local.get $out_ptr)
                  (call $samp (local.get $input_ptr) (i32.const 20)))
                (i32.const 4)))
        "#;
        let module_bytes = wat::parse_str(module_wat).expect("invalid WAT");
        let mut runtime = super::WasmiRuntime::new();

        let result = runtime.execute(
            &module_bytes, &[0u8; 20],
            crate::types::InstructionBudget { fuel: 1_000_000 },
        );

        match result {
            crate::types::ComputeResult::Complete { output } => {
                let code = i32::from_le_bytes([output[0], output[1], output[2], output[3]]);
                assert_eq!(code, -2, "expected -2 (no logits), got {code}");
            }
            other => panic!("expected Complete, got {other:?}"),
        }
    }
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p harmony-compute --features inference`
Expected: All tests pass (existing + 6 new inference tests)

Run: `cargo fmt -p harmony-compute -- --check`
Expected: Clean

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-compute/src/wasmi_runtime.rs
git commit -m "test(compute): add wasmi inference host function unit tests"
```

---

### Task 4: WasmtimeRuntime — Host Functions + Replay Adaptation

Add inference fields to wasmtime's `HostState` and `WasmtimeSession`, register 6 host functions with replay/cache support, and update `resume_with_io` for `LoadModel`.

**Files:**
- Modify: `crates/harmony-compute/src/wasmtime_runtime.rs`

**Key difference from wasmi:** Wasmtime doesn't support resumable calls. It replays the entire module from scratch with cached I/O responses. For `model_load`, the response (gguf_data + tokenizer_data) is cached in `WasmtimeSession`. On replay, the `model_load` host function checks the cache and constructs the engine directly (no trap).

**Note:** Engine reconstruction on replay is a known cost for this first pass. The GGUF is re-parsed on each replay. This is acceptable because after `model_load` succeeds, subsequent inference operations are synchronous (no more I/O traps, no more replays).

- [ ] **Step 1: Add inference fields to wasmtime HostState**

```rust
struct HostState {
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    not_found_cids: HashSet<[u8; 32]>,
    io_request: Option<crate::types::IORequest>,
    io_write_target: Option<(u32, u32)>,
    #[cfg(feature = "inference")]
    inference_engine: Option<harmony_inference::QwenEngine>,
    #[cfg(feature = "inference")]
    last_logits: Option<Vec<f32>>,
    #[cfg(feature = "inference")]
    model_cache: Option<(Vec<u8>, Vec<u8>)>,
    #[cfg(feature = "inference")]
    model_not_found: Option<i32>,
}
```

Update `HostState::new()` and `HostState::with_cache()` to initialize the new fields. Add a new constructor:

```rust
    #[cfg(feature = "inference")]
    fn with_inference_cache(
        io_cache: HashMap<[u8; 32], Vec<u8>>,
        not_found_cids: HashSet<[u8; 32]>,
        model_cache: Option<(Vec<u8>, Vec<u8>)>,
        model_not_found: Option<i32>,
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
```

- [ ] **Step 2: Add inference fields to WasmtimeSession**

```rust
struct WasmtimeSession {
    module_bytes: Vec<u8>,
    module_hash: [u8; 32],
    input: Vec<u8>,
    io_cache: HashMap<[u8; 32], Vec<u8>>,
    not_found_cids: HashSet<[u8; 32]>,
    total_fuel_consumed: u64,
    pending_cid: Option<[u8; 32]>,
    memory_snapshot: Vec<u8>,
    #[cfg(feature = "inference")]
    model_cache: Option<(Vec<u8>, Vec<u8>)>,
    #[cfg(feature = "inference")]
    model_not_found: Option<i32>,
    #[cfg(feature = "inference")]
    pending_model_load: bool,
}
```

Update all `WasmtimeSession` construction sites to include the new fields (initialized to `None`/`false`).

- [ ] **Step 3: Register model_load host function (with replay)**

After the existing `fetch_content` registration, add a `#[cfg(feature = "inference")]` block. The `model_load` function checks the cache before trapping:

```rust
#[cfg(feature = "inference")]
{
    linker
        .func_wrap(
            "harmony",
            "model_load",
            |mut caller: wasmtime::Caller<'_, HostState>,
             gguf_cid_ptr: i32,
             tokenizer_cid_ptr: i32|
             -> Result<i32, wasmtime::Error> {
                // Check cached model_not_found first (replay path)
                if let Some(code) = caller.data().model_not_found {
                    return Ok(code);
                }

                // Check cached model data (replay path)
                if caller.data().model_cache.is_some() {
                    use harmony_inference::InferenceEngine;
                    let (gguf_data, tokenizer_data) =
                        caller.data().model_cache.clone().unwrap();
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

                // Cache miss — read CIDs and trap
                let memory = caller
                    .get_export("memory")
                    .and_then(|e| e.into_memory())
                    .ok_or_else(|| wasmtime::Error::msg("missing exported memory"))?;

                let mut gguf_cid = [0u8; 32];
                memory.read(&caller, gguf_cid_ptr as usize, &mut gguf_cid)
                    .map_err(|e| wasmtime::Error::msg(format!("read GGUF CID: {e}")))?;

                let mut tokenizer_cid = [0u8; 32];
                memory.read(&caller, tokenizer_cid_ptr as usize, &mut tokenizer_cid)
                    .map_err(|e| wasmtime::Error::msg(format!("read tokenizer CID: {e}")))?;

                let host = caller.data_mut();
                host.io_request = Some(crate::types::IORequest::LoadModel {
                    gguf_cid,
                    tokenizer_cid,
                });

                Err(wasmtime::Error::msg("harmony_io_trap"))
            },
        )
        .expect("failed to register harmony.model_load");
}
```

- [ ] **Step 4: Register remaining 5 host functions**

Register `tokenize`, `detokenize`, `forward`, `sample`, `model_reset` following the same patterns as the wasmi versions (Task 2, Steps 3-7). Copy each closure body verbatim from the wasmi implementations, making only these type substitutions:

- `wasmi::Caller<'_, HostState>` → `wasmtime::Caller<'_, HostState>`
- `wasmi::Error::new(msg)` → `wasmtime::Error::msg(msg)`
- `wasmi::Extern::into_memory` → `.into_memory()`
- `Result<i32, wasmi::Error>` → `Result<i32, wasmtime::Error>`

The closure logic (memory reads, engine calls, return codes) is identical. These 5 functions are synchronous — no caching/replay needed. The `sample` closure must clone `last_logits` before borrowing the engine (same borrow-conflict fix as the wasmi version).

- [ ] **Step 5: Update resume_with_io for LoadModel**

In `resume_with_io`, after extracting `pending_cid`, add handling for model load responses:

```rust
    // Check if this is a model load response
    #[cfg(feature = "inference")]
    let is_model_load = session.pending_model_load;
    #[cfg(not(feature = "inference"))]
    let is_model_load = false;

    if is_model_load {
        #[cfg(feature = "inference")]
        {
            session.pending_model_load = false;
            match response {
                crate::types::IOResponse::ModelReady { gguf_data, tokenizer_data } => {
                    session.model_cache = Some((gguf_data, tokenizer_data));
                }
                crate::types::IOResponse::ModelGgufNotFound => {
                    session.model_not_found = Some(-1);
                }
                crate::types::IOResponse::ModelTokenizerNotFound => {
                    session.model_not_found = Some(-2);
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

            // Re-execute with updated model cache
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
                    ComputeResult::NeedsIO { request } => match request {
                        crate::types::IORequest::FetchContent { cid } => Some(*cid),
                        _ => None,
                    },
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
                    pending_model_load: false,
                });
            }

            return result;
        }
    }
```

Also update the `NeedsIO` result handling at the end of `resume_with_io` to detect `LoadModel`:

```rust
let new_pending_cid = match &result {
    ComputeResult::NeedsIO { request } => match request {
        crate::types::IORequest::FetchContent { cid } => Some(*cid),
        crate::types::IORequest::LoadModel { .. } => None,
    },
    _ => None,
};

#[cfg(feature = "inference")]
let new_pending_model_load = matches!(
    &result,
    ComputeResult::NeedsIO { request }
    if matches!(request, crate::types::IORequest::LoadModel { .. })
);
```

And include `pending_model_load: new_pending_model_load` in the `WasmtimeSession` construction.

- [ ] **Step 6: Update execute() to set pending_model_load**

In the `execute()` method, where `WasmtimeSession` is constructed from a `NeedsIO` result, also detect `LoadModel`:

```rust
#[cfg(feature = "inference")]
let pending_model_load = matches!(
    &result,
    ComputeResult::NeedsIO { request }
    if matches!(request, crate::types::IORequest::LoadModel { .. })
);
```

And include it in the session construction.

- [ ] **Step 7: Verify compilation and tests**

Run: `cargo check -p harmony-compute --features inference,wasmtime`
Expected: Compiles

Run: `cargo test -p harmony-compute --features inference`
Expected: All tests pass

Run: `cargo clippy -p harmony-compute --features inference,wasmtime -- -D warnings`
Expected: No warnings

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-compute/src/wasmtime_runtime.rs
git commit -m "feat(compute): add inference host functions to WasmtimeRuntime with replay/cache support"
```

---

### Task 5: WasmtimeRuntime — Unit Tests

Write WAT-based unit tests for wasmtime inference host functions, mirroring the wasmi tests from Task 3.

**Files:**
- Modify: `crates/harmony-compute/src/wasmtime_runtime.rs` (add to existing `#[cfg(test)] mod tests`)

- [ ] **Step 1: Write wasmtime inference tests**

Add the same 6 test cases as Task 3, but using `super::WasmtimeRuntime::new()` instead of `super::WasmiRuntime::new()`. The WAT modules are identical — only the runtime differs.

Tests to add (all `#[cfg(feature = "inference")]`):
1. `inference_model_load_triggers_needs_io` — same WAT, WasmtimeRuntime
2. `inference_model_load_invalid_gguf_returns_neg3` — same WAT, WasmtimeRuntime
3. `inference_model_load_gguf_not_found` — same WAT, WasmtimeRuntime
4. `inference_forward_before_model_load` — same WAT, WasmtimeRuntime
5. `inference_tokenize_before_model_load` — same WAT, WasmtimeRuntime
6. `inference_model_reset_empty_engine` — same WAT, WasmtimeRuntime
7. `inference_sample_before_forward` — same WAT, WasmtimeRuntime

Follow the exact same test structure and assertions as Task 3, replacing `WasmiRuntime` with `WasmtimeRuntime`.

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-compute --features inference,wasmtime`
Expected: All tests pass (existing + 6 wasmi inference + 6 wasmtime inference)

Run: `cargo fmt -p harmony-compute -- --check`
Expected: Clean

Run: `cargo clippy -p harmony-compute --features inference,wasmtime -- -D warnings`
Expected: No warnings

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-compute/src/wasmtime_runtime.rs
git commit -m "test(compute): add wasmtime inference host function unit tests"
```

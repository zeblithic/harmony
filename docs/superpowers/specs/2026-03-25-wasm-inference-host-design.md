# WASM Host Functions for Model Inference

## Goal

Expose inference operations as WASM host imports in harmony-compute, so WASM guests can orchestrate LLM inference (prompt formatting, sampling parameters, tool calling, stopping criteria) while the actual matrix ops run natively on the host.

## Architecture

Six new host functions under the `"harmony"` namespace, gated behind an `inference` feature flag on `harmony-compute`. The WASM guest drives the autoregressive loop; the host holds the `QwenEngine`, model weights, and logits in native memory. Model loading uses the existing I/O suspension mechanism for CAS resolution. All other operations are synchronous.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-inference` | `QwenEngine`, `InferenceEngine` trait, `SamplingParams` (optional dep) |
| `candle-core` | `Device` type for engine construction (transitive via harmony-inference) |

### Feature Flag

```toml
[features]
inference = ["harmony-inference"]
```

When disabled, no inference host functions are registered and `HostState` has no inference fields. This keeps harmony-compute pure for consumers that don't need inference.

## Host Function API

All functions use the `"harmony"` import namespace. Return codes follow the existing `fetch_content` convention: `>= 0` is success, negative values are typed errors.

All pointer+length arguments are bounds-checked against WASM linear memory. Out-of-bounds access produces a trap (not a return code), consistent with `fetch_content` behavior.

Token IDs fit in `i32` (`>= 0`). Qwen3's vocabulary is ~152K, well within `i32::MAX` (~2.1B).

### `model_load`

```
harmony.model_load(gguf_cid_ptr: i32, tokenizer_cid_ptr: i32) -> i32
```

Load a GGUF model and tokenizer from CAS by their content IDs.

**Parameters:**
- `gguf_cid_ptr` — pointer to 32-byte GGUF content ID in WASM memory
- `tokenizer_cid_ptr` — pointer to 32-byte tokenizer.json content ID in WASM memory

**Behavior:** Reads both CIDs from WASM memory, stores a `LoadModel` I/O request in `HostState`, and traps to suspend execution. The caller resolves both CIDs from CAS and resumes with `ModelReady` (or `ModelLoadFailed`). On resume, the host constructs a `QwenEngine`, calls `load_gguf()` and `load_tokenizer()`, and writes the return code.

**Return codes:**
- `0` — success, model and tokenizer loaded
- `-1` — GGUF content not found (CID resolution failed)
- `-2` — tokenizer content not found (CID resolution failed)
- `-3` — invalid GGUF data (parse/load error)
- `-4` — invalid tokenizer JSON

**Replaces any previously loaded model.** Clears `last_logits`.

### `tokenize`

```
harmony.tokenize(text_ptr: i32, text_len: i32, out_ptr: i32, out_cap: i32) -> i32
```

Tokenize UTF-8 text into token IDs.

**Parameters:**
- `text_ptr` — pointer to UTF-8 text in WASM memory
- `text_len` — length of text in bytes
- `out_ptr` — pointer to output buffer in WASM memory
- `out_cap` — output buffer capacity in bytes

**Output format:** Token IDs as consecutive little-endian `u32` values.

**Return codes:**
- `>= 0` — number of bytes written (= num_tokens × 4)
- `-1` — no tokenizer loaded
- `-2` — output buffer too small

### `detokenize`

```
harmony.detokenize(tokens_ptr: i32, tokens_len: i32, out_ptr: i32, out_cap: i32) -> i32
```

Convert token IDs back to UTF-8 text.

**Parameters:**
- `tokens_ptr` — pointer to token IDs (little-endian `u32` array) in WASM memory
- `tokens_len` — number of bytes (must be divisible by 4)
- `out_ptr` — pointer to output buffer in WASM memory
- `out_cap` — output buffer capacity in bytes

**Return codes:**
- `>= 0` — number of UTF-8 bytes written
- `-1` — no tokenizer loaded
- `-2` — output buffer too small
- `-3` — `tokens_len` not divisible by 4

### `forward`

```
harmony.forward(tokens_ptr: i32, tokens_len: i32) -> i32
```

Run a forward pass on the given token IDs. Logits are stored host-side in `last_logits`.

**Parameters:**
- `tokens_ptr` — pointer to token IDs (little-endian `u32` array) in WASM memory
- `tokens_len` — number of bytes (must be divisible by 4)

**Behavior:** Reads token IDs from WASM memory, calls `engine.forward()`, stores the resulting logits in `HostState.last_logits`. The WASM guest never sees raw logits — it calls `sample` to consume them.

**Return codes:**
- `0` — success, logits stored host-side
- `-1` — no model loaded
- `-2` — forward pass failed
- `-3` — `tokens_len` not divisible by 4 or empty

### `sample`

```
harmony.sample(params_ptr: i32, params_len: i32) -> i32
```

Sample the next token from the most recent `forward` call's logits.

**Parameters:**
- `params_ptr` — pointer to sampling parameters in WASM memory
- `params_len` — must be 20 (5 fields × 4 bytes)

**Parameter format (20 bytes, all little-endian):**

| Offset | Type | Field |
|--------|------|-------|
| 0 | f32 | temperature (0.0 = greedy) |
| 4 | f32 | top_p (1.0 = disabled) |
| 8 | u32 | top_k (0 = disabled) |
| 12 | f32 | repeat_penalty (1.0 = no penalty) |
| 16 | u32 | repeat_last_n (0 = unbounded, widened to `usize` on host) |

**Return codes:**
- `>= 0` — sampled token ID
- `-1` — no model loaded
- `-2` — sampling failed (invalid params or internal error)
- `-3` — no logits available (call `forward` first)

### `model_reset`

```
harmony.model_reset() -> i32
```

Reset the KV cache, position counter, token history, and last logits. Starts a new conversation without reloading the model.

**Return codes:**
- `0` — success (also succeeds if no model is loaded — no-op)

## HostState Changes

```rust
struct HostState {
    io_request: Option<IORequest>,
    io_write_target: Option<(u32, u32)>,
    #[cfg(feature = "inference")]
    inference_engine: Option<harmony_inference::QwenEngine>,
    #[cfg(feature = "inference")]
    last_logits: Option<Vec<f32>>,
}
```

`inference_engine` is `None` until `model_load` succeeds. `last_logits` is populated by each `forward()` call and read by `sample()`. `model_reset()` calls `engine.reset()` (clears KV cache, position, token history) and clears `last_logits`. `model_load()` replaces `inference_engine` entirely and clears `last_logits`.

## IORequest / IOResponse Extensions

```rust
pub enum IORequest {
    FetchContent { cid: [u8; 32] },
    LoadModel { gguf_cid: [u8; 32], tokenizer_cid: [u8; 32] },
}

pub enum IOResponse {
    ContentReady { data: Vec<u8> },
    ContentNotFound,
    ModelReady { gguf_data: Vec<u8>, tokenizer_data: Vec<u8> },
    ModelGgufNotFound,
    ModelTokenizerNotFound,
}
```

The caller (harmony-node or test harness) receives `NeedsIO { LoadModel { ... } }`, resolves both CIDs from CAS (potentially assembling DAG bundles), and calls `resume_with_io()`:

- `ModelReady { gguf_data, tokenizer_data }` — both CIDs resolved successfully
- `ModelGgufNotFound` — GGUF CID could not be resolved
- `ModelTokenizerNotFound` — tokenizer CID could not be resolved

On `ModelReady`, the host function resumption:

1. Constructs `QwenEngine::new(Device::Cpu)`
2. Calls `engine.load_gguf(&gguf_data)` — returns -3 on failure
3. Calls `engine.load_tokenizer(&tokenizer_data)` — returns -4 on failure
4. Stores engine in `HostState.inference_engine`
5. Returns `0i32` to the WASM guest

On `ModelGgufNotFound`, returns -1. On `ModelTokenizerNotFound`, returns -2.

### Resume path differences

`model_load` does **not** need `io_write_target` — the return code is the host function's return value, delivered directly via the trap resumption API (`invocation.resume(&mut store, &[Val::I32(code)])` in wasmi, or function return in wasmtime's replay). The `resume_with_io` handler checks the `IORequest` variant to branch:

- `FetchContent` → writes data to `io_write_target` buffer, resumes with byte count
- `LoadModel` → constructs engine from response, resumes with status code

### WasmtimeRuntime replay adaptation

The existing wasmtime runtime uses `pending_cid: Option<[u8; 32]>` and `io_cache: HashMap<[u8; 32], Vec<u8>>` for `FetchContent` replay. For `LoadModel`, the wasmtime `HostState` is generalized:

- `pending_cid` becomes `pending_io: Option<IORequest>` to hold either variant
- A new `model_response: Option<IOResponse>` field caches the `ModelReady`/`ModelGgufNotFound`/`ModelTokenizerNotFound` response
- On replay, the `model_load` host function checks `model_response`: if present, it processes the cached response and returns the code directly (no trap)
- The existing `io_cache` remains unchanged for `FetchContent` replay

## Registration

Both `WasmiRuntime` and `WasmtimeRuntime` get `#[cfg(feature = "inference")]` blocks in their linker registration that call `linker.func_wrap("harmony", "model_load", ...)` etc. for all 6 functions.

The existing `fetch_content` registration is unchanged.

## Data Flow

### Autoregressive Generation (WASM guest perspective)

```wasm
;; 1. Load model (suspends for I/O)
(call $model_load (i32.const 0) (i32.const 32))    ;; CIDs at memory[0..64]

;; 2. Tokenize prompt
(call $tokenize (local.get $prompt_ptr) (local.get $prompt_len)
                (local.get $tok_ptr) (i32.const 16384))

;; 3. Prefill
(call $forward (local.get $tok_ptr) (local.get $tok_bytes))

;; 4. Sample first token
(call $sample (local.get $params_ptr) (i32.const 20))

;; 5. Decode loop
(loop $gen
  ;; Store sampled token, call forward with it
  (call $forward (local.get $next_tok_ptr) (i32.const 4))
  (call $sample (local.get $params_ptr) (i32.const 20))
  ;; Check for EOS, loop if not done
  (br_if $gen ...))

;; 6. Detokenize generated tokens
(call $detokenize (local.get $gen_ptr) (local.get $gen_bytes)
                   (local.get $out_ptr) (i32.const 4096))
```

### I/O Suspension Flow (model_load)

```
WASM guest                    Host runtime              Caller
    |                              |                       |
    |-- model_load(cid1, cid2) --> |                       |
    |                              |-- trap ---------> NeedsIO(LoadModel{cid1, cid2})
    |                              |                       |
    |                              |                  resolve CIDs from CAS
    |                              |                       |
    |                              | <-- resume_with_io(ModelReady{gguf, tok})
    |                              |                       |
    |                              | load_gguf + load_tokenizer
    |                              |                       |
    | <--- return 0 (success) ---- |                       |
```

## Testing Strategy

### Unit tests (no real model)

All tests follow the existing WAT module pattern from `fetch_content` tests. Each test creates a small WAT module that calls the host function and writes the return code to output memory.

- **model_load triggers NeedsIO** — call model_load, verify `ComputeResult::NeedsIO { LoadModel { ... } }`
- **model_load resume with ModelReady (invalid GGUF)** — resume with garbage bytes, verify return code -3
- **model_load resume with ModelGgufNotFound** — verify return code -1
- **model_load resume with ModelTokenizerNotFound** — verify return code -2
- **tokenize before model_load** — verify return code -1
- **forward before model_load** — verify return code -1
- **sample before forward** — verify return code -2
- **model_reset on empty engine** — verify return code 0

Tests run against both wasmi and wasmtime runtimes.

### Integration tests (`#[ignore]`)

Require real GGUF model files (env vars `HARMONY_TEST_GGUF`, `HARMONY_TEST_TOKENIZER`):

- **Full cycle** — model_load → resume → tokenize → forward → sample → verify valid token
- **Multi-turn** — forward → sample → forward → sample → reset → forward
- **Detokenize roundtrip** — tokenize text → detokenize tokens → verify content

## Scope Exclusions

- **GPU/Metal/CUDA** — future feature flags on harmony-inference
- **Logits in WASM memory** — logits stay host-side; WASM guests use `sample`
- **Engram embedding injection** — future integration point between forward and sample
- **Zenoh queryable service** — harmony-yku, blocked by this bead
- **Streaming generation API** — trivial to build on token-by-token primitives
- **Shared engine across modules** — per-module engine for now

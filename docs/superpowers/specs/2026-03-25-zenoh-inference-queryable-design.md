# Zenoh Inference Queryable Service

## Goal

Expose LLM inference as a Zenoh queryable on `harmony/compute/activity/inference`, so any node on the mesh can request text generation from a node with a loaded model. Text in, text out. Single reply per query.

## Architecture

A node configured with a model CID loads the model at startup, declares a Zenoh queryable, and advertises its capacity. Incoming inference queries are dispatched as WASM workflows using a built-in inference runner module that calls the `harmony-compute` inference host functions (model_load, tokenize, forward, sample, detokenize). The WorkflowEngine handles cooperative scheduling, dedup, and history tracking — inference is just another compute job.

### Dependencies

| Crate | Purpose |
|-------|---------|
| `harmony-zenoh` | QueryableRouter, namespace key expressions |
| `harmony-compute` | ComputeRuntime, inference host functions (feature = "inference") |
| `harmony-inference` | QwenEngine, InferenceEngine trait |
| `harmony-workflow` | WorkflowEngine, cooperative scheduling |
| `harmony-node` | Integration point — all wiring happens here |

No new crate. This is integration work in `harmony-node`.

## Query Protocol

### Key Expression

`harmony/compute/activity/inference` — constructed via `harmony_zenoh::namespace::compute::activity_key("inference")`. Add a `pub const INFERENCE_ACTIVITY: &str = "harmony/compute/activity/inference"` constant to the compute namespace module.

### Request Payload

```
[0x02] [prompt_len: u32 LE] [prompt_utf8] [sampling_params: 20 bytes (optional)]
```

- Tag `0x02` distinguishes inference from existing compute payloads (`0x00` inline WASM, `0x01` CID ref)
- `prompt_len` is the byte length of the UTF-8 prompt
- `sampling_params` uses the same 20-byte layout as the `sample` host function:

| Offset | Type | Field |
|--------|------|-------|
| 0 | f32 | temperature (0.0 = greedy) |
| 4 | f32 | top_p (1.0 = disabled) |
| 8 | u32 | top_k (0 = disabled) |
| 12 | f32 | repeat_penalty (1.0 = no penalty) |
| 16 | u32 | repeat_last_n (0 = unbounded) |

If the payload ends after the prompt (no sampling params), greedy defaults apply.

### Response Payload

```
[0x00] [generated_text_utf8]    — success
[0x01] [error_message_utf8]     — failure
```

Reuses the existing compute reply format.

## Model Lifecycle

### Startup Sequence

1. Node config specifies `inference_model_gguf_cid` and `inference_model_tokenizer_cid` (hex-encoded 32-byte CIDs)
2. If both are set, the node resolves them from CAS and calls `QwenEngine::new()` → `load_gguf()` → `load_tokenizer()`
3. On success: declare queryable on `harmony/compute/activity/inference`, publish capacity
4. On failure: log error, skip inference — node operates normally without inference capability

### Capacity Advertisement

Published to `harmony/compute/capacity/{node_addr}`:

```
[model_gguf_cid: 32 bytes] [status: u8]
```

- `0x01` = ready (can accept inference queries)
- `0x00` = busy (inference slot occupied, queries will queue)

One inference request runs at a time. Concurrent queries queue in the WorkflowEngine.

### No Hot-Reload

Changing the model requires a node restart. Multi-model and hot-swap are future beads.

## Inference Runner WASM Module

A built-in WAT module embedded in `harmony-node` as a constant byte slice. It drives the autoregressive loop using the inference host functions from harmony-compute (e0s).

### Input Layout (constructed by NodeRuntime)

```
[gguf_cid: 32 bytes] [tokenizer_cid: 32 bytes] [prompt_len: u32 LE] [prompt_utf8] [sampling_params: 20 bytes] [max_tokens: u32 LE]
```

The node prepends the model CIDs (from its config) and appends `max_tokens` before submitting to WorkflowEngine. Default: `const DEFAULT_MAX_INFERENCE_TOKENS: u32 = 512` in `harmony-node`.

### Execution Flow

1. Read CIDs from input → call `model_load(gguf_cid_ptr, tokenizer_cid_ptr)`
2. Read prompt → call `tokenize(prompt_ptr, prompt_len, out_ptr, out_cap)`
3. Call `forward(tokens_ptr, tokens_len)` for prefill
4. Loop: `sample(params_ptr, 20)` → check EOS → `forward(token_ptr, 4)` → repeat until EOS or max_tokens
5. Call `detokenize(generated_tokens_ptr, generated_len, out_ptr, out_cap)`
6. Return generated text as output

### Model Loading Optimization

The node pre-loads the model at startup. When the WASM runner calls `model_load`, the host function detects the engine is already loaded with matching CIDs and returns 0 without re-parsing. This requires a check in the wasmi `resume_with_io` path: if `inference_engine` is already populated and the pending `LoadModel` CIDs match, skip I/O suspension and return success directly.

For wasmtime, the cached model data in the session serves the same purpose — the model_load host function builds the engine from cache on replay.

## Integration with NodeRuntime

### Query Dispatch

When `DeliverQuery` arrives for the inference queryable:

1. Parse request payload (tag 0x02, prompt, optional sampling params)
2. Construct WASM input: `[gguf_cid] [tokenizer_cid] [prompt_len] [prompt] [sampling_params] [max_tokens]`
3. Submit to WorkflowEngine with the built-in inference runner module bytes
4. Store mapping: `workflow_id → query_id`

### Workflow Completion

When WorkflowEngine produces `Complete { output }`:

1. Look up `query_id` from `workflow_id` mapping
2. Format reply: `[0x00] [output]` (output is UTF-8 generated text)
3. Call `queryable_router.reply(query_id, payload)`

On workflow failure: reply with `[0x01] [error_message]`.

### Cooperative Scheduling

The inference WASM workflow yields like any other — WorkflowEngine gives it fuel slices, and other workflows get interleaved. No special treatment needed.

## Testing Strategy

### Unit tests (no real model)

- Parse/serialize request payload (tag 0x02, prompt, sampling params, defaults)
- Parse/serialize response payload (success, error)
- Capacity payload round-trip (model CID + status byte)
- Query routing: inference queryable ID detection + dispatch to WorkflowEngine
- Workflow-to-query mapping: complete → reply, fail → error reply
- Built-in WAT module compiles and validates

### Integration tests with WorkflowEngine (no real model)

- Submit inference runner → verify `NeedsIO(LoadModel)` emitted
- Resume with `ModelReady` (invalid GGUF) → verify error reply
- Verify cooperative yielding doesn't corrupt inference state

### Integration tests with real model (`#[ignore]`)

- Full cycle: query payload → inference runner → generated text reply
- Verify generated text is non-empty and valid UTF-8
- Uses `HARMONY_TEST_GGUF` / `HARMONY_TEST_TOKENIZER` env vars

## Scope Exclusions

- **Streaming responses** — single reply only; streaming needs a different Zenoh pattern
- **Multi-model** — one model per node, configured at startup
- **Token-level API** — text in, text out only (tracked as harmony-2cx)
- **Smart routing / load balancing** — capacity is advertised but routing is requester's responsibility
- **GPU/Metal acceleration** — CPU only (existing harmony-inference feature flags)
- **Custom inference WASM modules** — only the built-in runner for now
- **Configurable max_tokens** — hardcoded default (512) in request; future bead can add it to the protocol
- **Hot-reload / model swap** — requires node restart

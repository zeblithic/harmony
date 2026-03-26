# Zenoh Inference Queryable Service Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose LLM inference as a Zenoh queryable so mesh nodes can request text generation from nodes with loaded models.

**Architecture:** Integration work across harmony-zenoh (namespace constant), harmony-workflow (LoadModel I/O support), harmony-node (inference module, query routing, model loading). A built-in WAT inference runner module calls e0s host functions. Queries arrive via Zenoh queryable, get dispatched as WASM workflows, and replies route back through the queryable router.

**Tech Stack:** harmony-zenoh, harmony-compute (inference feature), harmony-inference, harmony-workflow, harmony-node

**Spec:** `docs/superpowers/specs/2026-03-25-zenoh-inference-queryable-design.md`

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-zenoh/src/namespace.rs` | Add `INFERENCE_ACTIVITY` constant |
| `crates/harmony-node/src/config.rs` | Add inference model CID fields to `ConfigFile` |
| `crates/harmony-node/src/inference.rs` | Payload parsing/serialization, constants, WAT embed |
| `crates/harmony-node/src/inference_runner.wat` | Built-in WAT module for autoregressive inference loop |
| `crates/harmony-node/build.rs` | Compile WAT → WASM at build time |
| `crates/harmony-workflow/src/types.rs` | Add `LoadModel` action/event variants |
| `crates/harmony-workflow/src/engine.rs` | Handle `IORequest::LoadModel` in `handle_compute_result` |
| `crates/harmony-node/src/runtime.rs` | Inference fields on `NodeConfig`/`NodeRuntime`, model loading, queryable declaration, query dispatch, `ParsedCompute::Inference` variant |

---

### Task 1: Namespace Constant + Config + Inference Payload Module

Add namespace constant, config fields, and the inference module with payload parsing, constants, and tests.

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`
- Modify: `crates/harmony-node/src/config.rs`
- Create: `crates/harmony-node/src/inference.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod inference;`)

- [ ] **Step 1: Add inference activity constant**

In `crates/harmony-zenoh/src/namespace.rs`, in the `pub mod compute` block after `ACTIVITY_SUB`:

```rust
/// Key expression for inference queryable.
pub const INFERENCE_ACTIVITY: &str = "harmony/compute/activity/inference";
```

- [ ] **Step 2: Add config fields**

In `crates/harmony-node/src/config.rs`, add to `ConfigFile`:

```rust
/// Hex-encoded 32-byte CID of the GGUF model file in CAS.
pub inference_model_gguf_cid: Option<String>,
/// Hex-encoded 32-byte CID of the tokenizer.json file in CAS.
pub inference_model_tokenizer_cid: Option<String>,
```

- [ ] **Step 3: Create inference.rs**

Create `crates/harmony-node/src/inference.rs` with:

- `pub const DEFAULT_MAX_INFERENCE_TOKENS: u32 = 512;`
- `pub const INFERENCE_TAG: u8 = 0x02;`
- `pub const CAPACITY_READY: u8 = 0x01;`
- `pub const CAPACITY_BUSY: u8 = 0x00;`
- `pub const INFERENCE_RUNNER_WASM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inference_runner.wasm"));` — compiled WASM from build.rs (placeholder: use `b""` until Task 3 creates the WAT + build.rs)
- `pub struct InferenceRequest { pub prompt: String, pub sampling_params: [u8; 20] }`
- `impl InferenceRequest { pub fn parse(payload: &[u8]) -> Result<Self, String>` — parses tag 0x02 + prompt_len + prompt + optional sampling params (greedy defaults if absent)
- `pub fn build_runner_input(gguf_cid, tokenizer_cid, request) -> Vec<u8>` — constructs `[gguf_cid:32][tok_cid:32][prompt_len:u32][prompt][params:20][max_tokens:u32]`
- `pub fn build_capacity_payload(model_cid, ready) -> Vec<u8>` — constructs `[cid:32][status:u8]`
- Unit tests: parse with params, parse greedy defaults, wrong tag, truncated, build_runner_input layout, capacity payload

For the WASM constant, temporarily use:
```rust
// Placeholder until build.rs compiles inference_runner.wat (Task 3)
pub const INFERENCE_RUNNER_WASM: &[u8] = &[];
```

- [ ] **Step 4: Add `mod inference;` to main.rs**

- [ ] **Step 5: Verify and commit**

Run: `cargo test -p harmony-node --lib inference`
Run: `cargo check -p harmony-zenoh`

```bash
git add crates/harmony-zenoh/src/namespace.rs crates/harmony-node/src/config.rs crates/harmony-node/src/inference.rs crates/harmony-node/src/main.rs
git commit -m "feat: add inference namespace constant, config fields, and payload module"
```

---

### Task 2: WorkflowEngine LoadModel Support

The WorkflowEngine's `handle_compute_result` currently destructures `IORequest::FetchContent { cid }` exhaustively. With the `LoadModel` variant added in e0s, this needs a new code path.

**Files:**
- Modify: `crates/harmony-workflow/src/types.rs`
- Modify: `crates/harmony-workflow/src/engine.rs`

**Key context:**
- `engine.rs:511-512`: `ComputeResult::NeedsIO { request } => { let IORequest::FetchContent { cid } = request;` — this is an exhaustive destructure that fails to compile with `LoadModel`
- Need new `WorkflowAction::LoadModel` and `WorkflowEvent::ModelLoaded`/`ModelLoadFailed`
- The engine should emit `LoadModel` action so the caller (NodeRuntime) can provide the model bytes
- The engine needs new `WorkflowStatus::WaitingForModel` or reuse `WaitingForIo`

- [ ] **Step 1: Add types**

In `crates/harmony-workflow/src/types.rs`:

Add to `WorkflowAction`:
```rust
/// Request model loading (inference host function called model_load).
LoadModel {
    workflow_id: WorkflowId,
    gguf_cid: [u8; 32],
    tokenizer_cid: [u8; 32],
},
```

Add to `WorkflowEvent`:
```rust
/// Model data resolved for an inference workflow.
ModelLoaded {
    gguf_cid: [u8; 32],
    tokenizer_cid: [u8; 32],
    gguf_data: Vec<u8>,
    tokenizer_data: Vec<u8>,
},
/// Model loading failed.
ModelLoadFailed {
    gguf_cid: [u8; 32],
    tokenizer_cid: [u8; 32],
    reason: String,
},
```

Add to `HistoryEvent`:
```rust
/// WASM module requested model loading.
ModelRequested { gguf_cid: [u8; 32], tokenizer_cid: [u8; 32] },
/// Model was resolved.
ModelResolved { gguf_cid: [u8; 32], tokenizer_cid: [u8; 32] },
```

Add to `WorkflowStatus`:
```rust
/// Suspended waiting for model loading.
WaitingForModel { gguf_cid: [u8; 32], tokenizer_cid: [u8; 32] },
```

- [ ] **Step 2: Handle LoadModel in engine.rs handle_compute_result**

In `engine.rs`, the `ComputeResult::NeedsIO` match arm currently does `let IORequest::FetchContent { cid } = request;`. Replace with a `match`:

```rust
ComputeResult::NeedsIO { request } => {
    match request {
        IORequest::FetchContent { cid } => {
            // ... existing FetchContent handling (unchanged) ...
        }
        IORequest::LoadModel { gguf_cid, tokenizer_cid } => {
            if let Some(state) = self.workflows.get_mut(&wf_id) {
                state.history.events.push(HistoryEvent::ModelRequested {
                    gguf_cid,
                    tokenizer_cid,
                });

                let saved_session = self
                    .runtime
                    .take_session()
                    .expect("NeedsIO implies session");
                state.saved_session = Some(saved_session);
                state.status = WorkflowStatus::WaitingForModel {
                    gguf_cid,
                    tokenizer_cid,
                };
                self.active = None;

                return vec![
                    WorkflowAction::LoadModel {
                        workflow_id: wf_id,
                        gguf_cid,
                        tokenizer_cid,
                    },
                    WorkflowAction::PersistHistory { workflow_id: wf_id },
                ];
            }
            self.active = None;
            return Vec::new();
        }
    }
}
```

- [ ] **Step 3: Handle ModelLoaded/ModelLoadFailed events in engine.rs**

In the `handle` method, add cases for the new events. The pattern mirrors `ContentFetched`/`ContentFetchFailed`:

For `ModelLoaded`: find the workflow in `WaitingForModel` state with matching CIDs, restore session, resume with `IOResponse::ModelReady { gguf_data, tokenizer_data }`, return the resulting actions.

For `ModelLoadFailed`: find the workflow, resume with `IOResponse::ModelGgufNotFound` (or `ModelTokenizerNotFound` based on the reason), return failure actions.

- [ ] **Step 4: Verify compilation and tests**

Run: `cargo test -p harmony-workflow`
Run: `cargo check -p harmony-workflow`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-workflow/src/types.rs crates/harmony-workflow/src/engine.rs
git commit -m "feat(workflow): add LoadModel I/O support for inference workflows"
```

---

### Task 3: Inference Runner WAT Module + build.rs

Create the built-in WAT module that drives the autoregressive inference loop and the build.rs that compiles it to WASM.

**Files:**
- Create: `crates/harmony-node/src/inference_runner.wat`
- Create: `crates/harmony-node/build.rs`
- Modify: `crates/harmony-node/Cargo.toml` (add `wat` build-dependency)
- Modify: `crates/harmony-node/src/inference.rs` (update WASM constant)

**Key context:**
- The WAT module imports 5 host functions from the "harmony" namespace: `model_load`, `tokenize`, `detokenize`, `forward`, `sample`
- It exports `memory` (2 pages = 128KB) and `compute(input_ptr: i32, input_len: i32) -> i32`
- Memory layout: input at [0..input_len], token buffer at 32768, generated tokens at 40960, detokenize output at 49152, final output at [input_len..]

- [ ] **Step 1: Create inference_runner.wat**

See the WAT module in the spec design section "Inference Runner WASM Module". The module:

1. Calls `model_load(input_ptr, input_ptr+32)` — CIDs at input[0..64]
2. Reads `prompt_len` from input[64..68], prompt at input[68..]
3. Calls `tokenize(prompt_ptr, prompt_len, 32768, 8192)` → token_bytes
4. Calls `forward(32768, token_bytes)` — prefill
5. Reads sampling params offset and max_tokens from input
6. Loop: `sample(params_ptr, 20)` → `forward(32764, 4)` → store token → check max_tokens
7. Calls `detokenize(40960, gen_count*4, 49152, 16384)` → output_bytes
8. `memory.copy(input_ptr+input_len, 49152, output_bytes)` — copy to output area
9. Returns output_bytes

- [ ] **Step 2: Create build.rs**

Create `crates/harmony-node/build.rs`:

```rust
fn main() {
    let wat_path = std::path::Path::new("src/inference_runner.wat");
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let wasm_path = std::path::Path::new(&out_dir).join("inference_runner.wasm");

    let wat_source = std::fs::read_to_string(wat_path)
        .expect("failed to read inference_runner.wat");
    let wasm_bytes = wat::parse_str(&wat_source)
        .expect("failed to compile inference_runner.wat");
    std::fs::write(&wasm_path, &wasm_bytes)
        .expect("failed to write inference_runner.wasm");

    println!("cargo:rerun-if-changed=src/inference_runner.wat");
}
```

- [ ] **Step 3: Add wat build-dependency to Cargo.toml**

In `crates/harmony-node/Cargo.toml`:

```toml
[build-dependencies]
wat = { version = "1" }
```

Check if `wat` is in the workspace dependencies already. If not, add it to the workspace `Cargo.toml` first.

- [ ] **Step 4: Update inference.rs WASM constant**

Replace the placeholder:
```rust
pub const INFERENCE_RUNNER_WASM: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/inference_runner.wasm"));
```

- [ ] **Step 5: Verify WAT compiles and produces valid WASM**

Run: `cargo build -p harmony-node`
Expected: build.rs compiles WAT to WASM, include_bytes embeds it

Run: `cargo test -p harmony-node --lib inference`
Expected: tests pass (add a test that `INFERENCE_RUNNER_WASM` is non-empty and starts with WASM magic bytes `\0asm`)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/inference_runner.wat crates/harmony-node/build.rs crates/harmony-node/Cargo.toml crates/harmony-node/src/inference.rs
git commit -m "feat(node): add inference runner WAT module with build.rs WASM compilation"
```

---

### Task 4: NodeRuntime Integration

Wire up model loading at startup, inference queryable declaration, query dispatch through WorkflowEngine, and reply routing.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/src/main.rs` (NodeConfig construction)
- Modify: `crates/harmony-node/Cargo.toml` (add inference feature + deps)

**Key context:**
- `NodeConfig` is constructed in `main.rs` and passed to `NodeRuntime::new()` which returns `(Self, Vec<RuntimeAction>)`
- Queryable IDs are stored in `compute_queryable_ids: HashSet<QueryableId>` and checked in `route_query()`
- Workflow-to-query mapping: `workflow_to_query: HashMap<WorkflowId, Vec<u64>>`
- `dispatch_workflow_actions()` converts `WorkflowComplete/Failed` to `SendReply`
- `parse_compute_payload()` returns `ParsedCompute` enum (needs new `Inference` variant)
- `RuntimeAction::Publish { key_expr, payload }` already exists for capacity advertisement
- `RuntimeAction::FetchContent { cid }` already exists for CAS resolution

- [ ] **Step 1: Add inference feature to harmony-node Cargo.toml**

```toml
[features]
inference = ["harmony-compute/inference", "dep:harmony-inference", "dep:candle-core"]

[dependencies]
harmony-inference = { workspace = true, optional = true }
candle-core = { workspace = true, optional = true }
```

Also ensure `harmony-workflow` forwards the inference feature to `harmony-compute`:
In `crates/harmony-workflow/Cargo.toml`:
```toml
[features]
inference = ["harmony-compute/inference"]
```
And in `harmony-node/Cargo.toml`:
```toml
inference = ["harmony-compute/inference", "harmony-workflow/inference", "dep:harmony-inference", "dep:candle-core"]
```

- [ ] **Step 2: Add inference fields to NodeConfig**

In `runtime.rs`, add to `NodeConfig`:
```rust
pub inference_gguf_cid: Option<[u8; 32]>,
pub inference_tokenizer_cid: Option<[u8; 32]>,
```

Update `NodeConfig` construction in `main.rs` — parse hex strings from `ConfigFile`:
```rust
let inference_gguf_cid = config_file.inference_model_gguf_cid.as_deref()
    .and_then(|s| hex::decode(s).ok())
    .and_then(|v| <[u8; 32]>::try_from(v).ok());
let inference_tokenizer_cid = config_file.inference_model_tokenizer_cid.as_deref()
    .and_then(|s| hex::decode(s).ok())
    .and_then(|v| <[u8; 32]>::try_from(v).ok());
```

Update all other `NodeConfig` construction sites (tests) to include the new fields as `None`.

- [ ] **Step 3: Add inference state to NodeRuntime**

Add fields:
```rust
inference_queryable_id: Option<harmony_zenoh::queryable::QueryableId>,
inference_model_cid: Option<[u8; 32]>,
inference_tokenizer_cid: Option<[u8; 32]>,
```

Initialize to `None` in the struct literal of `NodeRuntime::new()`.

- [ ] **Step 4: Add ParsedCompute::Inference variant**

Add to the `ParsedCompute` enum:
```rust
Inference {
    request: crate::inference::InferenceRequest,
},
```

Add `0x02` branch to `parse_compute_payload()`:
```rust
0x02 => {
    match crate::inference::InferenceRequest::parse(&payload) {
        Ok(request) => Some(ParsedCompute::Inference { request }),
        Err(_) => None,
    }
}
```

Add `ParsedCompute::Inference` match arm to `route_compute_query()`:
```rust
ParsedCompute::Inference { request } => {
    let (gguf_cid, tok_cid) = match (self.inference_model_cid, self.inference_tokenizer_cid) {
        (Some(g), Some(t)) => (g, t),
        _ => {
            let mut payload = vec![0x01];
            payload.extend_from_slice(b"no inference model loaded");
            self.pending_direct_actions.push(RuntimeAction::SendReply { query_id, payload });
            return Vec::new();
        }
    };

    let input = crate::inference::build_runner_input(&gguf_cid, &tok_cid, &request);
    let module = crate::inference::INFERENCE_RUNNER_WASM.to_vec();
    let module_hash = harmony_crypto::hash::blake3_hash(&module);
    let wf_id = WorkflowId::new(&module_hash, &input);
    self.workflow_to_query.entry(wf_id).or_default().push(query_id);
    self.workflow.handle(WorkflowEvent::Submit {
        module,
        input,
        hint: ComputeHint::PreferLocal,
    })
}
```

- [ ] **Step 5: Handle model loading lifecycle**

In `NodeRuntime::new()`, if inference CIDs are configured, emit `RuntimeAction::FetchContent` for the GGUF CID (and tokenizer CID). Store the CIDs in the runtime fields.

When the content arrives (via the existing `ContentFetchResponse` event path), detect that it's an inference model CID and handle accordingly:
- Store gguf_data and tokenizer_data
- Once both are resolved, construct `QwenEngine`, load model
- Declare queryable on `INFERENCE_ACTIVITY`, add to `compute_queryable_ids`
- Emit `RuntimeAction::Publish` for capacity

This reuses the existing CAS fetch infrastructure rather than adding new `RuntimeAction` variants.

- [ ] **Step 6: Handle WorkflowAction::LoadModel**

In `dispatch_workflow_actions()`, add handling for the new `LoadModel` action. When a workflow needs a model:
- If the node has a pre-loaded engine with matching CIDs, provide the model bytes from CAS cache
- Resume the workflow with `WorkflowEvent::ModelLoaded`
- If CIDs don't match the loaded model, respond with `ModelLoadFailed`

- [ ] **Step 7: Verify compilation and existing tests**

Run: `cargo check -p harmony-node --features inference`
Run: `cargo test -p harmony-node --lib`

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/ crates/harmony-workflow/Cargo.toml
git commit -m "feat(node): integrate inference queryable with model loading, query dispatch, and reply routing"
```

---

### Task 5: Integration Tests

Write tests verifying the inference query flow.

**Files:**
- Modify: `crates/harmony-node/src/inference.rs` (add tests if not already there)
- Modify: `crates/harmony-node/src/runtime.rs` (add inference integration tests)
- Modify: `crates/harmony-workflow/src/engine.rs` (add LoadModel handling test)

- [ ] **Step 1: WorkflowEngine LoadModel test**

In `engine.rs` tests: submit a WASM module that calls `model_load`, verify the engine emits `WorkflowAction::LoadModel` with the correct CIDs.

- [ ] **Step 2: NodeRuntime — inference query with no model returns error**

Create `NodeRuntime` without inference CIDs. Route an inference query (tag 0x02). Verify `SendReply` with error payload.

- [ ] **Step 3: NodeRuntime — inference query produces workflow submit**

Create `NodeRuntime` with inference CIDs configured and model "loaded" (mock). Route an inference query. Verify `WorkflowEvent::Submit` is produced with the runner module and correct input layout.

- [ ] **Step 4: NodeRuntime — startup emits fetch actions for model CIDs**

Create `NodeConfig` with inference CIDs. Verify `new()` returns `FetchContent` actions for both CIDs.

- [ ] **Step 5: WASM runner validation test**

Verify `INFERENCE_RUNNER_WASM` starts with WASM magic bytes (`\0asm`), is non-empty, and can be compiled by wasmi.

- [ ] **Step 6: Run all tests**

Run: `cargo test -p harmony-node --features inference`
Run: `cargo test -p harmony-workflow`
Run: `cargo clippy -p harmony-node --features inference -- -D warnings`

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/ crates/harmony-workflow/
git commit -m "test: add inference queryable integration tests"
```

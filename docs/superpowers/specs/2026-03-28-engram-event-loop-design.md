# Engram Event Loop Wiring Design

**Goal:** Wire the Engram forward pass API (harmony-5h0b) into harmony-node's event loop so that inference queries use Engram-augmented forward passes when an Engram manifest is configured.

**Motivation:** The inference-side API is complete: `engram_bridge` (two-phase lookup), `EngramGatedResidual` (gated injection module), and `QwenEngine::forward_with_engram()`. This bead connects them to the real CAS network — fetching the manifest at startup, fetching shards per-query via Zenoh, and passing pre-resolved embeddings to the blocking inference task.

**Scope:** harmony-node only. Config, runtime fields, manifest loading, RunInference intercept expansion. No changes to harmony-inference or harmony-engram.

---

## Architecture

Pre-fetch approach: all Engram shards are fetched before spawning the blocking inference task. The event loop tokenizes early, computes N-grams, fetches shards in parallel via Zenoh, resolves embeddings, then spawns `run_inference_loop` with pre-built Engram context. Decode steps use normal `forward()` (no Engram) — streaming shard fetch during decode is deferred to harmony-geef.

Single code path with optional Engram: if `engram_client` is `None`, the existing non-Engram inference path runs unchanged.

---

## NodeConfig + Runtime Fields

### NodeConfig

```rust
pub engram_manifest_cid: Option<[u8; 32]>,
```

If `None`, Engram is disabled entirely.

### NodeRuntime new fields

```rust
#[cfg(feature = "inference")]
engram_manifest_data: Option<Vec<u8>>,
#[cfg(feature = "inference")]
engram_client: Option<harmony_engram::EngramClient>,
#[cfg(feature = "inference")]
engram_module: Option<harmony_inference::EngramGatedResidual>,
#[cfg(feature = "inference")]
engram_injection_layers: Vec<usize>,
```

### Manifest Loading

Follows the inference GGUF/tokenizer pattern:

1. At startup, if `engram_manifest_cid` is `Some`, emit `RuntimeAction::FetchContent { cid }`
2. In `ContentFetchResponse` handler, check if CID matches `engram_manifest_cid`
3. If match: store raw bytes in `engram_manifest_data`
4. Call `check_engram_ready()`:
   - Parse `ManifestHeader::from_bytes(&data)` for config
   - Parse `parse_shard_cids(&data[header_len..])` for shard CID list
   - Construct `EngramClient::from_manifest(config, shard_cids)`
   - Store in `engram_client`
5. `engram_module` starts as `None` — for integration testing, constructed with `EngramGatedResidual::new()` (random weights). Production weight loading from CAS is a separate concern (future bead).
6. `engram_injection_layers` defaults to `vec![2, 14]` (from DeepSeek research).

---

## RunInference Intercept Expansion

### Tokenize Early

Currently `run_inference_loop` receives `InferenceInput` (either `Text` or `TokenIds`) and tokenizes inside the blocking task. With Engram, tokens are needed before spawning (for N-gram extraction).

Change: tokenize in the async event loop before spawn. The blocking task always receives `Vec<u32>` tokens.

```rust
// In RunInference intercept:
let (tokens, is_token_mode) = match input {
    InferenceInput::Text(prompt) => {
        let t = engine.tokenize(&prompt).map_err(...)?;
        (t, false)
    }
    InferenceInput::TokenIds(ids) => (ids, true),
};
```

### Engram Branch

```
if engram_client is Some AND engram_module is Some:
    tokio::spawn async {
        1. prepare_engram_request(client, &tokens) → EngramRequest
        2. For each required_shard:
             fetch_via_zenoh(session, content::fetch_key(hex(cid))) → await
           (all in parallel via futures::future::join_all)
        3. Build HashMap<u64, Vec<u8>> from results
        4. resolve_engram_embeddings(client, request, shard_data, device) → Tensor
        5. Build EngramPrefill { module, embeddings, injection_layers }
        6. spawn_blocking(run_inference_loop(engine, tx, ..., Some(engram_prefill)))
        7. Panic monitor (same as existing pattern)
    }
else:
    spawn_blocking(run_inference_loop(engine, tx, ..., None))
```

If any shard fetch fails: log warning, fall back to non-Engram inference (send `run_inference_loop` with `None` engram). Engram is a quality enhancement, not a hard requirement — graceful degradation.

### run_inference_loop Signature Change

```rust
fn run_inference_loop(
    engine: harmony_inference::QwenEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    tokens: Vec<u32>,
    is_token_mode: bool,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
    engram: Option<EngramPrefill>,
)
```

Where:

```rust
struct EngramPrefill {
    module: harmony_inference::EngramGatedResidual,
    embeddings: candle_core::Tensor,
    injection_layers: Vec<usize>,
}
```

Inside the loop:
- Prefill call: if `engram` is `Some`, use `engine.forward_with_engram(&tokens, &mut cache, &ctx)` where `ctx` is built from `EngramPrefill`
- Decode calls: use `engine.forward(&[next_token], &mut cache)` (no Engram for single-token steps — that's harmony-geef's scope)

---

## Graceful Degradation

Engram failures should never break inference:

- Manifest fetch fails → log error, `engram_client` stays `None`, all inference uses normal `forward()`
- Manifest parse fails → same: log error, `engram_client` stays `None`
- Shard fetch fails (any shard) → log warning, this query falls back to non-Engram `forward()`
- Shard resolve fails → same: log warning, fall back

The user gets inference output regardless — just without Engram augmentation.

---

## File Map

**Modified:**

| File | Change |
|------|--------|
| `crates/harmony-node/src/runtime.rs` | Add `engram_*` fields to NodeRuntime, manifest parse in ContentFetchResponse, `check_engram_ready()` |
| `crates/harmony-node/src/event_loop.rs` | Tokenize-early in RunInference intercept, Engram shard fetch branch, updated `run_inference_loop` signature |
| `crates/harmony-node/Cargo.toml` | Add `harmony-engram` to inference feature deps |

**No new files.** Integration work into existing files.

**No changes to:** harmony-inference, harmony-engram, harmony-compute, InferenceEngine trait.

---

## What This Does NOT Include

- Decode-step Engram (streaming shard fetch during generation) — harmony-geef
- Trained Engram weight loading from CAS — future bead
- Engram table training — harmony-ws11
- Async prefetch pipelining — harmony-geef
- DAG manifest (multi-book) — deferred until table sizes demand it

---

## Testing Strategy

**Unit tests in runtime.rs:**

1. **Manifest parse round-trip** — Construct fake manifest bytes (ManifestHeader + shard CIDs), call the parse logic, verify `EngramClient` has correct config and shard count
2. **Missing manifest CID → no fetch** — `engram_manifest_cid: None` → verify no `FetchContent` emitted at startup

**Existing tests preserved:**

3. All existing inference tests pass unchanged — non-Engram path is identical
4. All existing DSD tests pass — DSD doesn't use Engram

**Integration testing** (requires real Zenoh + CAS, deferred):

5. End-to-end: configure manifest CID → fetch → parse → inference query → shard fetch → Engram-augmented logits

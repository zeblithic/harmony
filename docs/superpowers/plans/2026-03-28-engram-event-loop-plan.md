# Engram Event Loop Wiring Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire Engram conditional memory into harmony-node's event loop so inference queries use Engram-augmented forward passes when configured.

**Architecture:** Manifest fetched at startup via `FetchContent`. RunInference intercept tokenizes early, calls `prepare_engram_request` (sync), fetches shards in parallel (async Zenoh), then spawns blocking task with shard data. The blocking task calls `resolve_engram_embeddings` + `forward_with_engram`. Non-Engram path unchanged.

**Tech Stack:** harmony-node, harmony-engram, harmony-inference, tokio, zenoh, futures

**Spec:** `docs/superpowers/specs/2026-03-28-engram-event-loop-design.md`

**Key design note:** `resolve_engram_embeddings` needs `&EngramClient` which can't cross the async boundary without `Clone`. We clone the client per-request (cheap — just config + Vec<[u8;32]>). The async block fetches shards only. The blocking task receives the cloned client + shard data and does resolve + inference.

---

## File Map

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/Cargo.toml` | Add `harmony-engram` optional dep behind `inference` feature |
| `crates/harmony-node/src/config.rs` | Add `engram_manifest_cid` to ConfigFile |
| `crates/harmony-node/src/main.rs` | Hex-decode engram manifest CID into NodeConfig |
| `crates/harmony-node/src/runtime.rs` | Add Engram fields to NodeConfig + NodeRuntime, manifest parse in ContentFetchResponse, `parse_engram_manifest()`, startup fetch |
| `crates/harmony-node/src/event_loop.rs` | Tokenize-early in RunInference, Engram shard fetch branch, updated `run_inference_loop` signature with `Option<EngramPrefill>` |
| `crates/harmony-engram/src/lib.rs` | Add `Clone` derive to `EngramClient` |

---

### Task 1: Config + Dependencies

Add the `engram_manifest_cid` config field and the `harmony-engram` dependency.

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`
- Modify: `crates/harmony-node/src/config.rs`
- Modify: `crates/harmony-node/src/main.rs`
- Modify: `crates/harmony-node/src/runtime.rs` (NodeConfig only)

- [ ] **Step 1: Add harmony-engram dependency to Cargo.toml**

In `crates/harmony-node/Cargo.toml`, add to `[dependencies]` (after `harmony-speculative`, line 41):

```toml
harmony-engram = { workspace = true, optional = true }
```

Update the `inference` feature (line 16) to include it:

```toml
inference = ["harmony-compute/inference", "harmony-workflow/inference", "dep:harmony-inference", "dep:candle-core", "dep:harmony-agent", "dep:harmony-engram"]
```

- [ ] **Step 2: Add engram_manifest_cid to ConfigFile**

In `crates/harmony-node/src/config.rs`, add after `inference_model_tokenizer_cid` (line 86):

```rust
    /// Hex-encoded 32-byte CID of the Engram manifest in CAS.
    pub engram_manifest_cid: Option<String>,
```

- [ ] **Step 3: Add engram_manifest_cid to NodeConfig**

In `crates/harmony-node/src/runtime.rs`, add after `inference_tokenizer_cid` (line 76) in the `NodeConfig` struct:

```rust
    /// Hex-decoded 32-byte CID of the Engram manifest in CAS.
    pub engram_manifest_cid: Option<[u8; 32]>,
```

- [ ] **Step 4: Add hex-decode plumbing in main.rs**

In `crates/harmony-node/src/main.rs`, in the `NodeConfig` construction block, add after the `inference_tokenizer_cid` field (after line 636):

```rust
                engram_manifest_cid: config_file
                    .engram_manifest_cid
                    .as_deref()
                    .and_then(|s| {
                        hex::decode(s)
                            .ok()
                            .and_then(|v| <[u8; 32]>::try_from(v).ok())
                            .or_else(|| {
                                tracing::warn!("engram_manifest_cid is not a valid 32-byte hex string; Engram disabled");
                                None
                            })
                    }),
```

- [ ] **Step 5: Verify workspace compiles**

Run: `cargo check --workspace`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/config.rs crates/harmony-node/src/runtime.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add engram_manifest_cid config and harmony-engram dependency"
```

---

### Task 2: Runtime Engram Fields + Manifest Loading

Add Engram fields to NodeRuntime, manifest fetch at startup, and manifest parsing in ContentFetchResponse.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add Engram fields to NodeRuntime**

In the `NodeRuntime` struct (after the inference-related fields, around line 725), add:

```rust
    /// Engram manifest CID (from config). Set to None on parse failure.
    #[cfg(feature = "inference")]
    engram_manifest_cid: Option<[u8; 32]>,
    /// Engram client for shard lookups (constructed after manifest parsed).
    #[cfg(feature = "inference")]
    engram_client: Option<harmony_engram::EngramClient>,
    /// Engram gated residual module (random-init until trained weights loaded).
    #[cfg(feature = "inference")]
    engram_module: Option<harmony_inference::EngramGatedResidual>,
    /// Which transformer layers to inject Engram at.
    #[cfg(feature = "inference")]
    engram_injection_layers: Vec<usize>,
```

Initialize them in `NodeRuntime::new()` (around line 927):

```rust
            #[cfg(feature = "inference")]
            engram_manifest_cid: config.engram_manifest_cid,
            #[cfg(feature = "inference")]
            engram_client: None,
            #[cfg(feature = "inference")]
            engram_module: None,
            #[cfg(feature = "inference")]
            engram_injection_layers: vec![2, 14],
```

- [ ] **Step 2: Add manifest FetchContent at startup**

In the startup block (around line 976, after the existing inference CID fetches), add:

```rust
            #[cfg(feature = "inference")]
            if let Some(engram_cid) = rt.engram_manifest_cid {
                rt.pending_direct_actions
                    .push(RuntimeAction::FetchContent { cid: engram_cid });
                tracing::info!("fetching Engram manifest: {}", hex::encode(engram_cid));
            }
```

- [ ] **Step 3: Add ContentFetchResponse handler for Engram manifest**

In the `ContentFetchResponse` handler (around line 1346), add Engram manifest handling. The existing handler has `if is_inference_gguf || is_inference_tok { ... }`. Add an `else if` after it (or handle the remaining `result` for Engram). The key is: if the CID matches `engram_manifest_cid` and `engram_client` is still `None`, parse the manifest.

After the inference block's closing brace but before the general workflow fallthrough, add:

```rust
                #[cfg(feature = "inference")]
                {
                    let is_engram = Some(cid) == self.engram_manifest_cid
                        && self.engram_client.is_none();
                    if is_engram {
                        match &result {
                            Ok(data) => self.parse_engram_manifest(data),
                            Err(e) => {
                                tracing::error!(
                                    "Engram manifest fetch failed ({}): {e}; Engram disabled",
                                    hex::encode(cid)
                                );
                                self.engram_manifest_cid = None;
                            }
                        }
                    }
                }
```

Note: use `&result` (borrow) so the workflow handler can still process the result below.

- [ ] **Step 4: Implement parse_engram_manifest**

Add near `check_inference_model_ready` (around line 2589):

```rust
    #[cfg(feature = "inference")]
    fn parse_engram_manifest(&mut self, data: &[u8]) {
        use harmony_engram::{manifest::parse_shard_cids, EngramClient, ManifestHeader};

        let (header, remaining) = match postcard::take_from_bytes::<ManifestHeader>(data) {
            Ok(pair) => pair,
            Err(e) => {
                tracing::error!("Engram manifest header parse failed: {e}; Engram disabled");
                self.engram_manifest_cid = None;
                return;
            }
        };

        let shard_cids = match parse_shard_cids(remaining) {
            Ok(cids) => cids,
            Err(e) => {
                tracing::error!("Engram shard CIDs parse failed: {e}; Engram disabled");
                self.engram_manifest_cid = None;
                return;
            }
        };

        if shard_cids.len() != header.num_shards as usize {
            tracing::error!(
                "Engram shard count mismatch: header={}, actual={}; Engram disabled",
                header.num_shards,
                shard_cids.len()
            );
            self.engram_manifest_cid = None;
            return;
        }

        let config = header.to_config();
        let engram_dim = config.embedding_dim;
        tracing::info!(
            "Engram manifest loaded: {} shards, dim={}, {} heads",
            shard_cids.len(),
            engram_dim,
            config.num_heads,
        );
        self.engram_client = Some(EngramClient::from_manifest(config, shard_cids));

        // Random-init module for integration testing. Trained weights loaded separately.
        let hidden_dim = 1536; // Qwen3-0.6B
        match harmony_inference::EngramGatedResidual::new(engram_dim, hidden_dim, 3, &candle_core::Device::Cpu) {
            Ok(m) => {
                self.engram_module = Some(m);
                tracing::info!("Engram module initialized (random weights)");
            }
            Err(e) => tracing::error!("Engram module creation failed: {e}"),
        }
    }
```

- [ ] **Step 5: Verify workspace compiles**

Run: `cargo check --workspace`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): add Engram runtime fields, manifest fetch, and parse"
```

---

### Task 3: Make EngramClient Clone + RunInference Engram Branch

Add `Clone` to `EngramClient`, then expand the RunInference intercept for Engram.

**Files:**
- Modify: `crates/harmony-engram/src/lib.rs`
- Modify: `crates/harmony-node/src/event_loop.rs`

- [ ] **Step 1: Add Clone derive to EngramClient**

In `crates/harmony-engram/src/lib.rs`, change line 68:

```rust
#[derive(Debug)]
pub struct EngramClient {
```

to:

```rust
#[derive(Debug, Clone)]
pub struct EngramClient {
```

- [ ] **Step 2: Add EngramPrefill struct**

In `crates/harmony-node/src/event_loop.rs`, add before `run_inference_loop` (around line 1999):

```rust
/// Pre-fetched Engram data for the blocking inference task.
///
/// The async event loop fetches shards, then this struct carries everything
/// the blocking task needs to resolve embeddings and call forward_with_engram.
#[cfg(feature = "inference")]
struct EngramPrefill {
    client: harmony_engram::EngramClient,
    module: harmony_inference::EngramGatedResidual,
    request: harmony_inference::engram_bridge::EngramRequest,
    shard_data: std::collections::HashMap<u64, Vec<u8>>,
    injection_layers: Vec<usize>,
}
```

- [ ] **Step 3: Update run_inference_loop signature**

Change the `run_inference_loop` signature to accept pre-tokenized tokens and optional Engram:

```rust
#[cfg(feature = "inference")]
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

Update the function body:
- Remove the `InferenceInput` match block (tokens already provided)
- Add `history.extend_from_slice(&tokens)` at the start
- For the prefill forward call, branch on `engram`:

```rust
    let mut logits = if let Some(ref engram) = engram {
        match harmony_inference::engram_bridge::resolve_engram_embeddings(
            &engram.client,
            &engram.request,
            &engram.shard_data,
            &candle_core::Device::Cpu,
        ) {
            Ok(embeddings) => {
                let ctx = harmony_inference::EngramContext {
                    module: &engram.module,
                    embeddings,
                    injection_layers: &engram.injection_layers,
                };
                match engine.forward_with_engram(&tokens, &mut cache, &ctx) {
                    Ok(l) => l,
                    Err(e) => { /* send Failed, return */ }
                }
            }
            Err(e) => {
                tracing::warn!("Engram resolve failed: {e}; falling back to non-Engram");
                match engine.forward(&tokens, &mut cache) {
                    Ok(l) => l,
                    Err(e) => { /* send Failed, return */ }
                }
            }
        }
    } else {
        match engine.forward(&tokens, &mut cache) {
            Ok(l) => l,
            Err(e) => { /* send Failed, return */ }
        }
    };
```

- Decode loop: always `engine.forward(&[next_token], &mut cache)` (no Engram)

- [ ] **Step 4: Expand the RunInference intercept**

Replace the existing RunInference intercept block (lines ~1157-1220) with the expanded version:

1. **Tokenize before taking engine** — borrow `verification_engine` via `as_ref()`, tokenize, handle errors without taking engine
2. **Take engine** after successful tokenization
3. **Prepare Engram (sync)** — if `engram_client` + `engram_module` both present, call `prepare_engram_request` synchronously. Clone client + module for the async/blocking task.
4. **Engram branch** — `tokio::spawn` async block: fetch all shards in parallel via `join_all`, build shard_data HashMap, on any failure fall back to non-Engram. Then `spawn_blocking(run_inference_loop(..., Some(engram_prefill)))`.
5. **Non-Engram branch** — `spawn_blocking(run_inference_loop(..., None))` (same as today with pre-tokenized input).

The implementer should read the existing intercept code carefully and follow its patterns for engine take/return, panic monitoring, and error reply formatting.

- [ ] **Step 5: Verify workspace compiles**

Run: `cargo check --workspace`

- [ ] **Step 6: Run all tests**

Run: `cargo test --workspace --exclude harmony-tunnel`

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-engram/src/lib.rs crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): wire Engram shard fetching into RunInference intercept"
```

---

### Task 4: Verify and Clean Up

Final verification.

**Files:**
- All modified files

- [ ] **Step 1: Run clippy**

Run: `cargo clippy --workspace`

- [ ] **Step 2: Run format check**

Run: `cargo fmt --all -- --check`

- [ ] **Step 3: Run full test suite**

Run: `cargo test --workspace --exclude harmony-tunnel`

- [ ] **Step 4: Commit any fixes**

```bash
git add crates/harmony-node/ crates/harmony-engram/
git commit -m "chore: clippy and fmt fixes for Engram event loop wiring"
```

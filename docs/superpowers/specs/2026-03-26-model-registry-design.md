# Model Registry and CAS Distribution

## Overview

Structured model metadata and mesh-wide discovery for ML models stored in Harmony CAS. A `ModelManifest` describes each model (family, parameters, quantization, capabilities, memory requirements) and points to the model data CID. Nodes advertise available models via Zenoh push-based publications. The existing CAS pipeline (cache → disk → S3 fallback) handles all data storage and retrieval.

**Goal:** Any node on the mesh can discover which models are available, where they live, and whether they'll fit — enabling the agent protocol to route inference tasks to capable nodes.

**Scope:** Metadata types, wire format, Zenoh discovery namespace, and a local ModelRegistry state machine. Read-side only — model ingestion tooling, automatic download, and eviction are out of scope.

## What Already Exists

- **CAS books + DAG chunking:** `ContentId::for_book()` for ≤1MB, `dag::ingest()` for larger files via FastCDC + Merkle tree. Models of any size are already storable.
- **S3 archivist:** Auto-archives public durable content. Model data and manifests persist to S3 automatically.
- **S3 fallback:** StorageTier falls back to S3 on cache+disk miss for durable CIDs (PR #134).
- **InferenceEngine:** `load_gguf(&[u8])` takes raw bytes — no registry coupling needed.
- **Engram precedent:** Manifest-as-CID pattern for distributed data (embedding table shards).

No new infrastructure needed for storage or retrieval. The gap is structured metadata and discovery.

## Design Decisions

### Separate metadata CID (not a bundle wrapper)

Model metadata is its own small book CID containing a postcard-encoded `ModelManifest`. The manifest includes a `data_cid` field pointing to the model data's DAG root. This avoids bundle nesting complexity — unlike engram (which enumerates thousands of shard CIDs), a model manifest has just one data pointer. A single book fetch retrieves the full manifest.

### Push-based Zenoh discovery

Each node publishes a `ModelAdvertisement` to `harmony/model/{manifest_cid}/{node_addr}` for each locally available model. Subscribers see all advertisements on the mesh. This matches the existing capacity advertisement pattern and works well for the mesh sizes Harmony targets. Bandwidth-efficient alternatives (Bloom filters, queryables) are deferred to harmony-eo6f.

### Operational metadata, not provenance

The manifest captures what's needed for routing and compatibility: family, parameter count, quantization, context length, vocab size, memory estimate, supported tasks. Provenance fields (source URL, license, lineage, benchmarks) are deferred to harmony-rijm.

### New crate, not extension

`harmony-model` gets its own crate — same pattern as `harmony-agent`. This keeps the dependency graph clean: consumers that only need metadata types don't pull in candle (via harmony-inference) or the full content pipeline (via harmony-content beyond ContentId).

### Public durable content flags

Both manifests and model data use content flags `00` (public durable). The archivist auto-archives them to S3, and S3 fallback auto-retrieves on miss. No special content-class handling needed.

### Reuse existing RuntimeAction/RuntimeEvent pattern

No new RuntimeAction or RuntimeEvent variants. Publications go through the existing `RuntimeAction::Publish { key_expr, payload }`. Inbound advertisements arrive as `RuntimeEvent::SubscriptionMessage { key_expr, payload }` and are demuxed by key prefix in `route_subscription()`. This follows the established pattern and avoids touching match arms across the codebase.

### Empty-payload retraction

When a node removes a model, it publishes an empty payload to the same Zenoh key. The receiver's `route_subscription` handler checks for empty payload and removes the corresponding entry from `remote_models`. This avoids needing a Zenoh delete API or new event loop plumbing — empty-payload-as-tombstone is a standard Zenoh pattern.

## New Types

### ModelManifest

```rust
/// Describes a model stored in Harmony CAS.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelManifest {
    /// Human-readable name (e.g., "Qwen3-0.6B-Q4_K_M").
    pub name: String,
    /// Model family identifier (e.g., "qwen3", "llama", "phi").
    pub family: String,
    /// Parameter count (e.g., 600_000_000 for 0.6B).
    pub parameter_count: u64,
    /// Serialization format.
    pub format: ModelFormat,
    /// Quantization method (e.g., "Q4_K_M", "F16", "BF16"). None for full precision.
    pub quantization: Option<String>,
    /// Context window size in tokens.
    pub context_length: u32,
    /// Vocabulary size.
    pub vocab_size: u32,
    /// Estimated memory required in bytes.
    pub memory_estimate: u64,
    /// Tasks this model supports.
    pub tasks: Vec<ModelTask>,
    /// CID of the model data (DAG root for large models, book for small).
    pub data_cid: ContentId,
    /// CID of the tokenizer (JSON book). None if embedded in model file.
    pub tokenizer_cid: Option<ContentId>,
}
```

### ModelFormat

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}
```

### ModelTask

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTask {
    TextGeneration,
    Embedding,
    Vision,
    AudioTranscription,
}
```

### ModelAdvertisement

```rust
/// Compact advertisement published via Zenoh for quick filtering.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAdvertisement {
    /// Manifest CID (redundant with key, self-contained for consumers).
    pub manifest_cid: ContentId,
    /// Subset of manifest fields for filtering without fetching full manifest.
    pub name: String,
    pub family: String,
    pub parameter_count: u64,
    pub quantization: Option<String>,
    pub tasks: Vec<ModelTask>,
    pub memory_estimate: u64,
}
```

### ModelError

```rust
#[derive(Debug)]
pub enum ModelError {
    EncodeFailed,
    DecodeFailed,
    ManifestTooLarge,
}
```

Implements `Display` and `std::error::Error`.

## Wire Format

Postcard encoding for all types (matches engram convention).

```rust
pub fn encode_manifest(manifest: &ModelManifest) -> Result<Vec<u8>, ModelError>;
pub fn decode_manifest(data: &[u8]) -> Result<ModelManifest, ModelError>;
pub fn manifest_cid(data: &[u8]) -> ContentId;

pub fn encode_advertisement(ad: &ModelAdvertisement) -> Result<Vec<u8>, ModelError>;
pub fn decode_advertisement(data: &[u8]) -> Result<ModelAdvertisement, ModelError>;
```

`manifest_cid` hardcodes public durable flags (`00`) — manifests are always public durable per the design decision above.

Safety: `encode_manifest` returns `ManifestTooLarge` if the encoded size exceeds 64KB (manifests should be ~300 bytes).

## ModelRegistry State Machine

Sans-I/O state machine tracking local and remote model availability.

### State

```rust
pub struct ModelRegistry {
    /// Models this node has locally (manifest CID → manifest).
    local_models: HashMap<ContentId, ModelManifest>,
    /// Remote advertisements (manifest CID → map of node address → advertisement).
    remote_models: HashMap<ContentId, HashMap<[u8; 16], ModelAdvertisement>>,
    /// This node's address (for constructing Zenoh keys).
    local_addr: [u8; 16],
}
```

### Events

```rust
pub enum ModelRegistryEvent {
    /// Local model registered (e.g., after ingesting a GGUF file).
    RegisterLocal { manifest_cid: ContentId, manifest: ModelManifest },
    /// Local model removed.
    UnregisterLocal { manifest_cid: ContentId },
    /// Remote advertisement received via Zenoh.
    AdvertisementReceived { manifest_cid: ContentId, node_addr: [u8; 16], ad: ModelAdvertisement },
    /// Remote node departed (stale timeout or explicit).
    NodeDeparted { node_addr: [u8; 16] },
}
```

### Actions

```rust
pub enum ModelRegistryAction {
    /// Publish advertisement to Zenoh (via existing RuntimeAction::Publish).
    PublishAdvertisement { key_expr: String, payload: Vec<u8> },
    /// Retract advertisement: publish empty payload to same key (tombstone).
    RetractAdvertisement { key_expr: String },
}
```

Both actions map to the existing `RuntimeAction::Publish { key_expr, payload }`. `RetractAdvertisement` uses `payload: vec![]` (empty payload as tombstone).

### Query Methods

```rust
impl ModelRegistry {
    pub fn local_models(&self) -> &HashMap<ContentId, ModelManifest>;
    pub fn find_by_task(&self, task: ModelTask) -> Vec<(&ContentId, Source)>;
    pub fn find_by_family(&self, family: &str) -> Vec<(&ContentId, Source)>;
    pub fn nodes_for_model(&self, manifest_cid: &ContentId) -> Option<Vec<[u8; 16]>>;
    /// Remove a specific node's advertisement (called on empty-payload retraction).
    pub fn remove_advertisement(&mut self, manifest_cid: &ContentId, node_addr: &[u8; 16]);
}

pub enum Source {
    Local,
    Remote(Vec<[u8; 16]>),
    Both(Vec<[u8; 16]>),
}
```

### Behavior

- `RegisterLocal`: stores manifest, emits `PublishAdvertisement` with encoded `ModelAdvertisement`.
- `UnregisterLocal`: removes from local_models, emits `RetractAdvertisement`.
- `AdvertisementReceived`: stores advertisement in remote_models (manifest_cid → node_addr → ad). This allows `find_by_task` and `find_by_family` to filter remote models by advertisement metadata without fetching manifests from CAS.
- `NodeDeparted`: removes node_addr from all remote_models entries; prunes entries with no remaining nodes.
- Query methods return `Source` to distinguish local, remote, or both.

## Zenoh Namespace

```
harmony/model/{manifest_cid_hex}/{node_addr_hex}
```

- Subscribe `harmony/model/**` → all model advertisements on mesh.
- Subscribe `harmony/model/{cid}/*` → all nodes advertising a specific model.
- `{node_addr_hex}` suffix prevents collisions for the same model on multiple nodes.

### Namespace Builders

Added to `harmony-zenoh::namespace::model`:

```rust
pub const PREFIX: &str = "harmony/model";
pub fn advertisement_key(manifest_cid: &ContentId, node_addr: &[u8; 16]) -> String;
pub fn advertisement_sub_all() -> &'static str;
pub fn advertisement_sub_model(manifest_cid: &ContentId) -> String;
```

## NodeRuntime Integration

### No new RuntimeAction/RuntimeEvent variants

Follow the existing pattern: all Zenoh publications go through `RuntimeAction::Publish { key_expr, payload }` and all inbound subscriptions arrive as `RuntimeEvent::SubscriptionMessage { key_expr, payload }`, demuxed by key prefix in `route_subscription()`.

### Wiring

- `ModelRegistryAction::PublishAdvertisement` → `RuntimeAction::Publish { key_expr, payload }`
- `ModelRegistryAction::RetractAdvertisement` → `RuntimeAction::Publish { key_expr, payload: vec![] }`
- Zenoh subscriber for `harmony/model/**` → `RuntimeEvent::SubscriptionMessage` → `route_subscription()` matches `harmony/model/` prefix → decodes advertisement, extracts manifest_cid and node_addr from key → dispatches to `ModelRegistry::handle_event(AdvertisementReceived { ... })`
- Empty payload in `route_subscription()`: calls `ModelRegistry::remove_advertisement(manifest_cid, node_addr)` directly (same pattern as `peer_filters.upsert_*()` — no event variant needed for retraction)

## Event Loop Integration

- On startup, subscribe to `harmony/model/**` (same mechanism as existing subscriptions).
- No new action handling needed — `RuntimeAction::Publish` already exists.
- `route_subscription()` gains a new prefix branch for `harmony/model/` that decodes advertisements and routes to ModelRegistry.

## Testing

### harmony-model unit tests

1. **Manifest round-trip** — encode → decode preserves all fields.
2. **Manifest CID deterministic** — same manifest bytes → same ContentId.
3. **Advertisement round-trip** — encode → decode preserves all fields.
4. **ManifestTooLarge rejected** — manifest exceeding 64KB → error.
5. **Registry RegisterLocal** — stores manifest, emits PublishAdvertisement with correct key.
6. **Registry UnregisterLocal** — removes manifest, emits RetractAdvertisement with empty payload.
7. **Registry AdvertisementReceived** — tracks remote node and ad metadata for manifest CID.
8. **Registry NodeDeparted** — removes node from all entries, prunes empty.
9. **Registry find_by_task** — returns matching models with correct Source (local and remote).
10. **Registry find_by_family** — returns matching models with correct Source.
11. **Registry nodes_for_model** — returns node list for known CID, None for unknown.
12. **Registry Source::Both** — model available locally AND remotely.
13. **Registry remove_advertisement** — `remove_advertisement(cid, node_addr)` removes entry, prunes empty.

### harmony-zenoh namespace tests

14. **advertisement_key format** — correct key structure.
15. **advertisement_sub_all** — returns `harmony/model/**`.
16. **advertisement_sub_model** — correct wildcard pattern.

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-model/Cargo.toml` | New crate: depends on harmony-content, serde, postcard |
| `crates/harmony-model/src/lib.rs` | Public API re-exports |
| `crates/harmony-model/src/manifest.rs` | ModelManifest, ModelFormat, ModelTask |
| `crates/harmony-model/src/wire.rs` | encode/decode manifest & advertisement, ModelError |
| `crates/harmony-model/src/registry.rs` | ModelRegistry state machine |
| `crates/harmony-zenoh/src/namespace.rs` | `pub mod model` with key builders |
| `crates/harmony-node/src/runtime.rs` | Wire ModelRegistry into route_subscription, add model subscription |
| `crates/harmony-node/src/event_loop.rs` | Subscribe to harmony/model/** on startup |
| `Cargo.toml` (workspace) | Add harmony-model to workspace members |

## Dependencies

No new external crates. `postcard` and `serde` are already workspace dependencies. `harmony-content` is the only harmony dependency needed — `manifest_cid()` delegates to `ContentId::for_book()` which handles crypto internally. Direct `harmony-crypto` dependency is not needed.

## What This Does NOT Include

- **Model ingestion CLI/API** — how models get into CAS initially. Any tool that writes books + publishes a manifest works.
- **Automatic model download** — agent-level policy, not registry concern.
- **Model eviction/garbage collection** — deferred.
- **Rich provenance metadata** — deferred to harmony-rijm.
- **Scalable discovery (Bloom filters/queryables)** — deferred to harmony-eo6f.
- **harmony-inference changes** — engine already loads GGUF bytes, no coupling needed.

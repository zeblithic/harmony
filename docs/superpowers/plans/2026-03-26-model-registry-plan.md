# Model Registry and CAS Distribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add structured model metadata, wire encoding, Zenoh discovery namespace, and a local ModelRegistry state machine in a new `harmony-model` crate.

**Architecture:** New `harmony-model` crate follows the `harmony-engram` pattern — postcard-encoded types with `to_bytes()`/`from_bytes()` methods. ModelRegistry is a sans-I/O state machine (Event → State → Actions) following `StorageTier`'s pattern. Zenoh namespace builders follow the `harmony-zenoh::namespace::agent` module pattern.

**Tech Stack:** Rust (edition 2021), postcard (workspace dep), serde (workspace dep), harmony-content (for ContentId)

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-model/Cargo.toml` | Crate manifest — depends on harmony-content, serde, postcard |
| `crates/harmony-model/src/lib.rs` | Public API re-exports |
| `crates/harmony-model/src/manifest.rs` | `ModelManifest`, `ModelFormat`, `ModelTask`, `ModelAdvertisement` types |
| `crates/harmony-model/src/wire.rs` | `encode_manifest`/`decode_manifest`, `encode_advertisement`/`decode_advertisement`, `manifest_cid()`, `ModelError` |
| `crates/harmony-model/src/registry.rs` | `ModelRegistry` state machine, `ModelRegistryEvent`, `ModelRegistryAction`, `Source` |
| `crates/harmony-zenoh/src/namespace.rs` | New `pub mod model` with `PREFIX`, key builders |

**Out of scope for this plan:** NodeRuntime integration (`runtime.rs`, `event_loop.rs`). Those files are large, tightly coupled, and the spec says the wiring is straightforward (`route_subscription` prefix branch + existing `RuntimeAction::Publish`). A follow-up plan should handle integration once the library crate is solid and tested.

---

### Task 1: Crate Scaffold and Types

**Files:**
- Create: `crates/harmony-model/Cargo.toml`
- Create: `crates/harmony-model/src/lib.rs`
- Create: `crates/harmony-model/src/manifest.rs`
- Modify: `Cargo.toml` (workspace root — add to `members` list)

**Context:** Follow the `harmony-engram` crate structure. Types use `#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]`. `ContentId` already implements `Serialize`/`Deserialize` (as `[u8; 32]`), so it works with postcard.

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-model"
version.workspace = true
edition.workspace = true
license.workspace = true
rust-version.workspace = true
repository.workspace = true
description = "Model metadata and registry for Harmony CAS-distributed ML models"

[dependencies]
harmony-content = { path = "../harmony-content" }
harmony-zenoh = { path = "../harmony-zenoh" }
hex.workspace = true
postcard = { workspace = true }
serde = { workspace = true, features = ["derive"] }
```

- [ ] **Step 2: Add to workspace members**

In `Cargo.toml` (workspace root), add `"crates/harmony-model"` to the `members` array. Insert alphabetically after `"crates/harmony-inference"`.

- [ ] **Step 3: Create manifest.rs with types**

```rust
//! Model metadata types for Harmony CAS-distributed ML models.

use harmony_content::ContentId;
use serde::{Deserialize, Serialize};

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
    /// Quantization method (e.g., "Q4_K_M", "F16"). None for full precision.
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

/// Model serialization format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    Gguf,
    Safetensors,
}

/// Task type a model supports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelTask {
    TextGeneration,
    Embedding,
    Vision,
    AudioTranscription,
}

/// Compact advertisement for Zenoh discovery.
///
/// Contains enough metadata for quick filtering without fetching the full
/// manifest from CAS. Published to `harmony/model/{manifest_cid}/{node_addr}`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ModelAdvertisement {
    /// Manifest CID (redundant with key, self-contained for consumers).
    pub manifest_cid: ContentId,
    /// Human-readable model name.
    pub name: String,
    /// Model family identifier.
    pub family: String,
    /// Parameter count.
    pub parameter_count: u64,
    /// Quantization method.
    pub quantization: Option<String>,
    /// Supported tasks.
    pub tasks: Vec<ModelTask>,
    /// Estimated memory required in bytes.
    pub memory_estimate: u64,
}
```

- [ ] **Step 4: Create lib.rs with re-exports**

```rust
pub mod manifest;
pub mod wire;
pub mod registry;

pub use manifest::{ModelAdvertisement, ModelFormat, ModelManifest, ModelTask};
pub use wire::{
    decode_advertisement, decode_manifest, encode_advertisement, encode_manifest, manifest_cid,
    ModelError,
};
pub use registry::{ModelRegistry, ModelRegistryAction, ModelRegistryEvent, Source};
```

Note: `wire` and `registry` modules don't exist yet. Create them as empty placeholder files so the crate compiles:

`wire.rs`:
```rust
//! Wire encoding/decoding for model metadata.

/// Errors from model wire operations.
#[derive(Debug)]
pub enum ModelError {}

pub fn encode_manifest(_manifest: &super::manifest::ModelManifest) -> Result<Vec<u8>, ModelError> { todo!() }
pub fn decode_manifest(_data: &[u8]) -> Result<super::manifest::ModelManifest, ModelError> { todo!() }
pub fn manifest_cid(_data: &[u8]) -> harmony_content::ContentId { todo!() }
pub fn encode_advertisement(_ad: &super::manifest::ModelAdvertisement) -> Result<Vec<u8>, ModelError> { todo!() }
pub fn decode_advertisement(_data: &[u8]) -> Result<super::manifest::ModelAdvertisement, ModelError> { todo!() }
```

`registry.rs`:
```rust
//! Sans-I/O model registry state machine.

/// Tracks local and remote model availability.
pub struct ModelRegistry;

pub enum ModelRegistryEvent {}
pub enum ModelRegistryAction {}

/// Whether a model is local, remote, or both.
pub enum Source {}
```

- [ ] **Step 5: Verify crate compiles**

Run: `cargo check -p harmony-model`
Expected: compiles with no errors (warnings about unused/todo are OK)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-model/ Cargo.toml Cargo.lock
git commit -m "feat(model): scaffold harmony-model crate with manifest types"
```

---

### Task 2: Wire Encoding

**Files:**
- Modify: `crates/harmony-model/src/wire.rs`
- Test: `crates/harmony-model/src/wire.rs` (inline `#[cfg(test)]` module)

**Context:** Follow the `harmony-engram::manifest::ManifestHeader` pattern — `postcard::to_allocvec()` / `postcard::from_bytes()`. Map postcard errors to `ModelError` variants. `manifest_cid()` uses `ContentId::for_book()` with hardcoded public durable flags (all flags false = `ContentFlags::default()`).

**Reference files:**
- `crates/harmony-engram/src/manifest.rs:46-52` — postcard encode/decode pattern
- `crates/harmony-content/src/cid.rs:229-241` — `ContentId::for_book()` signature

- [ ] **Step 1: Write failing tests**

Add to `wire.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::{ModelAdvertisement, ModelFormat, ModelManifest, ModelTask};
    use harmony_content::{ContentFlags, ContentId};

    fn sample_manifest() -> ModelManifest {
        let data_cid = ContentId::for_book(b"model-data", ContentFlags::default()).unwrap();
        ModelManifest {
            name: "Qwen3-0.6B-Q4_K_M".into(),
            family: "qwen3".into(),
            parameter_count: 600_000_000,
            format: ModelFormat::Gguf,
            quantization: Some("Q4_K_M".into()),
            context_length: 32768,
            vocab_size: 151936,
            memory_estimate: 512_000_000,
            tasks: vec![ModelTask::TextGeneration],
            data_cid,
            tokenizer_cid: None,
        }
    }

    fn sample_advertisement() -> ModelAdvertisement {
        let manifest_cid = ContentId::for_book(b"manifest-bytes", ContentFlags::default()).unwrap();
        ModelAdvertisement {
            manifest_cid,
            name: "Qwen3-0.6B-Q4_K_M".into(),
            family: "qwen3".into(),
            parameter_count: 600_000_000,
            quantization: Some("Q4_K_M".into()),
            tasks: vec![ModelTask::TextGeneration],
            memory_estimate: 512_000_000,
        }
    }

    #[test]
    fn manifest_round_trip() {
        let manifest = sample_manifest();
        let encoded = encode_manifest(&manifest).unwrap();
        let decoded = decode_manifest(&encoded).unwrap();
        assert_eq!(manifest, decoded);
    }

    #[test]
    fn manifest_cid_deterministic() {
        let manifest = sample_manifest();
        let encoded = encode_manifest(&manifest).unwrap();
        let cid1 = manifest_cid(&encoded);
        let cid2 = manifest_cid(&encoded);
        assert_eq!(cid1, cid2);
    }

    #[test]
    fn advertisement_round_trip() {
        let ad = sample_advertisement();
        let encoded = encode_advertisement(&ad).unwrap();
        let decoded = decode_advertisement(&encoded).unwrap();
        assert_eq!(ad, decoded);
    }

    #[test]
    fn manifest_too_large_rejected() {
        // Create a manifest with a huge name to exceed 64KB.
        let mut manifest = sample_manifest();
        manifest.name = "x".repeat(70_000);
        let result = encode_manifest(&manifest);
        assert!(matches!(result, Err(ModelError::ManifestTooLarge)));
    }

    #[test]
    fn decode_manifest_empty_fails() {
        let result = decode_manifest(&[]);
        assert!(matches!(result, Err(ModelError::DecodeFailed)));
    }

    #[test]
    fn decode_advertisement_empty_fails() {
        let result = decode_advertisement(&[]);
        assert!(matches!(result, Err(ModelError::DecodeFailed)));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-model -- wire`
Expected: FAIL (all tests hit `todo!()` panics)

- [ ] **Step 3: Implement wire.rs**

Replace the placeholder `wire.rs` with the full implementation:

```rust
//! Wire encoding/decoding for model metadata.
//!
//! Uses postcard for compact binary serialization (matches harmony-engram convention).

use harmony_content::{ContentFlags, ContentId};

use crate::manifest::{ModelAdvertisement, ModelManifest};

/// Maximum encoded manifest size (safety check).
const MAX_MANIFEST_SIZE: usize = 64 * 1024;

/// Errors from model wire operations.
#[derive(Debug)]
pub enum ModelError {
    /// Postcard serialization failed.
    EncodeFailed,
    /// Postcard deserialization failed.
    DecodeFailed,
    /// Encoded manifest exceeds 64KB safety limit.
    ManifestTooLarge,
}

impl core::fmt::Display for ModelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EncodeFailed => write!(f, "manifest encode failed"),
            Self::DecodeFailed => write!(f, "manifest decode failed"),
            Self::ManifestTooLarge => write!(f, "encoded manifest exceeds 64KB limit"),
        }
    }
}

impl std::error::Error for ModelError {}

/// Serialize a [`ModelManifest`] to postcard bytes.
///
/// Returns [`ModelError::ManifestTooLarge`] if the encoded size exceeds 64KB.
pub fn encode_manifest(manifest: &ModelManifest) -> Result<Vec<u8>, ModelError> {
    let bytes = postcard::to_allocvec(manifest).map_err(|_| ModelError::EncodeFailed)?;
    if bytes.len() > MAX_MANIFEST_SIZE {
        return Err(ModelError::ManifestTooLarge);
    }
    Ok(bytes)
}

/// Deserialize a [`ModelManifest`] from postcard bytes.
pub fn decode_manifest(data: &[u8]) -> Result<ModelManifest, ModelError> {
    postcard::from_bytes(data).map_err(|_| ModelError::DecodeFailed)
}

/// Compute the [`ContentId`] for an encoded manifest.
///
/// Manifests are always public durable (flags `00`).
pub fn manifest_cid(data: &[u8]) -> ContentId {
    ContentId::for_book(data, ContentFlags::default())
        .expect("encoded manifest should be within book size limit")
}

/// Serialize a [`ModelAdvertisement`] to postcard bytes.
pub fn encode_advertisement(ad: &ModelAdvertisement) -> Result<Vec<u8>, ModelError> {
    postcard::to_allocvec(ad).map_err(|_| ModelError::EncodeFailed)
}

/// Deserialize a [`ModelAdvertisement`] from postcard bytes.
pub fn decode_advertisement(data: &[u8]) -> Result<ModelAdvertisement, ModelError> {
    postcard::from_bytes(data).map_err(|_| ModelError::DecodeFailed)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-model -- wire`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-model/src/wire.rs
git commit -m "feat(model): wire encoding with postcard for manifest and advertisement"
```

---

### Task 3: Zenoh Namespace Builders

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs` (add `pub mod model`)
- Test: inline `#[cfg(test)]` within the new module

**Context:** Follow the `pub mod agent` pattern exactly — `PREFIX` constant, `format!` builders returning `String`. The key structure is `harmony/model/{manifest_cid_hex}/{node_addr_hex}`. CID hex is the full 64-char lowercase hex of `ContentId::to_bytes()`. Node addr hex is the 32-char lowercase hex of the 16-byte address.

**Reference:** `crates/harmony-zenoh/src/namespace.rs:684-725` — the `agent` module pattern.

- [ ] **Step 1: Write failing tests**

Add a `pub mod model` block at the end of `namespace.rs` (before the closing of the file), with tests:

```rust
pub mod model {
    use alloc::{format, string::String};

    /// Base prefix for model advertisements.
    pub const PREFIX: &str = "harmony/model";

    /// Full key for a model advertisement: `harmony/model/{manifest_cid_hex}/{node_addr_hex}`.
    pub fn advertisement_key(manifest_cid_hex: &str, node_addr_hex: &str) -> String {
        format!("{PREFIX}/{manifest_cid_hex}/{node_addr_hex}")
    }

    /// Subscribe to all model advertisements: `harmony/model/**`.
    pub fn advertisement_sub_all() -> &'static str {
        "harmony/model/**"
    }

    /// Subscribe to all nodes advertising a specific model: `harmony/model/{manifest_cid_hex}/*`.
    pub fn advertisement_sub_model(manifest_cid_hex: &str) -> String {
        format!("{PREFIX}/{manifest_cid_hex}/*")
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn advertisement_key_format() {
            let key = advertisement_key("abcd1234", "deadbeef");
            assert_eq!(key, "harmony/model/abcd1234/deadbeef");
        }

        #[test]
        fn advertisement_sub_all_pattern() {
            assert_eq!(advertisement_sub_all(), "harmony/model/**");
        }

        #[test]
        fn advertisement_sub_model_pattern() {
            let pattern = advertisement_sub_model("abcd1234");
            assert_eq!(pattern, "harmony/model/abcd1234/*");
        }
    }
}
```

Note: The functions take pre-formatted hex strings (not raw ContentId/bytes). This keeps the namespace module dependency-free (it doesn't depend on harmony-content). The caller (ModelRegistry) is responsible for hex-encoding.

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh -- model`
Expected: all 3 tests PASS (these are implementation + test in one step since the module is simple)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add model namespace builders for advertisement discovery"
```

---

### Task 4: ModelRegistry State Machine

**Files:**
- Modify: `crates/harmony-model/src/registry.rs`
- Modify: `crates/harmony-model/Cargo.toml` (add `hex` dependency)
- Test: `crates/harmony-model/src/registry.rs` (inline `#[cfg(test)]` module)

**Context:** Sans-I/O state machine following `StorageTier` pattern: `handle_event(event) -> Vec<Action>`. The registry tracks local models (full manifests) and remote models (advertisements keyed by node address). Query methods are pure — no actions emitted.

**Reference:**
- `crates/harmony-content/src/storage_tier.rs` — Event/Action pattern
- `crates/harmony-zenoh/src/namespace.rs` — `model::advertisement_key()` for key construction

- [ ] **Step 1: Write failing tests**

Replace the placeholder `registry.rs` with tests first (implementation bodies as `todo!()`):

```rust
//! Sans-I/O model registry state machine.
//!
//! Tracks local and remote model availability. Emits Zenoh publish actions
//! when local models are registered or unregistered. Receives remote
//! advertisements and answers queries about model availability.

use std::collections::{HashMap, HashSet};

use harmony_content::ContentId;

use crate::manifest::{ModelAdvertisement, ModelFormat, ModelManifest, ModelTask};
use crate::wire;

/// Tracks local and remote model availability.
pub struct ModelRegistry {
    /// Models this node has locally (manifest CID → manifest).
    local_models: HashMap<ContentId, ModelManifest>,
    /// Remote advertisements (manifest CID → node address → advertisement).
    remote_models: HashMap<ContentId, HashMap<[u8; 16], ModelAdvertisement>>,
    /// This node's address (for constructing Zenoh keys).
    local_addr: [u8; 16],
}

/// Inbound events for the registry.
pub enum ModelRegistryEvent {
    /// A local model was registered (e.g., after ingesting a GGUF file).
    RegisterLocal {
        manifest_cid: ContentId,
        manifest: ModelManifest,
    },
    /// A local model was removed.
    UnregisterLocal { manifest_cid: ContentId },
    /// A remote advertisement was received via Zenoh subscription.
    AdvertisementReceived {
        manifest_cid: ContentId,
        node_addr: [u8; 16],
        ad: ModelAdvertisement,
    },
    /// A remote node departed (all its advertisements should be removed).
    NodeDeparted { node_addr: [u8; 16] },
}

/// Outbound actions from the registry.
#[derive(Debug, PartialEq, Eq)]
pub enum ModelRegistryAction {
    /// Publish advertisement to Zenoh (maps to RuntimeAction::Publish).
    PublishAdvertisement { key_expr: String, payload: Vec<u8> },
    /// Retract advertisement: empty-payload publish (tombstone).
    RetractAdvertisement { key_expr: String },
}

/// Whether a model is available locally, remotely, or both.
#[derive(Debug, PartialEq, Eq)]
pub enum Source {
    Local,
    Remote(Vec<[u8; 16]>),
    Both(Vec<[u8; 16]>),
}

impl ModelRegistry {
    /// Create a new registry for the given node address.
    pub fn new(local_addr: [u8; 16]) -> Self {
        todo!()
    }

    /// Process an event and return any resulting actions.
    pub fn handle_event(&mut self, event: ModelRegistryEvent) -> Vec<ModelRegistryAction> {
        todo!()
    }

    /// Remove a specific node's advertisement for a model.
    ///
    /// Called by `route_subscription` when an empty-payload tombstone is received.
    pub fn remove_advertisement(&mut self, manifest_cid: &ContentId, node_addr: &[u8; 16]) {
        todo!()
    }

    /// All locally available models.
    pub fn local_models(&self) -> &HashMap<ContentId, ModelManifest> {
        todo!()
    }

    /// Find models matching a task type (searches both local and remote).
    pub fn find_by_task(&self, task: ModelTask) -> Vec<(ContentId, Source)> {
        todo!()
    }

    /// Find models by family name (searches both local and remote).
    pub fn find_by_family(&self, family: &str) -> Vec<(ContentId, Source)> {
        todo!()
    }

    /// Which nodes have a specific model? Returns None if unknown.
    pub fn nodes_for_model(&self, manifest_cid: &ContentId) -> Option<Vec<[u8; 16]>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    const LOCAL_ADDR: [u8; 16] = [0xAA; 16];
    const REMOTE_ADDR_1: [u8; 16] = [0xBB; 16];
    const REMOTE_ADDR_2: [u8; 16] = [0xCC; 16];

    fn make_manifest(name: &str, family: &str, tasks: Vec<ModelTask>) -> (ContentId, ModelManifest) {
        let data_cid = ContentId::for_book(name.as_bytes(), ContentFlags::default()).unwrap();
        let manifest = ModelManifest {
            name: name.into(),
            family: family.into(),
            parameter_count: 600_000_000,
            format: ModelFormat::Gguf,
            quantization: Some("Q4_K_M".into()),
            context_length: 32768,
            vocab_size: 151936,
            memory_estimate: 512_000_000,
            tasks,
            data_cid,
            tokenizer_cid: None,
        };
        let encoded = wire::encode_manifest(&manifest).unwrap();
        let cid = wire::manifest_cid(&encoded);
        (cid, manifest)
    }

    fn make_advertisement(manifest_cid: ContentId, manifest: &ModelManifest) -> ModelAdvertisement {
        ModelAdvertisement {
            manifest_cid,
            name: manifest.name.clone(),
            family: manifest.family.clone(),
            parameter_count: manifest.parameter_count,
            quantization: manifest.quantization.clone(),
            tasks: manifest.tasks.clone(),
            memory_estimate: manifest.memory_estimate,
        }
    }

    #[test]
    fn register_local_emits_publish() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("test-model", "qwen3", vec![ModelTask::TextGeneration]);
        let actions = reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest: manifest.clone(),
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ModelRegistryAction::PublishAdvertisement { key_expr, payload } => {
                assert!(key_expr.starts_with("harmony/model/"));
                assert!(!payload.is_empty());
                // Verify the payload decodes to a valid advertisement.
                let ad = wire::decode_advertisement(payload).unwrap();
                assert_eq!(ad.name, "test-model");
                assert_eq!(ad.manifest_cid, cid);
            }
            other => panic!("expected PublishAdvertisement, got {:?}", other),
        }
        // Model is now in local_models.
        assert!(reg.local_models().contains_key(&cid));
    }

    #[test]
    fn unregister_local_emits_retract() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("test-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid,
            manifest,
        });
        let actions = reg.handle_event(ModelRegistryEvent::UnregisterLocal { manifest_cid: cid });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ModelRegistryAction::RetractAdvertisement { key_expr } => {
                assert!(key_expr.starts_with("harmony/model/"));
            }
            other => panic!("expected RetractAdvertisement, got {:?}", other),
        }
        assert!(!reg.local_models().contains_key(&cid));
    }

    #[test]
    fn advertisement_received_tracks_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("remote-model", "llama", vec![ModelTask::TextGeneration]);
        let ad = make_advertisement(cid, &manifest);
        let actions = reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid,
            node_addr: REMOTE_ADDR_1,
            ad,
        });
        assert!(actions.is_empty(), "receiving an ad should not emit actions");
        let nodes = reg.nodes_for_model(&cid).unwrap();
        assert_eq!(nodes, vec![REMOTE_ADDR_1]);
    }

    #[test]
    fn node_departed_removes_all_entries() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid1, m1) = make_manifest("model-a", "qwen3", vec![ModelTask::TextGeneration]);
        let (cid2, m2) = make_manifest("model-b", "llama", vec![ModelTask::Embedding]);
        let ad1 = make_advertisement(cid1, &m1);
        let ad2 = make_advertisement(cid2, &m2);

        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid1, node_addr: REMOTE_ADDR_1, ad: ad1,
        });
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid2, node_addr: REMOTE_ADDR_1, ad: ad2,
        });

        let actions = reg.handle_event(ModelRegistryEvent::NodeDeparted {
            node_addr: REMOTE_ADDR_1,
        });
        assert!(actions.is_empty());
        assert!(reg.nodes_for_model(&cid1).is_none());
        assert!(reg.nodes_for_model(&cid2).is_none());
    }

    #[test]
    fn find_by_task_local_and_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid_local, manifest_local) = make_manifest("local-gen", "qwen3", vec![ModelTask::TextGeneration]);
        let (cid_remote, manifest_remote) = make_manifest("remote-embed", "bge", vec![ModelTask::Embedding]);

        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid_local, manifest: manifest_local,
        });
        let ad = make_advertisement(cid_remote, &manifest_remote);
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid_remote, node_addr: REMOTE_ADDR_1, ad,
        });

        let text_gen = reg.find_by_task(ModelTask::TextGeneration);
        assert_eq!(text_gen.len(), 1);
        assert_eq!(text_gen[0].0, cid_local);
        assert_eq!(text_gen[0].1, Source::Local);

        let embed = reg.find_by_task(ModelTask::Embedding);
        assert_eq!(embed.len(), 1);
        assert_eq!(embed[0].0, cid_remote);
        assert!(matches!(embed[0].1, Source::Remote(_)));
    }

    #[test]
    fn find_by_family() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("qwen-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid, manifest,
        });
        let results = reg.find_by_family("qwen3");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, cid);

        let no_results = reg.find_by_family("llama");
        assert!(no_results.is_empty());
    }

    #[test]
    fn source_both_when_local_and_remote() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("shared-model", "qwen3", vec![ModelTask::TextGeneration]);
        reg.handle_event(ModelRegistryEvent::RegisterLocal {
            manifest_cid: cid, manifest: manifest.clone(),
        });
        let ad = make_advertisement(cid, &manifest);
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid, node_addr: REMOTE_ADDR_1, ad,
        });

        let results = reg.find_by_task(ModelTask::TextGeneration);
        assert_eq!(results.len(), 1);
        assert!(matches!(results[0].1, Source::Both(_)));
    }

    #[test]
    fn remove_advertisement_specific_node() {
        let mut reg = ModelRegistry::new(LOCAL_ADDR);
        let (cid, manifest) = make_manifest("multi-node", "llama", vec![ModelTask::TextGeneration]);
        let ad1 = make_advertisement(cid, &manifest);
        let ad2 = make_advertisement(cid, &manifest);

        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid, node_addr: REMOTE_ADDR_1, ad: ad1,
        });
        reg.handle_event(ModelRegistryEvent::AdvertisementReceived {
            manifest_cid: cid, node_addr: REMOTE_ADDR_2, ad: ad2,
        });

        // Remove only node 1.
        reg.remove_advertisement(&cid, &REMOTE_ADDR_1);
        let nodes = reg.nodes_for_model(&cid).unwrap();
        assert_eq!(nodes, vec![REMOTE_ADDR_2]);

        // Remove node 2 — entry should be pruned entirely.
        reg.remove_advertisement(&cid, &REMOTE_ADDR_2);
        assert!(reg.nodes_for_model(&cid).is_none());
    }

    #[test]
    fn nodes_for_unknown_model_returns_none() {
        let reg = ModelRegistry::new(LOCAL_ADDR);
        let fake_cid = ContentId::for_book(b"nonexistent", ContentFlags::default()).unwrap();
        assert!(reg.nodes_for_model(&fake_cid).is_none());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-model -- registry`
Expected: FAIL (all tests hit `todo!()` panics)

- [ ] **Step 3: Implement registry.rs**

Replace all `todo!()` bodies with implementations:

```rust
impl ModelRegistry {
    pub fn new(local_addr: [u8; 16]) -> Self {
        Self {
            local_models: HashMap::new(),
            remote_models: HashMap::new(),
            local_addr,
        }
    }

    pub fn handle_event(&mut self, event: ModelRegistryEvent) -> Vec<ModelRegistryAction> {
        match event {
            ModelRegistryEvent::RegisterLocal { manifest_cid, manifest } => {
                let ad = ModelAdvertisement {
                    manifest_cid,
                    name: manifest.name.clone(),
                    family: manifest.family.clone(),
                    parameter_count: manifest.parameter_count,
                    quantization: manifest.quantization.clone(),
                    tasks: manifest.tasks.clone(),
                    memory_estimate: manifest.memory_estimate,
                };
                self.local_models.insert(manifest_cid, manifest);
                let key_expr = harmony_zenoh::namespace::model::advertisement_key(
                    &hex::encode(manifest_cid.to_bytes()),
                    &hex::encode(self.local_addr),
                );
                match wire::encode_advertisement(&ad) {
                    Ok(payload) => vec![ModelRegistryAction::PublishAdvertisement { key_expr, payload }],
                    Err(_) => vec![],
                }
            }
            ModelRegistryEvent::UnregisterLocal { manifest_cid } => {
                self.local_models.remove(&manifest_cid);
                let key_expr = harmony_zenoh::namespace::model::advertisement_key(
                    &hex::encode(manifest_cid.to_bytes()),
                    &hex::encode(self.local_addr),
                );
                vec![ModelRegistryAction::RetractAdvertisement { key_expr }]
            }
            ModelRegistryEvent::AdvertisementReceived { manifest_cid, node_addr, ad } => {
                self.remote_models
                    .entry(manifest_cid)
                    .or_default()
                    .insert(node_addr, ad);
                vec![]
            }
            ModelRegistryEvent::NodeDeparted { node_addr } => {
                self.remote_models.retain(|_, nodes| {
                    nodes.remove(&node_addr);
                    !nodes.is_empty()
                });
                vec![]
            }
        }
    }

    pub fn remove_advertisement(&mut self, manifest_cid: &ContentId, node_addr: &[u8; 16]) {
        if let Some(nodes) = self.remote_models.get_mut(manifest_cid) {
            nodes.remove(node_addr);
            if nodes.is_empty() {
                self.remote_models.remove(manifest_cid);
            }
        }
    }

    pub fn local_models(&self) -> &HashMap<ContentId, ModelManifest> {
        &self.local_models
    }

    pub fn find_by_task(&self, task: ModelTask) -> Vec<(ContentId, Source)> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        // Local models matching the task.
        for (cid, manifest) in &self.local_models {
            if manifest.tasks.contains(&task) {
                let remote_nodes: Vec<[u8; 16]> = self.remote_models
                    .get(cid)
                    .map(|nodes| nodes.keys().copied().collect())
                    .unwrap_or_default();
                let source = if remote_nodes.is_empty() {
                    Source::Local
                } else {
                    Source::Both(remote_nodes)
                };
                results.push((*cid, source));
                seen.insert(*cid);
            }
        }

        // Remote-only models matching the task.
        for (cid, nodes) in &self.remote_models {
            if seen.contains(cid) {
                continue;
            }
            let any_match = nodes.values().any(|ad| ad.tasks.contains(&task));
            if any_match {
                let node_addrs: Vec<[u8; 16]> = nodes.keys().copied().collect();
                results.push((*cid, Source::Remote(node_addrs)));
            }
        }

        results
    }

    pub fn find_by_family(&self, family: &str) -> Vec<(ContentId, Source)> {
        let mut results = Vec::new();
        let mut seen = HashSet::new();

        for (cid, manifest) in &self.local_models {
            if manifest.family == family {
                let remote_nodes: Vec<[u8; 16]> = self.remote_models
                    .get(cid)
                    .map(|nodes| nodes.keys().copied().collect())
                    .unwrap_or_default();
                let source = if remote_nodes.is_empty() {
                    Source::Local
                } else {
                    Source::Both(remote_nodes)
                };
                results.push((*cid, source));
                seen.insert(*cid);
            }
        }

        for (cid, nodes) in &self.remote_models {
            if seen.contains(cid) {
                continue;
            }
            let any_match = nodes.values().any(|ad| ad.family == family);
            if any_match {
                let node_addrs: Vec<[u8; 16]> = nodes.keys().copied().collect();
                results.push((*cid, Source::Remote(node_addrs)));
            }
        }

        results
    }

    pub fn nodes_for_model(&self, manifest_cid: &ContentId) -> Option<Vec<[u8; 16]>> {
        self.remote_models.get(manifest_cid).map(|nodes| {
            nodes.keys().copied().collect()
        })
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-model -- registry`
Expected: all 9 tests PASS

- [ ] **Step 5: Run full crate tests**

Run: `cargo test -p harmony-model`
Expected: all 15 tests PASS (6 wire + 9 registry)

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p harmony-model`
Expected: no warnings

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-model/src/registry.rs
git commit -m "feat(model): ModelRegistry state machine with event/action pattern"
```

---

### Task 5: Final Integration Verification

**Files:**
- Modify: `crates/harmony-model/src/lib.rs` (verify re-exports match actual public API)

- [ ] **Step 1: Verify lib.rs re-exports compile**

The `lib.rs` from Task 1 already has the re-exports. Verify they match the actual public types:

Run: `cargo doc -p harmony-model --no-deps`
Expected: docs build with no warnings, all public types documented

- [ ] **Step 2: Run format check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues

- [ ] **Step 3: Run full workspace check**

Run: `cargo check --workspace`
Expected: entire workspace compiles (harmony-model doesn't break anything)

- [ ] **Step 4: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass (existing + new)

- [ ] **Step 5: Commit any fixups**

If Steps 1-4 revealed issues, fix and commit:

```bash
git add -u
git commit -m "fix(model): address integration issues from workspace check"
```

If no issues, skip this step.

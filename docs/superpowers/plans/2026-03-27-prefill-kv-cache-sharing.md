# Prefill KV Cache Sharing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable nodes to serialize compressed KV caches, store them in CAS, and let other nodes fetch + resume generation without re-evaluating the prompt.

**Architecture:** Two crate changes: (1) harmony-inference gains serde derives on CompressedKvLayer/QuantizedVec and serialize/deserialize methods on InferenceCache (behind `kv-compress`), (2) harmony-speculative gains a `prefill` feature with PrefillCacheHeader, store/load functions using harmony-content's Merkle DAG, and SHA-256 token hashing via harmony-crypto.

**Tech Stack:** Rust, `postcard` (compact serialization), `serde`, harmony-content (CAS/DAG), harmony-crypto (SHA-256)

**Spec:** `docs/superpowers/specs/2026-03-27-prefill-kv-cache-sharing-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/harmony-inference/Cargo.toml` | Modify | Add serde + postcard as optional deps behind kv-compress |
| `crates/harmony-inference/src/kv_compress.rs` | Modify | Make types pub, add serde derives |
| `crates/harmony-inference/src/lib.rs` | Modify | Add serialize_compressed/deserialize_compressed, SerializationFailed error |
| `crates/harmony-inference/src/error.rs` | Modify | Add SerializationFailed variant |
| `crates/harmony-speculative/Cargo.toml` | Modify | Add prefill feature + deps |
| `crates/harmony-speculative/src/lib.rs` | Modify | Add prefill module declaration |
| `crates/harmony-speculative/src/prefill.rs` | Create | PrefillCacheHeader, PrefillError, store/load/token_hash functions |

---

### Task 1: Add serde + postcard deps to harmony-inference

**Files:**
- Modify: `crates/harmony-inference/Cargo.toml`

- [ ] **Step 1: Update Cargo.toml**

Add `serde` and `postcard` as optional dependencies gated behind `kv-compress`:

```toml
[features]
default = []
cuda = ["candle-core/cuda", "candle-nn/cuda"]
metal = ["candle-core/metal", "candle-nn/metal"]
kv-compress = ["dep:serde", "dep:postcard"]

[dependencies]
candle-core = { workspace = true }
candle-nn = { workspace = true }
tokenizers = { workspace = true }
rand = { workspace = true }
thiserror = { workspace = true, features = ["std"] }
tracing = { workspace = true }
serde = { workspace = true, features = ["derive", "alloc"], optional = true }
postcard = { workspace = true, optional = true }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-inference --features kv-compress`
Expected: OK

Run: `cargo check -p harmony-inference`
Expected: OK (serde/postcard not compiled without feature)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-inference/Cargo.toml
git commit -m "feat(inference): add serde + postcard deps behind kv-compress feature"
```

---

### Task 2: Make kv_compress types public with serde derives

**Files:**
- Modify: `crates/harmony-inference/src/kv_compress.rs`

- [ ] **Step 1: Write failing test**

Add to the existing `kv_compress::tests` module:

```rust
    #[test]
    fn quantized_vec_serde_roundtrip() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        let bytes = postcard::to_allocvec(&qv).unwrap();
        let restored: QuantizedVec = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(qv.min, restored.min);
        assert_eq!(qv.scale, restored.scale);
        assert_eq!(qv.packed, restored.packed);
    }

    #[test]
    fn compressed_kv_layer_serde_roundtrip() {
        let data = vec![1.0_f32; 128];
        let qv = quantize_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 4],
            v: vec![qv; 4],
            seq_len: 2,
        };
        let bytes = postcard::to_allocvec(&layer).unwrap();
        let restored: CompressedKvLayer = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(layer.seq_len, restored.seq_len);
        assert_eq!(layer.k.len(), restored.k.len());
        assert_eq!(layer.v.len(), restored.v.len());
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress::tests::quantized_vec_serde`
Expected: FAIL — serde derives missing

- [ ] **Step 3: Add serde derives and make types public**

In `crates/harmony-inference/src/kv_compress.rs`:

Add `use serde::{Serialize, Deserialize};` to the imports.

Change `QuantizedVec`:
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedVec {
    pub min: f32,
    pub scale: f32,
    pub packed: Vec<u8>,
}
```

Change `CompressedKvLayer`:
```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    pub k: Vec<QuantizedVec>,
    pub v: Vec<QuantizedVec>,
    pub seq_len: usize,
}
```

Note: `quantize_vec` and `dequantize_vec` remain private — they're internal helpers. The types are public for cross-crate serialization.

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress -- kv_compress::tests`
Expected: ALL PASS (existing 9 + 2 new)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/kv_compress.rs
git commit -m "feat(inference): make kv_compress types pub with serde derives

QuantizedVec and CompressedKvLayer gain Serialize/Deserialize derives
and pub visibility for cross-crate serialization."
```

---

### Task 3: SerializationFailed error variant + serialize/deserialize methods

**Files:**
- Modify: `crates/harmony-inference/src/error.rs`
- Modify: `crates/harmony-inference/src/lib.rs`

- [ ] **Step 1: Write failing tests**

Add `SerializationFailed` test to `error.rs` tests module (which is `#[cfg(feature = "kv-compress")]`):

```rust
    #[test]
    fn serialization_failed_displays_message() {
        let err = InferenceError::SerializationFailed("bad data".into());
        assert_eq!(err.to_string(), "serialization failed: bad data");
    }
```

Add serialization tests to the `kv_compress_cache_tests` module in `lib.rs`:

```rust
    #[test]
    fn serialize_deserialize_roundtrip() {
        let mut cache = cache_with_data(2, 8, 128, 16);
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();

        assert!(restored.is_compressed());
        assert_eq!(restored.position, 16);
        assert_eq!(restored.num_layers, 2);
        assert!(restored.compressed[0].is_some());
        assert!(restored.compressed[1].is_none());
    }

    #[test]
    fn serialize_uncompressed_errors() {
        let cache = cache_with_data(2, 8, 128, 16);
        let result = cache.serialize_compressed();
        assert!(matches!(result, Err(InferenceError::SerializationFailed(_))));
    }

    #[test]
    fn deserialize_validates_num_layers() {
        let mut cache = cache_with_data(2, 8, 128, 4);
        cache.compress().unwrap();
        let bytes = cache.serialize_compressed().unwrap();

        // Wrong num_layers
        let result = InferenceCache::deserialize_compressed(&bytes, 99, 128, 8);
        assert!(matches!(result, Err(InferenceError::SerializationFailed(_))));
    }

    #[test]
    fn serialize_empty_compressed_cache() {
        let mut cache = InferenceCache::new(2, 128, 8);
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();

        assert!(restored.is_compressed());
        assert_eq!(restored.position, 0);
        assert!(restored.compressed.iter().all(|c| c.is_none()));
    }

    #[test]
    fn serialize_preserves_position() {
        let mut cache = cache_with_data(2, 8, 128, 42);
        cache.position = 42;
        cache.compress().unwrap();

        let bytes = cache.serialize_compressed().unwrap();
        let restored = InferenceCache::deserialize_compressed(&bytes, 2, 128, 8).unwrap();
        assert_eq!(restored.position, 42);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-inference --features kv-compress -- serialization`
Expected: FAIL — methods don't exist

- [ ] **Step 3: Add SerializationFailed error variant**

In `crates/harmony-inference/src/error.rs`, add before the closing `}`:

```rust
    /// Serialization or deserialization of compressed cache failed.
    #[cfg(feature = "kv-compress")]
    #[error("serialization failed: {0}")]
    SerializationFailed(String),
```

- [ ] **Step 4: Implement serialize/deserialize on InferenceCache**

In `crates/harmony-inference/src/lib.rs`, add a private serialization payload type at the top of the `#[cfg(feature = "kv-compress")]` impl block:

```rust
#[cfg(feature = "kv-compress")]
#[derive(serde::Serialize, serde::Deserialize)]
struct CompressedCachePayload {
    position: usize,
    layers: Vec<Option<kv_compress::CompressedKvLayer>>,
}
```

Then add methods to the `#[cfg(feature = "kv-compress")] impl InferenceCache` block:

```rust
    /// Serialize the compressed cache to bytes.
    /// Returns Err if the cache is not compressed.
    pub fn serialize_compressed(&self) -> Result<Vec<u8>, InferenceError> {
        if !self.is_compressed {
            return Err(InferenceError::SerializationFailed(
                "cache is not compressed — call compress() first".into(),
            ));
        }
        let payload = CompressedCachePayload {
            position: self.position,
            layers: self.compressed.clone(),
        };
        postcard::to_allocvec(&payload)
            .map_err(|e| InferenceError::SerializationFailed(e.to_string()))
    }

    /// Deserialize a compressed cache from bytes.
    /// Validates that the serialized layer count matches `num_layers`.
    /// `head_dim` and `num_kv_heads` are trusted inputs (from PrefillCacheHeader).
    pub fn deserialize_compressed(
        data: &[u8],
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Result<InferenceCache, InferenceError> {
        let payload: CompressedCachePayload = postcard::from_bytes(data)
            .map_err(|e| InferenceError::SerializationFailed(e.to_string()))?;

        if payload.layers.len() != num_layers {
            return Err(InferenceError::SerializationFailed(format!(
                "layer count mismatch: expected {num_layers}, got {}",
                payload.layers.len()
            )));
        }

        Ok(InferenceCache {
            layers: (0..num_layers).map(|_| None).collect(),
            position: payload.position,
            num_layers,
            head_dim,
            num_kv_heads,
            compressed: payload.layers,
            is_compressed: true,
        })
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-inference --features kv-compress`
Expected: ALL PASS

Run: `cargo test -p harmony-inference`
Expected: ALL PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-inference/src/error.rs crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): add serialize/deserialize for compressed KV cache

serialize_compressed() postcard-encodes the compressed layers + position.
deserialize_compressed() validates layer count and constructs a cache
in compressed state. Behind kv-compress feature."
```

---

### Task 4: prefill feature flag + deps on harmony-speculative

**Files:**
- Modify: `crates/harmony-speculative/Cargo.toml`
- Modify: `Cargo.toml` (workspace root — add postcard to harmony-speculative's workspace dep if needed)

- [ ] **Step 1: Update Cargo.toml**

In `crates/harmony-speculative/Cargo.toml`:

```toml
[package]
name = "harmony-speculative"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Decentralized Speculative Decoding protocol for Harmony"

[features]
default = []
prefill = ["dep:harmony-inference", "dep:harmony-content", "dep:harmony-crypto", "dep:serde", "dep:postcard", "dep:thiserror"]

[dependencies]
harmony-inference = { workspace = true, features = ["kv-compress"], optional = true }
harmony-content = { workspace = true, optional = true }
harmony-crypto = { workspace = true, optional = true }
serde = { workspace = true, features = ["derive", "alloc"], optional = true }
postcard = { workspace = true, optional = true }
thiserror = { workspace = true, optional = true }

[dev-dependencies]
candle-core = { workspace = true }
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-speculative`
Expected: OK (no prefill deps)

Run: `cargo check -p harmony-speculative --features prefill`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-speculative/Cargo.toml
git commit -m "feat(speculative): add prefill feature flag with inference/content/crypto deps"
```

---

### Task 5: PrefillCacheHeader and PrefillError types

**Files:**
- Create: `crates/harmony-speculative/src/prefill.rs`
- Modify: `crates/harmony-speculative/src/lib.rs`

- [ ] **Step 1: Create prefill.rs with types and token_hash**

Create `crates/harmony-speculative/src/prefill.rs`:

```rust
//! Prefill KV cache sharing protocol.
//!
//! Enables powerful nodes to serialize compressed KV caches into CAS,
//! so edge nodes can fetch and resume generation without re-evaluating
//! the prompt. Feature-gated behind `prefill`.

use harmony_content::book::BookStore;
use harmony_content::cid::ContentId;
use harmony_content::chunker::ChunkerConfig;
use harmony_content::dag;
use harmony_content::error::ContentError;
use harmony_crypto::hash::full_hash;
use harmony_inference::{InferenceCache, InferenceError};
use serde::{Deserialize, Serialize};

/// Magic bytes identifying a prefill KV cache blob.
pub const PREFILL_MAGIC: [u8; 4] = *b"HKV\x01";

/// Supported quantization bit width.
const SUPPORTED_QUANT_BITS: u8 = 3;

/// Header prepended to the serialized KV cache payload in CAS.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefillCacheHeader {
    /// Magic bytes + format version.
    pub magic: [u8; 4],
    /// CAS CID of the GGUF model that produced this cache.
    pub model_cid: [u8; 32],
    /// SHA-256 of the token IDs (as contiguous little-endian u32 bytes).
    pub token_hash: [u8; 32],
    /// Number of tokens in the prefill (= cache.position).
    pub token_count: u32,
    /// Architecture params for validation.
    pub num_layers: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,
    /// Compression config.
    pub quant_bits: u8,
}

/// Errors from prefill cache operations.
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("invalid magic: expected HKV\\x01")]
    InvalidMagic,

    #[error("unsupported quant_bits: {0}")]
    UnsupportedQuantBits(u8),

    #[error("model mismatch: expected {expected:?}, got {actual:?}")]
    ModelMismatch { expected: [u8; 32], actual: [u8; 32] },

    #[error("cache is not compressed")]
    CacheNotCompressed,

    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    #[error("storage failed: {0}")]
    StorageFailed(String),
}

impl From<InferenceError> for PrefillError {
    fn from(e: InferenceError) -> Self {
        PrefillError::SerializationFailed(e.to_string())
    }
}

impl From<ContentError> for PrefillError {
    fn from(e: ContentError) -> Self {
        PrefillError::StorageFailed(e.to_string())
    }
}

/// Compute the token hash for a prefill sequence.
/// SHA-256 of token IDs as contiguous little-endian u32 bytes.
pub fn token_hash(token_ids: &[u32]) -> [u8; 32] {
    let bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    full_hash(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_deterministic() {
        let ids = vec![100u32, 200, 300];
        let h1 = token_hash(&ids);
        let h2 = token_hash(&ids);
        assert_eq!(h1, h2);
    }

    #[test]
    fn token_hash_different_for_different_tokens() {
        let h1 = token_hash(&[1, 2, 3]);
        let h2 = token_hash(&[3, 2, 1]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn token_hash_empty() {
        let h = token_hash(&[]);
        // SHA-256 of empty input — should be the well-known constant
        assert_eq!(h, full_hash(b""));
    }

    #[test]
    fn header_serde_roundtrip() {
        let header = PrefillCacheHeader {
            magic: PREFILL_MAGIC,
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 2048,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
        };
        let bytes = postcard::to_allocvec(&header).unwrap();
        let restored: PrefillCacheHeader = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(header, restored);
    }
}
```

- [ ] **Step 2: Add module declaration to lib.rs**

In `crates/harmony-speculative/src/lib.rs`, add:

```rust
#[cfg(feature = "prefill")]
pub mod prefill;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-speculative --features prefill -- prefill::tests`
Expected: PASS (4 tests)

Run: `cargo test -p harmony-speculative`
Expected: PASS (existing tests unaffected)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-speculative/src/prefill.rs crates/harmony-speculative/src/lib.rs
git commit -m "feat(speculative): add PrefillCacheHeader, PrefillError, and token_hash

Types for prefill KV cache sharing. PrefillCacheHeader carries model CID,
token hash, architecture params. token_hash computes SHA-256 of token IDs."
```

---

### Task 6: store_prefill_cache and load_prefill_cache

**Files:**
- Modify: `crates/harmony-speculative/src/prefill.rs`

- [ ] **Step 1: Write failing tests**

Add to the `prefill::tests` module:

```rust
    use candle_core::{DType, Device, Tensor};
    use harmony_content::book::MemoryBookStore;
    use harmony_content::cid::ContentId;

    /// Create a compressed InferenceCache with synthetic data.
    fn compressed_cache(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            cache.layers[0] = Some((k, v));
            cache.position = n_tokens;
        }
        cache.compress().unwrap();
        cache
    }

    fn fake_model_cid() -> ContentId {
        ContentId::for_book(b"fake-model-bytes", Default::default()).unwrap()
    }

    #[test]
    fn store_load_roundtrip() {
        let cache = compressed_cache(2, 8, 128, 16);
        let model_cid = fake_model_cid();
        let token_ids = vec![1u32, 2, 3, 4];
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();

        let (restored, header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert!(restored.is_compressed());
        assert_eq!(restored.position, 16);
        assert_eq!(header.token_count, 16);
        assert_eq!(header.num_layers, 2);
        assert_eq!(header.num_kv_heads, 8);
        assert_eq!(header.head_dim, 128);
        assert_eq!(header.quant_bits, 3);
        assert_eq!(header.token_hash, token_hash(&token_ids));
    }

    #[test]
    fn load_rejects_wrong_model() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &[1, 2], &mut store).unwrap();

        let wrong_cid = ContentId::for_book(b"wrong-model", Default::default()).unwrap();
        let result = load_prefill_cache(&root, &wrong_cid, &store);
        assert!(matches!(result, Err(PrefillError::ModelMismatch { .. })));
    }

    #[test]
    fn store_requires_compressed() {
        let cache = InferenceCache::new(2, 128, 8); // not compressed
        let model_cid = fake_model_cid();
        let mut store = MemoryBookStore::new();

        let result = store_prefill_cache(&cache, &model_cid, &[1], &mut store);
        assert!(matches!(result, Err(PrefillError::CacheNotCompressed)));
    }

    #[test]
    fn header_magic_validation() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &[1], &mut store).unwrap();

        // Corrupt the first byte of the stored blob (magic)
        // Reassemble, corrupt, re-store
        let mut blob = dag::reassemble(&root, &store).unwrap();
        blob[4] = 0xFF; // corrupt magic (skip header_len prefix)
        let corrupt_root = dag::ingest(&blob, &ChunkerConfig::DEFAULT, &mut store).unwrap();

        let result = load_prefill_cache(&corrupt_root, &model_cid, &store);
        assert!(matches!(result, Err(PrefillError::InvalidMagic)));
    }

    #[test]
    fn unsupported_quant_bits_rejected() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &[1], &mut store).unwrap();

        // Reassemble, patch quant_bits in header, re-store
        let mut blob = dag::reassemble(&root, &store).unwrap();
        let header_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        // Decode header, change quant_bits, re-encode
        let mut header: PrefillCacheHeader = postcard::from_bytes(&blob[4..4 + header_len]).unwrap();
        header.quant_bits = 5; // unsupported
        let new_header_bytes = postcard::to_allocvec(&header).unwrap();
        // Rebuild blob with new header
        let mut new_blob = Vec::new();
        new_blob.extend_from_slice(&(new_header_bytes.len() as u32).to_le_bytes());
        new_blob.extend_from_slice(&new_header_bytes);
        new_blob.extend_from_slice(&blob[4 + header_len..]);
        let corrupt_root = dag::ingest(&new_blob, &ChunkerConfig::DEFAULT, &mut store).unwrap();

        let result = load_prefill_cache(&corrupt_root, &model_cid, &store);
        assert!(matches!(result, Err(PrefillError::UnsupportedQuantBits(5))));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-speculative --features prefill -- prefill::tests::store`
Expected: FAIL — functions don't exist

- [ ] **Step 3: Implement store_prefill_cache and load_prefill_cache**

Add to `crates/harmony-speculative/src/prefill.rs`:

```rust
/// Store a compressed KV cache in CAS. Returns the root CID.
///
/// The cache must be compressed (`is_compressed() == true`).
/// `model_cid` identifies the GGUF model. `token_ids` is the
/// prefill token sequence (hashed into the header).
pub fn store_prefill_cache(
    cache: &InferenceCache,
    model_cid: &ContentId,
    token_ids: &[u32],
    store: &mut dyn BookStore,
) -> Result<ContentId, PrefillError> {
    if !cache.is_compressed() {
        return Err(PrefillError::CacheNotCompressed);
    }

    let header = PrefillCacheHeader {
        magic: PREFILL_MAGIC,
        model_cid: model_cid.to_bytes(),
        token_hash: token_hash(token_ids),
        token_count: cache.position as u32,
        num_layers: cache.num_layers as u16,
        num_kv_heads: cache.num_kv_heads as u16,
        head_dim: cache.head_dim as u16,
        quant_bits: SUPPORTED_QUANT_BITS,
    };

    let header_bytes = postcard::to_allocvec(&header)
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;

    let payload_bytes = cache.serialize_compressed()?;

    // Wire blob: [header_len: u32 LE][header][payload]
    let mut blob = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
    blob.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    blob.extend_from_slice(&header_bytes);
    blob.extend_from_slice(&payload_bytes);

    let root = dag::ingest(&blob, &ChunkerConfig::DEFAULT, store)?;
    Ok(root)
}

/// Load a prefill cache from CAS.
///
/// Validates the header magic, quant_bits, and model CID. Returns the cache
/// in compressed state (caller must call `decompress()` before `forward()`).
pub fn load_prefill_cache(
    root_cid: &ContentId,
    expected_model_cid: &ContentId,
    store: &dyn BookStore,
) -> Result<(InferenceCache, PrefillCacheHeader), PrefillError> {
    let blob = dag::reassemble(root_cid, store)?;

    if blob.len() < 4 {
        return Err(PrefillError::SerializationFailed("blob too short".into()));
    }

    let header_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    if blob.len() < 4 + header_len {
        return Err(PrefillError::SerializationFailed("truncated header".into()));
    }

    let header: PrefillCacheHeader = postcard::from_bytes(&blob[4..4 + header_len])
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;

    if header.magic != PREFILL_MAGIC {
        return Err(PrefillError::InvalidMagic);
    }

    if header.quant_bits != SUPPORTED_QUANT_BITS {
        return Err(PrefillError::UnsupportedQuantBits(header.quant_bits));
    }

    let expected_bytes = expected_model_cid.to_bytes();
    if header.model_cid != expected_bytes {
        return Err(PrefillError::ModelMismatch {
            expected: expected_bytes,
            actual: header.model_cid,
        });
    }

    let payload = &blob[4 + header_len..];
    let cache = InferenceCache::deserialize_compressed(
        payload,
        header.num_layers as usize,
        header.head_dim as usize,
        header.num_kv_heads as usize,
    )?;

    Ok((cache, header))
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-speculative --features prefill -- prefill::tests`
Expected: ALL PASS

- [ ] **Step 5: Run clippy**

Run: `cargo clippy -p harmony-speculative --features prefill -- -D warnings`
Expected: Clean

Run: `cargo clippy -p harmony-speculative -- -D warnings`
Expected: Clean

Run: `cargo clippy -p harmony-inference --features kv-compress -- -D warnings`
Expected: Clean

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-speculative/src/prefill.rs
git commit -m "feat(speculative): implement store/load prefill cache via CAS

store_prefill_cache serializes compressed KV cache + header, chunks via
FastCDC, stores as Merkle DAG. load_prefill_cache reassembles, validates
magic/quant_bits/model_cid, deserializes to compressed InferenceCache."
```

---

## Test Matrix Summary

| Test | Crate | What | Task |
|------|-------|------|------|
| `quantized_vec_serde_roundtrip` | inference | Postcard roundtrip for QuantizedVec | 2 |
| `compressed_kv_layer_serde_roundtrip` | inference | Postcard roundtrip for CompressedKvLayer | 2 |
| `serialization_failed_displays_message` | inference | Error variant display | 3 |
| `serialize_deserialize_roundtrip` | inference | Full cache serialize/deserialize cycle | 3 |
| `serialize_uncompressed_errors` | inference | Reject serialization of uncompressed cache | 3 |
| `deserialize_validates_num_layers` | inference | Reject mismatched num_layers | 3 |
| `serialize_empty_compressed_cache` | inference | Empty cache roundtrips | 3 |
| `serialize_preserves_position` | inference | Position survives roundtrip | 3 |
| `token_hash_deterministic` | speculative | Same input → same hash | 5 |
| `token_hash_different_for_different_tokens` | speculative | Different input → different hash | 5 |
| `token_hash_empty` | speculative | Empty token list → SHA-256 of empty | 5 |
| `header_serde_roundtrip` | speculative | Postcard roundtrip for PrefillCacheHeader | 5 |
| `store_load_roundtrip` | speculative | Full pipeline: compress → store → load → verify | 6 |
| `load_rejects_wrong_model` | speculative | Model CID mismatch → error | 6 |
| `store_requires_compressed` | speculative | Uncompressed cache → error | 6 |
| `header_magic_validation` | speculative | Corrupted magic → error | 6 |
| `unsupported_quant_bits_rejected` | speculative | quant_bits != 3 → error | 6 |

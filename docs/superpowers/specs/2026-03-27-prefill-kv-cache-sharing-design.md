# Prefill KV Cache Sharing via CAS Distribution

**Date:** 2026-03-27
**Status:** Draft
**Bead:** harmony-hbf0
**Depends on:** harmony-6b0 (externalized KV cache), harmony-p3zv (KV compression)

## Problem

On a heterogeneous mesh, powerful nodes (Tier 2: RPi5, 4-8GB) can evaluate prompts much faster than constrained nodes (Tier 1: MT7621, 512MB). Prefill — the initial forward pass over the full prompt — is the most expensive phase of inference. If a Tier 2 node has already prefilled a prompt, its KV cache contains all the information a Tier 1 node needs to continue generation without re-evaluating the prompt.

This spec defines how to serialize a compressed KV cache, store it in CAS as a Merkle DAG, and let any node on the mesh fetch it by CID and resume generation.

## Constraints

- **No new crates.** Serialization lives in harmony-inference (behind `kv-compress` feature). Protocol and CAS integration live in harmony-speculative.
- **Compressed only.** Only compressed caches can be serialized. Uncompressed caches are too large for mesh distribution (~224 MB at 2048 context) and would need compression anyway.
- **Full fetch.** No partial/range-based fetch. CAS chunk deduplication handles the common case. Partial fetch is a follow-up optimization if needed.
- **No verification.** Trust between nodes is assumed for now. TOPLOC verification is a follow-up bead (harmony-mssf).
- **Postcard serialization.** Deterministic, compact, `no_std`-compatible. Already a workspace dependency.

## Architecture

```
Producer node (Tier 2):                      Consumer node (Tier 1/2):

  prompt → tokenize → forward()              fetch root CID from mesh
         → cache with KV state                     ↓
         → cache.compress()               dag::reassemble(root_cid, store)
         → serialize_compressed()                  ↓
         → prepend PrefillCacheHeader      parse PrefillCacheHeader
         → dag::ingest() → root CID        validate model_cid, magic
         → publish root CID to mesh         deserialize_compressed()
                                            → InferenceCache (compressed)
                                            → cache.decompress()
                                            → continue generation with forward()
```

## Part 1: Serialization (harmony-inference)

### Visibility Changes

`QuantizedVec` and `CompressedKvLayer` change from `pub(crate)` to `pub`, and gain serde derives:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantizedVec {
    pub min: f32,
    pub scale: f32,
    pub packed: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    pub k: Vec<QuantizedVec>,
    pub v: Vec<QuantizedVec>,
    pub seq_len: usize,
}
```

### Serialized Payload

The serialized payload is a postcard-encoded struct:

```rust
#[derive(Serialize, Deserialize)]
struct CompressedCachePayload {
    position: usize,
    layers: Vec<Option<CompressedKvLayer>>,
}
```

This is an internal type — not exposed publicly. It exists only to group `position` with the layer data for serialization.

### New Methods on InferenceCache

```rust
#[cfg(feature = "kv-compress")]
impl InferenceCache {
    /// Serialize the compressed cache to bytes.
    /// Returns Err if the cache is not compressed.
    pub fn serialize_compressed(&self) -> Result<Vec<u8>, InferenceError>;

    /// Deserialize a compressed cache from bytes.
    /// Validates that the serialized data matches the given architecture params.
    pub fn deserialize_compressed(
        data: &[u8],
        num_layers: usize,
        head_dim: usize,
        num_kv_heads: usize,
    ) -> Result<InferenceCache, InferenceError>;
}
```

**`serialize_compressed`:** Checks `is_compressed`, clones the `compressed` vec and `position` into a `CompressedCachePayload`, postcard-encodes it.

**`deserialize_compressed`:** Postcard-decodes the payload, validates `layers.len() == num_layers`, constructs an `InferenceCache` in compressed state (is_compressed = true, layers all None, compressed populated). Note: `head_dim` and `num_kv_heads` are trusted inputs from the caller (sourced from the `PrefillCacheHeader`), not validated against the payload — the payload stores `QuantizedVec` packed bytes but not the original tensor dimensions. TOPLOC verification (harmony-mssf) addresses the trust gap.

### Error Handling

New `InferenceError` variant (behind `kv-compress`):

```rust
#[cfg(feature = "kv-compress")]
#[error("serialization failed: {0}")]
SerializationFailed(String),
```

### Dependencies

Add `serde` and `postcard` to harmony-inference's `[dependencies]` (both already workspace deps), gated behind `kv-compress`:

```toml
[features]
kv-compress = ["dep:serde", "dep:postcard"]

[dependencies]
serde = { workspace = true, features = ["derive", "alloc"], optional = true }
postcard = { workspace = true, optional = true }
```

## Part 2: Protocol and CAS Integration (harmony-speculative)

### PrefillCacheHeader

```rust
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PrefillCacheHeader {
    /// Magic bytes + format version.
    pub magic: [u8; 4],          // b"HKV\x01"

    /// CAS CID of the GGUF model that produced this cache.
    pub model_cid: [u8; 32],

    /// SHA-256 of the token IDs (as little-endian u32 bytes).
    pub token_hash: [u8; 32],

    /// Number of tokens in the prefill (= cache.position).
    pub token_count: u32,

    /// Architecture params for validation.
    pub num_layers: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,

    /// Compression config.
    pub quant_bits: u8,          // 3 for current 3-bit quantizer
}
```

**Size:** ~79 bytes raw, but postcard uses varint encoding for integers, so the serialized size varies slightly. The `header_len` prefix in the wire format handles this — readers never need to assume a fixed header size.

**`token_hash` computation:** SHA-256 over the `Vec<u32>` token IDs serialized as contiguous little-endian bytes. This ensures the hash is a function of the exact token sequence, not the text.

### Wire Blob Format

The blob stored in CAS is:

```
[header_len: u32 LE][postcard(PrefillCacheHeader)][postcard(CompressedCachePayload)]
```

The 4-byte `header_len` prefix enables the reader to split header from payload without parsing the payload. This is important for validation — a node can check the header (model CID, magic) before committing to deserialize the full ~49 MB payload.

### Public API

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
) -> Result<ContentId, PrefillError>;

/// Load a prefill cache from CAS.
///
/// Validates the header magic and model CID. Returns the cache
/// in compressed state (caller must call `decompress()` before
/// `forward()`).
pub fn load_prefill_cache(
    root_cid: &ContentId,
    expected_model_cid: &ContentId,
    store: &dyn BookStore,
) -> Result<(InferenceCache, PrefillCacheHeader), PrefillError>;

/// Compute the token hash for a prefill sequence.
/// SHA-256 of token IDs as contiguous little-endian u32 bytes.
pub fn token_hash(token_ids: &[u32]) -> [u8; 32];
```

### store_prefill_cache Flow

1. Check `cache.is_compressed()` — return `PrefillError::CacheNotCompressed` if false.
2. Build `PrefillCacheHeader`:
   - magic: `b"HKV\x01"`
   - model_cid: `model_cid.to_bytes()` (ContentId → [u8; 32])
   - token_hash: SHA-256 of token_ids as LE bytes
   - token_count: `cache.position` as u32
   - num_layers, num_kv_heads, head_dim: from cache fields
   - quant_bits: 3
3. `cache.serialize_compressed()` → payload bytes.
4. Encode header with postcard → header bytes.
5. Build wire blob: `[header_len as u32 LE][header_bytes][payload_bytes]`.
6. `dag::ingest(wire_blob, default_chunker_config, store)` → root CID.
7. Return root CID.

### load_prefill_cache Flow

1. `dag::reassemble(root_cid, store)` → wire blob bytes.
2. Read `header_len` (first 4 bytes, u32 LE).
3. Postcard-decode `PrefillCacheHeader` from bytes `[4..4+header_len]`.
4. Validate magic == `b"HKV\x01"` — return `PrefillError::InvalidMagic` if not.
5. Validate `header.quant_bits == 3` — return `PrefillError::UnsupportedQuantBits` if not (forward-compat guard).
6. Validate `header.model_cid == expected_model_cid.to_bytes()` — return `PrefillError::ModelMismatch` if not.
7. Remaining bytes `[4+header_len..]` are the payload.
8. `InferenceCache::deserialize_compressed(payload, header.num_layers, header.head_dim, header.num_kv_heads)` → cache.
9. Return (cache, header).

### Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("invalid magic: expected HKV\\x01")]
    InvalidMagic,

    #[error("model mismatch: expected {expected:?}, got {actual:?}")]
    ModelMismatch { expected: [u8; 32], actual: [u8; 32] },

    #[error("unsupported quant_bits: {0}")]
    UnsupportedQuantBits(u8),

    #[error("cache is not compressed")]
    CacheNotCompressed,

    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    #[error("storage failed: {0}")]
    StorageFailed(String),
}
```

### Dependencies

The prefill functionality is behind a `prefill` feature flag on harmony-speculative. This avoids pulling candle (via harmony-inference) into every consumer of harmony-speculative.

```toml
[features]
default = []
prefill = ["dep:harmony-inference", "dep:harmony-content", "dep:harmony-crypto", "dep:postcard"]

[dependencies]
harmony-inference = { workspace = true, features = ["kv-compress"], optional = true }
harmony-content = { workspace = true, optional = true }
harmony-crypto = { workspace = true, optional = true }
serde = { workspace = true, features = ["derive"] }
postcard = { workspace = true, optional = true }
thiserror = { workspace = true }
```

All prefill types and functions are `#[cfg(feature = "prefill")]`.

## Wire Size Analysis

At 2048 context tokens with 3-bit compression:

| Component | Size |
|---|---|
| header_len prefix | 4 bytes |
| PrefillCacheHeader (postcard, varint) | ~75-79 bytes |
| Compressed KV payload (postcard) | ~49 MB |
| **Total wire blob** | **~49 MB** |
| FastCDC chunks (512KB avg) | ~96 books |
| Merkle DAG bundles | 1 bundle (96 × 32B CIDs = 3 KB) |

At 1024 context: ~24.5 MB → ~48 chunks.
At 512 context: ~12.2 MB → ~24 chunks.

## Testing Strategy

### harmony-inference tests (kv-compress feature)

| Test | What |
|---|---|
| `serialize_deserialize_roundtrip` | Compress, serialize, deserialize, verify layers match |
| `serialize_uncompressed_errors` | Serialize on uncompressed cache → SerializationFailed |
| `deserialize_validates_params` | Mismatched num_layers/head_dim → error |
| `serialize_empty_compressed_cache` | All-None layers roundtrip correctly |
| `serialize_preserves_position` | position value survives roundtrip |

### harmony-speculative tests

| Test | What |
|---|---|
| `store_load_roundtrip` | Full pipeline: compress → store → load → verify |
| `load_rejects_wrong_model` | Mismatched model CID → ModelMismatch |
| `header_magic_validation` | Corrupted magic → InvalidMagic |
| `token_hash_deterministic` | Same token IDs → same hash |
| `store_requires_compressed` | Uncompressed cache → CacheNotCompressed |
| `unsupported_quant_bits_rejected` | Header with quant_bits != 3 → UnsupportedQuantBits |

## Out of Scope

- **TOPLOC verification** — follow-up bead harmony-mssf
- **Partial/range fetch** — future optimization
- **Protocol negotiation** — mesh-layer concern, not this spec
- **Uncompressed cache serialization** — too large for mesh distribution, no use case
- **GPU device threading** — decompress() hardcodes Device::Cpu (known limitation from harmony-p3zv, tracked in harmony-51dy)

# Oluo Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build Harmony's semantic search data layer (`harmony-semantic`) and search engine (`harmony-oluo`) as sans-I/O crates with content-addressed binary vector indexing.

**Architecture:** Two crates following the sans-I/O pattern. `harmony-semantic` handles binary formats (sidecar encode/decode, collection blobs, Hamming distance, quantization, CBOR metadata, overlay merge). `harmony-oluo` builds the adaptive-depth embedding trie and search engine on top, with trait-based contracts for Jain (privacy gate) and Wylene (user intent). Both are `no_std + alloc` at their core.

**Tech Stack:** Rust (edition 2021, MSRV 1.81), `harmony-crypto` (SHA-256, BLAKE3), `harmony-content` (ContentId, CidType::ReservedA), `ciborium` (CBOR encoding, std-gated), `serde` (serialization), `hashbrown` (no_std maps).

**Design docs:**
- `docs/plans/2026-03-07-harmony-semantic-index-design.md` — HSI v1 format spec
- `docs/plans/2026-03-08-oluo-design.md` — Enriched sidecars, trie index, Jain/Oluo contracts

**Existing beads:** harmony-e67, harmony-5s7, harmony-c8w, harmony-0oi (all P4, OPEN, harmony-semantic label)

---

## Phase 1: harmony-semantic (data formats & vector ops)

### Task 1: Crate skeleton + error types + embedding tier enum

**Context:** Create the `harmony-semantic` crate with `no_std + alloc` support. This is the foundation everything else builds on. Follow the same crate structure pattern as `harmony-kitri`.

**Files:**
- Create: `crates/harmony-semantic/Cargo.toml`
- Create: `crates/harmony-semantic/src/lib.rs`
- Create: `crates/harmony-semantic/src/error.rs`
- Create: `crates/harmony-semantic/src/tier.rs`
- Modify: `Cargo.toml` (workspace — add member + ciborium dep)

**Step 1: Add `ciborium` to workspace dependencies and add `harmony-semantic` as a member**

In root `Cargo.toml`, add to `[workspace]` members:
```toml
"crates/harmony-semantic",
```

Add to `[workspace.dependencies]`:
```toml
ciborium = { version = "0.2", default-features = false }
```

**Step 2: Create `crates/harmony-semantic/Cargo.toml`**

```toml
[package]
name = "harmony-semantic"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Harmony Semantic Index — binary vector embeddings, sidecar format, and collection blobs"

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-content/std",
    "serde/std",
    "dep:ciborium",
]

[dependencies]
harmony-crypto = { workspace = true, default-features = false }
harmony-content = { workspace = true, default-features = false }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
ciborium = { workspace = true, optional = true }
hashbrown = { workspace = true }

[dev-dependencies]
```

**Step 3: Write the failing tests for `SemanticError` and `EmbeddingTier`**

In `src/error.rs`:
```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for harmony-semantic.

use alloc::string::String;
use core::fmt;

/// Errors from semantic index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticError {
    /// Sidecar magic bytes don't match expected format.
    InvalidMagic,
    /// Sidecar data is too short to contain the fixed header.
    TruncatedHeader { expected: usize, actual: usize },
    /// Collection entry count exceeds maximum (256).
    CollectionOverflow { count: u32 },
    /// Model fingerprint mismatch during overlay merge.
    FingerprintMismatch,
    /// CBOR metadata decoding failed.
    MetadataInvalid { reason: String },
    /// Privacy tier 3 (encrypted-ephemeral) cannot be indexed.
    PrivacyBlocked,
    /// Input vector has wrong number of dimensions for quantization.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid sidecar magic bytes"),
            Self::TruncatedHeader { expected, actual } => {
                write!(f, "truncated header: expected {expected} bytes, got {actual}")
            }
            Self::CollectionOverflow { count } => {
                write!(f, "collection overflow: {count} entries exceeds max 256")
            }
            Self::FingerprintMismatch => write!(f, "model fingerprint mismatch"),
            Self::MetadataInvalid { reason } => {
                write!(f, "invalid CBOR metadata: {reason}")
            }
            Self::PrivacyBlocked => {
                write!(f, "encrypted-ephemeral content cannot be indexed")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

pub type SemanticResult<T> = Result<T, SemanticError>;
```

In `src/tier.rs`:
```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Embedding tier definitions — progressive resolution from 64 to 1024 bits.

/// The five embedding tiers, each a nested prefix of the MRL vector.
///
/// Tier N contains the first 2^(N+5) bits of the binary-quantized vector.
/// Higher tiers are strict supersets of lower tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EmbeddingTier {
    /// 64-bit (8 bytes) — single GPR comparison. Network fanout tier.
    T1 = 1,
    /// 128-bit (16 bytes) — SSE register. Lightweight indexing.
    T2 = 2,
    /// 256-bit (32 bytes) — AVX2 register. Community search tier.
    T3 = 3,
    /// 512-bit (64 bytes) — AVX-512 register. High precision.
    T4 = 4,
    /// 1024-bit (128 bytes) — 2×AVX-512. Maximum precision.
    T5 = 5,
}

impl EmbeddingTier {
    /// Number of bits in this tier's binary vector.
    pub const fn bit_count(self) -> usize {
        match self {
            Self::T1 => 64,
            Self::T2 => 128,
            Self::T3 => 256,
            Self::T4 => 512,
            Self::T5 => 1024,
        }
    }

    /// Number of bytes in this tier's binary vector.
    pub const fn byte_count(self) -> usize {
        self.bit_count() / 8
    }

    /// Offset of this tier's data within the sidecar fixed header.
    /// Tier data starts at byte 40 (after magic + fingerprint + target CID).
    pub const fn sidecar_offset(self) -> usize {
        match self {
            Self::T1 => 40,
            Self::T2 => 48,
            Self::T3 => 64,
            Self::T4 => 96,
            Self::T5 => 160,
        }
    }
}
```

In `src/lib.rs`:
```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Harmony Semantic Index — binary vector embeddings, sidecar format,
//! and collection blobs for content-addressed semantic search.

#![no_std]
extern crate alloc;

pub mod error;
pub mod tier;

pub use error::{SemanticError, SemanticResult};
pub use tier::EmbeddingTier;
```

**Step 4: Run tests to verify compilation**

```bash
cargo test -p harmony-semantic
```

Expected: compiles, 0 tests (we'll add tests in the next tasks with the types they test).

**Step 5: Commit**

```bash
git add crates/harmony-semantic/ Cargo.toml Cargo.lock
git commit -m "feat(semantic): add harmony-semantic crate skeleton with error types and tier enum"
```

---

### Task 2: Model fingerprint + sidecar v1 fixed header codec

**Context:** The sidecar is the core data structure — a 288-byte binary blob containing multi-tier embedding vectors for a piece of content. The fixed header layout is defined in the HSI design doc. The model fingerprint is `SHA-256(model_id)[:4]`. This task implements the v1 sidecar (fixed header only, no CBOR trailer).

**Bead:** harmony-e67 (HSI: sidecar blob codec)

**Files:**
- Create: `crates/harmony-semantic/src/fingerprint.rs`
- Create: `crates/harmony-semantic/src/sidecar.rs`
- Modify: `crates/harmony-semantic/src/lib.rs`

**Step 1: Write fingerprint module**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Model fingerprint — first 4 bytes of SHA-256(model_id).

/// A 4-byte model fingerprint identifying the embedding model.
///
/// Computed as `SHA-256(model_id_string)[:4]`. Enables mixed-model
/// coexistence: sidecars from different models can coexist, and the
/// fingerprint disambiguates during overlay merge and search.
pub type ModelFingerprint = [u8; 4];

/// Compute a model fingerprint from a model identifier string.
///
/// ```ignore
/// let fp = model_fingerprint("pplx-embed-v1-0.6B");
/// assert_eq!(fp.len(), 4);
/// ```
pub fn model_fingerprint(model_id: &str) -> ModelFingerprint {
    let hash = harmony_crypto::hash::full_hash(model_id.as_bytes());
    let mut fp = [0u8; 4];
    fp.copy_from_slice(&hash[..4]);
    fp
}
```

**Step 2: Write sidecar v1 codec**

The sidecar struct holds the decoded fixed header. Encoding writes the 288-byte binary format; decoding reads it back. Key design: tier data fields use fixed-size arrays matching each tier's byte count. The `SIDECAR_HEADER_SIZE` constant is 288.

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Sidecar blob codec — encode/decode the 288-byte HSI fixed header.

use crate::error::{SemanticError, SemanticResult};
use crate::fingerprint::ModelFingerprint;

/// HSI v1 magic bytes: "HSI" + version 1.
pub const SIDECAR_V1_MAGIC: [u8; 4] = [0x48, 0x53, 0x49, 0x01];
/// HSI v2 magic bytes: "HSI" + version 2 (enriched sidecar with CBOR trailer).
pub const SIDECAR_V2_MAGIC: [u8; 4] = [0x48, 0x53, 0x49, 0x02];
/// Size of the fixed header (both v1 and v2).
pub const SIDECAR_HEADER_SIZE: usize = 288;

/// Decoded sidecar fixed header.
///
/// Contains the multi-tier binary embedding vectors for a single piece
/// of content. Tiers are nested prefixes of the same MRL vector — Tier 1
/// is the first 64 bits, Tier 2 is the first 128, and so on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SidecarHeader {
    /// Model fingerprint (SHA-256(model_id)[:4]).
    pub fingerprint: ModelFingerprint,
    /// CID of the content this sidecar describes (32 bytes).
    pub target_cid: [u8; 32],
    /// Tier 1: 64-bit binary vector (8 bytes).
    pub tier1: [u8; 8],
    /// Tier 2: 128-bit binary vector (16 bytes).
    pub tier2: [u8; 16],
    /// Tier 3: 256-bit binary vector (32 bytes).
    pub tier3: [u8; 32],
    /// Tier 4: 512-bit binary vector (64 bytes).
    pub tier4: [u8; 64],
    /// Tier 5: 1024-bit binary vector (128 bytes).
    pub tier5: [u8; 128],
}

impl SidecarHeader {
    /// Encode the header as a 288-byte v1 sidecar blob.
    pub fn encode_v1(&self) -> [u8; SIDECAR_HEADER_SIZE] {
        let mut buf = [0u8; SIDECAR_HEADER_SIZE];
        buf[0..4].copy_from_slice(&SIDECAR_V1_MAGIC);
        buf[4..8].copy_from_slice(&self.fingerprint);
        buf[8..40].copy_from_slice(&self.target_cid);
        buf[40..48].copy_from_slice(&self.tier1);
        buf[48..64].copy_from_slice(&self.tier2);
        buf[64..96].copy_from_slice(&self.tier3);
        buf[96..160].copy_from_slice(&self.tier4);
        buf[160..288].copy_from_slice(&self.tier5);
        buf
    }

    /// Decode a sidecar header from bytes. Accepts both v1 and v2 magic
    /// (v2 has additional CBOR trailer after the header, parsed separately).
    pub fn decode(data: &[u8]) -> SemanticResult<Self> {
        if data.len() < SIDECAR_HEADER_SIZE {
            return Err(SemanticError::TruncatedHeader {
                expected: SIDECAR_HEADER_SIZE,
                actual: data.len(),
            });
        }

        let magic = &data[0..4];
        if magic != SIDECAR_V1_MAGIC && magic != SIDECAR_V2_MAGIC {
            return Err(SemanticError::InvalidMagic);
        }

        let mut fingerprint = [0u8; 4];
        fingerprint.copy_from_slice(&data[4..8]);
        let mut target_cid = [0u8; 32];
        target_cid.copy_from_slice(&data[8..40]);
        let mut tier1 = [0u8; 8];
        tier1.copy_from_slice(&data[40..48]);
        let mut tier2 = [0u8; 16];
        tier2.copy_from_slice(&data[48..64]);
        let mut tier3 = [0u8; 32];
        tier3.copy_from_slice(&data[64..96]);
        let mut tier4 = [0u8; 64];
        tier4.copy_from_slice(&data[96..160]);
        let mut tier5 = [0u8; 128];
        tier5.copy_from_slice(&data[160..288]);

        Ok(Self {
            fingerprint,
            target_cid,
            tier1,
            tier2,
            tier3,
            tier4,
            tier5,
        })
    }

    /// Returns `true` if the given data starts with v2 magic (has CBOR trailer).
    pub fn is_v2(data: &[u8]) -> bool {
        data.len() >= 4 && data[0..4] == SIDECAR_V2_MAGIC
    }

    /// Extract the tier data for a given embedding tier.
    pub fn tier_data(&self, tier: crate::tier::EmbeddingTier) -> &[u8] {
        use crate::tier::EmbeddingTier;
        match tier {
            EmbeddingTier::T1 => &self.tier1,
            EmbeddingTier::T2 => &self.tier2,
            EmbeddingTier::T3 => &self.tier3,
            EmbeddingTier::T4 => &self.tier4,
            EmbeddingTier::T5 => &self.tier5,
        }
    }
}
```

**Tests to write (in `sidecar.rs` `#[cfg(test)] mod tests`):**

1. `sidecar_v1_encode_decode_roundtrip` — encode then decode, assert fields match
2. `sidecar_decode_truncated_rejects` — data shorter than 288 bytes → `TruncatedHeader`
3. `sidecar_decode_bad_magic_rejects` — wrong magic → `InvalidMagic`
4. `sidecar_v2_magic_accepted` — v2 magic bytes decode header correctly
5. `sidecar_tier_data_returns_correct_slice` — `tier_data(T1)` returns tier1, etc.
6. `model_fingerprint_deterministic` — same model ID → same fingerprint

**Step 3: Run tests**
```bash
cargo test -p harmony-semantic
```

**Step 4: Commit**
```bash
git add crates/harmony-semantic/src/
git commit -m "feat(semantic): sidecar v1 fixed header codec + model fingerprint"
```

---

### Task 3: Hamming distance + binary quantization

**Context:** The core search primitive is XOR + POPCNT over binary vectors. Quantization converts float32 embeddings to binary (> 0 → 1, ≤ 0 → 0). These are used everywhere — sidecar creation, collection search, trie traversal.

**Bead:** harmony-5s7 (HSI: binary quantization and Hamming distance)

**Files:**
- Create: `crates/harmony-semantic/src/distance.rs`
- Create: `crates/harmony-semantic/src/quantize.rs`
- Modify: `crates/harmony-semantic/src/lib.rs`

**Step 1: Write distance module**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Hamming distance — XOR + POPCNT over binary vectors.

/// Compute the Hamming distance between two binary vectors of equal length.
///
/// This is the number of bit positions where the two vectors differ.
/// Implemented as XOR each byte pair, then count set bits (POPCNT).
/// The compiler auto-vectorizes this for SSE/AVX when available.
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must be equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute the Hamming distance using u64 chunks for better throughput.
///
/// Processes 8 bytes at a time. Falls back to byte-level for the remainder.
pub fn hamming_distance_fast(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must be equal length");
    let chunks = a.len() / 8;
    let mut dist = 0u32;

    for i in 0..chunks {
        let offset = i * 8;
        let va = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        dist += (va ^ vb).count_ones();
    }

    // Handle remaining bytes.
    let remainder = chunks * 8;
    for i in remainder..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}
```

**Step 2: Write quantize module**

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Binary quantization — float32 → binary vector conversion.

use alloc::vec::Vec;

use crate::error::{SemanticError, SemanticResult};
use crate::sidecar::SidecarHeader;

/// Quantize a float32 MRL vector to a binary vector.
///
/// Each dimension: if value > 0.0, output bit 1; else 0.
/// The output is packed into bytes, MSB first within each byte.
pub fn quantize_to_binary(float_vec: &[f32]) -> Vec<u8> {
    let byte_count = (float_vec.len() + 7) / 8;
    let mut bytes = alloc::vec![0u8; byte_count];

    for (i, &val) in float_vec.iter().enumerate() {
        if val > 0.0 {
            bytes[i / 8] |= 1 << (7 - (i % 8));
        }
    }

    bytes
}

/// Pack a quantized binary vector into the five sidecar tiers.
///
/// The input must be at least 1024 dimensions (128 bytes after quantization).
/// If shorter, the remaining tiers are zero-filled.
pub fn pack_tiers(
    float_vec: &[f32],
    fingerprint: [u8; 4],
    target_cid: [u8; 32],
) -> SemanticResult<SidecarHeader> {
    if float_vec.is_empty() {
        return Err(SemanticError::DimensionMismatch {
            expected: 64,
            actual: 0,
        });
    }

    let binary = quantize_to_binary(float_vec);

    // Tier 1: first 8 bytes (64 bits). Minimum required.
    if binary.len() < 8 {
        return Err(SemanticError::DimensionMismatch {
            expected: 64,
            actual: float_vec.len(),
        });
    }

    let mut tier1 = [0u8; 8];
    tier1.copy_from_slice(&binary[..8]);

    let mut tier2 = [0u8; 16];
    let t2_len = binary.len().min(16);
    tier2[..t2_len].copy_from_slice(&binary[..t2_len]);

    let mut tier3 = [0u8; 32];
    let t3_len = binary.len().min(32);
    tier3[..t3_len].copy_from_slice(&binary[..t3_len]);

    let mut tier4 = [0u8; 64];
    let t4_len = binary.len().min(64);
    tier4[..t4_len].copy_from_slice(&binary[..t4_len]);

    let mut tier5 = [0u8; 128];
    let t5_len = binary.len().min(128);
    tier5[..t5_len].copy_from_slice(&binary[..t5_len]);

    Ok(SidecarHeader {
        fingerprint,
        target_cid,
        tier1,
        tier2,
        tier3,
        tier4,
        tier5,
    })
}
```

**Tests to write:**

Distance tests:
1. `hamming_distance_identical_is_zero` — same bytes → 0
2. `hamming_distance_all_different` — `0x00` vs `0xFF` → 8 per byte
3. `hamming_distance_known_vector` — hand-computed example
4. `hamming_distance_fast_matches_simple` — both functions return same result for various inputs
5. `hamming_distance_tier3_scale` — 32-byte vectors (256-bit, realistic tier 3)

Quantize tests:
1. `quantize_positive_to_one` — `[1.0, -1.0, 0.5, -0.5]` → `0b1010_0000` = `0xA0`
2. `quantize_zero_to_zero` — `0.0` → bit 0
3. `quantize_1024_dimensions` — 1024 floats → 128 bytes
4. `pack_tiers_roundtrip` — pack then read back tiers, verify nested prefixes
5. `pack_tiers_rejects_empty` — empty input → error
6. `tiers_are_nested_prefixes` — tier1 == tier2[..8] == tier3[..8], etc.

**Step 3: Run tests**
```bash
cargo test -p harmony-semantic
```

**Step 4: Commit**
```bash
git add crates/harmony-semantic/src/
git commit -m "feat(semantic): Hamming distance + binary quantization with tier packing"
```

---

### Task 4: Collection blob codec

**Context:** Collection blobs group up to 256 semantically similar content CIDs into a navigable cluster. They're the leaf nodes of the future trie index. Format: 4-byte magic + 4-byte fingerprint + 4-byte count + 32-byte centroid + N×64 entries (32-byte CID + 32-byte Tier 3 vector).

**Bead:** harmony-c8w (HSI: collection blob codec)

**Files:**
- Create: `crates/harmony-semantic/src/collection.rs`
- Modify: `crates/harmony-semantic/src/lib.rs`

**Step 1: Write collection module**

Key types:
- `CollectionEntry { target_cid: [u8; 32], tier3: [u8; 32] }` — a single entry
- `CollectionBlob { fingerprint, centroid, entries }` — the full collection
- `COLLECTION_MAGIC: [u8; 4]` = `[0x48, 0x53, 0x43, 0x01]` ("HSC" v1)
- `COLLECTION_HEADER_SIZE: usize` = 44 (magic + fingerprint + count + centroid)
- `COLLECTION_ENTRY_SIZE: usize` = 64 (CID + tier3 vector)
- `MAX_COLLECTION_ENTRIES: u32` = 256

Methods:
- `CollectionBlob::encode(&self) -> Vec<u8>` — serialize to binary
- `CollectionBlob::decode(data: &[u8]) -> SemanticResult<Self>` — deserialize
- `CollectionBlob::compute_centroid(entries: &[CollectionEntry]) -> [u8; 32]` — majority-vote centroid (for each bit position, the most common value wins)

**Tests to write:**
1. `collection_encode_decode_roundtrip` — encode then decode, assert all fields match
2. `collection_decode_truncated_rejects` — too short → error
3. `collection_decode_bad_magic_rejects` — wrong magic → error
4. `collection_overflow_rejects` — >256 entries → `CollectionOverflow`
5. `collection_empty_allowed` — 0 entries is valid
6. `centroid_computation` — known vectors → expected centroid
7. `collection_entry_size_is_64` — constant check

**Step 2: Run tests**
```bash
cargo test -p harmony-semantic
```

**Step 3: Commit**
```bash
git add crates/harmony-semantic/src/
git commit -m "feat(semantic): collection blob codec with centroid computation"
```

---

### Task 5: CBOR metadata + enriched sidecar (v2)

**Context:** The enriched sidecar extends the v1 fixed header with a CBOR-encoded metadata trailer. This is std-gated (uses `ciborium`). The trailer contains privacy tier, timestamps, content type, tags, geo coordinates, and an extension map. The `PrivacyTier` enum is used by both Oluo (indexing decisions) and Jain (gate decisions).

**Files:**
- Create: `crates/harmony-semantic/src/metadata.rs`
- Modify: `crates/harmony-semantic/src/sidecar.rs` (add v2 encode/decode)
- Modify: `crates/harmony-semantic/src/lib.rs`

**Step 1: Write metadata module**

Key types (available in all builds, `no_std + alloc`):
```rust
/// Privacy tier for content indexing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[derive(serde::Serialize, serde::Deserialize)]
pub enum PrivacyTier {
    PublicDurable = 0,
    PublicEphemeral = 1,
    EncryptedDurable = 2,
    EncryptedEphemeral = 3,
}

/// Sidecar metadata — parsed from the CBOR trailer.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SidecarMetadata {
    pub privacy_tier: Option<PrivacyTier>,
    pub created_at: Option<u64>,
    pub content_type: Option<String>,
    pub language: Option<String>,
    pub geo: Option<(f64, f64)>,
    pub description: Option<String>,
    pub tags: Option<Vec<String>>,
    pub refs: Option<Vec<[u8; 32]>>,
    pub source_device: Option<String>,
    pub ext: Option<BTreeMap<String, Vec<u8>>>,
}
```

**Step 2: Write v2 sidecar encode/decode (std-gated)**

In `sidecar.rs`, add `EnrichedSidecar` struct (wraps `SidecarHeader` + `SidecarMetadata`):
```rust
/// An enriched sidecar — fixed header + CBOR metadata trailer.
#[derive(Debug, Clone, PartialEq)]
pub struct EnrichedSidecar {
    pub header: SidecarHeader,
    pub metadata: SidecarMetadata,
}
```

The encode method:
1. Write 288-byte header with v2 magic
2. CBOR-encode the metadata map
3. Write 4-byte trailer length (u32 BE) at byte 288
4. Write CBOR bytes after that

The decode method:
1. Decode the fixed header (reuses `SidecarHeader::decode`)
2. If data is exactly 288 bytes, return empty metadata (v1 compat)
3. Otherwise read 4-byte trailer length at 288, decode CBOR from 292..292+len

Gate the encode/decode methods behind `#[cfg(feature = "std")]`.

**Tests to write (all `#[cfg(feature = "std")]`):**
1. `enriched_sidecar_roundtrip` — full encode/decode with all metadata fields
2. `enriched_sidecar_minimal_metadata` — only privacy_tier and created_at
3. `enriched_sidecar_v1_compat` — 288-byte blob → empty metadata
4. `privacy_tier_ordering` — `PublicDurable < PublicEphemeral < ... < EncryptedEphemeral`
5. `metadata_with_tags_and_refs` — multiple tags, multiple refs

**Step 3: Run tests**
```bash
cargo test -p harmony-semantic
```

**Step 4: Commit**
```bash
git add crates/harmony-semantic/src/
git commit -m "feat(semantic): CBOR metadata trailer + enriched sidecar v2 format"
```

---

### Task 6: Overlay merge logic

**Context:** Multiple sidecars can target the same content CID (original + AI tags + user tags). At query time, Oluo merges them. Merge rules: privacy_tier → most restrictive, created_at → earliest, tags → union, description → concatenate, embedding → from blessed model, latest created_at.

**Files:**
- Create: `crates/harmony-semantic/src/overlay.rs`
- Modify: `crates/harmony-semantic/src/lib.rs`

**Step 1: Write overlay merge**

Key function: `merge_metadata(base: &SidecarMetadata, overlay: &SidecarMetadata) -> SidecarMetadata`

Merge rules from design doc (lines 260-280):
- `privacy_tier`: most restrictive (max value)
- `created_at`: earliest timestamp
- `content_type`: first non-null
- `language`: concatenate unique values (both may be multilingual)
- `geo`: latest wins (most recent knowledge)
- `description`: concatenate with ` | ` separator
- `tags`: union (set merge)
- `refs`: union (set merge)
- `source_device`: first non-null
- `ext`: union merge, key conflicts → latest created_at wins

Also: `select_embedding_header(headers: &[&SidecarHeader], blessed_fingerprint: &ModelFingerprint) -> Option<&SidecarHeader>` — pick the header matching the blessed model; if multiple match, prefer latest (by position, since we don't have created_at on headers themselves).

**Tests to write:**
1. `merge_privacy_most_restrictive` — PublicDurable + EncryptedDurable → EncryptedDurable
2. `merge_created_at_earliest` — 1000 + 500 → 500
3. `merge_tags_union` — `["a", "b"]` + `["b", "c"]` → `["a", "b", "c"]`
4. `merge_description_concatenates` — `"foo"` + `"bar"` → `"foo | bar"`
5. `merge_content_type_first_non_null` — None + Some("image/jpeg") → Some("image/jpeg")
6. `merge_geo_latest_wins` — (1.0, 2.0) then (3.0, 4.0) → (3.0, 4.0)
7. `merge_refs_union_dedup` — overlapping CID refs → union without dups
8. `select_embedding_matching_fingerprint` — picks header with blessed fingerprint

**Step 2: Run tests, commit**
```bash
cargo test -p harmony-semantic
git add crates/harmony-semantic/src/
git commit -m "feat(semantic): overlay merge logic for multi-sidecar content"
```

---

## Phase 2: harmony-oluo (search engine)

### Task 7: Crate skeleton + error types + Zenoh key expressions

**Context:** `harmony-oluo` is the search engine built on `harmony-semantic`. Sans-I/O state machine pattern. Depends on `harmony-semantic`, `harmony-crypto`, `harmony-content`.

**Files:**
- Create: `crates/harmony-oluo/Cargo.toml`
- Create: `crates/harmony-oluo/src/lib.rs`
- Create: `crates/harmony-oluo/src/error.rs`
- Create: `crates/harmony-oluo/src/zenoh_keys.rs`
- Modify: `Cargo.toml` (workspace — add member)

**Step 1: Add `harmony-oluo` workspace member and create crate**

```toml
# Cargo.toml (harmony-oluo)
[package]
name = "harmony-oluo"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Oluo — Harmony's semantic search engine (adaptive-depth trie, sans-I/O)"

[features]
default = ["std"]
std = [
    "harmony-semantic/std",
    "harmony-crypto/std",
    "harmony-content/std",
]

[dependencies]
harmony-semantic = { path = "../harmony-semantic", default-features = false }
harmony-crypto = { workspace = true, default-features = false }
harmony-content = { workspace = true, default-features = false }
hashbrown = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }

[dev-dependencies]
```

**Step 2: Write error types, Zenoh key constants, lib.rs**

Zenoh keys from design doc (lines 391-407):
```rust
pub const INGEST_PREFIX: &str = "oluo/ingest";
pub const QUERY_PREFIX: &str = "oluo/query";
pub const RESULTS_PREFIX: &str = "oluo/results";
pub const INDEX_STATS_PREFIX: &str = "oluo/index/stats";
pub const INDEX_SYNC_PREFIX: &str = "oluo/index/sync";
```

**Step 3: Verify compilation, commit**
```bash
cargo test -p harmony-oluo
git add crates/harmony-oluo/ Cargo.toml Cargo.lock
git commit -m "feat(oluo): add harmony-oluo crate skeleton with error types and Zenoh key expressions"
```

---

### Task 8: IngestGate + RetrievalFilter traits + search scope

**Context:** These are the contract traits between Oluo and Jain. The engine doesn't implement Jain — it defines the interface. Also define `SearchScope` (Personal/Community/NetworkWide) and `SearchQuery`.

**Files:**
- Create: `crates/harmony-oluo/src/ingest.rs`
- Create: `crates/harmony-oluo/src/filter.rs`
- Create: `crates/harmony-oluo/src/scope.rs`
- Modify: `crates/harmony-oluo/src/lib.rs`

**Step 1: Write trait definitions**

These come directly from the design doc (lines 322-402). Key types:
- `IngestDecision { IndexFull, IndexLightweight { ttl_secs }, Reject }`
- `RawSearchResult { target_cid, score: f32, metadata, overlays }`
- `FilteredSearchResult { target_cid, score: f32, metadata }`
- `RetrievalContext { requester, social_context, device_context }`
- `SearchScope { Personal, Community, NetworkWide }`
- `SearchQuery { embedding: [u8; 32], tier: EmbeddingTier, scope: SearchScope, max_results: u32 }`

**Tests:** Mostly type construction tests — these are trait definitions, the implementation is in Jain.

**Step 2: Commit**
```bash
git add crates/harmony-oluo/src/
git commit -m "feat(oluo): IngestGate/RetrievalFilter traits + SearchScope + SearchQuery"
```

---

### Task 9: Adaptive-depth trie node format + encode/decode

**Context:** The trie is a binary space partition tree over embedding bits. Internal nodes have: magic + fingerprint + split_bit + child0_cid + child1_cid + centroid. Leaves are collection blobs (already implemented in harmony-semantic). This task implements the internal node format.

**Files:**
- Create: `crates/harmony-oluo/src/trie.rs`
- Modify: `crates/harmony-oluo/src/lib.rs`

**Step 1: Write trie node codec**

```rust
/// Internal trie node magic: "HSN" + version 1.
pub const TRIE_NODE_MAGIC: [u8; 4] = [0x48, 0x53, 0x4E, 0x01];
/// Size of an internal trie node blob.
pub const TRIE_NODE_SIZE: usize = 106;

/// An internal node in the adaptive-depth embedding trie.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrieNode {
    pub fingerprint: [u8; 4],
    /// The embedding bit position used to split at this node.
    pub split_bit: u16,
    /// CID of the child-0 subtree (bit = 0 at split_bit position).
    pub child0: [u8; 32],
    /// CID of the child-1 subtree (bit = 1 at split_bit position).
    pub child1: [u8; 32],
    /// Centroid of the entire subtree (256-bit binary vector).
    pub centroid: [u8; 32],
}
```

Methods: `encode() -> [u8; TRIE_NODE_SIZE]`, `decode(data) -> Result<Self>`.

Helper: `fn get_bit(vector: &[u8], bit_position: u16) -> bool` — extract a single bit from a binary vector (used during trie traversal to decide which child to descend into).

**Tests:**
1. `trie_node_encode_decode_roundtrip`
2. `trie_node_decode_bad_magic`
3. `trie_node_decode_truncated`
4. `get_bit_extracts_correct_bit` — test MSB-first bit extraction
5. `get_bit_boundary_values` — bit 0, bit 255, etc.

**Step 2: Commit**
```bash
git add crates/harmony-oluo/src/
git commit -m "feat(oluo): trie internal node format + bit extraction helper"
```

---

### Task 10: OluoEngine sans-I/O state machine

**Context:** The engine is the central coordinator. It receives events (Ingest, Search, SyncReceived, EvictExpired) and emits actions (IndexUpdated, SearchResults, FetchTrieNode, PersistBlob, etc.). It manages an in-memory trie root and delegates to trie traversal/search logic. Follows the same Event → handle() → Vec<Action> pattern as KitriEngine.

**Files:**
- Create: `crates/harmony-oluo/src/engine.rs`
- Modify: `crates/harmony-oluo/src/lib.rs`

**Step 1: Write the engine**

Events and actions from design doc (lines 527-541):
```rust
pub enum OluoEvent {
    /// Submit a sidecar for indexing (Jain has approved).
    Ingest {
        sidecar: EnrichedSidecar,
        decision: IngestDecision,
    },
    /// Execute a search query.
    Search {
        query_id: u64,
        query: SearchQuery,
    },
    /// A community published an updated trie root.
    SyncReceived {
        community_id: [u8; 32],
        trie_root: [u8; 32],
    },
    /// Timer tick: evict expired lightweight entries.
    EvictExpired { now_ms: u64 },
}

pub enum OluoAction {
    /// The local index trie root changed.
    IndexUpdated { trie_root: [u8; 32] },
    /// Search results ready for Jain filtering.
    SearchResults {
        query_id: u64,
        results: Vec<RawSearchResult>,
    },
    /// Request a trie node blob from content store.
    FetchTrieNode { cid: [u8; 32] },
    /// Request a sidecar blob from content store.
    FetchSidecar { cid: [u8; 32] },
    /// Publish our trie root to a community.
    PublishTrieRoot {
        community_id: [u8; 32],
        root: [u8; 32],
    },
    /// Persist a blob to content-addressed storage.
    PersistBlob { cid: [u8; 32], data: Vec<u8> },
}
```

The engine tracks:
- `personal_index: Option<[u8; 32]>` — CID of the local trie root
- `entries: BTreeMap<[u8; 32], IndexEntry>` — in-memory index entries (target_cid → tier3 vector + metadata ref)
- `collection_threshold: u32` — when entries exceed this, build/rebuild the trie (default 256)

Initial implementation: flat scan search (no trie yet). The trie-building logic (split when a collection exceeds 256 entries) can be deferred to a follow-up task.

**Tests:**
1. `engine_ingest_stores_entry` — ingest → entry exists
2. `engine_ingest_reject_blocks` — `IngestDecision::Reject` → no-op
3. `engine_ingest_privacy_tier_3_blocked` — encrypted-ephemeral → rejected regardless of decision
4. `engine_search_empty_returns_empty` — no entries → empty results
5. `engine_search_returns_nearest` — ingest 3 items, search, verify ranking by Hamming distance
6. `engine_evict_removes_expired` — lightweight entries with TTL → removed after expiry

**Step 2: Commit**
```bash
git add crates/harmony-oluo/src/
git commit -m "feat(oluo): OluoEngine sans-I/O state machine with flat-scan search"
```

---

### Task 11: Trie search traversal

**Context:** The trie traversal algorithm walks the binary space partition tree. At each internal node, check the query's bit at the split position. Descend into the matching child. Optionally check the far child if its centroid is close enough (backtracking). At leaves, scan the collection blob entries.

**Files:**
- Create: `crates/harmony-oluo/src/search.rs`
- Create: `crates/harmony-oluo/src/ranking.rs`
- Modify: `crates/harmony-oluo/src/lib.rs`

**Step 1: Write trie search**

Key function: Given a root trie node and a query vector, produce a list of `(target_cid, hamming_distance)` candidates. Since this is sans-I/O, the search emits `FetchTrieNode` actions when it needs to load children. The engine accumulates loaded nodes and resumes search when data arrives.

For the initial implementation, the search operates on pre-loaded collection blobs (personal search where all data is local). The iterative fetch pattern (for community/network search) is deferred.

```rust
/// Search a single collection blob for nearest neighbors.
pub fn scan_collection(
    collection: &CollectionBlob,
    query_tier3: &[u8; 32],
    max_results: usize,
) -> Vec<SearchHit> { ... }

/// A search hit with distance score.
pub struct SearchHit {
    pub target_cid: [u8; 32],
    pub distance: u32,
}
```

**Step 2: Write ranking module**

Convert `SearchHit` → `RawSearchResult` with normalized score (0.0 = identical, 1.0 = maximally different). Score = `distance as f32 / max_distance as f32` where `max_distance = tier_bit_count`.

**Tests:**
1. `scan_collection_finds_nearest` — 5 entries, query closest to entry 3 → entry 3 first
2. `scan_collection_respects_max_results` — max_results=2 → only 2 returned
3. `scan_collection_empty` — empty collection → empty results
4. `score_normalization` — distance 0 → score 0.0, distance 256 → score 1.0

**Step 3: Commit**
```bash
git add crates/harmony-oluo/src/
git commit -m "feat(oluo): trie search traversal + collection scanning + result ranking"
```

---

## Summary

| Task | Crate | Module(s) | Bead |
|------|-------|-----------|------|
| 1 | harmony-semantic | skeleton, error, tier | — |
| 2 | harmony-semantic | fingerprint, sidecar (v1) | harmony-e67 |
| 3 | harmony-semantic | distance, quantize | harmony-5s7 |
| 4 | harmony-semantic | collection | harmony-c8w |
| 5 | harmony-semantic | metadata, sidecar (v2) | — |
| 6 | harmony-semantic | overlay | — |
| 7 | harmony-oluo | skeleton, error, zenoh_keys | — |
| 8 | harmony-oluo | ingest, filter, scope | — |
| 9 | harmony-oluo | trie (node format) | — |
| 10 | harmony-oluo | engine | — |
| 11 | harmony-oluo | search, ranking | — |

**Total:** 11 tasks across 2 crates, ~20 source files, estimated 86+ tests.

**YAGNI deferred:**
- Network-wide search protocol (live Zenoh fanout)
- Encrypted index entries (Nakaiah integration)
- Trie split/rebuild (can use flat scan initially)
- Float32 re-ranking (binary-only sufficient)
- Community index sync protocol
- Self-tagging (athenaeum chunk integration)
- Basis vector navigation primitives (bead harmony-0oi, deferred)

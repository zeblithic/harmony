# Engram-as-CAS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sans-I/O `harmony-engram` crate that performs O(1) hash-based embedding lookups against sharded Engram tables stored as Harmony CAS books.

**Architecture:** Pure state-machine client — no network, no async, no runtime dependencies. Caller provides N-gram tokens, crate computes which shards to fetch and where to read. Caller handles network I/O (Zenoh queries), then passes shard bytes back for vector extraction and f16→f32 aggregation. Manifest serialization uses postcard for no_std portability.

**Tech Stack:** Rust (no_std + alloc), xxhash-rust (xxhash64), half (f16), postcard + serde (manifest serialization)

**Spec:** `docs/superpowers/specs/2026-03-24-engram-as-cas-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/harmony-engram/Cargo.toml` | Crate manifest with xxhash-rust, half, postcard, serde deps |
| `crates/harmony-engram/src/lib.rs` | Public API: `EngramConfig`, `EngramLookup`, `EngramClient` |
| `crates/harmony-engram/src/error.rs` | `EngramError` enum |
| `crates/harmony-engram/src/hash.rs` | xxhash64 multi-head hashing + shard resolution |
| `crates/harmony-engram/src/resolve.rs` | f16→f32 vector extraction and multi-head aggregation |
| `crates/harmony-engram/src/manifest.rs` | `ManifestHeader` postcard serialization |

### Modified Files

| File | Change |
|------|--------|
| `Cargo.toml` (workspace root) | Add `harmony-engram` member + `xxhash-rust` and `half` workspace deps |
| `crates/harmony-zenoh/src/namespace.rs` | Add `engram` namespace module |

---

### Task 1: Crate Scaffold + Core Types

**Files:**
- Create: `crates/harmony-engram/Cargo.toml`
- Create: `crates/harmony-engram/src/error.rs`
- Create: `crates/harmony-engram/src/lib.rs`
- Modify: `Cargo.toml` (workspace root)

- [ ] **Step 1: Add workspace dependencies**

Add `xxhash-rust` and `half` to `[workspace.dependencies]` in the workspace root `Cargo.toml`. Insert in the `# Utilities` section, keeping alphabetical order within the section. Also add `harmony-engram` to the `[workspace.dependencies]` internal crates section and to the `members` list (alphabetically between `harmony-discovery` and `harmony-jain`).

In `Cargo.toml` (workspace root), add to `members` list:

```rust
    "crates/harmony-engram",
```

In `[workspace.dependencies]`, in the `# Utilities` section:

```toml
half = { version = "2", default-features = false }
xxhash-rust = { version = "0.8", default-features = false, features = ["xxh64"] }
```

In `[workspace.dependencies]`, in the internal crates section:

```toml
harmony-engram = { path = "crates/harmony-engram", default-features = false }
```

- [ ] **Step 2: Create crate Cargo.toml**

```toml
[package]
name = "harmony-engram"
version.workspace = true
edition.workspace = true
license.workspace = true
rust-version.workspace = true

[features]
default = []
std = []

[dependencies]
half = { workspace = true }
postcard = { workspace = true }
serde = { workspace = true, features = ["derive"] }
xxhash-rust = { workspace = true }
```

No `std` feature propagation needed — all three deps are `no_std` by default.

- [ ] **Step 3: Create error.rs**

```rust
/// Errors that can occur during Engram operations.
#[derive(Debug)]
pub enum EngramError {
    /// Shard index exceeds the manifest's shard count.
    ShardIndexOutOfBounds { index: u64, num_shards: u64 },
    /// The number of shard data slices doesn't match the lookup's head count.
    ShardCountMismatch { expected: usize, got: usize },
    /// A shard's byte slice is too short to extract the vector at the given offset.
    ShardTooShort {
        shard_index: u64,
        offset: usize,
        vector_bytes: usize,
        shard_len: usize,
    },
    /// Manifest deserialization failed.
    ManifestDeserialize,
}

impl core::fmt::Display for EngramError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShardIndexOutOfBounds { index, num_shards } => {
                write!(f, "shard index {index} out of bounds (num_shards={num_shards})")
            }
            Self::ShardCountMismatch { expected, got } => {
                write!(f, "expected {expected} shard slices, got {got}")
            }
            Self::ShardTooShort {
                shard_index,
                offset,
                vector_bytes,
                shard_len,
            } => {
                write!(
                    f,
                    "shard {shard_index} too short: need {vector_bytes} bytes at offset {offset}, \
                     but shard is {shard_len} bytes"
                )
            }
            Self::ManifestDeserialize => write!(f, "failed to deserialize manifest header"),
        }
    }
}
```

- [ ] **Step 4: Create lib.rs with core types**

```rust
//! Sans-I/O client for Engram conditional memory table lookups.
//!
//! Engram tables are sharded embedding tables stored as Harmony CAS books.
//! This crate computes which shards to fetch and extracts/aggregates vectors
//! from fetched shard bytes.  The caller handles all network I/O.
//!
//! # Lookup flow
//!
//! 1. Call [`EngramClient::lookup`] with N-gram tokens → [`EngramLookup`]
//! 2. Fetch each shard by CID from the network (caller's responsibility)
//! 3. Call [`EngramClient::resolve`] with the lookup + shard bytes → aggregated f32 embedding

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{string::String, vec::Vec};

pub mod error;
pub mod hash;
pub mod manifest;
pub mod resolve;

pub use error::EngramError;
pub use manifest::ManifestHeader;

/// Configuration for an Engram table (derived from the manifest).
#[derive(Debug, Clone)]
pub struct EngramConfig {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Number of components per embedding vector.
    pub embedding_dim: usize,
    /// Bytes per component (2 for f16).
    pub dtype_bytes: usize,
    /// Number of independent hash heads.
    pub num_heads: u32,
    /// Number of embedding vectors per shard.
    pub shard_size: u32,
    /// Total number of shards in the table.
    pub num_shards: u64,
    /// Total number of entries across all shards.
    pub total_entries: u64,
    /// Per-head xxhash64 seeds.  Length must equal `num_heads`.
    pub hash_seeds: Vec<u64>,
}

impl EngramConfig {
    /// Bytes per embedding vector: `embedding_dim * dtype_bytes`.
    pub fn vector_bytes(&self) -> usize {
        self.embedding_dim * self.dtype_bytes
    }
}

/// Result of hashing an N-gram — which shards to fetch and where to read.
#[derive(Debug, Clone)]
pub struct EngramLookup {
    /// Shard index for each head (length = num_heads).
    pub shard_indices: Vec<u64>,
    /// Byte offset within each shard (length = num_heads).
    pub entry_offsets: Vec<usize>,
}

/// Sans-I/O client for performing Engram lookups.
///
/// Holds the table configuration and the shard-to-CID mapping.
/// All methods are pure computation — no I/O.
#[derive(Debug)]
pub struct EngramClient {
    config: EngramConfig,
    /// Shard index → CID (32-byte raw content identifier).
    manifest_cids: Vec<[u8; 32]>,
}

impl EngramClient {
    /// Construct a client from a parsed manifest.
    ///
    /// `shard_cids` must have exactly `config.num_shards` entries.
    pub fn from_manifest(config: EngramConfig, shard_cids: Vec<[u8; 32]>) -> Self {
        debug_assert_eq!(
            shard_cids.len() as u64,
            config.num_shards,
            "shard_cids length must match config.num_shards"
        );
        Self {
            config,
            manifest_cids: shard_cids,
        }
    }

    /// Access the table configuration.
    pub fn config(&self) -> &EngramConfig {
        &self.config
    }

    /// Get the CID for a shard index.
    ///
    /// Returns `None` if `shard_index >= num_shards`.
    pub fn shard_cid(&self, shard_index: u64) -> Option<&[u8; 32]> {
        self.manifest_cids.get(shard_index as usize)
    }

    /// Compute which shards and byte offsets are needed for an N-gram.
    ///
    /// Pure computation — no I/O.  The caller fetches each shard by its CID
    /// (via [`shard_cid`](Self::shard_cid)) and passes the bytes to
    /// [`resolve`](Self::resolve).
    pub fn lookup(&self, ngram_tokens: &[u32]) -> EngramLookup {
        hash::compute_lookup(&self.config, ngram_tokens)
    }

    /// Extract and aggregate embedding vectors from fetched shard bytes.
    ///
    /// `shard_data` must contain one byte slice per head, in the same order
    /// as `lookup.shard_indices`.  Each slice is the raw bytes of that shard
    /// (typically 64KB).
    ///
    /// Returns the summed embedding as f32 little-endian bytes
    /// (length = `embedding_dim * 4`).
    pub fn resolve(
        &self,
        lookup: &EngramLookup,
        shard_data: &[&[u8]],
    ) -> Result<Vec<u8>, EngramError> {
        resolve::aggregate(&self.config, lookup, shard_data)
    }
}
```

- [ ] **Step 5: Verify the crate compiles**

Run: `cargo check -p harmony-engram`
Expected: success (warnings about unused modules are fine — they'll be populated in later tasks)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-engram/ Cargo.toml
git commit -m "feat(engram): scaffold harmony-engram crate with core types

Add EngramConfig, EngramLookup, EngramClient structs and EngramError
enum. Sans-I/O client for Engram conditional memory table lookups.

Adds xxhash-rust and half as workspace dependencies."
```

---

### Task 2: xxhash64 Multi-Head Hashing + `lookup()`

**Files:**
- Create: `crates/harmony-engram/src/hash.rs`

**Reference docs:**
- Spec section "Multi-Head Hashing" — xxhash64 per-head seeds, table index mapping
- Spec section "Addressing" — shard_index = h/shard_size, byte_offset = (h%shard_size)*vector_bytes

- [ ] **Step 1: Write failing tests for hash determinism and shard resolution**

In `crates/harmony-engram/src/hash.rs`:

```rust
//! xxhash64 multi-head hashing and shard resolution.
//!
//! For each N-gram, computes `num_heads` independent xxhash64 values using
//! per-head seeds.  Each hash maps to a table index, which determines the
//! shard index and byte offset within that shard.

use alloc::vec::Vec;
use crate::{EngramConfig, EngramLookup};

/// Compute the [`EngramLookup`] for an N-gram given the table config.
///
/// 1. Encode tokens as little-endian bytes
/// 2. For each head seed: `table_index = xxhash64(bytes, seed) % total_entries`
/// 3. `shard_index = table_index / shard_size`
/// 4. `byte_offset = (table_index % shard_size) * vector_bytes`
pub fn compute_lookup(config: &EngramConfig, ngram_tokens: &[u32]) -> EngramLookup {
    todo!()
}

/// Hash an N-gram's token bytes with a single seed.
///
/// Tokens are encoded as contiguous little-endian u32 bytes.
fn hash_ngram(tokens: &[u32], seed: u64) -> u64 {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::String, vec};

    fn test_config() -> EngramConfig {
        EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 2,
            shard_size: 3,
            num_shards: 4,
            total_entries: 12, // 4 shards × 3 entries each
            hash_seeds: vec![42, 99],
        }
    }

    #[test]
    fn hash_determinism() {
        // Same input + seed must produce identical output across calls.
        let tokens = [1u32, 2, 3];
        let h1 = hash_ngram(&tokens, 42);
        let h2 = hash_ngram(&tokens, 42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_seeds_produce_different_hashes() {
        let tokens = [1u32, 2, 3];
        let h1 = hash_ngram(&tokens, 42);
        let h2 = hash_ngram(&tokens, 99);
        assert_ne!(h1, h2);
    }

    #[test]
    fn lookup_returns_correct_head_count() {
        let config = test_config();
        let lookup = compute_lookup(&config, &[1, 2, 3]);
        assert_eq!(lookup.shard_indices.len(), 2);
        assert_eq!(lookup.entry_offsets.len(), 2);
    }

    #[test]
    fn lookup_shard_indices_in_bounds() {
        let config = test_config();
        let lookup = compute_lookup(&config, &[100, 200, 300]);
        for &idx in &lookup.shard_indices {
            assert!(idx < config.num_shards, "shard index {idx} >= {}", config.num_shards);
        }
    }

    #[test]
    fn lookup_entry_offsets_in_bounds() {
        let config = test_config();
        let vector_bytes = config.vector_bytes(); // 4 * 2 = 8
        let shard_bytes = config.shard_size as usize * vector_bytes; // 3 * 8 = 24
        let lookup = compute_lookup(&config, &[7, 8, 9]);
        for &offset in &lookup.entry_offsets {
            assert!(
                offset + vector_bytes <= shard_bytes,
                "offset {offset} + {vector_bytes} exceeds shard size {shard_bytes}"
            );
        }
    }

    #[test]
    fn lookup_boundary_single_shard_table() {
        // Edge case: table with 1 shard, 1 entry.
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 2,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 1,
            num_shards: 1,
            total_entries: 1,
            hash_seeds: vec![0],
        };
        let lookup = compute_lookup(&config, &[42]);
        assert_eq!(lookup.shard_indices, vec![0]);
        assert_eq!(lookup.entry_offsets, vec![0]);
    }

    #[test]
    fn hash_empty_tokens() {
        // Empty N-gram is a degenerate case but must not panic.
        let config = test_config();
        let lookup = compute_lookup(&config, &[]);
        assert_eq!(lookup.shard_indices.len(), 2);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram hash::tests -- --no-capture`
Expected: FAIL — `todo!()` panics

- [ ] **Step 3: Implement hash_ngram() and compute_lookup()**

Replace the `todo!()` bodies in `crates/harmony-engram/src/hash.rs`:

```rust
fn hash_ngram(tokens: &[u32], seed: u64) -> u64 {
    // Encode tokens as contiguous little-endian bytes.
    // Stack buffer for N-grams up to 32 tokens (128 bytes).
    // Larger N-grams spill to Vec (rare in practice).
    let byte_len = tokens.len() * 4;
    if byte_len <= 128 {
        let mut buf = [0u8; 128];
        for (i, t) in tokens.iter().enumerate() {
            buf[i * 4..(i + 1) * 4].copy_from_slice(&t.to_le_bytes());
        }
        xxhash_rust::xxh64::xxh64(&buf[..byte_len], seed)
    } else {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        xxhash_rust::xxh64::xxh64(&bytes, seed)
    }
}

pub fn compute_lookup(config: &EngramConfig, ngram_tokens: &[u32]) -> EngramLookup {
    let vector_bytes = config.vector_bytes();
    let mut shard_indices = Vec::with_capacity(config.num_heads as usize);
    let mut entry_offsets = Vec::with_capacity(config.num_heads as usize);

    for seed in &config.hash_seeds {
        let raw_hash = hash_ngram(ngram_tokens, *seed);
        let table_index = raw_hash % config.total_entries;
        let shard_index = table_index / config.shard_size as u64;
        let entry_within_shard = (table_index % config.shard_size as u64) as usize;
        let byte_offset = entry_within_shard * vector_bytes;

        shard_indices.push(shard_index);
        entry_offsets.push(byte_offset);
    }

    EngramLookup {
        shard_indices,
        entry_offsets,
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-engram hash::tests -- --no-capture`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-engram/src/hash.rs
git commit -m "feat(engram): implement xxhash64 multi-head hashing

compute_lookup() hashes N-gram tokens with per-head seeds via xxhash64,
maps to table indices, and resolves shard index + byte offset.
Stack-allocated buffer for N-grams up to 32 tokens."
```

---

### Task 3: f16→f32 Vector Extraction + `resolve()`

**Files:**
- Create: `crates/harmony-engram/src/resolve.rs`

**Reference docs:**
- Spec section "Vector extraction" — direct byte slice, no deserialization
- Spec section "Aggregation" — decode f16→f32, sum component-wise

- [ ] **Step 1: Write failing tests for vector extraction and aggregation**

In `crates/harmony-engram/src/resolve.rs`:

```rust
//! f16→f32 vector extraction and multi-head aggregation.
//!
//! Each shard is a contiguous block of f16 embedding vectors.  Given a byte
//! offset, extract the vector, decode each f16 component to f32, and sum
//! across all heads in f32 precision to avoid f16 overflow.

use alloc::vec::Vec;
use half::f16;
use crate::{EngramConfig, EngramError, EngramLookup};

/// Extract and aggregate embedding vectors from fetched shard bytes.
///
/// For each head, extracts `embedding_dim` f16 values at the corresponding
/// byte offset, converts to f32, and sums all heads component-wise.
///
/// Returns f32 little-endian bytes (length = `embedding_dim * 4`).
pub fn aggregate(
    config: &EngramConfig,
    lookup: &EngramLookup,
    shard_data: &[&[u8]],
) -> Result<Vec<u8>, EngramError> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::String, vec};

    fn make_f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn test_config() -> EngramConfig {
        EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 2,
            shard_size: 3,
            num_shards: 4,
            total_entries: 12,
            hash_seeds: vec![42, 99],
        }
    }

    #[test]
    fn single_head_extraction() {
        // 1-head config, extract vector at offset 0.
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![0],
        };
        let result = aggregate(&config, &lookup, &[&shard]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn single_head_nonzero_offset() {
        // Extract from the second vector slot (offset = 8 bytes).
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        let shard = make_f16_bytes(&[
            0.0, 0.0, 0.0, 0.0, // slot 0
            5.0, 6.0, 7.0, 8.0, // slot 1 ← target
            0.0, 0.0, 0.0, 0.0, // slot 2
        ]);
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![8], // 1 * 4 * 2 = 8
        };
        let result = aggregate(&config, &lookup, &[&shard]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn multi_head_sum() {
        // 2 heads, different shards, sum component-wise.
        let config = test_config();
        let shard_a = make_f16_bytes(&[
            1.0, 2.0, 3.0, 4.0, // slot 0 ← head 0 reads here
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]);
        let shard_b = make_f16_bytes(&[
            0.0, 0.0, 0.0, 0.0,
            0.5, 1.5, 2.5, 3.5, // slot 1 ← head 1 reads here
            0.0, 0.0, 0.0, 0.0,
        ]);
        let lookup = EngramLookup {
            shard_indices: vec![0, 1],
            entry_offsets: vec![0, 8], // head 0 at offset 0, head 1 at offset 8
        };
        let result = aggregate(&config, &lookup, &[&shard_a, &shard_b]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn shard_count_mismatch() {
        let config = test_config();
        let lookup = EngramLookup {
            shard_indices: vec![0, 1],
            entry_offsets: vec![0, 0],
        };
        // Only provide 1 shard for a 2-head lookup.
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let err = aggregate(&config, &lookup, &[&shard]).unwrap_err();
        assert!(matches!(err, EngramError::ShardCountMismatch { expected: 2, got: 1 }));
    }

    #[test]
    fn shard_too_short() {
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        // Shard only has 4 bytes, but we need 8 at offset 0.
        let shard = [0u8; 4];
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![0],
        };
        let err = aggregate(&config, &lookup, &[&shard[..]]).unwrap_err();
        assert!(matches!(err, EngramError::ShardTooShort { .. }));
    }

    #[test]
    fn output_length() {
        let config = test_config(); // embedding_dim=4
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lookup = EngramLookup {
            shard_indices: vec![0, 0],
            entry_offsets: vec![0, 0],
        };
        let result = aggregate(&config, &lookup, &[&shard, &shard]).unwrap();
        // embedding_dim * 4 bytes per f32 = 4 * 4 = 16
        assert_eq!(result.len(), 16);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram resolve::tests -- --no-capture`
Expected: FAIL — `todo!()` panics

- [ ] **Step 3: Implement aggregate()**

Replace the `todo!()` body in `aggregate()`:

```rust
pub fn aggregate(
    config: &EngramConfig,
    lookup: &EngramLookup,
    shard_data: &[&[u8]],
) -> Result<Vec<u8>, EngramError> {
    let num_heads = lookup.shard_indices.len();
    if shard_data.len() != num_heads {
        return Err(EngramError::ShardCountMismatch {
            expected: num_heads,
            got: shard_data.len(),
        });
    }

    let dim = config.embedding_dim;
    let vector_bytes = config.vector_bytes();

    // Accumulator in f32 precision.
    let mut acc = vec![0.0f32; dim];

    for (i, (shard, &offset)) in shard_data.iter().zip(&lookup.entry_offsets).enumerate() {
        if offset + vector_bytes > shard.len() {
            return Err(EngramError::ShardTooShort {
                shard_index: lookup.shard_indices[i],
                offset,
                vector_bytes,
                shard_len: shard.len(),
            });
        }

        let slice = &shard[offset..offset + vector_bytes];
        for (j, chunk) in slice.chunks_exact(2).enumerate() {
            let val = f16::from_le_bytes([chunk[0], chunk[1]]);
            acc[j] += val.to_f32();
        }
    }

    // Encode accumulator as f32 little-endian bytes.
    let mut out = Vec::with_capacity(dim * 4);
    for val in &acc {
        out.extend_from_slice(&val.to_le_bytes());
    }
    Ok(out)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-engram resolve::tests -- --no-capture`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-engram/src/resolve.rs
git commit -m "feat(engram): f16-to-f32 vector extraction and multi-head aggregation

resolve::aggregate() extracts embedding vectors from shard bytes, decodes
f16 components to f32, and sums across all heads in f32 precision to
avoid f16 overflow/saturation."
```

---

### Task 4: Manifest Serialization

**Files:**
- Create: `crates/harmony-engram/src/manifest.rs`

**Reference docs:**
- Spec section "Manifest" — table metadata + shard CID list
- Spec: "serialized via postcard"

- [ ] **Step 1: Write failing tests for manifest round-trip**

In `crates/harmony-engram/src/manifest.rs`:

```rust
//! Manifest header serialization and deserialization.
//!
//! The manifest consists of a [`ManifestHeader`] (table metadata) plus a
//! shard CID list.  The header is serialized with postcard for no_std
//! portability.  The shard CID list is stored as raw 32-byte entries
//! (not postcard-encoded) because postcard's variable-length encoding
//! would bloat 50 million fixed-size entries.
//!
//! In a Harmony Merkle DAG, the manifest is stored as:
//! - Root bundle child 0: postcard-encoded [`ManifestHeader`]
//! - Root bundle children 1..N: raw shard CID chunks
//!
//! This module handles header (de)serialization.  DAG assembly is the
//! caller's concern (typically harmony-content's `dag::ingest`).

use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::{EngramConfig, EngramError};

/// Serializable manifest header — table metadata without the shard CID list.
///
/// Postcard-encoded for compact, no_std-friendly serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestHeader {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Number of components per embedding vector.
    pub embedding_dim: u32,
    /// Bytes per component (2 for f16).
    pub dtype_bytes: u16,
    /// Number of independent hash heads.
    pub num_heads: u32,
    /// Per-head xxhash64 seeds.
    pub hash_seeds: Vec<u64>,
    /// Total entries across all shards.
    pub total_entries: u64,
    /// Embeddings per shard.
    pub shard_size: u32,
    /// Total shard count.
    pub num_shards: u64,
}

impl ManifestHeader {
    /// Serialize to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EngramError> {
        todo!()
    }

    /// Deserialize from postcard bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, EngramError> {
        todo!()
    }

    /// Convert to an [`EngramConfig`].
    pub fn to_config(&self) -> EngramConfig {
        todo!()
    }
}

/// Parse raw shard CID bytes into a Vec of 32-byte CID arrays.
///
/// `data` must be a multiple of 32 bytes.  Returns one `[u8; 32]` per shard.
pub fn parse_shard_cids(data: &[u8]) -> Result<Vec<[u8; 32]>, EngramError> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn test_header() -> ManifestHeader {
        ManifestHeader {
            version: String::from("v1"),
            embedding_dim: 160,
            dtype_bytes: 2,
            num_heads: 4,
            hash_seeds: vec![111, 222, 333, 444],
            total_entries: 10_000_000_000,
            shard_size: 200,
            num_shards: 50_000_000,
        }
    }

    #[test]
    fn header_round_trip() {
        let header = test_header();
        let bytes = header.to_bytes().unwrap();
        let decoded = ManifestHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn header_to_config() {
        let header = test_header();
        let config = header.to_config();
        assert_eq!(config.version, "v1");
        assert_eq!(config.embedding_dim, 160);
        assert_eq!(config.dtype_bytes, 2);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.shard_size, 200);
        assert_eq!(config.num_shards, 50_000_000);
        assert_eq!(config.total_entries, 10_000_000_000);
        assert_eq!(config.hash_seeds, vec![111, 222, 333, 444]);
    }

    #[test]
    fn parse_shard_cids_basic() {
        // 2 CIDs, each 32 bytes.
        let mut data = vec![0u8; 64];
        data[0] = 0xAA;
        data[32] = 0xBB;
        let cids = parse_shard_cids(&data).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0][0], 0xAA);
        assert_eq!(cids[1][0], 0xBB);
    }

    #[test]
    fn parse_shard_cids_empty() {
        let cids = parse_shard_cids(&[]).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn parse_shard_cids_not_aligned() {
        // 33 bytes — not a multiple of 32.
        let data = vec![0u8; 33];
        let err = parse_shard_cids(&data).unwrap_err();
        assert!(matches!(err, EngramError::ManifestDeserialize));
    }

    #[test]
    fn invalid_bytes_returns_error() {
        let err = ManifestHeader::from_bytes(&[0xFF, 0xFF]).unwrap_err();
        assert!(matches!(err, EngramError::ManifestDeserialize));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram manifest::tests -- --no-capture`
Expected: FAIL — `todo!()` panics

- [ ] **Step 3: Implement ManifestHeader methods and parse_shard_cids**

Replace the `todo!()` bodies:

```rust
impl ManifestHeader {
    pub fn to_bytes(&self) -> Result<Vec<u8>, EngramError> {
        postcard::to_allocvec(self).map_err(|_| EngramError::ManifestDeserialize)
    }

    pub fn from_bytes(data: &[u8]) -> Result<Self, EngramError> {
        postcard::from_bytes(data).map_err(|_| EngramError::ManifestDeserialize)
    }

    pub fn to_config(&self) -> EngramConfig {
        EngramConfig {
            version: self.version.clone(),
            embedding_dim: self.embedding_dim as usize,
            dtype_bytes: self.dtype_bytes as usize,
            num_heads: self.num_heads,
            shard_size: self.shard_size,
            num_shards: self.num_shards,
            total_entries: self.total_entries,
            hash_seeds: self.hash_seeds.clone(),
        }
    }
}

pub fn parse_shard_cids(data: &[u8]) -> Result<Vec<[u8; 32]>, EngramError> {
    if data.len() % 32 != 0 {
        return Err(EngramError::ManifestDeserialize);
    }
    let count = data.len() / 32;
    let mut cids = Vec::with_capacity(count);
    for chunk in data.chunks_exact(32) {
        let mut cid = [0u8; 32];
        cid.copy_from_slice(chunk);
        cids.push(cid);
    }
    Ok(cids)
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-engram manifest::tests -- --no-capture`
Expected: all 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-engram/src/manifest.rs
git commit -m "feat(engram): manifest header serialization with postcard

ManifestHeader (de)serializes via postcard for no_std portability.
parse_shard_cids() reads raw 32-byte CID entries from the shard list
portion of the manifest DAG."
```

---

### Task 5: Zenoh Engram Namespace

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

**Reference docs:**
- Spec section "Zenoh Key Expressions":
  - `harmony/engram/{version}/manifest` — root CID of manifest DAG
  - `harmony/engram/{version}/shard/{index}` — individual shard queryable
- Existing namespace modules (content, compute, etc.) for pattern reference

- [ ] **Step 1: Write failing tests for engram namespace**

Add the following test block at the end of the `#[cfg(test)] mod tests` in `crates/harmony-zenoh/src/namespace.rs` (before the closing `}`):

```rust
    // ── Engram ───────────────────────────────────────────────────

    #[test]
    fn engram_manifest_key() {
        assert_eq!(
            engram::manifest_key("v1"),
            "harmony/engram/v1/manifest"
        );
    }

    #[test]
    fn engram_shard_key() {
        assert_eq!(
            engram::shard_key("v1", 42),
            "harmony/engram/v1/shard/42"
        );
    }

    #[test]
    fn engram_shard_key_zero() {
        assert_eq!(
            engram::shard_key("v1", 0),
            "harmony/engram/v1/shard/0"
        );
    }

    #[test]
    fn engram_shard_queryable() {
        assert_eq!(
            engram::shard_queryable("v1"),
            "harmony/engram/v1/shard/**"
        );
    }

    #[test]
    fn engram_subscription_pattern() {
        assert_eq!(engram::SUB, "harmony/engram/**");
    }
```

Also add `engram::PREFIX` to the `all_prefixes_start_with_root` test's `prefixes` array.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh -- engram --no-capture`
Expected: FAIL — `engram` module doesn't exist yet

- [ ] **Step 3: Implement the engram namespace module**

Add the following module in `crates/harmony-zenoh/src/namespace.rs`, after the `page` module and before the `#[cfg(test)]` block:

```rust
/// Engram conditional memory table key expressions.
///
/// Namespace: `harmony/engram/{version}/`
///
/// Engram tables are sharded embedding tables stored as CAS books.
/// Nodes hosting shards declare queryables on `harmony/engram/{version}/shard/**`.
/// Any node holding a cached shard can respond.
pub mod engram {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/engram`
    pub const PREFIX: &str = "harmony/engram";

    /// Subscribe to all engram traffic: `harmony/engram/**`
    pub const SUB: &str = "harmony/engram/**";

    // ── Builders ────────────────────────────────────────────────

    /// Manifest key: `harmony/engram/{version}/manifest`
    pub fn manifest_key(version: &str) -> String {
        format!("{PREFIX}/{version}/manifest")
    }

    /// Individual shard key: `harmony/engram/{version}/shard/{index}`
    pub fn shard_key(version: &str, index: u64) -> String {
        format!("{PREFIX}/{version}/shard/{index}")
    }

    /// Shard queryable pattern: `harmony/engram/{version}/shard/**`
    ///
    /// Nodes hosting Engram shards register this pattern.
    pub fn shard_queryable(version: &str) -> String {
        format!("{PREFIX}/{version}/shard/**")
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh -- engram --no-capture`
Expected: all 5 engram tests PASS

Run: `cargo test -p harmony-zenoh -- all_prefixes_start_with_root --no-capture`
Expected: PASS (with engram::PREFIX added to the assertion array)

- [ ] **Step 5: Run full workspace check**

Run: `cargo test -p harmony-engram && cargo test -p harmony-zenoh && cargo clippy -p harmony-engram -p harmony-zenoh`
Expected: all tests pass, no clippy warnings

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add engram namespace for shard queryables

harmony/engram/{version}/manifest — root CID of the manifest DAG
harmony/engram/{version}/shard/{index} — individual shard queryable

Nodes hosting Engram shards declare queryables on the shard pattern.
Any node holding a cached shard can respond."
```

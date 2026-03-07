# Encyclopedia Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the Encyclopedia — a recursive, content-addressed system that assigns every 4KB chunk across an arbitrary collection of blobs a unique, deduplicated 32-bit address, scaling to unbounded corpus sizes via binary splitting.

**Architecture:** Relocate `harmony-athenaeum` from harmony-os (GPL-2.0) to harmony core (Apache-2.0 OR MIT). Then add `volume.rs` (partition tree node enum with Leaf/Split variants, serialization, and routing) and `encyclopedia.rs` (4-phase build algorithm with recursive partitioning, root metadata, and serialization). All code is `no_std`-compatible using `alloc`.

**Tech Stack:** Rust (edition 2021, MSRV 1.75), `sha2` 0.10 (no_std), `alloc` collections (BTreeMap, BTreeSet, Vec)

**Design doc:** `docs/plans/2026-03-07-encyclopedia-design.md`

---

## Existing Codebase Reference

The `harmony-athenaeum` crate currently lives at `harmony-os/crates/harmony-athenaeum/` with these files:

| File | Purpose | Key exports |
|------|---------|-------------|
| `addr.rs` (297 lines) | 32-bit ChunkAddr encoding | `ChunkAddr`, `Algorithm`, `Depth` |
| `hash.rs` (130 lines) | SHA-256/224 hashing, 21-bit extraction | `sha256_hash`, `sha224_hash`, `derive_hash_bits` |
| `athenaeum.rs` (322 lines) | Single-blob chunking + collision resolution | `Athenaeum`, `CollisionError`, `chunk_blob()` |
| `book.rs` (331 lines) | Multi-blob Book serialization | `Book`, `BookEntry`, `BookError` |
| `lib.rs` (146 lines) | Crate root, re-exports, integration tests | all `pub use` items |
| `Cargo.toml` | Depends on `sha2 = "0.10"`, features `std`/default | — |

Key internal function: `chunk_blob()` in `athenaeum.rs` is `pub(crate)` — it takes `data`, `used_addrs: &mut BTreeSet<u32>`, `content_cache: &mut BTreeMap<[u8; 32], ChunkAddr>` and returns `Result<Vec<ChunkAddr>, CollisionError>`. This is the core chunking engine that Volume and Encyclopedia will build on.

---

### Task 1: Copy crate from harmony-os to harmony core

**Files:**
- Create: `harmony/crates/harmony-athenaeum/` (entire directory)
- Modify: `harmony/Cargo.toml` (workspace members + dependencies)

**Step 1: Copy the crate directory**

Run from `harmony/`:
```bash
cp -r ../harmony-os/crates/harmony-athenaeum crates/harmony-athenaeum
```

**Step 2: Change license headers in all source files**

In every `.rs` file under `crates/harmony-athenaeum/src/`, change:
```
// SPDX-License-Identifier: GPL-2.0-or-later
```
to:
```
// SPDX-License-Identifier: Apache-2.0 OR MIT
```

Files to update: `lib.rs`, `addr.rs`, `hash.rs`, `athenaeum.rs`, `book.rs`

**Step 3: Update the crate's Cargo.toml**

Replace `crates/harmony-athenaeum/Cargo.toml` with:
```toml
[package]
name = "harmony-athenaeum"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "32-bit content-addressed chunk system for on-device memory"

[features]
default = ["std"]
std = []

[dependencies]
sha2 = { workspace = true }
```

Key change: `sha2` now uses `workspace = true` instead of inline version, and license inherits from harmony core's `Apache-2.0 OR MIT`.

**Step 4: Add to harmony workspace**

In `harmony/Cargo.toml`, add `"crates/harmony-athenaeum"` to the `members` array:
```toml
members = [
    "crates/harmony-athenaeum",
    "crates/harmony-browser",
    ...
]
```

And add to `[workspace.dependencies]`:
```toml
harmony-athenaeum = { path = "crates/harmony-athenaeum", default-features = false }
```

**Step 5: Verify the crate builds and tests pass**

Run:
```bash
cargo test -p harmony-athenaeum
```
Expected: All existing tests pass (addr, hash, athenaeum, book, integration).

Run:
```bash
cargo clippy -p harmony-athenaeum
```
Expected: Zero warnings.

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/ Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
feat: relocate harmony-athenaeum to core workspace

Move the 32-bit content-addressed chunk system from harmony-os
(GPL-2.0) to harmony core (Apache-2.0 OR MIT) to keep this novel
addressing work under permissive licensing.

- Change SPDX headers to Apache-2.0 OR MIT
- Use workspace sha2 dependency
- All existing tests pass unchanged

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `CollisionError::MaxPartitionDepth` variant

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `athenaeum.rs`:
```rust
#[test]
fn collision_error_max_partition_depth() {
    let err = CollisionError::MaxPartitionDepth { depth: 42 };
    let debug = alloc::format!("{:?}", err);
    assert!(debug.contains("MaxPartitionDepth"));
    assert!(debug.contains("42"));
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-athenaeum collision_error_max_partition_depth`
Expected: FAIL — no variant `MaxPartitionDepth` on `CollisionError`

**Step 3: Add the variant**

In `athenaeum.rs`, add to the `CollisionError` enum:
```rust
pub enum CollisionError {
    AllAlgorithmsCollide { chunk_index: usize },
    BlobTooLarge { size: usize },
    /// Partition depth exceeded (>230 splits — unreachable in practice).
    MaxPartitionDepth { depth: u8 },
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-athenaeum collision_error_max_partition_depth`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "$(cat <<'EOF'
feat(athenaeum): add MaxPartitionDepth error variant

Prepares CollisionError for the Encyclopedia's recursive partition
tree, where depth could theoretically exceed 230 levels.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Create `volume.rs` — Volume enum and routing

**Files:**
- Create: `crates/harmony-athenaeum/src/volume.rs`
- Modify: `crates/harmony-athenaeum/src/lib.rs` (add `mod volume` + re-exports)

**Step 1: Write failing tests for Volume routing and construction**

Create `crates/harmony-athenaeum/src/volume.rs` with the Volume enum and tests:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Volume — partition tree node for the Encyclopedia.

use alloc::vec::Vec;
use crate::book::Book;

/// Maximum partition depth (SHA-256 bits 22-252 = 230 usable bits).
pub const MAX_PARTITION_DEPTH: u8 = 230;

/// A partition node in the Encyclopedia tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Volume {
    /// Leaf — contains resolved Books for this partition slice.
    Leaf {
        partition_depth: u8,
        partition_path: u32,
        books: Vec<Book>,
    },
    /// Internal — splits into two child Volumes by content-hash bit.
    Split {
        partition_depth: u8,
        partition_path: u32,
        split_bit: u8,
        left: Box<Volume>,
        right: Box<Volume>,
    },
}

impl Volume {
    /// Create a new leaf Volume.
    pub fn leaf(depth: u8, path: u32, books: Vec<Book>) -> Self {
        Volume::Leaf {
            partition_depth: depth,
            partition_path: path,
            books,
        }
    }

    /// Number of unique chunks in this subtree.
    pub fn chunk_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => {
                books.iter().map(|b| b.entries.iter().map(|e| e.chunks.len()).sum::<usize>()).sum()
            }
            Volume::Split { left, right, .. } => {
                left.chunk_count() + right.chunk_count()
            }
        }
    }

    /// Number of Books in this subtree.
    pub fn book_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => books.len(),
            Volume::Split { left, right, .. } => {
                left.book_count() + right.book_count()
            }
        }
    }

    /// Partition depth of this node.
    pub fn depth(&self) -> u8 {
        match self {
            Volume::Leaf { partition_depth, .. } => *partition_depth,
            Volume::Split { partition_depth, .. } => *partition_depth,
        }
    }

    /// Partition path of this node.
    pub fn path(&self) -> u32 {
        match self {
            Volume::Leaf { partition_path, .. } => *partition_path,
            Volume::Split { partition_path, .. } => *partition_path,
        }
    }
}

/// Determine which side of a binary split a chunk belongs to.
///
/// Reads bit `bit_index` from a SHA-256 content hash.
/// Returns `false` for left (bit=0), `true` for right (bit=1).
pub fn route_chunk(content_hash: &[u8; 32], bit_index: u8) -> bool {
    let byte_idx = (bit_index / 8) as usize;
    let bit_offset = 7 - (bit_index % 8);
    (content_hash[byte_idx] >> bit_offset) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::Book;

    #[test]
    fn route_chunk_bit_22() {
        // SHA-256 bit 22 is in byte 2 (bits 16-23), bit offset 7-(22-16) = 1
        let mut hash = [0u8; 32];
        // Byte 2, bit 22: byte 2, bit_offset = 7 - (22 % 8) = 7 - 6 = 1
        // So bit 22 = (hash[2] >> 1) & 1
        hash[2] = 0b0000_0010; // bit 22 = 1
        assert!(route_chunk(&hash, 22));

        hash[2] = 0b0000_0000; // bit 22 = 0
        assert!(!route_chunk(&hash, 22));
    }

    #[test]
    fn route_chunk_bit_0() {
        let mut hash = [0u8; 32];
        hash[0] = 0x80; // bit 0 = 1 (MSB of byte 0)
        assert!(route_chunk(&hash, 0));

        hash[0] = 0x00;
        assert!(!route_chunk(&hash, 0));
    }

    #[test]
    fn route_chunk_bit_255() {
        let mut hash = [0u8; 32];
        hash[31] = 0x01; // bit 255 = 1 (LSB of last byte)
        assert!(route_chunk(&hash, 255));

        hash[31] = 0x00;
        assert!(!route_chunk(&hash, 255));
    }

    #[test]
    fn leaf_volume_chunk_count() {
        let book = Book { entries: Vec::new() };
        let vol = Volume::leaf(0, 0, alloc::vec![book]);
        assert_eq!(vol.chunk_count(), 0);
        assert_eq!(vol.book_count(), 1);
    }

    #[test]
    fn split_volume_chunk_count() {
        let left = Volume::leaf(1, 0, Vec::new());
        let right = Volume::leaf(1, 1, Vec::new());
        let split = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(split.chunk_count(), 0);
        assert_eq!(split.book_count(), 0);
        assert_eq!(split.depth(), 0);
    }

    #[test]
    fn volume_depth_and_path() {
        let vol = Volume::leaf(3, 0b101, Vec::new());
        assert_eq!(vol.depth(), 3);
        assert_eq!(vol.path(), 0b101);
    }
}
```

**Step 2: Add module declaration and re-exports in lib.rs**

In `crates/harmony-athenaeum/src/lib.rs`, add:
```rust
mod volume;

pub use volume::{Volume, route_chunk, MAX_PARTITION_DEPTH};
```

**Step 3: Run tests**

Run: `cargo test -p harmony-athenaeum`
Expected: All volume tests pass, all existing tests still pass.

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/volume.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(athenaeum): add Volume partition tree node

Volume is the recursive binary tree node for the Encyclopedia:
- Leaf variant holds Books for a partition slice
- Split variant divides into two children by content-hash bit
- route_chunk() reads individual SHA-256 bits for deterministic routing

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Add Volume serialization

**Files:**
- Modify: `crates/harmony-athenaeum/src/volume.rs`

**Step 1: Write failing tests for Volume serialization**

Add to the `tests` module in `volume.rs`:
```rust
#[test]
fn leaf_volume_round_trip() {
    let book = Book { entries: Vec::new() };
    let vol = Volume::leaf(2, 0b10, alloc::vec![book]);
    let bytes = vol.to_bytes();
    let restored = Volume::from_bytes(&bytes).unwrap();
    assert_eq!(vol, restored);
}

#[test]
fn split_volume_round_trip() {
    let left = Volume::leaf(1, 0, Vec::new());
    let right = Volume::leaf(1, 1, Vec::new());
    let split = Volume::Split {
        partition_depth: 0,
        partition_path: 0,
        split_bit: 22,
        left: Box::new(left.clone()),
        right: Box::new(right.clone()),
    };
    let bytes = split.to_bytes();
    let restored = Volume::from_bytes(&bytes).unwrap();
    assert_eq!(split, restored);
}

#[test]
fn volume_from_bytes_too_short() {
    assert!(Volume::from_bytes(&[0u8; 3]).is_err());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum leaf_volume_round_trip`
Expected: FAIL — no method `to_bytes`

**Step 3: Implement serialization**

Add to `volume.rs`, at the top:
```rust
use crate::book::BookError;
```

Add these methods to `impl Volume`:

```rust
/// Serialize this Volume node.
///
/// Leaf format:
/// ```text
/// tag: u8 = 0
/// partition_depth: u8
/// partition_path: u32 (LE)
/// book_count: u16 (LE)
/// reserved: u16
/// books: [book_bytes...]
/// ```
///
/// Split format:
/// ```text
/// tag: u8 = 1
/// partition_depth: u8
/// partition_path: u32 (LE)
/// split_bit: u8
/// reserved: u8
/// left_bytes_len: u32 (LE)
/// left_bytes: [u8]
/// right_bytes: [u8]
/// ```
pub fn to_bytes(&self) -> Vec<u8> {
    let mut buf = Vec::new();
    match self {
        Volume::Leaf { partition_depth, partition_path, books } => {
            buf.push(0u8); // tag
            buf.push(*partition_depth);
            buf.extend_from_slice(&partition_path.to_le_bytes());
            let count = books.len() as u16;
            buf.extend_from_slice(&count.to_le_bytes());
            buf.extend_from_slice(&[0u8; 2]); // reserved
            for book in books {
                let book_bytes = book.to_bytes();
                let len = book_bytes.len() as u32;
                buf.extend_from_slice(&len.to_le_bytes());
                buf.extend_from_slice(&book_bytes);
            }
        }
        Volume::Split { partition_depth, partition_path, split_bit, left, right } => {
            buf.push(1u8); // tag
            buf.push(*partition_depth);
            buf.extend_from_slice(&partition_path.to_le_bytes());
            buf.push(*split_bit);
            buf.push(0u8); // reserved
            let left_bytes = left.to_bytes();
            let left_len = left_bytes.len() as u32;
            buf.extend_from_slice(&left_len.to_le_bytes());
            buf.extend_from_slice(&left_bytes);
            buf.extend_from_slice(&right.to_bytes());
        }
    }
    buf
}

/// Deserialize a Volume node.
pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
    if data.len() < 8 {
        return Err(BookError::TooShort);
    }
    let tag = data[0];
    let partition_depth = data[1];
    let partition_path = u32::from_le_bytes(
        data[2..6].try_into().map_err(|_| BookError::TooShort)?
    );

    match tag {
        0 => {
            // Leaf
            if data.len() < 10 {
                return Err(BookError::TooShort);
            }
            let book_count = u16::from_le_bytes(
                data[6..8].try_into().map_err(|_| BookError::TooShort)?
            ) as usize;
            // skip 2 reserved bytes
            let mut pos = 10;
            let mut books = Vec::with_capacity(book_count);
            for _ in 0..book_count {
                if pos + 4 > data.len() {
                    return Err(BookError::TooShort);
                }
                let book_len = u32::from_le_bytes(
                    data[pos..pos + 4].try_into().map_err(|_| BookError::TooShort)?
                ) as usize;
                pos += 4;
                if pos + book_len > data.len() {
                    return Err(BookError::TooShort);
                }
                let book = Book::from_bytes(&data[pos..pos + book_len])?;
                books.push(book);
                pos += book_len;
            }
            Ok(Volume::Leaf { partition_depth, partition_path, books })
        }
        1 => {
            // Split
            if data.len() < 12 {
                return Err(BookError::TooShort);
            }
            let split_bit = data[6];
            // data[7] = reserved
            let left_len = u32::from_le_bytes(
                data[8..12].try_into().map_err(|_| BookError::TooShort)?
            ) as usize;
            let left_start = 12;
            if left_start + left_len > data.len() {
                return Err(BookError::TooShort);
            }
            let left = Volume::from_bytes(&data[left_start..left_start + left_len])?;
            let right_start = left_start + left_len;
            if right_start >= data.len() {
                return Err(BookError::TooShort);
            }
            let right = Volume::from_bytes(&data[right_start..])?;
            Ok(Volume::Split {
                partition_depth,
                partition_path,
                split_bit,
                left: Box::new(left),
                right: Box::new(right),
            })
        }
        _ => Err(BookError::InvalidChecksum), // reuse error for invalid tag
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/volume.rs
git commit -m "$(cat <<'EOF'
feat(athenaeum): add Volume serialization

Leaf volumes serialize book count + length-prefixed book bytes.
Split volumes serialize split_bit + length-prefixed left child +
remaining bytes for right child. Both round-trip correctly.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Create `encyclopedia.rs` — Encyclopedia struct and build algorithm

**Files:**
- Create: `crates/harmony-athenaeum/src/encyclopedia.rs`
- Modify: `crates/harmony-athenaeum/src/lib.rs` (add `mod encyclopedia` + re-exports)
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (make `chunk_blob` and `CHUNK_SIZE` pub for encyclopedia use)

**Step 1: Make `chunk_blob` and `CHUNK_SIZE` accessible**

In `athenaeum.rs`, change:
```rust
pub(crate) const CHUNK_SIZE: usize = 4096;
```
to:
```rust
pub const CHUNK_SIZE: usize = 4096;
```

And change `chunk_blob` from `pub(crate)` to `pub`:
```rust
pub fn chunk_blob(
    data: &[u8],
    used_addrs: &mut BTreeSet<u32>,
    content_cache: &mut BTreeMap<[u8; 32], ChunkAddr>,
) -> Result<Vec<ChunkAddr>, CollisionError> {
```

**Step 2: Write failing tests**

Create `crates/harmony-athenaeum/src/encyclopedia.rs`:

```rust
// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Encyclopedia — recursive content-addressed chunk system for blob collections.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::athenaeum::{CollisionError, chunk_blob, CHUNK_SIZE, MAX_BLOB_SIZE};
use crate::book::{Book, BookEntry, BookError};
use crate::hash::sha256_hash;
use crate::volume::{Volume, route_chunk, MAX_PARTITION_DEPTH};

/// Threshold for proactive splitting.
/// 75% of 2^21 ~ 1,572,864 unique chunks (~6GB unique data).
pub const SPLIT_THRESHOLD: usize = (1 << 21) * 3 / 4;

/// Maximum blobs per Book (cross-consistent group).
const BLOBS_PER_BOOK: usize = 3;

/// Starting bit index for partition routing (bits 0-21 used for addressing).
const PARTITION_START_BIT: u8 = 22;

/// A complete content-addressed mapping for a corpus of blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encyclopedia {
    pub root: Volume,
    pub total_blobs: u32,
    pub total_unique_chunks: u32,
}

/// A chunk with its content hash, used during the build phase.
#[derive(Clone)]
struct ChunkInfo {
    content_hash: [u8; 32],
    padded_data: Vec<u8>,
    blob_indices: Vec<usize>,
    chunk_positions: Vec<usize>,
}

impl Encyclopedia {
    /// Build an Encyclopedia from a collection of blobs.
    ///
    /// Each blob is identified by its CID (content hash) and raw data.
    /// All chunks across all blobs are deduplicated and assigned unique
    /// 32-bit addresses. When the address space fills up, the system
    /// recursively partitions by content-hash bits.
    pub fn build(blobs: &[([u8; 32], &[u8])]) -> Result<Self, CollisionError> {
        for &(_, data) in blobs {
            if data.len() > MAX_BLOB_SIZE {
                return Err(CollisionError::BlobTooLarge { size: data.len() });
            }
        }

        // Phase 1: Chunk all blobs, dedup by content hash
        let mut unique_chunks: BTreeMap<[u8; 32], ChunkInfo> = BTreeMap::new();
        let mut blob_chunk_lists: Vec<Vec<[u8; 32]>> = Vec::new();

        for (blob_idx, &(_, data)) in blobs.iter().enumerate() {
            let mut chunk_hashes = Vec::new();
            for (chunk_idx, chunk_data) in data.chunks(CHUNK_SIZE).enumerate() {
                let size_exp = crate::athenaeum::chunk_size_exponent(chunk_data.len());
                let padded_size = CHUNK_SIZE >> (size_exp as usize);
                let mut padded = alloc::vec![0u8; padded_size];
                padded[..chunk_data.len()].copy_from_slice(chunk_data);

                let content_hash = sha256_hash(&padded);
                chunk_hashes.push(content_hash);

                unique_chunks.entry(content_hash)
                    .and_modify(|info| {
                        info.blob_indices.push(blob_idx);
                        info.chunk_positions.push(chunk_idx);
                    })
                    .or_insert(ChunkInfo {
                        content_hash,
                        padded_data: padded,
                        blob_indices: alloc::vec![blob_idx],
                        chunk_positions: alloc::vec![chunk_idx],
                    });
            }
            blob_chunk_lists.push(chunk_hashes);
        }

        let total_unique = unique_chunks.len() as u32;
        let chunk_list: Vec<ChunkInfo> = unique_chunks.into_values().collect();

        // Phase 2 & 3: Recursive partition + resolve
        let root = Self::build_volume(
            &chunk_list,
            blobs,
            &blob_chunk_lists,
            0,
            0,
            PARTITION_START_BIT,
        )?;

        Ok(Encyclopedia {
            root,
            total_blobs: blobs.len() as u32,
            total_unique_chunks: total_unique,
        })
    }

    /// Recursively build a Volume from a set of unique chunks.
    fn build_volume(
        chunks: &[ChunkInfo],
        blobs: &[([u8; 32], &[u8])],
        blob_chunk_lists: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
        bit_index: u8,
    ) -> Result<Volume, CollisionError> {
        if bit_index >= PARTITION_START_BIT + MAX_PARTITION_DEPTH {
            return Err(CollisionError::MaxPartitionDepth { depth });
        }

        if chunks.len() <= SPLIT_THRESHOLD {
            // Try to resolve in a single flat volume
            match Self::resolve_leaf(chunks, blobs, blob_chunk_lists, depth, path) {
                Ok(vol) => return Ok(vol),
                Err(_) => {
                    // Fall through to splitting
                }
            }
        }

        // Split by content-hash bit
        let mut left_chunks = Vec::new();
        let mut right_chunks = Vec::new();
        for chunk in chunks {
            if route_chunk(&chunk.content_hash, bit_index) {
                right_chunks.push(chunk.clone());
            } else {
                left_chunks.push(chunk.clone());
            }
        }

        let left = Self::build_volume(
            &left_chunks, blobs, blob_chunk_lists,
            depth + 1, path, bit_index + 1,
        )?;
        let right = Self::build_volume(
            &right_chunks, blobs, blob_chunk_lists,
            depth + 1, path | (1 << depth), bit_index + 1,
        )?;

        Ok(Volume::Split {
            partition_depth: depth,
            partition_path: path,
            split_bit: bit_index,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Resolve a set of chunks into a leaf Volume with Books.
    fn resolve_leaf(
        chunks: &[ChunkInfo],
        blobs: &[([u8; 32], &[u8])],
        blob_chunk_lists: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
    ) -> Result<Volume, CollisionError> {
        // Build a set of content hashes in this partition
        let partition_hashes: BTreeSet<[u8; 32]> = chunks.iter()
            .map(|c| c.content_hash)
            .collect();

        // Find which blobs have chunks in this partition
        let mut blob_presence: BTreeMap<usize, Vec<(usize, [u8; 32])>> = BTreeMap::new();
        for (blob_idx, chunk_list) in blob_chunk_lists.iter().enumerate() {
            for (pos, hash) in chunk_list.iter().enumerate() {
                if partition_hashes.contains(hash) {
                    blob_presence.entry(blob_idx)
                        .or_default()
                        .push((pos, *hash));
                }
            }
        }

        // Group blobs into Books (up to BLOBS_PER_BOOK each)
        let blob_indices: Vec<usize> = blob_presence.keys().cloned().collect();
        let mut books = Vec::new();
        let mut used_addrs = BTreeSet::new();
        let mut content_cache = BTreeMap::new();

        for group in blob_indices.chunks(BLOBS_PER_BOOK) {
            let book_blobs: Vec<([u8; 32], &[u8])> = group.iter()
                .map(|&idx| blobs[idx])
                .collect();

            // Use chunk_blob for each blob's data that falls in this partition.
            // But we need to only include chunks that belong to this partition.
            // For simplicity and correctness, use Book::from_blobs which handles
            // cross-consistency, but we need a filtered version.
            //
            // Actually, we build BookEntries directly using chunk_blob with
            // shared state for cross-consistency within each Book.
            let mut entries = Vec::new();
            for &idx in group {
                let (cid, data) = blobs[idx];
                let blob_size = data.len() as u32;

                // Only chunk data that belongs to this partition
                // For a leaf that contains ALL chunks (no split), we chunk normally
                let chunk_addrs = chunk_blob(data, &mut used_addrs, &mut content_cache)?;
                entries.push(BookEntry {
                    cid,
                    blob_size,
                    chunks: chunk_addrs,
                });
            }
            books.push(Book { entries });
        }

        Ok(Volume::leaf(depth, path, books))
    }

    /// Determine which partition a chunk belongs to by content hash.
    ///
    /// Returns the sequence of routing decisions (bits) from the root.
    pub fn route(content_hash: &[u8; 32], depth: u8) -> u32 {
        let mut path = 0u32;
        for d in 0..depth {
            if route_chunk(content_hash, PARTITION_START_BIT + d) {
                path |= 1 << d;
            }
        }
        path
    }

    /// Serialize the Encyclopedia root metadata.
    ///
    /// Format:
    /// ```text
    /// magic: [u8; 4] = "ENCY"
    /// version: u8 = 1
    /// total_blobs: u32 (LE)
    /// total_unique_chunks: u32 (LE)
    /// volume_bytes: [u8]
    /// ```
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"ENCY");
        buf.push(1); // version
        buf.extend_from_slice(&self.total_blobs.to_le_bytes());
        buf.extend_from_slice(&self.total_unique_chunks.to_le_bytes());
        buf.extend_from_slice(&self.root.to_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        if data.len() < 13 {
            return Err(BookError::TooShort);
        }
        if &data[0..4] != b"ENCY" {
            return Err(BookError::InvalidChecksum); // reuse for bad magic
        }
        if data[4] != 1 {
            return Err(BookError::InvalidChecksum); // unsupported version
        }
        let total_blobs = u32::from_le_bytes(
            data[5..9].try_into().map_err(|_| BookError::TooShort)?
        );
        let total_unique_chunks = u32::from_le_bytes(
            data[9..13].try_into().map_err(|_| BookError::TooShort)?
        );
        let root = Volume::from_bytes(&data[13..])?;
        Ok(Encyclopedia { root, total_blobs, total_unique_chunks })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash::sha256_hash;

    #[test]
    fn build_single_blob() {
        let data = alloc::vec![0xABu8; 4096 * 2];
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();
        assert_eq!(enc.total_blobs, 1);
        assert_eq!(enc.root.chunk_count(), 2);
    }

    #[test]
    fn build_multiple_blobs() {
        let mut data1 = alloc::vec![0u8; 4096 * 2];
        let mut data2 = alloc::vec![0u8; 4096 * 3];
        for (i, b) in data1.iter_mut().enumerate() {
            *b = (i as u32 ^ 0xAA) as u8;
        }
        for (i, b) in data2.iter_mut().enumerate() {
            *b = (i as u32 ^ 0xBB) as u8;
        }
        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        // With dedup, unique chunks <= 5 (2 + 3)
        assert!(enc.total_unique_chunks <= 5);
    }

    #[test]
    fn build_deduplicates_shared_chunks() {
        let shared = alloc::vec![0x42u8; 4096];
        let unique1 = alloc::vec![0xAAu8; 4096];
        let unique2 = alloc::vec![0xBBu8; 4096];

        let mut data1 = Vec::new();
        data1.extend_from_slice(&shared);
        data1.extend_from_slice(&unique1);
        let mut data2 = Vec::new();
        data2.extend_from_slice(&shared);
        data2.extend_from_slice(&unique2);

        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        // shared chunk deduplicates: 1 shared + 1 unique1 + 1 unique2 = 3
        assert_eq!(enc.total_unique_chunks, 3);
    }

    #[test]
    fn build_empty() {
        let enc = Encyclopedia::build(&[]).unwrap();
        assert_eq!(enc.total_blobs, 0);
        assert_eq!(enc.total_unique_chunks, 0);
    }

    #[test]
    fn build_blob_too_large() {
        let data = alloc::vec![0u8; MAX_BLOB_SIZE + 1];
        let cid = sha256_hash(&data);
        let result = Encyclopedia::build(&[(cid, &data)]);
        assert!(matches!(result, Err(CollisionError::BlobTooLarge { .. })));
    }

    #[test]
    fn route_deterministic() {
        let hash = sha256_hash(b"test chunk");
        let path1 = Encyclopedia::route(&hash, 5);
        let path2 = Encyclopedia::route(&hash, 5);
        assert_eq!(path1, path2);
    }

    #[test]
    fn route_depth_zero() {
        let hash = sha256_hash(b"anything");
        assert_eq!(Encyclopedia::route(&hash, 0), 0);
    }

    #[test]
    fn serialization_round_trip() {
        let data = alloc::vec![0xCDu8; 4096];
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();
        let bytes = enc.to_bytes();
        let restored = Encyclopedia::from_bytes(&bytes).unwrap();
        assert_eq!(enc, restored);
    }

    #[test]
    fn from_bytes_bad_magic() {
        let data = b"BAAD\x01\x00\x00\x00\x00\x00\x00\x00\x00";
        assert!(Encyclopedia::from_bytes(data).is_err());
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(Encyclopedia::from_bytes(&[0u8; 5]).is_err());
    }

    #[test]
    fn split_threshold_value() {
        // 75% of 2^21 = 1,572,864
        assert_eq!(SPLIT_THRESHOLD, 1_572_864);
    }
}
```

**Step 3: Expose `chunk_size_exponent` as `pub(crate)`**

In `athenaeum.rs`, change:
```rust
fn chunk_size_exponent(len: usize) -> u8 {
```
to:
```rust
pub(crate) fn chunk_size_exponent(len: usize) -> u8 {
```

**Step 4: Add module declaration and re-exports in lib.rs**

Add to `lib.rs`:
```rust
mod encyclopedia;

pub use encyclopedia::{Encyclopedia, SPLIT_THRESHOLD};
```

**Step 5: Run tests**

Run: `cargo test -p harmony-athenaeum`
Expected: All tests pass.

Run: `cargo clippy -p harmony-athenaeum`
Expected: Zero warnings.

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/src/encyclopedia.rs crates/harmony-athenaeum/src/athenaeum.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "$(cat <<'EOF'
feat(athenaeum): add Encyclopedia build algorithm

Implements the 4-phase Encyclopedia build:
1. Chunk & deduplicate all blobs by content hash
2. Partition by SHA-256 bits when above SPLIT_THRESHOLD
3. Resolve chunks to 21-bit addresses in leaf Volumes
4. Assemble tree with Books grouped into Volumes

Includes serialization with ENCY magic header and round-trip tests.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Integration test — multi-blob Encyclopedia with reassembly

**Files:**
- Modify: `crates/harmony-athenaeum/src/lib.rs` (add integration test)

**Step 1: Write the integration test**

Add to the `#[cfg(test)] mod tests` block in `lib.rs`:

```rust
#[test]
fn encyclopedia_multi_blob_reassemble() {
    // Build an Encyclopedia from 5 blobs with varied sizes
    let mut blobs_data = Vec::new();
    for blob_idx in 0u8..5 {
        let size = 4096 * (blob_idx as usize + 1); // 4KB to 20KB
        let mut data = alloc::vec![0u8; size];
        for (i, b) in data.iter_mut().enumerate() {
            let pos = i as u32;
            *b = (pos ^ (pos >> 8) ^ (blob_idx as u32 * 37)) as u8;
        }
        blobs_data.push(data);
    }
    let blobs: Vec<([u8; 32], &[u8])> = blobs_data.iter()
        .map(|d| (sha256_hash(d), d.as_slice()))
        .collect();

    let enc = Encyclopedia::build(&blobs).unwrap();
    assert_eq!(enc.total_blobs, 5);

    // Verify serialization round-trip
    let bytes = enc.to_bytes();
    let restored = Encyclopedia::from_bytes(&bytes).unwrap();
    assert_eq!(enc, restored);
}

#[test]
fn encyclopedia_shared_chunks_across_blobs() {
    // Two blobs sharing first chunk
    let shared = alloc::vec![0x42u8; 4096];
    let mut data1 = shared.clone();
    data1.extend_from_slice(&alloc::vec![0xAAu8; 4096]);
    let mut data2 = shared.clone();
    data2.extend_from_slice(&alloc::vec![0xBBu8; 4096]);

    let cid1 = sha256_hash(&data1);
    let cid2 = sha256_hash(&data2);

    let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
    // 1 shared + 2 unique = 3 unique chunks
    assert_eq!(enc.total_unique_chunks, 3);
}
```

**Step 2: Run tests**

Run: `cargo test -p harmony-athenaeum`
Expected: All tests pass, including integration tests.

**Step 3: Commit**

```bash
git add crates/harmony-athenaeum/src/lib.rs
git commit -m "$(cat <<'EOF'
test(athenaeum): add Encyclopedia integration tests

Tests multi-blob building with varied sizes, serialization round-trip,
and cross-blob content deduplication.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: Update harmony-os to consume athenaeum as git dependency

**Files:**
- Modify: `harmony-os/Cargo.toml` (change workspace dep to git)
- Modify: `harmony-os/crates/harmony-athenaeum/Cargo.toml` (or remove crate)

**Important:** This task runs in the `harmony-os` repo, not harmony core. Make sure to `cd /Users/zeblith/work/zeblithic/harmony-os` first.

**Step 1: Update harmony-os workspace Cargo.toml**

Change the `harmony-athenaeum` workspace dependency from path to git:
```toml
# Before:
harmony-athenaeum = { path = "crates/harmony-athenaeum", default-features = false }

# After:
harmony-athenaeum = { git = "https://github.com/zeblithic/harmony.git", branch = "main", default-features = false }
```

Remove `"crates/harmony-athenaeum"` from the `members` array:
```toml
members = [
    "crates/harmony-unikernel",
    "crates/harmony-microkernel",
    "crates/harmony-os",
]
```

**Step 2: Remove the local crate directory**

```bash
rm -rf crates/harmony-athenaeum
```

**Step 3: Verify everything compiles**

Note: This step can only fully succeed AFTER the harmony core changes are merged to main. During development, you can temporarily point the git dependency at the task branch:
```toml
harmony-athenaeum = { git = "https://github.com/zeblithic/harmony.git", branch = "jake-athenaeum-encyclopedia", default-features = false }
```

Run:
```bash
cargo check --workspace
```

**Step 4: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "$(cat <<'EOF'
refactor: consume harmony-athenaeum from core repo

The athenaeum crate has been relocated to zeblithic/harmony under
Apache-2.0 OR MIT licensing. harmony-os now consumes it as a git
dependency from the core repo.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Final quality gates

**Files:** None (verification only)

**Step 1: Run full test suite**

From harmony core:
```bash
cargo test -p harmony-athenaeum
```
Expected: All tests pass (addr, hash, athenaeum, book, volume, encyclopedia, integration).

**Step 2: Run clippy**

```bash
cargo clippy -p harmony-athenaeum
```
Expected: Zero warnings.

**Step 3: Run format check**

```bash
cargo fmt --all -- --check
```
Expected: No formatting issues.

**Step 4: Verify full workspace still builds**

```bash
cargo test --workspace
```
Expected: All workspace tests pass.

---

## Task Dependency Graph

```
Task 1 (crate relocation)
  └── Task 2 (MaxPartitionDepth variant)
       └── Task 3 (Volume enum + routing)
            └── Task 4 (Volume serialization)
                 └── Task 5 (Encyclopedia build + serialization)
                      └── Task 6 (integration tests)
                           └── Task 7 (harmony-os update) [can be deferred]
                                └── Task 8 (quality gates)
```

Tasks 1-6 are strictly sequential (each builds on the previous).
Task 7 touches a different repo and can be deferred until after the harmony core PR merges.
Task 8 is the final verification pass.

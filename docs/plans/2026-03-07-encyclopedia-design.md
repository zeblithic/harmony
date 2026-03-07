# Encyclopedia Design

**Status:** Approved design
**Date:** 2026-03-07
**Scope:** harmony (crates/harmony-athenaeum)
**Bead:** harmony-c1e

## Goal

Design and implement the Encyclopedia — a recursive, content-addressed
system that assigns every 4KB chunk across an arbitrary collection of
blobs a unique, deduplicated 32-bit address. When collisions become
intractable in the 21-bit address space, the system binary-splits into
Volumes using content-hash bits, scaling to unbounded corpus sizes.

Also relocate the `harmony-athenaeum` crate from `harmony-os` (GPL-2.0)
to `harmony` core (Apache-2.0 OR MIT) to keep this novel work under
permissive licensing.

## Problem

The Athenaeum (already implemented) translates a single 1MB blob into
32-bit-addressed chunks. The Book extends this to 3 blobs with
cross-consistent addressing. But real-world use cases need to address
*collections* of blobs — a DVD's worth, a hard drive's worth, or
potentially exabytes of distributed data — with the same properties:
content-deduplication, 32-bit local addressing, and deterministic
routing without global coordination.

The 21-bit address space (~2M slots) with 4 algorithm choices handles
up to ~8GB of unique data in a single flat space. Beyond that, we need
a scaling mechanism that preserves the CAS properties.

## Architecture

The Encyclopedia is a recursive binary tree of address-space partitions.
Each leaf owns a non-overlapping slice of chunks, all addressed in the
same 21-bit space. Partition routing is determined entirely by
content-hash bits — fully deterministic, embarrassingly parallel, no
global state required.

```
Encyclopedia (root)
+-- Volume A (SHA-256 bit 22 = 0)
|   +-- Book: blobs 1,2,3 -> chunk addresses [...]
|   +-- Book: blobs 4,5 -> chunk addresses [...]
+-- Volume B (SHA-256 bit 22 = 1)
    +-- Volume B.0 (SHA-256 bit 23 = 0)    <- further split needed
    |   +-- Book: blobs 6,7,8 -> chunk addresses [...]
    +-- Volume B.1 (SHA-256 bit 23 = 1)
        +-- Book: blobs 9,10 -> chunk addresses [...]
```

### Naming Hierarchy

| Level | Name | Contains | Scale |
|-------|------|----------|-------|
| Single blob | Athenaeum | Ordered chunk addresses for one CID | up to 1MB |
| 1-3 blobs | Book | Cross-consistent chunk mappings | up to 3MB |
| Arbitrary | Volume | Books (leaf) or child Volumes (internal) | up to ~8GB per leaf |
| Corpus | Encyclopedia | Root Volume + metadata | unbounded |

### Chunk Routing Rule

To find which Volume a chunk belongs to, read SHA-256 bits 22, 23,
24... of its content hash, one per tree level. Each bit selects left
(0) or right (1) child. This is a trie over content-hash bits.

**Any device can route any chunk to its Volume independently** — the
chunk's own hash contains the routing decision. No coordination, no
build state, no global index required.

### Scale

Each tree level doubles capacity:

| Depth | Unique chunks | Data (4KB each) |
|-------|--------------|-----------------|
| 0 | ~2M | ~8GB |
| 10 | ~2B | ~8TB |
| 20 | ~2T | ~8PB |
| 30 | ~2,000T | ~8EB |

The tree grows as needed, never pre-allocated. With ~230 available
partition bits from a SHA-256 digest (bits 22-252), the theoretical
maximum depth is 230 levels — far beyond any foreseeable data volume.

## Build Algorithm

### Phase 1: Chunk & Deduplicate

Split all input blobs into 4KB chunks. SHA-256 hash each chunk to get
its full 32-byte digest. Deduplicate: identical chunks across all blobs
collapse to a single entry.

### Phase 2: Partition (recursive)

```
if unique_count <= SPLIT_THRESHOLD:
    try to resolve all chunks in one Volume (flat)
    if intractable collision:
        fall back to splitting
else:
    partition by content-hash bit (bit 22 + depth):
        bit = 0 -> left child Volume
        bit = 1 -> right child Volume
    recurse each half independently (bit 23, 24, ...)
```

**SPLIT_THRESHOLD:** 75% of 2^21 = ~1,572,864 unique chunks (~6GB of
unique data). Below this, flat resolution almost never fails. Above
it, proactive splitting avoids wasted work trying to place too many
chunks before discovering a split is needed.

### Phase 3: Resolve (per leaf Volume)

For each unique chunk in a leaf Volume, assign a 21-bit address using
the 4-algorithm power-of-choice collision resolution (existing logic
from `chunk_blob()`). Group CID-to-chunk mappings into Books (up to 3
blobs per Book, cross-consistent within each Book).

### Phase 4: Assemble

Each leaf Volume contains its Books. Internal nodes reference their two
children by CID. The root Volume becomes the Encyclopedia's entry point.
Content-address every node — it's CAS all the way down.

### Parallel Build

Since each Volume resolves independently after partitioning, a large
build can fan out to N workers. The partition itself is embarrassingly
parallel (each chunk routes itself by its own hash). The only sequential
part is assembling the tree at the end — linking content-addressed
references.

## Serialization

### Volume Metadata

Leaf Volume:

```
+---------------------------+
| partition_depth: u8       | 1 byte
| partition_path: u32       | 4 bytes  (bits taken to reach this node)
| book_count: u16           | 2 bytes
| reserved: u8              | 1 byte
| book_cid[0]: [u8; 32]    | \
| book_cid[1]: [u8; 32]    |  } content-addressed references to Books
| ...                       | /
+---------------------------+
```

Internal Volume:

```
+---------------------------+
| partition_depth: u8       | 1 byte
| partition_path: u32       | 4 bytes
| split_bit_index: u8       | 1 byte  (which SHA-256 bit: 22, 23, ...)
| reserved: u16             | 2 bytes
| left_cid: [u8; 32]       | 32 bytes  (child Volume with bit = 0)
| right_cid: [u8; 32]      | 32 bytes  (child Volume with bit = 1)
+---------------------------+
```

A leaf Volume with 100 Books = 8 + 100x32 = 3,208 bytes — fits in a
single 4KB chunk.

### Encyclopedia Root

```
+---------------------------+
| magic: [u8; 4]            | "ENCY"
| version: u8               | 1
| total_blobs: u32          | 4 bytes
| total_unique_chunks: u32  | 4 bytes
| root_volume_cid: [u8; 32] | 32 bytes
+---------------------------+
```

45 bytes — fits in the smallest mini-blob. The entire Encyclopedia is
reachable from this single root CID.

### Depth Field

The existing 2-bit depth field in ChunkAddr stays focused on
hardware-level nesting (Depth 0 = data, Depth 1 = Bundle). The
Encyclopedia tree is a logical structure above that — Volume and
Encyclopedia metadata are stored as regular Depth 0 blobs, identified
by their CIDs. CAS references do the job naturally without overloading
the depth field.

## Crate Relocation

Move `harmony-athenaeum` from `harmony-os/crates/` to `harmony/crates/`:

- Change license headers from `GPL-2.0-or-later` to `Apache-2.0 OR MIT`
- Reuse harmony core's existing workspace `sha2` dependency
- Keep `sha2` as the direct dependency (simpler than coupling to
  `harmony-crypto` for just hash digests)
- Update `harmony-os` to consume `harmony-athenaeum` as a git
  dependency from the core repo

### Module Structure (after relocation + Encyclopedia)

```
harmony-athenaeum/src/
  lib.rs          (existing + new re-exports)
  addr.rs         (existing, unchanged)
  hash.rs         (existing, unchanged)
  athenaeum.rs    (existing, unchanged)
  book.rs         (existing, unchanged)
  volume.rs       (NEW - partition tree node)
  encyclopedia.rs (NEW - build algorithm + root metadata)
```

## Public API (sketch)

```rust
/// Threshold for proactive splitting.
/// 75% of 2^21 ~ 1,572,864 unique chunks (~6GB unique data).
pub const SPLIT_THRESHOLD: usize = (1 << 21) * 3 / 4;

/// A partition node in the Encyclopedia tree.
pub enum Volume {
    /// Leaf - contains resolved Books.
    Leaf {
        partition_depth: u8,
        partition_path: u32,
        books: Vec<Book>,
    },
    /// Internal - splits into two child Volumes.
    Split {
        partition_depth: u8,
        partition_path: u32,
        split_bit: u8,
        left: Box<Volume>,   // chunk's SHA-256 bit = 0
        right: Box<Volume>,  // chunk's SHA-256 bit = 1
    },
}

/// A complete content-addressed mapping for a corpus of blobs.
pub struct Encyclopedia {
    pub root: Volume,
    pub total_blobs: u32,
    pub total_unique_chunks: u32,
}

impl Encyclopedia {
    /// Build an Encyclopedia from a collection of blobs.
    pub fn build(
        blobs: &[([u8; 32], &[u8])],
    ) -> Result<Self, CollisionError>;

    /// Determine which Volume a chunk belongs to by content hash.
    pub fn route(content_hash: &[u8; 32], depth: u8) -> u32;

    /// Serialize the Encyclopedia root metadata.
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ParseError>;
}

impl Volume {
    /// Serialize this Volume node.
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize a Volume node.
    pub fn from_bytes(data: &[u8]) -> Result<Self, ParseError>;

    /// Number of unique chunks in this subtree.
    pub fn chunk_count(&self) -> usize;
}
```

### Error Handling

```rust
pub enum CollisionError {
    AllAlgorithmsCollide { chunk_index: usize },
    BlobTooLarge { size: usize },
    /// Partition depth exceeded (>230 splits - unreachable in practice).
    MaxPartitionDepth { depth: u8 },
}
```

## What This Design Does NOT Cover

- **Incremental updates** — Adding/removing blobs from an existing
  Encyclopedia. Rebuild for now; CAS makes this correct if not fast.
  Incremental is a future optimization.
- **Distributed build coordination** — The algorithm is parallel-safe
  but we don't include a coordination protocol. Each worker can
  independently resolve its assigned Volume.
- **CID query index** — "Which Volume has blob X?" is a linear scan.
  A CID-to-Volume index is a future optimization.
- **Encyclopedia composition** — Merging two independently-built
  Encyclopedias is a future composition problem.
- **Storage backend** — How Volumes/Books are persisted (memory, disk,
  network cache). That's the caller's concern.

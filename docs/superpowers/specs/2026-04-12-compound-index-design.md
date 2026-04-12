# CompoundIndex: Base + Delta CAS-Aware Index Management (ZEB-105, Phase 2)

## Problem

USearch indexes are mutable HNSW graphs. Harmony's storage is immutable CAS (BLAKE3-hashed blobs). Inserting a vector changes the graph, but re-serializing and re-hashing the entire index for every insertion is prohibitively expensive on edge devices. We need a pattern that bridges mutable graphs with immutable storage.

## Solution

Split the index into two layers: an immutable Base (memory-mapped from a CAS blob, read-only, zero-copy) and a mutable Delta (small, in-RAM, holds recent additions). Searches query both and fuse results. Periodic compaction merges Delta into Base, serializes, and returns bytes for the caller to CAS-store. The cost of hashing and network propagation is amortized over thousands of insertions.

## CompoundIndex API

```rust
pub struct CompoundIndex {
    base: Option<VectorIndex>,  // memory-mapped, read-only
    delta: VectorIndex,         // in-RAM, mutable
    config: VectorIndexConfig,
    compact_threshold: usize,
}
```

### Core methods

- `new(config, compact_threshold)` — Empty delta, no base.
- `add(key, vector)` — Insert into delta. O(log N) via HNSW.
- `search(query, k)` — Search base and delta, merge by distance, deduplicate by key (delta wins for updates), return top-k.
- `should_compact()` — `delta.len() >= compact_threshold`.
- `compact()` — Load base as writable, insert all delta vectors, serialize to bytes, reset delta. Returns `Vec<u8>` for caller to CAS-store. Then caller provides the file path back via `load_base()`.
- `load_base(path)` — Memory-map from file (primary path, zero-copy).
- `load_base_from_bytes(bytes)` — Write to temp file, then memory-map (for bootstrap/tests).
- `len()` — Total vectors across base + delta.
- `delta_len()` — Pending vectors in delta.

### Search merging

When both base and delta exist:
1. Search base for top-k results
2. Search delta for top-k results
3. Merge both lists sorted by distance
4. Deduplicate by key (if same key in both, keep delta's entry)
5. Return top-k from merged list

### Compaction workflow

1. Load base as writable: `VectorIndex::load(path, config)` (not `.view()` — needs to be mutable)
2. Insert all delta vectors into the writable base
3. Save to temp file, read serialized bytes
4. Return bytes to caller
5. Reset delta to empty
6. Caller CAS-stores bytes, gets CID, broadcasts via Zenoh
7. Caller provides new file path → `load_base(path)` swaps to new memory-mapped view

During compaction, the old base view remains active for concurrent searches. The swap to the new base is atomic (replace the `Option<VectorIndex>`).

### Memory during compaction

Temporary spike: the base exists as both a read-only view (for ongoing searches) and a writable copy (for delta insertion). For typical index sizes (30-100MB), this temporary doubling fits within RPi5's 8GB budget. The spike is brief — compaction takes under a second for 1K delta insertions.

## CAS-agnostic design

CompoundIndex does NOT depend on harmony-content. It provides:
- `compact() -> Vec<u8>` — raw bytes, caller hashes and stores
- `load_base(path)` / `load_base_from_bytes(bytes)` — caller provides data

The CAS bridge is a few lines in the consumer (oluo, engram):
```
let bytes = compound.compact()?;
let cid = blake3::hash(&bytes);
cas_store(cid, bytes);
compound.load_base(&local_path_for(cid));
```

## How oluo consumes this (Phase 3 preview)

OluoEngine holds a CompoundIndex + a side-table `HashMap<u64, SidecarMetadata>` for metadata:

```
Ingest → compound.add(key, vector) + metadata_table.insert(key, metadata)
Search → compound.search(query, k) → look up metadata for each match
Compact → compound.compact() → emit OluoAction::PersistBlob { bytes }
LoadBase → compound.load_base(path) → on startup or after receiving new CID
```

The sans-I/O pattern is preserved: compaction is async, triggered by a hint (`should_compact()`), executed outside the event loop.

## Files to modify

1. **`crates/harmony-search/src/compound.rs`** (new) — CompoundIndex implementation
2. **`crates/harmony-search/src/lib.rs`** — re-export CompoundIndex
3. **`crates/harmony-search/src/index.rs`** — may need `serialize_to_bytes()` / `load_from_bytes()` helpers on VectorIndex
4. **Tests** — compound search merging, compaction round-trip, base loading

## Success criteria

- CompoundIndex search returns correct results from both base and delta
- Compaction produces bytes that can be loaded back as a new base
- Delta is empty after compaction
- Memory-mapped base doesn't copy data into RAM
- All operations work without any CAS dependency

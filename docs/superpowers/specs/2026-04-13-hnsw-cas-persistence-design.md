# HNSW CAS Persistence Design (ZEB-110)

## Overview

Persist the harmony-oluo HNSW search index as CAS blobs so it survives restarts and can sync between peers via the same DAG mechanism as harmony-db. The OluoEngine remains a sans-I/O state machine — it emits persistence actions and the caller handles all CAS/disk I/O.

## Design Decisions

- **Independent lifecycle:** Oluo persists its own state, not tied to harmony-db commits. The two systems can be linked later by storing a pointer in the db manifest.
- **Accept delta loss:** Only compacted state survives restarts. Entries ingested since the last compaction are lost. The caller re-ingests from its source of truth (harmony-db entries, incoming messages). The default compaction threshold is 1000 entries, bounding worst-case loss.
- **DAG ingest for large blobs:** The compacted HNSW index can be 10-50MB for 10k vectors. `harmony-content::dag::ingest()` chunks it into ~512KB CAS books automatically. No custom chunking code.
- **Postcard for metadata:** The metadata sidecar is serialized with postcard (already a workspace dependency) using deterministic key order (entries sorted by u64 key via BTreeMap before encoding) to ensure stable CAS CIDs. At ~100 bytes per entry, 10k entries yields ~1MB — fits in one or two CAS books.
- **Sans-I/O preserved:** OluoEngine emits a `PersistSnapshot` action after compaction. The caller does all DAG ingest, manifest construction, and head-file management.
- **Three-level pointer chain:** Local head file → snapshot manifest CID → (index CID + metadata CID). Same pattern as harmony-db's `index.json` → commit manifest → table roots.

## Snapshot Structure

```text
oluo_head.json              ← local file, stores one manifest CID
  └─ SnapshotManifest       ← CAS book (~100 bytes, postcard)
       ├─ index_cid         ← DAG root for HNSW bytes (multi-MB, chunked)
       └─ metadata_cid      ← DAG root for postcard metadata map
```

## Data Model

### SnapshotManifest

Serialized with postcard, stored as a single CAS book:

```rust
#[derive(Serialize, Deserialize)]
struct SnapshotManifest {
    version: u32,            // SNAPSHOT_VERSION = 1
    index_cid: [u8; 32],    // DAG root CID for HNSW index bytes
    metadata_cid: [u8; 32], // DAG root CID for metadata sidecar
    key_counter: u64,        // next key to assign on resume
    compact_generation: u64, // compaction generation counter
}
```

### Metadata Sidecar

The existing `EntryMetadata` gets `Serialize`/`Deserialize` derives. The metadata map is sorted by key (collected into a `BTreeMap<u64, EntryMetadata>`) before postcard serialization to ensure deterministic byte output and stable CAS CIDs. The serialized bytes are then DAG-ingested into CAS.

### Local Head File

`oluo_head.json` in the data directory:

```json
{"head": "hex-encoded-manifest-cid"}
```

Same pattern as harmony-db's `index.json`.

## OluoEngine API Changes

### New Action Variant

`OluoAction::CompactRequest` is replaced by `PersistSnapshot`:

```rust
pub enum OluoAction {
    // ... existing IndexUpdated, SearchResults, Error ...

    /// Compacted index ready for persistence. Caller should:
    /// 1. DAG-ingest index_bytes and metadata_bytes into CAS
    /// 2. Build SnapshotManifest with the resulting CIDs
    /// 3. Write manifest to CAS, update local head file
    /// 4. Send CompactComplete back to the engine
    PersistSnapshot {
        index_bytes: Vec<u8>,
        metadata_bytes: Vec<u8>,
        key_counter: u64,
        generation: u64,
    },
}
```

### New Constructor for Restore

```rust
impl OluoEngine {
    /// Restore from a persisted snapshot.
    pub fn from_snapshot(
        index_bytes: &[u8],
        metadata: HashMap<u64, EntryMetadata>,
        key_counter: u64,
        generation: u64,
        compact_threshold: usize,
    ) -> Result<Self, SearchError> {
        // 1. Create CompoundIndex with compact_threshold, load base from index_bytes
        // 2. Clamp key_counter to max(existing_keys) + 1
        // 3. Derive scope_counts by iterating metadata
        // 4. Derive cid_to_key from metadata target_cid fields
        // 5. Set has_compacted = true (we have a base)
    }
}
```

### Visibility Changes

- `EntryMetadata` — `pub(crate)` → `pub` (callers must deserialize it for `from_snapshot`)
- `SearchScope` — add `serde::Serialize, serde::Deserialize` derives
- `EntryMetadata` fields — add `serde::Serialize, serde::Deserialize` derives

### Unchanged

- `OluoEvent::CompactComplete { path, generation }` — reused as-is. The caller writes the `index_bytes` to a local file (for memory-mapping) and sends this event with the file path. The engine calls `load_base(path)` to swap to the mmap'd base, freeing the in-memory copy.
- `OluoEvent::Ingest` — unchanged; delta entries accumulate until next compaction
- Compaction trigger logic — unchanged

## Persistence Flow

### Write Path (compaction + persist)

```text
OluoEngine                           Caller (app layer)
──────────                           ──────────────────
delta reaches threshold
compact() merges base+delta
serialize metadata to postcard
emit PersistSnapshot {
  index_bytes, metadata_bytes,  ───> 1. dag::ingest(index_bytes, store) → index_cid
  key_counter, generation            2. dag::ingest(metadata_bytes, store) → metadata_cid
}                                    3. Write index_bytes to local file for mmap
                                     4. Build SnapshotManifest
                                     5. Serialize manifest, store as CAS book → manifest_cid
                                     6. Write manifest_cid to oluo_head.json
                                     7. Send CompactComplete { path, generation }
                              <─────
load_base(path) swaps to mmap'd
```

### Read Path (restore from snapshot)

```text
Caller (app layer)                   OluoEngine
──────────────────                   ──────────
1. Read manifest_cid from oluo_head.json
2. Fetch manifest from CAS
3. Deserialize SnapshotManifest
4. dag::reassemble(index_cid, store) → index_bytes
5. dag::reassemble(metadata_cid, store) → metadata_bytes
6. Deserialize metadata_bytes → HashMap
7. OluoEngine::from_snapshot(
     &index_bytes, metadata,    ───> Load CompoundIndex base
     key_counter, generation,        Clamp key_counter, derive state
     compact_threshold               Ready for queries + ingests
   )
```

### Crash Recovery

If the process crashes between compaction and writing `oluo_head.json`, the head file still points to the previous snapshot. CAS blobs from the incomplete write are orphans (harmless — future GC can clean them). On restart, the engine loads the last good snapshot and the caller re-ingests entries added since.

## Files Changed

### harmony-oluo

- `crates/harmony-oluo/Cargo.toml` — add `serde` and `postcard` workspace dependencies
- `crates/harmony-oluo/src/engine.rs` — `EntryMetadata` visibility + serde derives, `PersistSnapshot` action variant (replacing `CompactRequest`), `from_snapshot()` constructor, metadata serialization in compaction handler
- `crates/harmony-oluo/src/scope.rs` — add `Serialize, Deserialize` derives to `SearchScope`

### No changes to other crates

harmony-search, harmony-content, and harmony-db are unchanged. DAG ingest, CAS storage, and head-file management are all caller responsibilities using existing APIs.

## Derived State on Restore

These fields are not serialized — they are derived from the metadata HashMap during `from_snapshot`:

- `scope_counts: [usize; 3]` — iterate metadata values, count per `entry.scope.index()`
- `cid_to_key: HashMap<[u8; 32], u64>` — iterate metadata, map `entry.target_cid → key`
- `has_compacted` — set to `true` (we loaded a base)

## Testing Strategy

### Unit tests (harmony-oluo)

All tests are within harmony-oluo — no CAS or filesystem dependencies (sans-I/O):

- **Snapshot round-trip:** Create engine, ingest entries across multiple scopes, trigger compaction, capture `PersistSnapshot` payload, construct new engine via `from_snapshot`, verify identical search results and scope filtering
- **Metadata serialization:** Postcard round-trip for `HashMap<u64, EntryMetadata>` with varied entry types (different scopes, expiry values, overlay CIDs)
- **Derived state correctness:** After `from_snapshot`, verify `scope_counts` matches actual metadata distribution and `cid_to_key` correctly maps all target CIDs
- **Key counter continuity:** After restore, new ingests get keys strictly above the restored `key_counter`
- **Generation continuity:** After restore, next compaction uses generation > restored generation
- **Re-ingest after restore:** Ingest entries that existed pre-snapshot (same CIDs), verify dedup still works via the derived `cid_to_key`

### Not in scope for automated tests

- Actual CAS/DAG ingest and reassembly (caller responsibility, tested at application layer)
- Head file I/O (trivial JSON read/write, caller responsibility)

## Future Extensions (not in this spec)

- **harmony-db manifest pointer:** Add optional `index_cid` to `CommitManifest` so sync gets both db and search state
- **Delta WAL:** Append-only log for delta entries to survive crashes without waiting for compaction
- **Incremental metadata sync:** Per-entry metadata CIDs for structural sharing across compaction cycles
- **Index GC:** Garbage-collect orphaned CAS blobs from incomplete snapshots

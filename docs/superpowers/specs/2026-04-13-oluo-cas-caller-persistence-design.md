# Oluo CAS Caller-Side Persistence Design (ZEB-110 Part 2)

## Overview

Wire the caller-side persistence for OluoEngine's HNSW search index. The engine (sans-I/O) already emits `PersistSnapshot` actions and accepts `from_snapshot()` restoration (PR #223). This spec covers the code that catches those actions, DAG-ingests into CAS, manages the local head file, and restores on startup.

## Design Decisions

- **New `persist` module in harmony-oluo** — co-located with the engine, following harmony-db's pattern (`persist.rs` alongside `db.rs`). harmony-oluo gains a dependency on harmony-content.
- **Stateless free functions** — `persist_snapshot()` and `load_snapshot()` take `data_dir` + `&mut dyn BookStore` as parameters. No struct, no state. Mirrors `dag::ingest`'s own API style.
- **Postcard for manifest (CAS), JSON for head file (local)** — Manifest is compact binary in CAS. Head file is human-readable JSON for debugging.
- **Caller-provided data directory** — All local files (`oluo_head.json`, `oluo_base.bin`) live under a `data_dir: &Path` supplied by the caller.

## Data Model

### SnapshotManifest

Postcard-serialized, stored as a single CAS book (no DAG chunking — it's ~100 bytes):

```rust
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SnapshotManifest {
    pub version: u32,            // SNAPSHOT_VERSION = 1
    pub index_cid: [u8; 32],    // DAG root for HNSW index bytes
    pub metadata_cid: [u8; 32], // DAG root for metadata sidecar
    pub key_counter: u64,        // next key to assign on resume
    pub compact_generation: u64, // compaction generation counter
}
```

### Local Head File

`{data_dir}/oluo_head.json`:

```json
{"version": 1, "head": "hex-encoded-manifest-cid"}
```

Written atomically via tmp + rename. Same pattern as harmony-db's `index.json`.

### Local Index File

`{data_dir}/oluo_base.bin`:

Raw bytes from `PersistSnapshot::index_bytes`, written atomically. The engine memory-maps this via `CompoundIndex::load_base(path)` after the caller sends `CompactComplete`.

## Function Signatures

### persist_snapshot

```rust
/// Persist a snapshot emitted by OluoEngine after compaction.
///
/// DAG-ingests index and metadata bytes into CAS, builds a
/// SnapshotManifest, writes the local index file for mmap,
/// and updates oluo_head.json.
///
/// Returns (local_index_path, generation) so the caller can
/// send CompactComplete back to the engine.
pub fn persist_snapshot(
    data_dir: &Path,
    store: &mut dyn BookStore,
    index_bytes: &[u8],
    metadata_bytes: &[u8],
    key_counter: u64,
    generation: u64,
) -> Result<(PathBuf, u64), OluoPersistError>
```

### load_snapshot

```rust
/// Load a snapshot from CAS and construct a ready-to-use OluoEngine.
///
/// Reads oluo_head.json, fetches the manifest from CAS,
/// reassembles index + metadata via DAG, restores the engine
/// via from_snapshot, and writes the local index file for mmap.
///
/// Returns None if no head file exists (fresh start).
pub fn load_snapshot(
    data_dir: &Path,
    store: &dyn BookStore,
    compact_threshold: usize,
) -> Result<Option<(OluoEngine, PathBuf, u64)>, OluoPersistError>
```

The return tuple gives the caller the engine plus the path and generation needed to send `CompactComplete` so the engine swaps to the mmap'd base.

## Error Type

```rust
#[derive(Debug, thiserror::Error)]
pub enum OluoPersistError {
    /// CAS storage or DAG operation failed.
    Content(ContentError),
    /// Local filesystem I/O failed.
    Io(std::io::Error),
    /// Manifest serialization or deserialization failed.
    ManifestSerde(String),
    /// Metadata deserialization failed.
    MetadataDeserialize(String),
    /// Engine restoration failed.
    Engine(SearchError),
    /// Manifest version not supported.
    UnsupportedVersion(u32),
    /// Referenced CID not found in store (hex-encoded CID).
    NotFound(String),
}
```

Uses `thiserror` for `Error` and `Display` derives. `From` impls for `ContentError`, `io::Error`, and `SearchError`.

## Persistence Flow

### Write Path

Called when the caller receives `OluoAction::PersistSnapshot`:

```text
1. dag::ingest(index_bytes, config, store)    → index_cid
2. dag::ingest(metadata_bytes, config, store)  → metadata_cid
3. Build SnapshotManifest { version: 1, index_cid, metadata_cid, key_counter, generation }
4. postcard::to_allocvec(manifest)             → manifest_bytes
5. store.insert(manifest_bytes)                → manifest_cid
6. atomic_write(data_dir/oluo_base.bin, index_bytes)
7. atomic_write(data_dir/oluo_head.json, {"version":1,"head":"<manifest_cid_hex>"})
8. Return (path_to_oluo_base.bin, generation)
```

The caller then sends `CompactComplete { path, generation }` to the engine.

Step 5 uses `store.insert()` directly — the manifest is tiny, no DAG chunking needed. Step 7 is last so the head file only updates after everything else succeeds.

### Read Path

Called at startup:

```text
1. Read data_dir/oluo_head.json               → manifest_cid (or None → return Ok(None))
2. store.get(manifest_cid)                     → manifest_bytes (or error if missing)
3. postcard::from_bytes(manifest_bytes)         → SnapshotManifest
4. dag::reassemble(index_cid, store)           → index_bytes
5. dag::reassemble(metadata_cid, store)        → metadata_bytes
6. postcard::from_bytes(metadata_bytes)         → BTreeMap<u64, EntryMetadata>
7. OluoEngine::from_snapshot(index_bytes, metadata, key_counter, generation, threshold)
8. atomic_write(data_dir/oluo_base.bin, index_bytes)
9. Return Ok(Some((engine, path_to_oluo_base.bin, generation)))
```

The caller then sends `CompactComplete` so the engine swaps to the mmap'd base.

### Crash Safety

The head file is always the last thing written (write path) and the first thing read (read path). Any crash mid-persist leaves the previous snapshot intact. Orphaned CAS blobs from incomplete writes are harmless — future GC can clean them.

## Files Changed

### New

- `crates/harmony-oluo/src/persist.rs` — `SnapshotManifest`, `OluoPersistError`, `persist_snapshot()`, `load_snapshot()`, atomic write helpers, head file read/write

### Modified

- `crates/harmony-oluo/Cargo.toml` — add `harmony-content` workspace dependency, `serde_json` dependency
- `crates/harmony-oluo/src/lib.rs` — add `pub mod persist;`, re-export `SnapshotManifest` and `OluoPersistError`

## Testing Strategy

All tests in harmony-oluo using `MemoryBookStore` + `tempfile::tempdir()`:

- **Round-trip:** Create engine, ingest entries, trigger compaction, capture `PersistSnapshot`, call `persist_snapshot()`, then `load_snapshot()` from the same store. Verify restored engine produces identical search results.
- **Fresh start:** `load_snapshot()` returns `None` when no `oluo_head.json` exists.
- **Manifest integrity:** Persist, then manually fetch manifest CID from store, deserialize, verify fields match the original `PersistSnapshot` payload.
- **Crash simulation:** Call `persist_snapshot()`, delete the head file, verify `load_snapshot()` returns `None` (falls back to fresh start).
- **Deterministic CIDs:** Persist the same snapshot twice, verify manifest CID is identical (postcard determinism + CAS).

### Dependencies for tests

- `tempfile` (dev-dependency) for temp directories
- `harmony-content`'s `MemoryBookStore` for in-memory CAS

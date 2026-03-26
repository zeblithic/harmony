# data_dir Persistent CAS Storage

## Overview

Add `data_dir: Option<String>` to `ConfigFile`. When set, persist CAS books to disk using a two-level hex prefix layout. On startup, scan the directory to build a disk index. The W-TinyLFU cache stays in memory as the hot layer; disk is the cold layer for evicted-but-durable content.

**Goal:** CAS book data survives node restarts when `data_dir` is configured.

**Scope:** CAS books only. Config field, disk I/O module, event loop wiring, startup scan. No NAR archives, no disk eviction, no OpenWRT integration (all filed as separate beads).

## Config

New field in `ConfigFile` (`crates/harmony-node/src/config.rs`):

```rust
pub data_dir: Option<String>,
```

TOML example:

```toml
data_dir = "/mnt/sda1/harmony"
```

When absent (default), behavior is unchanged — pure in-memory, no persistence. This preserves backward compatibility with every existing deployment.

The resolved `data_dir` path is passed through `NodeConfig` to `NodeRuntime::new()`, and separately to the event loop for disk I/O operations.

## File Layout

```
{data_dir}/book/{byte4_5_hex}/{full_cid_hex}
```

256-way fan-out using bytes 4-5 of the CID hash as the prefix directory. Same layout as `harmony-ingest/src/storage.rs`. Example:

```
/mnt/sda1/harmony/book/a1/a1b2c3d4e5f6...64-hex-chars...
```

Each file contains the raw book bytes (up to 1MB). Filenames are the full hex-encoded 32-byte ContentId.

## Architecture

### Write Path (PersistToDisk)

`StorageTier` already emits `StorageTierAction::PersistToDisk { cid, data }` for durable content (flags 00/10). This spec wires that action through:

1. `NodeRuntime` receives `PersistToDisk` from `StorageTier`
2. Converts to `RuntimeAction::PersistToDisk { cid, data }`
3. Event loop receives the action
4. Spawns `tokio::task::spawn_blocking` with the disk write (books can be up to 1MB — synchronous writes on USB storage could take 40ms+, so blocking the tick loop is unacceptable)
5. On completion, adds CID to `disk_index` (or logs error — write failures are non-fatal, content stays in memory cache)

### Read Path (DiskLookup)

When a CID is evicted from the in-memory cache but exists in `disk_index`:

1. A query arrives for the CID — cache miss, but `disk_index.contains(cid)` is true
2. `StorageTier` emits `StorageTierAction::DiskLookup { cid, query_id }`
3. Event loop spawns `spawn_blocking` to read the file
4. Sends `RuntimeEvent::DiskReadComplete { cid, query_id, data }` (or `DiskReadFailed`) back to the runtime
5. `StorageTier` re-admits the book to the in-memory cache and serves the query

### Startup Reload

On startup, if `data_dir` is configured:

1. Scan `{data_dir}/book/` recursively for files
2. Parse each filename as a hex CID — skip invalid filenames (log warning)
3. Populate `StorageTier.disk_index` with the discovered CIDs
4. No data is loaded into memory — books are loaded lazily on first access via `DiskLookup`

This keeps startup fast even with thousands of books on disk. Memory stays bounded by `cache_capacity`.

## Disk I/O Module

New module in `harmony-node`: `disk_io.rs` (or within an existing module — follow crate conventions).

Synchronous functions designed to be called from `spawn_blocking`:

```rust
/// Write a book to disk. Creates prefix directory if needed.
pub fn write_book(data_dir: &Path, cid: &ContentId, data: &[u8]) -> Result<(), std::io::Error>

/// Read a book from disk.
pub fn read_book(data_dir: &Path, cid: &ContentId) -> Result<Vec<u8>, std::io::Error>

/// Scan the book directory and return all CIDs found.
pub fn scan_books(data_dir: &Path) -> Vec<ContentId>
```

File path construction: `{data_dir}/book/{hex_cid[8..10]}/{hex_cid}` (bytes 4-5 as the fan-out prefix, matching `harmony-ingest`).

The `scan_books` function logs warnings for:
- Files with non-hex names (skipped)
- Files with wrong-length hex names (skipped)
- Empty directories (ignored silently)

## Event Loop Wiring

The event loop (`event_loop.rs`) gains:

1. A `data_dir: Option<PathBuf>` field
2. A `tokio::sync::mpsc` channel for disk I/O completion events
3. Handling of `RuntimeAction::PersistToDisk` — spawn write task
4. Handling of `RuntimeAction::DiskLookup` — spawn read task
5. Channel receiver in the `select!` loop — convert completions to `RuntimeEvent`

When `data_dir` is `None`, `PersistToDisk` and `DiskLookup` actions are silently ignored (no-op).

## RuntimeAction / RuntimeEvent Changes

### New RuntimeAction variants

```rust
RuntimeAction::PersistToDisk { cid: ContentId, data: Vec<u8> }
RuntimeAction::DiskLookup { cid: ContentId, query_id: u64 }
```

These may already exist as scaffolding — if so, just wire them. If not, add them.

### New/Existing RuntimeEvent variants

```rust
RuntimeEvent::DiskReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> }
RuntimeEvent::DiskReadFailed { cid: ContentId, query_id: u64 }
RuntimeEvent::DiskWriteComplete { cid: ContentId }
```

Check existing `StorageTierEvent` variants — these may already map to existing event types.

## StorageTier Changes

The `disk_index: HashSet<ContentId>` field already exists (currently `#[allow(dead_code)]`). This spec activates it:

1. Remove `#[allow(dead_code)]`
2. Populate from startup scan results (new method: `load_disk_index(cids: Vec<ContentId>)`)
3. Add to index when `PersistToDisk` is emitted
4. Check index on cache miss before returning "not found"
5. Emit `DiskLookup` when index contains the CID but cache doesn't

## NodeConfig Changes

Add `data_dir: Option<PathBuf>` to `NodeConfig` (the resolved config struct passed to `NodeRuntime::new`). `NodeRuntime` passes it through to `StorageTier` for disk_index management, and the event loop uses it for I/O paths.

## Error Handling

- **Write failure:** Log error, continue. Content stays in memory cache. Next restart it won't be on disk, but that's acceptable — the mesh can re-fetch it.
- **Read failure:** Send `DiskReadFailed` to `StorageTier`. The tier falls back to mesh fetch (existing behavior for cache misses).
- **Scan failure:** Log error for each bad file, continue scanning. Partially populated index is better than no index.
- **Missing data_dir directory:** Create `{data_dir}/book/` on first write. Don't create on startup (if the user configured a path that doesn't exist, they should see an error on first write, not a silent mkdir).

## Testing

### Unit Tests (disk_io module)

1. **Write/read round-trip:** Write book → read back → bytes match.
2. **Prefix directory creation:** Write creates the fan-out directory.
3. **Scan discovers books:** Write several books → scan → all CIDs returned.
4. **Scan skips invalid files:** Create files with non-hex names → scan ignores them.
5. **Read missing file:** Returns error (not panic).

### Integration Tests (StorageTier)

6. **PersistToDisk emitted for durable content:** Insert durable book → StorageTier emits PersistToDisk.
7. **DiskLookup on evicted-but-indexed CID:** Populate disk_index, query for a CID not in cache → DiskLookup emitted.
8. **DiskReadComplete re-admits to cache:** Feed DiskReadComplete → book is queryable.

### Config Tests

9. **data_dir parsed from TOML:** Config with data_dir field → parsed correctly.
10. **data_dir absent → None:** Config without field → None (backward compatible).

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-node/src/config.rs` | Add `data_dir: Option<String>` to `ConfigFile` |
| `crates/harmony-node/src/disk_io.rs` | New: write_book, read_book, scan_books |
| `crates/harmony-node/src/runtime.rs` | Wire PersistToDisk/DiskLookup actions, pass data_dir through NodeConfig |
| `crates/harmony-node/src/event_loop.rs` | spawn_blocking for disk I/O, channel for completions |
| `crates/harmony-node/src/main.rs` | Pass data_dir from config to event loop |
| `crates/harmony-content/src/storage_tier.rs` | Activate disk_index, add load_disk_index, emit DiskLookup on indexed miss |

## Dependencies

No new crate dependencies. Uses `std::fs`, `hex`, and existing `ContentId` types.

## What This Does NOT Include

- **NAR archive storage** — filed as harmony-885 (P3)
- **Disk space management / eviction** — filed as harmony-mti6 (P3)
- **RemoveFromDisk wiring** — scaffolding exists, deferred to eviction bead
- **OpenWRT init script update** — filed as harmony-c2py (P2), trivial once this lands

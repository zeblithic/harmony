# Disk Space Management and Eviction for CAS Storage — Design Spec

## Goal

Add configurable disk quota enforcement to the `data_dir` CAS storage layer.
When persisted books exceed the quota, evict least-recently-used entries from
disk, emitting `RemoveFromDisk` actions through the existing sans-I/O pipeline.

## Background

The `data_dir` persistence layer (bead harmony-0vu) writes durable books to
disk via `PersistToDisk` actions and reads them back on cache miss via
`DiskLookup`. The `RemoveFromDisk` action variant was scaffolded but deferred
to this bead. Without quota enforcement, disk usage grows unbounded.

Books are stored as flat files under `{data_dir}/book/{byte4_hex}/{cid_hex}`,
up to 1 MB each. Only durable content classes (`PublicDurable`,
`EncryptedDurable`) are persisted.

## Design Decisions

- **Write-then-evict**: Always persist the new book first, then evict LRU
  entries until back under quota. Disk may temporarily exceed quota by one
  book (max 1 MB). Simplest model; new content is never lost.
- **In-memory LRU**: Access ordering tracked in `StorageTier` state, not on
  disk. Lost on restart; reconstructed from access patterns. Fits the
  sans-I/O pattern.
- **Single byte limit**: One `disk_quota` config field shared by all durable
  classes. No per-class quotas.
- **Unlimited by default**: If `data_dir` is set without `disk_quota`, no
  eviction occurs. Backward-compatible with existing behavior. A log message
  notes the unbounded usage.
- **Pinned books exempt**: Pinned CIDs are never evicted from disk. If
  pinned content alone exceeds quota, a warning is logged but no eviction
  occurs.

## StorageTier State Changes

### Existing field changed

```rust
disk_index: HashMap<ContentId, u64>,  // was HashSet<ContentId> — now CID → byte size
```

### New fields

```rust
disk_lru: VecDeque<ContentId>,   // front = oldest, back = most recent
disk_used_bytes: u64,            // running total from disk_index values
disk_quota: Option<u64>,         // None = unlimited (no eviction)
```

### `enable_disk` signature change

```rust
// Before (uses impl IntoIterator<Item = ContentId>):
pub fn enable_disk(&mut self, cids: impl IntoIterator<Item = ContentId>)

// After:
pub fn enable_disk(&mut self, entries: impl IntoIterator<Item = (ContentId, u64)>)
```

Populates `disk_index` as a `HashMap`, seeds `disk_lru` in scan order,
computes `disk_used_bytes` as the sum of all sizes.

### New methods

```rust
pub fn set_disk_quota(&mut self, quota_bytes: u64)
```

Sets `disk_quota` to `Some(quota_bytes)`. Called during `NodeRuntime`
construction when the config field is present.

```rust
fn is_pinned(&self, cid: &ContentId) -> bool
```

Delegates to `self.cache.is_pinned(cid)`. The `ContentStore` (cache layer)
owns the pinned set; `StorageTier` needs read access for eviction decisions.
Add a public `is_pinned(&self, cid: &ContentId) -> bool` method to
`ContentStore` if one does not already exist.

## Eviction Flow

Runs immediately after every `PersistToDisk` emission:

1. **Update bookkeeping**: Insert `(cid, data.len())` into `disk_index`, add
   size to `disk_used_bytes`, push `cid` to back of `disk_lru`.
2. **Check quota**: If `disk_quota` is `None` or `disk_used_bytes <= quota`,
   stop.
3. **Evict loop**: Track `skipped = 0`. While `disk_used_bytes > quota`:
   - Pop front of `disk_lru`.
   - If CID is pinned: re-append to back of `disk_lru`, increment
     `skipped`, continue.
   - If CID is the one just persisted: re-append to back, increment
     `skipped`, continue.
   - If `skipped >= disk_lru.len()`: all remaining entries are
     unevictable — break (safety valve).
   - Remove from `disk_index`, subtract size from `disk_used_bytes`.
   - Emit `RemoveFromDisk { cid }`. Reset `skipped = 0`.
4. **Safety valve**: When `skipped >= disk_lru.len()`, every entry in the
   LRU has been examined and skipped (all pinned + the just-persisted CID).
   Log a warning and stop. Quota is exceeded but nothing can be done.

## LRU Maintenance

Three touch points keep the ordering accurate:

| Event | Action |
|-------|--------|
| `PersistToDisk` (new CID) | Push to back. Skip if CID already in `disk_index` (dedup). |
| `DiskReadComplete` | `retain` to remove, then `push_back`. O(n) but infrequent. |
| `RemoveFromDisk` | Already popped from front during eviction. No extra work. |

**Startup ordering**: `scan_books` returns entries in filesystem walk order
(effectively random). Acceptable — the LRU warms up as real accesses move
entries to the back.

## Disk I/O Changes

### `scan_books` return type

```rust
// Before:
pub fn scan_books(data_dir: &Path) -> Vec<ContentId>

// After:
pub fn scan_books(data_dir: &Path) -> Vec<(ContentId, u64)>
```

Each entry includes file size from `fs::metadata().len()`.

### New function: `delete_book`

```rust
pub fn delete_book(data_dir: &Path, cid: &ContentId) -> Result<(), io::Error>
```

Calls `fs::remove_file(book_path(data_dir, cid))`. Crash-safe — a file is
either present or absent; no corrupt intermediate state.

## Config

### New field in `ConfigFile`

```rust
pub disk_quota: Option<String>,  // e.g. "10 GiB", "500 MB"
```

Parsed with a human-readable byte parser supporting suffixes: `B`, `KB`,
`MB`, `GB`, `KiB`, `MiB`, `GiB` (case-insensitive). A bare number without
a suffix is a parse error to avoid ambiguity. Stored as `Option<u64>` after
parsing.

### Startup behavior

- `data_dir` set, `disk_quota` set: quota enforced.
- `data_dir` set, `disk_quota` absent: no eviction; log info message:
  "disk persistence enabled without quota — disk usage is unbounded."
- `data_dir` absent: `disk_quota` is ignored (no disk persistence).

### NodeConfig threading

`NodeConfig` gains `disk_quota: Option<u64>`. The existing `disk_cids`
field changes type from `Vec<ContentId>` to `Vec<(ContentId, u64)>` (renamed
to `disk_entries` for clarity). Both are passed from parsed config through to
`StorageTier` in `NodeRuntime::new()`.

## Event Loop Wiring

### RemoveFromDisk handler

```rust
RuntimeAction::RemoveFromDisk { cid } => {
    if let Some(ref dir) = data_dir {
        let dir = dir.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = disk_io::delete_book(&dir, &cid) {
                tracing::warn!(?cid, error = %e, "failed to delete book from disk");
            }
        });
    }
}
```

### New RuntimeAction variant

Add to the `RuntimeAction` enum in `runtime.rs`:

```rust
RuntimeAction::RemoveFromDisk { cid: ContentId },
```

This mirrors the existing `PersistToDisk` and `DiskLookup` variants.

### Runtime dispatch

The existing placeholder in `runtime.rs`:

```rust
StorageTierAction::RemoveFromDisk { .. } => {
    // Deferred to disk eviction bead (harmony-mti6).
}
```

Becomes:

```rust
StorageTierAction::RemoveFromDisk { cid } => {
    out.push(RuntimeAction::RemoveFromDisk { cid });
}
```

## Testing

### StorageTier unit tests (sans-I/O)

- **Eviction at quota**: Insert books exceeding quota; verify
  `RemoveFromDisk` actions for oldest CIDs.
- **LRU ordering**: Insert A, B, C; read A; trigger eviction — B evicted
  before A.
- **Pinned books skipped**: Pin a book; trigger eviction; verify it stays
  in `disk_index`.
- **No eviction without quota**: `disk_quota = None`; arbitrary fill; no
  `RemoveFromDisk` emitted.
- **Dedup on re-persist**: Persist same CID twice; `disk_used_bytes`
  counted once; no duplicate in `disk_lru`.
- **Safety valve**: Pin everything; exceed quota; verify no panic, no
  infinite loop.
- **DiskReadComplete refreshes LRU**: Persist A, B, C; `DiskReadComplete`
  for A; evict — B evicted first.

### disk_io unit tests

- `delete_book` removes file.
- `delete_book` on missing file returns `NotFound`.
- `scan_books` returns `(cid, size)` tuples matching written files.

### Config parsing tests

- `"10 GiB"` → `10 * 1024^3`.
- `"500 MB"` → `500 * 1_000_000`.
- `"1234"` (no suffix) → parse error.
- Invalid strings → clear error message.

## Out of Scope

- Byte-based pin limit enforcement (`max_pinned_bytes`)
- Pin audit on cache eviction paths (window, protected→probation)
- Per-class disk quotas
- Disk usage stats queryable
- Persisting LRU ordering across restarts
- NAR sidecar `.meta` file management (harmony-os concern)
- O(1) LRU structure (e.g. `LinkedHashMap`) — `VecDeque` with O(n) `retain`
  is sufficient while disk reads are infrequent; upgrade if profiling shows
  otherwise

# data_dir Persistent CAS Storage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `data_dir` config option so CAS books persist to disk and survive node restarts.

**Architecture:** Write-through persistence. `StorageTier` optimistically adds to `disk_index` when emitting `PersistToDisk`. Event loop handles I/O via `spawn_blocking`. On startup, scan disk to populate `disk_index` (lazy — data loaded on demand via `DiskLookup`).

**Tech Stack:** Rust, tokio (spawn_blocking), harmony-content (StorageTier, ContentId), TOML config

**Spec:** `docs/superpowers/specs/2026-03-26-data-dir-persistence-design.md`

**Test command:** `cargo test -p harmony-node` and `cargo test -p harmony-content`
**Lint command:** `cargo clippy -p harmony-node -p harmony-content`

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-node/src/config.rs` | Config parsing | Modify: add `data_dir` field |
| `crates/harmony-node/src/disk_io.rs` | Synchronous disk I/O helpers | Create |
| `crates/harmony-content/src/storage_tier.rs` | Activate disk_index, emit PersistToDisk/DiskLookup | Modify |
| `crates/harmony-node/src/runtime.rs` | Convert StorageTierAction to RuntimeAction for disk I/O | Modify |
| `crates/harmony-node/src/event_loop.rs` | spawn_blocking for disk I/O, startup scan | Modify |
| `crates/harmony-node/src/main.rs` | Pass data_dir to event loop | Modify |

---

### Task 1: Config field and disk I/O helpers

Add `data_dir` to `ConfigFile` and create `disk_io.rs` with write/read/scan functions.

**Files:**
- Modify: `crates/harmony-node/src/config.rs`
- Create: `crates/harmony-node/src/disk_io.rs`

- [ ] **Step 1: Write failing tests for disk_io**

Create `crates/harmony-node/src/disk_io.rs` with test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::cid::{ContentFlags, ContentId};
    use tempfile::TempDir;

    fn make_durable_cid(data: &[u8]) -> ContentId {
        ContentId::for_book(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn write_and_read_round_trip() {
        let dir = TempDir::new().unwrap();
        let cid = make_durable_cid(b"hello world book data");
        let data = b"hello world book data";
        write_book(dir.path(), &cid, data).unwrap();
        let read_data = read_book(dir.path(), &cid).unwrap();
        assert_eq!(read_data, data);
    }

    #[test]
    fn write_creates_prefix_directory() {
        let dir = TempDir::new().unwrap();
        let cid = make_durable_cid(b"test data");
        write_book(dir.path(), &cid, b"test data").unwrap();

        let hex_cid = hex::encode(cid.to_bytes());
        let prefix_dir = dir.path().join("book").join(&hex_cid[8..10]);
        assert!(prefix_dir.exists(), "Prefix directory should be created");
    }

    #[test]
    fn scan_discovers_written_books() {
        let dir = TempDir::new().unwrap();
        let cid1 = make_durable_cid(b"book one");
        let cid2 = make_durable_cid(b"book two");
        write_book(dir.path(), &cid1, b"book one").unwrap();
        write_book(dir.path(), &cid2, b"book two").unwrap();

        let found = scan_books(dir.path());
        assert_eq!(found.len(), 2);
        assert!(found.contains(&cid1));
        assert!(found.contains(&cid2));
    }

    #[test]
    fn scan_skips_invalid_filenames() {
        let dir = TempDir::new().unwrap();
        // Write a valid book.
        let cid = make_durable_cid(b"valid");
        write_book(dir.path(), &cid, b"valid").unwrap();

        // Create a file with a non-hex name in the same prefix directory.
        let hex_cid = hex::encode(cid.to_bytes());
        let prefix_dir = dir.path().join("book").join(&hex_cid[8..10]);
        std::fs::write(prefix_dir.join("not-a-hex-cid.txt"), b"junk").unwrap();

        let found = scan_books(dir.path());
        assert_eq!(found.len(), 1, "Should skip invalid filename");
        assert!(found.contains(&cid));
    }

    #[test]
    fn read_missing_file_returns_error() {
        let dir = TempDir::new().unwrap();
        let cid = make_durable_cid(b"nonexistent");
        let result = read_book(dir.path(), &cid);
        assert!(result.is_err());
    }

    #[test]
    fn scan_empty_directory_returns_empty() {
        let dir = TempDir::new().unwrap();
        let found = scan_books(dir.path());
        assert!(found.is_empty());
    }
}
```

- [ ] **Step 2: Implement disk_io.rs**

Add above the test module:

```rust
//! Synchronous disk I/O helpers for CAS book persistence.
//!
//! Designed to be called from `tokio::task::spawn_blocking` — all functions
//! are synchronous and may block on filesystem operations.
//!
//! File layout: `{data_dir}/book/{hex_cid[8..10]}/{hex_cid}`
//! (256-way fan-out using byte 4 of the CID, matching harmony-ingest).

use harmony_content::cid::ContentId;
use std::path::{Path, PathBuf};

/// Compute the file path for a book on disk.
pub fn book_path(data_dir: &Path, cid: &ContentId) -> PathBuf {
    let hex_cid = hex::encode(cid.to_bytes());
    data_dir.join("book").join(&hex_cid[8..10]).join(&hex_cid)
}

/// Write a book to disk. Creates prefix directory if needed.
pub fn write_book(data_dir: &Path, cid: &ContentId, data: &[u8]) -> Result<(), std::io::Error> {
    let path = book_path(data_dir, cid);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, data)
}

/// Read a book from disk.
pub fn read_book(data_dir: &Path, cid: &ContentId) -> Result<Vec<u8>, std::io::Error> {
    let path = book_path(data_dir, cid);
    std::fs::read(&path)
}

/// Scan the book directory and return all valid CIDs found.
///
/// Logs warnings for files with invalid names (non-hex, wrong length).
/// Returns an empty Vec if the book directory doesn't exist.
pub fn scan_books(data_dir: &Path) -> Vec<ContentId> {
    let book_dir = data_dir.join("book");
    if !book_dir.exists() {
        return Vec::new();
    }

    let mut cids = Vec::new();
    let prefix_dirs = match std::fs::read_dir(&book_dir) {
        Ok(entries) => entries,
        Err(e) => {
            tracing::warn!("Failed to read book directory {}: {e}", book_dir.display());
            return Vec::new();
        }
    };

    for prefix_entry in prefix_dirs.flatten() {
        if !prefix_entry.path().is_dir() {
            continue;
        }
        let files = match std::fs::read_dir(prefix_entry.path()) {
            Ok(entries) => entries,
            Err(_) => continue,
        };
        for file_entry in files.flatten() {
            let filename = file_entry.file_name();
            let name = filename.to_string_lossy();
            // CID is 32 bytes = 64 hex chars.
            if name.len() != 64 || !name.chars().all(|c| c.is_ascii_hexdigit()) {
                tracing::debug!("Skipping non-CID file: {}", file_entry.path().display());
                continue;
            }
            match hex::decode(name.as_ref()) {
                Ok(bytes) => {
                    let arr: [u8; 32] = match bytes.try_into() {
                        Ok(a) => a,
                        Err(_) => continue,
                    };
                    if let Ok(cid) = ContentId::from_bytes(arr) {
                        cids.push(cid);
                    }
                }
                Err(_) => continue,
            }
        }
    }
    cids
}
```

Note: Check if `ContentId::from_bytes(&[u8])` exists. If not, you may need `ContentId::from_bytes_exact([u8; 32])` or construct it differently. Read the `ContentId` type to find the right constructor. Also check if `tracing` is already a dependency of `harmony-node` (it should be — it's used throughout the crate).

- [ ] **Step 3: Add `data_dir` to ConfigFile**

In `crates/harmony-node/src/config.rs`, add to the `ConfigFile` struct:

```rust
/// Directory for persistent CAS book storage. When set, durable books
/// are written to disk and reloaded on restart. When absent, all content
/// is in-memory only (lost on restart).
pub data_dir: Option<String>,
```

Add it alphabetically among the other fields.

- [ ] **Step 4: Register disk_io module**

Add `mod disk_io;` to `crates/harmony-node/src/main.rs` (harmony-node is a binary crate — all module declarations are in main.rs, not lib.rs). Place it near the other `mod` declarations.

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-node disk_io -- --nocapture`
Expected: ALL PASS (6 tests)

- [ ] **Step 6: Run clippy**

Run: `cargo clippy -p harmony-node 2>&1`
Expected: No errors

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/config.rs crates/harmony-node/src/disk_io.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add data_dir config option and disk I/O helpers for CAS persistence"
```

---

### Task 2: Activate StorageTier disk_index and emit PersistToDisk/DiskLookup

Un-defer the commented-out `PersistToDisk` emissions and `DiskLookup` on cache miss. Activate `disk_index`. Add `load_disk_index` method. Gate all disk behavior on a `disk_enabled` flag (set when `data_dir` is configured).

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

- [ ] **Step 1: Add `disk_enabled` flag to StorageTier**

Add a `disk_enabled: bool` field to `StorageTier`. Set it to `false` in the constructor. Add a method:

```rust
/// Enable disk persistence and populate the initial disk index from a startup scan.
pub fn enable_disk(&mut self, cids: Vec<ContentId>) {
    self.disk_enabled = true;
    for cid in cids {
        self.disk_index.insert(cid);
    }
}
```

Remove `#[allow(dead_code)]` from `disk_index`.

- [ ] **Step 2: Emit PersistToDisk for durable content**

In `handle_transit` and `handle_publish`, replace the "PersistToDisk deferred" comments with actual emission. **Important:** The data must be cloned BEFORE it's consumed by `cache.store_preadmitted(cid, data)` (which moves `data`). Clone early:

```rust
// Clone data for disk persistence BEFORE cache consumes it.
let persist_data = if self.disk_enabled && Self::is_durable_class(&cid) {
    Some(data.clone())
} else {
    None
};

// ... existing cache insertion (consumes data) ...

// After cache insertion, persist durable content to disk.
if let Some(persist_bytes) = persist_data {
    self.disk_index.insert(cid);
    actions.push(StorageTierAction::PersistToDisk {
        cid,
        data: persist_bytes,
    });
}
```

- [ ] **Step 3: Emit DiskLookup on cache miss**

In the cache miss branch of `handle_content_query`, restore the commented-out disk_index check:

```rust
None => {
    self.metrics.cache_misses += 1;
    if self.disk_enabled && self.disk_index.contains(cid) {
        return vec![StorageTierAction::DiskLookup { cid: *cid, query_id }];
    }
    vec![]
}
```

- [ ] **Step 4: Update tests**

Update the `persist_to_disk_deferred_until_runtime_wired` test to check that PersistToDisk IS emitted when disk is enabled, and NOT emitted when disabled. Update `cache_miss_returns_empty` to test DiskLookup emission when disk is enabled.

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): activate disk_index, emit PersistToDisk and DiskLookup when disk enabled"
```

---

### Task 3: Wire disk I/O actions through NodeRuntime

Convert `StorageTierAction::PersistToDisk` and `DiskLookup` to `RuntimeAction` variants in `NodeRuntime`.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add RuntimeAction variants**

Add to the `RuntimeAction` enum:

```rust
/// Persist a CAS book to disk (spawned as blocking I/O by event loop).
PersistToDisk { cid: ContentId, data: Vec<u8> },
/// Read a CAS book from disk (spawned as blocking I/O by event loop).
DiskLookup { cid: ContentId, query_id: u64 },
```

Note: `ContentId` needs to be imported. Check if it's already imported — if not, add `use harmony_content::cid::ContentId;`.

- [ ] **Step 2: Add RuntimeEvent variants for disk I/O completions**

Add to the `RuntimeEvent` enum:

```rust
/// Disk read completed — content fetched from persistent storage.
DiskReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
/// Disk read failed — file missing or corrupted.
DiskReadFailed { cid: ContentId, query_id: u64 },
```

In `NodeRuntime::push_event`, add handling for these:

```rust
RuntimeEvent::DiskReadComplete { cid, query_id, data } => {
    let storage_event = StorageTierEvent::DiskReadComplete { cid, query_id, data };
    let actions = self.storage.handle(storage_event);
    self.dispatch_storage_actions(actions);
}
RuntimeEvent::DiskReadFailed { cid, query_id } => {
    let storage_event = StorageTierEvent::DiskReadFailed { cid, query_id };
    let actions = self.storage.handle(storage_event);
    self.dispatch_storage_actions(actions);
}
```

- [ ] **Step 3: Wire StorageTierAction to RuntimeAction**

In the `dispatch_storage_actions` method (around line 1785), replace the no-op arm:

```rust
StorageTierAction::PersistToDisk { cid, data } => {
    self.pending_direct_actions.push(RuntimeAction::PersistToDisk { cid, data });
}
StorageTierAction::DiskLookup { cid, query_id } => {
    self.pending_direct_actions.push(RuntimeAction::DiskLookup { cid, query_id });
}
StorageTierAction::RemoveFromDisk { .. } => {
    // Deferred to disk eviction bead (harmony-mti6).
}
```

- [ ] **Step 3: Add data_dir to NodeConfig and wire enable_disk**

Add `data_dir: Option<PathBuf>` to `NodeConfig`. In `NodeRuntime::new()`, after constructing `StorageTier`, call `storage.enable_disk(cids)` if `data_dir` is set. The CIDs come from a startup scan — but the scan needs to happen before `NodeRuntime::new()`. So: accept `disk_cids: Vec<ContentId>` as a parameter to `new()`, and call `enable_disk` if the list is non-empty.

Actually, simpler: just add `disk_cids` to `NodeConfig`:

```rust
pub struct NodeConfig {
    // ... existing fields ...
    /// CIDs discovered on disk during startup scan (empty if no data_dir).
    pub disk_cids: Vec<ContentId>,
}
```

In `new()`, after creating StorageTier:

```rust
if !config.disk_cids.is_empty() {
    storage.enable_disk(config.disk_cids);
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-node -- --nocapture`
Run: `cargo clippy -p harmony-node`
Expected: ALL PASS (existing tests may need minor updates for the new NodeConfig field)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire PersistToDisk and DiskLookup through NodeRuntime"
```

---

### Task 4: Event loop — spawn_blocking for disk I/O and startup scan

Wire the event loop to handle `PersistToDisk` and `DiskLookup` via `spawn_blocking`, add a completion channel, and scan the disk directory on startup.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/main.rs`

- [ ] **Step 1: Add disk I/O channel to event loop**

In the `run()` function, create an mpsc channel for disk I/O completions:

```rust
let (disk_tx, mut disk_rx) = mpsc::channel::<DiskIoResult>(64);
```

Define the `DiskIoResult` enum (at the top of event_loop.rs or in a local module):

```rust
enum DiskIoResult {
    ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },
    ReadFailed { cid: ContentId, query_id: u64 },
    WriteComplete { cid: ContentId },
    WriteFailed { cid: ContentId, error: String },
}
```

- [ ] **Step 2: Handle PersistToDisk in dispatch_action**

Add to `dispatch_action`:

```rust
RuntimeAction::PersistToDisk { cid, data } => {
    if let Some(ref data_dir) = data_dir {
        let dir = data_dir.clone();
        let tx = disk_tx.clone();
        tokio::task::spawn_blocking(move || {
            match crate::disk_io::write_book(&dir, &cid, &data) {
                Ok(()) => {
                    let _ = tx.blocking_send(DiskIoResult::WriteComplete { cid });
                }
                Err(e) => {
                    tracing::error!("Disk write failed for {}: {e}", hex::encode(cid.to_bytes()));
                    let _ = tx.blocking_send(DiskIoResult::WriteFailed {
                        cid,
                        error: e.to_string(),
                    });
                }
            }
        });
    }
}
RuntimeAction::DiskLookup { cid, query_id } => {
    if let Some(ref data_dir) = data_dir {
        let dir = data_dir.clone();
        let tx = disk_tx.clone();
        tokio::task::spawn_blocking(move || {
            match crate::disk_io::read_book(&dir, &cid) {
                Ok(data) => {
                    let _ = tx.blocking_send(DiskIoResult::ReadComplete { cid, query_id, data });
                }
                Err(_) => {
                    let _ = tx.blocking_send(DiskIoResult::ReadFailed { cid, query_id });
                }
            }
        });
    }
}
```

Note: `dispatch_action` needs `data_dir: &Option<PathBuf>` and `disk_tx: &mpsc::Sender<DiskIoResult>` as additional parameters.

- [ ] **Step 3: Add disk completion receiver to select! loop**

In the main `tokio::select!` loop, add a branch:

```rust
Some(disk_result) = disk_rx.recv() => {
    match disk_result {
        DiskIoResult::ReadComplete { cid, query_id, data } => {
            runtime.push_event(RuntimeEvent::DiskReadComplete { cid, query_id, data });
        }
        DiskIoResult::ReadFailed { cid, query_id } => {
            runtime.push_event(RuntimeEvent::DiskReadFailed { cid, query_id });
        }
        DiskIoResult::WriteComplete { cid } => {
            tracing::debug!("Disk write complete: {}", hex::encode(cid.to_bytes()));
        }
        DiskIoResult::WriteFailed { cid, error } => {
            tracing::warn!("Disk write failed for {}: {error}", hex::encode(cid.to_bytes()));
        }
    }
}
```

Check if `RuntimeEvent` already has `DiskReadComplete`/`DiskReadFailed` variants. These need to map to `StorageTierEvent` variants, which DO exist. The runtime should convert:
- `DiskReadComplete` → `StorageTierEvent::DiskReadComplete`
- `DiskReadFailed` → `StorageTierEvent::DiskReadFailed`

If `RuntimeEvent` doesn't have these variants, add them and wire the conversion in `NodeRuntime::push_event`.

- [ ] **Step 4: Add startup scan**

In `main.rs`, after loading config but before constructing `NodeRuntime`, scan the disk:

```rust
let disk_cids = match &config_file.data_dir {
    Some(dir) => {
        let path = PathBuf::from(dir);
        let cids = crate::disk_io::scan_books(&path);
        tracing::info!("Loaded {} CIDs from disk at {}", cids.len(), path.display());
        cids
    }
    None => Vec::new(),
};
```

Pass `disk_cids` into `NodeConfig` and `data_dir` into the event loop's `run()` function.

- [ ] **Step 5: Update run() signature**

Add `data_dir: Option<PathBuf>` parameter to the event loop's `run()` function.

- [ ] **Step 6: Run tests and clippy**

Run: `cargo test -p harmony-node -- --nocapture`
Run: `cargo clippy -p harmony-node`
Expected: ALL PASS

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/main.rs crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire disk I/O via spawn_blocking, startup scan, event loop integration"
```

---

### Task 5: Final integration and workspace test

Run full workspace tests, verify disk persistence works end-to-end.

- [ ] **Step 1: Run harmony-content tests**

Run: `cargo test -p harmony-content -- --nocapture`
Expected: ALL PASS

- [ ] **Step 2: Run harmony-node tests**

Run: `cargo test -p harmony-node -- --nocapture`
Expected: ALL PASS

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -p harmony-node -p harmony-content`
Expected: Clean

- [ ] **Step 4: Final commit if needed**

```bash
git add -A
git commit -m "chore: final cleanup after data_dir persistence integration"
```

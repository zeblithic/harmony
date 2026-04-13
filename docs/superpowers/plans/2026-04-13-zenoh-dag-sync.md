# harmony-db Zenoh DAG Sync Implementation Plan (ZEB-108)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable peers to sync harmony-db databases over Zenoh by publishing root CIDs and pulling missing blocks on demand.

**Architecture:** Key expression helpers in harmony-zenoh define the namespace. A `DiskBookStore` in harmony-db implements `BookStore` backed by the local CAS filesystem with a pluggable fetch callback for network fallback. The actual Zenoh wiring (pub/sub for root announcements, queryable for block serving) lives at the application layer where both crates are available.

**Tech Stack:** Rust, harmony-db (Prolly Tree CAS), harmony-zenoh (Zenoh key expressions), harmony-content (BookStore trait, ContentId)

---

## File Structure

| File | Responsibility | Change |
|------|---------------|--------|
| `crates/harmony-zenoh/src/keyspace.rs` | Zenoh key expression builders | Add `db_root_key`, `db_root_sub`, `db_block_key`, `db_block_queryable` functions |
| `crates/harmony-db/src/store.rs` | Disk-backed BookStore with network fallback | New file: `DiskBookStore` implementing `BookStore` |
| `crates/harmony-db/src/lib.rs` | Crate exports | Add `mod store` and re-export `DiskBookStore` |

---

### Task 1: Add db sync key expression helpers to harmony-zenoh

**Files:**
- Modify: `crates/harmony-zenoh/src/keyspace.rs`

- [ ] **Step 1: Write the failing tests**

Add these tests to the existing `mod tests` block in `crates/harmony-zenoh/src/keyspace.rs`:

```rust
// ── DB sync key expression tests ────────────────────────────────

#[test]
fn db_root_key_builds_correctly() {
    let ke = db_root_key("abcd1234", "mail").unwrap();
    assert_eq!(ke.as_str(), "harmony/db/abcd1234/mail/root");
}

#[test]
fn db_root_sub_matches_root_key() {
    let sub = db_root_sub("abcd1234", "mail").unwrap();
    let key = db_root_key("abcd1234", "mail").unwrap();
    assert!(sub.intersects(&key));
}

#[test]
fn db_root_sub_does_not_match_other_peer() {
    let sub = db_root_sub("abcd1234", "mail").unwrap();
    let key = db_root_key("deadbeef", "mail").unwrap();
    assert!(!sub.intersects(&key));
}

#[test]
fn db_block_key_builds_correctly() {
    let ke = db_block_key("abcd1234", "mail", "ff00ff00").unwrap();
    assert_eq!(ke.as_str(), "harmony/db/abcd1234/mail/block/ff00ff00");
}

#[test]
fn db_block_queryable_matches_any_block() {
    let queryable = db_block_queryable("abcd1234", "mail").unwrap();
    let key1 = db_block_key("abcd1234", "mail", "aabbccdd").unwrap();
    let key2 = db_block_key("abcd1234", "mail", "11223344").unwrap();
    assert!(queryable.intersects(&key1));
    assert!(queryable.intersects(&key2));
}

#[test]
fn db_block_queryable_does_not_match_other_db() {
    let queryable = db_block_queryable("abcd1234", "mail").unwrap();
    let key = db_block_key("abcd1234", "contacts", "aabbccdd").unwrap();
    assert!(!queryable.intersects(&key));
}

#[test]
fn db_key_rejects_slashes_in_owner() {
    assert!(db_root_key("ab/cd", "mail").is_err());
}

#[test]
fn db_key_rejects_slashes_in_db_name() {
    assert!(db_root_key("abcd1234", "my/db").is_err());
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh db_root_key db_root_sub db_block_key db_block_queryable db_key_rejects -- --nocapture 2>&1`
Expected: FAIL — functions don't exist.

- [ ] **Step 3: Implement key expression builders**

Add these functions to `crates/harmony-zenoh/src/keyspace.rs`, in a new section after the book/page key expressions:

```rust
// ── DB sync key expressions ─────────────────────────────────────────

/// Build a db root announcement key expression.
///
/// Pattern: `harmony/db/{owner_addr_hex}/{db_name}/root`
pub fn db_root_key(owner_addr_hex: &str, db_name: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(owner_addr_hex)?;
    reject_slashes(db_name)?;
    ke(&format!("harmony/db/{owner_addr_hex}/{db_name}/root"))
}

/// Build a db root announcement subscription pattern.
///
/// Pattern: `harmony/db/{owner_addr_hex}/{db_name}/root`
pub fn db_root_sub(owner_addr_hex: &str, db_name: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(owner_addr_hex)?;
    reject_slashes(db_name)?;
    ke(&format!("harmony/db/{owner_addr_hex}/{db_name}/root"))
}

/// Build a db block fetch key expression.
///
/// Pattern: `harmony/db/{owner_addr_hex}/{db_name}/block/{cid_hex}`
pub fn db_block_key(
    owner_addr_hex: &str,
    db_name: &str,
    cid_hex: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(owner_addr_hex)?;
    reject_slashes(db_name)?;
    reject_slashes(cid_hex)?;
    ke(&format!(
        "harmony/db/{owner_addr_hex}/{db_name}/block/{cid_hex}"
    ))
}

/// Build a db block queryable pattern (wildcard for any CID).
///
/// Pattern: `harmony/db/{owner_addr_hex}/{db_name}/block/*`
pub fn db_block_queryable(
    owner_addr_hex: &str,
    db_name: &str,
) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(owner_addr_hex)?;
    reject_slashes(db_name)?;
    ke(&format!(
        "harmony/db/{owner_addr_hex}/{db_name}/block/*"
    ))
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh db_root_key db_root_sub db_block_key db_block_queryable db_key_rejects -- --nocapture 2>&1`
Expected: PASS (all 8 tests)

- [ ] **Step 5: Run all harmony-zenoh tests**

Run: `cargo test -p harmony-zenoh -- --nocapture 2>&1`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-zenoh/src/keyspace.rs
git commit -m "feat(zenoh): add db sync key expression helpers (ZEB-108)"
```

---

### Task 2: Create DiskBookStore with network fallback

**Files:**
- Create: `crates/harmony-db/src/store.rs`
- Modify: `crates/harmony-db/src/lib.rs`

This creates a `BookStore` implementation backed by the local CAS filesystem (`commits/` and `blobs/` directories) with an optional closure for fetching missing CIDs from the network.

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-db/src/store.rs` with just the test module first:

```rust
//! Disk-backed BookStore with optional network fallback.

use std::path::{Path, PathBuf};

use harmony_content::book::BookStore;
use harmony_content::cid::{ContentFlags, ContentId};
use harmony_content::error::ContentError;

use crate::error::DbError;
use crate::persist;

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn get_returns_committed_node() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        // Write a fake node to commits/
        let data = b"test node data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let hex = hex::encode(cid.to_bytes());
        std::fs::write(
            data_dir.join("commits").join(format!("{hex}.bin")),
            data,
        )
        .unwrap();

        let store = DiskBookStore::new(data_dir);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn get_returns_blob() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        // Write a fake blob
        let data = b"test blob data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let hex = hex::encode(cid.to_bytes());
        std::fs::write(
            data_dir.join("blobs").join(format!("{hex}.bin")),
            data,
        )
        .unwrap();

        let store = DiskBookStore::new(data_dir);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn get_returns_none_for_missing() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        let cid = ContentId::for_book(b"not stored", ContentFlags::default()).unwrap();
        let store = DiskBookStore::new(data_dir);
        assert!(store.get(&cid).is_none());
    }

    #[test]
    fn contains_checks_local_only() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        let data = b"exists";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let hex = hex::encode(cid.to_bytes());
        std::fs::write(
            data_dir.join("commits").join(format!("{hex}.bin")),
            data,
        )
        .unwrap();

        let store = DiskBookStore::new(data_dir);
        assert!(store.contains(&cid));

        let missing = ContentId::for_book(b"nope", ContentFlags::default()).unwrap();
        assert!(!store.contains(&missing));
    }

    #[test]
    fn store_writes_to_commits() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        let data = b"stored data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();

        let mut store = DiskBookStore::new(data_dir);
        store.store(cid, data.to_vec());

        // Should now be readable
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn insert_with_flags_computes_cid_and_stores() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        let data = b"inserted data";
        let mut store = DiskBookStore::new(data_dir);
        let cid = store.insert_with_flags(data, ContentFlags::default()).unwrap();

        assert!(store.contains(&cid));
        assert_eq!(store.get(&cid).unwrap(), data);
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

In `crates/harmony-db/src/lib.rs`, add:

```rust
mod store;

pub use store::DiskBookStore;
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-db disk_book_store -- --nocapture 2>&1`
Expected: FAIL — `DiskBookStore` struct doesn't exist.

- [ ] **Step 4: Implement DiskBookStore**

Add the struct and `BookStore` impl to `crates/harmony-db/src/store.rs`, before the `#[cfg(test)]` block:

```rust
/// A [`BookStore`] backed by the harmony-db CAS filesystem.
///
/// Reads from both `commits/` (tree nodes, manifests) and `blobs/`
/// (value data). Writes go to `commits/`. An optional fetch callback
/// enables transparent network fallback for missing CIDs — the callback
/// is invoked on local miss, and fetched data is cached locally.
///
/// The `cache` field holds fetched/stored data in memory for the
/// lifetime of this store, satisfying `BookStore::get`'s `&[u8]`
/// return type without re-reading from disk.
pub struct DiskBookStore {
    data_dir: PathBuf,
    /// In-memory cache of fetched blocks. `BookStore::get` returns `&[u8]`
    /// which requires the data to live as long as `&self`, so we cache
    /// disk reads here.
    cache: std::collections::HashMap<ContentId, Vec<u8>>,
    /// Optional network fetch callback: given a CID, returns bytes if available.
    fetcher: Option<Box<dyn Fn(&ContentId) -> Option<Vec<u8>>>>,
}

impl DiskBookStore {
    /// Create a store backed by the given data directory (no network fallback).
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: std::collections::HashMap::new(),
            fetcher: None,
        }
    }

    /// Create a store with a network fetch callback for missing CIDs.
    ///
    /// The callback is invoked when `get()` can't find a CID locally.
    /// Fetched data is verified (CID must match) and cached locally
    /// in both the in-memory cache and on disk.
    pub fn with_fetcher(
        data_dir: &Path,
        fetcher: impl Fn(&ContentId) -> Option<Vec<u8>> + 'static,
    ) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: std::collections::HashMap::new(),
            fetcher: Some(Box::new(fetcher)),
        }
    }

    /// Try to read a CID from local CAS (commits/ then blobs/).
    fn read_local(&self, cid: &ContentId) -> Option<Vec<u8>> {
        let hex = hex::encode(cid.to_bytes());
        // Check commits/ first (tree nodes, manifests)
        let commit_path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&commit_path) {
            return Some(data);
        }
        // Check blobs/ (value data)
        let blob_path = self.data_dir.join("blobs").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&blob_path) {
            return Some(data);
        }
        None
    }

    /// Write data to commits/ directory.
    fn write_to_commits(&self, cid: &ContentId, data: &[u8]) {
        let hex = hex::encode(cid.to_bytes());
        let path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        if !path.exists() {
            let tmp = self.data_dir.join("commits").join(format!("{hex}.bin.tmp"));
            let _ = std::fs::write(&tmp, data);
            let _ = std::fs::rename(&tmp, &path);
        }
    }
}

impl BookStore for DiskBookStore {
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_book(data, flags)?;
        self.write_to_commits(&cid, data);
        self.cache.insert(cid, data.to_vec());
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.write_to_commits(&cid, &data);
        self.cache.insert(cid, data);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        // Check in-memory cache first
        if let Some(data) = self.cache.get(cid) {
            return Some(data);
        }
        // Check local disk
        if let Some(data) = self.read_local(cid) {
            // We need to cache to satisfy the &[u8] lifetime.
            // Use interior mutability pattern — but BookStore::get takes &self.
            // For now, return None and let the caller use the mutable path.
            // Actually, we can't cache here with &self. Return the data
            // from disk by stashing it in the cache via unsafe... no.
            //
            // The clean solution: the caller should call get_or_fetch() which
            // takes &mut self. But BookStore::get is &self.
            //
            // Workaround: use a RefCell for the cache.
            return None; // placeholder — see Step 5
        }
        None
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.cache.contains_key(cid) || self.read_local(cid).is_some()
    }

    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        self.cache.remove(cid)
    }
}
```

Wait — there's a problem. `BookStore::get(&self, cid) -> Option<&[u8]>` takes `&self` but we need to cache disk reads to return a `&[u8]` that lives long enough. Let me redesign this using `RefCell`.

- [ ] **Step 5: Fix the lifetime issue with RefCell**

Replace the `DiskBookStore` struct and impl with this corrected version:

```rust
use std::cell::RefCell;

/// A [`BookStore`] backed by the harmony-db CAS filesystem.
///
/// Reads from both `commits/` (tree nodes, manifests) and `blobs/`
/// (value data). Writes go to `commits/`. An optional fetch callback
/// enables transparent network fallback for missing CIDs — the callback
/// is invoked on local miss, and fetched data is cached locally.
pub struct DiskBookStore {
    data_dir: PathBuf,
    /// In-memory cache of fetched blocks. Uses RefCell because
    /// `BookStore::get(&self)` needs to cache disk reads to satisfy
    /// the `&[u8]` lifetime requirement.
    cache: RefCell<std::collections::HashMap<ContentId, Vec<u8>>>,
    /// Optional network fetch callback: given a CID, returns bytes if available.
    fetcher: Option<Box<dyn Fn(&ContentId) -> Option<Vec<u8>>>>,
}

impl DiskBookStore {
    /// Create a store backed by the given data directory (no network fallback).
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: RefCell::new(std::collections::HashMap::new()),
            fetcher: None,
        }
    }

    /// Create a store with a network fetch callback for missing CIDs.
    ///
    /// The callback is invoked when `get()` can't find a CID locally.
    /// Fetched data is verified (CID must match BLAKE3 hash) and cached
    /// locally in both the in-memory cache and on disk.
    pub fn with_fetcher(
        data_dir: &Path,
        fetcher: impl Fn(&ContentId) -> Option<Vec<u8>> + 'static,
    ) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: RefCell::new(std::collections::HashMap::new()),
            fetcher: Some(Box::new(fetcher)),
        }
    }

    /// Try to read a CID from local CAS (commits/ then blobs/).
    fn read_local(&self, cid: &ContentId) -> Option<Vec<u8>> {
        let hex = hex::encode(cid.to_bytes());
        let commit_path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&commit_path) {
            return Some(data);
        }
        let blob_path = self.data_dir.join("blobs").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&blob_path) {
            return Some(data);
        }
        None
    }

    /// Write data to commits/ directory (atomic via tmp + rename).
    fn write_to_commits(&self, cid: &ContentId, data: &[u8]) {
        let hex = hex::encode(cid.to_bytes());
        let path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        if !path.exists() {
            let tmp = self.data_dir.join("commits").join(format!("{hex}.bin.tmp"));
            let _ = std::fs::write(&tmp, data);
            let _ = std::fs::rename(&tmp, &path);
        }
    }

    /// Verify that data matches the expected CID.
    fn verify_cid(cid: &ContentId, data: &[u8]) -> bool {
        ContentId::for_book(data, ContentFlags::default())
            .map(|computed| computed == *cid)
            .unwrap_or(false)
    }
}

impl BookStore for DiskBookStore {
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_book(data, flags)?;
        self.write_to_commits(&cid, data);
        self.cache.borrow_mut().insert(cid, data.to_vec());
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.write_to_commits(&cid, &data);
        self.cache.borrow_mut().insert(cid, data);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        // Populate cache from disk or network if needed
        if !self.cache.borrow().contains_key(cid) {
            if let Some(data) = self.read_local(cid) {
                self.cache.borrow_mut().insert(*cid, data);
            } else if let Some(ref fetcher) = self.fetcher {
                if let Some(data) = fetcher(cid) {
                    if Self::verify_cid(cid, &data) {
                        // Cache on disk and in memory
                        self.write_to_commits(cid, &data);
                        self.cache.borrow_mut().insert(*cid, data);
                    }
                    // CID mismatch: corrupted data from network, ignore
                }
            }
        }

        // SAFETY: we hold a Ref for the lifetime of &self. This works
        // because BookStore::get borrows &self, and the RefCell borrow
        // lives as long as the returned reference. We use unsafe to
        // extend the Ref lifetime to match &self.
        //
        // This is sound because:
        // 1. &self ensures no &mut self calls happen concurrently
        // 2. The HashMap only grows (no removes during get), so
        //    existing pointers remain valid
        let cache = self.cache.borrow();
        cache.get(cid).map(|v| {
            let ptr = v.as_ptr();
            let len = v.len();
            unsafe { std::slice::from_raw_parts(ptr, len) }
        })
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.cache.borrow().contains_key(cid) || self.read_local(cid).is_some()
    }

    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        self.cache.borrow_mut().remove(cid)
    }
}
```

Note on the `unsafe`: `BookStore::get(&self) -> Option<&[u8]>` requires the returned slice to live as long as `&self`. With `RefCell`, the `Ref` guard would drop at the end of `get()`, invalidating the slice. The unsafe block extends the lifetime by creating a raw slice that lives as long as the cache's allocation. This is sound because: (1) we hold `&self` so no `&mut self` methods can run, (2) the HashMap only grows during `get()` (we only insert, never remove), so existing allocations are stable.

If you're uncomfortable with this unsafe, an alternative is to change the approach: instead of implementing `BookStore` directly, provide a `get_or_fetch(&mut self, cid) -> Option<&[u8]>` method and have `rebuild_from` accept a different trait. But that would require modifying harmony-db's API. The `unsafe` here is a pragmatic choice that fits the existing trait.

- [ ] **Step 6: Run tests to verify they pass**

Run: `cargo test -p harmony-db disk_book -- --nocapture 2>&1`
Expected: PASS (all 6 tests)

- [ ] **Step 7: Run all harmony-db tests**

Run: `cargo test -p harmony-db -- --nocapture 2>&1`
Expected: All tests PASS.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-db/src/store.rs crates/harmony-db/src/lib.rs
git commit -m "feat(db): add DiskBookStore with network fetch fallback (ZEB-108)"
```

---

### Task 3: Add network fetch callback test

**Files:**
- Modify: `crates/harmony-db/src/store.rs` (tests)

- [ ] **Step 1: Write the test**

Add to the test module in `crates/harmony-db/src/store.rs`:

```rust
#[test]
fn get_falls_through_to_fetcher_on_miss() {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path();
    persist::ensure_dirs(data_dir).unwrap();

    let data = b"remote data";
    let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();

    // Create store with a fetcher that returns the data
    let store = DiskBookStore::with_fetcher(data_dir, move |requested_cid| {
        if *requested_cid == cid {
            Some(data.to_vec())
        } else {
            None
        }
    });

    // Not on disk, but fetcher provides it
    assert_eq!(store.get(&cid).unwrap(), data);

    // Should now be cached on disk too
    let hex = hex::encode(cid.to_bytes());
    assert!(data_dir.join("commits").join(format!("{hex}.bin")).exists());
}

#[test]
fn get_rejects_corrupted_fetch() {
    let dir = TempDir::new().unwrap();
    let data_dir = dir.path();
    persist::ensure_dirs(data_dir).unwrap();

    let real_data = b"real data";
    let cid = ContentId::for_book(real_data, ContentFlags::default()).unwrap();

    // Fetcher returns wrong data for the CID
    let store = DiskBookStore::with_fetcher(data_dir, move |_| {
        Some(b"corrupted data".to_vec())
    });

    // Should reject because BLAKE3 hash won't match
    assert!(store.get(&cid).is_none());
}

#[test]
fn fetcher_not_called_when_local_hit() {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    let dir = TempDir::new().unwrap();
    let data_dir = dir.path();
    persist::ensure_dirs(data_dir).unwrap();

    // Write data locally
    let data = b"local data";
    let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
    let hex = hex::encode(cid.to_bytes());
    std::fs::write(
        data_dir.join("commits").join(format!("{hex}.bin")),
        data,
    )
    .unwrap();

    let fetcher_called = Arc::new(AtomicBool::new(false));
    let fetcher_called_clone = fetcher_called.clone();

    let store = DiskBookStore::with_fetcher(data_dir, move |_| {
        fetcher_called_clone.store(true, Ordering::SeqCst);
        None
    });

    assert_eq!(store.get(&cid).unwrap(), data);
    assert!(!fetcher_called.load(Ordering::SeqCst), "fetcher should not be called for local hit");
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-db fetcher corrupted fetcher_not_called -- --nocapture 2>&1`
Expected: PASS (all 3 tests)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/store.rs
git commit -m "test(db): add DiskBookStore network fetch and integrity tests (ZEB-108)"
```

---

### Task 4: End-to-end sync integration test

**Files:**
- Modify: `crates/harmony-db/src/store.rs` (tests)

This test simulates the full sync flow: writer commits, reader syncs via `rebuild_from` using a `DiskBookStore` backed by the writer's CAS.

- [ ] **Step 1: Write the integration test**

Add to the test module in `crates/harmony-db/src/store.rs`:

```rust
#[test]
fn end_to_end_sync_via_disk_book_store() {
    let writer_dir = TempDir::new().unwrap();
    let reader_dir = TempDir::new().unwrap();

    // Writer: create db, insert entries, commit
    let mut writer_db = crate::HarmonyDb::open(writer_dir.path()).unwrap();
    writer_db
        .insert(
            "mail",
            b"msg-001",
            b"Hello from writer",
            crate::EntryMeta {
                flags: 0,
                snippet: "Hello".into(),
            },
        )
        .unwrap();
    writer_db
        .insert(
            "mail",
            b"msg-002",
            b"Second message",
            crate::EntryMeta {
                flags: 0,
                snippet: "Second".into(),
            },
        )
        .unwrap();
    let root_cid = writer_db.commit(None).unwrap();

    // Reader: create empty db, sync from writer's CAS
    let mut reader_db = crate::HarmonyDb::open(reader_dir.path()).unwrap();

    // Simulate network: fetcher reads from writer's data_dir
    let writer_path = writer_dir.path().to_path_buf();
    let fetcher_store = DiskBookStore::with_fetcher(reader_dir.path(), move |cid| {
        let hex = hex::encode(cid.to_bytes());
        // Try commits/ then blobs/ in writer's directory
        let commit_path = writer_path.join("commits").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&commit_path) {
            return Some(data);
        }
        let blob_path = writer_path.join("blobs").join(format!("{hex}.bin"));
        if let Ok(data) = std::fs::read(&blob_path) {
            return Some(data);
        }
        None
    });

    reader_db.rebuild_from(root_cid, Some(&fetcher_store)).unwrap();

    // Verify reader has same state
    assert_eq!(reader_db.head(), Some(root_cid));
    assert_eq!(reader_db.table_names(), writer_db.table_names());

    let reader_val = reader_db.get("mail", b"msg-001").unwrap();
    assert_eq!(reader_val.as_deref(), Some(b"Hello from writer".as_slice()));

    let reader_val2 = reader_db.get("mail", b"msg-002").unwrap();
    assert_eq!(reader_val2.as_deref(), Some(b"Second message".as_slice()));
}

#[test]
fn incremental_sync_reuses_existing_blocks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    let writer_dir = TempDir::new().unwrap();
    let reader_dir = TempDir::new().unwrap();

    // Writer: create db with initial data
    let mut writer_db = crate::HarmonyDb::open(writer_dir.path()).unwrap();
    for i in 0..10 {
        writer_db
            .insert(
                "data",
                format!("key-{i:03}").as_bytes(),
                format!("value-{i}").as_bytes(),
                crate::EntryMeta {
                    flags: 0,
                    snippet: format!("v{i}"),
                },
            )
            .unwrap();
    }
    let root_v1 = writer_db.commit(None).unwrap();

    // Initial sync
    let mut reader_db = crate::HarmonyDb::open(reader_dir.path()).unwrap();
    let writer_path = writer_dir.path().to_path_buf();
    let wp = writer_path.clone();
    let store_v1 = DiskBookStore::with_fetcher(reader_dir.path(), move |cid| {
        let hex = hex::encode(cid.to_bytes());
        std::fs::read(wp.join("commits").join(format!("{hex}.bin")))
            .or_else(|_| std::fs::read(wp.join("blobs").join(format!("{hex}.bin"))))
            .ok()
    });
    reader_db.rebuild_from(root_v1, Some(&store_v1)).unwrap();

    // Writer adds one more entry
    writer_db
        .insert(
            "data",
            b"key-010",
            b"value-10",
            crate::EntryMeta {
                flags: 0,
                snippet: "v10".into(),
            },
        )
        .unwrap();
    let root_v2 = writer_db.commit(None).unwrap();

    // Incremental sync — count fetcher calls
    let fetch_count = Arc::new(AtomicUsize::new(0));
    let fc = fetch_count.clone();
    let wp2 = writer_path.clone();
    let store_v2 = DiskBookStore::with_fetcher(reader_dir.path(), move |cid| {
        fc.fetch_add(1, Ordering::SeqCst);
        let hex = hex::encode(cid.to_bytes());
        std::fs::read(wp2.join("commits").join(format!("{hex}.bin")))
            .or_else(|_| std::fs::read(wp2.join("blobs").join(format!("{hex}.bin"))))
            .ok()
    });
    reader_db.rebuild_from(root_v2, Some(&store_v2)).unwrap();

    assert_eq!(reader_db.head(), Some(root_v2));
    let val = reader_db.get("data", b"key-010").unwrap();
    assert_eq!(val.as_deref(), Some(b"value-10".as_slice()));

    // Structural sharing: should fetch far fewer blocks than total
    // (only changed nodes + new value blob, not the whole tree)
    let fetches = fetch_count.load(Ordering::SeqCst);
    assert!(
        fetches < 10,
        "incremental sync should fetch fewer blocks than total entries, got {fetches}"
    );
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-db end_to_end_sync incremental_sync -- --nocapture 2>&1`
Expected: PASS (both tests)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/store.rs
git commit -m "test(db): add end-to-end sync and incremental sync tests (ZEB-108)"
```

---

### Task 5: Final verification

**Files:**
- None (verification only)

- [ ] **Step 1: Run full test suite for both crates**

Run: `cargo test -p harmony-zenoh -p harmony-db 2>&1`
Expected: All tests PASS.

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-zenoh -p harmony-db -- -D warnings 2>&1`
Expected: No warnings.

- [ ] **Step 3: Workspace check**

Run: `cargo check --workspace 2>&1`
Expected: No errors.

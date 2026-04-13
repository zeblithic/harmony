//! Disk-backed BookStore with optional network fallback.

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use harmony_content::book::BookStore;
use harmony_content::cid::{ContentFlags, ContentId};
use harmony_content::error::ContentError;


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
    cache: RefCell<HashMap<ContentId, Vec<u8>>>,
    /// Optional network fetch callback: given a CID, returns bytes if available.
    fetcher: Option<Box<dyn Fn(&ContentId) -> Option<Vec<u8>>>>,
}

impl DiskBookStore {
    /// Create a store backed by the given data directory (no network fallback).
    ///
    /// The `commits/` and `blobs/` subdirectories must already exist
    /// (call `persist::ensure_dirs` first, or use `HarmonyDb::open`
    /// which does this automatically).
    pub fn new(data_dir: &Path) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: RefCell::new(HashMap::new()),
            fetcher: None,
        }
    }

    /// Create a store with a network fetch callback for missing CIDs.
    ///
    /// The callback is invoked when `get()` can't find a CID locally.
    /// Fetched data is verified (CID must match BLAKE3 hash) and cached
    /// locally in both the in-memory cache and on disk.
    ///
    /// The `commits/` and `blobs/` subdirectories must already exist.
    pub fn with_fetcher(
        data_dir: &Path,
        fetcher: impl Fn(&ContentId) -> Option<Vec<u8>> + 'static,
    ) -> Self {
        Self {
            data_dir: data_dir.to_path_buf(),
            cache: RefCell::new(HashMap::new()),
            fetcher: Some(Box::new(fetcher)),
        }
    }

    /// Check if a CID exists on local disk (commits/ or blobs/) without reading contents.
    fn exists_local(&self, cid: &ContentId) -> bool {
        let hex = hex::encode(cid.to_bytes());
        self.data_dir.join("commits").join(format!("{hex}.bin")).exists()
            || self.data_dir.join("blobs").join(format!("{hex}.bin")).exists()
    }

    /// Try to read a CID from local CAS (commits/ then blobs/).
    ///
    /// Returns `Ok(None)` only when both paths are `NotFound`.
    /// Non-NotFound IO errors (permissions, disk failure) propagate
    /// so callers don't silently fall through to network fetch.
    fn read_local(&self, cid: &ContentId) -> std::io::Result<Option<Vec<u8>>> {
        let hex = hex::encode(cid.to_bytes());
        let commit_path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        match std::fs::read(&commit_path) {
            Ok(data) => return Ok(Some(data)),
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            Err(_) => {}
        }
        let blob_path = self.data_dir.join("blobs").join(format!("{hex}.bin"));
        match std::fs::read(&blob_path) {
            Ok(data) => return Ok(Some(data)),
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(e),
            Err(_) => {}
        }
        Ok(None)
    }

    /// Write data to commits/ directory (atomic via tmp + rename).
    fn write_to_commits(&self, cid: &ContentId, data: &[u8]) -> std::io::Result<()> {
        let hex = hex::encode(cid.to_bytes());
        let path = self.data_dir.join("commits").join(format!("{hex}.bin"));
        if !path.exists() {
            let tmp = self.data_dir.join("commits").join(format!("{hex}.bin.tmp"));
            std::fs::write(&tmp, data)?;
            std::fs::rename(&tmp, &path)?;
        }
        Ok(())
    }

    /// Verify that data matches the expected CID (any CID type).
    fn verify_cid(cid: &ContentId, data: &[u8]) -> bool {
        cid.verify_hash(data)
    }
}

impl BookStore for DiskBookStore {
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_book(data, flags)?;
        self.write_to_commits(&cid, data)
            .map_err(|_| ContentError::StorageFailed)?;
        self.cache.borrow_mut().insert(cid, data.to_vec());
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        // store() returns () per trait — skip cache if disk write fails
        // to avoid RAM-only ghost data.
        if self.write_to_commits(&cid, &data).is_ok() {
            self.cache.borrow_mut().insert(cid, data);
        }
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        // Populate cache from disk or network if needed
        if !self.cache.borrow().contains_key(cid) {
            match self.read_local(cid) {
                Ok(Some(data)) => {
                    self.cache.borrow_mut().insert(*cid, data);
                }
                Ok(None) => {
                    // Not on disk — try network fetch
                    if let Some(ref fetcher) = self.fetcher {
                        if let Some(data) = fetcher(cid) {
                            if Self::verify_cid(cid, &data) {
                                // Unlike insert/store, this is a read-side cache:
                                // the caller needs this data now and it's CID-verified.
                                // Disk write is best-effort for future sessions.
                                let _ = self.write_to_commits(cid, &data);
                                self.cache.borrow_mut().insert(*cid, data);
                            }
                            // CID mismatch: corrupted data from network, ignore
                        }
                    }
                }
                Err(_) => {
                    // Non-NotFound IO error — local CAS is broken, don't
                    // fall through to network fetch.
                }
            }
        }

        // SAFETY: We need to return &[u8] with lifetime tied to &self,
        // but RefCell::borrow() returns a Ref guard that drops at end
        // of this function. We use unsafe to create a slice from the
        // cached Vec's allocation.
        //
        // This is sound because:
        // 1. &self prevents concurrent &mut self calls
        // 2. The HashMap only grows during get() (insert, never remove)
        //    so existing Vec allocations remain stable
        // 3. The Vec inside the HashMap is never modified after insertion
        let cache = self.cache.borrow();
        cache.get(cid).map(|v| {
            let ptr = v.as_ptr();
            let len = v.len();
            unsafe { std::slice::from_raw_parts(ptr, len) }
        })
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.cache.borrow().contains_key(cid) || self.exists_local(cid)
    }

    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        let data = self.cache.borrow_mut().remove(cid)
            .or_else(|| self.read_local(cid).unwrap_or(None));

        // Remove from disk (both possible locations).
        let hex = hex::encode(cid.to_bytes());
        let _ = std::fs::remove_file(self.data_dir.join("commits").join(format!("{hex}.bin")));
        let _ = std::fs::remove_file(self.data_dir.join("blobs").join(format!("{hex}.bin")));

        data
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use tempfile::TempDir;

    #[test]
    fn get_returns_committed_node() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

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

    #[test]
    fn get_falls_through_to_fetcher_on_miss() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

        let data = b"remote data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();

        let store = DiskBookStore::with_fetcher(data_dir, move |requested_cid| {
            if *requested_cid == cid {
                Some(data.to_vec())
            } else {
                None
            }
        });

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

        let store = DiskBookStore::with_fetcher(data_dir, move |_| {
            Some(b"corrupted data".to_vec())
        });

        assert!(store.get(&cid).is_none());
    }

    #[test]
    fn fetcher_not_called_when_local_hit() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        let dir = TempDir::new().unwrap();
        let data_dir = dir.path();
        persist::ensure_dirs(data_dir).unwrap();

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

    #[test]
    fn insert_with_flags_returns_err_on_disk_failure() {
        let dir = TempDir::new().unwrap();
        // Don't call ensure_dirs — commits/ missing → write fails
        let mut store = DiskBookStore::new(dir.path());
        let result = store.insert_with_flags(b"data", ContentFlags::default());
        assert!(matches!(result, Err(ContentError::StorageFailed)));
    }

    #[test]
    fn store_skips_cache_on_disk_failure() {
        let dir = TempDir::new().unwrap();
        // Don't call ensure_dirs — commits/ missing → write fails
        let data = b"data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let mut store = DiskBookStore::new(dir.path());
        store.store(cid, data.to_vec());
        assert!(!store.contains(&cid));
        assert!(store.get(&cid).is_none());
    }

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
            std::fs::read(writer_path.join("commits").join(format!("{hex}.bin")))
                .or_else(|_| std::fs::read(writer_path.join("blobs").join(format!("{hex}.bin"))))
                .ok()
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
        let fetches = fetch_count.load(Ordering::SeqCst);
        assert!(
            fetches < 10,
            "incremental sync should fetch fewer blocks than total entries, got {fetches}"
        );
    }
}

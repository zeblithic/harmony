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
        self.cache.borrow().contains_key(cid) || self.read_local(cid).is_some()
    }

    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        self.cache.borrow_mut().remove(cid)
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
}

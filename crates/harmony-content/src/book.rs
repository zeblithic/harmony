use alloc::vec::Vec;
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

use crate::cid::{ContentFlags, ContentId};
use crate::error::ContentError;

/// A content-addressed store for book data (CID-addressed units up to 1 MB).
pub trait BookStore {
    /// Insert raw book data with explicit flags, returning the book's ContentId.
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError>;

    /// Insert raw book data, returning the book's ContentId.
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        self.insert_with_flags(data, ContentFlags::default())
    }

    /// Store data under a pre-computed CID (used for bundles).
    fn store(&mut self, cid: ContentId, data: Vec<u8>);

    /// Retrieve data by CID.
    fn get(&self, cid: &ContentId) -> Option<&[u8]>;

    /// Check if a CID exists in the store.
    fn contains(&self, cid: &ContentId) -> bool;

    /// Remove data by CID, returning the data if it was present.
    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>>;
}

/// In-memory content-addressed store backed by a HashMap.
pub struct MemoryBookStore {
    data: HashMap<ContentId, Vec<u8>>,
}

impl MemoryBookStore {
    pub fn new() -> Self {
        MemoryBookStore {
            data: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the store and return all (CID, data) pairs.
    pub fn into_books(self) -> impl Iterator<Item = (ContentId, Vec<u8>)> {
        self.data.into_iter()
    }
}

impl Default for MemoryBookStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BookStore for MemoryBookStore {
    fn insert_with_flags(
        &mut self,
        data: &[u8],
        flags: ContentFlags,
    ) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_book(data, flags)?;
        self.data.entry(cid).or_insert_with(|| data.to_vec());
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.data.entry(cid).or_insert(data);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.data.get(cid).map(|v| v.as_slice())
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.data.contains_key(cid)
    }

    fn remove(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        self.data.remove(cid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::CidType;

    #[test]
    fn insert_and_get_round_trip() {
        let mut store = MemoryBookStore::new();
        let data = b"hello harmony book store";
        let cid = store.insert(data).unwrap();
        assert_eq!(cid.cid_type(), CidType::Book);
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn duplicate_insert_returns_same_cid() {
        let mut store = MemoryBookStore::new();
        let data = b"duplicate data";
        let cid1 = store.insert(data).unwrap();
        let cid2 = store.insert(data).unwrap();
        assert_eq!(cid1, cid2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn get_unknown_returns_none() {
        let store = MemoryBookStore::new();
        let cid = ContentId::for_book(b"not stored", ContentFlags::default()).unwrap();
        assert!(store.get(&cid).is_none());
        assert!(!store.contains(&cid));
    }

    #[test]
    fn contains_reflects_state() {
        let mut store = MemoryBookStore::new();
        let cid = store.insert(b"exists").unwrap();
        assert!(store.contains(&cid));
    }

    #[test]
    fn insert_with_flags_encrypted_book() {
        let mut store = MemoryBookStore::new();
        let flags = ContentFlags {
            encrypted: true,
            ..ContentFlags::default()
        };
        let data = b"encrypted payload";
        let cid = store.insert_with_flags(data, flags).unwrap();
        assert_eq!(cid.cid_type(), CidType::Book);
        assert_eq!(cid.flags().encrypted, true);
        assert_eq!(store.get(&cid).unwrap(), data);
        assert!(store.contains(&cid));
    }

    #[test]
    fn store_raw_for_bundle_data() {
        let mut store = MemoryBookStore::new();
        let blob_a = ContentId::for_book(b"aaa", ContentFlags::default()).unwrap();
        let blob_b = ContentId::for_book(b"bbb", ContentFlags::default()).unwrap();

        // Build bundle bytes manually
        let mut bundle_bytes = Vec::new();
        bundle_bytes.extend_from_slice(&blob_a.to_bytes());
        bundle_bytes.extend_from_slice(&blob_b.to_bytes());
        let bundle_cid =
            ContentId::for_bundle(&bundle_bytes, &[blob_a, blob_b], ContentFlags::default())
                .unwrap();

        store.store(bundle_cid, bundle_bytes.clone());
        assert_eq!(store.get(&bundle_cid).unwrap(), bundle_bytes.as_slice());
    }
}

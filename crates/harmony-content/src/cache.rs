use std::collections::HashSet;

use crate::blob::BlobStore;
use crate::cid::ContentId;
use crate::error::ContentError;
use crate::lru::Lru;
use crate::sketch::CountMinSketch;

/// A content store that wraps any [`BlobStore`] with W-TinyLFU admission.
///
/// The cache tracks access frequency via a [`CountMinSketch`] and maintains
/// three LRU segments (window, probation, protected) following the W-TinyLFU
/// design. A pinned set allows callers to protect specific CIDs from eviction
/// in future tasks.
///
/// This is the scaffold: insertion delegates to the backing store and records
/// the CID in the window segment. The full admission challenge (window victim
/// vs. probation victim frequency comparison) will be added in Task 5.
pub struct ContentStore<S: BlobStore> {
    store: S,
    sketch: CountMinSketch,
    window: Lru,
    probation: Lru,
    protected: Lru,
    #[allow(dead_code)] // Used in Task 8: pin/unpin support
    pinned: HashSet<ContentId>,
}

impl<S: BlobStore> ContentStore<S> {
    /// Create a new `ContentStore` wrapping the given backing store.
    ///
    /// The `capacity` is split across three LRU segments:
    /// - **window:** 1% of capacity (at least 1)
    /// - **protected:** 20% of capacity
    /// - **probation:** remainder (at least 1)
    ///
    /// The sketch width is `capacity * 2` and the halving threshold is
    /// `capacity * 10`.
    pub fn new(store: S, capacity: usize) -> Self {
        let window_cap = std::cmp::max(1, capacity / 100);
        let protected_cap = capacity / 5;
        let probation_cap = std::cmp::max(1, capacity.saturating_sub(window_cap + protected_cap));

        let sketch_width = capacity * 2;
        let halving_threshold = (capacity as u64) * 10;

        ContentStore {
            store,
            sketch: CountMinSketch::new(sketch_width, halving_threshold),
            window: Lru::new(window_cap),
            probation: Lru::new(probation_cap),
            protected: Lru::new(protected_cap),
            pinned: HashSet::new(),
        }
    }

    /// Record a CID in the cache metadata.
    ///
    /// Increments the sketch frequency counter and inserts into the window
    /// segment if the CID is not already tracked in any segment.
    /// The full admission challenge (comparing window victim vs. probation
    /// victim frequencies) will be added in Task 5.
    fn admit(&mut self, cid: ContentId) {
        self.sketch.increment(&cid);
        if !self.window.contains(&cid)
            && !self.probation.contains(&cid)
            && !self.protected.contains(&cid)
        {
            self.window.insert(cid);
        }
    }
}

impl<S: BlobStore> BlobStore for ContentStore<S> {
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        let cid = self.store.insert(data)?;
        self.admit(cid);
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.store.store(cid, data);
        self.admit(cid);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.store.get(cid)
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.store.contains(cid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;

    #[test]
    fn content_store_implements_blobstore() {
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 100);
        let data = b"hello cache";
        let cid = cs.insert(data).unwrap();
        assert!(cs.contains(&cid));
        assert_eq!(cs.get(&cid).unwrap(), data);
    }
}

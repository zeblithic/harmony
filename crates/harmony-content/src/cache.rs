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
/// On insertion, new CIDs enter the **window** segment. When the window
/// overflows, the evicted candidate faces an **admission challenge** against
/// the probation segment's LRU victim: the candidate is admitted to probation
/// only if its estimated frequency exceeds the victim's. On cache hits,
/// probation entries are promoted to protected; protected entries are touched
/// in place.
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

    /// Record an access for a CID already in the cache.
    ///
    /// Called by callers on cache hits to maintain frequency data and adjust
    /// segment placement:
    /// - **Protected:** touch (move to head).
    /// - **Probation:** promote to protected. If protected overflows, demote
    ///   the victim back to probation head.
    /// - **Window:** touch (move to head).
    pub fn record_access(&mut self, cid: &ContentId) {
        self.sketch.increment(cid);

        if self.protected.contains(cid) {
            self.protected.touch(cid);
        } else if self.probation.contains(cid) {
            // Promote from probation to protected.
            self.probation.remove(cid);
            if let Some(demoted) = self.protected.insert(*cid) {
                // Protected overflowed — demote victim back to probation head.
                self.probation.insert(demoted);
            }
        } else if self.window.contains(cid) {
            self.window.touch(cid);
        }
    }

    /// Record a CID in the cache metadata with full W-TinyLFU admission.
    ///
    /// Increments the sketch frequency counter. If the CID is already tracked
    /// in any segment, delegates to [`record_access`]. Otherwise inserts into
    /// the window segment. If the window overflows, the evicted candidate
    /// faces an admission challenge against the probation LRU victim.
    fn admit(&mut self, cid: ContentId) {
        // Already tracked — just update placement.
        if self.window.contains(&cid)
            || self.probation.contains(&cid)
            || self.protected.contains(&cid)
        {
            self.record_access(&cid);
            return;
        }

        // New CID — increment sketch once, then insert into window.
        self.sketch.increment(&cid);
        if let Some(candidate) = self.window.insert(cid) {
            // Window overflowed — run admission challenge.
            self.admission_challenge(candidate);
        }
    }

    /// Try to admit the window evictee (candidate) into probation.
    ///
    /// Peeks at probation's LRU victim and compares frequencies. If the
    /// candidate has higher frequency, it replaces the victim. Otherwise
    /// the candidate is dropped and probation is untouched.
    fn admission_challenge(&mut self, candidate: ContentId) {
        if let Some(victim) = self.probation.peek_lru() {
            if self.probation.len() < self.probation.capacity() {
                // Probation has space — admit directly.
                self.probation.insert(candidate);
                return;
            }
            // Probation full — compare frequencies.
            let candidate_freq = self.sketch.estimate(&candidate);
            let victim_freq = self.sketch.estimate(&victim);

            if candidate_freq > victim_freq {
                // Candidate wins — evict victim, admit candidate.
                self.probation.remove(&victim);
                self.store_remove(&victim);
                self.probation.insert(candidate);
            } else {
                // Victim wins — drop candidate entirely.
                self.store_remove(&candidate);
            }
        } else {
            // Probation is empty — admit directly.
            self.probation.insert(candidate);
        }
    }

    /// Remove a CID from the backing store.
    ///
    /// No-op for now: `MemoryBlobStore` does not support removal. The CID is
    /// no longer tracked by the cache but data stays in the store.
    fn store_remove(&mut self, _cid: &ContentId) {
        // No-op — backing store removal not yet supported.
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

    #[test]
    fn frequency_based_admission() {
        // Capacity 10: window=1, protected=2, probation=7.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);

        // Fill probation with 7 items, each accessed 5 times to build frequency.
        let hot: Vec<ContentId> = (0..7)
            .map(|i| {
                let data = format!("hot-{i}");
                cs.insert(data.as_bytes()).unwrap()
            })
            .collect();
        for cid in &hot {
            for _ in 0..5 {
                cs.record_access(cid);
            }
        }

        // Insert a cold item — it enters window, then faces admission challenge.
        // Its frequency (1) should lose against probation's tail (5+).
        let _cold = cs.insert(b"cold-newcomer").unwrap();

        // The hot items should still be accessible in the cache segments.
        for cid in &hot {
            assert!(
                cs.window.contains(cid)
                    || cs.probation.contains(cid)
                    || cs.protected.contains(cid),
                "hot CID should still be in cache"
            );
        }
    }

    #[test]
    fn probation_to_protected_promotion() {
        // Capacity 20: window=1, protected=4, probation=15.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);

        // Insert an item — it enters window, then gets pushed to probation
        // when the next item takes its window slot.
        let target = cs.insert(b"target").unwrap();
        let _pusher = cs.insert(b"pusher").unwrap();

        // target should now be in probation (pushed out of window by pusher).
        // Access it to promote to protected.
        cs.record_access(&target);
        assert!(cs.protected.contains(&target));
        assert!(!cs.probation.contains(&target));
    }

    #[test]
    fn scan_resistance() {
        // The core W-TinyLFU property: a sequential scan of cold data
        // should NOT evict a frequently-accessed hot item.

        // Capacity 20: window=1, protected=4, probation=15.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);

        // Create a hot item and give it lots of frequency.
        let hot = cs.insert(b"frequently-accessed").unwrap();
        for _ in 0..10 {
            cs.record_access(&hot);
        }

        // Sequential scan: insert 50 unique cold items (never accessed again).
        for i in 0..50 {
            let data = format!("scan-item-{i}");
            cs.insert(data.as_bytes()).unwrap();
        }

        // The hot item should survive the scan.
        assert!(
            cs.window.contains(&hot)
                || cs.probation.contains(&hot)
                || cs.protected.contains(&hot),
            "hot CID should survive sequential scan"
        );
    }
}

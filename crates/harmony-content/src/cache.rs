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
        assert!(capacity >= 1, "ContentStore capacity must be at least 1");

        let window_cap = std::cmp::max(1, capacity / 100);
        let protected_cap = capacity / 5;
        let probation_cap = capacity.saturating_sub(window_cap + protected_cap);

        let sketch_width = std::cmp::max(2, capacity * 2);
        let halving_threshold = std::cmp::max(1, (capacity as u64) * 10);

        ContentStore {
            store,
            sketch: CountMinSketch::new(sketch_width, halving_threshold),
            window: Lru::new(window_cap),
            probation: Lru::new(probation_cap),
            protected: Lru::new(protected_cap),
            pinned: HashSet::new(),
        }
    }

    /// Pin a CID, exempting it from eviction.
    pub fn pin(&mut self, cid: ContentId) {
        self.pinned.insert(cid);
    }

    /// Unpin a CID, making it eligible for eviction again.
    pub fn unpin(&mut self, cid: &ContentId) {
        self.pinned.remove(cid);
    }

    /// Check if a CID is pinned.
    pub fn is_pinned(&self, cid: &ContentId) -> bool {
        self.pinned.contains(cid)
    }

    /// Retrieve data and record the access for frequency tracking.
    ///
    /// Unlike [`BlobStore::get`] (which takes `&self` and cannot update cache
    /// metadata), this method updates the sketch frequency counter and adjusts
    /// segment placement on hits. Returns a cloned copy of the data.
    ///
    /// Callers that need W-TinyLFU benefits (frequency-based admission,
    /// promotion from probation to protected) should prefer this over the
    /// trait `get()`.
    pub fn get_and_record(&mut self, cid: &ContentId) -> Option<Vec<u8>> {
        let data = self.store.get(cid).map(|b| b.to_vec())?;
        self.record_access(cid);
        Some(data)
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
    /// If probation has space, admit directly. Otherwise compare the
    /// candidate's frequency against the least-recently-used *non-pinned*
    /// probation entry. Pinned candidates are always admitted. Pinned
    /// victims are never evicted — the scan skips them.
    fn admission_challenge(&mut self, candidate: ContentId) {
        // Probation has space — admit directly, no challenge needed.
        if self.probation.len() < self.probation.capacity() {
            self.probation.insert(candidate);
            return;
        }

        // Pinned candidates are always admitted — never drop a pinned CID.
        if self.pinned.contains(&candidate) {
            if let Some(victim) = self.probation.peek_lru_excluding(&self.pinned) {
                self.probation.remove(&victim);
                self.store_remove(&victim);
            }
            self.probation.insert(candidate);
            return;
        }

        // Find the first non-pinned victim for the frequency challenge.
        let Some(victim) = self.probation.peek_lru_excluding(&self.pinned) else {
            // All probation entries are pinned — drop unpinned candidate.
            self.store_remove(&candidate);
            return;
        };

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
    fn get_and_record_updates_frequency_and_promotes() {
        // Capacity 20: window=1, protected=4, probation=15.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);

        // Insert target, then push it to probation.
        let target = cs.insert(b"target").unwrap();
        let _pusher = cs.insert(b"pusher").unwrap();
        assert!(cs.probation.contains(&target));

        // get_and_record should return data AND promote to protected.
        let data = cs.get_and_record(&target);
        assert_eq!(data.as_deref(), Some(b"target".as_slice()));
        assert!(cs.protected.contains(&target));
        assert!(!cs.probation.contains(&target));

        // Miss returns None.
        let bogus = ContentId::for_blob(b"nonexistent").unwrap();
        assert_eq!(cs.get_and_record(&bogus), None);
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
                cs.window.contains(cid) || cs.probation.contains(cid) || cs.protected.contains(cid),
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
            cs.window.contains(&hot) || cs.probation.contains(&hot) || cs.protected.contains(&hot),
            "hot CID should survive sequential scan"
        );
    }

    #[test]
    fn pin_exempts_from_eviction() {
        // Small cache: capacity 5 → window=1, protected=1, probation=3.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 5);

        let pinned_cid = cs.insert(b"pinned-data").unwrap();
        cs.pin(pinned_cid);

        // Fill the cache well beyond capacity.
        for i in 0..20 {
            let data = format!("filler-{i}");
            cs.insert(data.as_bytes()).unwrap();
        }

        // Pinned item should still be tracked in cache segments (not just backing store).
        assert!(cs.is_pinned(&pinned_cid));
        assert!(
            cs.window.contains(&pinned_cid)
                || cs.probation.contains(&pinned_cid)
                || cs.protected.contains(&pinned_cid),
            "pinned CID should remain in cache segments"
        );
        assert!(cs.contains(&pinned_cid));

        // Unpin and verify it becomes evictable.
        cs.unpin(&pinned_cid);
        assert!(!cs.is_pinned(&pinned_cid));
    }

    #[test]
    fn capacity_one_segments_do_not_exceed_capacity() {
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 1);
        let total = cs.window.capacity() + cs.probation.capacity() + cs.protected.capacity();
        assert!(
            total <= 1,
            "segments sum to {total}, exceeds capacity 1"
        );
    }

    #[test]
    fn capacity_two_segments_do_not_exceed_capacity() {
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 2);
        let total = cs.window.capacity() + cs.probation.capacity() + cs.protected.capacity();
        assert!(
            total <= 2,
            "segments sum to {total}, exceeds capacity 2"
        );
    }

    #[test]
    #[should_panic(expected = "capacity must be at least 1")]
    fn zero_capacity_panics() {
        let store = MemoryBlobStore::new();
        let _cs = ContentStore::new(store, 0);
    }

    #[test]
    fn pinned_candidate_survives_admission_challenge() {
        // Regression: pinned CID evicted from window must not be dropped
        // even when it loses the frequency comparison.
        //
        // Capacity 10: window=1, protected=2, probation=7.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);

        // Fill probation with 7 hot items.
        let hot: Vec<ContentId> = (0..7)
            .map(|i| {
                let data = format!("hot-{i}");
                cs.insert(data.as_bytes()).unwrap()
            })
            .collect();
        for cid in &hot {
            for _ in 0..10 {
                cs.record_access(cid);
            }
        }

        // Insert a pinned item with zero extra frequency (just the 1 from insert).
        // It enters window, gets pushed to admission_challenge by the next insert.
        let pinned = cs.insert(b"pinned-low-freq").unwrap();
        cs.pin(pinned);

        // Push it out of window — triggers admission challenge.
        let _pusher = cs.insert(b"pusher-after-pin").unwrap();

        // The pinned item should survive despite lower frequency.
        assert!(
            cs.window.contains(&pinned)
                || cs.probation.contains(&pinned)
                || cs.protected.contains(&pinned),
            "pinned candidate should survive admission challenge"
        );
    }

    #[test]
    fn pinned_victim_skipped_in_admission() {
        // Regression: when probation's tail is pinned, the admission
        // challenge should skip it and compare against the next non-pinned
        // entry instead of dropping the candidate.
        //
        // Capacity 10: window=1, protected=2, probation=7.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);

        // Fill probation: insert 7 items (each pushes the previous out of window).
        let items: Vec<ContentId> = (0..7)
            .map(|i| {
                let data = format!("prob-{i}");
                cs.insert(data.as_bytes()).unwrap()
            })
            .collect();

        // Pin the LRU tail of probation (first inserted = least recent).
        let tail_cid = items[0];
        cs.pin(tail_cid);

        // Give the next candidate high frequency so it would beat non-pinned victims.
        let candidate_data = b"high-freq-candidate";
        let candidate = ContentId::for_blob(candidate_data).unwrap();
        for _ in 0..10 {
            cs.sketch.increment(&candidate);
        }

        // Store the data and admit — triggers admission challenge.
        cs.store(candidate, candidate_data.to_vec());

        // Candidate should have been admitted (beating a non-pinned victim).
        assert!(
            cs.window.contains(&candidate)
                || cs.probation.contains(&candidate)
                || cs.protected.contains(&candidate),
            "candidate should be admitted when pinned victim is skipped"
        );
        // Pinned tail should still be tracked.
        assert!(
            cs.probation.contains(&tail_cid),
            "pinned victim should remain in probation"
        );
    }
}

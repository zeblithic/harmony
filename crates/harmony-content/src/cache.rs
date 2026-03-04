use std::collections::HashSet;
use std::fmt;

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
///
/// ## Pin Scope Limitation
///
/// Pin checking currently only covers the **admission challenge** path
/// (window→probation eviction). Two other eviction paths do **not** check
/// pins:
///
/// 1. **Window eviction:** When the window segment overflows, the tail is
///    evicted unconditionally and sent to the admission challenge. If a pinned
///    CID is the window tail, it will be evicted from window — but it survives
///    because `admission_challenge` always admits pinned candidates.
///
/// 2. **Protected→probation demotion:** When a probation item is promoted to
///    protected and protected overflows, the protected tail is demoted back to
///    probation unconditionally. A pinned CID in protected can be demoted to
///    probation, but it remains tracked (not lost).
///
/// In both cases, pinned CIDs are never fully dropped from the cache — they
/// just move between segments. However, when `store_remove` gets a real
/// implementation (backing store cleanup on eviction), any eviction path that
/// doesn't check pins could incorrectly remove backing data for a pinned CID.
/// Before implementing `store_remove`, all eviction paths must be audited to
/// skip pinned CIDs.
pub struct ContentStore<S: BlobStore> {
    store: S,
    sketch: CountMinSketch,
    window: Lru,
    probation: Lru,
    protected: Lru,
    pinned: HashSet<ContentId>,
}

impl<S: BlobStore> fmt::Debug for ContentStore<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ContentStore")
            .field("sketch", &self.sketch)
            .field("window", &self.window)
            .field("probation", &self.probation)
            .field("protected", &self.protected)
            .field("pinned_count", &self.pinned.len())
            .finish_non_exhaustive()
    }
}

impl<S: BlobStore> ContentStore<S> {
    /// Create a new `ContentStore` wrapping the given backing store.
    ///
    /// The `capacity` is split across three LRU segments:
    /// - **window:** 1% of capacity (at least 1)
    /// - **protected:** 20% of capacity
    /// - **probation:** remainder
    ///
    /// At small capacities the splits degenerate: capacity=1 gives a single
    /// window slot with no probation or protected (effectively MRU).
    /// capacity=2..4 gives window=1 + probation only (no protected).
    /// capacity=5 is the minimum for all three segments to be non-zero.
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

    /// Maximum number of CIDs that can be pinned.
    ///
    /// Equal to half the total segment capacity. This enforces the storage
    /// model where at most 50% of cache capacity is reserved for pinned
    /// (user-owned, replicated) content, leaving the rest for shared
    /// network caching.
    pub fn pin_limit(&self) -> usize {
        let total = self.window.capacity() + self.probation.capacity() + self.protected.capacity();
        total / 2
    }

    /// Pin a CID, exempting it from eviction.
    ///
    /// Returns `true` if the CID was pinned (or was already pinned).
    /// Returns `false` if the pin quota ([`pin_limit`](Self::pin_limit))
    /// is reached.
    pub fn pin(&mut self, cid: ContentId) -> bool {
        if self.pinned.contains(&cid) {
            return true;
        }
        if self.pinned.len() >= self.pin_limit() {
            return false;
        }
        self.pinned.insert(cid);
        true
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

    /// Check whether a CID is tracked in the cache's LRU segments.
    ///
    /// Unlike [`BlobStore::contains`] (which checks the backing store and may
    /// find data that W-TinyLFU rejected but `store_remove` didn't clean up),
    /// this checks the actual admission state in window/probation/protected.
    pub fn is_admitted(&self, cid: &ContentId) -> bool {
        self.window.contains(cid) || self.probation.contains(cid) || self.protected.contains(cid)
    }

    /// Pre-store admission check for transit/opportunistic content.
    ///
    /// Returns `true` if the CID should be stored based on W-TinyLFU frequency
    /// estimation. Always increments the sketch counter so repeated transits of
    /// the same CID build frequency over time — even rejected items accumulate
    /// popularity, making future admission more likely.
    ///
    /// This is separate from [`store`](BlobStore::store) because `store()`
    /// unconditionally adds the item to the window segment. For transit content,
    /// we want to reject cold items upfront rather than storing and announcing
    /// them only for them to be immediately evicted.
    pub fn should_admit(&mut self, cid: &ContentId) -> bool {
        // Already in cache — always admit.
        if self.window.contains(cid) || self.probation.contains(cid) || self.protected.contains(cid)
        {
            return true;
        }

        // Track popularity even for rejected items.
        self.sketch.increment(cid);

        // If probation has space, any new item can be admitted.
        if self.probation.len() < self.probation.capacity() {
            return true;
        }

        // Compare against probation's LRU victim.
        let freq = self.sketch.estimate(cid);
        match self.probation.peek_lru_excluding(&self.pinned) {
            Some(victim) => freq > self.sketch.estimate(&victim),
            None => true, // All pinned — admit by default.
        }
    }

    /// Store data for a CID that was already checked by [`should_admit`].
    ///
    /// Skips the sketch increment in [`admit`] since `should_admit()` already
    /// incremented it. Use this to avoid double-counting frequency for transit
    /// content that goes through the pre-store admission check.
    pub fn store_preadmitted(&mut self, cid: ContentId, data: Vec<u8>) {
        self.store.store(cid, data);
        self.admit(cid, true);
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
    /// Increments the sketch frequency counter (unless `skip_increment` is true,
    /// indicating the caller already incremented via [`should_admit`]). If the
    /// CID is already tracked in any segment, delegates to [`record_access`].
    /// Otherwise inserts into the window segment. If the window overflows, the
    /// evicted candidate faces an admission challenge against the probation LRU
    /// victim.
    fn admit(&mut self, cid: ContentId, skip_increment: bool) {
        // Already tracked — just update placement.
        if self.window.contains(&cid)
            || self.probation.contains(&cid)
            || self.protected.contains(&cid)
        {
            self.record_access(&cid);
            return;
        }

        // New CID — increment sketch (unless pre-incremented by should_admit).
        if !skip_increment {
            self.sketch.increment(&cid);
        }
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
        // The pin quota (≤ capacity/2) guarantees that at least one non-pinned
        // victim exists in probation when a pinned candidate arrives here.
        if self.pinned.contains(&candidate) {
            let Some(victim) = self.probation.peek_lru_excluding(&self.pinned) else {
                // Quota invariant broken — all probation entries are pinned.
                // Refuse to insert rather than risk evicting a pinned tail.
                // This path is unreachable under correct quota enforcement.
                return;
            };
            self.probation.remove(&victim);
            self.store_remove(&victim);
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
    ///
    /// **WARNING:** Before implementing this, audit ALL eviction paths to check
    /// pins. Currently only `admission_challenge` checks the pinned set.
    /// Window eviction and protected→probation demotion do not. A real
    /// `store_remove` on those paths would destroy backing data for pinned CIDs.
    /// See the "Pin Scope Limitation" section on [`ContentStore`].
    fn store_remove(&mut self, _cid: &ContentId) {
        // No-op — backing store removal not yet supported.
    }
}

impl<S: BlobStore> BlobStore for ContentStore<S> {
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        let cid = self.store.insert(data)?;
        self.admit(cid, false);
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.store.store(cid, data);
        self.admit(cid, false);
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
    fn debug_impl_shows_summary() {
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 100);
        let cid = cs.insert(b"test").unwrap();
        cs.pin(cid);
        let dbg = format!("{cs:?}");
        assert!(dbg.contains("ContentStore"));
        assert!(dbg.contains("sketch:"), "sketch field should appear");
        assert!(dbg.contains("CountMinSketch"));
        assert!(dbg.contains("window:"), "window Lru field should appear");
        assert!(dbg.contains("probation:"), "probation Lru field should appear");
        assert!(dbg.contains("protected:"), "protected Lru field should appear");
        assert!(dbg.contains("Lru"), "Lru Debug impl should be used for segments");
        assert!(dbg.contains("pinned_count: 1, .."), "should use finish_non_exhaustive()");
    }

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
    fn store_preadmitted_does_not_double_increment() {
        // Verify that should_admit + store_preadmitted results in exactly
        // one sketch increment, same as a plain store().
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 100);

        let data_a = b"via-preadmitted";
        let cid_a = ContentId::for_blob(data_a).unwrap();
        assert!(cs.should_admit(&cid_a));
        cs.store_preadmitted(cid_a, data_a.to_vec());
        let freq_a = cs.sketch.estimate(&cid_a);

        let data_b = b"via-plain-store";
        let cid_b = ContentId::for_blob(data_b).unwrap();
        cs.store(cid_b, data_b.to_vec());
        let freq_b = cs.sketch.estimate(&cid_b);

        assert_eq!(
            freq_a, freq_b,
            "preadmitted and plain store should have same frequency"
        );
    }

    #[test]
    fn should_admit_rejects_cold_item_against_hot_probation() {
        // A cold CID (freq 1) should lose the admission check against
        // hot probation items (freq 10+).
        //
        // Capacity 3: window=1, protected=0, probation=2.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 3);

        // Fill probation: insert 3 items → window=1 + probation=2.
        let hot_a = cs.insert(b"hot-a").unwrap();
        let hot_b = cs.insert(b"hot-b").unwrap();
        let _window_item = cs.insert(b"window").unwrap();

        // Boost hot items' frequency (they stay in probation since protected=0).
        for _ in 0..10 {
            cs.record_access(&hot_a);
            cs.record_access(&hot_b);
        }

        // Cold CID with no frequency history should be rejected.
        let cold_cid = ContentId::for_blob(b"cold-newcomer").unwrap();
        assert!(!cs.should_admit(&cold_cid), "cold CID should be rejected");

        // But calling should_admit built some frequency. After enough calls,
        // the CID should eventually be admitted.
        for _ in 0..15 {
            cs.should_admit(&cold_cid);
        }
        assert!(
            cs.should_admit(&cold_cid),
            "warmed-up CID should be admitted"
        );
    }

    #[test]
    fn is_admitted_reflects_lru_not_backing_store() {
        // Verify that is_admitted checks LRU segments, not the backing store.
        // A CID stored via the backing store directly won't appear in LRU.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);

        // Store directly into backing store (bypassing admit).
        let cid = ContentId::for_blob(b"test").unwrap();
        cs.store.store(cid, b"test".to_vec());

        // Backing store has it, but LRU segments don't.
        assert!(cs.contains(&cid), "backing store has the data");
        assert!(!cs.is_admitted(&cid), "LRU segments don't track it");
    }

    #[test]
    fn frequency_based_admission() {
        // Capacity 4: window=1, protected=0, probation=3.
        // Zero protected capacity means record_access builds sketch frequency
        // without moving items out of probation (promote bounces back).
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 4);

        // Fill probation with 3 hot items, each accessed 5 times.
        let hot: Vec<ContentId> = (0..3)
            .map(|i| {
                let data = format!("hot-{i}");
                cs.insert(data.as_bytes()).unwrap()
            })
            .collect();
        // Push the last hot item out of window so all 3 are in probation.
        let _spacer = cs.insert(b"spacer").unwrap();
        for cid in &hot {
            for _ in 0..5 {
                cs.record_access(cid);
            }
        }
        assert_eq!(cs.probation.len(), 3, "probation should be full");

        // Insert a cold item — it enters window, pushing _spacer to admission.
        // _spacer has low frequency so it may be rejected. Then insert another
        // to push the cold item itself into the admission challenge.
        let cold = cs.insert(b"cold-newcomer").unwrap();
        let _pusher = cs.insert(b"pusher").unwrap();

        // The cold item should have been rejected by the admission challenge:
        // its frequency (1) loses against the probation tail's frequency (5+).
        assert!(
            !cs.probation.contains(&cold) && !cs.protected.contains(&cold),
            "cold item should be rejected from main cache segments"
        );

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
    fn scan_resistance_probation_resident() {
        // Variant: hot item stays in probation (not promoted to protected),
        // exercising the admission challenge frequency comparison directly.
        //
        // Capacity 4: window=1, protected=0, probation=3.
        // Zero protected cap means record_access builds sketch frequency
        // without promoting items out of probation.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 4);

        // Insert a hot item, push it to probation, build frequency.
        let hot = cs.insert(b"frequently-accessed").unwrap();
        let _pusher = cs.insert(b"push-to-probation").unwrap();
        assert!(cs.probation.contains(&hot));
        for _ in 0..10 {
            cs.record_access(&hot);
        }

        // Sequential scan: insert 20 cold items.
        // Each scan item enters window, then faces admission_challenge
        // against the probation tail. The hot item's high frequency should
        // protect it from eviction.
        for i in 0..20 {
            let data = format!("scan-item-{i}");
            cs.insert(data.as_bytes()).unwrap();
        }

        // The hot item should survive — its frequency beats every cold item.
        assert!(
            cs.probation.contains(&hot),
            "high-frequency probation item should survive sequential scan"
        );
    }

    #[test]
    fn pin_exempts_from_eviction() {
        // Small cache: capacity 5 → window=1, protected=1, probation=3.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 5);

        let pinned_cid = cs.insert(b"pinned-data").unwrap();
        assert!(cs.pin(pinned_cid));

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
    fn capacity_one_segment_splits() {
        // capacity=1: window gets the minimum of 1, leaving 0 for both
        // probation and protected. Effectively a 1-entry MRU cache with
        // no admission challenge (nothing to challenge against).
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 1);
        assert_eq!(cs.window.capacity(), 1);
        assert_eq!(cs.probation.capacity(), 0);
        assert_eq!(cs.protected.capacity(), 0);
    }

    #[test]
    fn capacity_two_segment_splits() {
        // capacity=2: window=1, probation=1, protected=0.
        // Minimum viable W-TinyLFU: one window slot feeds admission
        // challenge against a single probation entry.
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 2);
        assert_eq!(cs.window.capacity(), 1);
        assert_eq!(cs.probation.capacity(), 1);
        assert_eq!(cs.protected.capacity(), 0);
    }

    #[test]
    fn capacity_four_segment_splits() {
        // capacity=4: window=1, probation=3, protected=0.
        // Still no protected segment (4/5 = 0).
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 4);
        assert_eq!(cs.window.capacity(), 1);
        assert_eq!(cs.probation.capacity(), 3);
        assert_eq!(cs.protected.capacity(), 0);
    }

    #[test]
    fn capacity_five_segment_splits() {
        // capacity=5: window=1, probation=3, protected=1.
        // First capacity where all three segments are non-zero.
        let store = MemoryBlobStore::new();
        let cs = ContentStore::new(store, 5);
        assert_eq!(cs.window.capacity(), 1);
        assert_eq!(cs.probation.capacity(), 3);
        assert_eq!(cs.protected.capacity(), 1);
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
        assert!(cs.pin(pinned));

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
        assert!(cs.pin(tail_cid));

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

    #[test]
    fn pin_quota_rejects_over_limit() {
        // Capacity 10: pin limit = 5.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);
        assert_eq!(cs.pin_limit(), 5);

        let cids: Vec<ContentId> = (0..6)
            .map(|i| cs.insert(format!("item-{i}").as_bytes()).unwrap())
            .collect();

        // First 5 pins succeed.
        for cid in &cids[..5] {
            assert!(cs.pin(*cid), "pin within quota should succeed");
        }

        // 6th pin is rejected.
        assert!(!cs.pin(cids[5]), "pin over quota should fail");

        // Re-pinning an already-pinned CID still succeeds (idempotent).
        assert!(cs.pin(cids[0]), "re-pinning should succeed");

        // After unpin, a new pin succeeds.
        cs.unpin(&cids[0]);
        assert!(cs.pin(cids[5]), "pin after unpin should succeed");
    }

    #[test]
    fn pin_quota_zero_for_capacity_one() {
        // Capacity 1 → total segments = 1 → pin_limit = 0.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 1);
        assert_eq!(cs.pin_limit(), 0);

        let cid = cs.insert(b"data").unwrap();
        assert!(!cs.pin(cid), "capacity-1 cache cannot pin anything");
    }

    #[test]
    fn zero_cap_protected_bounces_to_probation() {
        // Capacity 4: window=1, protected=0, probation=3.
        // When record_access tries to promote from probation to protected,
        // the zero-cap protected segment rejects, and the item stays in probation.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 4);
        assert_eq!(cs.protected.capacity(), 0);

        // Insert target, push to probation.
        let target = cs.insert(b"target").unwrap();
        let _p = cs.insert(b"pusher").unwrap();
        assert!(cs.probation.contains(&target));

        // Record access — should try to promote but bounce back.
        cs.record_access(&target);

        // With zero-cap protected, item stays in probation (not lost).
        assert!(
            cs.probation.contains(&target) || cs.protected.contains(&target),
            "item should still be tracked after promotion bounce"
        );
    }
}

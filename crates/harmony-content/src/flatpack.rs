//! FlatpackIndex: reverse-map index from child CIDs to their containing bundles,
//! backed by a cuckoo filter for fast approximate membership queries.
//!
//! Maintains a forward map (bundle → children) for cleanup on eviction
//! and a reverse map (child → bundles) for lookups. The cuckoo filter
//! summarizes which child CIDs have entries, enabling fast negative lookups
//! without traversing the full reverse map.

use alloc::vec::Vec;
use core::fmt;

#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

use crate::cid::ContentId;
use crate::cuckoo::CuckooFilter;

/// Index tracking which child CIDs are contained within which bundle CIDs.
///
/// The reverse map (`child_cid → {bundle_cids}`) enables lookup of all bundles
/// containing a given child. The forward map (`bundle_cid → [child_cids]`)
/// enables efficient cleanup when a bundle is evicted. The cuckoo filter
/// provides a fast approximate check for whether a child CID is referenced
/// at all, avoiding hash-map lookups on the hot path.
///
/// The `mutations_since_broadcast` counter tracks how many structural changes
/// have occurred since the last filter broadcast, allowing callers to decide
/// when to re-broadcast the filter to peers.
pub struct FlatpackIndex {
    /// child_cid → set of bundle_cids that contain it.
    reverse: HashMap<ContentId, HashSet<ContentId>>,
    /// bundle_cid → ordered list of child_cids (for cleanup on eviction).
    forward: HashMap<ContentId, Vec<ContentId>>,
    /// Approximate membership filter over child CIDs in the reverse map.
    filter: CuckooFilter,
    /// Children that were successfully inserted into the cuckoo filter.
    /// Tracks actual filter membership so that `on_bundle_evicted` only
    /// calls `filter.delete` for children that are genuinely present —
    /// deleting a never-inserted child could remove an unrelated CID's
    /// fingerprint that happens to share the same 12-bit value.
    filter_members: HashSet<ContentId>,
    /// Number of structural mutations since the last broadcast reset.
    mutations_since_broadcast: u32,
}

impl fmt::Debug for FlatpackIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatpackIndex")
            .field("bundles", &self.forward.len())
            .field("child_cids", &self.reverse.len())
            .field("filter", &self.filter)
            .field("mutations_since_broadcast", &self.mutations_since_broadcast)
            .finish_non_exhaustive()
    }
}

impl FlatpackIndex {
    /// Create a new empty index with the given cuckoo filter capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero (forwarded from [`CuckooFilter::new`]).
    pub fn new(capacity: u32) -> Self {
        Self {
            reverse: HashMap::new(),
            forward: HashMap::new(),
            filter: CuckooFilter::new(capacity),
            filter_members: HashSet::new(),
            mutations_since_broadcast: 0,
        }
    }

    /// Record that `bundle_cid` contains the given `child_cids`.
    ///
    /// For each child CID that is new to the reverse map (has no existing
    /// bundle references), the child is also inserted into the cuckoo filter.
    /// The bundle is then added to each child's reverse-map entry, and the
    /// full child list is stored in the forward map for later cleanup.
    ///
    /// Increments the mutation counter.
    pub fn on_bundle_admitted(&mut self, bundle_cid: ContentId, child_cids: Vec<ContentId>) {
        // If this bundle was already admitted (re-admission with possibly
        // different children), clean up old entries first to prevent orphaned
        // reverse-map references.
        if self.forward.contains_key(&bundle_cid) {
            self.on_bundle_evicted(&bundle_cid);
        }

        for child in &child_cids {
            let set = self.reverse.entry(*child).or_default();
            if set.is_empty() {
                // New child — insert into cuckoo filter. Track success so
                // that on_bundle_evicted only deletes confirmed members.
                // FilterFull is tolerable; rebuild_filter will resize later.
                if self.filter.insert(child).is_ok() {
                    self.filter_members.insert(*child);
                }
            }
            set.insert(bundle_cid);
        }
        self.forward.insert(bundle_cid, child_cids);
        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
    }

    /// Remove all index entries associated with `bundle_cid`.
    ///
    /// Looks up the forward map to find the bundle's children. For each child,
    /// removes the bundle from the reverse-map set. If a child's set becomes
    /// empty, the child is removed from the reverse map and deleted from the
    /// cuckoo filter. Finally, the forward-map entry is removed.
    ///
    /// If the bundle is not in the forward map, this is a no-op.
    ///
    /// Increments the mutation counter (unless the bundle was unknown).
    pub fn on_bundle_evicted(&mut self, bundle_cid: &ContentId) {
        let Some(children) = self.forward.remove(bundle_cid) else {
            return;
        };

        for child in &children {
            if let Some(set) = self.reverse.get_mut(child) {
                set.remove(bundle_cid);
                if set.is_empty() {
                    self.reverse.remove(child);
                    // Only delete from cuckoo filter if we successfully
                    // inserted it — otherwise we'd remove an unrelated
                    // CID's fingerprint that shares the same 12-bit value.
                    if self.filter_members.remove(child) {
                        self.filter.delete(child);
                    }
                }
            }
        }

        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
    }

    /// Look up which bundles reference `child_cid`.
    ///
    /// Returns `Some(&HashSet<ContentId>)` if the child has at least one
    /// bundle reference, or `None` if it is not tracked.
    pub fn lookup(&self, child_cid: &ContentId) -> Option<&HashSet<ContentId>> {
        self.reverse.get(child_cid).filter(|s| !s.is_empty())
    }

    /// Returns `true` if `child_cid` is referenced by at least one bundle.
    pub fn is_referenced(&self, child_cid: &ContentId) -> bool {
        self.reverse.get(child_cid).is_some_and(|s| !s.is_empty())
    }

    /// Returns the number of structural mutations since the last broadcast reset.
    pub fn mutations_since_broadcast(&self) -> u32 {
        self.mutations_since_broadcast
    }

    /// Reset the mutation counter to zero (typically after broadcasting the filter).
    pub fn reset_mutation_counter(&mut self) {
        self.mutations_since_broadcast = 0;
    }

    /// Returns a reference to the underlying cuckoo filter.
    pub fn filter(&self) -> &CuckooFilter {
        &self.filter
    }

    /// Rebuild the cuckoo filter from scratch using the current reverse map.
    ///
    /// If the reverse map has more unique children than the current filter
    /// can hold (based on its bucket count), the filter is replaced with a
    /// larger one sized for the actual population. This prevents silent
    /// false negatives from `FilterFull` errors during re-insertion.
    pub fn rebuild_filter(&mut self) {
        let child_count = self.reverse.len() as u32;
        // Each bucket holds 4 fingerprints. Resize if population exceeds
        // total slot count (conservative — actual capacity depends on load
        // factor, but oversizing is cheap and prevents false negatives).
        let total_slots = self.filter.num_buckets().saturating_mul(4);
        if child_count > 0 && child_count > total_slots {
            self.filter = CuckooFilter::new(child_count);
        } else {
            self.filter.clear();
        }
        self.filter_members.clear();
        for child in self.reverse.keys() {
            if self.filter.insert(child).is_ok() {
                self.filter_members.insert(*child);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::ContentFlags;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(
            format!("flatpack-test-{i}").as_bytes(),
            ContentFlags::default(),
        )
        .unwrap()
    }

    #[test]
    fn admitted_bundle_creates_reverse_entries() {
        let mut idx = FlatpackIndex::new(100);
        let bundle = make_cid(0);
        let child_a = make_cid(1);
        let child_b = make_cid(2);

        idx.on_bundle_admitted(bundle, vec![child_a, child_b]);

        assert!(idx.is_referenced(&child_a));
        assert!(idx.is_referenced(&child_b));

        let set_a = idx
            .lookup(&child_a)
            .expect("child_a should be in reverse map");
        assert!(set_a.contains(&bundle));
        assert_eq!(set_a.len(), 1);

        let set_b = idx
            .lookup(&child_b)
            .expect("child_b should be in reverse map");
        assert!(set_b.contains(&bundle));
        assert_eq!(set_b.len(), 1);
    }

    #[test]
    fn evicted_bundle_removes_reverse_entries() {
        let mut idx = FlatpackIndex::new(100);
        let bundle = make_cid(0);
        let child_a = make_cid(1);
        let child_b = make_cid(2);

        idx.on_bundle_admitted(bundle, vec![child_a, child_b]);
        idx.on_bundle_evicted(&bundle);

        assert!(!idx.is_referenced(&child_a));
        assert!(!idx.is_referenced(&child_b));
        assert!(idx.lookup(&child_a).is_none());
        assert!(idx.lookup(&child_b).is_none());
    }

    #[test]
    fn shared_child_survives_partial_eviction() {
        let mut idx = FlatpackIndex::new(100);
        let bundle_1 = make_cid(10);
        let bundle_2 = make_cid(20);
        let shared_child = make_cid(99);
        let unique_child = make_cid(100);

        idx.on_bundle_admitted(bundle_1, vec![shared_child, unique_child]);
        idx.on_bundle_admitted(bundle_2, vec![shared_child]);

        // Evict bundle_1 — shared child still referenced by bundle_2.
        idx.on_bundle_evicted(&bundle_1);

        assert!(idx.is_referenced(&shared_child));
        let set = idx
            .lookup(&shared_child)
            .expect("shared child should survive");
        assert!(set.contains(&bundle_2));
        assert!(!set.contains(&bundle_1));

        // unique_child should be gone.
        assert!(!idx.is_referenced(&unique_child));
        assert!(idx.lookup(&unique_child).is_none());

        // Evict bundle_2 — shared child now unreferenced.
        idx.on_bundle_evicted(&bundle_2);
        assert!(!idx.is_referenced(&shared_child));
        assert!(idx.lookup(&shared_child).is_none());
    }

    #[test]
    fn cuckoo_filter_tracks_child_cids() {
        let mut idx = FlatpackIndex::new(100);
        let bundle = make_cid(0);
        let child = make_cid(1);

        idx.on_bundle_admitted(bundle, vec![child]);
        assert!(idx.filter().may_contain(&child));

        idx.on_bundle_evicted(&bundle);
        assert!(!idx.filter().may_contain(&child));
    }

    #[test]
    fn lookup_missing_returns_none() {
        let idx = FlatpackIndex::new(100);
        let missing = make_cid(42);
        assert!(idx.lookup(&missing).is_none());
        assert!(!idx.is_referenced(&missing));
    }

    #[test]
    fn evict_unknown_bundle_is_noop() {
        let mut idx = FlatpackIndex::new(100);
        let unknown = make_cid(999);
        // Should not panic.
        idx.on_bundle_evicted(&unknown);
        assert_eq!(idx.mutations_since_broadcast(), 0);
    }

    // ---- Integration tests: full cycle, GC safety, rebuild, mutation counter ----

    #[test]
    fn full_cycle_admit_broadcast_deserialize_lookup() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid(1000);
        let child_a = make_cid(1001);
        let child_b = make_cid(1002);

        // Admit bundle.
        idx.on_bundle_admitted(bundle, vec![child_a, child_b]);

        // Serialize cuckoo filter (simulates broadcast).
        let bytes = idx.filter().to_bytes();

        // Deserialize (simulates receiving peer).
        let remote_filter = CuckooFilter::from_bytes(&bytes).unwrap();

        // Remote filter should say "maybe" for both children.
        assert!(remote_filter.may_contain(&child_a));
        assert!(remote_filter.may_contain(&child_b));

        // Verify count matches.
        assert_eq!(remote_filter.count(), 2);
    }

    #[test]
    fn gc_safety_blocks_referenced_eviction() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid(2000);
        let blob = make_cid(2001);

        idx.on_bundle_admitted(bundle, vec![blob]);

        // Simulate GC check before evicting blob.
        assert!(
            idx.is_referenced(&blob),
            "referenced blob must not be evicted"
        );

        // After bundle eviction, blob becomes unreferenced.
        idx.on_bundle_evicted(&bundle);
        assert!(
            !idx.is_referenced(&blob),
            "unreferenced blob can be evicted"
        );
    }

    #[test]
    fn rebuild_filter_matches_current_state() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid(3000);
        let child_a = make_cid(3001);
        let child_b = make_cid(3002);
        let child_c = make_cid(3003);

        let bundle_2 = make_cid(3004);
        idx.on_bundle_admitted(bundle, vec![child_a, child_b]);
        idx.on_bundle_admitted(bundle_2, vec![child_c]);
        idx.on_bundle_evicted(&bundle_2);

        // Rebuild filter from scratch.
        idx.rebuild_filter();

        // Filter should match: child_a and child_b yes, child_c no.
        assert!(idx.filter().may_contain(&child_a));
        assert!(idx.filter().may_contain(&child_b));
        assert!(!idx.filter().may_contain(&child_c));
    }

    #[test]
    fn mutation_counter_increments() {
        let mut idx = FlatpackIndex::new(1000);
        assert_eq!(idx.mutations_since_broadcast(), 0);

        idx.on_bundle_admitted(make_cid(4000), vec![make_cid(4001)]);
        assert_eq!(idx.mutations_since_broadcast(), 1);

        idx.on_bundle_evicted(&make_cid(4000));
        assert_eq!(idx.mutations_since_broadcast(), 2);

        idx.reset_mutation_counter();
        assert_eq!(idx.mutations_since_broadcast(), 0);
    }

    #[test]
    fn readmit_bundle_with_different_children_cleans_old_entries() {
        let mut idx = FlatpackIndex::new(100);
        let bundle = make_cid(5000);
        let old_child = make_cid(5001);
        let new_child = make_cid(5002);

        // First admission with old_child.
        idx.on_bundle_admitted(bundle, vec![old_child]);
        assert!(idx.is_referenced(&old_child));

        // Re-admit same bundle with different children.
        idx.on_bundle_admitted(bundle, vec![new_child]);

        // Old child should be cleaned up.
        assert!(
            !idx.is_referenced(&old_child),
            "old child should be unreferenced after re-admission"
        );
        assert!(
            idx.lookup(&old_child).is_none(),
            "old child should have no reverse entries"
        );

        // New child should be present.
        assert!(idx.is_referenced(&new_child));
        let set = idx.lookup(&new_child).expect("new child should exist");
        assert!(set.contains(&bundle));
    }
}

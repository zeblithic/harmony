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
        for child in &child_cids {
            let set = self.reverse.entry(*child).or_default();
            if set.is_empty() {
                // New child — insert into cuckoo filter. Ignore FilterFull
                // errors; the filter is best-effort and rebuild_filter can
                // recover later.
                let _ = self.filter.insert(child);
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
                    self.filter.delete(child);
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
        self.reverse
            .get(child_cid)
            .is_some_and(|s| !s.is_empty())
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
    /// Clears the existing filter and re-inserts all child CIDs that have
    /// at least one bundle reference. Useful after many insertions and
    /// deletions to eliminate false positives from deleted entries.
    pub fn rebuild_filter(&mut self) {
        self.filter.clear();
        for child in self.reverse.keys() {
            let _ = self.filter.insert(child);
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

        let set_a = idx.lookup(&child_a).expect("child_a should be in reverse map");
        assert!(set_a.contains(&bundle));
        assert_eq!(set_a.len(), 1);

        let set_b = idx.lookup(&child_b).expect("child_b should be in reverse map");
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
        let set = idx.lookup(&shared_child).expect("shared child should survive");
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
}

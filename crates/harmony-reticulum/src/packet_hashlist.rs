#[cfg(not(feature = "std"))]
use hashbrown::HashSet;
#[cfg(feature = "std")]
use std::collections::HashSet;

/// Default maximum capacity for the packet hashlist.
pub const HASHLIST_DEFAULT_CAPACITY: usize = 1_000_000;

/// Bounded rotating set for duplicate packet detection.
///
/// Uses a two-generation design (matching Python `Transport.py`):
/// - `current`: the active set where new hashes are inserted
/// - `previous`: the prior generation, still checked for lookups
///
/// When `current` reaches half of `max_capacity`, it rotates: the current set
/// becomes `previous` (dropping the old previous), and a fresh empty set
/// becomes `current`. This guarantees bounded memory while ensuring that
/// any hash survives for at least `max_capacity / 2` insertions.
pub struct PacketHashlist {
    current: HashSet<[u8; 32]>,
    previous: HashSet<[u8; 32]>,
    max_capacity: usize,
}

impl PacketHashlist {
    /// Create a hashlist with the default capacity (1M entries).
    pub fn new() -> Self {
        Self::with_capacity(HASHLIST_DEFAULT_CAPACITY)
    }

    /// Create a hashlist with a custom maximum capacity.
    pub fn with_capacity(max_capacity: usize) -> Self {
        Self {
            current: HashSet::new(),
            previous: HashSet::new(),
            max_capacity,
        }
    }

    /// Check whether a packet hash has been seen (in either generation).
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.current.contains(hash) || self.previous.contains(hash)
    }

    /// Insert a packet hash. Returns `true` if the hash is new (not a duplicate),
    /// `false` if it was already present.
    ///
    /// Automatically rotates generations when the current set reaches half capacity.
    pub fn insert(&mut self, hash: [u8; 32]) -> bool {
        if self.contains(&hash) {
            return false;
        }

        self.current.insert(hash);

        if self.current.len() >= self.max_capacity / 2 {
            self.rotate();
        }

        true
    }

    /// Total number of tracked hashes across both generations.
    pub fn len(&self) -> usize {
        self.current.len() + self.previous.len()
    }

    /// Whether the hashlist is empty.
    pub fn is_empty(&self) -> bool {
        self.current.is_empty() && self.previous.is_empty()
    }

    /// Clear all tracked hashes.
    pub fn clear(&mut self) {
        self.current.clear();
        self.previous.clear();
    }

    /// Rotate: current becomes previous, previous is dropped.
    fn rotate(&mut self) {
        self.previous = core::mem::take(&mut self.current);
    }
}

impl Default for PacketHashlist {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a distinct [u8; 32] from a single distinguishing byte.
    fn h(b: u8) -> [u8; 32] {
        let mut a = [0u8; 32];
        a[0] = b;
        a
    }

    // ── Basic insert / contains ─────────────────────────────────────────

    #[test]
    fn insert_and_contains() {
        let mut hl = PacketHashlist::with_capacity(100);
        assert!(hl.is_empty());

        assert!(hl.insert(h(1)));
        assert!(hl.contains(&h(1)));
        assert!(!hl.contains(&h(2)));
        assert_eq!(hl.len(), 1);
    }

    #[test]
    fn duplicate_returns_false() {
        let mut hl = PacketHashlist::with_capacity(100);
        assert!(hl.insert(h(1)));
        assert!(!hl.insert(h(1))); // duplicate
        assert_eq!(hl.len(), 1);
    }

    // ── Rotation ────────────────────────────────────────────────────────

    #[test]
    fn rotation_triggers_at_half_capacity() {
        let mut hl = PacketHashlist::with_capacity(10);
        // Half capacity = 5. Insert 5 items to trigger rotation.
        for i in 0..5u8 {
            hl.insert(h(i));
        }
        // After rotation: current is empty, previous has 5
        assert_eq!(hl.current.len(), 0);
        assert_eq!(hl.previous.len(), 5);
        assert_eq!(hl.len(), 5);
    }

    #[test]
    fn previous_set_still_checked() {
        let mut hl = PacketHashlist::with_capacity(10);
        hl.insert(h(42));

        // Fill to trigger rotation
        for i in 1..5u8 {
            hl.insert(h(i));
        }

        // h(42) should still be found in previous
        assert!(hl.contains(&h(42)));
        // Inserting it again should return false (duplicate in previous)
        assert!(!hl.insert(h(42)));
    }

    // ── Double rotation evicts old entries ───────────────────────────────

    #[test]
    fn double_rotation_evicts() {
        let mut hl = PacketHashlist::with_capacity(10);

        // Insert item in first generation
        hl.insert(h(0xFF));

        // First rotation (insert 4 more to hit 5 total)
        for i in 1..5u8 {
            hl.insert(h(i));
        }
        assert!(hl.contains(&h(0xFF))); // still in previous

        // Second rotation (insert 5 more unique items)
        for i in 10..15u8 {
            hl.insert(h(i));
        }

        // 0xFF was in previous, which just got dropped
        assert!(!hl.contains(&h(0xFF)));
    }

    // ── Clear ───────────────────────────────────────────────────────────

    #[test]
    fn clear_empties_both_sets() {
        let mut hl = PacketHashlist::with_capacity(100);
        hl.insert(h(1));
        hl.insert(h(2));
        assert_eq!(hl.len(), 2);

        hl.clear();
        assert!(hl.is_empty());
        assert_eq!(hl.len(), 0);
        assert!(!hl.contains(&h(1)));
    }

    // ── Len tracks both sets ────────────────────────────────────────────

    #[test]
    fn len_spans_both_generations() {
        let mut hl = PacketHashlist::with_capacity(20);
        // Insert 5 items
        for i in 0..5u8 {
            hl.insert(h(i));
        }
        assert_eq!(hl.len(), 5);
        assert_eq!(hl.current.len(), 5);
        assert_eq!(hl.previous.len(), 0);

        // Trigger rotation (at 10 items = half of 20)
        for i in 5..10u8 {
            hl.insert(h(i));
        }
        // After rotation: all 10 moved to previous, current empty
        assert_eq!(hl.len(), 10);
        assert_eq!(hl.current.len(), 0);
        assert_eq!(hl.previous.len(), 10);

        // Insert one more into current
        hl.insert(h(20));
        assert_eq!(hl.len(), 11);
        assert_eq!(hl.current.len(), 1);
    }

    // ── Duplicate across generations ────────────────────────────────────

    #[test]
    fn duplicate_found_in_previous_not_moved_to_current() {
        let mut hl = PacketHashlist::with_capacity(10);
        hl.insert(h(99));

        // Trigger rotation
        for i in 1..5u8 {
            hl.insert(h(i));
        }

        // 99 is in previous. Attempting insert should fail without adding to current.
        assert!(!hl.insert(h(99)));
        assert!(!hl.current.contains(&h(99)));
        assert!(hl.previous.contains(&h(99)));
    }
}

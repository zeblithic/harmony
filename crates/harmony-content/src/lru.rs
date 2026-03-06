use alloc::vec::Vec;
use core::fmt;
use hashbrown::{HashMap, HashSet};

use crate::cid::ContentId;

/// A slab-allocated doubly-linked-list LRU cache.
///
/// Used as the underlying eviction structure for each segment
/// (Window, Probation, Protected) in the W-TinyLFU cache.
///
/// Entries are stored in a flat `Vec` (the "slab"), with freed slots
/// recycled via a free list. A `HashMap` provides O(1) CID-to-index
/// lookup. The doubly-linked list maintains recency order: head is
/// the most recently used, tail is the eviction candidate.
pub struct Lru {
    entries: Vec<LruEntry>,
    map: HashMap<ContentId, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    free: Vec<usize>,
    len: usize,
    capacity: usize,
}

struct LruEntry {
    cid: ContentId,
    prev: Option<usize>,
    next: Option<usize>,
    occupied: bool,
}

impl fmt::Debug for Lru {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Lru")
            .field("len", &self.len)
            .field("capacity", &self.capacity)
            .field("recycled_slots", &self.free.len())
            .finish_non_exhaustive()
    }
}

impl Lru {
    /// Create an empty LRU with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            map: HashMap::with_capacity(capacity),
            head: None,
            tail: None,
            free: Vec::new(),
            len: 0,
            capacity,
        }
    }

    /// Insert a CID at the head (most recently used position).
    ///
    /// If the CID is already present, it is promoted to head (touch).
    /// If the LRU is at capacity, the tail (least recently used) is
    /// evicted and its CID is returned.
    pub fn insert(&mut self, cid: ContentId) -> Option<ContentId> {
        // Zero-capacity LRU rejects all inserts immediately.
        if self.capacity == 0 {
            return Some(cid);
        }

        // If already present, just touch and return.
        if self.map.contains_key(&cid) {
            self.touch(&cid);
            return None;
        }

        // Evict tail if at capacity.
        let evicted = if self.len >= self.capacity {
            self.evict_tail()
        } else {
            None
        };

        let idx = self.alloc_slot(cid);
        self.map.insert(cid, idx);
        self.link_at_head(idx);
        self.len += 1;

        evicted
    }

    /// Move a CID to the head if present. Returns `true` if the CID was found.
    pub fn touch(&mut self, cid: &ContentId) -> bool {
        if let Some(&idx) = self.map.get(cid) {
            self.unlink(idx);
            self.link_at_head(idx);
            true
        } else {
            false
        }
    }

    /// Remove a CID from the LRU, recycling its slab slot.
    /// Returns `true` if the CID was found and removed.
    pub fn remove(&mut self, cid: &ContentId) -> bool {
        if let Some(idx) = self.map.remove(cid) {
            self.unlink(idx);
            self.entries[idx].occupied = false;
            self.free.push(idx);
            self.len -= 1;
            true
        } else {
            false
        }
    }

    /// Peek at the tail (least recently used) CID without removing it.
    pub fn peek_lru(&self) -> Option<ContentId> {
        self.tail.map(|idx| self.entries[idx].cid)
    }

    /// Find the least-recently-used CID that is NOT in `exclude`.
    ///
    /// Walks from tail toward head, skipping excluded entries. Returns
    /// `None` if every entry is in the exclusion set (or the LRU is empty).
    pub fn peek_lru_excluding(&self, exclude: &HashSet<ContentId>) -> Option<ContentId> {
        let mut idx = self.tail;
        while let Some(i) = idx {
            let cid = self.entries[i].cid;
            if !exclude.contains(&cid) {
                return Some(cid);
            }
            idx = self.entries[i].prev;
        }
        None
    }

    /// Returns `true` if the LRU contains the given CID.
    pub fn contains(&self, cid: &ContentId) -> bool {
        self.map.contains_key(cid)
    }

    /// Returns the number of entries currently in the LRU.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the LRU contains no entries.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the maximum capacity of this LRU.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Allocate a slab slot for a new entry, reusing a freed slot if available.
    fn alloc_slot(&mut self, cid: ContentId) -> usize {
        if let Some(idx) = self.free.pop() {
            self.entries[idx] = LruEntry {
                cid,
                prev: None,
                next: None,
                occupied: true,
            };
            idx
        } else {
            let idx = self.entries.len();
            self.entries.push(LruEntry {
                cid,
                prev: None,
                next: None,
                occupied: true,
            });
            idx
        }
    }

    /// Link a node at the head of the doubly-linked list.
    fn link_at_head(&mut self, idx: usize) {
        self.entries[idx].prev = None;
        self.entries[idx].next = self.head;

        if let Some(old_head) = self.head {
            self.entries[old_head].prev = Some(idx);
        }

        self.head = Some(idx);

        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    /// Unlink a node from its current position in the doubly-linked list.
    fn unlink(&mut self, idx: usize) {
        let prev = self.entries[idx].prev;
        let next = self.entries[idx].next;

        match prev {
            Some(p) => self.entries[p].next = next,
            None => self.head = next, // idx was head
        }

        match next {
            Some(n) => self.entries[n].prev = prev,
            None => self.tail = prev, // idx was tail
        }

        self.entries[idx].prev = None;
        self.entries[idx].next = None;
    }

    /// Evict the tail (least recently used) entry, returning its CID.
    fn evict_tail(&mut self) -> Option<ContentId> {
        let tail_idx = self.tail?;
        let cid = self.entries[tail_idx].cid;
        self.map.remove(&cid);
        self.unlink(tail_idx);
        self.entries[tail_idx].occupied = false;
        self.free.push(tail_idx);
        self.len -= 1;
        Some(cid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(format!("lru-test-{i}").as_bytes()).unwrap()
    }

    #[test]
    fn debug_impl_shows_summary() {
        let mut lru = Lru::new(4);
        lru.insert(make_cid(0));
        lru.insert(make_cid(1));
        let dbg = format!("{lru:?}");
        assert!(dbg.contains("len: 2"));
        assert!(dbg.contains("capacity: 4"));
        assert!(dbg.contains("recycled_slots: 0"));
        assert!(dbg.contains(".."), "should use finish_non_exhaustive()");
    }

    #[test]
    fn eviction_order() {
        let mut lru = Lru::new(3);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        let c3 = make_cid(3);
        assert_eq!(lru.insert(c0), None);
        assert_eq!(lru.insert(c1), None);
        assert_eq!(lru.insert(c2), None);
        assert_eq!(lru.insert(c3), Some(c0));
        assert!(!lru.contains(&c0));
        assert!(lru.contains(&c3));
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn touch_promotes_to_head() {
        let mut lru = Lru::new(3);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        let c3 = make_cid(3);
        lru.insert(c0);
        lru.insert(c1);
        lru.insert(c2);
        assert!(lru.touch(&c0));
        assert_eq!(lru.insert(c3), Some(c1));
    }

    #[test]
    fn remove_frees_slot() {
        let mut lru = Lru::new(2);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        lru.insert(c0);
        lru.insert(c1);
        assert_eq!(lru.len(), 2);
        assert!(lru.remove(&c0));
        assert_eq!(lru.len(), 1);
        assert_eq!(lru.insert(c2), None);
        assert_eq!(lru.len(), 2);
    }

    #[test]
    fn peek_lru_excluding_skips_excluded() {
        let mut lru = Lru::new(4);
        let c0 = make_cid(0); // oldest (tail)
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        let c3 = make_cid(3); // newest (head)
        lru.insert(c0);
        lru.insert(c1);
        lru.insert(c2);
        lru.insert(c3);

        // Exclude the two oldest entries — should return c2.
        let mut exclude = HashSet::new();
        exclude.insert(c0);
        exclude.insert(c1);
        assert_eq!(lru.peek_lru_excluding(&exclude), Some(c2));

        // Exclude all — should return None.
        exclude.insert(c2);
        exclude.insert(c3);
        assert_eq!(lru.peek_lru_excluding(&exclude), None);

        // Empty exclude — should return tail.
        let empty = HashSet::new();
        assert_eq!(lru.peek_lru_excluding(&empty), Some(c0));
    }

    #[test]
    fn zero_capacity_rejects_insert() {
        let mut lru = Lru::new(0);
        let c0 = make_cid(0);

        // Insert returns the CID back immediately (rejected).
        assert_eq!(lru.insert(c0), Some(c0));
        assert_eq!(lru.len(), 0);
        assert!(!lru.contains(&c0));
    }
}

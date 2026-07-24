//! Generic multi-device last-writer-wins reachability substrate: the
//! `(owner, node_id)` keying + LWW comparator + async fallback trait. App-specific
//! source arbitration / reconnect / liveness / refresh policy is layered on top
//! by the consumer (see harmony-client's `ReachabilityResolver`).

use async_trait::async_trait;
use std::collections::BTreeMap;

use crate::record::ReachabilityAnnouncePayload;

/// A record the kernel can order and index: it exposes a 32-byte node id and an
/// author-stamped announce time. Implemented by the reachability record and by
/// any app record that wants the LWW/multi-device machinery.
pub trait ReachabilityRecord {
    fn node_id(&self) -> [u8; 32];
    fn announced_at_ms(&self) -> u64;
}

impl ReachabilityRecord for ReachabilityAnnouncePayload {
    fn node_id(&self) -> [u8; 32] {
        self.iroh_node_id
    }
    fn announced_at_ms(&self) -> u64 {
        self.announced_at_ms
    }
}

/// Same-source LWW comparator: is `next` strictly newer than `prev`? Ordering:
/// primary by `clock` (`Ord`), ties by greater `announced_at_ms()`, remaining
/// ties by lexicographically greater `node_id()`. Full equality returns `false`
/// (a byte-identical replay is not a change). The caller supplies the clock's
/// ordering (e.g. an HLC compared as `(wall_ms, logical, device_id)`).
pub fn lww_newer<C, R>(prev_clock: &C, prev_rec: &R, next_clock: &C, next_rec: &R) -> bool
where
    C: Ord,
    R: ReachabilityRecord,
{
    use std::cmp::Ordering;
    match next_clock.cmp(prev_clock) {
        Ordering::Greater => true,
        Ordering::Less => false,
        Ordering::Equal => match next_rec.announced_at_ms().cmp(&prev_rec.announced_at_ms()) {
            Ordering::Greater => true,
            Ordering::Less => false,
            Ordering::Equal => next_rec.node_id() > prev_rec.node_id(),
        },
    }
}

/// A `BTreeMap` keyed by `(owner, node_id)` so a peer's multiple devices coexist
/// under one owner. Wraps the two non-trivial access patterns the reachability
/// resolver relies on: an owner-prefix range and a reverse-by-node-id scan.
#[derive(Debug, Clone)]
pub struct MultiDeviceMap<Owner: Ord + Copy, V> {
    inner: BTreeMap<(Owner, [u8; 32]), V>,
}

impl<Owner: Ord + Copy, V> Default for MultiDeviceMap<Owner, V> {
    fn default() -> Self {
        Self {
            inner: BTreeMap::new(),
        }
    }
}

impl<Owner: Ord + Copy, V> MultiDeviceMap<Owner, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, key: (Owner, [u8; 32]), value: V) -> Option<V> {
        self.inner.insert(key, value)
    }

    pub fn get(&self, key: &(Owner, [u8; 32])) -> Option<&V> {
        self.inner.get(key)
    }

    pub fn get_mut(&mut self, key: &(Owner, [u8; 32])) -> Option<&mut V> {
        self.inner.get_mut(key)
    }

    pub fn entry(
        &mut self,
        key: (Owner, [u8; 32]),
    ) -> std::collections::btree_map::Entry<'_, (Owner, [u8; 32]), V> {
        self.inner.entry(key)
    }

    pub fn remove(&mut self, key: &(Owner, [u8; 32])) -> Option<V> {
        self.inner.remove(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&(Owner, [u8; 32]), &V)> {
        self.inner.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// All `(key, value)` entries for one owner (every device), via the
    /// `(owner, 0..)..=(owner, 0xFF..)` prefix range.
    pub fn range_owner(&self, owner: &Owner) -> impl Iterator<Item = (&(Owner, [u8; 32]), &V)> {
        self.inner
            .range((*owner, [0u8; 32])..=(*owner, [0xFFu8; 32]))
    }

    /// Owner keys of this owner's devices (helper for bulk removal).
    pub fn owner_keys(&self, owner: &Owner) -> Vec<(Owner, [u8; 32])> {
        self.range_owner(owner).map(|(k, _)| *k).collect()
    }

    /// All entries whose node-id half matches, across owners (reverse lookup).
    pub fn find_by_node_id<'a>(
        &'a self,
        node_id: &'a [u8; 32],
    ) -> impl Iterator<Item = (&'a (Owner, [u8; 32]), &'a V)> {
        self.inner.iter().filter(move |((_, n), _)| n == node_id)
    }
}

/// Async fallback consulted on a cache miss (in the app: the pkarr resolver
/// adapter). Kept behind this trait so the kernel takes no transport dependency.
#[async_trait]
pub trait ReachabilityFallback<Owner>: Send + Sync {
    async fn resolve(&self, owner: &Owner) -> Vec<ReachabilityAnnouncePayload>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Rec {
        node: [u8; 32],
        at: u64,
    }
    impl ReachabilityRecord for Rec {
        fn node_id(&self) -> [u8; 32] {
            self.node
        }
        fn announced_at_ms(&self) -> u64 {
            self.at
        }
    }
    fn rec(n: u8, at: u64) -> Rec {
        Rec { node: [n; 32], at }
    }

    #[test]
    fn lww_newer_orders_by_clock_then_announced_then_node() {
        // Higher clock wins regardless of announce.
        assert!(lww_newer(&1u64, &rec(1, 9), &2u64, &rec(1, 0)));
        assert!(!lww_newer(&2u64, &rec(1, 0), &1u64, &rec(1, 9)));
        // Equal clock: higher announced_at wins.
        assert!(lww_newer(&5u64, &rec(1, 100), &5u64, &rec(1, 200)));
        // Equal clock + announced: greater node_id wins.
        assert!(lww_newer(&5u64, &rec(1, 100), &5u64, &rec(2, 100)));
        // Full equality is NOT newer (byte-identical replay is a no-op).
        assert!(!lww_newer(&5u64, &rec(1, 100), &5u64, &rec(1, 100)));
    }

    #[test]
    fn multi_device_map_range_and_reverse() {
        let mut m: MultiDeviceMap<[u8; 16], u32> = MultiDeviceMap::new();
        let owner_a = [0xAA; 16];
        let owner_b = [0xBB; 16];
        m.insert((owner_a, [1; 32]), 10);
        m.insert((owner_a, [2; 32]), 11);
        m.insert((owner_b, [1; 32]), 20);
        // Owner-prefix range returns only that owner's devices.
        let a: Vec<u32> = m.range_owner(&owner_a).map(|(_, v)| *v).collect();
        assert_eq!(a.len(), 2);
        assert!(a.contains(&10) && a.contains(&11));
        // Reverse-by-node finds across owners (both owners have node [1;32]).
        let hits: Vec<[u8; 16]> = m.find_by_node_id(&[1; 32]).map(|((o, _), _)| *o).collect();
        assert_eq!(hits.len(), 2);
    }
}

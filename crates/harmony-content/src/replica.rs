//! Replica storage: stores encrypted books on behalf of other peers.
//!
//! Keyed by (peer_identity, content_id). Separate from the node's own
//! content store — replicated data doesn't compete with local cache.

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use hashbrown::HashMap;

/// 16-byte peer identity hash (matches `harmony_identity::IdentityHash`).
/// Defined locally to avoid adding a dependency on `harmony-identity`.
type PeerHash = [u8; 16];

/// Error from replica storage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicaError {
    /// A single book exceeds the entire quota.
    ExceedsQuota { book_size: usize, quota: u64 },
}

/// Stores encrypted books on behalf of other peers.
///
/// Keyed by (peer_identity, content_id). Separate from the node's
/// own content store — replicated data doesn't compete with local cache.
pub trait ReplicaStore {
    /// Store encrypted data for a peer. If `usage + data.len() > quota`,
    /// evict oldest entries until room. If a single book exceeds the quota,
    /// return `ExceedsQuota`.
    fn store(
        &mut self,
        peer: PeerHash,
        cid: [u8; 32],
        data: Vec<u8>,
        quota: u64,
    ) -> Result<(), ReplicaError>;

    /// Retrieve a stored book for a peer.
    fn retrieve(&self, peer: &PeerHash, cid: &[u8; 32]) -> Option<Vec<u8>>;

    /// Total bytes stored for a peer.
    fn usage(&self, peer: &PeerHash) -> u64;

    /// Evict entries for a peer until usage <= target_bytes.
    fn evict_to(&mut self, peer: &PeerHash, target_bytes: u64);
}

/// Entry in the in-memory replica store.
struct ReplicaEntry {
    cid: [u8; 32],
    data: Vec<u8>,
}

/// Per-peer storage with insertion-order tracking for oldest-first eviction.
struct PeerReplica {
    /// Maps insertion_counter -> entry. BTreeMap gives ordered iteration.
    entries: BTreeMap<u64, ReplicaEntry>,
    /// Maps CID -> insertion_counter for O(1) dedup lookup.
    cid_to_seq: HashMap<[u8; 32], u64>,
    /// Monotonically increasing insertion counter.
    next_seq: u64,
    /// Cached total bytes for this peer.
    total_bytes: u64,
}

impl PeerReplica {
    fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            cid_to_seq: HashMap::new(),
            next_seq: 0,
            total_bytes: 0,
        }
    }

    fn contains(&self, cid: &[u8; 32]) -> bool {
        self.cid_to_seq.contains_key(cid)
    }

    fn insert(&mut self, cid: [u8; 32], data: Vec<u8>) {
        let size = data.len() as u64;
        let seq = self.next_seq;
        self.next_seq += 1;
        self.cid_to_seq.insert(cid, seq);
        self.entries.insert(seq, ReplicaEntry { cid, data });
        self.total_bytes += size;
    }

    fn evict_oldest(&mut self) -> Option<u64> {
        if let Some((&seq, _)) = self.entries.iter().next() {
            let entry = self.entries.remove(&seq).unwrap();
            let size = entry.data.len() as u64;
            self.cid_to_seq.remove(&entry.cid);
            self.total_bytes -= size;
            Some(size)
        } else {
            None
        }
    }

    fn get(&self, cid: &[u8; 32]) -> Option<&[u8]> {
        let seq = self.cid_to_seq.get(cid)?;
        self.entries.get(seq).map(|e| e.data.as_slice())
    }
}

/// In-memory implementation of [`ReplicaStore`].
pub struct MemoryReplicaStore {
    peers: HashMap<PeerHash, PeerReplica>,
}

impl MemoryReplicaStore {
    /// Create a new empty store.
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }
}

impl Default for MemoryReplicaStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ReplicaStore for MemoryReplicaStore {
    fn store(
        &mut self,
        peer: PeerHash,
        cid: [u8; 32],
        data: Vec<u8>,
        quota: u64,
    ) -> Result<(), ReplicaError> {
        let book_size = data.len();

        // Single book exceeds entire quota — reject outright.
        if book_size as u64 > quota {
            return Err(ReplicaError::ExceedsQuota { book_size, quota });
        }

        let peer_store = self.peers.entry(peer).or_insert_with(PeerReplica::new);

        // Idempotent: if we already have this CID, no-op.
        if peer_store.contains(&cid) {
            return Ok(());
        }

        // Evict oldest until there's room.
        while peer_store.total_bytes + book_size as u64 > quota {
            if peer_store.evict_oldest().is_none() {
                break;
            }
        }

        peer_store.insert(cid, data);
        Ok(())
    }

    fn retrieve(&self, peer: &PeerHash, cid: &[u8; 32]) -> Option<Vec<u8>> {
        self.peers.get(peer)?.get(cid).map(|s| s.to_vec())
    }

    fn usage(&self, peer: &PeerHash) -> u64 {
        self.peers.get(peer).map_or(0, |p| p.total_bytes)
    }

    fn evict_to(&mut self, peer: &PeerHash, target_bytes: u64) {
        if let Some(peer_store) = self.peers.get_mut(peer) {
            while peer_store.total_bytes > target_bytes {
                if peer_store.evict_oldest().is_none() {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn store_and_retrieve() {
        let mut store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        let cid = [0xBB; 32];
        let data = vec![1, 2, 3, 4, 5];
        store.store(peer, cid, data.clone(), 1024).unwrap();
        assert_eq!(store.retrieve(&peer, &cid), Some(data));
        assert_eq!(store.usage(&peer), 5);
    }

    #[test]
    fn quota_evicts_oldest() {
        let mut store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        // Store 3 books of 100 bytes each
        for i in 0..3u8 {
            let mut cid = [0u8; 32];
            cid[0] = i;
            store.store(peer, cid, vec![i; 100], 300).unwrap();
        }
        assert_eq!(store.usage(&peer), 300);

        // Store a 4th book — should evict the oldest (cid[0]=0)
        let mut cid4 = [0u8; 32];
        cid4[0] = 3;
        store.store(peer, cid4, vec![3; 100], 300).unwrap();
        assert_eq!(store.usage(&peer), 300);
        // Oldest (cid[0]=0) should be evicted
        let mut cid0 = [0u8; 32];
        cid0[0] = 0;
        assert_eq!(store.retrieve(&peer, &cid0), None);
        // Newest should still be there
        assert_eq!(store.retrieve(&peer, &cid4), Some(vec![3; 100]));
    }

    #[test]
    fn book_exceeding_quota_rejected() {
        let mut store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        let cid = [0xBB; 32];
        let result = store.store(peer, cid, vec![0; 1000], 500);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            ReplicaError::ExceedsQuota {
                book_size: 1000,
                quota: 500,
            }
        );
    }

    #[test]
    fn idempotent_store() {
        let mut store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        let cid = [0xBB; 32];
        store.store(peer, cid, vec![1; 100], 1024).unwrap();
        store.store(peer, cid, vec![1; 100], 1024).unwrap(); // same CID
        assert_eq!(store.usage(&peer), 100); // not doubled
    }

    #[test]
    fn evict_to_reduces_usage() {
        let mut store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        for i in 0..5u8 {
            let mut cid = [0u8; 32];
            cid[0] = i;
            store.store(peer, cid, vec![i; 100], 1000).unwrap();
        }
        assert_eq!(store.usage(&peer), 500);
        store.evict_to(&peer, 200);
        assert!(store.usage(&peer) <= 200);
    }

    #[test]
    fn retrieve_unknown_peer_returns_none() {
        let store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        let cid = [0xBB; 32];
        assert_eq!(store.retrieve(&peer, &cid), None);
    }

    #[test]
    fn usage_unknown_peer_returns_zero() {
        let store = MemoryReplicaStore::new();
        let peer = [0xAA; 16];
        assert_eq!(store.usage(&peer), 0);
    }
}

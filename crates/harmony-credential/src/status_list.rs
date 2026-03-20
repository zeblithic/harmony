use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use harmony_identity::IdentityRef;

/// Default status list capacity: 16,384 credential slots (2KB).
pub const DEFAULT_CAPACITY: u32 = 16_384;

/// A compact bitstring for tracking credential revocation status.
///
/// Each bit represents one credential slot: 0 = valid, 1 = revoked.
/// Managed by the credential issuer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusList {
    bits: Vec<u8>,
    capacity: u32,
}

impl StatusList {
    /// Create a new status list with the given capacity (number of credential slots).
    pub fn new(capacity: u32) -> Self {
        let byte_len = ((capacity + 7) / 8) as usize;
        Self {
            bits: vec![0u8; byte_len],
            capacity,
        }
    }

    /// Check whether the credential at `index` has been revoked.
    ///
    /// Returns `false` for out-of-bounds indices (treat unknown as valid).
    pub fn is_revoked(&self, index: u32) -> bool {
        if index >= self.capacity {
            return false;
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] & (1 << bit_idx) != 0
    }

    /// Revoke the credential at `index`. No-op if out of bounds.
    pub fn revoke(&mut self, index: u32) {
        if index >= self.capacity {
            return;
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    /// Total number of credential slots in this list.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

impl Default for StatusList {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// Check whether a credential has been revoked.
///
/// Returns `Some(true)` if revoked, `Some(false)` if valid,
/// `None` if the issuer's status list could not be resolved.
pub trait StatusListResolver {
    fn is_revoked(&self, issuer: &IdentityRef, index: u32) -> Option<bool>;
}

/// In-memory status list resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryStatusListResolver {
    lists: hashbrown::HashMap<harmony_identity::IdentityHash, StatusList>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryStatusListResolver {
    pub fn new() -> Self {
        Self {
            lists: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, issuer_hash: harmony_identity::IdentityHash, list: StatusList) {
        self.lists.insert(issuer_hash, list);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl StatusListResolver for MemoryStatusListResolver {
    fn is_revoked(&self, issuer: &IdentityRef, index: u32) -> Option<bool> {
        self.lists.get(&issuer.hash).map(|list| list.is_revoked(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_list_all_valid() {
        let list = StatusList::new(128);
        for i in 0..128 {
            assert!(!list.is_revoked(i));
        }
    }

    #[test]
    fn revoke_sets_bit() {
        let mut list = StatusList::new(128);
        list.revoke(42);
        assert!(list.is_revoked(42));
        assert!(!list.is_revoked(41));
        assert!(!list.is_revoked(43));
    }

    #[test]
    fn revoke_is_idempotent() {
        let mut list = StatusList::new(128);
        list.revoke(10);
        list.revoke(10);
        assert!(list.is_revoked(10));
    }

    #[test]
    fn capacity_matches_construction() {
        let list = StatusList::new(1000);
        assert_eq!(list.capacity(), 1000);
    }

    #[test]
    fn default_capacity() {
        let list = StatusList::default();
        assert_eq!(list.capacity(), DEFAULT_CAPACITY);
    }

    #[test]
    fn boundary_indices() {
        let mut list = StatusList::new(16);
        list.revoke(0);
        list.revoke(7);
        list.revoke(8);
        list.revoke(15);
        assert!(list.is_revoked(0));
        assert!(list.is_revoked(7));
        assert!(list.is_revoked(8));
        assert!(list.is_revoked(15));
        assert!(!list.is_revoked(1));
        assert!(!list.is_revoked(9));
    }

    #[test]
    fn out_of_bounds_is_valid() {
        let list = StatusList::new(16);
        assert!(!list.is_revoked(16));
        assert!(!list.is_revoked(9999));
    }

    #[test]
    fn revoke_out_of_bounds_is_noop() {
        let mut list = StatusList::new(16);
        list.revoke(16);
        list.revoke(9999);
        assert_eq!(list.capacity(), 16);
    }

    #[test]
    fn serde_round_trip() {
        let mut list = StatusList::new(256);
        list.revoke(0);
        list.revoke(100);
        list.revoke(255);

        let bytes = postcard::to_allocvec(&list).unwrap();
        let decoded: StatusList = postcard::from_bytes(&bytes).unwrap();
        assert!(decoded.is_revoked(0));
        assert!(decoded.is_revoked(100));
        assert!(decoded.is_revoked(255));
        assert!(!decoded.is_revoked(1));
        assert_eq!(decoded.capacity(), 256);
    }
}

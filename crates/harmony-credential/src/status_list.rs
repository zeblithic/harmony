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
        let byte_len = capacity.div_ceil(8) as usize;
        Self {
            bits: vec![0u8; byte_len],
            capacity,
        }
    }

    /// Check whether the credential at `index` has been revoked.
    ///
    /// Returns `None` for out-of-bounds indices so the verifier fails
    /// closed rather than treating un-representable indices as valid.
    pub fn is_revoked(&self, index: u32) -> Option<bool> {
        if index >= self.capacity {
            return None;
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        Some(self.bits[byte_idx] & (1 << bit_idx) != 0)
    }

    /// Revoke the credential at `index`.
    ///
    /// Returns `IndexOutOfBounds` if the index exceeds this list's capacity.
    pub fn revoke(&mut self, index: u32) -> Result<(), crate::error::CredentialError> {
        if index >= self.capacity {
            return Err(crate::error::CredentialError::IndexOutOfBounds);
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] |= 1 << bit_idx;
        Ok(())
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
        self.lists
            .get(&issuer.hash)
            .and_then(|list| list.is_revoked(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_list_all_valid() {
        let list = StatusList::new(128);
        for i in 0..128 {
            assert_eq!(list.is_revoked(i), Some(false));
        }
    }

    #[test]
    fn revoke_sets_bit() {
        let mut list = StatusList::new(128);
        list.revoke(42).unwrap();
        assert_eq!(list.is_revoked(42), Some(true));
        assert_eq!(list.is_revoked(41), Some(false));
        assert_eq!(list.is_revoked(43), Some(false));
    }

    #[test]
    fn revoke_is_idempotent() {
        let mut list = StatusList::new(128);
        list.revoke(10).unwrap();
        list.revoke(10).unwrap();
        assert_eq!(list.is_revoked(10), Some(true));
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
        list.revoke(0).unwrap();
        list.revoke(7).unwrap();
        list.revoke(8).unwrap();
        list.revoke(15).unwrap();
        assert_eq!(list.is_revoked(0), Some(true));
        assert_eq!(list.is_revoked(7), Some(true));
        assert_eq!(list.is_revoked(8), Some(true));
        assert_eq!(list.is_revoked(15), Some(true));
        assert_eq!(list.is_revoked(1), Some(false));
        assert_eq!(list.is_revoked(9), Some(false));
    }

    #[test]
    fn out_of_bounds_returns_none() {
        let list = StatusList::new(16);
        assert_eq!(list.is_revoked(16), None);
        assert_eq!(list.is_revoked(9999), None);
    }

    #[test]
    fn revoke_out_of_bounds_returns_error() {
        let mut list = StatusList::new(16);
        assert_eq!(
            list.revoke(16).unwrap_err(),
            crate::error::CredentialError::IndexOutOfBounds
        );
        assert_eq!(
            list.revoke(9999).unwrap_err(),
            crate::error::CredentialError::IndexOutOfBounds
        );
    }

    #[test]
    fn serde_round_trip() {
        let mut list = StatusList::new(256);
        list.revoke(0).unwrap();
        list.revoke(100).unwrap();
        list.revoke(255).unwrap();

        let bytes = postcard::to_allocvec(&list).unwrap();
        let decoded: StatusList = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.is_revoked(0), Some(true));
        assert_eq!(decoded.is_revoked(100), Some(true));
        assert_eq!(decoded.is_revoked(255), Some(true));
        assert_eq!(decoded.is_revoked(1), Some(false));
        assert_eq!(decoded.capacity(), 256);
    }
}

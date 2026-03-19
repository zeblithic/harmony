use serde::{Deserialize, Serialize};

use crate::crypto_suite::CryptoSuite;
use crate::identity::{Identity, IdentityHash};
use crate::pq_identity::PqIdentity;

/// A lightweight reference to an identity: address hash + crypto suite.
///
/// 17 bytes total. Use this when you need to know "which identity" and
/// "what kind" without carrying full public key material.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdentityRef {
    /// The 128-bit address hash: SHA256(pub_keys)[:16].
    pub hash: IdentityHash,
    /// The cryptographic suite backing this identity.
    pub suite: CryptoSuite,
}

impl IdentityRef {
    pub fn new(hash: IdentityHash, suite: CryptoSuite) -> Self {
        Self { hash, suite }
    }

    /// Whether this identity uses post-quantum cryptography.
    pub fn is_post_quantum(&self) -> bool {
        self.suite.is_post_quantum()
    }
}

impl From<&Identity> for IdentityRef {
    fn from(id: &Identity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::Ed25519,
        }
    }
}

impl From<&PqIdentity> for IdentityRef {
    fn from(id: &PqIdentity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::MlDsa65,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_classical_identity() {
        use rand::rngs::OsRng;
        let private = crate::identity::PrivateIdentity::generate(&mut OsRng);
        let id = private.public_identity();
        let id_ref = IdentityRef::from(id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::Ed25519);
        assert!(!id_ref.is_post_quantum());
    }

    #[test]
    fn from_pq_identity() {
        use rand::rngs::OsRng;
        let private = crate::pq_identity::PqPrivateIdentity::generate(&mut OsRng);
        let id = private.public_identity();
        let id_ref = IdentityRef::from(id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::MlDsa65);
        assert!(id_ref.is_post_quantum());
    }

    #[test]
    fn new_constructor() {
        let id_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);
        assert_eq!(id_ref.hash, [0xCC; 16]);
        assert!(id_ref.is_post_quantum());
    }

    #[test]
    fn equality_same_hash_same_suite() {
        let a = IdentityRef::new([0xDD; 16], CryptoSuite::Ed25519);
        let b = IdentityRef::new([0xDD; 16], CryptoSuite::Ed25519);
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_same_hash_different_suite() {
        let a = IdentityRef::new([0xEE; 16], CryptoSuite::Ed25519);
        let b = IdentityRef::new([0xEE; 16], CryptoSuite::MlDsa65);
        assert_ne!(a, b);
    }

    #[test]
    fn inequality_different_hash_same_suite() {
        let a = IdentityRef::new([0x11; 16], CryptoSuite::MlDsa65);
        let b = IdentityRef::new([0x22; 16], CryptoSuite::MlDsa65);
        assert_ne!(a, b);
    }

    #[test]
    fn serde_round_trip() {
        let id_ref = IdentityRef::new([0xFF; 16], CryptoSuite::MlDsa65);
        let bytes = postcard::to_allocvec(&id_ref).unwrap();
        let decoded: IdentityRef = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, id_ref);
    }
}

use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use serde::{Deserialize, Serialize};

/// A single claim: opaque type ID + opaque value.
///
/// The credential crate does not interpret claim semantics.
/// `type_id` is application-defined; `value` is an arbitrary payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub type_id: u16,
    pub value: Vec<u8>,
}

/// A claim prepared for selective disclosure.
///
/// Each claim is paired with a random salt. The signed credential
/// stores only the BLAKE3 digest; the holder retains the full
/// `SaltedClaim` for later selective revelation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltedClaim {
    pub claim: Claim,
    pub salt: [u8; 16],
}

impl SaltedClaim {
    /// Compute the BLAKE3 digest: `BLAKE3(salt || type_id.to_le_bytes() || value)`.
    ///
    /// Including `type_id` in the hash prevents type reinterpretation attacks.
    pub fn digest(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(16 + 2 + self.claim.value.len());
        buf.extend_from_slice(&self.salt);
        buf.extend_from_slice(&self.claim.type_id.to_le_bytes());
        buf.extend_from_slice(&self.claim.value);
        blake3_hash(&buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic() {
        let sc = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA, 0xBB],
            },
            salt: [0x11; 16],
        };
        assert_eq!(sc.digest(), sc.digest());
    }

    #[test]
    fn different_salts_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x22; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn different_type_ids_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 2,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn different_values_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xBB],
            },
            salt: [0x11; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn serde_round_trip() {
        let sc = SaltedClaim {
            claim: Claim {
                type_id: 42,
                value: alloc::vec![1, 2, 3],
            },
            salt: [0xFF; 16],
        };
        let bytes = postcard::to_allocvec(&sc).unwrap();
        let decoded: SaltedClaim = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.claim.type_id, 42);
        assert_eq!(decoded.claim.value, alloc::vec![1, 2, 3]);
        assert_eq!(decoded.salt, [0xFF; 16]);
        assert_eq!(decoded.digest(), sc.digest());
    }
}

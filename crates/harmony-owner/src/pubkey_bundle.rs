use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PubKeyBundle {
    pub classical: ClassicalKeys,
    pub post_quantum: Option<PqKeys>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassicalKeys {
    #[serde(with = "serde_bytes")]
    pub ed25519_verify: [u8; 32],
    #[serde(with = "serde_bytes")]
    pub x25519_pub: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PqKeys {
    #[serde(with = "serde_bytes")]
    pub ml_dsa_verify: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub ml_kem_pub: Vec<u8>,
}

impl PubKeyBundle {
    /// Derive the 128-bit IdentityHash from SIGNING-only material:
    /// `SHA256(ed25519_verify || optional ml_dsa_verify)[:16]`.
    ///
    /// Encryption keys (`x25519_pub`, `ml_kem_pub`) are deliberately NOT
    /// included so encryption-key rotation does not change identity. This
    /// mirrors Matrix's master signing key and Signal's identity key model.
    pub fn identity_hash(&self) -> [u8; 16] {
        // Build a stable, signing-only payload for hashing.
        #[derive(serde::Serialize)]
        struct SigningMaterial<'a> {
            #[serde(with = "serde_bytes")]
            ed25519_verify: &'a [u8; 32],
            #[serde(with = "serde_bytes")]
            ml_dsa_verify: Option<&'a [u8]>,
        }
        let payload = SigningMaterial {
            ed25519_verify: &self.classical.ed25519_verify,
            ml_dsa_verify: self.post_quantum.as_ref().map(|p| p.ml_dsa_verify.as_slice()),
        };
        let bytes = crate::cbor::to_canonical(&payload).expect("signing payload always encodes");
        let digest: [u8; 32] = harmony_crypto::hash::full_hash(&bytes);
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hash_is_deterministic() {
        let bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: [1u8; 32],
                x25519_pub: [2u8; 32],
            },
            post_quantum: None,
        };
        let h1 = bundle.identity_hash();
        let h2 = bundle.identity_hash();
        assert_eq!(h1, h2, "identity hash must be deterministic");
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn different_bundles_yield_different_hashes() {
        let a = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: [1u8; 32], x25519_pub: [2u8; 32] },
            post_quantum: None,
        };
        let b = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: [3u8; 32], x25519_pub: [2u8; 32] },
            post_quantum: None,
        };
        assert_ne!(a.identity_hash(), b.identity_hash());
    }

    #[test]
    fn x25519_rotation_does_not_change_identity_hash() {
        let mut bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: [1u8; 32], x25519_pub: [2u8; 32] },
            post_quantum: None,
        };
        let h1 = bundle.identity_hash();
        bundle.classical.x25519_pub = [99u8; 32];
        let h2 = bundle.identity_hash();
        assert_eq!(h1, h2, "rotating X25519 must not change the identity hash");
    }
}

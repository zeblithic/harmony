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
    /// Convenience constructor for a classical-only bundle from a verifying key.
    /// `x25519_pub` is derived from `ed25519_verify` via the RFC 7748 §5
    /// birational map ([`crate::x25519::ed25519_pub_to_x25519`]).
    ///
    /// Contract: `ed25519_verify` must be the caller's OWN valid Ed25519
    /// verifying key (callers are VouchingCert / LivenessCert `sign()`
    /// deriving the signer identity hash from the signing key) — passing a
    /// non-canonical or small-order point panics.
    pub fn classical_only(ed25519_verify: [u8; 32]) -> Self {
        Self {
            classical: ClassicalKeys {
                ed25519_verify,
                x25519_pub: crate::x25519::ed25519_pub_to_x25519(&ed25519_verify)
                    .expect("caller's own ed25519 verify key is a valid non-small-order point"),
            },
            post_quantum: None,
        }
    }

    /// Derive the 128-bit IdentityHash from SIGNING-only material:
    /// `SHA256(ed25519_verify || optional ml_dsa_verify)[:16]`.
    ///
    /// Encryption keys (`x25519_pub`, `ml_kem_pub`) are deliberately NOT
    /// included so encryption-key rotation does not change identity. This
    /// mirrors Matrix's master signing key and Signal's identity key model.
    pub fn identity_hash(&self) -> [u8; 16] {
        Self::signing_identity_hash(
            &self.classical.ed25519_verify,
            self.post_quantum
                .as_ref()
                .map(|p| p.ml_dsa_verify.as_slice()),
        )
    }

    /// Identity hash for a classical-only signer key WITHOUT constructing a
    /// bundle (and without deriving its X25519). Cert `verify()` paths use
    /// this for externally-supplied keys, where `classical_only`'s
    /// own-valid-key contract must not apply (it panics on non-canonical or
    /// small-order points — a crafted cert could DoS a verifier). Any 32-byte
    /// key hashes here; a weak key simply fails the signer-hash comparison or
    /// the signature check downstream. Because `identity_hash()` excludes
    /// encryption keys, this equals `classical_only(k).identity_hash()` for
    /// every valid `k`.
    pub fn classical_identity_hash(ed25519_verify: &[u8; 32]) -> [u8; 16] {
        Self::signing_identity_hash(ed25519_verify, None)
    }

    fn signing_identity_hash(ed25519_verify: &[u8; 32], ml_dsa_verify: Option<&[u8]>) -> [u8; 16] {
        // Build a stable, signing-only payload for hashing.
        #[derive(serde::Serialize)]
        struct SigningMaterial<'a> {
            #[serde(with = "serde_bytes")]
            ed25519_verify: &'a [u8; 32],
            #[serde(with = "serde_bytes")]
            ml_dsa_verify: Option<&'a [u8]>,
        }
        let payload = SigningMaterial {
            ed25519_verify,
            ml_dsa_verify,
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
            classical: ClassicalKeys {
                ed25519_verify: [1u8; 32],
                x25519_pub: [2u8; 32],
            },
            post_quantum: None,
        };
        let b = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: [3u8; 32],
                x25519_pub: [2u8; 32],
            },
            post_quantum: None,
        };
        assert_ne!(a.identity_hash(), b.identity_hash());
    }

    #[test]
    fn classical_identity_hash_matches_classical_only_bundle() {
        // The verify-path helper must agree with the bundle path forever:
        // certs store identity hashes computed at sign() time via
        // classical_only(); verify() recomputes via classical_identity_hash().
        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x09u8; 32]);
        let vk = sk.verifying_key().to_bytes();
        assert_eq!(
            PubKeyBundle::classical_identity_hash(&vk),
            PubKeyBundle::classical_only(vk).identity_hash(),
        );
    }

    #[test]
    fn classical_identity_hash_accepts_small_order_key_without_panic() {
        // External (attacker-supplied) keys flow through this helper in cert
        // verify paths; it must hash ANY 32 bytes rather than panic.
        let mut small_order = [0u8; 32];
        small_order[0] = 1; // compressed identity point
        let _ = PubKeyBundle::classical_identity_hash(&small_order); // must not panic
    }

    #[test]
    fn classical_only_populates_birational_x25519() {
        let sk = ed25519_dalek::SigningKey::from_bytes(&[0x07u8; 32]);
        let vk = sk.verifying_key().to_bytes();
        let bundle = PubKeyBundle::classical_only(vk);
        assert_eq!(
            bundle.classical.x25519_pub,
            crate::x25519::ed25519_pub_to_x25519(&vk).unwrap(),
            "classical_only must carry the birational X25519, not zeros"
        );
        assert_ne!(bundle.classical.x25519_pub, [0u8; 32]);
    }

    #[test]
    fn x25519_rotation_does_not_change_identity_hash() {
        let mut bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: [1u8; 32],
                x25519_pub: [2u8; 32],
            },
            post_quantum: None,
        };
        let h1 = bundle.identity_hash();
        bundle.classical.x25519_pub = [99u8; 32];
        let h2 = bundle.identity_hash();
        assert_eq!(h1, h2, "rotating X25519 must not change the identity hash");
    }
}

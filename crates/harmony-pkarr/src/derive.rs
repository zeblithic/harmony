//! HKDF-SHA256 derivation of ephemeral Ed25519 keys per case.
//!
//! Domain separation: each case uses a distinct salt
//! (`harmony.pkarr.v1.{invite|identity|community}`) so reusing the same
//! `ikm` across cases does NOT produce the same derived key.
//!
//! Spec Section 5.3.

use ed25519_dalek::SigningKey;
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;

/// Which Phase 2 case the derivation is for. Picks the salt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkarrCase {
    /// Case A: invite-redemption. `ikm` = `invite_token.sig` (64 bytes).
    Invite,
    /// Case B: opt-in identity-keyed. `ikm` = `owner_identity_pub` (64 bytes).
    ///
    /// # Security note — intentional public-key derivation
    ///
    /// For case B, `ikm = owner_identity_pub` is the **public** identity key.
    /// This is intentional: the pkarr slot is deterministically discoverable by
    /// anyone who holds the owner's public identity — that is the whole point of
    /// opt-in identity routing (the owner makes themselves discoverable).
    ///
    /// The derived ephemeral key signs BEP44 envelopes only.  It has **no
    /// relationship** to any private identity material: the owner's Ed25519
    /// private key is never used as, or mixed into, `ikm`.  The derivation
    /// does not leak any secret.
    Identity,
    /// Case C: in-community fallback. `ikm` = `EpochKey` (32 bytes).
    Community,
}

impl PkarrCase {
    pub fn salt(self) -> &'static [u8] {
        match self {
            Self::Invite => b"harmony.pkarr.v1.invite",
            // Case B derives from a *public* identity key by design — see the
            // `Identity` variant doc comment and spec §5.3 for the security
            // rationale (opt-in discoverability; authenticity via inner sig).
            Self::Identity => b"harmony.pkarr.v1.identity",
            Self::Community => b"harmony.pkarr.v1.community",
        }
    }
}

/// Derive an ephemeral Ed25519 signing key from per-case input.
///
/// Both publisher and resolver call this with identical `(case, ikm, info)`
/// inputs and obtain identical keys. The publisher signs BEP44 records
/// under the resulting key; the resolver derives the corresponding verifying
/// key from `signing.verifying_key()` and queries the DHT under it.
pub fn derive_ephemeral_key(case: PkarrCase, ikm: &[u8], info: &[u8]) -> SigningKey {
    let hkdf = Hkdf::<Sha256>::new(Some(case.salt()), ikm);
    let mut seed = [0u8; 32];
    hkdf.expand(info, &mut seed)
        .expect("HKDF-SHA256 always produces 32 bytes for our 32-byte output");
    let key = SigningKey::from_bytes(&seed);
    seed.zeroize();
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector — pins the entire keying scheme. If this test breaks,
    /// every published pkarr record under the v1 scheme becomes irretrievable
    /// without a v2 migration. DO NOT regenerate the expected hex without
    /// understanding the consequences.
    #[test]
    fn reference_vector_case_invite() {
        // ikm = 64 zero bytes (placeholder invite_token.sig)
        // info = epoch_id 12345 in big-endian
        let ikm = [0u8; 64];
        let info = 12345u64.to_be_bytes();
        let key = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        assert_eq!(
            vk_hex.len(),
            64,
            "Ed25519 verifying keys are 32 bytes = 64 hex chars"
        );
        // Pin: compute once, paste here. Regenerating breaks v1 records.
        let expected = "cf717966d2aaa0e7b3e75f759bbea3d46ae5e8d015d2d8b63fdf3a0d2eb1a7fd";
        assert_eq!(vk_hex, expected, "case-invite v1 keying must not drift");
    }

    #[test]
    fn reference_vector_case_identity() {
        let ikm = [0u8; 64];
        let info = 12345u64.to_be_bytes();
        let key = derive_ephemeral_key(PkarrCase::Identity, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        let expected = "855c400352869f52b19cdf1b784e72605e32c8635d9a758b516fbf8885e7e197";
        assert_eq!(vk_hex, expected, "case-identity v1 keying must not drift");
    }

    #[test]
    fn reference_vector_case_community() {
        let ikm = [0u8; 32];
        let mut info = [0u8; 72]; // 64 bytes member_pub + 8 bytes epoch_id
        info[64..].copy_from_slice(&12345u64.to_be_bytes());
        let key = derive_ephemeral_key(PkarrCase::Community, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        let expected = "623f89fc01f3538044560543b4fe4cbb43e848c2cd9031fb965523fccfa5477a";
        assert_eq!(vk_hex, expected, "case-community v1 keying must not drift");
    }

    #[test]
    fn different_cases_produce_different_keys() {
        let ikm = [1u8; 64];
        let info = [2u8; 8];
        let k1 = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info);
        let k2 = derive_ephemeral_key(PkarrCase::Identity, &ikm, &info);
        let k3 = derive_ephemeral_key(PkarrCase::Community, &ikm, &info);
        assert_ne!(k1.verifying_key(), k2.verifying_key());
        assert_ne!(k1.verifying_key(), k3.verifying_key());
        assert_ne!(k2.verifying_key(), k3.verifying_key());
    }

    #[test]
    fn different_info_produces_different_keys() {
        let ikm = [1u8; 64];
        let info_a = [2u8; 8];
        let info_b = [3u8; 8];
        let ka = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info_a);
        let kb = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info_b);
        assert_ne!(ka.verifying_key(), kb.verifying_key());
    }
}

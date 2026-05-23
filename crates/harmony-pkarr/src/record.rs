//! `PkarrRoutingRecord` — the opaque BEP44 inner payload.
//!
//! Spec Section 5.1. 2-char field keys per harmony convention.
//! Inner signature binds `(routing_blob, harmony_identity_pub,
//! announced_at_ms)` to the publisher's harmony Ed25519 identity key —
//! verified independently of the BEP44 outer (ephemeral) signature.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::PkarrError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PkarrRoutingRecord {
    /// Opaque routing blob. harmony-client encodes iroh routing here;
    /// harmony-pkarr treats as bytes.
    #[serde(rename = "rd", with = "serde_bytes")]
    pub routing_blob: alloc::vec::Vec<u8>,

    /// 64 bytes = X25519_pub(32) ‖ Ed25519_pub(32). The last 32 bytes are
    /// the Ed25519 verifying key used to verify `inner_sig`.
    #[serde(rename = "ip", with = "serde_bytes")]
    pub harmony_identity_pub: [u8; 64],

    /// Wall-clock publication time, ms since unix epoch.
    #[serde(rename = "at")]
    pub announced_at_ms: u64,

    /// Ed25519 sig over canonical-CBOR((routing_blob, harmony_identity_pub,
    /// announced_at_ms)) using the harmony identity Ed25519 key.
    #[serde(rename = "sg", with = "serde_bytes")]
    pub inner_sig: [u8; 64],
}

/// Maximum permitted skew between `announced_at_ms` and verifier's `now_ms`.
pub const SKEW_TOLERANCE_MS: u64 = 30 * 60 * 1000;

impl PkarrRoutingRecord {
    /// Build + inner-sign a record. `identity_signing_key` is the harmony
    /// identity Ed25519 key (NOT the ephemeral pkarr key — that one wraps
    /// this struct in the BEP44 envelope).
    pub fn sign_new(
        routing_blob: alloc::vec::Vec<u8>,
        harmony_identity_pub: [u8; 64],
        announced_at_ms: u64,
        identity_signing_key: &SigningKey,
    ) -> Result<Self, PkarrError> {
        let to_sign =
            canonical_signed_bytes(&routing_blob, &harmony_identity_pub, announced_at_ms)?;
        let sig = identity_signing_key.sign(&to_sign);
        Ok(Self {
            routing_blob,
            harmony_identity_pub,
            announced_at_ms,
            inner_sig: sig.to_bytes(),
        })
    }

    /// Verify the inner identity signature against the embedded
    /// `harmony_identity_pub`. RPK2 silent-drop on failure.
    pub fn verify_inner_sig(&self) -> Result<(), PkarrError> {
        let ed_pub_bytes: [u8; 32] = self.harmony_identity_pub[32..]
            .try_into()
            .map_err(|_| PkarrError::InvalidRecord)?;
        let vk = VerifyingKey::from_bytes(&ed_pub_bytes)
            .map_err(|_| PkarrError::InnerSignatureInvalid)?;
        let sig = Signature::from_bytes(&self.inner_sig);
        let to_verify = canonical_signed_bytes(
            &self.routing_blob,
            &self.harmony_identity_pub,
            self.announced_at_ms,
        )?;
        vk.verify(&to_verify, &sig)
            .map_err(|_| PkarrError::InnerSignatureInvalid)
    }

    /// Check `announced_at_ms` is within ±SKEW_TOLERANCE_MS of `now_ms`.
    /// RPK4 silent-drop on failure.
    pub fn verify_skew(&self, now_ms: u64) -> Result<(), PkarrError> {
        let diff = self.announced_at_ms.abs_diff(now_ms);
        if diff > SKEW_TOLERANCE_MS {
            Err(PkarrError::StaleOrSkewed)
        } else {
            Ok(())
        }
    }

    /// Check `harmony_identity_pub` matches an expected identity.
    /// RPK3 silent-drop on failure. Used by callers verifying records they
    /// queried by identity (case B) or by community-member context (case C).
    /// Case A skips this since the inner-sig already binds to `admin_identity_pub`
    /// from the invite payload.
    pub fn verify_identity_match(&self, expected: &[u8; 64]) -> Result<(), PkarrError> {
        if &self.harmony_identity_pub != expected {
            Err(PkarrError::IdentityMismatch)
        } else {
            Ok(())
        }
    }

    /// Canonical CBOR encoding of self, suitable for embedding in a BEP44
    /// envelope payload field.
    pub fn to_canonical_cbor(&self) -> Result<alloc::vec::Vec<u8>, PkarrError> {
        let mut out = alloc::vec::Vec::new();
        ciborium::into_writer(self, &mut out)
            .map_err(|_| PkarrError::SerializeError("PkarrRoutingRecord"))?;
        Ok(out)
    }

    pub fn from_canonical_cbor(bytes: &[u8]) -> Result<Self, PkarrError> {
        ciborium::from_reader(bytes).map_err(|_| PkarrError::DeserializeError("PkarrRoutingRecord"))
    }
}

fn canonical_signed_bytes(
    routing_blob: &[u8],
    harmony_identity_pub: &[u8; 64],
    announced_at_ms: u64,
) -> Result<alloc::vec::Vec<u8>, PkarrError> {
    // Tuple-as-array: deterministic CBOR ordering. ciborium encodes tuples
    // as CBOR arrays.
    let mut out = alloc::vec::Vec::new();
    ciborium::into_writer(
        &(
            serde_bytes::Bytes::new(routing_blob),
            serde_bytes::Bytes::new(harmony_identity_pub.as_ref()),
            announced_at_ms,
        ),
        &mut out,
    )
    .map_err(|_| PkarrError::SerializeError("canonical_signed_bytes"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_identity_pubkey(signing_key: &SigningKey) -> [u8; 64] {
        // Mimic harmony_identity::Identity::to_public_bytes() shape:
        // 32 zero bytes (X25519 placeholder) ‖ 32 bytes Ed25519 pub.
        let mut out = [0u8; 64];
        out[32..].copy_from_slice(&signing_key.verifying_key().to_bytes());
        out
    }

    #[test]
    fn round_trip_canonical_cbor() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(
            b"opaque-routing-blob".to_vec(),
            identity_pub,
            1_000_000,
            &sk,
        )
        .expect("sign");
        let cbor = rec.to_canonical_cbor().expect("encode");
        let decoded = PkarrRoutingRecord::from_canonical_cbor(&cbor).expect("decode");
        assert_eq!(rec, decoded);
    }

    #[test]
    fn verify_inner_sig_accepts_valid() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk)
            .expect("sign");
        assert!(rec.verify_inner_sig().is_ok());
    }

    #[test]
    fn verify_inner_sig_rejects_tampered_blob() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let mut rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk)
            .expect("sign");
        rec.routing_blob[0] ^= 1;
        assert_eq!(
            rec.verify_inner_sig(),
            Err(PkarrError::InnerSignatureInvalid)
        );
    }

    #[test]
    fn verify_inner_sig_rejects_tampered_at() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let mut rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk)
            .expect("sign");
        rec.announced_at_ms += 1;
        assert_eq!(
            rec.verify_inner_sig(),
            Err(PkarrError::InnerSignatureInvalid)
        );
    }

    #[test]
    fn verify_skew_accepts_within_window() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        // Use 10_000_000 so that subtracting SKEW_TOLERANCE_MS (1_800_000)
        // does not underflow the u64.
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 10_000_000, &sk)
            .expect("sign");
        assert!(rec.verify_skew(10_000_000 + SKEW_TOLERANCE_MS).is_ok());
        assert!(rec.verify_skew(10_000_000 - SKEW_TOLERANCE_MS).is_ok());
    }

    #[test]
    fn verify_skew_rejects_outside_window() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 10_000_000, &sk)
            .expect("sign");
        assert_eq!(
            rec.verify_skew(10_000_000 + SKEW_TOLERANCE_MS + 1),
            Err(PkarrError::StaleOrSkewed)
        );
        assert_eq!(
            rec.verify_skew(10_000_000 - SKEW_TOLERANCE_MS - 1),
            Err(PkarrError::StaleOrSkewed)
        );
    }

    #[test]
    fn verify_identity_match_rejects_substitution() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk)
            .expect("sign");
        let mut wrong = identity_pub;
        wrong[32] ^= 1;
        assert_eq!(
            rec.verify_identity_match(&wrong),
            Err(PkarrError::IdentityMismatch)
        );
        assert!(rec.verify_identity_match(&identity_pub).is_ok());
    }
}

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

    /// Signed validity horizon, ms since unix epoch. The record is honored
    /// while `now <= valid_until_ms`. Covered by `inner_sig`, so it cannot be
    /// forged or extended by a relay/attacker. No `serde(default)`: a record
    /// without this field (old wire format, or a stripped field) fails to
    /// decode and is dropped — the hard cutover that prevents downgrade.
    #[serde(rename = "vu")]
    pub valid_until_ms: u64,

    /// Ed25519 sig over canonical-CBOR((routing_blob, harmony_identity_pub,
    /// announced_at_ms, valid_until_ms)) using the harmony identity Ed25519 key.
    #[serde(rename = "sg", with = "serde_bytes")]
    pub inner_sig: [u8; 64],
}

/// Clock-skew allowance for the future-strict lower bound. A record whose
/// `announced_at_ms` is more than this far in the future is rejected as forged.
pub const FUTURE_TOLERANCE_MS: u64 = 5 * 60 * 1000;

impl PkarrRoutingRecord {
    /// Build + inner-sign a record. `identity_signing_key` is the harmony
    /// identity Ed25519 key (NOT the ephemeral pkarr key — that one wraps
    /// this struct in the BEP44 envelope).
    ///
    /// Returns `Err(PkarrError::IdentityMismatch)` if the verifying key
    /// derived from `identity_signing_key` does not match the Ed25519 portion
    /// of `harmony_identity_pub` (bytes [32..64]).  This guards against a
    /// mis-matched key pair being passed by the caller, which would produce a
    /// record that fails `verify_inner_sig` for every recipient.
    pub fn sign_new(
        routing_blob: alloc::vec::Vec<u8>,
        harmony_identity_pub: [u8; 64],
        announced_at_ms: u64,
        valid_until_ms: u64,
        identity_signing_key: &SigningKey,
    ) -> Result<Self, PkarrError> {
        // Key-match guard: verifying key from signing key must match the Ed25519
        // public key embedded in harmony_identity_pub[32..64].
        let derived_vk = identity_signing_key.verifying_key();
        let expected_ed_bytes: &[u8; 32] = harmony_identity_pub[32..]
            .try_into()
            .map_err(|_| PkarrError::IdentityMismatch)?;
        if derived_vk.as_bytes() != expected_ed_bytes {
            return Err(PkarrError::IdentityMismatch);
        }
        let to_sign = canonical_signed_bytes(
            &routing_blob,
            &harmony_identity_pub,
            announced_at_ms,
            valid_until_ms,
        )?;
        let sig = identity_signing_key.sign(&to_sign);
        Ok(Self {
            routing_blob,
            harmony_identity_pub,
            announced_at_ms,
            valid_until_ms,
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
            self.valid_until_ms,
        )?;
        vk.verify(&to_verify, &sig)
            .map_err(|_| PkarrError::InnerSignatureInvalid)
    }

    /// Freshness check (RPK4): reject a record that is forged-future
    /// (`announced_at_ms > now + FUTURE_TOLERANCE_MS`) or expired
    /// (`now > valid_until_ms`). The upper bound is the publisher's signed TTL,
    /// not a wall-clock window — so a valid record stays resolvable for its
    /// whole committed validity period regardless of republish cadence.
    pub fn verify_freshness(&self, now_ms: u64) -> Result<(), PkarrError> {
        if self.announced_at_ms > now_ms.saturating_add(FUTURE_TOLERANCE_MS) {
            return Err(PkarrError::StaleOrSkewed);
        }
        if now_ms > self.valid_until_ms {
            return Err(PkarrError::StaleOrSkewed);
        }
        Ok(())
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
    valid_until_ms: u64,
) -> Result<alloc::vec::Vec<u8>, PkarrError> {
    // Tuple-as-array: deterministic CBOR ordering. ciborium encodes tuples
    // as CBOR arrays.
    let mut out = alloc::vec::Vec::new();
    ciborium::into_writer(
        &(
            serde_bytes::Bytes::new(routing_blob),
            serde_bytes::Bytes::new(harmony_identity_pub.as_ref()),
            announced_at_ms,
            valid_until_ms,
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
            1_000_000 + 604_800_000,
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
        let rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000,
            &sk,
        )
        .expect("sign");
        assert!(rec.verify_inner_sig().is_ok());
    }

    #[test]
    fn verify_inner_sig_rejects_tampered_blob() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let mut rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000,
            &sk,
        )
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
        let mut rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000,
            &sk,
        )
        .expect("sign");
        rec.announced_at_ms += 1;
        assert_eq!(
            rec.verify_inner_sig(),
            Err(PkarrError::InnerSignatureInvalid)
        );
    }

    fn signed(at: u64, valid_until: u64) -> PkarrRoutingRecord {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, at, valid_until, &sk)
            .expect("sign")
    }

    #[test]
    fn valid_until_is_signed_and_tamper_proof() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000, // valid_until = announced + 7d
            &sk,
        )
        .expect("sign");
        assert_eq!(rec.valid_until_ms, 1_000_000 + 604_800_000);
        assert!(rec.verify_inner_sig().is_ok());

        // Tampering valid_until must break the inner signature.
        let mut tampered = rec.clone();
        tampered.valid_until_ms += 1;
        assert_eq!(
            tampered.verify_inner_sig(),
            Err(PkarrError::InnerSignatureInvalid)
        );
    }

    #[test]
    fn verify_freshness_accepts_old_record_within_ttl() {
        // Published 1 hour ago, TTL 7 days → resolvable (impossible under ±30min).
        let now = 10_000_000_000u64;
        let at = now - 60 * 60 * 1000;
        let rec = signed(at, at + 604_800_000);
        assert!(rec.verify_freshness(now).is_ok());
    }

    #[test]
    fn verify_freshness_rejects_expired() {
        let now = 10_000_000_000u64;
        let at = now - 8 * 24 * 60 * 60 * 1000; // 8 days ago
        let rec = signed(at, at + 604_800_000); // expired 1 day ago
        assert_eq!(rec.verify_freshness(now), Err(PkarrError::StaleOrSkewed));
    }

    #[test]
    fn verify_freshness_rejects_forged_future() {
        let now = 10_000_000_000u64;
        let at = now + FUTURE_TOLERANCE_MS + 1; // beyond skew allowance
        let rec = signed(at, at + 604_800_000);
        assert_eq!(rec.verify_freshness(now), Err(PkarrError::StaleOrSkewed));
    }

    #[test]
    fn verify_freshness_allows_small_future_skew() {
        let now = 10_000_000_000u64;
        let at = now + FUTURE_TOLERANCE_MS - 1; // within allowance
        let rec = signed(at, at + 604_800_000);
        assert!(rec.verify_freshness(now).is_ok());
    }

    #[test]
    fn sign_new_rejects_mismatched_key() {
        let sk = SigningKey::generate(&mut OsRng);
        let other_sk = SigningKey::generate(&mut OsRng);
        // identity_pub encodes `sk`'s verifying key, but we pass `other_sk` as signer.
        let identity_pub = fixture_identity_pubkey(&sk);
        let result = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000,
            &other_sk,
        );
        assert_eq!(result, Err(PkarrError::IdentityMismatch));
    }

    #[test]
    fn verify_identity_match_rejects_substitution() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(
            b"blob".to_vec(),
            identity_pub,
            1_000_000,
            1_000_000 + 604_800_000,
            &sk,
        )
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

//! Sans-I/O helpers for ZEB-113 online-recipient remote mail delivery.
//!
//! The SMTP handler calls into this module from the `DeliverToHarmony`
//! action when a recipient is not homed on the local gateway. The helpers
//! here perform no I/O: they seal plaintext via `HarmonyEnvelope` and
//! construct an `Identity` from an `AnnounceRecord`. The caller is
//! responsible for (a) resolving the `AnnounceRecord` via the
//! `RecipientResolver` trait and (b) publishing the sealed bytes via
//! `ZenohPublisher::publish_sealed_unicast`.

use harmony_discovery::AnnounceRecord;
use harmony_identity::{CryptoSuite, Identity, IdentityError, IdentityHash, PrivateIdentity};
use harmony_zenoh::envelope::{HarmonyEnvelope, MessageType};
use harmony_zenoh::ZenohError;
use rand_core::CryptoRngCore;

/// Errors surfaced by the remote delivery helpers.
#[derive(Debug)]
pub enum RemoteDeliveryError {
    /// An `AnnounceRecord` field had the wrong byte length for the
    /// classical identity format. `field` is either `"public_key"` or
    /// `"encryption_key"`; `got` is the observed length; `expected` is 32.
    InvalidKeyLength {
        field: &'static str,
        got: usize,
        expected: usize,
    },
    /// `Identity::from_public_keys` rejected the keys (e.g., 32 bytes but
    /// not a valid Ed25519 point).
    InvalidIdentity(IdentityError),
    /// The `AnnounceRecord` used a crypto suite this helper does not
    /// handle (e.g., post-quantum ML-DSA-65). Classical Ed25519/X25519
    /// is the only supported path in this PR.
    UnsupportedSuite(CryptoSuite),
    /// `HarmonyEnvelope::seal` failed (ECDH / AEAD / serialization).
    Seal(ZenohError),
}

impl std::fmt::Display for RemoteDeliveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidKeyLength { field, got, expected } => write!(
                f,
                "announce record field {field} has wrong length: got {got}, expected {expected}",
            ),
            Self::InvalidIdentity(e) => write!(f, "invalid identity keys: {e}"),
            Self::UnsupportedSuite(suite) => write!(f, "unsupported crypto suite: {suite:?}"),
            Self::Seal(e) => write!(f, "seal failed: {e}"),
        }
    }
}

impl std::error::Error for RemoteDeliveryError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidIdentity(e) => Some(e),
            Self::Seal(e) => Some(e),
            Self::InvalidKeyLength { .. } | Self::UnsupportedSuite(_) => None,
        }
    }
}

/// Seal `plaintext` addressed from `sender` (gateway's own identity) to
/// `recipient` (the remote user's `Identity`). Returns the fully-framed
/// envelope bytes ready for `session.put(...)`.
///
/// The sequence number is fresh-random per call — `HarmonyEnvelope::open`
/// does not enforce monotonicity on the receiver side, and no persistent
/// per-recipient counter exists at the gateway today. Random sequence
/// preserves the AAD entropy contract without new plumbing.
pub fn seal_for_recipient(
    rng: &mut impl CryptoRngCore,
    sender: &PrivateIdentity,
    recipient: &Identity,
    plaintext: &[u8],
) -> Result<Vec<u8>, RemoteDeliveryError> {
    let sequence = rng.next_u32();
    HarmonyEnvelope::seal(
        rng,
        MessageType::Put,
        sender,
        recipient,
        sequence,
        plaintext,
    )
    .map_err(RemoteDeliveryError::Seal)
}

/// Build a classical `Identity` from the `public_key` (Ed25519) and
/// `encryption_key` (X25519) fields of an `AnnounceRecord`.
///
/// Both byte vectors must be exactly 32 bytes for the classical identity
/// path. Post-quantum records (suite = `MlDsa65`) are outside this helper's
/// contract — they would need a separate PQ variant.
pub fn identity_from_announce_record(
    rec: &AnnounceRecord,
) -> Result<Identity, RemoteDeliveryError> {
    if rec.identity_ref.suite != CryptoSuite::Ed25519 {
        return Err(RemoteDeliveryError::UnsupportedSuite(rec.identity_ref.suite));
    }
    let x25519_pub: &[u8; 32] = rec.encryption_key.as_slice().try_into().map_err(|_| {
        RemoteDeliveryError::InvalidKeyLength {
            field: "encryption_key",
            got: rec.encryption_key.len(),
            expected: 32,
        }
    })?;
    let ed25519_pub: &[u8; 32] = rec.public_key.as_slice().try_into().map_err(|_| {
        RemoteDeliveryError::InvalidKeyLength {
            field: "public_key",
            got: rec.public_key.len(),
            expected: 32,
        }
    })?;
    Identity::from_public_keys(x25519_pub, ed25519_pub)
        .map_err(RemoteDeliveryError::InvalidIdentity)
}

/// Runtime contract for looking up a recipient's `Identity` by their
/// 16-byte address hash. Plumbed through the SMTP handler and consulted
/// once per non-local recipient in `DeliverToHarmony`.
///
/// `None` means "no announce record cached / not announced" — the
/// caller treats this as a warn-and-skip (per-recipient MX behavior);
/// the SMTP transaction still succeeds overall if other recipients
/// succeeded. A future `OfflineResolver` integration (ZEB-113 PR B) will
/// be wired behind this same trait, so the SMTP-handler side does not
/// need to change when store-and-forward lands.
///
/// # Contract
///
/// If `resolve(h)` returns `Some(id)`, then `id.address_hash == *h`.
/// The caller verifies this at runtime and skips (warn-log) on
/// mismatch to guard against buggy/stale resolvers that would cause
/// silent data loss by publishing to the correct topic but sealing to
/// the wrong public keys.
pub trait RecipientResolver: Send + Sync + 'static {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_discovery::AnnounceRecord;
    use harmony_identity::{CryptoSuite, IdentityRef, PrivateIdentity};
    use rand_core::OsRng;

    fn make_rec(
        public_key: Vec<u8>,
        encryption_key: Vec<u8>,
        suite: CryptoSuite,
    ) -> AnnounceRecord {
        AnnounceRecord {
            identity_ref: IdentityRef::new([0; 16], suite),
            public_key,
            encryption_key,
            routing_hints: vec![],
            published_at: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            signature: vec![],
        }
    }

    #[test]
    fn seal_for_recipient_round_trips_via_envelope_open() {
        let mut rng = OsRng;
        let gateway_priv = PrivateIdentity::generate(&mut rng);
        let recipient_priv = PrivateIdentity::generate(&mut rng);

        let plaintext = b"harmony-message-bytes-go-here";
        let sealed = seal_for_recipient(
            &mut rng,
            &gateway_priv,
            recipient_priv.public_identity(),
            plaintext,
        )
        .expect("seal should succeed");

        let opened = HarmonyEnvelope::open(
            &recipient_priv,
            gateway_priv.public_identity(),
            &sealed,
        )
        .expect("open should succeed");
        assert_eq!(opened.plaintext, plaintext);
        assert_eq!(
            opened.sender_address,
            gateway_priv.public_identity().address_hash
        );
    }

    #[test]
    fn identity_from_announce_record_extracts_classical_keys() {
        let mut rng = OsRng;
        let priv_id = PrivateIdentity::generate(&mut rng);
        let pub_id = priv_id.public_identity();

        // AnnounceRecord layout: public_key = Ed25519 (32B verifying),
        // encryption_key = X25519 (32B public).
        let pub_bytes = pub_id.to_public_bytes();
        let mut rec = make_rec(
            pub_bytes[32..].to_vec(),    // Ed25519
            pub_bytes[..32].to_vec(),    // X25519
            CryptoSuite::Ed25519,
        );
        rec.identity_ref = IdentityRef::new(pub_id.address_hash, CryptoSuite::Ed25519);

        let derived = identity_from_announce_record(&rec).expect("conversion should succeed");
        // address_hash = H(x25519 || ed25519), so a swap of public_key ↔
        // encryption_key on construction would fail this equality.
        assert_eq!(derived.address_hash, pub_id.address_hash);
    }

    #[test]
    fn identity_from_announce_record_rejects_wrong_length() {
        let rec = make_rec(
            vec![0u8; 10], // wrong length
            vec![0u8; 32],
            CryptoSuite::Ed25519,
        );
        let err = identity_from_announce_record(&rec).expect_err("should reject");
        assert!(
            matches!(
                err,
                RemoteDeliveryError::InvalidKeyLength { field: "public_key", got: 10, expected: 32 }
            ),
            "unexpected error: {err:?}"
        );
    }

    #[test]
    fn identity_from_announce_record_rejects_post_quantum_suite() {
        let rec = make_rec(vec![0u8; 32], vec![0u8; 32], CryptoSuite::MlDsa65);
        let err = identity_from_announce_record(&rec).expect_err("should reject PQ suite");
        assert!(
            matches!(err, RemoteDeliveryError::UnsupportedSuite(CryptoSuite::MlDsa65)),
            "unexpected error: {err:?}",
        );
    }
}

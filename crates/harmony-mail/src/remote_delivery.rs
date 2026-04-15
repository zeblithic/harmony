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
use harmony_identity::{Identity, IdentityError, IdentityHash, PrivateIdentity};
use harmony_zenoh::envelope::{HarmonyEnvelope, MessageType};
use harmony_zenoh::ZenohError;
use rand_core::CryptoRngCore;

/// Errors surfaced by the remote delivery helpers.
#[derive(Debug)]
pub enum RemoteDeliveryError {
    /// An `AnnounceRecord` carried a public or encryption key whose byte
    /// length did not match the classical (X25519 + Ed25519) identity
    /// format.
    InvalidAnnounceKey(IdentityError),
    /// `HarmonyEnvelope::seal` failed (ECDH / AEAD / serialization).
    Seal(ZenohError),
}

impl core::fmt::Display for RemoteDeliveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidAnnounceKey(e) => write!(f, "announce record keys invalid: {e:?}"),
            Self::Seal(e) => write!(f, "seal failed: {e:?}"),
        }
    }
}

impl std::error::Error for RemoteDeliveryError {}

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
    let x25519_pub: &[u8; 32] = rec.encryption_key.as_slice().try_into().map_err(|_| {
        RemoteDeliveryError::InvalidAnnounceKey(IdentityError::InvalidPublicKeyLength(
            rec.encryption_key.len(),
        ))
    })?;
    let ed25519_pub: &[u8; 32] = rec.public_key.as_slice().try_into().map_err(|_| {
        RemoteDeliveryError::InvalidAnnounceKey(IdentityError::InvalidPublicKeyLength(
            rec.public_key.len(),
        ))
    })?;
    Identity::from_public_keys(x25519_pub, ed25519_pub).map_err(RemoteDeliveryError::InvalidAnnounceKey)
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
pub trait RecipientResolver: Send + Sync {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seal_for_recipient_round_trips_via_envelope_open() {
        use harmony_identity::PrivateIdentity;
        use harmony_zenoh::envelope::HarmonyEnvelope;
        use rand_core::OsRng;

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
        use harmony_discovery::AnnounceRecord;
        use harmony_identity::{CryptoSuite, IdentityRef, PrivateIdentity};
        use rand_core::OsRng;

        let mut rng = OsRng;
        let priv_id = PrivateIdentity::generate(&mut rng);
        let pub_id = priv_id.public_identity();

        // AnnounceRecord layout: public_key = Ed25519 (32B verifying),
        // encryption_key = X25519 (32B public).
        let pub_bytes = pub_id.to_public_bytes();
        let rec = AnnounceRecord {
            identity_ref: IdentityRef::new(pub_id.address_hash, CryptoSuite::Ed25519),
            public_key: pub_bytes[32..].to_vec(),     // Ed25519
            encryption_key: pub_bytes[..32].to_vec(), // X25519
            routing_hints: vec![],
            published_at: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            signature: vec![],
        };

        let derived = identity_from_announce_record(&rec).expect("conversion should succeed");
        assert_eq!(derived.address_hash, pub_id.address_hash);
    }

    #[test]
    fn identity_from_announce_record_rejects_wrong_length() {
        use harmony_discovery::AnnounceRecord;
        use harmony_identity::{CryptoSuite, IdentityRef};

        let rec = AnnounceRecord {
            identity_ref: IdentityRef::new([0; 16], CryptoSuite::Ed25519),
            public_key: vec![0u8; 10], // wrong length
            encryption_key: vec![0u8; 32],
            routing_hints: vec![],
            published_at: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            signature: vec![],
        };
        let err = identity_from_announce_record(&rec).expect_err("should reject");
        assert!(
            matches!(err, RemoteDeliveryError::InvalidAnnounceKey(_)),
            "unexpected error: {err:?}"
        );
    }
}

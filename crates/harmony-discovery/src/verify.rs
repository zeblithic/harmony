use harmony_identity::CryptoSuite;

use crate::error::DiscoveryError;
use crate::record::AnnounceRecord;

/// Maximum allowed clock skew for announce timestamps (60 seconds).
const MAX_CLOCK_SKEW: u64 = 60;

/// Verify an announce record's signature and time bounds.
///
/// Checks:
/// 1. Record hasn't expired (`now < expires_at`)
/// 2. Record isn't future-stamped (`published_at <= now + MAX_CLOCK_SKEW`)
/// 3. Signature is valid for the included public key and crypto suite
///
/// # Security
///
/// **V1 limitation:** This function does NOT verify that `public_key`
/// hashes to `identity_ref.hash`. Any actor with a valid keypair can
/// craft an announce claiming an arbitrary identity address. Callers
/// MUST NOT rely solely on `verify_announce` to authenticate an
/// identity — the announce proves the signer controls the included
/// key, not that the key belongs to the claimed address. Address
/// re-derivation requires the encryption key (not carried in the
/// announce) and is deferred to a future version.
pub fn verify_announce(record: &AnnounceRecord, now: u64) -> Result<(), DiscoveryError> {
    // 1. Expiry check
    if now >= record.expires_at {
        return Err(DiscoveryError::Expired);
    }

    // 2. Reject future-stamped records (prevents cache poisoning via
    //    unreachable published_at values that block legitimate updates)
    if record.published_at > now.saturating_add(MAX_CLOCK_SKEW) {
        return Err(DiscoveryError::Expired);
    }

    let payload = record.signable_bytes();
    verify_signature(
        record.identity_ref.suite,
        &record.public_key,
        &payload,
        &record.signature,
    )
}

fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), DiscoveryError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| DiscoveryError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| DiscoveryError::SignatureInvalid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{AnnounceBuilder, RoutingHint};
    use harmony_identity::IdentityRef;
    use rand::rngs::OsRng;

    fn build_signed_announce() -> (harmony_identity::PrivateIdentity, AnnounceRecord) {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);

        let mut builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        (private, record)
    }

    #[test]
    fn valid_announce_passes() {
        let (_, record) = build_signed_announce();
        assert!(verify_announce(&record, 1500).is_ok());
    }

    #[test]
    fn expired_announce_rejected() {
        let (_, record) = build_signed_announce();
        assert_eq!(
            verify_announce(&record, 3000).unwrap_err(),
            DiscoveryError::Expired
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let (_, mut record) = build_signed_announce();
        record.signature = alloc::vec![0xFF; 64];
        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::SignatureInvalid
        );
    }

    #[test]
    fn wrong_public_key_rejected() {
        let (_, mut record) = build_signed_announce();
        let other = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        record.public_key = other.public_identity().verifying_key.to_bytes().to_vec();
        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::SignatureInvalid
        );
    }

    #[test]
    fn future_stamped_announce_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);

        // published_at far in the future
        let builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            u64::MAX - 100,
            u64::MAX - 1,
            [0x01; 16],
        );
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::Expired
        );
    }

    #[test]
    fn slight_clock_skew_allowed() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);

        // published_at is 30 seconds ahead (within MAX_CLOCK_SKEW of 60s)
        let mut builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            1030,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        assert!(verify_announce(&record, 1000).is_ok());
    }

    #[test]
    fn ml_dsa65_announce_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);

                let pq_id =
                    harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let identity_ref = IdentityRef::from(&pq_id);

                let mut builder = AnnounceBuilder::new(
                    identity_ref,
                    sign_pk.as_bytes(),
                    1000,
                    2000,
                    [0x05; 16],
                );
                builder.add_routing_hint(RoutingHint::Reticulum {
                    destination_hash: [0xDD; 16],
                });

                let payload = builder.signable_payload();
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(signature.as_bytes().to_vec());

                assert!(verify_announce(&record, 1500).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }
}

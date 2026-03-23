use alloc::vec::Vec;

use harmony_identity::CryptoSuite;

use crate::error::DiscoveryError;
use crate::record::AnnounceRecord;

/// Maximum allowed clock skew for announce timestamps.
/// All timestamps in this crate are Unix epoch seconds.
const MAX_CLOCK_SKEW: u64 = 60;

/// Verify an announce record's signature, time bounds, and pubkey→hash binding.
///
/// Checks:
/// 1. Record hasn't expired (`now < expires_at`)
/// 2. Record isn't future-stamped (`published_at <= now + MAX_CLOCK_SKEW`)
/// 3. Signature is valid for the included public key and crypto suite
/// 4. Public keys derive to the claimed identity address hash
pub fn verify_announce(record: &AnnounceRecord, now: u64) -> Result<(), DiscoveryError> {
    // 0. Structural validity — must have a positive validity window
    if record.published_at >= record.expires_at {
        return Err(DiscoveryError::InvalidRecord);
    }

    // 1. Expiry check
    if now >= record.expires_at {
        return Err(DiscoveryError::Expired);
    }

    // 2. Reject future-stamped records (prevents cache poisoning via
    //    unreachable published_at values that block legitimate updates)
    if record.published_at > now.saturating_add(MAX_CLOCK_SKEW) {
        return Err(DiscoveryError::FutureTimestamp);
    }

    let payload = record.signable_bytes();
    verify_signature(
        record.identity_ref.suite,
        &record.public_key,
        &payload,
        &record.signature,
    )?;

    // Verify that the included public keys derive to the claimed identity hash.
    // This prevents forged announces where an attacker substitutes their own keys
    // for a victim's identity address.
    let mut combined = Vec::with_capacity(
        record.encryption_key.len() + record.public_key.len(),
    );
    combined.extend_from_slice(&record.encryption_key);
    combined.extend_from_slice(&record.public_key);
    let derived_hash = harmony_crypto::hash::truncated_hash(&combined);
    if derived_hash != record.identity_ref.hash {
        return Err(DiscoveryError::AddressMismatch);
    }

    Ok(())
}

fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), DiscoveryError> {
    harmony_identity::verify_signature(suite, key_bytes, message, signature)
        .map_err(|_| DiscoveryError::SignatureInvalid)
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
            identity.encryption_key.as_bytes().to_vec(),
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
            identity.encryption_key.as_bytes().to_vec(),
            u64::MAX - 100,
            u64::MAX - 1,
            [0x01; 16],
        );
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::FutureTimestamp
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
            identity.encryption_key.as_bytes().to_vec(),
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

                let enc_pk_bytes = enc_pk.as_bytes();
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let identity_ref = IdentityRef::from(&pq_id);

                let mut builder = AnnounceBuilder::new(
                    identity_ref,
                    sign_pk.as_bytes(),
                    enc_pk_bytes,
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

    #[test]
    fn forged_announce_with_wrong_pubkey_rejected() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                // Generate two PQ identities
                let real_owner = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
                let attacker = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);

                let real_pub = real_owner.public_identity();
                let attacker_pub = attacker.public_identity();

                // Build announce with real_owner's identity_hash but attacker's keys
                let builder = AnnounceBuilder::new(
                    IdentityRef {
                        hash: real_pub.address_hash,
                        suite: harmony_identity::CryptoSuite::MlDsa65,
                    },
                    attacker_pub.verifying_key.as_bytes(),
                    attacker_pub.encryption_key.as_bytes(),
                    1000,
                    2000,
                    [0u8; 16],
                );
                let payload = builder.signable_payload();
                let sig = attacker.sign(&payload).unwrap();
                let record = builder.build(sig);

                // Signature is valid (attacker signed with their own key)
                // but address binding must fail: hash(attacker_keys) != real_owner.address_hash
                let result = verify_announce(&record, 1500);
                assert_eq!(result.unwrap_err(), DiscoveryError::AddressMismatch);
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    #[test]
    fn valid_announce_with_matching_pq_keys_passes() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let owner = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
                let pub_id = owner.public_identity();

                let builder = AnnounceBuilder::new(
                    IdentityRef::from(pub_id),
                    pub_id.verifying_key.as_bytes(),
                    pub_id.encryption_key.as_bytes(),
                    1000,
                    2000,
                    [0u8; 16],
                );
                let payload = builder.signable_payload();
                let sig = owner.sign(&payload).unwrap();
                let record = builder.build(sig);

                assert!(verify_announce(&record, 1500).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    #[test]
    fn forged_ed25519_announce_rejected() {
        let real_owner = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let attacker = harmony_identity::PrivateIdentity::generate(&mut OsRng);

        let real_pub = real_owner.public_identity();
        let attacker_pub = attacker.public_identity();

        // Build announce with real_owner's identity_hash but attacker's keys
        let builder = AnnounceBuilder::new(
            IdentityRef::from(real_pub),
            attacker_pub.verifying_key.to_bytes().to_vec(),
            attacker_pub.encryption_key.as_bytes().to_vec(),
            1000,
            2000,
            [0u8; 16],
        );
        let payload = builder.signable_payload();
        let sig = attacker.sign(&payload);
        let record = builder.build(sig.to_vec());

        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::AddressMismatch
        );
    }
}

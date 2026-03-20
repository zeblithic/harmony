use alloc::vec::Vec;
use harmony_identity::{CryptoSuite, IdentityRef};

use crate::endorsement::EndorsementRecord;
use crate::error::ProfileError;
use crate::profile::ProfileRecord;

/// Maximum allowed clock skew for record timestamps.
/// All timestamps in this crate are Unix epoch seconds.
const MAX_CLOCK_SKEW: u64 = 60;

/// Resolve an identity's verifying public key for profile and
/// endorsement verification.
///
/// Returns the raw verifying key bytes:
/// - Ed25519: 32-byte verifying key
/// - ML-DSA-65: 1952-byte signing public key
pub trait ProfileKeyResolver {
    fn resolve(&self, identity: &IdentityRef) -> Option<Vec<u8>>;
}

/// Verify a profile record's signature and time bounds.
pub fn verify_profile(
    record: &ProfileRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError> {
    check_time_bounds(record.published_at, record.expires_at, now)?;

    let key_bytes = keys
        .resolve(&record.identity_ref)
        .ok_or(ProfileError::KeyNotFound)?;

    let payload = record.signable_bytes();
    verify_signature(record.identity_ref.suite, &key_bytes, &payload, &record.signature)
}

/// Verify an endorsement record's signature and time bounds.
///
/// Rejects self-endorsements (`endorser == endorsee`) as structurally
/// invalid, even if the record was constructed bypassing the builder.
pub fn verify_endorsement(
    record: &EndorsementRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError> {
    if record.endorser.hash == record.endorsee.hash {
        return Err(ProfileError::InvalidRecord);
    }
    check_time_bounds(record.published_at, record.expires_at, now)?;

    let key_bytes = keys
        .resolve(&record.endorser)
        .ok_or(ProfileError::KeyNotFound)?;

    let payload = record.signable_bytes();
    verify_signature(record.endorser.suite, &key_bytes, &payload, &record.signature)
}

fn check_time_bounds(published_at: u64, expires_at: u64, now: u64) -> Result<(), ProfileError> {
    if published_at >= expires_at {
        return Err(ProfileError::InvalidRecord);
    }
    if now >= expires_at {
        return Err(ProfileError::Expired);
    }
    if published_at > now.saturating_add(MAX_CLOCK_SKEW) {
        return Err(ProfileError::FutureTimestamp);
    }
    Ok(())
}

/// Dispatch signature verification based on CryptoSuite.
// TODO(v2): MlDsa65Rotatable needs rotation-aware verification via
// KEL chain walk. See harmony-discovery verify.rs for the same note.
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), ProfileError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| ProfileError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| ProfileError::SignatureInvalid)
        }
    }
}

/// In-memory key resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryKeyResolver {
    keys: hashbrown::HashMap<harmony_identity::IdentityHash, Vec<u8>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl Default for MemoryKeyResolver {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryKeyResolver {
    pub fn new() -> Self {
        Self {
            keys: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, hash: harmony_identity::IdentityHash, key_bytes: Vec<u8>) {
        self.keys.insert(hash, key_bytes);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl ProfileKeyResolver for MemoryKeyResolver {
    fn resolve(&self, identity: &IdentityRef) -> Option<Vec<u8>> {
        self.keys.get(&identity.hash).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::endorsement::EndorsementBuilder;
    use crate::profile::ProfileBuilder;
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    fn setup_resolver(
        identity: &harmony_identity::Identity,
        identity_ref: &IdentityRef,
    ) -> MemoryKeyResolver {
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(identity_ref.hash, identity.verifying_key.to_bytes().to_vec());
        resolver
    }

    // ── ML-DSA-65 (primary path) ────────────────────────────────

    #[test]
    fn ml_dsa65_profile_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let id_ref = IdentityRef::from(&pq_id);

                let mut builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
                builder.display_name(alloc::string::String::from("PQ Alice"));

                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(sig.as_bytes().to_vec());

                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(id_ref.hash, sign_pk.as_bytes());

                assert!(verify_profile(&record, 1500, &resolver).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    #[test]
    fn ml_dsa65_endorsement_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let endorser_ref = IdentityRef::from(&pq_id);
                let endorsee_ref = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);

                let builder = EndorsementBuilder::new(
                    endorser_ref, endorsee_ref, 42, 1000, 2000, [0x01; 16],
                );
                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(sig.as_bytes().to_vec());

                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(endorser_ref.hash, sign_pk.as_bytes());

                assert!(verify_endorsement(&record, 1500, &resolver).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    // ── Ed25519 (compat path) ───────────────────────────────────

    #[test]
    fn ed25519_profile_verification() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let mut builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        builder.display_name(alloc::string::String::from("Ed Alice"));

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert!(verify_profile(&record, 1500, &resolver).is_ok());
    }

    #[test]
    fn ed25519_endorsement_verification() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let endorser_ref = IdentityRef::from(identity);
        let endorsee_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        let builder = EndorsementBuilder::new(
            endorser_ref, endorsee_ref, 42, 1000, 2000, [0x01; 16],
        );
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &endorser_ref);
        assert!(verify_endorsement(&record, 1500, &resolver).is_ok());
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn expired_profile_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 3000, &resolver).unwrap_err(),
            ProfileError::Expired
        );
    }

    #[test]
    fn future_stamped_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, u64::MAX - 100, u64::MAX - 1, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::FutureTimestamp
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let record = builder.build(alloc::vec![0xFF; 64]);

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::SignatureInvalid
        );
    }

    #[test]
    fn unknown_key_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let id_ref = IdentityRef::from(private.public_identity());

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = MemoryKeyResolver::new(); // empty
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::KeyNotFound
        );
    }

    #[test]
    fn invalid_record_rejected() {
        // Bypass builder to construct a record with published_at >= expires_at
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let record = ProfileRecord {
            identity_ref: id_ref,
            display_name: None,
            status_text: None,
            avatar_cid: None,
            published_at: 2000,
            expires_at: 1000, // invalid: expires before published
            nonce: [0x01; 16],
            signature: alloc::vec![0xDE, 0xAD],
        };

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::InvalidRecord
        );
    }

    #[test]
    fn self_endorsement_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        // Bypass builder to construct self-endorsement directly
        let record = EndorsementRecord {
            endorser: id_ref,
            endorsee: id_ref, // same identity
            type_id: 42,
            reason: None,
            published_at: 1000,
            expires_at: 2000,
            nonce: [0x01; 16],
            signature: alloc::vec![0xDE, 0xAD],
        };

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_endorsement(&record, 1500, &resolver).unwrap_err(),
            ProfileError::InvalidRecord
        );
    }
}

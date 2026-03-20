use alloc::vec::Vec;
use harmony_identity::{CryptoSuite, IdentityRef};

use crate::credential::Credential;
use crate::disclosure::Presentation;
use crate::error::CredentialError;
use crate::status_list::StatusListResolver;

/// Resolve an issuer's verifying public key for credential verification.
///
/// Returns the raw public key bytes used for signature verification:
/// - Ed25519: 32-byte verifying key
/// - ML-DSA-65: 1952-byte signing public key
///
/// The `issued_at` parameter enables KEL-backed resolvers to return
/// the key that was active at credential issuance time.
/// For non-rotatable identities, `issued_at` can be ignored.
pub trait CredentialKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, issued_at: u64) -> Option<Vec<u8>>;
}

/// Verify a credential's time bounds, signature, and revocation status.
pub fn verify_credential(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError> {
    // 1. Time bounds
    if now < credential.not_before {
        return Err(CredentialError::NotYetValid);
    }
    if now >= credential.expires_at {
        return Err(CredentialError::Expired);
    }

    // 2. Issuer resolution
    let key_bytes = keys
        .resolve(&credential.issuer, credential.issued_at)
        .ok_or(CredentialError::IssuerNotFound)?;

    // 3. Signature verification
    let payload = credential.signable_bytes();
    verify_signature(
        credential.issuer.suite,
        &key_bytes,
        &payload,
        &credential.signature,
    )?;

    // 4. Revocation check
    if let Some(idx) = credential.status_list_index {
        match status_lists.is_revoked(&credential.issuer, idx) {
            Some(true) => return Err(CredentialError::Revoked),
            Some(false) => {}
            None => return Err(CredentialError::StatusListNotFound),
        }
    }

    Ok(())
}

/// Verify a presentation: credential verification + subject identity +
/// disclosure integrity.
///
/// When `expected_subject` is `Some`, the credential's subject must match
/// the given identity. This prevents replay of intercepted presentations
/// by a different party.
pub fn verify_presentation(
    presentation: &Presentation,
    now: u64,
    expected_subject: Option<&IdentityRef>,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError> {
    verify_credential(&presentation.credential, now, keys, status_lists)?;
    if let Some(subject) = expected_subject {
        if &presentation.credential.subject != subject {
            return Err(CredentialError::SubjectMismatch);
        }
    }
    presentation.verify_disclosures()?;
    Ok(())
}

/// Dispatch signature verification based on CryptoSuite.
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), CredentialError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| CredentialError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| CredentialError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| CredentialError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| CredentialError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| CredentialError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| CredentialError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| CredentialError::SignatureInvalid)
        }
    }
}

/// In-memory key resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryKeyResolver {
    keys: hashbrown::HashMap<harmony_identity::IdentityHash, Vec<u8>>,
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
impl CredentialKeyResolver for MemoryKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, _issued_at: u64) -> Option<Vec<u8>> {
        self.keys.get(&issuer.hash).cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::SaltedClaim;
    use crate::credential::CredentialBuilder;
    use crate::disclosure::Presentation;
    use crate::status_list::{MemoryStatusListResolver, StatusList};
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    /// Build a signed credential using a real Ed25519 keypair.
    fn build_signed_credential() -> (
        harmony_identity::PrivateIdentity,
        Credential,
        Vec<SaltedClaim>,
        IdentityRef,
    ) {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        let mut builder = CredentialBuilder::new(issuer_ref, subject_ref, 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, claims) = builder.build(signature.to_vec());

        (private, cred, claims, issuer_ref)
    }

    fn setup_resolver(
        identity: &harmony_identity::Identity,
        issuer_ref: &IdentityRef,
    ) -> MemoryKeyResolver {
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.verifying_key.to_bytes().to_vec());
        resolver
    }

    fn empty_status() -> MemoryStatusListResolver {
        MemoryStatusListResolver::new()
    }

    // ---- verify_credential tests ----

    #[test]
    fn valid_credential_passes() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());
    }

    #[test]
    fn expired_credential_rejected() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 3000, &resolver, &status).unwrap_err(),
            CredentialError::Expired
        );
    }

    #[test]
    fn not_yet_valid_rejected() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 500, &resolver, &status).unwrap_err(),
            CredentialError::NotYetValid
        );
    }

    #[test]
    fn unknown_issuer_rejected() {
        let (_, cred, _, _) = build_signed_credential();
        let resolver = MemoryKeyResolver::new(); // empty
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::IssuerNotFound
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let (private, mut cred, _, issuer_ref) = build_signed_credential();
        cred.signature = alloc::vec![0xFF; 64]; // wrong signature
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::SignatureInvalid
        );
    }

    #[test]
    fn revoked_credential_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);

        let mut builder = CredentialBuilder::new(
            issuer_ref,
            IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519),
            1000,
            2000,
            [0x02; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.status_list_index(5);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &issuer_ref);
        let mut status = MemoryStatusListResolver::new();
        let mut list = StatusList::new(128);
        list.revoke(5).unwrap();
        status.insert(issuer_ref.hash, list);

        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::Revoked
        );
    }

    #[test]
    fn missing_status_list_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);

        let mut builder = CredentialBuilder::new(
            issuer_ref,
            IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519),
            1000,
            2000,
            [0x03; 16],
        );
        builder.status_list_index(5);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &issuer_ref);
        let status = empty_status(); // no list registered

        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::StatusListNotFound
        );
    }

    #[test]
    fn non_revocable_skips_status_check() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        assert!(cred.status_list_index.is_none());
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status(); // no lists at all -- fine for non-revocable
        assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());
    }

    // ---- verify_presentation tests ----

    #[test]
    fn valid_presentation_passes() {
        let (private, cred, claims, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert!(verify_presentation(&presentation, 1500, None, &resolver, &status).is_ok());
    }

    #[test]
    fn valid_presentation_with_expected_subject() {
        let (private, cred, claims, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let subject = cred.subject;
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert!(
            verify_presentation(&presentation, 1500, Some(&subject), &resolver, &status).is_ok()
        );
    }

    #[test]
    fn wrong_subject_rejected() {
        let (private, cred, claims, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let wrong_subject = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert_eq!(
            verify_presentation(&presentation, 1500, Some(&wrong_subject), &resolver, &status)
                .unwrap_err(),
            CredentialError::SubjectMismatch
        );
    }

    #[test]
    fn presentation_with_tampered_claim_rejected() {
        let (private, cred, mut claims, issuer_ref) = build_signed_credential();
        claims[0].claim.value = alloc::vec![0xFF]; // tamper
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };
        assert_eq!(
            verify_presentation(&presentation, 1500, None, &resolver, &status).unwrap_err(),
            CredentialError::DisclosureMismatch
        );
    }

    // ---- ML-DSA-65 issuer test ----

    #[test]
    fn ml_dsa65_issuer_verification() {
        // ML-DSA-65 keygen is expensive; run in a thread with larger stack
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);

                // Derive PqIdentity to get address hash
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let issuer_ref = IdentityRef::from(&pq_id);
                let subject_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);

                let mut builder =
                    CredentialBuilder::new(issuer_ref, subject_ref, 1000, 2000, [0x05; 16]);
                builder.add_claim(1, alloc::vec![0xDD], [0x44; 16]);

                let payload = builder.signable_payload();
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let (cred, claims) = builder.build(signature.as_bytes().to_vec());

                // Resolver stores ML-DSA-65 public key bytes (signing key only)
                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(issuer_ref.hash, sign_pk.as_bytes());
                let status = MemoryStatusListResolver::new();

                assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());

                // Full presentation flow
                let presentation = Presentation {
                    credential: cred,
                    disclosed_claims: claims,
                };
                assert!(verify_presentation(&presentation, 1500, None, &resolver, &status).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    // ---- Integration tests ----

    #[test]
    fn full_flow_issue_present_verify() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        // 1. Issue
        let mut builder = CredentialBuilder::new(issuer_ref, subject_ref, 1000, 5000, [0x10; 16]);
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]); // age claim
        builder.add_claim(2, alloc::vec![0x02], [0xA2; 16]); // org claim
        builder.add_claim(3, alloc::vec![0x03], [0xA3; 16]); // license claim
        builder.status_list_index(0);

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, claims) = builder.build(signature.to_vec());

        // 2. Holder selects subset to disclose (only age claim)
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };

        // 3. Verify
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.verifying_key.to_bytes().to_vec());
        let mut status = MemoryStatusListResolver::new();
        status.insert(issuer_ref.hash, StatusList::new(128));

        assert!(verify_presentation(&presentation, 2000, None, &resolver, &status).is_ok());
    }

    #[test]
    fn revocation_after_issuance() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        // Issue credential at index 7
        let mut builder = CredentialBuilder::new(issuer_ref, subject_ref, 1000, 5000, [0x20; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0xB1; 16]);
        builder.status_list_index(7);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.verifying_key.to_bytes().to_vec());

        // Valid before revocation
        let mut status = MemoryStatusListResolver::new();
        status.insert(issuer_ref.hash, StatusList::new(128));
        assert!(verify_credential(&cred, 2000, &resolver, &status).is_ok());

        // Revoke and verify again
        let mut list = StatusList::new(128);
        list.revoke(7).unwrap();
        let mut status2 = MemoryStatusListResolver::new();
        status2.insert(issuer_ref.hash, list);

        assert_eq!(
            verify_credential(&cred, 2000, &resolver, &status2).unwrap_err(),
            CredentialError::Revoked
        );
    }
}

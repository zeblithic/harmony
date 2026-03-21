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
///
/// **Important:** For `CryptoSuite::MlDsa65Rotatable` issuers, implementations
/// MUST resolve the historical key that was active at `issued_at` by walking
/// the issuer's Key Event Log. A static key store is only correct if the
/// issuer has never rotated keys — after rotation, credentials signed with
/// the previous key will return `SignatureInvalid` unless the resolver is
/// KEL-aware.
pub trait CredentialKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, issued_at: u64) -> Option<Vec<u8>>;
}

/// Resolve a credential by its content hash for chain verification.
///
/// Used by `verify_chain` to walk delegation chains. Implementations
/// may fetch from a local cache, database, or network.
pub trait CredentialResolver {
    fn resolve(&self, content_hash: &[u8; 32]) -> Option<Credential>;
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

/// Maximum delegation chain depth. Chains deeper than this are rejected.
pub const MAX_CHAIN_DEPTH: usize = 8;

/// Verify a credential and its full delegation chain.
///
/// Recursively verifies every ancestor credential -- time bounds,
/// signature, and revocation status. If any ancestor is expired,
/// revoked, or invalid, the entire chain fails.
///
/// For root credentials (no `proof` field), this behaves identically
/// to `verify_credential`.
pub fn verify_chain(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
    credentials: &impl CredentialResolver,
) -> Result<(), CredentialError> {
    let mut seen = [[0u8; 32]; MAX_CHAIN_DEPTH + 1];
    seen[0] = credential.content_hash();
    let mut seen_len = 1;

    verify_chain_inner(
        credential,
        now,
        keys,
        status_lists,
        credentials,
        &mut seen,
        &mut seen_len,
        0,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_chain_inner(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
    credentials: &impl CredentialResolver,
    seen: &mut [[u8; 32]; MAX_CHAIN_DEPTH + 1],
    seen_len: &mut usize,
    depth: usize,
) -> Result<(), CredentialError> {
    verify_credential(credential, now, keys, status_lists)?;

    if let Some(parent_hash) = credential.proof {
        if seen[..*seen_len].contains(&parent_hash) {
            return Err(CredentialError::ChainLoop);
        }

        if depth + 1 > MAX_CHAIN_DEPTH {
            return Err(CredentialError::ChainTooDeep);
        }

        seen[*seen_len] = parent_hash;
        *seen_len += 1;

        let parent = credentials
            .resolve(&parent_hash)
            .ok_or(CredentialError::ProofNotFound)?;

        // Verify the resolved credential matches the referenced hash.
        // A buggy or compromised resolver could return a different credential.
        if parent.content_hash() != parent_hash {
            return Err(CredentialError::ProofNotFound);
        }

        if parent.subject != credential.issuer {
            return Err(CredentialError::ChainBroken);
        }

        verify_chain_inner(
            &parent,
            now,
            keys,
            status_lists,
            credentials,
            seen,
            seen_len,
            depth + 1,
        )?;
    }

    Ok(())
}

/// Delegate signature verification to `harmony_identity`, mapping errors to [`CredentialError::SignatureInvalid`].
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), CredentialError> {
    harmony_identity::verify_signature(suite, key_bytes, message, signature)
        .map_err(|_| CredentialError::SignatureInvalid)
}

/// In-memory key resolver for testing.
///
/// Ignores `issued_at` -- only correct for non-rotatable identities or
/// rotatable identities that have never rotated. Not suitable for
/// production use with `MlDsa65Rotatable` issuers that have undergone
/// key rotation.
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

/// In-memory credential resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryCredentialResolver {
    credentials: hashbrown::HashMap<[u8; 32], Credential>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryCredentialResolver {
    pub fn new() -> Self {
        Self {
            credentials: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, credential: Credential) {
        let hash = credential.content_hash();
        self.credentials.insert(hash, credential);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl CredentialResolver for MemoryCredentialResolver {
    fn resolve(&self, content_hash: &[u8; 32]) -> Option<Credential> {
        self.credentials.get(content_hash).cloned()
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

    use super::MemoryCredentialResolver;

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

    fn build_delegation(
        issuer_priv: &harmony_identity::PrivateIdentity,
        subject_ref: IdentityRef,
        proof: Option<[u8; 32]>,
    ) -> Credential {
        let issuer_ref = IdentityRef::from(issuer_priv.public_identity());
        let mut builder = CredentialBuilder::new(issuer_ref, subject_ref, 1000, 5000, [0x30; 16]);
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
        if let Some(hash) = proof {
            builder.proof(hash);
        }
        let payload = builder.signable_payload();
        let signature = issuer_priv.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());
        cred
    }

    fn setup_chain_resolvers(
        identities: &[(&harmony_identity::PrivateIdentity, &IdentityRef)],
        credentials: &[&Credential],
    ) -> (
        MemoryKeyResolver,
        MemoryStatusListResolver,
        MemoryCredentialResolver,
    ) {
        let mut keys = MemoryKeyResolver::new();
        for (priv_id, id_ref) in identities {
            keys.insert(
                id_ref.hash,
                priv_id.public_identity().verifying_key.to_bytes().to_vec(),
            );
        }
        let status = MemoryStatusListResolver::new();
        let mut creds = MemoryCredentialResolver::new();
        for cred in credentials {
            creds.insert((*cred).clone());
        }
        (keys, status, creds)
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
            verify_presentation(
                &presentation,
                1500,
                Some(&wrong_subject),
                &resolver,
                &status
            )
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

    // ---- verify_chain tests ----

    #[test]
    fn chain_root_credential_no_proof() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(private.public_identity(), &issuer_ref);
        let status = empty_status();
        let creds = MemoryCredentialResolver::new();
        assert!(verify_chain(&cred, 1500, &resolver, &status, &creds).is_ok());
    }

    #[test]
    fn chain_valid_two_level() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let delegation = build_delegation(&gov_priv, uni_ref, None);
        let delegation_hash = delegation.content_hash();
        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let (keys, status, creds) = setup_chain_resolvers(
            &[(&gov_priv, &gov_ref), (&uni_priv, &uni_ref)],
            &[&delegation],
        );
        assert!(verify_chain(&degree, 1500, &keys, &status, &creds).is_ok());
    }

    #[test]
    fn chain_valid_three_level() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let dept_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let dept_ref = IdentityRef::from(dept_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let gov_cred = build_delegation(&gov_priv, uni_ref, None);
        let uni_cred = build_delegation(&uni_priv, dept_ref, Some(gov_cred.content_hash()));
        let dept_cred = build_delegation(&dept_priv, student_ref, Some(uni_cred.content_hash()));

        let (keys, status, creds) = setup_chain_resolvers(
            &[
                (&gov_priv, &gov_ref),
                (&uni_priv, &uni_ref),
                (&dept_priv, &dept_ref),
            ],
            &[&gov_cred, &uni_cred],
        );
        assert!(verify_chain(&dept_cred, 1500, &keys, &status, &creds).is_ok());
    }

    #[test]
    fn chain_revoked_ancestor_fails() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let mut builder = CredentialBuilder::new(gov_ref, uni_ref, 1000, 5000, [0x30; 16]);
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
        builder.status_list_index(0);
        let payload = builder.signable_payload();
        let delegation = {
            let sig = gov_priv.sign(&payload);
            let (cred, _) = builder.build(sig.to_vec());
            cred
        };
        let delegation_hash = delegation.content_hash();
        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let mut keys = MemoryKeyResolver::new();
        keys.insert(
            gov_ref.hash,
            gov_priv.public_identity().verifying_key.to_bytes().to_vec(),
        );
        keys.insert(
            uni_ref.hash,
            uni_priv.public_identity().verifying_key.to_bytes().to_vec(),
        );

        let mut status = MemoryStatusListResolver::new();
        let mut list = StatusList::new(128);
        list.revoke(0).unwrap();
        status.insert(gov_ref.hash, list);

        let mut creds = MemoryCredentialResolver::new();
        creds.insert(delegation);

        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::Revoked
        );
    }

    #[test]
    fn chain_broken_subject_issuer_mismatch() {
        let gov_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let gov_ref = IdentityRef::from(gov_priv.public_identity());
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let wrong_ref = IdentityRef::new([0xEE; 16], CryptoSuite::Ed25519);
        let delegation = build_delegation(&gov_priv, wrong_ref, None);
        let delegation_hash = delegation.content_hash();
        let degree = build_delegation(&uni_priv, student_ref, Some(delegation_hash));

        let (keys, status, creds) = setup_chain_resolvers(
            &[(&gov_priv, &gov_ref), (&uni_priv, &uni_ref)],
            &[&delegation],
        );
        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ChainBroken
        );
    }

    #[test]
    fn chain_proof_not_found() {
        let uni_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let uni_ref = IdentityRef::from(uni_priv.public_identity());
        let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);

        let degree = build_delegation(&uni_priv, student_ref, Some([0xFF; 32]));

        let mut keys = MemoryKeyResolver::new();
        keys.insert(
            uni_ref.hash,
            uni_priv.public_identity().verifying_key.to_bytes().to_vec(),
        );
        let status = empty_status();
        let creds = MemoryCredentialResolver::new();

        assert_eq!(
            verify_chain(&degree, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ProofNotFound
        );
    }

    #[test]
    fn chain_too_deep() {
        let privates: Vec<harmony_identity::PrivateIdentity> = (0..10)
            .map(|_| harmony_identity::PrivateIdentity::generate(&mut OsRng))
            .collect();
        let refs: Vec<IdentityRef> = privates
            .iter()
            .map(|p| IdentityRef::from(p.public_identity()))
            .collect();

        let mut keys = MemoryKeyResolver::new();
        for (i, p) in privates.iter().enumerate() {
            keys.insert(
                refs[i].hash,
                p.public_identity().verifying_key.to_bytes().to_vec(),
            );
        }

        let mut creds = MemoryCredentialResolver::new();
        let mut prev_hash: Option<[u8; 32]> = None;

        for i in 0..9 {
            let cred = build_delegation(&privates[i], refs[i + 1], prev_hash);
            prev_hash = Some(cred.content_hash());
            creds.insert(cred);
        }

        let leaf_subject = IdentityRef::new([0xCC; 16], CryptoSuite::Ed25519);
        let leaf = build_delegation(&privates[9], leaf_subject, prev_hash);

        let status = empty_status();
        assert_eq!(
            verify_chain(&leaf, 1500, &keys, &status, &creds).unwrap_err(),
            CredentialError::ChainTooDeep
        );
    }

    #[test]
    fn chain_dishonest_resolver_caught_by_hash_check() {
        // Content hashes make honest cycles impossible (chicken-and-egg).
        // A dishonest resolver returning the wrong credential is caught
        // by the content_hash verification — returns ProofNotFound.
        // The ChainLoop code path is defense-in-depth, unreachable with
        // honest resolvers, but kept for safety.
        let alice_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let bob_priv = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let alice_ref = IdentityRef::from(alice_priv.public_identity());
        let bob_ref = IdentityRef::from(bob_priv.public_identity());

        let fake_parent_hash = [0xAA; 32];
        let leaf = build_delegation(&alice_priv, bob_ref, Some(fake_parent_hash));
        let leaf_hash = leaf.content_hash();

        // Parent whose proof points back to the leaf
        let parent = build_delegation(&bob_priv, alice_ref, Some(leaf_hash));

        // Dishonest resolver: returns parent regardless of requested hash
        struct DishonestResolver(Credential);
        impl CredentialResolver for DishonestResolver {
            fn resolve(&self, _content_hash: &[u8; 32]) -> Option<Credential> {
                Some(self.0.clone())
            }
        }

        let mut keys = MemoryKeyResolver::new();
        keys.insert(
            alice_ref.hash,
            alice_priv
                .public_identity()
                .verifying_key
                .to_bytes()
                .to_vec(),
        );
        keys.insert(
            bob_ref.hash,
            bob_priv.public_identity().verifying_key.to_bytes().to_vec(),
        );
        let status = empty_status();

        // Caught by content hash verification — resolver returned wrong credential
        let result = verify_chain(&leaf, 1500, &keys, &status, &DishonestResolver(parent));
        assert_eq!(result.unwrap_err(), CredentialError::ProofNotFound);
    }

    #[test]
    fn chain_valid_two_level_ml_dsa65() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (gov_sign_pk, gov_sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (gov_enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let gov_pq =
                    harmony_identity::PqIdentity::from_public_keys(gov_enc_pk, gov_sign_pk.clone());
                let gov_ref = IdentityRef::from(&gov_pq);

                let (uni_sign_pk, uni_sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (uni_enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let uni_pq =
                    harmony_identity::PqIdentity::from_public_keys(uni_enc_pk, uni_sign_pk.clone());
                let uni_ref = IdentityRef::from(&uni_pq);

                let student_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);

                let mut builder = CredentialBuilder::new(gov_ref, uni_ref, 1000, 5000, [0x30; 16]);
                builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]);
                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&gov_sign_sk, &payload).unwrap();
                let (delegation, _) = builder.build(sig.as_bytes().to_vec());
                let delegation_hash = delegation.content_hash();

                let mut builder2 =
                    CredentialBuilder::new(uni_ref, student_ref, 1000, 5000, [0x31; 16]);
                builder2.add_claim(1, alloc::vec![0x02], [0xA2; 16]);
                builder2.proof(delegation_hash);
                let payload2 = builder2.signable_payload();
                let sig2 = harmony_crypto::ml_dsa::sign(&uni_sign_sk, &payload2).unwrap();
                let (degree, _) = builder2.build(sig2.as_bytes().to_vec());

                let mut keys = MemoryKeyResolver::new();
                keys.insert(gov_ref.hash, gov_sign_pk.as_bytes());
                keys.insert(uni_ref.hash, uni_sign_pk.as_bytes());
                let status = MemoryStatusListResolver::new();
                let mut creds = MemoryCredentialResolver::new();
                creds.insert(delegation);

                assert!(verify_chain(&degree, 1500, &keys, &status, &creds).is_ok());
            })
            .expect("spawn")
            .join()
            .expect("join");
    }
}

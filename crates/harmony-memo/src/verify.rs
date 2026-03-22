use alloc::vec::Vec;
use harmony_credential::{verify_credential, CredentialKeyResolver, StatusListResolver};
use harmony_credential::claim::{Claim, SaltedClaim};
use harmony_identity::IdentityRef;

use crate::{Memo, MemoError, MEMO_CLAIM_TYPE};

/// A no-op status list resolver that reports no revocations.
///
/// Memos are self-attested and do not use revocation lists, so this
/// resolver always returns `Some(false)` (not revoked).
pub struct NoOpStatusList;

impl StatusListResolver for NoOpStatusList {
    fn is_revoked(&self, _issuer: &IdentityRef, _index: u32) -> Option<bool> {
        Some(false)
    }
}

/// Verify a memo's credential and input/output binding.
///
/// 1. Verifies credential signature, time bounds, and issuer resolution.
/// 2. Re-derives the claim digest from `memo.input`, `memo.output`, and
///    `memo.claim_salt`, then checks it matches `credential.claim_digests[0]`.
///    This prevents tampering with input/output fields after signing.
///
/// Uses a no-op status list resolver since memos do not support revocation.
pub fn verify_memo(
    memo: &Memo,
    now: u64,
    keys: &impl CredentialKeyResolver,
) -> Result<(), MemoError> {
    // 1. Verify credential signature + time bounds
    verify_credential(&memo.credential, now, keys, &NoOpStatusList)?;

    // 2. Enforce self-attestation: issuer must equal subject
    if memo.credential.issuer != memo.credential.subject {
        return Err(MemoError::Credential(
            harmony_credential::CredentialError::SignatureInvalid,
        ));
    }

    // 3. Verify input/output binding against the signed claim digest
    if memo.credential.claim_digests.is_empty() {
        return Err(MemoError::ClaimDecodingFailed);
    }

    // Reconstruct the claim value: input || output (64 bytes)
    let mut claim_value = Vec::with_capacity(64);
    claim_value.extend_from_slice(&memo.input.to_bytes());
    claim_value.extend_from_slice(&memo.output.to_bytes());

    // Re-derive the digest using the stored salt
    let salted = SaltedClaim {
        claim: Claim {
            type_id: MEMO_CLAIM_TYPE,
            value: claim_value,
        },
        salt: memo.claim_salt,
    };
    let expected_digest = salted.digest();

    if memo.credential.claim_digests[0] != expected_digest {
        return Err(MemoError::InputOutputMismatch);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::create::create_memo;
    use alloc::vec::Vec;
    use harmony_content::ContentId;
    use harmony_identity::{IdentityRef, PqPrivateIdentity};
    use rand::rngs::OsRng;

    fn dummy_content_id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    /// A key resolver that stores a single PQ identity's verifying key.
    struct SingleKeyResolver {
        id_ref: IdentityRef,
        key_bytes: Vec<u8>,
    }

    impl SingleKeyResolver {
        fn from_identity(identity: &PqPrivateIdentity) -> Self {
            let pub_id = identity.public_identity();
            Self {
                id_ref: IdentityRef::from(pub_id),
                key_bytes: pub_id.verifying_key.as_bytes().to_vec(),
            }
        }
    }

    impl CredentialKeyResolver for SingleKeyResolver {
        fn resolve(&self, issuer: &IdentityRef, _issued_at: u64) -> Option<Vec<u8>> {
            if issuer == &self.id_ref {
                Some(self.key_bytes.clone())
            } else {
                None
            }
        }
    }

    #[test]
    fn create_and_verify_roundtrip() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        let keys = SingleKeyResolver::from_identity(&identity);
        verify_memo(&memo, 1500, &keys).expect("verify_memo should succeed");
    }

    #[test]
    fn verify_expired_memo_fails() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        let keys = SingleKeyResolver::from_identity(&identity);
        let result = verify_memo(&memo, 3000, &keys);
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoError::Credential(harmony_credential::CredentialError::Expired) => {}
            other => panic!("expected Expired, got: {other:?}"),
        }
    }

    #[test]
    fn verify_not_yet_valid_memo_fails() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        let keys = SingleKeyResolver::from_identity(&identity);
        let result = verify_memo(&memo, 500, &keys);
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoError::Credential(harmony_credential::CredentialError::NotYetValid) => {}
            other => panic!("expected NotYetValid, got: {other:?}"),
        }
    }

    #[test]
    fn verify_tampered_input_fails() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let mut memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        // Tamper with the input field
        memo.input = dummy_content_id(0xFF);

        let keys = SingleKeyResolver::from_identity(&identity);
        let result = verify_memo(&memo, 1500, &keys);
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoError::InputOutputMismatch => {}
            other => panic!("expected InputOutputMismatch, got: {other:?}"),
        }
    }

    #[test]
    fn verify_tampered_output_fails() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let mut memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        // Tamper with the output field
        memo.output = dummy_content_id(0xFF);

        let keys = SingleKeyResolver::from_identity(&identity);
        let result = verify_memo(&memo, 1500, &keys);
        assert!(result.is_err());
        match result.unwrap_err() {
            MemoError::InputOutputMismatch => {}
            other => panic!("expected InputOutputMismatch, got: {other:?}"),
        }
    }

    #[test]
    fn verify_unknown_issuer_fails() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let other_identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        // Use a resolver that only knows about a different identity
        let keys = SingleKeyResolver::from_identity(&other_identity);
        let result = verify_memo(&memo, 1500, &keys);
        assert!(result.is_err());
    }

    #[test]
    fn serialize_deserialize_then_verify() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        // Serialize and deserialize
        let bytes = crate::serialize(&memo).expect("serialize");
        let restored = crate::deserialize(&bytes).expect("deserialize");

        // Verify the deserialized memo
        let keys = SingleKeyResolver::from_identity(&identity);
        verify_memo(&restored, 1500, &keys).expect("verify_memo should succeed on deserialized memo");

        // Verify fields match
        assert_eq!(restored.input, memo.input);
        assert_eq!(restored.output, memo.output);
    }
}

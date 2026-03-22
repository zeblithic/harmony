use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_credential::CredentialBuilder;
use harmony_identity::{IdentityRef, PqPrivateIdentity};
use rand_core::CryptoRngCore;

use crate::{Memo, MemoError, MEMO_CLAIM_TYPE};

/// Create a self-attested memo binding `input` to `output`.
///
/// The memo is a credential where issuer = subject (self-attestation).
/// The claim value is `input.to_bytes() || output.to_bytes()` (64 bytes),
/// hashed with a random salt for selective disclosure.
///
/// # Parameters
///
/// - `input`: the content that was consumed/read
/// - `output`: the content that was produced
/// - `identity`: the signer's PQ private identity
/// - `rng`: cryptographic RNG (sans-I/O: caller provides)
/// - `now`: current timestamp as Unix seconds (sans-I/O: caller provides)
/// - `expires_at`: credential expiry as Unix seconds
///
/// # Errors
///
/// Returns `MemoError::Credential` if signing fails.
pub fn create_memo(
    input: ContentId,
    output: ContentId,
    identity: &PqPrivateIdentity,
    rng: &mut impl CryptoRngCore,
    now: u64,
    expires_at: u64,
) -> Result<Memo, MemoError> {
    let id_ref = IdentityRef::from(identity.public_identity());

    // Build claim value: input || output (64 bytes)
    let mut claim_value = Vec::with_capacity(64);
    claim_value.extend_from_slice(&input.to_bytes());
    claim_value.extend_from_slice(&output.to_bytes());

    // Random salt for selective disclosure
    let mut salt = [0u8; 16];
    rng.fill_bytes(&mut salt);

    // Random nonce for the credential
    let mut nonce = [0u8; 16];
    rng.fill_bytes(&mut nonce);

    // Build the credential (issuer = subject for self-attestation)
    let mut builder = CredentialBuilder::new(id_ref, id_ref, now, expires_at, nonce);
    builder.add_claim(MEMO_CLAIM_TYPE, claim_value, salt);

    // Sans-I/O signing: get payload, sign externally, finalize
    let payload = builder.signable_payload();
    let signature = identity
        .sign(&payload)
        .map_err(|_| MemoError::Credential(harmony_credential::CredentialError::SignatureInvalid))?;

    let (credential, _salted_claims) = builder.build(signature);

    Ok(Memo {
        input,
        output,
        credential,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn dummy_content_id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    #[test]
    fn create_memo_produces_valid_structure() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);

        let memo = create_memo(input, output, &identity, &mut OsRng, 1000, 2000)
            .expect("create_memo should succeed");

        assert_eq!(memo.input, input);
        assert_eq!(memo.output, output);

        let id_ref = IdentityRef::from(identity.public_identity());
        assert_eq!(memo.credential.issuer, id_ref);
        assert_eq!(memo.credential.subject, id_ref);
        assert_eq!(memo.credential.issued_at, 1000);
        assert_eq!(memo.credential.expires_at, 2000);
        assert_eq!(memo.credential.claim_digests.len(), 1);
    }

    #[test]
    fn create_memo_different_inputs_produce_different_digests() {
        let identity = PqPrivateIdentity::generate(&mut OsRng);
        let input_a = dummy_content_id(0x11);
        let input_b = dummy_content_id(0x33);
        let output = dummy_content_id(0x22);

        let memo_a = create_memo(input_a, output, &identity, &mut OsRng, 1000, 2000).unwrap();
        let memo_b = create_memo(input_b, output, &identity, &mut OsRng, 1000, 2000).unwrap();

        // Different inputs should produce different claim digests
        assert_ne!(
            memo_a.credential.claim_digests[0],
            memo_b.credential.claim_digests[0]
        );
    }
}

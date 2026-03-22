#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod create;
pub mod store;
pub mod verify;

use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_credential::{Credential, CredentialError};
use serde::{Deserialize, Serialize};

/// Claim type identifier for memo attestations.
///
/// `0x4D45` = ASCII "ME" — identifies a claim as a memo input/output binding.
pub const MEMO_CLAIM_TYPE: u16 = 0x4D45;

/// Wire format version byte, prefixed to serialized memos.
pub const FORMAT_VERSION: u8 = 1;

/// A signed attestation that an identity produced `output` from `input`.
///
/// The `credential` field carries the cryptographic proof (issuer = subject
/// for self-attestation). The claim value inside the credential is
/// `input.to_bytes() || output.to_bytes()` (64 bytes), salted and hashed.
/// The `claim_salt` is stored so receivers can re-derive the claim digest
/// and verify the input/output binding is authentic (not tampered after signing).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Memo {
    pub input: ContentId,
    pub output: ContentId,
    pub credential: Credential,
    /// Salt used for the memo claim's selective disclosure hash.
    /// Required for receivers to verify the input/output binding.
    pub claim_salt: [u8; 16],
}

/// Errors from memo operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MemoError {
    /// The underlying credential failed verification.
    Credential(CredentialError),
    /// Could not decode the claim inside the credential.
    ClaimDecodingFailed,
    /// Memo violates self-attestation: issuer != subject.
    SelfAttestationViolated,
    /// The input/output in the memo do not match the claim payload.
    InputOutputMismatch,
    /// Serialization or deserialization failed.
    SerializationError,
}

impl core::fmt::Display for MemoError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Credential(e) => write!(f, "credential error: {e}"),
            Self::SelfAttestationViolated => write!(f, "memo issuer != subject (not self-attested)"),
            Self::ClaimDecodingFailed => write!(f, "memo claim decoding failed"),
            Self::InputOutputMismatch => write!(f, "memo input/output does not match claim"),
            Self::SerializationError => write!(f, "memo serialization error"),
        }
    }
}

impl From<CredentialError> for MemoError {
    fn from(e: CredentialError) -> Self {
        MemoError::Credential(e)
    }
}

/// Serialize a memo to bytes: `[FORMAT_VERSION][postcard payload]`.
pub fn serialize(memo: &Memo) -> Result<Vec<u8>, MemoError> {
    let mut buf = Vec::new();
    buf.push(FORMAT_VERSION);
    let inner =
        postcard::to_allocvec(memo).map_err(|_| MemoError::SerializationError)?;
    buf.extend_from_slice(&inner);
    Ok(buf)
}

/// Deserialize a memo from bytes (expects a version prefix).
pub fn deserialize(data: &[u8]) -> Result<Memo, MemoError> {
    if data.is_empty() {
        return Err(MemoError::SerializationError);
    }
    if data[0] != FORMAT_VERSION {
        return Err(MemoError::SerializationError);
    }
    postcard::from_bytes(&data[1..]).map_err(|_| MemoError::SerializationError)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentId;
    use harmony_credential::CredentialBuilder;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn dummy_content_id(byte: u8) -> ContentId {
        ContentId::from_bytes([byte; 32])
    }

    fn dummy_memo() -> Memo {
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65);
        let mut builder = CredentialBuilder::new(issuer, issuer, 1000, 2000, [0x01; 16]);
        let input = dummy_content_id(0x11);
        let output = dummy_content_id(0x22);
        let mut claim_value = Vec::new();
        claim_value.extend_from_slice(&input.to_bytes());
        claim_value.extend_from_slice(&output.to_bytes());
        builder.add_claim(MEMO_CLAIM_TYPE, claim_value, [0x33; 16]);
        let payload = builder.signable_payload();
        // NB: payload used as stand-in for a real signature. This credential
        // is NOT cryptographically valid — use only for serialization tests.
        let (credential, _claims) = builder.build(payload);
        Memo {
            input,
            output,
            credential,
            claim_salt: [0x33; 16],
        }
    }

    #[test]
    fn claim_type_constant() {
        // 0x4D = 'M', 0x45 = 'E'
        assert_eq!(MEMO_CLAIM_TYPE, 0x4D45);
        assert_eq!(MEMO_CLAIM_TYPE, u16::from_be_bytes([b'M', b'E']));
    }

    #[test]
    fn format_version() {
        assert_eq!(FORMAT_VERSION, 1);
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let memo = dummy_memo();
        let bytes = serialize(&memo).expect("serialize");
        assert_eq!(bytes[0], FORMAT_VERSION);

        let restored = deserialize(&bytes).expect("deserialize");
        assert_eq!(restored.input, memo.input);
        assert_eq!(restored.output, memo.output);
        assert_eq!(restored.credential.issuer, memo.credential.issuer);
        assert_eq!(restored.credential.subject, memo.credential.subject);
        assert_eq!(
            restored.credential.claim_digests,
            memo.credential.claim_digests
        );
        assert_eq!(restored.credential.nonce, memo.credential.nonce);
    }

    #[test]
    fn deserialize_wrong_version() {
        let memo = dummy_memo();
        let mut bytes = serialize(&memo).unwrap();
        bytes[0] = 0xFF; // corrupt version
        assert!(matches!(
            deserialize(&bytes),
            Err(MemoError::SerializationError)
        ));
    }

    #[test]
    fn deserialize_empty() {
        assert!(matches!(
            deserialize(&[]),
            Err(MemoError::SerializationError)
        ));
    }
}

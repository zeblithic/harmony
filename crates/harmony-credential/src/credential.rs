use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::claim::SaltedClaim;
use crate::error::CredentialError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of a credential.
/// Everything except the signature.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    issuer: IdentityRef,
    subject: IdentityRef,
    claim_digests: Vec<[u8; 32]>,
    status_list_index: Option<u32>,
    not_before: u64,
    expires_at: u64,
    issued_at: u64,
    nonce: [u8; 16],
}

/// A compact binary verifiable credential.
///
/// Binds an issuer's signature to a set of claim digests about a subject.
/// Claims are stored as BLAKE3 hashes for selective disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub issuer: IdentityRef,
    pub subject: IdentityRef,
    pub claim_digests: Vec<[u8; 32]>,
    pub status_list_index: Option<u32>,
    pub not_before: u64,
    pub expires_at: u64,
    pub issued_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl Credential {
    /// BLAKE3 hash of the signable payload. Stable identifier for this
    /// credential, usable for indexing, deduplication, and future
    /// delegation chain references.
    pub fn content_hash(&self) -> [u8; 32] {
        blake3_hash(&self.signable_bytes())
    }

    /// Reconstruct the signable payload bytes (everything except signature).
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: self.claim_digests.clone(),
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize the credential to bytes with a format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, CredentialError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| CredentialError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize a credential from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, CredentialError> {
        if data.is_empty() {
            return Err(CredentialError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(CredentialError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| CredentialError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a credential.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final credential.
pub struct CredentialBuilder {
    issuer: IdentityRef,
    subject: IdentityRef,
    issued_at: u64,
    expires_at: u64,
    not_before: u64,
    nonce: [u8; 16],
    claims: Vec<SaltedClaim>,
    status_list_index: Option<u32>,
}

impl CredentialBuilder {
    /// Create a new builder.
    ///
    /// `not_before` defaults to `issued_at`. `expires_at` and `nonce` are
    /// required to prevent accidental omission.
    pub fn new(
        issuer: IdentityRef,
        subject: IdentityRef,
        issued_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        Self {
            issuer,
            subject,
            issued_at,
            expires_at,
            not_before: issued_at,
            nonce,
            claims: Vec::new(),
            status_list_index: None,
        }
    }

    /// Override the `not_before` timestamp (defaults to `issued_at`).
    pub fn not_before(&mut self, not_before: u64) -> &mut Self {
        self.not_before = not_before;
        self
    }

    /// Add a claim with the given type, value, and salt.
    pub fn add_claim(&mut self, type_id: u16, value: Vec<u8>, salt: [u8; 16]) -> &mut Self {
        self.claims.push(SaltedClaim {
            claim: crate::claim::Claim { type_id, value },
            salt,
        });
        self
    }

    /// Set the revocation status list index.
    pub fn status_list_index(&mut self, index: u32) -> &mut Self {
        self.status_list_index = Some(index);
        self
    }

    /// Produce the signable payload bytes.
    ///
    /// The caller signs these bytes externally with their private key.
    pub fn signable_payload(&self) -> Vec<u8> {
        let digests: Vec<[u8; 32]> = self.claims.iter().map(|c| c.digest()).collect();
        let payload = SignablePayload {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: digests,
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature to produce the Credential + SaltedClaims.
    ///
    /// The `SaltedClaim`s should be stored by the holder for later
    /// selective disclosure.
    pub fn build(self, signature: Vec<u8>) -> (Credential, Vec<SaltedClaim>) {
        let digests: Vec<[u8; 32]> = self.claims.iter().map(|c| c.digest()).collect();
        let credential = Credential {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: digests,
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
            signature,
        };
        (credential, self.claims)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_issuer() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    fn test_subject() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_digest_count() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);

        let payload = builder.signable_payload();
        let (cred, claims) = builder.build(payload.clone());

        assert_eq!(cred.claim_digests.len(), 2);
        assert_eq!(claims.len(), 2);
        assert_eq!(cred.issuer, test_issuer());
        assert_eq!(cred.subject, test_subject());
        assert_eq!(cred.not_before, 1000);
        assert_eq!(cred.expires_at, 2000);
        assert_eq!(cred.issued_at, 1000);
        assert_eq!(cred.nonce, [0x01; 16]);
        assert!(cred.status_list_index.is_none());
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);

        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    fn content_hash_is_deterministic() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);

        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.content_hash(), cred.content_hash());
    }

    #[test]
    fn not_before_defaults_to_issued_at() {
        let builder = CredentialBuilder::new(test_issuer(), test_subject(), 500, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.not_before, 500);
    }

    #[test]
    fn not_before_can_be_overridden() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 500, 2000, [0x01; 16]);
        builder.not_before(700);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.not_before, 700);
    }

    #[test]
    fn status_list_index_set() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
        builder.status_list_index(42);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.status_list_index, Some(42));
    }

    #[test]
    fn self_issued_credential() {
        let id = test_issuer();
        let builder = CredentialBuilder::new(id, id, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.issuer, cred.subject);
    }

    #[test]
    fn serde_round_trip() {
        let mut builder =
            CredentialBuilder::new(test_issuer(), test_subject(), 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.status_list_index(5);

        let _payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = cred.serialize().unwrap();
        let restored = Credential::deserialize(&bytes).unwrap();

        assert_eq!(restored.issuer, cred.issuer);
        assert_eq!(restored.subject, cred.subject);
        assert_eq!(restored.claim_digests, cred.claim_digests);
        assert_eq!(restored.status_list_index, cred.status_list_index);
        assert_eq!(restored.not_before, cred.not_before);
        assert_eq!(restored.expires_at, cred.expires_at);
        assert_eq!(restored.issued_at, cred.issued_at);
        assert_eq!(restored.nonce, cred.nonce);
        assert_eq!(restored.signature, cred.signature);
        assert_eq!(restored.content_hash(), cred.content_hash());
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            Credential::deserialize(&[]),
            Err(CredentialError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            Credential::deserialize(&[0xFF]),
            Err(CredentialError::DeserializeError(
                "unsupported format version"
            ))
        ));
    }
}

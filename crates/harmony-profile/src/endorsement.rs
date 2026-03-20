use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::ProfileError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of an endorsement record.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    endorser: IdentityRef,
    endorsee: IdentityRef,
    type_id: u32,
    reason: Option<String>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// A signed endorsement from one identity about another.
///
/// Endorsements are attestations of fact ("I verified their credential"),
/// not expressions of trust. They are signed by the endorser and
/// published at `harmony/endorsement/{endorser_hex}/{endorsee_hex}`.
///
/// # Construction
///
/// Produce records via [`EndorsementBuilder`]; direct struct construction
/// bypasses validity checks.
///
/// Verified by [`verify_endorsement()`](crate::verify::verify_endorsement).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndorsementRecord {
    pub endorser: IdentityRef,
    pub endorsee: IdentityRef,
    /// Endorsement category. Interpreted as a Q8 page address
    /// (2-28-2 format) for content-addressed type definitions.
    pub type_id: u32,
    pub reason: Option<String>,
    pub published_at: u64,
    pub expires_at: u64,
    /// Must be cryptographically random and unique per publication.
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl EndorsementRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    ///
    /// Uses the compile-time `FORMAT_VERSION` constant. Records from
    /// persistent storage MUST go through [`EndorsementRecord::deserialize`].
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    pub fn serialize(&self) -> Result<Vec<u8>, ProfileError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| ProfileError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.is_empty() {
            return Err(ProfileError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ProfileError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ProfileError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a signed endorsement record.
pub struct EndorsementBuilder {
    endorser: IdentityRef,
    endorsee: IdentityRef,
    type_id: u32,
    reason: Option<String>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl EndorsementBuilder {
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at` or if `endorser == endorsee`.
    pub fn new(
        endorser: IdentityRef,
        endorsee: IdentityRef,
        type_id: u32,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        assert!(
            expires_at > published_at,
            "expires_at ({expires_at}) must be > published_at ({published_at})"
        );
        assert_ne!(
            endorser.hash, endorsee.hash,
            "endorser and endorsee must be different identities"
        );
        Self {
            endorser,
            endorsee,
            type_id,
            reason: None,
            published_at,
            expires_at,
            nonce,
        }
    }

    pub fn reason(&mut self, reason: String) -> &mut Self {
        self.reason = Some(reason);
        self
    }

    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    pub fn build(self, signature: Vec<u8>) -> EndorsementRecord {
        EndorsementRecord {
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
            signature,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_endorser() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65)
    }

    fn test_endorsee() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_fields() {
        let mut builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        builder.reason(alloc::string::String::from("Verified in person"));

        let payload = builder.signable_payload();
        let record = builder.build(payload.clone());

        assert_eq!(record.endorser, test_endorser());
        assert_eq!(record.endorsee, test_endorsee());
        assert_eq!(record.type_id, 42);
        assert_eq!(record.reason.as_deref(), Some("Verified in person"));
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_before_published_at() {
        EndorsementBuilder::new(test_endorser(), test_endorsee(), 42, 2000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        EndorsementBuilder::new(test_endorser(), test_endorsee(), 42, 1000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "endorser and endorsee must be different")]
    fn rejects_self_endorsement() {
        let id = test_endorser();
        EndorsementBuilder::new(id, id, 42, 1000, 2000, [0x01; 16]);
    }

    #[test]
    fn endorsement_without_reason() {
        let builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            99,
            1000,
            2000,
            [0x01; 16],
        );
        let record = builder.build(alloc::vec![0xDE, 0xAD]);
        assert!(record.reason.is_none());
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        builder.reason(alloc::string::String::from("Verified"));

        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = EndorsementRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored, record);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            EndorsementRecord::deserialize(&[]),
            Err(ProfileError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            EndorsementRecord::deserialize(&[0xFF]),
            Err(ProfileError::DeserializeError("unsupported format version"))
        ));
    }
}

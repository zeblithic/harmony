use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::ProfileError;

const FORMAT_VERSION: u8 = 1;

/// Domain separator for profile record signatures.
const RECORD_TYPE: u8 = 0x01;

/// Internal struct for the signable portion of a profile record.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    record_type: u8,
    identity_ref: IdentityRef,
    display_name: Option<String>,
    status_text: Option<String>,
    avatar_cid: Option<[u8; 32]>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// A signed profile record publishing an identity's public metadata.
///
/// # Construction
///
/// Produce records via [`ProfileBuilder`]; direct struct construction
/// bypasses validity checks (e.g. `expires_at > published_at`).
///
/// Verified by [`verify_profile()`](crate::verify::verify_profile).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub identity_ref: IdentityRef,
    pub display_name: Option<String>,
    /// Freetext status. Future convention: `/book/<address>` for CAS
    /// content bundles.
    pub status_text: Option<String>,
    pub avatar_cid: Option<[u8; 32]>,
    pub published_at: u64,
    pub expires_at: u64,
    /// Must be cryptographically random and unique per publication.
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl ProfileRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    ///
    /// Uses the compile-time `FORMAT_VERSION` constant. Records from
    /// persistent storage MUST go through [`ProfileRecord::deserialize`].
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            record_type: RECORD_TYPE,
            identity_ref: self.identity_ref,
            display_name: self.display_name.clone(),
            status_text: self.status_text.clone(),
            avatar_cid: self.avatar_cid,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize with format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, ProfileError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| ProfileError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.is_empty() {
            return Err(ProfileError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ProfileError::DeserializeError("unsupported format version"));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ProfileError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a signed profile record.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final record.
pub struct ProfileBuilder {
    identity_ref: IdentityRef,
    display_name: Option<String>,
    status_text: Option<String>,
    avatar_cid: Option<[u8; 32]>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl ProfileBuilder {
    /// Create a new builder.
    ///
    /// All timestamps are unix epoch seconds. `nonce` must be
    /// cryptographically random.
    ///
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        identity_ref: IdentityRef,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        assert!(
            expires_at > published_at,
            "expires_at ({expires_at}) must be > published_at ({published_at})"
        );
        Self {
            identity_ref,
            display_name: None,
            status_text: None,
            avatar_cid: None,
            published_at,
            expires_at,
            nonce,
        }
    }

    pub fn display_name(&mut self, name: String) -> &mut Self {
        self.display_name = Some(name);
        self
    }

    pub fn status_text(&mut self, text: String) -> &mut Self {
        self.status_text = Some(text);
        self
    }

    pub fn avatar_cid(&mut self, cid: [u8; 32]) -> &mut Self {
        self.avatar_cid = Some(cid);
        self
    }

    /// Produce the signable payload bytes.
    ///
    /// Call this **after** all optional fields are set. Mutating the
    /// builder after this point will produce a record whose fields no
    /// longer match the signed payload, causing `SignatureInvalid`.
    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            record_type: RECORD_TYPE,
            identity_ref: self.identity_ref,
            display_name: self.display_name.clone(),
            status_text: self.status_text.clone(),
            avatar_cid: self.avatar_cid,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature.
    pub fn build(self, signature: Vec<u8>) -> ProfileRecord {
        ProfileRecord {
            identity_ref: self.identity_ref,
            display_name: self.display_name,
            status_text: self.status_text,
            avatar_cid: self.avatar_cid,
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

    fn test_identity() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_fields() {
        let id = test_identity();
        let mut builder = ProfileBuilder::new(id, 1000, 2000, [0x01; 16]);
        builder
            .display_name(alloc::string::String::from("Alice"))
            .status_text(alloc::string::String::from("Hello world"))
            .avatar_cid([0xCC; 32]);

        let payload = builder.signable_payload();
        let record = builder.build(payload.clone());

        assert_eq!(record.identity_ref, id);
        assert_eq!(record.display_name.as_deref(), Some("Alice"));
        assert_eq!(record.status_text.as_deref(), Some("Hello world"));
        assert_eq!(record.avatar_cid, Some([0xCC; 32]));
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_before_published_at() {
        ProfileBuilder::new(test_identity(), 2000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        ProfileBuilder::new(test_identity(), 1000, 1000, [0x01; 16]);
    }

    #[test]
    fn sparse_profile_works() {
        let builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(payload);
        assert!(record.display_name.is_none());
        assert!(record.status_text.is_none());
        assert!(record.avatar_cid.is_none());
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        builder.display_name(alloc::string::String::from("Bob"));

        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = ProfileRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored, record);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            ProfileRecord::deserialize(&[]),
            Err(ProfileError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            ProfileRecord::deserialize(&[0xFF]),
            Err(ProfileError::DeserializeError("unsupported format version"))
        ));
    }
}

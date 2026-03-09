//! License manifest types and signing.

use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_identity::{Identity, PrivateIdentity};
use serde::{Deserialize, Serialize};

use crate::error::RoxyError;
use crate::types::{LicenseType, Price, UsageRights};

/// Custom serde helper for 64-byte signature arrays.
///
/// Postcard does not handle large fixed-size arrays well, so we serialize
/// the signature as a `Vec<u8>` on the wire and reconstitute on deserialize.
mod serde_sig {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(sig: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error> {
        sig.as_slice().serialize(serializer)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(deserializer: D) -> Result<[u8; 64], D::Error> {
        let v: alloc::vec::Vec<u8> = Deserialize::deserialize(deserializer)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("signature must be exactly 64 bytes"))
    }
}

/// A license manifest that an artist publishes to describe licensing terms
/// for a piece of content.
///
/// The manifest is serialized with postcard, content-addressed (gets its own
/// CID), and signed by the artist's Ed25519 key.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseManifest {
    /// Artist's 128-bit address hash.
    pub creator: [u8; 16],
    /// CID of the licensed content (single blob or bundle).
    pub content_cid: ContentId,
    /// Schema version for forward compatibility.
    pub manifest_version: u8,
    /// Type of license offered.
    pub license_type: LicenseType,
    /// Price for the license. `None` for free content.
    pub price: Option<Price>,
    /// Access window per grant in seconds. `None` means perpetual.
    pub duration_secs: Option<u64>,
    /// What the consumer is allowed to do with the content.
    pub usage_rights: UsageRights,
    /// Seconds before expiry to send a renewal notice.
    pub expiry_notice_secs: u32,
    /// CID of the encrypted symmetric key blob.
    pub content_key_cid: ContentId,
    /// Ed25519 signature over the manifest (zeroed during signing).
    #[serde(with = "serde_sig")]
    pub signature: [u8; 64],
}

impl LicenseManifest {
    /// Serialize the manifest to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, RoxyError> {
        postcard::to_allocvec(self).map_err(RoxyError::from)
    }

    /// Deserialize a manifest from postcard bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, RoxyError> {
        postcard::from_bytes(bytes).map_err(RoxyError::from)
    }

    /// Produce the canonical bytes used for signing/verification.
    ///
    /// This is the manifest serialized with a zeroed signature field,
    /// ensuring deterministic output regardless of current signature value.
    pub fn signable_bytes(&self) -> Result<Vec<u8>, RoxyError> {
        let mut copy = self.clone();
        copy.signature = [0u8; 64];
        copy.to_bytes()
    }

    /// Sign the manifest with the artist's private identity.
    ///
    /// Sets `self.signature` to the Ed25519 signature over
    /// [`Self::signable_bytes`].
    pub fn sign(&mut self, artist: &PrivateIdentity) {
        let signable = self
            .signable_bytes()
            .expect("signable serialization should not fail");
        self.signature = artist.sign(&signable);
    }

    /// Verify the manifest signature against the given public identity.
    ///
    /// Returns `Err(RoxyError::CreatorMismatch)` if the identity's address
    /// hash does not match `self.creator`, or `Err(RoxyError::InvalidSignature)`
    /// if the signature does not verify.
    pub fn verify(&self, identity: &Identity) -> Result<(), RoxyError> {
        if identity.address_hash != self.creator {
            return Err(RoxyError::CreatorMismatch);
        }
        let signable = self.signable_bytes()?;
        identity
            .verify(&signable, &self.signature)
            .map_err(|_| RoxyError::InvalidSignature)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{LicenseType, Price, PricePer, UsageRights};
    use harmony_content::ContentFlags;
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    fn make_test_manifest(artist: &PrivateIdentity) -> LicenseManifest {
        let content_cid =
            harmony_content::ContentId::for_blob(b"test content", ContentFlags::default()).unwrap();
        let key_cid = harmony_content::ContentId::for_blob(
            b"encrypted key blob",
            ContentFlags {
                encrypted: true,
                ..Default::default()
            },
        )
        .unwrap();

        LicenseManifest {
            creator: artist.public_identity().address_hash,
            content_cid,
            manifest_version: 1,
            license_type: LicenseType::Subscription,
            price: Some(Price {
                amount: 500,
                currency: alloc::string::String::from("USD"),
                per: PricePer::Month,
            }),
            duration_secs: Some(30 * 24 * 3600),
            usage_rights: UsageRights::STREAM | UsageRights::DOWNLOAD,
            expiry_notice_secs: 3 * 24 * 3600,
            content_key_cid: key_cid,
            signature: [0u8; 64],
        }
    }

    #[test]
    fn manifest_serialization_round_trip() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let manifest = make_test_manifest(&artist);
        let bytes = manifest.to_bytes().unwrap();
        let decoded = LicenseManifest::from_bytes(&bytes).unwrap();
        assert_eq!(manifest.creator, decoded.creator);
        assert_eq!(manifest.content_cid, decoded.content_cid);
        assert_eq!(manifest.manifest_version, decoded.manifest_version);
        assert_eq!(manifest.license_type, decoded.license_type);
        assert_eq!(
            manifest.price.as_ref().map(|p| p.amount),
            decoded.price.as_ref().map(|p| p.amount)
        );
        assert_eq!(manifest.duration_secs, decoded.duration_secs);
        assert_eq!(manifest.usage_rights, decoded.usage_rights);
        assert_eq!(manifest.expiry_notice_secs, decoded.expiry_notice_secs);
        assert_eq!(manifest.content_key_cid, decoded.content_key_cid);
    }

    #[test]
    fn manifest_sign_and_verify() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);
        assert!(manifest.verify(&artist.public_identity()).is_ok());
    }

    #[test]
    fn manifest_verify_rejects_wrong_signer() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let imposter = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);
        let result = manifest.verify(&imposter.public_identity());
        assert!(result.is_err());
    }

    #[test]
    fn manifest_verify_rejects_tampered_data() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let mut manifest = make_test_manifest(&artist);
        manifest.sign(&artist);
        manifest.price = Some(Price {
            amount: 0,
            currency: alloc::string::String::from("USD"),
            per: PricePer::Month,
        });
        let result = manifest.verify(&artist.public_identity());
        assert!(result.is_err());
    }

    #[test]
    fn free_manifest_has_no_price_or_duration() {
        let artist = PrivateIdentity::generate(&mut OsRng);
        let content_cid =
            harmony_content::ContentId::for_blob(b"free content", ContentFlags::default()).unwrap();
        let key_cid =
            harmony_content::ContentId::for_blob(b"public key", ContentFlags::default()).unwrap();
        let mut manifest = LicenseManifest {
            creator: artist.public_identity().address_hash,
            content_cid,
            manifest_version: 1,
            license_type: LicenseType::Free,
            price: None,
            duration_secs: None,
            usage_rights: UsageRights::STREAM | UsageRights::DOWNLOAD,
            expiry_notice_secs: 0,
            content_key_cid: key_cid,
            signature: [0u8; 64],
        };
        manifest.sign(&artist);
        assert!(manifest.verify(&artist.public_identity()).is_ok());
    }
}

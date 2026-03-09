//! UCAN resource field encoding for Roxy license grants.
//!
//! The UCAN token's `resource` field is opaque bytes (max 256).
//! Roxy encodes the manifest CID so grants can be verified against
//! specific license manifests.

use alloc::vec::Vec;
use harmony_content::ContentId;

use crate::error::RoxyError;

/// Current resource encoding version.
const RESOURCE_VERSION: u8 = 1;

/// Encode a manifest CID into UCAN resource bytes.
///
/// Format: `[version: 1B][manifest_cid: 32B]` = 33 bytes total.
pub fn encode_resource(manifest_cid: &ContentId) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(33);
    bytes.push(RESOURCE_VERSION);
    bytes.extend_from_slice(&manifest_cid.to_bytes());
    bytes
}

/// Decode a manifest CID from UCAN resource bytes.
pub fn decode_resource(bytes: &[u8]) -> Result<ContentId, RoxyError> {
    if bytes.len() < 33 {
        return Err(RoxyError::InvalidResource);
    }
    if bytes[0] != RESOURCE_VERSION {
        return Err(RoxyError::InvalidResource);
    }
    let mut cid_bytes = [0u8; 32];
    cid_bytes.copy_from_slice(&bytes[1..33]);
    Ok(ContentId::from_bytes(cid_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn encode_decode_round_trip() {
        let manifest_cid = make_cid(b"manifest");
        let encoded = encode_resource(&manifest_cid);
        let decoded = decode_resource(&encoded).unwrap();
        assert_eq!(manifest_cid, decoded);
    }

    #[test]
    fn encoded_resource_fits_ucan() {
        let manifest_cid = make_cid(b"manifest");
        let encoded = encode_resource(&manifest_cid);
        assert!(encoded.len() <= 256);
    }

    #[test]
    fn decode_rejects_wrong_version() {
        let mut encoded = alloc::vec![99u8]; // wrong version
        encoded.extend_from_slice(&[0u8; 32]);
        let result = decode_resource(&encoded);
        assert!(result.is_err());
    }

    #[test]
    fn decode_rejects_short_input() {
        let result = decode_resource(&[1u8; 5]);
        assert!(result.is_err());
    }
}

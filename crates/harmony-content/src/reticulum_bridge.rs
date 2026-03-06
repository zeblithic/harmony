//! Reticulum content bridge — maps content blobs to resource transfers.
//!
//! Each blob (up to ~1MB) maps 1:1 to a single Reticulum resource transfer.
//! The CID is prepended as a 32-byte header so the receiver can verify
//! content integrity independently of the transport hash.

use alloc::vec::Vec;
use crate::cid::ContentId;
use crate::error::ContentError;

/// CID header length prepended to blob data for transport.
const CID_HEADER_LEN: usize = 32;

/// Pack a content blob for transport: CID header (32 bytes) + blob data.
///
/// The caller feeds the returned bytes into `ResourceSender::new()`.
pub fn pack_for_transport(cid: &ContentId, data: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(CID_HEADER_LEN + data.len());
    packed.extend_from_slice(&cid.to_bytes());
    packed.extend_from_slice(data);
    packed
}

/// Unpack received transport data: extract CID header and verify blob integrity.
///
/// Returns `(cid, blob_data)` if the CID matches the data's hash.
/// Returns `ContentError` if the data is too short or the hash doesn't match.
pub fn unpack_from_transport(transport_data: &[u8]) -> Result<(ContentId, Vec<u8>), ContentError> {
    if transport_data.len() < CID_HEADER_LEN {
        return Err(ContentError::TransportDataTooShort {
            len: transport_data.len(),
            min: CID_HEADER_LEN,
        });
    }

    let cid_bytes: [u8; 32] = transport_data[..CID_HEADER_LEN]
        .try_into()
        .expect("slice is exactly 32 bytes");
    let received_cid = ContentId::from_bytes(cid_bytes);
    let blob_data = &transport_data[CID_HEADER_LEN..];

    // Recompute CID from the blob data and verify it matches.
    let computed_cid = ContentId::for_blob(blob_data)?;

    if computed_cid != received_cid {
        return Err(ContentError::ChecksumMismatch);
    }

    Ok((received_cid, blob_data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::{BlobStore, MemoryBlobStore};

    #[test]
    fn pack_unpack_round_trip() {
        let mut store = MemoryBlobStore::new();
        let data = b"hello harmony reticulum bridge";
        let cid = store.insert(data).unwrap();

        let packed = pack_for_transport(&cid, data);
        assert_eq!(packed.len(), 32 + data.len());

        let (unpacked_cid, unpacked_data) = unpack_from_transport(&packed).unwrap();
        assert_eq!(unpacked_cid, cid);
        assert_eq!(unpacked_data, data);
    }

    #[test]
    fn unpack_detects_corrupt_content() {
        let mut store = MemoryBlobStore::new();
        let data = b"original data for corruption test";
        let cid = store.insert(data).unwrap();

        let mut packed = pack_for_transport(&cid, data);
        // Corrupt a byte in the blob data (after the 32-byte CID header).
        packed[CID_HEADER_LEN] ^= 0xFF;

        let result = unpack_from_transport(&packed);
        assert!(result.is_err(), "corrupted data should fail verification");
    }

    #[test]
    fn unpack_detects_wrong_cid_header() {
        let mut store = MemoryBlobStore::new();
        let data_a = b"blob A content";
        let data_b = b"blob B different content";
        let cid_a = store.insert(data_a).unwrap();
        let _cid_b = store.insert(data_b).unwrap();

        // Pack blob B's data with blob A's CID header.
        let packed = pack_for_transport(&cid_a, data_b);

        let result = unpack_from_transport(&packed);
        assert!(result.is_err(), "mismatched CID header should fail");
    }
}

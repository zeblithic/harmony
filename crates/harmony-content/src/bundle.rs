use crate::cid::ContentId;
use crate::error::ContentError;

/// Size of a single ContentId in bytes.
pub const CID_SIZE: usize = 32;

/// Maximum number of CIDs that fit in a single bundle (1MB payload / 32 bytes).
pub const MAX_BUNDLE_ENTRIES: usize = crate::cid::MAX_PAYLOAD_SIZE / CID_SIZE;

/// Parse a bundle's raw bytes into a slice of ContentIds (zero-copy).
///
/// # Safety invariants (all verified at compile time or by assertions):
/// - `ContentId` is `#[repr(C)]` with fields `[u8; 28]` + `[u8; 4]` = 32 bytes
/// - `ContentId` has alignment 1 (only byte-array fields)
/// - Input length is checked to be a multiple of 32
pub fn parse_bundle(data: &[u8]) -> Result<&[ContentId], ContentError> {
    if data.len() % CID_SIZE != 0 {
        return Err(ContentError::InvalidBundleLength { len: data.len() });
    }

    // Compile-time guarantee that ContentId is exactly 32 bytes.
    const _: () = assert!(std::mem::size_of::<ContentId>() == CID_SIZE);
    const _: () = assert!(std::mem::align_of::<ContentId>() == 1);

    let count = data.len() / CID_SIZE;

    // SAFETY: ContentId is #[repr(C)], exactly 32 bytes, alignment 1 (byte arrays only).
    // Input is checked to be a multiple of 32 bytes. &[u8] has alignment 1, matching
    // ContentId's alignment requirement.
    let cids = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const ContentId, count) };
    Ok(cids)
}

/// Validate that all children in a bundle have depth strictly less than the parent.
///
/// Returns `Ok(())` if valid, `Err(DepthViolation)` if any child has depth >= parent_depth.
pub fn validate_bundle_depth(
    parent_depth: u8,
    children: &[ContentId],
) -> Result<(), ContentError> {
    for child in children {
        let child_depth = child.cid_type().depth();
        if child_depth >= parent_depth {
            return Err(ContentError::DepthViolation {
                child: child_depth,
                parent: parent_depth,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::CidType;

    #[test]
    fn parse_valid_bundle() {
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&blob_a.to_bytes());
        bytes.extend_from_slice(&blob_b.to_bytes());

        let cids = parse_bundle(&bytes).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0], blob_a);
        assert_eq!(cids[1], blob_b);
    }

    #[test]
    fn parse_empty_bundle() {
        let cids = parse_bundle(&[]).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn parse_rejects_non_aligned_length() {
        let result = parse_bundle(&[0u8; 33]);
        assert!(matches!(
            result,
            Err(ContentError::InvalidBundleLength { len: 33 })
        ));
    }

    #[test]
    fn validate_depth_accepts_valid_sparse_tree() {
        // L4 bundle containing only L2 children — valid (sparse)
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));

        // L4 parent should accept L2 children
        assert!(validate_bundle_depth(4, &[l2, l2]).is_ok());
    }

    #[test]
    fn validate_depth_rejects_equal_depth() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        assert_eq!(l1.cid_type(), CidType::Bundle(1));

        // L1 parent cannot contain L1 children (depth not strictly less)
        assert!(validate_bundle_depth(1, &[l1]).is_err());
    }

    #[test]
    fn validate_depth_rejects_deeper_child() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));

        // L1 parent cannot contain L2 child
        assert!(validate_bundle_depth(1, &[l2]).is_err());
    }

    #[test]
    fn validate_depth_accepts_mixed_children() {
        // L3 bundle containing [blob, L1, L2] — valid (mixed depths, all < 3)
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();

        assert!(validate_bundle_depth(3, &[blob, l1, l2]).is_ok());
    }
}

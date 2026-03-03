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
pub fn validate_bundle_depth(parent_depth: u8, children: &[ContentId]) -> Result<(), ContentError> {
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

/// Builder for constructing bundles from child CIDs.
///
/// Collects child CIDs, optionally prepends an inline metadata CID,
/// validates depth constraints, and produces the bundle bytes + CID.
pub struct BundleBuilder {
    children: Vec<ContentId>,
    metadata: Option<ContentId>,
}

impl BundleBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        BundleBuilder {
            children: Vec::new(),
            metadata: None,
        }
    }

    /// Add a child CID to the bundle.
    pub fn add(&mut self, cid: ContentId) -> &mut Self {
        self.children.push(cid);
        self
    }

    /// Set inline metadata for a root bundle.
    ///
    /// The metadata CID will be prepended as the first entry.
    pub fn with_metadata(
        &mut self,
        total_size: u64,
        chunk_count: u32,
        timestamp: u64,
        mime: [u8; 8],
    ) -> &mut Self {
        self.metadata = Some(ContentId::inline_metadata(
            total_size,
            chunk_count,
            timestamp,
            mime,
        ));
        self
    }

    /// Build the bundle, returning the raw bytes and the bundle's CID.
    ///
    /// The bundle's depth is `max(child_depths) + 1`. If inline metadata
    /// was set, it is prepended as the first CID in the bundle bytes
    /// (metadata CIDs have depth 0, so they don't affect bundle depth).
    pub fn build(&self) -> Result<(Vec<u8>, ContentId), ContentError> {
        if self.children.is_empty() {
            return Err(ContentError::EmptyBundle);
        }

        // Collect all entries: optional metadata + children
        let mut entries: Vec<ContentId> = Vec::new();
        if let Some(meta) = &self.metadata {
            entries.push(*meta);
        }
        entries.extend_from_slice(&self.children);

        // Serialize to bytes
        let mut bundle_bytes = Vec::with_capacity(entries.len() * CID_SIZE);
        for cid in &entries {
            bundle_bytes.extend_from_slice(&cid.to_bytes());
        }

        // Compute bundle CID (depth based on all entries including metadata)
        let bundle_cid = ContentId::for_bundle(&bundle_bytes, &entries)?;

        Ok((bundle_bytes, bundle_cid))
    }

    /// Return the number of child CIDs (not counting metadata).
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Check if the builder has no children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl Default for BundleBuilder {
    fn default() -> Self {
        Self::new()
    }
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

    #[test]
    fn builder_basic_two_blobs() {
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();

        let mut builder = BundleBuilder::new();
        builder.add(blob_a).add(blob_b);

        let (bytes, cid) = builder.build().unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1));
        assert_eq!(cid.payload_size(), 64); // 2 * 32
        assert_eq!(bytes.len(), 64);

        // Parse back and verify
        let parsed = parse_bundle(&bytes).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], blob_a);
        assert_eq!(parsed[1], blob_b);
    }

    #[test]
    fn builder_with_metadata() {
        let blob = ContentId::for_blob(b"data").unwrap();
        let mut builder = BundleBuilder::new();
        builder.add(blob).with_metadata(1000, 1, 0, *b"text/pln");

        let (bytes, cid) = builder.build().unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1));
        // 2 entries: metadata + blob
        assert_eq!(bytes.len(), 64);

        let parsed = parse_bundle(&bytes).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].cid_type(), CidType::InlineMetadata);
        assert_eq!(parsed[1], blob);
    }

    #[test]
    fn builder_depth_cascade() {
        let blob = ContentId::for_blob(b"leaf").unwrap();

        // L1 bundle
        let mut b1 = BundleBuilder::new();
        b1.add(blob);
        let (_, l1_cid) = b1.build().unwrap();
        assert_eq!(l1_cid.cid_type(), CidType::Bundle(1));

        // L2 bundle wrapping the L1
        let mut b2 = BundleBuilder::new();
        b2.add(l1_cid);
        let (_, l2_cid) = b2.build().unwrap();
        assert_eq!(l2_cid.cid_type(), CidType::Bundle(2));
    }

    #[test]
    fn builder_rejects_empty() {
        let builder = BundleBuilder::new();
        assert!(matches!(builder.build(), Err(ContentError::EmptyBundle)));
    }

    #[test]
    fn builder_rejects_depth_overflow() {
        // Build up to depth 7 then try to wrap
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let mut current = blob;
        for _ in 0..7 {
            let mut b = BundleBuilder::new();
            b.add(current);
            let (_, cid) = b.build().unwrap();
            current = cid;
        }
        assert_eq!(current.cid_type(), CidType::Bundle(7));

        let mut b = BundleBuilder::new();
        b.add(current);
        assert!(b.build().is_err());
    }
}

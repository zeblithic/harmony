use crate::error::ContentError;

/// Length of the truncated SHA-256 content hash in bytes.
pub const CONTENT_HASH_LEN: usize = 28;

/// Maximum payload size expressible in 20 bits.
pub const MAX_PAYLOAD_SIZE: usize = 0xF_FFFF; // 1,048,575 bytes

/// Number of bits used for the type tag + checksum field.
pub const TAG_BITS: u32 = 12;

/// Bitmask for the 12-bit tag field (lower 12 bits of the last u32).
pub const TAG_MASK: u32 = 0xFFF;

/// A 32-byte content identifier.
///
/// Layout (big-endian):
/// - Bytes 0--27: truncated SHA-256 content hash (224 bits)
/// - Bytes 28--31: big-endian u32 where bits \[31:12\] = payload size,
///   bits \[11:0\] = type tag + checksum
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ContentId {
    /// First 28 bytes of SHA-256(content).
    pub hash: [u8; CONTENT_HASH_LEN],
    /// Big-endian u32: upper 20 bits = size, lower 12 bits = tag+checksum.
    pub size_and_tag: [u8; 4],
}

impl ContentId {
    /// Extract the payload size from the packed u32 (upper 20 bits).
    pub fn payload_size(&self) -> u32 {
        let packed = u32::from_be_bytes(self.size_and_tag);
        packed >> TAG_BITS
    }

    /// Extract the 12-bit tag field (lower 12 bits).
    pub fn tag(&self) -> u16 {
        let packed = u32::from_be_bytes(self.size_and_tag);
        (packed & TAG_MASK) as u16
    }

    /// Decode the CID type from the 12-bit tag field.
    pub fn cid_type(&self) -> CidType {
        let (cid_type, _checksum) = CidType::decode(self.tag());
        cid_type
    }

    /// Create a CID for a raw data blob.
    pub fn for_blob(data: &[u8]) -> Result<Self, ContentError> {
        if data.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let full = harmony_crypto::hash::full_hash(data);
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&full[..CONTENT_HASH_LEN]);

        let size = data.len() as u32;
        let cid_type = CidType::Blob;
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        Ok(ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        })
    }

    /// Create a CID for a bundle (array of child CIDs).
    ///
    /// `bundle_bytes` is the raw byte payload (concatenated child CIDs).
    /// `children` is the parsed slice of child CIDs (used for depth calculation).
    /// The bundle's depth is `max(child depths) + 1`.
    pub fn for_bundle(bundle_bytes: &[u8], children: &[ContentId]) -> Result<Self, ContentError> {
        if bundle_bytes.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: bundle_bytes.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let max_child_depth = children
            .iter()
            .map(|c| c.cid_type().depth())
            .max()
            .unwrap_or(0);
        let bundle_depth = max_child_depth + 1;

        if bundle_depth > 7 {
            return Err(ContentError::DepthViolation {
                child: max_child_depth,
                parent: bundle_depth,
            });
        }

        let full = harmony_crypto::hash::full_hash(bundle_bytes);
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&full[..CONTENT_HASH_LEN]);

        let size = bundle_bytes.len() as u32;
        let cid_type = CidType::Bundle(bundle_depth);
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        Ok(ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        })
    }

    /// Extract the checksum from the tag field.
    pub fn checksum(&self) -> u16 {
        let raw = u32::from_be_bytes(self.size_and_tag);
        let tag = (raw & TAG_MASK) as u16;
        let (_cid_type, checksum) = CidType::decode(tag);
        checksum
    }

    /// Verify the CID's checksum against its hash, size, and type.
    pub fn verify_checksum(&self) -> Result<(), ContentError> {
        let cid_type = self.cid_type();
        let expected = compute_checksum(&self.hash, self.payload_size(), &cid_type);
        if self.checksum() != expected {
            return Err(ContentError::ChecksumMismatch);
        }
        Ok(())
    }

    /// Create an inline metadata CID.
    ///
    /// The 28-byte hash field is repurposed to store:
    /// - bytes 0--7: total file size (u64 big-endian)
    /// - bytes 8--11: total chunk count (u32 big-endian)
    /// - bytes 12--19: creation timestamp (u64 big-endian, Unix epoch ms)
    /// - bytes 20--27: MIME type or extension (8 bytes, packed)
    pub fn inline_metadata(
        total_size: u64,
        chunk_count: u32,
        timestamp: u64,
        mime: [u8; 8],
    ) -> Self {
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash[0..8].copy_from_slice(&total_size.to_be_bytes());
        hash[8..12].copy_from_slice(&chunk_count.to_be_bytes());
        hash[12..20].copy_from_slice(&timestamp.to_be_bytes());
        hash[20..28].copy_from_slice(&mime);

        let cid_type = CidType::InlineMetadata;
        // Size field is 0 for inline metadata (the data is in the hash field).
        let size: u32 = 0;
        let checksum = compute_checksum(&hash, size, &cid_type);
        let tag = cid_type.encode(checksum);
        let size_and_tag_u32 = (size << TAG_BITS) | tag as u32;

        ContentId {
            hash,
            size_and_tag: size_and_tag_u32.to_be_bytes(),
        }
    }

    /// Parse an inline metadata CID, returning (total_size, chunk_count, timestamp, mime).
    pub fn parse_inline_metadata(&self) -> Result<(u64, u32, u64, [u8; 8]), ContentError> {
        if self.cid_type() != CidType::InlineMetadata {
            return Err(ContentError::NotInlineMetadata);
        }

        let total_size = u64::from_be_bytes(self.hash[0..8].try_into().unwrap());
        let chunk_count = u32::from_be_bytes(self.hash[8..12].try_into().unwrap());
        let timestamp = u64::from_be_bytes(self.hash[12..20].try_into().unwrap());
        let mut mime = [0u8; 8];
        mime.copy_from_slice(&self.hash[20..28]);

        Ok((total_size, chunk_count, timestamp, mime))
    }

    /// Serialize to a 32-byte array.
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..CONTENT_HASH_LEN].copy_from_slice(&self.hash);
        bytes[CONTENT_HASH_LEN..].copy_from_slice(&self.size_and_tag);
        bytes
    }

    /// Deserialize from a 32-byte array.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut hash = [0u8; CONTENT_HASH_LEN];
        hash.copy_from_slice(&bytes[..CONTENT_HASH_LEN]);
        let mut size_and_tag = [0u8; 4];
        size_and_tag.copy_from_slice(&bytes[CONTENT_HASH_LEN..]);
        ContentId { hash, size_and_tag }
    }
}

impl std::fmt::Display for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let cid_type = self.cid_type();
        let size = self.payload_size();
        write!(f, "{} {:?} {}B", hex_prefix(&self.hash), cid_type, size,)
    }
}

impl std::fmt::Debug for ContentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "ContentId({}, {}B, {:?})",
            hex_prefix(&self.hash),
            self.payload_size(),
            self.cid_type(),
        )
    }
}

/// Format the first 4 bytes of a hash as hex for debug display.
fn hex_prefix(hash: &[u8; CONTENT_HASH_LEN]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}...",
        hash[0], hash[1], hash[2], hash[3]
    )
}

// ---------------------------------------------------------------------------
// CidType enum with unary tag encode/decode
// ---------------------------------------------------------------------------

/// The type of a [`ContentId`], encoded as a unary prefix in the 12-bit tag field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CidType {
    /// Leaf data blob (depth 0). Tag prefix: `0`.
    Blob,
    /// Interior bundle node at the given depth (1--7). Tag prefix: `depth`
    /// leading 1-bits then `0`.
    Bundle(u8),
    /// Inline metadata (hash field repurposed). Tag prefix: `1111_1111_0`.
    InlineMetadata,
    /// Reserved type A. Tag prefix: `1111_1111_10`.
    ReservedA,
    /// Reserved type B. Tag prefix: `1111_1111_110`.
    ReservedB,
    /// Reserved type C. Tag prefix: `1111_1111_1110`.
    ReservedC,
    /// Reserved type D. Tag prefix: `1111_1111_1111`.
    ReservedD,
}

impl CidType {
    /// Return the tree depth for this CID type.
    ///
    /// - `Blob` => 0
    /// - `Bundle(d)` => d
    /// - All others => 0
    pub fn depth(&self) -> u8 {
        match self {
            CidType::Blob => 0,
            CidType::Bundle(d) => *d,
            _ => 0,
        }
    }

    /// Return the number of prefix bits consumed by the unary encoding.
    fn prefix_len(&self) -> u32 {
        match self {
            CidType::Blob => 1, // "0"
            CidType::Bundle(d) => {
                // d leading 1-bits + terminating 0 = d+1
                u32::from(*d) + 1
            }
            CidType::InlineMetadata => 9, // "1111_1111_0"
            CidType::ReservedA => 10,     // "1111_1111_10"
            CidType::ReservedB => 11,     // "1111_1111_110"
            CidType::ReservedC => 12,     // "1111_1111_1110"
            CidType::ReservedD => 12,     // "1111_1111_1111" (all bits used)
        }
    }

    /// Number of bits available for the checksum.
    pub fn checksum_bits(&self) -> u32 {
        TAG_BITS - self.prefix_len()
    }

    /// Encode this CID type and a checksum into a 12-bit tag value.
    ///
    /// The checksum is truncated to fit the available bits.
    pub fn encode(&self, checksum: u16) -> u16 {
        let prefix_len = self.prefix_len();
        let cksum_bits = TAG_BITS - prefix_len;

        // Build the prefix in the top bits of a 12-bit field.
        let prefix: u16 = match self {
            CidType::Blob => {
                // bit 11 = 0 => prefix is just 0 at the top
                0
            }
            CidType::Bundle(d) => {
                // d leading 1-bits at top, then a 0
                // e.g. depth=1: "10" at top => 0b10_0000_0000_00 shifted
                let ones = (1u16 << *d) - 1; // d bits of 1
                ones << (TAG_BITS as u16 - *d as u16)
                // The 0 terminator is implicit (next bit is 0)
            }
            CidType::InlineMetadata => {
                // "1111_1111_0" = 8 ones at top, then 0
                0xFF << (TAG_BITS as u16 - 8)
                // = 0xFF0 >> ... wait, let me be precise:
                // 8 ones shifted to top of 12 bits = 0b1111_1111_0000 = 0xFF0
            }
            CidType::ReservedA => {
                // "1111_1111_10" = 9 ones at top, then 0
                // 9 ones = 0b1_1111_1111 = 0x1FF
                0x1FF << (TAG_BITS as u16 - 9)
                // = 0xFF8
            }
            CidType::ReservedB => {
                // "1111_1111_110" = 10 ones, then 0
                0x3FF << (TAG_BITS as u16 - 10)
                // = 0xFFC
            }
            CidType::ReservedC => {
                // "1111_1111_1110" = 11 ones, then 0
                0x7FF << (TAG_BITS as u16 - 11)
                // = 0xFFE
            }
            CidType::ReservedD => {
                // "1111_1111_1111" = all 12 bits set
                0xFFF
            }
        };

        if cksum_bits == 0 {
            return prefix;
        }

        // Mask the checksum to fit
        let cksum_mask = (1u16 << cksum_bits) - 1;
        let masked_checksum = checksum & cksum_mask;

        prefix | masked_checksum
    }

    /// Decode a 12-bit tag value into a `CidType` and the embedded checksum.
    pub fn decode(tag: u16) -> (CidType, u16) {
        // Shift the 12-bit tag to the top of a u16 so we can use leading_zeros
        // to count unary 1-bits.
        let shifted = tag << (16 - TAG_BITS);
        let leading_ones = (!shifted).leading_zeros();

        match leading_ones {
            0 => {
                // Bit 11 = 0 => Blob, 11-bit checksum
                let checksum = tag & 0x7FF;
                (CidType::Blob, checksum)
            }
            d @ 1..=7 => {
                // Bundle(d): d leading 1-bits + terminating 0
                let cksum_bits = TAG_BITS - (d + 1);
                let checksum = if cksum_bits > 0 {
                    tag & ((1u16 << cksum_bits) - 1)
                } else {
                    0
                };
                (CidType::Bundle(d as u8), checksum)
            }
            8 => {
                // 1111_1111_0xxx => InlineMetadata, 3-bit checksum
                (CidType::InlineMetadata, tag & 0x7)
            }
            9 => {
                // 1111_1111_10xx => ReservedA, 2-bit checksum
                (CidType::ReservedA, tag & 0x3)
            }
            10 => {
                // 1111_1111_110x => ReservedB, 1-bit checksum
                (CidType::ReservedB, tag & 0x1)
            }
            11 => {
                // 1111_1111_1110 => ReservedC, 0-bit checksum
                (CidType::ReservedC, 0)
            }
            _ => {
                // 1111_1111_1111 => ReservedD, 0-bit checksum
                (CidType::ReservedD, 0)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Checksum computation
// ---------------------------------------------------------------------------

/// Compute a checksum over the CID's hash, size, and type, truncated to fit
/// the available checksum bits for the given CID type.
pub fn compute_checksum(hash: &[u8; CONTENT_HASH_LEN], size: u32, cid_type: &CidType) -> u16 {
    use harmony_crypto::hash::full_hash;

    let cksum_bits = cid_type.checksum_bits();
    if cksum_bits == 0 {
        return 0;
    }

    let mut input = [0u8; CONTENT_HASH_LEN + 4 + 1];
    input[..CONTENT_HASH_LEN].copy_from_slice(hash);
    input[CONTENT_HASH_LEN..CONTENT_HASH_LEN + 4].copy_from_slice(&size.to_be_bytes());
    input[CONTENT_HASH_LEN + 4] = cid_type.depth();
    let digest = full_hash(&input);

    let raw = u16::from_be_bytes([digest[0], digest[1]]);
    raw & ((1u16 << cksum_bits) - 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Task 2: ContentId struct
    // -----------------------------------------------------------------------

    #[test]
    fn content_id_is_32_bytes() {
        assert_eq!(std::mem::size_of::<ContentId>(), 32);
    }

    #[test]
    fn content_id_payload_size_and_tag() {
        let mut cid = ContentId {
            hash: [0u8; CONTENT_HASH_LEN],
            size_and_tag: [0; 4],
        };
        // Set size = 1000, tag = 0x123
        let packed: u32 = (1000 << TAG_BITS) | 0x123;
        cid.size_and_tag = packed.to_be_bytes();

        assert_eq!(cid.payload_size(), 1000);
        assert_eq!(cid.tag(), 0x123);
    }

    #[test]
    fn content_id_debug_format() {
        let cid = ContentId {
            hash: [0xAB; CONTENT_HASH_LEN],
            size_and_tag: ((512u32 << TAG_BITS) | 0).to_be_bytes(),
        };
        let debug = format!("{:?}", cid);
        assert!(debug.contains("ContentId("));
        assert!(debug.contains("abababab..."));
        assert!(debug.contains("512B"));
        assert!(debug.contains("Blob"));
    }

    // -----------------------------------------------------------------------
    // Task 3: CidType encode/decode
    // -----------------------------------------------------------------------

    #[test]
    fn blob_tag_round_trip() {
        let max_checksum: u16 = 0x7FF; // 11 bits
        let tag = CidType::Blob.encode(max_checksum);
        let (decoded_type, decoded_checksum) = CidType::decode(tag);
        assert_eq!(decoded_type, CidType::Blob);
        assert_eq!(decoded_checksum, max_checksum);
    }

    #[test]
    fn bundle_l1_tag_round_trip() {
        let max_checksum: u16 = 0x3FF; // 10 bits
        let tag = CidType::Bundle(1).encode(max_checksum);
        let (decoded_type, decoded_checksum) = CidType::decode(tag);
        assert_eq!(decoded_type, CidType::Bundle(1));
        assert_eq!(decoded_checksum, max_checksum);
    }

    #[test]
    fn bundle_l7_tag_round_trip() {
        let max_checksum: u16 = 0xF; // 4 bits
        let tag = CidType::Bundle(7).encode(max_checksum);
        let (decoded_type, decoded_checksum) = CidType::decode(tag);
        assert_eq!(decoded_type, CidType::Bundle(7));
        assert_eq!(decoded_checksum, max_checksum);
    }

    #[test]
    fn inline_metadata_tag_round_trip() {
        let max_checksum: u16 = 0x7; // 3 bits
        let tag = CidType::InlineMetadata.encode(max_checksum);
        let (decoded_type, decoded_checksum) = CidType::decode(tag);
        assert_eq!(decoded_type, CidType::InlineMetadata);
        assert_eq!(decoded_checksum, max_checksum);
    }

    #[test]
    fn all_bundle_depths_round_trip() {
        for depth in 1..=7u8 {
            let cid_type = CidType::Bundle(depth);
            let cksum_bits = cid_type.checksum_bits();
            let max_checksum = if cksum_bits > 0 {
                (1u16 << cksum_bits) - 1
            } else {
                0
            };
            let tag = cid_type.encode(max_checksum);
            let (decoded_type, decoded_checksum) = CidType::decode(tag);
            assert_eq!(
                decoded_type, cid_type,
                "round-trip failed for Bundle({})",
                depth
            );
            assert_eq!(
                decoded_checksum, max_checksum,
                "checksum round-trip failed for Bundle({})",
                depth
            );
        }
    }

    #[test]
    fn depth_returns_correct_values() {
        assert_eq!(CidType::Blob.depth(), 0);
        assert_eq!(CidType::Bundle(1).depth(), 1);
        assert_eq!(CidType::Bundle(5).depth(), 5);
        assert_eq!(CidType::Bundle(7).depth(), 7);
        assert_eq!(CidType::InlineMetadata.depth(), 0);
        assert_eq!(CidType::ReservedA.depth(), 0);
        assert_eq!(CidType::ReservedB.depth(), 0);
        assert_eq!(CidType::ReservedC.depth(), 0);
        assert_eq!(CidType::ReservedD.depth(), 0);
    }

    #[test]
    fn reserved_types_round_trip() {
        // ReservedA: 2-bit checksum
        let tag = CidType::ReservedA.encode(0x3);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::ReservedA);
        assert_eq!(c, 0x3);

        // ReservedB: 1-bit checksum
        let tag = CidType::ReservedB.encode(0x1);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::ReservedB);
        assert_eq!(c, 0x1);

        // ReservedC: 0-bit checksum
        let tag = CidType::ReservedC.encode(0);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::ReservedC);
        assert_eq!(c, 0);

        // ReservedD: 0-bit checksum
        let tag = CidType::ReservedD.encode(0);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::ReservedD);
        assert_eq!(c, 0);
    }

    #[test]
    fn blob_zero_checksum() {
        let tag = CidType::Blob.encode(0);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::Blob);
        assert_eq!(c, 0);
        assert_eq!(tag, 0);
    }

    #[test]
    fn checksum_bits_are_correct() {
        assert_eq!(CidType::Blob.checksum_bits(), 11);
        assert_eq!(CidType::Bundle(1).checksum_bits(), 10);
        assert_eq!(CidType::Bundle(2).checksum_bits(), 9);
        assert_eq!(CidType::Bundle(3).checksum_bits(), 8);
        assert_eq!(CidType::Bundle(4).checksum_bits(), 7);
        assert_eq!(CidType::Bundle(5).checksum_bits(), 6);
        assert_eq!(CidType::Bundle(6).checksum_bits(), 5);
        assert_eq!(CidType::Bundle(7).checksum_bits(), 4);
        assert_eq!(CidType::InlineMetadata.checksum_bits(), 3);
        assert_eq!(CidType::ReservedA.checksum_bits(), 2);
        assert_eq!(CidType::ReservedB.checksum_bits(), 1);
        assert_eq!(CidType::ReservedC.checksum_bits(), 0);
        assert_eq!(CidType::ReservedD.checksum_bits(), 0);
    }

    // -----------------------------------------------------------------------
    // Task 4: Checksum computation
    // -----------------------------------------------------------------------

    #[test]
    fn compute_checksum_deterministic() {
        let hash = [0xAA; CONTENT_HASH_LEN];
        let size = 1024u32;
        let cid_type = CidType::Blob;

        let c1 = compute_checksum(&hash, size, &cid_type);
        let c2 = compute_checksum(&hash, size, &cid_type);
        assert_eq!(c1, c2);
    }

    #[test]
    fn compute_checksum_varies_with_input() {
        let hash_a = [0xAA; CONTENT_HASH_LEN];
        let hash_b = [0xBB; CONTENT_HASH_LEN];
        let size = 1024u32;
        let cid_type = CidType::Blob;

        let ca = compute_checksum(&hash_a, size, &cid_type);
        let cb = compute_checksum(&hash_b, size, &cid_type);
        // Different inputs should (with overwhelming probability) yield
        // different checksums.
        assert_ne!(ca, cb);
    }

    #[test]
    fn checksum_fits_within_available_bits() {
        let hash = [0x42; CONTENT_HASH_LEN];
        let size = 999u32;

        for depth in 0..=7u8 {
            let cid_type = if depth == 0 {
                CidType::Blob
            } else {
                CidType::Bundle(depth)
            };
            let cksum_bits = cid_type.checksum_bits();
            let max_val = if cksum_bits > 0 {
                (1u16 << cksum_bits) - 1
            } else {
                0
            };
            let checksum = compute_checksum(&hash, size, &cid_type);
            assert!(
                checksum <= max_val,
                "checksum {} exceeds max {} for depth {}",
                checksum,
                max_val,
                depth
            );
        }

        // Also check InlineMetadata (3-bit checksum)
        let checksum = compute_checksum(&hash, size, &CidType::InlineMetadata);
        assert!(checksum <= 0x7);
    }

    // -----------------------------------------------------------------------
    // Task 5: ContentId constructors — for_blob and for_bundle
    // -----------------------------------------------------------------------

    #[test]
    fn for_blob_basic() {
        let data = b"hello harmony content addressing";
        let cid = ContentId::for_blob(data).unwrap();
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn for_blob_hash_is_truncated_sha256() {
        let data = b"test data";
        let cid = ContentId::for_blob(data).unwrap();
        let full = harmony_crypto::hash::full_hash(data);
        assert_eq!(&cid.hash, &full[..CONTENT_HASH_LEN]);
    }

    #[test]
    fn for_blob_rejects_oversized() {
        let data = vec![0u8; MAX_PAYLOAD_SIZE + 1];
        let result = ContentId::for_blob(&data);
        assert!(result.is_err());
    }

    #[test]
    fn for_blob_empty_data() {
        let cid = ContentId::for_blob(b"").unwrap();
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn for_blob_max_size() {
        let data = vec![0xFFu8; MAX_PAYLOAD_SIZE];
        let cid = ContentId::for_blob(&data).unwrap();
        assert_eq!(cid.payload_size(), MAX_PAYLOAD_SIZE as u32);
    }

    #[test]
    fn for_bundle_basic() {
        // Create two blob CIDs, then bundle them
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let cid = ContentId::for_bundle(&bundle_bytes, &children).unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1)); // one level above blobs
        assert_eq!(cid.payload_size(), bundle_bytes.len() as u32);
    }

    #[test]
    fn for_bundle_depth_is_max_child_plus_one() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_children = [blob];
        let l1_bytes = children_to_bytes(&l1_children);
        let l1 = ContentId::for_bundle(&l1_bytes, &l1_children).unwrap();
        assert_eq!(l1.cid_type(), CidType::Bundle(1));

        let l2_children = [l1];
        let l2_bytes = children_to_bytes(&l2_children);
        let l2 = ContentId::for_bundle(&l2_bytes, &l2_children).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));
    }

    #[test]
    fn for_bundle_rejects_depth_overflow() {
        // Can't create a Bundle(8) — max is 7
        let blob = ContentId::for_blob(b"leaf").unwrap();
        // Build up to depth 7
        let mut current = blob;
        for _ in 0..7 {
            let children = [current];
            let bytes = children_to_bytes(&children);
            current = ContentId::for_bundle(&bytes, &children).unwrap();
        }
        assert_eq!(current.cid_type(), CidType::Bundle(7));

        // Trying to wrap a depth-7 bundle should fail
        let children = [current];
        let bytes = children_to_bytes(&children);
        let result = ContentId::for_bundle(&bytes, &children);
        assert!(result.is_err());
    }

    /// Helper: serialize CID array to bytes (the bundle payload).
    fn children_to_bytes(children: &[ContentId]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(children.len() * 32);
        for cid in children {
            bytes.extend_from_slice(&cid.to_bytes());
        }
        bytes
    }

    // -----------------------------------------------------------------------
    // Task 6: Inline Metadata CID
    // -----------------------------------------------------------------------

    #[test]
    fn inline_metadata_round_trip() {
        let meta = ContentId::inline_metadata(
            100_000_000,   // 100MB total size
            100,           // 100 chunks
            1709337600000, // timestamp
            *b"text/pln",  // MIME type (8 bytes)
        );
        assert_eq!(meta.cid_type(), CidType::InlineMetadata);

        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, 100_000_000);
        assert_eq!(chunk_count, 100);
        assert_eq!(timestamp, 1709337600000);
        assert_eq!(&mime, b"text/pln");
    }

    #[test]
    fn inline_metadata_zero_values() {
        let meta = ContentId::inline_metadata(0, 0, 0, [0u8; 8]);
        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, 0);
        assert_eq!(chunk_count, 0);
        assert_eq!(timestamp, 0);
        assert_eq!(mime, [0u8; 8]);
    }

    #[test]
    fn inline_metadata_max_file_size() {
        let meta = ContentId::inline_metadata(u64::MAX, u32::MAX, u64::MAX, [0xFF; 8]);
        let (total_size, chunk_count, timestamp, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(total_size, u64::MAX);
        assert_eq!(chunk_count, u32::MAX);
        assert_eq!(timestamp, u64::MAX);
        assert_eq!(mime, [0xFF; 8]);
    }

    #[test]
    fn parse_inline_metadata_rejects_non_metadata_cid() {
        let blob = ContentId::for_blob(b"not metadata").unwrap();
        assert!(blob.parse_inline_metadata().is_err());
    }

    // -----------------------------------------------------------------------
    // Task 7: Checksum verification, Display, serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn checksum_verification_passes_for_valid_cid() {
        let cid = ContentId::for_blob(b"valid data").unwrap();
        assert!(cid.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_fails_for_corrupted_cid() {
        let cid = ContentId::for_blob(b"valid data").unwrap();
        let mut bytes = cid.to_bytes();
        bytes[0] ^= 0xFF; // corrupt the hash
        let corrupted = ContentId::from_bytes(bytes);
        assert!(corrupted.verify_checksum().is_err());
    }

    #[test]
    fn checksum_verification_passes_for_bundle() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let children = [blob];
        let bytes = children_to_bytes(&children);
        let bundle = ContentId::for_bundle(&bytes, &children).unwrap();
        assert!(bundle.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_passes_for_inline_metadata() {
        let meta = ContentId::inline_metadata(1000, 1, 0, [0; 8]);
        assert!(meta.verify_checksum().is_ok());
    }

    #[test]
    fn serialization_round_trip() {
        let cid = ContentId::for_blob(b"round trip test").unwrap();
        let bytes = cid.to_bytes();
        let restored = ContentId::from_bytes(bytes);
        assert_eq!(cid, restored);
    }

    #[test]
    fn display_shows_hex_type_and_size() {
        let cid = ContentId::for_blob(b"display test").unwrap();
        let s = format!("{cid}");
        assert!(s.contains("Blob"));
        assert!(s.contains("12")); // data length
    }

    // -----------------------------------------------------------------------
    // Task 8: Canonical test vectors
    // -----------------------------------------------------------------------
    // These exist so other language implementations can verify byte-identical CID output.

    #[test]
    fn canonical_vector_empty_blob() {
        let cid = ContentId::for_blob(b"").unwrap();
        let bytes = cid.to_bytes();
        // SHA-256("")[:28] = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b
        assert_eq!(
            hex::encode(&bytes[..CONTENT_HASH_LEN]),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b",
        );
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Blob);
        // Full 32-byte canonical hex (hash + packed size/tag/checksum)
        assert_eq!(
            hex::encode(bytes),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b0000026d",
        );
    }

    #[test]
    fn canonical_vector_hello_blob() {
        let cid = ContentId::for_blob(b"hello").unwrap();
        let bytes = cid.to_bytes();
        // SHA-256("hello")[:28] = 2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362
        assert_eq!(
            hex::encode(&bytes[..CONTENT_HASH_LEN]),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362",
        );
        assert_eq!(cid.payload_size(), 5);
        assert_eq!(cid.cid_type(), CidType::Blob);
        // Full 32-byte canonical hex
        assert_eq!(
            hex::encode(bytes),
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e7304336200005156",
        );
    }

    #[test]
    fn canonical_vector_bundle_of_two_blobs() {
        let blob_a = ContentId::for_blob(b"aaa").unwrap();
        let blob_b = ContentId::for_blob(b"bbb").unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let bundle = ContentId::for_bundle(&bundle_bytes, &children).unwrap();
        assert_eq!(bundle.payload_size(), 64); // 2 x 32 bytes
        assert_eq!(bundle.cid_type(), CidType::Bundle(1));
        // The hash is SHA-256 of the 64-byte bundle payload
        let full = harmony_crypto::hash::full_hash(&bundle_bytes);
        assert_eq!(&bundle.hash, &full[..CONTENT_HASH_LEN]);
        // Full 32-byte canonical hex
        assert_eq!(
            hex::encode(bundle.to_bytes()),
            "5c9f61c811d592b287972b0fa001f1d5d286b0f49c25e749b0eb83cd000409ca",
        );
    }

}

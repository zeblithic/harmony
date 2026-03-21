use crate::error::ContentError;
use alloc::{format, string::String};
use serde::{Deserialize, Serialize};

/// Length of the truncated SHA-256 content hash in bytes.
pub const CONTENT_HASH_LEN: usize = 28;

/// Maximum payload size expressible in 20 bits.
pub const MAX_PAYLOAD_SIZE: usize = 0xF_FFFF; // 1,048,575 bytes

/// Number of bits used for the type tag + checksum field.
pub const TAG_BITS: u32 = 12;

/// Bitmask for the 12-bit tag field (lower 12 bits of the last u32).
pub const TAG_MASK: u32 = 0xFFF;

/// Content class derived from the two leading classification bits of a CID.
///
/// The `(encrypted, ephemeral)` bits in `ContentFlags` (byte 0 of every CID)
/// define four content classes with distinct storage and publishing policies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentClass {
    /// `(false, false)` — Valuable public information. Disk-backed, LFU-managed.
    PublicDurable,
    /// `(false, true)` — Disposable-first content. Memory-only, evict first.
    PublicEphemeral,
    /// `(true, false)` — Content with intentional relationship. Configurable per-device.
    EncryptedDurable,
    /// `(true, true)` — Maximum privacy. Never persists, never enters Zenoh.
    EncryptedEphemeral,
}

impl ContentClass {
    /// Eviction priority: higher values are evicted first under pressure.
    ///
    /// Two tiers: ephemeral (evict first) and durable (evict last, frequency
    /// breaks ties). Both durable classes share priority 0 so EncryptedDurable
    /// can compete fairly against PublicDurable on frequency — otherwise
    /// `encrypted_durable_persist=true` would be a no-op for transit when
    /// probation fills with PublicDurable.
    /// EncryptedEphemeral never enters the cache, so its priority is irrelevant.
    pub fn eviction_priority(self) -> u8 {
        match self {
            ContentClass::PublicEphemeral => 2,
            ContentClass::PublicDurable | ContentClass::EncryptedDurable => 0,
            ContentClass::EncryptedEphemeral => u8::MAX, // unreachable in cache
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ContentFlags {
    pub encrypted: bool,
    pub ephemeral: bool,
    pub alt_hash: bool,
}

impl ContentFlags {
    pub fn to_bits(self) -> u8 {
        let mut bits = 0u8;
        if self.encrypted {
            bits |= 0x80;
        }
        if self.ephemeral {
            bits |= 0x40;
        }
        if self.alt_hash {
            bits |= 0x20;
        }
        bits
    }

    pub fn from_bits(byte: u8) -> Self {
        ContentFlags {
            encrypted: byte & 0x80 != 0,
            ephemeral: byte & 0x40 != 0,
            alt_hash: byte & 0x20 != 0,
        }
    }
}

/// A 32-byte content identifier.
///
/// Layout (big-endian):
/// - Bytes 0--27: truncated content hash (221 effective bits after flag reservation)
///   - Top 3 bits of byte 0 encode [`ContentFlags`]:
///     bit 7 = encrypted, bit 6 = ephemeral, bit 5 = alt_hash
///   - `verify_hash` masks byte 0 with `0x1F` on both sides to exclude flags
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
    /// Extract the content flags from the top 3 bits of hash\[0\].
    pub fn flags(&self) -> ContentFlags {
        ContentFlags::from_bits(self.hash[0])
    }

    /// Derive the content class from this CID's classification flags.
    pub fn content_class(&self) -> ContentClass {
        let flags = self.flags();
        match (flags.encrypted, flags.ephemeral) {
            (false, false) => ContentClass::PublicDurable,
            (false, true) => ContentClass::PublicEphemeral,
            (true, false) => ContentClass::EncryptedDurable,
            (true, true) => ContentClass::EncryptedEphemeral,
        }
    }

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

    /// Verify that this CID's hash matches the hash of the given data.
    ///
    /// Respects the `alt_hash` flag: uses SHA-224 when set, SHA-256 otherwise.
    /// The top 3 bits of hash\[0\] (flag bits) are masked out before comparison.
    pub fn verify_hash(&self, data: &[u8]) -> bool {
        let flags = self.flags();
        let digest = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(data)
        } else {
            let full = harmony_crypto::hash::full_hash(data);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };
        (self.hash[0] & 0x1F) == (digest[0] & 0x1F) && self.hash[1..] == digest[1..]
    }

    /// Create a CID for a raw data book (up to 1 MB).
    pub fn for_book(data: &[u8], flags: ContentFlags) -> Result<Self, ContentError> {
        if data.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }

        let mut hash = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(data)
        } else {
            let full = harmony_crypto::hash::full_hash(data);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };

        hash[0] = (hash[0] & 0x1F) | flags.to_bits();

        let size = data.len() as u32;
        let cid_type = CidType::Book;
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
    pub fn for_bundle(
        bundle_bytes: &[u8],
        children: &[ContentId],
        flags: ContentFlags,
    ) -> Result<Self, ContentError> {
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

        let mut hash = if flags.alt_hash {
            harmony_crypto::hash::sha224_hash(bundle_bytes)
        } else {
            let full = harmony_crypto::hash::full_hash(bundle_bytes);
            let mut trunc = [0u8; CONTENT_HASH_LEN];
            trunc.copy_from_slice(&full[..CONTENT_HASH_LEN]);
            trunc
        };

        hash[0] = (hash[0] & 0x1F) | flags.to_bits();

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

impl Serialize for ContentId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.to_bytes().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for ContentId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: [u8; 32] = Deserialize::deserialize(deserializer)?;
        Ok(ContentId::from_bytes(bytes))
    }
}

impl core::fmt::Display for ContentId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let cid_type = self.cid_type();
        let size = self.payload_size();
        write!(f, "{} {:?} {}B", hex_prefix(&self.hash), cid_type, size,)
    }
}

impl core::fmt::Debug for ContentId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
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
    /// Leaf data book (depth 0). Tag prefix: `0`.
    Book,
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
    /// - `Book` => 0
    /// - `Bundle(d)` => d
    /// - All others => 0
    pub fn depth(&self) -> u8 {
        match self {
            CidType::Book => 0,
            CidType::Bundle(d) => *d,
            _ => 0,
        }
    }

    /// Return a unique ordinal for each CID type variant.
    ///
    /// Used in checksum computation to ensure different types with the same
    /// depth (e.g. Book and InlineMetadata both have depth 0) produce
    /// different checksums.
    pub fn type_ordinal(&self) -> u8 {
        match self {
            CidType::Book => 0,
            CidType::Bundle(d) => *d, // 1..=7
            CidType::InlineMetadata => 8,
            CidType::ReservedA => 9,
            CidType::ReservedB => 10,
            CidType::ReservedC => 11,
            CidType::ReservedD => 12,
        }
    }

    /// Return the number of prefix bits consumed by the unary encoding.
    fn prefix_len(&self) -> u32 {
        match self {
            CidType::Book => 1, // "0"
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
            CidType::Book => {
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
                // Bit 11 = 0 => Book, 11-bit checksum
                let checksum = tag & 0x7FF;
                (CidType::Book, checksum)
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
    input[CONTENT_HASH_LEN + 4] = cid_type.type_ordinal();
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
        assert!(debug.contains("Book"));
    }

    // -----------------------------------------------------------------------
    // Task 3: CidType encode/decode
    // -----------------------------------------------------------------------

    #[test]
    fn book_tag_round_trip() {
        let max_checksum: u16 = 0x7FF; // 11 bits
        let tag = CidType::Book.encode(max_checksum);
        let (decoded_type, decoded_checksum) = CidType::decode(tag);
        assert_eq!(decoded_type, CidType::Book);
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
        assert_eq!(CidType::Book.depth(), 0);
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
    fn book_zero_checksum() {
        let tag = CidType::Book.encode(0);
        let (t, c) = CidType::decode(tag);
        assert_eq!(t, CidType::Book);
        assert_eq!(c, 0);
        assert_eq!(tag, 0);
    }

    #[test]
    fn checksum_bits_are_correct() {
        assert_eq!(CidType::Book.checksum_bits(), 11);
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
        let cid_type = CidType::Book;

        let c1 = compute_checksum(&hash, size, &cid_type);
        let c2 = compute_checksum(&hash, size, &cid_type);
        assert_eq!(c1, c2);
    }

    #[test]
    fn compute_checksum_varies_with_input() {
        let hash_a = [0xAA; CONTENT_HASH_LEN];
        let hash_b = [0xBB; CONTENT_HASH_LEN];
        let size = 1024u32;
        let cid_type = CidType::Book;

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
                CidType::Book
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
    // Task 5: ContentId constructors — for_book and for_bundle
    // -----------------------------------------------------------------------

    #[test]
    fn for_book_basic() {
        let data = b"hello harmony content addressing";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert_eq!(cid.cid_type(), CidType::Book);
    }

    #[test]
    fn for_book_hash_is_truncated_sha256() {
        let data = b"test data";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let full = harmony_crypto::hash::full_hash(data);
        // Top 3 bits are cleared by flag masking
        assert_eq!(cid.hash[0] & 0x1F, full[0] & 0x1F);
        assert_eq!(&cid.hash[1..], &full[1..CONTENT_HASH_LEN]);
    }

    #[test]
    fn verify_hash_matches_for_book() {
        let data = b"verify this";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert!(cid.verify_hash(data));
        assert!(!cid.verify_hash(b"wrong data"));
    }

    #[test]
    fn verify_hash_matches_for_bundle() {
        // Bundle CID has a different type tag than book, but verify_hash
        // should still pass because it only checks the hash portion.
        let blob_a = ContentId::for_book(b"child-a", ContentFlags::default()).unwrap();
        let blob_b = ContentId::for_book(b"child-b", ContentFlags::default()).unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes: Vec<u8> = children.iter().flat_map(|c| c.to_bytes()).collect();
        let bundle_cid =
            ContentId::for_bundle(&bundle_bytes, &children, ContentFlags::default()).unwrap();

        assert!(bundle_cid.verify_hash(&bundle_bytes));
        assert!(!bundle_cid.verify_hash(b"wrong data"));

        // Confirm it's actually a bundle type, not book.
        assert_ne!(bundle_cid.cid_type(), CidType::Book);
    }

    #[test]
    fn for_book_rejects_oversized() {
        let data = vec![0u8; MAX_PAYLOAD_SIZE + 1];
        let result = ContentId::for_book(&data, ContentFlags::default());
        assert!(result.is_err());
    }

    #[test]
    fn for_book_empty_data() {
        let cid = ContentId::for_book(b"", ContentFlags::default()).unwrap();
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Book);
    }

    #[test]
    fn for_book_max_size() {
        let data = vec![0xFFu8; MAX_PAYLOAD_SIZE];
        let cid = ContentId::for_book(&data, ContentFlags::default()).unwrap();
        assert_eq!(cid.payload_size(), MAX_PAYLOAD_SIZE as u32);
    }

    #[test]
    fn for_bundle_basic() {
        // Create two book CIDs, then bundle them
        let blob_a = ContentId::for_book(b"chunk a", ContentFlags::default()).unwrap();
        let blob_b = ContentId::for_book(b"chunk b", ContentFlags::default()).unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let cid = ContentId::for_bundle(&bundle_bytes, &children, ContentFlags::default()).unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1)); // one level above blobs
        assert_eq!(cid.payload_size(), bundle_bytes.len() as u32);
    }

    #[test]
    fn for_bundle_depth_is_max_child_plus_one() {
        let book = ContentId::for_book(b"leaf", ContentFlags::default()).unwrap();
        let l1_children = [book];
        let l1_bytes = children_to_bytes(&l1_children);
        let l1 = ContentId::for_bundle(&l1_bytes, &l1_children, ContentFlags::default()).unwrap();
        assert_eq!(l1.cid_type(), CidType::Bundle(1));

        let l2_children = [l1];
        let l2_bytes = children_to_bytes(&l2_children);
        let l2 = ContentId::for_bundle(&l2_bytes, &l2_children, ContentFlags::default()).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));
    }

    #[test]
    fn for_bundle_rejects_depth_overflow() {
        // Can't create a Bundle(8) — max is 7
        let book = ContentId::for_book(b"leaf", ContentFlags::default()).unwrap();
        // Build up to depth 7
        let mut current = book;
        for _ in 0..7 {
            let children = [current];
            let bytes = children_to_bytes(&children);
            current = ContentId::for_bundle(&bytes, &children, ContentFlags::default()).unwrap();
        }
        assert_eq!(current.cid_type(), CidType::Bundle(7));

        // Trying to wrap a depth-7 bundle should fail
        let children = [current];
        let bytes = children_to_bytes(&children);
        let result = ContentId::for_bundle(&bytes, &children, ContentFlags::default());
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
    // ContentFlags tests
    // -----------------------------------------------------------------------

    #[test]
    fn content_flags_default_is_cleartext_durable_sha256() {
        let f = ContentFlags::default();
        assert!(!f.encrypted);
        assert!(!f.ephemeral);
        assert!(!f.alt_hash);
        assert_eq!(f.to_bits(), 0x00);
    }

    #[test]
    fn content_flags_to_bits_round_trip() {
        for enc in [false, true] {
            for eph in [false, true] {
                for alt in [false, true] {
                    let flags = ContentFlags {
                        encrypted: enc,
                        ephemeral: eph,
                        alt_hash: alt,
                    };
                    let bits = flags.to_bits();
                    let restored = ContentFlags::from_bits(bits);
                    assert_eq!(
                        flags, restored,
                        "round trip failed for enc={enc}, eph={eph}, alt={alt}"
                    );
                }
            }
        }
    }

    #[test]
    fn content_flags_bits_are_top_3() {
        let enc = ContentFlags {
            encrypted: true,
            ephemeral: false,
            alt_hash: false,
        };
        assert_eq!(enc.to_bits(), 0x80);
        let eph = ContentFlags {
            encrypted: false,
            ephemeral: true,
            alt_hash: false,
        };
        assert_eq!(eph.to_bits(), 0x40);
        let alt = ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: true,
        };
        assert_eq!(alt.to_bits(), 0x20);
        let all = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: true,
        };
        assert_eq!(all.to_bits(), 0xE0);
    }

    // -----------------------------------------------------------------------
    // flags() accessor tests
    // -----------------------------------------------------------------------

    #[test]
    fn flags_accessor_reads_top_3_bits() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"flagged", flags).unwrap();
        let read = cid.flags();
        assert!(read.encrypted);
        assert!(read.ephemeral);
        assert!(!read.alt_hash);
    }

    #[test]
    fn flags_accessor_no_flags() {
        let cid = ContentId::for_book(b"plain", ContentFlags::default()).unwrap();
        let read = cid.flags();
        assert!(!read.encrypted);
        assert!(!read.ephemeral);
        assert!(!read.alt_hash);
    }

    // -----------------------------------------------------------------------
    // for_book with flags tests
    // -----------------------------------------------------------------------

    #[test]
    fn for_book_with_default_flags_matches_sha256() {
        let data = b"sha256 test";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert!(!cid.flags().alt_hash);
        assert!(cid.verify_hash(data));
    }

    #[test]
    fn for_book_with_alt_hash_uses_sha224() {
        let data = b"sha224 test";
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: true,
        };
        let cid = ContentId::for_book(data, flags).unwrap();
        assert!(cid.flags().alt_hash);
        assert!(cid.verify_hash(data));
        // Verify hash bytes match SHA-224
        let expected = harmony_crypto::hash::sha224_hash(data);
        assert_eq!(cid.hash[0] & 0x1F, expected[0] & 0x1F);
        assert_eq!(&cid.hash[1..], &expected[1..]);
    }

    #[test]
    fn for_book_same_data_different_flags_different_cid() {
        let data = b"same data";
        let cid_default = ContentId::for_book(data, ContentFlags::default()).unwrap();
        let cid_enc = ContentId::for_book(
            data,
            ContentFlags {
                encrypted: true,
                ..ContentFlags::default()
            },
        )
        .unwrap();
        let cid_alt = ContentId::for_book(
            data,
            ContentFlags {
                alt_hash: true,
                ..ContentFlags::default()
            },
        )
        .unwrap();
        assert_ne!(cid_default, cid_enc);
        assert_ne!(cid_default, cid_alt);
        assert_ne!(cid_enc, cid_alt);
    }

    // -----------------------------------------------------------------------
    // for_bundle with flags test
    // -----------------------------------------------------------------------

    #[test]
    fn for_bundle_with_flags() {
        let book = ContentId::for_book(b"child", ContentFlags::default()).unwrap();
        let children = [book];
        let bytes = children_to_bytes(&children);

        let flags = ContentFlags {
            encrypted: true,
            ephemeral: false,
            alt_hash: false,
        };
        let bundle = ContentId::for_bundle(&bytes, &children, flags).unwrap();
        assert!(bundle.flags().encrypted);
        assert!(!bundle.flags().ephemeral);
        assert!(bundle.verify_hash(&bytes));
    }

    // -----------------------------------------------------------------------
    // verify_hash tests
    // -----------------------------------------------------------------------

    #[test]
    fn verify_hash_sha256_default() {
        let data = b"sha256 verify";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert!(cid.verify_hash(data));
        assert!(!cid.verify_hash(b"wrong"));
    }

    #[test]
    fn verify_hash_sha224_alt() {
        let data = b"sha224 verify";
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: true,
        };
        let cid = ContentId::for_book(data, flags).unwrap();
        assert!(cid.verify_hash(data));
        assert!(!cid.verify_hash(b"wrong"));
    }

    #[test]
    fn verify_hash_ignores_flag_bits() {
        // Create CID with encrypted flag, then verify hash still works
        // (flag bits are masked out during comparison).
        let data = b"flag bits test";
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(data, flags).unwrap();
        assert!(cid.verify_hash(data));
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
        let book = ContentId::for_book(b"not metadata", ContentFlags::default()).unwrap();
        assert!(book.parse_inline_metadata().is_err());
    }

    // -----------------------------------------------------------------------
    // Task 7: Checksum verification, Display, serialization round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn checksum_verification_passes_for_valid_cid() {
        let cid = ContentId::for_book(b"valid data", ContentFlags::default()).unwrap();
        assert!(cid.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_fails_for_corrupted_cid() {
        let cid = ContentId::for_book(b"valid data", ContentFlags::default()).unwrap();
        let mut bytes = cid.to_bytes();
        bytes[0] ^= 0xFF; // corrupt the hash
        let corrupted = ContentId::from_bytes(bytes);
        assert!(corrupted.verify_checksum().is_err());
    }

    #[test]
    fn checksum_verification_passes_for_bundle() {
        let book = ContentId::for_book(b"leaf", ContentFlags::default()).unwrap();
        let children = [book];
        let bytes = children_to_bytes(&children);
        let bundle = ContentId::for_bundle(&bytes, &children, ContentFlags::default()).unwrap();
        assert!(bundle.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_verification_passes_for_inline_metadata() {
        let meta = ContentId::inline_metadata(1000, 1, 0, [0; 8]);
        assert!(meta.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_detects_type_corruption() {
        // A Book CID with its tag bits corrupted to InlineMetadata should
        // fail checksum verification, even though both types have depth 0.
        let book = ContentId::for_book(b"test", ContentFlags::default()).unwrap();
        let mut bytes = book.to_bytes();
        // Corrupt the tag: overwrite bottom 12 bits with an InlineMetadata tag.
        // InlineMetadata prefix = 0xFF0 (9 bits: 1111_1111_0), with 3-bit checksum.
        let packed = u32::from_be_bytes([bytes[28], bytes[29], bytes[30], bytes[31]]);
        let size_bits = packed & !TAG_MASK; // preserve the 20-bit size
        let fake_tag = CidType::InlineMetadata.encode(0); // InlineMetadata with checksum 0
        let corrupted_packed = size_bits | fake_tag as u32;
        bytes[28..32].copy_from_slice(&corrupted_packed.to_be_bytes());
        let corrupted = ContentId::from_bytes(bytes);
        assert_eq!(corrupted.cid_type(), CidType::InlineMetadata);
        assert!(
            corrupted.verify_checksum().is_err(),
            "type corruption (Book→InlineMetadata) should be detected by checksum"
        );
    }

    #[test]
    fn serialization_round_trip() {
        let cid = ContentId::for_book(b"round trip test", ContentFlags::default()).unwrap();
        let bytes = cid.to_bytes();
        let restored = ContentId::from_bytes(bytes);
        assert_eq!(cid, restored);
    }

    #[test]
    fn display_shows_hex_type_and_size() {
        let cid = ContentId::for_book(b"display test", ContentFlags::default()).unwrap();
        let s = format!("{cid}");
        assert!(s.contains("Book"));
        assert!(s.contains("12")); // data length
    }

    // -----------------------------------------------------------------------
    // Task 8: Canonical test vectors
    // -----------------------------------------------------------------------
    // These exist so other language implementations can verify byte-identical CID output.
    // hash[0] top 3 bits are cleared for default flags, changing the first byte
    // of the hash compared to raw SHA-256.

    #[test]
    fn canonical_vector_empty_blob() {
        let cid = ContentId::for_book(b"", ContentFlags::default()).unwrap();
        // SHA-256("")[:28] with top 3 bits cleared: e3 & 0x1F = 0x03
        assert_eq!(cid.hash[0], 0x03);
        assert_eq!(cid.payload_size(), 0);
        assert_eq!(cid.cid_type(), CidType::Book);
        // Full 32-byte canonical hex (independently verifiable by other implementations)
        assert_eq!(
            hex::encode(cid.to_bytes()),
            "03b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b000002ca",
        );
    }

    #[test]
    fn canonical_vector_hello_blob() {
        let cid = ContentId::for_book(b"hello", ContentFlags::default()).unwrap();
        // SHA-256("hello")[:28] with top 3 bits cleared: 2c & 0x1F = 0x0c
        assert_eq!(cid.hash[0], 0x0c);
        assert_eq!(cid.payload_size(), 5);
        assert_eq!(cid.cid_type(), CidType::Book);
        // Full 32-byte canonical hex (independently verifiable by other implementations)
        assert_eq!(
            hex::encode(cid.to_bytes()),
            "0cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e7304336200005020",
        );
    }

    #[test]
    fn canonical_vector_bundle_of_two_blobs() {
        let blob_a = ContentId::for_book(b"aaa", ContentFlags::default()).unwrap();
        let blob_b = ContentId::for_book(b"bbb", ContentFlags::default()).unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes = children_to_bytes(&children);
        let bundle =
            ContentId::for_bundle(&bundle_bytes, &children, ContentFlags::default()).unwrap();
        assert_eq!(bundle.payload_size(), 64); // 2 x 32 bytes
        assert_eq!(bundle.cid_type(), CidType::Bundle(1));
        assert_eq!(bundle.hash[0] & 0xE0, 0x00);
        assert!(bundle.verify_hash(&bundle_bytes));
        // Full 32-byte canonical hex (independently verifiable by other implementations).
        // Pin child CIDs so the bundle hash is fully determined:
        assert_eq!(
            hex::encode(blob_a.to_bytes()),
            "1834876dcfb05cb167a5c24953eba58c4ac89b1adf57f28f2f9d09af00003537",
        );
        assert_eq!(
            hex::encode(blob_b.to_bytes()),
            "1e744b9dc39389baf0c5a0660589b8402f3dbb49b89b3e75f2c93558000035db",
        );
        assert_eq!(
            hex::encode(bundle.to_bytes()),
            "05d0f2f83241a0222a2b84ac33993064e525a8b186fdf4b5784326b00004083c",
        );
    }

    // -----------------------------------------------------------------------
    // New canonical vectors with flags
    // -----------------------------------------------------------------------

    #[test]
    fn canonical_vector_hello_encrypted() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: false,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"hello", flags).unwrap();
        // SHA-256("hello")[0] = 0x2c, cleared = 0x0c, | 0x80 = 0x8c
        assert_eq!(cid.hash[0], 0x8c);
        assert!(cid.flags().encrypted);
        assert!(cid.verify_hash(b"hello"));
    }

    #[test]
    fn canonical_vector_hello_ephemeral() {
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"hello", flags).unwrap();
        // SHA-256("hello")[0] = 0x2c, cleared = 0x0c, | 0x40 = 0x4c
        assert_eq!(cid.hash[0], 0x4c);
        assert!(cid.flags().ephemeral);
        assert!(cid.verify_hash(b"hello"));
    }

    #[test]
    fn canonical_vector_hello_alt_hash() {
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: true,
        };
        let cid = ContentId::for_book(b"hello", flags).unwrap();
        assert!(cid.flags().alt_hash);
        assert!(cid.verify_hash(b"hello"));
        // Verify it uses SHA-224, not SHA-256
        let sha224 = harmony_crypto::hash::sha224_hash(b"hello");
        assert_eq!(cid.hash[0] & 0x1F, sha224[0] & 0x1F);
        assert_eq!(&cid.hash[1..], &sha224[1..]);
        // alt_hash bit set
        assert_eq!(cid.hash[0] & 0x20, 0x20);
    }

    #[test]
    fn canonical_vector_hello_private_ephemeral() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"hello", flags).unwrap();
        // SHA-256("hello")[0] = 0x2c, cleared = 0x0c, | 0xC0 = 0xcc
        assert_eq!(cid.hash[0], 0xcc);
        assert!(cid.flags().encrypted);
        assert!(cid.flags().ephemeral);
        assert!(cid.verify_hash(b"hello"));
    }

    #[test]
    fn canonical_vector_hello_all_flags() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: true,
        };
        let cid = ContentId::for_book(b"hello", flags).unwrap();
        assert_eq!(cid.hash[0] & 0xE0, 0xE0);
        assert!(cid.flags().encrypted);
        assert!(cid.flags().ephemeral);
        assert!(cid.flags().alt_hash);
        assert!(cid.verify_hash(b"hello"));
    }

    // -----------------------------------------------------------------------
    // ContentClass tests
    // -----------------------------------------------------------------------

    #[test]
    fn content_class_public_durable() {
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"test", flags).unwrap();
        assert_eq!(cid.content_class(), ContentClass::PublicDurable);
    }

    #[test]
    fn content_class_public_ephemeral() {
        let flags = ContentFlags {
            encrypted: false,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"test", flags).unwrap();
        assert_eq!(cid.content_class(), ContentClass::PublicEphemeral);
    }

    #[test]
    fn content_class_encrypted_durable() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: false,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"test", flags).unwrap();
        assert_eq!(cid.content_class(), ContentClass::EncryptedDurable);
    }

    #[test]
    fn content_class_encrypted_ephemeral() {
        let flags = ContentFlags {
            encrypted: true,
            ephemeral: true,
            alt_hash: false,
        };
        let cid = ContentId::for_book(b"test", flags).unwrap();
        assert_eq!(cid.content_class(), ContentClass::EncryptedEphemeral);
    }

    #[test]
    fn cross_flag_verify_hash_rejection() {
        // CID created with SHA-256 should fail verification if the alt_hash
        // bit is flipped (verify would use SHA-224, producing a different digest).
        let cid_256 = ContentId::for_book(b"hello", ContentFlags::default()).unwrap();
        let cid_224 = ContentId::for_book(
            b"hello",
            ContentFlags {
                alt_hash: true,
                ..ContentFlags::default()
            },
        )
        .unwrap();

        // Both verify against the same data with their own algorithm.
        assert!(cid_256.verify_hash(b"hello"));
        assert!(cid_224.verify_hash(b"hello"));

        // But they produce different CIDs (different hash bytes).
        assert_ne!(cid_256, cid_224);
        assert_ne!(cid_256.hash[1..], cid_224.hash[1..]);

        // Manually flip alt_hash bit on the SHA-256 CID — verify must fail
        // because the hash body is SHA-256 but verify now uses SHA-224.
        let mut corrupted = cid_256;
        corrupted.hash[0] |= 0x20; // set alt_hash bit
        assert!(!corrupted.verify_hash(b"hello"));
    }
}

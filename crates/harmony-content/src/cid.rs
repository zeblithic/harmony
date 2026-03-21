use crate::error::ContentError;
use alloc::{format, string::String};
use serde::{Deserialize, Serialize};

/// Length of the hash/payload portion of a ContentId in bytes.
pub const HASH_LEN: usize = 28;

/// Size of the header portion of a ContentId in bytes.
pub const HEADER_SIZE: usize = 4;

/// Maximum payload size for a leaf book (2^20 - 1 bytes).
pub const MAX_PAYLOAD_SIZE: usize = 0xF_FFFF; // 1,048,575 bytes

/// Maximum inline data length in bytes.
///
/// The 28-byte hash area stores: 1 byte overhead (top nibble = 4-bit XOR
/// checksum, bottom nibble reserved) + 27 bytes of data.
pub const MAX_INLINE_DATA: usize = 27;

/// Maximum depth for a bundle (depths 1-62 are bundles; 63 is stream).
pub const MAX_BUNDLE_DEPTH: u8 = 62;

/// Depth value reserved for streams.
pub const STREAM_DEPTH: u8 = 63;

/// Content class derived from the (encrypted, ephemeral) flag bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentClass {
    PublicDurable,
    PublicEphemeral,
    EncryptedDurable,
    EncryptedEphemeral,
}

impl ContentClass {
    pub fn eviction_priority(self) -> u8 {
        match self {
            ContentClass::PublicEphemeral => 2,
            ContentClass::PublicDurable | ContentClass::EncryptedDurable => 0,
            ContentClass::EncryptedEphemeral => u8::MAX,
        }
    }
}

/// Mode flags occupying the top 4 bits (bits 31-28) of the header.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ContentFlags {
    pub encrypted: bool,
    pub ephemeral: bool,
    pub sha224: bool,
    pub lsb_mode: bool,
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
        if self.sha224 {
            bits |= 0x20;
        }
        if self.lsb_mode {
            bits |= 0x10;
        }
        bits
    }

    pub fn from_bits(byte: u8) -> Self {
        ContentFlags {
            encrypted: byte & 0x80 != 0,
            ephemeral: byte & 0x40 != 0,
            sha224: byte & 0x20 != 0,
            lsb_mode: byte & 0x10 != 0,
        }
    }

    pub fn is_inline(&self) -> bool {
        self.sha224 && self.lsb_mode
    }
}

/// The logical type of a [`ContentId`], derived from depth and mode bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CidType {
    Book,
    Bundle(u8),
    Stream,
    InlineData,
}

impl CidType {
    pub fn depth(&self) -> u8 {
        match self {
            CidType::Book => 0,
            CidType::Bundle(d) => *d,
            CidType::Stream => STREAM_DEPTH,
            CidType::InlineData => 0,
        }
    }
}

/// A 32-byte content identifier.
///
/// Bytes 0-3: header (mode|depth|size|checksum). Bytes 4-31: hash or payload.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct ContentId {
    pub header: [u8; HEADER_SIZE],
    pub hash: [u8; HASH_LEN],
}

impl ContentId {
    fn header_u32(&self) -> u32 {
        u32::from_be_bytes(self.header)
    }

    fn mode_bits(&self) -> u8 {
        (self.header_u32() >> 28) as u8 & 0x0F
    }

    pub fn flags(&self) -> ContentFlags {
        ContentFlags::from_bits(self.mode_bits() << 4)
    }

    pub fn content_class(&self) -> ContentClass {
        let f = self.flags();
        match (f.encrypted, f.ephemeral) {
            (false, false) => ContentClass::PublicDurable,
            (false, true) => ContentClass::PublicEphemeral,
            (true, false) => ContentClass::EncryptedDurable,
            (true, true) => ContentClass::EncryptedEphemeral,
        }
    }

    pub fn depth(&self) -> u8 {
        ((self.header_u32() >> 22) & 0x3F) as u8
    }

    pub fn cid_type(&self) -> CidType {
        if self.is_inline() {
            return CidType::InlineData;
        }
        match self.depth() {
            0 => CidType::Book,
            STREAM_DEPTH => CidType::Stream,
            d => CidType::Bundle(d),
        }
    }

    pub fn raw_size_field(&self) -> u32 {
        (self.header_u32() >> 2) & 0xF_FFFF
    }

    pub fn payload_size(&self) -> u32 {
        let raw = self.raw_size_field();
        match self.depth() {
            0 | STREAM_DEPTH => raw,
            _ => {
                let full = decode_bundle_size(raw);
                if full > u32::MAX as u64 {
                    u32::MAX
                } else {
                    full as u32
                }
            }
        }
    }

    pub fn payload_size_bytes(&self) -> u64 {
        let raw = self.raw_size_field();
        match self.depth() {
            0 | STREAM_DEPTH => raw as u64,
            _ => decode_bundle_size(raw),
        }
    }

    pub fn is_inline(&self) -> bool {
        self.flags().is_inline()
    }

    pub fn is_sentinel(&self) -> bool {
        let bytes = self.to_bytes();
        compute_pairwise_xor(&bytes) == 0x03
    }

    pub fn verify_checksum(&self) -> Result<(), ContentError> {
        let bytes = self.to_bytes();
        let fold = compute_pairwise_xor(&bytes);
        // Non-sentinel CIDs are assembled so their pairwise XOR folds to 00.
        // Sentinel inline CIDs are assembled so their fold is 11 (checksum
        // XORed with 0x03). A non-inline CID with fold==11 is corrupted.
        //
        // - is_inline() → fold can be 0x00 (normal inline data) or 0x03
        //   (sentinel metadata)
        // - NOT is_inline() → fold must be 0x00; 0x03 means corruption
        if self.is_inline() {
            if fold != 0x00 && fold != 0x03 {
                return Err(ContentError::ChecksumMismatch);
            }
        } else if fold != 0x00 {
            return Err(ContentError::ChecksumMismatch);
        }
        Ok(())
    }

    pub fn verify_hash(&self, data: &[u8]) -> bool {
        if self.is_inline() {
            return false;
        }
        compute_hash(data, &self.flags()) == self.hash
    }

    pub fn for_book(data: &[u8], flags: ContentFlags) -> Result<Self, ContentError> {
        if flags.is_inline() {
            return Err(ContentError::InvalidFlags);
        }
        if data.len() > MAX_PAYLOAD_SIZE {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_PAYLOAD_SIZE,
            });
        }
        let hash = compute_hash(data, &flags);
        Ok(Self::assemble(flags, 0, data.len() as u32, hash, false))
    }

    pub fn for_bundle(
        bundle_bytes: &[u8],
        children: &[ContentId],
        flags: ContentFlags,
    ) -> Result<Self, ContentError> {
        if flags.is_inline() {
            return Err(ContentError::InvalidFlags);
        }
        let max_child_depth = children.iter().map(|c| c.depth()).max().unwrap_or(0);
        let bundle_depth = max_child_depth + 1;
        if bundle_depth > MAX_BUNDLE_DEPTH {
            return Err(ContentError::DepthViolation {
                child: max_child_depth,
                parent: MAX_BUNDLE_DEPTH,
            });
        }
        let hash = compute_hash(bundle_bytes, &flags);
        let raw_size = encode_bundle_size(bundle_bytes.len() as u64);
        Ok(Self::assemble(flags, bundle_depth, raw_size, hash, false))
    }

    pub fn for_stream(
        hash: [u8; HASH_LEN],
        flags: ContentFlags,
        chunk_index: u32,
    ) -> Result<Self, ContentError> {
        if flags.is_inline() {
            return Err(ContentError::InvalidFlags);
        }
        if chunk_index > 0xF_FFFF {
            return Err(ContentError::ChunkIndexTooLarge {
                index: chunk_index,
                max: 0xF_FFFF,
            });
        }
        Ok(Self::assemble(flags, STREAM_DEPTH, chunk_index, hash, false))
    }

    pub fn inline_data(data: &[u8], flags: ContentFlags) -> Result<Self, ContentError> {
        if data.len() > MAX_INLINE_DATA {
            return Err(ContentError::PayloadTooLarge {
                size: data.len(),
                max: MAX_INLINE_DATA,
            });
        }
        let mut flags = flags;
        flags.sha224 = true;
        flags.lsb_mode = true;

        let mut payload = [0u8; HASH_LEN];
        let mut xor4: u8 = 0;
        for &b in data {
            xor4 ^= (b >> 4) ^ (b & 0x0F);
        }
        payload[0] = (xor4 & 0x0F) << 4;
        payload[1..1 + data.len()].copy_from_slice(data);

        Ok(Self::assemble(flags, 0, data.len() as u32, payload, false))
    }

    pub fn extract_inline_data(&self) -> Result<alloc::vec::Vec<u8>, ContentError> {
        if !self.is_inline() || self.is_sentinel() {
            return Err(ContentError::NotInlineData);
        }
        let len = self.raw_size_field() as usize;
        if len > MAX_INLINE_DATA {
            return Err(ContentError::NotInlineData);
        }
        let data = &self.hash[1..1 + len];
        // Verify 4-bit XOR checksum stored in top nibble of hash[0].
        let stored_xor = (self.hash[0] >> 4) & 0x0F;
        let mut computed_xor: u8 = 0;
        for &b in data {
            computed_xor ^= (b >> 4) ^ (b & 0x0F);
        }
        computed_xor &= 0x0F;
        if stored_xor != computed_xor {
            return Err(ContentError::ChecksumMismatch);
        }
        Ok(data.to_vec())
    }

    pub fn inline_metadata(
        total_size: u64,
        chunk_count: u32,
        timestamp: u64,
        mime: [u8; 8],
    ) -> Self {
        let mut hash = [0u8; HASH_LEN];
        hash[0..8].copy_from_slice(&total_size.to_be_bytes());
        hash[8..12].copy_from_slice(&chunk_count.to_be_bytes());
        hash[12..20].copy_from_slice(&timestamp.to_be_bytes());
        hash[20..28].copy_from_slice(&mime);
        let flags = ContentFlags {
            sha224: true,
            lsb_mode: true,
            ..ContentFlags::default()
        };
        Self::assemble(flags, 0, 0, hash, true)
    }

    pub fn parse_inline_metadata(&self) -> Result<(u64, u32, u64, [u8; 8]), ContentError> {
        if !self.is_inline() || !self.is_sentinel() {
            return Err(ContentError::NotInlineData);
        }
        let total_size = u64::from_be_bytes(self.hash[0..8].try_into().unwrap());
        let chunk_count = u32::from_be_bytes(self.hash[8..12].try_into().unwrap());
        let timestamp = u64::from_be_bytes(self.hash[12..20].try_into().unwrap());
        let mut mime = [0u8; 8];
        mime.copy_from_slice(&self.hash[20..28]);
        Ok((total_size, chunk_count, timestamp, mime))
    }

    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        bytes[..HEADER_SIZE].copy_from_slice(&self.header);
        bytes[HEADER_SIZE..].copy_from_slice(&self.hash);
        bytes
    }

    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        let mut header = [0u8; HEADER_SIZE];
        header.copy_from_slice(&bytes[..HEADER_SIZE]);
        let mut hash = [0u8; HASH_LEN];
        hash.copy_from_slice(&bytes[HEADER_SIZE..]);
        ContentId { header, hash }
    }

    fn assemble(
        flags: ContentFlags,
        depth: u8,
        raw_size: u32,
        hash: [u8; HASH_LEN],
        sentinel: bool,
    ) -> Self {
        let mode = (flags.to_bits() >> 4) as u32;
        let header_no_cksum: u32 =
            (mode << 28) | ((depth as u32 & 0x3F) << 22) | ((raw_size & 0xF_FFFF) << 2);
        let mut bytes = [0u8; 32];
        bytes[..HEADER_SIZE].copy_from_slice(&header_no_cksum.to_be_bytes());
        bytes[HEADER_SIZE..].copy_from_slice(&hash);
        let fold = compute_pairwise_xor(&bytes);
        let cksum = if sentinel { fold ^ 0x03 } else { fold };
        let header_u32 = header_no_cksum | (cksum as u32 & 0x03);
        ContentId {
            header: header_u32.to_be_bytes(),
            hash,
        }
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
        write!(
            f,
            "{} {:?} {}B",
            hex_prefix(&self.hash),
            self.cid_type(),
            self.payload_size()
        )
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

fn hex_prefix(hash: &[u8; HASH_LEN]) -> String {
    format!(
        "{:02x}{:02x}{:02x}{:02x}...",
        hash[0], hash[1], hash[2], hash[3]
    )
}

fn compute_hash(data: &[u8], flags: &ContentFlags) -> [u8; HASH_LEN] {
    debug_assert!(!flags.is_inline(), "compute_hash called with inline flags");
    if flags.sha224 && !flags.lsb_mode {
        harmony_crypto::hash::sha224_hash(data)
    } else if flags.lsb_mode && !flags.sha224 {
        let full = harmony_crypto::hash::full_hash(data);
        let mut trunc = [0u8; HASH_LEN];
        trunc.copy_from_slice(&full[4..32]);
        trunc
    } else {
        let full = harmony_crypto::hash::full_hash(data);
        let mut trunc = [0u8; HASH_LEN];
        trunc.copy_from_slice(&full[..HASH_LEN]);
        trunc
    }
}

/// XOR fold all 128 two-bit pairs in a 256-bit (32-byte) value.
pub fn compute_pairwise_xor(bytes: &[u8; 32]) -> u8 {
    let mut fold: u8 = 0;
    for &b in bytes {
        fold ^= (b >> 6) & 0x03;
        fold ^= (b >> 4) & 0x03;
        fold ^= (b >> 2) & 0x03;
        fold ^= b & 0x03;
    }
    fold & 0x03
}

/// Encode a byte size into a 20-bit bundle size field (12-bit mantissa + 8-bit exponent).
///
/// The formula is `(1 + M/4096) * 2^(E+20)` = `(4096 + M) << (E+8)`.
/// Raw value 0 (M=0, E=0) encodes exactly 1MB = 1,048,576 bytes, the minimum valid bundle size.
/// Values below 1MB are rounded up to the 1MB minimum.
/// The encoding always rounds up so that the decoded value is >= the actual size.
pub fn encode_bundle_size(size_bytes: u64) -> u32 {
    if size_bytes <= (1u64 << 20) {
        // Minimum bundle size is 1MB. Round up to the minimum encoding (raw=0).
        return 0;
    }
    let bit_len = 64 - size_bytes.leading_zeros();
    let exp_raw = bit_len - 21; // bits above 2^20
    let exponent = if exp_raw > 255 { 255u32 } else { exp_raw };
    let shift = exponent + 20;
    let mantissa = if shift < 64 {
        let scaled = size_bytes >> (shift - 12);
        let m = scaled.saturating_sub(4096);
        if m > 4095 {
            4095u32
        } else {
            m as u32
        }
    } else {
        0u32
    };
    // Round up: if the encoded value decodes to less than size_bytes, bump mantissa.
    let encoded = (mantissa << 8) | exponent;
    let decoded = decode_bundle_size(encoded);
    if decoded < size_bytes {
        if mantissa < 4095 {
            ((mantissa + 1) << 8) | exponent
        } else if exponent < 255 {
            // Mantissa maxed out — roll to next exponent with mantissa 0
            (0u32 << 8) | (exponent + 1)
        } else {
            // Both maxed — return maximum representable value
            (4095u32 << 8) | 255
        }
    } else {
        encoded
    }
}

/// Decode a 20-bit bundle size field to a byte count.
///
/// The formula is `(4096 + M) << (E+8)`.
/// Raw value 0 (M=0, E=0) decodes to exactly 1MB = 1,048,576 bytes.
pub fn decode_bundle_size(raw: u32) -> u64 {
    let mantissa = (raw >> 8) & 0xFFF;
    let exponent = raw & 0xFF;
    let base = 4096u64 + mantissa as u64; // (1 + M/4096) * 4096
    let shift = exponent + 8; // 2^(E+20) / 4096 = 2^(E+8)
    if shift >= 64 {
        return u64::MAX;
    }
    base.checked_shl(shift).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn content_id_is_32_bytes() {
        assert_eq!(core::mem::size_of::<ContentId>(), 32);
    }

    #[test]
    fn header_bit_extraction() {
        let mode: u32 = 0b1010;
        let depth: u32 = 0b000101;
        let size: u32 = 0x12345;
        let cksum: u32 = 0b10;
        let header_u32 = (mode << 28) | (depth << 22) | (size << 2) | cksum;
        let cid = ContentId {
            header: header_u32.to_be_bytes(),
            hash: [0u8; HASH_LEN],
        };
        let flags = cid.flags();
        assert!(flags.encrypted);
        assert!(!flags.ephemeral);
        assert!(flags.sha224);
        assert!(!flags.lsb_mode);
        assert_eq!(cid.depth(), 5);
        assert_eq!(cid.raw_size_field(), 0x12345);
        assert_eq!(cid.header_u32() & 0x03, 0b10);
    }

    #[test]
    fn for_book_basic() {
        let data = b"hello harmony content addressing";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert_eq!(cid.depth(), 0);
        assert_eq!(cid.cid_type(), CidType::Book);
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert!(cid.verify_hash(data));
    }

    #[test]
    fn for_book_empty() {
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
    fn for_book_rejects_oversized() {
        let data = vec![0u8; MAX_PAYLOAD_SIZE + 1];
        assert!(ContentId::for_book(&data, ContentFlags::default()).is_err());
    }

    #[test]
    fn for_bundle_basic() {
        let book_a = ContentId::for_book(b"chunk a", ContentFlags::default()).unwrap();
        let book_b = ContentId::for_book(b"chunk b", ContentFlags::default()).unwrap();
        let children = [book_a, book_b];
        let bytes = children_to_bytes(&children);
        let cid = ContentId::for_bundle(&bytes, &children, ContentFlags::default()).unwrap();
        assert_eq!(cid.depth(), 1);
        assert_eq!(cid.cid_type(), CidType::Bundle(1));
        assert!(cid.verify_hash(&bytes));
    }

    #[test]
    fn for_bundle_depth_cascade() {
        let book = ContentId::for_book(b"leaf", ContentFlags::default()).unwrap();
        let mut current = book;
        for d in 1..=6u8 {
            let children = [current];
            let bytes = children_to_bytes(&children);
            current = ContentId::for_bundle(&bytes, &children, ContentFlags::default()).unwrap();
            assert_eq!(current.depth(), d);
            assert_eq!(current.cid_type(), CidType::Bundle(d));
        }
    }

    #[test]
    fn for_bundle_rejects_depth_63() {
        let book = ContentId::for_book(b"leaf", ContentFlags::default()).unwrap();
        let mut current = book;
        for _ in 0..MAX_BUNDLE_DEPTH {
            let children = [current];
            let bytes = children_to_bytes(&children);
            current = ContentId::for_bundle(&bytes, &children, ContentFlags::default()).unwrap();
        }
        assert_eq!(current.depth(), MAX_BUNDLE_DEPTH);
        let children = [current];
        let bytes = children_to_bytes(&children);
        assert!(ContentId::for_bundle(&bytes, &children, ContentFlags::default()).is_err());
    }

    #[test]
    fn for_stream_basic() {
        let hash = [0xABu8; HASH_LEN];
        let cid = ContentId::for_stream(hash, ContentFlags::default(), 42).unwrap();
        assert_eq!(cid.depth(), STREAM_DEPTH);
        assert_eq!(cid.cid_type(), CidType::Stream);
        assert_eq!(cid.raw_size_field(), 42);
        assert_eq!(cid.hash, hash);
    }

    #[test]
    fn inline_data_basic() {
        let data = b"short";
        let cid = ContentId::inline_data(data, ContentFlags::default()).unwrap();
        assert!(cid.is_inline());
        assert_eq!(cid.cid_type(), CidType::InlineData);
        assert_eq!(cid.extract_inline_data().unwrap(), data);
    }

    #[test]
    fn inline_data_max_27_bytes() {
        let data = [0xAB; MAX_INLINE_DATA];
        let cid = ContentId::inline_data(&data, ContentFlags::default()).unwrap();
        assert_eq!(&cid.extract_inline_data().unwrap()[..], &data[..]);
        assert!(
            ContentId::inline_data(&[0u8; MAX_INLINE_DATA + 1], ContentFlags::default()).is_err()
        );
    }

    #[test]
    fn checksum_valid_on_creation() {
        let cid = ContentId::for_book(b"valid data", ContentFlags::default()).unwrap();
        assert!(cid.verify_checksum().is_ok());
    }

    #[test]
    fn checksum_detects_corruption() {
        let cid = ContentId::for_book(b"valid data", ContentFlags::default()).unwrap();
        let mut bytes = cid.to_bytes();
        bytes[10] ^= 0x01;
        assert!(ContentId::from_bytes(bytes).verify_checksum().is_err());
    }

    #[test]
    fn checksum_rejects_fold_03_for_non_inline() {
        // A non-inline CID whose pairwise XOR fold is 0x03 is corrupted:
        // only inline sentinels intentionally produce fold==0x03.
        // Craft a non-inline CID with fold==0x03 by flipping one bit.
        let mut cid = ContentId::for_book(b"valid data", ContentFlags::default()).unwrap();
        let mut bytes = cid.to_bytes();
        // Flip the lowest bit of the checksum nibble (bits 1-0 of header byte 3)
        // so that fold goes from 0x00 to 0x03 without touching mode/depth/size.
        // XOR two low-order bits of byte 3 to change fold by 0x03.
        bytes[3] ^= 0x03;
        cid = ContentId::from_bytes(bytes);
        assert!(!cid.is_inline(), "test setup: expected non-inline CID");
        assert!(
            cid.verify_checksum().is_err(),
            "non-inline CID with fold==0x03 must fail verify_checksum"
        );
    }

    #[test]
    fn verify_hash_sha256_msb() {
        let data = b"sha256 msb test";
        let cid = ContentId::for_book(data, ContentFlags::default()).unwrap();
        assert!(cid.verify_hash(data));
        assert!(!cid.verify_hash(b"wrong"));
        let full = harmony_crypto::hash::full_hash(data);
        assert_eq!(&cid.hash[..], &full[..HASH_LEN]);
    }

    #[test]
    fn verify_hash_sha256_lsb() {
        let data = b"sha256 lsb test";
        let flags = ContentFlags {
            lsb_mode: true,
            ..ContentFlags::default()
        };
        let cid = ContentId::for_book(data, flags).unwrap();
        assert!(cid.verify_hash(data));
        let full = harmony_crypto::hash::full_hash(data);
        assert_eq!(&cid.hash[..], &full[4..32]);
    }

    #[test]
    fn verify_hash_sha224() {
        let data = b"sha224 test";
        let flags = ContentFlags {
            sha224: true,
            ..ContentFlags::default()
        };
        let cid = ContentId::for_book(data, flags).unwrap();
        assert!(cid.verify_hash(data));
        assert_eq!(&cid.hash[..], &harmony_crypto::hash::sha224_hash(data)[..]);
    }

    #[test]
    fn flags_round_trip() {
        for enc in [false, true] {
            for eph in [false, true] {
                for sha in [false, true] {
                    for lsb in [false, true] {
                        let f = ContentFlags {
                            encrypted: enc,
                            ephemeral: eph,
                            sha224: sha,
                            lsb_mode: lsb,
                        };
                        assert_eq!(f, ContentFlags::from_bits(f.to_bits()));
                    }
                }
            }
        }
    }

    #[test]
    fn content_class_four_variants() {
        assert_eq!(
            ContentId::for_book(b"a", ContentFlags::default())
                .unwrap()
                .content_class(),
            ContentClass::PublicDurable
        );
        assert_eq!(
            ContentId::for_book(
                b"a",
                ContentFlags {
                    ephemeral: true,
                    ..ContentFlags::default()
                }
            )
            .unwrap()
            .content_class(),
            ContentClass::PublicEphemeral
        );
        assert_eq!(
            ContentId::for_book(
                b"a",
                ContentFlags {
                    encrypted: true,
                    ..ContentFlags::default()
                }
            )
            .unwrap()
            .content_class(),
            ContentClass::EncryptedDurable
        );
        assert_eq!(
            ContentId::for_book(
                b"a",
                ContentFlags {
                    encrypted: true,
                    ephemeral: true,
                    ..ContentFlags::default()
                }
            )
            .unwrap()
            .content_class(),
            ContentClass::EncryptedEphemeral
        );
    }

    #[test]
    fn serialization_round_trip() {
        let cid = ContentId::for_book(b"round trip test", ContentFlags::default()).unwrap();
        assert_eq!(cid, ContentId::from_bytes(cid.to_bytes()));
    }

    #[test]
    fn bundle_size_encode_decode() {
        // Sub-1MB values round up to 1MB minimum.
        assert_eq!(decode_bundle_size(encode_bundle_size(0)), 1_048_576);
        assert_eq!(decode_bundle_size(encode_bundle_size(512)), 1_048_576);
        // Exactly 1MB encodes as raw=0, decodes back to 1MB.
        assert_eq!(encode_bundle_size(1_048_576), 0);
        assert_eq!(decode_bundle_size(encode_bundle_size(1_048_576)), 1_048_576);
        // 2MB encodes and decodes with <=3% tolerance.
        let decoded_2mb = decode_bundle_size(encode_bundle_size(2_097_152));
        let ratio = decoded_2mb as f64 / 2_097_152f64;
        assert!((0.97..=1.03).contains(&ratio));
        // Encoding always rounds up (decoded >= input).
        for &size in &[1_048_577u64, 1_500_000, 10_000_000, 1_000_000_000] {
            let decoded = decode_bundle_size(encode_bundle_size(size));
            assert!(decoded >= size, "encode_bundle_size({size}) decoded {decoded} < input");
        }
    }

    #[test]
    fn bundle_size_min_max_range() {
        // raw=0 is 1MB (minimum valid bundle size), not a sentinel.
        assert_eq!(decode_bundle_size(0), 1_048_576);
        assert_eq!(decode_bundle_size((0 << 8) | 1), 2_097_152);
        assert_eq!(decode_bundle_size((4095 << 8) | 255), u64::MAX);
    }

    #[test]
    fn inline_metadata_round_trip() {
        let meta = ContentId::inline_metadata(100_000_000, 100, 1709337600000, *b"text/pln");
        assert!(meta.is_inline());
        assert!(meta.is_sentinel());
        let (sz, cnt, ts, mime) = meta.parse_inline_metadata().unwrap();
        assert_eq!(
            (sz, cnt, ts, &mime),
            (100_000_000, 100, 1709337600000, b"text/pln")
        );
    }

    #[test]
    fn inline_metadata_checksum_valid() {
        assert!(ContentId::inline_metadata(1000, 1, 0, [0; 8])
            .verify_checksum()
            .is_ok());
    }

    fn children_to_bytes(children: &[ContentId]) -> Vec<u8> {
        children.iter().flat_map(|c| c.to_bytes()).collect()
    }
}

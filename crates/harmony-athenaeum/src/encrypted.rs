// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Encrypted book metadata format.
//!
//! The metadata is stored in the first N pages of an encrypted book,
//! each prefixed with the 11 sentinel (0xFFFFFFFC). This module handles
//! serialization/deserialization of the metadata payload independently
//! of any cryptographic types — all key material is raw bytes.

use alloc::vec::Vec;

use crate::addr::{PAGE_SIZE, SELF_INDEX_SENTINEL_11};
use crate::athenaeum::BookError;

/// Sentinel value for encrypted book metadata pages.
///
/// Uses the `11` self-indexing sentinel (`0xFFFFFFFC`) to mark pages
/// that carry encrypted book metadata rather than data content.
pub const ENCRYPTED_SENTINEL: u32 = SELF_INDEX_SENTINEL_11;

/// Usable payload bytes per metadata page (PAGE_SIZE minus 4-byte sentinel).
pub const METADATA_PAGE_PAYLOAD: usize = PAGE_SIZE - 4;

/// Metadata for an encrypted book.
///
/// All cryptographic key material is stored as raw `Vec<u8>` — this type
/// is intentionally independent of `harmony-crypto`. The caller is
/// responsible for serializing PQC types into these byte vectors.
///
/// ## Wire Format (payload, without sentinel prefix)
///
/// ```text
/// Offset     Size    Field
/// 0          2       version (u16 LE)
/// 2          1       flags (bit 0 = has_expiry, bit 1 = has_tags)
/// 3          1       encryption_algo (0x00 = ChaCha20-Poly1305)
/// 4          2       owner_key_len (u16 LE)
/// 6          K       owner_public_key
/// 6+K        2       encapsulated_key_len (u16 LE)
/// 8+K        C       encapsulated_key
/// 8+K+C      2       signature_len (u16 LE)
/// 10+K+C     S       signature
/// 10+K+C+S   8       expiry (u64 LE) — if has_expiry
/// ...        2       tags_len (u16 LE) — if has_tags
/// ...        T       tags (opaque bytes) — if has_tags
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedBookMetadata {
    /// Format version (currently 1).
    pub version: u16,
    /// Flags: bit 0 = has_expiry, bit 1 = has_tags.
    pub flags: u8,
    /// Encryption algorithm identifier (0x00 = ChaCha20-Poly1305).
    pub encryption_algo: u8,
    /// Owner's public key (e.g., ML-KEM-768 public key, 1184 bytes).
    pub owner_public_key: Vec<u8>,
    /// Encapsulated key (e.g., ML-KEM-768 ciphertext, 1088 bytes).
    pub encapsulated_key: Vec<u8>,
    /// Signature over the book (e.g., ML-DSA-65 signature, 3309 bytes).
    pub signature: Vec<u8>,
    /// Optional expiry timestamp (Unix seconds, u64).
    pub expiry: Option<u64>,
    /// Optional opaque tag data.
    pub tags: Option<Vec<u8>>,
}

impl EncryptedBookMetadata {
    /// Flag bit: metadata includes an expiry timestamp.
    const FLAG_HAS_EXPIRY: u8 = 0x01;
    /// Flag bit: metadata includes tags.
    const FLAG_HAS_TAGS: u8 = 0x02;

    /// Serialize the metadata payload (without sentinel prefix).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();

        // version (u16 LE)
        buf.extend_from_slice(&self.version.to_le_bytes());
        // flags (u8)
        buf.push(self.flags);
        // encryption_algo (u8)
        buf.push(self.encryption_algo);

        // owner_public_key: len (u16 LE) + data
        buf.extend_from_slice(&(self.owner_public_key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.owner_public_key);

        // encapsulated_key: len (u16 LE) + data
        buf.extend_from_slice(&(self.encapsulated_key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.encapsulated_key);

        // signature: len (u16 LE) + data
        buf.extend_from_slice(&(self.signature.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.signature);

        // expiry (u64 LE) — if has_expiry
        if self.flags & Self::FLAG_HAS_EXPIRY != 0 {
            if let Some(exp) = self.expiry {
                buf.extend_from_slice(&exp.to_le_bytes());
            }
        }

        // tags: len (u16 LE) + data — if has_tags
        if self.flags & Self::FLAG_HAS_TAGS != 0 {
            if let Some(ref tags) = self.tags {
                buf.extend_from_slice(&(tags.len() as u16).to_le_bytes());
                buf.extend_from_slice(tags);
            }
        }

        buf
    }

    /// Deserialize from a payload byte slice (without sentinel prefix).
    ///
    /// Performs bounds checking before every field extraction.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        let mut offset = 0usize;

        // version (u16 LE)
        if offset + 2 > data.len() {
            return Err(BookError::TooShort);
        }
        let version = u16::from_le_bytes([data[offset], data[offset + 1]]);
        offset += 2;

        // flags (u8)
        if offset + 1 > data.len() {
            return Err(BookError::TooShort);
        }
        let flags = data[offset];
        offset += 1;

        // encryption_algo (u8)
        if offset + 1 > data.len() {
            return Err(BookError::TooShort);
        }
        let encryption_algo = data[offset];
        offset += 1;

        // owner_public_key: len (u16 LE) + data
        if offset + 2 > data.len() {
            return Err(BookError::TooShort);
        }
        let owner_key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + owner_key_len > data.len() {
            return Err(BookError::TooShort);
        }
        let owner_public_key = data[offset..offset + owner_key_len].to_vec();
        offset += owner_key_len;

        // encapsulated_key: len (u16 LE) + data
        if offset + 2 > data.len() {
            return Err(BookError::TooShort);
        }
        let encap_key_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + encap_key_len > data.len() {
            return Err(BookError::TooShort);
        }
        let encapsulated_key = data[offset..offset + encap_key_len].to_vec();
        offset += encap_key_len;

        // signature: len (u16 LE) + data
        if offset + 2 > data.len() {
            return Err(BookError::TooShort);
        }
        let sig_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
        offset += 2;
        if offset + sig_len > data.len() {
            return Err(BookError::TooShort);
        }
        let signature = data[offset..offset + sig_len].to_vec();
        offset += sig_len;

        // expiry (u64 LE) — if has_expiry
        let expiry = if flags & Self::FLAG_HAS_EXPIRY != 0 {
            if offset + 8 > data.len() {
                return Err(BookError::TooShort);
            }
            let exp = u64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]);
            offset += 8;
            Some(exp)
        } else {
            None
        };

        // tags: len (u16 LE) + data — if has_tags
        let tags = if flags & Self::FLAG_HAS_TAGS != 0 {
            if offset + 2 > data.len() {
                return Err(BookError::TooShort);
            }
            let tags_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            if offset + tags_len > data.len() {
                return Err(BookError::TooShort);
            }
            let tags_data = data[offset..offset + tags_len].to_vec();
            #[allow(unused_assignments)]
            {
                offset += tags_len;
            }
            Some(tags_data)
        } else {
            None
        };

        Ok(EncryptedBookMetadata {
            version,
            flags,
            encryption_algo,
            owner_public_key,
            encapsulated_key,
            signature,
            expiry,
            tags,
        })
    }

    /// How many 4KB pages are needed to store this metadata.
    ///
    /// Each page has a 4-byte sentinel prefix, leaving 4092 bytes for payload.
    pub fn pages_needed(&self) -> u8 {
        let payload_len = self.to_bytes().len();
        payload_len.div_ceil(METADATA_PAGE_PAYLOAD) as u8
    }

    /// Serialize the metadata into sentinel-prefixed 4KB pages.
    ///
    /// Each page starts with `ENCRYPTED_SENTINEL` (4 bytes LE), followed by
    /// up to `METADATA_PAGE_PAYLOAD` (4092) bytes of payload. The last page
    /// is zero-padded to a full 4KB.
    pub fn to_pages(&self) -> Vec<[u8; PAGE_SIZE]> {
        let payload = self.to_bytes();
        let page_count = payload.len().div_ceil(METADATA_PAGE_PAYLOAD);
        let sentinel_bytes = ENCRYPTED_SENTINEL.to_le_bytes();

        let mut pages = Vec::with_capacity(page_count);
        for i in 0..page_count {
            let mut page = [0u8; PAGE_SIZE];
            // Write sentinel prefix
            page[0..4].copy_from_slice(&sentinel_bytes);
            // Write payload chunk
            let chunk_start = i * METADATA_PAGE_PAYLOAD;
            let chunk_end = core::cmp::min(chunk_start + METADATA_PAGE_PAYLOAD, payload.len());
            let chunk = &payload[chunk_start..chunk_end];
            page[4..4 + chunk.len()].copy_from_slice(chunk);
            pages.push(page);
        }

        pages
    }

    /// Deserialize metadata from sentinel-prefixed 4KB pages.
    ///
    /// Validates that each page starts with `ENCRYPTED_SENTINEL`, then
    /// concatenates the payload portions and deserializes.
    pub fn from_pages(pages: &[[u8; PAGE_SIZE]]) -> Result<Self, BookError> {
        if pages.is_empty() {
            return Err(BookError::TooShort);
        }

        let mut payload = Vec::with_capacity(pages.len() * METADATA_PAGE_PAYLOAD);
        for page in pages {
            let sentinel = u32::from_le_bytes([page[0], page[1], page[2], page[3]]);
            if sentinel != ENCRYPTED_SENTINEL {
                return Err(BookError::BadFormat);
            }
            payload.extend_from_slice(&page[4..]);
        }

        Self::from_bytes(&payload)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> EncryptedBookMetadata {
        EncryptedBookMetadata {
            version: 1,
            flags: 0,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: None,
            tags: None,
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let meta = sample_metadata();
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta.version, meta2.version);
        assert_eq!(meta.owner_public_key, meta2.owner_public_key);
        assert_eq!(meta.encapsulated_key, meta2.encapsulated_key);
        assert_eq!(meta.signature, meta2.signature);
        assert_eq!(meta.expiry, meta2.expiry);
        assert_eq!(meta.tags, meta2.tags);
    }

    #[test]
    fn serialize_with_expiry() {
        let mut meta = sample_metadata();
        meta.flags = 0x01; // has_expiry
        meta.expiry = Some(1_700_000_000);
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta2.expiry, Some(1_700_000_000));
    }

    #[test]
    fn serialize_with_tags() {
        let mut meta = sample_metadata();
        meta.flags = 0x02; // has_tags
        meta.tags = Some(b"topic:science".to_vec());
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta2.tags.as_deref(), Some(b"topic:science".as_slice()));
    }

    #[test]
    fn minimum_size_is_5591_bytes() {
        let meta = sample_metadata();
        let bytes = meta.to_bytes();
        assert_eq!(bytes.len(), 10 + 1184 + 1088 + 3309); // 5591
    }

    #[test]
    fn pages_needed_for_minimum_metadata() {
        let meta = sample_metadata();
        assert_eq!(meta.pages_needed(), 2);
    }

    #[test]
    fn page_roundtrip() {
        let meta = sample_metadata();
        let pages = meta.to_pages();
        assert_eq!(pages.len(), 2);
        // Each page starts with the encrypted sentinel
        for page in &pages {
            let sentinel = u32::from_le_bytes([page[0], page[1], page[2], page[3]]);
            assert_eq!(sentinel, ENCRYPTED_SENTINEL);
        }
        let meta2 = EncryptedBookMetadata::from_pages(&pages).unwrap();
        assert_eq!(meta.owner_public_key, meta2.owner_public_key);
    }

    #[test]
    fn serialize_with_expiry_and_tags() {
        let mut meta = sample_metadata();
        meta.flags = 0x03; // has_expiry + has_tags
        meta.expiry = Some(1_700_000_000);
        meta.tags = Some(b"classification:secret".to_vec());
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta2.expiry, Some(1_700_000_000));
        assert_eq!(
            meta2.tags.as_deref(),
            Some(b"classification:secret".as_slice())
        );
    }

    #[test]
    fn encrypted_sentinel_matches_self_index_sentinel_11() {
        assert_eq!(ENCRYPTED_SENTINEL, 0xFFFF_FFFC);
        assert_eq!(ENCRYPTED_SENTINEL, SELF_INDEX_SENTINEL_11);
    }

    #[test]
    fn metadata_page_payload_is_page_size_minus_4() {
        assert_eq!(METADATA_PAGE_PAYLOAD, PAGE_SIZE - 4);
        assert_eq!(METADATA_PAGE_PAYLOAD, 4092);
    }

    #[test]
    fn from_bytes_rejects_too_short() {
        assert_eq!(
            EncryptedBookMetadata::from_bytes(&[]),
            Err(BookError::TooShort)
        );
        assert_eq!(
            EncryptedBookMetadata::from_bytes(&[0x01]),
            Err(BookError::TooShort)
        );
    }

    #[test]
    fn from_bytes_rejects_truncated_key_data() {
        // version + flags + algo + owner_key_len=1184 but only 10 bytes of key data
        let mut buf = Vec::new();
        buf.extend_from_slice(&1u16.to_le_bytes()); // version
        buf.push(0); // flags
        buf.push(0); // encryption_algo
        buf.extend_from_slice(&1184u16.to_le_bytes()); // owner_key_len
        buf.extend_from_slice(&[0xAA; 10]); // only 10 bytes, not 1184
        assert_eq!(
            EncryptedBookMetadata::from_bytes(&buf),
            Err(BookError::TooShort)
        );
    }

    #[test]
    fn from_pages_rejects_empty() {
        let empty: &[[u8; PAGE_SIZE]] = &[];
        assert_eq!(
            EncryptedBookMetadata::from_pages(empty),
            Err(BookError::TooShort)
        );
    }

    #[test]
    fn from_pages_rejects_bad_sentinel() {
        let mut page = [0u8; PAGE_SIZE];
        // Write wrong sentinel
        page[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert_eq!(
            EncryptedBookMetadata::from_pages(&[page]),
            Err(BookError::BadFormat)
        );
    }

    #[test]
    fn full_roundtrip_with_all_options() {
        let meta = EncryptedBookMetadata {
            version: 1,
            flags: 0x03,
            encryption_algo: 0,
            owner_public_key: vec![0x11; 1184],
            encapsulated_key: vec![0x22; 1088],
            signature: vec![0x33; 3309],
            expiry: Some(9_999_999_999),
            tags: Some(b"access:restricted;domain:medical".to_vec()),
        };
        // Full cycle: struct -> bytes -> pages -> bytes -> struct
        let pages = meta.to_pages();
        let meta2 = EncryptedBookMetadata::from_pages(&pages).unwrap();
        assert_eq!(meta, meta2);
    }
}

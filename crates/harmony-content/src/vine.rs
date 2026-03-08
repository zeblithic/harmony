//! Vine — short-form video metadata for the Harmony content layer.

use alloc::string::String;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// MIME type marker for a single vine video clip.
pub const MIME_VINE_VIDEO: [u8; 8] = *b"vine/vid";

/// MIME type marker for a compilation of vine clips.
pub const MIME_VINE_COMPILATION: [u8; 8] = *b"vine/cmp";

/// Maximum allowed length for a vine title, in bytes.
///
/// Byte-based (not character-based) because the wire format and storage
/// layer care about serialized size. This means multi-byte scripts (CJK,
/// emoji) get fewer visible characters — an intentional trade-off for
/// predictable payload sizes.
pub const MAX_TITLE_LEN: usize = 140;

/// Descriptor for a vine video, carrying creator identity and optional metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VineDescriptor {
    /// 128-bit Harmony address of the creator.
    pub creator_address: [u8; 16],
    /// Unix timestamp (seconds) when the vine was created.
    pub created_at: u64,
    /// CID of the raw video content blob.
    pub video_cid: [u8; 32],
    /// Optional human-readable title (max 140 bytes).
    pub title: Option<String>,
    /// If this vine is a reshare, the CID hash of the original.
    pub reshare_of: Option<[u8; 32]>,
}

impl VineDescriptor {
    /// Serialize to bytes using postcard.
    pub fn to_bytes(&self) -> Vec<u8> {
        postcard::to_allocvec(self).expect("VineDescriptor serialization should not fail")
    }

    /// Deserialize from postcard-encoded bytes.
    ///
    /// Note: this does **not** call [`Self::validate`]. Callers should
    /// validate the returned descriptor before trusting its contents
    /// (e.g. title length).
    pub fn from_bytes(data: &[u8]) -> Result<Self, postcard::Error> {
        postcard::from_bytes(data)
    }

    /// Validate descriptor constraints.
    ///
    /// Returns an error if the title exceeds [`MAX_TITLE_LEN`] bytes.
    pub fn validate(&self) -> Result<(), &'static str> {
        if let Some(ref title) = self.title {
            if title.len() > MAX_TITLE_LEN {
                return Err("title exceeds maximum length");
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vine_descriptor_round_trip() {
        let desc = VineDescriptor {
            creator_address: [0xAB; 16],
            created_at: 1_700_000_000,
            video_cid: [0xCC; 32],
            title: Some(String::from("My first vine")),
            reshare_of: None,
        };

        let bytes = desc.to_bytes();
        let recovered = VineDescriptor::from_bytes(&bytes).unwrap();
        assert_eq!(desc, recovered);
    }

    #[test]
    fn vine_descriptor_reshare_round_trip() {
        let desc = VineDescriptor {
            creator_address: [0x01; 16],
            created_at: 1_700_000_001,
            video_cid: [0xBB; 32],
            title: None,
            reshare_of: Some([0xDE; 32]),
        };

        let bytes = desc.to_bytes();
        let recovered = VineDescriptor::from_bytes(&bytes).unwrap();
        assert_eq!(desc, recovered);
    }

    #[test]
    fn vine_descriptor_title_too_long() {
        let desc = VineDescriptor {
            creator_address: [0x00; 16],
            created_at: 0,
            video_cid: [0x00; 32],
            title: Some("A".repeat(141)),
            reshare_of: None,
        };

        assert!(desc.validate().is_err());

        // Exactly 140 should be fine.
        let ok_desc = VineDescriptor {
            creator_address: [0x00; 16],
            created_at: 0,
            video_cid: [0x00; 32],
            title: Some("B".repeat(140)),
            reshare_of: None,
        };
        assert!(ok_desc.validate().is_ok());
    }

    #[test]
    fn mime_constants_are_8_bytes() {
        assert_eq!(MIME_VINE_VIDEO.len(), 8);
        assert_eq!(MIME_VINE_COMPILATION.len(), 8);
    }
}

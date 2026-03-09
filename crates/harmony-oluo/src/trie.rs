// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Trie node — internal node format for the adaptive-depth embedding trie.

use crate::error::{OluoError, OluoResult};

/// Internal trie node magic: "HSN" + version 1.
pub const TRIE_NODE_MAGIC: [u8; 4] = [0x48, 0x53, 0x4E, 0x01];
/// Size of an internal trie node blob.
pub const TRIE_NODE_SIZE: usize = 106;
/// Maximum valid `split_bit` value (exclusive). Centroid is 256-bit (T3).
pub const TRIE_MAX_SPLIT_BIT: u16 = 256;

/// An internal node in the adaptive-depth embedding trie.
///
/// Binary layout (106 bytes total):
/// - `[0..4]`   magic (`TRIE_NODE_MAGIC`)
/// - `[4..8]`   fingerprint (SHA-256(model_id)\[:4\])
/// - `[8..10]`  split_bit (u16 big-endian)
/// - `[10..42]` child0 CID (32 bytes)
/// - `[42..74]` child1 CID (32 bytes)
/// - `[74..106]` centroid (256-bit binary vector)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrieNode {
    /// Model fingerprint (SHA-256(model_id)[:4]).
    pub fingerprint: [u8; 4],
    /// The embedding bit position used to split at this node.
    pub split_bit: u16,
    /// CID of the child-0 subtree (bit = 0 at split_bit position).
    pub child0: [u8; 32],
    /// CID of the child-1 subtree (bit = 1 at split_bit position).
    pub child1: [u8; 32],
    /// Centroid of the entire subtree (256-bit binary vector).
    pub centroid: [u8; 32],
}

impl TrieNode {
    /// Encode this trie node into a fixed-size byte array.
    pub fn encode(&self) -> [u8; TRIE_NODE_SIZE] {
        let mut buf = [0u8; TRIE_NODE_SIZE];
        buf[0..4].copy_from_slice(&TRIE_NODE_MAGIC);
        buf[4..8].copy_from_slice(&self.fingerprint);
        buf[8..10].copy_from_slice(&self.split_bit.to_be_bytes());
        buf[10..42].copy_from_slice(&self.child0);
        buf[42..74].copy_from_slice(&self.child1);
        buf[74..106].copy_from_slice(&self.centroid);
        buf
    }

    /// Decode a trie node from a byte slice.
    ///
    /// Returns [`OluoError::TruncatedTrieNode`] if `data` is too short,
    /// or [`OluoError::InvalidTrieNode`] if the magic bytes don't match.
    pub fn decode(data: &[u8]) -> OluoResult<Self> {
        if data.len() < TRIE_NODE_SIZE {
            return Err(OluoError::TruncatedTrieNode {
                expected: TRIE_NODE_SIZE,
                actual: data.len(),
            });
        }
        if data[0..4] != TRIE_NODE_MAGIC {
            return Err(OluoError::InvalidTrieNode);
        }

        let mut fingerprint = [0u8; 4];
        fingerprint.copy_from_slice(&data[4..8]);

        let split_bit = u16::from_be_bytes([data[8], data[9]]);
        if split_bit >= TRIE_MAX_SPLIT_BIT {
            return Err(OluoError::InvalidTrieNode);
        }

        let mut child0 = [0u8; 32];
        child0.copy_from_slice(&data[10..42]);

        let mut child1 = [0u8; 32];
        child1.copy_from_slice(&data[42..74]);

        let mut centroid = [0u8; 32];
        centroid.copy_from_slice(&data[74..106]);

        Ok(Self {
            fingerprint,
            split_bit,
            child0,
            child1,
            centroid,
        })
    }
}

/// Extract a single bit from a binary vector (MSB-first convention).
///
/// `bit_position` is the overall bit index (0..N). Bit 0 is the most
/// significant bit of byte 0, matching the convention used by
/// `quantize_to_binary` in harmony-semantic.
///
/// Returns `false` if `bit_position` is out of bounds.
pub fn get_bit(vector: &[u8], bit_position: u16) -> bool {
    let byte_index = (bit_position / 8) as usize;
    if byte_index >= vector.len() {
        return false;
    }
    let bit_within_byte = 7 - (bit_position % 8);
    (vector[byte_index] >> bit_within_byte) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_node() -> TrieNode {
        TrieNode {
            fingerprint: [0xDE, 0xAD, 0xBE, 0xEF],
            split_bit: 42,
            child0: [0xAA; 32],
            child1: [0xBB; 32],
            centroid: [0xCC; 32],
        }
    }

    #[test]
    fn trie_node_encode_decode_roundtrip() {
        let node = sample_node();
        let encoded = node.encode();
        assert_eq!(encoded.len(), TRIE_NODE_SIZE);

        let decoded = TrieNode::decode(&encoded).expect("decode should succeed");
        assert_eq!(decoded, node);
    }

    #[test]
    fn trie_node_decode_bad_magic() {
        let node = sample_node();
        let mut encoded = node.encode();
        // Corrupt the magic bytes
        encoded[0] = 0xFF;

        let err = TrieNode::decode(&encoded).unwrap_err();
        assert_eq!(err, OluoError::InvalidTrieNode);
    }

    #[test]
    fn trie_node_decode_truncated() {
        let data = [0u8; 50]; // too short
        let err = TrieNode::decode(&data).unwrap_err();
        assert_eq!(
            err,
            OluoError::TruncatedTrieNode {
                expected: TRIE_NODE_SIZE,
                actual: 50,
            }
        );
    }

    #[test]
    fn trie_node_decode_invalid_split_bit() {
        let mut node = sample_node();
        node.split_bit = TRIE_MAX_SPLIT_BIT; // out of range for 256-bit centroid
        let encoded = node.encode();
        let err = TrieNode::decode(&encoded).unwrap_err();
        assert_eq!(err, OluoError::InvalidTrieNode);
    }

    #[test]
    fn get_bit_out_of_bounds_returns_false() {
        let vector = [0xFFu8; 4]; // 32 bits
        assert!(!get_bit(&vector, 32)); // first out-of-bounds bit
        assert!(!get_bit(&vector, 255)); // far out of bounds
    }

    #[test]
    fn get_bit_extracts_correct_bit() {
        // 0b1010_0101 = 0xA5
        let vector = [0xA5u8];
        // MSB-first: bit 0 = 1, bit 1 = 0, bit 2 = 1, bit 3 = 0,
        //            bit 4 = 0, bit 5 = 1, bit 6 = 0, bit 7 = 1
        assert!(get_bit(&vector, 0)); // bit 7 of byte = 1
        assert!(!get_bit(&vector, 1)); // bit 6 of byte = 0
        assert!(get_bit(&vector, 2)); // bit 5 of byte = 1
        assert!(!get_bit(&vector, 3)); // bit 4 of byte = 0
        assert!(!get_bit(&vector, 4)); // bit 3 of byte = 0
        assert!(get_bit(&vector, 5)); // bit 2 of byte = 1
        assert!(!get_bit(&vector, 6)); // bit 1 of byte = 0
        assert!(get_bit(&vector, 7)); // bit 0 of byte = 1
    }

    #[test]
    fn get_bit_boundary_values() {
        // 32-byte vector (256 bits), set known patterns at boundaries
        let mut vector = [0u8; 32];
        // Byte 0 = 0b1000_0001 = 0x81 → bit 0 = 1, bit 7 = 1
        vector[0] = 0x81;
        // Byte 1 = 0b1000_0000 = 0x80 → bit 8 = 1
        vector[1] = 0x80;
        // Byte 31 = 0b0000_0001 = 0x01 → bit 255 = 1
        vector[31] = 0x01;

        // bit 0: MSB of byte 0 → 1
        assert!(get_bit(&vector, 0));
        // bit 7: LSB of byte 0 → 1
        assert!(get_bit(&vector, 7));
        // bit 8: MSB of byte 1 → 1
        assert!(get_bit(&vector, 8));
        // bit 9: second bit of byte 1 → 0
        assert!(!get_bit(&vector, 9));
        // bit 255: LSB of byte 31 → 1
        assert!(get_bit(&vector, 255));
        // bit 248: MSB of byte 31 → 0
        assert!(!get_bit(&vector, 248));
    }
}

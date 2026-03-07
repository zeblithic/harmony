// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Volume — partition tree node for the Encyclopedia.

use alloc::vec::Vec;
use crate::book::Book;

/// Maximum partition depth (SHA-256 bits 22-252 = 230 usable bits).
pub const MAX_PARTITION_DEPTH: u8 = 230;

/// A partition node in the Encyclopedia tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Volume {
    /// Leaf — contains resolved Books for this partition slice.
    Leaf {
        partition_depth: u8,
        partition_path: u32,
        books: Vec<Book>,
    },
    /// Internal — splits into two child Volumes by content-hash bit.
    Split {
        partition_depth: u8,
        partition_path: u32,
        split_bit: u8,
        left: Box<Volume>,
        right: Box<Volume>,
    },
}

impl Volume {
    /// Create a new leaf Volume.
    pub fn leaf(depth: u8, path: u32, books: Vec<Book>) -> Self {
        Volume::Leaf {
            partition_depth: depth,
            partition_path: path,
            books,
        }
    }

    /// Number of unique chunks in this subtree.
    pub fn chunk_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => {
                books.iter()
                    .map(|b| b.entries.iter().map(|e| e.chunks.len()).sum::<usize>())
                    .sum()
            }
            Volume::Split { left, right, .. } => {
                left.chunk_count() + right.chunk_count()
            }
        }
    }

    /// Number of Books in this subtree.
    pub fn book_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => books.len(),
            Volume::Split { left, right, .. } => {
                left.book_count() + right.book_count()
            }
        }
    }

    /// Partition depth of this node.
    pub fn depth(&self) -> u8 {
        match self {
            Volume::Leaf { partition_depth, .. } | Volume::Split { partition_depth, .. } => {
                *partition_depth
            }
        }
    }

    /// Partition path of this node.
    pub fn path(&self) -> u32 {
        match self {
            Volume::Leaf { partition_path, .. } | Volume::Split { partition_path, .. } => {
                *partition_path
            }
        }
    }
}

/// Determine which side of a binary split a chunk belongs to.
///
/// Reads bit `bit_index` from a SHA-256 content hash.
/// Returns `false` for left (bit=0), `true` for right (bit=1).
pub fn route_chunk(content_hash: &[u8; 32], bit_index: u8) -> bool {
    let byte_idx = (bit_index / 8) as usize;
    let bit_offset = 7 - (bit_index % 8);
    (content_hash[byte_idx] >> bit_offset) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_chunk_bit_22() {
        let mut hash = [0u8; 32];
        // Bit 22 is in byte 2 (bits 16-23), bit_offset = 7 - (22 % 8) = 7 - 6 = 1
        hash[2] = 0b0000_0010; // bit 22 = 1
        assert!(route_chunk(&hash, 22));

        hash[2] = 0b0000_0000; // bit 22 = 0
        assert!(!route_chunk(&hash, 22));
    }

    #[test]
    fn route_chunk_bit_0() {
        let mut hash = [0u8; 32];
        hash[0] = 0x80; // bit 0 = 1 (MSB of byte 0)
        assert!(route_chunk(&hash, 0));

        hash[0] = 0x00;
        assert!(!route_chunk(&hash, 0));
    }

    #[test]
    fn route_chunk_bit_255() {
        let mut hash = [0u8; 32];
        hash[31] = 0x01; // bit 255 = 1 (LSB of last byte)
        assert!(route_chunk(&hash, 255));

        hash[31] = 0x00;
        assert!(!route_chunk(&hash, 255));
    }

    #[test]
    fn leaf_volume_chunk_count() {
        let book = Book { entries: Vec::new() };
        let vol = Volume::leaf(0, 0, alloc::vec![book]);
        assert_eq!(vol.chunk_count(), 0);
        assert_eq!(vol.book_count(), 1);
    }

    #[test]
    fn split_volume_chunk_count() {
        let left = Volume::leaf(1, 0, Vec::new());
        let right = Volume::leaf(1, 1, Vec::new());
        let split = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(split.chunk_count(), 0);
        assert_eq!(split.book_count(), 0);
        assert_eq!(split.depth(), 0);
    }

    #[test]
    fn volume_depth_and_path() {
        let vol = Volume::leaf(3, 0b101, Vec::new());
        assert_eq!(vol.depth(), 3);
        assert_eq!(vol.path(), 0b101);
    }
}

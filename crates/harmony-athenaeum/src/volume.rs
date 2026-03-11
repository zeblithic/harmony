// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Volume — partition tree node for the Encyclopedia.

use crate::addr::{PageAddr, ALGO_COUNT};
use crate::athenaeum::{Book, BookError};
use alloc::vec::Vec;

/// Maximum partition depth (SHA-256 bits 28-255 = 228 usable bits).
pub const MAX_PARTITION_DEPTH: u8 = 228;

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

    /// Total page references in this subtree.
    ///
    /// Sums `book.page_count()` across all Books in the subtree.
    pub fn page_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => books.iter().map(|b| b.page_count()).sum(),
            Volume::Split { left, right, .. } => left.page_count() + right.page_count(),
        }
    }

    /// Number of Books in this subtree.
    pub fn book_count(&self) -> usize {
        match self {
            Volume::Leaf { books, .. } => books.len(),
            Volume::Split { left, right, .. } => left.book_count() + right.book_count(),
        }
    }

    /// Partition depth of this node.
    pub fn depth(&self) -> u8 {
        match self {
            Volume::Leaf {
                partition_depth, ..
            }
            | Volume::Split {
                partition_depth, ..
            } => *partition_depth,
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

    /// Serialize this Volume node.
    ///
    /// Leaf format:
    /// - tag: u8 = 0
    /// - partition_depth: u8
    /// - partition_path: u32 (LE)
    /// - book_count: u32 (LE)
    /// - for each book: book_len: u32 (LE), book_bytes
    ///
    /// Split format:
    /// - tag: u8 = 1
    /// - partition_depth: u8
    /// - partition_path: u32 (LE)
    /// - split_bit: u8
    /// - reserved: u8
    /// - left_bytes_len: u32 (LE)
    /// - left_bytes
    /// - right_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        match self {
            Volume::Leaf {
                partition_depth,
                partition_path,
                books,
            } => {
                buf.push(0u8); // tag
                buf.push(*partition_depth);
                buf.extend_from_slice(&partition_path.to_le_bytes());
                let count = books.len() as u32;
                buf.extend_from_slice(&count.to_le_bytes());
                for book in books {
                    let book_bytes = serialize_book(book);
                    let len = book_bytes.len() as u32;
                    buf.extend_from_slice(&len.to_le_bytes());
                    buf.extend_from_slice(&book_bytes);
                }
            }
            Volume::Split {
                partition_depth,
                partition_path,
                split_bit,
                left,
                right,
            } => {
                buf.push(1u8); // tag
                buf.push(*partition_depth);
                buf.extend_from_slice(&partition_path.to_le_bytes());
                buf.push(*split_bit);
                buf.push(0u8); // reserved
                let left_bytes = left.to_bytes();
                let left_len = left_bytes.len() as u32;
                buf.extend_from_slice(&left_len.to_le_bytes());
                buf.extend_from_slice(&left_bytes);
                buf.extend_from_slice(&right.to_bytes());
            }
        }
        buf
    }

    /// Deserialize a Volume node.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        let (vol, consumed) = Self::parse(data)?;
        if consumed != data.len() {
            return Err(BookError::BadFormat);
        }
        Ok(vol)
    }

    /// Internal parser that returns the Volume and byte count consumed.
    fn parse(data: &[u8]) -> Result<(Self, usize), BookError> {
        if data.len() < 8 {
            return Err(BookError::TooShort);
        }
        let tag = data[0];
        let partition_depth = data[1];
        let partition_path =
            u32::from_le_bytes(data[2..6].try_into().map_err(|_| BookError::TooShort)?);

        match tag {
            0 => {
                // Leaf
                if data.len() < 10 {
                    return Err(BookError::TooShort);
                }
                let book_count =
                    u32::from_le_bytes(data[6..10].try_into().map_err(|_| BookError::TooShort)?)
                        as usize;
                let mut pos = 10;
                let mut books = Vec::with_capacity(book_count);
                for _ in 0..book_count {
                    if pos + 4 > data.len() {
                        return Err(BookError::TooShort);
                    }
                    let book_len = u32::from_le_bytes(
                        data[pos..pos + 4]
                            .try_into()
                            .map_err(|_| BookError::TooShort)?,
                    ) as usize;
                    pos += 4;
                    if pos + book_len > data.len() {
                        return Err(BookError::TooShort);
                    }
                    let book = deserialize_book(&data[pos..pos + book_len])?;
                    books.push(book);
                    pos += book_len;
                }
                Ok((
                    Volume::Leaf {
                        partition_depth,
                        partition_path,
                        books,
                    },
                    pos,
                ))
            }
            1 => {
                // Split
                if data.len() < 12 {
                    return Err(BookError::TooShort);
                }
                let split_bit = data[6];
                // data[7] = reserved
                let left_len =
                    u32::from_le_bytes(data[8..12].try_into().map_err(|_| BookError::TooShort)?)
                        as usize;
                let left_start = 12;
                if left_start + left_len > data.len() {
                    return Err(BookError::TooShort);
                }
                let (left, left_consumed) = Self::parse(&data[left_start..left_start + left_len])?;
                if left_consumed != left_len {
                    return Err(BookError::BadFormat);
                }
                let right_start = left_start + left_len;
                if right_start >= data.len() {
                    return Err(BookError::TooShort);
                }
                let (right, right_consumed) = Self::parse(&data[right_start..])?;
                let total = right_start + right_consumed;
                Ok((
                    Volume::Split {
                        partition_depth,
                        partition_path,
                        split_bit,
                        left: Box::new(left),
                        right: Box::new(right),
                    },
                    total,
                ))
            }
            _ => Err(BookError::BadFormat),
        }
    }
}

/// Serialize a Book for Volume leaf storage.
///
/// Format:
/// ```text
///   32 bytes: cid
///    4 bytes: blob_size (u32 LE)
///    2 bytes: page_count (u16 LE)
///    2 bytes: reserved (0)
///   For each page (page_count entries):
///     4 × 4 bytes: PageAddr.0 for each algorithm variant (u32 LE) = 16 bytes per page
/// ```
fn serialize_book(book: &Book) -> Vec<u8> {
    let pc = book.page_count();
    let size = 32 + 4 + 2 + 2 + pc * ALGO_COUNT * 4;
    let mut buf = Vec::with_capacity(size);

    buf.extend_from_slice(&book.cid);
    buf.extend_from_slice(&book.blob_size.to_le_bytes());
    buf.extend_from_slice(&(pc as u16).to_le_bytes());
    buf.extend_from_slice(&[0u8; 2]); // reserved

    for page_addrs in &book.pages {
        for addr in page_addrs {
            buf.extend_from_slice(&addr.0.to_le_bytes());
        }
    }

    buf
}

/// Deserialize a Book from Volume leaf storage.
fn deserialize_book(data: &[u8]) -> Result<Book, BookError> {
    // Minimum header: 32 (cid) + 4 (blob_size) + 2 (page_count) + 2 (reserved) = 40
    if data.len() < 40 {
        return Err(BookError::TooShort);
    }

    let mut cid = [0u8; 32];
    cid.copy_from_slice(&data[..32]);

    let blob_size = u32::from_le_bytes(data[32..36].try_into().map_err(|_| BookError::TooShort)?);
    if blob_size as usize > crate::addr::BOOK_MAX_SIZE {
        return Err(BookError::BadFormat);
    }

    let page_count =
        u16::from_le_bytes(data[36..38].try_into().map_err(|_| BookError::TooShort)?) as usize;

    if page_count > crate::addr::PAGES_PER_BOOK {
        return Err(BookError::BadFormat);
    }

    // blob_size must be consistent with page_count
    if blob_size as usize > page_count * crate::addr::PAGE_SIZE {
        return Err(BookError::BadFormat);
    }

    // data[38..40] = reserved, skip

    let pages_start = 40;
    let pages_bytes = page_count * ALGO_COUNT * 4;
    if data.len() < pages_start + pages_bytes {
        return Err(BookError::TooShort);
    }
    if data.len() != pages_start + pages_bytes {
        return Err(BookError::BadFormat);
    }

    let mut pages = Vec::with_capacity(page_count);
    let mut pos = pages_start;
    for _ in 0..page_count {
        let mut addrs = [PageAddr(0); ALGO_COUNT];
        for addr in &mut addrs {
            let raw = u32::from_le_bytes(
                data[pos..pos + 4]
                    .try_into()
                    .map_err(|_| BookError::TooShort)?,
            );
            *addr = PageAddr(raw);
            if !addr.verify_checksum() {
                return Err(BookError::BadFormat);
            }
            pos += 4;
        }
        pages.push(addrs);
    }

    Ok(Book {
        cid,
        pages,
        blob_size,
    })
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
    use crate::addr::PAGE_SIZE;

    fn sample_book() -> Book {
        Book::from_blob([0xAA; 32], &[0x42u8; PAGE_SIZE * 2]).unwrap()
    }

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
    fn leaf_volume_page_count() {
        let book = sample_book();
        assert_eq!(book.page_count(), 2);
        let vol = Volume::leaf(0, 0, alloc::vec![book]);
        assert_eq!(vol.page_count(), 2);
        assert_eq!(vol.book_count(), 1);
    }

    #[test]
    fn split_volume_page_count() {
        let left = Volume::leaf(1, 0, alloc::vec![sample_book()]);
        let right = Volume::leaf(1, 1, alloc::vec![sample_book()]);
        let split = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(split.page_count(), 4); // 2 pages per book × 2 books
        assert_eq!(split.book_count(), 2);
        assert_eq!(split.depth(), 0);
    }

    #[test]
    fn volume_depth_and_path() {
        let vol = Volume::leaf(3, 0b101, Vec::new());
        assert_eq!(vol.depth(), 3);
        assert_eq!(vol.path(), 0b101);
    }

    #[test]
    fn leaf_volume_round_trip() {
        let vol = Volume::leaf(2, 0b10, alloc::vec![sample_book()]);
        let bytes = vol.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(vol, restored);
    }

    #[test]
    fn split_volume_round_trip() {
        let left = Volume::leaf(1, 0, alloc::vec![sample_book()]);
        let right = Volume::leaf(1, 1, alloc::vec![sample_book()]);
        let split = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        let bytes = split.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(split, restored);
    }

    #[test]
    fn volume_from_bytes_too_short() {
        assert!(Volume::from_bytes(&[0u8; 3]).is_err());
    }

    #[test]
    fn volume_from_bytes_unknown_tag() {
        let mut data = [0u8; 10];
        data[0] = 99; // unknown tag
        assert_eq!(Volume::from_bytes(&data), Err(BookError::BadFormat));
    }

    #[test]
    fn volume_from_bytes_rejects_trailing_bytes() {
        let vol = Volume::leaf(0, 0, Vec::new());
        let mut bytes = vol.to_bytes();
        bytes.push(0xFF); // trailing garbage
        assert_eq!(Volume::from_bytes(&bytes), Err(BookError::BadFormat));
    }

    #[test]
    fn split_from_bytes_rejects_trailing_bytes() {
        let left = Volume::leaf(1, 0, Vec::new());
        let right = Volume::leaf(1, 1, Vec::new());
        let split = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        let mut bytes = split.to_bytes();
        bytes.push(0xFF); // trailing garbage
        assert_eq!(Volume::from_bytes(&bytes), Err(BookError::BadFormat));
    }

    #[test]
    fn empty_leaf_round_trip() {
        let vol = Volume::leaf(5, 0b11010, Vec::new());
        let bytes = vol.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(vol, restored);
        assert_eq!(restored.page_count(), 0);
        assert_eq!(restored.book_count(), 0);
    }

    #[test]
    fn book_serialization_round_trip() {
        let book = sample_book();
        let serialized = serialize_book(&book);
        let deserialized = deserialize_book(&serialized).unwrap();
        assert_eq!(book, deserialized);
    }

    #[test]
    fn empty_book_serialization_round_trip() {
        let book = Book::from_blob([0xBB; 32], &[]).unwrap();
        assert_eq!(book.page_count(), 0);
        let serialized = serialize_book(&book);
        let deserialized = deserialize_book(&serialized).unwrap();
        assert_eq!(book, deserialized);
    }

    #[test]
    fn max_partition_depth_value() {
        assert_eq!(MAX_PARTITION_DEPTH, 228);
    }
}

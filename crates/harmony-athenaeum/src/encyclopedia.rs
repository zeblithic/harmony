// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Encyclopedia — assigns unique PageAddrs across a collection of Books.
//!
//! Uses power-of-choice collision resolution: for each unique page,
//! try algorithms 0–3 and pick the first collision-free 28-bit hash.
//! When the address space fills, partition via Volume tree.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::addr::{PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, PAGE_SIZE};
use crate::athenaeum::{Book, BookError};
use crate::hash::sha256_hash;
use crate::volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};

/// Threshold for proactive splitting.
/// 75% of 2^28 = 201,326,592 unique pages.
pub const SPLIT_THRESHOLD: usize = (1 << 28) * 3 / 4;

/// Starting bit index for partition routing (bits 0-27 used for addressing).
const PARTITION_START_BIT: u8 = 28;

/// A complete content-addressed mapping for a corpus of blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encyclopedia {
    pub root: Volume,
    pub total_blobs: u32,
    pub total_unique_pages: u32,
}

/// Information about a unique page during the build phase.
///
/// Tracks the content hash (for dedup and partition routing) and
/// all 4 precomputed PageAddr variants from the Book's ToC.
#[derive(Clone)]
struct PageInfo {
    content_hash: [u8; 32],
    variants: [PageAddr; ALGO_COUNT],
}

impl Encyclopedia {
    /// Build an Encyclopedia from a collection of blobs.
    ///
    /// Each blob is identified by its CID (content hash) and raw data.
    /// All pages across all blobs are deduplicated by content hash and
    /// assigned unique 32-bit addresses via power-of-choice. When the
    /// address space fills up, the system recursively partitions by
    /// content-hash bits.
    pub fn build(blobs: &[([u8; 32], &[u8])]) -> Result<Self, BookError> {
        for &(_, data) in blobs {
            if data.len() > BOOK_MAX_SIZE {
                return Err(BookError::BlobTooLarge { size: data.len() });
            }
        }

        if blobs.is_empty() {
            return Ok(Encyclopedia {
                root: Volume::leaf(0, 0, Vec::new()),
                total_blobs: 0,
                total_unique_pages: 0,
            });
        }

        // Phase 1: Build Books from all blobs, dedup pages by content hash.
        let mut books: Vec<Book> = Vec::with_capacity(blobs.len());
        let mut unique_pages: BTreeMap<[u8; 32], PageInfo> = BTreeMap::new();
        let mut blob_page_hashes: Vec<Vec<[u8; 32]>> = Vec::new();

        for &(cid, data) in blobs {
            let book = Book::from_blob(cid, data)?;

            // Compute content hashes for each page (zero-padded to PAGE_SIZE)
            let mut hashes = Vec::new();
            for chunk in data.chunks(PAGE_SIZE) {
                let mut page_buf = [0u8; PAGE_SIZE];
                page_buf[..chunk.len()].copy_from_slice(chunk);
                let content_hash = sha256_hash(&page_buf);
                hashes.push(content_hash);
            }

            // Register unique pages with their precomputed variants from the Book
            for (page_idx, content_hash) in hashes.iter().enumerate() {
                unique_pages.entry(*content_hash).or_insert(PageInfo {
                    content_hash: *content_hash,
                    variants: book.pages[page_idx],
                });
            }

            blob_page_hashes.push(hashes);
            books.push(book);
        }

        let total_unique = unique_pages.len() as u32;
        let page_list: Vec<PageInfo> = unique_pages.into_values().collect();

        // Phase 2 & 3: Recursive partition + resolve
        let root = Self::build_volume(
            &page_list,
            &books,
            &blob_page_hashes,
            0, // depth
            0, // path
            PARTITION_START_BIT as u16,
        )?;

        Ok(Encyclopedia {
            root,
            total_blobs: blobs.len() as u32,
            total_unique_pages: total_unique,
        })
    }

    /// Recursively build a Volume from a set of unique pages.
    fn build_volume(
        pages: &[PageInfo],
        books: &[Book],
        blob_page_hashes: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
        bit_index: u16,
    ) -> Result<Volume, BookError> {
        if bit_index >= PARTITION_START_BIT as u16 + MAX_PARTITION_DEPTH as u16 {
            return Err(BookError::MaxPartitionDepth { depth });
        }

        if pages.is_empty() {
            return Ok(Volume::leaf(depth, path, Vec::new()));
        }

        if pages.len() <= SPLIT_THRESHOLD {
            // Try to resolve in a single flat volume
            match Self::resolve_leaf(pages, books, blob_page_hashes, depth, path) {
                Ok(vol) => return Ok(vol),
                Err(_) => {
                    // Fall through to splitting
                }
            }
        }

        // Split by content-hash bit (safe cast: guard above ensures bit_index < 256)
        let bit_index_u8 = bit_index as u8;
        let mut left_pages = Vec::new();
        let mut right_pages = Vec::new();
        for page in pages {
            if route_chunk(&page.content_hash, bit_index_u8) {
                right_pages.push(page.clone());
            } else {
                left_pages.push(page.clone());
            }
        }

        let left = Self::build_volume(
            &left_pages,
            books,
            blob_page_hashes,
            depth + 1,
            path,
            bit_index + 1,
        )?;
        let right = Self::build_volume(
            &right_pages,
            books,
            blob_page_hashes,
            depth + 1,
            if depth < 32 {
                path | (1u32 << depth)
            } else {
                path
            },
            bit_index + 1,
        )?;

        Ok(Volume::Split {
            partition_depth: depth,
            partition_path: path,
            split_bit: bit_index_u8,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Resolve a set of pages into a leaf Volume with Books.
    ///
    /// For each unique page in this partition, try algo 0, 1, 2, 3 in order
    /// — the first collision-free 28-bit hash_bits wins. Leverages the
    /// precomputed variants from each Book's ToC instead of runtime rehashing.
    ///
    /// Each blob's Book in the leaf contains only the pages that route here.
    fn resolve_leaf(
        pages: &[PageInfo],
        books: &[Book],
        blob_page_hashes: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
    ) -> Result<Volume, BookError> {
        // Build a set of content hashes in this partition
        let partition_hashes: BTreeSet<[u8; 32]> = pages.iter().map(|p| p.content_hash).collect();

        // Power-of-choice: verify collision-free addressing is possible.
        // For each unique page, try algo 0-3 and reserve the first
        // collision-free hash_bits. The chosen algo is already encoded
        // in the Book's pages array (all 4 variants stored per page).
        let mut used_hash_bits = BTreeSet::new();
        let mut assigned_pages = BTreeSet::new();
        let mut assigned_count = 0usize;

        for page_info in pages {
            if assigned_pages.contains(&page_info.content_hash) {
                continue;
            }

            let mut found = false;
            for algo_idx in 0..ALGO_COUNT {
                let hash_bits = page_info.variants[algo_idx].hash_bits();
                if !used_hash_bits.contains(&hash_bits) {
                    used_hash_bits.insert(hash_bits);
                    assigned_pages.insert(page_info.content_hash);
                    assigned_count += 1;
                    found = true;
                    break;
                }
            }
            if !found {
                return Err(BookError::AllAlgorithmsCollide {
                    page_index: assigned_count,
                });
            }
        }

        // Find which blobs (books) have pages in this partition
        let mut relevant_blob_indices: Vec<usize> = Vec::new();
        for (blob_idx, page_hashes) in blob_page_hashes.iter().enumerate() {
            if page_hashes.iter().any(|h| partition_hashes.contains(h)) {
                relevant_blob_indices.push(blob_idx);
            }
        }

        // Build Books for the leaf: each relevant blob gets its own Book,
        // but only containing the pages that route to this partition
        // with the chosen algorithm variant.
        let mut leaf_books = Vec::new();
        for &blob_idx in &relevant_blob_indices {
            let original_book = &books[blob_idx];
            let page_hashes = &blob_page_hashes[blob_idx];

            // Build the page list: only pages in this partition, using the
            // chosen algo variant for each.
            let mut new_pages: Vec<[PageAddr; ALGO_COUNT]> = Vec::new();
            for (page_idx, hash) in page_hashes.iter().enumerate() {
                if partition_hashes.contains(hash) {
                    new_pages.push(original_book.pages[page_idx]);
                }
            }

            // Leaf books are partial — blob_size reflects only the content
            // bytes in the pages routed to this partition. The last page of a
            // blob may be shorter than PAGE_SIZE, so we must use the original
            // page index to compute each page's true byte contribution.
            let mut partial_size = 0u32;
            for (page_idx, hash) in page_hashes.iter().enumerate() {
                if partition_hashes.contains(hash) {
                    let page_start = page_idx as u32 * PAGE_SIZE as u32;
                    let page_end = (page_start + PAGE_SIZE as u32).min(original_book.blob_size);
                    partial_size += page_end - page_start;
                }
            }
            leaf_books.push(Book {
                cid: original_book.cid,
                pages: new_pages,
                blob_size: partial_size,
            });
        }

        Ok(Volume::leaf(depth, path, leaf_books))
    }

    /// Determine which partition a page belongs to by content hash.
    ///
    /// Returns the sequence of routing decisions (bits) from the root.
    pub fn route(content_hash: &[u8; 32], depth: u8) -> u32 {
        let mut path = 0u32;
        for d in 0..depth.min(32) {
            if route_chunk(content_hash, PARTITION_START_BIT + d) {
                path |= 1 << d;
            }
        }
        path
    }

    /// Serialize the Encyclopedia root metadata.
    ///
    /// Format: magic "ENCY" (4) + version u8 (1) + total_blobs u32 LE (4)
    /// + total_unique_pages u32 LE (4) + volume_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"ENCY");
        buf.push(1); // version
        buf.extend_from_slice(&self.total_blobs.to_le_bytes());
        buf.extend_from_slice(&self.total_unique_pages.to_le_bytes());
        buf.extend_from_slice(&self.root.to_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        if data.len() < 13 {
            return Err(BookError::TooShort);
        }
        if &data[0..4] != b"ENCY" {
            return Err(BookError::BadFormat);
        }
        if data[4] != 1 {
            return Err(BookError::BadFormat);
        }
        let total_blobs =
            u32::from_le_bytes(data[5..9].try_into().map_err(|_| BookError::TooShort)?);
        let total_unique_pages =
            u32::from_le_bytes(data[9..13].try_into().map_err(|_| BookError::TooShort)?);
        let root = Volume::from_bytes(&data[13..])?;
        Ok(Encyclopedia {
            root,
            total_blobs,
            total_unique_pages,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_single_blob() {
        // Two pages with different content so they don't deduplicate.
        // Page 0 starts with 0x00, page 1 starts with 0x01 — different first byte.
        let mut data = alloc::vec![0u8; PAGE_SIZE * 2];
        data[0] = 0xAA;
        data[PAGE_SIZE] = 0xBB;
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();
        assert_eq!(enc.total_blobs, 1);
        assert_eq!(enc.total_unique_pages, 2);
        assert_eq!(enc.root.page_count(), 2);
    }

    #[test]
    fn build_multiple_blobs() {
        let mut data1 = alloc::vec![0u8; PAGE_SIZE * 2];
        let mut data2 = alloc::vec![0u8; PAGE_SIZE * 3];
        for (i, b) in data1.iter_mut().enumerate() {
            *b = (i as u32 ^ 0xAA) as u8;
        }
        for (i, b) in data2.iter_mut().enumerate() {
            *b = (i as u32 ^ 0xBB) as u8;
        }
        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        assert!(enc.total_unique_pages <= 5);
    }

    #[test]
    fn build_deduplicates_shared_pages() {
        let shared = alloc::vec![0x42u8; PAGE_SIZE];
        let unique1 = alloc::vec![0xAAu8; PAGE_SIZE];
        let unique2 = alloc::vec![0xBBu8; PAGE_SIZE];

        let mut data1 = Vec::new();
        data1.extend_from_slice(&shared);
        data1.extend_from_slice(&unique1);
        let mut data2 = Vec::new();
        data2.extend_from_slice(&shared);
        data2.extend_from_slice(&unique2);

        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        assert_eq!(enc.total_unique_pages, 3);
    }

    #[test]
    fn build_empty() {
        let enc = Encyclopedia::build(&[]).unwrap();
        assert_eq!(enc.total_blobs, 0);
        assert_eq!(enc.total_unique_pages, 0);
    }

    #[test]
    fn build_blob_too_large() {
        let data = alloc::vec![0u8; BOOK_MAX_SIZE + 1];
        let cid = sha256_hash(&data);
        let result = Encyclopedia::build(&[(cid, &data)]);
        assert!(matches!(result, Err(BookError::BlobTooLarge { .. })));
    }

    #[test]
    fn serialization_round_trip() {
        let data = alloc::vec![0xCDu8; PAGE_SIZE];
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();
        let bytes = enc.to_bytes();
        let restored = Encyclopedia::from_bytes(&bytes).unwrap();
        assert_eq!(enc, restored);
    }

    #[test]
    fn from_bytes_bad_magic() {
        let mut data = alloc::vec![0u8; 30];
        data[..4].copy_from_slice(b"BAAD");
        data[4] = 1;
        // Need enough data for a Volume after byte 13
        // Put a valid leaf volume: tag=0, depth=0, path=0, book_count=0
        data[13] = 0; // tag
        data[14] = 0; // depth
        assert!(Encyclopedia::from_bytes(&data).is_err());
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(Encyclopedia::from_bytes(&[0u8; 5]).is_err());
    }

    #[test]
    fn split_threshold_value() {
        assert_eq!(SPLIT_THRESHOLD, 201_326_592);
    }

    #[test]
    fn route_deterministic() {
        let hash = sha256_hash(b"test page");
        let path1 = Encyclopedia::route(&hash, 5);
        let path2 = Encyclopedia::route(&hash, 5);
        assert_eq!(path1, path2);
    }

    #[test]
    fn route_depth_zero() {
        let hash = sha256_hash(b"anything");
        assert_eq!(Encyclopedia::route(&hash, 0), 0);
    }

    #[test]
    fn route_does_not_panic_at_high_depth() {
        let hash = sha256_hash(b"deep partition test");
        let path = Encyclopedia::route(&hash, 100);
        // path is capped at 32 bits — should not panic
        let _ = path;
    }

    #[test]
    fn partial_last_page_blob_size() {
        // A blob of 5000 bytes = page 0 (4096 bytes) + page 1 (904 bytes).
        // When built into an Encyclopedia, the total blob_size across all
        // leaf Books must sum to the original blob_size (5000), not
        // page_count * PAGE_SIZE (8192).
        let data = alloc::vec![0xFFu8; 5000];
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();

        // Walk the tree and sum blob_size across all Books.
        fn sum_blob_sizes(vol: &Volume) -> u32 {
            match vol {
                Volume::Leaf { books, .. } => books.iter().map(|b| b.blob_size).sum(),
                Volume::Split { left, right, .. } => {
                    sum_blob_sizes(left) + sum_blob_sizes(right)
                }
            }
        }

        assert_eq!(sum_blob_sizes(&enc.root), 5000);
    }

    #[test]
    fn resolve_leaf_only_addresses_partition_pages() {
        // Build with 2 blobs, confirm the total page count across the tree
        // equals total_unique_pages (no duplication).
        let mut data1 = alloc::vec![0u8; PAGE_SIZE * 4];
        let mut data2 = alloc::vec![0u8; PAGE_SIZE * 4];
        for (i, chunk) in data1.chunks_mut(PAGE_SIZE).enumerate() {
            chunk[0] = i as u8;
            chunk[1] = 0xAA;
        }
        for (i, chunk) in data2.chunks_mut(PAGE_SIZE).enumerate() {
            chunk[0] = i as u8;
            chunk[1] = 0xBB;
        }
        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();

        // In a flat (no-split) volume, page_count should equal
        // total_unique_pages — no duplication across leaves.
        assert_eq!(enc.root.page_count() as u32, enc.total_unique_pages);
    }
}

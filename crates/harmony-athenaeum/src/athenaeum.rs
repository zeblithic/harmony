// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Book — a single CID blob stored as fixed-size 4KB pages.
//!
//! Each page has all 4 algorithm variants precomputed, enabling
//! collision-free routing without runtime rehashing.

use alloc::vec;
use alloc::vec::Vec;

use crate::addr::{
    Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE,
};

/// Error when constructing or reassembling a Book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookError {
    /// Blob exceeds the 1MB maximum.
    BlobTooLarge { size: usize },
    /// All 4 algorithm variants produced the same address for a page.
    AllAlgorithmsCollide { page_index: usize },
    /// A required page was not available during reassembly.
    MissingPage { page_index: u8 },
    /// Partition tree exceeded maximum depth — address space exhausted.
    MaxPartitionDepth { depth: u8 },
    /// Serialized data has an invalid format.
    BadFormat,
    /// Serialized data is too short to be valid.
    TooShort,
}

/// A Book maps a single CID blob into up to 256 fixed-size 4KB pages.
///
/// Every page has all 4 algorithm variants precomputed, stored as
/// `[PageAddr; ALGO_COUNT]` in Algorithm selector order:
/// `[Sha256Msb, Sha256Lsb, Sha224Msb, Sha224Lsb]`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Book {
    /// 256-bit blob identifier (content ID).
    pub cid: [u8; 32],
    /// Page addresses — up to 256 entries, each with all 4 algorithm variants.
    pub pages: Vec<[PageAddr; ALGO_COUNT]>,
    /// Actual byte count of the original blob (at most 1MB).
    pub blob_size: u32,
    /// When true, page 0 is an embedded Table of Contents.
    pub self_indexing: bool,
}

impl Book {
    /// Build a Book by splitting a blob into 4KB pages and computing all
    /// algorithm variants for each page.
    ///
    /// - Empty data produces a Book with 0 pages.
    /// - The last page is zero-padded to a full 4KB before hashing.
    /// - `page_count = ceil(blob_size / PAGE_SIZE)`.
    pub fn from_blob(cid: [u8; 32], data: &[u8]) -> Result<Self, BookError> {
        if data.len() > BOOK_MAX_SIZE {
            return Err(BookError::BlobTooLarge { size: data.len() });
        }

        if data.is_empty() {
            return Ok(Book {
                cid,
                pages: Vec::new(),
                blob_size: 0,
                self_indexing: false,
            });
        }

        let page_count = data.len().div_ceil(PAGE_SIZE);
        let mut pages = Vec::with_capacity(page_count);

        for (i, chunk) in data.chunks(PAGE_SIZE).enumerate() {
            let mut page_buf = [0u8; PAGE_SIZE];
            page_buf[..chunk.len()].copy_from_slice(chunk);

            let variants = [
                PageAddr::from_data(&page_buf, Algorithm::Sha256Msb),
                PageAddr::from_data(&page_buf, Algorithm::Sha256Lsb),
                PageAddr::from_data(&page_buf, Algorithm::Sha224Msb),
                PageAddr::from_data(&page_buf, Algorithm::Sha224Lsb),
            ];

            // Verify that at least one variant is unique (non-degenerate).
            // In practice, 4 independent hashes producing identical 28-bit
            // values is astronomically unlikely, but we check anyway.
            let all_same = variants[1..]
                .iter()
                .all(|v| v.hash_bits() == variants[0].hash_bits());
            if all_same {
                return Err(BookError::AllAlgorithmsCollide { page_index: i });
            }

            pages.push(variants);
        }

        Ok(Book {
            cid,
            pages,
            blob_size: data.len() as u32,
            self_indexing: false,
        })
    }

    /// Number of pages in this book.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Whether this book embeds its ToC at page 0.
    pub fn is_self_indexing(&self) -> bool {
        self.self_indexing
    }

    /// Number of data pages (excludes the ToC page for self-indexing books).
    pub fn data_page_count(&self) -> usize {
        if self.self_indexing {
            self.pages.len().saturating_sub(1)
        } else {
            self.pages.len()
        }
    }

    /// Slice of data page addresses (excludes ToC for self-indexing books).
    pub fn data_pages(&self) -> &[[PageAddr; ALGO_COUNT]] {
        if self.self_indexing && !self.pages.is_empty() {
            &self.pages[1..]
        } else {
            &self.pages
        }
    }

    /// Returns the ToC page bytes if this is a self-indexing book.
    pub fn toc_page(&self) -> Option<Vec<u8>> {
        if self.self_indexing {
            Some(self.toc())
        } else {
            None
        }
    }

    /// Generate a 4KB Table of Contents.
    ///
    /// Layout: 4 sections (one per algorithm) x 256 entries x 4 bytes (LE u32).
    ///
    /// - Section 0 (Sha256Msb): bytes 0..1024
    /// - Section 1 (Sha256Lsb): bytes 1024..2048
    /// - Section 2 (Sha224Msb): bytes 2048..3072
    /// - Section 3 (Sha224Lsb): bytes 3072..4096
    ///
    /// Existing pages write `PageAddr.0` as u32 LE.
    /// Non-existing pages (past page_count) write `NULL_PAGE` as u32 LE.
    pub fn toc(&self) -> Vec<u8> {
        let mut buf = vec![0u8; PAGE_SIZE];

        for algo_idx in 0..ALGO_COUNT {
            let section_offset = algo_idx * PAGES_PER_BOOK * 4;
            for page_idx in 0..PAGES_PER_BOOK {
                let entry_offset = section_offset + page_idx * 4;
                let value = if page_idx < self.pages.len() {
                    self.pages[page_idx][algo_idx].0
                } else {
                    NULL_PAGE
                };
                buf[entry_offset..entry_offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }

        buf
    }

    /// Split the original blob into page-sized buffers (4KB each).
    ///
    /// The last page is zero-padded to a full 4KB.
    pub fn page_data_from_blob(&self, data: &[u8]) -> Vec<Vec<u8>> {
        debug_assert_eq!(
            data.len(),
            self.blob_size as usize,
            "data length does not match blob_size"
        );
        let mut pages = Vec::new();
        for chunk in data.chunks(PAGE_SIZE) {
            let mut page_buf = vec![0u8; PAGE_SIZE];
            page_buf[..chunk.len()].copy_from_slice(chunk);
            pages.push(page_buf);
        }
        pages
    }

    /// Reassemble the original blob by fetching pages by index.
    ///
    /// Pages are fetched by index (0..page_count), concatenated, and
    /// truncated to `blob_size`. A missing page returns `MissingPage` error.
    pub fn reassemble(&self, fetch: impl Fn(u8) -> Option<Vec<u8>>) -> Result<Vec<u8>, BookError> {
        let mut result = Vec::with_capacity(self.blob_size as usize);

        for i in 0..self.pages.len() {
            let page_data = fetch(i as u8).ok_or(BookError::MissingPage {
                page_index: i as u8,
            })?;
            if page_data.len() != PAGE_SIZE {
                return Err(BookError::BadFormat);
            }
            result.extend_from_slice(&page_data);
        }

        result.truncate(self.blob_size as usize);
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cid() -> [u8; 32] {
        crate::hash::sha256_hash(b"test book cid")
    }

    #[test]
    fn from_blob_single_page() {
        let data = vec![0xABu8; 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.blob_size, 100);
    }

    #[test]
    fn from_blob_full_page() {
        let data = vec![0xCDu8; PAGE_SIZE];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.blob_size, PAGE_SIZE as u32);
    }

    #[test]
    fn from_blob_multiple_pages() {
        // 3 full pages + 1 partial = 4 pages
        let data = vec![0xEFu8; PAGE_SIZE * 3 + 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.page_count(), 4);
        assert_eq!(book.blob_size, (PAGE_SIZE * 3 + 100) as u32);
    }

    #[test]
    fn from_blob_max_size() {
        let data = vec![0u8; BOOK_MAX_SIZE];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.page_count(), PAGES_PER_BOOK);
        assert_eq!(book.blob_size, BOOK_MAX_SIZE as u32);
    }

    #[test]
    fn from_blob_too_large() {
        let data = vec![0u8; BOOK_MAX_SIZE + 1];
        let err = Book::from_blob(test_cid(), &data).unwrap_err();
        assert_eq!(
            err,
            BookError::BlobTooLarge {
                size: BOOK_MAX_SIZE + 1
            }
        );
    }

    #[test]
    fn from_blob_empty() {
        let book = Book::from_blob(test_cid(), &[]).unwrap();
        assert_eq!(book.page_count(), 0);
        assert_eq!(book.blob_size, 0);
    }

    #[test]
    fn from_blob_cid_preserved() {
        let cid = test_cid();
        let data = vec![1u8; 4096];
        let book = Book::from_blob(cid, &data).unwrap();
        assert_eq!(book.cid, cid);
    }

    #[test]
    fn last_page_zero_padded_produces_valid_addr() {
        // 100 bytes of data → 1 page, zero-padded to 4KB
        let data = vec![0x42u8; 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.page_count(), 1);

        // All 4 variants should have valid checksums
        for addr in &book.pages[0] {
            assert!(addr.verify_checksum());
        }

        // Verify the address matches the zero-padded buffer
        let mut padded = [0u8; PAGE_SIZE];
        padded[..100].copy_from_slice(&data);
        for (i, algo) in Algorithm::ALL.iter().enumerate() {
            let expected = PageAddr::from_data(&padded, *algo);
            assert_eq!(book.pages[0][i], expected);
        }
    }

    #[test]
    fn all_four_variants_precomputed() {
        let data = vec![0xFFu8; PAGE_SIZE];
        let book = Book::from_blob(test_cid(), &data).unwrap();

        // Verify the order matches Algorithm::ALL
        let expected_algos = [
            Algorithm::Sha256Msb,
            Algorithm::Sha256Lsb,
            Algorithm::Sha224Msb,
            Algorithm::Sha224Lsb,
        ];

        for (i, expected_algo) in expected_algos.iter().enumerate() {
            assert_eq!(
                book.pages[0][i].algorithm(),
                *expected_algo,
                "index {i} should be {expected_algo:?}"
            );
        }

        // All variants should have valid checksums
        for addr in &book.pages[0] {
            assert!(addr.verify_checksum());
        }

        // All 4 variants should produce different raw values (different algo bits at minimum)
        let raw_values: Vec<u32> = book.pages[0].iter().map(|a| a.0).collect();
        for i in 0..raw_values.len() {
            for j in (i + 1)..raw_values.len() {
                assert_ne!(
                    raw_values[i], raw_values[j],
                    "variants {i} and {j} should differ"
                );
            }
        }
    }

    #[test]
    fn toc_is_4kb() {
        let data = vec![0u8; PAGE_SIZE * 3];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let toc = book.toc();
        assert_eq!(toc.len(), PAGE_SIZE);
    }

    #[test]
    fn toc_layout_sections() {
        let data = vec![0xAAu8; PAGE_SIZE * 2];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let toc = book.toc();

        // Check section offsets and values for existing pages
        for algo_idx in 0..ALGO_COUNT {
            let section_offset = algo_idx * PAGES_PER_BOOK * 4;

            // First 2 pages should have valid addresses
            for page_idx in 0..2 {
                let entry_offset = section_offset + page_idx * 4;
                let value = u32::from_le_bytes([
                    toc[entry_offset],
                    toc[entry_offset + 1],
                    toc[entry_offset + 2],
                    toc[entry_offset + 3],
                ]);
                let expected = book.pages[page_idx][algo_idx].0;
                assert_eq!(
                    value, expected,
                    "section {algo_idx}, page {page_idx}: expected {expected:#x}, got {value:#x}"
                );
                // Valid addresses should pass checksum
                let addr = PageAddr(value);
                assert!(addr.verify_checksum());
            }

            // Pages 2..256 should be NULL_PAGE
            for page_idx in 2..PAGES_PER_BOOK {
                let entry_offset = section_offset + page_idx * 4;
                let value = u32::from_le_bytes([
                    toc[entry_offset],
                    toc[entry_offset + 1],
                    toc[entry_offset + 2],
                    toc[entry_offset + 3],
                ]);
                assert_eq!(
                    value, NULL_PAGE,
                    "section {algo_idx}, page {page_idx}: expected NULL_PAGE, got {value:#x}"
                );
            }
        }
    }

    #[test]
    fn reassemble_round_trip() {
        let mut data = vec![0u8; PAGE_SIZE * 3 + 500];
        for (i, b) in data.iter_mut().enumerate() {
            let pos = i as u32;
            *b = (pos ^ (pos >> 8) ^ (pos >> 16)) as u8;
        }
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let page_bufs = book.page_data_from_blob(&data);

        let reassembled = book
            .reassemble(|idx| page_bufs.get(idx as usize).cloned())
            .unwrap();

        assert_eq!(reassembled.len(), data.len());
        assert_eq!(reassembled, data);
    }

    #[test]
    fn reassemble_missing_page_fails() {
        let data = vec![1u8; PAGE_SIZE * 2];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let result = book.reassemble(|_idx| None);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err(),
            BookError::MissingPage { page_index: 0 }
        );
    }

    #[test]
    fn blob_size_preserved() {
        let data = vec![0xFFu8; 5000];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.blob_size, 5000);
    }

    #[test]
    fn toc_empty_book() {
        let book = Book::from_blob(test_cid(), &[]).unwrap();
        let toc = book.toc();
        assert_eq!(toc.len(), PAGE_SIZE);
        // All entries should be NULL_PAGE
        for i in 0..PAGES_PER_BOOK * ALGO_COUNT {
            let offset = i * 4;
            let value = u32::from_le_bytes([
                toc[offset],
                toc[offset + 1],
                toc[offset + 2],
                toc[offset + 3],
            ]);
            assert_eq!(value, NULL_PAGE, "entry {i} should be NULL_PAGE");
        }
    }

    #[test]
    fn raw_book_is_not_self_indexing() {
        let data = vec![0xABu8; 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert!(!book.is_self_indexing());
    }

    #[test]
    fn raw_book_data_page_count_equals_page_count() {
        let data = vec![0xABu8; PAGE_SIZE * 3];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert_eq!(book.data_page_count(), 3);
        assert_eq!(book.data_page_count(), book.page_count());
    }

    #[test]
    fn raw_book_data_pages_returns_all() {
        let data = vec![0xABu8; PAGE_SIZE * 2];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let dp = book.data_pages();
        assert_eq!(dp.len(), 2);
        assert_eq!(dp[0], book.pages[0]);
        assert_eq!(dp[1], book.pages[1]);
    }

    #[test]
    fn raw_book_toc_page_returns_none() {
        let data = vec![0xABu8; 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        assert!(book.toc_page().is_none());
    }

    #[test]
    fn page_data_from_blob_zero_pads() {
        let data = vec![0xBBu8; PAGE_SIZE + 100];
        let book = Book::from_blob(test_cid(), &data).unwrap();
        let page_bufs = book.page_data_from_blob(&data);

        assert_eq!(page_bufs.len(), 2);
        assert_eq!(page_bufs[0].len(), PAGE_SIZE);
        assert_eq!(page_bufs[1].len(), PAGE_SIZE);

        // First page: all 0xBB
        assert!(page_bufs[0].iter().all(|&b| b == 0xBB));

        // Second page: first 100 bytes 0xBB, rest 0x00
        assert!(page_bufs[1][..100].iter().all(|&b| b == 0xBB));
        assert!(page_bufs[1][100..].iter().all(|&b| b == 0x00));
    }
}

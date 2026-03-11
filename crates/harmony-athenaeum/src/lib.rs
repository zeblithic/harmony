// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Athenaeum — 32-bit content-addressed page system.
//!
//! Translates 256-bit CID-addressed blobs into 32-bit-addressed
//! 4KB pages optimized for CPU cache lines and register widths.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod addr;
mod athenaeum;
mod encyclopedia;
mod hash;
mod volume;

pub use addr::{
    Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE,
};
pub use athenaeum::{Book, BookError};
pub use encyclopedia::{Encyclopedia, SPLIT_THRESHOLD};
pub use hash::{sha224_hash, sha256_hash};
pub use volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_1mb_blob() {
        // Create 1MB blob, build Book, verify page count, reassemble
        let data = vec![0x42u8; BOOK_MAX_SIZE];
        let cid = sha256_hash(&data);
        let book = Book::from_blob(cid, &data).unwrap();
        assert_eq!(book.page_count(), PAGES_PER_BOOK);
        let pages = book.page_data_from_blob(&data);
        let reassembled = book
            .reassemble(|idx| pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn verify_individual_pages() {
        // Build book, split into pages, verify each page against its algo=0 address
        let data = [0x42u8; PAGE_SIZE * 3];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let pages = book.page_data_from_blob(&data);
        for (i, page_data) in pages.iter().enumerate() {
            let addr = book.pages[i][0];
            assert!(addr.verify_data(page_data));
        }
    }

    #[test]
    fn small_blob_round_trip() {
        let data = b"hello world";
        let cid = sha256_hash(data);
        let book = Book::from_blob(cid, data).unwrap();
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.blob_size, 11);
        let pages = book.page_data_from_blob(data);
        let reassembled = book
            .reassemble(|idx| pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn toc_round_trip() {
        // Build book, generate ToC, verify ToC entries match book.pages
        let data = [0x42u8; PAGE_SIZE * 5];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let toc = book.toc();
        assert_eq!(toc.len(), PAGE_SIZE);
        for (page_idx, variants) in book.pages.iter().enumerate() {
            for (algo_idx, addr) in variants.iter().enumerate() {
                let offset = algo_idx * PAGES_PER_BOOK * 4 + page_idx * 4;
                let stored = u32::from_le_bytes([
                    toc[offset],
                    toc[offset + 1],
                    toc[offset + 2],
                    toc[offset + 3],
                ]);
                assert_eq!(
                    stored, addr.0,
                    "mismatch at page {page_idx}, algo {algo_idx}"
                );
            }
        }
    }

    #[test]
    fn page_addr_debug_is_informative() {
        let addr = PageAddr::new(0x0ABC_0000, Algorithm::Sha256Msb);
        let dbg = format!("{:?}", addr);
        assert!(dbg.contains("PageAddr"));
    }

    #[test]
    fn encyclopedia_multi_blob() {
        let d1 = [0x42u8; PAGE_SIZE * 2];
        let d2 = [0x99u8; PAGE_SIZE * 3];
        let c1 = sha256_hash(&d1);
        let c2 = sha256_hash(&d2);
        let enc = Encyclopedia::build(&[(c1, d1.as_slice()), (c2, d2.as_slice())]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        assert!(enc.total_unique_pages > 0);
    }

    #[test]
    fn null_page_sentinel() {
        let null = PageAddr(NULL_PAGE);
        assert!(!null.verify_checksum(), "NULL_PAGE must fail checksum");
        assert_eq!(null.checksum(), 0b11);
    }
}

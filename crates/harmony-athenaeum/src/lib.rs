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
    is_toc_sentinel, sentinel_algo, toc_sentinel_for_algo, Algorithm, PageAddr, ALGO_COUNT,
    BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE, SELF_INDEXING_MAX_DATA_PAGES,
    SELF_INDEXING_MAX_DATA_SIZE, SELF_INDEX_SENTINELS, SELF_INDEX_SENTINEL_00,
    SELF_INDEX_SENTINEL_01, SELF_INDEX_SENTINEL_10, SELF_INDEX_SENTINEL_11,
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

    #[test]
    fn self_indexing_end_to_end() {
        let data = vec![0x42u8; PAGE_SIZE * 5 + 100];
        let cid = sha256_hash(&data);
        let book = Book::from_blob_self_indexing(cid, &data).unwrap();

        assert!(book.is_self_indexing());
        assert_eq!(book.data_page_count(), 6);
        assert_eq!(book.page_count(), 7);

        let toc = book.toc();
        for addr in &book.pages[0] {
            assert!(addr.verify_data(&toc));
        }

        let first_u32 = u32::from_le_bytes([toc[0], toc[1], toc[2], toc[3]]);
        assert_eq!(first_u32, SELF_INDEX_SENTINEL_00);

        let page_bufs = book.page_data_from_blob(&data);
        assert_eq!(page_bufs.len(), 7);
        let reassembled = book
            .reassemble(|idx| page_bufs.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(reassembled, data);

        let blob: Vec<u8> = page_bufs.into_iter().flatten().collect();
        assert!(Book::is_self_indexing_blob(&blob));
    }

    #[test]
    fn self_indexing_vs_raw_data_pages() {
        let data = vec![0x99u8; PAGE_SIZE * 3];
        let cid = sha256_hash(&data);
        let raw = Book::from_blob(cid, &data).unwrap();
        let si = Book::from_blob_self_indexing(cid, &data).unwrap();

        assert_eq!(raw.page_count(), 3);
        assert_eq!(raw.data_page_count(), 3);

        assert_eq!(si.page_count(), 4);
        assert_eq!(si.data_page_count(), 3);

        // Data page addresses should be identical
        for i in 0..3 {
            assert_eq!(raw.pages[i], si.pages[i + 1]);
        }
    }

    #[test]
    fn self_indexing_empty_book() {
        let book = Book::from_blob_self_indexing([0; 32], &[]).unwrap();
        assert!(book.is_self_indexing());
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.data_page_count(), 0);

        let toc = book.toc();
        for algo_idx in 0..ALGO_COUNT {
            let section_offset = algo_idx * PAGES_PER_BOOK * 4;
            let val = u32::from_le_bytes([
                toc[section_offset],
                toc[section_offset + 1],
                toc[section_offset + 2],
                toc[section_offset + 3],
            ]);
            assert!(is_toc_sentinel(val));

            for page_idx in 1..PAGES_PER_BOOK {
                let offset = section_offset + page_idx * 4;
                let val = u32::from_le_bytes([
                    toc[offset],
                    toc[offset + 1],
                    toc[offset + 2],
                    toc[offset + 3],
                ]);
                assert_eq!(val, NULL_PAGE);
            }
        }

        let page_bufs = book.page_data_from_blob(&[]);
        let reassembled = book
            .reassemble(|idx| page_bufs.get(idx as usize).cloned())
            .unwrap();
        assert!(reassembled.is_empty());
    }
}

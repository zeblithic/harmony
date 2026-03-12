// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Athenaeum — 32-bit content-addressed page system.
//!
//! Translates 256-bit CID-addressed blobs into 32-bit-addressed
//! 4KB pages optimized for CPU cache lines and register widths.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod addr;
mod athenaeum;
mod encrypted;
mod encyclopedia;
mod hash;
mod volume;

pub use addr::{
    is_toc_sentinel, sentinel_algo, toc_sentinel_for_algo, Algorithm, PageAddr, ALGO_COUNT,
    BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE, SELF_INDEXING_MAX_DATA_PAGES,
    SELF_INDEXING_MAX_DATA_SIZE, SELF_INDEX_SENTINELS, SELF_INDEX_SENTINEL_00,
    SELF_INDEX_SENTINEL_01, SELF_INDEX_SENTINEL_10, SELF_INDEX_SENTINEL_11,
};
pub use athenaeum::{Book, BookError, BookType};
pub use encrypted::{EncryptedBookMetadata, ENCRYPTED_SENTINEL, METADATA_PAGE_PAYLOAD};
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

    // --- Encrypted book integration tests ---

    #[test]
    fn encrypted_book_end_to_end() {
        // Build encrypted book with expiry
        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: Some(1_700_000_000),
            tags: None,
        };

        let data = vec![0x42u8; PAGE_SIZE * 5 + 123]; // 5+ pages of data
        let book = Book::from_blob_encrypted([0xDD; 32], &data, &meta).unwrap();

        assert!(book.is_encrypted());
        assert_eq!(book.data_page_count(), 6); // ceil((5*4096 + 123) / 4096) = 6
        assert_eq!(book.page_count(), 8); // 2 meta + 6 data

        // Generate all page data
        let all_pages = book.page_data_from_blob_encrypted(&data, &meta);
        assert_eq!(all_pages.len(), 8);

        // First 2 pages should start with encrypted sentinel
        assert!(Book::is_encrypted_blob(&all_pages[0]));
        assert!(Book::is_encrypted_blob(&all_pages[1]));
        assert!(!Book::is_encrypted_blob(&all_pages[2])); // data page

        // Every page address should verify against its page data
        for (i, page_data) in all_pages.iter().enumerate() {
            for addr in &book.pages[i] {
                assert!(
                    addr.verify_data(page_data),
                    "page {i}: address does not verify against page data"
                );
            }
        }

        // Reassemble should recover original data (skipping metadata pages)
        let reassembled = book
            .reassemble(|idx| all_pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(reassembled, data);

        // Metadata should be recoverable from pages
        let meta_page_arrays: Vec<[u8; PAGE_SIZE]> = all_pages[..2]
            .iter()
            .map(|p| {
                let mut arr = [0u8; PAGE_SIZE];
                arr.copy_from_slice(p);
                arr
            })
            .collect();
        let recovered = EncryptedBookMetadata::from_pages(&meta_page_arrays).unwrap();
        assert_eq!(recovered.owner_public_key, meta.owner_public_key);
        assert_eq!(recovered.encapsulated_key, meta.encapsulated_key);
        assert_eq!(recovered.signature, meta.signature);
        assert_eq!(recovered.expiry, Some(1_700_000_000));
        assert_eq!(recovered, meta);
    }

    #[test]
    fn encrypted_book_end_to_end_with_tags() {
        // Build encrypted book with both expiry and tags
        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0x11; 1184],
            encapsulated_key: vec![0x22; 1088],
            signature: vec![0x33; 3309],
            expiry: Some(9_999_999_999),
            tags: Some(b"access:restricted;domain:medical".to_vec()),
        };

        let data = vec![0xABu8; PAGE_SIZE * 3 + 500];
        let book = Book::from_blob_encrypted([0xEE; 32], &data, &meta).unwrap();

        let all_pages = book.page_data_from_blob_encrypted(&data, &meta);
        let reassembled = book
            .reassemble(|idx| all_pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(reassembled, data);

        // Recover metadata and verify all fields
        let meta_page_arrays: Vec<[u8; PAGE_SIZE]> = all_pages[..2]
            .iter()
            .map(|p| {
                let mut arr = [0u8; PAGE_SIZE];
                arr.copy_from_slice(p);
                arr
            })
            .collect();
        let recovered = EncryptedBookMetadata::from_pages(&meta_page_arrays).unwrap();
        assert_eq!(recovered, meta);
    }

    #[test]
    fn book_type_backward_compat() {
        // Raw books still work
        let raw = Book::from_blob([0xAA; 32], &[0x42; PAGE_SIZE]).unwrap();
        assert!(!raw.is_self_indexing());
        assert!(!raw.is_encrypted());
        assert_eq!(raw.data_page_count(), 1);
        assert_eq!(raw.page_count(), 1);
        assert_eq!(raw.book_type, BookType::Raw);

        // Self-indexing books still work
        let si = Book::from_blob_self_indexing([0xBB; 32], &[0x42; PAGE_SIZE]).unwrap();
        assert!(si.is_self_indexing());
        assert!(!si.is_encrypted());
        assert_eq!(si.data_page_count(), 1);
        assert_eq!(si.page_count(), 2); // 1 ToC + 1 data
        assert_eq!(si.book_type, BookType::SelfIndexing);

        // Encrypted books
        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: None,
            tags: None,
        };
        let enc = Book::from_blob_encrypted([0xCC; 32], &[0x42; PAGE_SIZE], &meta).unwrap();
        assert!(!enc.is_self_indexing());
        assert!(enc.is_encrypted());
        assert_eq!(enc.data_page_count(), 1);
        assert_eq!(enc.page_count(), 3); // 2 meta + 1 data
        assert_eq!(enc.book_type, BookType::Encrypted { metadata_pages: 2 });

        // All three types reassemble correctly
        let raw_pages = raw.page_data_from_blob(&[0x42; PAGE_SIZE]);
        let raw_data = raw
            .reassemble(|idx| raw_pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(raw_data, vec![0x42u8; PAGE_SIZE]);

        let si_pages = si.page_data_from_blob(&[0x42; PAGE_SIZE]);
        let si_data = si
            .reassemble(|idx| si_pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(si_data, vec![0x42u8; PAGE_SIZE]);

        let enc_pages = enc.page_data_from_blob_encrypted(&[0x42; PAGE_SIZE], &meta);
        let enc_data = enc
            .reassemble(|idx| enc_pages.get(idx as usize).cloned())
            .unwrap();
        assert_eq!(enc_data, vec![0x42u8; PAGE_SIZE]);
    }

    #[test]
    fn volume_serialization_encrypted_book() {
        // Test that encrypted books survive volume serialization round-trip
        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: None,
            tags: None,
        };
        let data = vec![0x42u8; PAGE_SIZE * 2];
        let book = Book::from_blob_encrypted([0xDD; 32], &data, &meta).unwrap();

        // Wrap in a Volume leaf, serialize, deserialize
        let vol = Volume::leaf(0, 0, alloc::vec![book.clone()]);
        let bytes = vol.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();

        // Extract the book from the restored volume
        if let Volume::Leaf { books, .. } = &restored {
            assert_eq!(books.len(), 1);
            let deserialized = &books[0];
            assert_eq!(
                deserialized.book_type,
                BookType::Encrypted { metadata_pages: 2 }
            );
            assert_eq!(deserialized.blob_size, book.blob_size);
            assert_eq!(deserialized.cid, book.cid);
            assert_eq!(deserialized.pages.len(), book.pages.len());
            assert_eq!(deserialized, &book);
        } else {
            panic!("expected Volume::Leaf after deserialization");
        }
    }

    #[test]
    fn volume_serialization_all_book_types() {
        // Verify all three book types survive volume serialization
        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: Some(42),
            tags: None,
        };

        let raw = Book::from_blob([0x11; 32], &[0xAA; PAGE_SIZE]).unwrap();
        let si = Book::from_blob_self_indexing([0x22; 32], &[0xBB; PAGE_SIZE]).unwrap();
        let enc = Book::from_blob_encrypted([0x33; 32], &[0xCC; PAGE_SIZE], &meta).unwrap();

        let vol = Volume::leaf(0, 0, alloc::vec![raw.clone(), si.clone(), enc.clone()]);
        let bytes = vol.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(vol, restored);

        if let Volume::Leaf { books, .. } = &restored {
            assert_eq!(books[0].book_type, BookType::Raw);
            assert_eq!(books[1].book_type, BookType::SelfIndexing);
            assert_eq!(
                books[2].book_type,
                BookType::Encrypted { metadata_pages: 2 }
            );
        } else {
            panic!("expected Volume::Leaf");
        }
    }

    #[test]
    fn encrypted_book_data_pages_match_raw() {
        // Data page addresses for the same data should be identical
        // between raw and encrypted books (only metadata pages differ)
        let data = vec![0x55u8; PAGE_SIZE * 3];
        let cid = sha256_hash(&data);
        let raw = Book::from_blob(cid, &data).unwrap();

        let meta = EncryptedBookMetadata {
            version: 1,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: None,
            tags: None,
        };
        let enc = Book::from_blob_encrypted(cid, &data, &meta).unwrap();

        // Data pages in encrypted book should match raw book pages
        let enc_data_pages = enc.data_pages();
        assert_eq!(enc_data_pages.len(), raw.pages.len());
        for i in 0..raw.pages.len() {
            assert_eq!(
                raw.pages[i], enc_data_pages[i],
                "data page {i}: addresses differ between raw and encrypted"
            );
        }
    }
}

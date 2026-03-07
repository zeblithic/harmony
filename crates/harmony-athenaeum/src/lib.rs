// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Athenaeum — 32-bit content-addressed chunk system.
//!
//! Translates 256-bit CID-addressed blobs into 32-bit-addressed
//! mini-blobs optimized for CPU cache lines and register widths.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

mod addr;
mod athenaeum;
mod book;
mod encyclopedia;
mod hash;
mod volume;

pub use addr::{Algorithm, ChunkAddr, Depth};
pub use athenaeum::{Athenaeum, CollisionError, MissingChunkError, MAX_BLOB_SIZE};
pub use book::{Book, BookEntry, BookError};
pub use encyclopedia::{Encyclopedia, SPLIT_THRESHOLD};
pub use hash::{sha224_hash, sha256_hash};
pub use volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeMap;

    #[test]
    fn end_to_end_1mb_blob() {
        // Create a 1MB blob with unique chunks.
        // Use all 4 bytes of the position index so the pattern doesn't
        // repeat at 4KB boundaries (unlike `(i * 7) as u8` which wraps).
        let mut data = alloc::vec![0u8; 1024 * 1024];
        for (i, byte) in data.iter_mut().enumerate() {
            let pos = i as u32;
            *byte = (pos ^ (pos >> 8) ^ (pos >> 16)) as u8;
        }

        // Chunk it
        let cid = sha256_hash(&data);
        let ath = Athenaeum::from_blob(cid, &data).unwrap();
        assert_eq!(ath.chunks.len(), 256);

        // All addresses have valid checksums
        for chunk in &ath.chunks {
            assert!(chunk.verify_checksum());
        }

        // Serialize to book and back
        let book = Book::from_athenaeum(&ath);
        let book_bytes = book.to_bytes();
        assert!(
            book_bytes.len() <= 4096,
            "book fits in single 4KB chunk: {} bytes",
            book_bytes.len()
        );

        let restored_book = Book::from_bytes(&book_bytes).unwrap();
        assert_eq!(restored_book.entries.len(), 1);
        assert_eq!(restored_book.entries[0].chunks.len(), 256);
        assert_eq!(restored_book.entries[0].cid, cid);

        // Build a chunk store
        let mut store: BTreeMap<u32, alloc::vec::Vec<u8>> = BTreeMap::new();
        for (i, &addr) in ath.chunks.iter().enumerate() {
            let start = i * 4096;
            let end = core::cmp::min(start + 4096, data.len());
            let mut chunk = data[start..end].to_vec();
            chunk.resize(addr.size_bytes(), 0);
            store.insert(addr.0, chunk);
        }

        // Reassemble and verify
        let reassembled = ath.reassemble(|addr| store.get(&addr.0).cloned()).unwrap();
        assert_eq!(reassembled.len(), data.len());
        assert_eq!(reassembled, data);
    }

    #[test]
    fn verify_individual_chunks() {
        let mut data = alloc::vec![0xCDu8; 4096 * 4];
        // Make each chunk unique
        for (i, chunk) in data.chunks_mut(4096).enumerate() {
            chunk[0] = i as u8;
        }
        let cid = sha256_hash(&data);
        let ath = Athenaeum::from_blob(cid, &data).unwrap();

        for (i, addr) in ath.chunks.iter().enumerate() {
            let start = i * 4096;
            let end = core::cmp::min(start + 4096, data.len());
            let mut chunk = data[start..end].to_vec();
            chunk.resize(addr.size_bytes(), 0);
            assert!(addr.verify_data(&chunk), "chunk {} failed verification", i);
        }
    }

    #[test]
    fn small_blob_round_trip() {
        // A blob smaller than one chunk
        let data = b"hello athenaeum, this is a small blob!";
        let cid = sha256_hash(data);
        let ath = Athenaeum::from_blob(cid, data).unwrap();
        assert_eq!(ath.chunks.len(), 1);
        // Size should be smallest power-of-two >= 37 = 64 bytes
        assert_eq!(ath.chunks[0].size_bytes(), 64);

        let mut padded = alloc::vec![0u8; 64];
        padded[..data.len()].copy_from_slice(data);

        let reassembled = ath.reassemble(|_addr| Some(padded.clone())).unwrap();
        assert_eq!(&reassembled, data);
    }

    #[test]
    fn multi_blob_book_round_trip() {
        let data1 = alloc::vec![0xAAu8; 4096 * 2];
        let data2 = alloc::vec![0xBBu8; 4096 * 3];
        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let ath1 = Athenaeum::from_blob(cid1, &data1).unwrap();
        let ath2 = Athenaeum::from_blob(cid2, &data2).unwrap();

        let mut book = Book::from_athenaeum(&ath1);
        book.add_athenaeum(&ath2);

        let bytes = book.to_bytes();
        let restored = Book::from_bytes(&bytes).unwrap();

        assert_eq!(restored.entries.len(), 2);
        assert_eq!(restored.entries[0].cid, cid1);
        assert_eq!(restored.entries[0].chunks.len(), 2);
        assert_eq!(restored.entries[1].cid, cid2);
        assert_eq!(restored.entries[1].chunks.len(), 3);
    }

    #[test]
    fn chunk_addr_debug_is_informative() {
        let data = b"debug test data chunk that is long enough";
        let addr = ChunkAddr::from_data(data, Depth::Blob, 0);
        let debug = alloc::format!("{:?}", addr);
        assert!(debug.contains("ChunkAddr("));
        assert!(debug.contains("Blob"));
        assert!(debug.contains("4096B"));
    }

    #[test]
    fn encyclopedia_multi_blob_reassemble() {
        // Build an Encyclopedia from 5 blobs with varied sizes
        let mut blobs_data = Vec::new();
        for blob_idx in 0u8..5 {
            let size = 4096 * (blob_idx as usize + 1); // 4KB to 20KB
            let mut data = alloc::vec![0u8; size];
            for (i, b) in data.iter_mut().enumerate() {
                let pos = i as u32;
                *b = (pos ^ (pos >> 8) ^ (blob_idx as u32 * 37)) as u8;
            }
            blobs_data.push(data);
        }
        let blobs: Vec<([u8; 32], &[u8])> = blobs_data
            .iter()
            .map(|d| (sha256_hash(d), d.as_slice()))
            .collect();

        let enc = Encyclopedia::build(&blobs).unwrap();
        assert_eq!(enc.total_blobs, 5);

        // Verify serialization round-trip
        let bytes = enc.to_bytes();
        let restored = Encyclopedia::from_bytes(&bytes).unwrap();
        assert_eq!(enc, restored);
    }

    #[test]
    fn encyclopedia_shared_chunks_across_blobs() {
        // Two blobs sharing first chunk
        let shared = alloc::vec![0x42u8; 4096];
        let mut data1 = shared.clone();
        data1.extend_from_slice(&alloc::vec![0xAAu8; 4096]);
        let mut data2 = shared.clone();
        data2.extend_from_slice(&alloc::vec![0xBBu8; 4096]);

        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        // 1 shared + 2 unique = 3 unique chunks
        assert_eq!(enc.total_unique_chunks, 3);
    }
}

// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Encyclopedia — recursive content-addressed chunk system for blob collections.

use alloc::collections::BTreeMap;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::addr::Depth;
use crate::athenaeum::{
    address_with_collision_resolution, chunk_size_exponent, CollisionError, CHUNK_SIZE,
    MAX_BLOB_SIZE,
};
use crate::book::{Book, BookEntry, BookError};
use crate::hash::sha256_hash;
use crate::volume::{route_chunk, Volume, MAX_PARTITION_DEPTH};

/// Threshold for proactive splitting.
/// 75% of 2^21 ~ 1,572,864 unique chunks (~6GB unique data).
pub const SPLIT_THRESHOLD: usize = (1 << 21) * 3 / 4;

/// Maximum blobs per Book (cross-consistent group).
const BLOBS_PER_BOOK: usize = 3;

/// Starting bit index for partition routing (bits 0-21 used for addressing).
const PARTITION_START_BIT: u8 = 22;

/// A complete content-addressed mapping for a corpus of blobs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encyclopedia {
    pub root: Volume,
    pub total_blobs: u32,
    pub total_unique_chunks: u32,
}

/// A chunk with its content hash and padded data, used during the build phase.
#[derive(Clone)]
struct ChunkInfo {
    content_hash: [u8; 32],
    padded_data: Vec<u8>,
}

impl Encyclopedia {
    /// Build an Encyclopedia from a collection of blobs.
    ///
    /// Each blob is identified by its CID (content hash) and raw data.
    /// All chunks across all blobs are deduplicated and assigned unique
    /// 32-bit addresses. When the address space fills up, the system
    /// recursively partitions by content-hash bits.
    pub fn build(blobs: &[([u8; 32], &[u8])]) -> Result<Self, CollisionError> {
        for &(_, data) in blobs {
            if data.len() > MAX_BLOB_SIZE {
                return Err(CollisionError::BlobTooLarge { size: data.len() });
            }
        }

        if blobs.is_empty() {
            return Ok(Encyclopedia {
                root: Volume::leaf(0, 0, Vec::new()),
                total_blobs: 0,
                total_unique_chunks: 0,
            });
        }

        // Phase 1: Chunk all blobs, dedup by content hash, track which
        // blob indices use each unique chunk.
        let mut unique_chunks: BTreeMap<[u8; 32], ChunkInfo> = BTreeMap::new();
        let mut blob_chunk_hashes: Vec<Vec<[u8; 32]>> = Vec::new();

        for &(_, data) in blobs {
            let mut hashes = Vec::new();
            for chunk_data in data.chunks(CHUNK_SIZE) {
                let size_exp = chunk_size_exponent(chunk_data.len());
                let padded_size = CHUNK_SIZE >> (size_exp as usize);
                let mut padded = alloc::vec![0u8; padded_size];
                padded[..chunk_data.len()].copy_from_slice(chunk_data);

                let content_hash = sha256_hash(&padded);
                hashes.push(content_hash);

                unique_chunks.entry(content_hash).or_insert(ChunkInfo {
                    content_hash,
                    padded_data: padded,
                });
            }
            blob_chunk_hashes.push(hashes);
        }

        let total_unique = unique_chunks.len() as u32;
        let chunk_list: Vec<ChunkInfo> = unique_chunks.into_values().collect();

        // Phase 2 & 3: Recursive partition + resolve
        let root = Self::build_volume(
            &chunk_list,
            blobs,
            &blob_chunk_hashes,
            0, // depth
            0, // path
            PARTITION_START_BIT,
        )?;

        Ok(Encyclopedia {
            root,
            total_blobs: blobs.len() as u32,
            total_unique_chunks: total_unique,
        })
    }

    /// Recursively build a Volume from a set of unique chunks.
    fn build_volume(
        chunks: &[ChunkInfo],
        blobs: &[([u8; 32], &[u8])],
        blob_chunk_hashes: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
        bit_index: u8,
    ) -> Result<Volume, CollisionError> {
        if bit_index >= PARTITION_START_BIT.saturating_add(MAX_PARTITION_DEPTH) {
            return Err(CollisionError::MaxPartitionDepth { depth });
        }

        if chunks.is_empty() {
            return Ok(Volume::leaf(depth, path, Vec::new()));
        }

        if chunks.len() <= SPLIT_THRESHOLD {
            // Try to resolve in a single flat volume
            match Self::resolve_leaf(chunks, blobs, blob_chunk_hashes, depth, path) {
                Ok(vol) => return Ok(vol),
                Err(_) => {
                    // Fall through to splitting
                }
            }
        }

        // Split by content-hash bit
        let mut left_chunks = Vec::new();
        let mut right_chunks = Vec::new();
        for chunk in chunks {
            if route_chunk(&chunk.content_hash, bit_index) {
                right_chunks.push(chunk.clone());
            } else {
                left_chunks.push(chunk.clone());
            }
        }

        let left = Self::build_volume(
            &left_chunks,
            blobs,
            blob_chunk_hashes,
            depth + 1,
            path,
            bit_index + 1,
        )?;
        let right = Self::build_volume(
            &right_chunks,
            blobs,
            blob_chunk_hashes,
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
            split_bit: bit_index,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Resolve a set of chunks into a leaf Volume with Books.
    ///
    /// Addresses ONLY the chunks whose content hashes belong to this
    /// partition. Each blob's BookEntry contains only the subset of
    /// chunk addresses that route here.
    fn resolve_leaf(
        chunks: &[ChunkInfo],
        blobs: &[([u8; 32], &[u8])],
        blob_chunk_hashes: &[Vec<[u8; 32]>],
        depth: u8,
        path: u32,
    ) -> Result<Volume, CollisionError> {
        // Build a set of content hashes in this partition
        let partition_hashes: BTreeSet<[u8; 32]> = chunks.iter().map(|c| c.content_hash).collect();

        // Address ONLY the unique chunks in this partition
        let mut used_addrs = BTreeSet::new();
        let mut content_cache: BTreeMap<[u8; 32], crate::addr::ChunkAddr> = BTreeMap::new();

        for chunk_info in chunks {
            if content_cache.contains_key(&chunk_info.content_hash) {
                continue;
            }
            let size_exp = chunk_size_exponent(chunk_info.padded_data.len());
            let addr = address_with_collision_resolution(
                &chunk_info.padded_data,
                Depth::Blob,
                size_exp,
                &used_addrs,
            )
            .ok_or(CollisionError::AllAlgorithmsCollide { chunk_index: 0 })?;
            used_addrs.insert(addr.hash_bits());
            content_cache.insert(chunk_info.content_hash, addr);
        }

        // Find which blobs have chunks in this partition
        let mut relevant_blob_indices: Vec<usize> = Vec::new();
        for (blob_idx, chunk_list) in blob_chunk_hashes.iter().enumerate() {
            if chunk_list.iter().any(|h| partition_hashes.contains(h)) {
                relevant_blob_indices.push(blob_idx);
            }
        }

        // Build BookEntries: for each relevant blob, include only the
        // chunk addresses that route to this partition.
        let mut books = Vec::new();
        for group in relevant_blob_indices.chunks(BLOBS_PER_BOOK) {
            let mut entries = Vec::new();
            for &blob_idx in group {
                let (cid, data) = blobs[blob_idx];
                let blob_size = data.len() as u32;
                let mut chunk_addrs = Vec::new();
                for &hash in &blob_chunk_hashes[blob_idx] {
                    if let Some(&addr) = content_cache.get(&hash) {
                        chunk_addrs.push(addr);
                    }
                }
                entries.push(BookEntry {
                    cid,
                    blob_size,
                    chunks: chunk_addrs,
                });
            }
            books.push(Book { entries });
        }

        Ok(Volume::leaf(depth, path, books))
    }

    /// Determine which partition a chunk belongs to by content hash.
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
    /// + total_unique_chunks u32 LE (4) + volume_bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"ENCY");
        buf.push(1); // version
        buf.extend_from_slice(&self.total_blobs.to_le_bytes());
        buf.extend_from_slice(&self.total_unique_chunks.to_le_bytes());
        buf.extend_from_slice(&self.root.to_bytes());
        buf
    }

    /// Deserialize from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        if data.len() < 13 {
            return Err(BookError::TooShort);
        }
        if &data[0..4] != b"ENCY" {
            return Err(BookError::InvalidChecksum);
        }
        if data[4] != 1 {
            return Err(BookError::InvalidChecksum);
        }
        let total_blobs =
            u32::from_le_bytes(data[5..9].try_into().map_err(|_| BookError::TooShort)?);
        let total_unique_chunks =
            u32::from_le_bytes(data[9..13].try_into().map_err(|_| BookError::TooShort)?);
        let root = Volume::from_bytes(&data[13..])?;
        Ok(Encyclopedia {
            root,
            total_blobs,
            total_unique_chunks,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_single_blob() {
        let data = alloc::vec![0xABu8; 4096 * 2];
        let cid = sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, &data)]).unwrap();
        assert_eq!(enc.total_blobs, 1);
        assert_eq!(enc.root.chunk_count(), 2);
    }

    #[test]
    fn build_multiple_blobs() {
        let mut data1 = alloc::vec![0u8; 4096 * 2];
        let mut data2 = alloc::vec![0u8; 4096 * 3];
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
        assert!(enc.total_unique_chunks <= 5);
    }

    #[test]
    fn build_deduplicates_shared_chunks() {
        let shared = alloc::vec![0x42u8; 4096];
        let unique1 = alloc::vec![0xAAu8; 4096];
        let unique2 = alloc::vec![0xBBu8; 4096];

        let mut data1 = Vec::new();
        data1.extend_from_slice(&shared);
        data1.extend_from_slice(&unique1);
        let mut data2 = Vec::new();
        data2.extend_from_slice(&shared);
        data2.extend_from_slice(&unique2);

        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();
        assert_eq!(enc.total_unique_chunks, 3);
    }

    #[test]
    fn build_empty() {
        let enc = Encyclopedia::build(&[]).unwrap();
        assert_eq!(enc.total_blobs, 0);
        assert_eq!(enc.total_unique_chunks, 0);
    }

    #[test]
    fn build_blob_too_large() {
        let data = alloc::vec![0u8; MAX_BLOB_SIZE + 1];
        let cid = sha256_hash(&data);
        let result = Encyclopedia::build(&[(cid, &data)]);
        assert!(matches!(result, Err(CollisionError::BlobTooLarge { .. })));
    }

    #[test]
    fn route_deterministic() {
        let hash = sha256_hash(b"test chunk");
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
    fn serialization_round_trip() {
        let data = alloc::vec![0xCDu8; 4096];
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
        // Put a valid leaf volume: tag=0, depth=0, path=0, book_count=0, reserved=0
        data[13] = 0; // tag
        data[14] = 0; // depth
                      // rest zeros are fine for path, book_count, reserved
        assert!(Encyclopedia::from_bytes(&data).is_err());
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(Encyclopedia::from_bytes(&[0u8; 5]).is_err());
    }

    #[test]
    fn split_threshold_value() {
        assert_eq!(SPLIT_THRESHOLD, 1_572_864);
    }

    #[test]
    fn resolve_leaf_only_addresses_partition_chunks() {
        // Force a split by setting a tiny threshold, then verify each leaf
        // only contains chunks that route to its partition.
        //
        // We can't easily set SPLIT_THRESHOLD per-test, but we CAN verify
        // the fix indirectly: build with 2 blobs, confirm the total chunk
        // count across the tree equals total_unique_chunks (no duplication).
        let mut data1 = alloc::vec![0u8; 4096 * 4];
        let mut data2 = alloc::vec![0u8; 4096 * 4];
        for (i, chunk) in data1.chunks_mut(4096).enumerate() {
            chunk[0] = i as u8;
            chunk[1] = 0xAA;
        }
        for (i, chunk) in data2.chunks_mut(4096).enumerate() {
            chunk[0] = i as u8;
            chunk[1] = 0xBB;
        }
        let cid1 = sha256_hash(&data1);
        let cid2 = sha256_hash(&data2);

        let enc = Encyclopedia::build(&[(cid1, &data1), (cid2, &data2)]).unwrap();

        // In a flat (no-split) volume, chunk_count should equal
        // total_unique_chunks — no duplication across leaves.
        assert_eq!(enc.root.chunk_count() as u32, enc.total_unique_chunks);
    }

    #[test]
    fn route_does_not_panic_at_high_depth() {
        // Verify route() handles depth > 32 without panicking
        let hash = sha256_hash(b"deep partition test");
        let path = Encyclopedia::route(&hash, 100);
        // path is capped at 32 bits — should not panic
        let _ = path;
    }
}

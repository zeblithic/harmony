//! Manifest header serialization and deserialization.
//!
//! The manifest consists of a [`ManifestHeader`] (table metadata) plus a
//! shard CID list.  The header is serialized with postcard for no_std
//! portability.  The shard CID list is stored as raw 32-byte entries
//! (not postcard-encoded) because postcard's variable-length encoding
//! would bloat 50 million fixed-size entries.
//!
//! In a Harmony Merkle DAG, the manifest is stored as:
//! - Root bundle child 0: postcard-encoded [`ManifestHeader`]
//! - Root bundle children 1..N: raw shard CID chunks
//!
//! This module handles header (de)serialization.  DAG assembly is the
//! caller's concern (typically harmony-content's `dag::ingest`).

use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};

use crate::{EngramConfig, EngramError};

/// Serializable manifest header — table metadata without the shard CID list.
///
/// Postcard-encoded for compact, no_std-friendly serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestHeader {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Number of components per embedding vector.
    pub embedding_dim: u32,
    /// Bytes per component (2 for f16).  Widened to `usize` in [`EngramConfig`].
    pub dtype_bytes: u16,
    /// Number of independent hash heads.
    pub num_heads: u32,
    /// Per-head xxhash64 seeds.
    pub hash_seeds: Vec<u64>,
    /// Total entries across all shards.
    pub total_entries: u64,
    /// Embeddings per shard.
    pub shard_size: u32,
    /// Total shard count.
    pub num_shards: u64,
}

impl ManifestHeader {
    /// Serialize to postcard bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>, EngramError> {
        postcard::to_allocvec(self).map_err(|_| EngramError::ManifestSerialize)
    }

    /// Deserialize from postcard bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, EngramError> {
        postcard::from_bytes(data).map_err(|_| EngramError::ManifestDeserialize)
    }

    /// Convert to an [`EngramConfig`] for use with [`EngramClient`](crate::EngramClient).
    pub fn to_config(&self) -> EngramConfig {
        EngramConfig {
            version: self.version.clone(),
            embedding_dim: self.embedding_dim as usize,
            dtype_bytes: self.dtype_bytes as usize,
            num_heads: self.num_heads,
            shard_size: self.shard_size,
            num_shards: self.num_shards,
            total_entries: self.total_entries,
            hash_seeds: self.hash_seeds.clone(),
        }
    }
}

/// Parse raw shard CID bytes into a Vec of 32-byte CID arrays.
///
/// `data` must be a multiple of 32 bytes.  Returns one `[u8; 32]` per shard.
pub fn parse_shard_cids(data: &[u8]) -> Result<Vec<[u8; 32]>, EngramError> {
    if data.len() % 32 != 0 {
        return Err(EngramError::ManifestDeserialize);
    }
    let count = data.len() / 32;
    let mut cids = Vec::with_capacity(count);
    for chunk in data.chunks_exact(32) {
        let mut cid = [0u8; 32];
        cid.copy_from_slice(chunk);
        cids.push(cid);
    }
    Ok(cids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn test_header() -> ManifestHeader {
        ManifestHeader {
            version: String::from("v1"),
            embedding_dim: 160,
            dtype_bytes: 2,
            num_heads: 4,
            hash_seeds: vec![111, 222, 333, 444],
            total_entries: 10_000_000_000,
            shard_size: 200,
            num_shards: 50_000_000,
        }
    }

    #[test]
    fn header_round_trip() {
        let header = test_header();
        let bytes = header.to_bytes().unwrap();
        let decoded = ManifestHeader::from_bytes(&bytes).unwrap();
        assert_eq!(header, decoded);
    }

    #[test]
    fn header_to_config() {
        let header = test_header();
        let config = header.to_config();
        assert_eq!(config.version, "v1");
        assert_eq!(config.embedding_dim, 160);
        assert_eq!(config.dtype_bytes, 2);
        assert_eq!(config.num_heads, 4);
        assert_eq!(config.shard_size, 200);
        assert_eq!(config.num_shards, 50_000_000);
        assert_eq!(config.total_entries, 10_000_000_000);
        assert_eq!(config.hash_seeds, vec![111, 222, 333, 444]);
    }

    #[test]
    fn parse_shard_cids_basic() {
        // 2 CIDs, each 32 bytes.
        let mut data = vec![0u8; 64];
        data[0] = 0xAA;
        data[32] = 0xBB;
        let cids = parse_shard_cids(&data).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0][0], 0xAA);
        assert_eq!(cids[1][0], 0xBB);
    }

    #[test]
    fn parse_shard_cids_empty() {
        let cids = parse_shard_cids(&[]).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn parse_shard_cids_not_aligned() {
        // 33 bytes — not a multiple of 32.
        let data = vec![0u8; 33];
        let err = parse_shard_cids(&data).unwrap_err();
        assert!(matches!(err, EngramError::ManifestDeserialize));
    }

    #[test]
    fn invalid_bytes_returns_error() {
        let err = ManifestHeader::from_bytes(&[0xFF, 0xFF]).unwrap_err();
        assert!(matches!(err, EngramError::ManifestDeserialize));
    }
}

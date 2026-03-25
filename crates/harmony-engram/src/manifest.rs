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
        todo!()
    }

    /// Deserialize from postcard bytes.
    pub fn from_bytes(_data: &[u8]) -> Result<Self, EngramError> {
        todo!()
    }

    /// Convert to an [`EngramConfig`] for use with [`EngramClient`](crate::EngramClient).
    pub fn to_config(&self) -> EngramConfig {
        todo!()
    }
}

/// Parse raw shard CID bytes into a Vec of 32-byte CID arrays.
///
/// `data` must be a multiple of 32 bytes.  Returns one `[u8; 32]` per shard.
pub fn parse_shard_cids(_data: &[u8]) -> Result<Vec<[u8; 32]>, EngramError> {
    todo!()
}

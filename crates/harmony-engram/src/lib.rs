//! Sans-I/O client for Engram conditional memory table lookups.
//!
//! Engram tables are sharded embedding tables stored as Harmony CAS books.
//! This crate computes which shards to fetch and extracts/aggregates vectors
//! from fetched shard bytes.  The caller handles all network I/O.
//!
//! # Lookup flow
//!
//! 1. Call [`EngramClient::lookup`] with N-gram tokens → [`EngramLookup`]
//! 2. Fetch each shard by CID from the network (caller's responsibility)
//! 3. Call [`EngramClient::resolve`] with the lookup + shard bytes → aggregated f32 embedding

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::{string::String, vec::Vec};

pub mod chronos;
pub mod error;
pub mod hash;
pub mod manifest;
pub mod resolve;

pub use chronos::{ChronosTier, EngramMetadata, compute_decay, temporal_decay};
pub use error::EngramError;
pub use manifest::ManifestHeader;

/// Configuration for an Engram table (derived from the manifest).
#[derive(Debug, Clone)]
pub struct EngramConfig {
    /// Table version identifier (e.g. "v1").
    pub version: String,
    /// Number of components per embedding vector.
    pub embedding_dim: usize,
    /// Bytes per component (2 for f16).
    pub dtype_bytes: usize,
    /// Number of independent hash heads.
    pub num_heads: u32,
    /// Number of embedding vectors per shard.
    pub shard_size: u32,
    /// Total number of shards in the table.
    pub num_shards: u64,
    /// Total number of entries across all shards.
    pub total_entries: u64,
    /// Per-head xxhash64 seeds.  Length must equal `num_heads`.
    pub hash_seeds: Vec<u64>,
}

impl EngramConfig {
    /// Bytes per embedding vector: `embedding_dim * dtype_bytes`.
    pub fn vector_bytes(&self) -> usize {
        self.embedding_dim * self.dtype_bytes
    }
}

/// Result of hashing an N-gram — which shards to fetch and where to read.
#[derive(Debug, Clone)]
pub struct EngramLookup {
    /// Shard index for each head (length = num_heads).
    pub shard_indices: Vec<u64>,
    /// Byte offset within each shard (length = num_heads).
    pub entry_offsets: Vec<usize>,
}

/// Sans-I/O client for performing Engram lookups.
///
/// Holds the table configuration and the shard-to-CID mapping.
/// All methods are pure computation — no I/O.
#[derive(Debug, Clone)]
pub struct EngramClient {
    config: EngramConfig,
    /// Shard index → CID (32-byte raw content identifier).
    manifest_cids: Vec<[u8; 32]>,
}

impl EngramClient {
    /// Construct a client from a parsed manifest.
    ///
    /// `shard_cids` must have exactly `config.num_shards` entries.
    pub fn from_manifest(config: EngramConfig, shard_cids: Vec<[u8; 32]>) -> Self {
        debug_assert_eq!(
            shard_cids.len() as u64,
            config.num_shards,
            "shard_cids length must match config.num_shards"
        );
        Self {
            config,
            manifest_cids: shard_cids,
        }
    }

    /// Access the table configuration.
    pub fn config(&self) -> &EngramConfig {
        &self.config
    }

    /// Get the CID for a shard index.
    ///
    /// Returns `None` if `shard_index >= num_shards`.
    pub fn shard_cid(&self, shard_index: u64) -> Option<&[u8; 32]> {
        self.manifest_cids.get(shard_index as usize)
    }

    /// Compute which shards and byte offsets are needed for an N-gram.
    ///
    /// Pure computation — no I/O.  The caller fetches each shard by its CID
    /// (via [`shard_cid`](Self::shard_cid)) and passes the bytes to
    /// [`resolve`](Self::resolve).
    pub fn lookup(&self, ngram_tokens: &[u32]) -> EngramLookup {
        hash::compute_lookup(&self.config, ngram_tokens)
    }

    /// Extract and aggregate embedding vectors from fetched shard bytes.
    ///
    /// `shard_data` must contain one byte slice per head, in the same order
    /// as `lookup.shard_indices`.  Each slice is the raw bytes of that shard
    /// (typically 64KB).
    ///
    /// Returns the summed embedding as f32 little-endian bytes
    /// (length = `embedding_dim * 4`).
    pub fn resolve(
        &self,
        lookup: &EngramLookup,
        shard_data: &[&[u8]],
    ) -> Result<Vec<u8>, EngramError> {
        resolve::aggregate(&self.config, lookup, shard_data)
    }
}

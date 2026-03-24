use alloc::{string::String, vec::Vec};
use serde::{Deserialize, Serialize};
use crate::{EngramConfig, EngramError};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ManifestHeader {
    pub version: String,
    pub embedding_dim: u32,
    pub dtype_bytes: u16,
    pub num_heads: u32,
    pub hash_seeds: Vec<u64>,
    pub total_entries: u64,
    pub shard_size: u32,
    pub num_shards: u64,
}

impl ManifestHeader {
    pub fn to_bytes(&self) -> Result<Vec<u8>, EngramError> {
        todo!()
    }
    pub fn from_bytes(_data: &[u8]) -> Result<Self, EngramError> {
        todo!()
    }
    pub fn to_config(&self) -> EngramConfig {
        todo!()
    }
}

pub fn parse_shard_cids(_data: &[u8]) -> Result<Vec<[u8; 32]>, EngramError> {
    todo!()
}

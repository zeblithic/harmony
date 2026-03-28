//! Prefill KV cache sharing protocol.
//!
//! Enables powerful nodes to serialize compressed KV caches into CAS,
//! so edge nodes can fetch and resume generation without re-evaluating
//! the prompt. Feature-gated behind `prefill`.

// Imports used by serialize/store functions added in subsequent tasks.
#[allow(unused_imports)]
use harmony_content::book::BookStore;
#[allow(unused_imports)]
use harmony_content::cid::ContentId;
#[allow(unused_imports)]
use harmony_content::chunker::ChunkerConfig;
#[allow(unused_imports)]
use harmony_content::dag;
use harmony_content::error::ContentError;
use harmony_crypto::hash::full_hash;
#[allow(unused_imports)]
use harmony_inference::InferenceCache;
use harmony_inference::InferenceError;
use serde::{Deserialize, Serialize};

/// Magic bytes identifying a prefill KV cache blob.
pub const PREFILL_MAGIC: [u8; 4] = *b"HKV\x01";

/// Supported quantization bit width.
#[allow(dead_code)]
const SUPPORTED_QUANT_BITS: u8 = 3;

/// Header prepended to the serialized KV cache payload in CAS.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrefillCacheHeader {
    /// Magic bytes + format version.
    pub magic: [u8; 4],
    /// CAS CID of the GGUF model that produced this cache.
    pub model_cid: [u8; 32],
    /// SHA-256 of the token IDs (as contiguous little-endian u32 bytes).
    pub token_hash: [u8; 32],
    /// Number of tokens in the prefill (= cache.position).
    pub token_count: u32,
    /// Architecture params for validation.
    pub num_layers: u16,
    pub num_kv_heads: u16,
    pub head_dim: u16,
    /// Compression config.
    pub quant_bits: u8,
}

/// Errors from prefill cache operations.
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("invalid magic: expected HKV\\x01")]
    InvalidMagic,

    #[error("unsupported quant_bits: {0}")]
    UnsupportedQuantBits(u8),

    #[error("model mismatch: expected {expected:?}, got {actual:?}")]
    ModelMismatch { expected: [u8; 32], actual: [u8; 32] },

    #[error("cache is not compressed")]
    CacheNotCompressed,

    #[error("serialization failed: {0}")]
    SerializationFailed(String),

    #[error("storage failed: {0}")]
    StorageFailed(String),
}

impl From<InferenceError> for PrefillError {
    fn from(e: InferenceError) -> Self {
        PrefillError::SerializationFailed(e.to_string())
    }
}

impl From<ContentError> for PrefillError {
    fn from(e: ContentError) -> Self {
        PrefillError::StorageFailed(e.to_string())
    }
}

/// Compute the token hash for a prefill sequence.
/// SHA-256 of token IDs as contiguous little-endian u32 bytes.
pub fn token_hash(token_ids: &[u32]) -> [u8; 32] {
    let bytes: Vec<u8> = token_ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    full_hash(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn token_hash_deterministic() {
        let ids = vec![100u32, 200, 300];
        let h1 = token_hash(&ids);
        let h2 = token_hash(&ids);
        assert_eq!(h1, h2);
    }

    #[test]
    fn token_hash_different_for_different_tokens() {
        let h1 = token_hash(&[1, 2, 3]);
        let h2 = token_hash(&[3, 2, 1]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn token_hash_empty() {
        let h = token_hash(&[]);
        assert_eq!(h, full_hash(b""));
    }

    #[test]
    fn header_serde_roundtrip() {
        let header = PrefillCacheHeader {
            magic: PREFILL_MAGIC,
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 2048,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
        };
        let bytes = postcard::to_allocvec(&header).unwrap();
        let restored: PrefillCacheHeader = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(header, restored);
    }
}

//! Prefill KV cache sharing protocol.
//!
//! Enables powerful nodes to serialize compressed KV caches into CAS,
//! so edge nodes can fetch and resume generation without re-evaluating
//! the prompt. Feature-gated behind `prefill`.

use harmony_content::book::BookStore;
use harmony_content::cid::ContentId;
use harmony_content::chunker::ChunkerConfig;
use harmony_content::dag;
use harmony_content::error::ContentError;
use harmony_crypto::hash::full_hash;
use harmony_inference::InferenceCache;
use harmony_inference::InferenceError;
use serde::{Deserialize, Serialize};

/// Magic bytes identifying a prefill KV cache blob.
pub const PREFILL_MAGIC: [u8; 4] = *b"HKV\x02";

/// Supported quantization bit width.
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
    /// TOPLOC proofs for this cache (empty if not yet generated).
    pub proofs: Vec<crate::toploc_proof::TocProof>,
}

/// Errors from prefill cache operations.
#[derive(Debug, thiserror::Error)]
pub enum PrefillError {
    #[error("invalid magic: expected HKV\\x02")]
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

/// Store a compressed KV cache in CAS. Returns the root CID.
///
/// The cache must be compressed (`is_compressed() == true`).
/// `model_cid` identifies the GGUF model. `token_ids` is the
/// prefill token sequence (hashed into the header).
pub fn store_prefill_cache(
    cache: &InferenceCache,
    model_cid: &ContentId,
    token_ids: &[u32],
    store: &mut dyn BookStore,
) -> Result<ContentId, PrefillError> {
    if !cache.is_compressed() {
        return Err(PrefillError::CacheNotCompressed);
    }

    if token_ids.len() != cache.position {
        return Err(PrefillError::SerializationFailed(format!(
            "token_ids length {} does not match cache position {}",
            token_ids.len(),
            cache.position
        )));
    }

    let header = PrefillCacheHeader {
        magic: PREFILL_MAGIC,
        model_cid: model_cid.to_bytes(),
        token_hash: token_hash(token_ids),
        token_count: u32::try_from(cache.position).map_err(|_| {
            PrefillError::SerializationFailed(format!("position {} exceeds u32", cache.position))
        })?,
        num_layers: u16::try_from(cache.num_layers).map_err(|_| {
            PrefillError::SerializationFailed(format!("num_layers {} exceeds u16", cache.num_layers))
        })?,
        num_kv_heads: u16::try_from(cache.num_kv_heads).map_err(|_| {
            PrefillError::SerializationFailed(format!("num_kv_heads {} exceeds u16", cache.num_kv_heads))
        })?,
        head_dim: u16::try_from(cache.head_dim).map_err(|_| {
            PrefillError::SerializationFailed(format!("head_dim {} exceeds u16", cache.head_dim))
        })?,
        quant_bits: SUPPORTED_QUANT_BITS,
        proofs: vec![],
    };

    let header_bytes = postcard::to_allocvec(&header)
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;

    let payload_bytes = cache.serialize_compressed()?;

    // Wire blob: [header_len: u32 LE][header][payload]
    let mut blob = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
    blob.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    blob.extend_from_slice(&header_bytes);
    blob.extend_from_slice(&payload_bytes);

    let root = dag::ingest(&blob, &ChunkerConfig::DEFAULT, store)?;
    Ok(root)
}

/// Load a prefill cache from CAS.
///
/// Validates the header magic, quant_bits, and model CID. Returns the cache
/// in compressed state (caller must call `decompress()` before `forward()`).
pub fn load_prefill_cache(
    root_cid: &ContentId,
    expected_model_cid: &ContentId,
    store: &dyn BookStore,
) -> Result<(InferenceCache, PrefillCacheHeader), PrefillError> {
    let blob = dag::reassemble(root_cid, store)?;

    if blob.len() < 4 {
        return Err(PrefillError::SerializationFailed("blob too short".into()));
    }

    let header_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
    let header_end = 4usize.checked_add(header_len).ok_or_else(|| {
        PrefillError::SerializationFailed("header_len overflow".into())
    })?;
    if blob.len() < header_end {
        return Err(PrefillError::SerializationFailed("truncated header".into()));
    }

    let header: PrefillCacheHeader = postcard::from_bytes(&blob[4..header_end])
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;

    if header.magic != PREFILL_MAGIC {
        return Err(PrefillError::InvalidMagic);
    }

    if header.quant_bits != SUPPORTED_QUANT_BITS {
        return Err(PrefillError::UnsupportedQuantBits(header.quant_bits));
    }

    let expected_bytes = expected_model_cid.to_bytes();
    if header.model_cid != expected_bytes {
        return Err(PrefillError::ModelMismatch {
            expected: expected_bytes,
            actual: header.model_cid,
        });
    }

    let payload = &blob[header_end..];
    let cache = InferenceCache::deserialize_compressed(
        payload,
        header.num_layers as usize,
        header.head_dim as usize,
        header.num_kv_heads as usize,
    )?;

    Ok((cache, header))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use harmony_content::book::MemoryBookStore;

    /// Create a compressed InferenceCache with synthetic data.
    fn compressed_cache(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            cache.layers[0] = Some((k, v));
            cache.position = n_tokens;
        }
        cache.compress().unwrap();
        cache
    }

    fn fake_model_cid() -> ContentId {
        ContentId::for_book(b"fake-model-bytes", Default::default()).unwrap()
    }

    #[test]
    fn store_load_roundtrip() {
        let cache = compressed_cache(2, 8, 128, 16);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..16).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();

        let (restored, header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert!(restored.is_compressed());
        assert_eq!(restored.position, 16);
        assert_eq!(header.token_count, 16);
        assert_eq!(header.num_layers, 2);
        assert_eq!(header.num_kv_heads, 8);
        assert_eq!(header.head_dim, 128);
        assert_eq!(header.quant_bits, 3);
        assert_eq!(header.token_hash, token_hash(&token_ids));
    }

    #[test]
    fn load_rejects_wrong_model() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..4).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();

        let wrong_cid = ContentId::for_book(b"wrong-model", Default::default()).unwrap();
        let result = load_prefill_cache(&root, &wrong_cid, &store);
        assert!(matches!(result, Err(PrefillError::ModelMismatch { .. })));
    }

    #[test]
    fn store_requires_compressed() {
        let cache = InferenceCache::new(2, 128, 8); // not compressed
        let model_cid = fake_model_cid();
        let mut store = MemoryBookStore::new();

        let result = store_prefill_cache(&cache, &model_cid, &[1], &mut store);
        assert!(matches!(result, Err(PrefillError::CacheNotCompressed)));
    }

    #[test]
    fn header_magic_validation() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..4).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();

        // Corrupt the blob: reassemble, corrupt magic, re-store
        let mut blob = dag::reassemble(&root, &store).unwrap();
        blob[4] = 0xFF; // corrupt first byte of header (after header_len prefix)
        let corrupt_root = dag::ingest(&blob, &ChunkerConfig::DEFAULT, &mut store).unwrap();

        let result = load_prefill_cache(&corrupt_root, &model_cid, &store);
        assert!(matches!(result, Err(PrefillError::InvalidMagic)));
    }

    #[test]
    fn unsupported_quant_bits_rejected() {
        let cache = compressed_cache(2, 8, 128, 4);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..4).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();

        // Reassemble, patch quant_bits in header, re-store
        let blob = dag::reassemble(&root, &store).unwrap();
        let header_len = u32::from_le_bytes(blob[0..4].try_into().unwrap()) as usize;
        let mut header: PrefillCacheHeader = postcard::from_bytes(&blob[4..4 + header_len]).unwrap();
        header.quant_bits = 5; // unsupported
        let new_header_bytes = postcard::to_allocvec(&header).unwrap();
        let mut new_blob = Vec::new();
        new_blob.extend_from_slice(&(new_header_bytes.len() as u32).to_le_bytes());
        new_blob.extend_from_slice(&new_header_bytes);
        new_blob.extend_from_slice(&blob[4 + header_len..]);
        let corrupt_root = dag::ingest(&new_blob, &ChunkerConfig::DEFAULT, &mut store).unwrap();

        let result = load_prefill_cache(&corrupt_root, &model_cid, &store);
        assert!(matches!(result, Err(PrefillError::UnsupportedQuantBits(5))));
    }

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
            proofs: vec![],
        };
        let bytes = postcard::to_allocvec(&header).unwrap();
        let restored: PrefillCacheHeader = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(header, restored);
    }
}

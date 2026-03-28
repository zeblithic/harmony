//! Two-phase sans-I/O bridge between harmony-engram (hash lookups) and
//! EngramGatedResidual (tensor math).
//!
//! Phase 1: [`prepare_engram_request`] — extract N-grams from tokens, compute
//! hash lookups, return which shards to fetch.
//!
//! Phase 2: [`resolve_engram_embeddings`] — assemble shard data, resolve
//! embeddings, aggregate per position, return candle Tensor.

use std::collections::{HashMap, HashSet};

use candle_core::{Device, Result, Tensor};
use harmony_engram::{EngramClient, EngramLookup};

/// A shard that needs to be fetched by the caller.
#[derive(Debug, Clone)]
pub struct ShardRequest {
    /// Index into the Engram table's shard list.
    pub shard_index: u64,
    /// Content identifier for CAS fetch.
    pub cid: [u8; 32],
}

/// Per-N-gram lookup result with position attribution.
#[derive(Debug, Clone)]
pub struct NgramLookup {
    /// Position in the token sequence this N-gram's embedding covers.
    /// This is the **last token's position** in the N-gram.
    pub token_position: usize,
    /// Lookup result from harmony-engram (shard indices + byte offsets per head).
    pub lookup: EngramLookup,
}

/// Result of Phase 1: which shards to fetch and how to resolve them.
#[derive(Debug, Clone)]
pub struct EngramRequest {
    /// Deduplicated shards needed — caller fetches these by CID.
    pub required_shards: Vec<ShardRequest>,
    /// Per-N-gram lookups with position attribution.
    pub lookups: Vec<NgramLookup>,
    /// Input sequence length (for output tensor sizing).
    pub seq_len: usize,
}

/// Phase 1: Extract N-grams from tokens and compute hash lookups.
///
/// Extracts bigrams and trigrams, attributed to the **last token's position**.
/// For tokens `[A, B, C, D]`:
/// - Bigrams: `[A,B]` at pos 1, `[B,C]` at pos 2, `[C,D]` at pos 3
/// - Trigrams: `[A,B,C]` at pos 2, `[B,C,D]` at pos 3
///
/// Returns an [`EngramRequest`] with deduplicated shard requirements.
/// For seq_len < 2, returns an empty request (no N-grams possible).
pub fn prepare_engram_request(client: &EngramClient, tokens: &[u32]) -> Result<EngramRequest> {
    let seq_len = tokens.len();
    let mut lookups = Vec::new();
    let mut seen_shards = HashSet::new();
    let mut required_shards = Vec::new();

    // Extract bigrams (need at least 2 tokens)
    for i in 0..seq_len.saturating_sub(1) {
        let bigram = &tokens[i..i + 2];
        let lookup = client.lookup(bigram);
        collect_shards(client, &lookup, &mut seen_shards, &mut required_shards)?;
        lookups.push(NgramLookup {
            token_position: i + 1, // last token of bigram
            lookup,
        });
    }

    // Extract trigrams (need at least 3 tokens)
    for i in 0..seq_len.saturating_sub(2) {
        let trigram = &tokens[i..i + 3];
        let lookup = client.lookup(trigram);
        collect_shards(client, &lookup, &mut seen_shards, &mut required_shards)?;
        lookups.push(NgramLookup {
            token_position: i + 2, // last token of trigram
            lookup,
        });
    }

    Ok(EngramRequest {
        required_shards,
        lookups,
        seq_len,
    })
}

/// Helper: collect unique shard requests from a lookup.
/// Returns an error if a shard index is out of bounds in the manifest.
fn collect_shards(
    client: &EngramClient,
    lookup: &EngramLookup,
    seen: &mut HashSet<u64>,
    shards: &mut Vec<ShardRequest>,
) -> Result<()> {
    for &shard_idx in &lookup.shard_indices {
        if seen.insert(shard_idx) {
            let &cid = client.shard_cid(shard_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!(
                    "shard index {shard_idx} out of bounds in Engram manifest \
                     (num_shards={})",
                    client.config().num_shards
                ))
            })?;
            shards.push(ShardRequest {
                shard_index: shard_idx,
                cid,
            });
        }
    }
    Ok(())
}

/// Phase 2: Resolve shard data into a candle Tensor of embeddings.
///
/// The caller fetches each shard by CID and stores results in a
/// `HashMap<u64, Vec<u8>>` keyed by `shard_index` (not CID).
///
/// For each N-gram lookup, assembles per-head shard slices and calls
/// `client.resolve()` to get f32 bytes. Multiple N-grams at the same
/// position are summed element-wise. Positions with no N-gram coverage
/// get zero embeddings.
///
/// Returns a Tensor `[1, seq_len, embedding_dim]`.
pub fn resolve_engram_embeddings(
    client: &EngramClient,
    request: &EngramRequest,
    shard_data: &HashMap<u64, Vec<u8>>,
    device: &Device,
) -> Result<Tensor> {
    let embedding_dim = client.config().embedding_dim;
    let num_heads = client.config().num_heads as usize;

    // Empty sequence → zero-sized tensor.
    if request.seq_len == 0 {
        return Tensor::zeros(
            (1usize, 0usize, embedding_dim),
            candle_core::DType::F32,
            device,
        );
    }

    // Aggregation buffer: seq_len x embedding_dim, initialized to zero.
    let mut buffer = vec![0.0f32; request.seq_len * embedding_dim];

    for ngram in &request.lookups {
        // Assemble per-head shard slices for client.resolve().
        let mut head_slices: Vec<&[u8]> = Vec::with_capacity(num_heads);
        for &shard_idx in &ngram.lookup.shard_indices {
            let data = shard_data.get(&shard_idx).ok_or_else(|| {
                candle_core::Error::Msg(format!("missing shard data for shard_index {shard_idx}"))
            })?;
            head_slices.push(data.as_slice());
        }

        // Resolve: extract + aggregate f16->f32 across heads.
        let f32_bytes = client
            .resolve(&ngram.lookup, &head_slices)
            .map_err(|e| candle_core::Error::Msg(format!("engram resolve failed: {e}")))?;

        debug_assert_eq!(
            f32_bytes.len(),
            embedding_dim * 4,
            "resolve returned {} bytes; expected {}",
            f32_bytes.len(),
            embedding_dim * 4,
        );

        // Interpret as f32 little-endian and sum into position buffer.
        let pos_offset = ngram.token_position * embedding_dim;
        for (i, chunk) in f32_bytes.chunks_exact(4).enumerate() {
            let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            buffer[pos_offset + i] += val;
        }
    }

    // Convert aggregated buffer to Tensor [1, seq_len, embedding_dim].
    Tensor::from_vec(buffer, (1, request.seq_len, embedding_dim), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_engram::EngramConfig;

    /// Create a minimal EngramClient for testing.
    fn test_client() -> EngramClient {
        let config = EngramConfig {
            version: "test".into(),
            embedding_dim: 4,
            dtype_bytes: 2, // f16
            num_heads: 2,
            shard_size: 100,
            num_shards: 10,
            total_entries: 1000,
            hash_seeds: vec![42, 99],
        };
        // 10 shards with dummy CIDs
        let shard_cids: Vec<[u8; 32]> = (0..10)
            .map(|i| {
                let mut cid = [0u8; 32];
                cid[0] = i as u8;
                cid
            })
            .collect();
        EngramClient::from_manifest(config, shard_cids)
    }

    /// Create mock shard data: all zeros (valid f16 zero = 0x0000).
    fn zero_shard_data(num_shards: usize, shard_bytes: usize) -> HashMap<u64, Vec<u8>> {
        (0..num_shards as u64)
            .map(|i| (i, vec![0u8; shard_bytes]))
            .collect()
    }

    #[test]
    fn ngram_extraction_bigrams_and_trigrams() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[1, 2, 3, 4]).unwrap();

        // 3 bigrams + 2 trigrams = 5 lookups
        assert_eq!(request.lookups.len(), 5);
        assert_eq!(request.seq_len, 4);

        // Bigrams at positions 1, 2, 3
        assert_eq!(request.lookups[0].token_position, 1);
        assert_eq!(request.lookups[1].token_position, 2);
        assert_eq!(request.lookups[2].token_position, 3);

        // Trigrams at positions 2, 3
        assert_eq!(request.lookups[3].token_position, 2);
        assert_eq!(request.lookups[4].token_position, 3);
    }

    #[test]
    fn single_token_produces_empty_request() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[42]).unwrap();

        assert!(request.lookups.is_empty());
        assert!(request.required_shards.is_empty());
        assert_eq!(request.seq_len, 1);
    }

    #[test]
    fn empty_tokens_produces_empty_request() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[]).unwrap();

        assert!(request.lookups.is_empty());
        assert!(request.required_shards.is_empty());
        assert_eq!(request.seq_len, 0);
    }

    #[test]
    fn shard_deduplication() {
        let client = test_client();
        let request = prepare_engram_request(&client, &[1, 2, 3, 4]).unwrap();

        // Multiple lookups may hit the same shard — required_shards should be deduplicated
        let shard_indices: Vec<u64> = request
            .required_shards
            .iter()
            .map(|s| s.shard_index)
            .collect();
        let unique: HashSet<u64> = shard_indices.iter().copied().collect();
        assert_eq!(
            shard_indices.len(),
            unique.len(),
            "shards should be deduplicated"
        );
    }

    #[test]
    fn resolve_produces_correct_shape() {
        let client = test_client();
        let tokens = [1u32, 2, 3, 4];
        let request = prepare_engram_request(&client, &tokens).unwrap();

        // Each shard needs to be large enough: shard_size(100) * vector_bytes(4*2=8) = 800 bytes
        let shard_data = zero_shard_data(10, 800);

        let tensor = resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu)
            .expect("resolve failed");

        assert_eq!(tensor.dims(), &[1, 4, 4]); // [1, seq_len=4, embedding_dim=4]
    }

    #[test]
    fn resolve_missing_shard_returns_error() {
        let client = test_client();
        let tokens = [1u32, 2, 3];
        let request = prepare_engram_request(&client, &tokens).unwrap();

        // Provide empty HashMap — all shards missing
        let shard_data = HashMap::new();

        let result = resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu);

        assert!(result.is_err(), "should fail when shard data is missing");
    }

    #[test]
    fn resolve_zero_shards_produces_zero_tensor() {
        let client = test_client();
        let tokens = [1u32, 2, 3, 4];
        let request = prepare_engram_request(&client, &tokens).unwrap();

        let shard_data = zero_shard_data(10, 800);

        let tensor = resolve_engram_embeddings(&client, &request, &shard_data, &Device::Cpu)
            .expect("resolve failed");

        // All-zero shards -> all-zero embeddings
        let max_val: f32 = tensor
            .abs()
            .unwrap()
            .max_all()
            .unwrap()
            .to_scalar()
            .unwrap();
        assert!(
            max_val < 1e-6,
            "zero shards should produce zero embeddings, got {max_val}"
        );
    }
}

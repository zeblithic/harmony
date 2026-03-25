//! f16→f32 vector extraction and multi-head aggregation.
//!
//! Each shard is a contiguous block of f16 embedding vectors.  Given a byte
//! offset, extract the vector, decode each f16 component to f32, and sum
//! across all heads in f32 precision to avoid f16 overflow.

use alloc::vec;
use alloc::vec::Vec;
use half::f16;
use crate::{EngramConfig, EngramError, EngramLookup};

/// Extract and aggregate embedding vectors from fetched shard bytes.
///
/// For each head, extracts `embedding_dim` f16 values at the corresponding
/// byte offset, converts to f32, and sums all heads component-wise.
///
/// Returns f32 little-endian bytes (length = `embedding_dim * 4`).
pub fn aggregate(
    config: &EngramConfig,
    lookup: &EngramLookup,
    shard_data: &[&[u8]],
) -> Result<Vec<u8>, EngramError> {
    let num_heads = lookup.shard_indices.len();
    if shard_data.len() != num_heads {
        return Err(EngramError::ShardCountMismatch {
            expected: num_heads,
            got: shard_data.len(),
        });
    }

    let dim = config.embedding_dim;
    let vector_bytes = config.vector_bytes();

    // Accumulator in f32 precision.
    let mut acc = vec![0.0f32; dim];

    for (i, (shard, &offset)) in shard_data.iter().zip(&lookup.entry_offsets).enumerate() {
        let end = match offset.checked_add(vector_bytes) {
            Some(end) => end,
            None => {
                return Err(EngramError::ShardTooShort {
                    shard_index: lookup.shard_indices[i],
                    offset,
                    vector_bytes,
                    shard_len: shard.len(),
                });
            }
        };
        if end > shard.len() {
            return Err(EngramError::ShardTooShort {
                shard_index: lookup.shard_indices[i],
                offset,
                vector_bytes,
                shard_len: shard.len(),
            });
        }

        let slice = &shard[offset..end];
        for (j, chunk) in slice.chunks_exact(2).enumerate() {
            let val = f16::from_le_bytes([chunk[0], chunk[1]]);
            acc[j] += val.to_f32();
        }
    }

    // Encode accumulator as f32 little-endian bytes.
    let mut out = Vec::with_capacity(dim * 4);
    for val in &acc {
        out.extend_from_slice(&val.to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::String;

    fn make_f16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|v| f16::from_f32(*v).to_le_bytes())
            .collect()
    }

    fn read_f32_vec(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn test_config() -> EngramConfig {
        EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 2,
            shard_size: 3,
            num_shards: 4,
            total_entries: 12,
            hash_seeds: vec![42, 99],
        }
    }

    #[test]
    fn single_head_extraction() {
        // 1-head config, extract vector at offset 0.
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![0],
        };
        let result = aggregate(&config, &lookup, &[&shard]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn single_head_nonzero_offset() {
        // Extract from the second vector slot (offset = 8 bytes).
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        let shard = make_f16_bytes(&[
            0.0, 0.0, 0.0, 0.0, // slot 0
            5.0, 6.0, 7.0, 8.0, // slot 1 — target
            0.0, 0.0, 0.0, 0.0, // slot 2
        ]);
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![8], // 1 * 4 * 2 = 8
        };
        let result = aggregate(&config, &lookup, &[&shard]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn multi_head_sum() {
        // 2 heads, different shards, sum component-wise.
        let config = test_config();
        let shard_a = make_f16_bytes(&[
            1.0, 2.0, 3.0, 4.0, // slot 0 — head 0 reads here
            0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ]);
        let shard_b = make_f16_bytes(&[
            0.0, 0.0, 0.0, 0.0,
            0.5, 1.5, 2.5, 3.5, // slot 1 — head 1 reads here
            0.0, 0.0, 0.0, 0.0,
        ]);
        let lookup = EngramLookup {
            shard_indices: vec![0, 1],
            entry_offsets: vec![0, 8],
        };
        let result = aggregate(&config, &lookup, &[&shard_a, &shard_b]).unwrap();
        let values = read_f32_vec(&result);
        assert_eq!(values, vec![1.5, 3.5, 5.5, 7.5]);
    }

    #[test]
    fn shard_count_mismatch() {
        let config = test_config();
        let lookup = EngramLookup {
            shard_indices: vec![0, 1],
            entry_offsets: vec![0, 0],
        };
        // Only provide 1 shard for a 2-head lookup.
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let err = aggregate(&config, &lookup, &[&shard]).unwrap_err();
        assert!(matches!(err, EngramError::ShardCountMismatch { expected: 2, got: 1 }));
    }

    #[test]
    fn shard_too_short() {
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 4,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 3,
            num_shards: 1,
            total_entries: 3,
            hash_seeds: vec![0],
        };
        // Shard only has 4 bytes, but we need 8 at offset 0.
        let shard = [0u8; 4];
        let lookup = EngramLookup {
            shard_indices: vec![0],
            entry_offsets: vec![0],
        };
        let err = aggregate(&config, &lookup, &[&shard[..]]).unwrap_err();
        assert!(matches!(err, EngramError::ShardTooShort { .. }));
    }

    #[test]
    fn output_length() {
        let config = test_config(); // embedding_dim=4
        let shard = make_f16_bytes(&[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let lookup = EngramLookup {
            shard_indices: vec![0, 0],
            entry_offsets: vec![0, 0],
        };
        let result = aggregate(&config, &lookup, &[&shard, &shard]).unwrap();
        // embedding_dim * 4 bytes per f32 = 4 * 4 = 16
        assert_eq!(result.len(), 16);
    }
}

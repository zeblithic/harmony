//! xxhash64 multi-head hashing and shard resolution.
//!
//! For each N-gram, computes `num_heads` independent xxhash64 values using
//! per-head seeds.  Each hash maps to a table index, which determines the
//! shard index and byte offset within that shard.

use crate::{EngramConfig, EngramLookup};
use alloc::vec::Vec;

/// Compute the [`EngramLookup`] for an N-gram given the table config.
///
/// 1. Encode tokens as little-endian bytes
/// 2. For each head seed: `table_index = xxhash64(bytes, seed) % total_entries`
/// 3. `shard_index = table_index / shard_size`
/// 4. `byte_offset = (table_index % shard_size) * vector_bytes`
///
/// # Panics
///
/// Panics if `config.total_entries == 0` or `config.shard_size == 0` (division
/// by zero).  In debug builds these are caught by `debug_assert`; in release
/// builds the modulo/division will panic.  Callers must ensure the config is
/// constructed from a valid [`ManifestHeader`](crate::ManifestHeader).
pub fn compute_lookup(config: &EngramConfig, ngram_tokens: &[u32]) -> EngramLookup {
    debug_assert!(config.total_entries > 0, "total_entries must be positive");
    debug_assert!(config.shard_size > 0, "shard_size must be positive");
    debug_assert_eq!(
        config.hash_seeds.len(),
        config.num_heads as usize,
        "hash_seeds length must match num_heads"
    );

    let vector_bytes = config.vector_bytes();
    let mut shard_indices = Vec::with_capacity(config.num_heads as usize);
    let mut entry_offsets = Vec::with_capacity(config.num_heads as usize);

    for seed in &config.hash_seeds {
        let raw_hash = hash_ngram(ngram_tokens, *seed);
        let table_index = raw_hash % config.total_entries;
        let shard_index = table_index / config.shard_size as u64;
        let entry_within_shard = (table_index % config.shard_size as u64) as usize;
        let byte_offset = entry_within_shard * vector_bytes;

        shard_indices.push(shard_index);
        entry_offsets.push(byte_offset);
    }

    EngramLookup {
        shard_indices,
        entry_offsets,
    }
}

/// Hash arbitrary bytes with a single seed.
fn hash_bytes(bytes: &[u8], seed: u64) -> u64 {
    xxhash_rust::xxh64::xxh64(bytes, seed)
}

/// Hash an N-gram's token bytes with a single seed.
///
/// Tokens are encoded as contiguous little-endian u32 bytes.
fn hash_ngram(tokens: &[u32], seed: u64) -> u64 {
    let byte_len = tokens.len() * 4;
    if byte_len <= 128 {
        let mut buf = [0u8; 128];
        for (i, t) in tokens.iter().enumerate() {
            buf[i * 4..(i + 1) * 4].copy_from_slice(&t.to_le_bytes());
        }
        hash_bytes(&buf[..byte_len], seed)
    } else {
        let bytes: Vec<u8> = tokens.iter().flat_map(|t| t.to_le_bytes()).collect();
        hash_bytes(&bytes, seed)
    }
}

/// Compute the [`EngramLookup`] for arbitrary key bytes.
///
/// Same hashing logic as [`compute_lookup`] but accepts raw bytes instead of
/// token N-grams. Used by latent projection to hash binary LSH codes.
pub fn compute_lookup_from_bytes(config: &EngramConfig, key_bytes: &[u8]) -> EngramLookup {
    debug_assert!(config.total_entries > 0, "total_entries must be positive");
    debug_assert!(config.shard_size > 0, "shard_size must be positive");
    debug_assert_eq!(
        config.hash_seeds.len(),
        config.num_heads as usize,
        "hash_seeds length must match num_heads"
    );

    let vector_bytes = config.vector_bytes();
    let mut shard_indices = Vec::with_capacity(config.num_heads as usize);
    let mut entry_offsets = Vec::with_capacity(config.num_heads as usize);

    for seed in &config.hash_seeds {
        let raw_hash = hash_bytes(key_bytes, *seed);
        let table_index = raw_hash % config.total_entries;
        let shard_index = table_index / config.shard_size as u64;
        let entry_within_shard = (table_index % config.shard_size as u64) as usize;
        let byte_offset = entry_within_shard * vector_bytes;

        shard_indices.push(shard_index);
        entry_offsets.push(byte_offset);
    }

    EngramLookup {
        shard_indices,
        entry_offsets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::{string::String, vec};

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
    fn hash_determinism() {
        // Same input + seed must produce identical output across calls.
        let tokens = [1u32, 2, 3];
        let h1 = hash_ngram(&tokens, 42);
        let h2 = hash_ngram(&tokens, 42);
        assert_eq!(h1, h2);
    }

    #[test]
    fn different_seeds_produce_different_hashes() {
        let tokens = [1u32, 2, 3];
        let h1 = hash_ngram(&tokens, 42);
        let h2 = hash_ngram(&tokens, 99);
        assert_ne!(h1, h2);
        // Pin concrete values — xxhash64 is a fixed algorithm.
        // Any change here means the hash function changed, which would
        // silently corrupt all existing Engram table lookups.
        assert_eq!(h1, 11547749587308120431_u64);
        assert_eq!(h2, 469971568748895552_u64);
    }

    #[test]
    fn lookup_returns_correct_head_count() {
        let config = test_config();
        let lookup = compute_lookup(&config, &[1, 2, 3]);
        assert_eq!(lookup.shard_indices.len(), 2);
        assert_eq!(lookup.entry_offsets.len(), 2);
    }

    #[test]
    fn lookup_shard_indices_in_bounds() {
        let config = test_config();
        let lookup = compute_lookup(&config, &[100, 200, 300]);
        for &idx in &lookup.shard_indices {
            assert!(
                idx < config.num_shards,
                "shard index {idx} >= {}",
                config.num_shards
            );
        }
    }

    #[test]
    fn lookup_entry_offsets_in_bounds() {
        let config = test_config();
        let vector_bytes = config.vector_bytes();
        let shard_bytes = config.shard_size as usize * vector_bytes;
        let lookup = compute_lookup(&config, &[7, 8, 9]);
        for &offset in &lookup.entry_offsets {
            assert!(
                offset + vector_bytes <= shard_bytes,
                "offset {offset} + {vector_bytes} exceeds shard size {shard_bytes}"
            );
        }
    }

    #[test]
    fn lookup_boundary_single_shard_table() {
        // Edge case: table with 1 shard, 1 entry.
        let config = EngramConfig {
            version: String::from("v1"),
            embedding_dim: 2,
            dtype_bytes: 2,
            num_heads: 1,
            shard_size: 1,
            num_shards: 1,
            total_entries: 1,
            hash_seeds: vec![0],
        };
        let lookup = compute_lookup(&config, &[42]);
        assert_eq!(lookup.shard_indices, vec![0]);
        assert_eq!(lookup.entry_offsets, vec![0]);
    }

    #[test]
    fn hash_empty_tokens() {
        // Empty N-gram is a degenerate case but must not panic.
        let config = test_config();
        let lookup = compute_lookup(&config, &[]);
        assert_eq!(lookup.shard_indices.len(), 2);
    }

    #[test]
    fn compute_lookup_pinned_shard_and_offset_values() {
        // Pin concrete shard_indices and entry_offsets for test_config() + [1,2,3].
        // Any change here means compute_lookup changed behavior, which would
        // silently corrupt existing Engram table lookups.
        let config = test_config();
        let lookup = compute_lookup(&config, &[1, 2, 3]);
        // Derived from xxhash64([1,2,3] as LE bytes, seed) % 12, then /3 and %3*8
        assert_eq!(lookup.shard_indices, vec![3, 0]);
        assert_eq!(lookup.entry_offsets, vec![16, 0]);
    }

    #[test]
    fn compute_lookup_from_bytes_matches_manual() {
        let config = test_config();
        let bytes: Vec<u8> = [1u32, 2, 3]
            .iter()
            .flat_map(|t| t.to_le_bytes())
            .collect();
        let from_bytes = compute_lookup_from_bytes(&config, &bytes);
        let from_tokens = compute_lookup(&config, &[1, 2, 3]);
        assert_eq!(from_bytes.shard_indices, from_tokens.shard_indices);
        assert_eq!(from_bytes.entry_offsets, from_tokens.entry_offsets);
    }
}

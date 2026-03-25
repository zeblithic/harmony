/// Errors that can occur during Engram operations.
#[derive(Debug, thiserror::Error)]
pub enum EngramError {
    /// Shard index exceeds the manifest's shard count.
    #[error("shard index {index} out of bounds (num_shards={num_shards})")]
    ShardIndexOutOfBounds { index: u64, num_shards: u64 },
    /// The number of shard data slices doesn't match the lookup's head count.
    #[error("expected {expected} shard slices, got {got}")]
    ShardCountMismatch { expected: usize, got: usize },
    /// A shard's byte slice is too short to extract the vector at the given offset.
    #[error("shard {shard_index} too short: need {vector_bytes} bytes at offset {offset}, but shard is {shard_len} bytes")]
    ShardTooShort {
        shard_index: u64,
        offset: usize,
        vector_bytes: usize,
        shard_len: usize,
    },
    /// Manifest deserialization failed.
    #[error("failed to deserialize manifest header")]
    ManifestDeserialize,
    /// Manifest serialization failed.
    #[error("failed to serialize manifest header")]
    ManifestSerialize,
}

/// Errors that can occur during Engram operations.
#[derive(Debug)]
pub enum EngramError {
    /// Shard index exceeds the manifest's shard count.
    ShardIndexOutOfBounds { index: u64, num_shards: u64 },
    /// The number of shard data slices doesn't match the lookup's head count.
    ShardCountMismatch { expected: usize, got: usize },
    /// A shard's byte slice is too short to extract the vector at the given offset.
    ShardTooShort {
        shard_index: u64,
        offset: usize,
        vector_bytes: usize,
        shard_len: usize,
    },
    /// Manifest deserialization failed.
    ManifestDeserialize,
}

impl core::fmt::Display for EngramError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::ShardIndexOutOfBounds { index, num_shards } => {
                write!(f, "shard index {index} out of bounds (num_shards={num_shards})")
            }
            Self::ShardCountMismatch { expected, got } => {
                write!(f, "expected {expected} shard slices, got {got}")
            }
            Self::ShardTooShort {
                shard_index,
                offset,
                vector_bytes,
                shard_len,
            } => {
                write!(
                    f,
                    "shard {shard_index} too short: need {vector_bytes} bytes at offset {offset}, \
                     but shard is {shard_len} bytes"
                )
            }
            Self::ManifestDeserialize => write!(f, "failed to deserialize manifest header"),
        }
    }
}

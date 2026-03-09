/// Errors produced by content lifecycle operations.
#[derive(Debug, thiserror::Error)]
pub enum JainError {
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("key expression segment contains Zenoh metacharacters")]
    InvalidKeySegment,
}

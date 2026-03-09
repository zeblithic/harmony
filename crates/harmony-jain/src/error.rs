use alloc::string::String;

/// Errors produced by content lifecycle operations.
#[derive(Debug, thiserror::Error)]
pub enum JainError {
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("key expression segment contains Zenoh metacharacters")]
    InvalidKeySegment,

    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

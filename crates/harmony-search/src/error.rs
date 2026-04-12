//! Error types for harmony-search.

/// Errors from vector index operations.
#[derive(Debug, thiserror::Error)]
pub enum SearchError {
    /// USearch operation failed.
    #[error("index error: {0}")]
    Index(String),

    /// Invalid configuration.
    #[error("invalid config: {0}")]
    InvalidConfig(String),

    /// Serialization/deserialization failed.
    #[error("serialization error: {0}")]
    Serialization(String),
}

/// Result type for vector index operations.
pub type SearchResult<T> = Result<T, SearchError>;

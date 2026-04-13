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

    /// A replace-add failed and the old vector could not be restored.
    ///
    /// The index is in a degraded state: `delta_keys` still claims ownership
    /// of the key, but neither delta nor base has a valid vector for it.
    /// Callers must not attempt to roll back metadata — the entry is orphaned
    /// and should be cleaned up.
    #[error("rollback failed: add error: {add_error}, restore error: {restore_error}")]
    RollbackFailed {
        /// The original add error.
        add_error: String,
        /// The error from trying to restore the old vector.
        restore_error: String,
    },
}

/// Result type for vector index operations.
pub type SearchResult<T> = Result<T, SearchError>;

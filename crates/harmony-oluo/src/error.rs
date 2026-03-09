// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for harmony-oluo.

use core::fmt;

/// Errors from Oluo search engine operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OluoError {
    /// The underlying semantic index operation failed.
    Semantic(harmony_semantic::SemanticError),
    /// Trie node magic bytes don't match expected format.
    InvalidTrieNode,
    /// Trie node data is too short.
    TruncatedTrieNode { expected: usize, actual: usize },
    /// Search query references a tier not available in the index.
    TierUnavailable,
    /// Ingest was rejected by the privacy gate.
    IngestRejected,
}

impl fmt::Display for OluoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Semantic(e) => write!(f, "semantic error: {e}"),
            Self::InvalidTrieNode => write!(f, "invalid trie node magic bytes"),
            Self::TruncatedTrieNode { expected, actual } => {
                write!(
                    f,
                    "truncated trie node: expected {expected} bytes, got {actual}"
                )
            }
            Self::TierUnavailable => write!(f, "requested tier not available in index"),
            Self::IngestRejected => write!(f, "ingest rejected by privacy gate"),
        }
    }
}

impl From<harmony_semantic::SemanticError> for OluoError {
    fn from(e: harmony_semantic::SemanticError) -> Self {
        Self::Semantic(e)
    }
}

pub type OluoResult<T> = Result<T, OluoError>;

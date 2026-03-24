//! Error types for the harmony-s3 crate.

use std::fmt;

/// Errors that can occur in S3 library operations.
#[derive(Debug)]
pub enum S3Error {
    /// A PutObject request failed.
    PutFailed(String),
    /// A GetObject request failed.
    GetFailed(String),
    /// A HeadObject request failed.
    HeadFailed(String),
    /// Client configuration failed.
    ConfigError(String),
}

impl fmt::Display for S3Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PutFailed(msg) => write!(f, "S3 put failed: {msg}"),
            Self::GetFailed(msg) => write!(f, "S3 get failed: {msg}"),
            Self::HeadFailed(msg) => write!(f, "S3 head failed: {msg}"),
            Self::ConfigError(msg) => write!(f, "S3 config error: {msg}"),
        }
    }
}

impl std::error::Error for S3Error {}

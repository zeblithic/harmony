// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for harmony-semantic.

use alloc::string::String;
use core::fmt;

/// Errors from semantic index operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SemanticError {
    /// Sidecar magic bytes don't match expected format.
    InvalidMagic,
    /// Sidecar data is too short to contain the fixed header.
    TruncatedHeader { expected: usize, actual: usize },
    /// Collection entry count exceeds maximum (256).
    CollectionOverflow { count: u32 },
    /// Model fingerprint mismatch during overlay merge.
    FingerprintMismatch,
    /// CBOR metadata decoding failed.
    MetadataInvalid { reason: String },
    /// Privacy tier 3 (encrypted-ephemeral) cannot be indexed.
    PrivacyBlocked,
    /// Input vector has wrong number of dimensions for quantization.
    DimensionMismatch { expected: usize, actual: usize },
}

impl fmt::Display for SemanticError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidMagic => write!(f, "invalid sidecar magic bytes"),
            Self::TruncatedHeader { expected, actual } => {
                write!(
                    f,
                    "truncated header: expected {expected} bytes, got {actual}"
                )
            }
            Self::CollectionOverflow { count } => {
                write!(f, "collection overflow: {count} entries exceeds max 256")
            }
            Self::FingerprintMismatch => write!(f, "model fingerprint mismatch"),
            Self::MetadataInvalid { reason } => {
                write!(f, "invalid CBOR metadata: {reason}")
            }
            Self::PrivacyBlocked => {
                write!(f, "encrypted-ephemeral content cannot be indexed")
            }
            Self::DimensionMismatch { expected, actual } => {
                write!(f, "dimension mismatch: expected {expected}, got {actual}")
            }
        }
    }
}

pub type SemanticResult<T> = Result<T, SemanticError>;

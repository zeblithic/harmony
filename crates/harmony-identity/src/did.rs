//! DID (Decentralized Identifier) resolution for Harmony.
//!
//! Supports:
//! - `did:key` — multicodec-prefixed public keys (Ed25519, ML-DSA-65)
//! - `did:jwk` — JSON Web Key encoded DIDs (std only)

use alloc::string::String;
use alloc::vec::Vec;

use crate::crypto_suite::CryptoSuite;

/// A resolved DID containing the cryptographic suite and raw public key bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedDid {
    /// The cryptographic suite identified by the DID.
    pub suite: CryptoSuite,
    /// The raw public key bytes.
    pub public_key: Vec<u8>,
}

/// Errors that can occur during DID resolution.
#[non_exhaustive]
#[derive(Debug)]
pub enum DidError {
    /// The DID method is not supported.
    UnsupportedMethod(String),
    /// The DID string is malformed.
    MalformedDid(String),
    /// Base58 or base64 decoding failed.
    DecodingError(String),
    /// The multicodec prefix is not recognized.
    UnknownMulticodec(u32),
}

impl core::fmt::Display for DidError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedMethod(method) => {
                write!(f, "unsupported DID method: {method}")
            }
            Self::MalformedDid(msg) => write!(f, "malformed DID: {msg}"),
            Self::DecodingError(msg) => write!(f, "decoding error: {msg}"),
            Self::UnknownMulticodec(code) => {
                write!(f, "unknown multicodec: 0x{code:04x}")
            }
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DidError {}

/// Trait for DID resolution.
pub trait DidResolver {
    /// Resolve a DID string into its public key and crypto suite.
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError>;
}

/// Default DID resolver supporting `did:key` and `did:jwk` (std only).
pub struct DefaultDidResolver;

impl DidResolver for DefaultDidResolver {
    fn resolve(&self, did: &str) -> Result<ResolvedDid, DidError> {
        resolve_did(did)
    }
}

/// Resolve a DID string into its public key and crypto suite.
///
/// Dispatches on method prefix:
/// - `did:key:` — multicodec-prefixed public key
/// - `did:jwk:` — JSON Web Key (std only)
/// - `did:web:` — not supported
pub fn resolve_did(_did: &str) -> Result<ResolvedDid, DidError> {
    todo!()
}

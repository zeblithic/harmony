//! UCAN capability tokens — Harmony-native authorization primitives.
//!
//! Implements delegation chains with Ed25519 signatures, content-addressed
//! proof references, and configurable revocation. Follows UCAN authorization
//! principles with a compact binary format optimized for kernel verification.

use alloc::string::String;
use alloc::vec::Vec;

use harmony_crypto::hash;
use harmony_platform::EntropySource;

use crate::identity::{Identity, PrivateIdentity, ADDRESS_HASH_LENGTH, SIGNATURE_LENGTH};
use crate::IdentityError;

/// Maximum size of the resource field in bytes.
pub const MAX_RESOURCE_SIZE: usize = 256;

/// Capability types that can be granted by a UCAN token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapabilityType {
    /// Read/write access to content-addressed memory.
    Memory = 0,
    /// Access to network transport interfaces.
    Network = 1,
    /// Access to identity operations (signing, key management).
    Identity = 2,
    /// Access to persistent storage.
    Storage = 3,
    /// Access to publish/subscribe messaging.
    Messaging = 4,
    /// Access to workflow execution.
    Workflow = 5,
    /// Access to compute resources (WASM execution).
    Compute = 6,
}

impl TryFrom<u8> for CapabilityType {
    type Error = UcanError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Memory),
            1 => Ok(Self::Network),
            2 => Ok(Self::Identity),
            3 => Ok(Self::Storage),
            4 => Ok(Self::Messaging),
            5 => Ok(Self::Workflow),
            6 => Ok(Self::Compute),
            _ => Err(UcanError::InvalidEncoding),
        }
    }
}

/// Errors produced by UCAN operations.
#[derive(Debug, thiserror::Error)]
pub enum UcanError {
    /// The token's `expires_at` timestamp has passed.
    #[error("token has expired")]
    Expired,

    /// The token's `not_before` timestamp has not yet been reached.
    #[error("token is not yet valid")]
    NotYetValid,

    /// The Ed25519 signature on the token is invalid.
    #[error("signature verification failed")]
    SignatureInvalid,

    /// The proof chain exceeds the maximum allowed depth.
    #[error("proof chain too deep: {0} levels")]
    ChainTooDeep(usize),

    /// A proof referenced by hash could not be resolved.
    #[error("proof not found in resolver")]
    ProofNotFound,

    /// The proof chain is broken (audience/issuer mismatch).
    #[error("proof chain is broken")]
    ChainBroken,

    /// The token's capability type does not match its proof's capability.
    #[error("capability type mismatch between token and proof")]
    CapabilityMismatch,

    /// The delegated token attempts to escalate beyond its proof's authority.
    #[error("attenuation violation: delegated token exceeds proof authority")]
    AttenuationViolation,

    /// The token has been explicitly revoked.
    #[error("token has been revoked")]
    Revoked,

    /// The token's issuer could not be resolved to a known identity.
    #[error("issuer identity not found")]
    IssuerNotFound,

    /// The binary encoding is malformed or contains invalid values.
    #[error("invalid binary encoding")]
    InvalidEncoding,

    /// The resource field exceeds the maximum allowed size.
    #[error("resource exceeds maximum size of {MAX_RESOURCE_SIZE} bytes")]
    ResourceTooLarge,

    /// Wraps an identity error.
    #[error(transparent)]
    Identity(#[from] IdentityError),
}

/// A UCAN capability token.
#[derive(Debug, Clone)]
pub struct UcanToken {
    _placeholder: (),
}

/// A revocation record for a UCAN token.
#[derive(Debug, Clone)]
pub struct Revocation {
    /// The hash of the token being revoked.
    pub token_hash: [u8; 32],
}

/// Resolves proof chain references to their full tokens.
pub trait ProofResolver {
    /// Look up a token by its content hash.
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken>;
}

/// Resolves address hashes to public identities.
pub trait IdentityResolver {
    /// Look up an identity by its address hash.
    fn resolve(&self, address_hash: &[u8; ADDRESS_HASH_LENGTH]) -> Option<Identity>;
}

/// Checks whether a token has been revoked.
pub trait RevocationSet {
    /// Returns true if the token with the given hash has been revoked.
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool;
}

/// In-memory proof store for testing.
#[derive(Debug, Default)]
pub struct MemoryProofStore {
    _placeholder: (),
}

/// In-memory identity store for testing.
#[derive(Debug, Default)]
pub struct MemoryIdentityStore {
    _placeholder: (),
}

/// In-memory revocation set for testing.
#[derive(Debug, Default)]
pub struct MemoryRevocationSet {
    _placeholder: (),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capability_type_u8_roundtrip() {
        for val in 0u8..=6 {
            let cap = CapabilityType::try_from(val).unwrap();
            assert_eq!(cap as u8, val);
        }
    }

    #[test]
    fn capability_type_invalid_value_rejected() {
        assert!(CapabilityType::try_from(7u8).is_err());
        assert!(CapabilityType::try_from(255u8).is_err());
    }

    #[test]
    fn ucan_error_display() {
        let err = UcanError::Expired;
        assert_eq!(alloc::format!("{err}"), "token has expired");
    }
}

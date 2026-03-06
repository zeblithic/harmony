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
    /// Placeholder — will be defined in Task 2.
    Memory = 0,
}

/// Errors produced by UCAN operations.
#[derive(Debug, thiserror::Error)]
pub enum UcanError {
    /// Placeholder — will be defined in Task 2.
    #[error("token has expired")]
    Expired,

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

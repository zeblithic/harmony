//! UCAN capability tokens — Harmony-native authorization primitives.
//!
//! Implements delegation chains with Ed25519 signatures, content-addressed
//! proof references, and configurable revocation. Follows UCAN authorization
//! principles with a compact binary format optimized for kernel verification.

use alloc::vec::Vec;

use harmony_crypto::hash;
use harmony_platform::EntropySource;
#[cfg(any(test, feature = "test-utils"))]
use hashbrown::{HashMap, HashSet};

use crate::identity::{Identity, ADDRESS_HASH_LENGTH, SIGNATURE_LENGTH};
use crate::IdentityError;
use crate::PrivateIdentity;

/// Maximum size of the resource field in bytes.
pub const MAX_RESOURCE_SIZE: usize = 256;

/// ML-DSA-65 signature length (FIPS 204).
const PQ_SIGNATURE_LENGTH: usize = harmony_crypto::ml_dsa::SIG_LENGTH; // 3309

/// Identifies which cryptographic suite a serialized token uses.
///
/// The first byte of the wire format distinguishes classical Ed25519 tokens
/// from post-quantum ML-DSA-65 tokens so that parsers can dispatch to the
/// correct deserialization logic without trial-and-error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CryptoSuite {
    /// Classical Ed25519 signatures (64-byte).
    Ed25519 = 0x00,
    /// Post-quantum ML-DSA-65 signatures (3,309-byte, FIPS 204).
    MlDsa65 = 0x01,
}

/// Capability types matching the Ring 2 microkernel's resource model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CapabilityType {
    /// Access to a memory region.
    Memory = 0,
    /// Permission to send 9P messages to a server.
    Endpoint = 1,
    /// Permission to receive a hardware interrupt.
    Interrupt = 2,
    /// Access to a hardware I/O port range.
    IOPort = 3,
    /// Right to sign with a specific Harmony identity.
    Identity = 4,
    /// Right to read/write a CID range in the blob store.
    Content = 5,
    /// Right to execute WASM with a fuel budget.
    Compute = 6,
}

impl TryFrom<u8> for CapabilityType {
    type Error = UcanError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Memory),
            1 => Ok(Self::Endpoint),
            2 => Ok(Self::Interrupt),
            3 => Ok(Self::IOPort),
            4 => Ok(Self::Identity),
            5 => Ok(Self::Content),
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
    #[error("proof chain too deep: reached depth {depth}, limit {limit}")]
    ChainTooDeep {
        /// How deep the chain was when the limit was hit.
        depth: usize,
        /// The configured maximum depth.
        limit: usize,
    },

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

    /// The token's `not_before` timestamp exceeds its `expires_at`.
    #[error("invalid time window: not_before exceeds expires_at")]
    InvalidTimeWindow,

    /// The revocation's token hash does not match the supplied token.
    #[error("revocation targets a different token")]
    TokenHashMismatch,

    /// The resource field exceeds the maximum allowed size.
    #[error("resource exceeds maximum size of {MAX_RESOURCE_SIZE} bytes")]
    ResourceTooLarge,

    /// The caller is not the token's issuer (cannot revoke).
    #[error("only the token issuer can revoke it")]
    NotIssuer,

    /// Wraps an identity error.
    #[error(transparent)]
    Identity(#[from] IdentityError),
}

/// A UCAN capability token.
///
/// Wire format (big-endian):
/// ```text
/// [16B issuer][16B audience][1B capability][2B resource_len][NB resource]
/// [8B not_before][8B expires_at][16B nonce][1B has_proof][32B proof?]
/// [64B signature]
/// ```
///
/// Minimum size (no resource, no proof): 16+16+1+2+0+8+8+16+1+0+64 = 132 bytes.
#[derive(Debug, Clone)]
pub struct UcanToken {
    /// Address hash of the token issuer (signer).
    pub issuer: [u8; ADDRESS_HASH_LENGTH],
    /// Address hash of the token audience (recipient of the capability).
    pub audience: [u8; ADDRESS_HASH_LENGTH],
    /// The type of capability this token grants.
    pub capability: CapabilityType,
    /// Opaque resource identifier scoped by the capability type.
    pub resource: Vec<u8>,
    /// Unix timestamp (seconds) before which this token is not valid.
    pub not_before: u64,
    /// Unix timestamp (seconds) after which this token expires.
    pub expires_at: u64,
    /// Random nonce for uniqueness (prevents content-hash collisions).
    pub nonce: [u8; 16],
    /// BLAKE3 hash of the parent proof token, if this is a delegated token.
    pub proof: Option<[u8; 32]>,
    /// Ed25519 signature over the signable portion of this token.
    pub signature: [u8; SIGNATURE_LENGTH],
}

/// Minimum wire size of a serialized token (no resource, no proof).
const MIN_TOKEN_SIZE: usize = ADDRESS_HASH_LENGTH  // issuer
    + ADDRESS_HASH_LENGTH                           // audience
    + 1                                             // capability
    + 2                                             // resource_len
    + 8                                             // not_before
    + 8                                             // expires_at
    + 16                                            // nonce
    + 1                                             // has_proof
    + SIGNATURE_LENGTH; // signature

impl UcanToken {
    /// Serialize the token to its binary wire format.
    ///
    /// The caller must ensure `resource.len() <= MAX_RESOURCE_SIZE` (enforced
    /// by the creation APIs `issue_root_token` / `delegate`). The length is
    /// stored as a `u16`, so resources exceeding 65 535 bytes would be silently
    /// truncated — but `MAX_RESOURCE_SIZE` (256) makes that unreachable.
    pub fn to_bytes(&self) -> Vec<u8> {
        debug_assert!(
            self.resource.len() <= u16::MAX as usize,
            "resource exceeds u16 length limit; use issue_root_token/delegate to create tokens"
        );
        let resource_len = self.resource.len() as u16;
        let proof_size = if self.proof.is_some() { 32 } else { 0 };
        let total = MIN_TOKEN_SIZE + self.resource.len() + proof_size;

        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.audience);
        buf.push(self.capability as u8);
        buf.extend_from_slice(&resource_len.to_be_bytes());
        buf.extend_from_slice(&self.resource);
        buf.extend_from_slice(&self.not_before.to_be_bytes());
        buf.extend_from_slice(&self.expires_at.to_be_bytes());
        buf.extend_from_slice(&self.nonce);
        if let Some(ref proof) = self.proof {
            buf.push(1);
            buf.extend_from_slice(proof);
        } else {
            buf.push(0);
        }
        buf.extend_from_slice(&self.signature);
        buf
    }

    /// Deserialize a token from its binary wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, UcanError> {
        if data.len() < MIN_TOKEN_SIZE {
            return Err(UcanError::InvalidEncoding);
        }

        let mut pos = 0;

        let mut issuer = [0u8; ADDRESS_HASH_LENGTH];
        issuer.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LENGTH]);
        pos += ADDRESS_HASH_LENGTH;

        let mut audience = [0u8; ADDRESS_HASH_LENGTH];
        audience.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LENGTH]);
        pos += ADDRESS_HASH_LENGTH;

        let capability = CapabilityType::try_from(data[pos])?;
        pos += 1;

        let resource_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        if resource_len > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }

        if data.len() < MIN_TOKEN_SIZE + resource_len {
            return Err(UcanError::InvalidEncoding);
        }

        let resource = data[pos..pos + resource_len].to_vec();
        pos += resource_len;

        let not_before = u64::from_be_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| UcanError::InvalidEncoding)?,
        );
        pos += 8;

        let expires_at = u64::from_be_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| UcanError::InvalidEncoding)?,
        );
        pos += 8;

        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&data[pos..pos + 16]);
        pos += 16;

        let has_proof = data[pos];
        pos += 1;

        let proof = match has_proof {
            0 => None,
            1 => {
                if data.len() < pos + 32 + SIGNATURE_LENGTH {
                    return Err(UcanError::InvalidEncoding);
                }
                let mut proof_hash = [0u8; 32];
                proof_hash.copy_from_slice(&data[pos..pos + 32]);
                pos += 32;
                Some(proof_hash)
            }
            _ => return Err(UcanError::InvalidEncoding),
        };

        if data.len() != pos + SIGNATURE_LENGTH {
            return Err(UcanError::InvalidEncoding);
        }

        let mut signature = [0u8; SIGNATURE_LENGTH];
        signature.copy_from_slice(&data[pos..pos + SIGNATURE_LENGTH]);

        Ok(Self {
            issuer,
            audience,
            capability,
            resource,
            not_before,
            expires_at,
            nonce,
            proof,
            signature,
        })
    }

    /// Compute the BLAKE3 content hash of this token's wire representation.
    ///
    /// The hash covers the entire serialized form including the signature,
    /// so each signed token instance has a unique content hash. This is how
    /// child tokens reference their parent in the proof chain.
    pub fn content_hash(&self) -> [u8; 32] {
        hash::blake3_hash(&self.to_bytes())
    }

    /// Return the signable portion of the token (everything except the signature).
    pub fn signable_bytes(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        // The signature is always the last SIGNATURE_LENGTH bytes.
        bytes[..bytes.len() - SIGNATURE_LENGTH].to_vec()
    }
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
///
/// Stores tokens keyed by their BLAKE3 content hash.
/// Only available with the `test-utils` feature or in test builds.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Default)]
pub struct MemoryProofStore {
    tokens: HashMap<[u8; 32], UcanToken>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryProofStore {
    /// Create a new empty proof store.
    pub fn new() -> Self {
        Self {
            tokens: HashMap::new(),
        }
    }

    /// Insert a token, keyed by its content hash.
    pub fn insert(&mut self, token: UcanToken) {
        let hash = token.content_hash();
        self.tokens.insert(hash, token);
    }

    /// Insert a token under an explicit key, regardless of the token's actual
    /// content hash. Useful for testing adversarial `ProofResolver` behavior
    /// (e.g. returning the wrong token for a given hash).
    pub fn insert_with_key(&mut self, key: [u8; 32], token: UcanToken) {
        self.tokens.insert(key, token);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl ProofResolver for MemoryProofStore {
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken> {
        self.tokens.get(hash).cloned()
    }
}

/// In-memory identity store for testing.
///
/// Stores identities keyed by their address hash.
/// Only available with the `test-utils` feature or in test builds.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Default)]
pub struct MemoryIdentityStore {
    identities: HashMap<[u8; ADDRESS_HASH_LENGTH], Identity>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryIdentityStore {
    /// Create a new empty identity store.
    pub fn new() -> Self {
        Self {
            identities: HashMap::new(),
        }
    }

    /// Insert an identity, keyed by its address hash.
    pub fn insert(&mut self, identity: Identity) {
        let hash = identity.address_hash;
        self.identities.insert(hash, identity);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl IdentityResolver for MemoryIdentityStore {
    fn resolve(&self, address_hash: &[u8; ADDRESS_HASH_LENGTH]) -> Option<Identity> {
        self.identities.get(address_hash).cloned()
    }
}

/// In-memory revocation set for testing.
///
/// Stores the BLAKE3 content hashes of revoked tokens.
/// Only available with the `test-utils` feature or in test builds.
#[cfg(any(test, feature = "test-utils"))]
#[derive(Debug, Default)]
pub struct MemoryRevocationSet {
    revoked: HashSet<[u8; 32]>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryRevocationSet {
    /// Create a new empty revocation set.
    pub fn new() -> Self {
        Self {
            revoked: HashSet::new(),
        }
    }

    /// Mark a token hash as revoked.
    pub fn insert(&mut self, hash: [u8; 32]) {
        self.revoked.insert(hash);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl RevocationSet for MemoryRevocationSet {
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool {
        self.revoked.contains(token_hash)
    }
}

/// A post-quantum UCAN capability token using ML-DSA-65 signatures.
///
/// Wire format (big-endian):
/// ```text
/// [1B crypto_suite=0x01][16B issuer][16B audience][1B capability]
/// [2B resource_len][NB resource][8B not_before][8B expires_at]
/// [16B nonce][1B has_proof][32B proof?]
/// [3309B ML-DSA-65 signature]
/// ```
///
/// Minimum size (no resource, no proof):
/// 1+16+16+1+2+0+8+8+16+1+0+3309 = 3378 bytes.
#[derive(Debug, Clone)]
pub struct PqUcanToken {
    /// Address hash of the token issuer (signer).
    pub issuer: [u8; ADDRESS_HASH_LENGTH],
    /// Address hash of the token audience (recipient of the capability).
    pub audience: [u8; ADDRESS_HASH_LENGTH],
    /// The type of capability this token grants.
    pub capability: CapabilityType,
    /// Opaque resource identifier scoped by the capability type.
    pub resource: Vec<u8>,
    /// Unix timestamp (seconds) before which this token is not valid.
    pub not_before: u64,
    /// Unix timestamp (seconds) after which this token expires.
    pub expires_at: u64,
    /// Random nonce for uniqueness (prevents content-hash collisions).
    pub nonce: [u8; 16],
    /// BLAKE3 hash of the parent proof token, if this is a delegated token.
    pub proof: Option<[u8; 32]>,
    /// ML-DSA-65 signature over the signable portion of this token (3,309 bytes).
    pub signature: Vec<u8>,
}

/// Minimum wire size of a serialized PQ token (no resource, no proof).
const PQ_MIN_TOKEN_SIZE: usize = 1                   // crypto_suite
    + ADDRESS_HASH_LENGTH                             // issuer
    + ADDRESS_HASH_LENGTH                             // audience
    + 1                                               // capability
    + 2                                               // resource_len
    + 8                                               // not_before
    + 8                                               // expires_at
    + 16                                              // nonce
    + 1                                               // has_proof
    + PQ_SIGNATURE_LENGTH; // signature

impl PqUcanToken {
    /// Return the signable portion of the token (everything except the signature).
    ///
    /// This is the crypto_suite prefix + all fields before the signature.
    /// Safe to call even when `self.signature` is empty (e.g. before signing).
    pub fn signable_bytes(&self) -> Vec<u8> {
        debug_assert!(
            self.resource.len() <= u16::MAX as usize,
            "resource exceeds u16 length limit"
        );
        let resource_len = self.resource.len() as u16;
        let proof_size = if self.proof.is_some() { 32 } else { 0 };
        let signable_size =
            PQ_MIN_TOKEN_SIZE - PQ_SIGNATURE_LENGTH + self.resource.len() + proof_size;

        let mut buf = Vec::with_capacity(signable_size);
        buf.push(CryptoSuite::MlDsa65 as u8); // 0x01
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.audience);
        buf.push(self.capability as u8);
        buf.extend_from_slice(&resource_len.to_be_bytes());
        buf.extend_from_slice(&self.resource);
        buf.extend_from_slice(&self.not_before.to_be_bytes());
        buf.extend_from_slice(&self.expires_at.to_be_bytes());
        buf.extend_from_slice(&self.nonce);
        if let Some(ref proof) = self.proof {
            buf.push(1);
            buf.extend_from_slice(proof);
        } else {
            buf.push(0);
        }
        buf
    }

    /// Serialize the token to its binary wire format.
    ///
    /// The wire format starts with `crypto_suite = 0x01` to identify this as
    /// a post-quantum token, followed by the same field layout as the classical
    /// `UcanToken`, but with a 3,309-byte ML-DSA-65 signature instead of 64-byte
    /// Ed25519.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = self.signable_bytes();
        buf.extend_from_slice(&self.signature);
        buf
    }

    /// Deserialize a token from its binary wire format.
    ///
    /// Expects the first byte to be `0x01` (ML-DSA-65 crypto suite).
    pub fn from_bytes(data: &[u8]) -> Result<Self, UcanError> {
        if data.len() < PQ_MIN_TOKEN_SIZE {
            return Err(UcanError::InvalidEncoding);
        }

        let mut pos = 0;

        // Verify crypto suite byte
        if data[pos] != CryptoSuite::MlDsa65 as u8 {
            return Err(UcanError::InvalidEncoding);
        }
        pos += 1;

        let mut issuer = [0u8; ADDRESS_HASH_LENGTH];
        issuer.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LENGTH]);
        pos += ADDRESS_HASH_LENGTH;

        let mut audience = [0u8; ADDRESS_HASH_LENGTH];
        audience.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LENGTH]);
        pos += ADDRESS_HASH_LENGTH;

        let capability = CapabilityType::try_from(data[pos])?;
        pos += 1;

        let resource_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;

        if resource_len > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }

        if data.len() < PQ_MIN_TOKEN_SIZE + resource_len {
            return Err(UcanError::InvalidEncoding);
        }

        let resource = data[pos..pos + resource_len].to_vec();
        pos += resource_len;

        let not_before = u64::from_be_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| UcanError::InvalidEncoding)?,
        );
        pos += 8;

        let expires_at = u64::from_be_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| UcanError::InvalidEncoding)?,
        );
        pos += 8;

        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&data[pos..pos + 16]);
        pos += 16;

        let has_proof = data[pos];
        pos += 1;

        let proof = match has_proof {
            0 => None,
            1 => {
                if data.len() < pos + 32 + PQ_SIGNATURE_LENGTH {
                    return Err(UcanError::InvalidEncoding);
                }
                let mut proof_hash = [0u8; 32];
                proof_hash.copy_from_slice(&data[pos..pos + 32]);
                pos += 32;
                Some(proof_hash)
            }
            _ => return Err(UcanError::InvalidEncoding),
        };

        if data.len() != pos + PQ_SIGNATURE_LENGTH {
            return Err(UcanError::InvalidEncoding);
        }

        let signature = data[pos..pos + PQ_SIGNATURE_LENGTH].to_vec();

        Ok(Self {
            issuer,
            audience,
            capability,
            resource,
            not_before,
            expires_at,
            nonce,
            proof,
            signature,
        })
    }

    /// Compute the BLAKE3 content hash of this token's wire representation.
    pub fn content_hash(&self) -> [u8; 32] {
        hash::blake3_hash(&self.to_bytes())
    }

    /// Verify the ML-DSA-65 signature on this token.
    ///
    /// The caller provides the issuer's PQ public identity, whose verifying key
    /// is used to check the signature over the signable bytes.
    pub fn verify(
        &self,
        verifying_key: &harmony_crypto::ml_dsa::MlDsaPublicKey,
    ) -> Result<(), UcanError> {
        let signable = self.signable_bytes();
        let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(&self.signature)
            .map_err(|_| UcanError::SignatureInvalid)?;
        harmony_crypto::ml_dsa::verify(verifying_key, &signable, &sig)
            .map_err(|_| UcanError::SignatureInvalid)
    }
}

/// A signed revocation record — invalidates a specific token and all its downstream delegations.
#[derive(Debug, Clone)]
pub struct Revocation {
    /// Address hash of the revoking identity (must be the token's issuer).
    pub issuer: [u8; ADDRESS_HASH_LENGTH],
    /// BLAKE3 hash of the token being revoked.
    pub token_hash: [u8; 32],
    /// Timestamp of revocation.
    pub revoked_at: u64,
    /// Ed25519 signature over issuer + token_hash + revoked_at.
    pub signature: [u8; SIGNATURE_LENGTH],
}

/// Exact wire size of a serialized revocation record.
const REVOCATION_SIZE: usize = ADDRESS_HASH_LENGTH + 32 + 8 + SIGNATURE_LENGTH;

impl Revocation {
    /// Serialize to binary: `[16B issuer][32B token_hash][8B revoked_at][64B signature]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(REVOCATION_SIZE);
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.token_hash);
        buf.extend_from_slice(&self.revoked_at.to_be_bytes());
        buf.extend_from_slice(&self.signature);
        buf
    }

    /// Deserialize from binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, UcanError> {
        if bytes.len() != REVOCATION_SIZE {
            return Err(UcanError::InvalidEncoding);
        }

        let mut pos = 0;

        let mut issuer = [0u8; ADDRESS_HASH_LENGTH];
        issuer.copy_from_slice(&bytes[pos..pos + ADDRESS_HASH_LENGTH]);
        pos += ADDRESS_HASH_LENGTH;

        let mut token_hash = [0u8; 32];
        token_hash.copy_from_slice(&bytes[pos..pos + 32]);
        pos += 32;

        let revoked_at = u64::from_be_bytes(
            bytes[pos..pos + 8]
                .try_into()
                .map_err(|_| UcanError::InvalidEncoding)?,
        );
        pos += 8;

        let mut signature = [0u8; SIGNATURE_LENGTH];
        signature.copy_from_slice(&bytes[pos..pos + SIGNATURE_LENGTH]);

        Ok(Self {
            issuer,
            token_hash,
            revoked_at,
            signature,
        })
    }

    /// Serialize the signable portion (everything except the signature).
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(ADDRESS_HASH_LENGTH + 32 + 8);
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.token_hash);
        buf.extend_from_slice(&self.revoked_at.to_be_bytes());
        buf
    }
}

/// Verify a UCAN token and its delegation chain.
///
/// Fully sans-I/O: the caller provides current time, proof resolution,
/// identity lookup, and revocation state.
///
/// # Arguments
/// - `token` — the token to verify
/// - `now` — current Unix timestamp in seconds
/// - `proofs` — resolves parent tokens by BLAKE3 hash
/// - `identities` — resolves issuer identities by address hash
/// - `revocations` — checks if a token hash has been revoked
/// - `max_depth` — maximum delegation chain depth (0 = root only)
pub fn verify_token(
    token: &UcanToken,
    now: u64,
    proofs: &impl ProofResolver,
    identities: &impl IdentityResolver,
    revocations: &impl RevocationSet,
    max_depth: usize,
) -> Result<(), UcanError> {
    verify_token_recursive(
        token,
        now,
        proofs,
        identities,
        revocations,
        max_depth,
        0,
        None,
    )
}

/// Verify a revocation record's signature and issuer match.
///
/// Checks that:
/// 1. `revocation.token_hash` matches `token.content_hash()` (targets this token)
/// 2. `revocation.issuer` matches `token.issuer` (only the issuer can revoke)
/// 3. The Ed25519 signature over the signable bytes is valid
pub fn verify_revocation(
    revocation: &Revocation,
    token: &UcanToken,
    identities: &impl IdentityResolver,
) -> Result<(), UcanError> {
    if revocation.token_hash != token.content_hash() {
        return Err(UcanError::TokenHashMismatch);
    }

    if revocation.issuer != token.issuer {
        return Err(UcanError::NotIssuer);
    }

    let identity = identities
        .resolve(&revocation.issuer)
        .ok_or(UcanError::IssuerNotFound)?;
    let signable = revocation.signable_bytes();
    identity
        .verify(&signable, &revocation.signature)
        .map_err(|_| UcanError::SignatureInvalid)?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn verify_token_recursive(
    token: &UcanToken,
    now: u64,
    proofs: &impl ProofResolver,
    identities: &impl IdentityResolver,
    revocations: &impl RevocationSet,
    max_depth: usize,
    current_depth: usize,
    // Hash the caller expects this token to have (integrity check from parent
    // frame). `None` for the outermost call. This avoids computing
    // `content_hash()` twice per interior chain link — once as "current token"
    // for revocation, and again as "resolved parent" for integrity.
    expected_hash: Option<&[u8; 32]>,
) -> Result<(), UcanError> {
    // 1. Check time bounds
    if token.not_before > now {
        return Err(UcanError::NotYetValid);
    }
    if token.expires_at != 0 && now > token.expires_at {
        return Err(UcanError::Expired);
    }

    // 2. Compute content hash once (used for revocation + integrity).
    let token_hash = token.content_hash();

    // If the caller told us what hash to expect, verify integrity.
    // A buggy or adversarial ProofResolver could return a different token.
    if let Some(expected) = expected_hash {
        if token_hash != *expected {
            return Err(UcanError::ChainBroken);
        }
    }

    // 3. Check revocation (cheap HashSet lookup; must precede signature
    //    verification so that a tombstoned/rotated issuer identity doesn't
    //    mask a revocation with IssuerNotFound)
    if revocations.is_revoked(&token_hash) {
        return Err(UcanError::Revoked);
    }

    // 4. Verify signature
    let issuer_identity = identities
        .resolve(&token.issuer)
        .ok_or(UcanError::IssuerNotFound)?;
    let signable = token.signable_bytes();
    issuer_identity
        .verify(&signable, &token.signature)
        .map_err(|_| UcanError::SignatureInvalid)?;

    // 5. If delegated, verify the chain
    if let Some(parent_hash) = &token.proof {
        if current_depth >= max_depth {
            return Err(UcanError::ChainTooDeep {
                depth: current_depth,
                limit: max_depth,
            });
        }

        let parent = proofs
            .resolve(parent_hash)
            .ok_or(UcanError::ProofNotFound)?;

        // Chain continuity: parent.audience must equal this token's issuer
        if parent.audience != token.issuer {
            return Err(UcanError::ChainBroken);
        }

        // Attenuation: capability type must match
        if parent.capability != token.capability {
            return Err(UcanError::CapabilityMismatch);
        }

        // Attenuation: capability type + time bounds only.
        // NOTE: resource bytes are intentionally not compared here; Ring 2 is
        // responsible for resource-level attenuation (see design decision table).
        if token.not_before < parent.not_before {
            return Err(UcanError::AttenuationViolation);
        }
        if parent.expires_at != 0 && (token.expires_at == 0 || token.expires_at > parent.expires_at)
        {
            return Err(UcanError::AttenuationViolation);
        }

        // Recursively verify parent — pass parent_hash so the next frame can
        // verify integrity without recomputing the hash.
        verify_token_recursive(
            &parent,
            now,
            proofs,
            identities,
            revocations,
            max_depth,
            current_depth + 1,
            Some(parent_hash),
        )?;
    }

    Ok(())
}

impl PrivateIdentity {
    /// Issue a root UCAN token — claims direct ownership of a resource.
    ///
    /// Root tokens have no `proof` field. The issuer is asserting they own
    /// the resource. Downstream verification trusts or denies this claim
    /// based on the issuer's identity.
    pub fn issue_root_token(
        &self,
        rng: &mut impl EntropySource,
        audience: &[u8; ADDRESS_HASH_LENGTH],
        capability: CapabilityType,
        resource: &[u8],
        not_before: u64,
        expires_at: u64,
    ) -> Result<UcanToken, UcanError> {
        if resource.len() > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }
        if expires_at != 0 && not_before > expires_at {
            return Err(UcanError::InvalidTimeWindow);
        }

        let mut nonce = [0u8; 16];
        rng.fill_bytes(&mut nonce);

        let mut token = UcanToken {
            issuer: self.identity.address_hash,
            audience: *audience,
            capability,
            resource: resource.to_vec(),
            not_before,
            expires_at,
            nonce,
            proof: None,
            signature: [0u8; SIGNATURE_LENGTH],
        };

        let signable = token.signable_bytes();
        token.signature = self.sign(&signable);
        Ok(token)
    }

    /// Delegate a capability to another identity.
    ///
    /// Creates a child token referencing the parent by BLAKE3 hash.
    /// Enforces attenuation: capability type must match, time bounds
    /// must be narrowed or equal. **Resource bytes are NOT checked** —
    /// resource-level attenuation (memory ranges, CID ranges, fuel budgets)
    /// is the responsibility of Ring 2 callers.
    #[allow(clippy::too_many_arguments)]
    pub fn delegate(
        &self,
        rng: &mut impl EntropySource,
        parent: &UcanToken,
        audience: &[u8; ADDRESS_HASH_LENGTH],
        capability: CapabilityType,
        resource: &[u8],
        not_before: u64,
        expires_at: u64,
    ) -> Result<UcanToken, UcanError> {
        if resource.len() > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }
        if expires_at != 0 && not_before > expires_at {
            return Err(UcanError::InvalidTimeWindow);
        }

        // Chain continuity: caller must be the parent's intended audience
        if parent.audience != self.identity.address_hash {
            return Err(UcanError::ChainBroken);
        }

        // Attenuation checks
        if capability != parent.capability {
            return Err(UcanError::CapabilityMismatch);
        }
        if not_before < parent.not_before {
            return Err(UcanError::AttenuationViolation);
        }
        if parent.expires_at != 0 && (expires_at == 0 || expires_at > parent.expires_at) {
            return Err(UcanError::AttenuationViolation);
        }

        let mut nonce = [0u8; 16];
        rng.fill_bytes(&mut nonce);

        let mut token = UcanToken {
            issuer: self.identity.address_hash,
            audience: *audience,
            capability,
            resource: resource.to_vec(),
            not_before,
            expires_at,
            nonce,
            proof: Some(parent.content_hash()),
            signature: [0u8; SIGNATURE_LENGTH],
        };

        let signable = token.signable_bytes();
        token.signature = self.sign(&signable);
        Ok(token)
    }

    /// Revoke a token that this identity issued.
    ///
    /// Creates a signed revocation record. Returns `NotIssuer` if this identity
    /// is not the token's issuer. Revoking a parent implicitly invalidates all
    /// downstream delegations (chain verification fails if any link is revoked).
    pub fn revoke(&self, token: &UcanToken, revoked_at: u64) -> Result<Revocation, UcanError> {
        if self.identity.address_hash != token.issuer {
            return Err(UcanError::NotIssuer);
        }

        let mut revocation = Revocation {
            issuer: self.identity.address_hash,
            token_hash: token.content_hash(),
            revoked_at,
            signature: [0u8; SIGNATURE_LENGTH],
        };

        let signable = revocation.signable_bytes();
        revocation.signature = self.sign(&signable);
        Ok(revocation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrivateIdentity;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

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

    // ── Task 3: Serialization tests ───────────────────────────────────

    /// Helper: build a root token (no proof) with deterministic fields.
    fn make_root_token() -> UcanToken {
        UcanToken {
            issuer: [0xAA; ADDRESS_HASH_LENGTH],
            audience: [0xBB; ADDRESS_HASH_LENGTH],
            capability: CapabilityType::Memory,
            resource: alloc::vec![0x01, 0x02, 0x03],
            not_before: 1_700_000_000,
            expires_at: 1_700_003_600,
            nonce: [0xCC; 16],
            proof: None,
            signature: [0xDD; SIGNATURE_LENGTH],
        }
    }

    /// Helper: build a delegated token (with proof) with deterministic fields.
    fn make_delegated_token() -> UcanToken {
        UcanToken {
            issuer: [0x11; ADDRESS_HASH_LENGTH],
            audience: [0x22; ADDRESS_HASH_LENGTH],
            capability: CapabilityType::Content,
            resource: alloc::vec![0xFF; 10],
            not_before: 1_700_000_000,
            expires_at: 1_700_086_400,
            nonce: [0x33; 16],
            proof: Some([0x44; 32]),
            signature: [0x55; SIGNATURE_LENGTH],
        }
    }

    #[test]
    fn token_serialize_deserialize_root() {
        let token = make_root_token();
        let bytes = token.to_bytes();
        let restored = UcanToken::from_bytes(&bytes).unwrap();

        assert_eq!(restored.issuer, token.issuer);
        assert_eq!(restored.audience, token.audience);
        assert_eq!(restored.capability, token.capability);
        assert_eq!(restored.resource, token.resource);
        assert_eq!(restored.not_before, token.not_before);
        assert_eq!(restored.expires_at, token.expires_at);
        assert_eq!(restored.nonce, token.nonce);
        assert!(restored.proof.is_none());
        assert_eq!(restored.signature, token.signature);
    }

    #[test]
    fn token_serialize_deserialize_delegated() {
        let token = make_delegated_token();
        let bytes = token.to_bytes();
        let restored = UcanToken::from_bytes(&bytes).unwrap();

        assert_eq!(restored.issuer, token.issuer);
        assert_eq!(restored.audience, token.audience);
        assert_eq!(restored.capability, token.capability);
        assert_eq!(restored.resource, token.resource);
        assert_eq!(restored.not_before, token.not_before);
        assert_eq!(restored.expires_at, token.expires_at);
        assert_eq!(restored.nonce, token.nonce);
        assert_eq!(restored.proof, token.proof);
        assert_eq!(restored.signature, token.signature);
    }

    #[test]
    fn token_content_hash_deterministic() {
        let token = make_root_token();
        let h1 = token.content_hash();
        let h2 = token.content_hash();
        assert_eq!(h1, h2);

        // A clone should produce the same hash.
        let cloned = token.clone();
        assert_eq!(cloned.content_hash(), h1);
    }

    #[test]
    fn token_resource_too_large_rejected() {
        let mut token = make_root_token();
        token.resource = alloc::vec![0u8; MAX_RESOURCE_SIZE + 1];
        let bytes = token.to_bytes();
        let result = UcanToken::from_bytes(&bytes);
        assert!(matches!(result, Err(UcanError::ResourceTooLarge)));
    }

    #[test]
    fn token_truncated_bytes_rejected() {
        // Anything shorter than the minimum size should fail.
        let result = UcanToken::from_bytes(&[0u8; 10]);
        assert!(matches!(result, Err(UcanError::InvalidEncoding)));

        // One byte short of minimum.
        let result = UcanToken::from_bytes(&[0u8; MIN_TOKEN_SIZE - 1]);
        assert!(matches!(result, Err(UcanError::InvalidEncoding)));
    }

    // ── Task 4: Resolver and store tests ──────────────────────────────

    #[test]
    fn memory_proof_store_roundtrip() {
        let token = make_root_token();
        let token_hash = token.content_hash();

        let mut store = MemoryProofStore::new();
        store.insert(token.clone());

        let resolved = store.resolve(&token_hash).unwrap();
        assert_eq!(resolved.issuer, token.issuer);
        assert_eq!(resolved.audience, token.audience);
        assert_eq!(resolved.capability, token.capability);
        assert_eq!(resolved.resource, token.resource);
    }

    #[test]
    fn memory_proof_store_missing_returns_none() {
        let store = MemoryProofStore::new();
        let missing_hash = [0xFF; 32];
        assert!(store.resolve(&missing_hash).is_none());
    }

    #[test]
    fn memory_identity_store_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let private_id = PrivateIdentity::generate(&mut rng);
        let identity = private_id.public_identity().clone();
        let address = identity.address_hash;

        let mut store = MemoryIdentityStore::new();
        store.insert(identity.clone());

        let resolved = store.resolve(&address).unwrap();
        assert_eq!(resolved.address_hash, address);
        assert_eq!(
            resolved.verifying_key.as_bytes(),
            identity.verifying_key.as_bytes()
        );
    }

    #[test]
    fn memory_revocation_set_check() {
        let mut revocations = MemoryRevocationSet::new();
        let hash_a = [0xAA; 32];
        let hash_b = [0xBB; 32];

        assert!(!revocations.is_revoked(&hash_a));
        assert!(!revocations.is_revoked(&hash_b));

        revocations.insert(hash_a);

        assert!(revocations.is_revoked(&hash_a));
        assert!(!revocations.is_revoked(&hash_b));
    }

    // ── Task 5: Root token creation tests ─────────────────────────────

    #[test]
    fn issue_root_token_signs_correctly() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let audience_hash = [0xAA; 16];

        let token = issuer
            .issue_root_token(
                &mut rng,
                &audience_hash,
                CapabilityType::Content,
                &[0xBB; 8],
                1000,
                2000,
            )
            .unwrap();

        assert_eq!(token.issuer, issuer.identity.address_hash);
        assert_eq!(token.audience, audience_hash);
        assert_eq!(token.capability, CapabilityType::Content);
        assert_eq!(token.resource, alloc::vec![0xBB; 8]);
        assert_eq!(token.not_before, 1000);
        assert_eq!(token.expires_at, 2000);
        assert!(token.proof.is_none());

        // Verify the signature is valid
        let signable = token.signable_bytes();
        issuer
            .public_identity()
            .verify(&signable, &token.signature)
            .unwrap();
    }

    #[test]
    fn issue_root_token_resource_too_large() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let result = issuer.issue_root_token(
            &mut rng,
            &[0u8; 16],
            CapabilityType::Memory,
            &[0u8; 257],
            0,
            0,
        );
        assert!(matches!(result, Err(UcanError::ResourceTooLarge)));
    }

    #[test]
    fn issue_root_token_nonce_is_random() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let t1 = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();
        let t2 = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();
        assert_ne!(t1.nonce, t2.nonce);
    }

    // ── Task 6: Verification tests ────────────────────────────────────

    /// Helper: set up a root token with stores for verification.
    fn setup_root_token() -> (
        UcanToken,
        MemoryProofStore,
        MemoryIdentityStore,
        MemoryRevocationSet,
    ) {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let audience_hash = [0xAA; 16];

        let token = issuer
            .issue_root_token(
                &mut rng,
                &audience_hash,
                CapabilityType::Content,
                &[1, 2, 3],
                1000,
                2000,
            )
            .unwrap();

        let proof_store = MemoryProofStore::new();
        let mut id_store = MemoryIdentityStore::new();
        id_store.insert(issuer.public_identity().clone());
        let revocations = MemoryRevocationSet::new();

        (token, proof_store, id_store, revocations)
    }

    #[test]
    fn verify_valid_root_token() {
        let (token, proofs, ids, revocations) = setup_root_token();
        let result = verify_token(&token, 1500, &proofs, &ids, &revocations, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn verify_expired_token() {
        let (token, proofs, ids, revocations) = setup_root_token();
        let result = verify_token(&token, 3000, &proofs, &ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::Expired)));
    }

    #[test]
    fn verify_not_yet_valid_token() {
        let (token, proofs, ids, revocations) = setup_root_token();
        let result = verify_token(&token, 500, &proofs, &ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::NotYetValid)));
    }

    #[test]
    fn verify_no_expiry_always_valid() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        let mut ids = MemoryIdentityStore::new();
        ids.insert(issuer.public_identity().clone());
        let result = verify_token(
            &token,
            u64::MAX,
            &MemoryProofStore::new(),
            &ids,
            &MemoryRevocationSet::new(),
            5,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn verify_revoked_token() {
        let (token, proofs, ids, mut revocations) = setup_root_token();
        revocations.insert(token.content_hash());
        let result = verify_token(&token, 1500, &proofs, &ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::Revoked)));
    }

    #[test]
    fn verify_unknown_issuer() {
        let (token, proofs, _, revocations) = setup_root_token();
        let empty_ids = MemoryIdentityStore::new();
        let result = verify_token(&token, 1500, &proofs, &empty_ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::IssuerNotFound)));
    }

    #[test]
    fn verify_tampered_signature() {
        let (mut token, proofs, ids, revocations) = setup_root_token();
        token.signature[0] ^= 0xFF;
        let result = verify_token(&token, 1500, &proofs, &ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::SignatureInvalid)));
    }

    // ── Task 7: Delegation tests ──────────────────────────────────────

    #[test]
    fn delegate_creates_valid_chain() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);
        let end_user = [0xCC; 16];

        let root_token = root_id
            .issue_root_token(
                &mut rng,
                &delegate_id.identity.address_hash,
                CapabilityType::Content,
                &[1, 2, 3],
                1000,
                5000,
            )
            .unwrap();

        let child_token = delegate_id
            .delegate(
                &mut rng,
                &root_token,
                &end_user,
                CapabilityType::Content,
                &[1, 2, 3],
                2000,
                4000,
            )
            .unwrap();

        assert_eq!(child_token.issuer, delegate_id.identity.address_hash);
        assert_eq!(child_token.audience, end_user);
        assert_eq!(child_token.proof, Some(root_token.content_hash()));

        // Verify the full chain
        let mut proofs = MemoryProofStore::new();
        proofs.insert(root_token);
        let mut ids = MemoryIdentityStore::new();
        ids.insert(root_id.public_identity().clone());
        ids.insert(delegate_id.public_identity().clone());
        let revocations = MemoryRevocationSet::new();

        let result = verify_token(&child_token, 3000, &proofs, &ids, &revocations, 5);
        assert!(result.is_ok());
    }

    #[test]
    fn delegate_wrong_audience_rejected() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);
        let wrong_id = PrivateIdentity::generate(&mut rng);

        // Root token is addressed to delegate_id, not wrong_id
        let root_token = root_id
            .issue_root_token(
                &mut rng,
                &delegate_id.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        // wrong_id tries to delegate — should fail at creation time
        let result = wrong_id.delegate(
            &mut rng,
            &root_token,
            &[0u8; 16],
            CapabilityType::Content,
            &[],
            0,
            0,
        );
        assert!(matches!(result, Err(UcanError::ChainBroken)));
    }

    #[test]
    fn verify_chain_broken_audience_mismatch() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);
        let wrong_id = PrivateIdentity::generate(&mut rng);

        // Root token addressed to delegate_id
        let root_token = root_id
            .issue_root_token(
                &mut rng,
                &delegate_id.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        // Manually construct a child that claims root_token as proof but
        // was signed by wrong_id (who is NOT the root's audience).
        let mut child = UcanToken {
            issuer: wrong_id.identity.address_hash,
            audience: [0xEE; ADDRESS_HASH_LENGTH],
            capability: CapabilityType::Content,
            resource: alloc::vec![],
            not_before: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            proof: Some(root_token.content_hash()),
            signature: [0u8; SIGNATURE_LENGTH],
        };
        let signable = child.signable_bytes();
        child.signature = wrong_id.sign(&signable);

        let mut proofs = MemoryProofStore::new();
        proofs.insert(root_token);
        let mut ids = MemoryIdentityStore::new();
        ids.insert(root_id.public_identity().clone());
        ids.insert(wrong_id.public_identity().clone());

        let result = verify_token(&child, 0, &proofs, &ids, &MemoryRevocationSet::new(), 5);
        assert!(matches!(result, Err(UcanError::ChainBroken)));
    }

    #[test]
    fn delegate_capability_mismatch_rejected() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);

        let root_token = root_id
            .issue_root_token(
                &mut rng,
                &delegate_id.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let result = delegate_id.delegate(
            &mut rng,
            &root_token,
            &[0u8; 16],
            CapabilityType::Memory, // Wrong type!
            &[],
            0,
            0,
        );
        assert!(matches!(result, Err(UcanError::CapabilityMismatch)));
    }

    #[test]
    fn delegate_time_expansion_rejected() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);

        let root_token = root_id
            .issue_root_token(
                &mut rng,
                &delegate_id.identity.address_hash,
                CapabilityType::Content,
                &[],
                1000,
                5000,
            )
            .unwrap();

        // Try to expand expiry beyond parent
        let result = delegate_id.delegate(
            &mut rng,
            &root_token,
            &[0u8; 16],
            CapabilityType::Content,
            &[],
            1000,
            6000, // Beyond parent's 5000
        );
        assert!(matches!(result, Err(UcanError::AttenuationViolation)));

        // Try to move not_before earlier than parent
        let result = delegate_id.delegate(
            &mut rng,
            &root_token,
            &[0u8; 16],
            CapabilityType::Content,
            &[],
            500, // Before parent's 1000
            5000,
        );
        assert!(matches!(result, Err(UcanError::AttenuationViolation)));
    }

    #[test]
    fn three_hop_delegation_chain() {
        let mut rng = test_rng();
        let root = PrivateIdentity::generate(&mut rng);
        let mid = PrivateIdentity::generate(&mut rng);
        let leaf = PrivateIdentity::generate(&mut rng);
        let end_user = [0xFF; 16];

        let t1 = root
            .issue_root_token(
                &mut rng,
                &mid.identity.address_hash,
                CapabilityType::Compute,
                &[10],
                0,
                10000,
            )
            .unwrap();

        let t2 = mid
            .delegate(
                &mut rng,
                &t1,
                &leaf.identity.address_hash,
                CapabilityType::Compute,
                &[10],
                1000,
                9000,
            )
            .unwrap();

        let t3 = leaf
            .delegate(
                &mut rng,
                &t2,
                &end_user,
                CapabilityType::Compute,
                &[10],
                2000,
                8000,
            )
            .unwrap();

        let mut proofs = MemoryProofStore::new();
        proofs.insert(t1);
        proofs.insert(t2);
        let mut ids = MemoryIdentityStore::new();
        ids.insert(root.public_identity().clone());
        ids.insert(mid.public_identity().clone());
        ids.insert(leaf.public_identity().clone());
        let revocations = MemoryRevocationSet::new();

        assert!(verify_token(&t3, 5000, &proofs, &ids, &revocations, 5).is_ok());
    }

    #[test]
    fn chain_exceeds_max_depth() {
        let mut rng = test_rng();
        let root = PrivateIdentity::generate(&mut rng);
        let mid = PrivateIdentity::generate(&mut rng);
        let leaf = PrivateIdentity::generate(&mut rng);
        let end_user = [0xFF; 16];

        let t1 = root
            .issue_root_token(
                &mut rng,
                &mid.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let t2 = mid
            .delegate(
                &mut rng,
                &t1,
                &leaf.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let t3 = leaf
            .delegate(&mut rng, &t2, &end_user, CapabilityType::Content, &[], 0, 0)
            .unwrap();

        let mut proofs = MemoryProofStore::new();
        proofs.insert(t1);
        proofs.insert(t2);
        let mut ids = MemoryIdentityStore::new();
        ids.insert(root.public_identity().clone());
        ids.insert(mid.public_identity().clone());
        ids.insert(leaf.public_identity().clone());

        // max_depth=1 means only 1 delegation hop allowed (root + 1 child)
        let result = verify_token(&t3, 0, &proofs, &ids, &MemoryRevocationSet::new(), 1);
        assert!(matches!(
            result,
            Err(UcanError::ChainTooDeep { depth: 1, limit: 1 })
        ));
    }

    // ── Task 8: Revocation tests ──────────────────────────────────────

    #[test]
    fn revocation_serialize_deserialize() {
        let revocation = Revocation {
            issuer: [1u8; 16],
            token_hash: [2u8; 32],
            revoked_at: 12345,
            signature: [3u8; 64],
        };
        let bytes = revocation.to_bytes();
        let restored = Revocation::from_bytes(&bytes).unwrap();
        assert_eq!(restored.issuer, [1u8; 16]);
        assert_eq!(restored.token_hash, [2u8; 32]);
        assert_eq!(restored.revoked_at, 12345);
        assert_eq!(restored.signature, [3u8; 64]);
    }

    #[test]
    fn revocation_truncated_bytes_rejected() {
        assert!(matches!(
            Revocation::from_bytes(&[0u8; 10]),
            Err(UcanError::InvalidEncoding)
        ));
    }

    #[test]
    fn create_and_verify_revocation() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);

        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        let revocation = issuer.revoke(&token, 5000).unwrap();

        // Verify using verify_revocation()
        let mut ids = MemoryIdentityStore::new();
        ids.insert(issuer.public_identity().clone());
        verify_revocation(&revocation, &token, &ids).unwrap();

        assert_eq!(revocation.token_hash, token.content_hash());
        assert_eq!(revocation.revoked_at, 5000);
    }

    #[test]
    fn revoke_by_non_issuer_rejected() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let other = PrivateIdentity::generate(&mut rng);

        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        let result = other.revoke(&token, 5000);
        assert!(matches!(result, Err(UcanError::NotIssuer)));
    }

    #[test]
    fn verify_revocation_wrong_issuer_rejected() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let attacker = PrivateIdentity::generate(&mut rng);

        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        // Attacker creates a token they issued, then forges a revocation
        // with mismatched issuer field (manually constructed).
        let forged = Revocation {
            issuer: attacker.identity.address_hash,
            token_hash: token.content_hash(),
            revoked_at: 5000,
            signature: [0u8; SIGNATURE_LENGTH],
        };

        let mut ids = MemoryIdentityStore::new();
        ids.insert(attacker.public_identity().clone());
        ids.insert(issuer.public_identity().clone());

        let result = verify_revocation(&forged, &token, &ids);
        assert!(matches!(result, Err(UcanError::NotIssuer)));
    }

    #[test]
    fn verify_revocation_bad_signature_rejected() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);

        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        // Create a valid revocation then corrupt the signature
        let mut revocation = issuer.revoke(&token, 5000).unwrap();
        revocation.signature[0] ^= 0xFF;

        let mut ids = MemoryIdentityStore::new();
        ids.insert(issuer.public_identity().clone());

        let result = verify_revocation(&revocation, &token, &ids);
        assert!(matches!(result, Err(UcanError::SignatureInvalid)));
    }

    #[test]
    fn verify_revocation_wrong_token_rejected() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);

        let token_a = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();
        let token_b = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Memory, &[], 0, 0)
            .unwrap();

        // Revoke token A, then try to verify against token B
        let revocation = issuer.revoke(&token_a, 5000).unwrap();

        let mut ids = MemoryIdentityStore::new();
        ids.insert(issuer.public_identity().clone());

        // Should pass against token A
        verify_revocation(&revocation, &token_a, &ids).unwrap();
        // Should fail against token B (wrong token hash)
        assert!(matches!(
            verify_revocation(&revocation, &token_b, &ids),
            Err(UcanError::TokenHashMismatch)
        ));
    }

    #[test]
    fn revoked_parent_invalidates_child() {
        let mut rng = test_rng();
        let root = PrivateIdentity::generate(&mut rng);
        let delegate = PrivateIdentity::generate(&mut rng);

        let root_token = root
            .issue_root_token(
                &mut rng,
                &delegate.identity.address_hash,
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let child_token = delegate
            .delegate(
                &mut rng,
                &root_token,
                &[0xDD; 16],
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let mut proofs = MemoryProofStore::new();
        proofs.insert(root_token.clone());
        let mut ids = MemoryIdentityStore::new();
        ids.insert(root.public_identity().clone());
        ids.insert(delegate.public_identity().clone());
        let mut revocations = MemoryRevocationSet::new();

        // Child verifies before revocation
        assert!(verify_token(&child_token, 0, &proofs, &ids, &revocations, 5).is_ok());

        // Revoke the root token
        revocations.insert(root_token.content_hash());

        // Child now fails because parent is revoked
        let result = verify_token(&child_token, 0, &proofs, &ids, &revocations, 5);
        assert!(matches!(result, Err(UcanError::Revoked)));
    }

    #[test]
    fn issue_rejects_not_before_after_expires() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let result =
            issuer.issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Memory, &[], 200, 100);
        assert!(matches!(result, Err(UcanError::InvalidTimeWindow)));
    }

    #[test]
    fn delegate_rejects_not_before_after_expires() {
        let mut rng = test_rng();
        let root = PrivateIdentity::generate(&mut rng);
        let delegate = PrivateIdentity::generate(&mut rng);
        let parent = root
            .issue_root_token(
                &mut rng,
                &delegate.identity.address_hash,
                CapabilityType::Memory,
                &[],
                0,
                1000,
            )
            .unwrap();

        // not_before (500) > expires_at (400)
        let result = delegate.delegate(
            &mut rng,
            &parent,
            &[0u8; 16],
            CapabilityType::Memory,
            &[],
            500,
            400,
        );
        assert!(matches!(result, Err(UcanError::InvalidTimeWindow)));
    }

    #[test]
    fn issue_allows_zero_expires_with_any_not_before() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        // expires_at = 0 means no expiry, should be fine regardless of not_before
        let result =
            issuer.issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Memory, &[], 500, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn revoked_token_returns_revoked_even_without_issuer_in_store() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Memory, &[], 0, 0)
            .unwrap();

        // Revoke the token but do NOT add the issuer to the identity store
        let mut revocations = MemoryRevocationSet::new();
        revocations.insert(token.content_hash());
        let ids = MemoryIdentityStore::new(); // empty — issuer not present
        let proofs = MemoryProofStore::new();

        // Should return Revoked, not IssuerNotFound
        let result = verify_token(&token, 100, &proofs, &ids, &revocations, 10);
        assert!(matches!(result, Err(UcanError::Revoked)));
    }

    #[test]
    fn from_bytes_rejects_trailing_garbage() {
        let mut rng = test_rng();
        let issuer = PrivateIdentity::generate(&mut rng);
        let token = issuer
            .issue_root_token(&mut rng, &[0u8; 16], CapabilityType::Memory, &[], 0, 0)
            .unwrap();

        let mut wire = token.to_bytes();
        wire.push(0xFF); // trailing garbage byte
        assert!(matches!(
            UcanToken::from_bytes(&wire),
            Err(UcanError::InvalidEncoding)
        ));
    }

    #[test]
    fn verify_rejects_proof_with_wrong_content_hash() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let child_id = PrivateIdentity::generate(&mut rng);

        let parent = root_id
            .issue_root_token(
                &mut rng,
                &child_id.identity.address_hash,
                CapabilityType::Memory,
                &[],
                0,
                0,
            )
            .unwrap();

        let child = child_id
            .delegate(
                &mut rng,
                &parent,
                &[0u8; 16],
                CapabilityType::Memory,
                &[],
                0,
                0,
            )
            .unwrap();

        // Build a proof store that returns the WRONG token for the parent hash.
        // The child's proof field points to parent's hash, but we store a
        // different token (the child itself) under that key.
        let mut proofs = MemoryProofStore::new();
        // Insert wrong token under parent's hash key — simulates adversarial resolver
        proofs.insert_with_key(parent.content_hash(), child.clone());

        let mut ids = MemoryIdentityStore::new();
        ids.insert(root_id.identity.clone());
        ids.insert(child_id.identity.clone());
        let revocations = MemoryRevocationSet::new();

        let result = verify_token(&child, 100, &proofs, &ids, &revocations, 10);
        assert!(matches!(result, Err(UcanError::ChainBroken)));
    }
}

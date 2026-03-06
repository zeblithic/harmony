//! UCAN capability tokens — Harmony-native authorization primitives.
//!
//! Implements delegation chains with Ed25519 signatures, content-addressed
//! proof references, and configurable revocation. Follows UCAN authorization
//! principles with a compact binary format optimized for kernel verification.

use alloc::vec::Vec;

use harmony_crypto::hash;
use hashbrown::{HashMap, HashSet};

use crate::identity::{Identity, ADDRESS_HASH_LENGTH, SIGNATURE_LENGTH};
use crate::IdentityError;

/// Maximum size of the resource field in bytes.
pub const MAX_RESOURCE_SIZE: usize = 256;

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

        if data.len() < pos + SIGNATURE_LENGTH {
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
    fn signable_bytes(&self) -> Vec<u8> {
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
#[derive(Debug, Default)]
pub struct MemoryProofStore {
    tokens: HashMap<[u8; 32], UcanToken>,
}

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
}

impl ProofResolver for MemoryProofStore {
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken> {
        self.tokens.get(hash).cloned()
    }
}

/// In-memory identity store for testing.
///
/// Stores identities keyed by their address hash.
#[derive(Debug, Default)]
pub struct MemoryIdentityStore {
    identities: HashMap<[u8; ADDRESS_HASH_LENGTH], Identity>,
}

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

impl IdentityResolver for MemoryIdentityStore {
    fn resolve(&self, address_hash: &[u8; ADDRESS_HASH_LENGTH]) -> Option<Identity> {
        self.identities.get(address_hash).cloned()
    }
}

/// In-memory revocation set for testing.
///
/// Stores the BLAKE3 content hashes of revoked tokens.
#[derive(Debug, Default)]
pub struct MemoryRevocationSet {
    revoked: HashSet<[u8; 32]>,
}

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

impl RevocationSet for MemoryRevocationSet {
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool {
        self.revoked.contains(token_hash)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PrivateIdentity;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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
}

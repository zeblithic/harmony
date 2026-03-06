# UCAN Capability Tokens Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add UCAN capability tokens to harmony-identity — token structure, signing, verification, delegation chains, and revocation.

**Architecture:** A new `ucan` module in harmony-identity implementing compact binary tokens with content-addressed proof chains. Three resolver traits (`ProofResolver`, `IdentityResolver`, `RevocationSet`) provide the sans-I/O boundary. Token creation lives on `PrivateIdentity`.

**Tech Stack:** Rust, Ed25519 (ed25519-dalek), BLAKE3 (harmony-crypto), no_std + alloc, TDD

**Design doc:** `docs/plans/2026-03-06-ucan-tokens-design.md`

---

### Task 1: Module Scaffolding and Dependencies

**Files:**
- Modify: `crates/harmony-identity/Cargo.toml`
- Modify: `crates/harmony-identity/src/lib.rs`
- Create: `crates/harmony-identity/src/ucan.rs`

**Step 1: Add harmony-platform dependency to Cargo.toml**

In `crates/harmony-identity/Cargo.toml`, add `harmony-platform` under `[dependencies]` and propagate its `std` feature:

```toml
[dependencies]
harmony-crypto = { workspace = true }
harmony-platform = { workspace = true }
ed25519-dalek = { workspace = true }
x25519-dalek = { workspace = true }
rand_core = { workspace = true }
zeroize = { workspace = true }
thiserror = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = ["harmony-crypto/std", "harmony-platform/std", "rand_core/getrandom"]
```

Note: `hashbrown` is added for in-memory test impls. `harmony-platform` is added for `EntropySource`.

**Step 2: Add the `ucan` module to lib.rs**

After the existing module declarations in `crates/harmony-identity/src/lib.rs`, add:

```rust
pub mod ucan;
```

And add re-exports at the bottom:

```rust
pub use ucan::{
    CapabilityType, IdentityResolver, MemoryIdentityStore, MemoryProofStore,
    MemoryRevocationSet, ProofResolver, Revocation, RevocationSet, UcanError, UcanToken,
};
```

**Step 3: Create the ucan.rs file with module structure**

Create `crates/harmony-identity/src/ucan.rs` with imports and placeholder types:

```rust
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
```

**Step 4: Verify it compiles**

Run: `cargo check -p harmony-identity`
Expected: Compiles with no errors (empty module is valid).

**Step 5: Commit**

```bash
git add crates/harmony-identity/Cargo.toml crates/harmony-identity/src/lib.rs crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): scaffold ucan module in harmony-identity"
```

---

### Task 2: UcanError and CapabilityType

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for CapabilityType**

Add at the bottom of `ucan.rs`:

```rust
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
```

Run: `cargo test -p harmony-identity ucan::tests`
Expected: FAIL — types not defined yet.

**Step 2: Implement CapabilityType and UcanError**

Add above the `#[cfg(test)]` block in `ucan.rs`:

```rust
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

/// Errors from UCAN operations.
#[derive(Debug, thiserror::Error)]
pub enum UcanError {
    #[error("token has expired")]
    Expired,
    #[error("token is not yet valid")]
    NotYetValid,
    #[error("signature verification failed")]
    SignatureInvalid,
    #[error("delegation chain exceeds maximum depth of {0}")]
    ChainTooDeep(usize),
    #[error("proof token not found")]
    ProofNotFound,
    #[error("chain continuity broken: parent audience != child issuer")]
    ChainBroken,
    #[error("capability type mismatch in delegation")]
    CapabilityMismatch,
    #[error("time bounds not attenuated")]
    AttenuationViolation,
    #[error("token has been revoked")]
    Revoked,
    #[error("issuer identity not found")]
    IssuerNotFound,
    #[error("invalid token encoding")]
    InvalidEncoding,
    #[error("resource field exceeds maximum size")]
    ResourceTooLarge,
    #[error("identity error: {0}")]
    Identity(#[from] IdentityError),
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 3 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): add CapabilityType enum and UcanError"
```

---

### Task 3: UcanToken Struct and Binary Serialization

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for token serialization**

Add to the `tests` module:

```rust
    #[test]
    fn token_serialize_deserialize_root() {
        let token = UcanToken {
            issuer: [1u8; 16],
            audience: [2u8; 16],
            capability: CapabilityType::Content,
            resource: alloc::vec![0xAA; 32],
            not_before: 1000,
            expires_at: 2000,
            nonce: [3u8; 16],
            proof: None,
            signature: [4u8; 64],
        };
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
    fn token_serialize_deserialize_delegated() {
        let token = UcanToken {
            issuer: [1u8; 16],
            audience: [2u8; 16],
            capability: CapabilityType::Memory,
            resource: alloc::vec![0xBB; 8],
            not_before: 500,
            expires_at: 1500,
            nonce: [5u8; 16],
            proof: Some([6u8; 32]),
            signature: [7u8; 64],
        };
        let bytes = token.to_bytes();
        let restored = UcanToken::from_bytes(&bytes).unwrap();
        assert_eq!(restored.proof, Some([6u8; 32]));
    }

    #[test]
    fn token_content_hash_deterministic() {
        let token = UcanToken {
            issuer: [1u8; 16],
            audience: [2u8; 16],
            capability: CapabilityType::Compute,
            resource: alloc::vec![],
            not_before: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            proof: None,
            signature: [0u8; 64],
        };
        let h1 = token.content_hash();
        let h2 = token.content_hash();
        assert_eq!(h1, h2);
        assert_ne!(h1, [0u8; 32]); // Not trivially zero
    }

    #[test]
    fn token_resource_too_large_rejected() {
        let bytes_with_huge_resource = {
            let mut buf = Vec::new();
            buf.extend_from_slice(&[0u8; 16]); // issuer
            buf.extend_from_slice(&[0u8; 16]); // audience
            buf.push(0); // capability
            buf.extend_from_slice(&(257u16).to_be_bytes()); // resource_len > 256
            buf.extend_from_slice(&[0u8; 257]); // resource
            buf.extend_from_slice(&[0u8; 8]); // not_before
            buf.extend_from_slice(&[0u8; 8]); // expires_at
            buf.extend_from_slice(&[0u8; 16]); // nonce
            buf.push(0); // has_proof = false
            buf.extend_from_slice(&[0u8; 64]); // signature
            buf
        };
        assert!(matches!(
            UcanToken::from_bytes(&bytes_with_huge_resource),
            Err(UcanError::ResourceTooLarge)
        ));
    }

    #[test]
    fn token_truncated_bytes_rejected() {
        assert!(matches!(
            UcanToken::from_bytes(&[0u8; 10]),
            Err(UcanError::InvalidEncoding)
        ));
    }
```

Run: `cargo test -p harmony-identity ucan::tests`
Expected: FAIL — `UcanToken` not defined.

**Step 2: Implement UcanToken struct and serialization**

Add above the tests module in `ucan.rs`:

```rust
/// A UCAN capability token — compact binary format with content-addressed proofs.
#[derive(Debug, Clone)]
pub struct UcanToken {
    /// Issuer: address hash of the signing identity.
    pub issuer: [u8; ADDRESS_HASH_LENGTH],
    /// Audience: address hash of the recipient identity.
    pub audience: [u8; ADDRESS_HASH_LENGTH],
    /// Capability type.
    pub capability: CapabilityType,
    /// Resource scope (opaque bytes, interpreted by Ring 2, max 256 bytes).
    pub resource: Vec<u8>,
    /// Not-before: Unix timestamp in seconds.
    pub not_before: u64,
    /// Expires-at: Unix timestamp in seconds (0 = never).
    pub expires_at: u64,
    /// Nonce: 16 random bytes for replay protection.
    pub nonce: [u8; 16],
    /// Parent proof: BLAKE3 hash of the parent token. None for root tokens.
    pub proof: Option<[u8; 32]>,
    /// Ed25519 signature over all preceding fields.
    pub signature: [u8; SIGNATURE_LENGTH],
}

impl UcanToken {
    /// Serialize the token to compact binary format.
    ///
    /// Wire format (big-endian):
    /// `[16B issuer][16B audience][1B cap][2B resource_len][NB resource]`
    /// `[8B not_before][8B expires_at][16B nonce][1B has_proof][32B proof?][64B signature]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let proof_size = if self.proof.is_some() { 32 } else { 0 };
        let total = 16 + 16 + 1 + 2 + self.resource.len() + 8 + 8 + 16 + 1 + proof_size + 64;
        let mut buf = Vec::with_capacity(total);

        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.audience);
        buf.push(self.capability as u8);
        buf.extend_from_slice(&(self.resource.len() as u16).to_be_bytes());
        buf.extend_from_slice(&self.resource);
        buf.extend_from_slice(&self.not_before.to_be_bytes());
        buf.extend_from_slice(&self.expires_at.to_be_bytes());
        buf.extend_from_slice(&self.nonce);
        match &self.proof {
            Some(hash) => {
                buf.push(1);
                buf.extend_from_slice(hash);
            }
            None => buf.push(0),
        }
        buf.extend_from_slice(&self.signature);
        buf
    }

    /// Deserialize a token from compact binary format.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, UcanError> {
        // Minimum size: 16+16+1+2+0+8+8+16+1+0+64 = 132
        if bytes.len() < 132 {
            return Err(UcanError::InvalidEncoding);
        }

        let mut pos = 0;

        let mut issuer = [0u8; 16];
        issuer.copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;

        let mut audience = [0u8; 16];
        audience.copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;

        let capability = CapabilityType::try_from(bytes[pos])?;
        pos += 1;

        let resource_len =
            u16::from_be_bytes(bytes[pos..pos + 2].try_into().unwrap()) as usize;
        pos += 2;

        if resource_len > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }
        if pos + resource_len > bytes.len() {
            return Err(UcanError::InvalidEncoding);
        }
        let resource = bytes[pos..pos + resource_len].to_vec();
        pos += resource_len;

        if pos + 8 + 8 + 16 + 1 > bytes.len() {
            return Err(UcanError::InvalidEncoding);
        }

        let not_before = u64::from_be_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let expires_at = u64::from_be_bytes(bytes[pos..pos + 8].try_into().unwrap());
        pos += 8;

        let mut nonce = [0u8; 16];
        nonce.copy_from_slice(&bytes[pos..pos + 16]);
        pos += 16;

        let has_proof = bytes[pos];
        pos += 1;

        let proof = if has_proof == 1 {
            if pos + 32 > bytes.len() {
                return Err(UcanError::InvalidEncoding);
            }
            let mut hash = [0u8; 32];
            hash.copy_from_slice(&bytes[pos..pos + 32]);
            pos += 32;
            Some(hash)
        } else if has_proof == 0 {
            None
        } else {
            return Err(UcanError::InvalidEncoding);
        };

        if pos + 64 > bytes.len() {
            return Err(UcanError::InvalidEncoding);
        }
        let mut signature = [0u8; 64];
        signature.copy_from_slice(&bytes[pos..pos + 64]);

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

    /// Compute the BLAKE3 content hash of this token's serialized form.
    pub fn content_hash(&self) -> [u8; 32] {
        hash::blake3_hash(&self.to_bytes())
    }

    /// Serialize the signable portion (everything except the signature).
    fn signable_bytes(&self) -> Vec<u8> {
        let bytes = self.to_bytes();
        bytes[..bytes.len() - 64].to_vec()
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 8 tests pass (3 from Task 2 + 5 new).

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): UcanToken struct with binary serialization"
```

---

### Task 4: Resolver Traits and In-Memory Test Implementations

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for in-memory stores**

Add to the `tests` module:

```rust
    #[test]
    fn memory_proof_store_roundtrip() {
        let token = UcanToken {
            issuer: [1u8; 16],
            audience: [2u8; 16],
            capability: CapabilityType::Content,
            resource: alloc::vec![],
            not_before: 0,
            expires_at: 0,
            nonce: [0u8; 16],
            proof: None,
            signature: [0u8; 64],
        };
        let hash = token.content_hash();
        let mut store = MemoryProofStore::new();
        store.insert(token.clone());
        let resolved = store.resolve(&hash).unwrap();
        assert_eq!(resolved.issuer, [1u8; 16]);
    }

    #[test]
    fn memory_proof_store_missing_returns_none() {
        let store = MemoryProofStore::new();
        assert!(store.resolve(&[0u8; 32]).is_none());
    }

    #[test]
    fn memory_identity_store_roundtrip() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let id = PrivateIdentity::generate(&mut rng);
        let identity = id.public_identity().clone();
        let hash = identity.address_hash;

        let mut store = MemoryIdentityStore::new();
        store.insert(identity);
        let resolved = store.resolve(&hash).unwrap();
        assert_eq!(resolved.address_hash, hash);
    }

    #[test]
    fn memory_revocation_set_check() {
        let mut revocations = MemoryRevocationSet::new();
        let hash = [99u8; 32];
        assert!(!revocations.is_revoked(&hash));
        revocations.insert(hash);
        assert!(revocations.is_revoked(&hash));
    }
```

Run: `cargo test -p harmony-identity ucan::tests`
Expected: FAIL — traits and stores not defined.

**Step 2: Implement the three resolver traits and in-memory stores**

Add above the tests module in `ucan.rs`:

```rust
/// Resolves parent tokens by their BLAKE3 content hash.
///
/// Ring 0 defines this trait; Ring 2 implements it against the content store.
pub trait ProofResolver {
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken>;
}

/// Resolves identities by their address hash.
///
/// Needed for signature verification — the verifier must look up the
/// issuer's public key to check the Ed25519 signature.
pub trait IdentityResolver {
    fn resolve(&self, address_hash: &[u8; ADDRESS_HASH_LENGTH]) -> Option<Identity>;
}

/// Checks whether a token has been revoked.
pub trait RevocationSet {
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool;
}

/// In-memory proof store for testing.
pub struct MemoryProofStore {
    tokens: hashbrown::HashMap<[u8; 32], UcanToken>,
}

impl MemoryProofStore {
    pub fn new() -> Self {
        Self {
            tokens: hashbrown::HashMap::new(),
        }
    }

    /// Insert a token, keyed by its content hash.
    pub fn insert(&mut self, token: UcanToken) {
        let hash = token.content_hash();
        self.tokens.insert(hash, token);
    }
}

impl Default for MemoryProofStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ProofResolver for MemoryProofStore {
    fn resolve(&self, hash: &[u8; 32]) -> Option<UcanToken> {
        self.tokens.get(hash).cloned()
    }
}

/// In-memory identity store for testing.
pub struct MemoryIdentityStore {
    identities: hashbrown::HashMap<[u8; ADDRESS_HASH_LENGTH], Identity>,
}

impl MemoryIdentityStore {
    pub fn new() -> Self {
        Self {
            identities: hashbrown::HashMap::new(),
        }
    }

    /// Insert an identity, keyed by its address hash.
    pub fn insert(&mut self, identity: Identity) {
        self.identities.insert(identity.address_hash, identity);
    }
}

impl Default for MemoryIdentityStore {
    fn default() -> Self {
        Self::new()
    }
}

impl IdentityResolver for MemoryIdentityStore {
    fn resolve(&self, address_hash: &[u8; ADDRESS_HASH_LENGTH]) -> Option<Identity> {
        self.identities.get(address_hash).cloned()
    }
}

/// In-memory revocation set for testing.
pub struct MemoryRevocationSet {
    revoked: hashbrown::HashSet<[u8; 32]>,
}

impl MemoryRevocationSet {
    pub fn new() -> Self {
        Self {
            revoked: hashbrown::HashSet::new(),
        }
    }

    /// Mark a token hash as revoked.
    pub fn insert(&mut self, token_hash: [u8; 32]) {
        self.revoked.insert(token_hash);
    }
}

impl Default for MemoryRevocationSet {
    fn default() -> Self {
        Self::new()
    }
}

impl RevocationSet for MemoryRevocationSet {
    fn is_revoked(&self, token_hash: &[u8; 32]) -> bool {
        self.revoked.contains(token_hash)
    }
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 12 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): resolver traits and in-memory test implementations"
```

---

### Task 5: Root Token Creation

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for root token issuance**

Add to the `tests` module:

```rust
    use rand::SeedableRng;

    fn test_rng() -> rand::rngs::StdRng {
        rand::rngs::StdRng::seed_from_u64(42)
    }

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
```

Run: `cargo test -p harmony-identity ucan::tests::issue_root`
Expected: FAIL — `issue_root_token` not defined.

**Step 2: Implement `issue_root_token` on PrivateIdentity**

Add an `impl` block for PrivateIdentity in `ucan.rs` (above the tests module):

```rust
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
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 15 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): root token issuance on PrivateIdentity"
```

---

### Task 6: Token Verification

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for verification**

Add to the `tests` module:

```rust
    /// Helper: set up a root token with stores for verification.
    fn setup_root_token() -> (UcanToken, MemoryProofStore, MemoryIdentityStore, MemoryRevocationSet)
    {
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
```

Run: `cargo test -p harmony-identity ucan::tests::verify`
Expected: FAIL — `verify_token` not defined.

**Step 2: Implement `verify_token`**

Add above the tests module in `ucan.rs`:

```rust
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
    verify_token_recursive(token, now, proofs, identities, revocations, max_depth, 0)
}

fn verify_token_recursive(
    token: &UcanToken,
    now: u64,
    proofs: &impl ProofResolver,
    identities: &impl IdentityResolver,
    revocations: &impl RevocationSet,
    max_depth: usize,
    current_depth: usize,
) -> Result<(), UcanError> {
    // 1. Check time bounds
    if token.not_before > now {
        return Err(UcanError::NotYetValid);
    }
    if token.expires_at != 0 && now > token.expires_at {
        return Err(UcanError::Expired);
    }

    // 2. Verify signature
    let issuer_identity = identities
        .resolve(&token.issuer)
        .ok_or(UcanError::IssuerNotFound)?;
    let signable = token.signable_bytes();
    issuer_identity
        .verify(&signable, &token.signature)
        .map_err(|_| UcanError::SignatureInvalid)?;

    // 3. Check revocation
    if revocations.is_revoked(&token.content_hash()) {
        return Err(UcanError::Revoked);
    }

    // 4. If delegated, verify the chain
    if let Some(parent_hash) = &token.proof {
        if current_depth >= max_depth {
            return Err(UcanError::ChainTooDeep(max_depth));
        }

        let parent = proofs.resolve(parent_hash).ok_or(UcanError::ProofNotFound)?;

        // Chain continuity: parent.audience must equal this token's issuer
        if parent.audience != token.issuer {
            return Err(UcanError::ChainBroken);
        }

        // Attenuation: capability type must match
        if parent.capability != token.capability {
            return Err(UcanError::CapabilityMismatch);
        }

        // Attenuation: time bounds must be narrowed
        if token.not_before < parent.not_before {
            return Err(UcanError::AttenuationViolation);
        }
        if parent.expires_at != 0
            && (token.expires_at == 0 || token.expires_at > parent.expires_at)
        {
            return Err(UcanError::AttenuationViolation);
        }

        // Recursively verify parent
        verify_token_recursive(
            &parent,
            now,
            proofs,
            identities,
            revocations,
            max_depth,
            current_depth + 1,
        )?;
    }

    Ok(())
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 22 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): token verification with chain validation"
```

---

### Task 7: Delegation

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for delegation**

Add to the `tests` module:

```rust
    #[test]
    fn delegate_creates_valid_chain() {
        let mut rng = test_rng();
        let root_id = PrivateIdentity::generate(&mut rng);
        let delegate_id = PrivateIdentity::generate(&mut rng);
        let end_user = [0xCC; 16];

        // Root issues token to delegate
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

        // Delegate issues sub-token to end user
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
        let result = verify_token(
            &t3,
            0,
            &proofs,
            &ids,
            &MemoryRevocationSet::new(),
            1,
        );
        assert!(matches!(result, Err(UcanError::ChainTooDeep(1))));
    }
```

Run: `cargo test -p harmony-identity ucan::tests::delegate`
Expected: FAIL — `delegate` not defined.

**Step 2: Implement `delegate` on PrivateIdentity**

Add to the existing `impl PrivateIdentity` block in `ucan.rs`:

```rust
    /// Delegate a capability to another identity.
    ///
    /// Creates a child token referencing the parent by BLAKE3 hash.
    /// Enforces attenuation: capability type must match, time bounds
    /// must be narrowed or equal.
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
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 27 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): delegation with attenuation enforcement"
```

---

### Task 8: Revocation

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests for revocation**

Add to the `tests` module:

```rust
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
            .issue_root_token(
                &mut rng,
                &[0u8; 16],
                CapabilityType::Content,
                &[],
                0,
                0,
            )
            .unwrap();

        let revocation = issuer.revoke(&token, 5000);

        // Verify the revocation signature
        let signable = revocation.signable_bytes();
        issuer
            .public_identity()
            .verify(&signable, &revocation.signature)
            .unwrap();
        assert_eq!(revocation.token_hash, token.content_hash());
        assert_eq!(revocation.revoked_at, 5000);
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
```

Run: `cargo test -p harmony-identity ucan::tests::revoc`
Expected: FAIL — `Revocation` not defined.

**Step 2: Implement Revocation struct**

Add above the tests module in `ucan.rs`:

```rust
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

impl Revocation {
    /// Serialize to binary: `[16B issuer][32B token_hash][8B revoked_at][64B signature]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + 32 + 8 + 64);
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.token_hash);
        buf.extend_from_slice(&self.revoked_at.to_be_bytes());
        buf.extend_from_slice(&self.signature);
        buf
    }

    /// Deserialize from binary.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, UcanError> {
        // Exact size: 16 + 32 + 8 + 64 = 120
        if bytes.len() != 120 {
            return Err(UcanError::InvalidEncoding);
        }

        let mut issuer = [0u8; 16];
        issuer.copy_from_slice(&bytes[0..16]);

        let mut token_hash = [0u8; 32];
        token_hash.copy_from_slice(&bytes[16..48]);

        let revoked_at = u64::from_be_bytes(bytes[48..56].try_into().unwrap());

        let mut signature = [0u8; 64];
        signature.copy_from_slice(&bytes[56..120]);

        Ok(Self {
            issuer,
            token_hash,
            revoked_at,
            signature,
        })
    }

    /// Serialize the signable portion (everything except the signature).
    pub fn signable_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(16 + 32 + 8);
        buf.extend_from_slice(&self.issuer);
        buf.extend_from_slice(&self.token_hash);
        buf.extend_from_slice(&self.revoked_at.to_be_bytes());
        buf
    }
}
```

Add `revoke` to the `impl PrivateIdentity` block in `ucan.rs`:

```rust
    /// Revoke a token that this identity issued.
    ///
    /// Creates a signed revocation record. Only the issuer of a token can
    /// revoke it. Revoking a parent implicitly invalidates all downstream
    /// delegations (chain verification fails if any link is revoked).
    pub fn revoke(&self, token: &UcanToken, revoked_at: u64) -> Revocation {
        let mut revocation = Revocation {
            issuer: self.identity.address_hash,
            token_hash: token.content_hash(),
            revoked_at,
            signature: [0u8; SIGNATURE_LENGTH],
        };

        let signable = revocation.signable_bytes();
        revocation.signature = self.sign(&signable);
        revocation
    }
```

**Step 3: Run tests**

Run: `cargo test -p harmony-identity ucan::tests`
Expected: 31 tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(ucan): revocation records with chain invalidation"
```

---

### Task 9: Final Verification

**Files:**
- Modify: `crates/harmony-identity/src/lib.rs` (ensure re-exports are correct)

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: All tests pass (existing + ~31 new UCAN tests). Zero regressions.

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: Zero warnings.

**Step 3: Verify no_std compilation**

Run: `cargo check -p harmony-identity --no-default-features`
Expected: Compiles without errors.

**Step 4: Verify no_std chain (bottom-up)**

```bash
cargo check -p harmony-crypto --no-default-features
cargo check -p harmony-platform --no-default-features
cargo check -p harmony-identity --no-default-features
```

Expected: All three compile in no_std mode.

**Step 5: Final commit if any cleanup was needed**

If any adjustments were made during verification, commit them.

---

## Files Modified Summary

**4 files total:** 1 Cargo.toml + 1 lib.rs + 1 new ucan.rs + 1 design doc (already committed)

## Risk Assessment

- **Zero behavior change** for existing code — new module only, no modifications to identity.rs or error.rs
- **no_std compatible** — uses alloc::vec::Vec, hashbrown, no std-only types
- **Reticulum interop unaffected** — no changes to existing identity or crypto code
- **All resolver traits are sans-I/O** — callers provide state, no embedded I/O
- **Memory safety** — sensitive token fields (nonce, signature) are stack-allocated arrays, not heap. No ZeroizeOnDrop needed because tokens are authorization proofs, not secrets.

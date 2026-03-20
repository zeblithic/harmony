# Verifiable Credentials Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `harmony-credential`, a compact binary credential system with selective disclosure and bitstring revocation.

**Architecture:** New crate with 6 modules: `claim.rs` (salted hashing), `credential.rs` (struct + builder), `disclosure.rs` (presentation), `status_list.rs` (bitfield revocation), `verify.rs` (trait-based verification), `error.rs`. Sans-I/O signing via builder pattern. TDD throughout.

**Tech Stack:** Rust 1.85+, no_std + alloc, serde + postcard, BLAKE3, harmony-identity (IdentityRef, CryptoSuite), harmony-crypto (blake3_hash, ml_dsa, ed25519 via identity)

**Spec:** `docs/plans/2026-03-19-verifiable-credentials-design.md`

---

## File Structure

```
Cargo.toml                                        — add harmony-credential to workspace members + deps
crates/harmony-credential/Cargo.toml              — crate manifest (create)
crates/harmony-credential/src/lib.rs              — no_std setup, module declarations, re-exports (create)
crates/harmony-credential/src/error.rs            — CredentialError enum (create)
crates/harmony-credential/src/claim.rs            — Claim, SaltedClaim, digest() (create)
crates/harmony-credential/src/status_list.rs      — StatusList, StatusListResolver, MemoryStatusListResolver (create)
crates/harmony-credential/src/credential.rs       — Credential, CredentialBuilder, serialization (create)
crates/harmony-credential/src/disclosure.rs       — Presentation struct (create)
crates/harmony-credential/src/verify.rs           — CredentialKeyResolver, verify_credential, verify_presentation, MemoryKeyResolver (create)
```

---

### Task 1: Scaffold crate with error module

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Create: `crates/harmony-credential/Cargo.toml`
- Create: `crates/harmony-credential/src/lib.rs`
- Create: `crates/harmony-credential/src/error.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-credential"
description = "Compact binary verifiable credentials with selective disclosure for the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-crypto = { workspace = true, features = ["serde"] }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[dependencies.hashbrown]
workspace = true
optional = true

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]
test-utils = ["hashbrown"]

[dev-dependencies]
rand = { workspace = true }
hashbrown = { workspace = true }
```

- [ ] **Step 2: Create error.rs**

```rust
/// Errors returned by credential operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialError {
    NotYetValid,
    Expired,
    IssuerNotFound,
    SignatureInvalid,
    Revoked,
    StatusListNotFound,
    DisclosureMismatch,
    DuplicateDisclosure,
    IndexOutOfBounds,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for CredentialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotYetValid => write!(f, "credential not yet valid"),
            Self::Expired => write!(f, "credential expired"),
            Self::IssuerNotFound => write!(f, "issuer not found"),
            Self::SignatureInvalid => write!(f, "invalid credential signature"),
            Self::Revoked => write!(f, "credential revoked"),
            Self::StatusListNotFound => write!(f, "status list not found for issuer"),
            Self::DisclosureMismatch => write!(f, "disclosed claim does not match any digest"),
            Self::DuplicateDisclosure => write!(f, "duplicate disclosed claim"),
            Self::IndexOutOfBounds => write!(f, "status list index out of bounds"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CredentialError {}
```

- [ ] **Step 3: Create lib.rs**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;

pub use error::CredentialError;
```

- [ ] **Step 4: Add to workspace Cargo.toml**

In the workspace root `Cargo.toml`, add `"crates/harmony-credential"` to the `members` list (after `harmony-contacts`) and add a workspace dependency entry:

```toml
harmony-credential = { path = "crates/harmony-credential", default-features = false }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 0 tests, compiles cleanly.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-credential/
git commit -m "feat(credential): scaffold harmony-credential crate with error module"
```

---

### Task 2: Claim module with salted hashing

**Files:**
- Create: `crates/harmony-credential/src/claim.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Create claim.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digest_is_deterministic() {
        let sc = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA, 0xBB],
            },
            salt: [0x11; 16],
        };
        assert_eq!(sc.digest(), sc.digest());
    }

    #[test]
    fn different_salts_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x22; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn different_type_ids_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 2,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn different_values_produce_different_digests() {
        let a = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xAA],
            },
            salt: [0x11; 16],
        };
        let b = SaltedClaim {
            claim: Claim {
                type_id: 1,
                value: alloc::vec![0xBB],
            },
            salt: [0x11; 16],
        };
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn serde_round_trip() {
        let sc = SaltedClaim {
            claim: Claim {
                type_id: 42,
                value: alloc::vec![1, 2, 3],
            },
            salt: [0xFF; 16],
        };
        let bytes = postcard::to_allocvec(&sc).unwrap();
        let decoded: SaltedClaim = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.claim.type_id, 42);
        assert_eq!(decoded.claim.value, alloc::vec![1, 2, 3]);
        assert_eq!(decoded.salt, [0xFF; 16]);
        assert_eq!(decoded.digest(), sc.digest());
    }
}
```

Implementation:

```rust
use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use serde::{Deserialize, Serialize};

/// A single claim: opaque type ID + opaque value.
///
/// The credential crate does not interpret claim semantics.
/// `type_id` is application-defined; `value` is an arbitrary payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub type_id: u16,
    pub value: Vec<u8>,
}

/// A claim prepared for selective disclosure.
///
/// Each claim is paired with a random salt. The signed credential
/// stores only the BLAKE3 digest; the holder retains the full
/// `SaltedClaim` for later selective revelation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SaltedClaim {
    pub claim: Claim,
    pub salt: [u8; 16],
}

impl SaltedClaim {
    /// Compute the BLAKE3 digest: `BLAKE3(salt || type_id.to_le_bytes() || value)`.
    ///
    /// Including `type_id` in the hash prevents type reinterpretation attacks.
    pub fn digest(&self) -> [u8; 32] {
        let mut buf = Vec::with_capacity(16 + 2 + self.claim.value.len());
        buf.extend_from_slice(&self.salt);
        buf.extend_from_slice(&self.claim.type_id.to_le_bytes());
        buf.extend_from_slice(&self.claim.value);
        blake3_hash(&buf)
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

```rust
pub mod claim;

pub use claim::{Claim, SaltedClaim};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 5 claim tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/
git commit -m "feat(credential): add Claim and SaltedClaim with BLAKE3 digest"
```

---

### Task 3: StatusList module with bitfield revocation

**Files:**
- Create: `crates/harmony-credential/src/status_list.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Create status_list.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_list_all_valid() {
        let list = StatusList::new(128);
        for i in 0..128 {
            assert!(!list.is_revoked(i));
        }
    }

    #[test]
    fn revoke_sets_bit() {
        let mut list = StatusList::new(128);
        list.revoke(42);
        assert!(list.is_revoked(42));
        assert!(!list.is_revoked(41));
        assert!(!list.is_revoked(43));
    }

    #[test]
    fn revoke_is_idempotent() {
        let mut list = StatusList::new(128);
        list.revoke(10);
        list.revoke(10);
        assert!(list.is_revoked(10));
    }

    #[test]
    fn capacity_matches_construction() {
        let list = StatusList::new(1000);
        assert_eq!(list.capacity(), 1000);
    }

    #[test]
    fn default_capacity() {
        let list = StatusList::default();
        assert_eq!(list.capacity(), DEFAULT_CAPACITY);
    }

    #[test]
    fn boundary_indices() {
        let mut list = StatusList::new(16);
        list.revoke(0);
        list.revoke(7);
        list.revoke(8);
        list.revoke(15);
        assert!(list.is_revoked(0));
        assert!(list.is_revoked(7));
        assert!(list.is_revoked(8));
        assert!(list.is_revoked(15));
        assert!(!list.is_revoked(1));
        assert!(!list.is_revoked(9));
    }

    #[test]
    fn out_of_bounds_is_valid() {
        let list = StatusList::new(16);
        assert!(!list.is_revoked(16));
        assert!(!list.is_revoked(9999));
    }

    #[test]
    fn revoke_out_of_bounds_is_noop() {
        let mut list = StatusList::new(16);
        list.revoke(16); // should not panic
        list.revoke(9999);
        assert_eq!(list.capacity(), 16);
    }

    #[test]
    fn serde_round_trip() {
        let mut list = StatusList::new(256);
        list.revoke(0);
        list.revoke(100);
        list.revoke(255);

        let bytes = postcard::to_allocvec(&list).unwrap();
        let decoded: StatusList = postcard::from_bytes(&bytes).unwrap();
        assert!(decoded.is_revoked(0));
        assert!(decoded.is_revoked(100));
        assert!(decoded.is_revoked(255));
        assert!(!decoded.is_revoked(1));
        assert_eq!(decoded.capacity(), 256);
    }
}
```

Implementation:

```rust
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use harmony_identity::IdentityRef;

/// Default status list capacity: 16,384 credential slots (2KB).
pub const DEFAULT_CAPACITY: u32 = 16_384;

/// A compact bitstring for tracking credential revocation status.
///
/// Each bit represents one credential slot: 0 = valid, 1 = revoked.
/// Managed by the credential issuer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusList {
    bits: Vec<u8>,
    capacity: u32,
}

impl StatusList {
    /// Create a new status list with the given capacity (number of credential slots).
    pub fn new(capacity: u32) -> Self {
        let byte_len = ((capacity + 7) / 8) as usize;
        Self {
            bits: vec![0u8; byte_len],
            capacity,
        }
    }

    /// Check whether the credential at `index` has been revoked.
    ///
    /// Returns `false` for out-of-bounds indices (treat unknown as valid).
    pub fn is_revoked(&self, index: u32) -> bool {
        if index >= self.capacity {
            return false;
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] & (1 << bit_idx) != 0
    }

    /// Revoke the credential at `index`. No-op if out of bounds.
    pub fn revoke(&mut self, index: u32) {
        if index >= self.capacity {
            return;
        }
        let byte_idx = (index / 8) as usize;
        let bit_idx = (index % 8) as u8;
        self.bits[byte_idx] |= 1 << bit_idx;
    }

    /// Total number of credential slots in this list.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

impl Default for StatusList {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// Check whether a credential has been revoked.
///
/// Returns `Some(true)` if revoked, `Some(false)` if valid,
/// `None` if the issuer's status list could not be resolved.
pub trait StatusListResolver {
    fn is_revoked(&self, issuer: &IdentityRef, index: u32) -> Option<bool>;
}

/// In-memory status list resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryStatusListResolver {
    lists: hashbrown::HashMap<harmony_identity::IdentityHash, StatusList>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryStatusListResolver {
    pub fn new() -> Self {
        Self {
            lists: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, issuer_hash: harmony_identity::IdentityHash, list: StatusList) {
        self.lists.insert(issuer_hash, list);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl StatusListResolver for MemoryStatusListResolver {
    fn is_revoked(&self, issuer: &IdentityRef, index: u32) -> Option<bool> {
        self.lists.get(&issuer.hash).map(|list| list.is_revoked(index))
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

```rust
pub mod status_list;

pub use status_list::{StatusList, StatusListResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 5 claim + 7 status list = 12 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/
git commit -m "feat(credential): add StatusList with bitfield revocation"
```

---

### Task 4: Credential struct, builder, and serialization

**Files:**
- Create: `crates/harmony-credential/src/credential.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Create credential.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_issuer() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    fn test_subject() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_digest_count() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);

        let payload = builder.signable_payload();
        let (cred, claims) = builder.build(payload.clone()); // use payload as fake sig

        assert_eq!(cred.claim_digests.len(), 2);
        assert_eq!(claims.len(), 2);
        assert_eq!(cred.issuer, test_issuer());
        assert_eq!(cred.subject, test_subject());
        assert_eq!(cred.not_before, 1000);
        assert_eq!(cred.expires_at, 2000);
        assert_eq!(cred.issued_at, 1000);
        assert_eq!(cred.nonce, [0x01; 16]);
        assert!(cred.status_list_index.is_none());
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);

        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    fn content_hash_is_deterministic() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);

        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.content_hash(), cred.content_hash());
    }

    #[test]
    fn not_before_defaults_to_issued_at() {
        let builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            500,
            2000,
            [0x01; 16],
        );
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.not_before, 500);
    }

    #[test]
    fn not_before_can_be_overridden() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            500,
            2000,
            [0x01; 16],
        );
        builder.not_before(700);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.not_before, 700);
    }

    #[test]
    fn status_list_index_set() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.status_list_index(42);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.status_list_index, Some(42));
    }

    #[test]
    fn self_issued_credential() {
        let id = test_issuer();
        let builder = CredentialBuilder::new(id, id, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let (cred, _) = builder.build(payload);
        assert_eq!(cred.issuer, cred.subject);
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = CredentialBuilder::new(
            test_issuer(),
            test_subject(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.status_list_index(5);

        let payload = builder.signable_payload();
        let (cred, _) = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = cred.serialize().unwrap();
        let restored = Credential::deserialize(&bytes).unwrap();

        assert_eq!(restored.issuer, cred.issuer);
        assert_eq!(restored.subject, cred.subject);
        assert_eq!(restored.claim_digests, cred.claim_digests);
        assert_eq!(restored.status_list_index, cred.status_list_index);
        assert_eq!(restored.not_before, cred.not_before);
        assert_eq!(restored.expires_at, cred.expires_at);
        assert_eq!(restored.issued_at, cred.issued_at);
        assert_eq!(restored.nonce, cred.nonce);
        assert_eq!(restored.signature, cred.signature);
        assert_eq!(restored.content_hash(), cred.content_hash());
    }
}
```

Implementation:

```rust
use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::claim::SaltedClaim;
use crate::error::CredentialError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of a credential.
/// Everything except the signature.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    issuer: IdentityRef,
    subject: IdentityRef,
    claim_digests: Vec<[u8; 32]>,
    status_list_index: Option<u32>,
    not_before: u64,
    expires_at: u64,
    issued_at: u64,
    nonce: [u8; 16],
}

/// A compact binary verifiable credential.
///
/// Binds an issuer's signature to a set of claim digests about a subject.
/// Claims are stored as BLAKE3 hashes for selective disclosure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Credential {
    pub issuer: IdentityRef,
    pub subject: IdentityRef,
    pub claim_digests: Vec<[u8; 32]>,
    pub status_list_index: Option<u32>,
    pub not_before: u64,
    pub expires_at: u64,
    pub issued_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl Credential {
    /// BLAKE3 hash of the signable payload. Stable identifier for this
    /// credential, usable for indexing, deduplication, and future
    /// delegation chain references.
    pub fn content_hash(&self) -> [u8; 32] {
        blake3_hash(&self.signable_bytes())
    }

    /// Reconstruct the signable payload bytes (everything except signature).
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: self.claim_digests.clone(),
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize the credential to bytes with a format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, CredentialError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| CredentialError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize a credential from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, CredentialError> {
        if data.is_empty() {
            return Err(CredentialError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(CredentialError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| CredentialError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a credential.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final credential.
pub struct CredentialBuilder {
    issuer: IdentityRef,
    subject: IdentityRef,
    issued_at: u64,
    expires_at: u64,
    not_before: u64,
    nonce: [u8; 16],
    claims: Vec<SaltedClaim>,
    status_list_index: Option<u32>,
}

impl CredentialBuilder {
    /// Create a new builder.
    ///
    /// `not_before` defaults to `issued_at`. `expires_at` and `nonce` are
    /// required to prevent accidental omission.
    pub fn new(
        issuer: IdentityRef,
        subject: IdentityRef,
        issued_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        Self {
            issuer,
            subject,
            issued_at,
            expires_at,
            not_before: issued_at,
            nonce,
            claims: Vec::new(),
            status_list_index: None,
        }
    }

    /// Override the `not_before` timestamp (defaults to `issued_at`).
    pub fn not_before(&mut self, not_before: u64) -> &mut Self {
        self.not_before = not_before;
        self
    }

    /// Add a claim with the given type, value, and salt.
    pub fn add_claim(&mut self, type_id: u16, value: Vec<u8>, salt: [u8; 16]) -> &mut Self {
        self.claims.push(SaltedClaim {
            claim: crate::claim::Claim { type_id, value },
            salt,
        });
        self
    }

    /// Set the revocation status list index.
    pub fn status_list_index(&mut self, index: u32) -> &mut Self {
        self.status_list_index = Some(index);
        self
    }

    /// Produce the signable payload bytes.
    ///
    /// The caller signs these bytes externally with their private key.
    pub fn signable_payload(&self) -> Vec<u8> {
        let digests: Vec<[u8; 32]> = self.claims.iter().map(|c| c.digest()).collect();
        let payload = SignablePayload {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: digests,
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature to produce the Credential + SaltedClaims.
    ///
    /// The `SaltedClaim`s should be stored by the holder for later
    /// selective disclosure.
    pub fn build(self, signature: Vec<u8>) -> (Credential, Vec<SaltedClaim>) {
        let digests: Vec<[u8; 32]> = self.claims.iter().map(|c| c.digest()).collect();
        let credential = Credential {
            issuer: self.issuer,
            subject: self.subject,
            claim_digests: digests,
            status_list_index: self.status_list_index,
            not_before: self.not_before,
            expires_at: self.expires_at,
            issued_at: self.issued_at,
            nonce: self.nonce,
            signature,
        };
        (credential, self.claims)
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

```rust
pub mod credential;

pub use credential::{Credential, CredentialBuilder};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 5 claim + 9 status list + 9 credential = 23 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/
git commit -m "feat(credential): add Credential struct, CredentialBuilder, and serialization"
```

---

### Task 5: Presentation (disclosure) module

**Files:**
- Create: `crates/harmony-credential/src/disclosure.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Create disclosure.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::{Claim, SaltedClaim};
    use crate::credential::CredentialBuilder;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn build_test_credential() -> (crate::credential::Credential, Vec<SaltedClaim>) {
        let issuer = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let subject = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);
        let mut builder = CredentialBuilder::new(issuer, subject, 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);
        builder.add_claim(3, alloc::vec![0xCC], [0x33; 16]);
        let payload = builder.signable_payload();
        builder.build(payload)
    }

    #[test]
    fn verify_all_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn verify_subset_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone(), claims[2].clone()],
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn verify_empty_disclosures() {
        let (cred, _claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: Vec::new(),
        };
        assert!(presentation.verify_disclosures().is_ok());
    }

    #[test]
    fn rejects_tampered_claim() {
        let (cred, mut claims) = build_test_credential();
        claims[0].claim.value = alloc::vec![0xFF]; // tamper
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            crate::error::CredentialError::DisclosureMismatch
        );
    }

    #[test]
    fn rejects_unknown_claim() {
        let (cred, _claims) = build_test_credential();
        let unknown = SaltedClaim {
            claim: Claim {
                type_id: 99,
                value: alloc::vec![0xDD],
            },
            salt: [0x99; 16],
        };
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![unknown],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            crate::error::CredentialError::DisclosureMismatch
        );
    }

    #[test]
    fn rejects_duplicate_disclosures() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone(), claims[0].clone()],
        };
        assert_eq!(
            presentation.verify_disclosures().unwrap_err(),
            crate::error::CredentialError::DuplicateDisclosure
        );
    }

    #[test]
    fn serde_round_trip() {
        let (cred, claims) = build_test_credential();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[1].clone()],
        };
        let bytes = postcard::to_allocvec(&presentation).unwrap();
        let decoded: Presentation = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.credential.issuer, presentation.credential.issuer);
        assert_eq!(decoded.disclosed_claims.len(), 1);
        assert!(decoded.verify_disclosures().is_ok());
    }
}
```

Implementation:

```rust
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::claim::SaltedClaim;
use crate::credential::Credential;
use crate::error::CredentialError;

/// A credential presentation with selectively disclosed claims.
///
/// The holder sends this to a verifier, revealing only the claims
/// they choose. The verifier checks that each disclosed claim's
/// digest appears in the signed credential.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Presentation {
    pub credential: Credential,
    pub disclosed_claims: Vec<SaltedClaim>,
}

impl Presentation {
    /// Verify that all disclosed claims match digests in the credential.
    ///
    /// Checks:
    /// 1. Each disclosed claim's digest appears in `credential.claim_digests`
    /// 2. No two disclosed claims map to the same digest (no duplicates)
    pub fn verify_disclosures(&self) -> Result<(), CredentialError> {
        let mut matched_indices = Vec::new();

        for disclosed in &self.disclosed_claims {
            let digest = disclosed.digest();
            let pos = self
                .credential
                .claim_digests
                .iter()
                .position(|d| *d == digest)
                .ok_or(CredentialError::DisclosureMismatch)?;

            if matched_indices.contains(&pos) {
                return Err(CredentialError::DuplicateDisclosure);
            }
            matched_indices.push(pos);
        }

        Ok(())
    }
}
```

- [ ] **Step 2: Add module to lib.rs**

```rust
pub mod disclosure;

pub use disclosure::Presentation;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 5 claim + 9 status list + 9 credential + 7 disclosure = 30 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/
git commit -m "feat(credential): add Presentation with selective disclosure verification"
```

---

### Task 6: Verification module with CredentialKeyResolver

**Files:**
- Create: `crates/harmony-credential/src/verify.rs`
- Modify: `crates/harmony-credential/src/lib.rs`

- [ ] **Step 1: Create verify.rs with tests first**

Tests (these require Ed25519 signing, so use `harmony_identity::PrivateIdentity`):

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::claim::SaltedClaim;
    use crate::credential::CredentialBuilder;
    use crate::disclosure::Presentation;
    use crate::status_list::{MemoryStatusListResolver, StatusList};
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    /// Build a signed credential using a real Ed25519 keypair.
    fn build_signed_credential() -> (
        harmony_identity::PrivateIdentity,
        crate::credential::Credential,
        Vec<SaltedClaim>,
        IdentityRef,
    ) {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        let mut builder =
            CredentialBuilder::new(issuer_ref, subject_ref, 1000, 2000, [0x01; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.add_claim(2, alloc::vec![0xBB], [0x22; 16]);

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, claims) = builder.build(signature.to_vec());

        (private, cred, claims, issuer_ref)
    }

    fn setup_resolver(
        identity: &harmony_identity::Identity,
        issuer_ref: &IdentityRef,
    ) -> MemoryKeyResolver {
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.to_public_bytes().to_vec());
        resolver
    }

    fn empty_status() -> MemoryStatusListResolver {
        MemoryStatusListResolver::new()
    }

    // ---- verify_credential tests ----

    #[test]
    fn valid_credential_passes() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());
    }

    #[test]
    fn expired_credential_rejected() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 3000, &resolver, &status).unwrap_err(),
            CredentialError::Expired
        );
    }

    #[test]
    fn not_yet_valid_rejected() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 500, &resolver, &status).unwrap_err(),
            CredentialError::NotYetValid
        );
    }

    #[test]
    fn unknown_issuer_rejected() {
        let (_, cred, _, _) = build_signed_credential();
        let resolver = MemoryKeyResolver::new(); // empty
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::IssuerNotFound
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let (private, mut cred, _, issuer_ref) = build_signed_credential();
        cred.signature = alloc::vec![0xFF; 64]; // wrong signature
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::SignatureInvalid
        );
    }

    #[test]
    fn revoked_credential_rejected() {
        let (private, mut cred, _, issuer_ref) = build_signed_credential();

        // Rebuild with status_list_index
        let identity = private.public_identity();
        let mut builder = CredentialBuilder::new(
            issuer_ref,
            IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519),
            1000,
            2000,
            [0x02; 16],
        );
        builder.add_claim(1, alloc::vec![0xAA], [0x11; 16]);
        builder.status_list_index(5);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let resolver = setup_resolver(&identity, &issuer_ref);
        let mut status = MemoryStatusListResolver::new();
        let mut list = StatusList::new(128);
        list.revoke(5);
        status.insert(issuer_ref.hash, list);

        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::Revoked
        );
    }

    #[test]
    fn missing_status_list_rejected() {
        let (private, _, _, issuer_ref) = build_signed_credential();
        let identity = private.public_identity();

        let mut builder = CredentialBuilder::new(
            issuer_ref,
            IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519),
            1000,
            2000,
            [0x03; 16],
        );
        builder.status_list_index(5);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let resolver = setup_resolver(&identity, &issuer_ref);
        let status = empty_status(); // no list registered

        assert_eq!(
            verify_credential(&cred, 1500, &resolver, &status).unwrap_err(),
            CredentialError::StatusListNotFound
        );
    }

    #[test]
    fn non_revocable_skips_status_check() {
        let (private, cred, _, issuer_ref) = build_signed_credential();
        assert!(cred.status_list_index.is_none());
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status(); // no lists at all — fine for non-revocable
        assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());
    }

    // ---- verify_presentation tests ----

    #[test]
    fn valid_presentation_passes() {
        let (private, cred, claims, issuer_ref) = build_signed_credential();
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: claims,
        };
        assert!(verify_presentation(&presentation, 1500, &resolver, &status).is_ok());
    }

    #[test]
    fn presentation_with_tampered_claim_rejected() {
        let (private, cred, mut claims, issuer_ref) = build_signed_credential();
        claims[0].claim.value = alloc::vec![0xFF]; // tamper
        let resolver = setup_resolver(&private.public_identity(), &issuer_ref);
        let status = empty_status();
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };
        assert_eq!(
            verify_presentation(&presentation, 1500, &resolver, &status).unwrap_err(),
            CredentialError::DisclosureMismatch
        );
    }

    // ---- ML-DSA-65 issuer test ----

    #[test]
    fn ml_dsa65_issuer_verification() {
        // ML-DSA-65 keygen is expensive; run in a thread with larger stack
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);

                // Derive address from PQ public keys
                let pq_id = harmony_identity::PqIdentity::from_public_keys(
                    enc_pk.clone(),
                    sign_pk.clone(),
                );
                let issuer_ref = IdentityRef::from(&pq_id);
                let subject_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);

                let mut builder =
                    CredentialBuilder::new(issuer_ref, subject_ref, 1000, 2000, [0x05; 16]);
                builder.add_claim(1, alloc::vec![0xDD], [0x44; 16]);

                let payload = builder.signable_payload();
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let (cred, claims) = builder.build(signature.as_bytes().to_vec());

                // Resolver stores ML-DSA-65 public key bytes
                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(issuer_ref.hash, sign_pk.as_bytes());
                let status = MemoryStatusListResolver::new();

                assert!(verify_credential(&cred, 1500, &resolver, &status).is_ok());

                // Full presentation flow
                let presentation = Presentation {
                    credential: cred,
                    disclosed_claims: claims,
                };
                assert!(verify_presentation(&presentation, 1500, &resolver, &status).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    // ---- Integration tests ----

    #[test]
    fn full_flow_issue_present_verify() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        // 1. Issue
        let mut builder =
            CredentialBuilder::new(issuer_ref, subject_ref, 1000, 5000, [0x10; 16]);
        builder.add_claim(1, alloc::vec![0x01], [0xA1; 16]); // age claim
        builder.add_claim(2, alloc::vec![0x02], [0xA2; 16]); // org claim
        builder.add_claim(3, alloc::vec![0x03], [0xA3; 16]); // license claim
        builder.status_list_index(0);

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, claims) = builder.build(signature.to_vec());

        // 2. Holder selects subset to disclose (only age claim)
        let presentation = Presentation {
            credential: cred,
            disclosed_claims: vec![claims[0].clone()],
        };

        // 3. Verify
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.to_public_bytes().to_vec());
        let mut status = MemoryStatusListResolver::new();
        status.insert(issuer_ref.hash, StatusList::new(128));

        assert!(verify_presentation(&presentation, 2000, &resolver, &status).is_ok());
    }

    #[test]
    fn revocation_after_issuance() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let issuer_ref = IdentityRef::from(identity);
        let subject_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        // Issue credential at index 7
        let mut builder =
            CredentialBuilder::new(issuer_ref, subject_ref, 1000, 5000, [0x20; 16]);
        builder.add_claim(1, alloc::vec![0xAA], [0xB1; 16]);
        builder.status_list_index(7);
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let (cred, _) = builder.build(signature.to_vec());

        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(issuer_ref.hash, identity.to_public_bytes().to_vec());
        let mut status = MemoryStatusListResolver::new();
        let mut list = StatusList::new(128);
        status.insert(issuer_ref.hash, list.clone());

        // Valid before revocation
        assert!(verify_credential(&cred, 2000, &resolver, &status).is_ok());

        // Revoke and update resolver
        list.revoke(7);
        status.insert(issuer_ref.hash, list);

        // Rejected after revocation
        assert_eq!(
            verify_credential(&cred, 2000, &resolver, &status).unwrap_err(),
            CredentialError::Revoked
        );
    }
}
```

Implementation:

```rust
use alloc::vec::Vec;
use harmony_identity::{CryptoSuite, IdentityRef};

use crate::credential::Credential;
use crate::disclosure::Presentation;
use crate::error::CredentialError;
use crate::status_list::StatusListResolver;

/// Resolve an issuer's signing public key for credential verification.
///
/// The `issued_at` parameter enables KEL-backed resolvers to return
/// the signing key that was active at credential issuance time.
/// For non-rotatable identities, `issued_at` can be ignored.
pub trait CredentialKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, issued_at: u64) -> Option<Vec<u8>>;
}

/// Verify a credential's time bounds, signature, and revocation status.
pub fn verify_credential(
    credential: &Credential,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError> {
    // 1. Time bounds
    if now < credential.not_before {
        return Err(CredentialError::NotYetValid);
    }
    if now >= credential.expires_at {
        return Err(CredentialError::Expired);
    }

    // 2. Issuer resolution
    let key_bytes = keys
        .resolve(&credential.issuer, credential.issued_at)
        .ok_or(CredentialError::IssuerNotFound)?;

    // 3. Signature verification
    let payload = credential.signable_bytes();
    verify_signature(
        credential.issuer.suite,
        &key_bytes,
        &payload,
        &credential.signature,
    )?;

    // 4. Revocation check
    if let Some(idx) = credential.status_list_index {
        match status_lists.is_revoked(&credential.issuer, idx) {
            Some(true) => return Err(CredentialError::Revoked),
            Some(false) => {}
            None => return Err(CredentialError::StatusListNotFound),
        }
    }

    Ok(())
}

/// Verify a presentation: credential verification + disclosure integrity.
pub fn verify_presentation(
    presentation: &Presentation,
    now: u64,
    keys: &impl CredentialKeyResolver,
    status_lists: &impl StatusListResolver,
) -> Result<(), CredentialError> {
    verify_credential(&presentation.credential, now, keys, status_lists)?;
    presentation.verify_disclosures()?;
    Ok(())
}

/// Dispatch signature verification based on CryptoSuite.
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), CredentialError> {
    match suite {
        CryptoSuite::Ed25519 => {
            let sig: [u8; 64] = signature
                .try_into()
                .map_err(|_| CredentialError::SignatureInvalid)?;
            let identity = harmony_identity::Identity::from_public_bytes(key_bytes)
                .map_err(|_| CredentialError::IssuerNotFound)?;
            identity
                .verify(message, &sig)
                .map_err(|_| CredentialError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| CredentialError::IssuerNotFound)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| CredentialError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| CredentialError::SignatureInvalid)
        }
    }
}

/// In-memory key resolver for testing.
#[cfg(any(test, feature = "test-utils"))]
pub struct MemoryKeyResolver {
    keys: hashbrown::HashMap<harmony_identity::IdentityHash, Vec<u8>>,
}

#[cfg(any(test, feature = "test-utils"))]
impl MemoryKeyResolver {
    pub fn new() -> Self {
        Self {
            keys: hashbrown::HashMap::new(),
        }
    }

    pub fn insert(&mut self, hash: harmony_identity::IdentityHash, key_bytes: Vec<u8>) {
        self.keys.insert(hash, key_bytes);
    }
}

#[cfg(any(test, feature = "test-utils"))]
impl CredentialKeyResolver for MemoryKeyResolver {
    fn resolve(&self, issuer: &IdentityRef, _issued_at: u64) -> Option<Vec<u8>> {
        self.keys.get(&issuer.hash).cloned()
    }
}
```

**API note:** `Identity::from_public_bytes(key_bytes)` takes the 64-byte combined public key (32 X25519 + 32 Ed25519). The resolver must store the full 64-byte output of `identity.to_public_bytes()`, not just the 32-byte verifying key.

- [ ] **Step 2: Add module and final re-exports to lib.rs**

Replace the entire `lib.rs` with the final version:

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod claim;
pub mod credential;
pub mod disclosure;
pub mod error;
pub mod status_list;
pub mod verify;

pub use claim::{Claim, SaltedClaim};
pub use credential::{Credential, CredentialBuilder};
pub use disclosure::Presentation;
pub use error::CredentialError;
pub use status_list::{StatusList, StatusListResolver};
pub use verify::{verify_credential, verify_presentation, CredentialKeyResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use status_list::MemoryStatusListResolver;
#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-credential`
Expected: 5 claim + 9 status list + 9 credential + 7 disclosure + 13 verify = 43 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-credential/
git commit -m "feat(credential): add verification with CredentialKeyResolver and presentation checks"
```

---

### Task 7: Final quality gate

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: All tests pass (existing + 43 new credential tests).

- [ ] **Step 2: Format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.

- [ ] **Step 4: Fix any issues found, commit**

If any quality gate fails, fix the issues and commit:
```bash
git add crates/harmony-credential/
git commit -m "fix(credential): address clippy/fmt findings"
```

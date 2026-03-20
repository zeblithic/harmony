# Profiles & Endorsements Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `harmony-profile` crate for self-signed profile records and endorser-signed endorsement records, with verification, serialization, and Zenoh endorsement namespace patterns.

**Architecture:** Two signed record types (ProfileRecord, EndorsementRecord) built via sans-I/O builders, verified via ProfileKeyResolver trait, serialized with postcard + format version prefix. Follows the exact patterns established by `harmony-discovery` (AnnounceRecord/AnnounceBuilder) and `harmony-credential` (CredentialBuilder).

**Tech Stack:** Rust (no_std + alloc), postcard serialization, ed25519-dalek, harmony-crypto (ML-DSA-65), harmony-identity (IdentityRef, CryptoSuite)

**Design spec:** `docs/plans/2026-03-20-profiles-endorsements-design.md`

---

## File Structure

```
crates/harmony-profile/
  Cargo.toml              — crate manifest
  src/
    lib.rs                — no_std setup, module declarations, re-exports
    error.rs              — ProfileError enum with Display + std::error::Error
    profile.rs            — ProfileRecord, ProfileBuilder, SignablePayload, serialization
    endorsement.rs        — EndorsementRecord, EndorsementBuilder, SignablePayload, serialization
    verify.rs             — verify_profile(), verify_endorsement(), ProfileKeyResolver trait,
                            MemoryKeyResolver, signature dispatch

crates/harmony-zenoh/src/
  namespace.rs            — add `pub mod endorsement { ... }` with key expression constants/builders
```

Also modify: `Cargo.toml` (workspace root) to add `harmony-profile` to members list.

---

### Task 1: Crate Scaffold and Error Type

**Files:**
- Modify: `Cargo.toml` (workspace root — add member + dependency)
- Create: `crates/harmony-profile/Cargo.toml`
- Create: `crates/harmony-profile/src/lib.rs`
- Create: `crates/harmony-profile/src/error.rs`

- [ ] **Step 1: Create `crates/harmony-profile/Cargo.toml`**

```toml
[package]
name = "harmony-profile"
description = "Signed identity profiles and endorsement records for the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
ed25519-dalek = { workspace = true }
harmony-crypto = { workspace = true, features = ["serde"] }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[dependencies.hashbrown]
workspace = true
optional = true

[dev-dependencies]
rand = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]
test-utils = ["hashbrown"]
```

- [ ] **Step 2: Create `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;

pub use error::ProfileError;
```

- [ ] **Step 3: Write `src/error.rs`**

```rust
/// Errors returned by profile and endorsement operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    Expired,
    FutureTimestamp,
    InvalidRecord,
    SignatureInvalid,
    KeyNotFound,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Expired => write!(f, "record expired"),
            Self::FutureTimestamp => write!(f, "published_at is too far in the future"),
            Self::InvalidRecord => write!(f, "record structurally invalid"),
            Self::SignatureInvalid => write!(f, "invalid signature"),
            Self::KeyNotFound => write!(f, "signing key not found"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProfileError {}
```

- [ ] **Step 4: Add to workspace root `Cargo.toml`**

Add `"crates/harmony-profile"` to the `[workspace] members` list (alphabetical, after `harmony-peers`). Add to `[workspace.dependencies]`:

```toml
harmony-profile = { path = "crates/harmony-profile", default-features = false }
```

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-profile`
Expected: compiles

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-profile/
git commit -m "feat(profile): scaffold harmony-profile crate with error type"
```

---

### Task 2: ProfileRecord and ProfileBuilder

**Files:**
- Create: `crates/harmony-profile/src/profile.rs`
- Modify: `crates/harmony-profile/src/lib.rs` (add module + re-exports)

**Reference:** `crates/harmony-discovery/src/record.rs` for the builder + SignablePayload + serialization pattern.

- [ ] **Step 1: Write tests in `src/profile.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_identity() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_fields() {
        let id = test_identity();
        let mut builder = ProfileBuilder::new(id, 1000, 2000, [0x01; 16]);
        builder
            .display_name(alloc::string::String::from("Alice"))
            .status_text(alloc::string::String::from("Hello world"))
            .avatar_cid([0xCC; 32]);

        let payload = builder.signable_payload();
        let record = builder.build(payload.clone());

        assert_eq!(record.identity_ref, id);
        assert_eq!(record.display_name.as_deref(), Some("Alice"));
        assert_eq!(record.status_text.as_deref(), Some("Hello world"));
        assert_eq!(record.avatar_cid, Some([0xCC; 32]));
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_before_published_at() {
        ProfileBuilder::new(test_identity(), 2000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        ProfileBuilder::new(test_identity(), 1000, 1000, [0x01; 16]);
    }

    #[test]
    fn sparse_profile_works() {
        let builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(payload);
        assert!(record.display_name.is_none());
        assert!(record.status_text.is_none());
        assert!(record.avatar_cid.is_none());
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = ProfileBuilder::new(test_identity(), 1000, 2000, [0x01; 16]);
        builder.display_name(alloc::string::String::from("Bob"));

        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = ProfileRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored, record);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            ProfileRecord::deserialize(&[]),
            Err(ProfileError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            ProfileRecord::deserialize(&[0xFF]),
            Err(ProfileError::DeserializeError("unsupported format version"))
        ));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-profile`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement ProfileRecord + ProfileBuilder**

```rust
use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::ProfileError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of a profile record.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    identity_ref: IdentityRef,
    display_name: Option<String>,
    status_text: Option<String>,
    avatar_cid: Option<[u8; 32]>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// A signed profile record publishing an identity's public metadata.
///
/// # Construction
///
/// Produce records via [`ProfileBuilder`]; direct struct construction
/// bypasses validity checks (e.g. `expires_at > published_at`).
///
/// Verified by [`verify_profile()`](crate::verify::verify_profile).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProfileRecord {
    pub identity_ref: IdentityRef,
    pub display_name: Option<String>,
    /// Freetext status. Future convention: `/book/<address>` for CAS
    /// content bundles.
    pub status_text: Option<String>,
    pub avatar_cid: Option<[u8; 32]>,
    pub published_at: u64,
    pub expires_at: u64,
    /// Must be cryptographically random and unique per publication.
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl ProfileRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    ///
    /// Uses the compile-time `FORMAT_VERSION` constant. Records from
    /// persistent storage MUST go through [`ProfileRecord::deserialize`].
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            display_name: self.display_name.clone(),
            status_text: self.status_text.clone(),
            avatar_cid: self.avatar_cid,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize with format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, ProfileError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| ProfileError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.is_empty() {
            return Err(ProfileError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ProfileError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ProfileError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a signed profile record.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final record.
pub struct ProfileBuilder {
    identity_ref: IdentityRef,
    display_name: Option<String>,
    status_text: Option<String>,
    avatar_cid: Option<[u8; 32]>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl ProfileBuilder {
    /// Create a new builder.
    ///
    /// All timestamps are unix epoch seconds. `nonce` must be
    /// cryptographically random.
    ///
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        identity_ref: IdentityRef,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        assert!(
            expires_at > published_at,
            "expires_at ({expires_at}) must be > published_at ({published_at})"
        );
        Self {
            identity_ref,
            display_name: None,
            status_text: None,
            avatar_cid: None,
            published_at,
            expires_at,
            nonce,
        }
    }

    pub fn display_name(&mut self, name: String) -> &mut Self {
        self.display_name = Some(name);
        self
    }

    pub fn status_text(&mut self, text: String) -> &mut Self {
        self.status_text = Some(text);
        self
    }

    pub fn avatar_cid(&mut self, cid: [u8; 32]) -> &mut Self {
        self.avatar_cid = Some(cid);
        self
    }

    /// Produce the signable payload bytes.
    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            display_name: self.display_name.clone(),
            status_text: self.status_text.clone(),
            avatar_cid: self.avatar_cid,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature.
    pub fn build(self, signature: Vec<u8>) -> ProfileRecord {
        ProfileRecord {
            identity_ref: self.identity_ref,
            display_name: self.display_name,
            status_text: self.status_text,
            avatar_cid: self.avatar_cid,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
            signature,
        }
    }
}
```

- [ ] **Step 4: Update `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod profile;

pub use error::ProfileError;
pub use profile::{ProfileBuilder, ProfileRecord};
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-profile`
Expected: all 7 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-profile/src/profile.rs crates/harmony-profile/src/lib.rs
git commit -m "feat(profile): add ProfileRecord and ProfileBuilder"
```

---

### Task 3: EndorsementRecord and EndorsementBuilder

**Files:**
- Create: `crates/harmony-profile/src/endorsement.rs`
- Modify: `crates/harmony-profile/src/lib.rs` (add module + re-exports)

- [ ] **Step 1: Write tests in `src/endorsement.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_endorser() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::MlDsa65)
    }

    fn test_endorsee() -> IdentityRef {
        IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65)
    }

    #[test]
    fn builder_produces_correct_fields() {
        let mut builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        builder.reason(alloc::string::String::from("Verified in person"));

        let payload = builder.signable_payload();
        let record = builder.build(payload.clone());

        assert_eq!(record.endorser, test_endorser());
        assert_eq!(record.endorsee, test_endorsee());
        assert_eq!(record.type_id, 42);
        assert_eq!(record.reason.as_deref(), Some("Verified in person"));
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        let p1 = builder.signable_payload();
        let p2 = builder.signable_payload();
        assert_eq!(p1, p2);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_before_published_at() {
        EndorsementBuilder::new(test_endorser(), test_endorsee(), 42, 2000, 1000, [0x01; 16]);
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        EndorsementBuilder::new(test_endorser(), test_endorsee(), 42, 1000, 1000, [0x01; 16]);
    }

    #[test]
    fn endorsement_without_reason() {
        let builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            99,
            1000,
            2000,
            [0x01; 16],
        );
        let record = builder.build(alloc::vec![0xDE, 0xAD]);
        assert!(record.reason.is_none());
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = EndorsementBuilder::new(
            test_endorser(),
            test_endorsee(),
            42,
            1000,
            2000,
            [0x01; 16],
        );
        builder.reason(alloc::string::String::from("Verified"));

        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = EndorsementRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored, record);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            EndorsementRecord::deserialize(&[]),
            Err(ProfileError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            EndorsementRecord::deserialize(&[0xFF]),
            Err(ProfileError::DeserializeError("unsupported format version"))
        ));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-profile`
Expected: FAIL — types not defined

- [ ] **Step 3: Implement EndorsementRecord + EndorsementBuilder**

```rust
use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::ProfileError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of an endorsement record.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    endorser: IdentityRef,
    endorsee: IdentityRef,
    type_id: u32,
    reason: Option<String>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// A signed endorsement from one identity about another.
///
/// Endorsements are attestations of fact ("I verified their credential"),
/// not expressions of trust. They are signed by the endorser and
/// published at `harmony/endorsement/{endorser_hex}/{endorsee_hex}`.
///
/// # Construction
///
/// Produce records via [`EndorsementBuilder`]; direct struct construction
/// bypasses validity checks.
///
/// Verified by [`verify_endorsement()`](crate::verify::verify_endorsement).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EndorsementRecord {
    pub endorser: IdentityRef,
    pub endorsee: IdentityRef,
    /// Endorsement category. Interpreted as a Q8 page address
    /// (2-28-2 format) for content-addressed type definitions.
    pub type_id: u32,
    pub reason: Option<String>,
    pub published_at: u64,
    pub expires_at: u64,
    /// Must be cryptographically random and unique per publication.
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl EndorsementRecord {
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    pub fn serialize(&self) -> Result<Vec<u8>, ProfileError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| ProfileError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, ProfileError> {
        if data.is_empty() {
            return Err(ProfileError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ProfileError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ProfileError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a signed endorsement record.
pub struct EndorsementBuilder {
    endorser: IdentityRef,
    endorsee: IdentityRef,
    type_id: u32,
    reason: Option<String>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl EndorsementBuilder {
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        endorser: IdentityRef,
        endorsee: IdentityRef,
        type_id: u32,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self {
        assert!(
            expires_at > published_at,
            "expires_at ({expires_at}) must be > published_at ({published_at})"
        );
        Self {
            endorser,
            endorsee,
            type_id,
            reason: None,
            published_at,
            expires_at,
            nonce,
        }
    }

    pub fn reason(&mut self, reason: String) -> &mut Self {
        self.reason = Some(reason);
        self
    }

    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    pub fn build(self, signature: Vec<u8>) -> EndorsementRecord {
        EndorsementRecord {
            endorser: self.endorser,
            endorsee: self.endorsee,
            type_id: self.type_id,
            reason: self.reason,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
            signature,
        }
    }
}
```

- [ ] **Step 4: Update `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod endorsement;
pub mod error;
pub mod profile;

pub use endorsement::{EndorsementBuilder, EndorsementRecord};
pub use error::ProfileError;
pub use profile::{ProfileBuilder, ProfileRecord};
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-profile`
Expected: all 14 tests pass (7 profile + 7 endorsement)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-profile/src/endorsement.rs crates/harmony-profile/src/lib.rs
git commit -m "feat(profile): add EndorsementRecord and EndorsementBuilder"
```

---

### Task 4: Verification (ProfileKeyResolver, verify_profile, verify_endorsement)

**Files:**
- Create: `crates/harmony-profile/src/verify.rs`
- Modify: `crates/harmony-profile/src/lib.rs` (add module + re-exports)

**Reference:** `crates/harmony-discovery/src/verify.rs` for the exact verification + signature dispatch pattern.

- [ ] **Step 1: Write tests in `src/verify.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::endorsement::EndorsementBuilder;
    use crate::profile::ProfileBuilder;
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    fn setup_resolver(
        identity: &harmony_identity::Identity,
        identity_ref: &IdentityRef,
    ) -> MemoryKeyResolver {
        let mut resolver = MemoryKeyResolver::new();
        resolver.insert(identity_ref.hash, identity.verifying_key.to_bytes().to_vec());
        resolver
    }

    // ── ML-DSA-65 (primary path) ────────────────────────────────

    #[test]
    fn ml_dsa65_profile_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let id_ref = IdentityRef::from(&pq_id);

                let mut builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
                builder.display_name(alloc::string::String::from("PQ Alice"));

                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(sig.as_bytes().to_vec());

                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(id_ref.hash, sign_pk.as_bytes());

                assert!(verify_profile(&record, 1500, &resolver).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    #[test]
    fn ml_dsa65_endorsement_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);
                let pq_id = harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let endorser_ref = IdentityRef::from(&pq_id);
                let endorsee_ref = IdentityRef::new([0xBB; 16], CryptoSuite::MlDsa65);

                let builder = EndorsementBuilder::new(
                    endorser_ref, endorsee_ref, 42, 1000, 2000, [0x01; 16],
                );
                let payload = builder.signable_payload();
                let sig = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(sig.as_bytes().to_vec());

                let mut resolver = MemoryKeyResolver::new();
                resolver.insert(endorser_ref.hash, sign_pk.as_bytes());

                assert!(verify_endorsement(&record, 1500, &resolver).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }

    // ── Ed25519 (compat path) ───────────────────────────────────

    #[test]
    fn ed25519_profile_verification() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let mut builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        builder.display_name(alloc::string::String::from("Ed Alice"));

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert!(verify_profile(&record, 1500, &resolver).is_ok());
    }

    #[test]
    fn ed25519_endorsement_verification() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let endorser_ref = IdentityRef::from(identity);
        let endorsee_ref = IdentityRef::new([0xBB; 16], CryptoSuite::Ed25519);

        let builder = EndorsementBuilder::new(
            endorser_ref, endorsee_ref, 42, 1000, 2000, [0x01; 16],
        );
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        let resolver = setup_resolver(identity, &endorser_ref);
        assert!(verify_endorsement(&record, 1500, &resolver).is_ok());
    }

    // ── Error cases ─────────────────────────────────────────────

    #[test]
    fn expired_profile_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 3000, &resolver).unwrap_err(),
            ProfileError::Expired
        );
    }

    #[test]
    fn future_stamped_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, u64::MAX - 100, u64::MAX - 1, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::FutureTimestamp
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let record = builder.build(alloc::vec![0xFF; 64]);

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::SignatureInvalid
        );
    }

    #[test]
    fn unknown_key_rejected() {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let id_ref = IdentityRef::from(private.public_identity());

        let builder = ProfileBuilder::new(id_ref, 1000, 2000, [0x01; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        let resolver = MemoryKeyResolver::new(); // empty
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::KeyNotFound
        );
    }

    #[test]
    fn invalid_record_rejected() {
        // Bypass builder to construct a record with published_at >= expires_at
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let id_ref = IdentityRef::from(identity);

        let record = ProfileRecord {
            identity_ref: id_ref,
            display_name: None,
            status_text: None,
            avatar_cid: None,
            published_at: 2000,
            expires_at: 1000, // invalid: expires before published
            nonce: [0x01; 16],
            signature: alloc::vec![0xDE, 0xAD],
        };

        let resolver = setup_resolver(identity, &id_ref);
        assert_eq!(
            verify_profile(&record, 1500, &resolver).unwrap_err(),
            ProfileError::InvalidRecord
        );
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-profile`
Expected: FAIL — verify functions not defined

- [ ] **Step 3: Implement verification**

```rust
use alloc::vec::Vec;
use harmony_identity::{CryptoSuite, IdentityRef};

use crate::endorsement::EndorsementRecord;
use crate::error::ProfileError;
use crate::profile::ProfileRecord;

/// Maximum allowed clock skew for record timestamps.
/// All timestamps in this crate are Unix epoch seconds.
const MAX_CLOCK_SKEW: u64 = 60;

/// Resolve an identity's verifying public key for profile and
/// endorsement verification.
///
/// Returns the raw verifying key bytes:
/// - Ed25519: 32-byte verifying key
/// - ML-DSA-65: 1952-byte signing public key
pub trait ProfileKeyResolver {
    fn resolve(&self, identity: &IdentityRef) -> Option<Vec<u8>>;
}

/// Verify a profile record's signature and time bounds.
pub fn verify_profile(
    record: &ProfileRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError> {
    check_time_bounds(record.published_at, record.expires_at, now)?;

    let key_bytes = keys
        .resolve(&record.identity_ref)
        .ok_or(ProfileError::KeyNotFound)?;

    let payload = record.signable_bytes();
    verify_signature(record.identity_ref.suite, &key_bytes, &payload, &record.signature)
}

/// Verify an endorsement record's signature and time bounds.
pub fn verify_endorsement(
    record: &EndorsementRecord,
    now: u64,
    keys: &impl ProfileKeyResolver,
) -> Result<(), ProfileError> {
    check_time_bounds(record.published_at, record.expires_at, now)?;

    let key_bytes = keys
        .resolve(&record.endorser)
        .ok_or(ProfileError::KeyNotFound)?;

    let payload = record.signable_bytes();
    verify_signature(record.endorser.suite, &key_bytes, &payload, &record.signature)
}

fn check_time_bounds(published_at: u64, expires_at: u64, now: u64) -> Result<(), ProfileError> {
    if published_at >= expires_at {
        return Err(ProfileError::InvalidRecord);
    }
    if now >= expires_at {
        return Err(ProfileError::Expired);
    }
    if published_at > now.saturating_add(MAX_CLOCK_SKEW) {
        return Err(ProfileError::FutureTimestamp);
    }
    Ok(())
}

/// Dispatch signature verification based on CryptoSuite.
// TODO(v2): MlDsa65Rotatable needs rotation-aware verification via
// KEL chain walk. See harmony-discovery verify.rs for the same note.
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), ProfileError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| ProfileError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| ProfileError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| ProfileError::SignatureInvalid)
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
impl ProfileKeyResolver for MemoryKeyResolver {
    fn resolve(&self, identity: &IdentityRef) -> Option<Vec<u8>> {
        self.keys.get(&identity.hash).cloned()
    }
}
```

- [ ] **Step 4: Update `src/lib.rs` with final re-exports**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod endorsement;
pub mod error;
pub mod profile;
pub mod verify;

pub use endorsement::{EndorsementBuilder, EndorsementRecord};
pub use error::ProfileError;
pub use profile::{ProfileBuilder, ProfileRecord};
pub use verify::{verify_endorsement, verify_profile, ProfileKeyResolver};

#[cfg(any(test, feature = "test-utils"))]
pub use verify::MemoryKeyResolver;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-profile`
Expected: all 24 tests pass (7 profile + 7 endorsement + 10 verify)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-profile/src/verify.rs crates/harmony-profile/src/lib.rs
git commit -m "feat(profile): add verification with ML-DSA-65 and Ed25519 support"
```

---

### Task 5: Zenoh Endorsement Key Expression Patterns

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs` (add `pub mod endorsement`)

**Reference:** Existing `identity` module in namespace.rs for pattern.

- [ ] **Step 1: Write tests**

Add to the existing `#[cfg(test)] mod tests` block in `namespace.rs`:

```rust
    // ── Endorsement ──

    #[test]
    fn endorsement_key_format() {
        let key = endorsement::key(
            "aa00bb11cc22dd33ee44ff5566778899",
            "1122334455667788aabbccddeeff0011",
        );
        assert_eq!(
            key,
            "harmony/endorsement/aa00bb11cc22dd33ee44ff5566778899/1122334455667788aabbccddeeff0011"
        );
    }

    #[test]
    fn endorsement_by_endorser_format() {
        let pattern = endorsement::by_endorser("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            pattern,
            "harmony/endorsement/aa00bb11cc22dd33ee44ff5566778899/*"
        );
    }

    #[test]
    fn endorsement_of_endorsee_format() {
        let pattern = endorsement::of_endorsee("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            pattern,
            "harmony/endorsement/*/aa00bb11cc22dd33ee44ff5566778899"
        );
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh endorsement`
Expected: FAIL — `endorsement` module not defined

- [ ] **Step 3: Implement the endorsement namespace module**

Add after the existing `identity` module in `namespace.rs`. Also add `endorsement::PREFIX` to the existing `all_prefixes_start_with_root` consistency test's `prefixes` array:

```rust
pub mod endorsement {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/endorsement";

    /// All endorsements by a specific endorser.
    ///
    /// `endorser_hex` must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn by_endorser(endorser_hex: &str) -> String {
        format!("{PREFIX}/{endorser_hex}/*")
    }

    /// All endorsements of a specific endorsee.
    ///
    /// `endorsee_hex` must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn of_endorsee(endorsee_hex: &str) -> String {
        format!("{PREFIX}/*/{endorsee_hex}")
    }

    /// Key expression for a specific endorsement.
    ///
    /// Both parameters must be the 32-character lowercase hex encoding
    /// of the 16-byte `IdentityHash`.
    pub fn key(endorser_hex: &str, endorsee_hex: &str) -> String {
        format!("{PREFIX}/{endorser_hex}/{endorsee_hex}")
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-zenoh`
Expected: all existing + 3 new tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add endorsement namespace key expression patterns"
```

---

### Task 6: Final Verification

**Files:** None new — verification pass only.

- [ ] **Step 1: Run all profile tests**

Run: `cargo test -p harmony-profile`
Expected: all 23 tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-profile`
Expected: no warnings

- [ ] **Step 3: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 4: Commit any fixes**

If clippy or workspace tests reveal issues, fix and commit.

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Crate scaffold + ProfileError | 0 (compile check) |
| 2 | ProfileRecord + ProfileBuilder | 7 |
| 3 | EndorsementRecord + EndorsementBuilder | 7 |
| 4 | Verification (ML-DSA-65 + Ed25519) | 10 |
| 5 | Zenoh endorsement key expressions | 3 |
| 6 | Final verification | 0 (verification pass) |
| **Total** | | **27** |

# Unified Identity Reference Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote `CryptoSuite` to crate-level and add `IdentityRef` (17-byte identity reference with crypto suite tag) to `harmony-identity`.

**Architecture:** Modifications to the existing `harmony-identity` crate. New files `crypto_suite.rs` and `identity_ref.rs`. UCAN's local `CryptoSuite` definition replaced with an import from the new module. No downstream crate changes.

**Tech Stack:** Rust 1.75+, no_std + alloc, serde, postcard (dev), TDD

**Spec:** `docs/plans/2026-03-19-unified-identity-ref-design.md`

---

## File Structure

```
crates/harmony-identity/Cargo.toml            — add serde + postcard deps (modify)
crates/harmony-identity/src/crypto_suite.rs    — CryptoSuite enum (create)
crates/harmony-identity/src/identity_ref.rs    — IdentityRef struct (create)
crates/harmony-identity/src/ucan.rs            — remove CryptoSuite def, add import (modify)
crates/harmony-identity/src/lib.rs             — add modules, update re-exports (modify)
```

---

### Task 1: Add serde dependency and create `CryptoSuite` module

**Files:**
- Modify: `crates/harmony-identity/Cargo.toml`
- Create: `crates/harmony-identity/src/crypto_suite.rs`
- Modify: `crates/harmony-identity/src/lib.rs`

- [ ] **Step 1: Add serde and postcard dependencies to Cargo.toml**

Add `serde` to `[dependencies]` and `postcard` to `[dev-dependencies]`. Add `serde/std` to the `std` feature:

In `[dependencies]`, after `thiserror`, add:
```toml
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
```

In `[features]` `std` list, add:
```toml
    "serde/std",
```

In `[dev-dependencies]`, add:
```toml
postcard = { workspace = true }
```

- [ ] **Step 2: Create crypto_suite.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ed25519_is_not_post_quantum() {
        assert!(!CryptoSuite::Ed25519.is_post_quantum());
    }

    #[test]
    fn ml_dsa65_is_post_quantum() {
        assert!(CryptoSuite::MlDsa65.is_post_quantum());
    }

    #[test]
    fn signing_multicodec_values() {
        assert_eq!(CryptoSuite::Ed25519.signing_multicodec(), 0x00ed);
        assert_eq!(CryptoSuite::MlDsa65.signing_multicodec(), 0x1211);
    }

    #[test]
    fn encryption_multicodec_values() {
        assert_eq!(CryptoSuite::Ed25519.encryption_multicodec(), 0x00ec);
        assert_eq!(CryptoSuite::MlDsa65.encryption_multicodec(), 0x120c);
    }

    #[test]
    fn from_signing_multicodec_round_trip() {
        assert_eq!(
            CryptoSuite::from_signing_multicodec(0x00ed),
            Some(CryptoSuite::Ed25519)
        );
        assert_eq!(
            CryptoSuite::from_signing_multicodec(0x1211),
            Some(CryptoSuite::MlDsa65)
        );
        assert_eq!(CryptoSuite::from_signing_multicodec(0xFFFF), None);
    }

    #[test]
    fn from_encryption_multicodec_round_trip() {
        assert_eq!(
            CryptoSuite::from_encryption_multicodec(0x00ec),
            Some(CryptoSuite::Ed25519)
        );
        assert_eq!(
            CryptoSuite::from_encryption_multicodec(0x120c),
            Some(CryptoSuite::MlDsa65)
        );
        assert_eq!(CryptoSuite::from_encryption_multicodec(0x0000), None);
    }

    #[test]
    fn from_byte_round_trip() {
        assert_eq!(CryptoSuite::from_byte(0x00), Some(CryptoSuite::Ed25519));
        assert_eq!(CryptoSuite::from_byte(0x01), Some(CryptoSuite::MlDsa65));
        assert_eq!(CryptoSuite::from_byte(0x02), None);
        assert_eq!(CryptoSuite::from_byte(0xFF), None);
    }

    #[test]
    fn wire_discriminant_values() {
        assert_eq!(CryptoSuite::Ed25519 as u8, 0x00);
        assert_eq!(CryptoSuite::MlDsa65 as u8, 0x01);
    }

    #[test]
    fn serde_round_trip() {
        let suite = CryptoSuite::MlDsa65;
        let bytes = postcard::to_allocvec(&suite).unwrap();
        let decoded: CryptoSuite = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, suite);
    }
}
```

Implementation:

```rust
use serde::{Deserialize, Serialize};

/// The cryptographic algorithm suite backing an identity.
///
/// Discriminant values match the UCAN wire format (first byte of token).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum CryptoSuite {
    /// Ed25519 signing + X25519 encryption (Reticulum-compatible).
    /// Backward-compatibility layer — NOT post-quantum secure.
    Ed25519 = 0x00,
    /// ML-DSA-65 signing + ML-KEM-768 encryption (NIST FIPS 203/204).
    /// Harmony-native, post-quantum secure.
    MlDsa65 = 0x01,
}

impl CryptoSuite {
    /// Whether this suite is post-quantum secure.
    pub fn is_post_quantum(self) -> bool {
        matches!(self, Self::MlDsa65)
    }

    /// Multicodec identifier for the signing algorithm.
    /// Ed25519 = 0x00ed, ML-DSA-65 = 0x1211 (draft).
    pub fn signing_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ed,
            Self::MlDsa65 => 0x1211,
        }
    }

    /// Multicodec identifier for the encryption/KEM algorithm.
    /// X25519 = 0x00ec, ML-KEM-768 = 0x120c (draft).
    pub fn encryption_multicodec(self) -> u16 {
        match self {
            Self::Ed25519 => 0x00ec,
            Self::MlDsa65 => 0x120c,
        }
    }

    /// Construct from a signing multicodec identifier.
    pub fn from_signing_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ed => Some(Self::Ed25519),
            0x1211 => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from an encryption multicodec identifier.
    pub fn from_encryption_multicodec(code: u16) -> Option<Self> {
        match code {
            0x00ec => Some(Self::Ed25519),
            0x120c => Some(Self::MlDsa65),
            _ => None,
        }
    }

    /// Construct from the wire discriminant byte (0x00 or 0x01).
    pub fn from_byte(byte: u8) -> Option<Self> {
        match byte {
            0x00 => Some(Self::Ed25519),
            0x01 => Some(Self::MlDsa65),
            _ => None,
        }
    }
}
```

- [ ] **Step 3: Add module to lib.rs (temporarily, before UCAN migration)**

Add `pub mod crypto_suite;` to lib.rs BEFORE the ucan module, but do NOT change re-exports yet (UCAN still has its own CryptoSuite definition, which would conflict). Just add the module:

```rust
pub mod crypto_suite;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-identity`
Expected: 9 new CryptoSuite tests pass, all existing tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-identity/
git commit -m "feat(identity): add CryptoSuite module with multicodec conversion"
```

---

### Task 2: Migrate UCAN to use the new CryptoSuite

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`
- Modify: `crates/harmony-identity/src/lib.rs`

- [ ] **Step 1: Remove CryptoSuite from ucan.rs, add import**

In `crates/harmony-identity/src/ucan.rs`:
- Remove lines 24-36 (the `CryptoSuite` enum definition and its doc comments)
- Add at the top of the file (after any existing `use` statements):
```rust
use crate::crypto_suite::CryptoSuite;
```

- [ ] **Step 2: Update lib.rs re-exports**

Change the UCAN re-export line from:
```rust
pub use ucan::{
    verify_revocation, verify_token, CapabilityType, CryptoSuite, IdentityResolver, PqUcanToken,
    ProofResolver, Revocation, RevocationSet, UcanError, UcanToken,
};
```
to:
```rust
pub use crypto_suite::CryptoSuite;
pub use ucan::{
    verify_revocation, verify_token, CapabilityType, IdentityResolver, PqUcanToken,
    ProofResolver, Revocation, RevocationSet, UcanError, UcanToken,
};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-identity`
Expected: ALL existing tests pass unchanged (101+ identity tests + 9 new CryptoSuite tests). The UCAN tests use `CryptoSuite::MlDsa65 as u8` which now resolves via the import.

- [ ] **Step 4: Run full workspace**

Run: `cargo test --workspace`
Expected: All workspace tests pass. No downstream crate references `CryptoSuite` directly.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs crates/harmony-identity/src/lib.rs
git commit -m "refactor(identity): migrate UCAN to use crate-level CryptoSuite"
```

---

### Task 3: Add `IdentityRef` type

**Files:**
- Create: `crates/harmony-identity/src/identity_ref.rs`
- Modify: `crates/harmony-identity/src/lib.rs`

- [ ] **Step 1: Create identity_ref.rs with tests first**

Tests:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto_suite::CryptoSuite;

    #[test]
    fn from_classical_identity() {
        let id = crate::identity::Identity::derive_for_test([0xAA; 16]);
        let id_ref = IdentityRef::from(&id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::Ed25519);
        assert!(!id_ref.is_post_quantum());
    }

    #[test]
    fn from_pq_identity() {
        let id = crate::pq_identity::PqIdentity::derive_for_test([0xBB; 16]);
        let id_ref = IdentityRef::from(&id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::MlDsa65);
        assert!(id_ref.is_post_quantum());
    }

    #[test]
    fn new_constructor() {
        let id_ref = IdentityRef::new([0xCC; 16], CryptoSuite::MlDsa65);
        assert_eq!(id_ref.hash, [0xCC; 16]);
        assert!(id_ref.is_post_quantum());
    }

    #[test]
    fn equality_same_hash_same_suite() {
        let a = IdentityRef::new([0xDD; 16], CryptoSuite::Ed25519);
        let b = IdentityRef::new([0xDD; 16], CryptoSuite::Ed25519);
        assert_eq!(a, b);
    }

    #[test]
    fn inequality_same_hash_different_suite() {
        let a = IdentityRef::new([0xEE; 16], CryptoSuite::Ed25519);
        let b = IdentityRef::new([0xEE; 16], CryptoSuite::MlDsa65);
        assert_ne!(a, b);
    }

    #[test]
    fn inequality_different_hash_same_suite() {
        let a = IdentityRef::new([0x11; 16], CryptoSuite::MlDsa65);
        let b = IdentityRef::new([0x22; 16], CryptoSuite::MlDsa65);
        assert_ne!(a, b);
    }

    #[test]
    fn serde_round_trip() {
        let id_ref = IdentityRef::new([0xFF; 16], CryptoSuite::MlDsa65);
        let bytes = postcard::to_allocvec(&id_ref).unwrap();
        let decoded: IdentityRef = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, id_ref);
    }
}
```

**IMPORTANT NOTE for the implementer:** The tests above use `Identity::derive_for_test` and `PqIdentity::derive_for_test` — these helper methods may not exist. If they don't, construct `IdentityRef` directly via `IdentityRef::new()` with known hashes instead. Replace the `from_classical_identity` and `from_pq_identity` tests with:

```rust
    #[test]
    fn from_classical_identity() {
        // Use a real keypair to test From<&Identity>
        use rand::rngs::OsRng;
        let private = crate::identity::PrivateIdentity::generate(&mut OsRng);
        let id = private.public_identity();
        let id_ref = IdentityRef::from(id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::Ed25519);
        assert!(!id_ref.is_post_quantum());
    }

    #[test]
    fn from_pq_identity() {
        use rand::rngs::OsRng;
        let private = crate::pq_identity::PqPrivateIdentity::generate(&mut OsRng);
        let id = private.public_identity();
        let id_ref = IdentityRef::from(id);
        assert_eq!(id_ref.hash, id.address_hash);
        assert_eq!(id_ref.suite, CryptoSuite::MlDsa65);
        assert!(id_ref.is_post_quantum());
    }
```

Implementation:

```rust
use serde::{Deserialize, Serialize};

use crate::crypto_suite::CryptoSuite;
use crate::identity::{Identity, IdentityHash};
use crate::pq_identity::PqIdentity;

/// A lightweight reference to an identity: address hash + crypto suite.
///
/// 17 bytes total. Use this when you need to know "which identity" and
/// "what kind" without carrying full public key material.
///
/// Construct from concrete identity types via `From` impls, or directly
/// via `IdentityRef::new()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IdentityRef {
    /// The 128-bit address hash: SHA256(pub_keys)[:16].
    pub hash: IdentityHash,
    /// The cryptographic suite backing this identity.
    pub suite: CryptoSuite,
}

impl IdentityRef {
    pub fn new(hash: IdentityHash, suite: CryptoSuite) -> Self {
        Self { hash, suite }
    }

    /// Whether this identity uses post-quantum cryptography.
    pub fn is_post_quantum(&self) -> bool {
        self.suite.is_post_quantum()
    }
}

impl From<&Identity> for IdentityRef {
    fn from(id: &Identity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::Ed25519,
        }
    }
}

impl From<&PqIdentity> for IdentityRef {
    fn from(id: &PqIdentity) -> Self {
        Self {
            hash: id.address_hash,
            suite: CryptoSuite::MlDsa65,
        }
    }
}
```

- [ ] **Step 2: Update lib.rs**

Add module and re-export:
```rust
pub mod identity_ref;
pub use identity_ref::IdentityRef;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-identity`
Expected: All tests pass (101+ existing + 9 CryptoSuite + 7 IdentityRef = ~117+).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-identity/
git commit -m "feat(identity): add IdentityRef for suite-aware identity references"
```

---

### Task 4: Final quality gate

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: All tests pass.

- [ ] **Step 2: Format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.

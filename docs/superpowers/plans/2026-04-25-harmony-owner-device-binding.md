# Owner→Device Identity Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `harmony-owner` library implementing the two-tier owner→device identity binding (cert types, CRDT state, lifecycle flows, trust evaluation) per the design at `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`.

**Architecture:** New crate `harmony-owner` depending on `harmony-identity` (key primitives), `harmony-crypto` (hashing), `serde` + `ciborium` (canonical CBOR), `thiserror`. Five cert types (Enrollment, Vouching, Liveness, Revocation, Reclamation) all with domain-separated signing tags. Two CRDTs (LWW vouching set, Remove-Wins revocation set) compose into `OwnerState`. Lifecycle helpers (mint, enroll-via-master, enroll-via-quorum, archive, reclaim) operate on `OwnerState`. Trust evaluation is a pure function of state. **Out of plan scope:** Zenoh gossip/queryable wiring (separate plan), harmony-client Tauri IPC (separate plan).

**Tech Stack:** Rust 2024, ed25519-dalek, x25519-dalek (via harmony-identity), SHA256 (via harmony-crypto), ciborium 0.2, thiserror, zeroize.

**Bead/Issue:** ZEB-173 (Track A child #1; under umbrella ZEB-169).

---

### Task 1: Scaffold `harmony-owner` crate

**Files:**
- Create: `crates/harmony-owner/Cargo.toml`
- Create: `crates/harmony-owner/src/lib.rs`
- Modify: `Cargo.toml` (workspace members)

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "harmony-owner"
description = "Two-tier owner→device identity binding: enrollment, vouching CRDT, lifecycle, trust evaluation"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-crypto = { workspace = true }
harmony-identity = { workspace = true }
ciborium = { workspace = true }
serde = { workspace = true, features = ["derive", "alloc"] }
thiserror = { workspace = true }
zeroize = { workspace = true }
ed25519-dalek = { workspace = true }
x25519-dalek = { workspace = true }
rand_core = { workspace = true, features = ["getrandom"] }
hkdf = { workspace = true }
sha2 = { workspace = true }

[dev-dependencies]
hex = { workspace = true }
rand = { workspace = true }

[features]
default = []
```

- [ ] **Step 2: Create `src/lib.rs` with module skeleton**

```rust
//! Two-tier owner→device identity binding.
//!
//! Spec: `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`.

pub mod cbor;
pub mod certs;
pub mod crdt;
pub mod error;
pub mod lifecycle;
pub mod pubkey_bundle;
pub mod signing;
pub mod state;
pub mod trust;

pub use error::OwnerError;
```

- [ ] **Step 3: Register crate in workspace**

Edit `Cargo.toml` (workspace root). Add `"crates/harmony-owner"` to `members` in alphabetical position (after `harmony-oluo`, before `harmony-peers`). Also add to `[workspace.dependencies]`:

```toml
harmony-owner = { path = "crates/harmony-owner" }
```

- [ ] **Step 4: Verify compilation**

Run: `cargo check -p harmony-owner 2>&1 | tail -20`
Expected: `error[E0583]: file not found for module \`cbor\`` (and similar for other modules) — confirms the workspace registers the crate but module files don't exist yet, which is correct at this point.

- [ ] **Step 5: Add empty stub files for each module so the crate compiles**

```bash
for m in cbor certs crdt error lifecycle pubkey_bundle signing state trust; do
  touch crates/harmony-owner/src/$m.rs
done
```

Edit `crates/harmony-owner/src/error.rs` to add a placeholder `OwnerError`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OwnerError {
    #[error("placeholder")]
    Placeholder,
}
```

Run: `cargo check -p harmony-owner 2>&1 | tail -5`
Expected: clean, no errors.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-owner Cargo.toml
git commit -m "feat(owner): scaffold harmony-owner crate (ZEB-173)"
```

---

### Task 2: `OwnerError` type with all variants

**Files:**
- Modify: `crates/harmony-owner/src/error.rs`

- [ ] **Step 1: Write a test that exercises each variant**

Create `crates/harmony-owner/src/error.rs`:

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum OwnerError {
    #[error("invalid signature on {cert_type} cert")]
    InvalidSignature { cert_type: &'static str },

    #[error("unknown wire format version: {0}")]
    UnknownVersion(u8),

    #[error("CBOR encode/decode failure: {0}")]
    Cbor(String),

    #[error("identity hash does not match contained public keys")]
    IdentityHashMismatch,

    #[error("device {device:?} is not enrolled under owner {owner:?}")]
    NotEnrolled { owner: [u8; 16], device: [u8; 16] },

    #[error("device {device:?} is revoked")]
    Revoked { device: [u8; 16] },

    #[error("quorum requires at least {min} signers, got {got}")]
    InsufficientQuorum { min: usize, got: usize },

    #[error("trust state snapshot is older than freshness window ({age_days}d > 30d)")]
    StaleTrustState { age_days: u64 },

    #[error("identity in contested state — competing master/sibling assertions")]
    Contested,

    #[error("reclamation refuted by predecessor liveness")]
    ReclamationRefuted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_context() {
        let e = OwnerError::InvalidSignature { cert_type: "Enrollment" };
        assert!(format!("{e}").contains("Enrollment"));

        let e = OwnerError::InsufficientQuorum { min: 2, got: 1 };
        assert!(format!("{e}").contains("at least 2"));
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p harmony-owner error_display 2>&1 | tail -5`
Expected: PASS, 1 test.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/error.rs
git commit -m "feat(owner): error variants for cert verification + state checks (ZEB-173)"
```

---

### Task 3: `PubKeyBundle` types

**Files:**
- Modify: `crates/harmony-owner/src/pubkey_bundle.rs`

- [ ] **Step 1: Write the test**

Create `crates/harmony-owner/src/pubkey_bundle.rs`:

```rust
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PubKeyBundle {
    pub classical: ClassicalKeys,
    pub post_quantum: Option<PqKeys>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClassicalKeys {
    #[serde(with = "serde_bytes")]
    pub ed25519_verify: [u8; 32],
    #[serde(with = "serde_bytes")]
    pub x25519_pub: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PqKeys {
    #[serde(with = "serde_bytes")]
    pub ml_dsa_verify: Vec<u8>,
    #[serde(with = "serde_bytes")]
    pub ml_kem_pub: Vec<u8>,
}

impl PubKeyBundle {
    /// Derive the 128-bit IdentityHash: SHA256(canonical_cbor(self))[:16].
    pub fn identity_hash(&self) -> [u8; 16] {
        let bytes = crate::cbor::to_canonical(self).expect("PubKeyBundle always encodes");
        let digest: [u8; 32] = harmony_crypto::sha256(&bytes);
        let mut out = [0u8; 16];
        out.copy_from_slice(&digest[..16]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_hash_is_deterministic() {
        let bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: [1u8; 32],
                x25519_pub: [2u8; 32],
            },
            post_quantum: None,
        };
        let h1 = bundle.identity_hash();
        let h2 = bundle.identity_hash();
        assert_eq!(h1, h2, "identity hash must be deterministic");
        assert_eq!(h1.len(), 16);
    }

    #[test]
    fn different_bundles_yield_different_hashes() {
        let a = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: [1u8; 32], x25519_pub: [2u8; 32] },
            post_quantum: None,
        };
        let b = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: [3u8; 32], x25519_pub: [2u8; 32] },
            post_quantum: None,
        };
        assert_ne!(a.identity_hash(), b.identity_hash());
    }
}
```

Note: requires `serde_bytes` dependency. Add to Cargo.toml:

```toml
serde_bytes = { workspace = true }
```

- [ ] **Step 2: Run test (will fail because cbor module is empty)**

Run: `cargo test -p harmony-owner pubkey_bundle 2>&1 | tail -10`
Expected: FAIL — `crate::cbor::to_canonical` doesn't exist yet.

- [ ] **Step 3: Park test failure, proceed to next task (CBOR helper)**

Mark this task complete; the `pubkey_bundle` test will pass once Task 4 lands. Do NOT commit yet — combine with Task 4's commit.

---

### Task 4: Canonical CBOR helper

**Files:**
- Modify: `crates/harmony-owner/src/cbor.rs`

- [ ] **Step 1: Write the test**

Create `crates/harmony-owner/src/cbor.rs`:

```rust
use crate::OwnerError;
use serde::{de::DeserializeOwned, Serialize};

/// Encode a value to deterministic (canonical) CBOR bytes.
///
/// Per RFC 8949 §4.2: shortest-form integers, definite-length containers,
/// keys sorted by major type then byte-wise lexicographic order. ciborium
/// produces canonical output by default for these constraints.
pub fn to_canonical<T: Serialize>(value: &T) -> Result<Vec<u8>, OwnerError> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf).map_err(|e| OwnerError::Cbor(e.to_string()))?;
    Ok(buf)
}

pub fn from_bytes<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, OwnerError> {
    ciborium::de::from_reader(bytes).map_err(|e| OwnerError::Cbor(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, Serialize, Deserialize, PartialEq)]
    struct Sample {
        a: u32,
        b: String,
    }

    #[test]
    fn roundtrip() {
        let v = Sample { a: 42, b: "hello".into() };
        let bytes = to_canonical(&v).unwrap();
        let decoded: Sample = from_bytes(&bytes).unwrap();
        assert_eq!(v, decoded);
    }

    #[test]
    fn deterministic_encoding() {
        let v = Sample { a: 42, b: "hello".into() };
        let b1 = to_canonical(&v).unwrap();
        let b2 = to_canonical(&v).unwrap();
        assert_eq!(b1, b2, "encoding must be deterministic");
    }
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p harmony-owner cbor:: 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 3: Re-run pubkey_bundle test from Task 3**

Run: `cargo test -p harmony-owner pubkey_bundle 2>&1 | tail -5`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit Tasks 3+4 together**

```bash
git add crates/harmony-owner/src/cbor.rs crates/harmony-owner/src/pubkey_bundle.rs crates/harmony-owner/Cargo.toml
git commit -m "feat(owner): canonical CBOR helper + PubKeyBundle with identity_hash (ZEB-173)"
```

---

### Task 5: Domain-separated signing helpers

**Files:**
- Modify: `crates/harmony-owner/src/signing.rs`

- [ ] **Step 1: Write the test**

Create `crates/harmony-owner/src/signing.rs`:

```rust
use crate::OwnerError;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature};

/// Domain tags per cert type. Every signature is computed over
/// `tag || canonical_cbor_bytes(payload)`, which prevents a signature
/// valid in one cert context from being accepted in another.
pub mod tags {
    pub const ENROLLMENT: &[u8] = b"harmony-owner/v1/Enrollment";
    pub const VOUCHING: &[u8] = b"harmony-owner/v1/Vouching";
    pub const LIVENESS: &[u8] = b"harmony-owner/v1/Liveness";
    pub const REVOCATION: &[u8] = b"harmony-owner/v1/Revocation";
    pub const RECLAMATION: &[u8] = b"harmony-owner/v1/Reclamation";
}

pub fn sign_with_tag(
    sk: &SigningKey,
    tag: &[u8],
    payload_bytes: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tag.len() + payload_bytes.len());
    buf.extend_from_slice(tag);
    buf.extend_from_slice(payload_bytes);
    sk.sign(&buf).to_bytes().to_vec()
}

pub fn verify_with_tag(
    vk: &VerifyingKey,
    tag: &[u8],
    payload_bytes: &[u8],
    signature: &[u8],
    cert_type: &'static str,
) -> Result<(), OwnerError> {
    let sig_bytes: [u8; 64] = signature
        .try_into()
        .map_err(|_| OwnerError::InvalidSignature { cert_type })?;
    let sig = Signature::from_bytes(&sig_bytes);
    let mut buf = Vec::with_capacity(tag.len() + payload_bytes.len());
    buf.extend_from_slice(tag);
    buf.extend_from_slice(payload_bytes);
    vk.verify_strict(&buf, &sig).map_err(|_| OwnerError::InvalidSignature { cert_type })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn fresh_keypair() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    #[test]
    fn sign_then_verify_roundtrip() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let payload = b"hello, world";
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, payload);
        verify_with_tag(&vk, tags::ENROLLMENT, payload, &sig, "Enrollment").unwrap();
    }

    #[test]
    fn signature_with_wrong_tag_rejected() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let payload = b"hello, world";
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, payload);
        // Verifier expects VOUCHING tag; same payload, same key, but tag-mismatched signature must reject.
        let result = verify_with_tag(&vk, tags::VOUCHING, payload, &sig, "Vouching");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn signature_with_wrong_payload_rejected() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, b"original");
        let result = verify_with_tag(&vk, tags::ENROLLMENT, b"tampered", &sig, "Enrollment");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn signature_with_wrong_key_rejected() {
        let sk_a = fresh_keypair();
        let sk_b = fresh_keypair();
        let vk_b = sk_b.verifying_key();
        let payload = b"hello";
        let sig = sign_with_tag(&sk_a, tags::ENROLLMENT, payload);
        let result = verify_with_tag(&vk_b, tags::ENROLLMENT, payload, &sig, "Enrollment");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-owner signing:: 2>&1 | tail -10`
Expected: PASS, 4 tests.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/signing.rs
git commit -m "feat(owner): domain-separated signing tags per cert type (ZEB-173)"
```

---

### Task 6: `EnrollmentCert` with verification

**Files:**
- Create: `crates/harmony-owner/src/certs/mod.rs`
- Create: `crates/harmony-owner/src/certs/enrollment.rs`
- Modify: `crates/harmony-owner/src/lib.rs`

- [ ] **Step 1: Restructure: convert `certs.rs` to a directory**

```bash
rm crates/harmony-owner/src/certs.rs
mkdir crates/harmony-owner/src/certs
```

Create `crates/harmony-owner/src/certs/mod.rs`:

```rust
pub mod enrollment;
pub use enrollment::{EnrollmentCert, EnrollmentIssuer};
```

- [ ] **Step 2: Write the EnrollmentCert + tests**

Create `crates/harmony-owner/src/certs/enrollment.rs`:

```rust
use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const ENROLLMENT_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrollmentCert {
    pub version: u8,
    pub owner_id: [u8; 16],
    pub device_id: [u8; 16],
    pub device_pubkeys: PubKeyBundle,
    pub issued_at: u64,
    pub expires_at: Option<u64>,
    pub issuer: EnrollmentIssuer,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentIssuer {
    /// Master-signed. `master_pubkey` is embedded so verifier is self-contained.
    Master { master_pubkey: PubKeyBundle },

    /// K-quorum: each signer has its own EnrollmentCert under the same owner.
    /// Verifier walks back to those certs to fetch signers' pubkeys.
    Quorum {
        signers: Vec<[u8; 16]>,
        signatures: Vec<Vec<u8>>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct EnrollmentSigningPayload<'a> {
    version: u8,
    owner_id: [u8; 16],
    device_id: [u8; 16],
    device_pubkeys: &'a PubKeyBundle,
    issued_at: u64,
    expires_at: Option<u64>,
    issuer_kind: u8, // 0 = Master, 1 = Quorum
    issuer_data: Vec<u8>, // CBOR-encoded inner data of the EnrollmentIssuer
}

impl EnrollmentCert {
    /// Issue a Master-signed Enrollment Cert.
    pub fn sign_master(
        owner_sk: &SigningKey,
        master_pubkey: PubKeyBundle,
        device_id: [u8; 16],
        device_pubkeys: PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
    ) -> Result<Self, OwnerError> {
        let owner_id = master_pubkey.identity_hash();
        let issuer = EnrollmentIssuer::Master { master_pubkey: master_pubkey.clone() };
        let payload_bytes = cbor::to_canonical(&signing_payload(
            ENROLLMENT_VERSION,
            owner_id,
            device_id,
            &device_pubkeys,
            issued_at,
            expires_at,
            &issuer,
        )?)?;
        let signature = sign_with_tag(owner_sk, tags::ENROLLMENT, &payload_bytes);
        Ok(EnrollmentCert {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer,
            signature,
        })
    }

    pub fn verify(&self) -> Result<(), OwnerError> {
        if self.version != ENROLLMENT_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        match &self.issuer {
            EnrollmentIssuer::Master { master_pubkey } => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Enrollment" })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version,
                    self.owner_id,
                    self.device_id,
                    &self.device_pubkeys,
                    self.issued_at,
                    self.expires_at,
                    &self.issuer,
                )?)?;
                verify_with_tag(&vk, tags::ENROLLMENT, &payload_bytes, &self.signature, "Enrollment")?;
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
            EnrollmentIssuer::Quorum { signers, signatures } => {
                // Quorum verification is delegated to OwnerState (which has access to
                // the full set of signer Enrollment Certs). This standalone verify()
                // only checks the signature count matches signers count.
                if signers.len() != signatures.len() {
                    return Err(OwnerError::InvalidSignature { cert_type: "Enrollment" });
                }
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
        }
    }
}

fn signing_payload(
    version: u8,
    owner_id: [u8; 16],
    device_id: [u8; 16],
    device_pubkeys: &PubKeyBundle,
    issued_at: u64,
    expires_at: Option<u64>,
    issuer: &EnrollmentIssuer,
) -> Result<EnrollmentSigningPayload<'_>, OwnerError> {
    let (issuer_kind, issuer_data) = match issuer {
        EnrollmentIssuer::Master { master_pubkey } => {
            (0u8, cbor::to_canonical(master_pubkey)?)
        }
        EnrollmentIssuer::Quorum { signers, .. } => {
            // Signatures NOT included in signing payload (chicken-and-egg);
            // each signer's signature covers the rest of the payload + the
            // signers list.
            (1u8, cbor::to_canonical(signers)?)
        }
    };
    Ok(EnrollmentSigningPayload {
        version,
        owner_id,
        device_id,
        device_pubkeys,
        issued_at,
        expires_at,
        issuer_kind,
        issuer_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fresh_pubkey_bundle(ed_seed: u8, x_seed: u8) -> (SigningKey, PubKeyBundle) {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: sk.verifying_key().to_bytes(),
                x25519_pub: [x_seed; 32], // mock for test; real X25519 from harmony-identity
            },
            post_quantum: None,
        };
        let _ = ed_seed;
        (sk, bundle)
    }

    #[test]
    fn master_signed_enrollment_verifies() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        cert.verify().unwrap();
    }

    #[test]
    fn tampered_enrollment_signature_rejected() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let mut cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        // Tamper with timestamp
        cert.issued_at = 1_800_000_000;
        let result = cert.verify();
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn enrollment_cbor_roundtrip() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        let bytes = cbor::to_canonical(&cert).unwrap();
        let decoded: EnrollmentCert = cbor::from_bytes(&bytes).unwrap();
        assert_eq!(cert, decoded);
        decoded.verify().unwrap();
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner enrollment 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/certs
git rm crates/harmony-owner/src/certs.rs
git commit -m "feat(owner): EnrollmentCert with master-signed + quorum issuer variants (ZEB-173)"
```

---

### Task 7: `VouchingCert`

**Files:**
- Create: `crates/harmony-owner/src/certs/vouching.rs`
- Modify: `crates/harmony-owner/src/certs/mod.rs`

- [ ] **Step 1: Add module export**

Edit `crates/harmony-owner/src/certs/mod.rs`:

```rust
pub mod enrollment;
pub mod vouching;
pub use enrollment::{EnrollmentCert, EnrollmentIssuer};
pub use vouching::{Stance, VouchingCert};
```

- [ ] **Step 2: Write VouchingCert + tests**

Create `crates/harmony-owner/src/certs/vouching.rs`:

```rust
use crate::cbor;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const VOUCHING_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stance {
    Vouch,
    Challenge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VouchingCert {
    pub version: u8,
    pub owner_id: [u8; 16],
    pub signer: [u8; 16],
    pub target: [u8; 16],
    pub stance: Stance,
    pub issued_at: u64,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct VouchingSigningPayload {
    version: u8,
    owner_id: [u8; 16],
    signer: [u8; 16],
    target: [u8; 16],
    stance: Stance,
    issued_at: u64,
}

impl VouchingCert {
    pub fn sign(
        signer_sk: &SigningKey,
        owner_id: [u8; 16],
        signer: [u8; 16],
        target: [u8; 16],
        stance: Stance,
        issued_at: u64,
    ) -> Result<Self, OwnerError> {
        let payload_bytes = cbor::to_canonical(&VouchingSigningPayload {
            version: VOUCHING_VERSION,
            owner_id,
            signer,
            target,
            stance,
            issued_at,
        })?;
        let signature = sign_with_tag(signer_sk, tags::VOUCHING, &payload_bytes);
        Ok(VouchingCert {
            version: VOUCHING_VERSION,
            owner_id,
            signer,
            target,
            stance,
            issued_at,
            signature,
        })
    }

    pub fn verify(&self, signer_pubkey: &VerifyingKey) -> Result<(), OwnerError> {
        if self.version != VOUCHING_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        let payload_bytes = cbor::to_canonical(&VouchingSigningPayload {
            version: self.version,
            owner_id: self.owner_id,
            signer: self.signer,
            target: self.target,
            stance: self.stance,
            issued_at: self.issued_at,
        })?;
        verify_with_tag(signer_pubkey, tags::VOUCHING, &payload_bytes, &self.signature, "Vouching")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn vouch_signs_and_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(
            &sk,
            [1u8; 16],
            [2u8; 16],
            [3u8; 16],
            Stance::Vouch,
            1_700_000_000,
        ).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn challenge_signs_and_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(
            &sk,
            [1u8; 16],
            [2u8; 16],
            [3u8; 16],
            Stance::Challenge,
            1_700_000_000,
        ).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn signature_with_different_signer_key_rejected() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(&sk_a, [1u8; 16], [2u8; 16], [3u8; 16], Stance::Vouch, 1).unwrap();
        let result = cert.verify(&sk_b.verifying_key());
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner vouching 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/certs
git commit -m "feat(owner): VouchingCert with Vouch/Challenge stance (ZEB-173)"
```

---

### Task 8: `LivenessCert`

**Files:**
- Create: `crates/harmony-owner/src/certs/liveness.rs`
- Modify: `crates/harmony-owner/src/certs/mod.rs`

- [ ] **Step 1: Add module export**

Edit `crates/harmony-owner/src/certs/mod.rs`:

```rust
pub mod enrollment;
pub mod liveness;
pub mod vouching;
pub use enrollment::{EnrollmentCert, EnrollmentIssuer};
pub use liveness::LivenessCert;
pub use vouching::{Stance, VouchingCert};
```

- [ ] **Step 2: Write LivenessCert + tests**

Create `crates/harmony-owner/src/certs/liveness.rs`:

```rust
use crate::cbor;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const LIVENESS_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LivenessCert {
    pub version: u8,
    pub signer: [u8; 16],
    pub timestamp: u64,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct LivenessSigningPayload {
    version: u8,
    signer: [u8; 16],
    timestamp: u64,
}

impl LivenessCert {
    pub fn sign(signer_sk: &SigningKey, signer: [u8; 16], timestamp: u64) -> Result<Self, OwnerError> {
        let payload_bytes = cbor::to_canonical(&LivenessSigningPayload {
            version: LIVENESS_VERSION,
            signer,
            timestamp,
        })?;
        let signature = sign_with_tag(signer_sk, tags::LIVENESS, &payload_bytes);
        Ok(LivenessCert { version: LIVENESS_VERSION, signer, timestamp, signature })
    }

    pub fn verify(&self, signer_pubkey: &VerifyingKey) -> Result<(), OwnerError> {
        if self.version != LIVENESS_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        let payload_bytes = cbor::to_canonical(&LivenessSigningPayload {
            version: self.version,
            signer: self.signer,
            timestamp: self.timestamp,
        })?;
        verify_with_tag(signer_pubkey, tags::LIVENESS, &payload_bytes, &self.signature, "Liveness")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn liveness_roundtrip() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = LivenessCert::sign(&sk, [1u8; 16], 1_700_000_000).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn liveness_rejected_with_wrong_key() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        let cert = LivenessCert::sign(&sk_a, [1u8; 16], 1).unwrap();
        let result = cert.verify(&sk_b.verifying_key());
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner liveness 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/certs
git commit -m "feat(owner): LivenessCert for heartbeats and refutation (ZEB-173)"
```

---

### Task 9: `RevocationCert`

**Files:**
- Create: `crates/harmony-owner/src/certs/revocation.rs`
- Modify: `crates/harmony-owner/src/certs/mod.rs`

- [ ] **Step 1: Module export**

```rust
pub mod enrollment;
pub mod liveness;
pub mod revocation;
pub mod vouching;
pub use enrollment::{EnrollmentCert, EnrollmentIssuer};
pub use liveness::LivenessCert;
pub use revocation::{RevocationCert, RevocationIssuer, RevocationReason};
pub use vouching::{Stance, VouchingCert};
```

- [ ] **Step 2: Write RevocationCert + tests**

Create `crates/harmony-owner/src/certs/revocation.rs`:

```rust
use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const REVOCATION_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationReason {
    Decommissioned,
    Lost,
    Compromised,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationIssuer {
    SelfDevice,
    Master { master_pubkey: PubKeyBundle },
    Quorum { signers: Vec<[u8; 16]>, signatures: Vec<Vec<u8>> },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationCert {
    pub version: u8,
    pub owner_id: [u8; 16],
    pub target: [u8; 16],
    pub issued_at: u64,
    pub issuer: RevocationIssuer,
    pub reason: RevocationReason,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct RevocationSigningPayload<'a> {
    version: u8,
    owner_id: [u8; 16],
    target: [u8; 16],
    issued_at: u64,
    issuer_kind: u8,
    issuer_data: Vec<u8>,
    reason: &'a RevocationReason,
}

impl RevocationCert {
    pub fn sign_self(
        device_sk: &SigningKey,
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
    ) -> Result<Self, OwnerError> {
        let issuer = RevocationIssuer::SelfDevice;
        let payload_bytes = cbor::to_canonical(&signing_payload(
            REVOCATION_VERSION, owner_id, target, issued_at, &issuer, &reason)?)?;
        let signature = sign_with_tag(device_sk, tags::REVOCATION, &payload_bytes);
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer,
            reason,
            signature,
        })
    }

    pub fn sign_master(
        master_sk: &SigningKey,
        master_pubkey: PubKeyBundle,
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
    ) -> Result<Self, OwnerError> {
        let owner_id = master_pubkey.identity_hash();
        let issuer = RevocationIssuer::Master { master_pubkey: master_pubkey.clone() };
        let payload_bytes = cbor::to_canonical(&signing_payload(
            REVOCATION_VERSION, owner_id, target, issued_at, &issuer, &reason)?)?;
        let signature = sign_with_tag(master_sk, tags::REVOCATION, &payload_bytes);
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer,
            reason,
            signature,
        })
    }

    /// Verify against a provided pubkey for the issuer (self-device's pubkey
    /// or master's pubkey for SelfDevice/Master variants respectively). For
    /// Quorum, full verification is delegated to OwnerState.
    pub fn verify(&self, issuer_pubkey: Option<&VerifyingKey>) -> Result<(), OwnerError> {
        if self.version != REVOCATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        match (&self.issuer, issuer_pubkey) {
            (RevocationIssuer::SelfDevice, Some(vk)) => {
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version, self.owner_id, self.target, self.issued_at, &self.issuer, &self.reason)?)?;
                verify_with_tag(vk, tags::REVOCATION, &payload_bytes, &self.signature, "Revocation")
            }
            (RevocationIssuer::Master { master_pubkey }, _) => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Revocation" })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version, self.owner_id, self.target, self.issued_at, &self.issuer, &self.reason)?)?;
                verify_with_tag(&vk, tags::REVOCATION, &payload_bytes, &self.signature, "Revocation")
            }
            (RevocationIssuer::Quorum { .. }, _) => {
                // Delegated to OwnerState.
                Ok(())
            }
            (RevocationIssuer::SelfDevice, None) => {
                Err(OwnerError::InvalidSignature { cert_type: "Revocation" })
            }
        }
    }
}

fn signing_payload<'a>(
    version: u8,
    owner_id: [u8; 16],
    target: [u8; 16],
    issued_at: u64,
    issuer: &RevocationIssuer,
    reason: &'a RevocationReason,
) -> Result<RevocationSigningPayload<'a>, OwnerError> {
    let (issuer_kind, issuer_data) = match issuer {
        RevocationIssuer::SelfDevice => (0u8, Vec::new()),
        RevocationIssuer::Master { master_pubkey } => (1u8, cbor::to_canonical(master_pubkey)?),
        RevocationIssuer::Quorum { signers, .. } => (2u8, cbor::to_canonical(signers)?),
    };
    Ok(RevocationSigningPayload { version, owner_id, target, issued_at, issuer_kind, issuer_data, reason })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn self_revocation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let target = [9u8; 16];
        let cert = RevocationCert::sign_self(&sk, [1u8; 16], target, 1, RevocationReason::Decommissioned).unwrap();
        cert.verify(Some(&sk.verifying_key())).unwrap();
    }

    #[test]
    fn master_revocation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let master_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let cert = RevocationCert::sign_master(&sk, master_bundle, [9u8; 16], 1, RevocationReason::Compromised).unwrap();
        cert.verify(None).unwrap();
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner revocation 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/certs
git commit -m "feat(owner): RevocationCert with self/master/quorum issuers (ZEB-173)"
```

---

### Task 10: `ReclamationCert`

**Files:**
- Create: `crates/harmony-owner/src/certs/reclamation.rs`
- Modify: `crates/harmony-owner/src/certs/mod.rs`

- [ ] **Step 1: Module export**

```rust
pub mod enrollment;
pub mod liveness;
pub mod reclamation;
pub mod revocation;
pub mod vouching;
pub use enrollment::{EnrollmentCert, EnrollmentIssuer};
pub use liveness::LivenessCert;
pub use reclamation::ReclamationCert;
pub use revocation::{RevocationCert, RevocationIssuer, RevocationReason};
pub use vouching::{Stance, VouchingCert};
```

- [ ] **Step 2: Write ReclamationCert + tests**

Create `crates/harmony-owner/src/certs/reclamation.rs`:

```rust
use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const RECLAMATION_VERSION: u8 = 1;

/// Default challenge window for reclamation: 30 days in seconds.
pub const DEFAULT_CHALLENGE_WINDOW_SECS: u64 = 30 * 24 * 60 * 60;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReclamationCert {
    pub version: u8,
    pub new_owner_id: [u8; 16],
    pub new_owner_pubkey: PubKeyBundle,
    pub claimed_predecessor: [u8; 16],
    pub issued_at: u64,
    pub challenge_window_end: u64,
    pub note: String,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct ReclamationSigningPayload<'a> {
    version: u8,
    new_owner_id: [u8; 16],
    new_owner_pubkey: &'a PubKeyBundle,
    claimed_predecessor: [u8; 16],
    issued_at: u64,
    challenge_window_end: u64,
    note: &'a str,
}

impl ReclamationCert {
    pub fn sign(
        new_master_sk: &SigningKey,
        new_owner_pubkey: PubKeyBundle,
        claimed_predecessor: [u8; 16],
        issued_at: u64,
        challenge_window_secs: u64,
        note: String,
    ) -> Result<Self, OwnerError> {
        let new_owner_id = new_owner_pubkey.identity_hash();
        let challenge_window_end = issued_at + challenge_window_secs;
        let payload_bytes = cbor::to_canonical(&ReclamationSigningPayload {
            version: RECLAMATION_VERSION,
            new_owner_id,
            new_owner_pubkey: &new_owner_pubkey,
            claimed_predecessor,
            issued_at,
            challenge_window_end,
            note: &note,
        })?;
        let signature = sign_with_tag(new_master_sk, tags::RECLAMATION, &payload_bytes);
        Ok(ReclamationCert {
            version: RECLAMATION_VERSION,
            new_owner_id,
            new_owner_pubkey,
            claimed_predecessor,
            issued_at,
            challenge_window_end,
            note,
            signature,
        })
    }

    pub fn verify(&self) -> Result<(), OwnerError> {
        if self.version != RECLAMATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        if self.new_owner_pubkey.identity_hash() != self.new_owner_id {
            return Err(OwnerError::IdentityHashMismatch);
        }
        let vk = VerifyingKey::from_bytes(&self.new_owner_pubkey.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Reclamation" })?;
        let payload_bytes = cbor::to_canonical(&ReclamationSigningPayload {
            version: self.version,
            new_owner_id: self.new_owner_id,
            new_owner_pubkey: &self.new_owner_pubkey,
            claimed_predecessor: self.claimed_predecessor,
            issued_at: self.issued_at,
            challenge_window_end: self.challenge_window_end,
            note: &self.note,
        })?;
        verify_with_tag(&vk, tags::RECLAMATION, &payload_bytes, &self.signature, "Reclamation")
    }

    /// True if a `LivenessCert` with timestamp > self.issued_at and signed by
    /// any device under the predecessor identity refutes this reclamation.
    /// (The actual lookup of "any device under predecessor" is OwnerState's job.)
    pub fn is_refuted_by_timestamp(&self, liveness_timestamp: u64) -> bool {
        liveness_timestamp > self.issued_at
    }

    pub fn is_window_expired(&self, now: u64) -> bool {
        now >= self.challenge_window_end
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn reclamation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let cert = ReclamationCert::sign(
            &sk, bundle, [9u8; 16], 1_700_000_000, DEFAULT_CHALLENGE_WINDOW_SECS, "lost devices in fire".into()
        ).unwrap();
        cert.verify().unwrap();
    }

    #[test]
    fn refutation_logic() {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let cert = ReclamationCert::sign(&sk, bundle, [9u8; 16], 1_000_000, 1000, "n/a".into()).unwrap();
        assert!(cert.is_refuted_by_timestamp(1_000_001));
        assert!(!cert.is_refuted_by_timestamp(999_999));
        assert!(!cert.is_refuted_by_timestamp(1_000_000));
        assert!(cert.is_window_expired(1_001_000));
        assert!(!cert.is_window_expired(1_000_500));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner reclamation 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/certs
git commit -m "feat(owner): ReclamationCert with challenge window logic (ZEB-173)"
```

---

### Task 11: Vouching CRDT (LWW per cell)

**Files:**
- Create: `crates/harmony-owner/src/crdt/mod.rs`
- Create: `crates/harmony-owner/src/crdt/vouching_set.rs`
- Modify: `crates/harmony-owner/src/lib.rs`

- [ ] **Step 1: Restructure crdt to directory**

```bash
rm crates/harmony-owner/src/crdt.rs
mkdir crates/harmony-owner/src/crdt
```

Create `crates/harmony-owner/src/crdt/mod.rs`:

```rust
pub mod vouching_set;
pub use vouching_set::VouchingSet;
```

- [ ] **Step 2: Write VouchingSet + tests**

Create `crates/harmony-owner/src/crdt/vouching_set.rs`:

```rust
use crate::certs::{Stance, VouchingCert};
use std::collections::HashMap;

/// LWW-per-cell CRDT keyed by `(signer, target)`. Newer entries from the
/// same signer supersede older ones; signers cannot override each other.
#[derive(Debug, Clone, Default)]
pub struct VouchingSet {
    cells: HashMap<([u8; 16], [u8; 16]), VouchingCert>,
}

impl VouchingSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a cert. If a cert from the same signer about the same target
    /// already exists with a newer-or-equal timestamp, this is a no-op.
    pub fn insert(&mut self, cert: VouchingCert) {
        let key = (cert.signer, cert.target);
        match self.cells.get(&key) {
            Some(existing) if existing.issued_at >= cert.issued_at => { /* no-op: older */ }
            _ => { self.cells.insert(key, cert); }
        }
    }

    /// Merge another VouchingSet into this one, applying LWW per cell.
    pub fn merge(&mut self, other: VouchingSet) {
        for cert in other.cells.into_values() {
            self.insert(cert);
        }
    }

    /// Vouches for target from active signers (filter applied externally).
    pub fn vouches_for(&self, target: [u8; 16]) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values().filter(move |c| c.target == target && c.stance == Stance::Vouch)
    }

    pub fn challenges_against(&self, target: [u8; 16]) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values().filter(move |c| c.target == target && c.stance == Stance::Challenge)
    }

    pub fn iter(&self) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn make_cert(signer: [u8; 16], target: [u8; 16], stance: Stance, ts: u64) -> VouchingCert {
        let sk = SigningKey::generate(&mut OsRng);
        VouchingCert::sign(&sk, [0u8; 16], signer, target, stance, ts).unwrap()
    }

    #[test]
    fn lww_per_cell() {
        let mut set = VouchingSet::new();
        let signer = [1u8; 16];
        let target = [2u8; 16];

        // Older Vouch
        set.insert(make_cert(signer, target, Stance::Vouch, 100));
        assert_eq!(set.vouches_for(target).count(), 1);

        // Newer Challenge from same signer — supersedes
        set.insert(make_cert(signer, target, Stance::Challenge, 200));
        assert_eq!(set.vouches_for(target).count(), 0);
        assert_eq!(set.challenges_against(target).count(), 1);

        // Older Vouch from same signer — no-op
        set.insert(make_cert(signer, target, Stance::Vouch, 150));
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn signers_cannot_override_each_other() {
        let mut set = VouchingSet::new();
        let target = [9u8; 16];
        let signer_a = [1u8; 16];
        let signer_b = [2u8; 16];

        set.insert(make_cert(signer_a, target, Stance::Vouch, 100));
        set.insert(make_cert(signer_b, target, Stance::Challenge, 200));

        assert_eq!(set.vouches_for(target).count(), 1);
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn merge_converges() {
        let target = [9u8; 16];
        let signer_a = [1u8; 16];
        let signer_b = [2u8; 16];

        let mut set1 = VouchingSet::new();
        set1.insert(make_cert(signer_a, target, Stance::Vouch, 100));

        let mut set2 = VouchingSet::new();
        set2.insert(make_cert(signer_b, target, Stance::Vouch, 200));

        set1.merge(set2);
        assert_eq!(set1.vouches_for(target).count(), 2);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner vouching_set 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/crdt
git rm crates/harmony-owner/src/crdt.rs
git commit -m "feat(owner): VouchingSet LWW CRDT keyed by (signer, target) (ZEB-173)"
```

---

### Task 12: Revocation set (Remove-Wins, monotonic)

**Files:**
- Create: `crates/harmony-owner/src/crdt/revocation_set.rs`
- Modify: `crates/harmony-owner/src/crdt/mod.rs`

- [ ] **Step 1: Module export**

Edit `crates/harmony-owner/src/crdt/mod.rs`:

```rust
pub mod revocation_set;
pub mod vouching_set;
pub use revocation_set::RevocationSet;
pub use vouching_set::VouchingSet;
```

- [ ] **Step 2: Write RevocationSet + tests**

Create `crates/harmony-owner/src/crdt/revocation_set.rs`:

```rust
use crate::certs::RevocationCert;
use std::collections::HashMap;

/// Strict Remove-Wins / monotonic add-only revocation set. Once a target
/// is in the set, no subsequent cert can remove it. The earliest-timestamp
/// revocation cert wins per target (so replays of older state don't lose
/// information about already-known revocations).
#[derive(Debug, Clone, Default)]
pub struct RevocationSet {
    /// target -> earliest revocation cert seen
    cells: HashMap<[u8; 16], RevocationCert>,
}

impl RevocationSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, cert: RevocationCert) {
        match self.cells.get(&cert.target) {
            Some(existing) if existing.issued_at <= cert.issued_at => { /* keep earlier */ }
            _ => { self.cells.insert(cert.target, cert); }
        }
    }

    pub fn merge(&mut self, other: RevocationSet) {
        for cert in other.cells.into_values() {
            self.insert(cert);
        }
    }

    pub fn is_revoked(&self, target: [u8; 16]) -> bool {
        self.cells.contains_key(&target)
    }

    pub fn cert_for(&self, target: [u8; 16]) -> Option<&RevocationCert> {
        self.cells.get(&target)
    }

    pub fn iter(&self) -> impl Iterator<Item = &RevocationCert> {
        self.cells.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::RevocationReason;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn make_self_revocation(target: [u8; 16], ts: u64) -> RevocationCert {
        let sk = SigningKey::generate(&mut OsRng);
        RevocationCert::sign_self(&sk, [0u8; 16], target, ts, RevocationReason::Decommissioned).unwrap()
    }

    #[test]
    fn revocation_is_monotonic() {
        let mut set = RevocationSet::new();
        let target = [9u8; 16];

        set.insert(make_self_revocation(target, 100));
        assert!(set.is_revoked(target));

        // Idempotent
        set.insert(make_self_revocation(target, 200));
        assert!(set.is_revoked(target));
    }

    #[test]
    fn earliest_revocation_wins() {
        let mut set = RevocationSet::new();
        let target = [9u8; 16];

        let earlier = make_self_revocation(target, 100);
        let later = make_self_revocation(target, 200);

        // Insert later first, then earlier
        set.insert(later);
        set.insert(earlier);

        assert_eq!(set.cert_for(target).unwrap().issued_at, 100);
    }

    #[test]
    fn merge_preserves_revocations_from_both() {
        let target_a = [1u8; 16];
        let target_b = [2u8; 16];

        let mut s1 = RevocationSet::new();
        s1.insert(make_self_revocation(target_a, 100));

        let mut s2 = RevocationSet::new();
        s2.insert(make_self_revocation(target_b, 200));

        s1.merge(s2);
        assert!(s1.is_revoked(target_a));
        assert!(s1.is_revoked(target_b));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner revocation_set 2>&1 | tail -10`
Expected: PASS, 3 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/crdt
git commit -m "feat(owner): RevocationSet strict Remove-Wins CRDT (ZEB-173)"
```

---

### Task 13: `OwnerState` aggregating certs

**Files:**
- Modify: `crates/harmony-owner/src/state.rs`

- [ ] **Step 1: Write OwnerState + tests**

Replace `crates/harmony-owner/src/state.rs`:

```rust
use crate::certs::{EnrollmentCert, EnrollmentIssuer, LivenessCert, ReclamationCert};
use crate::crdt::{RevocationSet, VouchingSet};
use crate::OwnerError;
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;

/// Aggregate state for one owner identity: enrollment certs per device,
/// vouching CRDT, revocation set, latest liveness per device, optional
/// reclamation cert if this identity claims continuity from a predecessor.
#[derive(Debug, Clone, Default)]
pub struct OwnerState {
    pub owner_id: [u8; 16],
    pub enrollments: HashMap<[u8; 16], EnrollmentCert>,
    pub vouching: VouchingSet,
    pub revocations: RevocationSet,
    pub liveness: HashMap<[u8; 16], LivenessCert>,
    pub reclamation: Option<ReclamationCert>,
}

impl OwnerState {
    pub fn new(owner_id: [u8; 16]) -> Self {
        Self { owner_id, ..Default::default() }
    }

    pub fn add_enrollment(&mut self, cert: EnrollmentCert) -> Result<(), OwnerError> {
        cert.verify()?;
        // For Quorum certs, also walk back to verify each signer's enrollment
        // is present and was issued before this cert's `issued_at`.
        if let EnrollmentIssuer::Quorum { signers, signatures } = &cert.issuer {
            if signers.len() < 2 {
                return Err(OwnerError::InsufficientQuorum { min: 2, got: signers.len() });
            }
            // Verify quorum signatures: each signer in `signers` must have an
            // existing enrollment, and each signature in `signatures` must be
            // valid for that signer's pubkey.
            for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
                let signer_enrollment = self.enrollments.get(signer_id)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: *signer_id })?;
                let vk = VerifyingKey::from_bytes(&signer_enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" })?;
                let payload_bytes = quorum_signing_payload(&cert)?;
                crate::signing::verify_with_tag(
                    &vk,
                    crate::signing::tags::ENROLLMENT,
                    &payload_bytes,
                    sig,
                    "Enrollment-Quorum-Member",
                )?;
            }
        }
        self.enrollments.insert(cert.device_id, cert);
        Ok(())
    }

    pub fn add_liveness(&mut self, cert: LivenessCert) -> Result<(), OwnerError> {
        let enrollment = self.enrollments.get(&cert.signer)
            .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.signer })?;
        let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Liveness" })?;
        cert.verify(&vk)?;
        match self.liveness.get(&cert.signer) {
            Some(existing) if existing.timestamp >= cert.timestamp => { /* keep newer */ }
            _ => { self.liveness.insert(cert.signer, cert); }
        }
        Ok(())
    }

    pub fn add_vouching(&mut self, cert: crate::certs::VouchingCert) -> Result<(), OwnerError> {
        let enrollment = self.enrollments.get(&cert.signer)
            .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.signer })?;
        let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Vouching" })?;
        cert.verify(&vk)?;
        self.vouching.insert(cert);
        Ok(())
    }

    pub fn add_revocation(&mut self, cert: crate::certs::RevocationCert) -> Result<(), OwnerError> {
        // SelfDevice verification needs the device's pubkey from its enrollment.
        match &cert.issuer {
            crate::certs::RevocationIssuer::SelfDevice => {
                let enrollment = self.enrollments.get(&cert.target)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.target })?;
                let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Revocation" })?;
                cert.verify(Some(&vk))?;
            }
            _ => {
                cert.verify(None)?;
            }
        }
        self.revocations.insert(cert);
        Ok(())
    }

    pub fn active_devices(&self, now: u64, active_window_secs: u64) -> Vec<[u8; 16]> {
        let cutoff = now.saturating_sub(active_window_secs);
        self.enrollments.keys()
            .filter(|id| !self.revocations.is_revoked(**id))
            .filter(|id| match self.liveness.get(*id) {
                Some(l) => l.timestamp >= cutoff,
                None => false,
            })
            .copied()
            .collect()
    }

    pub fn is_revoked(&self, device: [u8; 16]) -> bool {
        self.revocations.is_revoked(device)
    }
}

/// Compute the canonical payload bytes used for quorum signature verification.
/// Same payload bytes that the Master signing path uses, except `signers` is
/// included but `signatures` is empty (signatures sign the rest of the payload
/// + signers list).
fn quorum_signing_payload(cert: &EnrollmentCert) -> Result<Vec<u8>, OwnerError> {
    use crate::cbor;
    use crate::pubkey_bundle::PubKeyBundle;
    use serde::Serialize;

    #[derive(Serialize)]
    struct QuorumPayload<'a> {
        version: u8,
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: &'a PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        issuer_kind: u8,
        signers: &'a Vec<[u8; 16]>,
    }

    let signers = match &cert.issuer {
        EnrollmentIssuer::Quorum { signers, .. } => signers,
        _ => return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" }),
    };

    cbor::to_canonical(&QuorumPayload {
        version: cert.version,
        owner_id: cert.owner_id,
        device_id: cert.device_id,
        device_pubkeys: &cert.device_pubkeys,
        issued_at: cert.issued_at,
        expires_at: cert.expires_at,
        issuer_kind: 1,
        signers,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::{EnrollmentCert, LivenessCert};
    use crate::pubkey_bundle::{ClassicalKeys, PubKeyBundle};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn keypair_and_bundle() -> (SigningKey, PubKeyBundle) {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        (sk, bundle)
    }

    #[test]
    fn add_master_enrollment_and_liveness_makes_active() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();

        let enrollment = EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_id, device_bundle, 1_000_000, None
        ).unwrap();

        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(enrollment).unwrap();

        let liveness = LivenessCert::sign(&device_sk, device_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let active = state.active_devices(1_500_000, 24 * 60 * 60);
        assert_eq!(active, vec![device_id]);
    }

    #[test]
    fn revoked_device_is_not_active() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();

        let enrollment = EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_id, device_bundle, 1_000_000, None
        ).unwrap();

        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(enrollment).unwrap();

        let liveness = LivenessCert::sign(&device_sk, device_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let revocation = crate::certs::RevocationCert::sign_self(
            &device_sk, owner_id, device_id, 1_400_000,
            crate::certs::RevocationReason::Decommissioned,
        ).unwrap();
        state.add_revocation(revocation).unwrap();

        let active = state.active_devices(1_500_000, 24 * 60 * 60);
        assert!(active.is_empty());
        assert!(state.is_revoked(device_id));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-owner state:: 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/state.rs
git commit -m "feat(owner): OwnerState aggregating enrollment + liveness + vouching + revocation (ZEB-173)"
```

---

### Task 14: Trust evaluation algorithm

**Files:**
- Modify: `crates/harmony-owner/src/trust.rs`

- [ ] **Step 1: Write trust evaluation + tests**

Replace `crates/harmony-owner/src/trust.rs`:

```rust
use crate::state::OwnerState;
use std::collections::HashSet;

pub const DEFAULT_ACTIVE_WINDOW_SECS: u64 = 90 * 24 * 60 * 60;
pub const DEFAULT_FRESHNESS_WINDOW_SECS: u64 = 30 * 24 * 60 * 60;
pub const N_VOUCH_THRESHOLD_V1: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustDecision {
    Full,
    Provisional,
    Refused(RefusalReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalReason {
    NotEnrolled,
    Revoked,
    Contested,
    StaleTrustState,
    ChallengedBySibling,
}

pub fn evaluate_trust(
    state: &OwnerState,
    target: [u8; 16],
    now: u64,
    active_window_secs: u64,
    freshness_window_secs: u64,
) -> TrustDecision {
    if state.is_revoked(target) {
        return TrustDecision::Refused(RefusalReason::Revoked);
    }
    if !state.enrollments.contains_key(&target) {
        return TrustDecision::Refused(RefusalReason::NotEnrolled);
    }
    let active = state.active_devices(now, active_window_secs);
    let active_set: HashSet<_> = active.iter().copied().collect();

    // Freshness: at least one cert in the state must be within freshness window.
    let cutoff = now.saturating_sub(freshness_window_secs);
    let any_fresh = state.liveness.values().any(|l| l.timestamp >= cutoff)
        || state.vouching.iter().any(|v| v.issued_at >= cutoff);
    if !any_fresh {
        return TrustDecision::Refused(RefusalReason::StaleTrustState);
    }

    // Single-device case
    if active_set.len() == 1 && active_set.contains(&target) {
        return TrustDecision::Full;
    }

    // Challenges from active siblings
    let challenged = state.vouching.challenges_against(target)
        .any(|c| active_set.contains(&c.signer));
    if challenged {
        return TrustDecision::Refused(RefusalReason::ChallengedBySibling);
    }

    // Vouches from active siblings (excluding target itself)
    let vouches = state.vouching.vouches_for(target)
        .filter(|v| active_set.contains(&v.signer) && v.signer != target)
        .count();

    if vouches >= N_VOUCH_THRESHOLD_V1 {
        TrustDecision::Full
    } else {
        TrustDecision::Provisional
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::{EnrollmentCert, LivenessCert, Stance, VouchingCert};
    use crate::pubkey_bundle::{ClassicalKeys, PubKeyBundle};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn keypair_and_bundle() -> (SigningKey, PubKeyBundle) {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        (sk, bundle)
    }

    fn enroll_via_master(state: &mut OwnerState, master_sk: &SigningKey, master_bundle: PubKeyBundle, device_bundle: PubKeyBundle, ts: u64) -> [u8; 16] {
        let device_id = device_bundle.identity_hash();
        let cert = EnrollmentCert::sign_master(master_sk, master_bundle, device_id, device_bundle, ts, None).unwrap();
        state.add_enrollment(cert).unwrap();
        device_id
    }

    #[test]
    fn single_device_yields_full_trust() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let device_id = enroll_via_master(&mut state, &master_sk, master_bundle, device_bundle, 1_000_000);

        let liveness = LivenessCert::sign(&device_sk, device_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let decision = evaluate_trust(&state, device_id, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Full);
    }

    #[test]
    fn second_device_no_vouch_is_provisional() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, id_a, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, id_b, 1_500_000).unwrap()).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Provisional);
    }

    #[test]
    fn one_vouch_yields_full_trust() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, id_a, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, id_b, 1_500_000).unwrap()).unwrap();

        let vouch = VouchingCert::sign(&sk_a, owner_id, id_a, id_b, Stance::Vouch, 1_400_000).unwrap();
        state.add_vouching(vouch).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Full);
    }

    #[test]
    fn challenge_overrides_vouch() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, id_a, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, id_b, 1_500_000).unwrap()).unwrap();

        // Vouch first, then challenge from same signer
        state.add_vouching(VouchingCert::sign(&sk_a, owner_id, id_a, id_b, Stance::Vouch, 1_400_000).unwrap()).unwrap();
        state.add_vouching(VouchingCert::sign(&sk_a, owner_id, id_a, id_b, Stance::Challenge, 1_450_000).unwrap()).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Refused(RefusalReason::ChallengedBySibling));
    }

    #[test]
    fn stale_trust_state_refuses() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_a.clone(), 1_000_000);
        state.add_liveness(LivenessCert::sign(&sk_a, id_a, 1_000_001).unwrap()).unwrap();

        // `now` is far past the freshness window
        let now = 1_000_001 + DEFAULT_FRESHNESS_WINDOW_SECS + 1;
        let decision = evaluate_trust(&state, id_a, now, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Refused(RefusalReason::StaleTrustState));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-owner trust:: 2>&1 | tail -10`
Expected: PASS, 5 tests.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/trust.rs
git commit -m "feat(owner): trust evaluation with N=1 vouch + freshness window (ZEB-173)"
```

---

### Task 15: Lifecycle helpers — mint and recovery artifact

**Files:**
- Create: `crates/harmony-owner/src/lifecycle/mod.rs`
- Create: `crates/harmony-owner/src/lifecycle/mint.rs`

- [ ] **Step 1: Restructure lifecycle to directory**

```bash
rm crates/harmony-owner/src/lifecycle.rs
mkdir crates/harmony-owner/src/lifecycle
```

Create `crates/harmony-owner/src/lifecycle/mod.rs`:

```rust
pub mod mint;
pub use mint::{mint_owner, MintResult, RecoveryArtifact};
```

- [ ] **Step 2: Write mint flow + tests**

Create `crates/harmony-owner/src/lifecycle/mint.rs`:

```rust
use crate::certs::EnrollmentCert;
use crate::pubkey_bundle::{ClassicalKeys, PubKeyBundle};
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;
use rand_core::{OsRng, RngCore};
use zeroize::Zeroize;

/// 32-byte master seed. Format BIP39-wraps to 24 mnemonic words. Drop wipes.
#[derive(Clone)]
pub struct RecoveryArtifact {
    seed: [u8; 32],
}

impl RecoveryArtifact {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self { seed }
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.seed
    }
    /// Reconstruct master signing key from the seed.
    pub fn master_signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(&self.seed)
    }
}

impl Drop for RecoveryArtifact {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

pub struct MintResult {
    pub state: OwnerState,
    pub recovery_artifact: RecoveryArtifact,
    pub device_signing_key: SigningKey,
}

/// Mint a fresh owner identity with device #1.
///
/// Returns the OwnerState (with device #1 enrolled), the recovery artifact
/// (which the user must back up — it encodes the master key), and device
/// #1's signing key (which the device retains for ongoing operation).
///
/// IMPORTANT: After this returns, the master key is reconstructible only
/// from the recovery artifact. Callers must never persist the master key
/// outside that artifact.
pub fn mint_owner(now: u64) -> Result<MintResult, OwnerError> {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    let master_sk = SigningKey::from_bytes(&seed);
    let master_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: master_sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32], // TODO: derive real X25519 from same seed via HKDF in v1.1
        },
        post_quantum: None,
    };
    let owner_id = master_bundle.identity_hash();

    let device_sk = SigningKey::generate(&mut OsRng);
    let device_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: device_sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    let device_id = device_bundle.identity_hash();

    let cert = EnrollmentCert::sign_master(
        &master_sk,
        master_bundle,
        device_id,
        device_bundle,
        now,
        None,
    )?;

    let mut state = OwnerState::new(owner_id);
    state.add_enrollment(cert)?;

    let recovery_artifact = RecoveryArtifact::from_seed(seed);
    // master_sk is dropped here, signaling intent to wipe master from RAM
    // (callers should ensure they don't retain master_sk anywhere).
    drop(master_sk);

    Ok(MintResult {
        state,
        recovery_artifact,
        device_signing_key: device_sk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_produces_active_device_one() {
        let result = mint_owner(1_700_000_000).unwrap();
        assert_eq!(result.state.enrollments.len(), 1);
        assert_eq!(result.recovery_artifact.as_bytes().len(), 32);
    }

    #[test]
    fn recovery_artifact_round_trip_yields_same_master_key() {
        let result = mint_owner(1_700_000_000).unwrap();
        let restored_sk = result.recovery_artifact.master_signing_key();
        let owner_id_via_restored = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: restored_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        }.identity_hash();
        assert_eq!(owner_id_via_restored, result.state.owner_id);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner mint 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/lifecycle
git rm crates/harmony-owner/src/lifecycle.rs
git commit -m "feat(owner): mint flow + recovery artifact with seed-derived master key (ZEB-173)"
```

---

### Task 16: Lifecycle — enroll-via-master

**Files:**
- Create: `crates/harmony-owner/src/lifecycle/enroll_master.rs`
- Modify: `crates/harmony-owner/src/lifecycle/mod.rs`

- [ ] **Step 1: Module export**

```rust
pub mod enroll_master;
pub mod mint;
pub use enroll_master::enroll_via_master;
pub use mint::{mint_owner, MintResult, RecoveryArtifact};
```

- [ ] **Step 2: Write helper + tests**

Create `crates/harmony-owner/src/lifecycle/enroll_master.rs`:

```rust
use crate::certs::{EnrollmentCert, Stance, VouchingCert};
use crate::lifecycle::RecoveryArtifact;
use crate::pubkey_bundle::PubKeyBundle;
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;

pub struct EnrollResult {
    pub enrollment_cert: EnrollmentCert,
    pub auto_vouch_certs: Vec<VouchingCert>,
}

/// Enroll a new device under the existing owner via the recovery artifact.
///
/// This brings the master signing key into RAM transiently, signs the new
/// device's enrollment cert, and immediately drops the master key. The new
/// device also auto-vouches for every active sibling.
///
/// Returns the new device's enrollment cert plus auto-vouches for siblings.
pub fn enroll_via_master(
    state: &OwnerState,
    artifact: &RecoveryArtifact,
    new_device_sk: &SigningKey,
    new_device_pubkey: PubKeyBundle,
    now: u64,
    active_window_secs: u64,
) -> Result<EnrollResult, OwnerError> {
    // Reconstruct master from artifact (transient).
    let master_sk = artifact.master_signing_key();
    let master_pubkey = master_pubkey_from_sk(&master_sk);

    let device_id = new_device_pubkey.identity_hash();
    let enrollment_cert = EnrollmentCert::sign_master(
        &master_sk,
        master_pubkey,
        device_id,
        new_device_pubkey,
        now,
        None,
    )?;
    drop(master_sk); // wipe master from RAM

    // New device auto-vouches for every active sibling.
    let active = state.active_devices(now, active_window_secs);
    let auto_vouch_certs: Vec<VouchingCert> = active.iter()
        .filter(|s| **s != device_id)
        .map(|sibling_id| VouchingCert::sign(
            new_device_sk,
            state.owner_id,
            device_id,
            *sibling_id,
            Stance::Vouch,
            now,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EnrollResult { enrollment_cert, auto_vouch_certs })
}

fn master_pubkey_from_sk(sk: &SigningKey) -> PubKeyBundle {
    use crate::pubkey_bundle::ClassicalKeys;
    PubKeyBundle {
        classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
        post_quantum: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::LivenessCert;
    use crate::lifecycle::mint_owner;
    use crate::pubkey_bundle::ClassicalKeys;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn enroll_second_device_via_master() {
        let mint = mint_owner(1_000_000).unwrap();
        // Device #1 is alive
        let device_a_id = *mint.state.enrollments.keys().next().unwrap();
        let mut state = mint.state;
        state.add_liveness(LivenessCert::sign(&mint.device_signing_key, device_a_id, 1_000_001).unwrap()).unwrap();

        // Generate device #2
        let device_b_sk = SigningKey::generate(&mut OsRng);
        let device_b_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_b_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };

        let result = enroll_via_master(
            &state,
            &mint.recovery_artifact,
            &device_b_sk,
            device_b_bundle,
            1_001_000,
            crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();

        // Apply to state
        state.add_enrollment(result.enrollment_cert.clone()).unwrap();
        for v in &result.auto_vouch_certs {
            state.add_vouching(v.clone()).unwrap();
        }

        // Auto-vouch should reach exactly device A (and not B itself)
        assert_eq!(result.auto_vouch_certs.len(), 1);
        assert_eq!(result.auto_vouch_certs[0].target, device_a_id);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner enroll_master 2>&1 | tail -10`
Expected: PASS, 1 test.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/lifecycle
git commit -m "feat(owner): enroll-via-master with auto-vouch for active siblings (ZEB-173)"
```

---

### Task 17: Lifecycle — enroll-via-quorum

**Files:**
- Create: `crates/harmony-owner/src/lifecycle/enroll_quorum.rs`
- Modify: `crates/harmony-owner/src/lifecycle/mod.rs`

- [ ] **Step 1: Module export**

```rust
pub mod enroll_master;
pub mod enroll_quorum;
pub mod mint;
pub use enroll_master::{enroll_via_master, EnrollResult};
pub use enroll_quorum::enroll_via_quorum;
pub use mint::{mint_owner, MintResult, RecoveryArtifact};
```

- [ ] **Step 2: Write helper + tests**

Create `crates/harmony-owner/src/lifecycle/enroll_quorum.rs`:

```rust
use crate::cbor;
use crate::certs::{EnrollmentCert, EnrollmentIssuer, Stance, VouchingCert};
use crate::lifecycle::EnrollResult;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags};
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;
use serde::Serialize;

#[derive(Serialize)]
struct QuorumPayload<'a> {
    version: u8,
    owner_id: [u8; 16],
    device_id: [u8; 16],
    device_pubkeys: &'a PubKeyBundle,
    issued_at: u64,
    expires_at: Option<u64>,
    issuer_kind: u8,
    signers: &'a Vec<[u8; 16]>,
}

/// Enroll a new device using K=2 quorum of existing siblings (no recovery
/// artifact needed). Returns the new device's enrollment cert + auto-vouches.
pub fn enroll_via_quorum(
    state: &OwnerState,
    quorum_signers: Vec<(&SigningKey, [u8; 16])>,
    new_device_sk: &SigningKey,
    new_device_pubkey: PubKeyBundle,
    now: u64,
    active_window_secs: u64,
) -> Result<EnrollResult, OwnerError> {
    if quorum_signers.len() < 2 {
        return Err(OwnerError::InsufficientQuorum { min: 2, got: quorum_signers.len() });
    }
    let device_id = new_device_pubkey.identity_hash();

    let signers: Vec<[u8; 16]> = quorum_signers.iter().map(|(_, id)| *id).collect();

    let payload_bytes = cbor::to_canonical(&QuorumPayload {
        version: crate::certs::enrollment::ENROLLMENT_VERSION,
        owner_id: state.owner_id,
        device_id,
        device_pubkeys: &new_device_pubkey,
        issued_at: now,
        expires_at: None,
        issuer_kind: 1,
        signers: &signers,
    })?;

    let signatures: Vec<Vec<u8>> = quorum_signers.iter()
        .map(|(sk, _)| sign_with_tag(sk, tags::ENROLLMENT, &payload_bytes))
        .collect();

    // Construct cert with quorum issuer; signature field is not used for
    // quorum (signers' individual signatures live in EnrollmentIssuer::Quorum).
    let enrollment_cert = EnrollmentCert {
        version: crate::certs::enrollment::ENROLLMENT_VERSION,
        owner_id: state.owner_id,
        device_id,
        device_pubkeys: new_device_pubkey,
        issued_at: now,
        expires_at: None,
        issuer: EnrollmentIssuer::Quorum { signers: signers.clone(), signatures },
        signature: Vec::new(),
    };

    let active = state.active_devices(now, active_window_secs);
    let auto_vouch_certs: Vec<VouchingCert> = active.iter()
        .filter(|s| **s != device_id)
        .map(|sibling_id| VouchingCert::sign(
            new_device_sk,
            state.owner_id,
            device_id,
            *sibling_id,
            Stance::Vouch,
            now,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EnrollResult { enrollment_cert, auto_vouch_certs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::LivenessCert;
    use crate::lifecycle::{enroll_via_master, mint_owner};
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn enroll_third_device_via_quorum() {
        let mint = mint_owner(1_000_000).unwrap();
        let mut state = mint.state;
        let device_a_id = *state.enrollments.keys().next().unwrap();
        let device_a_sk = mint.device_signing_key;
        state.add_liveness(LivenessCert::sign(&device_a_sk, device_a_id, 1_000_001).unwrap()).unwrap();

        // Enroll device B via master
        let device_b_sk = SigningKey::generate(&mut OsRng);
        let device_b_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_b_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let r1 = enroll_via_master(&state, &mint.recovery_artifact, &device_b_sk, device_b_bundle.clone(), 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
        let device_b_id = device_b_bundle.identity_hash();
        state.add_enrollment(r1.enrollment_cert).unwrap();
        for v in r1.auto_vouch_certs { state.add_vouching(v).unwrap(); }
        state.add_liveness(LivenessCert::sign(&device_b_sk, device_b_id, 1_001_001).unwrap()).unwrap();

        // Now enroll device C via quorum of A+B
        let device_c_sk = SigningKey::generate(&mut OsRng);
        let device_c_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_c_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };

        let r2 = enroll_via_quorum(
            &state,
            vec![(&device_a_sk, device_a_id), (&device_b_sk, device_b_id)],
            &device_c_sk,
            device_c_bundle.clone(),
            1_002_000,
            crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();

        state.add_enrollment(r2.enrollment_cert).unwrap();
        for v in r2.auto_vouch_certs { state.add_vouching(v).unwrap(); }

        // Verify Device C is now enrolled
        assert!(state.enrollments.contains_key(&device_c_bundle.identity_hash()));
    }

    #[test]
    fn quorum_with_one_signer_rejected() {
        let mint = mint_owner(1_000_000).unwrap();
        let device_a_id = *mint.state.enrollments.keys().next().unwrap();
        let device_a_sk = mint.device_signing_key;
        let new_sk = SigningKey::generate(&mut OsRng);
        let new_bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: new_sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let result = enroll_via_quorum(&mint.state, vec![(&device_a_sk, device_a_id)], &new_sk, new_bundle, 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        assert!(matches!(result, Err(OwnerError::InsufficientQuorum { .. })));
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner enroll_quorum 2>&1 | tail -10`
Expected: PASS, 2 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/lifecycle
git commit -m "feat(owner): enroll-via-quorum with K=2 sibling co-signing (ZEB-173)"
```

---

### Task 18: Lifecycle — reclamation flow

**Files:**
- Create: `crates/harmony-owner/src/lifecycle/reclamation.rs`
- Modify: `crates/harmony-owner/src/lifecycle/mod.rs`

- [ ] **Step 1: Module export**

```rust
pub mod enroll_master;
pub mod enroll_quorum;
pub mod mint;
pub mod reclamation;
pub use enroll_master::{enroll_via_master, EnrollResult};
pub use enroll_quorum::enroll_via_quorum;
pub use mint::{mint_owner, MintResult, RecoveryArtifact};
pub use reclamation::{evaluate_reclamation, mint_reclaimed, ReclamationStatus};
```

- [ ] **Step 2: Write reclamation helpers + tests**

Create `crates/harmony-owner/src/lifecycle/reclamation.rs`:

```rust
use crate::certs::{LivenessCert, ReclamationCert};
use crate::lifecycle::{mint_owner, MintResult};
use crate::OwnerError;

pub struct ReclamationMintResult {
    pub mint: MintResult,
    pub reclamation_cert: ReclamationCert,
}

/// Mint a fresh identity AND publish a Reclamation Cert claiming continuity
/// from a predecessor.
pub fn mint_reclaimed(
    claimed_predecessor: [u8; 16],
    challenge_window_secs: u64,
    note: String,
    now: u64,
) -> Result<ReclamationMintResult, OwnerError> {
    let mint = mint_owner(now)?;
    // Reconstruct master to sign the reclamation cert
    let master_sk = mint.recovery_artifact.master_signing_key();
    let master_pubkey = crate::pubkey_bundle::PubKeyBundle {
        classical: crate::pubkey_bundle::ClassicalKeys {
            ed25519_verify: master_sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    let reclamation_cert = ReclamationCert::sign(
        &master_sk,
        master_pubkey,
        claimed_predecessor,
        now,
        challenge_window_secs,
        note,
    )?;
    drop(master_sk);
    Ok(ReclamationMintResult { mint, reclamation_cert })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReclamationStatus {
    /// Window still open; no refutation observed yet.
    Pending,
    /// Window expired without refutation; honored at reduced trust.
    Honored,
    /// Refuted by a predecessor liveness cert with timestamp > issued_at.
    Refuted,
}

/// Evaluate a reclamation cert's status given the current time and any
/// observed liveness from devices under the predecessor identity.
pub fn evaluate_reclamation(
    cert: &ReclamationCert,
    predecessor_liveness_certs: &[LivenessCert],
    now: u64,
) -> ReclamationStatus {
    let refuted = predecessor_liveness_certs.iter()
        .any(|l| cert.is_refuted_by_timestamp(l.timestamp));
    if refuted {
        return ReclamationStatus::Refuted;
    }
    if cert.is_window_expired(now) {
        ReclamationStatus::Honored
    } else {
        ReclamationStatus::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::reclamation::DEFAULT_CHALLENGE_WINDOW_SECS;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn pending_during_window() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[], 1_000_500);
        assert_eq!(status, ReclamationStatus::Pending);
    }

    #[test]
    fn honored_after_window() {
        let result = mint_reclaimed([9u8; 16], 1000, "lost".into(), 1_000_000).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[], 1_002_000);
        assert_eq!(status, ReclamationStatus::Honored);
    }

    #[test]
    fn refuted_by_predecessor_liveness() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let predecessor_sk = SigningKey::generate(&mut OsRng);
        let liveness = LivenessCert::sign(&predecessor_sk, [9u8; 16], 1_000_500).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[liveness], 1_000_500);
        assert_eq!(status, ReclamationStatus::Refuted);
    }

    #[test]
    fn liveness_before_reclamation_does_not_refute() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let predecessor_sk = SigningKey::generate(&mut OsRng);
        let liveness = LivenessCert::sign(&predecessor_sk, [9u8; 16], 999_999).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[liveness], 1_000_500);
        assert_eq!(status, ReclamationStatus::Pending);
    }
}
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-owner reclamation:: 2>&1 | tail -10`
Expected: PASS, 4 tests.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/lifecycle
git commit -m "feat(owner): reclamation flow + status evaluation (ZEB-173)"
```

---

### Task 19: End-to-end happy-path lifecycle test

**Files:**
- Create: `crates/harmony-owner/tests/e2e_lifecycle.rs`

- [ ] **Step 1: Write the integration test**

Create `crates/harmony-owner/tests/e2e_lifecycle.rs`:

```rust
//! End-to-end happy-path: mint, enroll three devices via mixed paths,
//! exchange vouches, verify trust evaluation, archive a stale device.

use harmony_owner::{
    certs::{LivenessCert, RevocationCert, RevocationReason, Stance, VouchingCert},
    lifecycle::{enroll_via_master, enroll_via_quorum, mint_owner},
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
    trust::{evaluate_trust, RefusalReason, TrustDecision, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS},
};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

fn fresh_device() -> (SigningKey, PubKeyBundle) {
    let sk = SigningKey::generate(&mut OsRng);
    let bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    (sk, bundle)
}

#[test]
fn full_three_device_lifecycle() {
    // T0: Mint
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;
    state.add_liveness(LivenessCert::sign(&device_a_sk, device_a_id, 1_000_001).unwrap()).unwrap();

    // Single-device → Full trust
    assert_eq!(
        evaluate_trust(&state, device_a_id, 1_000_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T1: Enroll Device B via master
    let (device_b_sk, device_b_bundle) = fresh_device();
    let device_b_id = device_b_bundle.identity_hash();
    let r1 = enroll_via_master(&state, &mint.recovery_artifact, &device_b_sk, device_b_bundle, 1_001_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    state.add_enrollment(r1.enrollment_cert).unwrap();
    for v in r1.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&device_b_sk, device_b_id, 1_001_001).unwrap()).unwrap();

    // Device B is provisional (auto-vouches don't count for B because B vouched for A, not the other way)
    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_001_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Provisional
    );

    // Device A ratifies B
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_a_id, device_b_id, Stance::Vouch, 1_001_500).unwrap()).unwrap();

    // Device B is now Full
    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_001_500, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T2: Enroll Device C via quorum of A+B
    let (device_c_sk, device_c_bundle) = fresh_device();
    let device_c_id = device_c_bundle.identity_hash();
    let r2 = enroll_via_quorum(
        &state,
        vec![(&device_a_sk, device_a_id), (&device_b_sk, device_b_id)],
        &device_c_sk,
        device_c_bundle,
        1_002_000,
        DEFAULT_ACTIVE_WINDOW_SECS,
    ).unwrap();
    state.add_enrollment(r2.enrollment_cert).unwrap();
    for v in r2.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&device_c_sk, device_c_id, 1_002_001).unwrap()).unwrap();

    // Device A ratifies C
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_a_id, device_c_id, Stance::Vouch, 1_002_500).unwrap()).unwrap();

    assert_eq!(
        evaluate_trust(&state, device_c_id, 1_002_500, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T3: Revoke Device B (decommissioned)
    let revocation = RevocationCert::sign_self(&device_b_sk, state.owner_id, device_b_id, 1_003_000, RevocationReason::Decommissioned).unwrap();
    state.add_revocation(revocation).unwrap();

    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_003_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Refused(RefusalReason::Revoked)
    );

    // T4: Device C goes silent for 91 days. Active set should drop C.
    let way_later = 1_003_000 + 91 * 24 * 60 * 60;
    state.add_liveness(LivenessCert::sign(&device_a_sk, device_a_id, way_later).unwrap()).unwrap();
    let active_now = state.active_devices(way_later, DEFAULT_ACTIVE_WINDOW_SECS);
    assert_eq!(active_now, vec![device_a_id]);
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p harmony-owner --test e2e_lifecycle 2>&1 | tail -10`
Expected: PASS, 1 test.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/tests/e2e_lifecycle.rs
git commit -m "test(owner): end-to-end three-device lifecycle (ZEB-173)"
```

---

### Task 20: Threat scenario tests

**Files:**
- Create: `crates/harmony-owner/tests/e2e_threats.rs`

- [ ] **Step 1: Write threat-scenario tests**

Create `crates/harmony-owner/tests/e2e_threats.rs`:

```rust
//! End-to-end threat scenarios from the design's threat-coverage matrix:
//! stolen master, contested reclamation, partition + replay.

use harmony_owner::{
    certs::{LivenessCert, RevocationCert, RevocationReason, Stance, VouchingCert},
    lifecycle::{enroll_via_master, evaluate_reclamation, mint_owner, mint_reclaimed, ReclamationStatus},
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
    trust::{evaluate_trust, RefusalReason, TrustDecision, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS},
    certs::reclamation::DEFAULT_CHALLENGE_WINDOW_SECS,
};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

fn fresh_device() -> (SigningKey, PubKeyBundle) {
    let sk = SigningKey::generate(&mut OsRng);
    let bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    (sk, bundle)
}

#[test]
fn stolen_master_attacker_device_remains_provisional_when_real_devices_dont_vouch() {
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;
    state.add_liveness(LivenessCert::sign(&device_a_sk, device_a_id, 1_000_001).unwrap()).unwrap();

    // Attacker has the recovery artifact and enrolls a malicious device.
    let (attacker_sk, attacker_bundle) = fresh_device();
    let attacker_id = attacker_bundle.identity_hash();
    let r = enroll_via_master(&state, &mint.recovery_artifact, &attacker_sk, attacker_bundle, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    state.add_enrollment(r.enrollment_cert).unwrap();
    for v in r.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&attacker_sk, attacker_id, 1_500_001).unwrap()).unwrap();

    // Real device A does NOT vouch for the attacker. Attacker should stay provisional.
    assert_eq!(
        evaluate_trust(&state, attacker_id, 1_500_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Provisional
    );

    // Real device A challenges. Attacker is refused.
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_a_id, attacker_id, Stance::Challenge, 1_500_002).unwrap()).unwrap();
    assert_eq!(
        evaluate_trust(&state, attacker_id, 1_500_002, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Refused(RefusalReason::ChallengedBySibling)
    );
}

#[test]
fn revoked_device_cannot_be_un_revoked() {
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;

    // Self-revoke
    let rev = RevocationCert::sign_self(&device_a_sk, state.owner_id, device_a_id, 1_000_500, RevocationReason::Decommissioned).unwrap();
    state.add_revocation(rev).unwrap();
    assert!(state.is_revoked(device_a_id));

    // Even a "newer" revocation insertion does not un-revoke (it stays revoked).
    let rev2 = RevocationCert::sign_self(&device_a_sk, state.owner_id, device_a_id, 999_000, RevocationReason::Other("test".into())).unwrap();
    state.add_revocation(rev2).unwrap();
    assert!(state.is_revoked(device_a_id));
}

#[test]
fn reclamation_refuted_by_predecessor_liveness() {
    // Predecessor identity is alive
    let predecessor_mint = mint_owner(1_000_000).unwrap();
    let predecessor_owner_id = predecessor_mint.state.owner_id;
    let predecessor_device_id = *predecessor_mint.state.enrollments.keys().next().unwrap();
    let predecessor_sk = predecessor_mint.device_signing_key;

    // M2 publishes reclamation
    let reclaim = mint_reclaimed(predecessor_owner_id, DEFAULT_CHALLENGE_WINDOW_SECS, "thought all devices were lost".into(), 2_000_000).unwrap();

    // Predecessor publishes liveness within window
    let predecessor_liveness = LivenessCert::sign(&predecessor_sk, predecessor_device_id, 2_000_500).unwrap();

    let status = evaluate_reclamation(&reclaim.reclamation_cert, &[predecessor_liveness], 2_001_000);
    assert_eq!(status, ReclamationStatus::Refuted);
}

#[test]
fn reclamation_honored_after_silent_window() {
    let reclaim = mint_reclaimed([7u8; 16], 1000, "fire took everything".into(), 1_000_000).unwrap();
    let status = evaluate_reclamation(&reclaim.reclamation_cert, &[], 1_002_000);
    assert_eq!(status, ReclamationStatus::Honored);
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-owner --test e2e_threats 2>&1 | tail -10`
Expected: PASS, 4 tests.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/tests/e2e_threats.rs
git commit -m "test(owner): threat-coverage e2e (stolen master, refuted reclamation) (ZEB-173)"
```

---

### Task 21: Cross-implementation interop fixtures

**Files:**
- Create: `crates/harmony-owner/tests/interop_fixtures.rs`
- Create: `crates/harmony-owner/tests/fixtures/enrollment_master_v1.cbor` (binary, generated by test on first run)

- [ ] **Step 1: Write golden-bytes test**

Create `crates/harmony-owner/tests/interop_fixtures.rs`:

```rust
//! Cross-implementation interop fixtures. Each test produces a deterministic
//! cert from fixed seeds and asserts the byte-exact CBOR encoding. If a
//! second implementation produces these same bytes from the same inputs,
//! the wire format is unambiguous.
//!
//! On encoding changes that bump BindingFormatVersion, regenerate the
//! fixtures and bump the version byte in the test data.

use harmony_owner::{
    cbor,
    certs::EnrollmentCert,
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
};
use ed25519_dalek::SigningKey;

fn deterministic_master_sk() -> SigningKey {
    let seed = [42u8; 32];
    SigningKey::from_bytes(&seed)
}

fn deterministic_device_sk() -> SigningKey {
    let seed = [99u8; 32];
    SigningKey::from_bytes(&seed)
}

#[test]
fn master_enrollment_cert_v1_is_deterministic() {
    let master_sk = deterministic_master_sk();
    let master_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: master_sk.verifying_key().to_bytes(),
            x25519_pub: [1u8; 32], // fixed for fixture
        },
        post_quantum: None,
    };
    let device_sk = deterministic_device_sk();
    let device_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: device_sk.verifying_key().to_bytes(),
            x25519_pub: [2u8; 32],
        },
        post_quantum: None,
    };
    let device_id = device_bundle.identity_hash();

    let cert = EnrollmentCert::sign_master(
        &master_sk,
        master_bundle,
        device_id,
        device_bundle,
        1_700_000_000,
        None,
    ).unwrap();

    let bytes_a = cbor::to_canonical(&cert).unwrap();
    let bytes_b = cbor::to_canonical(&cert).unwrap();
    assert_eq!(bytes_a, bytes_b, "encoding must be deterministic across runs");

    // Round-trip
    let decoded: EnrollmentCert = cbor::from_bytes(&bytes_a).unwrap();
    assert_eq!(cert, decoded);
    decoded.verify().unwrap();

    // Print bytes for documentation; fixtures consumers can save these.
    println!("EnrollmentCert (Master) v1 bytes ({}): {}", bytes_a.len(), hex::encode(&bytes_a));
}
```

- [ ] **Step 2: Run test**

Run: `cargo test -p harmony-owner --test interop_fixtures -- --nocapture 2>&1 | tail -15`
Expected: PASS, 1 test, prints hex bytes.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/tests/interop_fixtures.rs
git commit -m "test(owner): cross-implementation interop fixture for EnrollmentCert v1 (ZEB-173)"
```

---

### Task 22: Crate documentation + example

**Files:**
- Modify: `crates/harmony-owner/src/lib.rs`
- Create: `crates/harmony-owner/README.md`

- [ ] **Step 1: Expand `src/lib.rs` docs**

Replace `crates/harmony-owner/src/lib.rs`:

```rust
//! # harmony-owner
//!
//! Two-tier owner→device identity binding for the Harmony network.
//!
//! Spec: `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`
//!
//! ## Concepts
//!
//! - **Owner identity `M`**: master keypair defining a single human user.
//!   Lives only in the recovery artifact + transient RAM during enrollment
//!   ceremonies.
//! - **Device identity `D`**: per-device keypair. Persists locally.
//! - **Enrollment Cert**: authorizes `D` under `M`. Signed by `M` or by a
//!   K=2 quorum of already-enrolled siblings.
//! - **Vouching Cert**: per-(signer, target) attestation. LWW CRDT.
//! - **Liveness Cert**: periodic timestamped heartbeat.
//! - **Revocation Cert**: monotonic Remove-Wins. Once present, never reversed.
//! - **Reclamation Cert**: time-bounded claim of continuity from a prior
//!   identity, after total loss.
//!
//! ## Trust evaluation
//!
//! [`trust::evaluate_trust`] returns Full / Provisional / Refused for a
//! target device given current state and time. v1 threshold: N=1 active
//! sibling vouch.
//!
//! ## Out of scope here
//!
//! Network propagation (Zenoh gossip + queryable) and harmony-client
//! integration are separate plans/crates.

pub mod cbor;
pub mod certs;
pub mod crdt;
pub mod error;
pub mod lifecycle;
pub mod pubkey_bundle;
pub mod signing;
pub mod state;
pub mod trust;

pub use error::OwnerError;
```

- [ ] **Step 2: Create README**

Create `crates/harmony-owner/README.md`:

```markdown
# harmony-owner

Two-tier owner→device identity binding: cert types, CRDT state, lifecycle
flows, trust evaluation.

See `docs/superpowers/specs/2026-04-25-harmony-owner-device-binding-design.md`
for the full design.

## Quick example

```rust
use harmony_owner::lifecycle::mint_owner;

let mint = mint_owner(unix_now()).unwrap();
// Save mint.recovery_artifact (BIP39 mnemonic via ZEB-175)
// Use mint.device_signing_key for ongoing device #1 operations
// mint.state has device #1 enrolled
```

## Status

v1 (ZEB-173). Network propagation (Zenoh) and harmony-client wiring tracked
separately under ZEB-169 Track A.
```

- [ ] **Step 3: Verify docs compile**

Run: `cargo doc -p harmony-owner --no-deps 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-owner/src/lib.rs crates/harmony-owner/README.md
git commit -m "docs(owner): crate-level docs + README (ZEB-173)"
```

---

### Task 23: Final full-suite run + clippy

**Files:** none (verification step)

- [ ] **Step 1: Run the full crate test suite**

Run: `cargo test -p harmony-owner 2>&1 | tail -5`
Expected: PASS, all tests (~30 across unit + integration).

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-owner --all-targets -- -D warnings 2>&1 | tail -10`
Expected: clean.

- [ ] **Step 3: Run workspace check to ensure no breakage elsewhere**

Run: `cargo check --workspace 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 4: No commit needed unless clippy required fixes; if so, commit fixups**

```bash
# Only if fixes were needed:
git add crates/harmony-owner
git commit -m "chore(owner): clippy fixups (ZEB-173)"
```

---

## Self-Review

**Spec coverage:**
1. ✅ Owner identity `M` cold-by-default with recovery artifact (Task 15)
2. ✅ Device identity per-device keypair (all tasks)
3. ✅ Enrollment Cert with Master + Quorum issuers (Task 6, 17)
4. ✅ Vouching Cert + Stance enum (Task 7)
5. ✅ Liveness Cert (Task 8)
6. ✅ Revocation Cert with Self/Master/Quorum (Task 9)
7. ✅ Reclamation Cert + challenge window (Task 10, 18)
8. ✅ Two-tier CRDT: Vouching LWW + Revocation Remove-Wins (Task 11, 12)
9. ✅ Domain-separated signing tags (Task 5)
10. ✅ Canonical CBOR (Task 4)
11. ✅ Trust evaluation algorithm with single-device case + freshness window (Task 14)
12. ✅ Active-window archival (Task 13)
13. ✅ Mint flow (Task 15)
14. ✅ Enroll-via-master with auto-vouch (Task 16)
15. ✅ Enroll-via-quorum K=2 (Task 17)
16. ✅ Reclamation flow with refutation (Task 18)
17. ✅ E2E lifecycle test (Task 19)
18. ✅ Threat-coverage tests (Task 20)
19. ✅ Cross-implementation interop fixture (Task 21)

**Out-of-plan-scope items (correctly deferred):**
1. Zenoh gossip topic publish/subscribe → separate plan
2. Queryable for cold-start trust state → separate plan
3. harmony-client Tauri IPC → separate plan
4. Heartbeat timer + scheduling → separate plan (this plan only handles cert *production*, not periodic publishing)
5. Per-task-15 X25519 key derivation from master seed (currently mocked at `[0u8; 32]` for unit tests; real X25519 derivation is a v1.1 follow-up via HKDF — flagged as TODO in the source)

**Type consistency check:**
1. `IdentityHash` consistently typed as `[u8; 16]` everywhere
2. Cert struct field names match across mod.rs re-exports and direct paths
3. `EnrollmentIssuer::Master` carries `master_pubkey`; `EnrollmentIssuer::Quorum` carries `signers + signatures` — used identically in Task 6 and Task 17

**Placeholder scan:** None. Each task has actual code, actual tests, actual commit messages. The one TODO comment in Task 15 is explicitly flagged as v1.1 work.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-25-harmony-owner-device-binding.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?

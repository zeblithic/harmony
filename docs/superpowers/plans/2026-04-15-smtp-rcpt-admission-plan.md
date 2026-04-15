# SMTP RCPT Admission + Discovery-Backed Email→Hash Resolution — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the blanket non-local-domain rejection at SMTP RCPT TO with discovery-backed email→hash resolution, wired end-to-end through a new `harmony-mail-discovery` crate (DNS + HTTPS signed-claim lookup) into the existing PR #240 remote-delivery machinery.

**Architecture:** New leaf crate `harmony-mail-discovery` owns all cryptographic claim verification (sans-I/O `claim.rs`), caches (`cache.rs`), DNS (`dns.rs`) and HTTPS (`http.rs`) fetchers behind traits, and orchestration (`resolver.rs`). `harmony-mail` collapses its three `Option` parameters into a single `RemoteDeliveryContext`, calls `EmailResolver::resolve` at RCPT TO for non-local domains, and maps `ResolveOutcome` to SMTP codes per spec §7.

**Tech Stack:** Rust 1.85, `ciborium` (deterministic CBOR), `ed25519-dalek`, `hickory-resolver` (DNS), `reqwest` (rustls-backed HTTPS), `dashmap`, `tokio`, `tracing`.

**Spec:** `docs/superpowers/specs/2026-04-15-smtp-rcpt-admission-design.md` — referenced throughout as "spec §X.Y". Read it once before starting; it holds all canonical type definitions and invariants.

**Companion research:** `docs/research/2026-04-15-email-hash-resolution-gemini-research.md` — background only, no normative content.

---

## Conventions for this plan

- **TDD throughout.** Every production file begins with a failing test.
- **Commits are frequent and small** — every task ends with a commit. `git add` names specific files (no `git add -A`).
- **All code is Ed25519 only** this PR. The `Signature` / `MasterPubkey` / `SigningPubkey` enums are written to leave room for PQ variants but every call site exhaustively matches and returns `UnsupportedAlgorithm` for any non-`Ed25519` variant.
- **No `.unwrap()` / `.expect()` on deserialization paths.** Enforce crate-wide: `#![deny(clippy::unwrap_used, clippy::expect_used)]` in `src/lib.rs` (test code exempt via `#[cfg(test)]`).
- **Canonical CBOR** (RFC 8949 §4.2) is used for every signature-covered byte range. Use `ciborium::ser::into_writer` with no custom config — `ciborium` emits definite-length encoding which is canonical for our shapes (no maps with duplicate keys, no floats).
- **Test utilities** live in `harmony-mail-discovery::test_support`, gated behind the `test-support` Cargo feature so `harmony-mail` can dev-depend on them. Inline `#[cfg(test)] mod tests` covers tests that don't need cross-module fixtures.
- **Time injection is mandatory.** No `SystemTime::now()` calls anywhere except inside `SystemTimeSource::now()`. All code paths take `&dyn TimeSource` or a `u64` timestamp argument.

---

## File structure

| Action | Path | Responsibility |
|---|---|---|
| Create | `crates/harmony-mail-discovery/Cargo.toml` | Crate manifest with `test-support` feature gate |
| Create | `crates/harmony-mail-discovery/src/lib.rs` | Module roots + crate-wide `#![deny(...)]` |
| Create | `crates/harmony-mail-discovery/src/claim.rs` | Types, canonical CBOR, `SignedClaim::verify` |
| Create | `crates/harmony-mail-discovery/src/cache.rs` | `TimeSource` trait, `ResolverCaches`, TTL+LRU |
| Create | `crates/harmony-mail-discovery/src/dns.rs` | `DnsClient` trait, TXT parser, `HickoryDnsClient` |
| Create | `crates/harmony-mail-discovery/src/http.rs` | `HttpClient` trait, URL construction, `ReqwestHttpClient` |
| Create | `crates/harmony-mail-discovery/src/resolver.rs` | `EmailResolver` trait, `DefaultEmailResolver`, background task |
| Create | `crates/harmony-mail-discovery/src/test_support.rs` | `ClaimBuilder`, `FakeDnsClient`, `FakeHttpClient`, `FakeTimeSource` |
| Create | `crates/harmony-mail-discovery/bin/harmony-mail-discovery-debug.rs` | CLI: resolve `user@domain` once, print outcome |
| Create | `crates/harmony-mail-discovery/tests/resolver_integration.rs` | ~17 component tests orchestrating fakes |
| Modify | `Cargo.toml` (workspace) | Add crate to members; add `hickory-resolver`, `reqwest`, `dashmap`, `async-trait` workspace deps |
| Modify | `crates/harmony-mail/Cargo.toml` | Add `harmony-mail-discovery` (normal + dev-dep with `test-support`), `async-trait` |
| Modify | `crates/harmony-mail/src/server.rs` | `RemoteDeliveryContext`; collapsed `run()` signature; replace blanket-reject block at line 1107-1133 |
| Modify | `crates/harmony-mail/src/remote_delivery.rs` | Add `ZenohRecipientResolver` production impl |
| Modify | `crates/harmony-mail/src/lib.rs` | Re-export `RemoteDeliveryContext` |
| Modify | `crates/harmony-mail/src/main.rs` | Build `RemoteDeliveryContext` and pass to `run()` |
| Create | `crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs` | ~6 end-to-end SMTP integration tests |

---

## Task 1: Scaffold the `harmony-mail-discovery` crate

**Files:**
- Create: `crates/harmony-mail-discovery/Cargo.toml`
- Create: `crates/harmony-mail-discovery/src/lib.rs`
- Modify: `Cargo.toml` (workspace manifest)

- [ ] **Step 1: Add workspace dependencies**

In the root `Cargo.toml`, under `[workspace.dependencies]`, add:

```toml
async-trait = "0.1"
dashmap = "6"
hickory-resolver = { version = "0.24", default-features = false, features = ["tokio-runtime", "system-config"] }
reqwest = { version = "0.12", default-features = false, features = ["rustls-tls-native-roots", "http2"] }
```

Add the new crate to `[workspace.members]`:

```toml
    "crates/harmony-mail-discovery",
```

And register its path (alongside the other `harmony-*` entries):

```toml
harmony-mail-discovery = { path = "crates/harmony-mail-discovery", default-features = false }
```

- [ ] **Step 2: Create the crate Cargo.toml**

Write `crates/harmony-mail-discovery/Cargo.toml`:

```toml
[package]
name = "harmony-mail-discovery"
description = "DNS + HTTPS discovery-backed email→identity resolution for harmony-mail"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[[bin]]
name = "harmony-mail-discovery-debug"
path = "bin/harmony-mail-discovery-debug.rs"
required-features = ["test-support"]

[features]
default = []
test-support = []

[dependencies]
harmony-crypto = { workspace = true, features = ["std"] }
harmony-identity = { workspace = true, features = ["std"] }
ed25519-dalek.workspace = true
ciborium = { workspace = true, features = ["std"] }
sha2 = { workspace = true, features = ["std"] }
tokio = { workspace = true, features = ["rt", "time", "sync", "macros"] }
async-trait = { workspace = true }
dashmap = { workspace = true }
thiserror.workspace = true
tracing.workspace = true
hickory-resolver = { workspace = true }
reqwest = { workspace = true }
base64 = { workspace = true }
hex.workspace = true

[dev-dependencies]
tokio = { workspace = true, features = ["rt-multi-thread", "macros", "test-util"] }
rand_core = { workspace = true, features = ["std", "getrandom"] }
```

- [ ] **Step 3: Create the lib.rs skeleton**

Write `crates/harmony-mail-discovery/src/lib.rs`:

```rust
//! Discovery-backed email → `IdentityHash` resolution for harmony-mail.
//!
//! See `docs/superpowers/specs/2026-04-15-smtp-rcpt-admission-design.md`
//! for the complete design. Module ordering mirrors the cryptographic
//! dependency order: `claim` is leaf (pure verification), `cache`,
//! `dns`, `http` compose onto it, `resolver` orchestrates.

#![deny(clippy::unwrap_used, clippy::expect_used)]
#![forbid(unsafe_code)]

pub mod cache;
pub mod claim;
pub mod dns;
pub mod http;
pub mod resolver;

#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
```

Create empty module files so the crate compiles:

```bash
touch crates/harmony-mail-discovery/src/{cache,claim,dns,http,resolver}.rs
```

- [ ] **Step 4: Verify the crate compiles**

Run: `cargo check -p harmony-mail-discovery`
Expected: clean compile (0 warnings, 0 errors).

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/harmony-mail-discovery/
git commit -m "feat(mail-discovery): scaffold crate (ZEB-120)"
```

---

## Task 2: `claim.rs` — type definitions + canonical CBOR helper

**Files:**
- Modify: `crates/harmony-mail-discovery/src/claim.rs`

- [ ] **Step 1: Write the failing canonical-CBOR roundtrip test**

Append to `crates/harmony-mail-discovery/src/claim.rs`:

```rust
use serde::{Deserialize, Serialize};

// Re-exports for the test below will fail compilation until we define
// the types. That's the TDD red step — proceed to step 2.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_record_roundtrips_via_canonical_cbor() {
        let rec = DomainRecord {
            version: 1,
            master_pubkey: MasterPubkey::Ed25519([7u8; 32]),
            domain_salt: [0x5au8; 16],
            alg: SignatureAlg::Ed25519,
        };
        let bytes = canonical_cbor(&rec).expect("encode");
        let decoded: DomainRecord = ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, rec);
    }
}
```

Run: `cargo test -p harmony-mail-discovery --lib claim::tests::domain_record_roundtrips_via_canonical_cbor`
Expected: compile error (`DomainRecord`, `MasterPubkey`, `SignatureAlg`, `canonical_cbor` undefined).

- [ ] **Step 2: Define the four payload types and the signature/pubkey enums**

Replace the file content with:

```rust
//! Signed-claim types and sans-I/O verification logic. See spec §4 and §5.1.

use std::fmt;

use serde::{Deserialize, Serialize};

/// Every byte range covered by a signature is the canonical-CBOR encoding
/// of the payload struct. Signers encode *once* and keep that byte slice
/// until the signature is produced; verifiers re-encode and compare.
pub fn canonical_cbor<T: Serialize>(value: &T) -> Result<Vec<u8>, ciborium::ser::Error<std::io::Error>> {
    let mut buf = Vec::new();
    ciborium::ser::into_writer(value, &mut buf)?;
    Ok(buf)
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MasterPubkey {
    Ed25519([u8; 32]),
    // MlDsa65(Box<[u8; 1952]>) — reserved; see spec §2.1, §4.1.
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SigningPubkey {
    Ed25519([u8; 32]),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signature {
    Ed25519([u8; 64]),
    // MlDsa65(Box<[u8; 3309]>), Hybrid(Box<HybridSignature>) — reserved.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignatureAlg {
    Ed25519,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DomainRecord {
    pub version: u8,
    pub master_pubkey: MasterPubkey,
    pub domain_salt: [u8; 16],
    pub alg: SignatureAlg,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SigningKeyCert {
    pub version: u8,
    pub signing_key_id: [u8; 8],
    pub signing_pubkey: SigningPubkey,
    pub valid_from: u64,
    pub valid_until: u64,
    pub domain: String,
    pub master_signature: Signature,
}

/// Signable view of `SigningKeyCert`: everything the master key signs.
/// The `master_signature` field is excluded; encoding produces the exact
/// byte range a verifier re-encodes and hands to `Ed25519::verify`.
#[derive(Debug, Serialize)]
pub struct SigningKeyCertSignable<'a> {
    pub version: u8,
    pub signing_key_id: [u8; 8],
    pub signing_pubkey: &'a SigningPubkey,
    pub valid_from: u64,
    pub valid_until: u64,
    pub domain: &'a str,
}

impl SigningKeyCert {
    pub fn signable(&self) -> SigningKeyCertSignable<'_> {
        SigningKeyCertSignable {
            version: self.version,
            signing_key_id: self.signing_key_id,
            signing_pubkey: &self.signing_pubkey,
            valid_from: self.valid_from,
            valid_until: self.valid_until,
            domain: &self.domain,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ClaimPayload {
    pub version: u8,
    pub domain: String,
    pub hashed_local_part: [u8; 32],
    pub email: String,
    pub identity_hash: [u8; 16],
    pub issued_at: u64,
    pub expires_at: u64,
    pub serial: u64,
    pub signing_key_id: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedClaim {
    pub payload: ClaimPayload,
    pub cert: SigningKeyCert,
    pub claim_signature: Signature,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationList {
    pub version: u8,
    pub domain: String,
    pub issued_at: u64,
    pub revoked_certs: Vec<SigningKeyCert>,
    pub master_signature: Signature,
}

#[derive(Debug, Serialize)]
pub struct RevocationListSignable<'a> {
    pub version: u8,
    pub domain: &'a str,
    pub issued_at: u64,
    pub revoked_certs: &'a [SigningKeyCert],
}

impl RevocationList {
    pub fn signable(&self) -> RevocationListSignable<'_> {
        RevocationListSignable {
            version: self.version,
            domain: &self.domain,
            issued_at: self.issued_at,
            revoked_certs: &self.revoked_certs,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_record_roundtrips_via_canonical_cbor() {
        let rec = DomainRecord {
            version: 1,
            master_pubkey: MasterPubkey::Ed25519([7u8; 32]),
            domain_salt: [0x5au8; 16],
            alg: SignatureAlg::Ed25519,
        };
        let bytes = canonical_cbor(&rec).expect("encode");
        let decoded: DomainRecord = ciborium::de::from_reader(&bytes[..]).expect("decode");
        assert_eq!(decoded, rec);
    }
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p harmony-mail-discovery --lib claim::`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/claim.rs
git commit -m "feat(mail-discovery): claim types + canonical CBOR (ZEB-120)"
```

---

## Task 3: `claim.rs` — `VerifyError` + `VerifiedBinding` + hashed-local-part helper

**Files:**
- Modify: `crates/harmony-mail-discovery/src/claim.rs`

- [ ] **Step 1: Write the failing test for `hashed_local_part`**

Append inside the existing `mod tests`:

```rust
#[test]
fn hashed_local_part_matches_spec_formula() {
    // SHA-256("alice" || 0x00 || domain_salt)
    let salt = [0x11u8; 16];
    let h = hashed_local_part("alice", &salt);
    // Precomputed expected value for this input/salt pair.
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"alice");
    hasher.update([0x00]);
    hasher.update(salt);
    let expected: [u8; 32] = hasher.finalize().into();
    assert_eq!(h, expected);
}

#[test]
fn hashed_local_part_is_case_preserving() {
    // Local-parts are case-sensitive per RFC 5321 §2.3.11. Domain is
    // lowercased before salt lookup, but local-part is not.
    let salt = [0u8; 16];
    assert_ne!(
        hashed_local_part("Alice", &salt),
        hashed_local_part("alice", &salt),
    );
}
```

- [ ] **Step 2: Implement `hashed_local_part`**

Add to the module body of `claim.rs` (above the `#[cfg(test)]`):

```rust
use sha2::{Digest, Sha256};

/// Canonical hashed_local_part per spec §4.3.
///
/// Local-part is NOT lowercased (RFC 5321 §2.3.11 defines local-parts
/// as case-sensitive). Domain casing is the caller's responsibility —
/// resolver always lowercases before handing it to anything in this
/// module.
pub fn hashed_local_part(local_part: &str, domain_salt: &[u8; 16]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(local_part.as_bytes());
    hasher.update([0x00]);
    hasher.update(domain_salt);
    hasher.finalize().into()
}
```

- [ ] **Step 3: Add `VerifiedBinding` and `VerifyError`**

Append to the module body (before `#[cfg(test)]`):

```rust
use harmony_identity::IdentityHash;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBinding {
    pub domain: String,
    pub email: String,
    pub identity_hash: IdentityHash,
    pub serial: u64,
    pub claim_expires_at: u64,
    pub signing_key_id: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum VerifyError {
    #[error("claim.domain, cert.domain, and queried_domain must all agree")]
    DomainMismatch,
    #[error("hashed_local_part does not match SHA-256(local_part || 0x00 || salt)")]
    HashedLocalPartMismatch,
    #[error("master signature over signing-key cert is invalid")]
    CertSignatureInvalid,
    #[error("claim signature under signing key is invalid")]
    ClaimSignatureInvalid,
    #[error("cert not yet valid (valid_from = {valid_from})")]
    CertNotYetValid { valid_from: u64 },
    #[error("cert expired (valid_until = {valid_until})")]
    CertExpired { valid_until: u64 },
    #[error("cert revoked at {revoked_at}")]
    CertRevoked { revoked_at: u64 },
    #[error("claim expired (expires_at = {expires_at})")]
    ClaimExpired { expires_at: u64 },
    #[error("unsupported version byte: {0}")]
    UnsupportedVersion(u8),
    #[error("unsupported signature algorithm")]
    UnsupportedAlgorithm,
    #[error("canonical CBOR encoding failed (unexpected)")]
    EncodingFailed,
}

/// View over the revocation cache passed into `verify`. The resolver
/// fills this from its revocation cache; tests use a trivial empty or
/// hand-built view.
#[derive(Debug, Default, Clone)]
pub struct RevocationView {
    /// Map of `signing_key_id -> (revoked_at = cert.valid_until)`. All
    /// certs in this view have `valid_until` in the past (by
    /// construction — see `RevocationList::revoked_certs` spec §4.4).
    pub revoked: std::collections::HashMap<[u8; 8], u64>,
}

impl RevocationView {
    pub fn empty() -> Self {
        Self::default()
    }
    pub fn insert(&mut self, signing_key_id: [u8; 8], revoked_at: u64) {
        self.revoked.insert(signing_key_id, revoked_at);
    }
}
```

- [ ] **Step 4: Run the new tests**

Run: `cargo test -p harmony-mail-discovery --lib claim::`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail-discovery/src/claim.rs
git commit -m "feat(mail-discovery): VerifyError, VerifiedBinding, hashed_local_part (ZEB-120)"
```

---

## Task 4: `test_support.rs` — `ClaimBuilder`, keys, and helpers

This unlocks TDD for every subsequent verification test. It's built BEFORE `verify` so failing tests in Task 5 can construct realistic inputs.

**Files:**
- Modify: `crates/harmony-mail-discovery/src/test_support.rs`

- [ ] **Step 1: Write the module**

```rust
//! Test fixtures exposed to integration tests and external crates
//! (via the `test-support` Cargo feature).
//!
//! This module exists so that `harmony-mail`'s integration tests can
//! construct valid `SignedClaim`s without duplicating the signing
//! ceremony. It is NOT compiled into release binaries unless the
//! consumer explicitly enables `test-support`.

use ed25519_dalek::{Signer, SigningKey as EdSigningKey, Verifier, VerifyingKey as EdVerifyingKey};
use rand_core::{CryptoRng, RngCore};
use sha2::{Digest, Sha256};

use crate::claim::{
    canonical_cbor, hashed_local_part, ClaimPayload, DomainRecord, MasterPubkey, RevocationList,
    Signature, SignatureAlg, SignedClaim, SigningKeyCert, SigningPubkey,
};

/// A master keypair bound to a test domain. Mirrors spec §2.1.
pub struct TestDomain {
    pub domain: String,
    pub salt: [u8; 16],
    pub master_sk: EdSigningKey,
    pub master_pk: EdVerifyingKey,
}

impl TestDomain {
    pub fn new(rng: &mut (impl RngCore + CryptoRng), domain: impl Into<String>) -> Self {
        let master_sk = EdSigningKey::generate(rng);
        let master_pk = master_sk.verifying_key();
        let mut salt = [0u8; 16];
        rng.fill_bytes(&mut salt);
        Self {
            domain: domain.into(),
            salt,
            master_sk,
            master_pk,
        }
    }

    pub fn record(&self) -> DomainRecord {
        DomainRecord {
            version: 1,
            master_pubkey: MasterPubkey::Ed25519(self.master_pk.to_bytes()),
            domain_salt: self.salt,
            alg: SignatureAlg::Ed25519,
        }
    }

    pub fn mint_signing_key(
        &self,
        rng: &mut (impl RngCore + CryptoRng),
        valid_from: u64,
        valid_until: u64,
    ) -> TestSigningKey {
        let signing_sk = EdSigningKey::generate(rng);
        let signing_pk = signing_sk.verifying_key();
        let mut key_id = [0u8; 8];
        rng.fill_bytes(&mut key_id);
        let cert = SigningKeyCert {
            version: 1,
            signing_key_id: key_id,
            signing_pubkey: SigningPubkey::Ed25519(signing_pk.to_bytes()),
            valid_from,
            valid_until,
            domain: self.domain.clone(),
            master_signature: Signature::Ed25519([0u8; 64]), // placeholder
        };
        let bytes = canonical_cbor(&cert.signable()).expect("encode cert signable");
        let sig = self.master_sk.sign(&bytes);
        let cert = SigningKeyCert {
            master_signature: Signature::Ed25519(sig.to_bytes()),
            ..cert
        };
        TestSigningKey {
            cert,
            signing_sk,
        }
    }

    pub fn revocation_list(
        &self,
        issued_at: u64,
        revoked_certs: Vec<SigningKeyCert>,
    ) -> RevocationList {
        let list = RevocationList {
            version: 1,
            domain: self.domain.clone(),
            issued_at,
            revoked_certs,
            master_signature: Signature::Ed25519([0u8; 64]),
        };
        let bytes = canonical_cbor(&list.signable()).expect("encode rev-list");
        let sig = self.master_sk.sign(&bytes);
        RevocationList {
            master_signature: Signature::Ed25519(sig.to_bytes()),
            ..list
        }
    }
}

pub struct TestSigningKey {
    pub cert: SigningKeyCert,
    pub signing_sk: EdSigningKey,
}

impl TestSigningKey {
    /// Turn this cert into a revocation cert by setting `valid_until`
    /// to `revoked_at` and re-signing under the master key.
    pub fn revoke(mut self, master_sk: &EdSigningKey, revoked_at: u64) -> SigningKeyCert {
        self.cert.valid_until = revoked_at;
        self.cert.master_signature = Signature::Ed25519([0u8; 64]);
        let bytes = canonical_cbor(&self.cert.signable()).expect("encode cert signable");
        let sig = master_sk.sign(&bytes);
        self.cert.master_signature = Signature::Ed25519(sig.to_bytes());
        self.cert
    }
}

/// Fluent builder for `SignedClaim`. Defaults to a valid claim bound
/// to `TestDomain` + `TestSigningKey`; tests mutate fields then call
/// `build` (which re-signs under the signing key).
pub struct ClaimBuilder<'a> {
    domain: &'a TestDomain,
    sk: &'a TestSigningKey,
    email: String,
    identity_hash: [u8; 16],
    issued_at: u64,
    expires_at: u64,
    serial: u64,
    payload_version: u8,
    override_hashed_local_part: Option<[u8; 32]>,
    override_payload_domain: Option<String>,
    tamper_sig: bool,
}

impl<'a> ClaimBuilder<'a> {
    pub fn new(domain: &'a TestDomain, sk: &'a TestSigningKey, now: u64) -> Self {
        Self {
            domain,
            sk,
            email: format!("alice@{}", domain.domain),
            identity_hash: [0x11; 16],
            issued_at: now,
            expires_at: now + 7 * 86_400,
            serial: 1,
            payload_version: 1,
            override_hashed_local_part: None,
            override_payload_domain: None,
            tamper_sig: false,
        }
    }

    pub fn email(mut self, e: impl Into<String>) -> Self { self.email = e.into(); self }
    pub fn identity_hash(mut self, h: [u8; 16]) -> Self { self.identity_hash = h; self }
    pub fn issued_at(mut self, t: u64) -> Self { self.issued_at = t; self }
    pub fn expires_at(mut self, t: u64) -> Self { self.expires_at = t; self }
    pub fn serial(mut self, s: u64) -> Self { self.serial = s; self }
    pub fn payload_version(mut self, v: u8) -> Self { self.payload_version = v; self }
    pub fn hashed_local_part_override(mut self, h: [u8; 32]) -> Self {
        self.override_hashed_local_part = Some(h); self
    }
    pub fn payload_domain_override(mut self, d: impl Into<String>) -> Self {
        self.override_payload_domain = Some(d.into()); self
    }
    pub fn tamper_claim_signature(mut self) -> Self { self.tamper_sig = true; self }

    pub fn build(self) -> SignedClaim {
        let local_part = self.email.split('@').next().unwrap_or("").to_string();
        let h = self.override_hashed_local_part.unwrap_or_else(|| {
            hashed_local_part(&local_part, &self.domain.salt)
        });
        let payload = ClaimPayload {
            version: self.payload_version,
            domain: self.override_payload_domain.unwrap_or_else(|| self.domain.domain.clone()),
            hashed_local_part: h,
            email: self.email,
            identity_hash: self.identity_hash,
            issued_at: self.issued_at,
            expires_at: self.expires_at,
            serial: self.serial,
            signing_key_id: self.sk.cert.signing_key_id,
        };
        let bytes = canonical_cbor(&payload).expect("encode payload");
        let sig = self.sk.signing_sk.sign(&bytes);
        let mut sig_bytes = sig.to_bytes();
        if self.tamper_sig {
            sig_bytes[0] ^= 0xff;
        }
        SignedClaim {
            payload,
            cert: self.sk.cert.clone(),
            claim_signature: Signature::Ed25519(sig_bytes),
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-mail-discovery --features test-support`
Expected: clean compile.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/test_support.rs
git commit -m "feat(mail-discovery): test_support with ClaimBuilder and test fixtures (ZEB-120)"
```

---

## Task 5: `claim.rs` — `SignedClaim::verify` happy path (TDD)

**Files:**
- Modify: `crates/harmony-mail-discovery/src/claim.rs`

- [ ] **Step 1: Add the feature-gated test-support import to claim.rs tests**

Update the `#[cfg(test)] mod tests` block to pull test_support into scope:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{ClaimBuilder, TestDomain};
    use rand_core::OsRng;

    const NOW: u64 = 2_000_000_000;
    const TOLERANCE: u64 = 60;
    // ... existing tests above ...
}
```

- [ ] **Step 2: Write the failing happy-path test**

Add to the `tests` module:

```rust
#[test]
fn verify_accepts_well_formed_fresh_claim() {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    let sk = d.mint_signing_key(&mut rng, NOW - 1000, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();

    let binding = claim
        .verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE)
        .expect("verify");

    assert_eq!(binding.domain, "q8.fyi");
    assert_eq!(binding.email, "alice@q8.fyi");
    assert_eq!(binding.identity_hash.as_slice(), &[0x11; 16]);
    assert_eq!(binding.serial, 1);
    assert_eq!(binding.signing_key_id, sk.cert.signing_key_id);
}
```

Run: `cargo test -p harmony-mail-discovery --features test-support --lib claim::tests::verify_accepts_well_formed_fresh_claim`
Expected: FAIL (no method `verify` on `SignedClaim`).

- [ ] **Step 3: Implement the full verification algorithm**

Add to claim.rs (above `#[cfg(test)]`):

```rust
use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};

impl SignedClaim {
    /// Spec §4.3 verification algorithm. `queried_domain` must be the
    /// domain the resolver intended to look up — prevents reflect
    /// attacks where a signed claim for `attacker.example` is served
    /// in response to a query for `q8.fyi`.
    pub fn verify(
        &self,
        domain_record: &DomainRecord,
        revocations: &RevocationView,
        now: u64,
        clock_skew_tolerance_secs: u64,
    ) -> Result<VerifiedBinding, VerifyError> {
        self.verify_against(&self.payload.domain, domain_record, revocations, now, clock_skew_tolerance_secs)
    }

    /// Same as `verify`, but the caller supplies the queried domain
    /// explicitly. Used by the resolver, which knows what domain it
    /// queried. External callers that trust the claim's self-reported
    /// domain can use `verify` directly.
    pub fn verify_against(
        &self,
        queried_domain: &str,
        domain_record: &DomainRecord,
        revocations: &RevocationView,
        now: u64,
        tolerance: u64,
    ) -> Result<VerifiedBinding, VerifyError> {
        // 1. Domain agreement.
        if self.payload.domain != queried_domain || self.cert.domain != queried_domain {
            return Err(VerifyError::DomainMismatch);
        }

        // 2. Version bytes.
        if self.payload.version != 1 {
            return Err(VerifyError::UnsupportedVersion(self.payload.version));
        }
        if self.cert.version != 1 {
            return Err(VerifyError::UnsupportedVersion(self.cert.version));
        }

        // 3. hashed_local_part consistency.
        let local_part = self
            .payload
            .email
            .split('@')
            .next()
            .unwrap_or("");
        let computed = hashed_local_part(local_part, &domain_record.domain_salt);
        if computed != self.payload.hashed_local_part {
            return Err(VerifyError::HashedLocalPartMismatch);
        }

        // 4. Master signature over the cert.
        let master_vk = match domain_record.master_pubkey {
            MasterPubkey::Ed25519(bytes) => {
                VerifyingKey::from_bytes(&bytes).map_err(|_| VerifyError::CertSignatureInvalid)?
            }
        };
        let cert_bytes = canonical_cbor(&self.cert.signable())
            .map_err(|_| VerifyError::EncodingFailed)?;
        let master_sig = match self.cert.master_signature {
            Signature::Ed25519(s) => EdSignature::from_bytes(&s),
        };
        master_vk
            .verify(&cert_bytes, &master_sig)
            .map_err(|_| VerifyError::CertSignatureInvalid)?;

        // 5. Cert validity window (with tolerance).
        if now + tolerance < self.cert.valid_from {
            return Err(VerifyError::CertNotYetValid { valid_from: self.cert.valid_from });
        }
        if now > self.cert.valid_until.saturating_add(tolerance) {
            return Err(VerifyError::CertExpired { valid_until: self.cert.valid_until });
        }

        // 6. Revocation check with grandfathering.
        if let Some(&revoked_at) = revocations.revoked.get(&self.cert.signing_key_id) {
            if self.payload.issued_at > revoked_at {
                return Err(VerifyError::CertRevoked { revoked_at });
            }
            // Grandfathered: issued_at <= revoked_at; continue.
        }

        // 7. Claim signature.
        let signing_vk = match self.cert.signing_pubkey {
            SigningPubkey::Ed25519(bytes) => {
                VerifyingKey::from_bytes(&bytes).map_err(|_| VerifyError::ClaimSignatureInvalid)?
            }
        };
        let payload_bytes = canonical_cbor(&self.payload)
            .map_err(|_| VerifyError::EncodingFailed)?;
        let claim_sig = match self.claim_signature {
            Signature::Ed25519(s) => EdSignature::from_bytes(&s),
        };
        signing_vk
            .verify(&payload_bytes, &claim_sig)
            .map_err(|_| VerifyError::ClaimSignatureInvalid)?;

        // 8. Claim expiry.
        if now > self.payload.expires_at.saturating_add(tolerance) {
            return Err(VerifyError::ClaimExpired { expires_at: self.payload.expires_at });
        }

        Ok(VerifiedBinding {
            domain: self.payload.domain.clone(),
            email: self.payload.email.clone(),
            identity_hash: IdentityHash::from(self.payload.identity_hash),
            serial: self.payload.serial,
            claim_expires_at: self.payload.expires_at,
            signing_key_id: self.cert.signing_key_id,
        })
    }
}
```

Note: `IdentityHash::from([u8; 16])` — verify the actual constructor API in `harmony-identity`. If the type is a tuple struct or requires a named constructor (e.g. `IdentityHash::new(bytes)` or `IdentityHash::from_bytes(bytes)`), substitute accordingly. Run `cargo check` and follow the compiler's guidance.

- [ ] **Step 4: Run the test**

Run: `cargo test -p harmony-mail-discovery --features test-support --lib claim::tests::verify_accepts_well_formed_fresh_claim`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail-discovery/src/claim.rs
git commit -m "feat(mail-discovery): SignedClaim::verify happy path (ZEB-120)"
```

---

## Task 6: `claim.rs` — negative-path verification tests (one per `VerifyError`)

**Files:**
- Modify: `crates/harmony-mail-discovery/src/claim.rs`

- [ ] **Step 1: Add all negative-path tests in one block**

Append to the `tests` module:

```rust
fn fresh(now: u64) -> (TestDomain, crate::test_support::TestSigningKey) {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    let sk = d.mint_signing_key(&mut rng, now - 1000, now + 90 * 86_400);
    (d, sk)
}

#[test]
fn rejects_domain_mismatch_between_payload_and_cert() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW)
        .payload_domain_override("attacker.example")
        .build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert_eq!(err, VerifyError::DomainMismatch);
}

#[test]
fn rejects_queried_domain_mismatch_on_verify_against() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    let err = claim
        .verify_against("other.example", &d.record(), &RevocationView::empty(), NOW, TOLERANCE)
        .unwrap_err();
    assert_eq!(err, VerifyError::DomainMismatch);
}

#[test]
fn rejects_hashed_local_part_mismatch() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW)
        .hashed_local_part_override([0xaau8; 32])
        .build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert_eq!(err, VerifyError::HashedLocalPartMismatch);
}

#[test]
fn rejects_tampered_claim_signature() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW).tamper_claim_signature().build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert_eq!(err, VerifyError::ClaimSignatureInvalid);
}

#[test]
fn rejects_cert_not_yet_valid() {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    // valid_from in the far future, outside tolerance.
    let sk = d.mint_signing_key(&mut rng, NOW + 10_000, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert!(matches!(err, VerifyError::CertNotYetValid { .. }), "{err:?}");
}

#[test]
fn rejects_cert_expired() {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    let sk = d.mint_signing_key(&mut rng, NOW - 200_000, NOW - 100_000);
    let claim = ClaimBuilder::new(&d, &sk, NOW - 150_000).build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert!(matches!(err, VerifyError::CertExpired { .. }), "{err:?}");
}

#[test]
fn rejects_claim_expired() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW)
        .issued_at(NOW - 10 * 86_400)
        .expires_at(NOW - 3 * 86_400) // 3 days past, outside tolerance
        .build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert!(matches!(err, VerifyError::ClaimExpired { .. }), "{err:?}");
}

#[test]
fn rejects_revoked_cert_when_claim_issued_after_revocation() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW).issued_at(NOW).build();
    let mut revocations = RevocationView::empty();
    // Revoked 1000 seconds before claim issuance.
    revocations.insert(sk.cert.signing_key_id, NOW - 1000);
    let err = claim.verify(&d.record(), &revocations, NOW, TOLERANCE).unwrap_err();
    assert!(matches!(err, VerifyError::CertRevoked { .. }), "{err:?}");
}

#[test]
fn grandfathers_claim_issued_before_revocation() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW).issued_at(NOW - 10_000).build();
    let mut revocations = RevocationView::empty();
    revocations.insert(sk.cert.signing_key_id, NOW - 1000); // revoked AFTER claim issuance
    let binding = claim.verify(&d.record(), &revocations, NOW, TOLERANCE).expect("grandfathered");
    assert_eq!(binding.signing_key_id, sk.cert.signing_key_id);
}

#[test]
fn rejects_unsupported_claim_version() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW).payload_version(99).build();
    let err = claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).unwrap_err();
    assert_eq!(err, VerifyError::UnsupportedVersion(99));
}

#[test]
fn tolerance_allows_60s_clock_skew_on_cert_future() {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    // valid_from is 30s in the future — inside 60s tolerance.
    let sk = d.mint_signing_key(&mut rng, NOW + 30, NOW + 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW + 30).build();
    claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).expect("accepts with tolerance");
}

#[test]
fn tolerance_allows_60s_clock_skew_on_claim_expiry() {
    let (d, sk) = fresh(NOW);
    let claim = ClaimBuilder::new(&d, &sk, NOW - 100)
        .expires_at(NOW - 30) // expired 30s ago
        .build();
    claim.verify(&d.record(), &RevocationView::empty(), NOW, TOLERANCE).expect("accepts with tolerance");
}
```

- [ ] **Step 2: Run the full claim test suite**

Run: `cargo test -p harmony-mail-discovery --features test-support --lib claim::`
Expected: ALL PASS (including new 12 negative tests + 3 positive from earlier).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/claim.rs
git commit -m "test(mail-discovery): exhaustive VerifyError coverage (ZEB-120)"
```

---

## Task 7: `cache.rs` — `TimeSource`, `CacheEntry`, and `ResolverCaches` (TDD)

**Files:**
- Modify: `crates/harmony-mail-discovery/src/cache.rs`

- [ ] **Step 1: Write the failing test file**

Write `crates/harmony-mail-discovery/src/cache.rs`:

```rust
//! TTL + LRU caches for the resolver (spec §5.2).
//!
//! Four positive caches (claim, signing-key, master-key, revocation),
//! one negative cache, and one "domain last seen" sliding-window cache
//! for the 72h soft-fail. Time is injected via `TimeSource` so tests
//! control expiry with `FakeTimeSource::advance`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Clock abstraction. Production uses `SystemTimeSource`; tests use
/// `FakeTimeSource` (see `test_support`).
pub trait TimeSource: Send + Sync + 'static {
    fn now(&self) -> u64;
}

pub struct SystemTimeSource;

impl TimeSource for SystemTimeSource {
    fn now(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Cheap, monotonic counter used as an LRU-ish recency token. Not a
/// real LRU — we only need "evict oldest N when over bound," which a
/// monotonic counter + sort-and-drop achieves without a doubly-linked
/// list. Spec §7.8 allows coarse-grained eviction since the attack
/// model is "bound memory under adversarial input," not "serve hot
/// items preferentially."
#[derive(Debug)]
pub struct RecencyCounter(AtomicU64);

impl RecencyCounter {
    pub fn new() -> Self {
        Self(AtomicU64::new(1))
    }
    pub fn tick(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for RecencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: u64,
    pub recency: u64,
}

impl<T> CacheEntry<T> {
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::FakeTimeSource;

    #[test]
    fn cache_entry_is_expired_at_boundary() {
        let entry = CacheEntry {
            value: 42u32,
            expires_at: 100,
            recency: 1,
        };
        assert!(!entry.is_expired(99));
        assert!(entry.is_expired(100));
        assert!(entry.is_expired(101));
    }

    #[test]
    fn fake_time_source_advances() {
        let t = FakeTimeSource::new(1000);
        assert_eq!(t.now(), 1000);
        t.advance(500);
        assert_eq!(t.now(), 1500);
    }
}
```

- [ ] **Step 2: Add `FakeTimeSource` to `test_support.rs`**

Append to `crates/harmony-mail-discovery/src/test_support.rs`:

```rust
use std::sync::atomic::{AtomicU64, Ordering};

use crate::cache::TimeSource;

#[derive(Debug)]
pub struct FakeTimeSource(AtomicU64);

impl FakeTimeSource {
    pub fn new(start: u64) -> Self {
        Self(AtomicU64::new(start))
    }
    pub fn advance(&self, secs: u64) {
        self.0.fetch_add(secs, Ordering::Relaxed);
    }
    pub fn set(&self, secs: u64) {
        self.0.store(secs, Ordering::Relaxed);
    }
}

impl TimeSource for FakeTimeSource {
    fn now(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p harmony-mail-discovery --features test-support --lib cache::`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/cache.rs crates/harmony-mail-discovery/src/test_support.rs
git commit -m "feat(mail-discovery): TimeSource + CacheEntry primitives (ZEB-120)"
```

---

## Task 8: `cache.rs` — `ResolverCaches` with TTL, LRU bound, and eviction

**Files:**
- Modify: `crates/harmony-mail-discovery/src/cache.rs`

- [ ] **Step 1: Write failing tests for insert/get/TTL expiry/LRU bound**

Append to `cache.rs` (inside `mod tests`):

```rust
#[test]
fn claim_cache_inserts_and_retrieves() {
    let caches = ResolverCaches::new(CacheLimits::default());
    let key = ("q8.fyi".to_string(), [0x11u8; 32]);
    caches.put_claim(key.clone(), "claim-bytes-here".to_string(), 1000, 100);
    let got = caches.get_claim(&key, 500).expect("hit");
    assert_eq!(got, "claim-bytes-here");
}

#[test]
fn claim_cache_ttl_expiry_is_honored() {
    let caches = ResolverCaches::new(CacheLimits::default());
    let key = ("q8.fyi".to_string(), [0x11u8; 32]);
    caches.put_claim(key.clone(), "v".to_string(), 1000, 100);
    assert!(caches.get_claim(&key, 999).is_some());
    assert!(caches.get_claim(&key, 1000).is_none(), "expires_at is exclusive");
    assert!(caches.get_claim(&key, 1500).is_none());
}

#[test]
fn claim_cache_lru_bound_evicts_oldest() {
    let limits = CacheLimits { claim_max: 3, ..CacheLimits::default() };
    let caches = ResolverCaches::new(limits);
    for i in 0..5u8 {
        let key = ("q8.fyi".to_string(), [i; 32]);
        caches.put_claim(key, format!("v{i}"), 10_000, i as u64);
    }
    // After inserting 5, only the 3 most recent (i=2,3,4) should remain.
    assert!(caches.get_claim(&("q8.fyi".to_string(), [0; 32]), 100).is_none());
    assert!(caches.get_claim(&("q8.fyi".to_string(), [1; 32]), 100).is_none());
    assert!(caches.get_claim(&("q8.fyi".to_string(), [2; 32]), 100).is_some());
    assert!(caches.get_claim(&("q8.fyi".to_string(), [4; 32]), 100).is_some());
}

#[test]
fn negative_cache_expires_after_60s() {
    let caches = ResolverCaches::new(CacheLimits::default());
    let key = ("q8.fyi".to_string(), [0xff; 32]);
    caches.mark_negative(key.clone(), 1000 + 60);
    assert!(caches.is_negative(&key, 1050));
    assert!(!caches.is_negative(&key, 1060));
}

#[test]
fn domain_last_seen_supports_72h_soft_fail_query() {
    let caches = ResolverCaches::new(CacheLimits::default());
    caches.mark_domain_seen("q8.fyi", 1000);
    assert!(caches.was_domain_seen_within("q8.fyi", 1000 + 3600, 72 * 3600));
    assert!(caches.was_domain_seen_within("q8.fyi", 1000 + 72 * 3600 - 1, 72 * 3600));
    assert!(!caches.was_domain_seen_within("q8.fyi", 1000 + 72 * 3600, 72 * 3600));
}

#[test]
fn sweep_evicts_expired_entries_across_caches() {
    let caches = ResolverCaches::new(CacheLimits::default());
    caches.put_claim(("a".into(), [1; 32]), "v".into(), 50, 1);
    caches.put_claim(("b".into(), [2; 32]), "w".into(), 500, 2);
    caches.sweep_expired(100);
    assert!(caches.get_claim(&("a".into(), [1; 32]), 100).is_none());
    assert!(caches.get_claim(&("b".into(), [2; 32]), 100).is_some());
}
```

Run: expect compile errors — `ResolverCaches`, `CacheLimits`, etc., are undefined.

- [ ] **Step 2: Implement `ResolverCaches`**

Add to `cache.rs`:

```rust
use std::collections::HashMap;

use dashmap::DashMap;

use crate::claim::{DomainRecord, RevocationView, SignedClaim, SigningKeyCert};

#[derive(Debug, Clone)]
pub struct CacheLimits {
    pub claim_max: usize,
    pub signing_key_max: usize,
    pub master_key_max: usize,
    pub revocation_max: usize,
    pub negative_max: usize,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            claim_max: 10_000,
            signing_key_max: 10_000,
            master_key_max: 10_000,
            revocation_max: 10_000,
            negative_max: 10_000,
        }
    }
}

pub type HashedLocalPart = [u8; 32];
pub type SigningKeyId = [u8; 8];

pub struct ResolverCaches {
    limits: CacheLimits,
    recency: RecencyCounter,
    // Positive caches — value is stored in a CacheEntry that tracks
    // expires_at (set by caller to the correct TTL: claim.expires_at,
    // cert.valid_until, DNS TTL, now+6h respectively).
    claim: DashMap<(String, HashedLocalPart), CacheEntry<SignedClaim>>,
    signing_key: DashMap<(String, SigningKeyId), CacheEntry<SigningKeyCert>>,
    master_key: DashMap<String, CacheEntry<DomainRecord>>,
    revocation: DashMap<String, CacheEntry<RevocationView>>,
    // Sliding window: last time we saw this domain resolve successfully.
    domain_last_seen: DashMap<String, u64>,
    // Negative cache: key -> expires_at.
    negative: DashMap<(String, HashedLocalPart), u64>,
    // Track last successful revocation refresh per domain for 24h safety valve.
    revocation_last_refreshed: DashMap<String, u64>,
}

impl ResolverCaches {
    pub fn new(limits: CacheLimits) -> Self {
        Self {
            limits,
            recency: RecencyCounter::new(),
            claim: DashMap::new(),
            signing_key: DashMap::new(),
            master_key: DashMap::new(),
            revocation: DashMap::new(),
            domain_last_seen: DashMap::new(),
            negative: DashMap::new(),
            revocation_last_refreshed: DashMap::new(),
        }
    }

    // --- Claim cache ---
    pub fn put_claim(&self, key: (String, HashedLocalPart), value: SignedClaim, expires_at: u64, _recency_hint: u64) {
        let entry = CacheEntry { value, expires_at, recency: self.recency.tick() };
        self.claim.insert(key, entry);
        self.enforce_bound(&self.claim, self.limits.claim_max);
    }
    pub fn get_claim(&self, key: &(String, HashedLocalPart), now: u64) -> Option<SignedClaim> {
        let entry = self.claim.get(key)?;
        if entry.is_expired(now) { return None; }
        Some(entry.value.clone())
    }

    // --- Signing-key cache ---
    pub fn put_signing_key(&self, key: (String, SigningKeyId), cert: SigningKeyCert, expires_at: u64) {
        let entry = CacheEntry { value: cert, expires_at, recency: self.recency.tick() };
        self.signing_key.insert(key, entry);
        self.enforce_bound(&self.signing_key, self.limits.signing_key_max);
    }
    pub fn get_signing_key(&self, key: &(String, SigningKeyId), now: u64) -> Option<SigningKeyCert> {
        let e = self.signing_key.get(key)?;
        if e.is_expired(now) { return None; }
        Some(e.value.clone())
    }

    // --- Master-key (DomainRecord) cache ---
    pub fn put_master_key(&self, domain: &str, rec: DomainRecord, expires_at: u64) {
        let entry = CacheEntry { value: rec, expires_at, recency: self.recency.tick() };
        self.master_key.insert(domain.to_ascii_lowercase(), entry);
        self.enforce_bound(&self.master_key, self.limits.master_key_max);
    }
    pub fn get_master_key(&self, domain: &str, now: u64) -> Option<DomainRecord> {
        let e = self.master_key.get(&domain.to_ascii_lowercase())?;
        if e.is_expired(now) { return None; }
        Some(e.value.clone())
    }

    // --- Revocation cache ---
    pub fn put_revocation(&self, domain: &str, view: RevocationView, expires_at: u64, last_refreshed: u64) {
        let entry = CacheEntry { value: view, expires_at, recency: self.recency.tick() };
        let d = domain.to_ascii_lowercase();
        self.revocation.insert(d.clone(), entry);
        self.revocation_last_refreshed.insert(d, last_refreshed);
        self.enforce_bound(&self.revocation, self.limits.revocation_max);
    }
    /// Returns the cached revocation view WHETHER OR NOT it's past
    /// expires_at — the resolver uses expires_at to decide when to
    /// refresh, and the 24h safety valve to decide when to stop serving.
    pub fn get_revocation(&self, domain: &str) -> Option<(RevocationView, /*expires_at*/ u64, /*last_refreshed*/ u64)> {
        let d = domain.to_ascii_lowercase();
        let e = self.revocation.get(&d)?;
        let last_refreshed = self.revocation_last_refreshed.get(&d).map(|v| *v).unwrap_or(0);
        Some((e.value.clone(), e.expires_at, last_refreshed))
    }

    // --- Domain last-seen (72h soft-fail window) ---
    pub fn mark_domain_seen(&self, domain: &str, now: u64) {
        self.domain_last_seen.insert(domain.to_ascii_lowercase(), now);
    }
    pub fn was_domain_seen_within(&self, domain: &str, now: u64, window_secs: u64) -> bool {
        self.domain_last_seen
            .get(&domain.to_ascii_lowercase())
            .map(|last| now < last.saturating_add(window_secs))
            .unwrap_or(false)
    }

    // --- Negative cache ---
    pub fn mark_negative(&self, key: (String, HashedLocalPart), expires_at: u64) {
        self.negative.insert(key, expires_at);
        self.enforce_bound(&self.negative, self.limits.negative_max);
    }
    pub fn is_negative(&self, key: &(String, HashedLocalPart), now: u64) -> bool {
        self.negative.get(key).map(|e| now < *e).unwrap_or(false)
    }

    // --- Sweep ---
    pub fn sweep_expired(&self, now: u64) {
        self.claim.retain(|_, e| !e.is_expired(now));
        self.signing_key.retain(|_, e| !e.is_expired(now));
        self.master_key.retain(|_, e| !e.is_expired(now));
        // revocation cache is NOT swept here — resolver decides staleness
        // via the 24h safety valve, not by entry expires_at.
        self.negative.retain(|_, expires_at| now < *expires_at);
    }

    fn enforce_bound<K, V>(&self, map: &DashMap<K, CacheEntry<V>>, max: usize)
    where
        K: std::hash::Hash + Eq + Clone,
    {
        if map.len() <= max { return; }
        // Collect (recency, key) pairs, sort by recency ascending, drop
        // the oldest until within bound. Coarse but bounded.
        let mut pairs: Vec<_> = map.iter().map(|e| (e.value().recency, e.key().clone())).collect();
        pairs.sort_by_key(|(r, _)| *r);
        let excess = map.len().saturating_sub(max);
        for (_, k) in pairs.into_iter().take(excess) {
            map.remove(&k);
        }
    }
}
```

- [ ] **Step 3: Run the tests**

Run: `cargo test -p harmony-mail-discovery --features test-support --lib cache::`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/cache.rs
git commit -m "feat(mail-discovery): ResolverCaches with TTL + LRU bound (ZEB-120)"
```

---

## Task 9: `dns.rs` — `DnsClient` trait + TXT parser

**Files:**
- Modify: `crates/harmony-mail-discovery/src/dns.rs`

- [ ] **Step 1: Write failing parser tests**

Write `crates/harmony-mail-discovery/src/dns.rs`:

```rust
//! DNS TXT fetch + parse for `_harmony.<domain>` records (spec §4.1, §5.3).

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::claim::{DomainRecord, MasterPubkey, SignatureAlg};

#[async_trait]
pub trait DnsClient: Send + Sync + 'static {
    /// Return all TXT strings for `name`. Empty Vec means NODATA /
    /// NOERROR-no-TXT. `Err` means transient network failure.
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError>;
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DnsError {
    #[error("no such record (NXDOMAIN or NOERROR-no-TXT)")]
    NoRecord,
    #[error("transient DNS failure: {0}")]
    Transient(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DnsFetchError {
    #[error("no _harmony TXT record found")]
    NoRecord,
    #[error("malformed TXT record: {0}")]
    Malformed(String),
    #[error("transient: {0}")]
    Transient(String),
    #[error("unsupported version byte: {0}")]
    UnsupportedVersion(u8),
    #[error("multiple v=harmony1 records present")]
    MultipleRecords,
}

pub async fn fetch_domain_record(
    dns: &dyn DnsClient,
    domain: &str,
) -> Result<DomainRecord, DnsFetchError> {
    let name = format!("_harmony.{}", domain.to_ascii_lowercase());
    let txts = match dns.fetch_txt(&name).await {
        Ok(t) => t,
        Err(DnsError::NoRecord) => return Err(DnsFetchError::NoRecord),
        Err(DnsError::Transient(m)) => return Err(DnsFetchError::Transient(m)),
    };
    parse_harmony_txts(&txts)
}

pub fn parse_harmony_txts(txts: &[String]) -> Result<DomainRecord, DnsFetchError> {
    let harmony_txts: Vec<&str> = txts
        .iter()
        .map(|s| s.as_str())
        .filter(|s| s.trim_start().starts_with("v=harmony1"))
        .collect();
    match harmony_txts.len() {
        0 => Err(DnsFetchError::NoRecord),
        1 => parse_single_harmony_txt(harmony_txts[0]),
        _ => Err(DnsFetchError::MultipleRecords),
    }
}

fn parse_single_harmony_txt(txt: &str) -> Result<DomainRecord, DnsFetchError> {
    let mut version: Option<u8> = None;
    let mut k: Option<[u8; 32]> = None;
    let mut salt: Option<[u8; 16]> = None;
    let mut alg: Option<SignatureAlg> = None;
    for field in txt.split(';') {
        let field = field.trim();
        if field.is_empty() { continue; }
        let (name, value) = field.split_once('=').ok_or_else(|| {
            DnsFetchError::Malformed(format!("field missing '=': {field}"))
        })?;
        match name.trim() {
            "v" => {
                if value.trim() != "harmony1" {
                    return Err(DnsFetchError::UnsupportedVersion(0));
                }
                version = Some(1);
            }
            "k" => {
                let bytes = URL_SAFE_NO_PAD.decode(value.trim()).map_err(|e| {
                    DnsFetchError::Malformed(format!("k base64url: {e}"))
                })?;
                if bytes.len() != 32 {
                    return Err(DnsFetchError::Malformed(format!("k length {} != 32", bytes.len())));
                }
                let mut arr = [0u8; 32];
                arr.copy_from_slice(&bytes);
                k = Some(arr);
            }
            "salt" => {
                let bytes = URL_SAFE_NO_PAD.decode(value.trim()).map_err(|e| {
                    DnsFetchError::Malformed(format!("salt base64url: {e}"))
                })?;
                if bytes.len() != 16 {
                    return Err(DnsFetchError::Malformed(format!("salt length {} != 16", bytes.len())));
                }
                let mut arr = [0u8; 16];
                arr.copy_from_slice(&bytes);
                salt = Some(arr);
            }
            "alg" => match value.trim() {
                "ed25519" => alg = Some(SignatureAlg::Ed25519),
                other => return Err(DnsFetchError::Malformed(format!("unknown alg={other}"))),
            },
            _ => {} // unknown fields ignored (forward-compat)
        }
    }
    Ok(DomainRecord {
        version: version.ok_or_else(|| DnsFetchError::Malformed("missing v".into()))?,
        master_pubkey: MasterPubkey::Ed25519(k.ok_or_else(|| DnsFetchError::Malformed("missing k".into()))?),
        domain_salt: salt.ok_or_else(|| DnsFetchError::Malformed("missing salt".into()))?,
        alg: alg.ok_or_else(|| DnsFetchError::Malformed("missing alg".into()))?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_txt() -> String {
        let k = URL_SAFE_NO_PAD.encode([0xabu8; 32]);
        let salt = URL_SAFE_NO_PAD.encode([0xcd; 16]);
        format!("v=harmony1; k={k}; salt={salt}; alg=ed25519")
    }

    #[test]
    fn parses_well_formed_single_txt() {
        let rec = parse_harmony_txts(&[sample_txt()]).expect("parse");
        assert_eq!(rec.version, 1);
        assert_eq!(rec.alg, SignatureAlg::Ed25519);
        assert_eq!(rec.domain_salt, [0xcd; 16]);
    }

    #[test]
    fn rejects_multiple_harmony_records() {
        let err = parse_harmony_txts(&[sample_txt(), sample_txt()]).unwrap_err();
        assert_eq!(err, DnsFetchError::MultipleRecords);
    }

    #[test]
    fn ignores_unknown_fields_for_forward_compat() {
        let k = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let salt = URL_SAFE_NO_PAD.encode([0u8; 16]);
        let txt = format!("v=harmony1; k={k}; salt={salt}; alg=ed25519; future=xyz");
        parse_harmony_txts(&[txt]).expect("must accept unknown field");
    }

    #[test]
    fn rejects_unknown_v_version() {
        let txt = "v=harmony2; k=AA; salt=AA; alg=ed25519".to_string();
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::UnsupportedVersion(_)), "{err:?}");
    }

    #[test]
    fn rejects_missing_required_field() {
        let k = URL_SAFE_NO_PAD.encode([0u8; 32]);
        let txt = format!("v=harmony1; k={k}; alg=ed25519"); // no salt
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn rejects_malformed_base64_k() {
        let salt = URL_SAFE_NO_PAD.encode([0u8; 16]);
        let txt = format!("v=harmony1; k=!!!bad!!!; salt={salt}; alg=ed25519");
        let err = parse_harmony_txts(&[txt]).unwrap_err();
        assert!(matches!(err, DnsFetchError::Malformed(_)), "{err:?}");
    }

    #[test]
    fn ignores_non_harmony_txt_records() {
        let txt_a = "spf1 include:_spf.google.com".to_string();
        let txt_b = sample_txt();
        let rec = parse_harmony_txts(&[txt_a, txt_b]).expect("parse");
        assert_eq!(rec.version, 1);
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p harmony-mail-discovery --lib dns::`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/dns.rs
git commit -m "feat(mail-discovery): DnsClient trait + TXT parser (ZEB-120)"
```

---

## Task 10: `dns.rs` — `HickoryDnsClient`

**Files:**
- Modify: `crates/harmony-mail-discovery/src/dns.rs`

- [ ] **Step 1: Add the hickory-backed impl**

Append to `dns.rs`:

```rust
use std::time::Duration;

use hickory_resolver::config::{ResolverConfig, ResolverOpts};
use hickory_resolver::TokioAsyncResolver;

pub struct HickoryDnsClient {
    inner: TokioAsyncResolver,
}

impl HickoryDnsClient {
    /// Build with system-config defaults (honors /etc/resolv.conf on
    /// Unix). Falls back to Google DNS if system config lookup fails.
    pub fn from_system(timeout: Duration) -> Self {
        let (config, mut opts) = hickory_resolver::system_conf::read_system_conf()
            .unwrap_or_else(|_| (ResolverConfig::google(), ResolverOpts::default()));
        opts.timeout = timeout;
        Self {
            inner: TokioAsyncResolver::tokio(config, opts),
        }
    }
}

#[async_trait]
impl DnsClient for HickoryDnsClient {
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError> {
        match self.inner.txt_lookup(name).await {
            Ok(lookup) => {
                let mut out = Vec::new();
                for record in lookup.iter() {
                    // A TXT record can hold multiple <character-string>s;
                    // harmony records always fit in one, but we concat
                    // to be forgiving.
                    let joined: String = record
                        .txt_data()
                        .iter()
                        .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                        .collect();
                    out.push(joined);
                }
                Ok(out)
            }
            Err(e) => {
                use hickory_resolver::error::ResolveErrorKind;
                match e.kind() {
                    ResolveErrorKind::NoRecordsFound { .. } => Err(DnsError::NoRecord),
                    _ => Err(DnsError::Transient(e.to_string())),
                }
            }
        }
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-mail-discovery`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/dns.rs
git commit -m "feat(mail-discovery): HickoryDnsClient (ZEB-120)"
```

---

## Task 11: `http.rs` — `HttpClient` trait + URL construction + CBOR parsers

**Files:**
- Modify: `crates/harmony-mail-discovery/src/http.rs`

- [ ] **Step 1: Write the module with failing parser tests**

Write `crates/harmony-mail-discovery/src/http.rs`:

```rust
//! HTTPS fetch + parse for `.well-known/harmony-users` and
//! `.well-known/harmony-revocations` (spec §4.5, §5.4).

use async_trait::async_trait;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine;

use crate::claim::{RevocationList, SignedClaim};

/// Thin abstraction so we can inject fakes without spinning up a real
/// HTTP server. Production impl is `ReqwestHttpClient`.
#[async_trait]
pub trait HttpClient: Send + Sync + 'static {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError>;
}

#[derive(Debug, Clone)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HttpError {
    #[error("connect: {0}")]
    Connect(String),
    #[error("timeout")]
    Timeout,
    #[error("tls: {0}")]
    Tls(String),
    #[error("body too large (> cap)")]
    BodyTooLarge,
    #[error("redirect refused")]
    RedirectRefused,
    #[error("other: {0}")]
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum HttpFetchError {
    #[error("transport: {0}")]
    Transport(#[from] HttpError),
    #[error("server returned {0}")]
    Server(u16),
    #[error("malformed CBOR: {0}")]
    MalformedCbor(String),
}

#[derive(Debug)]
pub enum ClaimFetchResult {
    Found(SignedClaim),
    NotFound,
}

#[derive(Debug)]
pub enum RevocationFetchResult {
    Found(RevocationList),
    Empty, // 404 -> authoritative empty
}

pub fn claim_url(domain: &str, hashed_local_part: &[u8; 32]) -> String {
    let h = URL_SAFE_NO_PAD.encode(hashed_local_part);
    format!("https://{}/.well-known/harmony-users?h={}", domain.to_ascii_lowercase(), h)
}

pub fn revocation_url(domain: &str) -> String {
    format!("https://{}/.well-known/harmony-revocations", domain.to_ascii_lowercase())
}

pub async fn fetch_claim(
    http: &dyn HttpClient,
    domain: &str,
    hashed_local_part: &[u8; 32],
) -> Result<ClaimFetchResult, HttpFetchError> {
    let resp = http.get(&claim_url(domain, hashed_local_part)).await?;
    match resp.status {
        200 => {
            let claim: SignedClaim = ciborium::de::from_reader(&resp.body[..])
                .map_err(|e| HttpFetchError::MalformedCbor(e.to_string()))?;
            Ok(ClaimFetchResult::Found(claim))
        }
        404 => Ok(ClaimFetchResult::NotFound),
        code => Err(HttpFetchError::Server(code)),
    }
}

pub async fn fetch_revocation_list(
    http: &dyn HttpClient,
    domain: &str,
) -> Result<RevocationFetchResult, HttpFetchError> {
    let resp = http.get(&revocation_url(domain)).await?;
    match resp.status {
        200 => {
            let list: RevocationList = ciborium::de::from_reader(&resp.body[..])
                .map_err(|e| HttpFetchError::MalformedCbor(e.to_string()))?;
            Ok(RevocationFetchResult::Found(list))
        }
        404 => Ok(RevocationFetchResult::Empty),
        code => Err(HttpFetchError::Server(code)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn claim_url_uses_base64url_no_pad() {
        let h = [0xffu8; 32];
        let url = claim_url("q8.fyi", &h);
        assert!(url.starts_with("https://q8.fyi/.well-known/harmony-users?h="));
        assert!(!url.contains('='), "must use base64url no-pad: {url}");
    }

    #[test]
    fn revocation_url_is_well_known_path() {
        assert_eq!(revocation_url("Q8.fyi"), "https://q8.fyi/.well-known/harmony-revocations");
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p harmony-mail-discovery --lib http::tests`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/http.rs
git commit -m "feat(mail-discovery): HttpClient trait + URL builders + CBOR parsers (ZEB-120)"
```

---

## Task 12: `http.rs` — `ReqwestHttpClient`

**Files:**
- Modify: `crates/harmony-mail-discovery/src/http.rs`

- [ ] **Step 1: Add the reqwest-backed impl**

Append to `http.rs`:

```rust
use std::time::Duration;

use reqwest::redirect::Policy;

pub struct ReqwestHttpClient {
    inner: reqwest::Client,
    body_cap_bytes: usize,
}

impl ReqwestHttpClient {
    pub fn new(
        connect_timeout: Duration,
        total_timeout: Duration,
        body_cap_bytes: usize,
    ) -> Result<Self, HttpError> {
        let inner = reqwest::Client::builder()
            .redirect(Policy::none())
            .connect_timeout(connect_timeout)
            .timeout(total_timeout)
            .https_only(true)
            .build()
            .map_err(|e| HttpError::Other(e.to_string()))?;
        Ok(Self { inner, body_cap_bytes })
    }
}

#[async_trait]
impl HttpClient for ReqwestHttpClient {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
        let resp = self.inner.get(url).send().await.map_err(|e| {
            if e.is_timeout() { HttpError::Timeout }
            else if e.is_connect() { HttpError::Connect(e.to_string()) }
            else if e.is_redirect() { HttpError::RedirectRefused }
            else { HttpError::Other(e.to_string()) }
        })?;
        let status = resp.status().as_u16();
        // Bounded body read — prevents oversize responses from blowing
        // memory. reqwest doesn't expose chunked reads cleanly, so we
        // check Content-Length as a first line of defense, then cap on
        // the collected bytes as a second.
        if let Some(len) = resp.content_length() {
            if len as usize > self.body_cap_bytes {
                return Err(HttpError::BodyTooLarge);
            }
        }
        let bytes = resp.bytes().await.map_err(|e| HttpError::Other(e.to_string()))?;
        if bytes.len() > self.body_cap_bytes {
            return Err(HttpError::BodyTooLarge);
        }
        Ok(HttpResponse { status, body: bytes.to_vec() })
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-mail-discovery`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/http.rs
git commit -m "feat(mail-discovery): ReqwestHttpClient (ZEB-120)"
```

---

## Task 13: `test_support.rs` — `FakeDnsClient` + `FakeHttpClient`

**Files:**
- Modify: `crates/harmony-mail-discovery/src/test_support.rs`

- [ ] **Step 1: Append the fakes**

Append to `test_support.rs`:

```rust
use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;

use crate::dns::{DnsClient, DnsError};
use crate::http::{HttpClient, HttpError, HttpResponse};

/// Scripted DNS responder. Keys are fully-qualified names
/// (e.g. `_harmony.q8.fyi`). Default response is `NoRecord`.
#[derive(Default)]
pub struct FakeDnsClient {
    answers: Mutex<HashMap<String, Result<Vec<String>, DnsError>>>,
    call_count: Mutex<usize>,
}

impl FakeDnsClient {
    pub fn new() -> Self { Self::default() }
    pub fn set(&self, name: &str, answer: Result<Vec<String>, DnsError>) {
        self.answers.lock().unwrap().insert(name.to_string(), answer);
    }
    pub fn call_count(&self) -> usize { *self.call_count.lock().unwrap() }
}

#[async_trait]
impl DnsClient for FakeDnsClient {
    async fn fetch_txt(&self, name: &str) -> Result<Vec<String>, DnsError> {
        *self.call_count.lock().unwrap() += 1;
        self.answers
            .lock()
            .unwrap()
            .get(name)
            .cloned()
            .unwrap_or(Err(DnsError::NoRecord))
    }
}

/// Scripted HTTP responder keyed on exact URL. Default response is 404.
#[derive(Default)]
pub struct FakeHttpClient {
    answers: Mutex<HashMap<String, Result<HttpResponse, HttpError>>>,
    call_count: Mutex<usize>,
    calls_by_url: Mutex<HashMap<String, usize>>,
}

impl FakeHttpClient {
    pub fn new() -> Self { Self::default() }
    pub fn set(&self, url: &str, answer: Result<HttpResponse, HttpError>) {
        self.answers.lock().unwrap().insert(url.to_string(), answer);
    }
    pub fn set_not_found(&self, url: &str) {
        self.set(url, Ok(HttpResponse { status: 404, body: vec![] }));
    }
    pub fn call_count(&self) -> usize { *self.call_count.lock().unwrap() }
    pub fn call_count_for(&self, url: &str) -> usize {
        self.calls_by_url.lock().unwrap().get(url).copied().unwrap_or(0)
    }
}

#[async_trait]
impl HttpClient for FakeHttpClient {
    async fn get(&self, url: &str) -> Result<HttpResponse, HttpError> {
        *self.call_count.lock().unwrap() += 1;
        *self.calls_by_url.lock().unwrap().entry(url.to_string()).or_default() += 1;
        self.answers
            .lock()
            .unwrap()
            .get(url)
            .cloned()
            .unwrap_or(Ok(HttpResponse { status: 404, body: vec![] }))
    }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-mail-discovery --features test-support --tests`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/test_support.rs
git commit -m "test(mail-discovery): FakeDnsClient + FakeHttpClient (ZEB-120)"
```

---

## Task 14: `resolver.rs` — types, trait, and config

**Files:**
- Modify: `crates/harmony-mail-discovery/src/resolver.rs`

- [ ] **Step 1: Write the skeleton**

Write `crates/harmony-mail-discovery/src/resolver.rs`:

```rust
//! `EmailResolver` trait + `DefaultEmailResolver` orchestration
//! (spec §5.5 + §6).

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use harmony_identity::IdentityHash;
use tokio::sync::{Mutex, Semaphore};
use tokio::task::JoinHandle;
use tracing::{debug, info, warn, error};

use crate::cache::{CacheLimits, ResolverCaches, TimeSource};
use crate::claim::{hashed_local_part, DomainRecord, RevocationView, SignedClaim, VerifyError};
use crate::dns::{DnsClient, DnsFetchError};
use crate::http::{
    ClaimFetchResult, HttpClient, HttpFetchError, RevocationFetchResult,
};

/// Contract: translate `local_part@domain` to a Harmony `IdentityHash`
/// (or explain why not).
#[async_trait]
pub trait EmailResolver: Send + Sync + 'static {
    async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveOutcome {
    Resolved(IdentityHash),
    DomainDoesNotParticipate,
    UserUnknown,
    Transient { reason: &'static str },
    Revoked,
}

#[derive(Debug, Clone)]
pub struct ResolverConfig {
    pub soft_fail_window_secs: u64,
    pub negative_cache_secs: u64,
    pub revocation_refresh_secs: u64,
    pub revocation_safety_valve_secs: u64,
    pub clock_skew_tolerance_secs: u64,
    pub dns_ttl_default_secs: u64,
    pub cache_limits: CacheLimits,
    pub background_sweep_interval: Duration,
}

impl Default for ResolverConfig {
    fn default() -> Self {
        Self {
            soft_fail_window_secs: 72 * 3600,
            negative_cache_secs: 60,
            revocation_refresh_secs: 6 * 3600,
            revocation_safety_valve_secs: 24 * 3600,
            clock_skew_tolerance_secs: 60,
            dns_ttl_default_secs: 3600,
            cache_limits: CacheLimits::default(),
            background_sweep_interval: Duration::from_secs(15 * 60),
        }
    }
}

pub struct DefaultEmailResolver {
    dns: Arc<dyn DnsClient>,
    http: Arc<dyn HttpClient>,
    caches: Arc<ResolverCaches>,
    time: Arc<dyn TimeSource>,
    config: ResolverConfig,
    // Per-domain concurrency cap for revocation refresh (spec §7.8).
    revocation_refresh_locks: Arc<dashmap::DashMap<String, Arc<Semaphore>>>,
    background_task: Mutex<Option<JoinHandle<()>>>,
}

impl DefaultEmailResolver {
    pub fn new(
        dns: Arc<dyn DnsClient>,
        http: Arc<dyn HttpClient>,
        time: Arc<dyn TimeSource>,
        config: ResolverConfig,
    ) -> Self {
        let caches = Arc::new(ResolverCaches::new(config.cache_limits.clone()));
        Self {
            dns,
            http,
            caches,
            time,
            config,
            revocation_refresh_locks: Arc::new(dashmap::DashMap::new()),
            background_task: Mutex::new(None),
        }
    }

    pub fn caches(&self) -> &Arc<ResolverCaches> { &self.caches }
    pub fn config(&self) -> &ResolverConfig { &self.config }
    pub fn time(&self) -> &Arc<dyn TimeSource> { &self.time }
}
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p harmony-mail-discovery --features test-support`
Expected: clean.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/src/resolver.rs
git commit -m "feat(mail-discovery): EmailResolver trait + config + skeleton (ZEB-120)"
```

---

## Task 15: `resolver.rs` — cold-path resolution (DNS + claim fetch + verify)

**Files:**
- Modify: `crates/harmony-mail-discovery/src/resolver.rs`
- Create: `crates/harmony-mail-discovery/tests/resolver_integration.rs`

- [ ] **Step 1: Write the failing cold-path test**

Write `crates/harmony-mail-discovery/tests/resolver_integration.rs`:

```rust
//! Component tests for `DefaultEmailResolver` with injected fakes.
//! See spec §8.2.

use std::sync::Arc;

use harmony_mail_discovery::cache::TimeSource;
use harmony_mail_discovery::claim::canonical_cbor;
use harmony_mail_discovery::http::HttpResponse;
use harmony_mail_discovery::resolver::{
    DefaultEmailResolver, EmailResolver, ResolveOutcome, ResolverConfig,
};
use harmony_mail_discovery::test_support::{
    ClaimBuilder, FakeDnsClient, FakeHttpClient, FakeTimeSource, TestDomain,
};
use rand_core::OsRng;

const NOW: u64 = 2_000_000_000;

fn build_dns_txt(d: &TestDomain) -> String {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;
    let harmony_mail_discovery::claim::MasterPubkey::Ed25519(k) = d.record().master_pubkey;
    let k = URL_SAFE_NO_PAD.encode(k);
    let salt = URL_SAFE_NO_PAD.encode(d.salt);
    format!("v=harmony1; k={k}; salt={salt}; alg=ed25519")
}

fn setup() -> (TestDomain, Arc<FakeDnsClient>, Arc<FakeHttpClient>, Arc<FakeTimeSource>, DefaultEmailResolver) {
    let mut rng = OsRng;
    let d = TestDomain::new(&mut rng, "q8.fyi");
    let dns = Arc::new(FakeDnsClient::new());
    let http = Arc::new(FakeHttpClient::new());
    let time = Arc::new(FakeTimeSource::new(NOW));
    let resolver = DefaultEmailResolver::new(
        dns.clone(),
        http.clone(),
        time.clone(),
        ResolverConfig::default(),
    );
    (d, dns, http, time, resolver)
}

#[tokio::test]
async fn cold_path_resolves_and_caches() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).identity_hash([0x42; 16]).build();

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    let claim_url = harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part);
    http.set(&claim_url, Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));

    let out = resolver.resolve("alice", "q8.fyi").await;
    let harmony_identity::IdentityHash(bytes) = match &out {
        ResolveOutcome::Resolved(h) => *h,
        other => panic!("expected Resolved, got {other:?}"),
    };
    assert_eq!(bytes, [0x42; 16]);

    // Subsequent call: zero network.
    let pre_dns = dns.call_count();
    let pre_http = http.call_count();
    let _ = resolver.resolve("alice", "q8.fyi").await;
    assert_eq!(dns.call_count(), pre_dns, "no new DNS calls on cache hit");
    assert_eq!(http.call_count(), pre_http, "no new HTTP calls on cache hit");
}
```

Note: `IdentityHash` destructuring — verify the type's actual shape (`IdentityHash(pub [u8; 16])` or a named accessor like `.as_bytes()`). If the tuple destructure doesn't work, substitute the correct accessor pattern.

Run: `cargo test -p harmony-mail-discovery --features test-support --test resolver_integration cold_path_resolves_and_caches`
Expected: FAIL — `resolve` not yet implemented.

- [ ] **Step 2: Implement cold path in resolver.rs**

Append to `crates/harmony-mail-discovery/src/resolver.rs`:

```rust
#[async_trait]
impl EmailResolver for DefaultEmailResolver {
    async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome {
        let domain = domain.to_ascii_lowercase();
        let now = self.time.now();

        // 1. Get or fetch the domain record.
        let domain_record = match self.caches.get_master_key(&domain, now) {
            Some(rec) => rec,
            None => match crate::dns::fetch_domain_record(self.dns.as_ref(), &domain).await {
                Ok(rec) => {
                    self.caches.put_master_key(&domain, rec.clone(), now.saturating_add(self.config.dns_ttl_default_secs));
                    rec
                }
                Err(DnsFetchError::NoRecord) => {
                    if self.caches.was_domain_seen_within(&domain, now, self.config.soft_fail_window_secs) {
                        return ResolveOutcome::Transient { reason: "dns_no_record_soft_fail" };
                    }
                    return ResolveOutcome::DomainDoesNotParticipate;
                }
                Err(DnsFetchError::UnsupportedVersion(_)) => return ResolveOutcome::DomainDoesNotParticipate,
                Err(DnsFetchError::MultipleRecords) => {
                    return ResolveOutcome::Transient { reason: "dns_multiple_records" };
                }
                Err(DnsFetchError::Malformed(_)) => {
                    if self.caches.was_domain_seen_within(&domain, now, self.config.soft_fail_window_secs) {
                        return ResolveOutcome::Transient { reason: "dns_malformed" };
                    }
                    return ResolveOutcome::DomainDoesNotParticipate;
                }
                Err(DnsFetchError::Transient(_)) => {
                    return ResolveOutcome::Transient { reason: "dns_error" };
                }
            },
        };

        // 2. Compute cache key.
        let h = hashed_local_part(local_part, &domain_record.domain_salt);
        let cache_key = (domain.clone(), h);

        // 3. Negative-cache short-circuit.
        if self.caches.is_negative(&cache_key, now) {
            return ResolveOutcome::UserUnknown;
        }

        // 4. Ensure we have a usable revocation view (bootstrap closed).
        let revocations = match self.ensure_revocation_view(&domain, now).await {
            Ok(view) => view,
            Err(reason) => return ResolveOutcome::Transient { reason },
        };

        // 5. Claim cache hit?
        let claim = match self.caches.get_claim(&cache_key, now) {
            Some(cached) => cached,
            None => match crate::http::fetch_claim(self.http.as_ref(), &domain, &h).await {
                Ok(ClaimFetchResult::Found(c)) => c,
                Ok(ClaimFetchResult::NotFound) => {
                    self.caches.mark_negative(cache_key.clone(), now.saturating_add(self.config.negative_cache_secs));
                    return ResolveOutcome::UserUnknown;
                }
                Err(HttpFetchError::MalformedCbor(_)) => {
                    return ResolveOutcome::Transient { reason: "claim_parse" };
                }
                Err(HttpFetchError::Server(_)) | Err(HttpFetchError::Transport(_)) => {
                    return ResolveOutcome::Transient { reason: "http_error" };
                }
            },
        };

        // 6. Verify.
        match claim.verify_against(&domain, &domain_record, &revocations, now, self.config.clock_skew_tolerance_secs) {
            Ok(binding) => {
                self.caches.put_claim(cache_key, claim.clone(), claim.payload.expires_at, binding.serial);
                self.caches.put_signing_key(
                    (domain.clone(), claim.cert.signing_key_id),
                    claim.cert.clone(),
                    claim.cert.valid_until,
                );
                self.caches.mark_domain_seen(&domain, now);
                ResolveOutcome::Resolved(binding.identity_hash)
            }
            Err(VerifyError::CertRevoked { .. }) => ResolveOutcome::Revoked,
            Err(e) => {
                warn!(error = %e, %domain, local_part = %local_part, "claim verification failed");
                self.caches.mark_negative((domain.clone(), h), now.saturating_add(self.config.negative_cache_secs));
                ResolveOutcome::Transient { reason: verify_err_reason(&e) }
            }
        }
    }
}

fn verify_err_reason(e: &VerifyError) -> &'static str {
    match e {
        VerifyError::DomainMismatch => "claim_domain_mismatch",
        VerifyError::HashedLocalPartMismatch => "claim_lp_mismatch",
        VerifyError::CertSignatureInvalid => "cert_sig",
        VerifyError::ClaimSignatureInvalid => "claim_sig",
        VerifyError::CertNotYetValid { .. } => "cert_future",
        VerifyError::CertExpired { .. } => "cert_expired",
        VerifyError::CertRevoked { .. } => "cert_revoked",
        VerifyError::ClaimExpired { .. } => "claim_expired",
        VerifyError::UnsupportedVersion(_) => "unsupported_version",
        VerifyError::UnsupportedAlgorithm => "unsupported_alg",
        VerifyError::EncodingFailed => "encoding_failed",
    }
}

impl DefaultEmailResolver {
    /// Fetch or reuse the revocation view for `domain`. Spec §7.4:
    /// "No cached list ever + fetch fails" is `Transient` — fail
    /// closed on first-ever bootstrap. Subsequent refreshes leave the
    /// previous cache in place on failure.
    async fn ensure_revocation_view(
        &self,
        domain: &str,
        now: u64,
    ) -> Result<RevocationView, &'static str> {
        if let Some((view, expires_at, last_refreshed)) = self.caches.get_revocation(domain) {
            // 24h safety valve: if we haven't refreshed successfully in >24h, fail.
            if now.saturating_sub(last_refreshed) > self.config.revocation_safety_valve_secs {
                return Err("revocation_stale");
            }
            if now < expires_at {
                return Ok(view);
            }
            // Expired: try to refresh; on failure, keep serving previous.
            match self.refresh_revocation_view(domain, now).await {
                Ok(fresh) => Ok(fresh),
                Err(_) => Ok(view),
            }
        } else {
            // No prior cache — bootstrap-closed.
            self.refresh_revocation_view(domain, now).await.map_err(|e| e)
        }
    }

    async fn refresh_revocation_view(
        &self,
        domain: &str,
        now: u64,
    ) -> Result<RevocationView, &'static str> {
        // Per-domain concurrency cap.
        let sem = self
            .revocation_refresh_locks
            .entry(domain.to_string())
            .or_insert_with(|| Arc::new(Semaphore::new(1)))
            .clone();
        let _permit = sem.acquire_owned().await.map_err(|_| "revocation_bootstrap_failed")?;

        // Get master pubkey for verifying the list.
        let Some(domain_record) = self.caches.get_master_key(domain, now) else {
            return Err("revocation_bootstrap_failed");
        };

        match crate::http::fetch_revocation_list(self.http.as_ref(), domain).await {
            Ok(RevocationFetchResult::Found(list)) => {
                // Verify master signature over the list.
                if let Err(reason) = verify_revocation_list(&list, &domain_record) {
                    error!(%domain, %reason, "revocation list signature failed — potential attack");
                    return Err("revocation_bootstrap_failed");
                }
                let view = build_revocation_view(&list);
                self.caches.put_revocation(
                    domain,
                    view.clone(),
                    now.saturating_add(self.config.revocation_refresh_secs),
                    now,
                );
                Ok(view)
            }
            Ok(RevocationFetchResult::Empty) => {
                let view = RevocationView::empty();
                self.caches.put_revocation(
                    domain,
                    view.clone(),
                    now.saturating_add(self.config.revocation_refresh_secs),
                    now,
                );
                Ok(view)
            }
            Err(_) => Err("revocation_bootstrap_failed"),
        }
    }
}

fn verify_revocation_list(
    list: &crate::claim::RevocationList,
    domain_record: &DomainRecord,
) -> Result<(), &'static str> {
    use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};
    let crate::claim::MasterPubkey::Ed25519(k) = domain_record.master_pubkey;
    let vk = VerifyingKey::from_bytes(&k).map_err(|_| "master_key_parse")?;
    let bytes = crate::claim::canonical_cbor(&list.signable()).map_err(|_| "encoding_failed")?;
    let crate::claim::Signature::Ed25519(sig_bytes) = list.master_signature;
    let sig = EdSignature::from_bytes(&sig_bytes);
    vk.verify(&bytes, &sig).map_err(|_| "master_sig")?;
    Ok(())
}

fn build_revocation_view(list: &crate::claim::RevocationList) -> RevocationView {
    let mut view = RevocationView::empty();
    for cert in &list.revoked_certs {
        view.insert(cert.signing_key_id, cert.valid_until);
    }
    view
}
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p harmony-mail-discovery --features test-support --test resolver_integration`
Expected: PASS (the one cold-path test).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/resolver.rs crates/harmony-mail-discovery/tests/resolver_integration.rs
git commit -m "feat(mail-discovery): DefaultEmailResolver cold path (ZEB-120)"
```

---

## Task 16: `resolver.rs` — failure-mapping tests (DNS, HTTP, soft-fail)

**Files:**
- Modify: `crates/harmony-mail-discovery/tests/resolver_integration.rs`

- [ ] **Step 1: Add all failure-mapping tests**

Append to `resolver_integration.rs`:

```rust
use harmony_mail_discovery::dns::DnsError;
use harmony_mail_discovery::http::HttpError;

#[tokio::test]
async fn dns_nxdomain_returns_domain_does_not_participate() {
    let (_d, dns, _http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Err(DnsError::NoRecord));
    assert_eq!(
        resolver.resolve("alice", "q8.fyi").await,
        ResolveOutcome::DomainDoesNotParticipate,
    );
}

#[tokio::test]
async fn dns_nxdomain_within_soft_fail_window_returns_transient() {
    let (d, dns, http, time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    // First, succeed once so the domain is "seen".
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    assert!(matches!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Resolved(_)));

    // Now expire the master-key cache and flip DNS to NXDOMAIN.
    time.advance(2 * 3600); // past 1h DNS TTL
    dns.set("_harmony.q8.fyi", Err(DnsError::NoRecord));
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => {
            assert_eq!(reason, "dns_no_record_soft_fail");
        }
        other => panic!("expected Transient soft-fail, got {other:?}"),
    }
}

#[tokio::test]
async fn dns_timeout_returns_transient() {
    let (_d, dns, _http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Err(DnsError::Transient("timeout".into())));
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "dns_error"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn http_404_returns_user_unknown_and_is_negative_cached() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    // Claim 404 — but we still need revocation list to bootstrap.
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    // The claim URL is for "alice" so we don't set it — default is 404.
    let out = resolver.resolve("alice", "q8.fyi").await;
    assert_eq!(out, ResolveOutcome::UserUnknown);

    let pre_http = http.call_count();
    let _ = resolver.resolve("alice", "q8.fyi").await;
    // Re-query hits negative cache, no new HTTP call for the claim URL.
    assert_eq!(http.call_count(), pre_http, "negative cache short-circuits");
}

#[tokio::test]
async fn http_500_returns_transient() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    let claim_url = harmony_mail_discovery::http::claim_url(&d.domain, &[0u8; 32]);
    let _ = claim_url; // suppress unused if test is skipped
    // Any claim URL for this domain: return 500.
    // FakeHttpClient matches on exact URL, so we inject via "all URLs not otherwise mapped return default (404)".
    // For 500, we have to know the exact hash — compute it:
    let h = harmony_mail_discovery::claim::hashed_local_part("alice", &d.salt);
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &h),
        Ok(HttpResponse { status: 500, body: vec![] }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "http_error"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn http_malformed_cbor_returns_transient() {
    let (d, dns, http, _time, resolver) = setup();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));
    let h = harmony_mail_discovery::claim::hashed_local_part("alice", &d.salt);
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &h),
        Ok(HttpResponse { status: 200, body: vec![0xff, 0xff, 0xff] }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "claim_parse"),
        other => panic!("expected Transient, got {other:?}"),
    }
}

#[tokio::test]
async fn revocation_bootstrap_failure_fails_closed() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();
    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    // Revocation URL returns 5xx → no prior cache → fail closed.
    http.set(
        &harmony_mail_discovery::http::revocation_url(&d.domain),
        Ok(HttpResponse { status: 500, body: vec![] }),
    );
    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "revocation_bootstrap_failed"),
        other => panic!("expected bootstrap-closed Transient, got {other:?}"),
    }
}
```

- [ ] **Step 2: Run the tests**

Run: `cargo test -p harmony-mail-discovery --features test-support --test resolver_integration`
Expected: PASS (all tests).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/tests/resolver_integration.rs
git commit -m "test(mail-discovery): DNS/HTTP/soft-fail failure mapping (ZEB-120)"
```

---

## Task 17: `resolver.rs` — revocation behavior + serial rollback tests

**Files:**
- Modify: `crates/harmony-mail-discovery/tests/resolver_integration.rs`
- Modify: `crates/harmony-mail-discovery/src/resolver.rs`

- [ ] **Step 1: Write failing tests for revoked claim + grandfathering + serial rollback**

Append to `resolver_integration.rs`:

```rust
#[tokio::test]
async fn claim_signed_by_revoked_cert_returns_revoked() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    // Signing key that will be revoked.
    let sk = d.mint_signing_key(&mut rng, NOW - 10_000, NOW + 90 * 86_400);
    // Claim issued AFTER revocation time -> rejected.
    let claim = ClaimBuilder::new(&d, &sk, NOW).issued_at(NOW).build();
    let revocation = d.revocation_list(
        NOW,
        vec![sk.revoke(&d.master_sk, NOW - 100)],
    );

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    http.set(
        &harmony_mail_discovery::http::revocation_url(&d.domain),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&revocation).unwrap() }),
    );

    assert_eq!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Revoked);
}

#[tokio::test]
async fn claim_issued_before_revocation_is_grandfathered() {
    let (d, dns, http, _time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 20_000, NOW + 90 * 86_400);
    // Claim issued 10_000s ago — before revocation at NOW - 5_000.
    let claim = ClaimBuilder::new(&d, &sk, NOW).issued_at(NOW - 10_000).build();
    let revocation = d.revocation_list(
        NOW,
        vec![sk.revoke(&d.master_sk, NOW - 5_000)],
    );

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    http.set(
        &harmony_mail_discovery::http::revocation_url(&d.domain),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&revocation).unwrap() }),
    );

    assert!(matches!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Resolved(_)));
}

#[tokio::test]
async fn claim_serial_rollback_on_refetch_returns_transient() {
    let (d, dns, http, time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);

    // First claim: serial=5, expires soon so the cache key turns over.
    let claim_high = ClaimBuilder::new(&d, &sk, NOW).serial(5).expires_at(NOW + 60).build();
    // Second claim: serial=3 (rollback!), fresh.
    let claim_low = ClaimBuilder::new(&d, &sk, NOW).serial(3).expires_at(NOW + 10_000).build();

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    let claim_url = harmony_mail_discovery::http::claim_url(&d.domain, &claim_high.payload.hashed_local_part);
    http.set(&claim_url, Ok(HttpResponse { status: 200, body: canonical_cbor(&claim_high).unwrap() }));
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));

    assert!(matches!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Resolved(_)));

    // Expire the claim cache, serve a lower-serial claim.
    time.advance(120);
    http.set(&claim_url, Ok(HttpResponse { status: 200, body: canonical_cbor(&claim_low).unwrap() }));

    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "claim_serial_rollback"),
        other => panic!("expected serial rollback Transient, got {other:?}"),
    }
}
```

Run: the serial-rollback test will FAIL — resolver doesn't check serial yet.

- [ ] **Step 2: Add serial tracking in the resolver**

Update `DefaultEmailResolver` to track highest-seen serial per cache key. Append to `resolver.rs`:

```rust
use dashmap::DashMap;

impl DefaultEmailResolver {
    // Inside the struct, add a field (modify struct):
    // pub(crate) highest_serial: Arc<DashMap<(String, [u8; 32]), u64>>,
}
```

First, add the field to the struct definition:

```rust
pub struct DefaultEmailResolver {
    dns: Arc<dyn DnsClient>,
    http: Arc<dyn HttpClient>,
    caches: Arc<ResolverCaches>,
    time: Arc<dyn TimeSource>,
    config: ResolverConfig,
    revocation_refresh_locks: Arc<dashmap::DashMap<String, Arc<Semaphore>>>,
    background_task: Mutex<Option<JoinHandle<()>>>,
    highest_serial: Arc<DashMap<(String, [u8; 32]), u64>>,
}
```

Initialize in `new`:

```rust
highest_serial: Arc::new(DashMap::new()),
```

Modify the verify success branch in `resolve` (near line where `Ok(binding)` is produced):

```rust
Ok(binding) => {
    // Serial rollback check.
    let key = (domain.clone(), h);
    if let Some(prev) = self.highest_serial.get(&key) {
        if binding.serial < *prev {
            warn!(%domain, local_part = %local_part, prev = *prev, new = binding.serial, "claim serial rollback");
            return ResolveOutcome::Transient { reason: "claim_serial_rollback" };
        }
    }
    self.highest_serial.insert(key.clone(), binding.serial);
    self.caches.put_claim(key, claim.clone(), claim.payload.expires_at, binding.serial);
    self.caches.put_signing_key(
        (domain.clone(), claim.cert.signing_key_id),
        claim.cert.clone(),
        claim.cert.valid_until,
    );
    self.caches.mark_domain_seen(&domain, now);
    ResolveOutcome::Resolved(binding.identity_hash)
}
```

- [ ] **Step 3: Run all resolver tests**

Run: `cargo test -p harmony-mail-discovery --features test-support --test resolver_integration`
Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/resolver.rs crates/harmony-mail-discovery/tests/resolver_integration.rs
git commit -m "feat(mail-discovery): revocation + serial rollback enforcement (ZEB-120)"
```

---

## Task 18: `resolver.rs` — 24h safety valve + 72h soft-fail expiry tests

**Files:**
- Modify: `crates/harmony-mail-discovery/tests/resolver_integration.rs`

- [ ] **Step 1: Add the two remaining component tests**

Append to `resolver_integration.rs`:

```rust
#[tokio::test]
async fn stale_revocation_past_24h_returns_transient() {
    let (d, dns, http, time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 90 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));

    // First resolve succeeds, populates revocation cache with last_refreshed = NOW.
    assert!(matches!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Resolved(_)));

    // Advance past 24h. Future revocation fetches fail.
    time.advance(25 * 3600);
    http.set(
        &harmony_mail_discovery::http::revocation_url(&d.domain),
        Ok(HttpResponse { status: 500, body: vec![] }),
    );

    match resolver.resolve("alice", "q8.fyi").await {
        ResolveOutcome::Transient { reason } => assert_eq!(reason, "revocation_stale"),
        other => panic!("expected revocation_stale, got {other:?}"),
    }
}

#[tokio::test]
async fn soft_fail_expires_after_72h_and_returns_does_not_participate() {
    let (d, dns, http, time, resolver) = setup();
    let mut rng = OsRng;
    let sk = d.mint_signing_key(&mut rng, NOW - 100, NOW + 100 * 86_400);
    let claim = ClaimBuilder::new(&d, &sk, NOW).build();

    dns.set("_harmony.q8.fyi", Ok(vec![build_dns_txt(&d)]));
    http.set(
        &harmony_mail_discovery::http::claim_url(&d.domain, &claim.payload.hashed_local_part),
        Ok(HttpResponse { status: 200, body: canonical_cbor(&claim).unwrap() }),
    );
    http.set_not_found(&harmony_mail_discovery::http::revocation_url(&d.domain));

    // Establish domain as "seen".
    assert!(matches!(resolver.resolve("alice", "q8.fyi").await, ResolveOutcome::Resolved(_)));

    // Advance past 72h + past DNS TTL. Domain's seen-window has expired.
    time.advance(73 * 3600);
    dns.set("_harmony.q8.fyi", Err(DnsError::NoRecord));

    assert_eq!(
        resolver.resolve("alice", "q8.fyi").await,
        ResolveOutcome::DomainDoesNotParticipate,
    );
}
```

- [ ] **Step 2: Run all resolver tests**

Run: `cargo test -p harmony-mail-discovery --features test-support --test resolver_integration`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/tests/resolver_integration.rs
git commit -m "test(mail-discovery): 24h safety valve + 72h soft-fail expiry (ZEB-120)"
```

---

## Task 19: Background refresh task

**Files:**
- Modify: `crates/harmony-mail-discovery/src/resolver.rs`

- [ ] **Step 1: Implement `spawn_background_refresh`**

Append to `resolver.rs`:

```rust
impl DefaultEmailResolver {
    /// Spawn the background maintenance task. Safe to call multiple
    /// times — subsequent calls are no-ops. The task exits when the
    /// resolver is dropped (it holds a Weak reference to caches/time).
    pub async fn spawn_background_refresh(self: &Arc<Self>) {
        let mut guard = self.background_task.lock().await;
        if guard.is_some() { return; }

        let weak = Arc::downgrade(self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval({
                // Upgrade once to read interval, then drop.
                let Some(this) = weak.upgrade() else { return; };
                this.config.background_sweep_interval
            });
            ticker.tick().await; // immediate first tick — discard
            loop {
                ticker.tick().await;
                let Some(this) = weak.upgrade() else { break; };
                let now = this.time.now();
                this.caches.sweep_expired(now);
                // Per-domain revocation refresh scheduling: collect
                // domains whose last_refreshed + refresh_secs < now.
                let to_refresh: Vec<String> = this
                    .caches
                    .revocation_refresh_candidates(now, this.config.revocation_refresh_secs);
                for domain in to_refresh {
                    let this2 = this.clone();
                    tokio::spawn(async move {
                        let _ = this2.refresh_revocation_view(&domain, this2.time.now()).await;
                    });
                }
            }
        });
        *guard = Some(handle);
    }
}
```

- [ ] **Step 2: Add `revocation_refresh_candidates` to `ResolverCaches`**

In `cache.rs`, append:

```rust
impl ResolverCaches {
    pub fn revocation_refresh_candidates(&self, now: u64, refresh_secs: u64) -> Vec<String> {
        self.revocation_last_refreshed
            .iter()
            .filter(|e| now.saturating_sub(*e.value()) >= refresh_secs)
            .map(|e| e.key().clone())
            .collect()
    }
}
```

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p harmony-mail-discovery --features test-support`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail-discovery/src/resolver.rs crates/harmony-mail-discovery/src/cache.rs
git commit -m "feat(mail-discovery): background sweep + revocation refresh task (ZEB-120)"
```

---

## Task 20: Debug CLI binary

**Files:**
- Create: `crates/harmony-mail-discovery/bin/harmony-mail-discovery-debug.rs`

- [ ] **Step 1: Write the binary**

```rust
//! Debug CLI: resolve `user@domain` once and print the outcome.
//! Uses real DNS + HTTPS. Useful for operators verifying their
//! `.well-known/` deployment works end-to-end.
//!
//! Usage: `harmony-mail-discovery-debug alice@q8.fyi`

use std::env;
use std::sync::Arc;
use std::time::Duration;

use harmony_mail_discovery::cache::SystemTimeSource;
use harmony_mail_discovery::dns::HickoryDnsClient;
use harmony_mail_discovery::http::ReqwestHttpClient;
use harmony_mail_discovery::resolver::{DefaultEmailResolver, EmailResolver, ResolverConfig};

#[tokio::main(flavor = "current_thread")]
async fn main() {
    tracing_subscriber::fmt::init();
    let arg = env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: harmony-mail-discovery-debug <user@domain>");
        std::process::exit(2);
    });
    let (local, domain) = match arg.split_once('@') {
        Some(p) => p,
        None => {
            eprintln!("expected user@domain, got: {arg}");
            std::process::exit(2);
        }
    };

    let dns: Arc<dyn harmony_mail_discovery::dns::DnsClient> =
        Arc::new(HickoryDnsClient::from_system(Duration::from_secs(5)));
    let http: Arc<dyn harmony_mail_discovery::http::HttpClient> = Arc::new(
        ReqwestHttpClient::new(Duration::from_secs(5), Duration::from_secs(10), 1_000_000)
            .expect("build reqwest client"),
    );
    let time: Arc<dyn harmony_mail_discovery::cache::TimeSource> = Arc::new(SystemTimeSource);

    let resolver = DefaultEmailResolver::new(dns, http, time, ResolverConfig::default());
    let outcome = resolver.resolve(local, domain).await;
    println!("{outcome:?}");
}
```

- [ ] **Step 2: Build + run smoke**

Run: `cargo build -p harmony-mail-discovery --features test-support --bin harmony-mail-discovery-debug`
Expected: clean compile.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail-discovery/bin/
git commit -m "feat(mail-discovery): debug CLI binary (ZEB-120)"
```

---

## Task 21: `harmony-mail` — introduce `RemoteDeliveryContext` + refactor `run()` signature

**Files:**
- Modify: `crates/harmony-mail/Cargo.toml`
- Modify: `crates/harmony-mail/src/server.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

- [ ] **Step 1: Wire new dependency**

Add to `crates/harmony-mail/Cargo.toml` under `[dependencies]`:

```toml
harmony-mail-discovery = { workspace = true }
async-trait = { workspace = true }
```

Under `[dev-dependencies]`:

```toml
harmony-mail-discovery = { workspace = true, features = ["test-support"] }
```

- [ ] **Step 2: Define `RemoteDeliveryContext` in `server.rs`**

Near the top of `crates/harmony-mail/src/server.rs` (after the imports), add:

```rust
/// Bundle of dependencies needed for remote (non-local-domain) mail
/// delivery. Passing the three as a single struct instead of three
/// `Option`s removes the "two of three" invalid state that was easy
/// to silently hit when PR #240 landed with Options.
#[derive(Clone)]
pub struct RemoteDeliveryContext {
    pub gateway_identity: std::sync::Arc<harmony_identity::PrivateIdentity>,
    pub recipient_resolver: std::sync::Arc<dyn crate::remote_delivery::RecipientResolver>,
    pub email_resolver: std::sync::Arc<dyn harmony_mail_discovery::resolver::EmailResolver>,
}
```

- [ ] **Step 3: Change `run` signature**

Modify `crates/harmony-mail/src/server.rs` at the three `run`-like entry points (main `run`, the internal handler invocations). Replace:

```rust
gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
```

with a single:

```rust
remote_delivery: Option<RemoteDeliveryContext>,
```

Update all call sites — 8+ locations per the grep in Task 0. Each `gateway_identity.clone()` / `recipient_resolver.clone()` pair becomes `remote_delivery.as_ref().map(|ctx| (ctx.gateway_identity.clone(), ctx.recipient_resolver.clone()))` or, better, a dedicated helper:

```rust
fn split_remote(
    ctx: &Option<RemoteDeliveryContext>,
) -> (Option<Arc<harmony_identity::PrivateIdentity>>, Option<Arc<dyn crate::remote_delivery::RecipientResolver>>) {
    match ctx {
        Some(c) => (Some(c.gateway_identity.clone()), Some(c.recipient_resolver.clone())),
        None => (None, None),
    }
}
```

Keep the internal `process_async_actions` signature taking the two `Option<Arc<...>>` params for minimal diff. Only the outer `run()` changes shape. This scopes the refactor to its stated purpose without churning every internal caller.

- [ ] **Step 4: Re-export from lib.rs**

Append to `crates/harmony-mail/src/lib.rs`:

```rust
pub use server::RemoteDeliveryContext;
```

(Or whatever re-export style the lib already uses.)

- [ ] **Step 5: Verify compilation**

Run: `cargo check -p harmony-mail`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/Cargo.toml crates/harmony-mail/src/server.rs crates/harmony-mail/src/lib.rs
git commit -m "refactor(mail): collapse run() Options into RemoteDeliveryContext (ZEB-120)"
```

---

## Task 22: Update PR #240 tests for new `run()` shape

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` (tests `remote_only_250`, `resolver_miss_451`, `remote_not_configured_451`)

- [ ] **Step 1: Locate the three tests**

Run: `grep -n "remote_only_250\|resolver_miss_451\|remote_not_configured_451" crates/harmony-mail/src/server.rs`
Each test currently builds `gateway_identity`, `recipient_resolver` separately and passes both to `run()`. Update each test to construct `Some(RemoteDeliveryContext { .. })` with a stub `email_resolver` (or `None` if the test doesn't exercise remote RCPT).

- [ ] **Step 2: Add a stub `EmailResolver` helper in the test module**

In the test module at the bottom of `server.rs`:

```rust
#[cfg(test)]
struct StubEmailResolver;

#[cfg(test)]
#[async_trait::async_trait]
impl harmony_mail_discovery::resolver::EmailResolver for StubEmailResolver {
    async fn resolve(&self, _local_part: &str, _domain: &str) -> harmony_mail_discovery::resolver::ResolveOutcome {
        harmony_mail_discovery::resolver::ResolveOutcome::DomainDoesNotParticipate
    }
}
```

- [ ] **Step 3: Rewrite each test's setup block**

Replace the old two-Option construction with:

```rust
let remote_delivery = Some(RemoteDeliveryContext {
    gateway_identity: Arc::new(PrivateIdentity::generate(&mut OsRng)),
    recipient_resolver: Arc::new(/* existing stub */),
    email_resolver: Arc::new(StubEmailResolver),
});
```

And pass `remote_delivery` in place of the two `Option`s.

- [ ] **Step 4: Run the three tests**

Run: `cargo test -p harmony-mail remote_only_250 resolver_miss_451 remote_not_configured_451`
Expected: PASS.

- [ ] **Step 5: Full harmony-mail test suite sanity check**

Run: `cargo test -p harmony-mail`
Expected: all 386+ pre-existing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "test(mail): update PR #240 tests for RemoteDeliveryContext (ZEB-120)"
```

---

## Task 23: Wire `EmailResolver` into `run_async_address_resolution`

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` (around line 1107-1133)

- [ ] **Step 1: Write failing integration test for non-local RCPT admission**

This test asserts that a valid claim-backed resolve leads to `HarmonyResolved { identity: Some(_) }`. Write it as a unit test in the existing `server.rs` `#[cfg(test)] mod tests`:

```rust
#[tokio::test]
async fn non_local_rcpt_resolves_via_email_resolver() {
    use harmony_mail_discovery::resolver::ResolveOutcome;

    struct ResolvesAlice([u8; 16]);
    #[async_trait::async_trait]
    impl harmony_mail_discovery::resolver::EmailResolver for ResolvesAlice {
        async fn resolve(&self, local: &str, domain: &str) -> ResolveOutcome {
            if local == "alice" && domain == "remote.example" {
                ResolveOutcome::Resolved(harmony_identity::IdentityHash::from(self.0))
            } else {
                ResolveOutcome::UserUnknown
            }
        }
    }

    // Build a session + process_async_actions call for RCPT TO:<alice@remote.example>.
    // See existing resolver_miss_451 / remote_only_250 tests for harness pattern.
    // Assert the resulting SmtpEvent::HarmonyResolved carries identity = Some(hash).
    // TODO: mirror the harness of resolver_miss_451; this test block is the
    // canonical reference point for the shape.
    todo!("flesh out from resolver_miss_451 harness")
}
```

Actually, instead of a placeholder `todo!()`, copy the harness structure from `resolver_miss_451`. Since that test uses `process_async_actions` directly with hand-built `SmtpAction::ResolveHarmonyAddress`, mirror that exactly, substituting an `email_resolver` that returns `Resolved` for `alice@remote.example`. Keep the test assertion as: after the action processes, `resolved_recipients` contains the returned hash. (Read `resolver_miss_451` for the exact helper names.)

Run: expect FAIL — the existing `run_async_address_resolution` blanket-rejects non-local domains.

- [ ] **Step 2: Replace the blanket-reject block**

In `crates/harmony-mail/src/server.rs`, replace lines 1107-1133 (the `else` branch for non-local domains) with:

```rust
} else {
    // Non-local domain: consult the EmailResolver (ZEB-120).
    match remote_delivery {
        Some(ctx) => {
            match ctx.email_resolver.resolve(local_part, domain).await {
                harmony_mail_discovery::resolver::ResolveOutcome::Resolved(hash) => {
                    Some(hash)
                }
                harmony_mail_discovery::resolver::ResolveOutcome::UserUnknown => {
                    tracing::info!(local_part, %domain, "RCPT: user unknown at remote domain");
                    None
                }
                harmony_mail_discovery::resolver::ResolveOutcome::DomainDoesNotParticipate => {
                    tracing::info!(%domain, "RCPT: domain does not participate");
                    None
                }
                harmony_mail_discovery::resolver::ResolveOutcome::Revoked => {
                    tracing::warn!(local_part, %domain, "RCPT: binding revoked");
                    None
                }
                harmony_mail_discovery::resolver::ResolveOutcome::Transient { reason } => {
                    // Transient: we return None here, but the caller needs
                    // to distinguish transient from permanent to emit 451
                    // vs 550. Plumb this via a new `SmtpEvent::HarmonyResolveTransient`
                    // variant, or (simpler for this PR) signal through the
                    // `identity: None` path plus a side-channel state flag
                    // that the `HarmonyResolved` handler consults.
                    tracing::warn!(local_part, %domain, reason, "RCPT: transient resolver failure");
                    // TODO (see Task 24): wire transient → 451 properly.
                    None
                }
            }
        }
        None => {
            // No remote delivery context wired; behave as before
            // (blanket-reject non-local domains).
            tracing::debug!(%domain, "non-local domain, no remote delivery configured");
            None
        }
    }
};
```

Note: the `None` function argument at the outermost `run()` (before Task 21's refactor) is now `remote_delivery: &Option<RemoteDeliveryContext>` — pass it through `process_async_actions` instead of the two separate `Option<Arc<...>>` params, OR add a third parameter `email_resolver: &Option<Arc<dyn EmailResolver>>` alongside the existing two. The cleaner path: change `process_async_actions` to take `&Option<RemoteDeliveryContext>` and call a `split_remote` helper internally where it needs to hand individual Arcs to downstream logic. This is the right moment to do that consolidation since we're touching all the call sites anyway.

- [ ] **Step 3: Run the new test + the existing three**

Run: `cargo test -p harmony-mail remote_only_250 resolver_miss_451 remote_not_configured_451 non_local_rcpt_resolves_via_email_resolver`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): admit non-local RCPTs via EmailResolver (ZEB-120)"
```

---

## Task 24: Transient → 451 vs permanent → 550 signaling

**Files:**
- Modify: `crates/harmony-mail/src/smtp.rs` (add `SmtpEvent::HarmonyResolveTransient` OR a new field)
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Pick the signaling approach**

Read `crates/harmony-mail/src/smtp.rs` around the existing `SmtpEvent::HarmonyResolved` enum variant. If the state machine already distinguishes "permanent fail" (→ 550) from "transient fail" (→ 451) via `DeliveryResult { success }` or similar, mirror that pattern. Otherwise, add a new `SmtpEvent::HarmonyResolveTransient { local_part, reason }` variant that the state machine translates to a 451 response.

The spec §7 mapping requires:
- `Resolved(hash)` → 250 (via existing `HarmonyResolved { identity: Some(hash) }`)
- `UserUnknown`, `DomainDoesNotParticipate`, `Revoked` → 550 (via `HarmonyResolved { identity: None }`)
- `Transient { .. }` → 451 (NEW — needs its own signal)

- [ ] **Step 2: Add the transient SMTP event**

Add to the `SmtpEvent` enum in `smtp.rs`:

```rust
HarmonyResolveTransient {
    local_part: String,
    reason: &'static str,
},
```

Add a matching handler in the session state machine that emits the SMTP response `451 4.3.0 Temporary resolver failure (<reason>)` and leaves the recipient out of `resolved_recipients`. See the existing `HarmonyResolved` handler for the code shape.

- [ ] **Step 3: Dispatch the transient event from `run_async_address_resolution`**

Modify the `Transient { reason }` branch in `server.rs`:

```rust
harmony_mail_discovery::resolver::ResolveOutcome::Transient { reason } => {
    tracing::warn!(local_part, %domain, reason, "RCPT: transient resolver failure");
    // Emit transient event and skip the HarmonyResolved dispatch.
    let callback_actions = session.handle(SmtpEvent::HarmonyResolveTransient {
        local_part: local_part.clone(),
        reason,
    });
    execute_actions_generic(&callback_actions, writer).await?;
    continue; // skip the fall-through HarmonyResolved dispatch
}
```

Restructure the outer match so the `Transient` arm takes this path, and the other arms (`Resolved`, `UserUnknown`, etc.) continue to produce an `identity: Option<IdentityHash>` that gets dispatched via the existing `HarmonyResolved` event.

- [ ] **Step 4: Add a unit test for transient → 451**

Add to the `#[cfg(test)] mod tests` in `server.rs`:

```rust
#[tokio::test]
async fn non_local_rcpt_transient_returns_451() {
    struct TransientResolver;
    #[async_trait::async_trait]
    impl harmony_mail_discovery::resolver::EmailResolver for TransientResolver {
        async fn resolve(&self, _local: &str, _domain: &str) -> harmony_mail_discovery::resolver::ResolveOutcome {
            harmony_mail_discovery::resolver::ResolveOutcome::Transient { reason: "dns_timeout" }
        }
    }

    // (Mirror resolver_miss_451 harness, but inject TransientResolver.)
    // Assert the writer buffer contains "451".
    // Assert session has no resolved recipients.
}
```

- [ ] **Step 5: Run the new test + the full test suite**

Run: `cargo test -p harmony-mail`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/src/smtp.rs crates/harmony-mail/src/server.rs
git commit -m "feat(mail): HarmonyResolveTransient event → SMTP 451 (ZEB-120)"
```

---

## Task 25: Production `ZenohRecipientResolver`

**Files:**
- Modify: `crates/harmony-mail/src/remote_delivery.rs`

- [ ] **Step 1: Write failing test for Zenoh query → AnnounceRecord → Identity**

This is hard to unit-test without spinning up Zenoh sessions, which is slow. Instead, write one integration test in `tests/smtp_remote_delivery_integration.rs` (the existing file) that exercises `ZenohRecipientResolver::resolve` against a real session. But for TDD here, write a narrower unit test that mocks the "query result bytes" layer:

Add to the `#[cfg(test)] mod tests` in `remote_delivery.rs`:

```rust
#[test]
fn zenoh_recipient_resolver_parses_announce_record_bytes() {
    use harmony_discovery::AnnounceRecord;
    let mut rng = rand_core::OsRng;
    let priv_id = PrivateIdentity::generate(&mut rng);
    let pub_id = priv_id.public_identity();
    let pub_bytes = pub_id.to_public_bytes();
    let rec = AnnounceRecord {
        identity_ref: IdentityRef::new(pub_id.address_hash, CryptoSuite::Ed25519),
        public_key: pub_bytes[32..].to_vec(),
        encryption_key: pub_bytes[..32].to_vec(),
        routing_hints: vec![],
        published_at: 0,
        expires_at: 0,
        nonce: [0u8; 16],
        signature: vec![],
    };
    let bytes = bincode_or_whatever_the_codec_is(&rec).expect("serialize");
    // Function under test: parse_announce_record_bytes -> Identity
    let id = parse_announce_record_bytes(&bytes).expect("parse");
    assert_eq!(id.address_hash, pub_id.address_hash);
}
```

Note: check how `AnnounceRecord` is serialized over Zenoh elsewhere in the codebase (e.g., in the publisher side of the announce topic). Use the same codec. Search `crates/harmony-discovery/` for serialization calls.

- [ ] **Step 2: Implement `ZenohRecipientResolver`**

Append to `crates/harmony-mail/src/remote_delivery.rs`:

```rust
use std::sync::Arc;
use std::time::Duration;

use zenoh::Session;

/// Production `RecipientResolver`: queries the Zenoh
/// `harmony/identity/{hash_hex}/resolve` queryable namespace for an
/// `AnnounceRecord`, parses it via `identity_from_announce_record`.
///
/// `RecipientResolver::resolve` is synchronous; Zenoh queries are
/// async. Bridge with `tokio::runtime::Handle::block_on` — the trait
/// call happens from within the SMTP handler's async task, so a
/// blocking bridge is safe (tokio runs the query on a worker thread).
pub struct ZenohRecipientResolver {
    session: Arc<Session>,
    query_timeout: Duration,
}

impl ZenohRecipientResolver {
    pub fn new(session: Arc<Session>, query_timeout: Duration) -> Self {
        Self { session, query_timeout }
    }
}

impl RecipientResolver for ZenohRecipientResolver {
    fn resolve(&self, address_hash: &IdentityHash) -> Option<Identity> {
        let hex = hex::encode(address_hash.as_bytes());
        let key = format!("harmony/identity/{hex}/resolve");
        let session = self.session.clone();
        let timeout = self.query_timeout;
        let handle = tokio::runtime::Handle::try_current().ok()?;
        handle.block_on(async move {
            let replies = match tokio::time::timeout(timeout, session.get(&key).await.ok()?.into_future()).await {
                Ok(r) => r?,
                Err(_) => return None,
            };
            let bytes = replies.result().ok()?.payload().to_bytes();
            let rec: AnnounceRecord = postcard::from_bytes(&bytes).ok()?;
            identity_from_announce_record(&rec).ok()
        })
    }
}
```

Note: the exact `zenoh::Session::get` call shape, the reply iteration, and the serialization codec (`postcard` vs `bincode` vs CBOR) MUST match what the existing announcer publishes with. Check `crates/harmony-discovery/src/*.rs` and the announcer call site for the codec — substitute the correct one. Do not guess.

- [ ] **Step 3: Run the unit test**

Run: `cargo test -p harmony-mail remote_delivery::tests::zenoh_recipient_resolver_parses_announce_record_bytes`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/remote_delivery.rs
git commit -m "feat(mail): ZenohRecipientResolver via queryable (ZEB-120)"
```

---

## Task 26: `main.rs` wiring — build `RemoteDeliveryContext`

**Files:**
- Modify: `crates/harmony-mail/src/main.rs`
- Modify: `crates/harmony-mail/src/config.rs` (if the gateway identity path/format needs a config field)

- [ ] **Step 1: Identify where the gateway's PrivateIdentity comes from**

Run: `grep -n "PrivateIdentity\|gateway_identity\|identity_path" crates/harmony-mail/src/config.rs`

If there's no existing config field for the gateway identity key path, add one to the `Config` struct. Existing analogues are TLS cert paths, so mirror that pattern.

- [ ] **Step 2: Wire `RemoteDeliveryContext` construction in `Commands::Run`**

Replace the body of `Commands::Run { config: config_path }` in `main.rs` with:

```rust
Commands::Run { config: config_path } => {
    let config = harmony_mail::config::Config::from_file(Path::new(&config_path))
        .unwrap_or_else(|e| {
            eprintln!("Failed to load config from {config_path}: {e}");
            std::process::exit(1);
        });

    // Build remote-delivery context.
    let remote_delivery = match build_remote_delivery(&config).await {
        Ok(ctx) => Some(ctx),
        Err(e) => {
            eprintln!("Failed to build remote-delivery context: {e}");
            eprintln!("Proceeding without remote delivery (local-only mode).");
            None
        }
    };

    if let Err(e) = harmony_mail::server::run(config, remote_delivery).await {
        eprintln!("Server error: {e}");
        std::process::exit(1);
    }
}
```

And add the helper `async fn build_remote_delivery(config: &Config) -> Result<RemoteDeliveryContext, Box<dyn std::error::Error>>`:

```rust
async fn build_remote_delivery(
    config: &harmony_mail::config::Config,
) -> Result<harmony_mail::RemoteDeliveryContext, Box<dyn std::error::Error>> {
    use std::sync::Arc;
    use std::time::Duration;

    // Load gateway identity from config-specified key file.
    let identity_bytes = std::fs::read(&config.gateway.identity_key_path)?;
    let gateway_identity = Arc::new(harmony_identity::PrivateIdentity::from_bytes(&identity_bytes)?);

    // Zenoh session.
    let session = Arc::new(zenoh::open(zenoh::Config::default()).await?);

    // Production RecipientResolver.
    let recipient_resolver = Arc::new(harmony_mail::remote_delivery::ZenohRecipientResolver::new(
        session.clone(),
        Duration::from_secs(5),
    ));

    // Discovery-backed EmailResolver.
    let dns: Arc<dyn harmony_mail_discovery::dns::DnsClient> = Arc::new(
        harmony_mail_discovery::dns::HickoryDnsClient::from_system(Duration::from_secs(5)),
    );
    let http: Arc<dyn harmony_mail_discovery::http::HttpClient> = Arc::new(
        harmony_mail_discovery::http::ReqwestHttpClient::new(
            Duration::from_secs(5),
            Duration::from_secs(10),
            1_000_000,
        )?,
    );
    let time: Arc<dyn harmony_mail_discovery::cache::TimeSource> =
        Arc::new(harmony_mail_discovery::cache::SystemTimeSource);
    let default_resolver = Arc::new(harmony_mail_discovery::resolver::DefaultEmailResolver::new(
        dns,
        http,
        time,
        harmony_mail_discovery::resolver::ResolverConfig::default(),
    ));
    default_resolver.spawn_background_refresh().await;

    Ok(harmony_mail::RemoteDeliveryContext {
        gateway_identity,
        recipient_resolver,
        email_resolver: default_resolver,
    })
}
```

Note: the exact `PrivateIdentity::from_bytes` constructor and the config schema for `gateway.identity_key_path` must match what the rest of the codebase expects. If `Config` doesn't have a `[gateway]` section yet, extend `config.rs` with:

```rust
#[derive(Deserialize)]
pub struct GatewaySection {
    pub identity_key_path: String,
}
```

and a field on `Config`.

- [ ] **Step 3: Verify compilation**

Run: `cargo check -p harmony-mail`
Expected: clean.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/main.rs crates/harmony-mail/src/config.rs
git commit -m "feat(mail): main.rs wires RemoteDeliveryContext (ZEB-120)"
```

---

## Task 27: Integration test — full SMTP RCPT admission end-to-end

**Files:**
- Create: `crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs`

- [ ] **Step 1: Write the test harness + happy-path test**

Write `crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs`:

```rust
//! End-to-end RCPT admission integration test (spec §8.3).
//!
//! Harness:
//!   - Gateway A: full SMTP stack + RemoteDeliveryContext wired with
//!     a fake EmailResolver that resolves alice@remote.example to a
//!     known hash, plus a RecipientResolver returning Bob's Identity.
//!   - Zenoh pair (session_a, session_b) subscribes on Bob's unicast
//!     topic to capture the sealed envelope.

use std::sync::Arc;
use std::time::Duration;

use harmony_identity::{IdentityHash, PrivateIdentity};
use harmony_mail::mailbox_manager::ZenohPublisher;
use harmony_mail::remote_delivery::RecipientResolver;
use harmony_mail::server::{process_async_actions, RemoteDeliveryContext};
use harmony_mail::smtp::{SmtpAction, SmtpSession};
use harmony_mail_discovery::resolver::{EmailResolver, ResolveOutcome};
use harmony_zenoh::envelope::HarmonyEnvelope;
use harmony_zenoh::namespace::msg;
use rand_core::OsRng;

struct FakeEmailResolver {
    alice_hash: IdentityHash,
}

#[async_trait::async_trait]
impl EmailResolver for FakeEmailResolver {
    async fn resolve(&self, local: &str, domain: &str) -> ResolveOutcome {
        if local == "alice" && domain == "remote.example" {
            ResolveOutcome::Resolved(self.alice_hash)
        } else {
            ResolveOutcome::UserUnknown
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn smtp_rcpt_to_remote_domain_resolves_seals_publishes() {
    // [Mirror the harness of smtp_remote_delivery_integration.rs —
    // open sessions, subscribe Bob on msg::unicast_key(bob_hash),
    // run the SMTP RCPT TO → DATA sequence, assert sealed envelope
    // received + opens to plaintext.]
    //
    // Copy-paste the harness setup from smtp_remote_delivery_integration.rs,
    // then swap:
    //   - StubResolver (local test) for ZenohRecipientResolver (against session_a)
    //     OR keep StubResolver but add FakeEmailResolver atop it.
    //   - Drive RCPT TO:<alice@remote.example> through the smtp_parse → smtp
    //     state machine (vs. the old test which called DeliverToHarmony directly).
    //
    // Assertions:
    //   - SMTP response to RCPT TO is 250
    //   - Envelope captured on Bob's unicast topic
    //   - HarmonyEnvelope::open recovers plaintext
    //   - Plaintext parses back to HarmonyMessage with expected subject
}
```

Fill in the `// [Mirror ...]` block by reading `crates/harmony-mail/tests/smtp_remote_delivery_integration.rs` and copying the session setup, publisher, subscriber, probe convergence loop, and the DATA plaintext construction. Then replace the direct `DeliverToHarmony` call with a RCPT TO → DATA sequence driven through `smtp_parse::parse_command` + `session.handle(...)` + `process_async_actions`.

- [ ] **Step 2: Run the test**

Run: `cargo test -p harmony-mail --test smtp_remote_rcpt_admission_integration smtp_rcpt_to_remote_domain_resolves_seals_publishes`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs
git commit -m "test(mail): end-to-end RCPT admission integration (happy path, ZEB-120)"
```

---

## Task 28: Integration tests — failure/edge cases (550, 451, revoked, mixed)

**Files:**
- Modify: `crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs`

- [ ] **Step 1: Add the remaining five tests**

Append to `smtp_remote_rcpt_admission_integration.rs`. Each test reuses the harness but swaps in a different `EmailResolver` return value:

```rust
#[tokio::test]
async fn smtp_rcpt_to_unknown_user_returns_550() {
    // FakeEmailResolver returns UserUnknown for everything.
    // Drive RCPT TO:<ghost@remote.example>, assert "550" in writer buffer.
}

#[tokio::test]
async fn smtp_rcpt_to_non_participating_domain_returns_550() {
    // Returns DomainDoesNotParticipate. Assert 550.
}

#[tokio::test]
async fn smtp_rcpt_to_transient_dns_failure_returns_451() {
    // Returns Transient { reason: "dns_timeout" }. Assert 451.
}

#[tokio::test]
async fn smtp_rcpt_to_revoked_recipient_returns_550() {
    // Returns Revoked. Assert 550.
}

#[tokio::test]
async fn smtp_rcpt_to_mixed_local_and_remote_recipients_succeeds_per_recipient() {
    // One RCPT TO:<localuser@local.example> (succeeds via IMAP store).
    // One RCPT TO:<alice@remote.example> (succeeds via EmailResolver).
    // One RCPT TO:<ghost@remote.example> (fails UserUnknown -> 550 on that RCPT).
    // Assert DATA succeeds for the two accepted recipients; the sealed envelope
    // reaches Bob; the local mailbox receives the local delivery.
    // Confirms PR #240's per-recipient semantics unchanged.
}
```

- [ ] **Step 2: Run all integration tests**

Run: `cargo test -p harmony-mail --test smtp_remote_rcpt_admission_integration`
Expected: all 6 PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail/tests/smtp_remote_rcpt_admission_integration.rs
git commit -m "test(mail): RCPT admission edge cases (550/451/revoked/mixed, ZEB-120)"
```

---

## Task 29: Full workspace test + lint gate

**Files:** (no new files)

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: ALL tests pass. If any pre-existing test broke, investigate and either fix or document.

- [ ] **Step 2: Clippy clean**

Run: `cargo clippy --workspace --all-targets --features test-support -- -D warnings`
Expected: no warnings. Fix any newly-introduced lints (particularly `clippy::unwrap_used`, `clippy::expect_used`, `clippy::too_many_arguments`).

- [ ] **Step 3: Format check**

Run: `cargo fmt --all -- --check`
Expected: clean. If dirty, run `cargo fmt --all` and commit.

- [ ] **Step 4: Doc build**

Run: `cargo doc --workspace --no-deps`
Expected: clean.

- [ ] **Step 5: Commit any fixups**

```bash
git add -u
git commit -m "chore(mail-discovery): workspace lint/fmt pass (ZEB-120)" # only if anything changed
```

---

## Task 30: PR preparation

**Files:** (none)

- [ ] **Step 1: Rebase on latest origin/main**

```bash
git fetch origin
git rebase origin/main
```

Resolve any conflicts in `crates/harmony-mail/src/server.rs` or `main.rs` if concurrent PRs touched the same regions.

- [ ] **Step 2: Review diff size and split if needed**

Run: `git diff --stat origin/main..HEAD`

If the total diff is >2000 LOC or touches more than one surface area meaningfully (e.g., main.rs wiring could land as a separate ticket), consider splitting into two PRs along the `harmony-mail-discovery` ↔ `harmony-mail integration` seam. The crate scaffold through Task 20 is independently shippable and reviewable.

- [ ] **Step 3: Push and open PR**

```bash
git push -u origin zeb-120-rcpt-admission
```

Open PR against `main` with title "feat(mail): RCPT admission + discovery-backed email→hash resolution (ZEB-120)" and a summary linking the spec + research docs. Reference ZEB-113 and PR #240 as context.

---

## Self-review checklist

After completing every task, confirm the following before handoff:

- [ ] **Spec §1 (context):** scope matches — new crate, `RemoteDeliveryContext`, production `RecipientResolver`, main.rs wiring. (Covered by Tasks 1, 21, 25, 26.)
- [ ] **Spec §2 (north star / phasing):** Phase 1 ships claim + DNS + HTTPS end-to-end; Phase 2 = Zenoh is deferred (not in this plan by design).
- [ ] **Spec §3 (trust model):** 3-layer bootstrap is implemented in `resolver::resolve` — DNS-master-key → master-signed cert → signing-key-signed claim.
- [ ] **Spec §4 (wire formats):** every struct from §4.1–4.4 appears in `claim.rs` (Task 2); canonical CBOR helper in Task 2; version-byte checks in Task 5.
- [ ] **Spec §5 (components):** five `harmony-mail-discovery` modules + debug CLI + `harmony-mail` integration all covered.
- [ ] **Spec §6 (data flow):** cold path (Task 15), hot path (Task 15 assertion), background refresh (Task 19), revocation-triggered rejection (Task 17), soft-fail (Task 16, 18).
- [ ] **Spec §7 (error handling):** DNS + HTTP + verify error mappings all covered in Tasks 16-17; clock-skew tolerance in Task 5-6; serial rollback in Task 17; resource exhaustion (LRU bound) in Task 8; panic surface (clippy gate) in Task 29.
- [ ] **Spec §8 (testing):** unit (~25) + component (~17) + integration (~6) + PR #240 adjustments (~3) all present — count: Tasks 3+5+6 = ~17 claim tests; Task 7+8 = ~6 cache tests; Tasks 15-18 = ~10 component tests; Task 28 = 6 integration tests; Task 22 = 3 updates.
- [ ] **Spec §9 (migration to Phase 2):** no regressions. Phase 2 is additive — Zenoh transport added, wire formats unchanged.
- [ ] **Placeholder scan:** zero `TODO`/`TBD`/"fill in later" in production code paths. Comments referencing "mirror the harness of X" in tests are intentional pointers to existing code the worker must read.
- [ ] **Type consistency:** `RemoteDeliveryContext` struct fields match across `server.rs`, `main.rs`, and all tests (`gateway_identity`, `recipient_resolver`, `email_resolver`). `EmailResolver::resolve` signature is stable across all usages (`async fn resolve(&self, local_part: &str, domain: &str) -> ResolveOutcome`). `ResolveOutcome` variants used in tests match the enum in `resolver.rs`.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-04-15-smtp-rcpt-admission-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?

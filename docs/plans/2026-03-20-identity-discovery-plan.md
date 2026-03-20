# Identity Namespace & Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `harmony-discovery` crate for Zenoh-native identity discovery with signed announce records, queryable resolution, liveliness presence tracking, and add identity key expression patterns to `harmony-zenoh`.

**Architecture:** Sans-I/O state machine (`DiscoveryManager`) consuming events and emitting actions, with a self-signed `AnnounceRecord` type built via a builder pattern. Follows the exact patterns established by `harmony-credential` (builder, verification, serialization) and `harmony-peers` (event/action state machine).

**Tech Stack:** Rust (no_std + alloc), postcard serialization, ed25519-dalek, harmony-crypto (BLAKE3, ML-DSA-65), harmony-identity (IdentityRef, CryptoSuite), hashbrown (HashMap/HashSet)

**Design spec:** `docs/plans/2026-03-19-identity-discovery-design.md`

---

## File Structure

```
crates/harmony-discovery/
  Cargo.toml              — crate manifest
  src/
    lib.rs                — no_std setup, module declarations, re-exports
    error.rs              — DiscoveryError enum with Display + std::error::Error
    record.rs             — AnnounceRecord, RoutingHint, AnnounceBuilder, SignablePayload, serialization
    verify.rs             — verify_announce() with Ed25519/ML-DSA-65 dispatch
    manager.rs            — DiscoveryManager, DiscoveryEvent, DiscoveryAction
    resolve.rs            — OfflineResolver trait

crates/harmony-zenoh/src/
  namespace.rs            — add `pub mod identity { ... }` with key expression constants/builders
```

Also modify: `Cargo.toml` (workspace root) to add `harmony-discovery` to members list.

---

### Task 1: Crate Scaffold and Error Type

**Files:**
- Modify: `Cargo.toml` (workspace root — add member)
- Create: `crates/harmony-discovery/Cargo.toml`
- Create: `crates/harmony-discovery/src/lib.rs`
- Create: `crates/harmony-discovery/src/error.rs`

- [ ] **Step 1: Create `Cargo.toml`**

```toml
[package]
name = "harmony-discovery"
description = "Zenoh-native identity discovery with signed announce records and presence tracking"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
ed25519-dalek = { workspace = true }
hashbrown = { workspace = true }
harmony-crypto = { workspace = true, features = ["serde"] }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[dev-dependencies]
rand = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]
```

- [ ] **Step 2: Create `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;

pub use error::DiscoveryError;
```

- [ ] **Step 3: Write the error enum in `src/error.rs`**

```rust
/// Errors returned by discovery operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryError {
    Expired,
    SignatureInvalid,
    AddressMismatch,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Expired => write!(f, "announce record expired"),
            Self::SignatureInvalid => write!(f, "invalid announce signature"),
            Self::AddressMismatch => write!(f, "public key does not match identity address"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DiscoveryError {}
```

- [ ] **Step 4: Add `harmony-discovery` to workspace members in root `Cargo.toml`**

Add `"crates/harmony-discovery"` to the `[workspace] members` list (alphabetical order, after `harmony-credential`). Also add to `[workspace.dependencies]`:

```toml
harmony-discovery = { path = "crates/harmony-discovery", default-features = false }
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-discovery`
Expected: compiles with no errors

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock crates/harmony-discovery/
git commit -m "feat(discovery): scaffold harmony-discovery crate with error type"
```

---

### Task 2: AnnounceRecord, RoutingHint, and AnnounceBuilder

**Files:**
- Create: `crates/harmony-discovery/src/record.rs`
- Modify: `crates/harmony-discovery/src/lib.rs` (add module + re-exports)

**Reference:** `crates/harmony-credential/src/credential.rs` for the builder + SignablePayload + serialization pattern.

- [ ] **Step 1: Write tests for AnnounceBuilder and AnnounceRecord**

In `src/record.rs`, write the test module first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::{CryptoSuite, IdentityRef};

    fn test_identity_ref() -> IdentityRef {
        IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519)
    }

    #[test]
    fn builder_produces_correct_fields() {
        let identity_ref = test_identity_ref();
        let mut builder = AnnounceBuilder::new(
            identity_ref,
            alloc::vec![0x01; 32],
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });

        let payload = builder.signable_payload();
        let record = builder.build(payload.clone());

        assert_eq!(record.identity_ref, identity_ref);
        assert_eq!(record.public_key, alloc::vec![0x01; 32]);
        assert_eq!(record.routing_hints.len(), 1);
        assert_eq!(record.published_at, 1000);
        assert_eq!(record.expires_at, 2000);
        assert_eq!(record.nonce, [0x01; 16]);
    }

    #[test]
    fn signable_payload_is_deterministic() {
        let builder = AnnounceBuilder::new(
            test_identity_ref(),
            alloc::vec![0x01; 32],
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
        AnnounceBuilder::new(
            test_identity_ref(),
            alloc::vec![0x01; 32],
            2000,
            1000,
            [0x01; 16],
        );
    }

    #[test]
    #[should_panic(expected = "expires_at")]
    fn rejects_expires_at_equal_to_published_at() {
        AnnounceBuilder::new(
            test_identity_ref(),
            alloc::vec![0x01; 32],
            1000,
            1000,
            [0x01; 16],
        );
    }

    #[test]
    fn multiple_routing_hints() {
        let mut builder = AnnounceBuilder::new(
            test_identity_ref(),
            alloc::vec![0x01; 32],
            1000,
            2000,
            [0x01; 16],
        );
        builder
            .add_routing_hint(RoutingHint::Reticulum {
                destination_hash: [0xCC; 16],
            })
            .add_routing_hint(RoutingHint::Zenoh {
                locator: alloc::vec![0xDD; 8],
            });

        let payload = builder.signable_payload();
        let record = builder.build(payload);
        assert_eq!(record.routing_hints.len(), 2);
    }

    #[test]
    fn serde_round_trip() {
        let mut builder = AnnounceBuilder::new(
            test_identity_ref(),
            alloc::vec![0x01; 32],
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });

        let payload = builder.signable_payload();
        let record = builder.build(alloc::vec![0xDE, 0xAD]);

        let bytes = record.serialize().unwrap();
        let restored = AnnounceRecord::deserialize(&bytes).unwrap();

        assert_eq!(restored.identity_ref, record.identity_ref);
        assert_eq!(restored.public_key, record.public_key);
        assert_eq!(restored.routing_hints.len(), record.routing_hints.len());
        assert_eq!(restored.published_at, record.published_at);
        assert_eq!(restored.expires_at, record.expires_at);
        assert_eq!(restored.nonce, record.nonce);
        assert_eq!(restored.signature, record.signature);
    }

    #[test]
    fn deserialize_rejects_corrupt_data() {
        assert!(matches!(
            AnnounceRecord::deserialize(&[]),
            Err(DiscoveryError::DeserializeError("empty data"))
        ));
        assert!(matches!(
            AnnounceRecord::deserialize(&[0xFF]),
            Err(DiscoveryError::DeserializeError(
                "unsupported format version"
            ))
        ));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-discovery`
Expected: FAIL — types not defined yet

- [ ] **Step 3: Implement AnnounceRecord, RoutingHint, AnnounceBuilder**

In `src/record.rs`:

```rust
use alloc::vec::Vec;
use harmony_identity::IdentityRef;
use serde::{Deserialize, Serialize};

use crate::error::DiscoveryError;

const FORMAT_VERSION: u8 = 1;

/// Internal struct for the signable portion of an announce record.
/// Everything except the signature.
#[derive(Serialize, Deserialize)]
struct SignablePayload {
    format_version: u8,
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

/// How to reach an identity over a specific transport.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingHint {
    /// Reticulum destination hash (for Tier 1 link establishment).
    Reticulum { destination_hash: [u8; 16] },
    /// Zenoh locator (for direct Zenoh session).
    Zenoh { locator: Vec<u8> },
}

/// A signed identity announcement for Zenoh-based discovery.
///
/// Self-contained: includes the public key needed to verify the
/// signature. Any node can verify an announce without prior state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,
    pub public_key: Vec<u8>,
    pub routing_hints: Vec<RoutingHint>,
    pub published_at: u64,
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}

impl AnnounceRecord {
    /// Reconstruct the signable payload bytes (everything except signature).
    pub(crate) fn signable_bytes(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            public_key: self.public_key.clone(),
            routing_hints: self.routing_hints.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Serialize the announce record to bytes with a format version prefix.
    pub fn serialize(&self) -> Result<Vec<u8>, DiscoveryError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| DiscoveryError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    /// Deserialize an announce record from bytes (expects format version prefix).
    pub fn deserialize(data: &[u8]) -> Result<Self, DiscoveryError> {
        if data.is_empty() {
            return Err(DiscoveryError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(DiscoveryError::DeserializeError(
                "unsupported format version",
            ));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| DiscoveryError::DeserializeError("postcard decode failed"))
    }
}

/// Builder for constructing a signed announce record.
///
/// Sans-I/O: call `signable_payload()` to get the bytes to sign,
/// then `build(signature)` to produce the final record.
pub struct AnnounceBuilder {
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl AnnounceBuilder {
    /// Create a new builder.
    ///
    /// # Panics
    ///
    /// Panics if `expires_at <= published_at`.
    pub fn new(
        identity_ref: IdentityRef,
        public_key: Vec<u8>,
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
            public_key,
            routing_hints: Vec::new(),
            published_at,
            expires_at,
            nonce,
        }
    }

    /// Add a routing hint describing how to reach this identity.
    pub fn add_routing_hint(&mut self, hint: RoutingHint) -> &mut Self {
        self.routing_hints.push(hint);
        self
    }

    /// Produce the signable payload bytes.
    ///
    /// The caller signs these bytes externally with their private key.
    pub fn signable_payload(&self) -> Vec<u8> {
        let payload = SignablePayload {
            format_version: FORMAT_VERSION,
            identity_ref: self.identity_ref,
            public_key: self.public_key.clone(),
            routing_hints: self.routing_hints.clone(),
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
        };
        postcard::to_allocvec(&payload).expect("signable payload serialization cannot fail")
    }

    /// Finalize with a signature to produce the AnnounceRecord.
    pub fn build(self, signature: Vec<u8>) -> AnnounceRecord {
        AnnounceRecord {
            identity_ref: self.identity_ref,
            public_key: self.public_key,
            routing_hints: self.routing_hints,
            published_at: self.published_at,
            expires_at: self.expires_at,
            nonce: self.nonce,
            signature,
        }
    }
}
```

- [ ] **Step 4: Update `src/lib.rs` to add module and re-exports**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod record;

pub use error::DiscoveryError;
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-discovery`
Expected: all 7 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-discovery/src/record.rs crates/harmony-discovery/src/lib.rs
git commit -m "feat(discovery): add AnnounceRecord, RoutingHint, and AnnounceBuilder"
```

---

### Task 3: Announce Verification

**Files:**
- Create: `crates/harmony-discovery/src/verify.rs`
- Modify: `crates/harmony-discovery/src/lib.rs` (add module + re-export)

**Reference:** `crates/harmony-credential/src/verify.rs` for the `verify_signature` dispatch pattern with ed25519_dalek and ML-DSA-65.

- [ ] **Step 1: Write tests for announce verification**

In `src/verify.rs`, write the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{AnnounceBuilder, RoutingHint};
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    fn build_signed_announce() -> (harmony_identity::PrivateIdentity, AnnounceRecord) {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);

        let mut builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            1000,
            2000,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        (private, record)
    }

    #[test]
    fn valid_announce_passes() {
        let (_, record) = build_signed_announce();
        assert!(verify_announce(&record, 1500).is_ok());
    }

    #[test]
    fn expired_announce_rejected() {
        let (_, record) = build_signed_announce();
        assert_eq!(
            verify_announce(&record, 3000).unwrap_err(),
            DiscoveryError::Expired
        );
    }

    #[test]
    fn tampered_signature_rejected() {
        let (_, mut record) = build_signed_announce();
        record.signature = alloc::vec![0xFF; 64];
        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::SignatureInvalid
        );
    }

    #[test]
    fn wrong_public_key_rejected() {
        let (_, mut record) = build_signed_announce();
        // Replace public key with a different key's bytes
        let other = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        record.public_key = other.public_identity().verifying_key.to_bytes().to_vec();
        assert_eq!(
            verify_announce(&record, 1500).unwrap_err(),
            DiscoveryError::SignatureInvalid
        );
    }

    #[test]
    fn ml_dsa65_announce_verification() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (enc_pk, _) = harmony_crypto::ml_kem::generate(&mut OsRng);

                let pq_id =
                    harmony_identity::PqIdentity::from_public_keys(enc_pk, sign_pk.clone());
                let identity_ref = IdentityRef::from(&pq_id);

                let mut builder = AnnounceBuilder::new(
                    identity_ref,
                    sign_pk.as_bytes(),
                    1000,
                    2000,
                    [0x05; 16],
                );
                builder.add_routing_hint(RoutingHint::Reticulum {
                    destination_hash: [0xDD; 16],
                });

                let payload = builder.signable_payload();
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, &payload).unwrap();
                let record = builder.build(signature.as_bytes().to_vec());

                assert!(verify_announce(&record, 1500).is_ok());
            })
            .expect("failed to spawn thread")
            .join()
            .expect("thread panicked");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-discovery`
Expected: FAIL — `verify_announce` not defined

- [ ] **Step 3: Implement `verify_announce` and signature dispatch**

In `src/verify.rs`:

```rust
use harmony_identity::CryptoSuite;

use crate::error::DiscoveryError;
use crate::record::AnnounceRecord;

/// Verify an announce record's signature and expiry.
///
/// Checks:
/// 1. Record hasn't expired (`expires_at > now`)
/// 2. Signature is valid for the included public key and crypto suite
///
/// Address re-derivation (checking that `public_key` matches
/// `identity_ref.hash`) is deferred to a future version — see design
/// spec for rationale.
pub fn verify_announce(record: &AnnounceRecord, now: u64) -> Result<(), DiscoveryError> {
    // 1. Expiry check
    if now >= record.expires_at {
        return Err(DiscoveryError::Expired);
    }

    // 2. Signature verification
    let payload = record.signable_bytes();
    verify_signature(
        record.identity_ref.suite,
        &record.public_key,
        &payload,
        &record.signature,
    )
}

/// Dispatch signature verification based on CryptoSuite.
fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), DiscoveryError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| DiscoveryError::SignatureInvalid)
        }
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| DiscoveryError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| DiscoveryError::SignatureInvalid)
        }
    }
}
```

- [ ] **Step 4: Update `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod record;
pub mod verify;

pub use error::DiscoveryError;
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
pub use verify::verify_announce;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-discovery`
Expected: all 12 tests pass (7 record + 5 verify)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-discovery/src/verify.rs crates/harmony-discovery/src/lib.rs
git commit -m "feat(discovery): add announce verification with Ed25519 and ML-DSA-65 support"
```

---

### Task 4: DiscoveryManager State Machine

**Files:**
- Create: `crates/harmony-discovery/src/manager.rs`
- Modify: `crates/harmony-discovery/src/lib.rs` (add module + re-exports)

**Reference:** `crates/harmony-peers/src/manager.rs` and `crates/harmony-peers/src/event.rs` for the event/action sans-I/O pattern.

- [ ] **Step 1: Write tests for DiscoveryManager**

In `src/manager.rs`, write the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{AnnounceBuilder, RoutingHint};
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    fn build_valid_record(published_at: u64, expires_at: u64) -> AnnounceRecord {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);

        let mut builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            published_at,
            expires_at,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        builder.build(signature.to_vec())
    }

    fn build_invalid_record() -> AnnounceRecord {
        let identity_ref = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let builder = AnnounceBuilder::new(
            identity_ref,
            alloc::vec![0x01; 32],
            1000,
            2000,
            [0x01; 16],
        );
        let payload = builder.signable_payload();
        // Bogus signature — will fail verification
        builder.build(alloc::vec![0xFF; 64])
    }

    #[test]
    fn announce_received_valid_caches_and_emits() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record.clone(),
            now: 1500,
        });

        assert!(mgr.get_record(&addr).is_some());
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn announce_received_invalid_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_invalid_record();
        let addr = record.identity_ref.hash;

        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record,
            now: 1500,
        });

        assert!(mgr.get_record(&addr).is_none());
        assert!(actions.is_empty());
    }

    #[test]
    fn announce_received_expired_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record,
            now: 3000,
        });

        assert!(mgr.get_record(&addr).is_none());
        assert!(actions.is_empty());
    }

    #[test]
    fn fresher_record_replaces_older() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        // Insert older record
        let builder1 = AnnounceBuilder::new(identity_ref, pk.clone(), 1000, 3000, [0x01; 16]);
        let payload1 = builder1.signable_payload();
        let record1 = builder1.build(private.sign(&payload1).to_vec());
        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record1,
            now: 1500,
        });

        // Insert fresher record
        let builder2 = AnnounceBuilder::new(identity_ref, pk.clone(), 2000, 4000, [0x02; 16]);
        let payload2 = builder2.signable_payload();
        let record2 = builder2.build(private.sign(&payload2).to_vec());
        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record2,
            now: 2500,
        });

        let cached = mgr.get_record(&identity_ref.hash).unwrap();
        assert_eq!(cached.published_at, 2000);
    }

    #[test]
    fn stale_record_does_not_replace_fresher() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        // Insert fresher record first
        let builder1 = AnnounceBuilder::new(identity_ref, pk.clone(), 2000, 4000, [0x01; 16]);
        let payload1 = builder1.signable_payload();
        let record1 = builder1.build(private.sign(&payload1).to_vec());
        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record1,
            now: 2500,
        });

        // Try to insert older record
        let builder2 = AnnounceBuilder::new(identity_ref, pk.clone(), 1000, 3000, [0x02; 16]);
        let payload2 = builder2.signable_payload();
        let record2 = builder2.build(private.sign(&payload2).to_vec());
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record2,
            now: 2500,
        });

        let cached = mgr.get_record(&identity_ref.hash).unwrap();
        assert_eq!(cached.published_at, 2000); // unchanged
        assert!(actions.is_empty()); // no discovery emitted
    }

    #[test]
    fn query_received_known_identity() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record,
            now: 1500,
        });

        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: addr,
            query_id: 42,
        });

        assert!(actions.iter().any(|a| matches!(
            a,
            DiscoveryAction::RespondToQuery {
                query_id: 42,
                record: Some(_),
            }
        )));
    }

    #[test]
    fn query_received_unknown_identity() {
        let mut mgr = DiscoveryManager::new();

        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: [0xFF; 16],
            query_id: 99,
        });

        assert!(actions.iter().any(|a| matches!(
            a,
            DiscoveryAction::RespondToQuery {
                query_id: 99,
                record: None,
            }
        )));
    }

    #[test]
    fn liveliness_join_without_cached_record() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: addr,
            alive: true,
        });
        assert!(mgr.is_online(&addr));
        // No cached record → no IdentityDiscovered emitted
        assert!(actions.is_empty());
    }

    #[test]
    fn liveliness_join_with_cached_record_emits_discovered() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 5000);
        let addr = record.identity_ref.hash;

        // Cache a record first
        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record,
            now: 1500,
        });

        // Liveliness join should emit IdentityDiscovered
        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: addr,
            alive: true,
        });
        assert!(mgr.is_online(&addr));
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn liveliness_leave() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];

        mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: addr,
            alive: true,
        });

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: addr,
            alive: false,
        });
        assert!(!mgr.is_online(&addr));
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
    }

    #[test]
    fn tick_evicts_expired_records() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record,
            now: 1500,
        });
        assert!(mgr.get_record(&addr).is_some());

        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 3000 });
        assert!(mgr.get_record(&addr).is_none());
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
    }

    #[test]
    fn start_and_stop_announcing() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        mgr.set_local_record(record);

        let actions = mgr.start_announcing();
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::PublishAnnounce { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::SetLiveliness { alive: true })));

        let actions = mgr.stop_announcing();
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::SetLiveliness { alive: false })));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-discovery`
Expected: FAIL — `DiscoveryManager` not defined

- [ ] **Step 3: Implement DiscoveryManager**

In `src/manager.rs`:

```rust
use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};
use harmony_identity::IdentityHash;

use crate::record::AnnounceRecord;
use crate::verify::verify_announce;

/// Events consumed by the discovery state machine.
#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    /// Received a published announce from the network.
    AnnounceReceived { record: AnnounceRecord, now: u64 },
    /// Someone queried us for an identity's record.
    QueryReceived { address: IdentityHash, query_id: u64 },
    /// Liveliness change from Zenoh.
    LivelinessChange { address: IdentityHash, alive: bool },
    /// Periodic maintenance.
    Tick { now: u64 },
}

/// Actions emitted by the discovery state machine.
#[derive(Debug, Clone)]
pub enum DiscoveryAction {
    /// Publish our announce record to the network.
    PublishAnnounce { record: AnnounceRecord },
    /// Respond to a resolve query.
    RespondToQuery {
        query_id: u64,
        record: Option<AnnounceRecord>,
    },
    /// Declare or undeclare our liveliness token.
    SetLiveliness { alive: bool },
    /// Notify the application of a newly discovered/online identity.
    IdentityDiscovered { record: AnnounceRecord },
    /// Notify that an identity went offline (liveliness leave).
    IdentityOffline { address: IdentityHash },
    /// An expired record was evicted from cache.
    RecordExpired { address: IdentityHash },
}

/// Sans-I/O state machine for identity discovery.
///
/// Consumes [`DiscoveryEvent`]s and emits [`DiscoveryAction`]s.
/// The caller maps actions to actual Zenoh operations.
pub struct DiscoveryManager {
    local_record: Option<AnnounceRecord>,
    known_identities: HashMap<IdentityHash, AnnounceRecord>,
    online: HashSet<IdentityHash>,
    announcing: bool,
}

impl DiscoveryManager {
    pub fn new() -> Self {
        Self {
            local_record: None,
            known_identities: HashMap::new(),
            online: HashSet::new(),
            announcing: false,
        }
    }

    /// Set our local announce record for publishing.
    pub fn set_local_record(&mut self, record: AnnounceRecord) {
        self.local_record = Some(record);
    }

    /// Start announcing — emits PublishAnnounce + SetLiveliness.
    #[must_use]
    pub fn start_announcing(&mut self) -> Vec<DiscoveryAction> {
        self.announcing = true;
        let mut actions = Vec::new();
        if let Some(record) = self.local_record.clone() {
            actions.push(DiscoveryAction::PublishAnnounce { record });
        }
        actions.push(DiscoveryAction::SetLiveliness { alive: true });
        actions
    }

    /// Stop announcing — emits SetLiveliness(false).
    #[must_use]
    pub fn stop_announcing(&mut self) -> Vec<DiscoveryAction> {
        self.announcing = false;
        alloc::vec![DiscoveryAction::SetLiveliness { alive: false }]
    }

    /// Check if an identity is currently online.
    pub fn is_online(&self, address: &IdentityHash) -> bool {
        self.online.contains(address)
    }

    /// Get a cached announce record.
    pub fn get_record(&self, address: &IdentityHash) -> Option<&AnnounceRecord> {
        self.known_identities.get(address)
    }

    /// Process a discovery event and return resulting actions.
    #[must_use]
    pub fn on_event(&mut self, event: DiscoveryEvent) -> Vec<DiscoveryAction> {
        let mut actions = Vec::new();
        match event {
            DiscoveryEvent::AnnounceReceived { record, now } => {
                self.handle_announce(record, now, &mut actions);
            }
            DiscoveryEvent::QueryReceived { address, query_id } => {
                let record = self.known_identities.get(&address).cloned();
                actions.push(DiscoveryAction::RespondToQuery { query_id, record });
            }
            DiscoveryEvent::LivelinessChange { address, alive } => {
                if alive {
                    self.online.insert(address);
                    if let Some(record) = self.known_identities.get(&address).cloned() {
                        actions.push(DiscoveryAction::IdentityDiscovered { record });
                    }
                } else {
                    self.online.remove(&address);
                    actions.push(DiscoveryAction::IdentityOffline { address });
                }
            }
            DiscoveryEvent::Tick { now } => {
                self.evict_expired(now, &mut actions);
            }
        }
        actions
    }

    fn handle_announce(
        &mut self,
        record: AnnounceRecord,
        now: u64,
        actions: &mut Vec<DiscoveryAction>,
    ) {
        // Verify signature and expiry
        if verify_announce(&record, now).is_err() {
            return;
        }

        let addr = record.identity_ref.hash;

        // Freshness check: only accept if newer than cached
        if let Some(existing) = self.known_identities.get(&addr) {
            if record.published_at <= existing.published_at {
                return;
            }
        }

        self.known_identities.insert(addr, record.clone());
        actions.push(DiscoveryAction::IdentityDiscovered { record });
    }

    fn evict_expired(&mut self, now: u64, actions: &mut Vec<DiscoveryAction>) {
        let expired: Vec<IdentityHash> = self
            .known_identities
            .iter()
            .filter(|(_, record)| now >= record.expires_at)
            .map(|(addr, _)| *addr)
            .collect();

        for addr in expired {
            self.known_identities.remove(&addr);
            actions.push(DiscoveryAction::RecordExpired { address: addr });
        }
    }
}
```

- [ ] **Step 4: Update `src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod manager;
pub mod record;
pub mod verify;

pub use error::DiscoveryError;
pub use manager::{DiscoveryAction, DiscoveryEvent, DiscoveryManager};
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
pub use verify::verify_announce;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-discovery`
Expected: all 25 tests pass (7 record + 5 verify + 13 manager)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-discovery/src/manager.rs crates/harmony-discovery/src/lib.rs
git commit -m "feat(discovery): add DiscoveryManager sans-I/O state machine"
```

---

### Task 5: OfflineResolver Trait

**Files:**
- Create: `crates/harmony-discovery/src/resolve.rs`
- Modify: `crates/harmony-discovery/src/lib.rs` (add module + re-export)

- [ ] **Step 1: Create `src/resolve.rs`**

```rust
use harmony_identity::IdentityHash;

use crate::record::AnnounceRecord;

/// Resolve an identity's announce record from persistent storage.
///
/// Used as a fallback when no live Queryable answers. The content
/// layer can implement this by storing AnnounceRecords as content
/// objects.
///
/// The `DiscoveryManager` does not call this directly — the caller
/// (runtime integration layer) falls back to offline resolution
/// when a query yields no cached result.
pub trait OfflineResolver {
    fn resolve(&self, address: &IdentityHash) -> Option<AnnounceRecord>;
}
```

- [ ] **Step 2: Update `src/lib.rs` with final re-exports**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod manager;
pub mod record;
pub mod resolve;
pub mod verify;

pub use error::DiscoveryError;
pub use manager::{DiscoveryAction, DiscoveryEvent, DiscoveryManager};
pub use record::{AnnounceBuilder, AnnounceRecord, RoutingHint};
pub use resolve::OfflineResolver;
pub use verify::verify_announce;
```

- [ ] **Step 3: Verify it compiles and all tests pass**

Run: `cargo test -p harmony-discovery`
Expected: all 25 tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-discovery/src/resolve.rs crates/harmony-discovery/src/lib.rs
git commit -m "feat(discovery): add OfflineResolver trait for DHT/content-layer fallback"
```

---

### Task 6: Zenoh Identity Key Expression Patterns

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs` (add `pub mod identity`)

**Reference:** Existing modules in `namespace.rs` — `pub mod reticulum { ... }`, `pub mod content { ... }` — for the exact pattern of constants + builder functions.

- [ ] **Step 1: Write tests for key expression builders**

Add at the bottom of the existing tests module in `namespace.rs` (or in a new test block within the `identity` module):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn announce_key_format() {
        let key = identity::announce_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/announce"
        );
    }

    #[test]
    fn resolve_key_format() {
        let key = identity::resolve_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/resolve"
        );
    }

    #[test]
    fn alive_key_format() {
        let key = identity::alive_key("aa00bb11cc22dd33ee44ff5566778899");
        assert_eq!(
            key,
            "harmony/identity/aa00bb11cc22dd33ee44ff5566778899/alive"
        );
    }

    #[test]
    fn wildcard_constants() {
        assert_eq!(
            identity::ALL_ANNOUNCES,
            "harmony/identity/*/announce"
        );
        assert_eq!(identity::ALL_ALIVE, "harmony/identity/*/alive");
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh identity`
Expected: FAIL — `identity` module not defined

- [ ] **Step 3: Implement the identity namespace module**

Add to `crates/harmony-zenoh/src/namespace.rs`, after the existing modules (e.g., after the `reticulum` or `content` module):

```rust
pub mod identity {
    use alloc::{format, string::String};

    pub const PREFIX: &str = "harmony/identity";

    /// Subscribe to all identity announces.
    pub const ALL_ANNOUNCES: &str = "harmony/identity/*/announce";

    /// Subscribe to all identity presence changes.
    pub const ALL_ALIVE: &str = "harmony/identity/*/alive";

    /// Key expression for a specific identity's announce channel.
    pub fn announce_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/announce")
    }

    /// Key expression for a specific identity's resolve endpoint.
    pub fn resolve_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/resolve")
    }

    /// Key expression for a specific identity's liveliness token.
    pub fn alive_key(address_hex: &str) -> String {
        format!("{PREFIX}/{address_hex}/alive")
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-zenoh identity`
Expected: all 4 key expression tests pass

Also run full workspace check: `cargo test -p harmony-zenoh`
Expected: all existing tests still pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add identity namespace key expression patterns"
```

---

### Task 7: Integration Tests and Final Verification

**Files:**
- Modify: `crates/harmony-discovery/src/manager.rs` (add integration tests)

- [ ] **Step 1: Add integration tests**

Add to the existing `tests` module in `manager.rs`:

```rust
    #[test]
    fn full_flow_announce_cache_query() {
        let mut mgr = DiscoveryManager::new();

        // 1. Build and sign an announce
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let mut builder =
            AnnounceBuilder::new(identity_ref, pk, 1000, 5000, [0x10; 16]);
        builder
            .add_routing_hint(RoutingHint::Reticulum {
                destination_hash: [0xAA; 16],
            })
            .add_routing_hint(RoutingHint::Zenoh {
                locator: alloc::vec![0xBB; 4],
            });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        // 2. Receive the announce
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record.clone(),
            now: 2000,
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));

        // 3. Query for it
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: identity_ref.hash,
            query_id: 1,
        });
        match &actions[0] {
            DiscoveryAction::RespondToQuery {
                query_id: 1,
                record: Some(r),
            } => {
                assert_eq!(r.identity_ref, identity_ref);
                assert_eq!(r.routing_hints.len(), 2);
            }
            other => panic!("expected RespondToQuery, got {:?}", other),
        }

        // 4. Liveliness tracks online state
        mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: identity_ref.hash,
            alive: true,
        });
        assert!(mgr.is_online(&identity_ref.hash));

        // 5. Tick evicts after expiry
        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 6000 });
        assert!(mgr.get_record(&identity_ref.hash).is_none());
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
    }

    #[test]
    fn announce_serde_round_trip_through_manager() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let builder = AnnounceBuilder::new(identity_ref, pk, 1000, 5000, [0x20; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        // Serialize, deserialize, then feed to manager
        let bytes = record.serialize().unwrap();
        let restored = AnnounceRecord::deserialize(&bytes).unwrap();

        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: restored,
            now: 2000,
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
        assert!(mgr.get_record(&identity_ref.hash).is_some());
    }
```

- [ ] **Step 2: Run all tests**

Run: `cargo test -p harmony-discovery`
Expected: all 27 tests pass

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -p harmony-discovery`
Expected: no warnings

- [ ] **Step 4: Run full workspace tests to ensure no regressions**

Run: `cargo test --workspace`
Expected: all tests pass across all crates

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-discovery/src/manager.rs
git commit -m "feat(discovery): add integration tests for full announce-cache-query flow"
```

---

## Summary

| Task | Component | Tests |
|------|-----------|-------|
| 1 | Crate scaffold + DiscoveryError | 0 (compile check) |
| 2 | AnnounceRecord + RoutingHint + AnnounceBuilder | 7 |
| 3 | verify_announce (Ed25519 + ML-DSA-65) | 5 |
| 4 | DiscoveryManager state machine | 13 |
| 5 | OfflineResolver trait | 0 (trait only) |
| 6 | Zenoh identity key expressions | 4 |
| 7 | Integration tests + final verification | 2 |
| **Total** | | **31** |

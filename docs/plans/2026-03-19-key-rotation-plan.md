# Key Rotation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement KERI-inspired Key Event Log with pre-rotation commitments as a new `harmony-kel` crate, plus prerequisite crypto serde and `CryptoSuite::MlDsa65Rotatable` variant.

**Architecture:** Three changes: (1) Add serde/Debug to PQ crypto types in `harmony-crypto`, (2) Add `MlDsa65Rotatable` variant to `CryptoSuite` in `harmony-identity`, (3) New `harmony-kel` crate with `InceptionEvent`, `RotationEvent`, `InteractionEvent`, and `KeyEventLog`. All `no_std`-compatible, sans-I/O, TDD.

**Tech Stack:** Rust 1.75+, no_std + alloc, serde, postcard, BLAKE3, ML-DSA-65, ML-KEM-768

**Spec:** `docs/plans/2026-03-19-key-rotation-design.md`

---

## File Structure

```
crates/harmony-crypto/Cargo.toml              — add serde dep (modify)
crates/harmony-crypto/src/ml_dsa.rs           — add Debug + serde impls (modify)
crates/harmony-crypto/src/ml_kem.rs           — add Debug + serde impls (modify)
crates/harmony-identity/src/crypto_suite.rs   — add MlDsa65Rotatable variant (modify)
crates/harmony-identity/src/identity_ref.rs   — update hash field doc comment (modify)
crates/harmony-kel/Cargo.toml                 — crate manifest (create)
crates/harmony-kel/src/lib.rs                 — re-exports (create)
crates/harmony-kel/src/commitment.rs          — pre-rotation commitment (create)
crates/harmony-kel/src/event.rs               — KeyEvent types (create)
crates/harmony-kel/src/log.rs                 — KeyEventLog (create)
crates/harmony-kel/src/error.rs               — KelError (create)
Cargo.toml                                    — add workspace member + dep (modify)
```

---

### Task 1: Add serde and Debug to PQ crypto types

**Files:**
- Modify: `crates/harmony-crypto/Cargo.toml`
- Modify: `crates/harmony-crypto/src/ml_dsa.rs`
- Modify: `crates/harmony-crypto/src/ml_kem.rs`

The inner types from `ml-dsa` and `ml-kem` crates don't implement serde or Debug. We add manual byte-based serde impls and Debug impls to our wrapper types.

- [ ] **Step 1: Add serde dependency to harmony-crypto**

In `crates/harmony-crypto/Cargo.toml`, add to `[dependencies]`:
```toml
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
```

Add `"serde/std",` to the `std` feature list.

Add `postcard` to `[dev-dependencies]`:
```toml
postcard = { workspace = true }
```

- [ ] **Step 2: Add Debug and serde to MlDsaPublicKey**

In `ml_dsa.rs`, add after the existing `impl MlDsaPublicKey` block:

```rust
impl core::fmt::Debug for MlDsaPublicKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        write!(f, "MlDsaPublicKey({:02x}{:02x}..)", bytes[0], bytes[1])
    }
}

impl serde::Serialize for MlDsaPublicKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.as_bytes())
    }
}

impl<'de> serde::Deserialize<'de> for MlDsaPublicKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: alloc::vec::Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}
```

- [ ] **Step 3: Add Debug and serde to MlDsaSignature**

In `ml_dsa.rs`, add after the existing `impl MlDsaSignature` block:

```rust
impl core::fmt::Debug for MlDsaSignature {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "MlDsaSignature({}B)", self.bytes.len())
    }
}

impl serde::Serialize for MlDsaSignature {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.bytes)
    }
}

impl<'de> serde::Deserialize<'de> for MlDsaSignature {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: alloc::vec::Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}
```

- [ ] **Step 4: Add Debug and serde to MlKemPublicKey**

In `ml_kem.rs`, add after the existing `impl MlKemPublicKey` block:

```rust
impl core::fmt::Debug for MlKemPublicKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        write!(f, "MlKemPublicKey({:02x}{:02x}..)", bytes[0], bytes[1])
    }
}

impl serde::Serialize for MlKemPublicKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.as_bytes())
    }
}

impl<'de> serde::Deserialize<'de> for MlKemPublicKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: alloc::vec::Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}
```

- [ ] **Step 5: Add serde round-trip tests**

Add tests to the existing test modules in `ml_dsa.rs` and `ml_kem.rs`:

In `ml_dsa.rs` tests:
```rust
    #[test]
    fn public_key_serde_round_trip() {
        let (pk, _sk) = generate(&mut rand::rngs::OsRng);
        let bytes = postcard::to_allocvec(&pk).unwrap();
        let decoded: MlDsaPublicKey = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), decoded.as_bytes());
    }

    #[test]
    fn signature_serde_round_trip() {
        let (pk, sk) = generate(&mut rand::rngs::OsRng);
        let sig = sign(&sk, b"test message").unwrap();
        let bytes = postcard::to_allocvec(&sig).unwrap();
        let decoded: MlDsaSignature = postcard::from_bytes(&bytes).unwrap();
        assert!(verify(&pk, b"test message", &decoded).is_ok());
    }
```

In `ml_kem.rs` tests:
```rust
    #[test]
    fn public_key_serde_round_trip() {
        let (pk, _sk) = generate(&mut rand::rngs::OsRng);
        let bytes = postcard::to_allocvec(&pk).unwrap();
        let decoded: MlKemPublicKey = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), decoded.as_bytes());
    }
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-crypto`
Expected: All existing tests pass + 3 new serde tests.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-crypto/
git commit -m "feat(crypto): add Debug and serde impls to PQ key/signature types"
```

---

### Task 2: Add `CryptoSuite::MlDsa65Rotatable` variant

**Files:**
- Modify: `crates/harmony-identity/src/crypto_suite.rs`
- Modify: `crates/harmony-identity/src/identity_ref.rs` (doc comment only)

- [ ] **Step 1: Add variant and update all methods**

In `crypto_suite.rs`:
- Add `MlDsa65Rotatable = 0x02` variant with doc comment
- Update `is_post_quantum`: add `| Self::MlDsa65Rotatable` to the match
- Update `signing_multicodec`: `Self::MlDsa65Rotatable => 0x1211`
- Update `encryption_multicodec`: `Self::MlDsa65Rotatable => 0x120c`
- `from_signing_multicodec` and `from_encryption_multicodec`: NO change (they return `MlDsa65`, not `Rotatable` — multicodec identifies algorithm, not lifecycle)
- Update `from_byte`: add `0x02 => Some(Self::MlDsa65Rotatable)`
- Update `TryFrom<u8>`: add `0x02 => Ok(Self::MlDsa65Rotatable)`
- Update `From<CryptoSuite> for u8`: already works via `as u8` cast

- [ ] **Step 2: Add tests**

Add to the existing test module in `crypto_suite.rs`:

```rust
    #[test]
    fn ml_dsa65_rotatable_is_post_quantum() {
        assert!(CryptoSuite::MlDsa65Rotatable.is_post_quantum());
    }

    #[test]
    fn ml_dsa65_rotatable_from_byte() {
        assert_eq!(CryptoSuite::from_byte(0x02), Some(CryptoSuite::MlDsa65Rotatable));
    }

    #[test]
    fn ml_dsa65_rotatable_try_from() {
        assert_eq!(CryptoSuite::try_from(0x02), Ok(CryptoSuite::MlDsa65Rotatable));
    }

    #[test]
    fn ml_dsa65_rotatable_multicodec_same_as_static() {
        assert_eq!(
            CryptoSuite::MlDsa65Rotatable.signing_multicodec(),
            CryptoSuite::MlDsa65.signing_multicodec()
        );
        assert_eq!(
            CryptoSuite::MlDsa65Rotatable.encryption_multicodec(),
            CryptoSuite::MlDsa65.encryption_multicodec()
        );
    }

    #[test]
    fn multicodec_round_trip_lossy_for_rotatable() {
        // Multicodec identifies algorithm, not lifecycle — round-trip is lossy
        let code = CryptoSuite::MlDsa65Rotatable.signing_multicodec();
        assert_eq!(CryptoSuite::from_signing_multicodec(code), Some(CryptoSuite::MlDsa65));
    }

    #[test]
    fn ml_dsa65_rotatable_serde_round_trip() {
        let suite = CryptoSuite::MlDsa65Rotatable;
        let bytes = postcard::to_allocvec(&suite).unwrap();
        let decoded: CryptoSuite = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, suite);
    }

    #[test]
    fn ml_dsa65_rotatable_wire_discriminant() {
        assert_eq!(CryptoSuite::MlDsa65Rotatable as u8, 0x02);
    }
```

- [ ] **Step 3: Update IdentityRef doc comment**

In `identity_ref.rs`, update the `hash` field doc comment from:
```rust
    /// The 128-bit address hash: SHA256(pub_keys)[:16].
```
to:
```rust
    /// The 128-bit address hash. Derivation depends on suite:
    /// Ed25519/MlDsa65: SHA256(pub_keys)[:16].
    /// MlDsa65Rotatable: SHA256(inception_payload)[:16].
```

- [ ] **Step 4: Update serde_encodes_as_discriminant_byte test**

Update the existing test to include the new variant:
```rust
    #[test]
    fn serde_encodes_as_discriminant_byte() {
        let bytes = postcard::to_allocvec(&CryptoSuite::Ed25519).unwrap();
        assert_eq!(bytes, [0x00]);
        let bytes = postcard::to_allocvec(&CryptoSuite::MlDsa65).unwrap();
        assert_eq!(bytes, [0x01]);
        let bytes = postcard::to_allocvec(&CryptoSuite::MlDsa65Rotatable).unwrap();
        assert_eq!(bytes, [0x02]);
    }
```

Also update `serde_round_trip_both_variants` to include all three:
```rust
    #[test]
    fn serde_round_trip_all_variants() {
        for suite in [CryptoSuite::Ed25519, CryptoSuite::MlDsa65, CryptoSuite::MlDsa65Rotatable] {
            let bytes = postcard::to_allocvec(&suite).unwrap();
            let decoded: CryptoSuite = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(decoded, suite);
        }
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-identity`
Expected: All existing tests pass + ~8 new tests.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-identity/
git commit -m "feat(identity): add CryptoSuite::MlDsa65Rotatable variant"
```

---

### Task 3: Scaffold `harmony-kel` crate with error and commitment types

**Files:**
- Create: `crates/harmony-kel/Cargo.toml`
- Create: `crates/harmony-kel/src/lib.rs`
- Create: `crates/harmony-kel/src/error.rs`
- Create: `crates/harmony-kel/src/commitment.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-kel"
description = "KERI-inspired Key Event Log with pre-rotation for the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-crypto = { workspace = true }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]

[dev-dependencies]
rand = { workspace = true }
postcard = { workspace = true }
```

- [ ] **Step 2: Create error.rs**

```rust
/// Errors returned by Key Event Log operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KelError {
    /// Inception event has an invalid signature.
    InvalidInceptionSignature,
    /// Rotation event's keys don't match the pre-rotation commitment.
    PreRotationMismatch,
    /// Event's previous_hash doesn't match BLAKE3 of the prior event payload.
    HashChainBroken,
    /// Sequence number is not monotonically increasing.
    SequenceViolation,
    /// Event signature is invalid.
    InvalidSignature,
    /// Deserialized log contains a second InceptionEvent.
    DuplicateInception,
    /// KEL is empty (no inception event).
    EmptyLog,
    /// Serialization failed.
    SerializeError(&'static str),
    /// Deserialization failed.
    DeserializeError(&'static str),
}

impl core::fmt::Display for KelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInceptionSignature => write!(f, "invalid inception signature"),
            Self::PreRotationMismatch => write!(f, "pre-rotation commitment mismatch"),
            Self::HashChainBroken => write!(f, "hash chain broken"),
            Self::SequenceViolation => write!(f, "sequence number violation"),
            Self::InvalidSignature => write!(f, "invalid event signature"),
            Self::DuplicateInception => write!(f, "duplicate inception event"),
            Self::EmptyLog => write!(f, "empty key event log"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KelError {}
```

- [ ] **Step 3: Create commitment.rs with tests first**

Tests:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_crypto::ml_dsa;
    use harmony_crypto::ml_kem;

    #[test]
    fn commitment_is_deterministic() {
        let (sign_pk, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let c1 = compute_commitment(&sign_pk, &enc_pk);
        let c2 = compute_commitment(&sign_pk, &enc_pk);
        assert_eq!(c1, c2);
    }

    #[test]
    fn different_keys_produce_different_commitments() {
        let (sign_pk1, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk1, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let (sign_pk2, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk2, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let c1 = compute_commitment(&sign_pk1, &enc_pk1);
        let c2 = compute_commitment(&sign_pk2, &enc_pk2);
        assert_ne!(c1, c2);
    }

    #[test]
    fn verify_commitment_matches() {
        let (sign_pk, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let commitment = compute_commitment(&sign_pk, &enc_pk);
        assert!(verify_commitment(&sign_pk, &enc_pk, &commitment));
    }

    #[test]
    fn verify_commitment_rejects_wrong_keys() {
        let (sign_pk1, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk1, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let (sign_pk2, _) = ml_dsa::generate(&mut rand::rngs::OsRng);
        let (enc_pk2, _) = ml_kem::generate(&mut rand::rngs::OsRng);
        let commitment = compute_commitment(&sign_pk1, &enc_pk1);
        assert!(!verify_commitment(&sign_pk2, &enc_pk2, &commitment));
    }
}
```

Implementation:
```rust
use alloc::vec::Vec;
use harmony_crypto::hash::blake3_hash;
use harmony_crypto::ml_dsa::MlDsaPublicKey;
use harmony_crypto::ml_kem::MlKemPublicKey;

/// Compute the pre-rotation commitment: BLAKE3(signing_pub || encryption_pub).
pub fn compute_commitment(
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
) -> [u8; 32] {
    let mut buf = Vec::with_capacity(
        signing_key.as_bytes().len() + encryption_key.as_bytes().len(),
    );
    buf.extend_from_slice(&signing_key.as_bytes());
    buf.extend_from_slice(&encryption_key.as_bytes());
    blake3_hash(&buf)
}

/// Verify that the given keys match a pre-rotation commitment.
pub fn verify_commitment(
    signing_key: &MlDsaPublicKey,
    encryption_key: &MlKemPublicKey,
    commitment: &[u8; 32],
) -> bool {
    compute_commitment(signing_key, encryption_key) == *commitment
}
```

- [ ] **Step 4: Create lib.rs**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod commitment;
pub mod error;

pub use commitment::{compute_commitment, verify_commitment};
pub use error::KelError;
```

- [ ] **Step 5: Add workspace entries**

In root `Cargo.toml`, add `"crates/harmony-kel"` to `members` and add to `[workspace.dependencies]`:
```toml
harmony-kel = { path = "crates/harmony-kel", default-features = false }
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-kel`
Expected: 4 commitment tests pass.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-kel/ Cargo.toml Cargo.lock
git commit -m "feat(kel): scaffold harmony-kel with pre-rotation commitment"
```

---

### Task 4: Implement Key Event types

**Files:**
- Create: `crates/harmony-kel/src/event.rs`
- Modify: `crates/harmony-kel/src/lib.rs`

- [ ] **Step 1: Create event.rs**

This task creates the event types and payload serialization helpers. Tests will come in Task 5 with the KEL (events are validated in context of the log).

```rust
use alloc::vec::Vec;
use harmony_crypto::ml_dsa::{MlDsaPublicKey, MlDsaSignature};
use harmony_crypto::ml_kem::MlKemPublicKey;
use serde::{Deserialize, Serialize};

/// A key lifecycle event in the Key Event Log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum KeyEvent {
    Inception(InceptionEvent),
    Rotation(RotationEvent),
    Interaction(InteractionEvent),
}

/// Creates a rotatable identity. Address derived from unsigned payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InceptionEvent {
    pub signing_key: MlDsaPublicKey,
    pub encryption_key: MlKemPublicKey,
    pub next_key_commitment: [u8; 32],
    pub created_at: u64,
    pub signature: MlDsaSignature,
}

/// Retires current keys, activates pre-committed keys. Dual-signed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RotationEvent {
    pub sequence: u64,
    pub previous_hash: [u8; 32],
    pub signing_key: MlDsaPublicKey,
    pub encryption_key: MlKemPublicKey,
    pub next_key_commitment: [u8; 32],
    pub created_at: u64,
    pub old_signature: MlDsaSignature,
    pub new_signature: MlDsaSignature,
}

/// Anchors data to the log without rotating keys.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEvent {
    pub sequence: u64,
    pub previous_hash: [u8; 32],
    pub data_hash: [u8; 32],
    pub created_at: u64,
    pub signature: MlDsaSignature,
}

impl KeyEvent {
    /// The effective sequence number of this event.
    /// InceptionEvent = 0, others use their explicit sequence field.
    pub fn sequence(&self) -> u64 {
        match self {
            KeyEvent::Inception(_) => 0,
            KeyEvent::Rotation(e) => e.sequence,
            KeyEvent::Interaction(e) => e.sequence,
        }
    }

    /// The signing key introduced or active at this event.
    pub fn signing_key(&self) -> &MlDsaPublicKey {
        match self {
            KeyEvent::Inception(e) => &e.signing_key,
            KeyEvent::Rotation(e) => &e.signing_key,
            KeyEvent::Interaction(_) => panic!("interaction events don't introduce keys"),
        }
    }

    /// The encryption key introduced or active at this event.
    pub fn encryption_key(&self) -> &MlKemPublicKey {
        match self {
            KeyEvent::Inception(e) => &e.encryption_key,
            KeyEvent::Rotation(e) => &e.encryption_key,
            KeyEvent::Interaction(_) => panic!("interaction events don't introduce keys"),
        }
    }

    /// The next key commitment at this event.
    pub fn next_key_commitment(&self) -> &[u8; 32] {
        match self {
            KeyEvent::Inception(e) => &e.next_key_commitment,
            KeyEvent::Rotation(e) => &e.next_key_commitment,
            KeyEvent::Interaction(_) => panic!("interaction events don't have commitments"),
        }
    }
}

/// Serialize the unsigned payload of an inception event (excludes signature).
pub fn serialize_inception_payload(event: &InceptionEvent) -> Vec<u8> {
    // Serialize all fields except signature
    let mut buf = Vec::new();
    buf.extend_from_slice(&event.signing_key.as_bytes());
    buf.extend_from_slice(&event.encryption_key.as_bytes());
    buf.extend_from_slice(&event.next_key_commitment);
    buf.extend_from_slice(&event.created_at.to_le_bytes());
    buf
}

/// Serialize the unsigned payload of a rotation event (excludes signatures).
pub fn serialize_rotation_payload(event: &RotationEvent) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&event.sequence.to_le_bytes());
    buf.extend_from_slice(&event.previous_hash);
    buf.extend_from_slice(&event.signing_key.as_bytes());
    buf.extend_from_slice(&event.encryption_key.as_bytes());
    buf.extend_from_slice(&event.next_key_commitment);
    buf.extend_from_slice(&event.created_at.to_le_bytes());
    buf
}

/// Serialize the unsigned payload of an interaction event (excludes signature).
pub fn serialize_interaction_payload(event: &InteractionEvent) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&event.sequence.to_le_bytes());
    buf.extend_from_slice(&event.previous_hash);
    buf.extend_from_slice(&event.data_hash);
    buf.extend_from_slice(&event.created_at.to_le_bytes());
    buf
}

/// Serialize the payload of any event (for hash chaining).
pub fn serialize_event_payload(event: &KeyEvent) -> Vec<u8> {
    match event {
        KeyEvent::Inception(e) => serialize_inception_payload(e),
        KeyEvent::Rotation(e) => serialize_rotation_payload(e),
        KeyEvent::Interaction(e) => serialize_interaction_payload(e),
    }
}
```

- [ ] **Step 2: Update lib.rs**

Add:
```rust
pub mod event;
pub use event::{
    InceptionEvent, InteractionEvent, KeyEvent, RotationEvent,
    serialize_event_payload, serialize_inception_payload,
    serialize_interaction_payload, serialize_rotation_payload,
};
```

- [ ] **Step 3: Build check**

Run: `cargo check -p harmony-kel`
Expected: Compiles (no tests for event types alone — tested via KEL in Task 5).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-kel/
git commit -m "feat(kel): add KeyEvent types with payload serialization"
```

---

### Task 5: Implement `KeyEventLog`

**Files:**
- Create: `crates/harmony-kel/src/log.rs`
- Modify: `crates/harmony-kel/src/lib.rs`

This is the core of the crate — the validated, hash-chained event log.

- [ ] **Step 1: Write tests first**

Create `log.rs` with tests at the bottom. The tests generate real PQ keypairs and create/validate real events. This is a large test file — all tests use a shared helper that generates keypairs and builds signed events.

The implementer should write these tests:

1. `from_inception_creates_valid_kel` — create inception, verify address, signing key, sequence 0
2. `from_inception_rejects_invalid_signature` — tamper with signature, expect `InvalidInceptionSignature`
3. `address_is_deterministic` — same inception payload → same address
4. `apply_rotation_updates_keys` — rotate, verify `current_signing_key` changes
5. `apply_rotation_rejects_bad_commitment` — wrong keys for commitment, expect `PreRotationMismatch`
6. `apply_rotation_rejects_bad_hash_chain` — wrong previous_hash, expect `HashChainBroken`
7. `apply_rotation_rejects_bad_sequence` — wrong sequence number, expect `SequenceViolation`
8. `apply_rotation_rejects_bad_old_signature` — invalid old-key signature, expect `InvalidSignature`
9. `apply_rotation_rejects_bad_new_signature` — invalid new-key signature, expect `InvalidSignature`
10. `apply_interaction_appends` — append interaction, verify hash chain
11. `apply_interaction_rejects_bad_chain` — wrong previous_hash, expect `HashChainBroken`
12. `apply_interaction_rejects_bad_sequence` — wrong sequence, expect `SequenceViolation`
13. `apply_interaction_rejects_bad_signature` — invalid signature, expect `InvalidSignature`
14. `address_permanent_through_rotations` — rotate, verify address unchanged
15. `identity_ref_returns_rotatable` — verify suite is `MlDsa65Rotatable`
16. `double_rotation` — rotate twice, verify keys update each time
17. `interaction_after_rotation_uses_new_key` — rotate then interact with new key
18. `mixed_event_sequence` — inception → interaction → rotation → interaction
19. `serialize_deserialize_round_trip` — persist and restore full KEL
20. `deserialize_rejects_corrupt_data` — bad data returns error

- [ ] **Step 2: Implement KeyEventLog**

The implementation uses `harmony_crypto::hash::truncated_hash` for address derivation, `blake3_hash` for hash chaining, and `ml_dsa::verify` for signature verification.

Key implementation details:
- `from_inception`: verify signature over unsigned payload, derive address via `truncated_hash(serialize_inception_payload(...))`
- `apply_rotation`: verify hash chain, pre-rotation commitment, sequence, BOTH signatures (old under current key, new under rotation's key)
- `apply_interaction`: verify hash chain, sequence, signature under current active key
- Track the current active signing/encryption keys and the latest `next_key_commitment` by scanning the event list (or caching on append)

- [ ] **Step 3: Update lib.rs**

Add:
```rust
pub mod log;
pub use log::KeyEventLog;
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-kel`
Expected: All ~24 tests pass (4 commitment + 20 log).

- [ ] **Step 5: Run full workspace**

Run: `cargo test --workspace`
Expected: All tests pass.

- [ ] **Step 6: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-kel/
git commit -m "feat(kel): implement KeyEventLog with pre-rotation and dual signatures"
```

---

### Task 6: Final quality gate

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: All tests pass.

- [ ] **Step 2: Format check**

Run: `cargo fmt --all -- --check`
Expected: Clean.

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.

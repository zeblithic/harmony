# `harmony-tunnel` Crate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a sans-I/O state machine crate that manages PQ-authenticated, encrypted tunnels between remote Harmony peers, with dual-path multiplexing for Reticulum control-plane and Zenoh data-plane frames.

**Architecture:** Pure state machine (`TunnelSession`) consumes `TunnelEvent` and emits `TunnelAction`. Two-message PQ handshake (ML-KEM-768 + ML-DSA-65) establishes directional ChaCha20-Poly1305 session keys. Frames are tagged (`0x00` keepalive, `0x01` Reticulum, `0x02` Zenoh), encrypted as complete AEAD units with the remote peer's NodeId as AAD.

**Tech Stack:** Rust, `harmony-crypto` (HKDF-SHA256, ChaCha20-Poly1305, BLAKE3, ML-KEM, ML-DSA), `harmony-identity` (PqIdentity/PqPrivateIdentity), `zeroize`, `thiserror`

**Spec:** `docs/superpowers/specs/2026-03-20-tunnel-peer-infrastructure-design.md` — Section 1

**Scope:** This plan covers **Bead #1 (harmony-641)** only — the `harmony-tunnel` crate. Beads #2-5 (iroh-net integration, peer lifecycle, discovery hints, relay deployment) will have separate implementation plans.

---

## File Structure

```
crates/harmony-tunnel/
├── Cargo.toml
└── src/
    ├── lib.rs          — Module declarations, re-exports
    ├── error.rs        — TunnelError enum
    ├── event.rs        — TunnelEvent and TunnelAction enums
    ├── handshake.rs    — PQ handshake message types and logic (TunnelInit, TunnelAccept)
    ├── frame.rs        — Frame encoding/decoding and encryption/decryption
    └── session.rs      — TunnelSession state machine (Idle→Initiating→Active→Closed)
```

---

### Task 1: Scaffold crate and define error types

**Files:**
- Create: `crates/harmony-tunnel/Cargo.toml`
- Create: `crates/harmony-tunnel/src/lib.rs`
- Create: `crates/harmony-tunnel/src/error.rs`
- Modify: `Cargo.toml` (workspace root — add member + workspace dep)

- [ ] **Step 1: Add crate to workspace root `Cargo.toml`**

In `Cargo.toml` (workspace root), add `"crates/harmony-tunnel"` to the `members` array (after `harmony-trust`, before `harmony-oluo`):

```toml
    "crates/harmony-trust",
    "crates/harmony-tunnel",
    "crates/harmony-oluo",
```

Note: the workspace members list is not strictly alphabetical — insert after `harmony-trust`.

And add the workspace dependency (alphabetical, between `harmony-trust` and `harmony-workflow`):

```toml
harmony-tunnel = { path = "crates/harmony-tunnel", default-features = false }
```

- [ ] **Step 2: Create `crates/harmony-tunnel/Cargo.toml`**

```toml
[package]
name = "harmony-tunnel"
description = "PQ-authenticated encrypted tunnel state machine for Harmony peer-to-peer connections"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-crypto = { workspace = true }
harmony-identity = { workspace = true }
rand_core = { workspace = true }
zeroize = { workspace = true }
thiserror = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-crypto/std",
    "harmony-identity/std",
    "rand_core/getrandom",
    "thiserror/std",
]

[dev-dependencies]
rand = { workspace = true }
hex = { workspace = true }
```

- [ ] **Step 3: Create `crates/harmony-tunnel/src/error.rs`**

```rust
/// Errors that can occur during tunnel operations.
#[derive(Debug, thiserror::Error)]
pub enum TunnelError {
    #[error("tunnel is not in the expected state for this operation")]
    InvalidState,

    #[error("handshake signature verification failed")]
    SignatureVerificationFailed,

    #[error("ML-KEM decapsulation produced an invalid shared secret")]
    DecapsulationFailed,

    #[error("frame decryption failed (authentication tag mismatch)")]
    DecryptionFailed,

    #[error("frame too short: expected at least {expected} bytes, got {got}")]
    FrameTooShort { expected: usize, got: usize },

    #[error("unknown frame tag: 0x{tag:02x}")]
    UnknownFrameTag { tag: u8 },

    #[error("handshake message malformed: {reason}")]
    MalformedHandshake { reason: &'static str },

    #[error("keepalive timeout: peer unresponsive")]
    KeepaliveTimeout,

    #[error("cryptographic operation failed: {0}")]
    Crypto(#[from] harmony_crypto::CryptoError),
}
```

- [ ] **Step 4: Create `crates/harmony-tunnel/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;
pub mod event;
pub mod frame;
pub mod handshake;
pub mod session;

pub use error::TunnelError;
pub use event::{TunnelAction, TunnelEvent};
pub use session::TunnelSession;
```

- [ ] **Step 5: Add `signing_key()` accessor to `PqPrivateIdentity`**

The tunnel handshake needs direct access to the ML-DSA secret key for signing. Add this method to `PqPrivateIdentity` in `crates/harmony-identity/src/pq_identity.rs`:

```rust
    /// Returns a reference to the ML-DSA-65 signing key.
    pub fn signing_key(&self) -> &harmony_crypto::ml_dsa::MlDsaSecretKey {
        &self.signing_key
    }
```

Also add a public accessor for the `verifying_key` field on `PqIdentity` if not already present (the tunnel needs to access `remote_identity.verifying_key` directly). Check whether `verifying_key` is `pub` on the struct — if it is, no change needed.

- [ ] **Step 6: Create stub `event.rs`, `frame.rs`, `handshake.rs`, `session.rs`**

Create each with just enough to compile:

`crates/harmony-tunnel/src/event.rs`:
```rust
/// Events fed into the TunnelSession by the caller.
#[derive(Debug)]
pub enum TunnelEvent {
    /// Raw bytes received from the transport (iroh-net connection).
    InboundBytes { data: alloc::vec::Vec<u8> },
    /// Periodic timer tick for keepalive management.
    Tick { now_ms: u64 },
}

/// Actions the TunnelSession asks the caller to perform.
#[derive(Debug)]
pub enum TunnelAction {
    /// Encrypted bytes to write to the transport.
    OutboundBytes { data: alloc::vec::Vec<u8> },
}
```

`crates/harmony-tunnel/src/frame.rs`:
```rust
// Frame encoding/decoding — implemented in Task 3.
```

`crates/harmony-tunnel/src/handshake.rs`:
```rust
// PQ handshake messages — implemented in Task 2.
```

`crates/harmony-tunnel/src/session.rs`:
```rust
use crate::error::TunnelError;
use crate::event::{TunnelAction, TunnelEvent};

/// Tunnel session states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunnelState {
    Idle,
    Initiating,
    Active,
    Closed,
}

/// Sans-I/O state machine for a single tunnel connection.
pub struct TunnelSession {
    state: TunnelState,
}

impl TunnelSession {
    /// Returns the current state.
    pub fn state(&self) -> TunnelState {
        self.state
    }
}
```

- [ ] **Step 7: Verify crate compiles**

Run: `cargo check -p harmony-tunnel`
Expected: Compiles with no errors (warnings about unused are OK).

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-tunnel/ crates/harmony-identity/src/pq_identity.rs Cargo.toml
git commit -m "feat(tunnel): scaffold harmony-tunnel crate with error types

Adds signing_key() accessor to PqPrivateIdentity for tunnel handshake use."
```

---

### Task 2: PQ Handshake Messages

**Files:**
- Create: `crates/harmony-tunnel/src/handshake.rs` (replace stub)

This task implements the `TunnelInit` and `TunnelAccept` message types — serialization, deserialization, signature creation, and signature verification. No state machine logic yet.

**Reference:** The handshake uses:
- `harmony_crypto::ml_kem::{encapsulate, decapsulate, MlKemPublicKey, MlKemSecretKey, MlKemCiphertext, MlKemSharedSecret}` — `CT_LENGTH = 1088`, `SS_LENGTH = 32`
- `harmony_crypto::ml_dsa::{sign, verify, MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature}` — `PK_LENGTH = 1952`, `SIG_LENGTH = 3309`
- `harmony_crypto::hash::blake3_hash` — for transcript hash
- `harmony_crypto::hkdf::derive_key` — for session key derivation
- `harmony_crypto::aead::{KEY_LENGTH}` — 32 bytes

- [ ] **Step 1: Write tests for TunnelInit serialization roundtrip**

In `crates/harmony-tunnel/src/handshake.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn tunnel_init_roundtrip() {
        let (ml_kem_pk, _ml_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (ml_dsa_pk, ml_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init, _shared_secret) = TunnelInit::create(
            &mut OsRng,
            &ml_kem_pk,   // responder's ML-KEM public key
            &ml_dsa_pk,   // initiator's ML-DSA public key
            &ml_dsa_sk,   // initiator's ML-DSA secret key
        ).unwrap();

        let bytes = init.to_bytes();
        let parsed = TunnelInit::from_bytes(&bytes).unwrap();

        assert_eq!(init.ciphertext.as_bytes(), parsed.ciphertext.as_bytes());
        assert_eq!(init.initiator_pubkey.as_bytes(), parsed.initiator_pubkey.as_bytes());
        assert_eq!(init.nonce, parsed.nonce);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel tunnel_init_roundtrip`
Expected: FAIL — `TunnelInit` not defined.

- [ ] **Step 3: Implement TunnelInit**

Replace `crates/harmony-tunnel/src/handshake.rs` with:

```rust
use alloc::vec::Vec;
use harmony_crypto::ml_dsa::{MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature};
use harmony_crypto::ml_kem::{MlKemCiphertext, MlKemPublicKey, MlKemSharedSecret};
use harmony_crypto::hash::blake3_hash;
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::error::TunnelError;

/// ML-KEM ciphertext length.
const CT_LEN: usize = 1088;
/// ML-DSA-65 public key length.
const DSA_PK_LEN: usize = 1952;
/// ML-DSA-65 signature length.
const SIG_LEN: usize = 3309;
/// Nonce length.
const NONCE_LEN: usize = 32;

/// First handshake message: initiator → responder.
///
/// Wire format: [CT_LEN ciphertext][DSA_PK_LEN pubkey][NONCE_LEN nonce][SIG_LEN signature]
pub struct TunnelInit {
    pub ciphertext: MlKemCiphertext,
    pub initiator_pubkey: MlDsaPublicKey,
    pub nonce: [u8; NONCE_LEN],
    pub signature: MlDsaSignature,
}

/// The total byte length of a serialized TunnelInit.
pub const TUNNEL_INIT_LEN: usize = CT_LEN + DSA_PK_LEN + NONCE_LEN + SIG_LEN;

impl TunnelInit {
    /// Create a TunnelInit message.
    ///
    /// `responder_kem_pk` is the responder's ML-KEM-768 public key (from AnnounceRecord or contacts).
    /// `initiator_dsa_pk` / `initiator_dsa_sk` are the initiator's ML-DSA-65 keypair.
    pub fn create(
        rng: &mut impl CryptoRngCore,
        responder_kem_pk: &MlKemPublicKey,
        initiator_dsa_pk: &MlDsaPublicKey,
        initiator_dsa_sk: &MlDsaSecretKey,
    ) -> Result<(Self, MlKemSharedSecret), TunnelError> {
        // 1. ML-KEM encapsulate to responder's public key
        let (ciphertext, shared_secret) =
            harmony_crypto::ml_kem::encapsulate(rng, responder_kem_pk)?;

        // 2. Generate random nonce
        let mut nonce = [0u8; NONCE_LEN];
        rng.fill_bytes(&mut nonce);

        // 3. Build signed payload: ciphertext || pubkey || nonce
        let mut signed_payload = Vec::with_capacity(CT_LEN + DSA_PK_LEN + NONCE_LEN);
        signed_payload.extend_from_slice(ciphertext.as_bytes());
        signed_payload.extend_from_slice(&initiator_dsa_pk.as_bytes());
        signed_payload.extend_from_slice(&nonce);

        // 4. Sign
        let signature = harmony_crypto::ml_dsa::sign(initiator_dsa_sk, &signed_payload)?;

        Ok((
            Self {
                ciphertext,
                initiator_pubkey: MlDsaPublicKey::from_bytes(&initiator_dsa_pk.as_bytes())
                    .map_err(|_| TunnelError::MalformedHandshake { reason: "pubkey clone failed" })?,
                nonce,
                signature,
            },
            shared_secret,
        ))
    }

    /// Serialize to wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(TUNNEL_INIT_LEN);
        buf.extend_from_slice(self.ciphertext.as_bytes());
        buf.extend_from_slice(&self.initiator_pubkey.as_bytes());
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(self.signature.as_bytes());
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < TUNNEL_INIT_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: TUNNEL_INIT_LEN,
                got: data.len(),
            });
        }

        let mut offset = 0;
        let ciphertext = MlKemCiphertext::from_bytes(&data[offset..offset + CT_LEN])?;
        offset += CT_LEN;

        let initiator_pubkey = MlDsaPublicKey::from_bytes(&data[offset..offset + DSA_PK_LEN])
            .map_err(|_| TunnelError::MalformedHandshake { reason: "invalid ML-DSA public key" })?;
        offset += DSA_PK_LEN;

        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&data[offset..offset + NONCE_LEN]);
        offset += NONCE_LEN;

        let signature = MlDsaSignature::from_bytes(&data[offset..offset + SIG_LEN])
            .map_err(|_| TunnelError::MalformedHandshake { reason: "invalid ML-DSA signature" })?;

        Ok(Self {
            ciphertext,
            initiator_pubkey,
            nonce,
            signature,
        })
    }

    /// Verify the initiator's signature over the signed payload.
    pub fn verify(&self) -> Result<(), TunnelError> {
        let mut signed_payload = Vec::with_capacity(CT_LEN + DSA_PK_LEN + NONCE_LEN);
        signed_payload.extend_from_slice(self.ciphertext.as_bytes());
        signed_payload.extend_from_slice(&self.initiator_pubkey.as_bytes());
        signed_payload.extend_from_slice(&self.nonce);

        harmony_crypto::ml_dsa::verify(&self.initiator_pubkey, &signed_payload, &self.signature)
            .map_err(|_| TunnelError::SignatureVerificationFailed)
    }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-tunnel tunnel_init_roundtrip`
Expected: PASS

- [ ] **Step 5: Write tests for TunnelAccept and transcript verification**

Add to `handshake.rs` tests:

```rust
    #[test]
    fn tunnel_accept_roundtrip() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        // Initiator creates TunnelInit
        let (init_msg, init_shared_secret) = TunnelInit::create(
            &mut OsRng,
            &resp_kem_pk,
            &init_dsa_pk,
            &init_dsa_sk,
        ).unwrap();

        // Responder creates TunnelAccept
        let (accept_msg, resp_shared_secret) = TunnelAccept::create(
            &mut OsRng,
            &resp_kem_sk,
            &resp_dsa_pk,
            &resp_dsa_sk,
            &init_msg,
        ).unwrap();

        // Both derived the same shared secret
        assert_eq!(init_shared_secret.as_bytes(), resp_shared_secret.as_bytes());

        // Roundtrip serialization
        let bytes = accept_msg.to_bytes();
        let parsed = TunnelAccept::from_bytes(&bytes).unwrap();
        assert_eq!(accept_msg.nonce, parsed.nonce);
    }

    #[test]
    fn transcript_signature_verifies() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init_msg, _) = TunnelInit::create(
            &mut OsRng, &resp_kem_pk, &init_dsa_pk, &init_dsa_sk,
        ).unwrap();

        let (accept_msg, _) = TunnelAccept::create(
            &mut OsRng, &resp_kem_sk, &resp_dsa_pk, &resp_dsa_sk, &init_msg,
        ).unwrap();

        // Verify the transcript signature
        let transcript = compute_transcript_hash(&init_msg, &accept_msg);
        harmony_crypto::ml_dsa::verify(
            &accept_msg.responder_pubkey,
            &transcript,
            &accept_msg.signature,
        ).unwrap();
    }
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel tunnel_accept`
Expected: FAIL — `TunnelAccept` not defined.

- [ ] **Step 7: Implement TunnelAccept and transcript hash**

Add to `handshake.rs`:

```rust
/// Second handshake message: responder → initiator.
///
/// Wire format: [DSA_PK_LEN pubkey][NONCE_LEN nonce][SIG_LEN signature]
pub struct TunnelAccept {
    pub responder_pubkey: MlDsaPublicKey,
    pub nonce: [u8; NONCE_LEN],
    pub signature: MlDsaSignature,
}

/// The total byte length of a serialized TunnelAccept.
pub const TUNNEL_ACCEPT_LEN: usize = DSA_PK_LEN + NONCE_LEN + SIG_LEN;

/// Compute the BLAKE3 transcript hash over both handshake messages.
///
/// Includes all fields of TunnelInit and the non-signature fields of TunnelAccept,
/// binding both peers to the full handshake transcript.
pub fn compute_transcript_hash(init: &TunnelInit, accept: &TunnelAccept) -> [u8; 32] {
    let mut transcript = Vec::with_capacity(TUNNEL_INIT_LEN + DSA_PK_LEN + NONCE_LEN);
    // Full TunnelInit (including signature — binds initiator's identity)
    transcript.extend_from_slice(init.ciphertext.as_bytes());
    transcript.extend_from_slice(&init.initiator_pubkey.as_bytes());
    transcript.extend_from_slice(&init.nonce);
    transcript.extend_from_slice(init.signature.as_bytes());
    // TunnelAccept fields before signature
    transcript.extend_from_slice(&accept.responder_pubkey.as_bytes());
    transcript.extend_from_slice(&accept.nonce);

    blake3_hash(&transcript)
}

impl TunnelAccept {
    /// Create a TunnelAccept message in response to a TunnelInit.
    ///
    /// Decapsulates the shared secret, signs the transcript hash.
    pub fn create(
        rng: &mut impl CryptoRngCore,
        responder_kem_sk: &harmony_crypto::ml_kem::MlKemSecretKey,
        responder_dsa_pk: &MlDsaPublicKey,
        responder_dsa_sk: &MlDsaSecretKey,
        init: &TunnelInit,
    ) -> Result<(Self, MlKemSharedSecret), TunnelError> {
        // 1. Verify initiator's signature
        init.verify()?;

        // 2. Decapsulate shared secret
        let shared_secret =
            harmony_crypto::ml_kem::decapsulate(responder_kem_sk, &init.ciphertext)?;

        // 3. Generate responder nonce
        let mut nonce = [0u8; NONCE_LEN];
        rng.fill_bytes(&mut nonce);

        // 4. Build partial accept (without signature) for transcript
        let partial = Self {
            responder_pubkey: MlDsaPublicKey::from_bytes(&responder_dsa_pk.as_bytes())
                .map_err(|_| TunnelError::MalformedHandshake { reason: "pubkey clone failed" })?,
            nonce,
            signature: MlDsaSignature::from_bytes(
                &alloc::vec![0u8; SIG_LEN]
            ).map_err(|_| TunnelError::MalformedHandshake { reason: "zero sig" })?,
        };

        // 5. Compute transcript hash and sign
        let transcript = compute_transcript_hash(init, &partial);
        let signature = harmony_crypto::ml_dsa::sign(responder_dsa_sk, &transcript)?;

        Ok((
            Self {
                responder_pubkey: partial.responder_pubkey,
                nonce,
                signature,
            },
            shared_secret,
        ))
    }

    /// Serialize to wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(TUNNEL_ACCEPT_LEN);
        buf.extend_from_slice(&self.responder_pubkey.as_bytes());
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(self.signature.as_bytes());
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < TUNNEL_ACCEPT_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: TUNNEL_ACCEPT_LEN,
                got: data.len(),
            });
        }

        let mut offset = 0;
        let responder_pubkey = MlDsaPublicKey::from_bytes(&data[offset..offset + DSA_PK_LEN])
            .map_err(|_| TunnelError::MalformedHandshake { reason: "invalid ML-DSA public key" })?;
        offset += DSA_PK_LEN;

        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&data[offset..offset + NONCE_LEN]);
        offset += NONCE_LEN;

        let signature = MlDsaSignature::from_bytes(&data[offset..offset + SIG_LEN])
            .map_err(|_| TunnelError::MalformedHandshake { reason: "invalid ML-DSA signature" })?;

        Ok(Self {
            responder_pubkey,
            nonce,
            signature,
        })
    }

    /// Verify the responder's transcript signature.
    ///
    /// Caller must provide the original TunnelInit to reconstruct the transcript.
    pub fn verify(&self, init: &TunnelInit) -> Result<(), TunnelError> {
        let transcript = compute_transcript_hash(init, self);
        harmony_crypto::ml_dsa::verify(&self.responder_pubkey, &transcript, &self.signature)
            .map_err(|_| TunnelError::SignatureVerificationFailed)
    }
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cargo test -p harmony-tunnel`
Expected: All 3 tests pass.

- [ ] **Step 9: Write test for session key derivation**

Add to `handshake.rs` tests:

```rust
    #[test]
    fn session_keys_derived_correctly() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init_msg, init_ss) = TunnelInit::create(
            &mut OsRng, &resp_kem_pk, &init_dsa_pk, &init_dsa_sk,
        ).unwrap();

        let (accept_msg, resp_ss) = TunnelAccept::create(
            &mut OsRng, &resp_kem_sk, &resp_dsa_pk, &resp_dsa_sk, &init_msg,
        ).unwrap();

        // Both sides derive the same key pair
        let init_keys = derive_session_keys(init_ss.as_bytes(), &init_msg.nonce, &accept_msg.nonce);
        let resp_keys = derive_session_keys(resp_ss.as_bytes(), &init_msg.nonce, &accept_msg.nonce);

        // Initiator's send key == Responder's receive key
        assert_eq!(init_keys.i2r_key, resp_keys.i2r_key);
        // Responder's send key == Initiator's receive key
        assert_eq!(init_keys.r2i_key, resp_keys.r2i_key);
        // Keys are different from each other
        assert_ne!(init_keys.i2r_key, init_keys.r2i_key);
    }
```

- [ ] **Step 10: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel session_keys_derived`
Expected: FAIL — `derive_session_keys` and `SessionKeys` not defined.

- [ ] **Step 11: Implement session key derivation**

Add to `handshake.rs`:

```rust
use harmony_crypto::aead::KEY_LENGTH;

/// Directional session keys derived from the handshake.
pub struct SessionKeys {
    /// Initiator-to-responder encryption key (32 bytes).
    pub i2r_key: [u8; KEY_LENGTH],
    /// Responder-to-initiator encryption key (32 bytes).
    pub r2i_key: [u8; KEY_LENGTH],
}

impl Drop for SessionKeys {
    fn drop(&mut self) {
        self.i2r_key.zeroize();
        self.r2i_key.zeroize();
    }
}

/// Derive directional session keys from the ML-KEM shared secret and both nonces.
///
/// Uses HKDF-SHA256 with:
/// - IKM: the ML-KEM shared secret (32 bytes)
/// - Salt: nonce_i || nonce_r (64 bytes)
/// - Info: "i2r" or "r2i" (directional labels)
pub fn derive_session_keys(
    shared_secret: &[u8],
    nonce_i: &[u8; NONCE_LEN],
    nonce_r: &[u8; NONCE_LEN],
) -> SessionKeys {
    let mut salt = [0u8; NONCE_LEN * 2];
    salt[..NONCE_LEN].copy_from_slice(nonce_i);
    salt[NONCE_LEN..].copy_from_slice(nonce_r);

    let i2r_bytes = harmony_crypto::hkdf::derive_key(
        shared_secret,
        Some(&salt),
        b"i2r",
        KEY_LENGTH,
    )
    .expect("HKDF-SHA256 with 32-byte output cannot fail");

    let r2i_bytes = harmony_crypto::hkdf::derive_key(
        shared_secret,
        Some(&salt),
        b"r2i",
        KEY_LENGTH,
    )
    .expect("HKDF-SHA256 with 32-byte output cannot fail");

    let mut i2r_key = [0u8; KEY_LENGTH];
    let mut r2i_key = [0u8; KEY_LENGTH];
    i2r_key.copy_from_slice(&i2r_bytes);
    r2i_key.copy_from_slice(&r2i_bytes);

    SessionKeys { i2r_key, r2i_key }
}
```

- [ ] **Step 12: Run test to verify it passes**

Run: `cargo test -p harmony-tunnel session_keys_derived`
Expected: PASS

- [ ] **Step 13: Commit**

```bash
git add crates/harmony-tunnel/src/handshake.rs
git commit -m "feat(tunnel): PQ handshake messages with transcript signing and key derivation"
```

---

### Task 3: Frame Encoding, Encryption, and Decryption

**Files:**
- Create: `crates/harmony-tunnel/src/frame.rs` (replace stub)

Frames are the unit of data exchanged over an established tunnel. Each frame is encrypted as a complete AEAD ciphertext (tag + length + payload → encrypt → send).

- [ ] **Step 1: Write test for frame roundtrip (plaintext encoding)**

In `crates/harmony-tunnel/src/frame.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_encode_decode_roundtrip() {
        let frame = Frame {
            tag: FrameTag::Reticulum,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        let bytes = frame.encode();
        assert_eq!(bytes.len(), 1 + 2 + 4); // tag + length + payload

        let decoded = Frame::decode(&bytes).unwrap();
        assert_eq!(decoded.tag, FrameTag::Reticulum);
        assert_eq!(decoded.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn keepalive_frame_has_empty_payload() {
        let frame = Frame::keepalive();
        let bytes = frame.encode();
        assert_eq!(bytes.len(), 3); // tag + length(0)
        assert_eq!(bytes[0], 0x00);
        assert_eq!(&bytes[1..3], &[0x00, 0x00]);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel frame_encode_decode`
Expected: FAIL — `Frame`, `FrameTag` not defined.

- [ ] **Step 3: Implement Frame and FrameTag**

Replace `crates/harmony-tunnel/src/frame.rs`:

```rust
use alloc::vec::Vec;
use crate::error::TunnelError;

/// Frame type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameTag {
    Keepalive = 0x00,
    Reticulum = 0x01,
    Zenoh = 0x02,
}

impl FrameTag {
    pub fn from_byte(b: u8) -> Result<Self, TunnelError> {
        match b {
            0x00 => Ok(Self::Keepalive),
            0x01 => Ok(Self::Reticulum),
            0x02 => Ok(Self::Zenoh),
            other => Err(TunnelError::UnknownFrameTag { tag: other }),
        }
    }
}

/// A plaintext frame before encryption / after decryption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub tag: FrameTag,
    pub payload: Vec<u8>,
}

/// Frame header size: 1 byte tag + 2 bytes length.
const FRAME_HEADER_LEN: usize = 3;

impl Frame {
    /// Create a keepalive frame.
    pub fn keepalive() -> Self {
        Self {
            tag: FrameTag::Keepalive,
            payload: Vec::new(),
        }
    }

    /// Encode the frame to wire format (tag + big-endian length + payload).
    pub fn encode(&self) -> Vec<u8> {
        let len = self.payload.len() as u16;
        let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + self.payload.len());
        buf.push(self.tag as u8);
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < FRAME_HEADER_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: FRAME_HEADER_LEN,
                got: data.len(),
            });
        }

        let tag = FrameTag::from_byte(data[0])?;
        let len = u16::from_be_bytes([data[1], data[2]]) as usize;

        if data.len() < FRAME_HEADER_LEN + len {
            return Err(TunnelError::FrameTooShort {
                expected: FRAME_HEADER_LEN + len,
                got: data.len(),
            });
        }

        let payload = data[FRAME_HEADER_LEN..FRAME_HEADER_LEN + len].to_vec();

        Ok(Self { tag, payload })
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-tunnel frame_`
Expected: Both frame tests PASS.

- [ ] **Step 5: Write test for encrypted frame roundtrip**

Add to `frame.rs` tests:

```rust
    use rand::rngs::OsRng;
    use harmony_crypto::aead::KEY_LENGTH;

    #[test]
    fn encrypted_frame_roundtrip() {
        let key = [0x42u8; KEY_LENGTH];
        let aad = [0xABu8; 32]; // NodeId as AAD

        let frame = Frame {
            tag: FrameTag::Zenoh,
            payload: b"hello zenoh".to_vec(),
        };

        let mut nonce_counter: u64 = 0;
        let encrypted = encrypt_frame(&frame, &key, &aad, &mut nonce_counter).unwrap();
        assert_eq!(nonce_counter, 1); // Counter incremented

        let mut decrypt_counter: u64 = 0;
        let decrypted = decrypt_frame(&encrypted, &key, &aad, &mut decrypt_counter).unwrap();
        assert_eq!(decrypt_counter, 1);

        assert_eq!(decrypted.tag, FrameTag::Zenoh);
        assert_eq!(decrypted.payload, b"hello zenoh");
    }

    #[test]
    fn wrong_aad_fails_decryption() {
        let key = [0x42u8; KEY_LENGTH];
        let aad = [0xABu8; 32];
        let wrong_aad = [0xCDu8; 32];

        let frame = Frame {
            tag: FrameTag::Reticulum,
            payload: b"secret".to_vec(),
        };

        let mut enc_counter: u64 = 0;
        let encrypted = encrypt_frame(&frame, &key, &aad, &mut enc_counter).unwrap();

        let mut dec_counter: u64 = 0;
        let result = decrypt_frame(&encrypted, &key, &wrong_aad, &mut dec_counter);
        assert!(result.is_err());
    }
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel encrypted_frame`
Expected: FAIL — `encrypt_frame` / `decrypt_frame` not defined.

- [ ] **Step 7: Implement frame encryption and decryption**

Add to `frame.rs`:

```rust
use harmony_crypto::aead::{self, KEY_LENGTH, NONCE_LENGTH, TAG_LENGTH};

/// Build a 12-byte AEAD nonce from a 64-bit counter.
///
/// Format: [4 bytes zero padding][8 bytes big-endian counter]
fn counter_to_nonce(counter: u64) -> [u8; NONCE_LENGTH] {
    let mut nonce = [0u8; NONCE_LENGTH];
    nonce[4..].copy_from_slice(&counter.to_be_bytes());
    nonce
}

/// Encrypt a frame as a single AEAD ciphertext.
///
/// The entire plaintext frame (tag + length + payload) is encrypted.
/// The nonce counter is incremented after each call.
/// AAD should be the remote peer's NodeId (32 bytes).
pub fn encrypt_frame(
    frame: &Frame,
    key: &[u8; KEY_LENGTH],
    aad: &[u8],
    nonce_counter: &mut u64,
) -> Result<Vec<u8>, TunnelError> {
    let plaintext = frame.encode();
    let nonce = counter_to_nonce(*nonce_counter);
    *nonce_counter += 1;

    aead::encrypt(key, &nonce, &plaintext, aad).map_err(TunnelError::Crypto)
}

/// Decrypt a ciphertext back into a frame.
///
/// Verifies the AEAD tag, then decodes the plaintext frame.
/// The nonce counter is incremented after each call.
pub fn decrypt_frame(
    ciphertext: &[u8],
    key: &[u8; KEY_LENGTH],
    aad: &[u8],
    nonce_counter: &mut u64,
) -> Result<Frame, TunnelError> {
    let nonce = counter_to_nonce(*nonce_counter);
    *nonce_counter += 1;

    let plaintext =
        aead::decrypt(key, &nonce, ciphertext, aad).map_err(|_| TunnelError::DecryptionFailed)?;

    Frame::decode(&plaintext)
}
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `cargo test -p harmony-tunnel`
Expected: All frame tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-tunnel/src/frame.rs
git commit -m "feat(tunnel): frame encoding/decoding with AEAD encryption"
```

---

### Task 4: TunnelEvent and TunnelAction Enums

**Files:**
- Create: `crates/harmony-tunnel/src/event.rs` (replace stub)

- [ ] **Step 1: Implement complete event and action enums**

Replace `crates/harmony-tunnel/src/event.rs`:

```rust
use alloc::vec::Vec;

/// Events fed into the TunnelSession by the caller.
#[derive(Debug)]
pub enum TunnelEvent {
    /// Raw bytes received from the transport (iroh-net connection).
    InboundBytes { data: Vec<u8> },
    /// Send a Reticulum packet through this tunnel.
    SendReticulum { packet: Vec<u8> },
    /// Send a Zenoh message through this tunnel.
    SendZenoh { message: Vec<u8> },
    /// Periodic timer tick for keepalive management.
    Tick { now_ms: u64 },
    /// Request graceful tunnel shutdown.
    Close,
}

/// Actions the TunnelSession asks the caller to perform.
#[derive(Debug)]
pub enum TunnelAction {
    /// Encrypted bytes to write to the transport.
    OutboundBytes { data: Vec<u8> },
    /// A decrypted Reticulum packet received from the tunnel peer.
    ReticulumReceived { packet: Vec<u8> },
    /// A decrypted Zenoh message received from the tunnel peer.
    ZenohReceived { message: Vec<u8> },
    /// Handshake completed — the peer's PQ identity has been authenticated.
    HandshakeComplete {
        peer_dsa_pubkey: Vec<u8>,
        peer_node_id: [u8; 32],
    },
    /// An error occurred in the tunnel.
    Error { reason: alloc::string::String },
    /// The tunnel has been closed.
    Closed,
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p harmony-tunnel`
Expected: Compiles.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-tunnel/src/event.rs
git commit -m "feat(tunnel): define TunnelEvent and TunnelAction enums"
```

---

### Task 5: TunnelSession State Machine — Initiator Path

**Files:**
- Create: `crates/harmony-tunnel/src/session.rs` (replace stub)

The core state machine. This task implements the **initiator** side (the peer that sends TunnelInit and waits for TunnelAccept).

- [ ] **Step 1: Write test for initiator handshake flow**

In `crates/harmony-tunnel/src/session.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    /// Helper: create a paired initiator and responder identity.
    fn create_test_identities() -> (
        harmony_identity::PqPrivateIdentity,
        harmony_identity::PqPrivateIdentity,
    ) {
        let initiator = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        let responder = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
        (initiator, responder)
    }

    #[test]
    fn initiator_emits_tunnel_init_on_creation() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        let (session, actions) = TunnelSession::new_initiator(
            &mut OsRng,
            &initiator_id,
            responder_pub,
        ).unwrap();

        assert_eq!(session.state(), TunnelState::Initiating);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], TunnelAction::OutboundBytes { .. }));
    }

    #[test]
    fn initiator_completes_handshake_on_valid_accept() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        // Initiator sends TunnelInit
        let (mut initiator, init_actions) = TunnelSession::new_initiator(
            &mut OsRng,
            &initiator_id,
            responder_pub,
        ).unwrap();

        // Extract the TunnelInit bytes
        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Responder processes TunnelInit and creates TunnelAccept
        let (mut responder, accept_actions) = TunnelSession::new_responder(
            &mut OsRng,
            &responder_id,
            &init_bytes,
        ).unwrap();

        assert_eq!(responder.state(), TunnelState::Active);

        // Extract the TunnelAccept bytes
        let accept_bytes = accept_actions.iter().find_map(|a| match a {
            TunnelAction::OutboundBytes { data } => Some(data.clone()),
            _ => None,
        }).expect("responder should emit OutboundBytes");

        // Initiator processes TunnelAccept
        let actions = initiator.handle_event(
            TunnelEvent::InboundBytes { data: accept_bytes },
        ).unwrap();

        assert_eq!(initiator.state(), TunnelState::Active);
        assert!(actions.iter().any(|a| matches!(a, TunnelAction::HandshakeComplete { .. })));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-tunnel initiator_emits`
Expected: FAIL — `TunnelSession::new_initiator` not defined.

- [ ] **Step 3: Implement TunnelSession with initiator and responder constructors**

Replace `crates/harmony-tunnel/src/session.rs`:

```rust
use alloc::string::ToString;
use alloc::vec::Vec;
use harmony_crypto::aead::KEY_LENGTH;
use harmony_crypto::hash::blake3_hash;
use harmony_identity::{PqIdentity, PqPrivateIdentity};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::error::TunnelError;
use crate::event::{TunnelAction, TunnelEvent};
use crate::frame::{decrypt_frame, encrypt_frame, Frame, FrameTag};
use crate::handshake::{
    compute_transcript_hash, derive_session_keys, TunnelAccept, TunnelInit,
};

/// Keepalive interval in milliseconds.
const KEEPALIVE_INTERVAL_MS: u64 = 30_000;
/// Dead peer timeout: 3 missed keepalives.
const DEAD_TIMEOUT_MS: u64 = KEEPALIVE_INTERVAL_MS * 3;

/// Tunnel session states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunnelState {
    Idle,
    Initiating,
    Active,
    Closed,
}

/// Derive the 32-byte NodeId from an ML-DSA-65 public key.
pub fn node_id_from_dsa_pubkey(pubkey: &harmony_crypto::ml_dsa::MlDsaPublicKey) -> [u8; 32] {
    blake3_hash(&pubkey.as_bytes())
}

/// Sans-I/O state machine for a single tunnel connection.
pub struct TunnelSession {
    state: TunnelState,
    is_initiator: bool,
    /// Send key (our direction).
    send_key: [u8; KEY_LENGTH],
    /// Receive key (peer's direction).
    recv_key: [u8; KEY_LENGTH],
    /// Outbound nonce counter.
    send_nonce: u64,
    /// Inbound nonce counter.
    recv_nonce: u64,
    /// Remote peer's NodeId (BLAKE3 of their ML-DSA pubkey), used as AAD.
    remote_node_id: [u8; 32],
    /// Our NodeId, used as AAD for the remote's receive path.
    local_node_id: [u8; 32],
    /// Timestamp of last received data (for keepalive timeout).
    last_received_ms: u64,
    /// Timestamp of last sent data (for keepalive scheduling).
    last_sent_ms: u64,
    // -- Initiator-only state for completing handshake --
    /// Saved TunnelInit for transcript verification (initiator only).
    pending_init: Option<TunnelInit>,
    /// Saved shared secret for key derivation (initiator only, zeroized after use).
    pending_shared_secret: Option<Vec<u8>>,
    /// Initiator's nonce (saved for key derivation).
    init_nonce: [u8; 32],
    /// Expected responder's ML-DSA public key bytes (initiator only, for identity verification).
    expected_responder_pubkey: Option<alloc::vec::Vec<u8>>,
}

impl Drop for TunnelSession {
    fn drop(&mut self) {
        self.send_key.zeroize();
        self.recv_key.zeroize();
        if let Some(ref mut ss) = self.pending_shared_secret {
            ss.zeroize();
        }
    }
}

impl TunnelSession {
    /// Returns the current state.
    pub fn state(&self) -> TunnelState {
        self.state
    }

    /// Create a new tunnel session as the **initiator**.
    ///
    /// Generates and returns the TunnelInit message as an `OutboundBytes` action.
    /// The session transitions to `Initiating` and waits for a TunnelAccept.
    pub fn new_initiator(
        rng: &mut impl CryptoRngCore,
        local_identity: &PqPrivateIdentity,
        remote_identity: &PqIdentity,
    ) -> Result<(Self, Vec<TunnelAction>), TunnelError> {
        let local_pub = local_identity.public_identity();
        let local_dsa_pk = &local_pub.verifying_key;
        let local_dsa_sk = local_identity.signing_key();
        let remote_kem_pk = &remote_identity.encryption_key;

        let (init_msg, shared_secret) =
            TunnelInit::create(rng, remote_kem_pk, local_dsa_pk, local_dsa_sk)?;

        let init_bytes = init_msg.to_bytes();
        let init_nonce = init_msg.nonce;

        let local_node_id = node_id_from_dsa_pubkey(local_dsa_pk);
        let remote_node_id = node_id_from_dsa_pubkey(&remote_identity.verifying_key);

        let session = Self {
            state: TunnelState::Initiating,
            is_initiator: true,
            send_key: [0u8; KEY_LENGTH],
            recv_key: [0u8; KEY_LENGTH],
            send_nonce: 0,
            recv_nonce: 0,
            remote_node_id,
            local_node_id,
            last_received_ms: 0,
            last_sent_ms: 0,
            pending_init: Some(init_msg),
            pending_shared_secret: Some(shared_secret.as_bytes().to_vec()),
            init_nonce,
            expected_responder_pubkey: Some(remote_identity.verifying_key.as_bytes()),
        };

        Ok((session, vec![TunnelAction::OutboundBytes { data: init_bytes }]))
    }

    /// Create a new tunnel session as the **responder**.
    ///
    /// Processes the incoming TunnelInit, derives session keys, and returns
    /// the TunnelAccept as an `OutboundBytes` action plus a `HandshakeComplete`.
    pub fn new_responder(
        rng: &mut impl CryptoRngCore,
        local_identity: &PqPrivateIdentity,
        init_bytes: &[u8],
    ) -> Result<(Self, Vec<TunnelAction>), TunnelError> {
        let init_msg = TunnelInit::from_bytes(init_bytes)?;

        let local_pub = local_identity.public_identity();
        let local_dsa_pk = &local_pub.verifying_key;
        let local_dsa_sk = local_identity.signing_key();
        let local_kem_sk = local_identity.encryption_secret();

        let (accept_msg, shared_secret) =
            TunnelAccept::create(rng, local_kem_sk, local_dsa_pk, local_dsa_sk, &init_msg)?;

        let accept_bytes = accept_msg.to_bytes();

        // Derive directional keys
        let keys = derive_session_keys(
            shared_secret.as_bytes(),
            &init_msg.nonce,
            &accept_msg.nonce,
        );

        let remote_node_id = node_id_from_dsa_pubkey(&init_msg.initiator_pubkey);
        let local_node_id = node_id_from_dsa_pubkey(local_dsa_pk);

        let session = Self {
            state: TunnelState::Active,
            is_initiator: false,
            // Responder sends on r2i, receives on i2r
            send_key: keys.r2i_key,
            recv_key: keys.i2r_key,
            send_nonce: 0,
            recv_nonce: 0,
            remote_node_id,
            local_node_id,
            last_received_ms: 0,
            last_sent_ms: 0,
            pending_init: None,
            pending_shared_secret: None,
            init_nonce: [0u8; 32],
            expected_responder_pubkey: None,
        };

        let actions = vec![
            TunnelAction::OutboundBytes { data: accept_bytes },
            TunnelAction::HandshakeComplete {
                peer_dsa_pubkey: init_msg.initiator_pubkey.as_bytes(),
                peer_node_id: remote_node_id,
            },
        ];

        Ok((session, actions))
    }

    /// Process a single event and return resulting actions.
    pub fn handle_event(
        &mut self,
        event: TunnelEvent,
    ) -> Result<Vec<TunnelAction>, TunnelError> {
        match event {
            TunnelEvent::InboundBytes { data } => self.handle_inbound(data),
            TunnelEvent::SendReticulum { packet } => self.handle_send(FrameTag::Reticulum, packet),
            TunnelEvent::SendZenoh { message } => self.handle_send(FrameTag::Zenoh, message),
            TunnelEvent::Tick { now_ms } => self.handle_tick(now_ms),
            TunnelEvent::Close => self.handle_close(),
        }
    }

    fn handle_inbound(&mut self, data: Vec<u8>) -> Result<Vec<TunnelAction>, TunnelError> {
        match self.state {
            TunnelState::Initiating => self.handle_accept_response(data),
            TunnelState::Active => self.handle_encrypted_frame(data),
            _ => Err(TunnelError::InvalidState),
        }
    }

    /// Initiator processes TunnelAccept to complete handshake.
    fn handle_accept_response(
        &mut self,
        data: Vec<u8>,
    ) -> Result<Vec<TunnelAction>, TunnelError> {
        let accept_msg = TunnelAccept::from_bytes(&data)?;

        // Verify responder is the expected peer (MITM protection)
        if let Some(ref expected) = self.expected_responder_pubkey {
            if accept_msg.responder_pubkey.as_bytes() != *expected {
                return Err(TunnelError::SignatureVerificationFailed);
            }
        }

        let init_msg = self
            .pending_init
            .take()
            .ok_or(TunnelError::InvalidState)?;

        // Verify transcript signature
        accept_msg.verify(&init_msg)?;

        // Derive session keys
        let mut shared_secret = self
            .pending_shared_secret
            .take()
            .ok_or(TunnelError::InvalidState)?;

        let keys = derive_session_keys(&shared_secret, &init_msg.nonce, &accept_msg.nonce);
        shared_secret.zeroize();

        // Initiator sends on i2r, receives on r2i
        self.send_key = keys.i2r_key;
        self.recv_key = keys.r2i_key;
        self.state = TunnelState::Active;

        let remote_node_id = node_id_from_dsa_pubkey(&accept_msg.responder_pubkey);
        self.remote_node_id = remote_node_id;

        Ok(vec![TunnelAction::HandshakeComplete {
            peer_dsa_pubkey: accept_msg.responder_pubkey.as_bytes(),
            peer_node_id: remote_node_id,
        }])
    }

    fn handle_encrypted_frame(
        &mut self,
        data: Vec<u8>,
    ) -> Result<Vec<TunnelAction>, TunnelError> {
        let frame = decrypt_frame(&data, &self.recv_key, &self.local_node_id, &mut self.recv_nonce)?;

        match frame.tag {
            FrameTag::Keepalive => Ok(Vec::new()),
            FrameTag::Reticulum => Ok(vec![TunnelAction::ReticulumReceived {
                packet: frame.payload,
            }]),
            FrameTag::Zenoh => Ok(vec![TunnelAction::ZenohReceived {
                message: frame.payload,
            }]),
        }
    }

    fn handle_send(
        &mut self,
        tag: FrameTag,
        payload: Vec<u8>,
    ) -> Result<Vec<TunnelAction>, TunnelError> {
        if self.state != TunnelState::Active {
            return Err(TunnelError::InvalidState);
        }

        let frame = Frame { tag, payload };
        let encrypted =
            encrypt_frame(&frame, &self.send_key, &self.remote_node_id, &mut self.send_nonce)?;

        Ok(vec![TunnelAction::OutboundBytes { data: encrypted }])
    }

    fn handle_tick(&mut self, now_ms: u64) -> Result<Vec<TunnelAction>, TunnelError> {
        if self.state != TunnelState::Active {
            return Ok(Vec::new());
        }

        let mut actions = Vec::new();

        // Check for dead peer
        if self.last_received_ms > 0
            && now_ms.saturating_sub(self.last_received_ms) >= DEAD_TIMEOUT_MS
        {
            self.state = TunnelState::Closed;
            actions.push(TunnelAction::Error {
                reason: "keepalive timeout".to_string(),
            });
            actions.push(TunnelAction::Closed);
            return Ok(actions);
        }

        // Send keepalive if interval elapsed
        if now_ms.saturating_sub(self.last_sent_ms) >= KEEPALIVE_INTERVAL_MS {
            let frame = Frame::keepalive();
            let encrypted = encrypt_frame(
                &frame,
                &self.send_key,
                &self.remote_node_id,
                &mut self.send_nonce,
            )?;
            self.last_sent_ms = now_ms;
            actions.push(TunnelAction::OutboundBytes { data: encrypted });
        }

        Ok(actions)
    }

    fn handle_close(&mut self) -> Result<Vec<TunnelAction>, TunnelError> {
        self.state = TunnelState::Closed;
        self.send_key.zeroize();
        self.recv_key.zeroize();
        Ok(vec![TunnelAction::Closed])
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-tunnel`
Expected: All tests pass (initiator + responder tests, handshake tests, frame tests).

Note: this step may require adjusting the code based on exact `PqPrivateIdentity` API — specifically whether `signing_key()` and `encryption_secret()` are public methods. Check `crates/harmony-identity/src/pq_identity.rs` and adjust accessor names if needed. If these accessors don't exist yet, add them as `pub fn signing_key(&self) -> &MlDsaSecretKey` and `pub fn encryption_secret(&self) -> &MlKemSecretKey` to `PqPrivateIdentity`.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-tunnel/src/session.rs crates/harmony-tunnel/src/event.rs
git commit -m "feat(tunnel): TunnelSession state machine with initiator and responder paths"
```

---

### Task 6: Data Transfer Tests — Reticulum and Zenoh Frames

**Files:**
- Modify: `crates/harmony-tunnel/src/session.rs` (add tests)

End-to-end tests that verify data flows correctly through established tunnels in both directions.

- [ ] **Step 1: Write bidirectional data transfer test**

Add to `session.rs` tests:

```rust
    #[test]
    fn bidirectional_data_transfer() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        // Complete handshake
        let (mut initiator, init_actions) = TunnelSession::new_initiator(
            &mut OsRng, &initiator_id, &responder_pub,
        ).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let (mut responder, accept_actions) = TunnelSession::new_responder(
            &mut OsRng, &responder_id, &init_bytes,
        ).unwrap();

        let accept_bytes = accept_actions.iter().find_map(|a| match a {
            TunnelAction::OutboundBytes { data } => Some(data.clone()),
            _ => None,
        }).unwrap();

        initiator.handle_event(TunnelEvent::InboundBytes { data: accept_bytes }).unwrap();

        // Initiator sends Reticulum packet
        let actions = initiator.handle_event(TunnelEvent::SendReticulum {
            packet: b"reticulum-packet".to_vec(),
        }).unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Responder receives it
        let actions = responder.handle_event(TunnelEvent::InboundBytes {
            data: encrypted,
        }).unwrap();

        assert!(matches!(&actions[0], TunnelAction::ReticulumReceived { packet } if packet == b"reticulum-packet"));

        // Responder sends Zenoh message back
        let actions = responder.handle_event(TunnelEvent::SendZenoh {
            message: b"zenoh-message".to_vec(),
        }).unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Initiator receives it
        let actions = initiator.handle_event(TunnelEvent::InboundBytes {
            data: encrypted,
        }).unwrap();

        assert!(matches!(&actions[0], TunnelAction::ZenohReceived { message } if message == b"zenoh-message"));
    }

    #[test]
    fn truncated_tunnel_init_rejected() {
        let (_, responder_id) = create_test_identities();

        let result = TunnelSession::new_responder(
            &mut OsRng,
            &responder_id,
            &[0u8; 100], // Far too short for a valid TunnelInit
        );

        assert!(result.is_err());
    }

    #[test]
    fn wrong_responder_identity_rejected() {
        let (initiator_id, responder_id) = create_test_identities();
        let (_, impersonator_id) = create_test_identities();
        let responder_pub = responder_id.public_identity();

        // Initiator expects responder_id
        let (mut initiator, init_actions) = TunnelSession::new_initiator(
            &mut OsRng, &initiator_id, responder_pub,
        ).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Impersonator responds instead
        let (_, accept_actions) = TunnelSession::new_responder(
            &mut OsRng, &impersonator_id, &init_bytes,
        ).unwrap();

        let accept_bytes = accept_actions.iter().find_map(|a| match a {
            TunnelAction::OutboundBytes { data } => Some(data.clone()),
            _ => None,
        }).unwrap();

        // Initiator should reject — wrong responder identity
        let result = initiator.handle_event(
            TunnelEvent::InboundBytes { data: accept_bytes },
        );
        assert!(result.is_err());
    }

    #[test]
    fn send_before_handshake_fails() {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        let (mut session, _) = TunnelSession::new_initiator(
            &mut OsRng, &initiator_id, &responder_pub,
        ).unwrap();

        // Try to send before handshake completes
        let result = session.handle_event(TunnelEvent::SendReticulum {
            packet: b"too-early".to_vec(),
        });

        assert!(result.is_err());
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-tunnel`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-tunnel/src/session.rs
git commit -m "test(tunnel): bidirectional data transfer and pre-handshake rejection"
```

---

### Task 7: Keepalive and Timeout Tests

**Files:**
- Modify: `crates/harmony-tunnel/src/session.rs` (add tests)

- [ ] **Step 1: Write keepalive and timeout tests**

Add to `session.rs` tests:

```rust
    /// Helper: complete a full handshake and return both sessions.
    fn complete_handshake() -> (TunnelSession, TunnelSession) {
        let (initiator_id, responder_id) = create_test_identities();
        let responder_pub = responder_id.public_identity().clone();

        let (mut initiator, init_actions) = TunnelSession::new_initiator(
            &mut OsRng, &initiator_id, &responder_pub,
        ).unwrap();

        let init_bytes = match &init_actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        let (responder, accept_actions) = TunnelSession::new_responder(
            &mut OsRng, &responder_id, &init_bytes,
        ).unwrap();

        let accept_bytes = accept_actions.iter().find_map(|a| match a {
            TunnelAction::OutboundBytes { data } => Some(data.clone()),
            _ => None,
        }).unwrap();

        initiator.handle_event(TunnelEvent::InboundBytes { data: accept_bytes }).unwrap();

        (initiator, responder)
    }

    #[test]
    fn keepalive_sent_after_interval() {
        let (mut session, _) = complete_handshake();

        // First tick at 0ms — no keepalive yet (just started)
        session.last_sent_ms = 0;
        session.last_received_ms = 0;
        let actions = session.handle_event(TunnelEvent::Tick { now_ms: 0 }).unwrap();
        // At t=0 with last_sent=0, the interval check triggers
        assert!(actions.iter().any(|a| matches!(a, TunnelAction::OutboundBytes { .. })));

        // Tick before interval — no keepalive
        let actions = session.handle_event(TunnelEvent::Tick { now_ms: 15_000 }).unwrap();
        assert!(actions.is_empty());

        // Tick at interval — keepalive sent
        let actions = session.handle_event(TunnelEvent::Tick { now_ms: 30_001 }).unwrap();
        assert!(actions.iter().any(|a| matches!(a, TunnelAction::OutboundBytes { .. })));
    }

    #[test]
    fn dead_peer_timeout() {
        let (mut session, _) = complete_handshake();

        // Set last_received to some known time
        session.last_received_ms = 1000;

        // Tick well past dead timeout (1000 + 90000 = 91000)
        let actions = session.handle_event(TunnelEvent::Tick { now_ms: 91_001 }).unwrap();

        assert_eq!(session.state(), TunnelState::Closed);
        assert!(actions.iter().any(|a| matches!(a, TunnelAction::Closed)));
    }

    #[test]
    fn close_transitions_to_closed() {
        let (mut session, _) = complete_handshake();
        assert_eq!(session.state(), TunnelState::Active);

        let actions = session.handle_event(TunnelEvent::Close).unwrap();
        assert_eq!(session.state(), TunnelState::Closed);
        assert!(actions.iter().any(|a| matches!(a, TunnelAction::Closed)));
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-tunnel`
Expected: All tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-tunnel/src/session.rs
git commit -m "test(tunnel): keepalive scheduling, dead peer timeout, graceful close"
```

---

### Task 8: Update `last_received_ms` on inbound data and run full suite

**Files:**
- Modify: `crates/harmony-tunnel/src/session.rs`

The `handle_inbound` method needs to update `last_received_ms` so keepalive timeout tracking works correctly.

- [ ] **Step 1: Write test that verifies last_received_ms updates**

Add to tests:

```rust
    #[test]
    fn inbound_data_resets_keepalive_timer() {
        let (mut initiator, mut responder) = complete_handshake();

        // Set last_received to an old time
        initiator.last_received_ms = 1000;

        // Responder sends a Reticulum packet
        let actions = responder.handle_event(TunnelEvent::SendReticulum {
            packet: b"ping".to_vec(),
        }).unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        // Initiator receives at now_ms = 50_000
        // (We need to set now_ms somehow — add a now_ms field to InboundBytes
        //  or update last_received_ms in handle_inbound. Let's check the
        //  implementation and update handle_encrypted_frame.)
    }
```

Note: The `InboundBytes` event doesn't carry a timestamp. The tick event updates time. We need to either: (a) add `now_ms` to `InboundBytes`, or (b) have the caller set time via `Tick` before `InboundBytes`. Option (a) is cleaner — the caller already knows the current time.

- [ ] **Step 2: Add `now_ms` to `InboundBytes` event**

In `event.rs`, update:

```rust
    InboundBytes { data: Vec<u8>, now_ms: u64 },
```

- [ ] **Step 3: Update `handle_inbound` to use `now_ms`**

In `session.rs`, update `handle_event`:

```rust
    TunnelEvent::InboundBytes { data, now_ms } => {
        self.last_received_ms = now_ms;
        self.handle_inbound(data)
    }
```

- [ ] **Step 4: Update all existing tests to pass `now_ms`**

Update all `TunnelEvent::InboundBytes` in tests to include `now_ms: 0` (or appropriate value).

- [ ] **Step 5: Complete the `inbound_data_resets_keepalive_timer` test**

```rust
    #[test]
    fn inbound_data_resets_keepalive_timer() {
        let (mut initiator, mut responder) = complete_handshake();

        initiator.last_received_ms = 1000;

        let actions = responder.handle_event(TunnelEvent::SendReticulum {
            packet: b"ping".to_vec(),
        }).unwrap();

        let encrypted = match &actions[0] {
            TunnelAction::OutboundBytes { data } => data.clone(),
            _ => panic!("expected OutboundBytes"),
        };

        initiator.handle_event(TunnelEvent::InboundBytes {
            data: encrypted,
            now_ms: 50_000,
        }).unwrap();

        // Dead peer timeout should NOT trigger at 50_000 + 89_999
        let actions = initiator.handle_event(TunnelEvent::Tick { now_ms: 139_999 }).unwrap();
        assert_ne!(initiator.state(), TunnelState::Closed);

        // But SHOULD trigger at 50_000 + 90_001
        let actions = initiator.handle_event(TunnelEvent::Tick { now_ms: 140_001 }).unwrap();
        assert_eq!(initiator.state(), TunnelState::Closed);
    }
```

- [ ] **Step 6: Run full test suite**

Run: `cargo test -p harmony-tunnel`
Expected: All tests pass.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy -p harmony-tunnel`
Expected: No errors (warnings about unused are OK at this stage).

- [ ] **Step 8: Run workspace tests to ensure no breakage**

Run: `cargo test --workspace`
Expected: All 365+ existing tests still pass.

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-tunnel/
git commit -m "feat(tunnel): complete harmony-tunnel crate with keepalive and timestamp tracking"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 1 | Scaffold crate, errors, workspace integration, `signing_key()` accessor | Compiles |
| 2 | PQ handshake messages (TunnelInit, TunnelAccept, key derivation) | 4 tests |
| 3 | Frame encoding/decoding with AEAD encryption | 4 tests |
| 4 | TunnelEvent and TunnelAction enums | Compiles |
| 5 | TunnelSession state machine (initiator + responder) | 3 tests |
| 6 | Data transfer + adversarial handshake tests | 4 tests |
| 7 | Keepalive and timeout tests | 3 tests |
| 8 | Timestamp tracking, full suite validation | 1 test + workspace check |

**Total:** ~19 tests covering handshake, serialization, encryption, data transfer, identity verification, MITM rejection, keepalive, timeout, and error paths.

# Message Envelope + E2EE Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a binary E2EE message envelope to harmony-zenoh that seals/opens payloads using ChaCha20-Poly1305 with ECDH key agreement.

**Architecture:** Fixed 33-byte header (version|type, sender address, nonce, sequence) + variable ciphertext. Header bytes serve as AAD to bind routing metadata to the encryption. Key derived via ECDH + HKDF-SHA256 with directional salt.

**Tech Stack:** harmony-crypto (aead, hkdf), harmony-identity (Identity, PrivateIdentity, ECDH), rand_core

**Design doc:** `docs/plans/2026-03-02-message-envelope-e2ee-design.md`

---

### Task 1: Add dependencies to harmony-zenoh

**Files:**
- Modify: `crates/harmony-zenoh/Cargo.toml`

**Step 1: Add workspace dependencies**

Add under `[dependencies]`:

```toml
harmony-crypto.workspace = true
harmony-identity.workspace = true
rand_core.workspace = true
```

**Step 2: Verify it compiles**

Run: `cargo check -p harmony-zenoh`
Expected: success (no code uses them yet, but deps resolve)

**Step 3: Commit**

```bash
git add crates/harmony-zenoh/Cargo.toml
git commit -m "Add harmony-crypto, harmony-identity, rand_core deps to harmony-zenoh"
```

---

### Task 2: Add error variants and MessageType enum

**Files:**
- Modify: `crates/harmony-zenoh/src/error.rs`
- Create: `crates/harmony-zenoh/src/envelope.rs`
- Modify: `crates/harmony-zenoh/src/lib.rs`

**Step 1: Add error variants to ZenohError**

In `crates/harmony-zenoh/src/error.rs`, add these variants:

```rust
    #[error("envelope too short: {0} bytes, minimum {1}")]
    EnvelopeTooShort(usize, usize),

    #[error("unsupported envelope version: {0}")]
    UnsupportedVersion(u8),

    #[error("invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("envelope seal failed: {0}")]
    SealFailed(String),

    #[error("envelope open failed: {0}")]
    OpenFailed(String),
```

**Step 2: Create envelope.rs with MessageType and constants**

Create `crates/harmony-zenoh/src/envelope.rs`:

```rust
//! Binary E2EE message envelope for Harmony.
//!
//! Wire format (33-byte fixed header + variable ciphertext):
//!
//! ```text
//! [1B ver|type][16B sender_addr][12B nonce][4B sequence][N bytes ciphertext+tag]
//! ```
//!
//! The 33-byte header is passed as AAD to ChaCha20-Poly1305, cryptographically
//! binding the routing metadata to the encrypted payload.

use harmony_crypto::{aead, hkdf};
use harmony_identity::{Identity, PrivateIdentity};
use rand_core::CryptoRngCore;

use crate::ZenohError;

/// Current envelope format version.
pub const VERSION: u8 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 1 + 16 + 12 + 4; // 33

/// Minimum envelope size: header + Poly1305 tag (16 bytes for empty plaintext).
pub const MIN_ENVELOPE_SIZE: usize = HEADER_SIZE + aead::TAG_LENGTH;

/// HKDF info string for envelope key derivation.
const HKDF_INFO: &[u8] = b"harmony-envelope-v1";

/// Message type carried in the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Data message (publish).
    Put = 0,
    /// Tombstone (delete).
    Del = 1,
}

impl MessageType {
    /// Decode from the low 4 bits of the version|type byte.
    pub fn from_u8(val: u8) -> Result<Self, ZenohError> {
        match val {
            0 => Ok(Self::Put),
            1 => Ok(Self::Del),
            other => Err(ZenohError::InvalidMessageType(other)),
        }
    }
}
```

**Step 3: Register the module in lib.rs**

In `crates/harmony-zenoh/src/lib.rs`, add:

```rust
pub mod envelope;
```

And add to the re-exports:

```rust
pub use envelope::{HarmonyEnvelope, MessageType};
```

(The `HarmonyEnvelope` import will fail until Task 3, so for now just add `pub mod envelope;` and the `MessageType` re-export.)

**Step 4: Verify it compiles**

Run: `cargo check -p harmony-zenoh`
Expected: success

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/error.rs crates/harmony-zenoh/src/envelope.rs crates/harmony-zenoh/src/lib.rs
git commit -m "Add envelope module with MessageType, constants, and error variants"
```

---

### Task 3: Implement HarmonyEnvelope struct with seal() and open()

**Files:**
- Modify: `crates/harmony-zenoh/src/envelope.rs`

**Step 1: Write the failing test for seal/open roundtrip**

Add at the bottom of `envelope.rs`:

```rust
/// A decoded Harmony message envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyEnvelope {
    /// Envelope format version.
    pub version: u8,
    /// Message type (Put or Del).
    pub msg_type: MessageType,
    /// Sender's 16-byte address hash.
    pub sender_address: [u8; 16],
    /// Monotonic sequence number.
    pub sequence: u32,
    /// Decrypted plaintext payload.
    pub plaintext: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn seal_open_roundtrip() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let plaintext = b"hello harmony";
        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            42,
            plaintext,
        )
        .unwrap();

        let envelope = HarmonyEnvelope::open(
            &recipient,
            sender.public_identity(),
            &sealed,
        )
        .unwrap();

        assert_eq!(envelope.version, VERSION);
        assert_eq!(envelope.msg_type, MessageType::Put);
        assert_eq!(envelope.sender_address, sender.public_identity().address_hash);
        assert_eq!(envelope.sequence, 42);
        assert_eq!(envelope.plaintext, plaintext);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-zenoh seal_open_roundtrip`
Expected: FAIL — `seal` and `open` not defined yet

**Step 3: Implement seal() and open()**

Add to `envelope.rs` after the `HarmonyEnvelope` struct:

```rust
/// Derive a shared symmetric key from ECDH + HKDF.
///
/// Salt = sender_addr || recipient_addr for directionality.
fn derive_shared_key(
    sender: &PrivateIdentity,
    recipient: &Identity,
) -> Result<[u8; aead::KEY_LENGTH], ZenohError> {
    let shared_secret = sender.ecdh(&recipient.encryption_key);

    // Salt: sender || recipient address hashes (32 bytes)
    let mut salt = [0u8; 32];
    salt[..16].copy_from_slice(&sender.public_identity().address_hash);
    salt[16..].copy_from_slice(&recipient.address_hash);

    let key_bytes = hkdf::derive_key(
        shared_secret.as_bytes(),
        Some(&salt),
        HKDF_INFO,
        aead::KEY_LENGTH,
    )
    .map_err(|e| ZenohError::SealFailed(e.to_string()))?;

    let mut key = [0u8; aead::KEY_LENGTH];
    key.copy_from_slice(&key_bytes);
    Ok(key)
}

impl HarmonyEnvelope {
    /// Encrypt plaintext and produce a sealed binary envelope.
    ///
    /// Returns the wire bytes: `[33B header][ciphertext + tag]`.
    pub fn seal(
        rng: &mut impl CryptoRngCore,
        msg_type: MessageType,
        sender: &PrivateIdentity,
        recipient: &Identity,
        sequence: u32,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ZenohError> {
        let nonce = aead::generate_nonce(rng);
        let key = derive_shared_key(sender, recipient)?;

        // Build header
        let mut header = [0u8; HEADER_SIZE];
        header[0] = (VERSION << 4) | (msg_type as u8);
        header[1..17].copy_from_slice(&sender.public_identity().address_hash);
        header[17..29].copy_from_slice(&nonce);
        header[29..33].copy_from_slice(&sequence.to_be_bytes());

        // Encrypt with header as AAD
        let ciphertext = aead::encrypt(&key, &nonce, plaintext, &header)
            .map_err(|e| ZenohError::SealFailed(e.to_string()))?;

        // Concatenate header + ciphertext
        let mut result = Vec::with_capacity(HEADER_SIZE + ciphertext.len());
        result.extend_from_slice(&header);
        result.extend_from_slice(&ciphertext);
        Ok(result)
    }

    /// Parse and decrypt a sealed binary envelope.
    ///
    /// The caller must provide the sender's public identity (looked up from
    /// `sender_address` in the header). The session layer (harmony-0p6.3)
    /// will handle address → identity resolution.
    pub fn open(
        recipient: &PrivateIdentity,
        sender: &Identity,
        data: &[u8],
    ) -> Result<HarmonyEnvelope, ZenohError> {
        if data.len() < MIN_ENVELOPE_SIZE {
            return Err(ZenohError::EnvelopeTooShort(data.len(), MIN_ENVELOPE_SIZE));
        }

        let header = &data[..HEADER_SIZE];
        let ciphertext = &data[HEADER_SIZE..];

        // Parse header fields
        let version = header[0] >> 4;
        if version != VERSION {
            return Err(ZenohError::UnsupportedVersion(version));
        }
        let msg_type = MessageType::from_u8(header[0] & 0x0F)?;

        let mut sender_address = [0u8; 16];
        sender_address.copy_from_slice(&header[1..17]);

        let mut nonce = [0u8; aead::NONCE_LENGTH];
        nonce.copy_from_slice(&header[17..29]);

        let sequence = u32::from_be_bytes(header[29..33].try_into().unwrap());

        // Derive shared key (recipient side)
        let key = derive_shared_key_for_open(recipient, sender)?;

        // Decrypt with header as AAD
        let plaintext = aead::decrypt(&key, &nonce, ciphertext, header)
            .map_err(|e| ZenohError::OpenFailed(e.to_string()))?;

        Ok(HarmonyEnvelope {
            version,
            msg_type,
            sender_address,
            sequence,
            plaintext,
        })
    }
}

/// Derive the shared key from the recipient's perspective.
///
/// The salt must match what seal() used: sender_addr || recipient_addr.
fn derive_shared_key_for_open(
    recipient: &PrivateIdentity,
    sender: &Identity,
) -> Result<[u8; aead::KEY_LENGTH], ZenohError> {
    let shared_secret = recipient.ecdh(&sender.encryption_key);

    // Salt: sender || recipient (same order as seal)
    let mut salt = [0u8; 32];
    salt[..16].copy_from_slice(&sender.address_hash);
    salt[16..].copy_from_slice(&recipient.public_identity().address_hash);

    let key_bytes = hkdf::derive_key(
        shared_secret.as_bytes(),
        Some(&salt),
        HKDF_INFO,
        aead::KEY_LENGTH,
    )
    .map_err(|e| ZenohError::OpenFailed(e.to_string()))?;

    let mut key = [0u8; aead::KEY_LENGTH];
    key.copy_from_slice(&key_bytes);
    Ok(key)
}
```

**Step 4: Add re-export to lib.rs**

Update `crates/harmony-zenoh/src/lib.rs` to include the `HarmonyEnvelope` re-export:

```rust
pub use envelope::{HarmonyEnvelope, MessageType};
```

**Step 5: Add dev-dependency for rand**

In `crates/harmony-zenoh/Cargo.toml`, add:

```toml
[dev-dependencies]
rand = { workspace = true }
```

**Step 6: Run test to verify it passes**

Run: `cargo test -p harmony-zenoh seal_open_roundtrip`
Expected: PASS

**Step 7: Commit**

```bash
git add crates/harmony-zenoh/
git commit -m "Implement HarmonyEnvelope seal/open with ECDH + HKDF + ChaCha20-Poly1305"
```

---

### Task 4: Add comprehensive test coverage

**Files:**
- Modify: `crates/harmony-zenoh/src/envelope.rs` (tests module)

**Step 1: Add all remaining tests**

Add these tests to the existing `mod tests` block in `envelope.rs`:

```rust
    #[test]
    fn del_type_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Del, &sender, recipient.public_identity(), 0, b"",
        ).unwrap();

        let envelope = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert_eq!(envelope.msg_type, MessageType::Del);
        assert!(envelope.plaintext.is_empty());
    }

    #[test]
    fn empty_payload_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 0, b"",
        ).unwrap();

        assert_eq!(sealed.len(), MIN_ENVELOPE_SIZE);

        let envelope = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert!(envelope.plaintext.is_empty());
    }

    #[test]
    fn large_payload_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let plaintext = vec![0xAB; 8192];
        let sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 1, &plaintext,
        ).unwrap();

        let envelope = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert_eq!(envelope.plaintext, plaintext);
    }

    #[test]
    fn tampered_header_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 1, b"secret",
        ).unwrap();

        // Flip a byte in the sequence field (byte 30)
        sealed[30] ^= 0xFF;

        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed);
        assert!(result.is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 1, b"secret",
        ).unwrap();

        // Flip a byte in the ciphertext region
        let last = sealed.len() - 1;
        sealed[last] ^= 0xFF;

        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed);
        assert!(result.is_err());
    }

    #[test]
    fn wrong_recipient_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);
        let wrong_recipient = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 1, b"secret",
        ).unwrap();

        let result = HarmonyEnvelope::open(&wrong_recipient, sender.public_identity(), &sealed);
        assert!(result.is_err());
    }

    #[test]
    fn sequence_preserved() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        for seq in [0, 1, 255, 65535, u32::MAX] {
            let sealed = HarmonyEnvelope::seal(
                &mut rng, MessageType::Put, &sender, recipient.public_identity(), seq, b"x",
            ).unwrap();
            let envelope = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
            assert_eq!(envelope.sequence, seq);
        }
    }

    #[test]
    fn version_type_byte_encoding() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed_put = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 0, b"",
        ).unwrap();
        // Version 1 in high nibble, Put (0) in low nibble
        assert_eq!(sealed_put[0], 0x10);

        let sealed_del = HarmonyEnvelope::seal(
            &mut rng, MessageType::Del, &sender, recipient.public_identity(), 0, b"",
        ).unwrap();
        // Version 1 in high nibble, Del (1) in low nibble
        assert_eq!(sealed_del[0], 0x11);
    }

    #[test]
    fn envelope_too_short_rejected() {
        let mut rng = OsRng;
        let recipient = PrivateIdentity::generate(&mut rng);
        let sender = PrivateIdentity::generate(&mut rng);

        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &[0u8; 32]);
        assert!(result.is_err());
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &sender, recipient.public_identity(), 0, b"x",
        ).unwrap();

        // Set version to 2 (unsupported)
        sealed[0] = (2 << 4) | (sealed[0] & 0x0F);

        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed);
        assert!(matches!(result, Err(ZenohError::UnsupportedVersion(2))));
    }

    #[test]
    fn directionality_produces_different_ciphertexts() {
        let mut rng = OsRng;
        let alice = PrivateIdentity::generate(&mut rng);
        let bob = PrivateIdentity::generate(&mut rng);

        // Alice to Bob
        let a_to_b = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &alice, bob.public_identity(), 1, b"same",
        ).unwrap();

        // Bob to Alice
        let b_to_a = HarmonyEnvelope::seal(
            &mut rng, MessageType::Put, &bob, alice.public_identity(), 1, b"same",
        ).unwrap();

        // Different ciphertexts (different keys due to directional salt + different nonces)
        assert_ne!(a_to_b[HEADER_SIZE..], b_to_a[HEADER_SIZE..]);

        // Both decrypt correctly
        let env1 = HarmonyEnvelope::open(&bob, alice.public_identity(), &a_to_b).unwrap();
        let env2 = HarmonyEnvelope::open(&alice, bob.public_identity(), &b_to_a).unwrap();
        assert_eq!(env1.plaintext, b"same");
        assert_eq!(env2.plaintext, b"same");
    }
```

**Step 2: Run all tests**

Run: `cargo test -p harmony-zenoh`
Expected: all tests pass (roundtrip + 11 new tests)

**Step 3: Run clippy**

Run: `cargo clippy -p harmony-zenoh`
Expected: clean

**Step 4: Commit**

```bash
git add crates/harmony-zenoh/src/envelope.rs
git commit -m "Add comprehensive envelope test coverage"
```

---

### Task 5: Run full workspace verification

**Step 1: Run full test suite**

Run: `cargo test --workspace`
Expected: all tests pass (367 existing + ~12 new envelope tests)

**Step 2: Run clippy**

Run: `cargo clippy --workspace`
Expected: clean

**Step 3: Commit any fixups, then deliver**

Use `/deliver` to ship the work via PR.

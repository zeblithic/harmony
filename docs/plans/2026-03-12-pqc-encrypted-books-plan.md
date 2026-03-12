# Post-Quantum Cryptography & Encrypted Books Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add ML-KEM-768 and ML-DSA-65 post-quantum cryptography to Harmony, new PQC identity types, and encrypted books using the `11` sentinel for metadata pages.

**Architecture:** Three-phase implementation across `harmony-crypto` (PQC primitives), `harmony-identity` (PQC identity types + UCAN updates), and `harmony-athenaeum` (encrypted book type with metadata pages). Each phase builds on the previous. `harmony-athenaeum` stays independent of `harmony-crypto` — metadata uses raw bytes, callers serialize PQC types.

**Tech Stack:** Rust, RustCrypto (`ml-kem` 0.3, `ml-dsa` 0.1, `chacha20poly1305`, `sha2`, `hkdf`), `zeroize`, `rand_core`. MSRV bumped from 1.75 to 1.85.

**Design doc:** `docs/plans/2026-03-12-pqc-encrypted-books-design.md`

---

## Phase 1: PQC Crypto Primitives (`harmony-crypto`)

### Task 1: Add PQC Dependencies and Bump MSRV

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/harmony-crypto/Cargo.toml`

**Step 1: Add dependencies to workspace Cargo.toml**

Add to `[workspace.dependencies]`:

```toml
ml-kem = { version = "0.3.0-rc.0", default-features = false, features = ["zeroize"] }
ml-dsa = { version = "0.1.0-rc.7", default-features = false, features = ["rand_core", "zeroize"] }
hybrid-array = { version = "0.2", default-features = false }
```

**Step 2: Add dependencies to harmony-crypto Cargo.toml**

Add to `[dependencies]`:

```toml
ml-kem = { workspace = true }
ml-dsa = { workspace = true }
hybrid-array = { workspace = true }
```

Add `"ml-kem/std"`, `"ml-dsa/std"`, `"hybrid-array/std"` to the `std` feature list.

**Step 3: Bump MSRV**

Update `rust-version` in workspace `Cargo.toml` from `"1.75"` to `"1.85"` (both PQC crates require it). Update any MSRV references in CLAUDE.md.

**Step 4: Verify it compiles**

Run: `cargo check -p harmony-crypto`
Expected: compiles with no errors.

**Step 5: Commit**

```bash
git add Cargo.toml crates/harmony-crypto/Cargo.toml
git commit -m "build: add ml-kem and ml-dsa dependencies, bump MSRV to 1.85"
```

---

### Task 2: ML-KEM-768 Wrapper Module

**Files:**
- Create: `crates/harmony-crypto/src/ml_kem.rs`
- Modify: `crates/harmony-crypto/src/lib.rs`
- Modify: `crates/harmony-crypto/src/error.rs`

**Reference:** The `ml-kem` crate API uses:
- `MlKem768` as the parameter set type
- `KemCore::generate(&mut rng)` → `(DecapsulationKey, EncapsulationKey)`
- `EncapsulationKey::encapsulate(&mut rng)` → `Result<(Ciphertext, SharedKey), ()>`
- `DecapsulationKey::decapsulate(&ct)` → `Result<SharedKey, ()>`
- `EncodedSizeUser::{as_bytes(), from_bytes()}` for serialization
- Returns `hybrid_array::Array<u8, N>`, use `.as_slice()` to get `&[u8]`

**Step 1: Add error variants to `error.rs`**

```rust
#[error("ML-KEM encapsulation failed")]
MlKemEncapsulationFailed,

#[error("ML-KEM decapsulation failed")]
MlKemDecapsulationFailed,

#[error("ML-DSA signing failed")]
MlDsaSignFailed,

#[error("ML-DSA signature verification failed")]
MlDsaVerifyFailed,
```

**Step 2: Write the failing tests in `ml_kem.rs`**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn keygen_produces_valid_keys() {
        let (pk, sk) = generate(&mut OsRng);
        assert_eq!(pk.as_bytes().len(), PK_LENGTH);
        assert_eq!(sk.as_bytes().len(), SK_LENGTH);
    }

    #[test]
    fn encapsulate_decapsulate_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let (ct, ss_sender) = encapsulate(&mut OsRng, &pk).unwrap();
        let ss_receiver = decapsulate(&sk, &ct).unwrap();
        assert_eq!(ss_sender.as_bytes(), ss_receiver.as_bytes());
    }

    #[test]
    fn wrong_secret_key_fails_decapsulation() {
        let (pk, _sk1) = generate(&mut OsRng);
        let (_pk2, sk2) = generate(&mut OsRng);
        let (ct, ss_sender) = encapsulate(&mut OsRng, &pk).unwrap();
        // ML-KEM decapsulation with wrong key produces a different shared secret
        // (implicit rejection), it does not return an error
        let ss_wrong = decapsulate(&sk2, &ct).unwrap();
        assert_ne!(ss_sender.as_bytes(), ss_wrong.as_bytes());
    }

    #[test]
    fn public_key_serialization_roundtrip() {
        let (pk, _sk) = generate(&mut OsRng);
        let bytes = pk.as_bytes();
        let pk2 = MlKemPublicKey::from_bytes(bytes).unwrap();
        assert_eq!(pk.as_bytes(), pk2.as_bytes());
    }

    #[test]
    fn secret_key_serialization_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let sk_bytes = sk.as_bytes();
        let sk2 = MlKemSecretKey::from_bytes(sk_bytes).unwrap();
        // Verify round-trip by encapsulating with original pk, decapsulating with restored sk
        let (ct, ss1) = encapsulate(&mut OsRng, &pk).unwrap();
        let ss2 = decapsulate(&sk2, &ct).unwrap();
        assert_eq!(ss1.as_bytes(), ss2.as_bytes());
    }

    #[test]
    fn shared_secret_is_32_bytes() {
        let (pk, sk) = generate(&mut OsRng);
        let (ct, ss) = encapsulate(&mut OsRng, &pk).unwrap();
        assert_eq!(ss.as_bytes().len(), 32);
        let ss2 = decapsulate(&sk, &ct).unwrap();
        assert_eq!(ss2.as_bytes().len(), 32);
    }
}
```

**Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-crypto ml_kem`
Expected: compilation failure (module doesn't exist yet)

**Step 4: Implement the module**

Create `crates/harmony-crypto/src/ml_kem.rs`:

```rust
//! ML-KEM-768 key encapsulation (FIPS 203).
//!
//! Wraps the RustCrypto `ml-kem` crate with Harmony conventions:
//! Zeroize-on-drop for secret material, fixed-size byte accessors,
//! and CryptoError integration.

use crate::CryptoError;
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use ml_kem::{
    kem::{Decapsulate, Encapsulate},
    EncodedSizeUser, KemCore, MlKem768,
};

/// ML-KEM-768 public key length (1,184 bytes).
pub const PK_LENGTH: usize = 1184;

/// ML-KEM-768 secret key length (2,400 bytes).
pub const SK_LENGTH: usize = 2400;

/// ML-KEM-768 ciphertext length (1,088 bytes).
pub const CT_LENGTH: usize = 1088;

/// ML-KEM-768 shared secret length (32 bytes).
pub const SS_LENGTH: usize = 32;

/// ML-KEM-768 encapsulation (public) key.
pub struct MlKemPublicKey {
    inner: ml_kem::EncapsulationKey<ml_kem::MlKem768Params>,
}

impl MlKemPublicKey {
    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != PK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = hybrid_array::Array::try_from(bytes)
            .map_err(|_| CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            })?;
        Ok(Self {
            inner: ml_kem::EncapsulationKey::from_bytes(&arr),
        })
    }

    /// Serialize to bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // EncodedSizeUser::as_bytes returns an Array; we return slice
        // We need to store the encoded form — use a method that returns owned
        // Actually, EncapsulationKey implements as_bytes() which returns Array
        // We'll store and return
        self.inner.as_bytes().as_slice()
    }

    /// Serialize to owned byte vector.
    pub fn to_bytes(&self) -> alloc::vec::Vec<u8> {
        self.inner.as_bytes().to_vec()
    }
}

impl Clone for MlKemPublicKey {
    fn clone(&self) -> Self {
        Self {
            inner: ml_kem::EncapsulationKey::from_bytes(&self.inner.as_bytes()),
        }
    }
}

/// ML-KEM-768 decapsulation (secret) key. Zeroizes on drop.
pub struct MlKemSecretKey {
    inner: ml_kem::DecapsulationKey<ml_kem::MlKem768Params>,
}

impl MlKemSecretKey {
    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = hybrid_array::Array::try_from(bytes)
            .map_err(|_| CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            })?;
        Ok(Self {
            inner: ml_kem::DecapsulationKey::from_bytes(&arr),
        })
    }

    /// Serialize to bytes.
    pub fn as_bytes(&self) -> alloc::vec::Vec<u8> {
        self.inner.as_bytes().to_vec()
    }
}

/// ML-KEM-768 ciphertext (1,088 bytes).
pub struct MlKemCiphertext {
    bytes: [u8; CT_LENGTH],
}

impl MlKemCiphertext {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != CT_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: CT_LENGTH,
                got: bytes.len(),
            });
        }
        let mut ct = [0u8; CT_LENGTH];
        ct.copy_from_slice(bytes);
        Ok(Self { bytes: ct })
    }

    pub fn as_bytes(&self) -> &[u8; CT_LENGTH] {
        &self.bytes
    }
}

/// ML-KEM-768 shared secret (32 bytes). Zeroizes on drop.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct MlKemSharedSecret {
    bytes: [u8; SS_LENGTH],
}

impl MlKemSharedSecret {
    pub fn as_bytes(&self) -> &[u8; SS_LENGTH] {
        &self.bytes
    }
}

/// Generate an ML-KEM-768 keypair.
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlKemPublicKey, MlKemSecretKey) {
    let (dk, ek) = MlKem768::generate(rng);
    (MlKemPublicKey { inner: ek }, MlKemSecretKey { inner: dk })
}

/// Encapsulate: generate a shared secret encrypted to the given public key.
pub fn encapsulate(
    rng: &mut impl CryptoRngCore,
    pk: &MlKemPublicKey,
) -> Result<(MlKemCiphertext, MlKemSharedSecret), CryptoError> {
    let (ct, ss) = pk
        .inner
        .encapsulate(rng)
        .map_err(|_| CryptoError::MlKemEncapsulationFailed)?;

    let mut ct_bytes = [0u8; CT_LENGTH];
    ct_bytes.copy_from_slice(ct.as_slice());

    let mut ss_bytes = [0u8; SS_LENGTH];
    ss_bytes.copy_from_slice(ss.as_slice());

    Ok((
        MlKemCiphertext { bytes: ct_bytes },
        MlKemSharedSecret { bytes: ss_bytes },
    ))
}

/// Decapsulate: recover the shared secret from a ciphertext using the secret key.
pub fn decapsulate(
    sk: &MlKemSecretKey,
    ct: &MlKemCiphertext,
) -> Result<MlKemSharedSecret, CryptoError> {
    let ct_arr = hybrid_array::Array::try_from(ct.bytes.as_slice())
        .map_err(|_| CryptoError::MlKemDecapsulationFailed)?;
    let ss = sk
        .inner
        .decapsulate(&ct_arr)
        .map_err(|_| CryptoError::MlKemDecapsulationFailed)?;

    let mut ss_bytes = [0u8; SS_LENGTH];
    ss_bytes.copy_from_slice(ss.as_slice());

    Ok(MlKemSharedSecret { bytes: ss_bytes })
}
```

**Step 5: Register the module in `lib.rs`**

Add `pub mod ml_kem;` to `crates/harmony-crypto/src/lib.rs`.

**Step 6: Run tests to verify they pass**

Run: `cargo test -p harmony-crypto ml_kem`
Expected: all 6 tests pass.

**Step 7: Commit**

```bash
git add crates/harmony-crypto/src/ml_kem.rs crates/harmony-crypto/src/lib.rs crates/harmony-crypto/src/error.rs
git commit -m "feat(crypto): add ML-KEM-768 key encapsulation module"
```

**Implementation notes for the subagent:**
- The `as_bytes()` method on `MlKemPublicKey` may need adjustment — `EncodedSizeUser::as_bytes()` returns an owned `Array`, not a reference. You may need to store the serialized form or return `Vec<u8>` instead of `&[u8]`. Adapt the API to what actually compiles.
- ML-KEM decapsulation with the wrong key uses "implicit rejection" — it returns a valid-looking but wrong shared secret rather than an error. The test for wrong-key should assert `!=` on the shared secrets, not expect an `Err`.
- `hybrid_array::Array::try_from(&[u8])` may need different conversion. Check the actual `hybrid-array` API. You may need `GenericArray::clone_from_slice()` or similar.
- The exact import paths may differ between `ml-kem` versions. Check what's actually available and adjust.

---

### Task 3: ML-DSA-65 Wrapper Module

**Files:**
- Create: `crates/harmony-crypto/src/ml_dsa.rs`
- Modify: `crates/harmony-crypto/src/lib.rs`

**Reference:** The `ml-dsa` crate API uses:
- `MlDsa65` as the parameter set type
- `MlDsa65::key_gen(&mut rng)` → `KeyPair<MlDsa65>`
- `keypair.signing_key().sign(msg)` → `Signature<MlDsa65>` (via `Signer` trait)
- `keypair.verifying_key().verify(msg, &sig)` → `Result<(), Error>` (via `Verifier` trait)
- `encode()`/`decode()` for serialization
- Signing key: 4,032 bytes, Verifying key: 1,952 bytes, Signature: 3,309 bytes

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn keygen_produces_valid_keys() {
        let (pk, sk) = generate(&mut OsRng);
        assert_eq!(pk.as_bytes().len(), PK_LENGTH);
        assert_eq!(sk.as_bytes().len(), SK_LENGTH);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let msg = b"hello post-quantum world";
        let sig = sign(&sk, msg).unwrap();
        assert_eq!(sig.as_bytes().len(), SIG_LENGTH);
        assert!(verify(&pk, msg, &sig).is_ok());
    }

    #[test]
    fn verify_wrong_message_fails() {
        let (pk, sk) = generate(&mut OsRng);
        let sig = sign(&sk, b"correct message").unwrap();
        assert!(verify(&pk, b"wrong message", &sig).is_err());
    }

    #[test]
    fn verify_wrong_key_fails() {
        let (_pk1, sk1) = generate(&mut OsRng);
        let (pk2, _sk2) = generate(&mut OsRng);
        let sig = sign(&sk1, b"message").unwrap();
        assert!(verify(&pk2, b"message", &sig).is_err());
    }

    #[test]
    fn public_key_serialization_roundtrip() {
        let (pk, _sk) = generate(&mut OsRng);
        let bytes = pk.as_bytes();
        let pk2 = MlDsaPublicKey::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), pk2.as_bytes());
    }

    #[test]
    fn secret_key_serialization_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let sk_bytes = sk.as_bytes();
        let sk2 = MlDsaSecretKey::from_bytes(&sk_bytes).unwrap();
        let sig = sign(&sk2, b"test").unwrap();
        assert!(verify(&pk, b"test", &sig).is_ok());
    }

    #[test]
    fn signature_serialization_roundtrip() {
        let (_pk, sk) = generate(&mut OsRng);
        let sig = sign(&sk, b"data").unwrap();
        let sig_bytes = sig.as_bytes();
        let sig2 = MlDsaSignature::from_bytes(&sig_bytes).unwrap();
        assert_eq!(sig.as_bytes(), sig2.as_bytes());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-crypto ml_dsa`
Expected: compilation failure.

**Step 3: Implement the module**

Create `crates/harmony-crypto/src/ml_dsa.rs`:

```rust
//! ML-DSA-65 digital signatures (FIPS 204).
//!
//! Wraps the RustCrypto `ml-dsa` crate with Harmony conventions:
//! Zeroize-on-drop for secret material, fixed-size byte accessors,
//! and CryptoError integration.

use alloc::vec::Vec;
use crate::CryptoError;
use rand_core::CryptoRngCore;

use ml_dsa::{
    KeyGen, MlDsa65,
    signature::{Keypair, Signer, Verifier},
};

/// ML-DSA-65 verifying (public) key length (1,952 bytes).
pub const PK_LENGTH: usize = 1952;

/// ML-DSA-65 signing (secret) key length (4,032 bytes).
pub const SK_LENGTH: usize = 4032;

/// ML-DSA-65 signature length (3,309 bytes).
pub const SIG_LENGTH: usize = 3309;

/// ML-DSA-65 verifying (public) key.
pub struct MlDsaPublicKey {
    inner: ml_dsa::VerifyingKey<MlDsa65>,
}

impl MlDsaPublicKey {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != PK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = hybrid_array::Array::try_from(bytes)
            .map_err(|_| CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            })?;
        let vk = ml_dsa::VerifyingKey::<MlDsa65>::decode(&arr)
            .ok_or(CryptoError::MlDsaVerifyFailed)?;
        Ok(Self { inner: vk })
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.encode().to_vec()
    }
}

impl Clone for MlDsaPublicKey {
    fn clone(&self) -> Self {
        let encoded = self.inner.encode();
        Self {
            inner: ml_dsa::VerifyingKey::decode(&encoded).unwrap(),
        }
    }
}

/// ML-DSA-65 signing (secret) key. Zeroizes on drop.
pub struct MlDsaSecretKey {
    inner: ml_dsa::SigningKey<MlDsa65>,
}

impl MlDsaSecretKey {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = hybrid_array::Array::try_from(bytes)
            .map_err(|_| CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            })?;
        let sk = ml_dsa::SigningKey::<MlDsa65>::decode(&arr)
            .ok_or(CryptoError::MlDsaSignFailed)?;
        Ok(Self { inner: sk })
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.encode().to_vec()
    }
}

/// ML-DSA-65 signature (3,309 bytes).
pub struct MlDsaSignature {
    inner: ml_dsa::Signature<MlDsa65>,
}

impl MlDsaSignature {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SIG_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SIG_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = hybrid_array::Array::try_from(bytes)
            .map_err(|_| CryptoError::InvalidKeyLength {
                expected: SIG_LENGTH,
                got: bytes.len(),
            })?;
        let sig = ml_dsa::Signature::<MlDsa65>::decode(&arr)
            .ok_or(CryptoError::MlDsaVerifyFailed)?;
        Ok(Self { inner: sig })
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.encode().to_vec()
    }
}

/// Generate an ML-DSA-65 keypair.
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlDsaPublicKey, MlDsaSecretKey) {
    let kp = MlDsa65::key_gen(rng);
    let vk = kp.verifying_key().clone();
    // Extract signing key — need to encode/decode since KeyPair doesn't give ownership
    let sk_encoded = kp.signing_key().encode();
    let sk = ml_dsa::SigningKey::decode(&sk_encoded).unwrap();
    (
        MlDsaPublicKey { inner: vk },
        MlDsaSecretKey { inner: sk },
    )
}

/// Sign a message with ML-DSA-65.
pub fn sign(sk: &MlDsaSecretKey, message: &[u8]) -> Result<MlDsaSignature, CryptoError> {
    let sig = sk.inner.try_sign(message).map_err(|_| CryptoError::MlDsaSignFailed)?;
    Ok(MlDsaSignature { inner: sig })
}

/// Verify a signature with ML-DSA-65.
pub fn verify(
    pk: &MlDsaPublicKey,
    message: &[u8],
    sig: &MlDsaSignature,
) -> Result<(), CryptoError> {
    pk.inner
        .verify(message, &sig.inner)
        .map_err(|_| CryptoError::MlDsaVerifyFailed)
}
```

**Step 4: Register the module in `lib.rs`**

Add `pub mod ml_dsa;` to `crates/harmony-crypto/src/lib.rs`.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-crypto ml_dsa`
Expected: all 7 tests pass.

**Step 6: Commit**

```bash
git add crates/harmony-crypto/src/ml_dsa.rs crates/harmony-crypto/src/lib.rs
git commit -m "feat(crypto): add ML-DSA-65 digital signature module"
```

**Implementation notes for the subagent:**
- The `ml-dsa` `Signer` trait is re-exported from the `signature` crate. Import path: `ml_dsa::signature::Signer`.
- `KeyPair` gives references to signing/verifying keys. To extract owned keys, encode and re-decode.
- `SigningKey::decode()` and `VerifyingKey::decode()` return `Option<Self>`, not `Result`.
- `Signature::decode()` also returns `Option<Self>`.
- The `try_sign` method is from the `Signer` trait; `sign` is the panicking version.
- If `hybrid_array::Array::try_from` doesn't work for your version, try `GenericArray::clone_from_slice()` or construct from a slice differently.

---

### Task 4: Hybrid KEM Module

**Files:**
- Create: `crates/harmony-crypto/src/hybrid_kem.rs`
- Modify: `crates/harmony-crypto/src/lib.rs`

**Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn hybrid_kem_roundtrip() {
        let (ml_pk, ml_sk) = crate::ml_kem::generate(&mut OsRng);
        let x_sk = x25519_dalek::StaticSecret::random_from_rng(&mut OsRng);
        let x_pk = x25519_dalek::PublicKey::from(&x_sk);

        let (ct, x_eph_pk, shared_key_sender) =
            hybrid_encapsulate(&mut OsRng, &ml_pk, &x_pk, b"test-context").unwrap();
        let shared_key_receiver =
            hybrid_decapsulate(&ml_sk, &ct, &x_sk, &x_eph_pk, b"test-context").unwrap();

        assert_eq!(shared_key_sender, shared_key_receiver);
        assert_eq!(shared_key_sender.len(), 32);
    }

    #[test]
    fn different_contexts_produce_different_keys() {
        let (ml_pk, ml_sk) = crate::ml_kem::generate(&mut OsRng);
        let x_sk = x25519_dalek::StaticSecret::random_from_rng(&mut OsRng);
        let x_pk = x25519_dalek::PublicKey::from(&x_sk);

        let (ct, x_eph_pk, key1) =
            hybrid_encapsulate(&mut OsRng, &ml_pk, &x_pk, b"context-a").unwrap();
        // Decapsulate with different context should produce different key
        let key2 =
            hybrid_decapsulate(&ml_sk, &ct, &x_sk, &x_eph_pk, b"context-b").unwrap();

        assert_ne!(key1, key2);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-crypto hybrid_kem`
Expected: compilation failure.

**Step 3: Implement the module**

Create `crates/harmony-crypto/src/hybrid_kem.rs`:

```rust
//! Hybrid KEM: X25519 + ML-KEM-768.
//!
//! Combines a classical X25519 key exchange with ML-KEM-768 key encapsulation.
//! The two shared secrets are combined via HKDF-SHA256 to produce a single
//! 32-byte symmetric key. This provides defense-in-depth: the result is secure
//! if either algorithm holds.

use alloc::vec::Vec;
use crate::{ml_kem, CryptoError};
use rand_core::CryptoRngCore;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret as X25519Secret};

/// Hybrid encapsulate: generate a shared key using both X25519 and ML-KEM-768.
///
/// Returns (ML-KEM ciphertext, ephemeral X25519 public key, 32-byte shared key).
/// The caller must transmit both the ciphertext and ephemeral public key to the recipient.
pub fn hybrid_encapsulate(
    rng: &mut impl CryptoRngCore,
    ml_pk: &ml_kem::MlKemPublicKey,
    x_pk: &X25519PublicKey,
    context: &[u8],
) -> Result<(ml_kem::MlKemCiphertext, X25519PublicKey, [u8; 32]), CryptoError> {
    // ML-KEM encapsulation
    let (ct, ml_ss) = ml_kem::encapsulate(rng, ml_pk)?;

    // X25519 ephemeral key exchange
    let x_eph_sk = X25519Secret::random_from_rng(rng);
    let x_eph_pk = X25519PublicKey::from(&x_eph_sk);
    let x_ss = x_eph_sk.diffie_hellman(x_pk);

    // Combine via HKDF
    let mut ikm = Vec::with_capacity(64);
    ikm.extend_from_slice(x_ss.as_bytes());
    ikm.extend_from_slice(ml_ss.as_bytes());

    let combined = crate::hkdf::derive_key(
        &ikm,
        None,
        context,
        32,
    )?;

    let mut key = [0u8; 32];
    key.copy_from_slice(&combined);

    Ok((ct, x_eph_pk, key))
}

/// Hybrid decapsulate: recover the shared key using both X25519 and ML-KEM-768.
pub fn hybrid_decapsulate(
    ml_sk: &ml_kem::MlKemSecretKey,
    ct: &ml_kem::MlKemCiphertext,
    x_sk: &X25519Secret,
    x_eph_pk: &X25519PublicKey,
    context: &[u8],
) -> Result<[u8; 32], CryptoError> {
    // ML-KEM decapsulation
    let ml_ss = ml_kem::decapsulate(ml_sk, ct)?;

    // X25519 key exchange
    let x_ss = x_sk.diffie_hellman(x_eph_pk);

    // Combine via HKDF (same as encapsulate)
    let mut ikm = Vec::with_capacity(64);
    ikm.extend_from_slice(x_ss.as_bytes());
    ikm.extend_from_slice(ml_ss.as_bytes());

    let combined = crate::hkdf::derive_key(
        &ikm,
        None,
        context,
        32,
    )?;

    let mut key = [0u8; 32];
    key.copy_from_slice(&combined);

    Ok(key)
}
```

**Step 4: Register module and add x25519-dalek dependency**

Add `pub mod hybrid_kem;` to `lib.rs`. Add `x25519-dalek` to harmony-crypto's Cargo.toml dependencies (it's already in the workspace).

**Step 5: Run tests**

Run: `cargo test -p harmony-crypto hybrid_kem`
Expected: all 2 tests pass.

**Step 6: Run full test suite**

Run: `cargo test --workspace`
Expected: all existing tests still pass alongside new ones.

**Step 7: Commit**

```bash
git add crates/harmony-crypto/src/hybrid_kem.rs crates/harmony-crypto/src/lib.rs crates/harmony-crypto/Cargo.toml
git commit -m "feat(crypto): add hybrid X25519 + ML-KEM-768 key establishment"
```

---

## Phase 2: PQC Identity (`harmony-identity`)

### Task 5: PqIdentity Type (Public)

**Files:**
- Create: `crates/harmony-identity/src/pq_identity.rs`
- Modify: `crates/harmony-identity/src/lib.rs`
- Modify: `crates/harmony-identity/Cargo.toml` (if needed — harmony-crypto is already a dep)

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn address_is_16_bytes() {
        let (pk, _sk) = PqPrivateIdentity::generate(&mut OsRng);
        let identity = pk.public_identity();
        assert_eq!(identity.address_hash.len(), ADDRESS_HASH_LENGTH);
    }

    #[test]
    fn address_derived_from_public_keys() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let id = priv_id.public_identity();
        // Re-derive address manually
        let mut combined = Vec::new();
        combined.extend_from_slice(&id.encryption_key.to_bytes());
        combined.extend_from_slice(&id.verifying_key.as_bytes());
        let expected = harmony_crypto::hash::truncated_hash(&combined);
        assert_eq!(id.address_hash, expected);
    }

    #[test]
    fn public_bytes_roundtrip() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let id = priv_id.public_identity();
        let bytes = id.to_public_bytes();
        assert_eq!(bytes.len(), PQ_PUBLIC_KEY_LENGTH);
        let id2 = PqIdentity::from_public_bytes(&bytes).unwrap();
        assert_eq!(id.address_hash, id2.address_hash);
    }

    #[test]
    fn different_keypairs_produce_different_addresses() {
        let id1 = PqPrivateIdentity::generate(&mut OsRng);
        let id2 = PqPrivateIdentity::generate(&mut OsRng);
        assert_ne!(
            id1.public_identity().address_hash,
            id2.public_identity().address_hash
        );
    }
}
```

**Step 2: Implement**

```rust
//! Post-quantum identity types using ML-KEM-768 and ML-DSA-65.

use alloc::vec::Vec;
use harmony_crypto::ml_kem::{self, MlKemPublicKey, MlKemSecretKey};
use harmony_crypto::ml_dsa::{self, MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature};
use harmony_crypto::hash::truncated_hash;
use rand_core::CryptoRngCore;

use crate::error::IdentityError;

pub const ADDRESS_HASH_LENGTH: usize = 16;
pub const PQ_PUBLIC_KEY_LENGTH: usize = ml_kem::PK_LENGTH + ml_dsa::PK_LENGTH; // 3136
pub const PQ_PRIVATE_KEY_LENGTH: usize = ml_kem::SK_LENGTH + ml_dsa::SK_LENGTH; // 6432
pub const PQ_SIGNATURE_LENGTH: usize = ml_dsa::SIG_LENGTH; // 3309

/// Post-quantum public identity (ML-KEM-768 + ML-DSA-65).
pub struct PqIdentity {
    pub encryption_key: MlKemPublicKey,
    pub verifying_key: MlDsaPublicKey,
    pub address_hash: [u8; ADDRESS_HASH_LENGTH],
}

impl PqIdentity {
    pub fn from_public_keys(
        encryption_key: MlKemPublicKey,
        verifying_key: MlDsaPublicKey,
    ) -> Self {
        let mut combined = Vec::with_capacity(PQ_PUBLIC_KEY_LENGTH);
        combined.extend_from_slice(&encryption_key.to_bytes());
        combined.extend_from_slice(&verifying_key.as_bytes());
        let address_hash = truncated_hash(&combined);
        Self {
            encryption_key,
            verifying_key,
            address_hash,
        }
    }

    pub fn to_public_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PQ_PUBLIC_KEY_LENGTH);
        bytes.extend_from_slice(&self.encryption_key.to_bytes());
        bytes.extend_from_slice(&self.verifying_key.as_bytes());
        bytes
    }

    pub fn from_public_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        if bytes.len() != PQ_PUBLIC_KEY_LENGTH {
            return Err(IdentityError::InvalidPublicKeyLength(bytes.len()));
        }
        let ek = MlKemPublicKey::from_bytes(&bytes[..ml_kem::PK_LENGTH])
            .map_err(|e| IdentityError::Crypto(e))?;
        let vk = MlDsaPublicKey::from_bytes(&bytes[ml_kem::PK_LENGTH..])
            .map_err(|e| IdentityError::Crypto(e))?;
        Ok(Self::from_public_keys(ek, vk))
    }

    pub fn verify(
        &self,
        message: &[u8],
        signature: &MlDsaSignature,
    ) -> Result<(), IdentityError> {
        ml_dsa::verify(&self.verifying_key, message, signature)
            .map_err(|e| IdentityError::Crypto(e))
    }
}

/// Post-quantum private identity (full keypairs). Zeroizes on drop.
pub struct PqPrivateIdentity {
    identity: PqIdentity,
    encryption_secret: MlKemSecretKey,
    signing_key: MlDsaSecretKey,
}

impl PqPrivateIdentity {
    pub fn generate(rng: &mut impl CryptoRngCore) -> Self {
        let (ek, dk) = ml_kem::generate(rng);
        let (vk, sk) = ml_dsa::generate(rng);
        let identity = PqIdentity::from_public_keys(ek, vk);
        Self {
            identity,
            encryption_secret: dk,
            signing_key: sk,
        }
    }

    pub fn public_identity(&self) -> &PqIdentity {
        &self.identity
    }

    pub fn sign(&self, message: &[u8]) -> Result<MlDsaSignature, IdentityError> {
        ml_dsa::sign(&self.signing_key, message)
            .map_err(|e| IdentityError::Crypto(e))
    }

    pub fn to_private_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PQ_PRIVATE_KEY_LENGTH);
        bytes.extend_from_slice(&self.encryption_secret.as_bytes());
        bytes.extend_from_slice(&self.signing_key.as_bytes());
        bytes
    }

    pub fn from_private_bytes(
        bytes: &[u8],
        public_bytes: &[u8],
    ) -> Result<Self, IdentityError> {
        if bytes.len() != PQ_PRIVATE_KEY_LENGTH {
            return Err(IdentityError::InvalidPrivateKeyLength(bytes.len()));
        }
        let dk = MlKemSecretKey::from_bytes(&bytes[..ml_kem::SK_LENGTH])
            .map_err(|e| IdentityError::Crypto(e))?;
        let sk = MlDsaSecretKey::from_bytes(&bytes[ml_kem::SK_LENGTH..])
            .map_err(|e| IdentityError::Crypto(e))?;
        let identity = PqIdentity::from_public_bytes(public_bytes)?;
        Ok(Self {
            identity,
            encryption_secret: dk,
            signing_key: sk,
        })
    }
}
```

**Step 3: Register module and exports in `lib.rs`**

Add `pub mod pq_identity;` and appropriate `pub use` statements.

**Step 4: Run tests**

Run: `cargo test -p harmony-identity pq_identity`
Expected: all tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-identity/src/pq_identity.rs crates/harmony-identity/src/lib.rs
git commit -m "feat(identity): add PqIdentity and PqPrivateIdentity types"
```

---

### Task 6: PqIdentity Encrypt/Decrypt

**Files:**
- Modify: `crates/harmony-identity/src/pq_identity.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn encrypt_decrypt_roundtrip() {
    let sender = PqPrivateIdentity::generate(&mut OsRng);
    let recipient = PqPrivateIdentity::generate(&mut OsRng);
    let plaintext = b"secret post-quantum message";

    let ciphertext = recipient.public_identity()
        .encrypt(&mut OsRng, plaintext).unwrap();
    let decrypted = recipient.decrypt(&ciphertext).unwrap();
    assert_eq!(decrypted, plaintext);
}

#[test]
fn wrong_recipient_cannot_decrypt() {
    let recipient = PqPrivateIdentity::generate(&mut OsRng);
    let wrong = PqPrivateIdentity::generate(&mut OsRng);
    let ciphertext = recipient.public_identity()
        .encrypt(&mut OsRng, b"secret").unwrap();
    // Decryption with wrong key should fail (or produce wrong plaintext)
    assert!(wrong.decrypt(&ciphertext).is_err());
}

#[test]
fn ciphertext_format() {
    let recipient = PqPrivateIdentity::generate(&mut OsRng);
    let ciphertext = recipient.public_identity()
        .encrypt(&mut OsRng, b"test").unwrap();
    // Format: [1088B ML-KEM ct][12B nonce][ciphertext + 16B tag]
    assert!(ciphertext.len() >= ml_kem::CT_LENGTH + 12 + 16);
}
```

**Step 2: Implement encrypt on PqIdentity**

```rust
impl PqIdentity {
    /// Encrypt plaintext to this identity using ML-KEM-768 + ChaCha20-Poly1305.
    ///
    /// Wire format: [1088B ML-KEM ciphertext][12B nonce][encrypted data + 16B tag]
    pub fn encrypt(
        &self,
        rng: &mut impl CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, IdentityError> {
        // 1. ML-KEM encapsulate → shared secret
        let (ct, ss) = ml_kem::encapsulate(rng, &self.encryption_key)
            .map_err(|e| IdentityError::Crypto(e))?;

        // 2. HKDF → symmetric key
        let key_bytes = harmony_crypto::hkdf::derive_key(
            ss.as_bytes(),
            Some(&self.address_hash),
            b"harmony-pq-encrypt-v1",
            32,
        ).map_err(|e| IdentityError::Crypto(e))?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // 3. ChaCha20-Poly1305 encrypt
        let nonce = harmony_crypto::aead::generate_nonce(rng);
        let encrypted = harmony_crypto::aead::encrypt(&key, &nonce, plaintext, &[])
            .map_err(|e| IdentityError::Crypto(e))?;

        // 4. Assemble wire format
        let mut result = Vec::with_capacity(ml_kem::CT_LENGTH + 12 + encrypted.len());
        result.extend_from_slice(ct.as_bytes());
        result.extend_from_slice(&nonce);
        result.extend_from_slice(&encrypted);
        Ok(result)
    }
}
```

**Step 3: Implement decrypt on PqPrivateIdentity**

```rust
impl PqPrivateIdentity {
    /// Decrypt ciphertext encrypted to this identity.
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, IdentityError> {
        let min_len = ml_kem::CT_LENGTH + 12 + 16; // ct + nonce + tag
        if ciphertext.len() < min_len {
            return Err(IdentityError::DecryptionFailed);
        }

        // 1. Extract ML-KEM ciphertext
        let ct = ml_kem::MlKemCiphertext::from_bytes(&ciphertext[..ml_kem::CT_LENGTH])
            .map_err(|e| IdentityError::Crypto(e))?;

        // 2. Decapsulate → shared secret
        let ss = ml_kem::decapsulate(&self.encryption_secret, &ct)
            .map_err(|e| IdentityError::Crypto(e))?;

        // 3. HKDF → symmetric key
        let key_bytes = harmony_crypto::hkdf::derive_key(
            ss.as_bytes(),
            Some(&self.identity.address_hash),
            b"harmony-pq-encrypt-v1",
            32,
        ).map_err(|e| IdentityError::Crypto(e))?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // 4. Extract nonce and decrypt
        let nonce_start = ml_kem::CT_LENGTH;
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&ciphertext[nonce_start..nonce_start + 12]);

        let encrypted_data = &ciphertext[nonce_start + 12..];
        harmony_crypto::aead::decrypt(&key, &nonce, encrypted_data, &[])
            .map_err(|_| IdentityError::DecryptionFailed)
    }
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-identity pq_identity`
Expected: all tests pass (old + new).

**Step 5: Commit**

```bash
git add crates/harmony-identity/src/pq_identity.rs
git commit -m "feat(identity): add PQC encrypt/decrypt to PqIdentity"
```

---

### Task 7: UCAN Crypto Suite Byte

**Files:**
- Modify: `crates/harmony-identity/src/ucan.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn pq_ucan_roundtrip() {
    let issuer = PqPrivateIdentity::generate(&mut OsRng);
    let audience_addr = [0xBB; 16];
    let token = issuer.issue_pq_root_token(
        &mut OsRng,
        &audience_addr,
        CapabilityType::Content,
        b"cid:abc",
        0, 9999999999,
    ).unwrap();
    let bytes = token.to_bytes();
    assert_eq!(bytes[0], 0x01); // crypto_suite = PQ
    let token2 = PqUcanToken::from_bytes(&bytes).unwrap();
    assert_eq!(token.issuer, token2.issuer);
}
```

**Step 2: Implement**

Add a `CryptoSuite` enum:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CryptoSuite {
    Ed25519 = 0x00,
    MlDsa65 = 0x01,
}
```

Create `PqUcanToken` struct with ML-DSA-65 signatures (3,309 bytes). The wire format is:

```
[1B crypto_suite=0x01][16B issuer][16B audience][1B capability]
[2B resource_len][NB resource][8B not_before][8B expires_at]
[16B nonce][1B has_proof][32B proof?][3309B ML-DSA-65 signature]
```

Add `issue_pq_root_token`, `delegate_pq`, `verify_pq_token` methods to `PqPrivateIdentity`.

The existing `UcanToken` (Ed25519) stays unchanged. The new `PqUcanToken` is a parallel type.

**Step 3: Run all identity tests**

Run: `cargo test -p harmony-identity`
Expected: all old and new tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-identity/src/ucan.rs
git commit -m "feat(identity): add PQ UCAN tokens with ML-DSA-65 signatures"
```

---

## Phase 3: Encrypted Books (`harmony-athenaeum`)

### Task 8: BookType Enum (Replace `self_indexing` bool)

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs`
- Modify: `crates/harmony-athenaeum/src/encyclopedia.rs`
- Modify: `crates/harmony-athenaeum/src/volume.rs`
- Modify: `crates/harmony-athenaeum/src/lib.rs`

This is a refactor: replace `self_indexing: bool` with `book_type: BookType` across all files.

**Step 1: Define the enum in `athenaeum.rs`**

```rust
/// Classifies how a Book's pages are structured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BookType {
    /// No embedded metadata. All pages are data.
    Raw,
    /// Page 0 is a table-of-contents (00 sentinel). Pages 1-255 are data.
    SelfIndexing,
    /// Pages 0..metadata_pages carry encrypted book metadata (11 sentinel).
    /// Remaining pages are encrypted data.
    Encrypted { metadata_pages: u8 },
}
```

**Step 2: Replace `self_indexing: bool` with `book_type: BookType`**

In `Book`:
- `self_indexing: bool` → `book_type: BookType`
- `is_self_indexing()` → `matches!(self.book_type, BookType::SelfIndexing)`
- Add `is_encrypted()` → `matches!(self.book_type, BookType::Encrypted { .. })`
- Update `data_page_count()`: handle `Encrypted { metadata_pages }` by subtracting metadata_pages
- Update `data_pages()`: skip metadata_pages for encrypted
- Update `reassemble()`: skip metadata_pages for encrypted
- Update `from_blob()`: set `book_type: BookType::Raw`
- Update `from_blob_self_indexing()`: set `book_type: BookType::SelfIndexing`

**Step 3: Update `encyclopedia.rs`**

Change `self_indexing: false` → `book_type: BookType::Raw`.

**Step 4: Update `volume.rs` serialization**

Change flags byte encoding:
```rust
// serialize_book:
let flags: u8 = match book.book_type {
    BookType::Raw => 0b00,
    BookType::SelfIndexing => 0b01,
    BookType::Encrypted { .. } => 0b10,
};
buf.extend_from_slice(&[flags, 0u8]);

// For Encrypted, also encode metadata_pages in the reserved byte:
// Actually: flags byte bits 0-1 = type, byte 39 = metadata_pages (for encrypted)

// deserialize_book:
let type_bits = data[38] & 0x03;
let book_type = match type_bits {
    0b00 => BookType::Raw,
    0b01 => BookType::SelfIndexing,
    0b10 => BookType::Encrypted { metadata_pages: data[39] },
    _ => return Err(BookError::BadFormat),
};
```

Update the `blob_size` consistency check:
```rust
let overhead_pages = match book_type {
    BookType::Raw => 0,
    BookType::SelfIndexing => 1,
    BookType::Encrypted { metadata_pages } => metadata_pages as usize,
};
let data_pages = if page_count >= overhead_pages {
    page_count - overhead_pages
} else {
    return Err(BookError::BadFormat);
};
```

**Step 5: Update `lib.rs` exports**

Add `BookType` to the public exports.

**Step 6: Run all tests**

Run: `cargo test -p harmony-athenaeum`
Expected: all 108 existing tests pass. The refactor is behavior-preserving.

**Step 7: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs crates/harmony-athenaeum/src/encyclopedia.rs \
       crates/harmony-athenaeum/src/volume.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "refactor(athenaeum): replace self_indexing bool with BookType enum"
```

---

### Task 9: EncryptedBookMetadata Type

**Files:**
- Create: `crates/harmony-athenaeum/src/encrypted.rs`
- Modify: `crates/harmony-athenaeum/src/lib.rs`

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_metadata() -> EncryptedBookMetadata {
        EncryptedBookMetadata {
            version: 1,
            flags: 0,
            encryption_algo: 0,
            owner_public_key: vec![0xAA; 1184],
            encapsulated_key: vec![0xBB; 1088],
            signature: vec![0xCC; 3309],
            expiry: None,
            tags: None,
        }
    }

    #[test]
    fn serialize_deserialize_roundtrip() {
        let meta = sample_metadata();
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta.version, meta2.version);
        assert_eq!(meta.owner_public_key, meta2.owner_public_key);
        assert_eq!(meta.encapsulated_key, meta2.encapsulated_key);
        assert_eq!(meta.signature, meta2.signature);
        assert_eq!(meta.expiry, meta2.expiry);
        assert_eq!(meta.tags, meta2.tags);
    }

    #[test]
    fn serialize_with_expiry() {
        let mut meta = sample_metadata();
        meta.flags = 0x01; // has_expiry
        meta.expiry = Some(1_700_000_000);
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta2.expiry, Some(1_700_000_000));
    }

    #[test]
    fn serialize_with_tags() {
        let mut meta = sample_metadata();
        meta.flags = 0x02; // has_tags
        meta.tags = Some(b"topic:science".to_vec());
        let bytes = meta.to_bytes();
        let meta2 = EncryptedBookMetadata::from_bytes(&bytes).unwrap();
        assert_eq!(meta2.tags.as_deref(), Some(b"topic:science".as_slice()));
    }

    #[test]
    fn minimum_size_is_5591_bytes() {
        let meta = sample_metadata();
        let bytes = meta.to_bytes();
        assert_eq!(bytes.len(), 10 + 1184 + 1088 + 3309); // 5591
    }

    #[test]
    fn pages_needed_for_minimum_metadata() {
        let meta = sample_metadata();
        assert_eq!(meta.pages_needed(), 2);
    }
}
```

**Step 2: Implement**

```rust
//! Encrypted book metadata format.
//!
//! The metadata is stored in the first N pages of an encrypted book,
//! each prefixed with the 11 sentinel (0xFFFFFFFC). This module handles
//! serialization/deserialization of the metadata payload independently
//! of any cryptographic types — all key material is raw bytes.

use alloc::vec::Vec;
use crate::addr::PAGE_SIZE;
use crate::athenaeum::BookError;

/// Sentinel value marking encrypted book metadata pages.
pub const ENCRYPTED_SENTINEL: u32 = crate::addr::SELF_INDEX_SENTINEL_11;

/// Usable payload per metadata page (4096 - 4 byte sentinel prefix).
pub const METADATA_PAGE_PAYLOAD: usize = PAGE_SIZE - 4;

/// Metadata for an encrypted book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EncryptedBookMetadata {
    pub version: u16,
    pub flags: u8,
    pub encryption_algo: u8,
    pub owner_public_key: Vec<u8>,
    pub encapsulated_key: Vec<u8>,
    pub signature: Vec<u8>,
    pub expiry: Option<u64>,
    pub tags: Option<Vec<u8>>,
}

impl EncryptedBookMetadata {
    /// Serialize metadata to bytes (without sentinel prefixes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.push(self.flags);
        buf.push(self.encryption_algo);
        buf.extend_from_slice(&(self.owner_public_key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.owner_public_key);
        buf.extend_from_slice(&(self.encapsulated_key.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.encapsulated_key);
        buf.extend_from_slice(&(self.signature.len() as u16).to_le_bytes());
        buf.extend_from_slice(&self.signature);
        if self.flags & 0x01 != 0 {
            if let Some(expiry) = self.expiry {
                buf.extend_from_slice(&expiry.to_le_bytes());
            }
        }
        if self.flags & 0x02 != 0 {
            if let Some(ref tags) = self.tags {
                buf.extend_from_slice(&(tags.len() as u16).to_le_bytes());
                buf.extend_from_slice(tags);
            }
        }
        buf
    }

    /// Deserialize metadata from bytes (without sentinel prefixes).
    pub fn from_bytes(data: &[u8]) -> Result<Self, BookError> {
        // Parse fields sequentially...
        // (full implementation with bounds checks)
    }

    /// How many 4KB pages are needed to store this metadata (with sentinel prefixes).
    pub fn pages_needed(&self) -> u8 {
        let payload_len = self.to_bytes().len();
        ((payload_len + METADATA_PAGE_PAYLOAD - 1) / METADATA_PAGE_PAYLOAD) as u8
    }

    /// Build metadata pages: each page starts with 0xFFFFFFFC sentinel,
    /// followed by payload bytes, zero-padded to PAGE_SIZE.
    pub fn to_pages(&self) -> Vec<[u8; PAGE_SIZE]> {
        let payload = self.to_bytes();
        let mut pages = Vec::new();
        for chunk in payload.chunks(METADATA_PAGE_PAYLOAD) {
            let mut page = [0u8; PAGE_SIZE];
            page[..4].copy_from_slice(&ENCRYPTED_SENTINEL.to_le_bytes());
            page[4..4 + chunk.len()].copy_from_slice(chunk);
            pages.push(page);
        }
        pages
    }

    /// Parse metadata from concatenated page data.
    /// Each page must start with the 11 sentinel.
    pub fn from_pages(pages: &[[u8; PAGE_SIZE]]) -> Result<Self, BookError> {
        let mut payload = Vec::new();
        for page in pages {
            let sentinel = u32::from_le_bytes([page[0], page[1], page[2], page[3]]);
            if sentinel != ENCRYPTED_SENTINEL {
                return Err(BookError::BadFormat);
            }
            payload.extend_from_slice(&page[4..]);
        }
        Self::from_bytes(&payload)
    }
}
```

**Step 3: Register module and exports**

Add `pub mod encrypted;` to `lib.rs`. Export `EncryptedBookMetadata` and `ENCRYPTED_SENTINEL`.

**Step 4: Run tests**

Run: `cargo test -p harmony-athenaeum encrypted`
Expected: all tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/encrypted.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "feat(athenaeum): add EncryptedBookMetadata type"
```

---

### Task 10: Encrypted Book Construction and Detection

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn from_blob_encrypted_basic() {
    let meta = sample_encrypted_metadata();
    let data = vec![0x42u8; PAGE_SIZE * 2];
    let book = Book::from_blob_encrypted(test_cid(), &data, &meta).unwrap();
    assert!(book.is_encrypted());
    assert_eq!(book.book_type, BookType::Encrypted { metadata_pages: 2 });
    assert_eq!(book.data_page_count(), 2);
    // Total pages = 2 metadata + 2 data = 4
    assert_eq!(book.page_count(), 4);
}

#[test]
fn is_encrypted_blob_detection() {
    let meta = sample_encrypted_metadata();
    let data = vec![0x42u8; PAGE_SIZE];
    let book = Book::from_blob_encrypted(test_cid(), &data, &meta).unwrap();
    let pages = book.page_data_from_blob_encrypted(&data, &meta);
    assert!(Book::is_encrypted_blob(&pages[0]));
    assert!(!Book::is_encrypted_blob(&pages[2])); // data page
}

#[test]
fn metadata_roundtrip_via_book() {
    let meta = sample_encrypted_metadata();
    let data = vec![0x55u8; PAGE_SIZE * 3];
    let book = Book::from_blob_encrypted(test_cid(), &data, &meta).unwrap();
    let pages = book.page_data_from_blob_encrypted(&data, &meta);
    let meta_pages: Vec<[u8; PAGE_SIZE]> = pages[..2]
        .iter()
        .map(|p| {
            let mut arr = [0u8; PAGE_SIZE];
            arr.copy_from_slice(p);
            arr
        })
        .collect();
    let recovered = EncryptedBookMetadata::from_pages(&meta_pages).unwrap();
    assert_eq!(recovered.owner_public_key, meta.owner_public_key);
}

#[test]
fn encrypted_book_reassemble_skips_metadata() {
    let meta = sample_encrypted_metadata();
    let data = vec![0x77u8; PAGE_SIZE + 100];
    let book = Book::from_blob_encrypted(test_cid(), &data, &meta).unwrap();
    let all_pages = book.page_data_from_blob_encrypted(&data, &meta);
    let reassembled = book
        .reassemble(|idx| Some(all_pages[idx as usize].clone()))
        .unwrap();
    assert_eq!(reassembled, data);
}
```

**Step 2: Implement**

Add to `Book`:

```rust
/// Build an encrypted Book with metadata pages followed by data pages.
pub fn from_blob_encrypted(
    cid: [u8; 32],
    data: &[u8],
    metadata: &EncryptedBookMetadata,
) -> Result<Self, BookError> {
    let meta_page_count = metadata.pages_needed();
    let max_data = (PAGES_PER_BOOK - meta_page_count as usize) * PAGE_SIZE;
    if data.len() > max_data {
        return Err(BookError::BlobTooLarge { size: data.len() });
    }

    // Compute data page addresses (same as from_blob)
    let data_page_count = if data.is_empty() { 0 } else { (data.len() + PAGE_SIZE - 1) / PAGE_SIZE };
    let mut pages = Vec::with_capacity(meta_page_count as usize + data_page_count);

    // Metadata pages: compute addresses from metadata page content
    let meta_pages = metadata.to_pages();
    for page_buf in &meta_pages {
        let variants = [
            PageAddr::from_data(page_buf, Algorithm::Sha256Msb),
            PageAddr::from_data(page_buf, Algorithm::Sha256Lsb),
            PageAddr::from_data(page_buf, Algorithm::Sha224Msb),
            PageAddr::from_data(page_buf, Algorithm::Sha224Lsb),
        ];
        pages.push(variants);
    }

    // Data pages: compute addresses from data content
    for chunk in data.chunks(PAGE_SIZE) {
        let mut page_buf = [0u8; PAGE_SIZE];
        page_buf[..chunk.len()].copy_from_slice(chunk);
        let variants = [
            PageAddr::from_data(&page_buf, Algorithm::Sha256Msb),
            PageAddr::from_data(&page_buf, Algorithm::Sha256Lsb),
            PageAddr::from_data(&page_buf, Algorithm::Sha224Msb),
            PageAddr::from_data(&page_buf, Algorithm::Sha224Lsb),
        ];
        pages.push(variants);
    }

    Ok(Book {
        cid,
        pages,
        blob_size: data.len() as u32,
        book_type: BookType::Encrypted { metadata_pages: meta_page_count },
    })
}

/// Check if a blob starts with the encrypted book sentinel (0xFFFFFFFC).
pub fn is_encrypted_blob(blob: &[u8]) -> bool {
    if blob.len() < 4 {
        return false;
    }
    let first = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    first == SELF_INDEX_SENTINEL_11
}

/// Generate page data for an encrypted book (metadata pages + data pages).
pub fn page_data_from_blob_encrypted(
    &self,
    data: &[u8],
    metadata: &EncryptedBookMetadata,
) -> Vec<Vec<u8>> {
    let mut pages = Vec::new();

    // Metadata pages
    for page_buf in metadata.to_pages() {
        pages.push(page_buf.to_vec());
    }

    // Data pages (same as raw)
    for chunk in data.chunks(PAGE_SIZE) {
        let mut page_buf = vec![0u8; PAGE_SIZE];
        page_buf[..chunk.len()].copy_from_slice(chunk);
        pages.push(page_buf);
    }

    pages
}
```

**Step 3: Run tests**

Run: `cargo test -p harmony-athenaeum`
Expected: all tests pass (old + new).

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): add encrypted book construction and detection"
```

---

### Task 11: Integration Tests

**Files:**
- Modify: `crates/harmony-athenaeum/src/lib.rs` (test module)

**Step 1: Write integration tests**

```rust
#[test]
fn encrypted_book_end_to_end() {
    // Build encrypted book
    let meta = EncryptedBookMetadata {
        version: 1,
        flags: 0x01, // has_expiry
        encryption_algo: 0,
        owner_public_key: vec![0xAA; 1184],
        encapsulated_key: vec![0xBB; 1088],
        signature: vec![0xCC; 3309],
        expiry: Some(1_700_000_000),
        tags: None,
    };

    let data = vec![0x42u8; PAGE_SIZE * 5 + 123]; // 5+ pages of data
    let book = Book::from_blob_encrypted([0xDD; 32], &data, &meta).unwrap();

    assert!(book.is_encrypted());
    assert_eq!(book.data_page_count(), 6); // ceil(5*4096 + 123 / 4096) = 6
    assert_eq!(book.page_count(), 8); // 2 meta + 6 data

    // Generate all page data
    let all_pages = book.page_data_from_blob_encrypted(&data, &meta);
    assert_eq!(all_pages.len(), 8);

    // First 2 pages should start with encrypted sentinel
    assert!(Book::is_encrypted_blob(&all_pages[0]));
    assert!(Book::is_encrypted_blob(&all_pages[1]));
    assert!(!Book::is_encrypted_blob(&all_pages[2])); // data page

    // Reassemble should recover original data
    let reassembled = book
        .reassemble(|idx| Some(all_pages[idx as usize].clone()))
        .unwrap();
    assert_eq!(reassembled, data);

    // Metadata should be recoverable from pages
    let meta_page_arrays: Vec<[u8; PAGE_SIZE]> = all_pages[..2]
        .iter()
        .map(|p| {
            let mut arr = [0u8; PAGE_SIZE];
            arr.copy_from_slice(p);
            arr
        })
        .collect();
    let recovered = EncryptedBookMetadata::from_pages(&meta_page_arrays).unwrap();
    assert_eq!(recovered.owner_public_key, meta.owner_public_key);
    assert_eq!(recovered.expiry, Some(1_700_000_000));
}

#[test]
fn book_type_backward_compat() {
    // Raw books still work
    let raw = Book::from_blob([0xAA; 32], &[0x42; PAGE_SIZE]).unwrap();
    assert!(!raw.is_self_indexing());
    assert!(!raw.is_encrypted());
    assert_eq!(raw.data_page_count(), 1);

    // Self-indexing books still work
    let si = Book::from_blob_self_indexing([0xBB; 32], &[0x42; PAGE_SIZE]).unwrap();
    assert!(si.is_self_indexing());
    assert!(!si.is_encrypted());
    assert_eq!(si.data_page_count(), 1);
}
```

**Step 2: Run full test suite**

Run: `cargo test --workspace`
Expected: ALL tests pass across all crates.

**Step 3: Run clippy**

Run: `cargo clippy --workspace`
Expected: zero warnings.

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/lib.rs
git commit -m "test(athenaeum): add encrypted book integration tests"
```

---

## Final Verification

After all tasks are complete:

```bash
cargo test --workspace       # All tests pass
cargo clippy --workspace     # Zero warnings
cargo fmt --all -- --check   # Format clean
```

The implementation is ready for delivery via `/delivertask`.

# ZEB-746 PR 1 (core) — `harmony_crypto::password_envelope` + converge recovery

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared, feature-gated Argon2id+XChaCha20-Poly1305 password-envelope primitive to `harmony-crypto`, then rewire `harmony-owner/recovery` onto it with zero on-disk byte change.

**Architecture:** A format-agnostic `seal`/`open` kernel: the caller supplies Argon2id params, salt, nonce, the already-serialized header as opaque `aad`, and opaque plaintext. The primitive does Argon2id-derive-then-XChaCha20-Poly1305 and nothing format-specific. `harmony-owner/recovery` becomes the first consumer; the client's HRMI/HRSS follow in PR 2.

**Tech Stack:** Rust, `argon2 0.5`, `chacha20poly1305 0.10` (XChaCha20 path), `zeroize`. `harmony-crypto` is `no_std`+`alloc`.

**Spec:** `docs/specs/2026-07-24-zeb-746-at-rest-envelope-extraction-design.md`.

## Global Constraints

- **Byte-identity is the non-negotiable acceptance gate.** `harmony-owner`'s golden fixtures (`tests/fixtures/recovery_v1.bin`, `recovery_v1_no_metadata.bin`) and the `wire.rs` header golden MUST stay byte-identical after the rewire. Never set `HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE`.
- **Feature-gated, default-off.** The new code lives behind a new `harmony-crypto` feature `password-envelope` that turns on an **optional** `argon2` dep. Default `harmony-crypto` builds (the 20+ crates that never enable it) must gain no new dependency and must still compile `no_std`.
- **The primitive is format-agnostic.** It must never hard-code a header, magic, version, or CBOR body. Magic/version/AAD/plaintext-shape all stay caller-side. This is what lets the same function back HRMI, HRSS, and HRMR.
- **No new randomness in the primitive.** Salt and nonce are caller-supplied (preserves deterministic fixtures). Fixed invariants: derived key = 32 bytes; nonce must be exactly 24 bytes.
- **Exact Argon2 construction (byte-frozen):** `Algorithm::Argon2id`, `Version::V0x13`, output length 32. Caller passes `m_kib`/`t_cost`/`p_cost`. Recovery's pinned values: m=65536, t=3, p=1.
- **Panic-free on all caller inputs.** Wrong nonce length / invalid params / short salt return `CryptoError`, never panic.

---

## File Structure

- `crates/harmony-crypto/src/password_envelope.rs` — **new** module: `Argon2idParams`, `seal`, `open`, length consts, unit tests.
- `crates/harmony-crypto/src/lib.rs` — add feature-gated `pub mod password_envelope;`.
- `crates/harmony-crypto/src/error.rs` — add `InvalidNonceLength` + `InvalidKdfParams` variants.
- `crates/harmony-crypto/Cargo.toml` — add optional `argon2` dep + `password-envelope` feature.
- `crates/harmony-owner/src/recovery/encrypted_file.rs` — rewire `encrypt_core`/`decrypt_inner` onto the primitive; delete `derive_key_argon2id`.
- `crates/harmony-owner/Cargo.toml` — `recovery` feature enables `harmony-crypto/password-envelope`; drop now-unused direct `argon2`/`chacha20poly1305` deps.

---

### Task 1: `harmony_crypto::password_envelope` module + feature + error variants

**Files:**
- Create: `crates/harmony-crypto/src/password_envelope.rs`
- Modify: `crates/harmony-crypto/src/lib.rs`
- Modify: `crates/harmony-crypto/src/error.rs`
- Modify: `crates/harmony-crypto/Cargo.toml`

**Interfaces:**
- Produces (consumed by Task 2 and by PR 2):
  - `pub struct Argon2idParams` (fields private) with `pub fn new(m_kib: u32, t_cost: u32, p_cost: u32) -> Result<Self, CryptoError>`
  - `pub fn seal(password: &[u8], params: &Argon2idParams, salt: &[u8], nonce: &[u8], aad: &[u8], plaintext: &[u8]) -> Result<Vec<u8>, CryptoError>` (returns `ct‖tag`)
  - `pub fn open(password: &[u8], params: &Argon2idParams, salt: &[u8], nonce: &[u8], aad: &[u8], ciphertext: &[u8]) -> Result<Zeroizing<Vec<u8>>, CryptoError>`
  - `pub const KEY_LEN: usize = 32; pub const NONCE_LEN: usize = 24; pub const TAG_LEN: usize = 16;`

- [ ] **Step 1: Add the two `CryptoError` variants**

In `crates/harmony-crypto/src/error.rs`, add these variants to the `CryptoError` enum (place after `InvalidKeyLength`):

```rust
    #[error("invalid nonce length: expected {expected}, got {got}")]
    InvalidNonceLength { expected: usize, got: usize },

    #[error("invalid or out-of-range Argon2 KDF parameters")]
    InvalidKdfParams,
```

- [ ] **Step 2: Add the optional dep + feature to `Cargo.toml`**

In `crates/harmony-crypto/Cargo.toml`, add to `[dependencies]` (after `chacha20poly1305`):

```toml
argon2 = { workspace = true, optional = true }
```

And in `[features]`, add (after the `serde = [...]` line):

```toml
# Password-based at-rest encryption envelope (Argon2id + XChaCha20-Poly1305).
# Pulls the heavy `argon2` dep; default-off so the 20+ crates that never use it
# gain nothing. no_std+alloc clean (no getrandom — caller supplies salt/nonce).
password-envelope = ["dep:argon2"]
```

Do **not** add anything to the `std` feature list — the module needs neither `getrandom` nor `std`.

- [ ] **Step 3: Register the module in `lib.rs`**

In `crates/harmony-crypto/src/lib.rs`, add between `pub mod ml_kem;` and `pub mod sealed_box;`:

```rust
#[cfg(feature = "password-envelope")]
pub mod password_envelope;
```

- [ ] **Step 4: Write the module with `seal`/`open` + unit tests**

Create `crates/harmony-crypto/src/password_envelope.rs`:

```rust
//! Password-based at-rest encryption envelope: Argon2id key derivation followed
//! by XChaCha20-Poly1305 authenticated encryption.
//!
//! This is the shared kernel behind the app-side identity vault (`HRMI`), the
//! owner-state snapshot backup (`HRSS`), and the owner recovery file (`HRMR`).
//! Each caller frames its own header, chooses its own magic, and binds whatever
//! bytes it likes as `aad`; this module knows nothing about headers, versions,
//! or on-disk layout. That separation is what lets one primitive back three
//! byte-frozen formats — the format-distinguishing bytes live in the
//! caller-supplied `aad`, never in here.
//!
//! # Determinism / nonce reuse
//!
//! The caller supplies `salt` and `nonce`; this module generates no randomness,
//! so callers can reproduce byte-identical output for golden fixtures. **A
//! (key, nonce) pair must never be reused** — production callers draw a fresh
//! random salt and nonce per [`seal`].

use alloc::vec::Vec;

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
use zeroize::Zeroizing;

use crate::CryptoError;

/// Derived-key / AEAD-key length in bytes (fixed at the XChaCha20 key size).
pub const KEY_LEN: usize = 32;

/// XChaCha20-Poly1305 nonce length in bytes.
pub const NONCE_LEN: usize = 24;

/// Poly1305 authentication tag length in bytes.
pub const TAG_LEN: usize = 16;

/// Argon2id cost parameters. Caller-supplied because each on-disk format encodes
/// these values in its own header and must round-trip byte-for-byte. The derived
/// output length is fixed at [`KEY_LEN`] (32 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2idParams {
    m_kib: u32,
    t_cost: u32,
    p_cost: u32,
}

impl Argon2idParams {
    /// Validate and construct. Returns [`CryptoError::InvalidKdfParams`] if the
    /// triple is out of range for Argon2 (memory/iterations/lanes below the
    /// algorithm minimums), validated against the fixed 32-byte output length.
    pub fn new(m_kib: u32, t_cost: u32, p_cost: u32) -> Result<Self, CryptoError> {
        Params::new(m_kib, t_cost, p_cost, Some(KEY_LEN)).map_err(|_| CryptoError::InvalidKdfParams)?;
        Ok(Self {
            m_kib,
            t_cost,
            p_cost,
        })
    }
}

/// Argon2id-derive a 32-byte key. Fails (`InvalidKdfParams`) only on
/// out-of-range params or an Argon2-unacceptable salt length — never panics.
fn derive_key(
    password: &[u8],
    salt: &[u8],
    params: &Argon2idParams,
) -> Result<Zeroizing<[u8; KEY_LEN]>, CryptoError> {
    let argon_params = Params::new(params.m_kib, params.t_cost, params.p_cost, Some(KEY_LEN))
        .map_err(|_| CryptoError::InvalidKdfParams)?;
    let kdf = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);
    let mut out: Zeroizing<[u8; KEY_LEN]> = Zeroizing::new([0u8; KEY_LEN]);
    kdf.hash_password_into(password, salt, out.as_mut())
        .map_err(|_| CryptoError::InvalidKdfParams)?;
    Ok(out)
}

/// Argon2id-derive a key from `password`/`salt`/`params`, then
/// XChaCha20-Poly1305-seal `plaintext` under `nonce` (24 bytes), binding `aad`.
/// Returns `ciphertext ‖ tag` (16-byte Poly1305 tag). The derived key is
/// zeroized before return.
pub fn seal(
    password: &[u8],
    params: &Argon2idParams,
    salt: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::InvalidNonceLength {
            expected: NONCE_LEN,
            got: nonce.len(),
        });
    }
    let key = derive_key(password, salt, params)?;
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    cipher
        .encrypt(XNonce::from_slice(nonce), Payload { msg: plaintext, aad })
        .map_err(|_| CryptoError::AeadEncryptFailed)
}

/// Inverse of [`seal`]. Returns the zeroizing plaintext, or a [`CryptoError`] on
/// any failure (wrong key / tampered ciphertext / tag mismatch). Callers that
/// must not expose a decryption oracle collapse every error into one
/// indistinguishable message.
pub fn open(
    password: &[u8],
    params: &Argon2idParams,
    salt: &[u8],
    nonce: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::InvalidNonceLength {
            expected: NONCE_LEN,
            got: nonce.len(),
        });
    }
    let key = derive_key(password, salt, params)?;
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let pt = cipher
        .decrypt(XNonce::from_slice(nonce), Payload { msg: ciphertext, aad })
        .map_err(|_| CryptoError::AeadDecryptFailed)?;
    Ok(Zeroizing::new(pt))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cheap params keep the unit tests fast — the primitive is param-generic,
    // so correctness does not depend on recovery's production 64 MiB cost.
    fn cheap() -> Argon2idParams {
        Argon2idParams::new(8, 1, 1).expect("8 KiB / t=1 / p=1 is valid")
    }

    #[test]
    fn round_trip() {
        let p = cheap();
        let salt = [0xABu8; 16];
        let nonce = [0xCDu8; NONCE_LEN];
        let ct = seal(b"pw", &p, &salt, &nonce, b"hdr", b"hello harmony").unwrap();
        assert_eq!(ct.len(), b"hello harmony".len() + TAG_LEN);
        let pt = open(b"pw", &p, &salt, &nonce, b"hdr", &ct).unwrap();
        assert_eq!(&pt[..], b"hello harmony");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let p = cheap();
        let salt = [1u8; 16];
        let nonce = [2u8; NONCE_LEN];
        let a = seal(b"pw", &p, &salt, &nonce, b"aad", b"msg").unwrap();
        let b = seal(b"pw", &p, &salt, &nonce, b"aad", b"msg").unwrap();
        assert_eq!(a, b, "identical inputs must produce identical bytes");
    }

    #[test]
    fn wrong_password_fails() {
        let p = cheap();
        let salt = [3u8; 16];
        let nonce = [4u8; NONCE_LEN];
        let ct = seal(b"right", &p, &salt, &nonce, b"", b"data").unwrap();
        let err = open(b"wrong", &p, &salt, &nonce, b"", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn wrong_aad_fails() {
        let p = cheap();
        let salt = [5u8; 16];
        let nonce = [6u8; NONCE_LEN];
        let ct = seal(b"pw", &p, &salt, &nonce, b"aad-1", b"data").unwrap();
        let err = open(b"pw", &p, &salt, &nonce, b"aad-2", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let p = cheap();
        let salt = [7u8; 16];
        let nonce = [8u8; NONCE_LEN];
        let mut ct = seal(b"pw", &p, &salt, &nonce, b"", b"data").unwrap();
        ct[0] ^= 0x01;
        let err = open(b"pw", &p, &salt, &nonce, b"", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn wrong_nonce_length_rejected_not_panicked() {
        let p = cheap();
        let short = [0u8; 12];
        let err = seal(b"pw", &p, &[0u8; 16], &short, b"", b"data").unwrap_err();
        assert!(matches!(
            err,
            CryptoError::InvalidNonceLength {
                expected: 24,
                got: 12
            }
        ));
        let err = open(b"pw", &p, &[0u8; 16], &short, b"", b"xxxxxxxxxxxxxxxxx").unwrap_err();
        assert!(matches!(err, CryptoError::InvalidNonceLength { .. }));
    }

    #[test]
    fn invalid_params_rejected() {
        // t_cost = 0 is below Argon2's minimum (1).
        let err = Argon2idParams::new(65536, 0, 1).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidKdfParams));
    }
}
```

- [ ] **Step 5: Run the module tests (expect PASS)**

Run: `cargo nextest run -p harmony-crypto --features password-envelope -E 'test(password_envelope)'`
Expected: all 7 tests pass.

- [ ] **Step 6: Confirm the feature gate — default build excludes the module & dep**

Run: `cargo build -p harmony-crypto` (default features; `password-envelope` off)
Expected: builds clean; `argon2` is **not** compiled (it is optional and unreferenced).

Run: `cargo build -p harmony-crypto --no-default-features --features password-envelope`
Expected: builds clean — proves the module is `no_std`+`alloc` and pulls no `getrandom`/`std`.

- [ ] **Step 7: Clippy**

Run: `cargo clippy -p harmony-crypto --all-targets --features password-envelope -- -D warnings`
Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-crypto/
git commit -m "feat(harmony-crypto): add password_envelope (Argon2id+XChaCha20) behind a default-off feature"
```

---

### Task 2: Rewire `harmony-owner/recovery` onto the primitive

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`
- Modify: `crates/harmony-owner/Cargo.toml`

**Interfaces:**
- Consumes (from Task 1): `harmony_crypto::password_envelope::{seal, open, Argon2idParams}`.

- [ ] **Step 1: Wire the feature; drop the now-unused direct deps in `Cargo.toml`**

In `crates/harmony-owner/Cargo.toml`:

Remove these two lines from `[dependencies]` (they are used only in `encrypted_file.rs`, which no longer references them after this task):

```toml
argon2 = { workspace = true, optional = true }
chacha20poly1305 = { workspace = true, optional = true }
```

Change the `recovery` feature from:

```toml
recovery = ["dep:bip39", "dep:argon2", "dep:chacha20poly1305", "dep:secrecy"]
```

to:

```toml
recovery = ["dep:bip39", "dep:secrecy", "harmony-crypto/password-envelope"]
```

- [ ] **Step 2: Swap the imports in `encrypted_file.rs`**

Replace the `chacha20poly1305` import block (lines 12–15):

```rust
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
```

with:

```rust
use harmony_crypto::password_envelope::{self, Argon2idParams};
```

Then drop `KDF_OUT_LEN` from the `crate::recovery::wire::{...}` import list (it was used only by the deleted `derive_key_argon2id`). The list keeps `parse_header, serialize_header, HEADER_LEN, KDF_M_KIB, KDF_P, KDF_T, MAX_FILE_LEN, MIN_FILE_LEN, NONCE_LEN, SALT_LEN`.

- [ ] **Step 3: Add the params helper; rewrite `encrypt_core`; delete `derive_key_argon2id`**

Replace `encrypt_core` (lines 61–85) and the entire `derive_key_argon2id` fn (lines 87–99) with:

```rust
/// The recovery envelope's fixed Argon2id parameters. Equal to the `wire::`
/// constants that are serialized into (and strict-checked out of) the header.
fn recovery_kdf_params() -> Argon2idParams {
    Argon2idParams::new(KDF_M_KIB, KDF_T as u32, KDF_P as u32)
        .expect("recovery Argon2id params are constants known to validate")
}

/// Internal encrypt core. Takes already-built plaintext + already-chosen
/// salt + nonce. Both `to_encrypted_file` (production, random salt+nonce)
/// and `encrypt_with_params_for_test` (deterministic for fixtures) call
/// this.
fn encrypt_core(
    passphrase: &SecretString,
    plaintext: &[u8],
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
) -> Vec<u8> {
    let header = serialize_header();
    let ct = password_envelope::seal(
        passphrase.expose_secret().as_bytes(),
        &recovery_kdf_params(),
        salt,
        nonce,
        &header,
        plaintext,
    )
    .expect("sealing is infallible for a 24-byte nonce and validated params");
    let mut out = Vec::with_capacity(HEADER_LEN + SALT_LEN + NONCE_LEN + ct.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(salt);
    out.extend_from_slice(nonce);
    out.extend_from_slice(&ct);
    out
}
```

(The header is still serialized locally and passed as `aad`, so the AEAD tag over the header is unchanged — the output is byte-identical to the previous inline XChaCha20 path.)

- [ ] **Step 4: Rewrite the decrypt crypto section of `decrypt_inner`**

Replace lines 238–251 (from `let header = serialize_header();` through the `Zeroizing::new(plaintext_vec)` line) with:

```rust
    let header = serialize_header();
    let plaintext: Zeroizing<Vec<u8>> = password_envelope::open(
        passphrase.expose_secret().as_bytes(),
        &recovery_kdf_params(),
        salt,
        nonce,
        &header,
        ciphertext_and_tag,
    )
    .map_err(|_| RecoveryError::WrongPassphraseOrCorrupt)?;
```

Everything below (the `RecoveryFileBody` CBOR decode, `FORMAT_STRING` check, comment-length check, seed/metadata return) is unchanged. `open` already returns `Zeroizing<Vec<u8>>`, so the manual `Zeroizing::new(...)` wrapper is gone. Mapping every `open` error to `WrongPassphraseOrCorrupt` preserves the uniform, oracle-free error (header/param tampering is still caught earlier by `parse_header`).

- [ ] **Step 5: Run the recovery tests — byte-identity gate**

Run: `cargo nextest run -p harmony-owner --features test-fixtures`
Expected: all pass, including `recovery_wire_format_fixture::wire_format_v1_full_metadata_pinned` and `wire_format_v1_no_metadata_pinned` (the golden `.bin` byte-equality), all `decrypt_tests`, `prod_encrypt_tests`, and the `wire.rs` header golden. **If any golden test fails, STOP — the rewire changed the on-disk bytes; do not regenerate the fixture.**

- [ ] **Step 6: Clippy**

Run: `cargo clippy -p harmony-owner --all-targets --features test-fixtures -- -D warnings`
Expected: clean (no unused imports/deps left behind).

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-owner/
git commit -m "refactor(harmony-owner): converge recovery envelope onto harmony_crypto::password_envelope"
```

---

### Task 3: CI-parity verification sweep

No code changes — this is the authoritative gate for the dependency/feature change (a manifest change touches the whole workspace dep graph, so the scoped per-task gates are not sufficient on their own).

- [ ] **Step 1: Format**

Run: `cargo fmt --all -- --check`
Expected: clean.

- [ ] **Step 2: Default-feature workspace build (gate stays off)**

Run: `cargo build --workspace`
Expected: clean; `password-envelope` is off by default, `harmony-owner` builds with `recovery` (default) pulling the feature transitively.

- [ ] **Step 3: Workspace clippy**

Run: `cargo clippy --workspace --all-targets -- -D warnings`
Expected: clean.

- [ ] **Step 4: Targeted tests**

Run: `cargo nextest run -p harmony-crypto --features password-envelope`
Run: `cargo nextest run -p harmony-owner --features test-fixtures`
Expected: all pass.

- [ ] **Step 5: `no_std` gate for the new module**

Run: `cargo build -p harmony-crypto --no-default-features --features password-envelope`
Expected: clean — confirms the new deps stay `no_std`+`alloc` and never leak `std`/`getrandom` into no_std consumers.

- [ ] **Step 6: Confirm the golden fixtures are unchanged in git**

Run: `git status --porcelain crates/harmony-owner/tests/fixtures/`
Expected: **empty** (no fixture bytes modified anywhere in this PR).

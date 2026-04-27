# Identity Backup / Restore — Library Primitives — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `harmony-owner::recovery` module: BIP39-24 mnemonic encoding + Argon2id+XChaCha20-Poly1305 encrypted-file format for the existing 32-byte `RecoveryArtifact` seed, all gated behind a default-on `recovery` Cargo feature.

**Architecture:** New module under `crates/harmony-owner/src/recovery/` with submodules for `mnemonic`, `encrypted_file`, `wire`, and `error`. Public API exposed as inherent methods on the existing `RecoveryArtifact` (no refactor of `lifecycle/mint.rs`). Encrypted-file uses a 13-byte header bound as AAD (matches the ZEB-174 pattern but with magic `HRMR` instead of `HRMI`), variable-length CBOR payload (seed + optional `mint_at` + optional ≤256-byte `comment`), capped at 1024 bytes. A second `test-fixtures` Cargo feature exposes a deterministic-input encrypt helper used by the wire-format fixture test. **Out of plan scope:** harmony-client integration (separate ticket).

**Tech Stack:** Rust 2021, `bip39` v2 (English wordlist, `zeroize` feature on for mnemonic-entropy wiping), `argon2` v0.5 (Argon2id m=64MiB,t=3,p=1), `chacha20poly1305` v0.10 (XChaCha20-Poly1305), `ciborium` (already a workspace dep, used for canonical CBOR via existing `harmony-owner::cbor`), `secrecy` v0.10 (`SecretString` for passphrases), `subtle` (constant-time comparison), `zeroize` (already in harmony-owner).

**Spec:** `docs/superpowers/specs/2026-04-26-identity-backup-restore-design.md`

**Bead/Issue:** ZEB-175 (Track A child #3; under umbrella ZEB-169). Library half only — harmony-client integration is a separate follow-on ticket to be filed once this lands.

---

## File Structure

**Create:**
- `crates/harmony-owner/src/recovery/mod.rs` — public API re-exports + `RestoredArtifact` / `RecoveryMetadata` types + `impl RecoveryArtifact { to_mnemonic, from_mnemonic, to_encrypted_file, from_encrypted_file }`
- `crates/harmony-owner/src/recovery/error.rs` — `RecoveryError` enum
- `crates/harmony-owner/src/recovery/wire.rs` — encrypted-file constants + `serialize_header` + `parse_header`
- `crates/harmony-owner/src/recovery/mnemonic.rs` — BIP39-24 encode/decode internals
- `crates/harmony-owner/src/recovery/encrypted_file.rs` — Argon2id + XChaCha20-Poly1305 encode/decode internals + test-fixtures-gated deterministic helper
- `crates/harmony-owner/tests/recovery_wire_format_fixture.rs` — wire format byte-for-byte pin via committed fixture file
- `crates/harmony-owner/tests/fixtures/recovery_v1.bin` — committed binary fixture, regenerable via `HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1`

**Modify:**
- `crates/harmony-owner/src/lib.rs` — add `pub mod recovery;` gated by `#[cfg(feature = "recovery")]`
- `crates/harmony-owner/Cargo.toml` — add `[features]` (default-on `recovery`, opt-in `test-fixtures`); add optional deps; declare integration test's required feature
- `Cargo.toml` (workspace root) — add `bip39` and `secrecy` to `[workspace.dependencies]`

---

### Task 1: Workspace + crate Cargo.toml wiring

**Files:**
- Modify: `Cargo.toml` (workspace root, lines around 100–110)
- Modify: `crates/harmony-owner/Cargo.toml`

- [ ] **Step 1: Add bip39 and secrecy to workspace deps**

In `Cargo.toml` (workspace root), append to the `[workspace.dependencies]` block (alongside the existing entries near line 100):

```toml
# Identity recovery (BIP39 mnemonic + passphrase wrapping)
bip39 = { version = "2", default-features = false, features = ["std", "zeroize"] }
secrecy = "0.10"
```

- [ ] **Step 2: Modify `crates/harmony-owner/Cargo.toml` features and deps**

Replace the existing `[features]` block and append optional deps:

```toml
[dependencies]
harmony-crypto = { workspace = true }
harmony-identity = { workspace = true }
ciborium = { workspace = true }
serde = { workspace = true, features = ["derive", "alloc"] }
serde_bytes = { workspace = true }
thiserror = { workspace = true }
zeroize = { workspace = true }
ed25519-dalek = { workspace = true }
x25519-dalek = { workspace = true }
rand_core = { workspace = true, features = ["getrandom"] }
hkdf = { workspace = true }
sha2 = { workspace = true }
subtle = { workspace = true }

# Recovery-only — gated by the `recovery` feature.
bip39 = { workspace = true, optional = true }
argon2 = { workspace = true, optional = true }
chacha20poly1305 = { workspace = true, optional = true }
secrecy = { workspace = true, optional = true }

[dev-dependencies]
hex = { workspace = true }
rand = { workspace = true }

[features]
default = ["recovery"]
# Identity backup/restore: BIP39-24 mnemonic + Argon2id-XChaCha20-Poly1305
# encrypted file. Default-on for typical consumers (harmony-client). Disable
# via `default-features = false` to skip the heavy crypto deps.
recovery = ["dep:bip39", "dep:argon2", "dep:chacha20poly1305", "dep:secrecy"]
# Test-only: exposes deterministic-input encrypt helper for wire-format
# fixture generation. NOT default-on. Implies `recovery`.
test-fixtures = ["recovery"]

# The wire-format fixture integration test only builds when test-fixtures
# is enabled, so it can call the deterministic encrypt helper.
[[test]]
name = "recovery_wire_format_fixture"
required-features = ["test-fixtures"]
```

Note `subtle` was added as a non-optional dep — `harmony-crypto` already pulls it in transitively but we want a direct dep for the constant-time comparison in cross-encoding equivalence tests. (If `subtle` is missing from `[workspace.dependencies]`, add it: `subtle = { version = "2", default-features = false }`.)

- [ ] **Step 3: Verify build with both feature configurations**

```bash
cargo build -p harmony-owner
cargo build -p harmony-owner --no-default-features
cargo build -p harmony-owner --features test-fixtures
```

Expected: all three succeed. The `--no-default-features` build should NOT pull in bip39/argon2/chacha20poly1305/secrecy (sanity-check via `cargo tree -p harmony-owner --no-default-features` — those crates should NOT appear).

- [ ] **Step 4: Commit**

```bash
git add Cargo.toml crates/harmony-owner/Cargo.toml
git commit -m "build(owner): add recovery + test-fixtures Cargo features for ZEB-175"
```

---

### Task 2: Module skeleton + lib.rs registration

**Files:**
- Create: `crates/harmony-owner/src/recovery/mod.rs`
- Create: `crates/harmony-owner/src/recovery/error.rs`
- Create: `crates/harmony-owner/src/recovery/wire.rs`
- Create: `crates/harmony-owner/src/recovery/mnemonic.rs`
- Create: `crates/harmony-owner/src/recovery/encrypted_file.rs`
- Modify: `crates/harmony-owner/src/lib.rs`

- [ ] **Step 1: Create empty `recovery/mod.rs`**

```rust
//! Identity backup / restore — BIP39-24 mnemonic and encrypted-file
//! encodings of the master `RecoveryArtifact` seed.
//!
//! See `docs/superpowers/specs/2026-04-26-identity-backup-restore-design.md`
//! for the design and `docs/superpowers/plans/2026-04-26-identity-backup-restore.md`
//! for the build sequence.

pub mod error;
pub mod wire;
pub mod mnemonic;
pub mod encrypted_file;

pub use error::RecoveryError;
```

- [ ] **Step 2: Create empty stub files**

Each of `error.rs`, `wire.rs`, `mnemonic.rs`, `encrypted_file.rs` starts with just a module-level doc comment so the crate compiles:

```rust
// crates/harmony-owner/src/recovery/error.rs
//! `RecoveryError` enum — see plan Task 3.
```

```rust
// crates/harmony-owner/src/recovery/wire.rs
//! Encrypted-file wire format constants and header parsing — see plan Task 4.
```

```rust
// crates/harmony-owner/src/recovery/mnemonic.rs
//! BIP39-24 mnemonic encode/decode — see plan Tasks 5–7.
```

```rust
// crates/harmony-owner/src/recovery/encrypted_file.rs
//! Argon2id + XChaCha20-Poly1305 encrypted-file encode/decode — see plan Tasks 8–12.
```

- [ ] **Step 3: Add module registration to `lib.rs`**

After the existing `pub mod trust;` line (around line 40), add:

```rust
#[cfg(feature = "recovery")]
pub mod recovery;
```

- [ ] **Step 4: Verify both feature configurations still compile**

```bash
cargo build -p harmony-owner
cargo build -p harmony-owner --no-default-features
```

Expected: both succeed. With `--no-default-features` the `recovery` module isn't compiled at all.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-owner/src/lib.rs crates/harmony-owner/src/recovery/
git commit -m "feat(owner): scaffold recovery module under feature gate (ZEB-175)"
```

---

### Task 3: `RecoveryError` enum

**Files:**
- Modify: `crates/harmony-owner/src/recovery/error.rs`

- [ ] **Step 1: Write failing test for the Display format of every variant**

In `error.rs`:

```rust
//! `RecoveryError` enum — operator-readable failure modes for mnemonic and
//! encrypted-file encode/decode paths. Sibling to `OwnerError` (NOT folded
//! into it; recovery is a self-contained concern).

#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    // ── Mnemonic parse ─────────────────────────────────────────────
    #[error("expected 24 BIP39 words, got {0}")]
    WrongWordCount(usize),

    #[error("unknown word at position {position}: {word:?}")]
    UnknownWord { position: usize, word: String },

    #[error("mnemonic checksum mismatch — likely a typo somewhere in the 24 words")]
    BadChecksum,

    #[error("mnemonic contains non-ASCII characters; BIP39 English wordlist is ASCII-only")]
    NonAsciiInput,

    // ── Encrypted-file decode ──────────────────────────────────────
    #[error("recovery file is too small ({0} bytes; minimum 69)")]
    TooSmall(usize),

    #[error("recovery file is too large ({0} bytes; maximum 1024)")]
    TooLarge(usize),

    #[error("not a harmony recovery file (magic mismatch)")]
    UnrecognizedFormat,

    #[error("recovery file format version {0:#x} is not supported by this build")]
    UnsupportedVersion(u8),

    #[error("recovery file uses unsupported KDF id {0:#x} or non-standard parameters")]
    UnsupportedKdfParams(u8),

    #[error("wrong passphrase or corrupted recovery file (AEAD tag rejected)")]
    WrongPassphraseOrCorrupt,

    #[error("recovery file payload could not be decoded")]
    PayloadDecodeFailed,

    #[error("recovery file payload has unexpected format string {found:?}; expected {expected:?}")]
    UnexpectedPayloadFormat { found: String, expected: &'static str },

    // ── Encrypted-file encode ──────────────────────────────────────
    #[error("comment is {actual} bytes; max allowed is {max}")]
    CommentTooLong { actual: usize, max: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages_render_expected_text() {
        assert_eq!(
            RecoveryError::WrongWordCount(13).to_string(),
            "expected 24 BIP39 words, got 13"
        );
        assert_eq!(
            RecoveryError::UnknownWord { position: 7, word: "harmonny".into() }.to_string(),
            r#"unknown word at position 7: "harmonny""#
        );
        assert_eq!(
            RecoveryError::BadChecksum.to_string(),
            "mnemonic checksum mismatch — likely a typo somewhere in the 24 words"
        );
        assert_eq!(
            RecoveryError::TooSmall(50).to_string(),
            "recovery file is too small (50 bytes; minimum 69)"
        );
        assert_eq!(
            RecoveryError::UnsupportedVersion(0x02).to_string(),
            "recovery file format version 0x2 is not supported by this build"
        );
        assert_eq!(
            RecoveryError::UnexpectedPayloadFormat {
                found: "harmony-foo-v9".into(),
                expected: "harmony-owner-recovery-v1",
            }.to_string(),
            r#"recovery file payload has unexpected format string "harmony-foo-v9"; expected "harmony-owner-recovery-v1""#
        );
        assert_eq!(
            RecoveryError::CommentTooLong { actual: 300, max: 256 }.to_string(),
            "comment is 300 bytes; max allowed is 256"
        );
    }
}
```

- [ ] **Step 2: Run test, expect PASS**

```bash
cargo test -p harmony-owner --lib recovery::error
```

Expected: PASS (the impl IS the test target — both land together because thiserror generates Display from the attribute).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/error.rs
git commit -m "feat(owner): RecoveryError enum with Display tests (ZEB-175)"
```

---

### Task 4: Wire-format constants + header serialize/parse

**Files:**
- Modify: `crates/harmony-owner/src/recovery/wire.rs`

- [ ] **Step 1: Write failing tests for header round-trip and validation**

In `wire.rs`:

```rust
//! Encrypted-file wire format: 13-byte header bound as AAD, salt, nonce,
//! ciphertext, Poly1305 tag. The header layout matches the structure of
//! ZEB-174's `identity.enc` but uses magic `HRMR` instead of `HRMI` so
//! the two file types can never be confused.
//!
//! Layout:
//! ```text
//! offset  size  field
//! 0       4     magic = b"HRMR"
//! 4       1     format_version = 0x01
//! 5       1     kdf_id = 0x01 (Argon2id)
//! 6       4     kdf_m_kib (u32 BE) = 65536
//! 10      2     kdf_t (u16 BE) = 3
//! 12      1     kdf_p (u8) = 1
//! 13      16    salt
//! 29      24    nonce (XChaCha20-Poly1305 needs 24)
//! 53      var   ciphertext
//! end-16  16    Poly1305 tag
//! ```

use crate::recovery::error::RecoveryError;

pub const MAGIC: &[u8; 4] = b"HRMR";
pub const FORMAT_VERSION: u8 = 0x01;
pub const KDF_ID_ARGON2ID: u8 = 0x01;
pub const KDF_M_KIB: u32 = 65536;
pub const KDF_T: u16 = 3;
pub const KDF_P: u8 = 1;
pub const KDF_OUT_LEN: usize = 32;

pub const HEADER_LEN: usize = 13;
pub const SALT_LEN: usize = 16;
pub const NONCE_LEN: usize = 24;
pub const TAG_LEN: usize = 16;

/// Wire-layer minimum: header + salt + nonce + 0-byte ciphertext + tag.
/// A valid file in practice is larger because the smallest CBOR-encoded
/// `RecoveryFileBody` exceeds zero bytes; this is the parse-time guard.
pub const MIN_FILE_LEN: usize = HEADER_LEN + SALT_LEN + NONCE_LEN + TAG_LEN; // 69
pub const MAX_FILE_LEN: usize = 1024;

/// Build the 13-byte header bytes. Same content every time (no per-call
/// variability) — this is a constant-output helper that simply serializes
/// the locked KDF parameters.
pub fn serialize_header() -> [u8; HEADER_LEN] {
    let mut out = [0u8; HEADER_LEN];
    out[0..4].copy_from_slice(MAGIC);
    out[4] = FORMAT_VERSION;
    out[5] = KDF_ID_ARGON2ID;
    out[6..10].copy_from_slice(&KDF_M_KIB.to_be_bytes());
    out[10..12].copy_from_slice(&KDF_T.to_be_bytes());
    out[12] = KDF_P;
    out
}

/// Parse and validate a 13-byte header slice. STRICT EQUALITY on KDF
/// parameters: any deviation from the locked values returns
/// `UnsupportedKdfParams` BEFORE the (potentially attacker-controlled) bytes
/// are passed to Argon2id. This prevents a CPU/memory DoS via an
/// adversarially-large `kdf_m_kib`.
pub fn parse_header(bytes: &[u8]) -> Result<(), RecoveryError> {
    if bytes.len() < HEADER_LEN {
        return Err(RecoveryError::TooSmall(bytes.len()));
    }
    if &bytes[0..4] != MAGIC {
        return Err(RecoveryError::UnrecognizedFormat);
    }
    if bytes[4] != FORMAT_VERSION {
        return Err(RecoveryError::UnsupportedVersion(bytes[4]));
    }
    if bytes[5] != KDF_ID_ARGON2ID {
        return Err(RecoveryError::UnsupportedKdfParams(bytes[5]));
    }
    let m_kib = u32::from_be_bytes(bytes[6..10].try_into().unwrap());
    let t = u16::from_be_bytes(bytes[10..12].try_into().unwrap());
    let p = bytes[12];
    if m_kib != KDF_M_KIB || t != KDF_T || p != KDF_P {
        return Err(RecoveryError::UnsupportedKdfParams(bytes[5]));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_size_constants_add_up() {
        assert_eq!(HEADER_LEN, 13);
        assert_eq!(MIN_FILE_LEN, 13 + 16 + 24 + 16);
        assert_eq!(MIN_FILE_LEN, 69);
    }

    #[test]
    fn serialize_header_is_deterministic() {
        assert_eq!(serialize_header(), serialize_header());
    }

    #[test]
    fn serialize_header_layout_is_exact() {
        let h = serialize_header();
        assert_eq!(&h[0..4], b"HRMR");
        assert_eq!(h[4], 0x01);
        assert_eq!(h[5], 0x01);
        assert_eq!(u32::from_be_bytes(h[6..10].try_into().unwrap()), 65536);
        assert_eq!(u16::from_be_bytes(h[10..12].try_into().unwrap()), 3);
        assert_eq!(h[12], 1);
    }

    #[test]
    fn parse_header_round_trips() {
        assert!(parse_header(&serialize_header()).is_ok());
    }

    #[test]
    fn parse_header_rejects_short() {
        let err = parse_header(&[0u8; 5]).unwrap_err();
        assert!(matches!(err, RecoveryError::TooSmall(5)));
    }

    #[test]
    fn parse_header_rejects_wrong_magic() {
        let mut h = serialize_header();
        h[0] = b'X';
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnrecognizedFormat));
    }

    #[test]
    fn parse_header_rejects_wrong_version() {
        let mut h = serialize_header();
        h[4] = 0x02;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedVersion(0x02)));
    }

    #[test]
    fn parse_header_rejects_wrong_kdf_id() {
        let mut h = serialize_header();
        h[5] = 0x99;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x99)));
    }

    #[test]
    fn parse_header_rejects_wrong_m_kib() {
        let mut h = serialize_header();
        h[6..10].copy_from_slice(&32768u32.to_be_bytes());
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x01)));
    }

    #[test]
    fn parse_header_rejects_wrong_t() {
        let mut h = serialize_header();
        h[10..12].copy_from_slice(&7u16.to_be_bytes());
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x01)));
    }

    #[test]
    fn parse_header_rejects_wrong_p() {
        let mut h = serialize_header();
        h[12] = 4;
        let err = parse_header(&h).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x01)));
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-owner --lib recovery::wire
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/wire.rs
git commit -m "feat(owner): recovery wire-format constants + header parse/serialize (ZEB-175)"
```

---

### Task 5: Mnemonic encode (`to_mnemonic_inner`)

**Files:**
- Modify: `crates/harmony-owner/src/recovery/mnemonic.rs`

- [ ] **Step 1: Write failing test for encode + BIP39 known vector**

```rust
//! BIP39-24 mnemonic encode/decode for the 32-byte recovery seed.
//!
//! Uses the `bip39` crate (v2) with the English wordlist. The 32-byte seed
//! maps to exactly 24 BIP39 words: 256 bits payload + 8-bit checksum
//! (`SHA256(seed)[0]`) split into 24 × 11-bit groups. No PBKDF2 expansion
//! (the seed is the raw 32 bytes, not a BIP39-derived 64-byte expansion).

use crate::recovery::error::RecoveryError;

/// Encode a 32-byte seed as a 24-word BIP39 mnemonic (English wordlist).
pub(crate) fn to_mnemonic_inner(seed: &[u8; 32]) -> String {
    bip39::Mnemonic::from_entropy(seed)
        .expect("32 bytes is always valid BIP39-24 entropy")
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_produces_24_words() {
        let seed = [0u8; 32];
        let m = to_mnemonic_inner(&seed);
        let words: Vec<&str> = m.split_whitespace().collect();
        assert_eq!(words.len(), 24);
    }

    /// BIP39 canonical test vector: 32 bytes of zeroes maps to a known
    /// 24-word mnemonic starting with "abandon ... art". This locks
    /// interop with external BIP39 implementations.
    #[test]
    fn bip39_test_vector_all_zero_seed() {
        let seed = [0u8; 32];
        let m = to_mnemonic_inner(&seed);
        assert_eq!(
            m,
            "abandon abandon abandon abandon abandon abandon abandon abandon \
             abandon abandon abandon abandon abandon abandon abandon abandon \
             abandon abandon abandon abandon abandon abandon abandon art"
        );
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-owner --lib recovery::mnemonic
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/mnemonic.rs
git commit -m "feat(owner): mnemonic encode (to_mnemonic_inner) + BIP39 test vector (ZEB-175)"
```

---

### Task 6: Mnemonic decode happy path + leniency

**Files:**
- Modify: `crates/harmony-owner/src/recovery/mnemonic.rs`

- [ ] **Step 1: Write failing tests for round-trip and leniency rules**

Append to `recovery/mnemonic.rs`:

```rust
/// Parse a 24-word BIP39 mnemonic (English wordlist) into a 32-byte seed.
///
/// Leniency rules:
/// - Whitespace-tolerant: any run of whitespace collapses to a single space.
/// - Case-insensitive: lowercased before wordlist lookup.
/// - NO Unicode normalization: non-ASCII input is rejected outright.
///
/// Errors map explicitly to `RecoveryError::{NonAsciiInput, WrongWordCount,
/// UnknownWord, BadChecksum}`. We do not blanket-`From<bip39::Error>` so a
/// future bip39 crate update cannot widen our error contract.
pub(crate) fn from_mnemonic_inner(s: &str) -> Result<[u8; 32], RecoveryError> {
    if !s.is_ascii() {
        return Err(RecoveryError::NonAsciiInput);
    }
    // Whitespace normalize + lowercase.
    let normalized = s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase();
    let words: Vec<&str> = normalized.split(' ').collect();
    if words.len() != 24 {
        return Err(RecoveryError::WrongWordCount(words.len()));
    }
    let mnemonic = bip39::Mnemonic::parse_in_normalized(
        bip39::Language::English,
        &normalized,
    )
    .map_err(map_bip39_err_with_words(&words))?;
    let entropy = mnemonic.to_entropy();
    debug_assert_eq!(entropy.len(), 32, "BIP39-24 always decodes to 32 bytes");
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&entropy);
    Ok(seed)
}

fn map_bip39_err_with_words(words: &[&str]) -> impl FnOnce(bip39::Error) -> RecoveryError + '_ {
    move |e| match e {
        bip39::Error::UnknownWord(idx) => RecoveryError::UnknownWord {
            position: idx + 1, // 1-indexed for human display
            word: words.get(idx).map(|s| s.to_string()).unwrap_or_default(),
        },
        bip39::Error::InvalidChecksum => RecoveryError::BadChecksum,
        bip39::Error::BadWordCount(n) => RecoveryError::WrongWordCount(n),
        // Other variants (BadEntropyBitCount, AmbiguousLanguages, etc.) are
        // unreachable for our 24-word English-only path. Map to BadChecksum
        // as a conservative fallback rather than panicking.
        _ => RecoveryError::BadChecksum,
    }
}

#[cfg(test)]
mod decode_tests {
    use super::*;

    #[test]
    fn seed_to_mnemonic_roundtrips() {
        let mut seed = [0u8; 32];
        for (i, b) in seed.iter_mut().enumerate() {
            *b = (i * 7) as u8;
        }
        let m = to_mnemonic_inner(&seed);
        let restored = from_mnemonic_inner(&m).unwrap();
        assert_eq!(restored, seed);
    }

    #[test]
    fn case_insensitive_input_succeeds() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        let upper = m.to_uppercase();
        assert!(from_mnemonic_inner(&upper).is_ok());
        let mixed: String = m.chars().enumerate()
            .map(|(i, c)| if i % 2 == 0 { c.to_ascii_uppercase() } else { c })
            .collect();
        assert!(from_mnemonic_inner(&mixed).is_ok());
    }

    #[test]
    fn whitespace_normalization() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        // Collapse all spaces to triples-of-various-whitespace.
        let weird = m.replace(' ', "\t \n  ");
        assert!(from_mnemonic_inner(&weird).is_ok());
        // Leading and trailing whitespace
        let padded = format!("   \n{m}\t\t   ");
        assert!(from_mnemonic_inner(&padded).is_ok());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-owner --lib recovery::mnemonic
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/mnemonic.rs
git commit -m "feat(owner): mnemonic decode + case/whitespace leniency (ZEB-175)"
```

---

### Task 7: Mnemonic decode error paths

**Files:**
- Modify: `crates/harmony-owner/src/recovery/mnemonic.rs`

- [ ] **Step 1: Append error-path tests**

In `recovery/mnemonic.rs`'s `decode_tests` mod:

```rust
    #[test]
    fn non_ascii_rejected() {
        // Cyrillic 'a' inside one of the words.
        let m = to_mnemonic_inner(&[0u8; 32]);
        let mut chars: Vec<char> = m.chars().collect();
        chars[0] = 'а'; // U+0430 CYRILLIC SMALL LETTER A
        let bad: String = chars.into_iter().collect();
        assert!(matches!(
            from_mnemonic_inner(&bad).unwrap_err(),
            RecoveryError::NonAsciiInput
        ));
    }

    #[test]
    fn wrong_word_count_rejected_too_few() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        let words: Vec<&str> = m.split_whitespace().take(23).collect();
        let short = words.join(" ");
        let err = from_mnemonic_inner(&short).unwrap_err();
        assert!(matches!(err, RecoveryError::WrongWordCount(23)), "got: {err}");
    }

    #[test]
    fn wrong_word_count_rejected_too_many() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        let long = format!("{m} extra");
        let err = from_mnemonic_inner(&long).unwrap_err();
        assert!(matches!(err, RecoveryError::WrongWordCount(25)), "got: {err}");
    }

    #[test]
    fn unknown_word_reports_position() {
        let m = to_mnemonic_inner(&[0u8; 32]);
        let words: Vec<&str> = m.split_whitespace().collect();
        // Replace the 7th word (index 6) with a non-wordlist string.
        let mut mutated: Vec<String> = words.into_iter().map(String::from).collect();
        mutated[6] = "harmonny".into();
        let bad = mutated.join(" ");
        let err = from_mnemonic_inner(&bad).unwrap_err();
        match err {
            RecoveryError::UnknownWord { position, word } => {
                assert_eq!(position, 7); // 1-indexed
                assert_eq!(word, "harmonny");
            }
            other => panic!("expected UnknownWord, got: {other}"),
        }
    }

    #[test]
    fn bad_checksum_rejected() {
        // Swap two valid words to break the checksum without introducing
        // unknown words. The all-zero seed yields "abandon × 23 + art" —
        // swap "art" for another valid word like "ability" to break checksum.
        let m = to_mnemonic_inner(&[0u8; 32]);
        let bad = m.replace("art", "ability");
        let err = from_mnemonic_inner(&bad).unwrap_err();
        assert!(matches!(err, RecoveryError::BadChecksum), "got: {err}");
    }
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-owner --lib recovery::mnemonic
```

Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/mnemonic.rs
git commit -m "feat(owner): mnemonic decode error paths — non-ASCII, word count, unknown word, bad checksum (ZEB-175)"
```

---

### Task 8: CBOR payload struct (`RecoveryFileBody`)

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`

- [ ] **Step 1: Define the body struct + canonical-CBOR round-trip test**

```rust
//! Argon2id + XChaCha20-Poly1305 encrypted-file encode/decode for the
//! 32-byte recovery seed plus optional metadata. Format spec lives in
//! `crate::recovery::wire`.

use crate::recovery::error::RecoveryError;
use serde::{Deserialize, Serialize};

pub const FORMAT_STRING: &str = "harmony-owner-recovery-v1";
pub const MAX_COMMENT_LEN: usize = 256;

/// CBOR-encoded plaintext payload inside the AEAD ciphertext. The `format`
/// string is defense-in-depth: even though Poly1305 already proves the
/// payload was produced by someone with the passphrase, validating the
/// format string after decryption protects against future format
/// bifurcations being silently accepted by older parsers.
///
/// `pub` (not `pub(crate)`) because `encrypt_with_params_for_test` accepts
/// `&RecoveryFileBody` as a parameter, and that helper is callable from
/// integration tests when the `test-fixtures` feature is enabled. The
/// struct is only visible at all when the `recovery` feature is on
/// (the entire module is gated).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryFileBody {
    pub format: String,
    #[serde(with = "serde_bytes")]
    pub seed: [u8; 32],
    pub mint_at: Option<u64>,
    pub comment: Option<String>,
}

#[cfg(test)]
mod body_tests {
    use super::*;

    #[test]
    fn cbor_round_trip_minimal() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: None,
            comment: None,
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }

    #[test]
    fn cbor_round_trip_full() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("primary owner".into()),
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p harmony-owner --lib recovery::encrypted_file
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/encrypted_file.rs
git commit -m "feat(owner): RecoveryFileBody CBOR struct + roundtrip tests (ZEB-175)"
```

---

### Task 9: Deterministic encrypt helper (test-fixtures gated)

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`

- [ ] **Step 1: Add internal encrypt core + test-fixtures-gated public helper, with determinism test**

Append to `encrypted_file.rs`:

```rust
use crate::recovery::wire::{
    serialize_header, HEADER_LEN, KDF_M_KIB, KDF_OUT_LEN, KDF_P, KDF_T, NONCE_LEN, SALT_LEN, TAG_LEN,
};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
use secrecy::{ExposeSecret, SecretString};
use zeroize::Zeroizing;

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
    let key = derive_key_argon2id(passphrase, salt);
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let ct = cipher
        .encrypt(
            XNonce::from_slice(nonce),
            Payload { msg: plaintext, aad: &header },
        )
        .expect("XChaCha20-Poly1305 encryption is infallible for in-bounds inputs");
    let mut out = Vec::with_capacity(HEADER_LEN + SALT_LEN + NONCE_LEN + ct.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(salt);
    out.extend_from_slice(nonce);
    out.extend_from_slice(&ct);
    out
}

fn derive_key_argon2id(
    passphrase: &SecretString,
    salt: &[u8; SALT_LEN],
) -> Zeroizing<[u8; KDF_OUT_LEN]> {
    use argon2::{Algorithm, Argon2, Params, Version};
    let params = Params::new(KDF_M_KIB, KDF_T as u32, KDF_P as u32, Some(KDF_OUT_LEN))
        .expect("Argon2 params are constants known to validate");
    let kdf = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out: Zeroizing<[u8; KDF_OUT_LEN]> = Zeroizing::new([0u8; KDF_OUT_LEN]);
    kdf.hash_password_into(passphrase.expose_secret().as_bytes(), salt, out.as_mut())
        .expect("Argon2 derivation is infallible for valid params");
    out
}

/// Test-only deterministic encrypt: caller supplies salt and nonce so the
/// output is byte-stable for fixture pinning. Behind `test-fixtures` so it
/// is NOT part of the production API surface.
#[cfg(feature = "test-fixtures")]
pub fn encrypt_with_params_for_test(
    passphrase: &SecretString,
    body: &RecoveryFileBody,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
) -> Vec<u8> {
    let plaintext = crate::cbor::to_canonical(body).expect("body always encodes");
    encrypt_core(passphrase, &plaintext, salt, nonce)
}

#[cfg(all(test, feature = "test-fixtures"))]
mod fixture_helper_tests {
    use super::*;

    #[test]
    fn deterministic_with_fixed_inputs() {
        let pass = SecretString::from("test-passphrase".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [7u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("fixture".into()),
        };
        let salt = [0xAB; SALT_LEN];
        let nonce = [0xCD; NONCE_LEN];
        let a = encrypt_with_params_for_test(&pass, &body, &salt, &nonce);
        let b = encrypt_with_params_for_test(&pass, &body, &salt, &nonce);
        assert_eq!(a, b, "deterministic inputs must produce identical output");
        assert!(a.len() >= MIN_FILE_LEN_USE);
    }
    const MIN_FILE_LEN_USE: usize = HEADER_LEN + SALT_LEN + NONCE_LEN + TAG_LEN;
}
```

- [ ] **Step 2: Run with test-fixtures feature enabled**

```bash
cargo test -p harmony-owner --features test-fixtures --lib recovery::encrypted_file
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/encrypted_file.rs
git commit -m "feat(owner): encrypt_core + test-fixtures-gated deterministic helper (ZEB-175)"
```

---

### Task 10: Decrypt + happy-path round trip

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`

- [ ] **Step 1: Implement `decrypt_inner` + write the round-trip test**

Append:

```rust
use crate::recovery::wire::{parse_header, MAX_FILE_LEN, MIN_FILE_LEN};

/// Decrypt + parse a recovery-file byte slice into the seed and metadata.
/// Returns the raw seed (caller wraps it back into a `RecoveryArtifact`)
/// and the metadata fields.
pub(crate) fn decrypt_inner(
    bytes: &[u8],
    passphrase: &SecretString,
) -> Result<([u8; 32], Option<u64>, Option<String>), RecoveryError> {
    if bytes.len() < MIN_FILE_LEN {
        return Err(RecoveryError::TooSmall(bytes.len()));
    }
    if bytes.len() > MAX_FILE_LEN {
        return Err(RecoveryError::TooLarge(bytes.len()));
    }
    parse_header(&bytes[..HEADER_LEN])?;

    let salt: &[u8; SALT_LEN] = bytes[HEADER_LEN..HEADER_LEN + SALT_LEN]
        .try_into()
        .expect("range matches SALT_LEN");
    let nonce: &[u8; NONCE_LEN] = bytes[HEADER_LEN + SALT_LEN..HEADER_LEN + SALT_LEN + NONCE_LEN]
        .try_into()
        .expect("range matches NONCE_LEN");
    let ciphertext_and_tag = &bytes[HEADER_LEN + SALT_LEN + NONCE_LEN..];

    let header = serialize_header();
    let key = derive_key_argon2id(passphrase, salt);
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let plaintext_vec = cipher
        .decrypt(
            XNonce::from_slice(nonce),
            Payload { msg: ciphertext_and_tag, aad: &header },
        )
        .map_err(|_| RecoveryError::WrongPassphraseOrCorrupt)?;
    // Wrap plaintext in Zeroizing so it's wiped after we finish parsing.
    let plaintext: Zeroizing<Vec<u8>> = Zeroizing::new(plaintext_vec);

    let body: RecoveryFileBody = ciborium::de::from_reader(&plaintext[..])
        .map_err(|_| RecoveryError::PayloadDecodeFailed)?;

    if body.format != FORMAT_STRING {
        return Err(RecoveryError::UnexpectedPayloadFormat {
            found: body.format,
            expected: FORMAT_STRING,
        });
    }
    if let Some(c) = body.comment.as_ref() {
        if c.len() > MAX_COMMENT_LEN {
            return Err(RecoveryError::CommentTooLong {
                actual: c.len(),
                max: MAX_COMMENT_LEN,
            });
        }
    }
    Ok((body.seed, body.mint_at, body.comment))
}

#[cfg(all(test, feature = "test-fixtures"))]
mod decrypt_tests {
    use super::*;

    #[test]
    fn round_trip_minimal() {
        let pass = SecretString::from("rt-test".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [9u8; 32],
            mint_at: None,
            comment: None,
        };
        let bytes = encrypt_with_params_for_test(&pass, &body, &[1u8; SALT_LEN], &[2u8; NONCE_LEN]);
        let (seed, mint_at, comment) = decrypt_inner(&bytes, &pass).unwrap();
        assert_eq!(seed, [9u8; 32]);
        assert_eq!(mint_at, None);
        assert_eq!(comment, None);
    }

    #[test]
    fn round_trip_with_full_metadata() {
        let pass = SecretString::from("rt-full".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [11u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("primary owner".into()),
        };
        let bytes = encrypt_with_params_for_test(&pass, &body, &[3u8; SALT_LEN], &[4u8; NONCE_LEN]);
        let (seed, mint_at, comment) = decrypt_inner(&bytes, &pass).unwrap();
        assert_eq!(seed, [11u8; 32]);
        assert_eq!(mint_at, Some(1_700_000_000));
        assert_eq!(comment.as_deref(), Some("primary owner"));
    }
}
```

- [ ] **Step 2: Run with test-fixtures feature**

```bash
cargo test -p harmony-owner --features test-fixtures --lib recovery::encrypted_file
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/encrypted_file.rs
git commit -m "feat(owner): decrypt_inner + happy-path roundtrip tests (ZEB-175)"
```

---

### Task 11: Production encrypt (random salt+nonce) + comment cap

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`

- [ ] **Step 1: Add `encrypt_inner` + comment-cap tests**

Append:

```rust
use rand_core::{OsRng, RngCore};

/// Production encrypt entry point: random salt + nonce per call. The body
/// is built by the caller from the seed + metadata.
pub(crate) fn encrypt_inner(
    passphrase: &SecretString,
    seed: &[u8; 32],
    mint_at: Option<u64>,
    comment: Option<String>,
) -> Result<Vec<u8>, RecoveryError> {
    if let Some(c) = comment.as_ref() {
        if c.len() > MAX_COMMENT_LEN {
            return Err(RecoveryError::CommentTooLong {
                actual: c.len(),
                max: MAX_COMMENT_LEN,
            });
        }
    }
    let body = RecoveryFileBody {
        format: FORMAT_STRING.into(),
        seed: *seed,
        mint_at,
        comment,
    };
    let plaintext: Zeroizing<Vec<u8>> = Zeroizing::new(
        crate::cbor::to_canonical(&body).expect("body always encodes"),
    );
    let mut salt = [0u8; SALT_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce);
    Ok(encrypt_core(passphrase, &plaintext, &salt, &nonce))
}

#[cfg(test)]
mod prod_encrypt_tests {
    use super::*;

    #[test]
    fn salt_rotates_per_encode() {
        let pass = SecretString::from("salt-rot".to_string());
        let a = encrypt_inner(&pass, &[5u8; 32], None, None).unwrap();
        let b = encrypt_inner(&pass, &[5u8; 32], None, None).unwrap();
        // Same payload, fresh salt + nonce → different ciphertexts.
        assert_ne!(a, b, "salt + nonce regen must produce different bytes");
        // Salts (offset HEADER_LEN..HEADER_LEN+SALT_LEN) must differ.
        assert_ne!(
            &a[HEADER_LEN..HEADER_LEN + SALT_LEN],
            &b[HEADER_LEN..HEADER_LEN + SALT_LEN]
        );
    }

    #[test]
    fn comment_at_max_length_succeeds() {
        let pass = SecretString::from("max-len".to_string());
        let max_comment = "a".repeat(MAX_COMMENT_LEN);
        let r = encrypt_inner(&pass, &[5u8; 32], None, Some(max_comment));
        assert!(r.is_ok());
    }

    #[test]
    fn comment_over_max_fails_at_encode() {
        let pass = SecretString::from("over-len".to_string());
        let too_long = "a".repeat(MAX_COMMENT_LEN + 1);
        let err = encrypt_inner(&pass, &[5u8; 32], None, Some(too_long)).unwrap_err();
        assert!(matches!(
            err,
            RecoveryError::CommentTooLong { actual: 257, max: 256 }
        ));
    }
}
```

- [ ] **Step 2: Run**

```bash
cargo test -p harmony-owner --lib recovery::encrypted_file
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/encrypted_file.rs
git commit -m "feat(owner): production encrypt_inner with random salt+nonce + comment cap (ZEB-175)"
```

---

### Task 12: Decrypt error paths

**Files:**
- Modify: `crates/harmony-owner/src/recovery/encrypted_file.rs`

- [ ] **Step 1: Append negative-path tests**

In the `decrypt_tests` module (the one gated by `#[cfg(all(test, feature = "test-fixtures"))]`), append:

```rust
    fn fixture_bytes() -> Vec<u8> {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [13u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("neg".into()),
        };
        encrypt_with_params_for_test(
            &SecretString::from("correct".to_string()),
            &body,
            &[7u8; SALT_LEN],
            &[8u8; NONCE_LEN],
        )
    }

    #[test]
    fn wrong_passphrase_fails_aead() {
        let bytes = fixture_bytes();
        let err = decrypt_inner(&bytes, &SecretString::from("wrong".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::WrongPassphraseOrCorrupt));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let mut bytes = fixture_bytes();
        let len = bytes.len();
        // Flip a byte in the middle of the ciphertext region.
        let ct_start = HEADER_LEN + SALT_LEN + NONCE_LEN;
        bytes[ct_start + (len - ct_start) / 2] ^= 0x01;
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::WrongPassphraseOrCorrupt));
    }

    #[test]
    fn tampered_header_fails_via_aad() {
        let mut bytes = fixture_bytes();
        // Flip the kdf_t byte AFTER serialization but ONLY in the on-disk
        // bytes; parse_header's strict check would catch this first, so we
        // test a header byte the strict check ignores. Actually all 13 header
        // bytes are strict-checked, so tampering here returns one of the
        // strict errors instead of WrongPassphraseOrCorrupt. Test via the
        // strict path:
        bytes[10] ^= 0x01; // mutates kdf_t low byte: 3 → 2
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x01)));
    }

    #[test]
    fn tampered_kdf_params_rejected_before_argon2() {
        let mut bytes = fixture_bytes();
        // Set kdf_m_kib to a small value — the strict check must reject
        // this BEFORE we run Argon2id with attacker-controlled params.
        bytes[6..10].copy_from_slice(&1024u32.to_be_bytes());
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams(0x01)));
    }

    #[test]
    fn wrong_magic_rejected() {
        let mut bytes = fixture_bytes();
        bytes[0] = b'X';
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::UnrecognizedFormat));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut bytes = fixture_bytes();
        bytes[4] = 0x02;
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedVersion(0x02)));
    }

    #[test]
    fn too_small_rejected() {
        let bytes = vec![0u8; 50];
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::TooSmall(50)));
    }

    #[test]
    fn too_large_rejected() {
        let bytes = vec![0u8; 2048];
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::TooLarge(2048)));
    }
```

- [ ] **Step 2: Run with test-fixtures feature**

```bash
cargo test -p harmony-owner --features test-fixtures --lib recovery::encrypted_file
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/encrypted_file.rs
git commit -m "feat(owner): decrypt error paths — wrong pass, tamper, magic, version, size (ZEB-175)"
```

---

### Task 13: Public API on `RecoveryArtifact` + `RestoredArtifact` / `RecoveryMetadata`

**Files:**
- Modify: `crates/harmony-owner/src/recovery/mod.rs`

- [ ] **Step 1: Define types and impl block + cross-encoding equivalence test**

Replace `recovery/mod.rs` contents:

```rust
//! Identity backup / restore — BIP39-24 mnemonic and encrypted-file
//! encodings of the master `RecoveryArtifact` seed.
//!
//! ## Threat models
//!
//! The two encodings serve distinct threat models and are independently
//! complete recovery artifacts:
//!
//! - **Mnemonic (24 BIP39 English words):** defense against complete data
//!   loss. User writes the words on paper; theft of the paper = theft of
//!   the identity. No passphrase wrap.
//! - **Encrypted file (Argon2id + XChaCha20-Poly1305):** defense against
//!   file leak. The file is portable (USB / cloud / email); without the
//!   passphrase it is useless. Both must be lost simultaneously to lose
//!   the identity.
//!
//! ## Security-critical invariants
//!
//! - The 13-byte encrypted-file header is bound as AEAD AAD: any
//!   tampering — including a downgrade attack on KDF parameters — is
//!   rejected by Poly1305 before the payload is decrypted.
//! - KDF parameters are checked for **strict equality** against the
//!   locked v1 values BEFORE Argon2id runs. Prevents a CPU/memory DoS
//!   via attacker-controlled `kdf_m_kib`.
//! - Seeds never appear in error strings or `Debug` output.
//! - Passphrases are passed as `SecretString` (auto-zeroizes on drop).

pub mod error;
pub mod wire;
pub mod mnemonic;
pub mod encrypted_file;

pub use error::RecoveryError;

use crate::lifecycle::mint::RecoveryArtifact;
use secrecy::SecretString;

/// Encode-side input / decode-side output for the encrypted-file's
/// metadata. All fields are optional and non-secret on their own.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryMetadata {
    pub mint_at: Option<u64>,
    pub comment: Option<String>,
}

/// Result of decoding an encrypted recovery file. The artifact zeroizes
/// its seed on Drop (existing behavior from `lifecycle/mint.rs`).
pub struct RestoredArtifact {
    pub artifact: RecoveryArtifact,
    pub metadata: RecoveryMetadata,
}

impl RestoredArtifact {
    /// Discard the metadata and yield just the artifact.
    pub fn into_artifact(self) -> RecoveryArtifact {
        self.artifact
    }
}

impl RecoveryArtifact {
    /// Encode this artifact's seed as a 24-word BIP39 English mnemonic.
    pub fn to_mnemonic(&self) -> String {
        mnemonic::to_mnemonic_inner(self.as_bytes())
    }

    /// Decode a 24-word BIP39 English mnemonic into a `RecoveryArtifact`.
    /// Whitespace-tolerant, case-insensitive, ASCII-only.
    pub fn from_mnemonic(s: &str) -> Result<Self, RecoveryError> {
        let seed = mnemonic::from_mnemonic_inner(s)?;
        Ok(Self::from_seed(seed))
    }

    /// Encode this artifact + metadata as a passphrase-encrypted file.
    pub fn to_encrypted_file(
        &self,
        passphrase: &SecretString,
        metadata: &RecoveryMetadata,
    ) -> Result<Vec<u8>, RecoveryError> {
        encrypted_file::encrypt_inner(
            passphrase,
            self.as_bytes(),
            metadata.mint_at,
            metadata.comment.clone(),
        )
    }

    /// Decode a passphrase-encrypted recovery file.
    pub fn from_encrypted_file(
        bytes: &[u8],
        passphrase: &SecretString,
    ) -> Result<RestoredArtifact, RecoveryError> {
        let (seed, mint_at, comment) =
            encrypted_file::decrypt_inner(bytes, passphrase)?;
        Ok(RestoredArtifact {
            artifact: RecoveryArtifact::from_seed(seed),
            metadata: RecoveryMetadata { mint_at, comment },
        })
    }
}

#[cfg(test)]
mod equivalence_tests {
    use super::*;
    use subtle::ConstantTimeEq;

    /// Both encodings of the same seed must decode to artifacts that
    /// produce the same master `identity_hash` — this is the load-bearing
    /// correctness check that mnemonic and encrypted-file backups are
    /// truly equivalent.
    #[test]
    fn mnemonic_and_encrypted_file_yield_identical_master_pubkey() {
        let original = RecoveryArtifact::from_seed([42u8; 32]);
        let original_id = original.master_pubkey_bundle().identity_hash();

        // Round-trip via mnemonic.
        let m = original.to_mnemonic();
        let from_m = RecoveryArtifact::from_mnemonic(&m).unwrap();
        let id_via_m = from_m.master_pubkey_bundle().identity_hash();

        // Round-trip via encrypted file.
        let pass = SecretString::from("equiv-test".to_string());
        let bytes = original
            .to_encrypted_file(&pass, &RecoveryMetadata::default())
            .unwrap();
        let from_f = RecoveryArtifact::from_encrypted_file(&bytes, &pass)
            .unwrap()
            .into_artifact();
        let id_via_f = from_f.master_pubkey_bundle().identity_hash();

        // Constant-time equality on the identity hashes.
        assert!(bool::from(id_via_m.ct_eq(&original_id)));
        assert!(bool::from(id_via_f.ct_eq(&original_id)));
        assert!(bool::from(id_via_m.ct_eq(&id_via_f)));
    }

    #[test]
    fn restored_artifact_into_artifact_drops_metadata() {
        let original = RecoveryArtifact::from_seed([1u8; 32]);
        let pass = SecretString::from("io-test".to_string());
        let metadata = RecoveryMetadata {
            mint_at: Some(1_700_000_000),
            comment: Some("foo".into()),
        };
        let bytes = original.to_encrypted_file(&pass, &metadata).unwrap();
        let restored = RecoveryArtifact::from_encrypted_file(&bytes, &pass).unwrap();
        assert_eq!(restored.metadata, metadata);
        let _ = restored.into_artifact(); // type compiles, metadata gone
    }
}
```

- [ ] **Step 2: Run all recovery tests**

```bash
cargo test -p harmony-owner --features test-fixtures recovery
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-owner/src/recovery/mod.rs
git commit -m "feat(owner): public RecoveryArtifact API + cross-encoding equivalence test (ZEB-175)"
```

---

### Task 14: Wire-format fixture pinning

**Files:**
- Create: `crates/harmony-owner/tests/recovery_wire_format_fixture.rs`
- Create: `crates/harmony-owner/tests/fixtures/recovery_v1.bin` (regenerated by the test on first run)

- [ ] **Step 1: Write the integration test (initially fails because fixture doesn't exist)**

```rust
//! Wire-format pin for the encrypted recovery file. Decoding the committed
//! fixture must yield the expected seed and metadata; any accidental
//! serialization change (CBOR canonicalization drift, KDF param tweak,
//! header layout shift) breaks this test loudly.
//!
//! Regenerate the fixture for an intentional format change:
//!
//!     HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1 \
//!         cargo test --features test-fixtures \
//!         --test recovery_wire_format_fixture
//!
//! Then commit the new bytes alongside the version bump.
//!
//! Requires the `test-fixtures` Cargo feature for the deterministic
//! `encrypt_with_params_for_test` helper.

use harmony_owner::lifecycle::mint::RecoveryArtifact;
use harmony_owner::recovery::encrypted_file::{
    encrypt_with_params_for_test, FORMAT_STRING, RecoveryFileBody,
};
use harmony_owner::recovery::wire::{NONCE_LEN, SALT_LEN};
use secrecy::SecretString;
use std::path::PathBuf;

const FIXTURE_REL_PATH: &str = "tests/fixtures/recovery_v1.bin";
const FIXTURE_PASSPHRASE: &str = "harmony-recovery-fixture-v1";
const FIXTURE_SEED: [u8; 32] = [
    0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef,
    0xfe, 0xdc, 0xba, 0x98, 0x76, 0x54, 0x32, 0x10,
    0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77,
    0x88, 0x99, 0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff,
];
const FIXTURE_SALT: [u8; SALT_LEN] = [0x5A; SALT_LEN];
const FIXTURE_NONCE: [u8; NONCE_LEN] = [0xA5; NONCE_LEN];
const FIXTURE_MINT_AT: u64 = 1_700_000_000;
const FIXTURE_COMMENT: &str = "ZEB-175 wire format pin v1";

fn fixture_path() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push(FIXTURE_REL_PATH);
    p
}

fn deterministic_bytes() -> Vec<u8> {
    let body = RecoveryFileBody {
        format: FORMAT_STRING.into(),
        seed: FIXTURE_SEED,
        mint_at: Some(FIXTURE_MINT_AT),
        comment: Some(FIXTURE_COMMENT.into()),
    };
    encrypt_with_params_for_test(
        &SecretString::from(FIXTURE_PASSPHRASE.to_string()),
        &body,
        &FIXTURE_SALT,
        &FIXTURE_NONCE,
    )
}

#[test]
fn wire_format_v1_pinned() {
    let path = fixture_path();
    let regen = std::env::var("HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE").is_ok();

    if regen || !path.exists() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&path, deterministic_bytes()).unwrap();
        if regen {
            // Honor the explicit regen request — pass the test, the
            // operator just told us this run is for fixture rewrite.
            return;
        }
    }

    let on_disk = std::fs::read(&path).unwrap();
    let expected = deterministic_bytes();
    assert_eq!(
        on_disk, expected,
        "fixture {FIXTURE_REL_PATH} no longer matches the deterministic encode. \
         If this was an intentional format change, regenerate via \
         HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1 and commit the new bytes."
    );

    // Round-trip decode to confirm the format-string and metadata path.
    let restored = RecoveryArtifact::from_encrypted_file(
        &on_disk,
        &SecretString::from(FIXTURE_PASSPHRASE.to_string()),
    )
    .unwrap();
    assert_eq!(restored.artifact.as_bytes(), &FIXTURE_SEED);
    assert_eq!(restored.metadata.mint_at, Some(FIXTURE_MINT_AT));
    assert_eq!(restored.metadata.comment.as_deref(), Some(FIXTURE_COMMENT));
}
```

All items referenced (`encrypt_with_params_for_test`, `FORMAT_STRING`, `RecoveryFileBody`, `NONCE_LEN`, `SALT_LEN`) are already declared `pub` in tasks 4, 8, and 9 — they're each gated by either the `recovery` feature alone (the whole module is `#[cfg(feature = "recovery")]`) or the stricter `test-fixtures` feature, so widening to `pub` doesn't leak them into builds that don't opt in.

- [ ] **Step 2: Generate the initial fixture**

```bash
cargo test -p harmony-owner --features test-fixtures \
    --test recovery_wire_format_fixture
```

First run: the fixture doesn't exist, so the test creates it AND immediately re-reads + asserts. Should pass.

- [ ] **Step 3: Verify subsequent runs are idempotent**

```bash
cargo test -p harmony-owner --features test-fixtures \
    --test recovery_wire_format_fixture
```

Second run: fixture exists, bytes match, decode round-trips. PASS.

- [ ] **Step 4: Commit fixture + test**

```bash
git add crates/harmony-owner/tests/recovery_wire_format_fixture.rs
git add crates/harmony-owner/tests/fixtures/recovery_v1.bin
git commit -m "test(owner): pin recovery wire format v1 with binary fixture (ZEB-175)"
```

---

### Task 15: rustdoc polish + final verification

**Files:**
- Modify: `crates/harmony-owner/src/lifecycle/mint.rs` (add doc reference to recovery module)
- Modify: `crates/harmony-owner/src/lib.rs` (mention recovery in crate-level doc)

- [ ] **Step 1: Add a pointer in the existing `RecoveryArtifact` doc comment**

In `crates/harmony-owner/src/lifecycle/mint.rs`, modify the `RecoveryArtifact` doc:

```rust
/// 32-byte master seed. Format BIP39-wraps to 24 mnemonic words. Drop wipes.
///
/// Encoding/decoding for portable backup is in [`crate::recovery`] (gated by
/// the default-on `recovery` Cargo feature):
///
/// - [`RecoveryArtifact::to_mnemonic`] / [`RecoveryArtifact::from_mnemonic`]
/// - [`RecoveryArtifact::to_encrypted_file`] / [`RecoveryArtifact::from_encrypted_file`]
pub struct RecoveryArtifact {
    seed: [u8; 32],
}
```

- [ ] **Step 2: Update crate-level doc in `lib.rs`**

After the existing `## Out of scope here` paragraph, append:

```rust
//! ## Recovery
//!
//! Backup and restore of the master seed (mnemonic + encrypted-file)
//! lives in [`recovery`], gated by the default-on `recovery` Cargo
//! feature. See `docs/superpowers/specs/2026-04-26-identity-backup-restore-design.md`.
```

- [ ] **Step 3: Verify rustdoc builds clean**

```bash
cargo doc -p harmony-owner --no-deps
cargo doc -p harmony-owner --no-deps --no-default-features
```

Expected: both succeed with no warnings.

- [ ] **Step 4: Final test matrix**

```bash
# Default features (recovery on, test-fixtures off): unit tests only.
cargo test -p harmony-owner

# All features: unit tests + wire-format fixture.
cargo test -p harmony-owner --all-features

# No default features: recovery module excluded, only the binding-primitive
# tests run.
cargo test -p harmony-owner --no-default-features

# Confirm recovery deps don't leak into --no-default-features build.
cargo tree -p harmony-owner --no-default-features --target current \
    | grep -E '(bip39|argon2|chacha20poly1305|secrecy)' && \
    echo "FAIL: recovery deps leaked into --no-default-features build" || \
    echo "OK: --no-default-features build does not pull recovery crates"
```

Expected: all `cargo test` invocations pass; the `cargo tree` check prints `OK: ...`.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-owner/src/lifecycle/mint.rs crates/harmony-owner/src/lib.rs
git commit -m "docs(owner): cross-link RecoveryArtifact and recovery module (ZEB-175)"
```

---

## Verification summary

After all tasks complete, the following should hold:

1. `cargo test -p harmony-owner` — all default-feature unit tests pass.
2. `cargo test -p harmony-owner --all-features` — adds the wire-format fixture test; passes.
3. `cargo test -p harmony-owner --no-default-features` — only binding-primitive tests run (recovery module is `#[cfg(feature = "recovery")]`); passes.
4. `cargo doc -p harmony-owner --no-deps` — clean, no warnings.
5. `cargo tree -p harmony-owner --no-default-features` does NOT list bip39, argon2, chacha20poly1305, or secrecy.
6. `crates/harmony-owner/tests/fixtures/recovery_v1.bin` is committed.
7. Cross-encoding equivalence holds: a seed encoded via mnemonic and decoded via `from_mnemonic` produces the same `master_pubkey_bundle().identity_hash()` as the same seed encoded via `to_encrypted_file` and decoded via `from_encrypted_file`.

## Self-review

- **Spec coverage:** Every section of `docs/superpowers/specs/2026-04-26-identity-backup-restore-design.md` (Architecture / Mnemonic / Encrypted-file / Error model / Testing) has at least one task implementing it. Definition-of-done items 1–8 in the spec all map to tasks 1, 2, 3 (DoD #6), 4–12, 13 (DoD #1), 14 (DoD #4), 15 (DoD #5–8).
- **Type consistency:** `RecoveryError`, `RecoveryArtifact`, `RestoredArtifact`, `RecoveryMetadata`, `RecoveryFileBody`, `FORMAT_STRING`, `MAX_COMMENT_LEN`, header constants — all spelled identically across tasks 3 → 13.
- **API method names:** `to_mnemonic` / `from_mnemonic` / `to_encrypted_file` / `from_encrypted_file` — used identically in spec, in task 13's impl block, and in task 14's fixture test.
- **Visibility scoping:** `RecoveryFileBody` (task 8), `encrypt_with_params_for_test` (task 9), `FORMAT_STRING` / `MAX_COMMENT_LEN` (task 8), and the wire-format constants `NONCE_LEN` / `SALT_LEN` / `HEADER_LEN` (task 4) are all declared `pub` from the start — the integration test in task 14 imports them directly. They're each behind either the `recovery` feature gate (whole module) or the stricter `test-fixtures` feature, so widening to `pub` doesn't leak them into builds that don't opt in.
- **No placeholders:** scanned and clean.

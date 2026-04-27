# Identity Backup / Restore — Library Primitives

**Date:** 2026-04-26
**Status:** Draft
**Scope:** New module `harmony-owner::recovery` (gated by Cargo feature `recovery`, default-on). Library-only — no harmony-client integration.
**Resolves:** ZEB-175 (Track A umbrella ZEB-169), library half. The harmony-client wire-up is a separate follow-on ticket to be filed once this lands.

## Overview

ZEB-173 (PR #261, merged 2026-04-26) shipped the `harmony-owner` crate with the two-tier owner→device binding model. The master key `M` is encoded as a 32-byte seed inside a `RecoveryArtifact` type that lives only in the recovery artifact and transient RAM during enrollment ceremonies. The artifact is the load-bearing primitive that turns total-device-loss from identity-loss into a recoverable event.

Today the artifact has no portable encoding — `RecoveryArtifact::as_bytes() -> &[u8; 32]` exposes the raw seed, and that's it. Without a portable representation, no human user can actually back the artifact up. This design adds two encoding/decoding pathways:

1. **BIP39-24 mnemonic.** 24 English words. The standard format users already understand from cryptocurrency wallets. The mnemonic IS the secret — knowing the words is equivalent to knowing the seed.
2. **Encrypted file.** Argon2id + XChaCha20-Poly1305 envelope around a CBOR-encoded payload (seed + optional metadata). Passphrase-protected; can be safely transported (USB, cloud, email-to-self).

Both encodings round-trip to the same `RecoveryArtifact`. They serve two distinct threat models:
- **Mnemonic** — defense against complete data loss; user writes 24 words on paper, locks paper away. Theft of the paper = theft of the identity.
- **Encrypted file** — defense against file leak; the file alone is useless without the passphrase. Both must be lost simultaneously to lose the identity.

## Goals

1. A user with a `RecoveryArtifact` can produce a 24-word BIP39 mnemonic suitable for offline backup.
2. A user with the 24-word mnemonic can reconstruct the same `RecoveryArtifact` on any device.
3. A user with a `RecoveryArtifact` and a passphrase can produce a portable encrypted file with optional metadata (mint timestamp, user-supplied comment).
4. A user with the encrypted file and the correct passphrase can reconstruct the same `RecoveryArtifact` plus the metadata that was attached at export time.
5. Tampering with any byte of the encrypted file (header, salt, nonce, ciphertext, tag) is detected and rejected with a clear error before any seed material is materialized.
6. The wire format for the encrypted file is pinned by a committed test fixture. Any accidental serialization change breaks CI loudly.
7. The recovery module is gated by a Cargo feature so downstream crates that don't need it (e.g., a future server-side verifier) can opt out of the Argon2id compile cost.

## Non-Goals

1. **Harmony-client integration.** This design is library-only. A separate ticket will wire the new APIs into harmony-client's settings UI, the resolution chain in `identity.rs`, and the keychain/encrypted backends from ZEB-174. That ticket will also handle harmony-client's migration to actually generating its identity through `harmony-owner::lifecycle::mint_owner` instead of the current device-only `NodeIdentity` flow.
2. **CAS data / mailbox / profile recovery.** ZEB-175's open question 3 asks whether recovery should restore application data too. Out of scope here — this design recovers identity material only. Application data recovery is a separate concern (and likely a separate ticket).
3. **Mnemonic passphrase ("25th word").** BIP39's optional passphrase mechanism would require changing `harmony-owner` to derive seeds via BIP39's PBKDF2 expansion rather than `SigningKey::from_bytes`. The encrypted-file mode already provides passphrase protection (with Argon2id m=64MiB, which is cryptographically stronger than BIP39's PBKDF2-2048 anyway). Two clean modes for two threat models is preferable to a third "did you remember the passphrase" decision per restore.
4. **PQ key handling.** The current master `PubKeyBundle` has `post_quantum: None` always. When PQ master keys are added (the `TODO v1.1` in `mint.rs`), they will follow the same HKDF-from-seed pattern, so the seed alone will continue to encode the full master. No format change needed today.
5. **Cloud backup destinations** (Dropbox, iCloud, etc.). User stores the encrypted file wherever they want. The library produces and consumes the bytes; transport is not its concern.
6. **Property-based / fuzz testing.** The fixture file plus negative-path unit tests cover the realistic bug surface. A `cargo fuzz` target can be added later if a corpus surfaces real bugs.
7. **Word-by-word autocomplete on mnemonic input.** That belongs in client UX, not library scope.

## Architecture

### Module layout

All under `crates/harmony-owner/src/recovery/`, gated by `#[cfg(feature = "recovery")]`:

```text
src/recovery/
  mod.rs            — public API re-exports + RestoredArtifact / RecoveryMetadata types
                       + impl RecoveryArtifact { to_mnemonic, from_mnemonic,
                                                   to_encrypted_file, from_encrypted_file }
  mnemonic.rs       — BIP39-24 encode/decode internals
  encrypted_file.rs — Argon2id + XChaCha20-Poly1305 encode/decode internals
  wire.rs           — encrypted-file wire format constants + header parsing/validation
  error.rs          — RecoveryError enum
```

The existing `RecoveryArtifact` type stays where it is (`lifecycle/mint.rs`). The recovery module is purely additive — no refactor of `mint.rs`, no change to `from_seed` / `as_bytes` / `master_signing_key` / `master_pubkey_bundle`. New methods on `RecoveryArtifact` are added via an `impl` block in `recovery/mod.rs` that's only compiled when the feature is enabled.

### Cargo feature wiring

`crates/harmony-owner/Cargo.toml`:

```toml
[features]
default = ["recovery"]
recovery = ["dep:bip39", "dep:argon2", "dep:chacha20poly1305", "dep:secrecy"]

[dependencies]
bip39             = { version = "2",    optional = true, default-features = false, features = ["std", "zeroize"] }
argon2            = { version = "0.5",  optional = true }
chacha20poly1305  = { version = "0.10", optional = true }
secrecy           = { version = "0.10", optional = true }
```

Default-on means harmony-client and other typical consumers get recovery for free. Downstream crates that genuinely don't need it (e.g., a future server-side verifier that only inspects Enrollment Certs) can `default-features = false` to skip the Argon2id compile cost.

### Public API surface

Added to `RecoveryArtifact` via `impl` block in `recovery/mod.rs`:

```rust
impl RecoveryArtifact {
    pub fn to_mnemonic(&self) -> Zeroizing<String>;
    pub fn from_mnemonic(s: &str) -> Result<Self, RecoveryError>;

    pub fn to_encrypted_file(
        &self,
        passphrase: &SecretString,
        metadata: &RecoveryMetadata,
    ) -> Result<Vec<u8>, RecoveryError>;

    pub fn from_encrypted_file(
        bytes: &[u8],
        passphrase: &SecretString,
    ) -> Result<RestoredArtifact, RecoveryError>;
}
```

Plus the supporting types:

```rust
/// Encode-side input / decode-side output for the encrypted-file's metadata.
/// All fields are optional and non-secret on their own (the seed is the only
/// secret, and it lives in RecoveryArtifact, not here).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RecoveryMetadata {
    pub mint_at: Option<u64>,         // unix seconds of owner mint, if known
    pub comment: Option<String>,       // user-supplied label, ≤ 256 bytes
}

/// Result of decoding an encrypted recovery file. The artifact zeroizes its
/// seed on Drop (existing behavior from mint.rs).
pub struct RestoredArtifact {
    pub artifact: RecoveryArtifact,
    pub metadata: RecoveryMetadata,
}

impl RestoredArtifact {
    pub fn into_artifact(self) -> RecoveryArtifact { self.artifact }
}
```

Shape choices (intentional):
- **`pub` fields on `RestoredArtifact`** so callers can destructure (`let RestoredArtifact { artifact, metadata } = ...`).
- **No `Clone` on `RestoredArtifact`** — accidental cloning would defeat single-owner zeroization.
- **No `Debug` on `RestoredArtifact`** — `RecoveryArtifact` lacks `Debug` intentionally to keep seeds out of log output; `RestoredArtifact` follows suit.
- **No equality** — comparing two restored artifacts is rare; if a caller wants to verify "do these two backup files restore the same identity," they compare `master_pubkey_bundle().identity_hash()` of each in constant time.

The mnemonic path returns just `RecoveryArtifact` because the mnemonic encoding has no header, no comment, no timestamp. The artifact alone is the full result.

## Mnemonic encoding (BIP39-24)

Use the `bip39` crate (v2) with the English wordlist.

The 32-byte seed maps to exactly 24 BIP39 words: 256 bits of payload + 8-bit checksum (`SHA256(seed)[0]`) split into 24 × 11-bit groups. This is BIP39's natural encoding for a 256-bit secret — no PBKDF2 expansion needed, no padding.

### Encode

`to_mnemonic()` wraps the seed via `bip39::Mnemonic::from_entropy(self.as_bytes())` and returns its space-joined word string.

### Decode

`from_mnemonic(s: &str)` parsing rules:

1. **Whitespace-tolerant.** Collapse any run of whitespace (including newlines, tabs) to a single space; trim leading/trailing.
2. **Case-insensitive.** Lowercase before wordlist lookup. (BIP39 English wordlist is all lowercase; users writing in caps shouldn't be punished.)
3. **No Unicode normalization.** The English wordlist is pure ASCII. Anything outside ASCII is `RecoveryError::NonAsciiInput` rather than a silently-normalized success. Mirrors ZEB-174's "no normalization on passphrases" stance.
4. **Reject any word not in the wordlist** with `RecoveryError::UnknownWord { position, word }`. Position is 1-indexed for human readability ("position 7" not "index 6").
5. **Reject wrong word count** with `RecoveryError::WrongWordCount(actual)`.
6. **Reject bad checksum** with `RecoveryError::BadChecksum`.

### Output

Returns a `RecoveryArtifact`, which already zeroizes its 32-byte seed on Drop (existing code in `mint.rs`). The intermediate `bip39::Mnemonic` from the crate also zeroizes its internal entropy on drop, so the parsing path doesn't leak the seed in heap garbage.

### Explicitly excluded

- Word-by-word autocomplete or "did you mean?" suggestions. That's client UX, not library scope.
- Mnemonic passphrase / "25th word" support (see Non-Goals).
- Non-English wordlists. Adding them is a one-line `bip39::Language` change behind a future feature flag if needed; not required for v1.

## Encrypted-file wire format

### Cleartext header (13 bytes, bound as AAD via XChaCha20-Poly1305)

```text
offset  size  field             value
0       4     magic             b"HRMR"
4       1     format_version    0x01
5       1     kdf_id            0x01 (Argon2id)
6       4     kdf_m_kib (BE)    65536  (64 MiB)
10      2     kdf_t (BE)        3
12      1     kdf_p             1
```

The `HRMR` magic is deliberately distinct from ZEB-174's `HRMI` (identity.enc at-rest) so no parser can confuse the two file types. The 13 bytes are bound as AAD so any tampering — including a downgrade attack on KDF parameters — is rejected by Poly1305 before the payload is decrypted.

### Following the header

```text
offset      size       field
13          16         salt
29          24         nonce (XChaCha20-Poly1305 needs 24)
53          variable   ciphertext
end-16      16         Poly1305 tag
```

### Total file size

Variable but bounded:
- **Minimum 69 bytes** at the wire layer = 13 header + 16 salt + 24 nonce + 0 ciphertext + 16 tag. This is the smallest mathematically possible file. Note that a 69-byte file can never decrypt to a valid `RecoveryFileBody` because the empty plaintext won't CBOR-decode — the soft minimum (smallest valid file in practice) is closer to 110 bytes once a stub CBOR body is included. The 69-byte hard limit is a parse-time guard that fails fast on obviously-truncated input before paying the Argon2id cost.
- **Maximum 1024 bytes** — sanity cap to refuse absurd files before paying the Argon2id cost.

A typical file with a short comment lands around 100–200 bytes.

### Encrypted payload (CBOR via `harmony-owner::cbor::to_canonical`)

```rust
#[derive(Serialize, Deserialize)]
struct RecoveryFileBody {
    format: String,           // exactly "harmony-owner-recovery-v1"
    #[serde(with = "serde_bytes")]
    seed: [u8; 32],
    mint_at: Option<u64>,
    comment: Option<String>,  // max 256 bytes pre-encryption
}
```

The `format` string inside the encrypted payload is defense-in-depth: even though Poly1305 already proves the payload was produced by someone with the passphrase, validating `format == "harmony-owner-recovery-v1"` after decryption protects against future format bifurcations being silently accepted by older parsers.

### Encode pipeline (`to_encrypted_file`)

1. Validate `comment.as_ref().map(|s| s.len()).unwrap_or(0) <= 256`. Caller error if violated → `RecoveryError::CommentTooLong`. No silent truncation.
2. Build `RecoveryFileBody { format: "harmony-owner-recovery-v1", seed: self.seed, mint_at, comment }`.
3. CBOR-encode → plaintext bytes (zeroized after step 7).
4. Generate fresh 16-byte salt and 24-byte nonce via `OsRng`.
5. Derive 32-byte key via Argon2id with the locked params (m=64MiB, t=3, p=1), kept in `Zeroizing<[u8; 32]>`.
6. Encrypt plaintext with XChaCha20-Poly1305, AAD = the 13-byte header.
7. Concatenate `header || salt || nonce || ciphertext || tag`. Return.

### Decode pipeline (`from_encrypted_file`)

1. **Length checks.** `bytes.len() < 69` → `RecoveryError::TooSmall`. `bytes.len() > 1024` → `RecoveryError::TooLarge`.
2. **Header validation.**
   - First 4 bytes ≠ `b"HRMR"` → `RecoveryError::UnrecognizedFormat`.
   - `format_version ≠ 0x01` → `RecoveryError::UnsupportedVersion`.
   - `kdf_id ≠ 0x01` → `RecoveryError::UnsupportedKdfId`.
   - Any KDF param ≠ the locked value → `RecoveryError::UnsupportedKdfParams { id }`. (Strict equality, same DoS guard as ZEB-174's `identity.enc`. Prevents an attacker-controlled file from coercing the parser into running an absurd Argon2id m_kib value as a CPU/memory DoS.)
3. Slice salt (16B), nonce (24B), ciphertext (variable), tag (last 16B).
4. **Derive key** via Argon2id with the locked params. Output kept in `Zeroizing<[u8; 32]>`.
5. **Decrypt** with AAD = parsed 13-byte header. AEAD failure → `RecoveryError::WrongPassphraseOrCorrupt`. (Deliberately ambiguous; the AEAD does not — and should not — distinguish these.)
6. **CBOR-decode** plaintext as `RecoveryFileBody`. Failure → `RecoveryError::PayloadDecodeFailed(String)`.
7. **Validate** `body.format == "harmony-owner-recovery-v1"`. Mismatch → `RecoveryError::UnexpectedPayloadFormat { found, expected }`.
8. **Validate** `body.comment.as_ref().map(|s| s.len()).unwrap_or(0) <= 256`. Mismatch → `RecoveryError::CommentTooLong` (defense-in-depth on read; protects against a malicious file that somehow round-trips through CBOR with an over-cap comment).
9. Build `RestoredArtifact { artifact: RecoveryArtifact::from_seed(body.seed), metadata: RecoveryMetadata { mint_at: body.mint_at, comment: body.comment } }`.
10. Zeroize the plaintext buffer (the seed has already been copied into the artifact, which has its own Drop wipe).

### Why no fixed file size

ZEB-174's `identity.enc` is fixed at 230 bytes because its payload (161-byte device identity blob) is fixed. The recovery file's payload is variable (comment field), so a fixed file size would either pre-pad the comment to its max (waste of bytes for the common case) or refuse to support comments at all. Variable-length with a hard cap is the right shape for a portable artifact.

## Error model

Single `RecoveryError` enum, distinguishable variants per failure class. Operator-readable messages, no secret material in error strings.

```rust
#[derive(Debug, thiserror::Error)]
pub enum RecoveryError {
    // ── Mnemonic parse ───────────────────────────────────────
    #[error("expected 24 BIP39 words, got {0}")]
    WrongWordCount(usize),

    #[error("unknown word at position {position}: {word:?}")]
    UnknownWord { position: usize, word: String },

    #[error("mnemonic checksum mismatch — likely a typo somewhere in the 24 words")]
    BadChecksum,

    #[error("mnemonic contains non-ASCII characters; BIP39 English wordlist is ASCII-only")]
    NonAsciiInput,

    // ── Encrypted-file decode ────────────────────────────────
    #[error("recovery file is too small ({0} bytes; minimum 69)")]
    TooSmall(usize),

    #[error("recovery file is too large ({0} bytes; maximum 1024)")]
    TooLarge(usize),

    #[error("not a harmony recovery file (magic mismatch)")]
    UnrecognizedFormat,

    #[error("recovery file format version {0:#x} is not supported by this build")]
    UnsupportedVersion(u8),

    #[error("recovery file uses unsupported KDF id {0:#x}")]
    UnsupportedKdfId(u8),

    #[error("recovery file KDF id {id:#x} present but parameters are non-standard")]
    UnsupportedKdfParams { id: u8 },

    #[error("wrong passphrase or corrupted recovery file (AEAD tag rejected)")]
    WrongPassphraseOrCorrupt,

    #[error("recovery file payload could not be decoded: {0}")]
    PayloadDecodeFailed(String),

    #[error("recovery file payload has unexpected format string {found:?}; expected {expected:?}")]
    UnexpectedPayloadFormat { found: String, expected: &'static str },

    // ── Encrypted-file encode ────────────────────────────────
    #[error("comment is {actual} bytes; max allowed is {max}")]
    CommentTooLong { actual: usize, max: usize },
}
```

### Two design notes

- **`WrongPassphraseOrCorrupt` is deliberately ambiguous.** The AEAD tag failure could be either, and distinguishing them requires speculative information the AEAD doesn't reveal (and shouldn't, for security). Same posture as ZEB-174's `identity.enc` errors.
- **No `From<bip39::Error>` blanket impl.** Each `bip39::Error` variant is mapped explicitly inside `from_mnemonic` so the public surface stays in our control. A future `bip39` crate update can't accidentally widen our error contract.

### Relation to `OwnerError`

`RecoveryError` is a sibling top-level error, NOT a variant of `OwnerError`. Recovery is a self-contained concern; folding it into `OwnerError` (which covers cert validation, CRDT operations, etc.) muddies both error taxonomies. Callers that need to handle both can use `Box<dyn Error>` or build their own union enum.

## Testing strategy

Mirrors the ZEB-174 testing pattern. Three tiers: unit tests per module, a wire-format fixture, and a cross-encoding equivalence check.

### Unit tests in `recovery/mnemonic.rs`

- `seed_to_mnemonic_roundtrips` — `RecoveryArtifact::from_seed(s).to_mnemonic()`, parse back, seed equal.
- `bip39_word_count_is_24` — split-whitespace yields exactly 24.
- `case_insensitive_input_succeeds` — uppercase, mixed case both decode.
- `whitespace_normalization` — multi-line, tabs, multiple spaces between words all decode.
- `unknown_word_reports_position` — corrupt one word; expect `UnknownWord { position, word }` with correct values.
- `bad_checksum_rejected` — flip one valid word for another valid word; expect `BadChecksum`.
- `wrong_word_count_rejected` — both 23 and 25 words.
- `non_ascii_rejected` — Cyrillic / emoji input → `NonAsciiInput`.
- `bip39_test_vector` — at least one canonical BIP39 entropy/mnemonic pair (e.g., the all-zero 32-byte seed produces a known word string) to lock interop with external BIP39 implementations.

### Unit tests in `recovery/encrypted_file.rs`

- `roundtrip_minimal_metadata` — encode with `RecoveryMetadata::default()`, decode, seed + metadata equal.
- `roundtrip_with_full_metadata` — `mint_at: Some(t)`, `comment: Some("primary owner")`, all fields preserved.
- `comment_at_max_length_succeeds` — exactly 256 bytes.
- `comment_over_max_fails_at_encode` — 257 bytes → `CommentTooLong`.
- `wrong_passphrase_fails_aead` — encode with passphrase A, decode with B → `WrongPassphraseOrCorrupt`.
- `tampered_ciphertext_fails` — flip a byte in the ciphertext region → `WrongPassphraseOrCorrupt`.
- `tampered_header_caught_by_strict_check_pre_aad` — flip a header byte → strict `parse_header` validation rejects with `UnsupportedKdfId` / `UnsupportedKdfParams` BEFORE AEAD/KDF runs. (The 13-byte header is also bound as AEAD AAD, but the strict-equality v1 check is structurally redundant with — and stricter than — the AAD layer.)
- `tampered_kdf_params_rejected_before_argon2` — change `kdf_m_kib` from 65536 to 32768 → `UnsupportedKdfParams { id: 0x01 }` BEFORE the Argon2id pass even runs.
- `wrong_magic_rejected` — `UnrecognizedFormat`.
- `unsupported_version_rejected` — `format_version = 0x02`.
- `too_small_rejected` — 50-byte input.
- `too_large_rejected` — 2 KiB input.
- `salt_rotates_per_encode` — same seed + passphrase encoded twice → different ciphertexts (confirms salt is regenerated per call).

### Wire-format fixture in `crates/harmony-owner/tests/recovery_wire_format_fixture.rs`

Matches ZEB-174's pattern:

- Commit `crates/harmony-owner/tests/fixtures/recovery_v1.bin` produced from a deterministic seed + deterministic salt + deterministic nonce + known passphrase + known metadata. Expose a `pub fn encrypt_with_params_for_test(...)` behind a `test-fixtures` Cargo feature (NOT default-on, NOT `#[doc(hidden)] pub`) that lets tests inject the salt/nonce explicitly. The cleaner feature-gate pattern was suggested in PR #58 review for ZEB-174 and we should adopt it for new code rather than carry forward the legacy `#[doc(hidden)] pub` workaround.
- Test asserts decoding the fixture yields the expected seed and metadata.
- Honor a `HARMONY_REGENERATE_RECOVERY_WIRE_FIXTURE=1` env var to rewrite the fixture for intentional format changes (so version bumps land cleanly).
- This pins the v1 wire format byte-for-byte. Any accidental serialization change (CBOR canonicalization drift, KDF param tweak, header layout shift) breaks this test loudly.

### Cross-encoding equivalence test in `recovery/mod.rs`

- `mnemonic_and_encrypted_file_yield_identical_master_pubkey` — encode same seed both ways, decode both, assert the resulting artifacts produce the same `master_pubkey_bundle().identity_hash()` in constant time.

### Out of scope for v1

- Property-based / fuzz testing. The fixture file plus negative-path unit tests cover the realistic bug surface; a `cargo fuzz` target can be added later if a corpus surfaces real bugs.
- Cross-platform testing of the file format (Linux / macOS / Windows). The encoding is endianness-explicit and the AEAD is platform-independent; standard `cargo test` on the developer's primary platform is sufficient.

## Wire-format evolution policy

If the encrypted-file format ever needs to change in a way that breaks decode (new KDF params, payload schema change, etc.), bump `format_version` from `0x01` to `0x02` AND update the magic if the change is fundamental. Old clients reading new files emit `UnsupportedVersion` rather than crashing. New clients reading old files keep working until the policy explicitly drops v1 support.

The wire-format fixture pins v1 forever — even after v2 ships, the v1 fixture test stays in CI as a regression guard. New format versions get their own fixture file alongside.

## Definition of done

1. `harmony-owner` exports `RecoveryArtifact::to_mnemonic`, `from_mnemonic`, `to_encrypted_file`, `from_encrypted_file` plus `RecoveryMetadata` and `RestoredArtifact` types.
2. The `recovery` Cargo feature is default-on; building with `default-features = false` excludes the recovery code and its dependencies.
3. All unit tests above pass.
4. The wire-format fixture is committed and pins v1 byte-for-byte.
5. The cross-encoding equivalence test passes.
6. Public APIs have rustdoc explaining the threat model, the mnemonic-vs-file tradeoff, and the security-critical invariants (header is AAD; seeds are never logged; passphrases are `SecretString`).
7. `cargo doc --no-deps -p harmony-owner` builds without warnings.
8. `cargo test -p harmony-owner` passes with both `--all-features` and `--no-default-features`.

## Follow-on work

A separate ticket (to be filed once this lands) will:

1. Add `harmony-owner = { ... }` as a dependency of harmony-client's `src-tauri` crate.
2. Replace the current `identity.rs` device-only flow with `harmony-owner::lifecycle::mint_owner` on first launch, persisting OwnerState + the device signing key in the keychain/encrypted backends from ZEB-174.
3. Add settings UI: "Identity → Backup" (display mnemonic + offer encrypted-file export) and "Identity → Restore" (paste mnemonic OR import encrypted file, prompt for passphrase).
4. Run round-trip verification at restore time: derived owner_id matches the restore-time expectation; signature challenge proves the master_signing_key works.

That ticket's scope is large (resolution-chain refactor, UI design, integration testing); it deserves its own design pass and brainstorming session.

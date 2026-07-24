# ZEB-746 — Extract the Argon2id + XChaCha20 at-rest envelope into `harmony-crypto`

**Ticket:** ZEB-746 (ZEB-571 seam-audit item 8). **Date:** 2026-07-24. **Author:** Koya (agent), design approved by Jake 2026-07-24.

**Goal:** Replace three hand-rolled copies of the same Argon2id-then-XChaCha20-Poly1305 password-based at-rest encryption envelope with one shared, feature-gated primitive in `harmony-crypto`, leaving every on-disk byte unchanged.

**Lineage:** Same kernel-extract / core-first / cross-repo pattern as ZEB-736/737/738/744.

---

## 1. Background — the three copies

Two read-only forensic passes (2026-07-24, against client `main` `1e9644cf` and core `main` `3745744`) confirmed the audit's premise and surfaced a third copy it did not name:

| Copy | Location | Magic | AAD | Inner plaintext |
|---|---|---|---|---|
| Identity vault | client `src-tauri/src/identity.rs` (`HRMI`) | `HRMI` | 13-byte header | v0x01 raw 32-B seed (legacy) / v0x02 CBOR `SecretVault` (prod) |
| Owner-state snapshot | client `src-tauri/src/state_snapshot.rs` (`HRSS`) | `HRSS` | **string** `harmony-owner-state-snapshot-v1` | CBOR owner-state snapshot |
| Recovery file | core `harmony-owner/src/recovery/encrypted_file.rs` (`HRMR`) | `HRMR` | 13-byte header | CBOR `RecoveryFileBody` |

All three are **byte-structurally identical** in the parts that matter to a shared primitive:

- **Header schema (13 bytes):** `magic(4) · format_version(1) · kdf_id(1) · m_kib(u32 BE) · t(u16 BE) · p(u8)`. Note `t` is **u16** (not u32) — chosen so the header is exactly 13 bytes; endianness is load-bearing.
- **KDF:** Argon2id, version `V0x13`, `m = 65536 KiB (64 MiB)`, `t = 3`, `p = 1`, `salt = 16 B`, `out = 32 B`.
- **AEAD:** XChaCha20-Poly1305, `nonce = 24 B`, `tag = 16 B`, appended to the ciphertext.
- **Crates:** `argon2 0.5`, `chacha20poly1305 0.10` (same workspace pins both repos).

They differ only in the 4-byte magic, the AAD bytes (HRMI/HRMR use the header; HRSS uses a fixed string), and the inner plaintext shape. Core's own `recovery/wire.rs` documents the shared origin: *"matches ZEB-174's identity.enc but uses magic HRMR instead of HRMI so the two file types can never be confused."*

**`harmony-crypto` today has neither primitive:** its `aead` module is IETF ChaCha20-Poly1305 with a **12-byte** nonce (not XChaCha20's 24), and it has no `argon2` dependency at all. `harmony-crypto` is depended on by 22 workspace crates, several `no_std` — so the new code must be **feature-gated and default-off**.

---

## 2. The shared primitive — `harmony_crypto::password_envelope`

A new module behind a new **default-off** cargo feature `password-envelope` that turns on an **optional** `argon2` dependency and exercises the XChaCha20 path of the already-present `chacha20poly1305` dep.

The design rule that makes one function back three formats: **the caller supplies the already-serialized header (or string) as opaque `aad` bytes; the primitive never re-derives it.** The magic lives *inside* the AAD, so a primitive that hard-coded any header would change the other formats' Poly1305 tags and break their existing files. Everything format-specific stays in the caller; the primitive knows only "Argon2id-derive, then XChaCha20-Poly1305 over these exact bytes."

```rust
/// Argon2id cost parameters. Caller-supplied because each on-disk format encodes
/// these values in its own header and must round-trip unchanged. Fields are
/// private so the validating `new()` is the only way to obtain an instance — no
/// invalid params are constructible. Callers build from their own wire constants
/// and never read them back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2idParams {
    m_kib: u32,
    t_cost: u32,
    p_cost: u32,
}

impl Argon2idParams {
    /// Validates the triple via `argon2::Params::new` (with the fixed 32-byte
    /// output length). Returns `CryptoError` for out-of-range values.
    pub fn new(m_kib: u32, t_cost: u32, p_cost: u32) -> Result<Self, CryptoError>;
}

/// Argon2id-derive a 32-byte key from `password` + `salt` + `params`, then
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
) -> Result<Vec<u8>, CryptoError>;

/// Inverse of `seal`. Returns the zeroizing plaintext, or `CryptoError` on any
/// failure (wrong key / tampered ciphertext / tag mismatch). Callers that must
/// not leak a padding/format oracle collapse all error variants into one
/// indistinguishable message.
pub fn open(
    password: &[u8],
    params: &Argon2idParams,
    salt: &[u8],
    nonce: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Zeroizing<Vec<u8>>, CryptoError>;
```

Fixed invariants inside the primitive: key/out length = 32 (the XChaCha20 key size); `nonce` must be exactly 24 bytes (else a `CryptoError`, never a panic — the exact variant is an implementation choice consistent with the existing enum); `salt`, `aad`, `plaintext`/`ciphertext` are opaque byte slices. The module is pure `no_std` + `alloc` — it generates **no randomness** (salt and nonce are caller-supplied), which is what preserves the deterministic-fixture seam that every golden test relies on.

**Byte-identity by construction:** identical Argon2 inputs → identical 32-byte key → identical `XChaCha20Poly1305::encrypt(nonce, {msg, aad})` → identical `ct ‖ tag`. The existing golden vectors on both sides are the proof, not the claim.

### API-shape decision

Combined `seal`/`open` (derive-then-encrypt in one call) was chosen over a split `derive_key` + `xchacha_seal`/`open`: every current call site derives a fresh key per operation, and bundling keeps the 32-byte key from ever escaping into caller code (it is zeroized internally), removing the most error-prone step from three call sites. A stateful builder was rejected as over-engineered (YAGNI).

---

## 3. What stays caller-side (all three consumers, unchanged)

The primitive is deliberately ignorant of format. Each consumer keeps, byte-for-byte:

1. **Header serialization** — `magic / format_version / kdf_id / m / t / p` (u16-BE `t`), and prepending `header ‖ salt ‖ nonce` to the sealed `ct ‖ tag`.
2. **The strict param-equality DoS guard** — parse the header, reject unless `m/t/p` equal the pinned constants *before* calling `open` (so an attacker-controlled `m_kib` can never force a huge Argon2 allocation). This stays in each caller because the pinned values and the rejection contract are per-format policy.
3. **AAD construction** — HRMI/HRMR pass the 13-byte header; HRSS passes its fixed string. The primitive just receives `aad: &[u8]`.
4. **Inner plaintext** — client v0x01 raw seed / v0x02 CBOR `SecretVault`; HRSS CBOR snapshot; core CBOR `RecoveryFileBody` + its `format` string defense-in-depth check. Version dispatch (`bytes[4]`) and the legacy v0x01 read path stay in `identity.rs`.
5. **`secrecy::SecretString`** password handling and transient `expose_secret()` borrows.
6. **`subtle` constant-time compares** — `verify_round_trip` (re-read-after-write seed check) and `passphrase_eq` (no-op-rotation detection) in `identity.rs`; the equivalent in core.
7. **Salt/nonce RNG** — production callers generate fresh random salt+nonce (`OsRng`) and pass them in; deterministic fixtures pass fixed bytes.
8. **Error mapping** — the **client collapses every `open` failure into the one indistinguishable string** `"identity store could not be decrypted: wrong passphrase or corrupted file"`; core maps to `RecoveryError` variants. No new distinguishable error may leak out of the rewire.

---

## 4. Byte-identity gate (two-sided) + fixture-gap closure

Byte-identity is the non-negotiable acceptance gate, on **both** repos:

- **Core:** `harmony-owner/tests/recovery_wire_format_fixture.rs` (the committed `recovery_v1.bin` / `recovery_v1_no_metadata.bin`) and the `wire.rs` header golden stay green after the recovery rewire.
- **Client:** `tests/wire_format/fixture.rs` (`encrypted_v1.bin`), `identity.rs::wire_format::header_layout_is_exact`, and every round-trip / tamper / rotation / vault test stay green after the HRMI + HRSS rewire.

**Fixture gap (must close):** the client's **v0x02 production vault** and **HRSS** formats currently have **no byte-pinned golden vector** — only the legacy v0x01 identity format does. Since v0x02 is what production writes, an extraction could silently change its framing with every existing test still passing. Before/with the client rewire we add deterministic golden fixtures (fixed salt+nonce, behind `test-fixtures`) for the v0x02 vault and for HRSS, so the two-sided gate actually covers the formats in production use.

---

## 5. Sequencing — two PRs, core-first

### PR 1 — harmony (core)

1. Add `harmony-crypto/src/password_envelope.rs` + the `password-envelope` feature (`dep:argon2` optional; XChaCha20 via the existing `chacha20poly1305` dep) + unit tests: round-trip, wrong-password/tamper rejection, param-mismatch, wrong-nonce-length, and a fixed-input determinism vector.
2. Rewire `harmony-owner/src/recovery/encrypted_file.rs` `encrypt_core` / `decrypt_inner` onto `harmony_crypto::password_envelope::{seal, open}`. `harmony-owner`'s existing `recovery` feature adds `harmony-crypto/password-envelope`. Keep `wire.rs`, `RecoveryFileBody`, the format string, `RecoveryError`, and the DoS guard exactly as-is. `recovery_v1*.bin` fixtures + header golden stay green.
3. Gates: `cargo fmt --all --check`; `cargo clippy --all-targets -D warnings` (workspace); `cargo nextest` for `harmony-crypto` + `harmony-owner`; a `harmony-crypto` **default-feature (no_std-relevant) build check** to confirm the new deps stay behind the gate. CI + bots converge. Merge.

### PR 2 — harmony-client

1. Bump the 12-crate lockstep rev to the PR-1 merge rev (`harmony-pkarr` pin untouched); enable `harmony-crypto/password-envelope` on the client's `harmony-crypto` dep.
2. Rewire the four `identity.rs` HRMI crypto sites (`encrypt_with_params`, `decrypt`, `encrypt_vault`, `decrypt_v2_plaintext` — covering v0x01 and v0x02) onto the primitive.
3. Rewire `state_snapshot.rs` HRSS (string AAD) onto the primitive.
4. Add the deterministic v0x02 + HRSS golden fixtures (`test-fixtures`).
5. Drop the client's now-unused direct `argon2` / `chacha20poly1305` dependencies **only if** a workspace check confirms no other client code uses them.
6. Gates (fmt, clippy `--all-targets -D warnings`, scoped nextest per the `scripts/test-select` cost discipline, frontend unchanged) + CI + bots. Merge.

---

## 6. Testing summary

- **New (core):** `password_envelope` unit tests (round-trip, tamper, param/nonce validation, determinism).
- **Preserved (core):** `recovery_wire_format_fixture.rs` golden `.bin` + `wire.rs` header golden + `encrypted_file.rs` round-trip/tamper tests — all unchanged, all green (proves HRMR byte-identity post-rewire).
- **Preserved (client):** `encrypted_v1.bin` golden + `header_layout_is_exact` + all round-trip/tamper/rotation/vault tests — unchanged, green.
- **New (client):** deterministic golden fixtures for the v0x02 vault and HRSS.

The two-sided golden set is the definition of "no wire/disk change."

---

## 7. Feature-gating & dependency notes

- `harmony-crypto`: `password-envelope = ["dep:argon2"]`, default-off. `argon2 = { workspace = true, optional = true }`. XChaCha20 needs no new dep (`chacha20poly1305` is already present). The module is `alloc`-only; it must compile without the `std` feature so `no_std` consumers that never enable `password-envelope` are unaffected.
- `harmony-owner`: its default-on `recovery` feature gains `harmony-crypto/password-envelope`; `harmony-owner` already lists `argon2`/`chacha20poly1305` as optional under `recovery`, so after the rewire those direct deps may be droppable if nothing else in `harmony-owner` uses them (verify).
- Blast radius: the 20+ crates that depend on `harmony-crypto` but never enable `password-envelope` gain no new transitive dependency and no compile-time cost.

---

## 8. Definition of done

1. `harmony_crypto::password_envelope` merged behind a default-off feature; `harmony-owner/recovery` rewired onto it with `recovery_v1*.bin` fixtures green.
2. Client rewired: `identity.rs` (HRMI v0x01+v0x02) and `state_snapshot.rs` (HRSS) on the shared primitive; new v0x02 + HRSS golden fixtures; all existing wire/rotation/vault tests green.
3. Gates green both repos (fmt, clippy `--all-targets -D warnings`, scoped nextest, `harmony-crypto` no_std default build); CI + bots converged. No on-disk byte changed.

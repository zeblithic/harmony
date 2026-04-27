# ZEB-177 — Seeded Keygen for `harmony-identity` Design

## Goal

Add `PrivateIdentity::from_seed(&[u8; 32])` and `PqPrivateIdentity::from_seed(&[u8; 32])` to `harmony-identity` so a 32-byte master seed deterministically derives the same Ed25519 + X25519 + ML-KEM-768 + ML-DSA-65 keypairs every call. Hard prerequisite for [ZEB-176](https://linear.app/zeblith/issue/ZEB-176) (harmony-client identity backup/restore CLI), which needs a meaningful "restore" path — the current `generate(rng)` API produces fresh randomness each call, so a backed-up seed has nothing to derive.

## Why this is upstream of ZEB-176

The [ZEB-175](https://linear.app/zeblith/issue/ZEB-175) recovery library wraps a 32-byte master seed in a BIP39-24 mnemonic / Argon2id-encrypted file. ZEB-176 wires that into harmony-client. But harmony-client today calls `PqPrivateIdentity::generate(OsRng)` + `PrivateIdentity::generate(OsRng)` directly — there's no seed in the picture. Without seeded keygen, "restore" can only round-trip the 161-byte serialized blob, not a recoverable master seed. We've explicitly chosen (in the ZEB-176 brainstorming) to gain mnemonic recovery at the cost of breaking continuity with existing in-tree harmony-client identities, which are placeholder.

## Architecture

ZEB-177 spans two crates layered cleanly:

### `harmony-crypto` — low-level seed exposure

The underlying `ml_kem` crate already exposes a `DecapsulationKey::from_seed`, and `MlDsa65` already exposes `from_seed`. The harmony-crypto wrappers (`ml_kem::generate`, `ml_dsa::generate`) call those internally, but don't currently expose seeded variants publicly. ZEB-177 lifts those into the public API and refactors `generate` to delegate:

```rust
// harmony-crypto/src/ml_kem.rs
pub fn from_seed(seed: &[u8; SK_LENGTH]) -> (MlKemPublicKey, MlKemSecretKey) { ... }
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlKemPublicKey, MlKemSecretKey) {
    let mut seed = [0u8; SK_LENGTH];
    rng.fill_bytes(&mut seed);
    let kp = from_seed(&seed);
    seed.zeroize();
    kp
}
```

Same shape in `ml_dsa.rs`. The refactor is a net DRY win — `from_seed` becomes the canonical primitive and `generate` is "fill buffer, delegate, zeroize." Externally visible behavior is unchanged.

Ed25519 and X25519 don't need a `harmony-crypto` change — their underlying crates (`ed25519_dalek::SigningKey::from_bytes`, `x25519_dalek::StaticSecret::from`) already accept `[u8; 32]` directly.

### `harmony-identity` — high-level seeded composition

Derive 4 sub-keys from one master 32-byte seed via HKDF-SHA256 with disjoint info strings:

```text
master 32B seed
   │
   ├─ HKDF(info=b"harmony-identity-ed25519-v1", len=32) ─→ ed25519 SigningKey
   ├─ HKDF(info=b"harmony-identity-x25519-v1",  len=32) ─→ x25519 StaticSecret
   ├─ HKDF(info=b"harmony-identity-ml-kem-v1",  len=64) ─→ ml_kem::from_seed
   └─ HKDF(info=b"harmony-identity-ml-dsa-v1",  len=32) ─→ ml_dsa::from_seed
```

Each HKDF call is independent: `salt = None`, IKM = master seed, `info` = the per-key domain separator, `length` = whatever the underlying primitive needs. Adding a future sub-key (e.g., HPKE, per-device sub-signing, an additional MLS-style PSK) is purely additive: pick a new info string, no offset shifts, no version bump.

Two new entry points, both infallible:

```rust
impl PrivateIdentity {
    pub fn from_seed(seed: &[u8; 32]) -> Self;
}
impl PqPrivateIdentity {
    pub fn from_seed(seed: &[u8; 32]) -> Self;
}
```

Existing `generate(rng)` paths on `harmony-identity` are unchanged — they continue to call the underlying RNG-driven keygen. We're not collapsing them through `from_seed` because the harmony-identity-level `generate(rng)` produces 4 *independent* keypairs (each with its own RNG draw), which is a different security property than "derive 4 keys from one master seed." Both paths coexist.

## Why per-key HKDF (over single-call expansion-and-split)

The alternative was one HKDF call producing 160 bytes total, split by offset (`[0..32]=ed25519, [32..64]=x25519, [64..128]=ml-kem-seed, [128..160]=ml-dsa-seed`). Rejected because:

- Adding a future sub-key would either append past offset 160 (fine) or — more likely as the identity model evolves — require inserting in the middle, which forces a `format_version` bump and re-mints every existing identity.
- Per-key derivation with disjoint info strings is the standard NIST SP 800-108 / RFC 5869 pattern for purpose-separation. Mature systems (Tor, Signal, Noise) all do this.
- The single-call performance "savings" are imaginary: 4 HKDF-SHA256 calls each producing ≤64 bytes is sub-microsecond.

## API surface & implementation details

### `harmony-crypto/src/ml_kem.rs` — new public function

```rust
/// Derive an ML-KEM-768 keypair from a 64-byte seed.
///
/// Deterministic: same seed in → same keypair out. Used by
/// `harmony-identity::PqPrivateIdentity::from_seed` to build the PQ
/// identity from the master 32-byte seed via HKDF expansion.
pub fn from_seed(seed: &[u8; SK_LENGTH]) -> (MlKemPublicKey, MlKemSecretKey) {
    let mut seed_arr = Array::from(*seed);
    let dk = ml_kem::DecapsulationKey::<MlKem768>::from_seed(seed_arr);
    let ek = dk.encapsulation_key().clone();
    seed_arr.zeroize();
    (MlKemPublicKey { inner: ek }, MlKemSecretKey { inner: dk })
}
```

`generate(rng)` becomes a 5-line wrapper that fills a buffer and delegates. Existing `generate` tests still pass.

### `harmony-crypto/src/ml_dsa.rs` — same shape

```rust
/// Derive an ML-DSA-65 keypair from a 32-byte seed.
pub fn from_seed(seed: &[u8; SK_LENGTH]) -> (MlDsaPublicKey, MlDsaSecretKey) {
    let mut seed_arr = ml_dsa::B32::from(*seed);
    let kp = MlDsa65::from_seed(&seed_arr);
    let pk = MlDsaPublicKey { inner: kp.verifying_key().clone() };
    let sk = MlDsaSecretKey { seed: *seed };
    seed_arr.zeroize();
    (pk, sk)
}
```

### `harmony-identity/src/identity.rs` — `PrivateIdentity::from_seed`

```rust
const SEED_INFO_ED25519: &[u8] = b"harmony-identity-ed25519-v1";
const SEED_INFO_X25519:  &[u8] = b"harmony-identity-x25519-v1";

impl PrivateIdentity {
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let ed_bytes = hkdf::derive_key(seed, None, SEED_INFO_ED25519, 32)
            .expect("HKDF length 32 is within the SHA-256 limit");
        let x_bytes  = hkdf::derive_key(seed, None, SEED_INFO_X25519, 32)
            .expect("HKDF length 32 is within the SHA-256 limit");

        let mut ed_arr: [u8; 32] = ed_bytes.as_slice().try_into().unwrap();
        let mut x_arr:  [u8; 32] = x_bytes.as_slice().try_into().unwrap();
        let signing_key = SigningKey::from_bytes(&ed_arr);
        let encryption_secret = StaticSecret::from(x_arr);
        ed_arr.zeroize();
        x_arr.zeroize();

        let encryption_key = X25519PublicKey::from(&encryption_secret);
        let verifying_key = signing_key.verifying_key();
        let identity = Identity::from_public_keys(
            encryption_key.as_bytes(),
            verifying_key.as_bytes(),
        ).expect("from_seed-derived keys are always valid");

        Self { identity, encryption_secret, signing_key }
    }
}
```

The two `expect`s are sound:

- HKDF length: 32 is well under HKDF-SHA256's 8160-byte ceiling. The constant cannot fail.
- `Identity::from_public_keys`: deterministic Ed25519/X25519 from-bytes constructors don't fail on `[u8; 32]`. Existing `generate(rng)` uses the same `expect` pattern (identity.rs:174) for the same reason.

### `harmony-identity/src/pq_identity.rs` — `PqPrivateIdentity::from_seed`

```rust
const SEED_INFO_ML_KEM: &[u8] = b"harmony-identity-ml-kem-v1";
const SEED_INFO_ML_DSA: &[u8] = b"harmony-identity-ml-dsa-v1";

impl PqPrivateIdentity {
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let kem_seed_bytes = hkdf::derive_key(seed, None, SEED_INFO_ML_KEM, ml_kem::SK_LENGTH)
            .expect("HKDF length 64 is within the SHA-256 limit");
        let dsa_seed_bytes = hkdf::derive_key(seed, None, SEED_INFO_ML_DSA, ml_dsa::SK_LENGTH)
            .expect("HKDF length 32 is within the SHA-256 limit");

        let mut kem_seed: [u8; ml_kem::SK_LENGTH] = kem_seed_bytes.as_slice().try_into().unwrap();
        let mut dsa_seed: [u8; ml_dsa::SK_LENGTH] = dsa_seed_bytes.as_slice().try_into().unwrap();
        let (encryption_key, encryption_secret) = ml_kem::from_seed(&kem_seed);
        let (verifying_key,  signing_key)       = ml_dsa::from_seed(&dsa_seed);
        kem_seed.zeroize();
        dsa_seed.zeroize();

        let identity = PqIdentity::from_public_keys(encryption_key, verifying_key);
        Self { identity, encryption_secret, signing_key }
    }
}
```

### Info-string convention

All four constants are `b"harmony-identity-<algorithm>-v<N>"` where `<algorithm>` is the lowercased algorithm name and `<N>` is `1` for the initial release. Matches the existing `harmony-pq-encrypt-v1` convention used in `pq_identity.rs:117`. Future format bumps increment `N`.

### Salt choice

`salt = None` for all four HKDF calls. The IKM (master seed) is 32 bytes of high-entropy material — salt is cosmetic when IKM has the full strength of the desired output. Domain separation is already provided by the `info` parameter, which is the load-bearing separator.

## Testing strategy

Five test classes across the two crates:

### 1. Determinism

Same seed → same private bytes (the load-bearing invariant).

```rust
#[test]
fn private_identity_from_seed_is_deterministic() {
    let seed = [0x42u8; 32];
    let a = PrivateIdentity::from_seed(&seed);
    let b = PrivateIdentity::from_seed(&seed);
    assert_eq!(a.to_private_bytes(), b.to_private_bytes());
    assert_eq!(a.identity.address_hash, b.identity.address_hash);
}
```

Same shape for `PqPrivateIdentity`. The harmony-crypto leaf primitives `ml_kem::from_seed` and `ml_dsa::from_seed` get the same treatment, comparing their outputs via `pk.as_bytes()` / `sk.as_bytes()` byte-equality.

### 2. Disjointness

Different info strings produce different keys. Sanity check that the domain separators are actually doing work.

```rust
#[test]
fn classical_subkeys_are_disjoint() {
    let seed = [0x42u8; 32];
    let id = PrivateIdentity::from_seed(&seed);
    let ed_bytes: [u8; 32] = id.signing_key.to_bytes();
    let x_bytes: [u8; 32] = id.encryption_secret.to_bytes();
    assert_ne!(ed_bytes, x_bytes,
        "info-string separation failed: ed25519 and x25519 derived to same bytes");
}
```

Same for `PqPrivateIdentity` (kem_seed != dsa_seed). One cross-class assertion: classical Ed25519 bytes ≠ PQ ML-DSA seed bytes (proves the four info strings are all distinct in effect).

### 3. Round-trip via `to_private_bytes` / `from_private_bytes`

`from_seed`-derived identities serialize and deserialize identically to `generate`-derived ones.

```rust
#[test]
fn from_seed_round_trips_via_private_bytes() {
    let seed = [0x42u8; 32];
    let original = PrivateIdentity::from_seed(&seed);
    let bytes = original.to_private_bytes();
    let restored = PrivateIdentity::from_private_bytes(&bytes).unwrap();
    assert_eq!(original.identity.address_hash, restored.identity.address_hash);
}
```

Same for `PqPrivateIdentity`.

### 4. Functional equivalence

A from_seed identity actually works (sign/verify, encrypt/decrypt). Catches bugs where the HKDF expansion silently produces a structurally-valid-but-broken key.

```rust
#[test]
fn from_seed_identity_can_sign_and_verify() {
    let id = PrivateIdentity::from_seed(&[0x42u8; 32]);
    let sig = id.sign(b"hello");
    id.identity.verify(b"hello", &sig).unwrap();
}
#[test]
fn from_seed_pq_identity_can_encrypt_and_decrypt() {
    let id = PqPrivateIdentity::from_seed(&[0x42u8; 32]);
    let mut rng = rand::rngs::OsRng;
    let ct = id.public_identity().encrypt(&mut rng, b"hello").unwrap();
    let pt = id.decrypt(&ct).unwrap();
    assert_eq!(pt, b"hello");
}
```

### 5. Golden vector

Pin one specific (seed → public-key bytes) tuple inline as hex literals. Catches accidental info-string drift, KDF change, or upstream primitive ABI shift.

```rust
#[test]
fn from_seed_pins_public_keys_for_zero_seed() {
    let seed = [0u8; 32];
    let id = PrivateIdentity::from_seed(&seed);
    // Pinned hex strings filled in during impl from the first run.
    let expected_x25519_pub  = hex::decode("...").unwrap();  // 32 bytes
    let expected_ed25519_pub = hex::decode("...").unwrap();  // 32 bytes
    assert_eq!(&id.identity.encryption_key.as_bytes()[..], expected_x25519_pub.as_slice());
    assert_eq!(&id.identity.verifying_key.as_bytes()[..], expected_ed25519_pub.as_slice());
}
```

For `PqPrivateIdentity`, pin the truncated SHA-256 of the public-key bytes (16 bytes = `address_hash`) rather than the full ~3 KiB pubkey blob — proves derivation didn't change without bloating the test.

```rust
#[test]
fn pq_from_seed_pins_address_hash_for_zero_seed() {
    let seed = [0u8; 32];
    let id = PqPrivateIdentity::from_seed(&seed);
    let expected_address_hash = hex::decode("...").unwrap();  // 16 bytes
    assert_eq!(&id.public_identity().address_hash[..], expected_address_hash.as_slice());
}
```

Uses the `hex` crate already present as a dev-dep across the workspace (`hkdf.rs` tests already use `hex::decode`); avoids adding `hex-literal` as a new dependency.

### Test placement

- `harmony-crypto::ml_kem::from_seed` + `ml_dsa::from_seed` get determinism + round-trip tests (the existing `generate` tests already prove sign/verify/encrypt/decrypt work for arbitrary keys).
- `harmony-identity::PrivateIdentity::from_seed` + `PqPrivateIdentity::from_seed` get all 5 test classes.
- Existing `generate(rng)` tests keep passing — externally visible behavior is unchanged after the refactor through `from_seed`.

### Out of scope

- Negative tests for HKDF / from_seed errors — both are infallible by design.
- Cross-version interop test — there's only v1; future version bumps will add tests then.
- Migration tests for existing-on-disk identities — explicitly punted (ZEB-176 starts fresh).

## Definition of done

1. `harmony-crypto::ml_kem::from_seed` and `ml_dsa::from_seed` added as public functions.
2. `harmony-crypto::ml_kem::generate` and `ml_dsa::generate` refactored to delegate to `from_seed`. Existing tests pass unchanged.
3. `harmony-identity::PrivateIdentity::from_seed(&[u8; 32])` and `PqPrivateIdentity::from_seed(&[u8; 32])` added.
4. All 5 test classes pass for both `PrivateIdentity` and `PqPrivateIdentity` (determinism, disjointness, round-trip, functional, golden vector).
5. `cargo clippy --all-targets -- -D warnings` clean.
6. `cargo doc --no-deps` clean (rustdoc warnings are errors).

## Open questions resolved during brainstorming

For posterity:

1. **HKDF expansion strategy** — B: per-key expansion with disjoint info strings (over single 160-byte expansion + offset split). Rationale: future-additive, matches NIST SP 800-108 / Tor / Signal / Noise idiom.
2. **`harmony-crypto` `generate` refactor** — B: refactor `generate(rng)` to delegate to the new public `from_seed`. DRYer; one source of truth for "ML-KEM/ML-DSA keygen given seed bytes."
3. **KDF choice** — HKDF-SHA256 (already used throughout the codebase via `harmony_crypto::hkdf::derive_key`).
4. **API shape** — constructor `from_seed(&[u8; 32]) -> Self` matching existing `generate(rng) -> Self` style.
5. **Error handling** — infallible (`-> Self`, not `-> Result`). All four expand-then-derive operations are infallible for the fixed-length inputs we use; the only theoretically-fallible step (HKDF length) is bounded by compile-time constants well under the 8160-byte ceiling.
6. **Salt** — `None` for all HKDF calls. Domain separation is provided by `info`; the master seed already has full output strength.
7. **Info-string format** — `harmony-identity-<algorithm>-v<N>`, matching the existing `harmony-pq-encrypt-v1` convention.

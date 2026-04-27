# ZEB-177 — Seeded Keygen for `harmony-identity` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `PrivateIdentity::from_seed(&[u8; 32])` and `PqPrivateIdentity::from_seed(&[u8; 32])` to `harmony-identity` so a 32-byte master seed deterministically derives the same Ed25519 + X25519 + ML-KEM-768 + ML-DSA-65 keypairs every call. Hard prerequisite for ZEB-176 (harmony-client backup/restore CLI).

**Architecture:** Two-layer change. `harmony-crypto` exposes new public `from_seed` functions for the ML-KEM and ML-DSA primitives (refactoring existing `generate(rng)` to delegate). `harmony-identity` composes those with HKDF-SHA256 expansion of a master 32-byte seed using disjoint info strings (`harmony-identity-{ed25519,x25519,ml-kem,ml-dsa}-v1`) — one HKDF call per sub-key, salt = None. Both `from_seed` entry points are infallible.

**Tech Stack:** Rust 2021, `harmony-crypto` workspace crate (HKDF-SHA256 already exposed via `harmony_crypto::hkdf::derive_key`), `ml-kem` + `ml-dsa` (RustCrypto), `ed25519-dalek` 2.x, `x25519-dalek` 2.x.

---

## Task 1: `harmony-crypto::ml_kem::from_seed` + `generate` refactor

**Files:**
- Modify: `crates/harmony-crypto/src/ml_kem.rs` (add `from_seed`, refactor `generate` to delegate, add tests)

- [ ] **Step 1: Read the existing `generate` body to confirm the seed-handling pattern**

Run: `grep -n "pub fn generate" crates/harmony-crypto/src/ml_kem.rs`
Expected: line ~170, `pub fn generate(rng: &mut impl CryptoRngCore) -> (MlKemPublicKey, MlKemSecretKey)`. The existing body fills a `[u8; SK_LENGTH]` from `rng`, wraps it in `Array::from`, calls `ml_kem::DecapsulationKey::<MlKem768>::from_seed(seed)`, zeroizes both. The refactor lifts this into a new public `from_seed` and makes `generate` a 5-line wrapper.

- [ ] **Step 2: Add the failing determinism test**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `crates/harmony-crypto/src/ml_kem.rs`:

```rust
#[test]
fn from_seed_is_deterministic() {
    let seed = [0x42u8; SK_LENGTH];
    let (pk_a, sk_a) = from_seed(&seed);
    let (pk_b, sk_b) = from_seed(&seed);
    assert_eq!(pk_a.as_bytes(), pk_b.as_bytes(),
        "from_seed must be deterministic across calls (public key)");
    assert_eq!(sk_a.as_bytes(), sk_b.as_bytes(),
        "from_seed must be deterministic across calls (secret key)");
}

#[test]
fn from_seed_round_trips_via_bytes() {
    let seed = [0x42u8; SK_LENGTH];
    let (pk, sk) = from_seed(&seed);
    let pk_bytes = pk.as_bytes();
    let sk_bytes = sk.as_bytes();
    let pk_restored = MlKemPublicKey::from_bytes(&pk_bytes).unwrap();
    let sk_restored = MlKemSecretKey::from_bytes(&sk_bytes).unwrap();
    assert_eq!(pk.as_bytes(), pk_restored.as_bytes());
    assert_eq!(sk.as_bytes(), sk_restored.as_bytes());
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test -p harmony-crypto --lib ml_kem::tests::from_seed`
Expected: compile error — `cannot find function 'from_seed' in this scope`. (Test infrastructure will pick up the test once the function compiles.)

- [ ] **Step 4: Add the new public `from_seed` and refactor `generate`**

Replace the existing `pub fn generate(...)` body in `crates/harmony-crypto/src/ml_kem.rs` (around line 170) with:

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

/// Generate an ML-KEM-768 keypair.
///
/// Uses the provided RNG to generate a 64-byte seed, then derives the
/// keypair deterministically via [`from_seed`].
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlKemPublicKey, MlKemSecretKey) {
    let mut seed = [0u8; SK_LENGTH];
    rng.fill_bytes(&mut seed);
    let kp = from_seed(&seed);
    seed.zeroize();
    kp
}
```

- [ ] **Step 5: Run all `harmony-crypto::ml_kem` tests to verify nothing regresses**

Run: `cargo test -p harmony-crypto --lib ml_kem`
Expected: All existing tests + the two new ones pass. Existing `generate`-driven tests pass unchanged because `generate` still has the same external behavior (fill RNG → derive deterministic keypair → zeroize seed buffer).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-crypto/src/ml_kem.rs
git commit -m "feat(crypto): expose ml_kem::from_seed; refactor generate to delegate (ZEB-177)

Lifts the seed→keypair pattern that was already inside ml_kem::generate
into a new public from_seed function. generate becomes a 5-line wrapper
that fills a buffer from RNG and delegates. Externally visible behavior
is unchanged.

Required by harmony-identity::PqPrivateIdentity::from_seed (next task in
the ZEB-177 chain), which expands a 32-byte master seed via HKDF into
the 64-byte ML-KEM seed this function consumes.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `harmony-crypto::ml_dsa::from_seed` + `generate` refactor

**Files:**
- Modify: `crates/harmony-crypto/src/ml_dsa.rs` (add `from_seed`, refactor `generate` to delegate, add tests)

- [ ] **Step 1: Add the failing determinism + round-trip tests**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `crates/harmony-crypto/src/ml_dsa.rs`:

```rust
#[test]
fn from_seed_is_deterministic() {
    let seed = [0x42u8; SK_LENGTH];
    let (pk_a, sk_a) = from_seed(&seed);
    let (pk_b, sk_b) = from_seed(&seed);
    assert_eq!(pk_a.as_bytes(), pk_b.as_bytes(),
        "from_seed must be deterministic across calls (public key)");
    assert_eq!(sk_a.as_bytes(), sk_b.as_bytes(),
        "from_seed must be deterministic across calls (secret key)");
}

#[test]
fn from_seed_round_trips_via_bytes() {
    let seed = [0x42u8; SK_LENGTH];
    let (pk, sk) = from_seed(&seed);
    let pk_bytes = pk.as_bytes();
    let sk_bytes = sk.as_bytes();
    let pk_restored = MlDsaPublicKey::from_bytes(&pk_bytes).unwrap();
    let sk_restored = MlDsaSecretKey::from_bytes(&sk_bytes).unwrap();
    assert_eq!(pk.as_bytes(), pk_restored.as_bytes());
    assert_eq!(sk.as_bytes(), sk_restored.as_bytes());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-crypto --lib ml_dsa::tests::from_seed`
Expected: compile error — `cannot find function 'from_seed' in this scope`.

- [ ] **Step 3: Add the new public `from_seed` and refactor `generate`**

Replace the existing `pub fn generate(...)` body in `crates/harmony-crypto/src/ml_dsa.rs` (around line 189) with:

```rust
/// Derive an ML-DSA-65 keypair from a 32-byte seed.
///
/// Deterministic: same seed in → same keypair out. Used by
/// `harmony-identity::PqPrivateIdentity::from_seed` to build the PQ
/// identity from the master 32-byte seed via HKDF expansion.
pub fn from_seed(seed: &[u8; SK_LENGTH]) -> (MlDsaPublicKey, MlDsaSecretKey) {
    let mut seed_arr = ml_dsa::B32::from(*seed);
    let kp = MlDsa65::from_seed(&seed_arr);
    let pk = MlDsaPublicKey { inner: kp.verifying_key().clone() };
    let sk = MlDsaSecretKey { seed: *seed };
    seed_arr.zeroize();
    (pk, sk)
}

/// Generate an ML-DSA-65 keypair.
///
/// Uses the provided RNG to generate a 32-byte seed, then derives the
/// keypair deterministically via [`from_seed`].
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlDsaPublicKey, MlDsaSecretKey) {
    let mut seed = [0u8; SK_LENGTH];
    rng.fill_bytes(&mut seed);
    let kp = from_seed(&seed);
    seed.zeroize();
    kp
}
```

- [ ] **Step 4: Run all `harmony-crypto::ml_dsa` tests to verify nothing regresses**

Run: `cargo test -p harmony-crypto --lib ml_dsa`
Expected: All existing tests + the two new ones pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-crypto/src/ml_dsa.rs
git commit -m "feat(crypto): expose ml_dsa::from_seed; refactor generate to delegate (ZEB-177)

Same shape as ml_kem from_seed: lifts the seed→keypair pattern out of
generate and exposes it publicly. generate becomes a thin RNG wrapper
that delegates. Required by harmony-identity::PqPrivateIdentity::
from_seed (Task 4 in the ZEB-177 chain).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: `harmony-identity::PrivateIdentity::from_seed` (classical: Ed25519 + X25519)

**Files:**
- Modify: `crates/harmony-identity/src/identity.rs` (add info-string constants, `from_seed` impl, 5 test classes)

- [ ] **Step 1: Add the info-string constants near the existing public consts (top of file)**

After the existing constants block in `crates/harmony-identity/src/identity.rs` (around line 28, after `ADDRESS_HASH_LENGTH`), add:

```rust
/// HKDF info string for the Ed25519 sub-key derived from the master seed.
const SEED_INFO_ED25519: &[u8] = b"harmony-identity-ed25519-v1";
/// HKDF info string for the X25519 sub-key derived from the master seed.
const SEED_INFO_X25519:  &[u8] = b"harmony-identity-x25519-v1";
```

- [ ] **Step 2: Add the failing tests for `from_seed`**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `crates/harmony-identity/src/identity.rs`:

```rust
#[test]
fn from_seed_is_deterministic() {
    let seed = [0x42u8; 32];
    let a = PrivateIdentity::from_seed(&seed);
    let b = PrivateIdentity::from_seed(&seed);
    assert_eq!(a.to_private_bytes(), b.to_private_bytes(),
        "from_seed must produce identical private bytes for the same seed");
    assert_eq!(a.identity.address_hash, b.identity.address_hash,
        "from_seed must produce identical address hash for the same seed");
}

#[test]
fn from_seed_classical_subkeys_are_disjoint() {
    let seed = [0x42u8; 32];
    let id = PrivateIdentity::from_seed(&seed);
    // Both `StaticSecret::as_bytes()` and `SigningKey::as_bytes()` return
    // `&[u8; 32]`. Use the existing `to_private_bytes` access pattern from
    // `identity.rs:250-256` as a reference if these method names ever drift.
    let ed_bytes: &[u8; 32] = id.signing_key.as_bytes();
    let x_bytes:  &[u8; 32] = id.encryption_secret.as_bytes();
    assert_ne!(ed_bytes, x_bytes,
        "info-string separation failed: ed25519 and x25519 derived to same bytes");
}

#[test]
fn from_seed_round_trips_via_private_bytes() {
    let seed = [0x42u8; 32];
    let original = PrivateIdentity::from_seed(&seed);
    let bytes = original.to_private_bytes();
    let restored = PrivateIdentity::from_private_bytes(&bytes).unwrap();
    assert_eq!(original.identity.address_hash, restored.identity.address_hash);
}

#[test]
fn from_seed_identity_can_sign_and_verify() {
    let id = PrivateIdentity::from_seed(&[0x42u8; 32]);
    let sig = id.sign(b"hello");
    id.identity.verify(b"hello", &sig)
        .expect("from_seed-derived identity must produce verifiable signatures");
}

#[test]
fn from_seed_pins_public_keys_for_zero_seed() {
    let seed = [0u8; 32];
    let id = PrivateIdentity::from_seed(&seed);
    // Hex strings are filled in during this task's "pin the golden vector"
    // step (the test fails first with empty hex; the failure prints the
    // actual values; paste them back into these literals).
    let expected_x25519_pub_hex  = "PIN_ME_AFTER_FIRST_RUN";
    let expected_ed25519_pub_hex = "PIN_ME_AFTER_FIRST_RUN";
    assert_eq!(
        hex::encode(id.identity.encryption_key.as_bytes()),
        expected_x25519_pub_hex,
        "x25519 pub bytes drifted from pinned golden vector"
    );
    assert_eq!(
        hex::encode(id.identity.verifying_key.as_bytes()),
        expected_ed25519_pub_hex,
        "ed25519 pub bytes drifted from pinned golden vector"
    );
}
```

- [ ] **Step 3: Run tests to verify they fail (compile error)**

Run: `cargo test -p harmony-identity --lib identity::tests::from_seed`
Expected: compile error — `no function or associated item named 'from_seed' found for struct 'PrivateIdentity'`.

- [ ] **Step 4: Implement `PrivateIdentity::from_seed`**

In `crates/harmony-identity/src/identity.rs`, locate the `impl PrivateIdentity` block (starts around line 161). After the existing `pub fn generate(...)` (around line 163), add:

```rust
    /// Derive a `PrivateIdentity` deterministically from a 32-byte master seed.
    ///
    /// The master seed is HKDF-expanded into two disjoint sub-keys:
    /// - 32 bytes via `info=b"harmony-identity-ed25519-v1"` for the Ed25519 signing key
    /// - 32 bytes via `info=b"harmony-identity-x25519-v1"`  for the X25519 secret
    ///
    /// Same seed in → same keypairs out. Used by harmony-client to back up and
    /// restore identity via the [ZEB-175] recovery library.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let ed_bytes = harmony_crypto::hkdf::derive_key(seed, None, SEED_INFO_ED25519, 32)
            .expect("HKDF length 32 is within the SHA-256 limit");
        let x_bytes = harmony_crypto::hkdf::derive_key(seed, None, SEED_INFO_X25519, 32)
            .expect("HKDF length 32 is within the SHA-256 limit");

        let mut ed_arr: [u8; 32] = ed_bytes.as_slice().try_into()
            .expect("HKDF returned exactly 32 bytes");
        let mut x_arr: [u8; 32] = x_bytes.as_slice().try_into()
            .expect("HKDF returned exactly 32 bytes");
        let signing_key = SigningKey::from_bytes(&ed_arr);
        let encryption_secret = StaticSecret::from(x_arr);
        ed_arr.zeroize();
        x_arr.zeroize();

        let encryption_key = X25519PublicKey::from(&encryption_secret);
        let verifying_key = signing_key.verifying_key();
        let identity =
            Identity::from_public_keys(encryption_key.as_bytes(), verifying_key.as_bytes())
                .expect("from_seed-derived keys are always valid");

        Self {
            identity,
            encryption_secret,
            signing_key,
        }
    }
```

- [ ] **Step 5: Run the four non-golden-vector tests to verify they pass**

Run: `cargo test -p harmony-identity --lib identity::tests::from_seed_is_deterministic identity::tests::from_seed_classical_subkeys_are_disjoint identity::tests::from_seed_round_trips_via_private_bytes identity::tests::from_seed_identity_can_sign_and_verify`

Expected: 4 tests pass. The golden-vector test still fails because the hex strings are placeholders.

- [ ] **Step 6: Pin the golden vector**

Run: `cargo test -p harmony-identity --lib from_seed_pins_public_keys_for_zero_seed -- --nocapture 2>&1 | grep "drifted from pinned" -A 5 | head -20`

The assertion failure prints the actual `left` value (the hex of the produced pub key) — paste those into the test, replacing the `"PIN_ME_AFTER_FIRST_RUN"` placeholders. Specifically: the X25519 hex is the first failure; comment that line out temporarily, re-run, get the Ed25519 hex; uncomment X25519. Both are 64 hex chars (32 bytes each).

After pinning, the test body should look like:

```rust
    let expected_x25519_pub_hex  = "<actual-64-hex-chars-pinned-from-first-run>";
    let expected_ed25519_pub_hex = "<actual-64-hex-chars-pinned-from-first-run>";
```

- [ ] **Step 7: Run the golden-vector test to verify it passes**

Run: `cargo test -p harmony-identity --lib from_seed_pins_public_keys_for_zero_seed`
Expected: PASS.

- [ ] **Step 8: Run all `harmony-identity` tests to verify nothing regresses**

Run: `cargo test -p harmony-identity`
Expected: All tests pass (existing + 5 new).

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-identity/src/identity.rs
git commit -m "feat(identity): PrivateIdentity::from_seed for classical (Ed25519 + X25519) (ZEB-177)

Derives a PrivateIdentity deterministically from a 32-byte master seed
via two HKDF-SHA256 expansions with disjoint info strings:

  harmony-identity-ed25519-v1 -> Ed25519 SigningKey (32 bytes)
  harmony-identity-x25519-v1  -> X25519 StaticSecret (32 bytes)

Salt is None — the master seed has full output strength; domain
separation comes from info. Both expansions are infallible at 32 byte
length (well under HKDF-SHA256's 8160-byte ceiling).

Tests cover all 5 invariants: determinism, info-string disjointness,
round-trip via to_private_bytes/from_private_bytes, functional sign+
verify, and a golden-vector pin for the all-zero seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `harmony-identity::PqPrivateIdentity::from_seed` (PQ: ML-KEM-768 + ML-DSA-65)

**Files:**
- Modify: `crates/harmony-identity/src/pq_identity.rs` (add info-string constants, `from_seed` impl, 5 test classes)

- [ ] **Step 1: Add the info-string constants near the top of the file**

After the existing constants block in `crates/harmony-identity/src/pq_identity.rs` (around line 32, after `PQ_PRIVATE_KEY_LENGTH`), add:

```rust
/// HKDF info string for the ML-KEM-768 sub-key derived from the master seed.
const SEED_INFO_ML_KEM: &[u8] = b"harmony-identity-ml-kem-v1";
/// HKDF info string for the ML-DSA-65 sub-key derived from the master seed.
const SEED_INFO_ML_DSA: &[u8] = b"harmony-identity-ml-dsa-v1";
```

- [ ] **Step 2: Add the failing tests for `PqPrivateIdentity::from_seed`**

Append to the existing `#[cfg(test)] mod tests` block at the bottom of `crates/harmony-identity/src/pq_identity.rs`:

```rust
#[test]
fn pq_from_seed_is_deterministic() {
    let seed = [0x42u8; 32];
    let a = PqPrivateIdentity::from_seed(&seed);
    let b = PqPrivateIdentity::from_seed(&seed);
    assert_eq!(a.to_private_bytes(), b.to_private_bytes(),
        "from_seed must produce identical private bytes for the same seed");
    assert_eq!(a.public_identity().address_hash, b.public_identity().address_hash,
        "from_seed must produce identical address hash for the same seed");
}

#[test]
fn pq_from_seed_subkeys_are_disjoint() {
    // The ML-DSA secret-key bytes contain the 32-byte seed material directly,
    // and the ML-KEM secret-key bytes are derived from the 64-byte ML-KEM seed.
    // Both come out of HKDF with disjoint info strings — the first 32 bytes of
    // the ML-KEM seed must differ from the ML-DSA seed.
    let seed = [0x42u8; 32];
    let id = PqPrivateIdentity::from_seed(&seed);
    let kem_bytes = id.encryption_secret.as_bytes();  // 64-byte ML-KEM seed
    let dsa_bytes = id.signing_key.as_bytes();        // 32-byte ML-DSA seed
    assert_ne!(&kem_bytes[..32], &dsa_bytes[..],
        "info-string separation failed: ml-kem and ml-dsa derived to same bytes");
}

#[test]
fn pq_from_seed_round_trips_via_private_bytes() {
    let seed = [0x42u8; 32];
    let original = PqPrivateIdentity::from_seed(&seed);
    let bytes = original.to_private_bytes();
    let restored = PqPrivateIdentity::from_private_bytes(&bytes).unwrap();
    assert_eq!(
        original.public_identity().address_hash,
        restored.public_identity().address_hash,
        "round-trip via private bytes must preserve identity"
    );
}

#[test]
fn pq_from_seed_identity_can_encrypt_and_decrypt() {
    let id = PqPrivateIdentity::from_seed(&[0x42u8; 32]);
    let mut rng = rand::rngs::OsRng;
    let ct = id.public_identity().encrypt(&mut rng, b"hello").unwrap();
    let pt = id.decrypt(&ct).unwrap();
    assert_eq!(pt, b"hello");
}

#[test]
fn pq_from_seed_pins_address_hash_for_zero_seed() {
    let seed = [0u8; 32];
    let id = PqPrivateIdentity::from_seed(&seed);
    // Hex string is filled in during this task's "pin the golden vector" step.
    let expected_address_hash_hex = "PIN_ME_AFTER_FIRST_RUN";
    assert_eq!(
        hex::encode(id.public_identity().address_hash),
        expected_address_hash_hex,
        "PQ address hash drifted from pinned golden vector"
    );
}
```

- [ ] **Step 3: Run tests to verify they fail (compile error)**

Run: `cargo test -p harmony-identity --lib pq_identity::tests::pq_from_seed`
Expected: compile error — `no function or associated item named 'from_seed' found for struct 'PqPrivateIdentity'`.

- [ ] **Step 4: Implement `PqPrivateIdentity::from_seed`**

In `crates/harmony-identity/src/pq_identity.rs`, locate the `impl PqPrivateIdentity` block (starts around line 166). After the existing `pub fn generate(...)` (around line 168), add:

```rust
    /// Derive a `PqPrivateIdentity` deterministically from a 32-byte master seed.
    ///
    /// The master seed is HKDF-expanded into two disjoint sub-seeds:
    /// - 64 bytes via `info=b"harmony-identity-ml-kem-v1"` for the ML-KEM-768 keypair
    /// - 32 bytes via `info=b"harmony-identity-ml-dsa-v1"` for the ML-DSA-65 keypair
    ///
    /// Same seed in → same keypairs out. Used by harmony-client to back up and
    /// restore identity via the [ZEB-175] recovery library.
    pub fn from_seed(seed: &[u8; 32]) -> Self {
        let kem_seed_bytes = harmony_crypto::hkdf::derive_key(
            seed,
            None,
            SEED_INFO_ML_KEM,
            ml_kem::SK_LENGTH,
        )
        .expect("HKDF length 64 is within the SHA-256 limit");
        let dsa_seed_bytes = harmony_crypto::hkdf::derive_key(
            seed,
            None,
            SEED_INFO_ML_DSA,
            ml_dsa::SK_LENGTH,
        )
        .expect("HKDF length 32 is within the SHA-256 limit");

        let mut kem_seed: [u8; ml_kem::SK_LENGTH] = kem_seed_bytes
            .as_slice()
            .try_into()
            .expect("HKDF returned exactly ml_kem::SK_LENGTH bytes");
        let mut dsa_seed: [u8; ml_dsa::SK_LENGTH] = dsa_seed_bytes
            .as_slice()
            .try_into()
            .expect("HKDF returned exactly ml_dsa::SK_LENGTH bytes");
        let (encryption_key, encryption_secret) = ml_kem::from_seed(&kem_seed);
        let (verifying_key, signing_key) = ml_dsa::from_seed(&dsa_seed);
        kem_seed.zeroize();
        dsa_seed.zeroize();

        let identity = PqIdentity::from_public_keys(encryption_key, verifying_key);
        Self {
            identity,
            encryption_secret,
            signing_key,
        }
    }
```

- [ ] **Step 5: Run the four non-golden-vector tests to verify they pass**

Run: `cargo test -p harmony-identity --lib pq_identity::tests::pq_from_seed_is_deterministic pq_identity::tests::pq_from_seed_subkeys_are_disjoint pq_identity::tests::pq_from_seed_round_trips_via_private_bytes pq_identity::tests::pq_from_seed_identity_can_encrypt_and_decrypt`

Expected: 4 tests pass. The golden-vector test still fails because the hex string is a placeholder.

- [ ] **Step 6: Pin the golden vector**

Run: `cargo test -p harmony-identity --lib pq_from_seed_pins_address_hash_for_zero_seed -- --nocapture 2>&1 | grep "drifted from pinned" -A 5 | head -10`

The assertion failure prints the actual `left` value (32 hex chars = 16 bytes = address hash). Paste it into the test, replacing the `"PIN_ME_AFTER_FIRST_RUN"` placeholder.

After pinning:

```rust
    let expected_address_hash_hex = "<actual-32-hex-chars-pinned-from-first-run>";
```

- [ ] **Step 7: Run the golden-vector test to verify it passes**

Run: `cargo test -p harmony-identity --lib pq_from_seed_pins_address_hash_for_zero_seed`
Expected: PASS.

- [ ] **Step 8: Run all `harmony-identity` tests to verify nothing regresses**

Run: `cargo test -p harmony-identity`
Expected: All tests pass (existing + 5 from Task 3 + 5 from this task = 10 new).

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-identity/src/pq_identity.rs
git commit -m "feat(identity): PqPrivateIdentity::from_seed for ML-KEM-768 + ML-DSA-65 (ZEB-177)

Derives a PqPrivateIdentity deterministically from a 32-byte master
seed via two HKDF-SHA256 expansions with disjoint info strings:

  harmony-identity-ml-kem-v1 -> 64-byte ML-KEM-768 seed
  harmony-identity-ml-dsa-v1 -> 32-byte ML-DSA-65 seed

The 64-byte and 32-byte intermediate seeds are then handed to the
ml_kem::from_seed and ml_dsa::from_seed primitives exposed publicly in
the previous Task 1/2 commits. Both expansions are infallible at these
fixed lengths (well under HKDF-SHA256's 8160-byte ceiling).

Tests mirror the classical-side coverage: determinism, info-string
disjointness (ml-kem first 32 bytes != ml-dsa 32 bytes), round-trip
via to_private_bytes/from_private_bytes, functional encrypt+decrypt,
and a golden-vector pin on the 16-byte address hash for the all-zero
seed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Cross-class disjointness + final verification

**Files:**
- Modify: `crates/harmony-identity/src/pq_identity.rs` (add cross-class disjointness test that operates on public `to_private_bytes` outputs of both `PrivateIdentity` and `PqPrivateIdentity`)

- [ ] **Step 1: Confirm the `to_private_bytes` layouts the cross-class test will index into**

Run: `grep -n -A 12 "fn to_private_bytes" crates/harmony-identity/src/identity.rs crates/harmony-identity/src/pq_identity.rs`

Expected — from the existing implementations:
- `identity.rs::to_private_bytes` returns `[u8; 64]` with the first 32 bytes being one secret and the last 32 bytes being the other (read the body to confirm which order — at minimum it's two 32-byte halves of the X25519 + Ed25519 secrets). Note the order in a comment for step 2.
- `pq_identity.rs::to_private_bytes` returns `Vec<u8>` of length `ml_kem::SK_LENGTH + ml_dsa::SK_LENGTH`, with the ML-KEM bytes appearing before the ML-DSA bytes (read the body to confirm).

If either layout disagrees with the comment in step 2's test body, swap the indexing and update the comment to match.

- [ ] **Step 2: Add the failing cross-class disjointness test**

Append inside the existing `#[cfg(test)] mod tests` block at the bottom of `crates/harmony-identity/src/pq_identity.rs`:

```rust
/// Same master seed → classical and PQ derive to fully disjoint key bytes.
/// Sanity check that the four info strings (ed25519, x25519, ml-kem, ml-dsa)
/// are all distinct in their effect on HKDF output. Operates on the public
/// to_private_bytes outputs so the test does not depend on private struct
/// fields.
#[test]
fn classical_and_pq_subkeys_are_cross_class_disjoint() {
    use crate::PrivateIdentity;
    let seed = [0x42u8; 32];
    let classical = PrivateIdentity::from_seed(&seed);
    let pq = PqPrivateIdentity::from_seed(&seed);

    let classical_bytes = classical.to_private_bytes();
    let pq_bytes = pq.to_private_bytes();

    // Layout (confirmed in step 1):
    //   classical_bytes: 64 bytes = [X25519 secret (32) || Ed25519 secret (32)]
    //   pq_bytes:        ml_kem::SK_LENGTH + ml_dsa::SK_LENGTH bytes =
    //                    [ML-KEM seed (64) || ML-DSA seed (32)]
    let classical_x_bytes  = &classical_bytes[..32];
    let classical_ed_bytes = &classical_bytes[32..];
    let pq_kem_bytes = &pq_bytes[..ml_kem::SK_LENGTH];
    let pq_dsa_bytes = &pq_bytes[ml_kem::SK_LENGTH..];

    assert_ne!(classical_ed_bytes, pq_dsa_bytes,
        "ed25519 and ml-dsa derived to same 32 bytes — info-string separation failed");
    assert_ne!(classical_x_bytes, pq_dsa_bytes,
        "x25519 and ml-dsa derived to same 32 bytes — info-string separation failed");
    assert_ne!(classical_ed_bytes, &pq_kem_bytes[..32],
        "ed25519 and ml-kem (first 32B) derived to same bytes — info-string separation failed");
    assert_ne!(classical_x_bytes, &pq_kem_bytes[..32],
        "x25519 and ml-kem (first 32B) derived to same bytes — info-string separation failed");
}
```

If step 1 found that `identity.rs::to_private_bytes` puts Ed25519 first and X25519 second, swap the two `classical_*_bytes` slice indices accordingly and update the layout comment. Same swap rule for the PQ side.

- [ ] **Step 3: Run the cross-class test to verify it passes**

The test does not need a separate "first verify it fails" step: the implementations it depends on (`PrivateIdentity::from_seed`, `PqPrivateIdentity::from_seed`) already landed in Tasks 3 and 4, so the test compiles immediately. Either it passes (info strings really are disjoint) or it fails (a copy-paste bug in the constants made two info strings equal).

Run: `cargo test -p harmony-identity --lib classical_and_pq_subkeys_are_cross_class_disjoint`
Expected: PASS. If FAIL, re-check the four info-string constants — the most likely cause is two of them being byte-equal, which would mean classical and PQ derive to the same key from the same seed.

- [ ] **Step 4: Run the full `harmony-crypto` and `harmony-identity` test suites**

Run: `cargo test -p harmony-crypto -p harmony-identity`
Expected: All tests pass. Test count rises by 4 in harmony-crypto (2 per ml_kem + ml_dsa) and 11 in harmony-identity (5 classical + 5 PQ + 1 cross-class).

- [ ] **Step 5: Run clippy with warnings as errors**

Run: `cargo clippy -p harmony-crypto -p harmony-identity --all-targets -- -D warnings`
Expected: clean exit. No clippy warnings.

- [ ] **Step 6: Run rustdoc**

Run: `cargo doc -p harmony-crypto -p harmony-identity --no-deps`
Expected: clean exit. No rustdoc warnings.

- [ ] **Step 7: Commit the cross-class test**

```bash
git add crates/harmony-identity/src/pq_identity.rs
git commit -m "test(identity): cross-class disjointness for from_seed (ZEB-177)

Verifies that the four info strings (ed25519, x25519, ml-kem, ml-dsa)
all produce mutually distinct HKDF outputs from the same master seed.
Catches a regression where two info strings were copy-paste-equal
(would not be caught by the per-class disjointness tests).

Uses public to_private_bytes layouts for both classical and PQ — no
reliance on private struct fields — so the test stays correct under
internal refactors.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Definition of done (re-verified per spec)

After Task 5 completes, all six DoD items from the spec hold:

1. ✓ `harmony-crypto::ml_kem::from_seed` and `ml_dsa::from_seed` are public (Tasks 1, 2).
2. ✓ `harmony-crypto::ml_kem::generate` and `ml_dsa::generate` refactored to delegate (Tasks 1, 2). Existing tests still pass.
3. ✓ `harmony-identity::PrivateIdentity::from_seed(&[u8; 32])` (Task 3) and `PqPrivateIdentity::from_seed(&[u8; 32])` (Task 4) added.
4. ✓ All 5 test classes pass for both classical and PQ paths: determinism (Task 3 step 5, Task 4 step 5), disjointness (per-class steps + Task 5 cross-class), round-trip (Task 3 step 5, Task 4 step 5), functional (Task 3 step 5, Task 4 step 5), golden vector (Task 3 step 7, Task 4 step 7).
5. ✓ Clippy clean (Task 5 step 6).
6. ✓ Rustdoc clean (Task 5 step 7).

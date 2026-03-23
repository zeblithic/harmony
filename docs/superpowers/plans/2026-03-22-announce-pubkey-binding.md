# Announce Pubkey→Hash Binding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-3bu`.

**Goal:** Close the V1 announce cache poisoning vulnerability by including the ML-KEM encryption key in AnnounceRecords, enabling verifiers to re-derive `address_hash = SHA256(enc_pub || sign_pub)[:16]` and reject forged announces.

**Architecture:** Three changes in `harmony-discovery`: (1) add `encryption_key` field to `AnnounceRecord`, `SignablePayload`, and `AnnounceBuilder`, (2) add binding check in `verify_announce()`, (3) bump FORMAT_VERSION. Then update callers in `harmony-node` to provide the encryption key and downgrade security warnings.

**Tech Stack:** Rust, `harmony-discovery` (record.rs, verify.rs), `harmony-crypto` (truncated_hash), `harmony-node` (runtime.rs)

**Spec:** `docs/superpowers/specs/2026-03-22-announce-pubkey-binding-design.md`

---

## File Structure

```
crates/harmony-discovery/src/
├── record.rs     — Add encryption_key to AnnounceRecord, SignablePayload, AnnounceBuilder
└── verify.rs     — Add address re-derivation check after signature verification

crates/harmony-node/src/
└── runtime.rs    — Update pubkey_cache comments, provide encryption_key in announce building
```

---

### Task 1: Add encryption_key to AnnounceRecord and SignablePayload

**Files:**
- Modify: `crates/harmony-discovery/src/record.rs`

- [ ] **Step 1: Add encryption_key field to AnnounceRecord**

```rust
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,
    pub public_key: Vec<u8>,
    pub encryption_key: Vec<u8>,   // NEW: ML-KEM-768 public key (1184 bytes)
    pub routing_hints: Vec<RoutingHint>,
    pub published_at: u64,
    pub expires_at: u64,
    pub nonce: [u8; 16],
    pub signature: Vec<u8>,
}
```

- [ ] **Step 2: Add encryption_key to SignablePayload**

```rust
struct SignablePayload {
    format_version: u8,
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    encryption_key: Vec<u8>,   // NEW
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}
```

- [ ] **Step 3: Update signable_bytes() to include encryption_key**

```rust
pub(crate) fn signable_bytes(&self) -> Vec<u8> {
    let payload = SignablePayload {
        format_version: FORMAT_VERSION,
        identity_ref: self.identity_ref,
        public_key: self.public_key.clone(),
        encryption_key: self.encryption_key.clone(),  // NEW
        routing_hints: self.routing_hints.clone(),
        published_at: self.published_at,
        expires_at: self.expires_at,
        nonce: self.nonce,
    };
    // ...
}
```

- [ ] **Step 4: Add encryption_key to AnnounceBuilder**

Update constructor, field list, `signable_payload()`, and `build()`:

```rust
pub struct AnnounceBuilder {
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    encryption_key: Vec<u8>,   // NEW
    routing_hints: Vec<RoutingHint>,
    // ...
}

pub fn new(
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    encryption_key: Vec<u8>,   // NEW
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
) -> Self { ... }
```

Include `encryption_key` in `signable_payload()` (via SignablePayload) and `build()` (copy to AnnounceRecord).

- [ ] **Step 5: Bump FORMAT_VERSION from 2 to 3**

```rust
const FORMAT_VERSION: u8 = 3;
```

- [ ] **Step 6: Update ALL existing tests**

Every test that constructs `AnnounceRecord` directly or via `AnnounceBuilder::new()` needs the new `encryption_key` field. Search both `record.rs` and `verify.rs` test modules.

For tests that don't care about the encryption key, use `vec![0u8; 1184]` (dummy ML-KEM public key — wrong content but right length for postcard serialization).

For the binding test, use real PQ key material from `harmony_identity::PqPrivateIdentity::generate()`.

- [ ] **Step 7: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-discovery`

- [ ] **Step 8: Commit**

```bash
git commit -m "feat(discovery): add encryption_key to AnnounceRecord for pubkey→hash binding"
```

---

### Task 2: Add address binding check in verify_announce()

**Files:**
- Modify: `crates/harmony-discovery/src/verify.rs`
- Modify: `crates/harmony-discovery/src/error.rs` (remove dead_code allow on AddressMismatch)

- [ ] **Step 1: Write test for binding check**

```rust
#[test]
fn forged_announce_with_wrong_pubkey_rejected() {
    use rand::rngs::OsRng;

    // Generate two PQ identities
    let real_owner = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);
    let attacker = harmony_identity::PqPrivateIdentity::generate(&mut OsRng);

    let real_pub = real_owner.public_identity();
    let attacker_pub = attacker.public_identity();

    // Build an announce with the real owner's identity_hash
    // but the ATTACKER's public keys — and signed by the attacker
    let builder = AnnounceBuilder::new(
        IdentityRef { hash: real_pub.address_hash, suite: CryptoSuite::MlDsa65 },
        attacker_pub.verifying_key.as_bytes(),       // attacker's signing key
        attacker_pub.encryption_key.as_bytes(),       // attacker's encryption key
        1000,
        2000,
        [0u8; 16],
    );
    let payload = builder.signable_payload();
    let sig = attacker.sign(&payload).unwrap();
    let record = builder.build(sig);

    // Signature is valid (attacker signed with their own key)
    // But address binding should fail: hash(attacker_keys) != real_owner.address_hash
    let result = verify_announce(&record, 1500);
    assert_eq!(result.unwrap_err(), DiscoveryError::AddressMismatch);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-discovery forged_announce`
Expected: FAIL — AddressMismatch check not yet implemented (test passes signature check, then gets accepted).

- [ ] **Step 3: Implement binding check in verify_announce()**

In `verify.rs`, after the signature verification succeeds, add:

```rust
// Verify that the included public keys derive to the claimed identity hash.
// This prevents forged announces where an attacker substitutes their own keys
// for a victim's identity address.
let mut combined = Vec::with_capacity(
    record.encryption_key.len() + record.public_key.len()
);
combined.extend_from_slice(&record.encryption_key);
combined.extend_from_slice(&record.public_key);
let derived_hash = harmony_crypto::hash::truncated_hash(&combined);
if derived_hash != record.identity_ref.hash {
    return Err(DiscoveryError::AddressMismatch);
}
```

Note: `harmony_crypto::hash::truncated_hash()` computes `SHA256(data)[:16]` — check if `harmony-crypto` is already a dependency of `harmony-discovery`. If not, add it. Alternatively, compute `SHA256` via `sha2::Sha256` directly (already available via `harmony-identity`'s verify path).

Actually, check if `harmony-discovery` depends on `harmony-crypto`. If not, the simplest approach: add a `pub fn verify_address_binding(encryption_key: &[u8], signing_key: &[u8], expected_hash: &[u8; 16]) -> bool` function in `harmony-identity` (which already has the hash computation) and call it from `verify.rs`.

- [ ] **Step 4: Remove `#[allow(dead_code)]` from AddressMismatch**

In `error.rs`, remove the `#[allow(dead_code)]` annotation on `AddressMismatch` — it's now used.

- [ ] **Step 5: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-discovery`

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(discovery): verify pubkey→hash binding in verify_announce

Rejects forged announces where the public keys don't derive to the
claimed identity_ref.hash. DiscoveryError::AddressMismatch is no
longer dead code."
```

---

### Task 3: Update harmony-node callers and downgrade security warnings

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Update pubkey_cache security comments**

Replace the `SECURITY(V1)` warnings with informational notes:

```rust
    // Cache of ML-DSA public keys by identity hash.
    // Populated from HandshakeComplete and AnnounceRecord events.
    // Capped at MAX_PUBKEY_CACHE_SIZE entries — evicts an arbitrary entry
    // on overflow (not LRU).
    // Announce records are verified for pubkey→hash binding by
    // verify_announce() before reaching this point — forged announces
    // are rejected with DiscoveryError::AddressMismatch.
    pubkey_cache: HashMap<[u8; 16], Vec<u8>>,
```

Also update the comment in `process_discovered_tunnel_hints`.

- [ ] **Step 2: Update any test that constructs AnnounceRecord directly**

Search `runtime.rs` tests for `AnnounceRecord {` and add `encryption_key: vec![]` or appropriate values.

- [ ] **Step 3: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node`

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(node): downgrade pubkey_cache security warnings (binding now enforced)"
```

---

### Task 4: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-discovery -p harmony-node`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: clippy fixes for announce pubkey binding"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Add `encryption_key` to AnnounceRecord/Builder/SignablePayload | New field, FORMAT_VERSION 3 |
| 2 | Address binding check in `verify_announce()` | `AddressMismatch` error on forged announces |
| 3 | Update harmony-node callers and downgrade warnings | Security comments updated |
| 4 | Cleanup | Clippy clean, workspace tests pass |

**Security impact:** After this change, forged announces with substituted public keys are rejected at verification time. The `pubkey_cache` in harmony-node is safe from cache poisoning via discovery announces.

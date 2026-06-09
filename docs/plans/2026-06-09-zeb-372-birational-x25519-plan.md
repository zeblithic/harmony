# ZEB-372: Real birational X25519 in PubKeyBundle — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the zeroed `x25519_pub: [0u8; 32]` stub in every production `PubKeyBundle` with the RFC-7748 birational map of the bundle's Ed25519 verify key, so owner/device bundles carry a usable encryption key (unblocks ZEB-418 SP2 Butler sealed deposits and ZEB-369 targeted invites).

**Architecture:** One new pure helper in `harmony-owner` (`ed25519_pub_to_x25519`, ported from harmony-client `dm_signing.rs:136-156`), called from the three production constructors (`master_pubkey_bundle_from_sk`, the `mint_owner` device bundle, `PubKeyBundle::classical_only`). The matching private key is `clamp(sk.to_scalar_bytes())` — already shipped client-side as `ed25519_priv_to_x25519` — so nothing new is stored; the X25519 secret is derivable on demand from the Ed25519 signing key. `identity_hash()` excludes `x25519_pub` (pinned by the existing `x25519_rotation_does_not_change_identity_hash` test at `pubkey_bundle.rs:109-121`), so **no identity migration**. A second, later PR re-pins harmony-client (separate plan section, blocked on this PR merging).

**Tech stack:** Rust, `curve25519-dalek =4.1.3` (new dep, exact-pinned to match harmony-client), `ed25519-dalek 2` (already present).

**Branch:** `zeb-372-birational-x25519` off `origin/main` `1ccf42b`. **Gates (no CI in this repo — local gates are the gate):** `cargo fmt --all -- --check && cargo clippy --workspace --locked -- -D warnings && cargo test --workspace --locked`.

**Verified ground truth (recon 2026-06-09, zero drift from ticket):**
- `crates/harmony-owner/src/lifecycle/mint.rs:49-57` — `master_pubkey_bundle_from_sk` zeroes x25519 (TODO comment at :53).
- `crates/harmony-owner/src/lifecycle/mint.rs:88-94` — `mint_owner` device bundle zeroes x25519.
- `crates/harmony-owner/src/pubkey_bundle.rs:25-38` — `classical_only` zeroes x25519 ("stub" doc comment).
- `crates/harmony-owner/src/pubkey_bundle.rs:40-67` — `identity_hash()` hashes a `SigningMaterial` struct containing only `ed25519_verify` + `ml_dsa_verify`; x25519 excluded.
- `crates/harmony-owner/tests/interop_fixtures.rs:40` — `EXPECTED_ENROLLMENT_CERT_V1_HEX` pins cert CBOR **including the zeroed x25519** → this fixture WILL change; regenerate intentionally via its env-var gate.
- harmony root `Cargo.toml` has `ed25519-dalek = { version = "2", ... }` (line ~80), **no** `curve25519-dalek`.
- Test-only struct-literal bundles ([1u8;32] etc. in enroll_master.rs, enroll_quorum.rs, state.rs, trust.rs, certs/*, tests/e2e_*) are NOT touched — they construct literals, not via the production constructors, and fake ed25519 bytes may not decompress.

---

### Task 1: Birational conversion helper in harmony-owner

**Files:**
- Create: `crates/harmony-owner/src/x25519.rs`
- Modify: `crates/harmony-owner/src/lib.rs` (add `pub mod x25519;`)
- Modify: `crates/harmony-owner/Cargo.toml` + workspace `Cargo.toml` (add `curve25519-dalek = "=4.1.3"`)

- [ ] **Step 1: Add the dependency.** In the workspace `Cargo.toml` `[workspace.dependencies]` (follow how ed25519-dalek is declared; if deps are per-crate, put it in harmony-owner's Cargo.toml directly): `curve25519-dalek = "=4.1.3"`. Exact-pin matches harmony-client (`=4.1.3` at its Cargo.toml:128) so the two repos can never drift on curve arithmetic.

- [ ] **Step 2: Write the failing tests** in `crates/harmony-owner/src/x25519.rs`:

```rust
//! RFC 7748 §5 birational map: Ed25519 verify key → X25519 public key.
//!
//! ZEB-372. The matching PRIVATE key is `clamp(SigningKey::to_scalar_bytes())`
//! (shipped in harmony-client as `ed25519_priv_to_x25519`, dm_signing.rs) —
//! derivable on demand from the Ed25519 signing key, never stored. This module
//! must stay byte-compatible with harmony-client `ed25519_pub_to_x25519`
//! (dm_signing.rs:136-156): same decompress → small-order reject → Montgomery.

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    /// THE load-bearing parity test: the pub-side birational map must equal
    /// the public key of the priv-side clamped scalar (what harmony-client
    /// decrypts with). If these diverge, sealed deposits are undecryptable.
    #[test]
    fn pub_conversion_matches_priv_derived_public() {
        for seed in [[0x42u8; 32], [0x01u8; 32], [0xEEu8; 32]] {
            let sk = SigningKey::from_bytes(&seed);
            let via_pub = ed25519_pub_to_x25519(&sk.verifying_key().to_bytes())
                .expect("valid key converts");
            // mul_base_clamped applies RFC 7748 clamping internally — identical
            // to harmony-client's ed25519_priv_to_x25519 + X25519 base-point mul.
            let via_priv = curve25519_dalek::montgomery::MontgomeryPoint::mul_base_clamped(
                sk.to_scalar_bytes(),
            )
            .to_bytes();
            assert_eq!(via_pub, via_priv, "birational pub/priv parity (seed {:02x})", seed[0]);
        }
    }

    /// Reference vector, pinned. DO NOT REGENERATE casually: changing this
    /// derivation orphans every sealed blob addressed to existing bundles
    /// (mirrors the harmony-pkarr derive.rs case-vector discipline).
    /// Fill EXPECTED at implementation time from the first verified run of
    /// the parity test above, then treat as frozen.
    #[test]
    fn reference_vector_seed_42() {
        let sk = SigningKey::from_bytes(&[0x42u8; 32]);
        let x = ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).unwrap();
        const EXPECTED: &str = "<hex of x, filled at implementation time>";
        assert_eq!(hex::encode(x), EXPECTED, "ZEB-372 frozen derivation vector");
    }

    /// Small-order / invalid inputs must be rejected, mirroring the client's
    /// C2 torsion check.
    #[test]
    fn rejects_invalid_and_small_order_points() {
        assert!(ed25519_pub_to_x25519(&[0xFFu8; 32]).is_none(), "non-canonical point");
        // Compressed identity point (small order).
        let mut identity = [0u8; 32];
        identity[0] = 1;
        assert!(ed25519_pub_to_x25519(&identity).is_none(), "small-order point");
    }
}
```

(If `hex` isn't already a harmony-owner dev-dependency, add `hex = "0.4"` to `[dev-dependencies]` or pin the vector as a `[u8; 32]` literal instead — match whichever style `derive.rs` case vectors use.)

- [ ] **Step 3: Run tests to verify they fail** (module/function don't exist yet): `cargo test -p harmony-owner x25519 --locked` → compile error, as expected.

- [ ] **Step 4: Implement** (port of harmony-client `dm_signing.rs:136-156`, `Option` instead of the client's error enum):

```rust
/// Convert an Ed25519 verify key to its birational X25519 public key
/// (RFC 7748 §5). Returns `None` for non-canonical or small-order (torsion)
/// points — callers constructing bundles from freshly generated keys may
/// `expect()`; callers handling external bytes must propagate the `None`.
pub fn ed25519_pub_to_x25519(ed25519_pub: &[u8; 32]) -> Option<[u8; 32]> {
    use curve25519_dalek::edwards::CompressedEdwardsY;
    let edwards = CompressedEdwardsY(*ed25519_pub).decompress()?;
    if edwards.is_small_order() {
        return None;
    }
    Some(edwards.to_montgomery().to_bytes())
}
```

- [ ] **Step 5: Fill the reference vector.** Run `cargo test -p harmony-owner x25519 --locked`; parity + rejection tests pass; take the hex from the `reference_vector_seed_42` failure output, paste into `EXPECTED`, re-run → all pass.

- [ ] **Step 6: Commit.** `git add -A && git commit -m "feat(zeb-372): birational ed25519→x25519 helper in harmony-owner"`

### Task 2: Populate x25519_pub in the three production constructors

**Files:**
- Modify: `crates/harmony-owner/src/lifecycle/mint.rs:49-57, 88-94`
- Modify: `crates/harmony-owner/src/pubkey_bundle.rs:25-38`

- [ ] **Step 1: Write the failing test** (in `pubkey_bundle.rs` tests module):

```rust
#[test]
fn classical_only_populates_birational_x25519() {
    let sk = ed25519_dalek::SigningKey::from_bytes(&[0x07u8; 32]);
    let vk = sk.verifying_key().to_bytes();
    let bundle = PubKeyBundle::classical_only(vk);
    assert_eq!(
        bundle.classical.x25519_pub,
        crate::x25519::ed25519_pub_to_x25519(&vk).unwrap(),
        "classical_only must carry the birational X25519, not zeros"
    );
    assert_ne!(bundle.classical.x25519_pub, [0u8; 32]);
}
```

- [ ] **Step 2: Run to verify it fails:** `cargo test -p harmony-owner classical_only_populates --locked` → assertion failure (zeros).

- [ ] **Step 3: Implement.** `master_pubkey_bundle_from_sk` (mint.rs:49-57) — replace the stub line and delete the TODO comment:

```rust
pub(crate) fn master_pubkey_bundle_from_sk(sk: &SigningKey) -> PubKeyBundle {
    PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: sk.verifying_key().to_bytes(),
            x25519_pub: crate::x25519::ed25519_pub_to_x25519(&sk.verifying_key().to_bytes())
                .expect("freshly derived ed25519 verify key is a valid non-small-order point"),
        },
        post_quantum: None,
    }
}
```

`mint_owner` device bundle (mint.rs:88-94): same change with `device_sk`. `classical_only` (pubkey_bundle.rs:25-38): derive internally from the `ed25519_verify` parameter with the same `expect` + update its doc comment (drop the "X25519 stub" sentence; document the input-must-be-a-valid-own-key contract: callers are VouchingCert/LivenessCert `sign()` passing their own keys).

- [ ] **Step 4: Run the crate tests:** `cargo test -p harmony-owner --locked`. Expected: the new tests pass; `x25519_rotation_does_not_change_identity_hash` still passes (identity unaffected); **`master_enrollment_cert_v1_is_deterministic` / `EXPECTED_ENROLLMENT_CERT_V1_HEX` FAILS** — that is correct and intentional (the cert wire bytes now carry a real key).

- [ ] **Step 5: Regenerate the interop fixture intentionally** using its documented env-var gate (see `tests/recovery_wire_format_fixture.rs:94-100` / `interop_fixtures.rs` header for the exact variable), update the pinned hex, and add a comment line: `// ZEB-372 (2026-06-09): x25519_pub now birational-real; identity_hash unchanged (excludes x25519).` Re-run `cargo test -p harmony-owner --locked` → all green. If any OTHER pinned fixture fails, STOP and report (do not regenerate fixtures the plan didn't predict).

- [ ] **Step 6: Commit.** `git commit -am "feat(zeb-372): populate birational x25519_pub in master/device/classical_only bundles"`

### Task 3: Full-workspace gates + PR readiness

- [ ] **Step 1:** `cargo fmt --all -- --check` (run `cargo fmt --all` first if needed, then re-check).
- [ ] **Step 2:** `cargo clippy --workspace --locked -- -D warnings`.
- [ ] **Step 3:** `cargo test --workspace --locked` — full workspace, catches consumers of the constructors outside harmony-owner.
- [ ] **Step 4:** Commit any gate fallout as its own commit; re-run the failed gate. (Commit BEFORE running long gates so no work is lost.)

### Phase 2 (separate PR, AFTER this PR merges — harmony-client re-pin; for reference, executed later)

1. Bump all 8 `rev = "1ccf42bf…"` pins in `harmony-client/src-tauri/Cargo.toml` to the merge SHA; `cargo update` the harmony crates.
2. New integration test: mint an owner, build an `EnrollmentCert`, `seal_to_owner(cert.device_pubkeys.classical.x25519_pub, msg)` → `open_from_owner(ed25519_priv_to_x25519(device_sk), sealed)` round-trips — the cross-repo proof the whole ticket exists for.
3. Parity pin: assert `cert.classical.x25519_pub == dm_signing::ed25519_pub_to_x25519(&cert.classical.ed25519_verify)` (the two repos' implementations agree forever).
4. Carry the PR #218 CodeRabbit nitpick: add a plain-string `"superseded"` rejection case to `src/lib/notes-migrate.test.ts` (production Tauri rejects with strings, not Error objects).
5. Full harmony-client gates (fmt/clippy/nextest --all-targets/tsc/vitest) + PR + bot loop.

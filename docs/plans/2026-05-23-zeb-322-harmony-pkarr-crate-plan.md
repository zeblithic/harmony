# ZEB-322 Phase 2a: harmony-pkarr crate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new transport-agnostic `harmony-pkarr` crate in `harmony-core` that publishes and resolves BEP44 signed records to the BitTorrent Mainline DHT via HTTP relays, with HKDF-derived ephemeral Ed25519 keys. Becomes the foundation harmony-client's three case-specific policies (invite, identity, community) will layer on in Phase 2b (ZEB-323).

**Architecture:** Single crate with 7 source files + 1 test fixture module + 1 wire-format pin test. No iroh / zenoh / harmony-client deps — keeps it reusable. HTTP-relay-only backend (no embedded DHT). Stateless relay client with rotation + cooldown. In-memory LRU cache on the resolver. Mock pkarr-relay axum server shipped behind `test-fixtures` feature for downstream integration tests.

**Tech Stack:** Rust (workspace edition matches harmony-core defaults), `pkarr` crate (BEP44 envelope), `hkdf` + `sha2` (key derivation), `ed25519-dalek` (signatures), `serde` + `ciborium` (canonical CBOR — match harmony-discovery's CBOR choice if it has one; otherwise use ciborium which is the harmony-crypto convention), `axum` + `tokio` (mock relay), `tracing` (logging).

**Spec:** `harmony-client/docs/specs/2026-05-23-zeb-321-phase2-discovery-bootstrap-design.md` (commit `cb5cca5`), Sections 4.1, 5, 6, 7, 9, 13.1/13.3/13.4. **Linear:** ZEB-322 (parent: ZEB-321). **Branch:** `zeb-322-harmony-pkarr-crate` (already created off `origin/main` `04449d6`).

**HARD RULES from user memory (every implementer subagent prompt must enforce):**
- Quality gates from harmony root: `cargo fmt --all -- --check`, `cargo clippy --workspace --all-targets -- -D warnings`, `cargo nextest run --workspace --all-targets --features test-fixtures` (or `cargo test --workspace --no-fail-fast` if nextest not installed)
- Pipe exit codes: `set -o pipefail` or `${PIPESTATUS[0]}` when piping cargo output through `tail`/`grep`
- Implementer subagent gates per `feedback_implementer_gate_time_budget`: commit-before-gate + 10-min Bash wall-clock kill switch + DONE_WITH_CONCERNS escape hatch
- Long-running background per `feedback_long_running_background_supervision`: any cargo command potentially > 10 min MUST use foreground timeout (`timeout 600 cargo ...`) or ScheduleWakeup heartbeat
- `cargo fmt` MUST be in implementer verification, not just clippy (`feedback_cargo_fmt_gate`)
- No worktrees; `git checkout -b` only (`feedback_no_worktrees`)
- Pre-existing test failures in the workspace baseline are NOT blocking; new failures are blocking (`feedback_test_drift_is_our_fault`)
- Second-order correctness review (`feedback_second_order_correctness_review`): when fixing a bug, ask "does my fix introduce a new invariant violation?"

## File Structure

```
crates/harmony-pkarr/
├─ Cargo.toml                                            # NEW
├─ src/
│  ├─ lib.rs                                             # NEW — public API re-exports
│  ├─ error.rs                                           # NEW — PkarrError enum (mirror harmony-discovery style: manual Display + std::error::Error gated)
│  ├─ epoch.rs                                           # NEW — week-aligned epoch math
│  ├─ derive.rs                                          # NEW — HKDF-SHA256 key derivation per case
│  ├─ record.rs                                          # NEW — PkarrRoutingRecord + canonical CBOR + inner-sig verify
│  ├─ relay.rs                                           # NEW — HTTP pkarr-relay client (rotation, cooldown, retry)
│  ├─ publisher.rs                                       # NEW — PkarrPublisher background task
│  ├─ resolver.rs                                        # NEW — PkarrResolver with LRU cache + parallel epoch-window queries
│  └─ testing.rs                                         # NEW — #[cfg(any(test, feature = "test-fixtures"))] mock relay (axum)
└─ tests/
   └─ wire_format_pkarr_routing_record_fixtures.rs      # NEW — canonical CBOR pins for each case

Cargo.toml                                              # MODIFY — add `crates/harmony-pkarr` to workspace.members; add `pkarr` to workspace.dependencies
```

---

## Task 0: Pre-flight baseline (no commit)

**Purpose:** Establish a known-green baseline before introducing new code. Verify branch state. Capture any pre-existing test failures so they're not mistaken for regressions introduced by this PR.

- [ ] **Step 1: Verify branch state**

Run from harmony repo root:
```bash
git status --short                  # should show no staged/unstaged changes
git rev-parse HEAD                  # should be at origin/main (04449d6 or later)
git rev-parse --abbrev-ref HEAD     # should be zeb-322-harmony-pkarr-crate
```

- [ ] **Step 2: Capture workspace test baseline**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run --workspace --no-fail-fast 2>&1 | tee /tmp/zeb-322-baseline-test.log
```

If nextest not installed: `cargo test --workspace --no-fail-fast`. Record the count of failing tests (if any). These are pre-existing and NOT blocking for this PR. Save the list in `/tmp/zeb-322-baseline-failures.txt`.

- [ ] **Step 3: Verify cargo fmt + clippy currently pass on workspace**

```bash
timeout 300 cargo fmt --all -- --check
timeout 600 cargo clippy --workspace --all-targets -- -D warnings 2>&1 | tail -50
```

Both must pass before adding new code. If they fail on main, STOP and flag — that's a pre-existing problem to escalate before proceeding.

- [ ] **Step 4: Verify pkarr crate version exists on crates.io**

```bash
cargo search pkarr --limit 1
```

Expected: `pkarr = "6.0.0"` or later. Note the exact version for Cargo.toml pinning in Task 1.

**No commit for Task 0.**

---

## Task 1: Crate skeleton

**Purpose:** Stand up the empty crate with workspace integration. Verifies workspace.members + workspace.dependencies wiring before adding any real code.

**Files:**
- Create: `crates/harmony-pkarr/Cargo.toml`
- Create: `crates/harmony-pkarr/src/lib.rs`
- Create: `crates/harmony-pkarr/src/error.rs`
- Modify: `Cargo.toml` (workspace root) — add `crates/harmony-pkarr` to `workspace.members` (alphabetical position); add `pkarr = "6"` (or current version from Task 0) to `workspace.dependencies`

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/error.rs`:
```rust
//! Error types for harmony-pkarr operations.
//!
//! Mirrors harmony-discovery's manual Display + std::error::Error pattern
//! (no thiserror dep at the harmony-core layer).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PkarrError {
    /// BEP44 outer signature failed verification. (RPK1)
    OuterSignatureInvalid,
    /// Inner identity signature failed verification. (RPK2)
    InnerSignatureInvalid,
    /// `harmony_identity_pub` does not match the expected identity.
    /// (RPK3)
    IdentityMismatch,
    /// `announced_at_ms` outside ±30 min skew window. (RPK4)
    StaleOrSkewed,
    /// `routing_blob` could not be decoded by the caller's parser. (RPK5)
    /// Returned only by callers that wrap harmony-pkarr; harmony-pkarr itself
    /// treats `routing_blob` as opaque bytes.
    RoutingBlobInvalid,
    /// Relay returned a non-success HTTP status.
    RelayHttpError(u16),
    /// All configured relays are on cooldown or unreachable.
    NoRelaysAvailable,
    /// Relay returned a malformed BEP44 envelope.
    RelayResponseInvalid,
    /// CBOR serialize/deserialize failure on PkarrRoutingRecord payload.
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for PkarrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OuterSignatureInvalid => write!(f, "BEP44 outer signature invalid"),
            Self::InnerSignatureInvalid => write!(f, "inner identity signature invalid"),
            Self::IdentityMismatch => write!(f, "harmony_identity_pub does not match expected identity"),
            Self::StaleOrSkewed => write!(f, "announced_at_ms outside ±30min skew"),
            Self::RoutingBlobInvalid => write!(f, "routing_blob could not be decoded"),
            Self::RelayHttpError(status) => write!(f, "relay returned HTTP {status}"),
            Self::NoRelaysAvailable => write!(f, "no relays available (all on cooldown or unreachable)"),
            Self::RelayResponseInvalid => write!(f, "relay returned malformed response"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PkarrError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_status_code() {
        let e = PkarrError::RelayHttpError(429);
        let s = format!("{}", e);
        assert!(s.contains("429"));
    }

    #[test]
    fn errors_are_comparable() {
        // PartialEq is load-bearing for #[derive(PartialEq)] on downstream types
        // that include PkarrError; verify the derive.
        assert_eq!(PkarrError::OuterSignatureInvalid, PkarrError::OuterSignatureInvalid);
        assert_ne!(PkarrError::OuterSignatureInvalid, PkarrError::InnerSignatureInvalid);
    }
}
```

Create `crates/harmony-pkarr/src/lib.rs`:
```rust
//! Pkarr (Public-Key Addressable Resource Records) publish + resolve over
//! BitTorrent Mainline DHT via HTTP-relay. Transport-agnostic at the value
//! layer — see [`record::PkarrRoutingRecord`].
//!
//! See `harmony-client/docs/specs/2026-05-23-zeb-321-phase2-discovery-bootstrap-design.md`
//! for the cohesive Phase 2 design this crate is the harmony-core half of.
//!
//! # Crates this crate intentionally does NOT depend on
//!
//! - `iroh` / `zenoh` — transport-agnostic at the value layer; downstream
//!   harmony-client wraps the opaque `routing_blob` with iroh-specific
//!   routing data.
//! - `harmony-client` — this crate ships in harmony-core for reuse by
//!   harmony-arch / harmony-glitch / etc.

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(feature = "std")]
extern crate alloc;

pub mod error;

pub use error::PkarrError;
```

Create `crates/harmony-pkarr/Cargo.toml`:
```toml
[package]
name = "harmony-pkarr"
description = "BEP44 pkarr publish + resolve over Mainline DHT via HTTP-relay; transport-agnostic value layer"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = ["std"]
std = []
# Exposes the mock pkarr-relay axum server for downstream integration tests.
test-fixtures = ["dep:axum", "dep:tokio", "dep:hyper"]

[dependencies]
# Add later as tasks pull them in:
# pkarr, hkdf, sha2, ed25519-dalek, serde, ciborium, tracing, lru, reqwest

# test-fixtures-only:
axum = { workspace = true, optional = true }
tokio = { workspace = true, optional = true }
hyper = { workspace = true, optional = true }

[dev-dependencies]
# Add later as tasks pull them in.
```

Modify workspace root `Cargo.toml`:
- Add `"crates/harmony-pkarr"` to `[workspace] members` in alphabetical position (between `harmony-owner` and `harmony-peers`)
- Add `pkarr = "6"` (or version from Task 0) to `[workspace.dependencies]` (alphabetical position)
- Verify `hyper` is already in `[workspace.dependencies]`; if not, add `hyper = { version = "1", default-features = false }`

- [ ] **Step 2: Run tests to verify they fail (then pass)**

Actually error.rs tests should pass on first run because the tests only exercise the enum + Display impl which are complete. Run:
```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 300 cargo nextest run -p harmony-pkarr 2>&1 | tail -20
# OR if nextest not installed:
# timeout 300 cargo test -p harmony-pkarr 2>&1 | tail -20
```

Expected: 2 tests pass.

- [ ] **Step 3: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets -- -D warnings
```

Both must pass.

- [ ] **Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add Cargo.toml crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/lib.rs crates/harmony-pkarr/src/error.rs
git commit -m "feat(zeb-322): harmony-pkarr crate skeleton + PkarrError"
```

---

## Task 2: Epoch math

**Purpose:** Pure-function week-aligned epoch math + tolerance window. Easiest pure unit to start with; no external deps. Locks the epoch semantics before publisher/resolver depend on them.

**Files:**
- Create: `crates/harmony-pkarr/src/epoch.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod epoch;` + re-exports

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/epoch.rs` with the test module first (TDD red phase):
```rust
//! Week-aligned epoch math.
//!
//! Epoch identifier = `floor(unix_ms / EPOCH_DURATION_MS)`.
//! Resolver queries `[epoch-1, epoch, epoch+1]` in parallel for ±1-day
//! clock-skew tolerance across the 7-day boundary.

/// 1 week in milliseconds.
pub const EPOCH_DURATION_MS: u64 = 7 * 86_400_000;

/// Returns the epoch identifier containing `now_ms`.
pub fn current_epoch_id(now_ms: u64) -> u64 {
    todo!("implement in step 3")
}

/// Returns the tolerance window `[prev, current, next]` epoch IDs.
/// Resolver queries all three keys in parallel.
pub fn epoch_tolerance_window(now_ms: u64) -> [u64; 3] {
    todo!("implement in step 3")
}

/// Wall-clock time (ms) when the given epoch starts.
pub fn epoch_start_ms(epoch_id: u64) -> u64 {
    todo!("implement in step 3")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_zero_starts_at_unix_epoch() {
        assert_eq!(current_epoch_id(0), 0);
    }

    #[test]
    fn epoch_advances_at_week_boundary() {
        let just_before = EPOCH_DURATION_MS - 1;
        let just_after = EPOCH_DURATION_MS;
        assert_eq!(current_epoch_id(just_before), 0);
        assert_eq!(current_epoch_id(just_after), 1);
    }

    #[test]
    fn tolerance_window_returns_prev_current_next() {
        let now = 5 * EPOCH_DURATION_MS + EPOCH_DURATION_MS / 2; // mid-epoch 5
        let window = epoch_tolerance_window(now);
        assert_eq!(window, [4, 5, 6]);
    }

    #[test]
    fn tolerance_window_saturates_at_zero() {
        // Inside epoch 0: prev should saturate to 0, not underflow.
        let now = EPOCH_DURATION_MS / 4; // ~1.75 days in
        let window = epoch_tolerance_window(now);
        assert_eq!(window, [0, 0, 1]);
    }

    #[test]
    fn epoch_start_is_inverse_of_current_epoch_id() {
        let now = 12345 * EPOCH_DURATION_MS + 1000;
        let e = current_epoch_id(now);
        let start = epoch_start_ms(e);
        assert_eq!(start, 12345 * EPOCH_DURATION_MS);
        assert!(now - start < EPOCH_DURATION_MS);
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod epoch;
pub use epoch::{current_epoch_id, epoch_start_ms, epoch_tolerance_window, EPOCH_DURATION_MS};
```

- [ ] **Step 2: Run test to verify they fail**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo nextest run -p harmony-pkarr epoch:: 2>&1 | tail -20
```

Expected: 5 tests FAIL on `todo!()`.

- [ ] **Step 3: Write the implementation**

Replace the `todo!()` bodies:
```rust
pub fn current_epoch_id(now_ms: u64) -> u64 {
    now_ms / EPOCH_DURATION_MS
}

pub fn epoch_tolerance_window(now_ms: u64) -> [u64; 3] {
    let e = current_epoch_id(now_ms);
    [e.saturating_sub(1), e, e.saturating_add(1)]
}

pub fn epoch_start_ms(epoch_id: u64) -> u64 {
    epoch_id * EPOCH_DURATION_MS
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo nextest run -p harmony-pkarr epoch:: 2>&1 | tail -20
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 300 cargo clippy -p harmony-pkarr --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/src/epoch.rs crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-322): week-aligned epoch math + tolerance window"
```

---

## Task 3: HKDF key derivation (per-case)

**Purpose:** Single HKDF-SHA256 derivation function with three case-specific salts. Locks the keying scheme via reference vectors before publisher/resolver consume them. Reference vectors prevent future refactors from silently changing the keys (which would invalidate every published record).

**Files:**
- Create: `crates/harmony-pkarr/src/derive.rs`
- Modify: `crates/harmony-pkarr/Cargo.toml` — add `hkdf`, `sha2`, `ed25519-dalek`, `zeroize` to `[dependencies]`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod derive;`

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/derive.rs`:
```rust
//! HKDF-SHA256 derivation of ephemeral Ed25519 keys per case.
//!
//! Domain separation: each case uses a distinct salt
//! (`harmony.pkarr.v1.{invite|identity|community}`) so reusing the same
//! `ikm` across cases does NOT produce the same derived key.
//!
//! Spec Section 5.3.

use ed25519_dalek::SigningKey;
use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;

/// Which Phase 2 case the derivation is for. Picks the salt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PkarrCase {
    /// Case A: invite-redemption. `ikm` = `invite_token.sig` (64 bytes).
    Invite,
    /// Case B: opt-in identity-keyed. `ikm` = `owner_identity_pub` (64 bytes).
    Identity,
    /// Case C: in-community fallback. `ikm` = `EpochKey` (32 bytes).
    Community,
}

impl PkarrCase {
    pub fn salt(self) -> &'static [u8] {
        match self {
            Self::Invite => b"harmony.pkarr.v1.invite",
            Self::Identity => b"harmony.pkarr.v1.identity",
            Self::Community => b"harmony.pkarr.v1.community",
        }
    }
}

/// Derive an ephemeral Ed25519 signing key from per-case input.
///
/// Both publisher and resolver call this with identical `(case, ikm, info)`
/// inputs and obtain identical keys. The publisher signs BEP44 records
/// under the resulting key; the resolver derives the corresponding verifying
/// key from `signing.verifying_key()` and queries the DHT under it.
pub fn derive_ephemeral_key(case: PkarrCase, ikm: &[u8], info: &[u8]) -> SigningKey {
    let hkdf = Hkdf::<Sha256>::new(Some(case.salt()), ikm);
    let mut seed = [0u8; 32];
    hkdf.expand(info, &mut seed)
        .expect("HKDF-SHA256 always produces 32 bytes for our 32-byte output");
    let key = SigningKey::from_bytes(&seed);
    seed.zeroize();
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference vector — pins the entire keying scheme. If this test breaks,
    /// every published pkarr record under the v1 scheme becomes irretrievable
    /// without a v2 migration. DO NOT regenerate the expected hex without
    /// understanding the consequences.
    #[test]
    fn reference_vector_case_invite() {
        // ikm = 64 zero bytes (placeholder invite_token.sig)
        // info = epoch_id 12345 in big-endian
        let ikm = [0u8; 64];
        let info = 12345u64.to_be_bytes();
        let key = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        // Pin: compute once, paste here. Regenerating breaks v1 records.
        assert_eq!(vk_hex.len(), 64, "Ed25519 verifying keys are 32 bytes = 64 hex chars");
        // Implementer: run the test once with `eprintln!("{}", vk_hex);` then
        // pin the value here. Mark with a TODO if you want; the next reviewer
        // will need to commit the actual hex string.
        let expected = "TODO_PIN_HEX_AFTER_FIRST_RUN";
        assert_eq!(vk_hex, expected, "case-invite v1 keying must not drift");
    }

    #[test]
    fn reference_vector_case_identity() {
        let ikm = [0u8; 64];
        let info = 12345u64.to_be_bytes();
        let key = derive_ephemeral_key(PkarrCase::Identity, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        let expected = "TODO_PIN_HEX_AFTER_FIRST_RUN";
        assert_eq!(vk_hex, expected, "case-identity v1 keying must not drift");
    }

    #[test]
    fn reference_vector_case_community() {
        let ikm = [0u8; 32];
        let mut info = [0u8; 72]; // 64 bytes member_pub + 8 bytes epoch_id
        info[64..].copy_from_slice(&12345u64.to_be_bytes());
        let key = derive_ephemeral_key(PkarrCase::Community, &ikm, &info);
        let vk_hex = hex::encode(key.verifying_key().to_bytes());
        let expected = "TODO_PIN_HEX_AFTER_FIRST_RUN";
        assert_eq!(vk_hex, expected, "case-community v1 keying must not drift");
    }

    #[test]
    fn different_cases_produce_different_keys() {
        let ikm = [1u8; 64];
        let info = [2u8; 8];
        let k1 = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info);
        let k2 = derive_ephemeral_key(PkarrCase::Identity, &ikm, &info);
        let k3 = derive_ephemeral_key(PkarrCase::Community, &ikm, &info);
        assert_ne!(k1.verifying_key(), k2.verifying_key());
        assert_ne!(k1.verifying_key(), k3.verifying_key());
        assert_ne!(k2.verifying_key(), k3.verifying_key());
    }

    #[test]
    fn different_info_produces_different_keys() {
        let ikm = [1u8; 64];
        let info_a = [2u8; 8];
        let info_b = [3u8; 8];
        let ka = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info_a);
        let kb = derive_ephemeral_key(PkarrCase::Invite, &ikm, &info_b);
        assert_ne!(ka.verifying_key(), kb.verifying_key());
    }
}
```

Modify `crates/harmony-pkarr/Cargo.toml` `[dependencies]`:
```toml
[dependencies]
hkdf = { workspace = true }
sha2 = { workspace = true, features = ["alloc"] }
ed25519-dalek = { workspace = true }
zeroize = { workspace = true }

# test-fixtures-only:
axum = { workspace = true, optional = true }
tokio = { workspace = true, optional = true }
hyper = { workspace = true, optional = true }

[dev-dependencies]
hex = { workspace = true }
```

Verify `hex` is in workspace.dependencies; if not, add `hex = { version = "0.4", default-features = false, features = ["alloc"] }`.

Add to `src/lib.rs`:
```rust
pub mod derive;
pub use derive::{derive_ephemeral_key, PkarrCase};
```

- [ ] **Step 2: First run — capture reference hex**

The reference-vector tests will FAIL (assertion vs `TODO_PIN_HEX_AFTER_FIRST_RUN`). Run once to capture the actual derived hex:
```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo nextest run -p harmony-pkarr derive:: --no-capture 2>&1 | grep -E "(left|right|TODO)" | head -10
```

Or temporarily replace each `expected = "TODO_..."` with `eprintln!("{}", vk_hex); return;` before the assertion to print and exit. After capture, paste the actual hex into each `expected` constant.

- [ ] **Step 3: Pin the reference hex**

Edit `crates/harmony-pkarr/src/derive.rs` — replace each `TODO_PIN_HEX_AFTER_FIRST_RUN` with the actual 64-char lowercase hex captured in Step 2. Verify all three are different (sanity check on case separation).

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo nextest run -p harmony-pkarr derive:: 2>&1 | tail -10
```

Expected: 5 tests PASS.

- [ ] **Step 5: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 300 cargo clippy -p harmony-pkarr --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/derive.rs crates/harmony-pkarr/src/lib.rs Cargo.toml
git commit -m "feat(zeb-322): HKDF-SHA256 per-case key derivation + reference vectors"
```

---

## Task 4: PkarrRoutingRecord wire format

**Purpose:** Define the BEP44 inner payload struct + canonical CBOR + inner-sig sign/verify path. Pins the wire format with a fixture test before publisher/resolver depend on it.

**Files:**
- Create: `crates/harmony-pkarr/src/record.rs`
- Create: `crates/harmony-pkarr/tests/wire_format_pkarr_routing_record_fixtures.rs`
- Modify: `crates/harmony-pkarr/Cargo.toml` — add `serde`, `ciborium` to `[dependencies]`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod record;`

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/record.rs`:
```rust
//! `PkarrRoutingRecord` — the opaque BEP44 inner payload.
//!
//! Spec Section 5.1. 2-char field keys per harmony convention.
//! Inner signature binds `(routing_blob, harmony_identity_pub,
//! announced_at_ms)` to the publisher's harmony Ed25519 identity key —
//! verified independently of the BEP44 outer (ephemeral) signature.

use ed25519_dalek::{Signature, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::error::PkarrError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PkarrRoutingRecord {
    /// Opaque routing blob. harmony-client encodes iroh routing here;
    /// harmony-pkarr treats as bytes.
    #[serde(rename = "rd", with = "serde_bytes")]
    pub routing_blob: alloc::vec::Vec<u8>,

    /// 64 bytes = X25519_pub(32) ‖ Ed25519_pub(32). The last 32 bytes are
    /// the Ed25519 verifying key used to verify `inner_sig`.
    #[serde(rename = "ip", with = "serde_bytes")]
    pub harmony_identity_pub: [u8; 64],

    /// Wall-clock publication time, ms since unix epoch.
    #[serde(rename = "at")]
    pub announced_at_ms: u64,

    /// Ed25519 sig over canonical-CBOR((routing_blob, harmony_identity_pub,
    /// announced_at_ms)) using the harmony identity Ed25519 key.
    #[serde(rename = "sg", with = "serde_bytes")]
    pub inner_sig: [u8; 64],
}

/// Maximum permitted skew between `announced_at_ms` and verifier's `now_ms`.
pub const SKEW_TOLERANCE_MS: u64 = 30 * 60 * 1000;

impl PkarrRoutingRecord {
    /// Build + inner-sign a record. `identity_signing_key` is the harmony
    /// identity Ed25519 key (NOT the ephemeral pkarr key — that one wraps
    /// this struct in the BEP44 envelope).
    pub fn sign_new(
        routing_blob: alloc::vec::Vec<u8>,
        harmony_identity_pub: [u8; 64],
        announced_at_ms: u64,
        identity_signing_key: &SigningKey,
    ) -> Result<Self, PkarrError> {
        let to_sign = canonical_signed_bytes(&routing_blob, &harmony_identity_pub, announced_at_ms)?;
        let sig = identity_signing_key.sign(&to_sign);
        Ok(Self {
            routing_blob,
            harmony_identity_pub,
            announced_at_ms,
            inner_sig: sig.to_bytes(),
        })
    }

    /// Verify the inner identity signature against the embedded
    /// `harmony_identity_pub`. RPK2 silent-drop on failure.
    pub fn verify_inner_sig(&self) -> Result<(), PkarrError> {
        let ed_pub_bytes: [u8; 32] = self.harmony_identity_pub[32..]
            .try_into()
            .map_err(|_| PkarrError::InvalidRecord())?;
        let vk = VerifyingKey::from_bytes(&ed_pub_bytes)
            .map_err(|_| PkarrError::InnerSignatureInvalid)?;
        let sig = Signature::from_bytes(&self.inner_sig);
        let to_verify =
            canonical_signed_bytes(&self.routing_blob, &self.harmony_identity_pub, self.announced_at_ms)?;
        vk.verify(&to_verify, &sig)
            .map_err(|_| PkarrError::InnerSignatureInvalid)
    }

    /// Check `announced_at_ms` is within ±SKEW_TOLERANCE_MS of `now_ms`.
    /// RPK4 silent-drop on failure.
    pub fn verify_skew(&self, now_ms: u64) -> Result<(), PkarrError> {
        let diff = self.announced_at_ms.abs_diff(now_ms);
        if diff > SKEW_TOLERANCE_MS {
            Err(PkarrError::StaleOrSkewed)
        } else {
            Ok(())
        }
    }

    /// Check `harmony_identity_pub` matches an expected identity.
    /// RPK3 silent-drop on failure. Used by callers verifying records they
    /// queried by identity (case B) or by community-member context (case C).
    /// Case A skips this since the inner-sig already binds to `admin_identity_pub`
    /// from the invite payload.
    pub fn verify_identity_match(&self, expected: &[u8; 64]) -> Result<(), PkarrError> {
        if &self.harmony_identity_pub != expected {
            Err(PkarrError::IdentityMismatch)
        } else {
            Ok(())
        }
    }

    /// Canonical CBOR encoding of self, suitable for embedding in a BEP44
    /// envelope payload field.
    pub fn to_canonical_cbor(&self) -> Result<alloc::vec::Vec<u8>, PkarrError> {
        let mut out = alloc::vec::Vec::new();
        ciborium::into_writer(self, &mut out)
            .map_err(|_| PkarrError::SerializeError("PkarrRoutingRecord"))?;
        Ok(out)
    }

    pub fn from_canonical_cbor(bytes: &[u8]) -> Result<Self, PkarrError> {
        ciborium::from_reader(bytes).map_err(|_| PkarrError::DeserializeError("PkarrRoutingRecord"))
    }
}

fn canonical_signed_bytes(
    routing_blob: &[u8],
    harmony_identity_pub: &[u8; 64],
    announced_at_ms: u64,
) -> Result<alloc::vec::Vec<u8>, PkarrError> {
    // Tuple-as-array: deterministic CBOR ordering. ciborium encodes tuples
    // as CBOR arrays.
    let mut out = alloc::vec::Vec::new();
    ciborium::into_writer(
        &(serde_bytes::Bytes::new(routing_blob), serde_bytes::Bytes::new(harmony_identity_pub.as_ref()), announced_at_ms),
        &mut out,
    )
    .map_err(|_| PkarrError::SerializeError("canonical_signed_bytes"))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_identity_pubkey(signing_key: &SigningKey) -> [u8; 64] {
        // Mimic harmony_identity::Identity::to_public_bytes() shape:
        // 32 zero bytes (X25519 placeholder) ‖ 32 bytes Ed25519 pub.
        let mut out = [0u8; 64];
        out[32..].copy_from_slice(&signing_key.verifying_key().to_bytes());
        out
    }

    #[test]
    fn round_trip_canonical_cbor() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"opaque-routing-blob".to_vec(), identity_pub, 1_000_000, &sk)
            .expect("sign");
        let cbor = rec.to_canonical_cbor().expect("encode");
        let decoded = PkarrRoutingRecord::from_canonical_cbor(&cbor).expect("decode");
        assert_eq!(rec, decoded);
    }

    #[test]
    fn verify_inner_sig_accepts_valid() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        assert!(rec.verify_inner_sig().is_ok());
    }

    #[test]
    fn verify_inner_sig_rejects_tampered_blob() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let mut rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        rec.routing_blob[0] ^= 1;
        assert_eq!(rec.verify_inner_sig(), Err(PkarrError::InnerSignatureInvalid));
    }

    #[test]
    fn verify_inner_sig_rejects_tampered_at() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let mut rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        rec.announced_at_ms += 1;
        assert_eq!(rec.verify_inner_sig(), Err(PkarrError::InnerSignatureInvalid));
    }

    #[test]
    fn verify_skew_accepts_within_window() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        assert!(rec.verify_skew(1_000_000 + SKEW_TOLERANCE_MS).is_ok());
        assert!(rec.verify_skew(1_000_000 - SKEW_TOLERANCE_MS).is_ok());
    }

    #[test]
    fn verify_skew_rejects_outside_window() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        assert_eq!(rec.verify_skew(1_000_000 + SKEW_TOLERANCE_MS + 1), Err(PkarrError::StaleOrSkewed));
        assert_eq!(rec.verify_skew(1_000_000 - SKEW_TOLERANCE_MS - 1), Err(PkarrError::StaleOrSkewed));
    }

    #[test]
    fn verify_identity_match_rejects_substitution() {
        let sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&sk);
        let rec = PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, 1_000_000, &sk).expect("sign");
        let mut wrong = identity_pub;
        wrong[32] ^= 1;
        assert_eq!(rec.verify_identity_match(&wrong), Err(PkarrError::IdentityMismatch));
        assert!(rec.verify_identity_match(&identity_pub).is_ok());
    }
}
```

Add to `crates/harmony-pkarr/Cargo.toml`:
```toml
[dependencies]
hkdf = { workspace = true }
sha2 = { workspace = true, features = ["alloc"] }
ed25519-dalek = { workspace = true }
zeroize = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
serde_bytes = { workspace = true }
ciborium = { workspace = true }

# (existing test-fixtures-only deps)
[dev-dependencies]
hex = { workspace = true }
rand = { workspace = true }
```

Verify `serde_bytes` and `ciborium` are in workspace.dependencies. If not, add:
- `serde_bytes = { version = "0.11", default-features = false, features = ["alloc"] }`
- `ciborium = { version = "0.2", default-features = false }`

Add to `src/lib.rs`:
```rust
pub mod record;
pub use record::{PkarrRoutingRecord, SKEW_TOLERANCE_MS};
```

Add to `src/error.rs`: a new variant `InvalidRecord` (used by `verify_inner_sig` for the length-conversion arm):
```rust
    /// Record structurally invalid (e.g., identity_pub wrong length).
    InvalidRecord,
```

And add to the Display impl: `Self::InvalidRecord => write!(f, "record structurally invalid"),`

- [ ] **Step 2: Run tests to verify they fail at compile then implement until passing**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run -p harmony-pkarr record:: 2>&1 | tail -20
```

Expected: 7 tests PASS (no `todo!()` in this task — implementation is inline with the test code).

- [ ] **Step 3: Add the wire-format pin test**

Create `crates/harmony-pkarr/tests/wire_format_pkarr_routing_record_fixtures.rs`:
```rust
//! Wire-format pin: canonical CBOR bytes of one `PkarrRoutingRecord`
//! produced with deterministic-test inputs.
//!
//! If this test breaks, the canonical CBOR encoding of `PkarrRoutingRecord`
//! has drifted — every published pkarr record under the v1 wire format
//! becomes undecodable without a v2 migration. Regenerate the expected hex
//! only with full understanding.

use ed25519_dalek::SigningKey;
use harmony_pkarr::PkarrRoutingRecord;

#[test]
fn canonical_cbor_bytes_pinned() {
    // Deterministic inputs.
    let sk_bytes = [42u8; 32];
    let sk = SigningKey::from_bytes(&sk_bytes);
    let mut identity_pub = [0u8; 64];
    identity_pub[32..].copy_from_slice(&sk.verifying_key().to_bytes());
    let rec = PkarrRoutingRecord::sign_new(
        b"deterministic-routing-blob".to_vec(),
        identity_pub,
        1_700_000_000_000u64, // fixed wall-clock
        &sk,
    )
    .expect("sign");

    let cbor = rec.to_canonical_cbor().expect("encode");
    let cbor_hex = hex::encode(&cbor);

    // Pin: implementer runs once with `eprintln!("{}", cbor_hex);` then
    // replaces this expected string with the captured value.
    let expected = "TODO_PIN_CBOR_HEX_AFTER_FIRST_RUN";
    assert_eq!(cbor_hex, expected, "PkarrRoutingRecord canonical CBOR must not drift");
}
```

- [ ] **Step 4: Capture + pin the CBOR hex**

Run once, replace `eprintln!` macro inside the test temporarily or use the test output to capture the actual hex, then paste into `expected`. Re-run to verify pass.

- [ ] **Step 5: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets -- -D warnings
```

- [ ] **Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/record.rs crates/harmony-pkarr/src/error.rs crates/harmony-pkarr/src/lib.rs crates/harmony-pkarr/tests/wire_format_pkarr_routing_record_fixtures.rs Cargo.toml
git commit -m "feat(zeb-322): PkarrRoutingRecord wire format + inner-sig + skew verify + pin test"
```

---

## Task 5: Mock pkarr-relay (test fixture)

**Purpose:** Tiny in-process axum HTTP server that accepts BEP44 PUT/GET requests and stores them in-memory. Shipped behind `test-fixtures` feature so downstream harmony-client integration tests (Phase 2b) can reuse it. Building this BEFORE the real relay client gives us a controllable test target.

**Files:**
- Create: `crates/harmony-pkarr/src/testing.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs` — conditional `pub mod testing;`
- Modify: `crates/harmony-pkarr/Cargo.toml` — verify axum/tokio/hyper deps are correctly gated

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/testing.rs`:
```rust
//! Mock pkarr-relay HTTP server for testing.
//!
//! Implements the minimal subset of the BEP44-over-HTTP relay protocol
//! that real pkarr relays expose:
//! - `PUT /{z32_public_key}` — store a BEP44 envelope (signed payload).
//! - `GET /{z32_public_key}` — retrieve the most recent envelope.
//!
//! Storage is in-memory `HashMap<PublicKey, BEP44Envelope>`. No persistence.
//! No DHT-style propagation. Latest write wins.
//!
//! Spawn with `MockPkarrRelay::start().await` → returns `(handle, base_url)`.
//! Use `base_url` in your relay-client configuration.

#![cfg(any(test, feature = "test-fixtures"))]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, put},
    Router,
};

/// In-memory BEP44 store. Key is the z-base-32-encoded public key string
/// (as it appears in the URL path). Value is the raw envelope bytes
/// (whatever the client PUT).
type Store = Arc<RwLock<HashMap<String, Vec<u8>>>>;

/// Handle to a running mock relay. Drop to stop the server.
pub struct MockPkarrRelay {
    pub base_url: String,
    _shutdown: tokio::sync::oneshot::Sender<()>,
    _server_task: tokio::task::JoinHandle<()>,
}

impl MockPkarrRelay {
    /// Start a fresh mock relay on a random localhost port.
    pub async fn start() -> Self {
        let store: Store = Arc::new(RwLock::new(HashMap::new()));
        let app = Router::new()
            .route("/{key}", put(put_record).get(get_record))
            .with_state(store);

        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind 0");
        let addr = listener.local_addr().expect("addr");
        let base_url = format!("http://{}", addr);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server_task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("axum serve");
        });

        Self {
            base_url,
            _shutdown: shutdown_tx,
            _server_task: server_task,
        }
    }
}

async fn put_record(
    Path(key): Path<String>,
    State(store): State<Store>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    store.write().await.insert(key, body.to_vec());
    StatusCode::OK
}

async fn get_record(
    Path(key): Path<String>,
    State(store): State<Store>,
) -> impl IntoResponse {
    match store.read().await.get(&key) {
        Some(bytes) => (StatusCode::OK, bytes.clone()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trip_put_then_get() {
        let relay = MockPkarrRelay::start().await;
        let client = reqwest::Client::new();
        let put_resp = client
            .put(format!("{}/abc123", relay.base_url))
            .body(b"hello".to_vec())
            .send()
            .await
            .expect("put");
        assert_eq!(put_resp.status(), 200);

        let get_resp = client
            .get(format!("{}/abc123", relay.base_url))
            .send()
            .await
            .expect("get");
        assert_eq!(get_resp.status(), 200);
        assert_eq!(get_resp.bytes().await.expect("body").as_ref(), b"hello");
    }

    #[tokio::test]
    async fn get_missing_key_returns_404() {
        let relay = MockPkarrRelay::start().await;
        let client = reqwest::Client::new();
        let get_resp = client
            .get(format!("{}/missing", relay.base_url))
            .send()
            .await
            .expect("get");
        assert_eq!(get_resp.status(), 404);
    }
}
```

Modify `crates/harmony-pkarr/Cargo.toml` — make sure `tokio` and `axum` are full-featured under test-fixtures:
```toml
[features]
default = ["std"]
std = []
test-fixtures = ["dep:axum", "dep:tokio", "dep:hyper", "dep:reqwest"]

[dependencies]
# ... existing ...
axum = { workspace = true, optional = true, default-features = false, features = ["tokio", "http1"] }
tokio = { workspace = true, optional = true, features = ["rt-multi-thread", "macros", "net", "sync"] }
hyper = { workspace = true, optional = true }
reqwest = { workspace = true, optional = true, default-features = false, features = ["rustls-tls"] }

[dev-dependencies]
hex = { workspace = true }
rand = { workspace = true }
tokio = { workspace = true, features = ["rt-multi-thread", "macros", "time"] }
reqwest = { workspace = true, default-features = false, features = ["rustls-tls"] }

[[test]]
name = "wire_format_pkarr_routing_record_fixtures"
required-features = []

[features]  # (consolidated above)
```

Verify `reqwest` is in workspace.dependencies. If not, add `reqwest = { version = "0.12", default-features = false, features = ["rustls-tls"] }`.

Add to `src/lib.rs`:
```rust
#[cfg(any(test, feature = "test-fixtures"))]
pub mod testing;
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run -p harmony-pkarr --features test-fixtures testing:: 2>&1 | tail -20
```

Expected: 2 tests pass.

- [ ] **Step 3: Run cargo fmt + clippy with test-fixtures feature**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets --features test-fixtures -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/testing.rs crates/harmony-pkarr/src/lib.rs Cargo.toml
git commit -m "feat(zeb-322): mock pkarr-relay axum server behind test-fixtures feature"
```

---

## Task 6: Relay client (HTTP + rotation + cooldown)

**Purpose:** Production HTTP client that talks to one or more pkarr relays in rotation, with per-relay cooldown on timeout/429, and surfaces a clean `Result<Option<Bytes>>` API. Tests use the Task 5 mock relay.

**Files:**
- Create: `crates/harmony-pkarr/src/relay.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod relay;`
- Modify: `crates/harmony-pkarr/Cargo.toml` — promote `reqwest` to non-optional dep

- [ ] **Step 1: Write the failing test**

Create `crates/harmony-pkarr/src/relay.rs`:
```rust
//! HTTP pkarr-relay client.
//!
//! Pool of relay base URLs. On publish/resolve, iterates the pool until one
//! succeeds. On per-request timeout (5s) or HTTP 429, the offending relay
//! enters a 30s cooldown and is skipped on subsequent calls until expiry.
//!
//! Spec Section 6.4 (publication-side IP-hiding via rotation) + Section 14
//! (failure-modes table).

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::error::PkarrError;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);
const COOLDOWN: Duration = Duration::from_secs(30);

#[derive(Debug, Clone)]
pub struct RelayPool {
    relays: alloc::vec::Vec<alloc::string::String>,
}

impl RelayPool {
    pub fn new(relays: alloc::vec::Vec<alloc::string::String>) -> Self {
        Self { relays }
    }

    pub fn is_empty(&self) -> bool {
        self.relays.is_empty()
    }

    pub fn len(&self) -> usize {
        self.relays.len()
    }
}

pub struct RelayClient {
    pool: RelayPool,
    http: reqwest::Client,
    cooldown: Mutex<HashMap<alloc::string::String, Instant>>,
}

impl RelayClient {
    pub fn new(pool: RelayPool) -> Self {
        Self {
            pool,
            http: reqwest::Client::builder()
                .timeout(REQUEST_TIMEOUT)
                .build()
                .expect("reqwest client"),
            cooldown: Mutex::new(HashMap::new()),
        }
    }

    /// PUT the BEP44 envelope to the first available relay.
    pub async fn put(&self, key_z32: &str, envelope: &[u8]) -> Result<(), PkarrError> {
        for base in self.available_relays() {
            let url = format!("{}/{}", base, key_z32);
            match self
                .http
                .put(&url)
                .body(envelope.to_vec())
                .send()
                .await
            {
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    continue;
                }
                Ok(resp) => return Err(PkarrError::RelayHttpError(resp.status().as_u16())),
                Err(_) => {
                    self.mark_cooldown(&base);
                    continue;
                }
            }
        }
        Err(PkarrError::NoRelaysAvailable)
    }

    /// GET the BEP44 envelope. Returns `Ok(None)` if no relay has the key
    /// (404 across the pool); `Err` if all relays errored/cooldown.
    pub async fn get(&self, key_z32: &str) -> Result<Option<alloc::vec::Vec<u8>>, PkarrError> {
        let mut all_404 = true;
        for base in self.available_relays() {
            let url = format!("{}/{}", base, key_z32);
            match self.http.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp.bytes().await.map_err(|_| PkarrError::RelayResponseInvalid)?;
                    return Ok(Some(bytes.to_vec()));
                }
                Ok(resp) if resp.status().as_u16() == 404 => continue,
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    all_404 = false;
                    continue;
                }
                Ok(_) => {
                    all_404 = false;
                    continue;
                }
                Err(_) => {
                    self.mark_cooldown(&base);
                    all_404 = false;
                    continue;
                }
            }
        }
        if all_404 {
            Ok(None)
        } else {
            Err(PkarrError::NoRelaysAvailable)
        }
    }

    fn available_relays(&self) -> alloc::vec::Vec<alloc::string::String> {
        let now = Instant::now();
        let cd = self.cooldown.lock().expect("cooldown poisoned");
        self.pool
            .relays
            .iter()
            .filter(|r| cd.get(r.as_str()).is_none_or(|expiry| *expiry < now))
            .cloned()
            .collect()
    }

    fn mark_cooldown(&self, base: &str) {
        let mut cd = self.cooldown.lock().expect("cooldown poisoned");
        cd.insert(base.to_string(), Instant::now() + COOLDOWN);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockPkarrRelay;

    #[tokio::test]
    async fn put_then_get_via_pool() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        client.put("k1", b"envelope").await.expect("put");
        let got = client.get("k1").await.expect("get");
        assert_eq!(got.as_deref(), Some(b"envelope".as_ref()));
    }

    #[tokio::test]
    async fn get_returns_none_on_404() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        let got = client.get("missing").await.expect("get");
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn unreachable_relay_falls_through_to_next() {
        let alive = MockPkarrRelay::start().await;
        // Hard-coded unreachable port should fail fast (refused / dns / etc).
        let pool = RelayPool::new(vec!["http://127.0.0.1:1".to_string(), alive.base_url.clone()]);
        let client = RelayClient::new(pool);
        client.put("k1", b"hello").await.expect("put");
        let got = client.get("k1").await.expect("get");
        assert_eq!(got.as_deref(), Some(b"hello".as_ref()));
    }

    #[tokio::test]
    async fn all_relays_unavailable_yields_error() {
        let pool = RelayPool::new(vec!["http://127.0.0.1:1".to_string()]);
        let client = RelayClient::new(pool);
        assert!(matches!(
            client.get("k").await,
            Err(PkarrError::NoRelaysAvailable)
        ));
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod relay;
pub use relay::{RelayClient, RelayPool};
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run -p harmony-pkarr --features test-fixtures relay:: 2>&1 | tail -20
```

Expected: 4 tests pass. Note: the "unreachable relay" test depends on port 1 actually being unreachable; if your environment routes port 1 unexpectedly, replace with `127.0.0.1:0` (which is also unbindable for connect).

- [ ] **Step 3: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets --features test-fixtures -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/relay.rs crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-322): HTTP relay client + pool rotation + cooldown"
```

---

## Task 7: PkarrPublisher

**Purpose:** Background task that holds a set of "active publications" (key + record-builder + republish schedule) and pushes them to the relay pool at the right times. Caller adds/removes publications via `register`/`unregister`. Republish schedule: immediate + `epoch_start + 30min` + `epoch_start + 3.5d` per the spec.

**Files:**
- Create: `crates/harmony-pkarr/src/publisher.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod publisher;`

- [ ] **Step 1: Write the failing test + implementation in one pass**

Create `crates/harmony-pkarr/src/publisher.rs`:
```rust
//! `PkarrPublisher` — background task that publishes registered keys on a
//! schedule and republishes ahead of DHT-record TTL expiry.
//!
//! Spec Section 6 (publication lifecycles per case).
//!
//! Caller responsibility: `register(handle, key, record_builder)` adds an
//! active publication; `unregister(handle)` removes it. The publisher itself
//! is case-agnostic — case-specific lifecycle logic lives in harmony-client.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use ed25519_dalek::SigningKey;
use tokio::sync::{Mutex, Notify};

use crate::epoch::{current_epoch_id, epoch_start_ms, EPOCH_DURATION_MS};
use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;
use crate::relay::RelayClient;

/// Callback that produces a fresh `PkarrRoutingRecord` for a publication.
/// Closure-typed so callers can capture per-publication state (which iroh
/// routing to encode, what `announced_at_ms` to stamp).
pub type RecordBuilder = Arc<dyn Fn(u64) -> PkarrRoutingRecord + Send + Sync>;

#[derive(Clone)]
struct ActivePublication {
    handle: alloc::string::String,
    ephemeral_key: SigningKey,
    builder: RecordBuilder,
    next_publish_at: Instant,
}

pub struct PkarrPublisher {
    relay: Arc<RelayClient>,
    state: Arc<Mutex<HashMap<alloc::string::String, ActivePublication>>>,
    wakeup: Arc<Notify>,
}

impl PkarrPublisher {
    pub fn new(relay: Arc<RelayClient>) -> Self {
        Self {
            relay,
            state: Arc::new(Mutex::new(HashMap::new())),
            wakeup: Arc::new(Notify::new()),
        }
    }

    /// Add or replace an active publication. Schedules an immediate publish
    /// on the next loop tick.
    pub async fn register(
        &self,
        handle: alloc::string::String,
        ephemeral_key: SigningKey,
        builder: RecordBuilder,
    ) {
        let pub_state = ActivePublication {
            handle: handle.clone(),
            ephemeral_key,
            builder,
            next_publish_at: Instant::now(),
        };
        self.state.lock().await.insert(handle, pub_state);
        self.wakeup.notify_one();
    }

    pub async fn unregister(&self, handle: &str) {
        self.state.lock().await.remove(handle);
    }

    pub async fn active_handles(&self) -> alloc::vec::Vec<alloc::string::String> {
        self.state.lock().await.keys().cloned().collect()
    }

    /// Spawn the background driver. Caller keeps the returned `JoinHandle`
    /// so it can `abort()` on shutdown.
    pub fn spawn(self: Arc<Self>) -> tokio::task::JoinHandle<()> {
        tokio::spawn(async move {
            loop {
                // Wait until the next scheduled publish, or until someone
                // registers a new publication.
                let sleep_until = self.next_wakeup().await;
                tokio::select! {
                    _ = tokio::time::sleep_until(sleep_until.into()) => {}
                    _ = self.wakeup.notified() => {}
                }
                self.drive_pending().await;
            }
        })
    }

    /// Publish all publications whose `next_publish_at` is in the past.
    async fn drive_pending(&self) {
        let now = Instant::now();
        let mut due: alloc::vec::Vec<ActivePublication> = alloc::vec::Vec::new();
        {
            let mut state = self.state.lock().await;
            for pub_state in state.values_mut() {
                if pub_state.next_publish_at <= now {
                    due.push(pub_state.clone());
                    pub_state.next_publish_at = compute_next_publish_at(now_ms());
                }
            }
        }
        for pub_state in due {
            if let Err(e) = self.publish_one(&pub_state).await {
                tracing::warn!(handle = %pub_state.handle, error = ?e, "pkarr publish failed");
            }
        }
    }

    async fn publish_one(&self, pub_state: &ActivePublication) -> Result<(), PkarrError> {
        let now = now_ms();
        let record = (pub_state.builder)(now);
        let cbor = record.to_canonical_cbor()?;
        let envelope = wrap_bep44_envelope(&pub_state.ephemeral_key, &cbor, now as u32)?;
        let key_z32 = z32_encode_public(&pub_state.ephemeral_key.verifying_key().to_bytes());
        self.relay.put(&key_z32, &envelope).await
    }

    async fn next_wakeup(&self) -> Instant {
        let state = self.state.lock().await;
        state
            .values()
            .map(|p| p.next_publish_at)
            .min()
            .unwrap_or_else(|| Instant::now() + Duration::from_secs(3600))
    }
}

/// Compute the next scheduled publish time relative to `now_ms`.
///
/// Schedule per spec Section 5.4: `epoch_start + 30min` and `epoch_start + 3.5d`.
/// Returns whichever future time is sooner; if both are past, schedules
/// next-epoch's `+30min` mark.
pub(crate) fn compute_next_publish_at(now_ms: u64) -> Instant {
    let epoch_id = current_epoch_id(now_ms);
    let epoch_start = epoch_start_ms(epoch_id);
    let candidate_a = epoch_start + 30 * 60 * 1000;
    let candidate_b = epoch_start + 7 * 86_400_000 / 2; // 3.5 d
    let candidate_c = epoch_start_ms(epoch_id + 1) + 30 * 60 * 1000;

    let next_ms = [candidate_a, candidate_b, candidate_c]
        .iter()
        .copied()
        .filter(|t| *t > now_ms)
        .min()
        .unwrap_or(now_ms + EPOCH_DURATION_MS);

    Instant::now() + Duration::from_millis(next_ms - now_ms)
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock < UNIX epoch is unsupported")
        .as_millis() as u64
}

/// BEP44 envelope wrap. Uses the `pkarr` crate's primitives if available;
/// otherwise emits a manually-constructed signed envelope per BEP44.
/// Implementer: use `pkarr::SignedPacket` if compatible, else implement
/// BEP44 v=0 signing directly:
///   sig = Ed25519_sign(ephemeral_key, "3:seqi{seq}e1:v{payload_len}:{payload}")
fn wrap_bep44_envelope(
    ephemeral_key: &SigningKey,
    payload: &[u8],
    seq: u32,
) -> Result<alloc::vec::Vec<u8>, PkarrError> {
    // Manual BEP44 v=0 sign:
    use ed25519_dalek::Signer;
    let mut to_sign = alloc::vec::Vec::new();
    to_sign.extend_from_slice(format!("3:seqi{}e1:v{}:", seq, payload.len()).as_bytes());
    to_sign.extend_from_slice(payload);
    let sig = ephemeral_key.sign(&to_sign);

    // Envelope format: bencoded { sig: bytes, seq: int, v: bytes }
    // Use bencoded form so any standard pkarr-relay accepts it.
    // For mock-relay round-trip, simpler: concat sig(64) ‖ seq(4 LE) ‖ payload.
    // Implementer: pick one consistent form and document. Below uses simple form.
    let mut envelope = alloc::vec::Vec::with_capacity(64 + 4 + payload.len());
    envelope.extend_from_slice(&sig.to_bytes());
    envelope.extend_from_slice(&seq.to_le_bytes());
    envelope.extend_from_slice(payload);
    Ok(envelope)
}

/// Encode a 32-byte Ed25519 public key as the z-base-32 string pkarr-relays
/// expect in the URL path. For the mock relay (Task 5) which uses arbitrary
/// strings, just use lowercase hex.
fn z32_encode_public(pk: &[u8; 32]) -> alloc::string::String {
    // Implementer: replace with real z-base-32 (e.g., `zbase32` crate) for
    // production interop. Mock relay accepts any string.
    hex::encode(pk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockPkarrRelay;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn registered_publication_publishes_immediately() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(client));
        let _handle = Arc::clone(&publisher).spawn();

        let ephemeral = SigningKey::generate(&mut OsRng);
        let pk_hex = hex::encode(ephemeral.verifying_key().to_bytes());

        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(
                b"test-routing".to_vec(),
                identity_pub,
                now_ms,
                &identity_sk_for_builder,
            )
            .expect("sign")
        });

        publisher
            .register("test-pub".to_string(), ephemeral, builder)
            .await;

        // Wait briefly for the publish to land in the mock relay.
        let mut retries = 0;
        while retries < 50 {
            tokio::time::sleep(Duration::from_millis(50)).await;
            let resp = reqwest::get(format!("{}/{}", relay.base_url, pk_hex))
                .await
                .expect("get");
            if resp.status() == 200 {
                let body = resp.bytes().await.expect("body");
                // Envelope = 64 sig + 4 seq + CBOR payload (~70+ bytes).
                assert!(body.len() > 64 + 4);
                return;
            }
            retries += 1;
        }
        panic!("publisher did not push to mock relay within 2.5s");
    }

    #[tokio::test]
    async fn unregister_stops_future_publishes() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(client));
        let _handle = Arc::clone(&publisher).spawn();

        let ephemeral = SigningKey::generate(&mut OsRng);
        let identity_sk = SigningKey::generate(&mut OsRng);
        let mut identity_pub = [0u8; 64];
        identity_pub[32..].copy_from_slice(&identity_sk.verifying_key().to_bytes());
        let identity_sk_for_builder = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(b"blob".to_vec(), identity_pub, now_ms, &identity_sk_for_builder).expect("sign")
        });

        publisher
            .register("to-remove".to_string(), ephemeral, builder)
            .await;
        publisher.unregister("to-remove").await;

        assert!(publisher.active_handles().await.is_empty());
    }
}
```

Add to `src/lib.rs`:
```rust
pub mod publisher;
pub use publisher::{PkarrPublisher, RecordBuilder};
```

Modify `crates/harmony-pkarr/Cargo.toml` `[dependencies]` — promote `tokio` to required (the publisher needs it always; the `test-fixtures` gate is just for the mock relay):
```toml
[dependencies]
# ... existing ...
tokio = { workspace = true, features = ["rt", "macros", "sync", "time"] }
tracing = { workspace = true }
```

Wait — `tokio` was already optional under `test-fixtures`. We need it for the publisher even without test-fixtures. Move it out of optional. Update test-fixtures feature flag to NOT include `tokio` (since it's already always-on) and just keep `axum` + `reqwest` + `hyper`.

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run -p harmony-pkarr --features test-fixtures publisher:: 2>&1 | tail -30
```

Expected: 2 tests pass within ~3s each.

- [ ] **Step 3: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets --features test-fixtures -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/publisher.rs crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-322): PkarrPublisher background task + BEP44 envelope sign"
```

---

## Task 8: PkarrResolver with LRU cache + parallel epoch-window queries

**Purpose:** Counterpart to PkarrPublisher. Queries the relay pool for a key, parses the BEP44 envelope, verifies the outer Ed25519 sig, parses the inner CBOR record, returns it. Caches results in-memory. Queries the 3-key epoch tolerance window in parallel (per Section 5.4) and picks the freshest valid response.

**Files:**
- Create: `crates/harmony-pkarr/src/resolver.rs`
- Modify: `crates/harmony-pkarr/src/lib.rs` — `pub mod resolver;`
- Modify: `crates/harmony-pkarr/Cargo.toml` — add `lru` dep

- [ ] **Step 1: Write the failing test + implementation**

Create `crates/harmony-pkarr/src/resolver.rs`:
```rust
//! `PkarrResolver` — relay-pool resolver with parallel epoch-window queries
//! and in-memory LRU cache.
//!
//! Spec Sections 5.4 + 7.1.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use futures::future::join_all;
use lru::LruCache;

use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;
use crate::relay::RelayClient;

const POSITIVE_CACHE_TTL: Duration = Duration::from_secs(15 * 60);
const NEGATIVE_CACHE_TTL: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct CachedResolution {
    record: Option<PkarrRoutingRecord>,
    fetched_at: Instant,
    ttl: Duration,
}

pub struct PkarrResolver {
    relay: Arc<RelayClient>,
    cache: Arc<Mutex<LruCache<[u8; 32], CachedResolution>>>,
}

impl PkarrResolver {
    pub fn new(relay: Arc<RelayClient>) -> Self {
        Self {
            relay,
            cache: Arc::new(Mutex::new(LruCache::new(NonZeroUsize::new(1024).expect("nonzero")))),
        }
    }

    /// Resolve a single ephemeral public key. Returns `Ok(Some)` if a valid
    /// signed record is found; `Ok(None)` if confirmed-absent; `Err` on
    /// transport failures.
    pub async fn resolve(&self, pk: &VerifyingKey) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let pk_bytes = pk.to_bytes();
        if let Some(cached) = self.cache_get(&pk_bytes) {
            return Ok(cached.record.clone());
        }
        let key_z32 = hex::encode(pk_bytes);
        match self.relay.get(&key_z32).await? {
            None => {
                self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                Ok(None)
            }
            Some(envelope) => {
                let record = parse_and_verify(&envelope, pk)?;
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
            }
        }
    }

    /// Resolve any of `keys` (the 3-key epoch tolerance window) in parallel.
    /// Returns the freshest valid record by `announced_at_ms`, or `None` if
    /// none resolve.
    pub async fn resolve_window(
        &self,
        keys: &[VerifyingKey],
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let futures = keys.iter().map(|pk| self.resolve(pk));
        let results = join_all(futures).await;

        let mut best: Option<PkarrRoutingRecord> = None;
        let mut any_err: Option<PkarrError> = None;
        for r in results {
            match r {
                Ok(Some(rec)) => {
                    if best.as_ref().is_none_or(|b| rec.announced_at_ms > b.announced_at_ms) {
                        best = Some(rec);
                    }
                }
                Ok(None) => {}
                Err(e) => any_err = Some(e),
            }
        }

        match (best, any_err) {
            (Some(rec), _) => Ok(Some(rec)),
            (None, Some(e)) => Err(e),
            (None, None) => Ok(None),
        }
    }

    fn cache_get(&self, pk: &[u8; 32]) -> Option<CachedResolution> {
        let mut cache = self.cache.lock().expect("cache poisoned");
        let entry = cache.get(pk)?;
        if entry.fetched_at.elapsed() < entry.ttl {
            Some(entry.clone())
        } else {
            cache.pop(pk);
            None
        }
    }

    fn cache_put(&self, pk: [u8; 32], record: Option<PkarrRoutingRecord>, ttl: Duration) {
        let mut cache = self.cache.lock().expect("cache poisoned");
        cache.put(
            pk,
            CachedResolution {
                record,
                fetched_at: Instant::now(),
                ttl,
            },
        );
    }
}

/// Parse the BEP44 envelope, verify the outer Ed25519 sig under `expected_pk`,
/// then parse + return the inner `PkarrRoutingRecord` (does NOT verify the
/// inner sig — caller does that with the expected identity).
fn parse_and_verify(envelope: &[u8], expected_pk: &VerifyingKey) -> Result<PkarrRoutingRecord, PkarrError> {
    // Envelope = 64 sig + 4 seq (LE) + payload (matching publisher.rs format).
    if envelope.len() < 64 + 4 {
        return Err(PkarrError::RelayResponseInvalid);
    }
    let sig_bytes: [u8; 64] = envelope[..64].try_into().expect("64 == 64");
    let seq = u32::from_le_bytes(envelope[64..68].try_into().expect("4 == 4"));
    let payload = &envelope[68..];

    let mut to_verify = alloc::vec::Vec::new();
    to_verify.extend_from_slice(format!("3:seqi{}e1:v{}:", seq, payload.len()).as_bytes());
    to_verify.extend_from_slice(payload);

    let sig = Signature::from_bytes(&sig_bytes);
    expected_pk
        .verify(&to_verify, &sig)
        .map_err(|_| PkarrError::OuterSignatureInvalid)?;

    PkarrRoutingRecord::from_canonical_cbor(payload)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publisher::{PkarrPublisher, RecordBuilder};
    use crate::testing::MockPkarrRelay;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_identity_pubkey(sk: &SigningKey) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        out
    }

    #[tokio::test]
    async fn publish_then_resolve_round_trip() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));

        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _ph = Arc::clone(&publisher).spawn();
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let identity_sk_clone = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(b"r-blob".to_vec(), identity_pub, now_ms, &identity_sk_clone)
                .expect("sign")
        });

        publisher
            .register("round-trip".to_string(), ephemeral, builder)
            .await;

        // Wait for publish to land + then resolve.
        let mut attempts = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            attempts += 1;
            assert!(attempts < 60, "resolve timed out");
            let r = resolver.resolve(&vk).await.expect("resolve");
            if let Some(rec) = r {
                assert_eq!(rec.routing_blob, b"r-blob");
                assert!(rec.verify_inner_sig().is_ok());
                return;
            }
        }
    }

    #[tokio::test]
    async fn resolve_missing_returns_none() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let absent_key = SigningKey::generate(&mut OsRng).verifying_key();
        let result = resolver.resolve(&absent_key).await.expect("resolve");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn resolve_caches_positive_result() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _ph = Arc::clone(&publisher).spawn();
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let identity_sk_clone = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(b"cached".to_vec(), identity_pub, now_ms, &identity_sk_clone)
                .expect("sign")
        });
        publisher
            .register("cache-test".to_string(), ephemeral, builder)
            .await;

        // Wait for publish.
        let mut attempts = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            attempts += 1;
            assert!(attempts < 60, "first resolve timed out");
            if resolver.resolve(&vk).await.expect("resolve").is_some() {
                break;
            }
        }

        // Second resolve hits cache.
        let r = resolver.resolve(&vk).await.expect("resolve");
        assert!(r.is_some());
    }
}
```

Add to `crates/harmony-pkarr/Cargo.toml`:
```toml
[dependencies]
# ... existing ...
lru = { workspace = true }
futures = { workspace = true }
```

Verify `lru` and `futures` are in workspace.dependencies. If not, add:
- `lru = { version = "0.16", default-features = false }`
- `futures = { version = "0.3", default-features = false, features = ["alloc"] }`

Add to `src/lib.rs`:
```rust
pub mod resolver;
pub use resolver::PkarrResolver;
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 600 cargo nextest run -p harmony-pkarr --features test-fixtures resolver:: 2>&1 | tail -30
```

Expected: 3 tests pass within ~5s each.

- [ ] **Step 3: Run cargo fmt + clippy**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 600 cargo clippy -p harmony-pkarr --all-targets --features test-fixtures -- -D warnings
```

- [ ] **Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git add crates/harmony-pkarr/Cargo.toml crates/harmony-pkarr/src/resolver.rs crates/harmony-pkarr/src/lib.rs
git commit -m "feat(zeb-322): PkarrResolver with LRU cache + parallel epoch-window queries"
```

---

## Task 9: Final 5-gate sweep + push branch + open PR

**Purpose:** Run the full workspace quality gates one last time, push the branch, and create the PR with proper cross-repo coordination notes.

- [ ] **Step 1: Workspace-wide gates (clean run)**

```bash
cd /Users/zeblith/work/zeblithic/harmony
timeout 120 cargo fmt --all -- --check
timeout 900 cargo clippy --workspace --all-targets --features test-fixtures -- -D warnings 2>&1 | tail -50
timeout 1200 cargo nextest run --workspace --all-targets --features test-fixtures 2>&1 | tail -50
```

All three must pass. Compare nextest failure list to `/tmp/zeb-322-baseline-failures.txt` from Task 0 — any NEW failure not in the baseline is a regression and must be fixed before PR creation.

- [ ] **Step 2: Confirm branch is up to date with origin/main**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git fetch origin --quiet
git log --oneline HEAD..origin/main | head -5    # should be empty if branched from latest
```

If origin/main has advanced since branching, rebase:
```bash
git rebase origin/main
# re-run Step 1 after rebase
```

- [ ] **Step 3: Push branch**

```bash
cd /Users/zeblith/work/zeblithic/harmony
git push -u origin zeb-322-harmony-pkarr-crate
```

- [ ] **Step 4: Create PR**

PR title: `feat(zeb-322): harmony-pkarr crate — BEP44 publish/resolve via Mainline DHT HTTP-relay`

PR body — use a HEREDOC; DO NOT include the bare string `ZEB-321` anywhere in the title or body per the auto-close memory rule (use markdown link to the umbrella, but only via descriptive prose, not as a close-trigger):

```bash
cd /Users/zeblith/work/zeblithic/harmony
gh pr create --title "feat(zeb-322): harmony-pkarr crate — BEP44 publish/resolve via Mainline DHT HTTP-relay" --body "$(cat <<'EOF'
## Summary

New `harmony-pkarr` crate. Transport-agnostic BEP44 publish/resolve over the BitTorrent Mainline DHT via HTTP relays, with HKDF-derived ephemeral Ed25519 keys. Foundation for the cross-WAN first-contact discovery work in the multi-phase cross-WAN connectivity initiative.

- 8 new source files (lib.rs, error.rs, epoch.rs, derive.rs, record.rs, relay.rs, publisher.rs, resolver.rs, testing.rs).
- Mock pkarr-relay axum server behind \`test-fixtures\` feature (re-used by downstream harmony-client tests in [ZEB-323](https://linear.app/zeblith/issue/ZEB-323)).
- Three reference vectors pinning HKDF-derived keys per case (invite / identity / community).
- Wire-format pin test for \`PkarrRoutingRecord\` canonical CBOR.
- RPK1-RPK5 silent-drop verify rules covered by discrete unit tests.
- 5 RPK rules covered + skew tolerance + identity-match.

## Design

Full design: \`harmony-client/docs/specs/2026-05-23-zeb-321-phase2-discovery-bootstrap-design.md\` (commit cb5cca5).
Closes [ZEB-322](https://linear.app/zeblith/issue/ZEB-322).

## Test plan

- [ ] \`cargo fmt --all -- --check\` clean
- [ ] \`cargo clippy --workspace --all-targets --features test-fixtures -- -D warnings\` clean
- [ ] \`cargo nextest run --workspace --all-targets --features test-fixtures\` green (no regressions vs Task 0 baseline)
- [ ] Wire-format pin test (\`tests/wire_format_pkarr_routing_record_fixtures.rs\`) holds
- [ ] HKDF reference vectors (in \`derive.rs\` tests) hold

## Cross-repo coordination

This is the harmony-core half of a two-PR Phase 2 ship. The companion [ZEB-323](https://linear.app/zeblith/issue/ZEB-323) harmony-client PR pins this PR's merge commit SHA in its Cargo.toml. **Merge this PR first.**

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 5: Verify PR + return URL**

```bash
gh pr view --json url,number,title 2>&1 | tail -10
```

Capture the PR number + URL for the parent controller's tracking.

---

## End of Plan

All 9 tasks complete: ~1000-1200 LOC of Rust (8 source files + 1 test fixture), 1 wire-format pin test, ~20 unit + integration tests covering all 5 RPK silent-drop rules + HKDF reference vectors + publisher/resolver round-trips through the mock relay.

Downstream blocker: harmony-client Phase 2b PR (ZEB-323) cannot merge until this PR's merge commit SHA is reachable from harmony's main branch and pinned in harmony-client's Cargo.toml.

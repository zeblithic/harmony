# Phase 0c: Chronos Temporal Decay — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the Chronos temporal decay function that attenuates Engram embeddings based on knowledge freshness, so stale facts gracefully weaken toward zero.

**Architecture:** Pure function in `harmony-engram` crate. No tensor math — just timestamp arithmetic and a Gaussian decay curve. The caller multiplies resolved embedding tensors by the decay vector before passing to `EngramGatedResidual`. Operates in `no_std` + `alloc` environment using `libm::exp` for the Gaussian computation.

**Tech Stack:** Rust, `libm` (for `no_std`-compatible `exp`), `harmony-engram` crate

**Spec reference:** `docs/superpowers/specs/2026-04-06-harmony-ct87-custom-model-design.md`, Section 6 (lines 358–425)

---

## File Structure

| Action | File | Responsibility |
|--------|------|---------------|
| Create | `crates/harmony-engram/src/chronos.rs` | ChronosTier enum, EngramMetadata struct, decay math, `temporal_decay()` batch function, all tests |
| Modify | `crates/harmony-engram/src/lib.rs` | Add `pub mod chronos;` and re-exports |
| Modify | `crates/harmony-engram/Cargo.toml` | Add `libm` dependency |

---

## Context for Implementers

### Where Chronos fits in the pipeline

```
Tokens → N-gram hashing → Shard lookup → Resolve f16 embeddings
       → Chronos temporal_decay (THIS MODULE — attenuate stale entries)
       → EngramGatedResidual at Layer 2
```

Chronos acts **before** the Engram gate. When decay → 0, the attenuated embedding → zero vector. The existing `EngramGatedResidual` (in `engram_residual.rs`) already handles zero input correctly — it produces zero residual, so the model proceeds as if no Engram entry exists.

### The decay curve

```
age = now - entry.timestamp

if tier == Eternal:      decay = 1.0  (never decays)
if age <= ttl:           decay = 1.0  (still fresh)
else:
    staleness = (age - ttl) / ttl
    decay = exp(-staleness² / 2)       (Gaussian decay)
```

Reference values (from spec):
- At age = 1× TTL: decay = 1.0
- At age = 2× TTL: staleness = 1.0, decay ≈ 0.6065
- At age = 3× TTL: staleness = 2.0, decay ≈ 0.1353
- At age = 4× TTL: staleness = 3.0, decay ≈ 0.0111

### Crate environment

`harmony-engram` is `no_std` by default with `alloc`. Uses `extern crate alloc;` and `alloc::vec::Vec`. For `exp()`, use `libm::exp` (not `f64::exp()` which requires `std`). The `libm` crate is already a workspace dependency.

### Existing code patterns

- Tests live in `#[cfg(test)] mod tests` at the bottom of the same file
- Functions are `pub` with doc comments
- Error types go in `error.rs` but Chronos has no error cases (all inputs produce valid output)
- The crate uses `alloc::{vec::Vec}` for heap allocations in `no_std`

---

### Task 1: ChronosTier enum, EngramMetadata struct, and TTL mapping

**Files:**
- Create: `crates/harmony-engram/src/chronos.rs`

- [ ] **Step 1: Write failing tests for tier TTL mapping and metadata construction**

Create `crates/harmony-engram/src/chronos.rs` with only the test module:

```rust
//! Chronos temporal decay — attenuates Engram embeddings by knowledge freshness.
//!
//! Stale facts decay toward zero using a Gaussian curve past their TTL.
//! When decay reaches zero, the attenuated embedding becomes a zero vector,
//! and `EngramGatedResidual` produces zero residual — the model proceeds
//! as if no Engram entry exists.

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tier TTL defaults ──

    #[test]
    fn eternal_ttl_is_max() {
        assert_eq!(ChronosTier::Eternal.default_ttl_seconds(), u32::MAX);
    }

    #[test]
    fn near_eternal_ttl_is_ten_years() {
        // 10 * 365.25 * 24 * 3600 = 315_576_000
        assert_eq!(ChronosTier::NearEternal.default_ttl_seconds(), 315_576_000);
    }

    #[test]
    fn episodic_ttl_is_one_year() {
        // 365.25 * 24 * 3600 = 31_557_600
        assert_eq!(ChronosTier::Episodic.default_ttl_seconds(), 31_557_600);
    }

    #[test]
    fn regular_ttl_is_thirty_days() {
        // 30 * 24 * 3600 = 2_592_000
        assert_eq!(ChronosTier::Regular.default_ttl_seconds(), 2_592_000);
    }

    #[test]
    fn ephemeral_ttl_is_zero() {
        assert_eq!(ChronosTier::Ephemeral.default_ttl_seconds(), 0);
    }

    // ── Tier parsing ──

    #[test]
    fn from_u8_valid_tiers() {
        assert_eq!(ChronosTier::from_u8(1), Some(ChronosTier::Eternal));
        assert_eq!(ChronosTier::from_u8(2), Some(ChronosTier::NearEternal));
        assert_eq!(ChronosTier::from_u8(3), Some(ChronosTier::Episodic));
        assert_eq!(ChronosTier::from_u8(4), Some(ChronosTier::Regular));
        assert_eq!(ChronosTier::from_u8(5), Some(ChronosTier::Ephemeral));
    }

    #[test]
    fn from_u8_invalid_returns_none() {
        assert_eq!(ChronosTier::from_u8(0), None);
        assert_eq!(ChronosTier::from_u8(6), None);
        assert_eq!(ChronosTier::from_u8(255), None);
    }

    // ── Metadata construction ──

    #[test]
    fn new_derives_ttl_from_tier() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Regular);
        assert_eq!(meta.timestamp, 1_000_000);
        assert_eq!(meta.tier, ChronosTier::Regular);
        assert_eq!(meta.ttl_seconds, 2_592_000);
    }

    #[test]
    fn with_ttl_overrides_default() {
        let meta = EngramMetadata::with_ttl(1_000_000, ChronosTier::Regular, 86_400);
        assert_eq!(meta.ttl_seconds, 86_400);
        assert_eq!(meta.tier, ChronosTier::Regular);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram --lib chronos -- --nocapture 2>&1 | head -20`

Expected: Compilation errors — `ChronosTier` and `EngramMetadata` are not defined yet.

- [ ] **Step 3: Implement ChronosTier enum and EngramMetadata struct**

Add the following **above** the `#[cfg(test)]` block in `crates/harmony-engram/src/chronos.rs`:

```rust
//! Chronos temporal decay — attenuates Engram embeddings by knowledge freshness.
//!
//! Stale facts decay toward zero using a Gaussian curve past their TTL.
//! When decay reaches zero, the attenuated embedding becomes a zero vector,
//! and `EngramGatedResidual` produces zero residual — the model proceeds
//! as if no Engram entry exists.

/// Chronos frequency tier — classifies how quickly knowledge expires.
///
/// Tier numbering (1–5) matches the spec and is stored as `u8` in Engram
/// table entries. Use [`from_u8`](Self::from_u8) to parse from stored values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ChronosTier {
    /// Physical constants, mathematical axioms. Never expires.
    Eternal = 1,
    /// Historical facts, geography. TTL: ~10 years.
    NearEternal = 2,
    /// Political leaders, tech standards. TTL: ~1 year.
    Episodic = 3,
    /// Ages, populations. TTL: ~30 days.
    Regular = 4,
    /// Stock prices, live sensor data. Immediately stale.
    Ephemeral = 5,
}

impl ChronosTier {
    /// Default TTL in seconds for this tier.
    ///
    /// | Tier | TTL |
    /// |------|-----|
    /// | Eternal | `u32::MAX` (~136 years) |
    /// | NearEternal | 315,576,000 (~10 years) |
    /// | Episodic | 31,557,600 (~1 year) |
    /// | Regular | 2,592,000 (30 days) |
    /// | Ephemeral | 0 |
    pub fn default_ttl_seconds(self) -> u32 {
        match self {
            Self::Eternal => u32::MAX,
            Self::NearEternal => 315_576_000, // 10 * 365.25 * 86400
            Self::Episodic => 31_557_600,     // 365.25 * 86400
            Self::Regular => 2_592_000,       // 30 * 86400
            Self::Ephemeral => 0,
        }
    }

    /// Parse a tier from its `u8` discriminant (1–5).
    ///
    /// Returns `None` for values outside the valid range.
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Eternal),
            2 => Some(Self::NearEternal),
            3 => Some(Self::Episodic),
            4 => Some(Self::Regular),
            5 => Some(Self::Ephemeral),
            _ => None,
        }
    }
}

/// Per-entry metadata for Chronos temporal decay.
///
/// Carries the timestamp and freshness tier needed to compute the decay
/// factor. The embedding itself is not included — the caller multiplies
/// the resolved embedding by the decay factor returned by [`temporal_decay`].
#[derive(Debug, Clone, Copy)]
pub struct EngramMetadata {
    /// Unix epoch timestamp when this entry was last updated.
    pub timestamp: u32,
    /// Frequency tier (1–5).
    pub tier: ChronosTier,
    /// Time-to-live in seconds. Usually derived from tier via
    /// [`ChronosTier::default_ttl_seconds`], but may be overridden.
    pub ttl_seconds: u32,
}

impl EngramMetadata {
    /// Create metadata with the tier's default TTL.
    pub fn new(timestamp: u32, tier: ChronosTier) -> Self {
        Self {
            timestamp,
            tier,
            ttl_seconds: tier.default_ttl_seconds(),
        }
    }

    /// Create metadata with a custom TTL override.
    pub fn with_ttl(timestamp: u32, tier: ChronosTier, ttl_seconds: u32) -> Self {
        Self {
            timestamp,
            tier,
            ttl_seconds,
        }
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-engram --lib chronos`

Expected: 7 tests pass (5 TTL tests + 2 parsing tests + 2 construction tests — wait, that's 9 tests across 7 `#[test]` functions). All 7 should PASS.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-engram/src/chronos.rs
git commit -m "feat(engram): ChronosTier enum and EngramMetadata struct

Adds the five-tier temporal classification (Eternal through Ephemeral)
with default TTL mapping and per-entry metadata for Chronos decay.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Core decay math — `compute_decay()`

**Files:**
- Modify: `crates/harmony-engram/Cargo.toml` (add `libm` dependency)
- Modify: `crates/harmony-engram/src/chronos.rs` (add `compute_decay` function + tests)

- [ ] **Step 1: Add `libm` dependency to Cargo.toml**

Add to `[dependencies]` section of `crates/harmony-engram/Cargo.toml`:

```toml
libm = { workspace = true }
```

The full `[dependencies]` section should now be:

```toml
[dependencies]
half = { workspace = true }
libm = { workspace = true }
postcard = { workspace = true }
serde = { workspace = true, features = ["derive"] }
thiserror = { workspace = true }
xxhash-rust = { workspace = true }
```

- [ ] **Step 2: Write failing tests for decay computation**

Add these tests to the `mod tests` block in `crates/harmony-engram/src/chronos.rs`:

```rust
    // ── Decay: eternal tier ──

    #[test]
    fn eternal_never_decays() {
        let meta = EngramMetadata::new(0, ChronosTier::Eternal);
        // Even 100 years later, eternal knowledge is fully fresh.
        assert_eq!(compute_decay(&meta, 3_155_760_000), 1.0);
    }

    // ── Decay: fresh entries (within TTL) ──

    #[test]
    fn fresh_entry_returns_one() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Regular);
        // 15 days later (half the 30-day TTL) — still fresh.
        let now = 1_000_000 + 1_296_000; // +15 days
        assert_eq!(compute_decay(&meta, now), 1.0);
    }

    #[test]
    fn exactly_at_ttl_returns_one() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Regular);
        // Exactly at TTL boundary (30 days later).
        let now = 1_000_000 + 2_592_000;
        assert_eq!(compute_decay(&meta, now), 1.0);
    }

    // ── Decay: Gaussian curve past TTL ──

    #[test]
    fn at_two_times_ttl_decay_approx_0_61() {
        let meta = EngramMetadata::new(0, ChronosTier::Regular);
        let ttl = 2_592_000_u32; // 30 days
        let now = 2 * ttl;      // staleness = 1.0
        let decay = compute_decay(&meta, now);
        // exp(-0.5) ≈ 0.6065
        assert!((decay - 0.6065).abs() < 0.001, "got {decay}");
    }

    #[test]
    fn at_three_times_ttl_decay_approx_0_14() {
        let meta = EngramMetadata::new(0, ChronosTier::Regular);
        let ttl = 2_592_000_u32;
        let now = 3 * ttl; // staleness = 2.0
        let decay = compute_decay(&meta, now);
        // exp(-2.0) ≈ 0.1353
        assert!((decay - 0.1353).abs() < 0.001, "got {decay}");
    }

    #[test]
    fn at_four_times_ttl_decay_approx_0_01() {
        let meta = EngramMetadata::new(0, ChronosTier::Regular);
        let ttl = 2_592_000_u32;
        let now = 4 * ttl; // staleness = 3.0
        let decay = compute_decay(&meta, now);
        // exp(-4.5) ≈ 0.0111
        assert!((decay - 0.0111).abs() < 0.001, "got {decay}");
    }

    // ── Decay: ephemeral tier ──

    #[test]
    fn ephemeral_same_second_is_fresh() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Ephemeral);
        assert_eq!(compute_decay(&meta, 1_000_000), 1.0);
    }

    #[test]
    fn ephemeral_one_second_later_is_zero() {
        let meta = EngramMetadata::new(1_000_000, ChronosTier::Ephemeral);
        assert_eq!(compute_decay(&meta, 1_000_001), 0.0);
    }

    // ── Decay: edge cases ──

    #[test]
    fn future_timestamp_treated_as_fresh() {
        // Entry from "the future" (clock skew). age = 0 via saturating_sub.
        let meta = EngramMetadata::new(2_000_000, ChronosTier::Regular);
        assert_eq!(compute_decay(&meta, 1_000_000), 1.0);
    }

    #[test]
    fn decay_is_monotonically_decreasing() {
        let meta = EngramMetadata::new(0, ChronosTier::Episodic);
        let ttl = meta.ttl_seconds;
        let mut prev = 1.0_f32;
        // Sample from TTL to 5× TTL
        for multiplier in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0] {
            let now = (ttl as f64 * multiplier) as u32;
            let decay = compute_decay(&meta, now);
            assert!(
                decay <= prev,
                "decay not monotonically decreasing: at {multiplier}×TTL got {decay} > prev {prev}"
            );
            prev = decay;
        }
    }

    #[test]
    fn decay_always_in_zero_one_range() {
        // Spot-check various ages across tiers.
        for tier in [
            ChronosTier::Eternal,
            ChronosTier::NearEternal,
            ChronosTier::Episodic,
            ChronosTier::Regular,
            ChronosTier::Ephemeral,
        ] {
            let meta = EngramMetadata::new(0, tier);
            for now in [0, 1, 1000, 1_000_000, 100_000_000, u32::MAX] {
                let decay = compute_decay(&meta, now);
                assert!(
                    (0.0..=1.0).contains(&decay),
                    "tier {tier:?} at now={now}: decay {decay} out of range"
                );
            }
        }
    }
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test -p harmony-engram --lib chronos -- --nocapture 2>&1 | head -20`

Expected: Compilation error — `compute_decay` is not defined.

- [ ] **Step 4: Implement `compute_decay()`**

Add this function to `crates/harmony-engram/src/chronos.rs`, after the `EngramMetadata` impl block and before the `#[cfg(test)]` block:

```rust
/// Compute the decay factor for a single Engram entry.
///
/// Returns a value in `[0.0, 1.0]`:
/// - `1.0` — fully fresh (within TTL, or eternal tier)
/// - `0.0` — fully expired (the embedding should be treated as absent)
///
/// Past TTL, decay follows a Gaussian curve:
/// ```text
/// staleness = (age - ttl) / ttl
/// decay     = exp(-staleness² / 2)
/// ```
///
/// This gives a smooth, gradual decline: 0.61 at 2× TTL, 0.14 at 3× TTL,
/// 0.01 at 4× TTL. Knowledge gracefully dies over ~3–4× its TTL.
pub fn compute_decay(entry: &EngramMetadata, now: u32) -> f32 {
    // Eternal knowledge never decays.
    if matches!(entry.tier, ChronosTier::Eternal) {
        return 1.0;
    }

    // Ephemeral (ttl = 0): only fresh at exact timestamp or earlier.
    if entry.ttl_seconds == 0 {
        return if now <= entry.timestamp { 1.0 } else { 0.0 };
    }

    let age = now.saturating_sub(entry.timestamp);

    // Within TTL — fully fresh.
    if age <= entry.ttl_seconds {
        return 1.0;
    }

    // Past TTL — Gaussian decay.
    let staleness = (age - entry.ttl_seconds) as f64 / entry.ttl_seconds as f64;
    libm::exp(-staleness * staleness / 2.0) as f32
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-engram --lib chronos`

Expected: All 19 tests pass (7 from Task 1 + 12 new decay tests).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-engram/Cargo.toml crates/harmony-engram/src/chronos.rs
git commit -m "feat(engram): Chronos compute_decay — Gaussian temporal decay past TTL

Implements the core decay function: eternal entries never decay, fresh
entries return 1.0, and expired entries follow exp(-staleness²/2) where
staleness = (age - ttl) / ttl. Uses libm::exp for no_std compatibility.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Batch `temporal_decay()` function and module export

**Files:**
- Modify: `crates/harmony-engram/src/chronos.rs` (add `temporal_decay` function + tests)
- Modify: `crates/harmony-engram/src/lib.rs` (add module declaration and re-exports)

- [ ] **Step 1: Write failing tests for batch temporal_decay**

Add these tests to the `mod tests` block in `crates/harmony-engram/src/chronos.rs`:

```rust
    // ── Batch temporal_decay ──

    #[test]
    fn temporal_decay_empty_slice() {
        let result = temporal_decay(&[], 1_000_000);
        assert!(result.is_empty());
    }

    #[test]
    fn temporal_decay_mixed_tiers() {
        let now = 1_000_000_u32;
        let entries = [
            // Eternal — always 1.0
            EngramMetadata::new(0, ChronosTier::Eternal),
            // Regular, fresh (created 10 days ago, TTL = 30 days)
            EngramMetadata::new(now - 864_000, ChronosTier::Regular),
            // Ephemeral, stale (created 1 second ago)
            EngramMetadata::new(now - 1, ChronosTier::Ephemeral),
        ];
        let decays = temporal_decay(&entries, now);
        assert_eq!(decays.len(), 3);
        assert_eq!(decays[0], 1.0); // eternal
        assert_eq!(decays[1], 1.0); // fresh
        assert_eq!(decays[2], 0.0); // ephemeral, 1 second stale
    }

    #[test]
    fn temporal_decay_output_length_matches_input() {
        let entries: Vec<EngramMetadata> = (0..10)
            .map(|i| EngramMetadata::new(i * 1000, ChronosTier::Episodic))
            .collect();
        let decays = temporal_decay(&entries, 100_000_000);
        assert_eq!(decays.len(), entries.len());
    }

    #[test]
    fn temporal_decay_all_eternal_all_ones() {
        let entries: Vec<EngramMetadata> = (0..5)
            .map(|_| EngramMetadata::new(0, ChronosTier::Eternal))
            .collect();
        let decays = temporal_decay(&entries, u32::MAX);
        assert!(decays.iter().all(|&d| d == 1.0));
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-engram --lib chronos -- --nocapture 2>&1 | head -20`

Expected: Compilation error — `temporal_decay` is not defined.

- [ ] **Step 3: Implement `temporal_decay()` and add `alloc` import**

Add the `alloc` import at the top of `crates/harmony-engram/src/chronos.rs` (after the module doc comment, before the `ChronosTier` enum):

```rust
use alloc::vec::Vec;
```

Add the `temporal_decay` function after `compute_decay`, before `#[cfg(test)]`:

```rust
/// Compute decay factors for a batch of Engram entries.
///
/// Returns one `f32` per entry in `[0.0, 1.0]`. The caller multiplies each
/// resolved embedding by the corresponding decay factor before passing to
/// `EngramGatedResidual`.
pub fn temporal_decay(entries: &[EngramMetadata], now: u32) -> Vec<f32> {
    entries.iter().map(|e| compute_decay(e, now)).collect()
}
```

Also add the `alloc` import to the test module:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use alloc::vec::Vec;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-engram --lib chronos`

Expected: All 23 tests pass (19 from Tasks 1–2 + 4 new batch tests).

- [ ] **Step 5: Wire chronos module into lib.rs**

Add the module declaration and re-exports to `crates/harmony-engram/src/lib.rs`.

After the existing `pub mod resolve;` line (~line 23), add:

```rust
pub mod chronos;
```

After the existing `pub use manifest::ManifestHeader;` line (~line 26), add:

```rust
pub use chronos::{ChronosTier, EngramMetadata, compute_decay, temporal_decay};
```

- [ ] **Step 6: Run the full crate test suite to verify nothing is broken**

Run: `cargo test -p harmony-engram`

Expected: All tests pass — the existing `hash`, `resolve`, and `manifest` tests plus all 23 chronos tests.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-engram/src/chronos.rs crates/harmony-engram/src/lib.rs
git commit -m "feat(engram): temporal_decay batch function and module export

Adds the public temporal_decay(entries, now) -> Vec<f32> batch API
and wires the chronos module into harmony-engram's public interface.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review Checklist

### Spec Coverage

| Spec Requirement | Task |
|------------------|------|
| Five frequency tiers (Eternal → Ephemeral) | Task 1: `ChronosTier` enum |
| Per-entry metadata (timestamp, tier, ttl_seconds) | Task 1: `EngramMetadata` struct |
| Tier-derived default TTLs | Task 1: `default_ttl_seconds()` |
| Decay = 1.0 for eternal tier | Task 2: `compute_decay` + `eternal_never_decays` test |
| Decay = 1.0 within TTL | Task 2: `fresh_entry_returns_one`, `exactly_at_ttl_returns_one` |
| Gaussian decay past TTL: exp(-staleness²/2) | Task 2: `compute_decay` + curve validation tests |
| Decay ≈ 0.61 at 2× TTL, 0.14 at 3×, 0.01 at 4× | Task 2: three explicit curve tests |
| Ephemeral (TTL=0) instantly decays | Task 2: `ephemeral_same_second_is_fresh`, `ephemeral_one_second_later_is_zero` |
| Connection to gray vector (decay→0 = zero embedding) | Documented in module comment; verified by existing `zero_embedding_returns_zero_residual` test in `engram_residual.rs` |
| `pub fn temporal_decay(entries, now) -> Vec<f32>` | Task 3: batch function |
| Pure function in harmony-engram | All tasks: everything in `chronos.rs` |
| Caller multiplies embedding by decay | Documented in function docstrings; no code needed in this module |

### No Placeholders

Verified: Every step has complete code. No TBD/TODO/placeholder text.

### Type Consistency

- `ChronosTier` — consistent across all tasks
- `EngramMetadata` — consistent (fields: `timestamp: u32`, `tier: ChronosTier`, `ttl_seconds: u32`)
- `compute_decay(&EngramMetadata, u32) -> f32` — consistent
- `temporal_decay(&[EngramMetadata], u32) -> Vec<f32>` — matches spec signature

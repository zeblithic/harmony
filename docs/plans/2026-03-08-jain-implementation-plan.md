# Jain Content Lifecycle Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `harmony-jain`, a sans-I/O content lifecycle engine that serves as gatekeeper (ingest screening), filter (context-appropriate result screening), and housekeeper (staleness scoring, cleanup recommendations, health monitoring).

**Architecture:** Monolithic `JainEngine` struct with `HashMap<ContentId, ContentRecord>` tracking all content on a node. Driven by events (real-time), snapshots (periodic reconciliation), and ticks (periodic scoring). Staleness scoring is a pure function in its own module. Filter rules are evaluated in order, first match wins. All output is via `JainAction` enum variants — no I/O.

**Tech Stack:** Rust (`no_std` compatible), `serde`/`postcard` for serialization, `hashbrown` for `HashMap`, `hex` for Zenoh key encoding, `thiserror` for errors. Depends on `harmony-content` (ContentId) and `harmony-roxy` (ContentCategory).

**Design doc:** `docs/plans/2026-03-08-jain-design.md`

**Reference crate:** `harmony-roxy` — follow the same patterns for Cargo.toml, lib.rs, error module, catalog module, and test style.

---

### Task 1: Crate Scaffold & Error Types

**Files:**
- Create: `crates/harmony-jain/Cargo.toml`
- Create: `crates/harmony-jain/src/lib.rs`
- Create: `crates/harmony-jain/src/error.rs`
- Modify: `Cargo.toml` (workspace root — add member + dependency)

**Step 1: Add harmony-jain to workspace root Cargo.toml**

In `Cargo.toml` (workspace root), add `"crates/harmony-jain"` to the `[workspace] members` list (alphabetically after `harmony-identity`). Also add to `[workspace.dependencies]`:

```toml
harmony-jain = { path = "crates/harmony-jain", default-features = false }
```

**Step 2: Create `crates/harmony-jain/Cargo.toml`**

```toml
[package]
name = "harmony-jain"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "Sans-I/O content lifecycle engine — gatekeeper, filter, and housekeeper for the Harmony decentralized stack"

[features]
default = ["std"]
std = [
    "harmony-content/std",
    "harmony-roxy/std",
    "thiserror/std",
]

[dependencies]
harmony-content = { workspace = true, default-features = false }
harmony-roxy = { workspace = true, default-features = false }
thiserror.workspace = true
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }
hashbrown = { workspace = true }
hex = { workspace = true }
```

**Step 3: Create `crates/harmony-jain/src/error.rs`**

```rust
/// Errors produced by content lifecycle operations.
#[derive(Debug, thiserror::Error)]
pub enum JainError {
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("key expression segment contains Zenoh metacharacters")]
    InvalidKeySegment,
}
```

**Step 4: Create `crates/harmony-jain/src/lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

mod error;
pub use error::JainError;
```

**Step 5: Verify it compiles**

Run: `cargo check -p harmony-jain`
Expected: success, no errors.

**Step 6: Commit**

```bash
git add crates/harmony-jain/ Cargo.toml Cargo.lock
git commit -m "feat(jain): scaffold harmony-jain crate with error types"
```

---

### Task 2: Core Types — ContentOrigin, Sensitivity, SocialContext

**Files:**
- Create: `crates/harmony-jain/src/types.rs`
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod types`)

**Step 1: Write the failing tests**

In `crates/harmony-jain/src/types.rs`, add the module with tests at the bottom:

```rust
//! Core types for content lifecycle tracking.

use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_origin_serialization_round_trip() {
        for origin in [
            ContentOrigin::SelfCreated,
            ContentOrigin::PeerReplicated,
            ContentOrigin::Downloaded,
            ContentOrigin::CachedInTransit,
        ] {
            let bytes = postcard::to_allocvec(&origin).unwrap();
            let decoded: ContentOrigin = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(origin, decoded);
        }
    }

    #[test]
    fn sensitivity_ordering() {
        assert!(Sensitivity::Public < Sensitivity::Private);
        assert!(Sensitivity::Private < Sensitivity::Intimate);
        assert!(Sensitivity::Intimate < Sensitivity::Confidential);
    }

    #[test]
    fn sensitivity_serialization_round_trip() {
        for s in [
            Sensitivity::Public,
            Sensitivity::Private,
            Sensitivity::Intimate,
            Sensitivity::Confidential,
        ] {
            let bytes = postcard::to_allocvec(&s).unwrap();
            let decoded: Sensitivity = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(s, decoded);
        }
    }

    #[test]
    fn social_context_ordering() {
        assert!(SocialContext::Private < SocialContext::Companion);
        assert!(SocialContext::Companion < SocialContext::Social);
        assert!(SocialContext::Social < SocialContext::Professional);
    }

    #[test]
    fn social_context_serialization_round_trip() {
        for ctx in [
            SocialContext::Private,
            SocialContext::Companion,
            SocialContext::Social,
            SocialContext::Professional,
        ] {
            let bytes = postcard::to_allocvec(&ctx).unwrap();
            let decoded: SocialContext = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(ctx, decoded);
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `ContentOrigin`, `Sensitivity`, `SocialContext` not defined.

**Step 3: Write the type definitions (above the tests)**

```rust
/// How content arrived on this node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum ContentOrigin {
    /// User created this content on this device.
    SelfCreated = 0,
    /// Replicated from a trusted peer for safekeeping.
    PeerReplicated = 1,
    /// Downloaded/purchased via Roxy license.
    Downloaded = 2,
    /// Cached opportunistically from transit traffic (W-TinyLFU admitted).
    CachedInTransit = 3,
}

/// Sensitivity classification for content filtering.
///
/// Ordered from least to most sensitive. `PartialOrd`/`Ord` derive
/// uses discriminant order, so `Public < Private < Intimate < Confidential`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum Sensitivity {
    /// Safe in any context.
    Public = 0,
    /// Personal but not sensitive (journal entries, financial docs).
    Private = 1,
    /// Adult/romantic content.
    Intimate = 2,
    /// Secrets, keys, legal documents.
    Confidential = 3,
}

/// Social context hint from Wylene describing the user's current situation.
///
/// Ordered from most private to most public. `PartialOrd`/`Ord` derive
/// uses discriminant order, so `Private < Companion < Social < Professional`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum SocialContext {
    /// Alone, at home.
    Private = 0,
    /// With trusted partner/close friend.
    Companion = 1,
    /// Casual social setting.
    Social = 2,
    /// Work, presentation, public screen.
    Professional = 3,
}
```

**Step 4: Add `pub mod types` to `lib.rs`**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod types;

mod error;
pub use error::JainError;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: 5 tests pass.

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/types.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): core types — ContentOrigin, Sensitivity, SocialContext"
```

---

### Task 3: ContentRecord and StalenessScore

**Files:**
- Modify: `crates/harmony-jain/src/types.rs` (add ContentRecord, StalenessScore)

**Step 1: Write the failing tests**

Add to the `tests` module in `types.rs`:

```rust
    #[test]
    fn staleness_score_clamps_to_unit_range() {
        let low = StalenessScore::new(-0.5);
        assert_eq!(low.value(), 0.0);

        let high = StalenessScore::new(1.5);
        assert_eq!(high.value(), 1.0);

        let mid = StalenessScore::new(0.42);
        assert!((mid.value() - 0.42).abs() < f64::EPSILON);
    }

    #[test]
    fn staleness_score_fresh_and_stale() {
        let fresh = StalenessScore::FRESH;
        assert_eq!(fresh.value(), 0.0);

        let stale = StalenessScore::STALE;
        assert_eq!(stale.value(), 1.0);
    }

    #[test]
    fn content_record_tracks_basic_fields() {
        let cid = make_cid(b"test-content");
        let record = ContentRecord {
            cid,
            size_bytes: 1024,
            content_type: harmony_roxy::catalog::ContentCategory::Music,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            stored_at: 1000.0,
            last_accessed: 1000.0,
            access_count: 0,
            replica_count: 1,
            pinned: false,
            licensed: false,
        };
        assert_eq!(record.size_bytes, 1024);
        assert_eq!(record.origin, ContentOrigin::SelfCreated);
        assert!(!record.pinned);
    }
```

Also add the test helper above the `tests` module (inside `#[cfg(test)]`):

```rust
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `StalenessScore`, `ContentRecord` not defined.

**Step 3: Write the implementations**

Add to `types.rs` (after the existing enums, before `#[cfg(test)]`):

```rust
use harmony_content::ContentId;
use harmony_roxy::catalog::ContentCategory;

/// Jain's internal model of a tracked piece of content.
#[derive(Debug, Clone)]
pub struct ContentRecord {
    /// Content identifier.
    pub cid: ContentId,
    /// Size in bytes on disk.
    pub size_bytes: u64,
    /// What kind of content this is.
    pub content_type: ContentCategory,
    /// How this content arrived on the node.
    pub origin: ContentOrigin,
    /// Sensitivity classification for filtering.
    pub sensitivity: Sensitivity,
    /// Timestamp when content was first stored.
    pub stored_at: f64,
    /// Timestamp of most recent access.
    pub last_accessed: f64,
    /// Lifetime access count.
    pub access_count: u64,
    /// Known copies on trusted peers.
    pub replica_count: u8,
    /// User explicitly pinned this content.
    pub pinned: bool,
    /// Has an active Roxy license.
    pub licensed: bool,
}

/// Staleness score: 0.0 (fresh/vital) to 1.0 (almost certainly unneeded).
///
/// Clamped to `[0.0, 1.0]` on construction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StalenessScore(f64);

impl StalenessScore {
    /// Completely fresh — never suggest cleanup.
    pub const FRESH: Self = StalenessScore(0.0);

    /// Completely stale — highest priority for cleanup.
    pub const STALE: Self = StalenessScore(1.0);

    /// Create a new score, clamping to `[0.0, 1.0]`.
    pub fn new(value: f64) -> Self {
        StalenessScore(value.clamp(0.0, 1.0))
    }

    /// Get the score value.
    pub fn value(self) -> f64 {
        self.0
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (5 previous + 3 new = 8).

**Step 5: Commit**

```bash
git add crates/harmony-jain/src/types.rs
git commit -m "feat(jain): ContentRecord and StalenessScore types"
```

---

### Task 4: Configuration — JainConfig, FilterRule, FilterRuleSet

**Files:**
- Create: `crates/harmony-jain/src/config.rs`
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod config`)

**Step 1: Write the failing tests**

Create `crates/harmony-jain/src/config.rs` with tests:

```rust
//! Configuration for the Jain content lifecycle engine.

use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::types::{Sensitivity, SocialContext};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_thresholds() {
        let config = JainConfig::default();
        assert!(config.archive_threshold < config.burn_threshold);
        assert!(config.burn_threshold <= 1.0);
        assert!(config.min_replica_count >= 1);
        assert!(config.storage_alert_percent > 0.0);
        assert!(config.storage_alert_percent < 1.0);
    }

    #[test]
    fn default_filter_rules_block_intimate_in_professional() {
        let rules = FilterRuleSet::default();
        assert!(rules.enabled);
        assert!(!rules.rules.is_empty());
    }

    #[test]
    fn filter_rule_set_serialization_round_trip() {
        let rules = FilterRuleSet::default();
        let bytes = postcard::to_allocvec(&rules).unwrap();
        let decoded: FilterRuleSet = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.enabled, rules.enabled);
        assert_eq!(decoded.rules.len(), rules.rules.len());
    }

    #[test]
    fn config_serialization_round_trip() {
        let config = JainConfig::default();
        let bytes = postcard::to_allocvec(&config).unwrap();
        let decoded: JainConfig = postcard::from_bytes(&bytes).unwrap();
        assert!((decoded.archive_threshold - config.archive_threshold).abs() < f64::EPSILON);
        assert!((decoded.burn_threshold - config.burn_threshold).abs() < f64::EPSILON);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `JainConfig`, `FilterRule`, `FilterRuleSet` not defined.

**Step 3: Write the implementations**

Add above `#[cfg(test)]` in `config.rs`:

```rust
/// Tunable weights and thresholds for staleness scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JainConfig {
    /// Staleness above this → suggest archiving to cold storage.
    pub archive_threshold: f64,
    /// Staleness above this → suggest burning (permanent deletion).
    pub burn_threshold: f64,
    /// Below this replica count → RepairNeeded.
    pub min_replica_count: u8,
    /// Storage usage fraction that triggers HealthAlert (e.g. 0.85).
    pub storage_alert_percent: f64,
    /// Half-life in seconds for access recency decay.
    pub access_decay_half_life_secs: f64,
    /// Weight reduction for self-created content (0.0–1.0).
    /// Higher values reduce staleness more for user's own content.
    pub self_created_weight: f64,
}

impl Default for JainConfig {
    fn default() -> Self {
        Self {
            archive_threshold: 0.6,
            burn_threshold: 0.85,
            min_replica_count: 2,
            storage_alert_percent: 0.85,
            access_decay_half_life_secs: 30.0 * 24.0 * 3600.0, // 30 days
            self_created_weight: 0.3,
        }
    }
}

/// A single content filter rule.
///
/// Evaluated as: if content sensitivity >= `min_sensitivity` AND current
/// social context > `max_context`, then block (or ask for confirmation).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Applies to content at or above this sensitivity level.
    pub min_sensitivity: Sensitivity,
    /// Blocks when social context is strictly above this level.
    pub max_context: SocialContext,
    /// If true, ask user before surfacing (vs hard block).
    pub require_confirmation: bool,
}

/// Ordered filter rule set — evaluated in order, first match wins.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRuleSet {
    /// Rules evaluated in order. First matching rule determines the decision.
    pub rules: Vec<FilterRule>,
    /// User can disable the entire filter system.
    pub enabled: bool,
}

impl Default for FilterRuleSet {
    /// Ships with sensible defaults:
    /// 1. Intimate content blocked above Companion; confirmation at Companion.
    /// 2. Confidential content only in Private, hard block otherwise.
    /// 3. Private content blocked in Professional context.
    fn default() -> Self {
        Self {
            enabled: true,
            rules: alloc::vec![
                // Rule 1: Intimate → only Private/Companion, confirm at Companion
                FilterRule {
                    min_sensitivity: Sensitivity::Intimate,
                    max_context: SocialContext::Companion,
                    require_confirmation: true,
                },
                // Rule 2: Confidential → Private only, hard block
                FilterRule {
                    min_sensitivity: Sensitivity::Confidential,
                    max_context: SocialContext::Private,
                    require_confirmation: false,
                },
                // Rule 3: Private → blocked in Professional
                FilterRule {
                    min_sensitivity: Sensitivity::Private,
                    max_context: SocialContext::Social,
                    require_confirmation: false,
                },
            ],
        }
    }
}
```

**Step 4: Add `pub mod config` to `lib.rs`**

```rust
pub mod config;
pub mod types;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (8 previous + 4 new = 12).

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/config.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): JainConfig, FilterRule, FilterRuleSet with defaults"
```

---

### Task 5: Staleness Scoring — Pure Function

**Files:**
- Create: `crates/harmony-jain/src/scoring.rs`
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod scoring`)

**Step 1: Write the failing tests**

Create `crates/harmony-jain/src/scoring.rs` with tests:

```rust
//! Staleness scoring — pure function from ContentRecord + JainConfig → StalenessScore.

use crate::config::JainConfig;
use crate::types::{ContentRecord, StalenessScore};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentOrigin, Sensitivity};
    use harmony_content::ContentFlags;
    use harmony_roxy::catalog::ContentCategory;

    fn make_cid(data: &[u8]) -> harmony_content::ContentId {
        harmony_content::ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    fn base_record() -> ContentRecord {
        ContentRecord {
            cid: make_cid(b"test"),
            size_bytes: 1024,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            stored_at: 0.0,
            last_accessed: 0.0,
            access_count: 1,
            replica_count: 3,
            pinned: false,
            licensed: false,
        }
    }

    #[test]
    fn recently_accessed_content_scores_low() {
        let config = JainConfig::default();
        let mut record = base_record();
        record.last_accessed = 100.0;
        let score = staleness_score(&record, &config, 100.0);
        assert!(score.value() < 0.1, "score was {}", score.value());
    }

    #[test]
    fn old_untouched_content_scores_high() {
        let config = JainConfig::default();
        let mut record = base_record();
        record.last_accessed = 0.0;
        // 4 half-lives later
        let now = 4.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert!(score.value() > 0.8, "score was {}", score.value());
    }

    #[test]
    fn pinned_content_always_fresh() {
        let config = JainConfig::default();
        let mut record = base_record();
        record.pinned = true;
        record.last_accessed = 0.0;
        let now = 10.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert_eq!(score.value(), 0.0);
    }

    #[test]
    fn licensed_content_always_fresh() {
        let config = JainConfig::default();
        let mut record = base_record();
        record.licensed = true;
        record.last_accessed = 0.0;
        let now = 10.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert_eq!(score.value(), 0.0);
    }

    #[test]
    fn under_replicated_content_always_fresh() {
        let config = JainConfig::default();
        let mut record = base_record();
        record.replica_count = 1; // below min_replica_count (2)
        record.last_accessed = 0.0;
        let now = 10.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert_eq!(score.value(), 0.0);
    }

    #[test]
    fn self_created_content_scores_lower_than_transit() {
        let config = JainConfig::default();
        let now = 2.0 * config.access_decay_half_life_secs;

        let mut self_created = base_record();
        self_created.origin = ContentOrigin::SelfCreated;
        self_created.last_accessed = 0.0;

        let mut transit = base_record();
        transit.origin = ContentOrigin::CachedInTransit;
        transit.last_accessed = 0.0;

        let self_score = staleness_score(&self_created, &config, now);
        let transit_score = staleness_score(&transit, &config, now);
        assert!(
            self_score.value() < transit_score.value(),
            "self={} transit={}",
            self_score.value(),
            transit_score.value()
        );
    }

    #[test]
    fn frequently_accessed_content_scores_lower() {
        let config = JainConfig::default();
        let now = 2.0 * config.access_decay_half_life_secs;

        let mut frequent = base_record();
        frequent.last_accessed = 0.0;
        frequent.access_count = 100;

        let mut rare = base_record();
        rare.last_accessed = 0.0;
        rare.access_count = 1;

        let freq_score = staleness_score(&frequent, &config, now);
        let rare_score = staleness_score(&rare, &config, now);
        assert!(
            freq_score.value() < rare_score.value(),
            "frequent={} rare={}",
            freq_score.value(),
            rare_score.value()
        );
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `staleness_score` not defined.

**Step 3: Write the implementation**

Add above `#[cfg(test)]` in `scoring.rs`:

```rust
/// Compute the staleness score for a content record.
///
/// Returns `StalenessScore::FRESH` (0.0) for pinned, licensed, or
/// under-replicated content. Otherwise computes a score based on:
/// - Exponential decay of time since last access (half-life from config)
/// - Access frequency (higher count → lower staleness)
/// - Origin bonus (self-created content gets reduced staleness)
///
/// The result is clamped to `[0.0, 1.0]`.
pub fn staleness_score(record: &ContentRecord, config: &JainConfig, now: f64) -> StalenessScore {
    // Hard overrides — never suggest cleanup for these.
    if record.pinned || record.licensed || record.replica_count < config.min_replica_count {
        return StalenessScore::FRESH;
    }

    // Time-based decay: how many half-lives since last access?
    let elapsed = (now - record.last_accessed).max(0.0);
    let half_lives = elapsed / config.access_decay_half_life_secs;
    // Recency factor: 1.0 when just accessed, approaches 0.0 over time.
    let recency = 0.5_f64.powf(half_lives);

    // Frequency factor: diminishing returns on access count.
    // ln(1 + count) / ln(1 + count + 10) gives a 0..1 range that
    // increases with count but saturates.
    let freq = (1.0 + record.access_count as f64).ln() / (11.0 + record.access_count as f64).ln();

    // Combined freshness from recency and frequency.
    // Both contribute — recently accessed OR frequently accessed = fresh.
    let freshness = 0.7 * recency + 0.3 * freq;

    // Origin bonus: self-created content is less stale.
    let origin_bonus = match record.origin {
        ContentOrigin::SelfCreated => config.self_created_weight,
        _ => 0.0,
    };

    // Staleness is the inverse of freshness, reduced by origin bonus.
    let staleness = (1.0 - freshness) * (1.0 - origin_bonus);

    StalenessScore::new(staleness)
}
```

**Step 4: Add `pub mod scoring` to `lib.rs`**

```rust
pub mod config;
pub mod scoring;
pub mod types;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (12 previous + 7 new = 19).

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/scoring.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): staleness scoring pure function with decay and overrides"
```

---

### Task 6: Events, Actions & Decision Types

**Files:**
- Create: `crates/harmony-jain/src/actions.rs`
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod actions`)

**Step 1: Create `crates/harmony-jain/src/actions.rs` with all event/action types and tests**

```rust
//! Events, actions, and decisions for the Jain content lifecycle engine.

use alloc::string::String;
use harmony_content::ContentId;
use harmony_roxy::catalog::ContentCategory;
use serde::{Deserialize, Serialize};

use crate::types::{ContentOrigin, Sensitivity, StalenessScore};

/// Events Jain receives from the content subsystem.
#[derive(Debug, Clone)]
pub enum ContentEvent {
    /// New content stored on this node.
    Stored {
        cid: ContentId,
        size_bytes: u64,
        content_type: ContentCategory,
        origin: ContentOrigin,
        sensitivity: Sensitivity,
        timestamp: f64,
    },
    /// Existing content was accessed (read/streamed/used).
    Accessed { cid: ContentId, timestamp: f64 },
    /// Content was deleted from this node.
    Deleted { cid: ContentId },
    /// User pinned content (protect from cleanup).
    Pinned { cid: ContentId },
    /// User unpinned content (allow cleanup again).
    Unpinned { cid: ContentId },
    /// Roxy license granted for this content.
    LicenseGranted { cid: ContentId },
    /// Roxy license expired for this content.
    LicenseExpired { cid: ContentId },
    /// Replica count changed on trusted peers.
    ReplicaChanged { cid: ContentId, new_count: u8 },
}

/// Content presented for ingest evaluation (gatekeeper).
#[derive(Debug, Clone)]
pub struct IngestCandidate {
    pub cid: ContentId,
    pub size_bytes: u64,
    pub content_type: ContentCategory,
    pub origin: ContentOrigin,
    pub sensitivity: Sensitivity,
}

/// Jain's gatekeeper decision on incoming content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestDecision {
    /// Store and send to Oluo for indexing.
    IndexAndStore,
    /// Store but don't index (private, sensitive, or low-value).
    StoreOnly,
    /// Reject — don't store at all.
    Reject { reason: RejectReason },
}

/// Why content was rejected at ingest.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// Node storage budget would be exceeded.
    StorageBudgetExceeded,
    /// Exact-hash duplicate already tracked.
    DuplicateContent,
}

/// Jain's outbound filter decision on search results.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterDecision {
    /// Safe to show in current context.
    Allow,
    /// Blocked by filter rules — don't show.
    Block,
    /// Requires user confirmation before showing.
    Confirm,
}

/// Actions Jain emits for the caller to execute.
#[derive(Debug, Clone)]
pub enum JainAction {
    /// Recommend burning content (permanent deletion).
    RecommendBurn { recommendation: CleanupRecommendation },
    /// Recommend archiving to cold storage.
    RecommendArchive { recommendation: CleanupRecommendation },
    /// Exact-hash duplicate detected — recommend dedup.
    RecommendDedup { keep: ContentId, burn: ContentId },
    /// Ask Oluo if similar/equivalent content exists externally.
    QueryOluo { cid: ContentId, query_hint: QueryHint },
    /// Replica count below safety threshold.
    RepairNeeded { cid: ContentId, current_replicas: u8, desired: u8 },
    /// Operational health alert.
    HealthAlert { alert: HealthAlertKind },
}

/// A cleanup recommendation with supporting data.
#[derive(Debug, Clone)]
pub struct CleanupRecommendation {
    /// Content to clean up.
    pub cid: ContentId,
    /// Why Jain is suggesting this.
    pub reason: CleanupReason,
    /// How stale this content is.
    pub staleness: StalenessScore,
    /// Bytes recovered if this recommendation is accepted.
    pub space_recovered_bytes: u64,
    /// 0.0–1.0 confidence in this recommendation.
    pub confidence: f64,
}

/// Why Jain is suggesting cleanup.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupReason {
    /// Not accessed in a long time.
    Stale,
    /// Might have a public equivalent (encrypted version of public content).
    EncryptedVersionOfPublic,
    /// Node is running low on space.
    OverStorageBudget,
}

/// Hint for Oluo about what kind of similarity check to perform.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QueryHint {
    /// Check if a public/unencrypted version exists.
    FindPublicEquivalent,
    /// Check if near-duplicates exist on this node.
    FindLocalDuplicates,
}

/// Types of operational health alerts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HealthAlertKind {
    /// Storage usage above threshold.
    StorageNearFull { used_percent_x100: u32 },
    /// Records in Jain's tracker with no backing content on disk.
    StaleReconciliation { records_without_backing: u32 },
    /// Content below minimum replica count.
    ReplicaDeficit { affected_records: u32 },
}

/// Snapshot entry for periodic reconciliation.
#[derive(Debug, Clone)]
pub struct SnapshotEntry {
    /// Content identifier.
    pub cid: ContentId,
    /// Size on disk.
    pub size_bytes: u64,
    /// Whether the content actually exists on disk.
    pub exists_on_disk: bool,
}

/// Node health summary.
#[derive(Debug, Clone)]
pub struct HealthReport {
    pub total_records: u32,
    pub total_bytes: u64,
    pub storage_used_percent_x100: u32,
    pub under_replicated_count: u32,
    pub stale_count: u32,
    pub pinned_count: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    #[test]
    fn ingest_decision_equality() {
        assert_eq!(IngestDecision::IndexAndStore, IngestDecision::IndexAndStore);
        assert_eq!(IngestDecision::StoreOnly, IngestDecision::StoreOnly);
        assert_eq!(
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent
            },
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent
            }
        );
        assert_ne!(IngestDecision::IndexAndStore, IngestDecision::StoreOnly);
    }

    #[test]
    fn cleanup_reason_serialization_round_trip() {
        for reason in [
            CleanupReason::Stale,
            CleanupReason::EncryptedVersionOfPublic,
            CleanupReason::OverStorageBudget,
        ] {
            let bytes = postcard::to_allocvec(&reason).unwrap();
            let decoded: CleanupReason = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(reason, decoded);
        }
    }

    #[test]
    fn query_hint_serialization_round_trip() {
        for hint in [
            QueryHint::FindPublicEquivalent,
            QueryHint::FindLocalDuplicates,
        ] {
            let bytes = postcard::to_allocvec(&hint).unwrap();
            let decoded: QueryHint = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(hint, decoded);
        }
    }

    #[test]
    fn reject_reason_serialization_round_trip() {
        for reason in [
            RejectReason::StorageBudgetExceeded,
            RejectReason::DuplicateContent,
        ] {
            let bytes = postcard::to_allocvec(&reason).unwrap();
            let decoded: RejectReason = postcard::from_bytes(&bytes).unwrap();
            assert_eq!(reason, decoded);
        }
    }
}
```

**Note:** `HealthAlertKind` uses `used_percent_x100: u32` (e.g. 8500 = 85.00%) instead of `f64` to keep the type `Eq`-derivable without float comparison hassles. The engine converts from `f64` internally.

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (19 previous + 4 new = 23).

**Step 3: Add `pub mod actions` to `lib.rs`**

```rust
pub mod actions;
pub mod config;
pub mod scoring;
pub mod types;
```

**Step 4: Commit**

```bash
git add crates/harmony-jain/src/actions.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): events, actions, and decision types"
```

---

### Task 7: Zenoh Key Expression Builders

**Files:**
- Create: `crates/harmony-jain/src/catalog.rs`
- Modify: `crates/harmony-jain/src/error.rs` (already has `InvalidKeySegment`)
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod catalog`)

**Step 1: Write the failing tests**

Create `crates/harmony-jain/src/catalog.rs`:

```rust
//! Zenoh key expression patterns for Jain health and recommendation topics.

use alloc::string::String;

use crate::error::JainError;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn health_key_format() {
        let hash = [0xABu8; 16];
        let key = health_key(&hash);
        assert_eq!(key, "jain/health/abababababababababababababababab");
    }

    #[test]
    fn recommend_key_format() {
        let hash = [0xCDu8; 16];
        let key = recommend_key(&hash);
        assert_eq!(key, "jain/recommend/cdcdcdcdcdcdcdcdcdcdcdcdcdcdcdcd");
    }

    #[test]
    fn action_key_format() {
        let hash = [0xEFu8; 16];
        let key = action_key(&hash, "abc123").unwrap();
        assert_eq!(
            key,
            "jain/action/efefefefefefefefefefefefefefefef/abc123"
        );
    }

    #[test]
    fn stats_key_format() {
        let hash = [0x01u8; 16];
        let key = stats_key(&hash);
        assert_eq!(key, "jain/stats/01010101010101010101010101010101");
    }

    #[test]
    fn action_key_rejects_metacharacters() {
        let hash = [0xABu8; 16];
        assert!(action_key(&hash, "a/b").is_err());
        assert!(action_key(&hash, "a*b").is_err());
    }

    #[test]
    fn action_key_rejects_empty_segment() {
        let hash = [0xABu8; 16];
        assert!(action_key(&hash, "").is_err());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — functions not defined.

**Step 3: Write the implementations**

Add above `#[cfg(test)]` in `catalog.rs`:

```rust
/// Validate that a key expression segment contains no Zenoh metacharacters.
fn validate_segment(s: &str) -> Result<(), JainError> {
    if s.is_empty() || s.contains('/') || s.contains('*') {
        Err(JainError::InvalidKeySegment)
    } else {
        Ok(())
    }
}

/// Build a Zenoh key expression for a node health report.
///
/// Returns `"jain/health/{hex_hash}"`.
pub fn health_key(node_hash: &[u8; 16]) -> String {
    alloc::format!("jain/health/{}", hex::encode(node_hash))
}

/// Build a Zenoh key expression for cleanup recommendations.
///
/// Returns `"jain/recommend/{hex_hash}"`.
pub fn recommend_key(user_hash: &[u8; 16]) -> String {
    alloc::format!("jain/recommend/{}", hex::encode(user_hash))
}

/// Build a Zenoh key expression for a user-approved cleanup action.
///
/// Returns `"jain/action/{hex_hash}/{action_id}"`.
///
/// Returns `Err(JainError::InvalidKeySegment)` if `action_id` contains
/// Zenoh metacharacters (`/` or `*`).
pub fn action_key(user_hash: &[u8; 16], action_id: &str) -> Result<String, JainError> {
    validate_segment(action_id)?;
    Ok(alloc::format!(
        "jain/action/{}/{}",
        hex::encode(user_hash),
        action_id
    ))
}

/// Build a Zenoh key expression for node storage/metrics stats.
///
/// Returns `"jain/stats/{hex_hash}"`.
pub fn stats_key(node_hash: &[u8; 16]) -> String {
    alloc::format!("jain/stats/{}", hex::encode(node_hash))
}
```

**Step 4: Add `pub mod catalog` to `lib.rs`**

```rust
pub mod actions;
pub mod catalog;
pub mod config;
pub mod scoring;
pub mod types;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (23 previous + 6 new = 29).

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/catalog.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): Zenoh key expression builders for health, recommend, action, stats"
```

---

### Task 8: JainEngine — Constructor, record_count, health_report

**Files:**
- Create: `crates/harmony-jain/src/engine.rs`
- Modify: `crates/harmony-jain/src/lib.rs` (add `pub mod engine`)

**Step 1: Write the failing tests**

Create `crates/harmony-jain/src/engine.rs`:

```rust
//! Sans-I/O content lifecycle engine.

use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_content::ContentId;

use crate::actions::*;
use crate::config::{FilterRuleSet, JainConfig};
use crate::types::ContentRecord;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{ContentOrigin, Sensitivity};
    use harmony_content::ContentFlags;
    use harmony_roxy::catalog::ContentCategory;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    fn default_engine() -> JainEngine {
        JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 10_000)
    }

    fn base_record(cid: ContentId) -> ContentRecord {
        ContentRecord {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            stored_at: 0.0,
            last_accessed: 0.0,
            access_count: 1,
            replica_count: 3,
            pinned: false,
            licensed: false,
        }
    }

    #[test]
    fn new_engine_is_empty() {
        let engine = default_engine();
        assert_eq!(engine.record_count(), 0);
    }

    #[test]
    fn health_report_empty_engine() {
        let engine = default_engine();
        let report = engine.health_report();
        assert_eq!(report.total_records, 0);
        assert_eq!(report.total_bytes, 0);
        assert_eq!(report.storage_used_percent_x100, 0);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `JainEngine` not defined.

**Step 3: Write the engine struct, constructor, record_count, health_report**

Add above `#[cfg(test)]` in `engine.rs`:

```rust
/// Sans-I/O content lifecycle engine.
///
/// Tracks content on a node via a `HashMap<ContentId, ContentRecord>`.
/// Driven by events (real-time), snapshots (periodic), and ticks (periodic).
/// Emits `JainAction`s for the caller to execute. Never performs I/O.
pub struct JainEngine {
    records: HashMap<ContentId, ContentRecord>,
    config: JainConfig,
    filter_rules: FilterRuleSet,
    total_storage_bytes: u64,
    storage_capacity_bytes: u64,
}

impl JainEngine {
    /// Create a new engine with the given configuration.
    pub fn new(
        config: JainConfig,
        filter_rules: FilterRuleSet,
        storage_capacity_bytes: u64,
    ) -> Self {
        Self {
            records: HashMap::new(),
            config,
            filter_rules,
            total_storage_bytes: 0,
            storage_capacity_bytes,
        }
    }

    /// Number of tracked content records.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Current health summary.
    pub fn health_report(&self) -> HealthReport {
        let mut under_replicated = 0u32;
        let mut stale = 0u32;
        let mut pinned = 0u32;

        for record in self.records.values() {
            if record.replica_count < self.config.min_replica_count {
                under_replicated += 1;
            }
            if record.pinned {
                pinned += 1;
            }
        }

        let used_percent = if self.storage_capacity_bytes > 0 {
            (self.total_storage_bytes as f64 / self.storage_capacity_bytes as f64) * 100.0
        } else {
            0.0
        };

        HealthReport {
            total_records: self.records.len() as u32,
            total_bytes: self.total_storage_bytes,
            storage_used_percent_x100: (used_percent * 100.0) as u32,
            under_replicated_count: under_replicated,
            stale_count: stale,
            pinned_count: pinned,
        }
    }
}

impl Default for JainEngine {
    fn default() -> Self {
        Self::new(JainConfig::default(), FilterRuleSet::default(), 0)
    }
}
```

**Note:** `stale_count` in `health_report` is always 0 at this stage — it requires calling `staleness_score` for each record, which we'll add in Task 10 when we implement `tick()`. For now the field exists but isn't computed.

**Step 4: Add `pub mod engine` to `lib.rs`**

```rust
pub mod actions;
pub mod catalog;
pub mod config;
pub mod engine;
pub mod scoring;
pub mod types;
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (29 previous + 2 new = 31).

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/engine.rs crates/harmony-jain/src/lib.rs
git commit -m "feat(jain): JainEngine struct with constructor and health_report"
```

---

### Task 9: JainEngine — handle_event and evaluate_ingest

**Files:**
- Modify: `crates/harmony-jain/src/engine.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `engine.rs`:

```rust
    #[test]
    fn handle_stored_event_creates_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        let actions = engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        assert!(actions.is_empty());
        assert_eq!(engine.record_count(), 1);
        let report = engine.health_report();
        assert_eq!(report.total_bytes, 500);
    }

    #[test]
    fn handle_stored_duplicate_emits_dedup() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        let actions = engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 200.0,
        });
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], JainAction::RecommendDedup { .. }));
    }

    #[test]
    fn handle_accessed_updates_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::Accessed {
            cid,
            timestamp: 200.0,
        });
        // Record count unchanged, but internal state updated.
        assert_eq!(engine.record_count(), 1);
    }

    #[test]
    fn handle_deleted_removes_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::Deleted { cid });
        assert_eq!(engine.record_count(), 0);
        assert_eq!(engine.health_report().total_bytes, 0);
    }

    #[test]
    fn handle_pinned_and_unpinned() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::Pinned { cid });
        assert_eq!(engine.health_report().pinned_count, 1);
        engine.handle_event(ContentEvent::Unpinned { cid });
        assert_eq!(engine.health_report().pinned_count, 0);
    }

    #[test]
    fn handle_license_granted_and_expired() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::LicenseGranted { cid });
        // Licensed content is protected — verify via scoring later.
        engine.handle_event(ContentEvent::LicenseExpired { cid });
        assert_eq!(engine.record_count(), 1);
    }

    #[test]
    fn handle_replica_changed() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged {
            cid,
            new_count: 1,
        });
        // With min_replica_count=2, this should show as under-replicated.
        assert_eq!(engine.health_report().under_replicated_count, 1);
    }

    #[test]
    fn evaluate_ingest_allows_normal_content() {
        let engine = default_engine();
        let decision = engine.evaluate_ingest(&IngestCandidate {
            cid: make_cid(b"new-song"),
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
        });
        assert_eq!(decision, IngestDecision::IndexAndStore);
    }

    #[test]
    fn evaluate_ingest_rejects_duplicate() {
        let mut engine = default_engine();
        let cid = make_cid(b"song");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        let decision = engine.evaluate_ingest(&IngestCandidate {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
        });
        assert_eq!(
            decision,
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent
            }
        );
    }

    #[test]
    fn evaluate_ingest_rejects_over_budget() {
        let engine = JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 50);
        let decision = engine.evaluate_ingest(&IngestCandidate {
            cid: make_cid(b"huge-file"),
            size_bytes: 100,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
        });
        assert_eq!(
            decision,
            IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded
            }
        );
    }

    #[test]
    fn evaluate_ingest_stores_only_for_confidential() {
        let engine = default_engine();
        let decision = engine.evaluate_ingest(&IngestCandidate {
            cid: make_cid(b"secret"),
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Confidential,
        });
        assert_eq!(decision, IngestDecision::StoreOnly);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — methods not defined.

**Step 3: Write the implementations**

Add to the `impl JainEngine` block in `engine.rs`:

```rust
    /// Process a content lifecycle event. Returns immediate actions.
    pub fn handle_event(&mut self, event: ContentEvent) -> Vec<JainAction> {
        let mut actions = Vec::new();

        match event {
            ContentEvent::Stored {
                cid,
                size_bytes,
                content_type,
                origin,
                sensitivity,
                timestamp,
            } => {
                if self.records.contains_key(&cid) {
                    actions.push(JainAction::RecommendDedup {
                        keep: cid,
                        burn: cid,
                    });
                    return actions;
                }
                self.total_storage_bytes += size_bytes;
                self.records.insert(
                    cid,
                    ContentRecord {
                        cid,
                        size_bytes,
                        content_type,
                        origin,
                        sensitivity,
                        stored_at: timestamp,
                        last_accessed: timestamp,
                        access_count: 0,
                        replica_count: 1,
                        pinned: false,
                        licensed: false,
                    },
                );
            }
            ContentEvent::Accessed { cid, timestamp } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.last_accessed = timestamp;
                    record.access_count += 1;
                }
            }
            ContentEvent::Deleted { cid } => {
                if let Some(record) = self.records.remove(&cid) {
                    self.total_storage_bytes =
                        self.total_storage_bytes.saturating_sub(record.size_bytes);
                }
            }
            ContentEvent::Pinned { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.pinned = true;
                }
            }
            ContentEvent::Unpinned { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.pinned = false;
                }
            }
            ContentEvent::LicenseGranted { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.licensed = true;
                }
            }
            ContentEvent::LicenseExpired { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.licensed = false;
                }
            }
            ContentEvent::ReplicaChanged { cid, new_count } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.replica_count = new_count;
                }
            }
        }

        actions
    }

    /// Gatekeeper: evaluate incoming content before storing.
    ///
    /// Returns `IndexAndStore` for normal public content, `StoreOnly` for
    /// sensitive content (Confidential or Intimate), and `Reject` for
    /// duplicates or content that would exceed the storage budget.
    pub fn evaluate_ingest(&self, candidate: &IngestCandidate) -> IngestDecision {
        // Reject exact duplicates.
        if self.records.contains_key(&candidate.cid) {
            return IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            };
        }

        // Reject if over budget.
        if self.total_storage_bytes + candidate.size_bytes > self.storage_capacity_bytes {
            return IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            };
        }

        // Confidential or Intimate content: store but don't send to Oluo.
        if candidate.sensitivity >= Sensitivity::Intimate {
            return IngestDecision::StoreOnly;
        }

        IngestDecision::IndexAndStore
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (31 previous + 11 new = 42).

**Step 5: Commit**

```bash
git add crates/harmony-jain/src/engine.rs
git commit -m "feat(jain): handle_event and evaluate_ingest on JainEngine"
```

---

### Task 10: JainEngine — filter_result

**Files:**
- Modify: `crates/harmony-jain/src/engine.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `engine.rs`:

```rust
    use crate::types::SocialContext;

    #[test]
    fn filter_allows_public_content_in_any_context() {
        let mut engine = default_engine();
        let cid = make_cid(b"public-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_blocks_intimate_in_professional() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Block
        );
    }

    #[test]
    fn filter_blocks_intimate_in_social() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Social),
            FilterDecision::Block
        );
    }

    #[test]
    fn filter_confirms_intimate_at_companion() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Companion),
            FilterDecision::Confirm
        );
    }

    #[test]
    fn filter_allows_intimate_in_private() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Private),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_blocks_confidential_outside_private() {
        let mut engine = default_engine();
        let cid = make_cid(b"secret-doc");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Confidential,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Companion),
            FilterDecision::Block
        );
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Private),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_allows_everything_when_disabled() {
        let mut engine = JainEngine::new(
            JainConfig::default(),
            FilterRuleSet {
                enabled: false,
                rules: alloc::vec![],
            },
            10_000,
        );
        let cid = make_cid(b"intimate-photo");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 100.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_allows_unknown_cid() {
        let engine = default_engine();
        let cid = make_cid(b"not-tracked");
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `filter_result` not defined.

**Step 3: Write the implementation**

Add to the `impl JainEngine` block:

```rust
    /// Outbound filter: screen a search result for context appropriateness.
    ///
    /// Evaluates filter rules in order. First matching rule determines the
    /// decision. If no rule matches, or if the CID is unknown, returns `Allow`.
    /// If filtering is disabled, always returns `Allow`.
    pub fn filter_result(&self, cid: &ContentId, context: SocialContext) -> FilterDecision {
        if !self.filter_rules.enabled {
            return FilterDecision::Allow;
        }

        let record = match self.records.get(cid) {
            Some(r) => r,
            None => return FilterDecision::Allow,
        };

        for rule in &self.filter_rules.rules {
            if record.sensitivity >= rule.min_sensitivity && context > rule.max_context {
                return if rule.require_confirmation {
                    FilterDecision::Confirm
                } else {
                    FilterDecision::Block
                };
            }

            // Special case: sensitivity matches and context equals max_context
            // with require_confirmation — this is the "at boundary" case.
            if record.sensitivity >= rule.min_sensitivity
                && context == rule.max_context
                && rule.require_confirmation
            {
                return FilterDecision::Confirm;
            }
        }

        FilterDecision::Allow
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (42 previous + 8 new = 50).

**Step 5: Commit**

```bash
git add crates/harmony-jain/src/engine.rs
git commit -m "feat(jain): context-aware content filtering with default rules"
```

---

### Task 11: JainEngine — tick (staleness recommendations + health alerts)

**Files:**
- Modify: `crates/harmony-jain/src/engine.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `engine.rs`:

```rust
    #[test]
    fn tick_recommends_burn_for_very_stale_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"old-transit");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            stored_at: 0.0,
            timestamp: 0.0,
        });
        // 10 half-lives later — very stale.
        let now = 10.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        assert!(
            actions.iter().any(|a| matches!(a, JainAction::RecommendBurn { .. })),
            "expected RecommendBurn, got {:?}",
            actions
        );
    }

    #[test]
    fn tick_recommends_archive_for_moderately_stale_content() {
        let config = JainConfig {
            archive_threshold: 0.3,
            burn_threshold: 0.95,
            ..JainConfig::default()
        };
        let mut engine = JainEngine::new(config, FilterRuleSet::default(), 10_000);
        let cid = make_cid(b"medium-old");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        // ~2 half-lives — moderately stale but not burn-worthy.
        let now = 2.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, JainAction::RecommendArchive { .. })),
            "expected RecommendArchive, got {:?}",
            actions
        );
    }

    #[test]
    fn tick_skips_pinned_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"pinned-old");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::Pinned { cid });
        let now = 10.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        assert!(
            !actions.iter().any(|a| matches!(
                a,
                JainAction::RecommendBurn { .. } | JainAction::RecommendArchive { .. }
            )),
            "pinned content should not be recommended for cleanup"
        );
    }

    #[test]
    fn tick_emits_repair_needed_for_under_replicated() {
        let mut engine = default_engine();
        let cid = make_cid(b"lonely");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged {
            cid,
            new_count: 1,
        });
        let actions = engine.tick(100.0);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, JainAction::RepairNeeded { .. })),
            "expected RepairNeeded, got {:?}",
            actions
        );
    }

    #[test]
    fn tick_emits_storage_near_full_alert() {
        let mut engine = JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 1000);
        let cid = make_cid(b"big-file");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 900,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        let actions = engine.tick(100.0);
        assert!(
            actions.iter().any(|a| matches!(
                a,
                JainAction::HealthAlert {
                    alert: HealthAlertKind::StorageNearFull { .. }
                }
            )),
            "expected StorageNearFull alert, got {:?}",
            actions
        );
    }

    #[test]
    fn tick_no_actions_for_fresh_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"fresh");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        engine.handle_event(ContentEvent::Accessed {
            cid,
            timestamp: 100.0,
        });
        let actions = engine.tick(100.0);
        assert!(actions.is_empty(), "expected no actions, got {:?}", actions);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `tick` not defined.

**Step 3: Write the implementation**

Add to the `impl JainEngine` block:

```rust
    /// Periodic scoring pass.
    ///
    /// Re-scores all records for staleness, emits `RecommendBurn` or
    /// `RecommendArchive` for content above thresholds. Emits `RepairNeeded`
    /// for under-replicated content. Emits `HealthAlert` if storage is
    /// above the alert threshold.
    pub fn tick(&mut self, now: f64) -> Vec<JainAction> {
        let mut actions = Vec::new();

        // Storage health check.
        if self.storage_capacity_bytes > 0 {
            let used_pct =
                self.total_storage_bytes as f64 / self.storage_capacity_bytes as f64;
            if used_pct >= self.config.storage_alert_percent {
                actions.push(JainAction::HealthAlert {
                    alert: HealthAlertKind::StorageNearFull {
                        used_percent_x100: (used_pct * 10000.0) as u32,
                    },
                });
            }
        }

        // Per-record scoring.
        for record in self.records.values() {
            // Under-replicated check.
            if record.replica_count < self.config.min_replica_count {
                actions.push(JainAction::RepairNeeded {
                    cid: record.cid,
                    current_replicas: record.replica_count,
                    desired: self.config.min_replica_count,
                });
            }

            // Staleness scoring.
            let score = crate::scoring::staleness_score(record, &self.config, now);

            if score.value() >= self.config.burn_threshold {
                actions.push(JainAction::RecommendBurn {
                    recommendation: CleanupRecommendation {
                        cid: record.cid,
                        reason: CleanupReason::Stale,
                        staleness: score,
                        space_recovered_bytes: record.size_bytes,
                        confidence: score.value(),
                    },
                });
            } else if score.value() >= self.config.archive_threshold {
                actions.push(JainAction::RecommendArchive {
                    recommendation: CleanupRecommendation {
                        cid: record.cid,
                        reason: CleanupReason::Stale,
                        staleness: score,
                        space_recovered_bytes: record.size_bytes,
                        confidence: score.value(),
                    },
                });
            }
        }

        actions
    }
```

**Step 4: Make `config` field accessible to tests**

The `tick_recommends_archive_for_moderately_stale_content` test references `engine.config.access_decay_half_life_secs`. Since `config` is private, either:
- Make it `pub(crate)`: change `config: JainConfig` to `pub(crate) config: JainConfig` in the struct definition.
- Or store the half-life value separately in the test.

Recommended: change to `pub(crate) config: JainConfig` since other engine methods need it too and tests should be able to inspect config.

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (50 previous + 6 new = 56).

**Step 6: Commit**

```bash
git add crates/harmony-jain/src/engine.rs
git commit -m "feat(jain): tick — staleness recommendations and health alerts"
```

---

### Task 12: JainEngine — reconcile

**Files:**
- Modify: `crates/harmony-jain/src/engine.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `engine.rs`:

```rust
    #[test]
    fn reconcile_adds_missing_records() {
        let mut engine = default_engine();
        let cid = make_cid(b"found-in-attic");
        let actions = engine.reconcile(&[SnapshotEntry {
            cid,
            size_bytes: 200,
            exists_on_disk: true,
        }]);
        // No immediate actions, but the record should now be tracked.
        assert_eq!(engine.record_count(), 1);
        assert_eq!(engine.health_report().total_bytes, 200);
    }

    #[test]
    fn reconcile_detects_missing_backing_data() {
        let mut engine = default_engine();
        let cid = make_cid(b"ghost");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        // Snapshot says the file doesn't exist on disk.
        let actions = engine.reconcile(&[SnapshotEntry {
            cid,
            size_bytes: 100,
            exists_on_disk: false,
        }]);
        assert!(
            actions.iter().any(|a| matches!(
                a,
                JainAction::RepairNeeded { .. }
            )),
            "expected RepairNeeded for missing backing data, got {:?}",
            actions
        );
    }

    #[test]
    fn reconcile_detects_orphaned_records() {
        let mut engine = default_engine();
        let cid = make_cid(b"orphan");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 100.0,
        });
        // Empty snapshot — content is gone from disk without a delete event.
        let actions = engine.reconcile(&[]);
        assert!(
            actions.iter().any(|a| matches!(
                a,
                JainAction::HealthAlert {
                    alert: HealthAlertKind::StaleReconciliation { .. }
                }
            )),
            "expected StaleReconciliation alert, got {:?}",
            actions
        );
        // Orphaned record should be removed.
        assert_eq!(engine.record_count(), 0);
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-jain`
Expected: FAIL — `reconcile` not defined.

**Step 3: Write the implementation**

Add to the `impl JainEngine` block:

```rust
    /// Periodic inventory reconciliation.
    ///
    /// Compares Jain's internal records against a snapshot of what actually
    /// exists on disk. Detects:
    /// - Content on disk not tracked by Jain → adds a record.
    /// - Content tracked by Jain but missing from disk → emits alert, removes record.
    /// - Content tracked by Jain with `exists_on_disk: false` → emits `RepairNeeded`.
    pub fn reconcile(&mut self, snapshot: &[SnapshotEntry]) -> Vec<JainAction> {
        use hashbrown::HashSet;
        let mut actions = Vec::new();
        let mut seen_cids: HashSet<ContentId> = HashSet::new();

        for entry in snapshot {
            seen_cids.insert(entry.cid);

            if !entry.exists_on_disk {
                // Tracked content whose backing data is missing — needs repair.
                if self.records.contains_key(&entry.cid) {
                    actions.push(JainAction::RepairNeeded {
                        cid: entry.cid,
                        current_replicas: 0,
                        desired: self.config.min_replica_count,
                    });
                }
                continue;
            }

            // Content on disk not tracked by Jain — add a default record.
            if !self.records.contains_key(&entry.cid) {
                self.total_storage_bytes += entry.size_bytes;
                self.records.insert(
                    entry.cid,
                    ContentRecord {
                        cid: entry.cid,
                        size_bytes: entry.size_bytes,
                        content_type: harmony_roxy::catalog::ContentCategory::Bundle,
                        origin: crate::types::ContentOrigin::CachedInTransit,
                        sensitivity: crate::types::Sensitivity::Public,
                        stored_at: 0.0,
                        last_accessed: 0.0,
                        access_count: 0,
                        replica_count: 1,
                        pinned: false,
                        licensed: false,
                    },
                );
            }
        }

        // Detect orphaned records: tracked by Jain but absent from snapshot.
        let orphaned: Vec<ContentId> = self
            .records
            .keys()
            .filter(|cid| !seen_cids.contains(cid))
            .copied()
            .collect();

        if !orphaned.is_empty() {
            actions.push(JainAction::HealthAlert {
                alert: HealthAlertKind::StaleReconciliation {
                    records_without_backing: orphaned.len() as u32,
                },
            });
            for cid in orphaned {
                if let Some(record) = self.records.remove(&cid) {
                    self.total_storage_bytes =
                        self.total_storage_bytes.saturating_sub(record.size_bytes);
                }
            }
        }

        actions
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-jain`
Expected: all tests pass (56 previous + 3 new = 59).

**Step 5: Commit**

```bash
git add crates/harmony-jain/src/engine.rs
git commit -m "feat(jain): reconcile — snapshot-based inventory reconciliation"
```

---

### Task 13: Final Verification & Clippy

**Files:**
- No new files. Verification pass only.

**Step 1: Run all tests**

Run: `cargo test -p harmony-jain`
Expected: 59 tests pass.

**Step 2: Run clippy**

Run: `cargo clippy -p harmony-jain -- -D warnings`
Expected: zero warnings. Fix any that appear.

**Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues. Fix any that appear.

**Step 4: Run full workspace test**

Run: `cargo test --workspace`
Expected: all tests pass (existing + 59 new).

**Step 5: Run full workspace clippy**

Run: `cargo clippy --workspace -- -D warnings`
Expected: zero warnings.

**Step 6: Commit any fixes from steps 2-5**

```bash
git add -A
git commit -m "style: fix clippy and fmt issues in harmony-jain"
```

(Only if there were fixes needed. Skip if clean.)

---

## Summary

| Task | Module | Tests Added | Running Total |
|---|---|---|---|
| 1 | Scaffold + error | 0 (compile check) | 0 |
| 2 | types (enums) | 5 | 5 |
| 3 | types (ContentRecord, StalenessScore) | 3 | 8 |
| 4 | config | 4 | 12 |
| 5 | scoring | 7 | 19 |
| 6 | actions | 4 | 23 |
| 7 | catalog | 6 | 29 |
| 8 | engine (scaffold) | 2 | 31 |
| 9 | engine (handle_event, evaluate_ingest) | 11 | 42 |
| 10 | engine (filter_result) | 8 | 50 |
| 11 | engine (tick) | 6 | 56 |
| 12 | engine (reconcile) | 3 | 59 |
| 13 | Verification | 0 | 59 |

13 tasks, 13 commits, ~59 tests, 7 source files.

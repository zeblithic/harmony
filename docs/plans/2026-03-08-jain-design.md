# Jain: Content Lifecycle Engine (Green Butler)

**Status:** Design approved. Foundation layer — sans-I/O types and engine.

**Color:** Green (Maintenance / Entropy Reduction)

**Metaphor:** Jain is the trusted household manager. Watches the front door (gatekeeper), screens what guests see (filter), and periodically inventories the attic (housekeeper). Never throws anything away without asking. The butler, janitor, and nurse rolled into one.

---

## Overview

`harmony-jain` is a `no_std`-compatible, sans-I/O content lifecycle engine. It tracks content on a node, scores staleness, screens incoming and outgoing content for context-appropriateness, and emits recommendations for the caller to execute. Jain never performs I/O or deletes content directly — it computes decisions and the node runtime acts on them.

### Three Roles

1. **Gatekeeper (ingest)** — All incoming content passes through Jain first. Jain decides whether to store it, whether Oluo should index it, and whether to reject it (duplicate, over budget).

2. **Filter (outbound)** — When search results come back from Oluo, Jain screens them against the current social context and user-defined filter rules. Intimate photos don't surface during a work presentation.

3. **Housekeeper (maintenance)** — Periodically scores all tracked content for staleness, recommends burning (deletion) or archiving (cold storage), detects under-replicated content, and emits health alerts.

---

## Core Types

### Content Model

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

/// Social context hint from Wylene.
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

/// Jain's internal model of a tracked piece of content.
#[derive(Debug, Clone)]
pub struct ContentRecord {
    pub cid: ContentId,
    pub size_bytes: u64,
    pub content_type: ContentCategory, // reused from harmony-roxy
    pub origin: ContentOrigin,
    pub sensitivity: Sensitivity,
    pub stored_at: f64,
    pub last_accessed: f64,
    pub access_count: u64,
    pub replica_count: u8,
    pub pinned: bool,
    pub licensed: bool,
}

/// Staleness score: 0.0 = fresh/vital, 1.0 = almost certainly unneeded.
pub struct StalenessScore(f64);
```

### Configuration

```rust
/// Tunable weights and thresholds for staleness scoring.
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
    pub self_created_weight: f64,
}

/// A single content filter rule.
pub struct FilterRule {
    /// Applies to content at or above this sensitivity level.
    pub min_sensitivity: Sensitivity,
    /// Blocks when social context is above this level.
    pub max_context: SocialContext,
    /// If true, ask user before surfacing (vs hard block).
    pub require_confirmation: bool,
}

/// Filter rule set — evaluated in order, first match wins.
pub struct FilterRuleSet {
    pub rules: Vec<FilterRule>,
    /// User can disable the whole filter system.
    pub enabled: bool,
}
```

**Default filter rules:**

| Rule | Min Sensitivity | Max Context | Confirmation? | Effect |
|---|---|---|---|---|
| 1 | `Intimate` | `Companion` | yes | Intimate content blocked above Companion; asks first even at Companion |
| 2 | `Confidential` | `Private` | no | Confidential only in Private, hard block otherwise |
| 3 | `Private` | `Social` | no | Private blocked in Professional context |

---

## Events & Actions

### Input: Content Events

```rust
/// Events Jain receives from the content subsystem.
pub enum ContentEvent {
    Stored {
        cid: ContentId,
        size_bytes: u64,
        content_type: ContentCategory,
        origin: ContentOrigin,
        sensitivity: Sensitivity,
        timestamp: f64,
    },
    Accessed { cid: ContentId, timestamp: f64 },
    Deleted { cid: ContentId },
    Pinned { cid: ContentId },
    Unpinned { cid: ContentId },
    LicenseGranted { cid: ContentId },
    LicenseExpired { cid: ContentId },
    ReplicaChanged { cid: ContentId, new_count: u8 },
}
```

### Input: Ingest Candidate

```rust
/// Content presented for ingest evaluation (gatekeeper).
pub struct IngestCandidate {
    pub cid: ContentId,
    pub size_bytes: u64,
    pub content_type: ContentCategory,
    pub origin: ContentOrigin,
    pub sensitivity: Sensitivity,
}
```

### Output: Decisions

```rust
/// Jain's gatekeeper decision on incoming content.
pub enum IngestDecision {
    /// Store and send to Oluo for indexing.
    IndexAndStore,
    /// Store but don't index (private, sensitive, or low-value).
    StoreOnly,
    /// Reject — don't store at all.
    Reject { reason: RejectReason },
}

pub enum RejectReason {
    StorageBudgetExceeded,
    DuplicateContent,
}

/// Jain's outbound filter decision on search results.
pub enum FilterDecision {
    /// Safe to show in current context.
    Allow,
    /// Blocked by filter rules — don't show.
    Block,
    /// Requires user confirmation before showing.
    Confirm,
}
```

### Output: Actions

```rust
/// Actions Jain emits for the caller to execute.
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

pub struct CleanupRecommendation {
    pub cid: ContentId,
    pub reason: CleanupReason,
    pub staleness: StalenessScore,
    pub space_recovered_bytes: u64,
    /// 0.0–1.0 confidence in this recommendation.
    pub confidence: f64,
}

pub enum CleanupReason {
    /// Not accessed in a long time.
    Stale,
    /// Might have a public equivalent (encrypted version of public content).
    EncryptedVersionOfPublic,
    /// Node is running low on space.
    OverStorageBudget,
}

pub enum QueryHint {
    /// Check if a public/unencrypted version exists.
    FindPublicEquivalent,
    /// Check if near-duplicates exist on this node.
    FindLocalDuplicates,
}

pub enum HealthAlertKind {
    StorageNearFull { used_percent: f64 },
    StaleReconciliation { records_without_backing: u32 },
    ReplicaDeficit { affected_records: u32 },
}
```

---

## Engine Interface

```rust
/// Sans-I/O content lifecycle engine.
///
/// Caller drives with events, snapshots, and periodic ticks.
/// Jain computes decisions; the caller executes them.
pub struct JainEngine {
    records: HashMap<ContentId, ContentRecord>,
    config: JainConfig,
    filter_rules: FilterRuleSet,
    total_storage_bytes: u64,
    storage_capacity_bytes: u64,
}

impl JainEngine {
    /// Create a new engine with the given configuration.
    pub fn new(config: JainConfig, filter_rules: FilterRuleSet,
               storage_capacity_bytes: u64) -> Self;

    /// Gatekeeper: evaluate incoming content before storing.
    pub fn evaluate_ingest(&self, candidate: &IngestCandidate) -> IngestDecision;

    /// Outbound filter: screen a search result for context appropriateness.
    pub fn filter_result(&self, cid: &ContentId, context: SocialContext) -> FilterDecision;

    /// Process a content lifecycle event. Returns immediate actions
    /// (e.g., duplicate detection on Stored).
    pub fn handle_event(&mut self, event: ContentEvent) -> Vec<JainAction>;

    /// Periodic inventory reconciliation. Catches drift between Jain's
    /// records and actual on-disk content.
    pub fn reconcile(&mut self, snapshot: &[SnapshotEntry]) -> Vec<JainAction>;

    /// Periodic scoring pass. Re-scores all records, emits recommendations
    /// for anything above thresholds, emits health alerts.
    pub fn tick(&mut self, now: f64) -> Vec<JainAction>;

    /// Current health summary.
    pub fn health_report(&self) -> HealthReport;

    /// Number of tracked records.
    pub fn record_count(&self) -> usize;
}

/// Entry in a snapshot for reconciliation.
pub struct SnapshotEntry {
    pub cid: ContentId,
    pub size_bytes: u64,
    pub exists_on_disk: bool,
}

/// Node health summary.
pub struct HealthReport {
    pub total_records: u32,
    pub total_bytes: u64,
    pub storage_used_percent: f64,
    pub under_replicated_count: u32,
    pub stale_count: u32,
    pub pinned_count: u32,
}
```

### Input Model

**Event-driven (real-time):** `handle_event()` fires on every content store/access/delete. Keeps the internal model fresh.

**Snapshot-driven (periodic):** `reconcile()` receives a full inventory scan. Catches anything events missed — node crashes, corruption, content that appeared without an event. The "inventory the attic" pass.

**Tick (periodic):** `tick()` re-scores staleness, checks thresholds, emits recommendations and health alerts. Caller decides frequency.

### Staleness Scoring

Pure function: `fn staleness_score(record: &ContentRecord, config: &JainConfig, now: f64) -> StalenessScore`

Signals and weights:

| Signal | Direction | Notes |
|---|---|---|
| Time since last access | Staler with time | Exponential decay with configurable half-life |
| Access frequency | Fresher with frequency | Lifetime access count normalized by age |
| Origin = SelfCreated | Fresher | Weighted by `self_created_weight` |
| Pinned | Override → 0.0 | User explicitly said keep |
| Licensed (Roxy) | Override → 0.0 | Paid for = definitely wanted |
| Replica count < min | Override → 0.0 | Unsafe to suggest burning |
| Size | Amplifies score | Large stale content = higher priority recommendation |

---

## Zenoh Key Expressions

```
jain/health/{node_hash}                  # HealthReport published periodically
jain/recommend/{user_hash}               # CleanupRecommendations for Wylene
jain/action/{user_hash}/{action_id}      # user-approved cleanup actions
jain/stats/{node_hash}                   # detailed metrics
```

Key expression builder functions with `validate_segment()` metacharacter rejection, same pattern as `harmony-roxy::catalog`.

---

## Crate Structure

```
crates/harmony-jain/
├── Cargo.toml
└── src/
    ├── lib.rs          # pub mod declarations, crate-level docs
    ├── types.rs        # ContentOrigin, Sensitivity, SocialContext, ContentRecord, StalenessScore
    ├── config.rs       # JainConfig, FilterRule, FilterRuleSet, default rules
    ├── engine.rs       # JainEngine — all five methods
    ├── scoring.rs      # staleness_score() pure function
    ├── catalog.rs      # Zenoh key expression builders
    └── error.rs        # JainError enum
```

### Dependencies

| Crate | Why |
|---|---|
| `harmony-content` | `ContentId`, `ContentFlags` |
| `harmony-roxy` | `ContentCategory` (reuse) |
| `serde` | Serialization |
| `postcard` | Binary serialization |
| `hashbrown` | `HashMap` (no_std) |
| `hex` | Key expression encoding |
| `thiserror` | Error derive |

---

## Deferred (out of scope for foundation)

- Trusted peer replication protocol ("butler" — ensuring copies live on friends' nodes)
- Cold storage offload protocol (actual archive mechanism)
- Oluo integration (similarity queries — Oluo crate doesn't exist yet)
- Wylene integration (presenting recommendations to user)
- Perceptual hashing / near-duplicate detection (ML-based)
- Content sensitivity classification via ML (foundation uses manual/upstream tags)
- Persistence (Jain's state is in-memory, rebuilt from events/snapshots)

---

## Cross-Color Interactions

| Interaction | How it works |
|---|---|
| **Green + Yellow (Oluo)** | Jain emits `QueryOluo` actions; Oluo checks similarity/public equivalents |
| **Green + Magenta (Wylene)** | Jain publishes recommendations; Wylene surfaces them; user approves via Wylene |
| **Green + Red (Roxy)** | Licensed content immune from cleanup; license expiry triggers re-scoring |
| **Green + Cyan** | Cyan prevents damage (replication); Jain repairs what got through (RepairNeeded) |
| **Green + Lyll** | Content addressing provides the hash-based dedup foundation |

---

## Design Philosophy

> We all have so much digital data we don't want to take the effort to go through — some of it we want to keep, but so much we actually don't. Jain is the janitor/nurse that keeps systems clean, healthy, and functional.

Jain exists because digital hoarding is a real problem — psychologically parallel to material hoarding. We accumulate infinite data because we're afraid of losing it, but a trusted system that can preserve what matters (public library dedup, peer replication) means we can finally let go of what doesn't. Jain helps with that, gently and with consent. Jain never deletes without asking.

**API naming note:** `Burn` is used instead of `Evict` — a deliberate, humane design choice. Controlled burns keep forests healthy; eviction carries trauma. The API reflects the values of the system.

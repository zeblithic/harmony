# StorageTier Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `StorageTier<B: BlobStore>` to harmony-content — a sans-I/O wrapper that integrates `ContentStore` with Zenoh queryable and subscriber patterns.

**Architecture:** Event/action state machine. `StorageTier` accepts `StorageTierEvent` variants and returns `Vec<StorageTierAction>`. Uses `harmony_zenoh::namespace::content` for canonical key expressions. Replaces the ad-hoc `zenoh_bridge.rs` functions.

**Tech Stack:** Rust, harmony-content crate, harmony-zenoh dependency (for namespace constants), hex crate (already a dep).

---

### Task 1: Add harmony-zenoh dependency to harmony-content

**Files:**
- Modify: `crates/harmony-content/Cargo.toml`

**Step 1: Add the dependency**

Add `harmony-zenoh.workspace = true` to the `[dependencies]` section of `crates/harmony-content/Cargo.toml`, after the `hex` line.

**Step 2: Verify it compiles**

Run: `cargo check -p harmony-content`
Expected: success (no code uses it yet, just confirming no circular deps)

**Step 3: Commit**

```bash
git add crates/harmony-content/Cargo.toml
git commit -m "chore(content): add harmony-zenoh dependency for StorageTier"
```

---

### Task 2: Create StorageBudget and StorageMetrics types

**Files:**
- Create: `crates/harmony-content/src/storage_tier.rs`
- Modify: `crates/harmony-content/src/lib.rs` (add `pub mod storage_tier;`)

**Step 1: Write failing test for StorageBudget defaults**

In `crates/harmony-content/src/storage_tier.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_default_values() {
        let budget = StorageBudget {
            cache_capacity: 1000,
            max_pinned_bytes: 500_000_000,
        };
        assert_eq!(budget.cache_capacity, 1000);
        assert_eq!(budget.max_pinned_bytes, 500_000_000);
    }

    #[test]
    fn metrics_start_at_zero() {
        let m = StorageMetrics::default();
        assert_eq!(m.queries_served, 0);
        assert_eq!(m.cache_hits, 0);
        assert_eq!(m.cache_misses, 0);
        assert_eq!(m.transit_admitted, 0);
        assert_eq!(m.transit_rejected, 0);
        assert_eq!(m.publishes_stored, 0);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content storage_tier`
Expected: FAIL — module doesn't exist yet

**Step 3: Write minimal implementation**

At the top of `crates/harmony-content/src/storage_tier.rs`:

```rust
//! StorageTier: sans-I/O wrapper integrating ContentStore with Zenoh patterns.

/// Configuration for storage capacity limits.
#[derive(Debug, Clone)]
pub struct StorageBudget {
    /// Maximum items in the W-TinyLFU cache.
    pub cache_capacity: usize,
    /// Maximum bytes reserved for pinned content.
    pub max_pinned_bytes: u64,
}

/// Observable metrics for the storage tier.
#[derive(Debug, Clone, Default)]
pub struct StorageMetrics {
    pub queries_served: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub transit_admitted: u64,
    pub transit_rejected: u64,
    pub publishes_stored: u64,
}
```

Also add `pub mod storage_tier;` to `crates/harmony-content/src/lib.rs`.

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content storage_tier`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add StorageBudget and StorageMetrics types"
```

---

### Task 3: Define StorageTierEvent and StorageTierAction enums

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing test for event/action construction**

```rust
#[test]
fn event_and_action_types_exist() {
    let _event = StorageTierEvent::ContentQuery {
        query_id: 1,
        cid: ContentId::for_blob(b"test").unwrap(),
    };
    let _action = StorageTierAction::SendReply {
        query_id: 1,
        payload: vec![1, 2, 3],
    };
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content event_and_action`
Expected: FAIL — types not defined

**Step 3: Write the enums**

```rust
use crate::cid::ContentId;

/// Inbound events for the storage tier to process.
#[derive(Debug, Clone)]
pub enum StorageTierEvent {
    /// Content query on harmony/content/{prefix}/**
    ContentQuery { query_id: u64, cid: ContentId },
    /// Content transiting through router (harmony/content/transit/**)
    TransitContent { cid: ContentId, data: Vec<u8> },
    /// Explicit publish request (harmony/content/publish/*)
    PublishContent { cid: ContentId, data: Vec<u8> },
    /// Stats query on harmony/content/stats/{node_addr}
    StatsQuery { query_id: u64 },
}

/// Outbound actions returned by the storage tier for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StorageTierAction {
    /// Reply to a content query with data.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Announce content availability on harmony/announce/{cid_hex}.
    AnnounceContent { key_expr: String, payload: Vec<u8> },
    /// Reply with cache metrics.
    SendStatsReply { query_id: u64, payload: Vec<u8> },
    /// Register shard queryables (returned at startup).
    DeclareQueryables { key_exprs: Vec<String> },
    /// Register subscriptions (returned at startup).
    DeclareSubscribers { key_exprs: Vec<String> },
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content event_and_action`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): define StorageTierEvent and StorageTierAction enums"
```

---

### Task 4: StorageTier struct and startup actions

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing test for startup**

```rust
use crate::blob::MemoryBlobStore;

#[test]
fn startup_declares_queryables_and_subscribers() {
    let budget = StorageBudget {
        cache_capacity: 100,
        max_pinned_bytes: 1_000_000,
    };
    let (tier, actions) = StorageTier::new(MemoryBlobStore::new(), budget);
    let _ = tier; // just verify it was created

    // Should have exactly 2 startup actions: DeclareQueryables + DeclareSubscribers
    assert_eq!(actions.len(), 2);

    // First: 16 shard patterns + 1 stats key = 17 queryables
    match &actions[0] {
        StorageTierAction::DeclareQueryables { key_exprs } => {
            assert_eq!(key_exprs.len(), 17);
            assert!(key_exprs[0].starts_with("harmony/content/0/"));
            assert!(key_exprs[16].starts_with("harmony/content/stats"));
        }
        other => panic!("expected DeclareQueryables, got {other:?}"),
    }

    // Second: transit + publish = 2 subscribers
    match &actions[1] {
        StorageTierAction::DeclareSubscribers { key_exprs } => {
            assert_eq!(key_exprs.len(), 2);
            assert!(key_exprs.contains(&"harmony/content/transit/**".to_string()));
            assert!(key_exprs.contains(&"harmony/content/publish/*".to_string()));
        }
        other => panic!("expected DeclareSubscribers, got {other:?}"),
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content startup_declares`
Expected: FAIL — `StorageTier` not defined

**Step 3: Write implementation**

```rust
use crate::blob::BlobStore;
use crate::cache::ContentStore;
use harmony_zenoh::namespace::content as ns;

pub struct StorageTier<B: BlobStore> {
    cache: ContentStore<B>,
    budget: StorageBudget,
    metrics: StorageMetrics,
}

impl<B: BlobStore> StorageTier<B> {
    /// Create a new StorageTier with startup actions.
    ///
    /// Returns the tier and the initial actions the caller must execute
    /// (declaring queryables and subscribers on the Zenoh session).
    pub fn new(store: B, budget: StorageBudget) -> (Self, Vec<StorageTierAction>) {
        let cache = ContentStore::new(store, budget.cache_capacity);

        let mut queryable_keys = ns::all_shard_patterns();
        queryable_keys.push(ns::STATS.to_string());

        let subscriber_keys = vec![
            ns::TRANSIT_SUB.to_string(),
            ns::PUBLISH_SUB.to_string(),
        ];

        let actions = vec![
            StorageTierAction::DeclareQueryables { key_exprs: queryable_keys },
            StorageTierAction::DeclareSubscribers { key_exprs: subscriber_keys },
        ];

        let tier = Self {
            cache,
            budget,
            metrics: StorageMetrics::default(),
        };

        (tier, actions)
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content startup_declares`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): StorageTier struct with startup actions"
```

---

### Task 5: Handle ContentQuery event

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn content_query_hit_returns_reply() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

    // Store some content first
    let data = b"cached blob";
    let cid = tier.cache.insert(data).unwrap();

    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 42, cid });

    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::SendReply { query_id, payload } => {
            assert_eq!(*query_id, 42);
            assert_eq!(payload.as_slice(), data.as_slice());
        }
        other => panic!("expected SendReply, got {other:?}"),
    }
    assert_eq!(tier.metrics().queries_served, 1);
    assert_eq!(tier.metrics().cache_hits, 1);
}

#[test]
fn content_query_miss_returns_nothing() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
    let cid = ContentId::for_blob(b"not stored").unwrap();

    let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 99, cid });
    assert!(actions.is_empty());
    assert_eq!(tier.metrics().queries_served, 1);
    assert_eq!(tier.metrics().cache_misses, 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content content_query`
Expected: FAIL — `handle` method not defined

**Step 3: Write implementation**

Add `handle` method and `metrics` accessor to `impl<B: BlobStore> StorageTier<B>`:

```rust
    /// Process an event and return actions for the caller to execute.
    pub fn handle(&mut self, event: StorageTierEvent) -> Vec<StorageTierAction> {
        match event {
            StorageTierEvent::ContentQuery { query_id, cid } => {
                self.handle_content_query(query_id, &cid)
            }
            _ => vec![], // other handlers added in subsequent tasks
        }
    }

    /// Read-only access to metrics.
    pub fn metrics(&self) -> &StorageMetrics {
        &self.metrics
    }

    fn handle_content_query(&mut self, query_id: u64, cid: &ContentId) -> Vec<StorageTierAction> {
        self.metrics.queries_served += 1;
        match self.cache.get_and_record(cid) {
            Some(data) => {
                self.metrics.cache_hits += 1;
                vec![StorageTierAction::SendReply { query_id, payload: data }]
            }
            None => {
                self.metrics.cache_misses += 1;
                vec![]
            }
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content content_query`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): handle ContentQuery in StorageTier"
```

---

### Task 6: Handle TransitContent event (opportunistic caching)

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn transit_content_admitted_produces_announcement() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
    let data = b"transiting blob";
    let cid = ContentId::for_blob(data).unwrap();

    let actions = tier.handle(StorageTierEvent::TransitContent {
        cid,
        data: data.to_vec(),
    });

    // Should produce an announce action
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::AnnounceContent { key_expr, .. } => {
            assert!(key_expr.starts_with("harmony/announce/"));
        }
        other => panic!("expected AnnounceContent, got {other:?}"),
    }
    assert_eq!(tier.metrics().transit_admitted, 1);

    // Content should now be queryable
    let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
    assert_eq!(query_actions.len(), 1); // cache hit
}

#[test]
fn transit_duplicate_still_counted() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
    let data = b"repeated transit";
    let cid = ContentId::for_blob(data).unwrap();

    tier.handle(StorageTierEvent::TransitContent { cid, data: data.to_vec() });
    tier.handle(StorageTierEvent::TransitContent { cid, data: data.to_vec() });

    // Both transits are admitted (dedup happens at store level)
    assert_eq!(tier.metrics().transit_admitted, 2);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content transit_content`
Expected: FAIL — TransitContent arm returns empty vec

**Step 3: Write implementation**

Add to the `handle` match and add private method:

```rust
    // In handle():
    StorageTierEvent::TransitContent { cid, data } => {
        self.handle_transit(cid, data)
    }

    fn handle_transit(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        self.cache.store(cid, data);
        self.metrics.transit_admitted += 1;
        let cid_hex = hex::encode(cid.to_bytes());
        let key_expr = harmony_zenoh::namespace::announce::key(&cid_hex);
        let payload = cid.payload_size().to_be_bytes().to_vec();
        vec![StorageTierAction::AnnounceContent { key_expr, payload }]
    }
```

Note: W-TinyLFU admission is handled internally by `ContentStore::store()` which calls `admit()`. The transit handler always reports "admitted" because the data is offered to the cache — the cache itself decides whether to keep it via frequency comparison. A future refinement could check if the CID was actually retained, but that's not needed for this PR.

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content transit_content`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): handle TransitContent with opportunistic caching"
```

---

### Task 7: Handle PublishContent event (always-admit)

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing test**

```rust
#[test]
fn publish_content_always_stored_and_announced() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
    let data = b"explicitly published blob";
    let cid = ContentId::for_blob(data).unwrap();

    let actions = tier.handle(StorageTierEvent::PublishContent {
        cid,
        data: data.to_vec(),
    });

    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::AnnounceContent { key_expr, payload } => {
            assert!(key_expr.starts_with("harmony/announce/"));
            let announced_size = u32::from_be_bytes(payload[..4].try_into().unwrap());
            assert_eq!(announced_size, cid.payload_size());
        }
        other => panic!("expected AnnounceContent, got {other:?}"),
    }
    assert_eq!(tier.metrics().publishes_stored, 1);

    // Content should be queryable
    let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
    assert_eq!(query_actions.len(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content publish_content`
Expected: FAIL

**Step 3: Write implementation**

```rust
    // In handle():
    StorageTierEvent::PublishContent { cid, data } => {
        self.handle_publish(cid, data)
    }

    fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        self.cache.store(cid, data);
        self.metrics.publishes_stored += 1;
        let cid_hex = hex::encode(cid.to_bytes());
        let key_expr = harmony_zenoh::namespace::announce::key(&cid_hex);
        let payload = cid.payload_size().to_be_bytes().to_vec();
        vec![StorageTierAction::AnnounceContent { key_expr, payload }]
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content publish_content`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): handle PublishContent with always-admit"
```

---

### Task 8: Handle StatsQuery event

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Write failing test**

```rust
#[test]
fn stats_query_returns_serialized_metrics() {
    let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
    let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

    // Generate some activity
    let data = b"stats test blob";
    let cid = ContentId::for_blob(data).unwrap();
    tier.handle(StorageTierEvent::PublishContent { cid, data: data.to_vec() });
    tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });

    let actions = tier.handle(StorageTierEvent::StatsQuery { query_id: 77 });
    assert_eq!(actions.len(), 1);
    match &actions[0] {
        StorageTierAction::SendStatsReply { query_id, payload } => {
            assert_eq!(*query_id, 77);
            // Payload should be 6 u64s = 48 bytes
            assert_eq!(payload.len(), 48);
            // First u64: queries_served = 1
            let queries = u64::from_be_bytes(payload[0..8].try_into().unwrap());
            assert_eq!(queries, 1);
        }
        other => panic!("expected SendStatsReply, got {other:?}"),
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content stats_query`
Expected: FAIL

**Step 3: Write implementation**

Add serialization to `StorageMetrics` and handler:

```rust
impl StorageMetrics {
    /// Serialize metrics as 6 big-endian u64s (48 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(48);
        buf.extend_from_slice(&self.queries_served.to_be_bytes());
        buf.extend_from_slice(&self.cache_hits.to_be_bytes());
        buf.extend_from_slice(&self.cache_misses.to_be_bytes());
        buf.extend_from_slice(&self.transit_admitted.to_be_bytes());
        buf.extend_from_slice(&self.transit_rejected.to_be_bytes());
        buf.extend_from_slice(&self.publishes_stored.to_be_bytes());
        buf
    }
}
```

```rust
    // In handle():
    StorageTierEvent::StatsQuery { query_id } => {
        vec![StorageTierAction::SendStatsReply {
            query_id,
            payload: self.metrics.to_bytes(),
        }]
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content stats_query`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): handle StatsQuery with serialized metrics"
```

---

### Task 9: Replace zenoh_bridge.rs with StorageTier

**Files:**
- Modify: `crates/harmony-content/src/zenoh_bridge.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Verify existing zenoh_bridge tests still pass**

Run: `cargo test -p harmony-content zenoh_bridge`
Expected: 4 tests PASS (baseline)

**Step 2: Replace zenoh_bridge.rs**

Replace the contents of `zenoh_bridge.rs` with a thin re-export module that points users to `StorageTier`:

```rust
//! Legacy Zenoh content bridge — superseded by [`crate::storage_tier::StorageTier`].
//!
//! This module re-exports the canonical key expression helpers from
//! `harmony_zenoh::namespace` for callers that haven't migrated to StorageTier.

pub use harmony_zenoh::namespace::content::all_shard_patterns as content_queryable_key_exprs;
pub use harmony_zenoh::namespace::announce::key as cid_to_announce_key_expr;

use crate::cid::ContentId;

/// Convert a CID to its content key expression.
///
/// Delegates to [`harmony_zenoh::namespace::content::fetch_key`].
pub fn cid_to_key_expr(cid: &ContentId) -> String {
    let hex_cid = hex::encode(cid.to_bytes());
    harmony_zenoh::namespace::content::fetch_key(&hex_cid)
}
```

Remove `handle_content_query` and `handle_content_stored` — these are now `StorageTier::handle()`.

Remove `ContentBridgeAction` — replaced by `StorageTierAction`.

Remove `HEX_PREFIXES` — now canonical in `harmony_zenoh::namespace::content`.

**Step 3: Update tests in zenoh_bridge.rs**

Keep the structural tests that verify key expression format, update them to use the new functions:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::ContentId;

    #[test]
    fn cid_key_expr_matches_shard_structure() {
        let cid = ContentId::for_blob(b"shard routing test").unwrap();
        let key_expr = cid_to_key_expr(&cid);
        assert!(key_expr.starts_with("harmony/content/"));

        let shards = content_queryable_key_exprs();
        assert_eq!(shards.len(), 16);
        assert!(shards[0].starts_with("harmony/content/0/"));
    }

    #[test]
    fn announce_key_format() {
        let hex_cid = "abc123";
        let key = cid_to_announce_key_expr(hex_cid);
        assert_eq!(key, "harmony/announce/abc123");
    }
}
```

**Step 4: Run all tests**

Run: `cargo test -p harmony-content`
Expected: ALL tests pass (storage_tier + zenoh_bridge)

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-content`
Expected: no warnings

**Step 6: Commit**

```bash
git add crates/harmony-content/src/zenoh_bridge.rs crates/harmony-content/src/lib.rs
git commit -m "refactor(content): replace zenoh_bridge internals with namespace imports"
```

---

### Task 10: Final validation and cleanup

**Files:**
- All harmony-content source files

**Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: ALL tests pass (including other crates that may import from harmony-content)

**Step 2: Run clippy on workspace**

Run: `cargo clippy --workspace`
Expected: zero warnings

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: no formatting issues

**Step 4: Commit any fixups**

If clippy or fmt required changes, commit them.

**Step 5: Verify test count increased**

Run: `cargo test -p harmony-content -- --list 2>&1 | tail -1`
Expected: test count should be higher than before (was ~25 tests, should be ~35+)

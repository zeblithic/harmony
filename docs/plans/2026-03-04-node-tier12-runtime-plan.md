# Node Tier 1+2 Runtime Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform harmony-node from identity-only CLI into a full node runtime wiring Tier 1 (Reticulum Node + Zenoh QueryableRouter) and Tier 2 (StorageTier) together with a priority event loop.

**Architecture:** `NodeRuntime<B: BlobStore>` is a sans-I/O state machine that wraps the Reticulum `Node`, a `QueryableRouter`, and a `StorageTier<B>`. Events are pushed into internal priority queues. Each `tick()` drains ALL Tier 1 (router) events before processing ONE Tier 2 (storage) event, enforcing the design invariant that the router is never starved. The CLI adds a `run` subcommand that constructs the runtime.

**Tech Stack:** Rust, harmony-reticulum, harmony-zenoh, harmony-content, clap

**Design doc:** `docs/plans/2026-03-03-node-trinity-design.md` (sections 3, 4, 7)

---

## Background for implementer

### Key existing types you'll use

**harmony-reticulum (`crates/harmony-reticulum/src/node.rs`):**
- `Node::new() -> Self` — leaf-mode Reticulum router
- `Node::handle_event(event: NodeEvent) -> Vec<NodeAction>` — sans-I/O event processing
- `NodeEvent::InboundPacket { interface_name, raw, now }` / `NodeEvent::TimerTick { now }`
- `NodeAction::SendOnInterface { interface_name: Arc<str>, raw }` — outbound packet
- `NodeAction::DeliverLocally { destination_hash, packet, interface_name }` — local delivery

**harmony-zenoh (`crates/harmony-zenoh/src/queryable.rs`):**
- `QueryableRouter::new() -> Self` — sans-I/O query matcher (no Session dependency)
- `QueryableRouter::declare(key_expr) -> Result<(QueryableId, Vec<QueryableAction>), ZenohError>`
- `QueryableRouter::handle_event(QueryableEvent) -> Result<Vec<QueryableAction>, ZenohError>`
- `QueryableAction::DeliverQuery { queryable_id, query_id, key_expr, payload }`

**harmony-content (`crates/harmony-content/src/storage_tier.rs`):**
- `StorageTier::new(store, budget) -> (Self, Vec<StorageTierAction>)` — returns startup actions
- `StorageTier::handle(event: StorageTierEvent) -> Vec<StorageTierAction>`
- `StorageTierAction::DeclareQueryables { key_exprs }` — 16 shard patterns + stats
- `StorageTierAction::DeclareSubscribers { key_exprs }` — transit + publish subscriptions
- `ContentId::from_bytes([u8; 32]) -> Self` — reconstruct from raw bytes
- `MemoryBlobStore::new()` — in-memory backend (`crates/harmony-content/src/blob.rs`)

**harmony-zenoh namespace (`crates/harmony-zenoh/src/namespace.rs`):**
- `content::PREFIX` = `"harmony/content"`, `content::STATS` = `"harmony/content/stats"`
- `content::TRANSIT` = `"harmony/content/transit"`, `content::PUBLISH` = `"harmony/content/publish"`

---

### Task 1: Dependencies and type definitions

**Files:**
- Modify: `crates/harmony-node/Cargo.toml`
- Create: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod runtime;`)

**Step 1: Write the failing test**

Add to the bottom of `crates/harmony-node/src/runtime.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_event_variants_exist() {
        let _e1 = RuntimeEvent::InboundPacket {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            now: 1000,
        };
        let _e2 = RuntimeEvent::TimerTick { now: 1000 };
        let _e3 = RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/abc".into(),
            payload: vec![],
        };
        let _e4 = RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/content/transit/abc".into(),
            payload: vec![1, 2, 3],
        };
    }

    #[test]
    fn runtime_action_variants_exist() {
        let _a1 = RuntimeAction::SendOnInterface {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
        };
        let _a2 = RuntimeAction::SendReply {
            query_id: 1,
            payload: vec![1, 2, 3],
        };
        let _a3 = RuntimeAction::Publish {
            key_expr: "harmony/announce/abc".into(),
            payload: vec![],
        };
        let _a4 = RuntimeAction::DeclareQueryable {
            key_expr: "harmony/content/a/**".into(),
        };
        let _a5 = RuntimeAction::Subscribe {
            key_expr: "harmony/content/transit/**".into(),
        };
    }

    #[test]
    fn node_config_defaults() {
        let config = NodeConfig::default();
        assert_eq!(config.storage_budget.cache_capacity, 1024);
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node --lib runtime::tests 2>&1 | head -20`
Expected: FAIL — module `runtime` not found

**Step 3: Update Cargo.toml dependencies**

Replace the `[dependencies]` section in `crates/harmony-node/Cargo.toml`:

```toml
[dependencies]
harmony-content.workspace = true
harmony-identity = { path = "../harmony-identity" }
harmony-reticulum.workspace = true
harmony-zenoh.workspace = true
clap = { version = "4", features = ["derive"] }
hex.workspace = true
rand.workspace = true
```

Note: `harmony-reticulum` and `harmony-content` are already workspace dependencies (check root `Cargo.toml`). If `harmony-reticulum` is not in `[workspace.dependencies]`, add it: `harmony-reticulum = { path = "crates/harmony-reticulum" }`.

**Step 4: Write the type definitions**

Create `crates/harmony-node/src/runtime.rs`:

```rust
//! Node runtime: priority event loop wiring Tier 1 (Router) + Tier 2 (Storage).

use std::collections::VecDeque;
use std::sync::Arc;

use harmony_content::blob::BlobStore;
use harmony_content::cid::ContentId;
use harmony_content::storage_tier::{
    StorageBudget, StorageMetrics, StorageTier, StorageTierAction, StorageTierEvent,
};
use harmony_reticulum::node::{Node, NodeAction, NodeEvent};
use harmony_zenoh::queryable::{
    QueryableAction, QueryableEvent, QueryableId, QueryableRouter, QueryId,
};
use harmony_zenoh::namespace::content as content_ns;

/// Configuration for a Harmony node runtime.
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Storage tier capacity limits.
    pub storage_budget: StorageBudget,
}

impl Default for NodeConfig {
    fn default() -> Self {
        Self {
            storage_budget: StorageBudget {
                cache_capacity: 1024,
                max_pinned_bytes: 100_000_000,
            },
        }
    }
}

/// Inbound events fed into the node runtime.
#[derive(Debug)]
pub enum RuntimeEvent {
    /// Tier 1: Reticulum packet received on a network interface.
    InboundPacket {
        interface_name: String,
        raw: Vec<u8>,
        now: u64,
    },
    /// Tier 1: Periodic timer tick for path expiry, announce scheduling.
    TimerTick { now: u64 },
    /// Tier 2: Zenoh query received (content fetch or stats request).
    QueryReceived {
        query_id: u64,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Tier 2: Zenoh subscription message (transit or publish).
    SubscriptionMessage {
        key_expr: String,
        payload: Vec<u8>,
    },
}

/// Outbound actions returned by the runtime for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeAction {
    /// Tier 1: Send raw packet on a network interface.
    SendOnInterface { interface_name: Arc<str>, raw: Vec<u8> },
    /// Tier 2: Reply to a content or stats query.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Tier 2: Publish a message (e.g., content availability announcement).
    Publish { key_expr: String, payload: Vec<u8> },
    /// Setup: Declare a queryable key expression.
    DeclareQueryable { key_expr: String },
    /// Setup: Subscribe to a key expression.
    Subscribe { key_expr: String },
}
```

**Step 5: Add module declaration**

In `crates/harmony-node/src/main.rs`, add at the top (before `use clap`):

```rust
mod runtime;
```

**Step 6: Run tests to verify they pass**

Run: `cargo test -p harmony-node --lib runtime::tests`
Expected: 3 tests PASS

**Step 7: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/runtime.rs crates/harmony-node/src/main.rs
git commit -m "feat(node): add RuntimeEvent, RuntimeAction, and NodeConfig types

Foundation types for the Tier 1+2 priority event loop. RuntimeEvent
separates router events (InboundPacket, TimerTick) from storage events
(QueryReceived, SubscriptionMessage). RuntimeAction unifies outbound
actions across both tiers.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 2: NodeRuntime struct and constructor

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `runtime.rs`:

```rust
    use harmony_content::blob::MemoryBlobStore;

    fn make_runtime() -> (NodeRuntime<MemoryBlobStore>, Vec<RuntimeAction>) {
        let config = NodeConfig::default();
        NodeRuntime::new(config, MemoryBlobStore::new())
    }

    #[test]
    fn constructor_returns_startup_actions() {
        let (_, actions) = make_runtime();

        let queryable_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::DeclareQueryable { .. }))
            .count();
        let subscribe_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::Subscribe { .. }))
            .count();

        // 16 shard queryables + 1 stats queryable = 17
        assert_eq!(queryable_count, 17);
        // transit + publish subscriptions = 2
        assert_eq!(subscribe_count, 2);
    }

    #[test]
    fn metrics_start_at_zero() {
        let (rt, _) = make_runtime();
        let m = rt.metrics();
        assert_eq!(m.queries_served, 0);
        assert_eq!(m.cache_hits, 0);
    }

    #[test]
    fn queues_start_empty() {
        let (rt, _) = make_runtime();
        assert_eq!(rt.router_queue_len(), 0);
        assert_eq!(rt.storage_queue_len(), 0);
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-node --lib runtime::tests::constructor_returns_startup_actions 2>&1 | head -20`
Expected: FAIL — `NodeRuntime` not found

**Step 3: Write the NodeRuntime struct and constructor**

Add above the `#[cfg(test)]` block in `runtime.rs`:

```rust
use std::collections::HashSet;

/// Sans-I/O node runtime wiring Tier 1 (Router) and Tier 2 (Storage).
///
/// Events are pushed via [`push_event`](Self::push_event) into internal
/// priority queues. Each [`tick`](Self::tick) drains ALL Tier 1 events
/// before processing ONE Tier 2 event — the router is never starved.
pub struct NodeRuntime<B: BlobStore> {
    // Tier 1: Reticulum packet router
    router: Node,
    // Tier 1/2: Zenoh query dispatch
    queryable_router: QueryableRouter,
    // Tier 2: Content storage
    storage: StorageTier<B>,
    // Internal priority queues
    router_queue: VecDeque<NodeEvent>,
    storage_queue: VecDeque<StorageTierEvent>,
    // Queryable IDs belonging to the storage tier
    storage_queryable_ids: HashSet<QueryableId>,
}

impl<B: BlobStore> NodeRuntime<B> {
    /// Construct a new node runtime, returning startup actions the caller
    /// must execute (queryable declarations, subscriptions).
    pub fn new(config: NodeConfig, store: B) -> (Self, Vec<RuntimeAction>) {
        let router = Node::new();
        let mut queryable_router = QueryableRouter::new();

        let (storage, storage_startup) = StorageTier::new(store, config.storage_budget);

        let mut actions = Vec::new();
        let mut storage_queryable_ids = HashSet::new();

        // Process storage startup actions: register queryables and subscriptions
        for action in storage_startup {
            match action {
                StorageTierAction::DeclareQueryables { key_exprs } => {
                    for key_expr in &key_exprs {
                        match queryable_router.declare(key_expr) {
                            Ok((qid, _qactions)) => {
                                storage_queryable_ids.insert(qid);
                                actions.push(RuntimeAction::DeclareQueryable {
                                    key_expr: key_expr.clone(),
                                });
                            }
                            Err(_) => {} // invalid key expr — skip
                        }
                    }
                }
                StorageTierAction::DeclareSubscribers { key_exprs } => {
                    for key_expr in key_exprs {
                        actions.push(RuntimeAction::Subscribe { key_expr });
                    }
                }
                _ => {} // other startup actions not expected
            }
        }

        let rt = Self {
            router,
            queryable_router,
            storage,
            router_queue: VecDeque::new(),
            storage_queue: VecDeque::new(),
            storage_queryable_ids,
        };

        (rt, actions)
    }

    /// Read-only access to storage metrics.
    pub fn metrics(&self) -> &StorageMetrics {
        self.storage.metrics()
    }

    /// Number of pending Tier 1 (router) events.
    pub fn router_queue_len(&self) -> usize {
        self.router_queue.len()
    }

    /// Number of pending Tier 2 (storage) events.
    pub fn storage_queue_len(&self) -> usize {
        self.storage_queue.len()
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node --lib runtime::tests`
Expected: 6 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): NodeRuntime struct with Tier 1+2 constructor

NodeRuntime<B> wraps Reticulum Node, QueryableRouter, and StorageTier.
Constructor processes storage startup actions: registers 17 queryable
key expressions (16 shards + stats) with the QueryableRouter and
returns DeclareQueryable + Subscribe actions for the caller.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 3: push_event() and tick() with priority queues

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn push_event_classifies_router_events() {
        let (mut rt, _) = make_runtime();
        rt.push_event(RuntimeEvent::InboundPacket {
            interface_name: "lo".into(),
            raw: vec![0u8; 20],
            now: 1000,
        });
        rt.push_event(RuntimeEvent::TimerTick { now: 1001 });
        assert_eq!(rt.router_queue_len(), 2);
        assert_eq!(rt.storage_queue_len(), 0);
    }

    #[test]
    fn push_event_routes_query_to_storage() {
        let (mut rt, _) = make_runtime();
        // Push a query matching a declared content shard (harmony/content/a/**)
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/abc123".into(),
            payload: vec![],
        });
        // Query routing happens during push_event: QueryableRouter matches
        // and the result is queued as a StorageTierEvent
        assert_eq!(rt.storage_queue_len(), 1);
    }

    #[test]
    fn tick_drains_all_router_events_before_storage() {
        let (mut rt, _) = make_runtime();

        // Push 3 router events and 2 storage events
        for i in 0..3 {
            rt.push_event(RuntimeEvent::TimerTick { now: 1000 + i });
        }
        // Use stats queries as storage events (they always produce a reply)
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 10,
            key_expr: "harmony/content/stats/node1".into(),
            payload: vec![],
        });
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 11,
            key_expr: "harmony/content/stats/node1".into(),
            payload: vec![],
        });

        assert_eq!(rt.router_queue_len(), 3);
        assert_eq!(rt.storage_queue_len(), 2);

        // First tick: drains all 3 router events + 1 storage event
        let actions = rt.tick();
        assert_eq!(rt.router_queue_len(), 0);
        assert_eq!(rt.storage_queue_len(), 1); // one storage event remains

        // Should have exactly one SendReply (from the one stats query processed)
        let reply_count = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::SendReply { .. }))
            .count();
        assert_eq!(reply_count, 1);

        // Second tick: processes the remaining storage event
        let actions2 = rt.tick();
        assert_eq!(rt.storage_queue_len(), 0);
        let reply_count2 = actions2
            .iter()
            .filter(|a| matches!(a, RuntimeAction::SendReply { .. }))
            .count();
        assert_eq!(reply_count2, 1);
    }

    #[test]
    fn tick_with_empty_queues_returns_nothing() {
        let (mut rt, _) = make_runtime();
        let actions = rt.tick();
        assert!(actions.is_empty());
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node --lib runtime::tests::push_event_classifies 2>&1 | head -20`
Expected: FAIL — no method named `push_event`

**Step 3: Implement push_event() and tick()**

Add these methods to `impl<B: BlobStore> NodeRuntime<B>`:

```rust
    /// Push an event into the runtime's internal priority queues.
    ///
    /// Router events (InboundPacket, TimerTick) go to the Tier 1 queue.
    /// Query and subscription events are routed to the Tier 2 queue
    /// (queries go through the QueryableRouter for key expression matching).
    pub fn push_event(&mut self, event: RuntimeEvent) {
        match event {
            RuntimeEvent::InboundPacket {
                interface_name,
                raw,
                now,
            } => {
                self.router_queue.push_back(NodeEvent::InboundPacket {
                    interface_name,
                    raw,
                    now,
                });
            }
            RuntimeEvent::TimerTick { now } => {
                self.router_queue.push_back(NodeEvent::TimerTick { now });
            }
            RuntimeEvent::QueryReceived {
                query_id,
                key_expr,
                payload,
            } => {
                self.route_query(query_id, key_expr, payload);
            }
            RuntimeEvent::SubscriptionMessage { key_expr, payload } => {
                self.route_subscription(key_expr, payload);
            }
        }
    }

    /// Run one iteration of the priority event loop.
    ///
    /// 1. Drain ALL Tier 1 (router) events — router is never starved.
    /// 2. Process ONE Tier 2 (storage) event.
    pub fn tick(&mut self) -> Vec<RuntimeAction> {
        let mut actions = Vec::new();

        // Tier 1: drain ALL router events (highest priority)
        while let Some(event) = self.router_queue.pop_front() {
            let node_actions = self.router.handle_event(event);
            self.dispatch_router_actions(node_actions, &mut actions);
        }

        // Tier 2: process ONE storage event (middle priority)
        if let Some(event) = self.storage_queue.pop_front() {
            let storage_actions = self.storage.handle(event);
            self.dispatch_storage_actions(storage_actions, &mut actions);
        }

        actions
    }

    fn dispatch_router_actions(
        &mut self,
        node_actions: Vec<NodeAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in node_actions {
            match action {
                NodeAction::SendOnInterface {
                    interface_name,
                    raw,
                } => {
                    out.push(RuntimeAction::SendOnInterface {
                        interface_name,
                        raw,
                    });
                }
                // Other router actions are diagnostics — drop for now.
                // Future: emit RuntimeAction::RouterDiagnostic variants.
                _ => {}
            }
        }
    }

    fn dispatch_storage_actions(
        &mut self,
        storage_actions: Vec<StorageTierAction>,
        out: &mut Vec<RuntimeAction>,
    ) {
        for action in storage_actions {
            match action {
                StorageTierAction::SendReply { query_id, payload } => {
                    out.push(RuntimeAction::SendReply { query_id, payload });
                }
                StorageTierAction::AnnounceContent { key_expr, payload } => {
                    out.push(RuntimeAction::Publish { key_expr, payload });
                }
                StorageTierAction::SendStatsReply { query_id, payload } => {
                    out.push(RuntimeAction::SendReply { query_id, payload });
                }
                // DeclareQueryables/DeclareSubscribers only at startup
                _ => {}
            }
        }
    }

    /// Route a Zenoh query through the QueryableRouter to the correct tier.
    fn route_query(&mut self, query_id: u64, key_expr: String, payload: Vec<u8>) {
        let event = QueryableEvent::QueryReceived {
            query_id,
            key_expr: key_expr.clone(),
            payload,
        };
        let Ok(actions) = self.queryable_router.handle_event(event) else {
            return;
        };
        for action in actions {
            if let QueryableAction::DeliverQuery {
                queryable_id,
                query_id,
                key_expr,
                ..
            } = action
            {
                if self.storage_queryable_ids.contains(&queryable_id) {
                    if let Some(event) = self.parse_storage_query(query_id, &key_expr) {
                        self.storage_queue.push_back(event);
                    }
                }
            }
        }
    }

    /// Route a subscription message to the correct tier based on key prefix.
    fn route_subscription(&mut self, key_expr: String, payload: Vec<u8>) {
        if let Some(event) = self.parse_subscription_event(&key_expr, payload) {
            self.storage_queue.push_back(event);
        }
    }

    /// Parse a query key expression into a StorageTierEvent.
    fn parse_storage_query(&self, query_id: u64, key_expr: &str) -> Option<StorageTierEvent> {
        // Stats query: harmony/content/stats/{node_addr}
        if key_expr.starts_with(content_ns::STATS) {
            return Some(StorageTierEvent::StatsQuery { query_id });
        }

        // Content query: harmony/content/{prefix}/{cid_hex}
        // Strip "harmony/content/" prefix, skip shard char + "/", get CID hex
        let after_prefix = key_expr.strip_prefix(content_ns::PREFIX)?.strip_prefix('/')?;
        // after_prefix = "{shard}/{cid_hex}" — skip shard char and slash
        let cid_hex = after_prefix.get(2..)?;
        let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
        let cid = ContentId::from_bytes(cid_bytes);
        Some(StorageTierEvent::ContentQuery { query_id, cid })
    }

    /// Parse a subscription key expression + payload into a StorageTierEvent.
    fn parse_subscription_event(
        &self,
        key_expr: &str,
        payload: Vec<u8>,
    ) -> Option<StorageTierEvent> {
        if let Some(cid_hex) = key_expr.strip_prefix(content_ns::TRANSIT).and_then(|s| s.strip_prefix('/')) {
            let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
            let cid = ContentId::from_bytes(cid_bytes);
            return Some(StorageTierEvent::TransitContent { cid, data: payload });
        }
        if let Some(cid_hex) = key_expr.strip_prefix(content_ns::PUBLISH).and_then(|s| s.strip_prefix('/')) {
            let cid_bytes: [u8; 32] = hex::decode(cid_hex).ok()?.try_into().ok()?;
            let cid = ContentId::from_bytes(cid_bytes);
            return Some(StorageTierEvent::PublishContent { cid, data: payload });
        }
        None
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-node --lib runtime::tests`
Expected: 10 tests PASS

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-node -- -D warnings`
Expected: zero warnings

**Step 6: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): priority event loop with push_event() and tick()

Implements the core Tier 1+2 priority invariant: tick() drains ALL
router events before processing ONE storage event. Query routing
goes through QueryableRouter for proper key expression matching.
Subscription messages are routed by key prefix (transit/publish).

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 4: End-to-end storage round-trip tests

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs` (tests only)

These tests verify the full pipeline: publish content → query it back, transit content admission, subscription routing.

**Step 1: Write the tests**

Add to the `tests` module:

```rust
    use harmony_content::cid::ContentId;

    /// Helper: build a valid CID hex string for test data.
    fn cid_hex_for(data: &[u8]) -> (ContentId, String) {
        let cid = ContentId::for_blob(data).unwrap();
        let hex = hex::encode(cid.to_bytes());
        (cid, hex)
    }

    #[test]
    fn publish_then_query_round_trip() {
        let (mut rt, _) = make_runtime();
        let data = b"round trip test data";
        let (cid, cid_hex) = cid_hex_for(data);

        // Publish via subscription message
        let publish_key = format!("harmony/content/publish/{cid_hex}");
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: publish_key,
            payload: data.to_vec(),
        });
        let publish_actions = rt.tick();

        // Should get an AnnounceContent → Publish action
        assert!(
            publish_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::Publish { .. })),
            "expected Publish action from publish event"
        );

        // Query the same CID
        let first_char = &cid_hex[..1];
        let query_key = format!("harmony/content/{first_char}/{cid_hex}");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 42,
            key_expr: query_key,
            payload: vec![],
        });
        let query_actions = rt.tick();

        // Should get a SendReply with the original data
        let reply = query_actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 42, .. }));
        assert!(reply.is_some(), "expected SendReply for query_id 42");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            assert_eq!(payload.as_slice(), data);
        }
    }

    #[test]
    fn transit_content_admitted_and_queryable() {
        let (mut rt, _) = make_runtime();
        let data = b"transit data";
        let (cid, cid_hex) = cid_hex_for(data);

        // Transit via subscription
        let transit_key = format!("harmony/content/transit/{cid_hex}");
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: transit_key,
            payload: data.to_vec(),
        });
        let transit_actions = rt.tick();
        assert!(
            transit_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::Publish { .. })),
            "admitted transit should produce announcement"
        );

        // Query it back
        let first_char = &cid_hex[..1];
        let query_key = format!("harmony/content/{first_char}/{cid_hex}");
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 99,
            key_expr: query_key,
            payload: vec![],
        });
        let query_actions = rt.tick();
        assert!(
            query_actions
                .iter()
                .any(|a| matches!(a, RuntimeAction::SendReply { query_id: 99, .. })),
            "cached transit content should be queryable"
        );
    }

    #[test]
    fn stats_query_returns_metrics() {
        let (mut rt, _) = make_runtime();

        // Do a content query first (to bump metrics)
        let (_, cid_hex) = cid_hex_for(b"anything");
        let first_char = &cid_hex[..1];
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: format!("harmony/content/{first_char}/{cid_hex}"),
            payload: vec![],
        });
        rt.tick();

        // Now query stats
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 50,
            key_expr: "harmony/content/stats/mynode".into(),
            payload: vec![],
        });
        let actions = rt.tick();

        let reply = actions
            .iter()
            .find(|a| matches!(a, RuntimeAction::SendReply { query_id: 50, .. }));
        assert!(reply.is_some(), "stats query should produce reply");
        if let Some(RuntimeAction::SendReply { payload, .. }) = reply {
            // 7 metrics × 8 bytes = 56 bytes
            assert_eq!(payload.len(), 56);
            // First metric is queries_served, should be 1
            let queries = u64::from_be_bytes(payload[0..8].try_into().unwrap());
            assert_eq!(queries, 1);
        }
    }

    #[test]
    fn malformed_query_key_silently_dropped() {
        let (mut rt, _) = make_runtime();
        // Invalid CID hex (not 64 chars)
        rt.push_event(RuntimeEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/a/not_valid_hex".into(),
            payload: vec![],
        });
        // Should not panic, event is silently dropped
        assert_eq!(rt.storage_queue_len(), 0);
    }

    #[test]
    fn malformed_subscription_key_silently_dropped() {
        let (mut rt, _) = make_runtime();
        rt.push_event(RuntimeEvent::SubscriptionMessage {
            key_expr: "harmony/content/transit/bad_hex".into(),
            payload: vec![1, 2, 3],
        });
        assert_eq!(rt.storage_queue_len(), 0);
    }
```

**Step 2: Run tests**

Run: `cargo test -p harmony-node --lib runtime::tests`
Expected: 15 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "test(node): end-to-end storage round-trip and error handling tests

Verifies publish→query, transit→query, stats query, and malformed
input handling through the full NodeRuntime pipeline.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

### Task 5: CLI `run` subcommand

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

**Step 1: Write the failing test**

The CLI test is a simple integration check. Add a test at the bottom of `main.rs` (inside `#[cfg(test)] mod tests`):

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cli_parses_run_command() {
        let cli = Cli::try_parse_from(["harmony", "run"]).unwrap();
        assert!(matches!(cli.command, Commands::Run { .. }));
    }

    #[test]
    fn cli_parses_run_with_cache_capacity() {
        let cli = Cli::try_parse_from(["harmony", "run", "--cache-capacity", "2048"]).unwrap();
        if let Commands::Run { cache_capacity, .. } = cli.command {
            assert_eq!(cache_capacity, 2048);
        } else {
            panic!("expected Run command");
        }
    }
}
```

**Step 2: Run to verify it fails**

Run: `cargo test -p harmony-node --lib tests::cli_parses_run 2>&1 | head -20`
Expected: FAIL — no `Run` variant

**Step 3: Add the `run` subcommand**

Update `main.rs`. Add `Run` to the `Commands` enum:

```rust
#[derive(Subcommand)]
enum Commands {
    /// Identity management commands
    Identity {
        #[command(subcommand)]
        action: IdentityAction,
    },
    /// Start the Harmony node runtime
    Run {
        /// W-TinyLFU cache capacity (number of items)
        #[arg(long, default_value_t = 1024)]
        cache_capacity: usize,
    },
}
```

Add the `Run` handler in the `run()` function's match:

```rust
        Commands::Run { cache_capacity } => {
            use crate::runtime::{NodeConfig, NodeRuntime, RuntimeAction};
            use harmony_content::blob::MemoryBlobStore;
            use harmony_content::storage_tier::StorageBudget;

            let config = NodeConfig {
                storage_budget: StorageBudget {
                    cache_capacity,
                    max_pinned_bytes: 100_000_000,
                },
            };
            let (rt, startup_actions) = NodeRuntime::new(config, MemoryBlobStore::new());

            println!("Harmony node runtime initialized");
            println!("  Cache capacity: {cache_capacity} items");
            println!("  Router queue:   {} pending", rt.router_queue_len());
            println!("  Storage queue:  {} pending", rt.storage_queue_len());
            println!("\nStartup actions:");
            for action in &startup_actions {
                match action {
                    RuntimeAction::DeclareQueryable { key_expr } => {
                        println!("  queryable: {key_expr}");
                    }
                    RuntimeAction::Subscribe { key_expr } => {
                        println!("  subscribe: {key_expr}");
                    }
                    _ => {}
                }
            }
            println!(
                "\n{} queryables, {} subscriptions declared",
                startup_actions
                    .iter()
                    .filter(|a| matches!(a, RuntimeAction::DeclareQueryable { .. }))
                    .count(),
                startup_actions
                    .iter()
                    .filter(|a| matches!(a, RuntimeAction::Subscribe { .. }))
                    .count(),
            );
            println!("\nNode ready. (Event loop requires async runtime — not yet wired.)");
            Ok(())
        }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests PASS (runtime tests + CLI tests)

**Step 5: Run the CLI to verify output**

Run: `cargo run -p harmony-node -- run`
Expected output:
```
Harmony node runtime initialized
  Cache capacity: 1024 items
  Router queue:   0 pending
  Storage queue:  0 pending

Startup actions:
  queryable: harmony/content/0/**
  queryable: harmony/content/1/**
  ...
  queryable: harmony/content/f/**
  queryable: harmony/content/stats
  subscribe: harmony/content/transit/**
  subscribe: harmony/content/publish/*

17 queryables, 2 subscriptions declared

Node ready. (Event loop requires async runtime — not yet wired.)
```

**Step 6: Run full workspace checks**

Run: `cargo test --workspace && cargo clippy --workspace -- -D warnings`
Expected: all tests PASS, zero clippy warnings

**Step 7: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(node): add 'run' subcommand to CLI

The 'run' command constructs a NodeRuntime with MemoryBlobStore,
prints startup declarations (17 queryables + 2 subscriptions),
and reports readiness. The async event loop will be wired when
tokio integration is added.

Co-Authored-By: Claude Opus 4.6 <noreply@anthropic.com>"
```

---

## Summary

| Task | What it builds | Tests |
|------|---------------|-------|
| 1 | `RuntimeEvent`, `RuntimeAction`, `NodeConfig` types | 3 type existence tests |
| 2 | `NodeRuntime<B>` struct + `new()` constructor | 3 constructor/startup tests |
| 3 | `push_event()` + `tick()` priority loop + query/subscription routing | 4 priority + routing tests |
| 4 | End-to-end round-trip tests | 5 integration tests |
| 5 | CLI `run` subcommand | 2 CLI parsing tests |

Total: ~17 tests, 1 new file (`runtime.rs`), 2 modified files (`main.rs`, `Cargo.toml`).

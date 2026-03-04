//! Node runtime: priority event loop wiring Tier 1 (Router) + Tier 2 (Storage).

// Types defined here are consumed by later tasks in this binary crate.
#![allow(dead_code)]

use std::collections::{HashSet, VecDeque};
use std::sync::Arc;

use harmony_content::blob::BlobStore;
use harmony_content::storage_tier::{
    StorageBudget, StorageMetrics, StorageTier, StorageTierAction, StorageTierEvent,
};
use harmony_reticulum::node::{Node, NodeEvent};
use harmony_zenoh::queryable::{QueryableId, QueryableRouter};

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
                        if let Ok((qid, _qactions)) = queryable_router.declare(key_expr) {
                            storage_queryable_ids.insert(qid);
                            actions.push(RuntimeAction::DeclareQueryable {
                                key_expr: key_expr.clone(),
                            });
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
}

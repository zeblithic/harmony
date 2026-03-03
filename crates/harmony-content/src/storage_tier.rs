//! StorageTier: sans-I/O wrapper integrating ContentStore with Zenoh patterns.

use crate::blob::BlobStore;
use crate::cache::ContentStore;
use crate::cid::ContentId;
use harmony_zenoh::namespace::{announce as announce_ns, content as ns};

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

/// Sans-I/O storage tier integrating [`ContentStore`] with Zenoh key patterns.
///
/// On construction, returns startup actions (queryable and subscriber
/// declarations) that the caller must execute. Subsequent calls to
/// [`handle`](Self::handle) process inbound events and return outbound
/// actions.
pub struct StorageTier<B: BlobStore> {
    cache: ContentStore<B>,
    budget: StorageBudget,
    metrics: StorageMetrics,
}

impl<B: BlobStore> StorageTier<B> {
    /// Create a new StorageTier with startup actions.
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

    /// Read-only access to metrics.
    pub fn metrics(&self) -> &StorageMetrics {
        &self.metrics
    }

    /// Mutable access to the underlying content store (crate-internal).
    pub(crate) fn cache_mut(&mut self) -> &mut ContentStore<B> {
        &mut self.cache
    }

    /// Process an event and return actions for the caller to execute.
    pub fn handle(&mut self, event: StorageTierEvent) -> Vec<StorageTierAction> {
        match event {
            StorageTierEvent::ContentQuery { query_id, cid } => {
                self.handle_content_query(query_id, &cid)
            }
            StorageTierEvent::TransitContent { cid, data } => {
                self.handle_transit(cid, data)
            }
            StorageTierEvent::PublishContent { cid, data } => {
                self.handle_publish(cid, data)
            }
            StorageTierEvent::StatsQuery { query_id } => {
                self.handle_stats_query(query_id)
            }
        }
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

    fn handle_transit(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        self.cache.store(cid, data);
        self.metrics.transit_admitted += 1;
        let cid_hex = hex::encode(cid.to_bytes());
        let key_expr = announce_ns::key(&cid_hex);
        let payload = cid.payload_size().to_be_bytes().to_vec();
        vec![StorageTierAction::AnnounceContent { key_expr, payload }]
    }

    fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        self.cache.store(cid, data);
        self.metrics.publishes_stored += 1;
        let cid_hex = hex::encode(cid.to_bytes());
        let key_expr = announce_ns::key(&cid_hex);
        let payload = cid.payload_size().to_be_bytes().to_vec();
        vec![StorageTierAction::AnnounceContent { key_expr, payload }]
    }

    fn handle_stats_query(&mut self, _query_id: u64) -> Vec<StorageTierAction> {
        vec![]
    }
}

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

    use crate::blob::MemoryBlobStore;

    #[test]
    fn content_query_hit_returns_reply() {
        let budget = StorageBudget { cache_capacity: 100, max_pinned_bytes: 1_000_000 };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        let data = b"cached blob";
        let cid = tier.cache_mut().insert(data).unwrap();

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
        assert_eq!(query_actions.len(), 1);
    }

    #[test]
    fn startup_declares_queryables_and_subscribers() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (tier, actions) = StorageTier::new(MemoryBlobStore::new(), budget);
        let _ = tier;

        assert_eq!(actions.len(), 2);

        match &actions[0] {
            StorageTierAction::DeclareQueryables { key_exprs } => {
                assert_eq!(key_exprs.len(), 17); // 16 shards + 1 stats
                assert!(key_exprs[0].starts_with("harmony/content/0/"));
                assert!(key_exprs[16].starts_with("harmony/content/stats"));
            }
            other => panic!("expected DeclareQueryables, got {other:?}"),
        }

        match &actions[1] {
            StorageTierAction::DeclareSubscribers { key_exprs } => {
                assert_eq!(key_exprs.len(), 2);
                assert!(key_exprs.contains(&"harmony/content/transit/**".to_string()));
                assert!(key_exprs.contains(&"harmony/content/publish/*".to_string()));
            }
            other => panic!("expected DeclareSubscribers, got {other:?}"),
        }
    }
}

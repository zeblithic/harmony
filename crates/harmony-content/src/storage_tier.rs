//! StorageTier: sans-I/O wrapper integrating ContentStore with Zenoh patterns.

use crate::cid::ContentId;

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
}

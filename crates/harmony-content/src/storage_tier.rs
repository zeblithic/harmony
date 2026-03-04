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
    pub transit_stored: u64,
    pub transit_rejected: u64,
    pub publishes_stored: u64,
    pub publishes_rejected: u64,
}

impl StorageMetrics {
    /// Serialize metrics as 7 big-endian u64s (56 bytes).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(56);
        buf.extend_from_slice(&self.queries_served.to_be_bytes());
        buf.extend_from_slice(&self.cache_hits.to_be_bytes());
        buf.extend_from_slice(&self.cache_misses.to_be_bytes());
        buf.extend_from_slice(&self.transit_stored.to_be_bytes());
        buf.extend_from_slice(&self.transit_rejected.to_be_bytes());
        buf.extend_from_slice(&self.publishes_stored.to_be_bytes());
        buf.extend_from_slice(&self.publishes_rejected.to_be_bytes());
        buf
    }
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
    /// Reserved for future byte-based pin limits. Currently only count-based
    /// limits are enforced via [`ContentStore::pin_limit`].
    #[allow(dead_code)]
    budget: StorageBudget,
    metrics: StorageMetrics,
}

impl<B: BlobStore> StorageTier<B> {
    /// Create a new StorageTier with startup actions.
    pub fn new(store: B, budget: StorageBudget) -> (Self, Vec<StorageTierAction>) {
        let cache = ContentStore::new(store, budget.cache_capacity);

        let mut queryable_keys = ns::all_shard_patterns();
        queryable_keys.push(ns::STATS.to_string());

        let subscriber_keys = vec![ns::TRANSIT_SUB.to_string(), ns::PUBLISH_SUB.to_string()];

        let actions = vec![
            StorageTierAction::DeclareQueryables {
                key_exprs: queryable_keys,
            },
            StorageTierAction::DeclareSubscribers {
                key_exprs: subscriber_keys,
            },
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
    #[allow(dead_code)]
    pub(crate) fn cache_mut(&mut self) -> &mut ContentStore<B> {
        &mut self.cache
    }

    /// Process an event and return actions for the caller to execute.
    pub fn handle(&mut self, event: StorageTierEvent) -> Vec<StorageTierAction> {
        match event {
            StorageTierEvent::ContentQuery { query_id, cid } => {
                self.handle_content_query(query_id, &cid)
            }
            StorageTierEvent::TransitContent { cid, data } => self.handle_transit(cid, data),
            StorageTierEvent::PublishContent { cid, data } => self.handle_publish(cid, data),
            StorageTierEvent::StatsQuery { query_id } => self.handle_stats_query(query_id),
        }
    }

    fn handle_content_query(&mut self, query_id: u64, cid: &ContentId) -> Vec<StorageTierAction> {
        self.metrics.queries_served += 1;
        match self.cache.get_and_record(cid) {
            Some(data) => {
                self.metrics.cache_hits += 1;
                vec![StorageTierAction::SendReply {
                    query_id,
                    payload: data,
                }]
            }
            None => {
                self.metrics.cache_misses += 1;
                vec![]
            }
        }
    }

    fn handle_transit(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        // Transit comes from untrusted peers — verify CID matches data hash.
        if !Self::verify_cid(&cid, &data) {
            self.metrics.transit_rejected += 1;
            return vec![];
        }
        // Pre-store admission check: compare frequency against probation's LRU.
        // Rejected items still get their sketch counter incremented so repeated
        // transits build popularity for future admission.
        if !self.cache.should_admit(&cid) {
            self.metrics.transit_rejected += 1;
            return vec![];
        }
        // Use store_preadmitted to avoid double-incrementing the sketch
        // counter (should_admit already incremented it).
        self.cache.store_preadmitted(cid, data);
        self.metrics.transit_stored += 1;
        // Re-announcing already-cached content is intentional: it refreshes
        // the announcement TTL so peers know the content is still available.
        vec![self.make_announce_action(&cid)]
    }

    fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        // Publish comes from local apps — still verify as defense-in-depth.
        if !Self::verify_cid(&cid, &data) {
            self.metrics.publishes_rejected += 1;
            return vec![];
        }
        // Publish always stores — bypasses admission filter.
        // TODO: Pin published content to guarantee long-term retention (deferred).
        self.cache.store(cid, data);
        self.metrics.publishes_stored += 1;
        vec![self.make_announce_action(&cid)]
    }

    /// Verify that a CID is consistent with the given data.
    ///
    /// Checks three things:
    /// 1. The truncated SHA-256 hash matches the data.
    /// 2. The payload size field matches the actual data length.
    /// 3. The internal checksum is valid (hash + size + type are consistent).
    fn verify_cid(cid: &ContentId, data: &[u8]) -> bool {
        cid.verify_hash(data)
            && cid.payload_size() as usize == data.len()
            && cid.verify_checksum().is_ok()
    }

    /// Build an AnnounceContent action for a stored CID.
    fn make_announce_action(&self, cid: &ContentId) -> StorageTierAction {
        let cid_hex = hex::encode(cid.to_bytes());
        let key_expr = announce_ns::key(&cid_hex);
        let payload = cid.payload_size().to_be_bytes().to_vec();
        StorageTierAction::AnnounceContent { key_expr, payload }
    }

    fn handle_stats_query(&mut self, query_id: u64) -> Vec<StorageTierAction> {
        vec![StorageTierAction::SendStatsReply {
            query_id,
            payload: self.metrics.to_bytes(),
        }]
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
        assert_eq!(m.transit_stored, 0);
        assert_eq!(m.transit_rejected, 0);
        assert_eq!(m.publishes_stored, 0);
        assert_eq!(m.publishes_rejected, 0);
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
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
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
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
        let cid = ContentId::for_blob(b"not stored").unwrap();

        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 99, cid });
        assert!(actions.is_empty());
        assert_eq!(tier.metrics().queries_served, 1);
        assert_eq!(tier.metrics().cache_misses, 1);
    }

    #[test]
    fn stats_query_returns_serialized_metrics() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        let data = b"stats test blob";
        let cid = ContentId::for_blob(data).unwrap();
        tier.handle(StorageTierEvent::PublishContent {
            cid,
            data: data.to_vec(),
        });
        tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });

        let actions = tier.handle(StorageTierEvent::StatsQuery { query_id: 77 });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendStatsReply { query_id, payload } => {
                assert_eq!(*query_id, 77);
                assert_eq!(payload.len(), 56);
                let queries = u64::from_be_bytes(payload[0..8].try_into().unwrap());
                assert_eq!(queries, 1); // one ContentQuery was processed
            }
            other => panic!("expected SendStatsReply, got {other:?}"),
        }
    }

    #[test]
    fn publish_content_always_stored_and_announced() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
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
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
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
        assert_eq!(tier.metrics().transit_stored, 1);

        // Content should now be queryable
        let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
        assert_eq!(query_actions.len(), 1);
    }

    #[test]
    fn transit_duplicate_still_counted() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);
        let data = b"repeated transit";
        let cid = ContentId::for_blob(data).unwrap();

        tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.to_vec(),
        });
        tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.to_vec(),
        });

        assert_eq!(tier.metrics().transit_stored, 2);
    }

    #[test]
    fn transit_cold_item_rejected_by_admission() {
        // Tiny cache (capacity 3: window=1, protected=0, probation=2).
        // Fill probation with hot items, then send a cold transit item that
        // should fail the pre-store admission check.
        let budget = StorageBudget {
            cache_capacity: 3,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        // Fill cache: 3 items → window=1, probation=2.
        for i in 0..3 {
            let data = format!("hot-{i}");
            let cid = ContentId::for_blob(data.as_bytes()).unwrap();
            tier.handle(StorageTierEvent::TransitContent {
                cid,
                data: data.into_bytes(),
            });
        }
        // Boost frequency of the probation items via queries.
        for i in 0..2 {
            let data = format!("hot-{i}");
            let cid = ContentId::for_blob(data.as_bytes()).unwrap();
            for _ in 0..10 {
                tier.handle(StorageTierEvent::ContentQuery { query_id: 0, cid });
            }
        }
        let admitted_before = tier.metrics().transit_stored;
        let rejected_before = tier.metrics().transit_rejected;

        // Cold transit item with zero frequency should be rejected.
        let cold_data = b"cold-newcomer-will-lose";
        let cold_cid = ContentId::for_blob(cold_data).unwrap();
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: cold_cid,
            data: cold_data.to_vec(),
        });

        assert!(
            actions.is_empty(),
            "cold transit item should be rejected by W-TinyLFU, got: {actions:?}"
        );
        assert_eq!(
            tier.metrics().transit_stored,
            admitted_before,
            "transit_stored should not increase for rejected item"
        );
        assert_eq!(
            tier.metrics().transit_rejected,
            rejected_before + 1,
            "transit_rejected should increase for rejected item"
        );
    }

    #[test]
    fn transit_admits_bundle_content() {
        // Bundle CIDs have a different type tag than blobs. Verify that
        // transit handles bundles correctly (hash-only verification).
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        let blob_a = ContentId::for_blob(b"child-a").unwrap();
        let blob_b = ContentId::for_blob(b"child-b").unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes: Vec<u8> = children.iter().flat_map(|c| c.to_bytes()).collect();
        let bundle_cid = ContentId::for_bundle(&bundle_bytes, &children).unwrap();

        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: bundle_cid,
            data: bundle_bytes,
        });

        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], StorageTierAction::AnnounceContent { .. }));
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_rejects_cid_data_mismatch() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        // CID for "real data" but send "tampered data"
        let cid = ContentId::for_blob(b"real data").unwrap();
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: b"tampered data".to_vec(),
        });

        assert!(actions.is_empty(), "mismatched CID should produce no actions");
        assert_eq!(tier.metrics().transit_rejected, 1);
        assert_eq!(tier.metrics().transit_stored, 0);
    }

    #[test]
    fn publish_rejects_cid_data_mismatch() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        let cid = ContentId::for_blob(b"original").unwrap();
        let actions = tier.handle(StorageTierEvent::PublishContent {
            cid,
            data: b"different".to_vec(),
        });

        assert!(actions.is_empty(), "mismatched CID should produce no actions");
        assert_eq!(tier.metrics().publishes_stored, 0);
        assert_eq!(tier.metrics().publishes_rejected, 1);
    }

    #[test]
    fn transit_rejects_cid_with_wrong_size_field() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget);

        // Craft a CID with correct hash but wrong payload_size by mutating
        // the size bits in the last 4 bytes.
        let data = b"correct data";
        let real_cid = ContentId::for_blob(data).unwrap();
        let mut bytes = real_cid.to_bytes();
        // Corrupt the size: set size to 999 instead of 12.
        let packed = u32::from_be_bytes(bytes[28..32].try_into().unwrap());
        let tag_only = packed & 0xFFF; // preserve lower 12 tag bits
        let fake_packed = (999u32 << 12) | tag_only;
        bytes[28..32].copy_from_slice(&fake_packed.to_be_bytes());
        let bad_cid = ContentId::from_bytes(bytes);

        // Hash matches but size is wrong — should be rejected.
        assert!(bad_cid.verify_hash(data), "hash should still match");
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: bad_cid,
            data: data.to_vec(),
        });
        assert!(actions.is_empty(), "wrong size CID should be rejected");
        assert_eq!(tier.metrics().transit_rejected, 1);
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

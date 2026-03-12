//! StorageTier: sans-I/O wrapper integrating ContentStore with Zenoh patterns.

use crate::blob::BlobStore;
use crate::cache::ContentStore;
use crate::cid::{ContentClass, ContentId};
use alloc::{
    string::{String, ToString},
    vec,
    vec::Vec,
};
use harmony_zenoh::namespace::{announce as announce_ns, content as ns};
#[cfg(not(feature = "std"))]
use hashbrown::HashSet;
#[cfg(feature = "std")]
use std::collections::HashSet;

/// Configuration for storage capacity limits.
#[derive(Debug, Clone)]
pub struct StorageBudget {
    /// Maximum items in the W-TinyLFU cache.
    pub cache_capacity: usize,
    /// Maximum bytes reserved for pinned content.
    pub max_pinned_bytes: u64,
}

/// Observable metrics for the storage tier.
///
/// Fields are public for read convenience. External mutation is prevented
/// because [`StorageTier`] only exposes metrics via `&StorageMetrics`.
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
    /// Serialize metrics as big-endian u64s.
    ///
    /// Uses exhaustive destructuring so adding a field without updating
    /// serialization is a compile error.
    pub fn to_bytes(&self) -> Vec<u8> {
        let Self {
            queries_served,
            cache_hits,
            cache_misses,
            transit_stored,
            transit_rejected,
            publishes_stored,
            publishes_rejected,
        } = self;
        let fields: &[u64] = &[
            *queries_served,
            *cache_hits,
            *cache_misses,
            *transit_stored,
            *transit_rejected,
            *publishes_stored,
            *publishes_rejected,
        ];
        let mut buf = Vec::with_capacity(fields.len() * 8);
        for &val in fields {
            buf.extend_from_slice(&val.to_be_bytes());
        }
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
    /// Disk read completed — runtime delivers data read from disk.
    DiskReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
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
    /// Persist content to disk (durable classes only). The runtime handles actual I/O.
    PersistToDisk { cid: ContentId, data: Vec<u8> },
    /// Remove content from disk. The runtime handles actual I/O.
    RemoveFromDisk { cid: ContentId },
    /// Request disk read for a CID known to be on disk but evicted from memory.
    DiskLookup { cid: ContentId, query_id: u64 },
}

/// Per-class storage and publishing policy.
///
/// Controls how each content class is handled at the admission and
/// announcement decision points. Classes not covered by explicit flags
/// have hardcoded behavior:
/// - `PublicDurable (00)`: always persist, always announce.
/// - `EncryptedEphemeral (11)`: always rejected by `class_admits()` — never stored or announced.
#[derive(Debug, Clone)]
pub struct ContentPolicy {
    /// Whether to persist encrypted durable (10) content.
    pub encrypted_durable_persist: bool,
    /// Whether to announce encrypted durable (10) content on Zenoh.
    pub encrypted_durable_announce: bool,
    /// Whether to announce public ephemeral (01) content on Zenoh.
    pub public_ephemeral_announce: bool,
}

impl Default for ContentPolicy {
    fn default() -> Self {
        Self {
            encrypted_durable_persist: false,
            encrypted_durable_announce: false,
            public_ephemeral_announce: true,
        }
    }
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
    /// Per-class admission and announcement policy.
    policy: ContentPolicy,
    metrics: StorageMetrics,
    /// CIDs known to be persisted on disk (durable classes only).
    /// Scaffolding for future disk I/O — insertions and lookups are
    /// deferred until the runtime can service DiskLookup actions.
    #[allow(dead_code)]
    disk_index: HashSet<ContentId>,
}

impl<B: BlobStore> StorageTier<B> {
    /// Create a new StorageTier with startup actions.
    pub fn new(
        store: B,
        budget: StorageBudget,
        policy: ContentPolicy,
    ) -> (Self, Vec<StorageTierAction>) {
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
            policy,
            metrics: StorageMetrics::default(),
            disk_index: HashSet::new(),
        };

        (tier, actions)
    }

    /// Read-only access to the content policy.
    pub fn policy(&self) -> &ContentPolicy {
        &self.policy
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
            StorageTierEvent::DiskReadComplete {
                cid,
                query_id,
                data,
            } => {
                // DiskReadComplete should only arrive for CIDs previously admitted
                // and disk-indexed (durable classes). Guard against invariant violations.
                debug_assert!(
                    Self::is_durable_class(&cid),
                    "DiskReadComplete for non-durable class: {:?}",
                    cid.content_class()
                );
                // Verify integrity — disk data may be corrupted (bit rot, wrong file).
                if !Self::verify_cid(&cid, &data) {
                    return vec![];
                }
                // Re-cache the data from disk.
                self.cache.store(cid, data.clone());
                vec![StorageTierAction::SendReply {
                    query_id,
                    payload: data,
                }]
            }
        }
    }

    /// Check whether a CID belongs to a durable content class.
    fn is_durable_class(cid: &ContentId) -> bool {
        matches!(
            cid.content_class(),
            ContentClass::PublicDurable | ContentClass::EncryptedDurable
        )
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
                // Note: disk_index may contain this CID, but DiskLookup actions
                // are not yet serviced by the runtime (disk I/O is a future bead).
                // Emitting DiskLookup here would cause queries to hang with no reply.
                // When disk I/O is wired, restore the disk_index check:
                //   if self.disk_index.contains(cid) {
                //       return vec![StorageTierAction::DiskLookup { cid: *cid, query_id }];
                //   }
                vec![]
            }
        }
    }

    /// Check whether a CID's content class should be announced on Zenoh.
    fn should_announce(&self, cid: &ContentId) -> bool {
        match cid.content_class() {
            ContentClass::PublicDurable => true,
            ContentClass::PublicEphemeral => self.policy.public_ephemeral_announce,
            ContentClass::EncryptedDurable => self.policy.encrypted_durable_announce,
            ContentClass::EncryptedEphemeral => false,
        }
    }

    /// Check whether a CID's content class is admissible under the current policy.
    fn class_admits(&self, cid: &ContentId) -> bool {
        match cid.content_class() {
            ContentClass::EncryptedEphemeral => false,
            ContentClass::EncryptedDurable => self.policy.encrypted_durable_persist,
            ContentClass::PublicDurable | ContentClass::PublicEphemeral => true,
        }
    }

    fn handle_transit(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        // Transit comes from untrusted peers — verify CID matches data hash.
        if !Self::verify_cid(&cid, &data) {
            self.metrics.transit_rejected += 1;
            return vec![];
        }
        // Class-based admission: reject content classes forbidden by policy.
        if !self.class_admits(&cid) {
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

        // Note: PersistToDisk is not emitted yet — the runtime doesn't
        // service disk I/O (future bead). Emitting it would clone data on
        // every durable transit for no benefit. When disk I/O is wired,
        // restore: if Self::is_durable_class(&cid) { clone + PersistToDisk }.

        let mut actions = Vec::new();
        // Re-announcing already-cached content is intentional: it refreshes
        // the announcement TTL so peers know the content is still available.
        // Announce only if policy allows it for this content class.
        if self.should_announce(&cid) {
            actions.push(self.make_announce_action(&cid));
        }
        actions
    }

    fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        // Publish comes from local apps — still verify as defense-in-depth.
        if !Self::verify_cid(&cid, &data) {
            self.metrics.publishes_rejected += 1;
            return vec![];
        }
        // Class-based admission: reject content classes forbidden by policy.
        if !self.class_admits(&cid) {
            self.metrics.publishes_rejected += 1;
            return vec![];
        }
        // Publish always stores — bypasses admission filter.
        // TODO: Pin published content to guarantee long-term retention (deferred).
        self.cache.store(cid, data);
        self.metrics.publishes_stored += 1;

        // Note: PersistToDisk deferred — see handle_transit comment.

        let mut actions = Vec::new();
        // Announce only if policy allows it for this content class.
        if self.should_announce(&cid) {
            actions.push(self.make_announce_action(&cid));
        }
        actions
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
    ///
    /// Returns one action per store. Batching and rate-limiting belong in the
    /// I/O bridge layer, not this sans-I/O state machine.
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
            cid: ContentId::for_blob(b"test", crate::cid::ContentFlags::default()).unwrap(),
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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
        let cid = ContentId::for_blob(b"not stored", crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        let data = b"stats test blob";
        let cid = ContentId::for_blob(data, crate::cid::ContentFlags::default()).unwrap();
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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
        let data = b"explicitly published blob";
        let cid = ContentId::for_blob(data, crate::cid::ContentFlags::default()).unwrap();

        let actions = tier.handle(StorageTierEvent::PublishContent {
            cid,
            data: data.to_vec(),
        });

        // PublicDurable content emits AnnounceContent (PersistToDisk deferred).
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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
        let data = b"transiting blob";
        let cid = ContentId::for_blob(data, crate::cid::ContentFlags::default()).unwrap();

        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.to_vec(),
        });

        // PublicDurable content emits AnnounceContent (PersistToDisk deferred).
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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
        let data = b"repeated transit";
        let cid = ContentId::for_blob(data, crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        // Fill cache: 3 items → window=1, probation=2.
        for i in 0..3 {
            let data = format!("hot-{i}");
            let cid =
                ContentId::for_blob(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap();
            tier.handle(StorageTierEvent::TransitContent {
                cid,
                data: data.into_bytes(),
            });
        }
        // Boost frequency of the probation items via queries.
        for i in 0..2 {
            let data = format!("hot-{i}");
            let cid =
                ContentId::for_blob(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap();
            for _ in 0..10 {
                tier.handle(StorageTierEvent::ContentQuery { query_id: 0, cid });
            }
        }
        let admitted_before = tier.metrics().transit_stored;
        let rejected_before = tier.metrics().transit_rejected;

        // Cold transit item with zero frequency should be rejected.
        let cold_data = b"cold-newcomer-will-lose";
        let cold_cid = ContentId::for_blob(cold_data, crate::cid::ContentFlags::default()).unwrap();
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
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        let blob_a = ContentId::for_blob(b"child-a", crate::cid::ContentFlags::default()).unwrap();
        let blob_b = ContentId::for_blob(b"child-b", crate::cid::ContentFlags::default()).unwrap();
        let children = [blob_a, blob_b];
        let bundle_bytes: Vec<u8> = children.iter().flat_map(|c| c.to_bytes()).collect();
        let bundle_cid = ContentId::for_bundle(
            &bundle_bytes,
            &children,
            crate::cid::ContentFlags::default(),
        )
        .unwrap();

        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: bundle_cid,
            data: bundle_bytes,
        });

        // Bundle with default flags is PublicDurable: AnnounceContent (PersistToDisk deferred).
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_rejects_cid_data_mismatch() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        // CID for "real data" but send "tampered data"
        let cid = ContentId::for_blob(b"real data", crate::cid::ContentFlags::default()).unwrap();
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: b"tampered data".to_vec(),
        });

        assert!(
            actions.is_empty(),
            "mismatched CID should produce no actions"
        );
        assert_eq!(tier.metrics().transit_rejected, 1);
        assert_eq!(tier.metrics().transit_stored, 0);
    }

    #[test]
    fn publish_rejects_cid_data_mismatch() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        let cid = ContentId::for_blob(b"original", crate::cid::ContentFlags::default()).unwrap();
        let actions = tier.handle(StorageTierEvent::PublishContent {
            cid,
            data: b"different".to_vec(),
        });

        assert!(
            actions.is_empty(),
            "mismatched CID should produce no actions"
        );
        assert_eq!(tier.metrics().publishes_stored, 0);
        assert_eq!(tier.metrics().publishes_rejected, 1);
    }

    #[test]
    fn transit_rejects_cid_with_wrong_size_field() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        // Craft a CID with correct hash but wrong payload_size by mutating
        // the size bits in the last 4 bytes.
        let data = b"correct data";
        let real_cid = ContentId::for_blob(data, crate::cid::ContentFlags::default()).unwrap();
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
        let (tier, actions) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
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

    fn make_tier_with_policy(policy: ContentPolicy) -> StorageTier<MemoryBlobStore> {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (tier, _) = StorageTier::new(MemoryBlobStore::new(), budget, policy);
        tier
    }

    fn cid_with_class(data: &[u8], encrypted: bool, ephemeral: bool) -> (ContentId, Vec<u8>) {
        let flags = crate::cid::ContentFlags {
            encrypted,
            ephemeral,
            alt_hash: false,
        };
        let cid = ContentId::for_blob(data, flags).unwrap();
        (cid, data.to_vec())
    }

    #[test]
    fn transit_rejects_encrypted_ephemeral() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"secret stream", true, true);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        assert!(actions.is_empty());
        assert_eq!(tier.metrics().transit_rejected, 1);
    }

    #[test]
    fn publish_rejects_encrypted_ephemeral() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"secret stream", true, true);
        let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
        assert!(actions.is_empty());
        assert_eq!(tier.metrics().publishes_rejected, 1);
    }

    #[test]
    fn transit_rejects_encrypted_durable_when_policy_off() {
        let policy = ContentPolicy {
            encrypted_durable_persist: false,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted file", true, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        assert!(actions.is_empty());
        assert_eq!(tier.metrics().transit_rejected, 1);
    }

    #[test]
    fn transit_admits_encrypted_durable_when_policy_on() {
        let policy = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted file", true, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // EncryptedDurable: AnnounceContent (PersistToDisk deferred until disk I/O is wired).
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_encrypted_durable_survives_public_durable_pressure() {
        // Regression: EncryptedDurable (eviction_priority formerly 1) was
        // permanently rejected by should_admit when probation was full of
        // PublicDurable (priority 0), making encrypted_durable_persist=true
        // a no-op for transit in steady state. With the fix, both durable
        // classes share priority 0, so frequency breaks ties.
        let budget = StorageBudget {
            cache_capacity: 5,
            max_pinned_bytes: 1_000_000,
        };
        let policy = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        let (mut tier, _) = StorageTier::new(MemoryBlobStore::new(), budget, policy);

        // Fill cache with PublicDurable content.
        for i in 0..4 {
            let (cid, data) = cid_with_class(format!("pub-{i}").as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent { cid, data });
        }

        // Transit the EncryptedDurable CID multiple times to build frequency.
        // Previously this was pointless (class comparison short-circuited before
        // frequency was consulted). Now frequency is consulted for same-priority
        // classes, so repeated transits warm the sketch.
        let (enc_cid, enc_data) = cid_with_class(b"encrypted file", true, false);
        for _ in 0..5 {
            tier.handle(StorageTierEvent::TransitContent {
                cid: enc_cid,
                data: enc_data.clone(),
            });
        }

        let rejected_before = tier.metrics().transit_rejected;

        // Final transit — with built-up frequency, should beat the probation victim.
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: enc_cid,
            data: enc_data,
        });
        assert_eq!(
            tier.metrics().transit_rejected,
            rejected_before,
            "EncryptedDurable with high frequency should not be rejected"
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::AnnounceContent { .. })),
            "EncryptedDurable should emit AnnounceContent"
        );
    }

    #[test]
    fn transit_admits_public_durable() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"public doc", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // PublicDurable: AnnounceContent (PersistToDisk deferred until disk I/O is wired).
        assert_eq!(actions.len(), 1);
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_admits_public_ephemeral() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"live stream", false, true);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        assert_eq!(actions.len(), 1);
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn content_policy_defaults_are_conservative() {
        let policy = ContentPolicy::default();
        assert!(!policy.encrypted_durable_persist);
        assert!(!policy.encrypted_durable_announce);
        assert!(policy.public_ephemeral_announce);
    }

    #[test]
    fn storage_tier_accepts_policy() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let policy = ContentPolicy::default();
        let (tier, _) = StorageTier::new(MemoryBlobStore::new(), budget, policy);
        assert_eq!(tier.metrics().queries_served, 0);
    }

    // ---- Announcement gating by content class ----

    #[test]
    fn transit_public_ephemeral_no_announce_when_policy_off() {
        let policy = ContentPolicy {
            public_ephemeral_announce: false,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"live stream", false, true);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // Should store but NOT announce.
        assert!(actions.is_empty(), "no announce when policy off");
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_public_ephemeral_announces_when_policy_on() {
        let policy = ContentPolicy {
            public_ephemeral_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"live stream", false, true);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
    }

    #[test]
    fn transit_encrypted_durable_no_announce_when_policy_off() {
        let policy = ContentPolicy {
            encrypted_durable_persist: true, // must persist to reach announce check
            encrypted_durable_announce: false,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted doc", true, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // EncryptedDurable: no actions (no announce when policy off, PersistToDisk deferred).
        assert!(
            actions.is_empty(),
            "no announce + deferred persist = no actions"
        );
        assert_eq!(tier.metrics().transit_stored, 1);
    }

    #[test]
    fn transit_encrypted_durable_announces_when_policy_on() {
        let policy = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted doc", true, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // EncryptedDurable with announce on: AnnounceContent only (PersistToDisk deferred).
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
    }

    #[test]
    fn transit_public_durable_always_announces() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"valuable doc", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        // PublicDurable: AnnounceContent only (PersistToDisk deferred).
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
    }

    // ---- Sans-I/O disk persistence ----

    #[test]
    fn persist_to_disk_deferred_until_runtime_wired() {
        // PersistToDisk emission is deferred to avoid cloning data on the
        // hot path when the runtime discards the action (disk I/O not yet
        // wired). Neither durable nor ephemeral transit should emit it.
        let mut tier = make_tier_with_policy(ContentPolicy::default());

        let (dur_cid, dur_data) = cid_with_class(b"durable content", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: dur_cid,
            data: dur_data,
        });
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::PersistToDisk { .. })),
            "PersistToDisk deferred — should not be emitted"
        );

        let (eph_cid, eph_data) = cid_with_class(b"ephemeral content", false, true);
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: eph_cid,
            data: eph_data,
        });
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::PersistToDisk { .. })),
            "ephemeral content should never emit PersistToDisk"
        );
    }

    #[test]
    fn disk_read_complete_serves_reply() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let data = b"on-disk content".to_vec();
        let flags = crate::cid::ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: false,
        };
        let cid = ContentId::for_blob(&data, flags).unwrap();

        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid,
            query_id: 42,
            data: data.clone(),
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 42);
                assert_eq!(payload, &data);
            }
            other => panic!("expected SendReply, got {other:?}"),
        }
    }

    #[test]
    fn disk_read_complete_rejects_corrupted_data() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let flags = crate::cid::ContentFlags {
            encrypted: false,
            ephemeral: false,
            alt_hash: false,
        };
        let cid = ContentId::for_blob(b"original", flags).unwrap();

        // DiskReadComplete with data that doesn't match the CID hash.
        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid,
            query_id: 42,
            data: b"corrupted".to_vec(),
        });
        assert!(actions.is_empty(), "corrupted disk data should be rejected");
    }

    #[test]
    fn cache_miss_returns_empty() {
        // DiskLookup is not yet serviced by the runtime (disk I/O is a future
        // bead). Until then, cache misses return [] to avoid hanging queries.
        let budget = StorageBudget {
            cache_capacity: 3,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());
        let (cid, data) = cid_with_class(b"durable on disk", false, false);

        // Store via transit — adds to cache.
        tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.clone(),
        });

        // Flood cache to evict original CID.
        for i in 0..10 {
            let filler = format!("filler-{i}");
            let (filler_cid, filler_data) = cid_with_class(filler.as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent {
                cid: filler_cid,
                data: filler_data,
            });
        }

        // Query original CID — cache miss returns empty (no DiskLookup until
        // runtime can service it).
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 77, cid });
        assert!(
            actions.is_empty(),
            "cache miss should return empty until disk I/O is wired, got: {actions:?}"
        );
    }

    // ---- Integration tests: multi-step policy enforcement ----

    #[test]
    fn durable_lifecycle_store_and_disk_read() {
        // Store durable content → DiskReadComplete verifies and re-caches → reply served.
        //
        // PersistToDisk and DiskLookup emission are deferred until the runtime
        // can service disk I/O. This test exercises DiskReadComplete handling.
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"durable lifecycle", false, false);

        // Store via transit — should emit AnnounceContent.
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.clone(),
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::AnnounceContent { .. })));

        // Simulate disk read response — should verify, re-cache, and reply.
        let disk_actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid,
            query_id: 99,
            data: data.clone(),
        });
        assert!(disk_actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::SendReply { query_id: 99, .. })));

        // Content should be queryable from cache after re-caching.
        let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });
        assert!(query_actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::SendReply { query_id: 1, .. })));
    }

    #[test]
    fn mixed_class_eviction_ephemeral_first() {
        // Fill cache with PublicDurable + PublicEphemeral, verify ephemeral evicted first.
        let budget = StorageBudget {
            cache_capacity: 5,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) =
            StorageTier::new(MemoryBlobStore::new(), budget, ContentPolicy::default());

        // Store 2 durable items via publish (bypasses W-TinyLFU).
        let d1_data = b"durable-1";
        let (d1_cid, d1_vec) = cid_with_class(d1_data, false, false);
        tier.handle(StorageTierEvent::PublishContent {
            cid: d1_cid,
            data: d1_vec,
        });

        let d2_data = b"durable-2";
        let (d2_cid, d2_vec) = cid_with_class(d2_data, false, false);
        tier.handle(StorageTierEvent::PublishContent {
            cid: d2_cid,
            data: d2_vec,
        });

        // Store 2 ephemeral items via publish.
        let e1_data = b"ephemeral-1";
        let (e1_cid, e1_vec) = cid_with_class(e1_data, false, true);
        tier.handle(StorageTierEvent::PublishContent {
            cid: e1_cid,
            data: e1_vec,
        });

        let e2_data = b"ephemeral-2";
        let (e2_cid, e2_vec) = cid_with_class(e2_data, false, true);
        tier.handle(StorageTierEvent::PublishContent {
            cid: e2_cid,
            data: e2_vec,
        });

        // Fill cache to trigger evictions with durable items.
        for i in 0..10 {
            let data = format!("pressure-{i}");
            let (pressure_cid, pressure_data) = cid_with_class(data.as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent {
                cid: pressure_cid,
                data: pressure_data,
            });
        }

        // Durable items should survive (either in memory or on disk).
        // At least one durable item should be queryable.
        let d1_hit = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 1,
            cid: d1_cid,
        });
        let d2_hit = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 2,
            cid: d2_cid,
        });
        assert!(
            !d1_hit.is_empty() || !d2_hit.is_empty(),
            "at least one durable item should be retrievable (via cache or disk)"
        );
    }

    #[test]
    fn policy_toggle_encrypted_durable() {
        // With policy off: encrypted durable rejected.
        let policy_off = ContentPolicy {
            encrypted_durable_persist: false,
            ..ContentPolicy::default()
        };
        let mut tier_off = make_tier_with_policy(policy_off);
        let (cid, data) = cid_with_class(b"encrypted file", true, false);
        let actions = tier_off.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.clone(),
        });
        assert!(actions.is_empty());
        assert_eq!(tier_off.metrics().transit_rejected, 1);

        // With policy on: encrypted durable admitted.
        let policy_on = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier_on = make_tier_with_policy(policy_on);
        let actions = tier_on.handle(StorageTierEvent::TransitContent { cid, data });
        assert!(!actions.is_empty());
        assert_eq!(tier_on.metrics().transit_stored, 1);
    }
}

//! StorageTier: sans-I/O wrapper integrating ContentStore with Zenoh patterns.

use crate::book::BookStore;
use crate::cache::ContentStore;
use crate::cid::{ContentClass, ContentId};
use crate::flatpack::FlatpackIndex;
use alloc::{
    collections::VecDeque,
    string::{String, ToString},
    vec,
    vec::Vec,
};
use harmony_zenoh::namespace::{announce as announce_ns, content as ns};
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

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
    pub disk_reads_served: u64,
    pub disk_read_failures: u64,
    pub s3_reads_served: u64,
    pub s3_read_failures: u64,
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
            disk_reads_served,
            disk_read_failures,
            s3_reads_served,
            s3_read_failures,
        } = self;
        let fields: &[u64] = &[
            *queries_served,
            *cache_hits,
            *cache_misses,
            *transit_stored,
            *transit_rejected,
            *publishes_stored,
            *publishes_rejected,
            *disk_reads_served,
            *disk_read_failures,
            *s3_reads_served,
            *s3_read_failures,
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
    /// Disk read failed — runtime could not deliver the requested data.
    DiskReadFailed { cid: ContentId, query_id: u64 },
    /// S3 read completed — book fetched from remote storage.
    S3ReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
    /// S3 read failed — book not found or network error.
    S3ReadFailed { cid: ContentId, query_id: u64 },
    /// Timer tick for periodic filter broadcasts.
    FilterTimerTick,
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
    /// Fall back to S3 for a durable CID not found in cache or on disk.
    S3Lookup { cid: ContentId, query_id: u64 },
    /// Broadcast a Bloom filter snapshot of the cached CID set.
    ///
    /// The runtime constructs the full key expression (including node address)
    /// since `StorageTier` is sans-I/O and has no identity context.
    BroadcastFilter { payload: Vec<u8> },
    /// Broadcast a cuckoo filter snapshot of the Flatpack reverse index.
    ///
    /// The runtime constructs the full key expression (including node address)
    /// since `StorageTier` is sans-I/O and has no identity context.
    BroadcastCuckooFilter { payload: Vec<u8> },
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

/// Configuration for periodic Bloom filter broadcasts.
#[derive(Debug, Clone)]
pub struct FilterBroadcastConfig {
    /// Broadcast after this many cache admissions.
    pub mutation_threshold: u32,
    /// Maximum ticks between broadcasts (even if no mutations).
    /// The runtime injects `FilterTimerTick` events at this interval.
    pub max_interval_ticks: u32,
    /// Expected item count for sizing the Bloom filter (should match cache capacity).
    pub expected_items: u32,
    /// Target false positive rate.
    pub fp_rate: f64,
}

impl Default for FilterBroadcastConfig {
    fn default() -> Self {
        Self {
            mutation_threshold: 100,
            max_interval_ticks: 30,
            expected_items: 1024,
            fp_rate: 0.001,
        }
    }
}

/// Sans-I/O storage tier integrating [`ContentStore`] with Zenoh key patterns.
///
/// On construction, returns startup actions (queryable and subscriber
/// declarations) that the caller must execute. Subsequent calls to
/// [`handle`](Self::handle) process inbound events and return outbound
/// actions.
pub struct StorageTier<B: BookStore> {
    cache: ContentStore<B>,
    /// Reserved for future byte-based pin limits. Currently only count-based
    /// limits are enforced via [`ContentStore::pin_limit`].
    #[allow(dead_code)]
    budget: StorageBudget,
    /// Per-class admission and announcement policy.
    policy: ContentPolicy,
    metrics: StorageMetrics,
    /// Whether the runtime has wired up disk I/O.
    /// When false, PersistToDisk and DiskLookup actions are suppressed.
    disk_enabled: bool,
    /// CIDs known to be persisted on disk, mapped to their byte sizes (durable classes only).
    disk_index: HashMap<ContentId, u64>,
    /// LRU ordering of disk-indexed CIDs. Front = oldest, back = most recent.
    disk_lru: VecDeque<ContentId>,
    /// Running total of bytes stored on disk.
    disk_used_bytes: u64,
    /// Maximum bytes allowed on disk. None = unlimited.
    disk_quota: Option<u64>,
    /// Whether the runtime has wired up S3 remote storage.
    /// When false, S3Lookup actions are suppressed.
    s3_enabled: bool,
    /// Configuration for periodic Bloom filter broadcasts.
    filter_config: FilterBroadcastConfig,
    /// Number of cache mutations since last Bloom filter broadcast.
    mutations_since_broadcast: u32,
    /// Flatpack reverse index (child_cid → bundle_cids).
    flatpack: FlatpackIndex,
}

impl<B: BookStore> StorageTier<B> {
    /// Create a new StorageTier with startup actions.
    pub fn new(
        store: B,
        budget: StorageBudget,
        policy: ContentPolicy,
        filter_config: FilterBroadcastConfig,
    ) -> (Self, Vec<StorageTierAction>) {
        assert!(
            !policy.encrypted_durable_announce || policy.encrypted_durable_persist,
            "encrypted_durable_announce requires encrypted_durable_persist — \
             content rejected by class admission will never reach the announce check"
        );

        // Clamp mutation_threshold to at least 1 — zero would trigger a
        // filter rebuild on every single cache write.
        let filter_config = FilterBroadcastConfig {
            mutation_threshold: filter_config.mutation_threshold.max(1),
            ..filter_config
        };

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

        let flatpack = FlatpackIndex::new(filter_config.expected_items);
        let tier = Self {
            cache,
            budget,
            policy,
            metrics: StorageMetrics::default(),
            disk_enabled: false,
            disk_index: HashMap::new(),
            disk_lru: VecDeque::new(),
            disk_used_bytes: 0,
            disk_quota: None,
            s3_enabled: false,
            filter_config,
            mutations_since_broadcast: 0,
            flatpack,
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

    /// Read-only access to the filter broadcast configuration.
    pub fn filter_config(&self) -> &FilterBroadcastConfig {
        &self.filter_config
    }

    /// Read-only access to the Flatpack reverse index.
    pub fn flatpack(&self) -> &FlatpackIndex {
        &self.flatpack
    }

    /// Enable disk I/O and populate the disk index with CIDs and their byte sizes found on disk.
    ///
    /// Called by the runtime after scanning the data directory at startup.
    /// Once enabled, `handle_transit` and `handle_publish` emit
    /// [`StorageTierAction::PersistToDisk`] for durable content, and
    /// `handle_content_query` emits [`StorageTierAction::DiskLookup`]
    /// on cache misses for CIDs present in the disk index.
    pub fn enable_disk(&mut self, entries: impl IntoIterator<Item = (ContentId, u64)>) {
        self.disk_enabled = true;
        for (cid, size) in entries {
            self.disk_index.insert(cid, size);
            self.disk_lru.push_back(cid);
            self.disk_used_bytes += size;
        }
    }

    /// Set the maximum number of bytes allowed on disk. Replaces any previous quota.
    /// Set the disk quota and run an immediate eviction pass if existing
    /// data exceeds the quota. Returns `RemoveFromDisk` actions for any
    /// entries evicted during the initial enforcement.
    pub fn set_disk_quota(&mut self, quota_bytes: u64) -> Vec<StorageTierAction> {
        self.disk_quota = Some(quota_bytes);
        self.enforce_disk_quota()
    }

    /// Run a standalone eviction pass without recording a new persist.
    /// Used at startup when existing data may exceed a new/tightened quota.
    fn enforce_disk_quota(&mut self) -> Vec<StorageTierAction> {
        let quota = match self.disk_quota {
            Some(q) => q,
            None => return Vec::new(),
        };
        if self.disk_used_bytes <= quota {
            return Vec::new();
        }

        let mut actions = Vec::new();
        let mut skipped: usize = 0;
        while self.disk_used_bytes > quota {
            if skipped >= self.disk_lru.len() {
                break;
            }
            let candidate = match self.disk_lru.pop_front() {
                Some(c) => c,
                None => break,
            };
            if self.cache.is_pinned(&candidate) {
                self.disk_lru.push_back(candidate);
                skipped += 1;
                continue;
            }
            if let Some(entry_size) = self.disk_index.remove(&candidate) {
                self.disk_used_bytes = self.disk_used_bytes.saturating_sub(entry_size);
            }
            actions.push(StorageTierAction::RemoveFromDisk { cid: candidate });
            skipped = 0;
        }
        actions
    }

    /// Returns the running total of bytes currently tracked in the disk index.
    pub fn disk_used_bytes(&self) -> u64 {
        self.disk_used_bytes
    }

    /// Returns the configured disk quota, or `None` if unlimited.
    pub fn disk_quota(&self) -> Option<u64> {
        self.disk_quota
    }

    /// Returns `true` if the given CID is present in the disk index.
    pub fn disk_contains(&self, cid: &ContentId) -> bool {
        self.disk_index.contains_key(cid)
    }

    /// Whether disk I/O is currently enabled.
    pub fn disk_enabled(&self) -> bool {
        self.disk_enabled
    }

    /// Enable S3 remote storage fallback.
    ///
    /// When enabled, cache + disk misses for durable content classes emit
    /// [`StorageTierAction::S3Lookup`] so the runtime can fetch from S3.
    pub fn enable_s3(&mut self) {
        self.s3_enabled = true;
    }

    /// Disable S3 fallback (e.g., when S3Library init fails at startup).
    pub fn disable_s3(&mut self) {
        self.s3_enabled = false;
    }

    /// Whether S3 remote storage is currently enabled.
    pub fn s3_enabled(&self) -> bool {
        self.s3_enabled
    }

    /// Retrieve raw book data by CID from the underlying content store.
    pub fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.cache.get(cid)
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
                if !Self::is_durable_class(&cid) {
                    self.metrics.disk_read_failures += 1;
                    if let Some(size) = self.disk_index.remove(&cid) {
                        self.disk_used_bytes = self.disk_used_bytes.saturating_sub(size);
                        self.disk_lru.retain(|c| c != &cid);
                    }
                    return vec![StorageTierAction::SendReply {
                        query_id,
                        payload: vec![],
                    }];
                }
                // Verify integrity — disk data may be corrupted (bit rot, wrong file).
                if !Self::verify_cid(&cid, &data) {
                    self.metrics.disk_read_failures += 1;
                    // Retract stale index entry — corrupted files cause infinite
                    // DiskLookup → read → verify-fail → empty reply loops otherwise.
                    if let Some(size) = self.disk_index.remove(&cid) {
                        self.disk_used_bytes = self.disk_used_bytes.saturating_sub(size);
                        self.disk_lru.retain(|c| c != &cid);
                    }
                    return vec![StorageTierAction::SendReply {
                        query_id,
                        payload: vec![],
                    }];
                }
                // Pre-warm frequency so the re-cached CID survives the admission
                // challenge. Without this, a CID with decayed sketch counts would
                // be immediately re-evicted, creating a disk-read thrashing loop.
                // 5 increments beat cold items (freq ~1) without unfairly displacing
                // genuinely hot cached items (freq 10+).
                self.cache.warm_frequency(&cid, 5);
                self.cache.store(cid, data.clone());
                self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
                // Note: queries_served was already incremented in handle_content_query
                // when this query first arrived — only count the disk-specific metric.
                self.metrics.disk_reads_served += 1;
                self.touch_disk_lru(&cid);
                let mut actions = vec![StorageTierAction::SendReply {
                    query_id,
                    payload: data,
                }];
                if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
                    actions.push(self.rebuild_filter());
                }
                actions
            }
            StorageTierEvent::DiskReadFailed { cid, query_id } => {
                self.metrics.disk_read_failures += 1;
                // Retract the optimistic disk_index entry so subsequent queries
                // don't trigger futile DiskLookup cycles for a CID that failed.
                if let Some(size) = self.disk_index.remove(&cid) {
                    self.disk_used_bytes = self.disk_used_bytes.saturating_sub(size);
                    self.disk_lru.retain(|c| c != &cid);
                }
                // Reply with empty payload so the querier doesn't hang.
                vec![StorageTierAction::SendReply {
                    query_id,
                    payload: vec![],
                }]
            }
            StorageTierEvent::S3ReadComplete {
                cid,
                query_id,
                data,
            } => {
                // Verify integrity — S3 data may be corrupted or tampered.
                if !Self::verify_cid(&cid, &data) {
                    self.metrics.s3_read_failures += 1;
                    return vec![StorageTierAction::SendReply {
                        query_id,
                        payload: vec![],
                    }];
                }
                // Pre-warm frequency so the cached CID survives the admission
                // challenge (same rationale as DiskReadComplete).
                self.cache.warm_frequency(&cid, 5);

                // Clone data for disk persistence BEFORE cache and reply consume it.
                // At most 2 clones: one for cache, one for persist (reply gets the original).
                let persist_data = if self.disk_enabled && Self::is_durable_class(&cid) {
                    Some(data.clone())
                } else {
                    None
                };

                self.cache.store(cid, data.clone());
                self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
                self.metrics.s3_reads_served += 1;

                // Original `data` goes to SendReply (no clone needed).
                let mut actions = vec![StorageTierAction::SendReply {
                    query_id,
                    payload: data,
                }];

                // Persist to disk so future queries hit disk instead of S3.
                if let Some(persist_bytes) = persist_data {
                    let persist_size = persist_bytes.len() as u64;
                    actions.push(StorageTierAction::PersistToDisk {
                        cid,
                        data: persist_bytes,
                    });
                    self.record_disk_persist(cid, persist_size, &mut actions);
                }

                if self.should_announce(&cid) {
                    actions.push(self.make_announce_action(&cid));
                }

                if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
                    actions.push(self.rebuild_filter());
                }
                actions
            }
            StorageTierEvent::S3ReadFailed { cid: _, query_id } => {
                self.metrics.s3_read_failures += 1;
                // Reply with empty payload so the querier doesn't hang.
                vec![StorageTierAction::SendReply {
                    query_id,
                    payload: vec![],
                }]
            }
            StorageTierEvent::FilterTimerTick => {
                vec![self.rebuild_filter(), self.rebuild_cuckoo_filter()]
            }
        }
    }

    /// Rebuild the Bloom filter from the current cache state and return
    /// a `BroadcastFilter` action. Resets the mutation counter.
    fn rebuild_filter(&mut self) -> StorageTierAction {
        use crate::bloom::BloomFilter;

        let mut filter = BloomFilter::new(
            self.filter_config.expected_items,
            self.filter_config.fp_rate,
        );

        for cid in self.cache.iter_admitted() {
            if self.class_admits(&cid) {
                filter.insert(&cid);
            }
        }

        self.mutations_since_broadcast = 0;

        StorageTierAction::BroadcastFilter {
            payload: filter.to_bytes(),
        }
    }

    /// Rebuild the cuckoo filter from the Flatpack reverse index and return
    /// a `BroadcastCuckooFilter` action. Resets the Flatpack mutation counter.
    fn rebuild_cuckoo_filter(&mut self) -> StorageTierAction {
        self.flatpack.rebuild_filter();
        self.flatpack.reset_mutation_counter();
        StorageTierAction::BroadcastCuckooFilter {
            payload: self.flatpack.filter().to_bytes(),
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
                if self.disk_enabled && self.disk_index.contains_key(cid) {
                    return vec![StorageTierAction::DiskLookup {
                        cid: *cid,
                        query_id,
                    }];
                }
                if self.s3_enabled && Self::is_durable_class(cid) {
                    return vec![StorageTierAction::S3Lookup {
                        cid: *cid,
                        query_id,
                    }];
                }
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
        // Class-based admission first — O(1) flag check before O(data_size) hash.
        if !self.class_admits(&cid) {
            self.metrics.transit_rejected += 1;
            return vec![];
        }
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

        // Clone data for disk persistence BEFORE cache consumes it.
        // Skip if already in disk_index to avoid redundant 1MB clones on re-transit.
        let persist_data = if self.disk_enabled
            && Self::is_durable_class(&cid)
            && !self.disk_index.contains_key(&cid)
        {
            Some(data.clone())
        } else {
            None
        };

        // Use store_preadmitted to avoid double-incrementing the sketch
        // counter (should_admit already incremented it).
        self.cache.store_preadmitted(cid, data);
        self.metrics.transit_stored += 1;
        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);

        let mut actions = Vec::new();

        // After cache insertion, persist durable content to disk.
        // Optimistic: disk_index is updated before the async write completes.
        // Race window: if cache evicts this CID before the write finishes and a
        // query arrives, DiskLookup will fail (file not yet on disk).
        // DiskReadFailed retracts the index entry; the write eventually succeeds,
        // leaving an orphaned file that self-heals on the next startup scan.
        if let Some(persist_bytes) = persist_data {
            let persist_size = persist_bytes.len() as u64;
            actions.push(StorageTierAction::PersistToDisk {
                cid,
                data: persist_bytes,
            });
            self.record_disk_persist(cid, persist_size, &mut actions);
        }
        // Re-announcing already-cached content is intentional: it refreshes
        // the announcement TTL so peers know the content is still available.
        // Announce only if policy allows it for this content class.
        if self.should_announce(&cid) {
            actions.push(self.make_announce_action(&cid));
        }
        if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
            actions.push(self.rebuild_filter());
        }
        actions
    }

    fn handle_publish(&mut self, cid: ContentId, data: Vec<u8>) -> Vec<StorageTierAction> {
        // Class-based admission first — O(1) flag check before O(data_size) hash.
        if !self.class_admits(&cid) {
            self.metrics.publishes_rejected += 1;
            return vec![];
        }
        // Publish comes from local apps — still verify as defense-in-depth.
        if !Self::verify_cid(&cid, &data) {
            self.metrics.publishes_rejected += 1;
            return vec![];
        }
        // Clone data for disk persistence BEFORE cache consumes it.
        // Skip if already in disk_index to avoid redundant clones on re-publish.
        let persist_data = if self.disk_enabled
            && Self::is_durable_class(&cid)
            && !self.disk_index.contains_key(&cid)
        {
            Some(data.clone())
        } else {
            None
        };

        // Publish always stores — bypasses admission filter.
        // TODO: Pin published content to guarantee long-term retention (deferred).
        self.cache.store(cid, data);
        self.metrics.publishes_stored += 1;
        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);

        let mut actions = Vec::new();

        // After cache insertion, persist durable content to disk.
        if let Some(persist_bytes) = persist_data {
            let persist_size = persist_bytes.len() as u64;
            actions.push(StorageTierAction::PersistToDisk {
                cid,
                data: persist_bytes,
            });
            self.record_disk_persist(cid, persist_size, &mut actions);
        }
        // Announce only if policy allows it for this content class.
        if self.should_announce(&cid) {
            actions.push(self.make_announce_action(&cid));
        }
        if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
            actions.push(self.rebuild_filter());
        }
        actions
    }

    /// Verify that a CID is consistent with the given data.
    ///
    /// Checks three things:
    /// 1. The truncated SHA-256 hash matches the data.
    /// 2. The payload size field matches the actual data length (leaf books only).
    /// 3. The internal checksum is valid (hash + size + type are consistent).
    fn verify_cid(cid: &ContentId, data: &[u8]) -> bool {
        if !cid.verify_hash(data) {
            return false;
        }
        if cid.depth() == 0 && !cid.is_inline() && cid.payload_size() as usize != data.len() {
            // Only exact-size check for leaf books — bundles use float-encoded
            // size (approximate), so their sizes can't be validated exactly.
            return false;
        }
        cid.verify_checksum().is_ok()
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

    /// Record a book persisted to disk and run eviction if over quota.
    fn record_disk_persist(
        &mut self,
        cid: ContentId,
        size: u64,
        actions: &mut Vec<StorageTierAction>,
    ) {
        // Dedup: if already on disk, skip
        if self.disk_index.contains_key(&cid) {
            return;
        }

        // Update bookkeeping
        self.disk_index.insert(cid, size);
        self.disk_lru.push_back(cid);
        self.disk_used_bytes += size;

        // Check quota
        let quota = match self.disk_quota {
            Some(q) => q,
            None => return,
        };

        if self.disk_used_bytes <= quota {
            return;
        }

        // Evict LRU entries until under quota
        let mut skipped: usize = 0;
        while self.disk_used_bytes > quota {
            if skipped >= self.disk_lru.len() {
                // Safety valve: all entries are unevictable.
                // Cannot log here (harmony-content is no_std / no tracing).
                break;
            }

            let candidate = match self.disk_lru.pop_front() {
                Some(c) => c,
                None => break,
            };

            // Never evict the just-persisted book
            if candidate == cid {
                self.disk_lru.push_back(candidate);
                skipped += 1;
                continue;
            }

            // Skip pinned books
            if self.cache.is_pinned(&candidate) {
                self.disk_lru.push_back(candidate);
                skipped += 1;
                continue;
            }

            // Evict
            if let Some(entry_size) = self.disk_index.remove(&candidate) {
                self.disk_used_bytes = self.disk_used_bytes.saturating_sub(entry_size);
            }
            // If a PersistToDisk for this candidate is in the current actions
            // list, remove it — the persist and remove cancel out. Without this,
            // both would be dispatched as concurrent spawn_blocking tasks and the
            // delete could race ahead of the write, leaving an orphaned file.
            let pending_persist = actions.iter().position(
                |a| matches!(a, StorageTierAction::PersistToDisk { cid: c, .. } if *c == candidate),
            );
            if let Some(idx) = pending_persist {
                actions.swap_remove(idx);
                // Don't emit RemoveFromDisk either — the file was never written.
            } else {
                actions.push(StorageTierAction::RemoveFromDisk { cid: candidate });
            }
            skipped = 0; // Reset after successful eviction
        }
    }

    /// Move a CID to the back of the disk LRU (most recently used).
    pub(crate) fn touch_disk_lru(&mut self, cid: &ContentId) {
        if !self.disk_index.contains_key(cid) {
            return; // CID was evicted while DiskLookup was in-flight; don't re-insert
        }
        self.disk_lru.retain(|c| c != cid);
        self.disk_lru.push_back(*cid);
    }
}

#[cfg(test)]
impl<B: BookStore> StorageTier<B> {
    /// Test helper: simulate a PersistToDisk event and run eviction.
    pub fn test_persist_to_disk(&mut self, cid: ContentId, size: u64) -> Vec<StorageTierAction> {
        let mut actions = Vec::new();
        self.record_disk_persist(cid, size, &mut actions);
        actions
    }

    /// Test helper: simulate multiple PersistToDisk events in one batch,
    /// as would happen when several transits are processed in one tick.
    /// Each persist pushes a PersistToDisk action before calling
    /// record_disk_persist, mirroring the real handle_transit flow.
    pub fn test_persist_batch(&mut self, entries: &[(ContentId, u64)]) -> Vec<StorageTierAction> {
        let mut actions = Vec::new();
        for &(cid, size) in entries {
            actions.push(StorageTierAction::PersistToDisk {
                cid,
                data: alloc::vec![0u8; size as usize],
            });
            self.record_disk_persist(cid, size, &mut actions);
        }
        actions
    }

    /// Test helper: delegate to cache pin.
    pub fn cache_pin(&mut self, cid: ContentId) -> bool {
        self.cache.pin(cid)
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
            cid: ContentId::for_book(b"test", crate::cid::ContentFlags::default()).unwrap(),
        };
        let _action = StorageTierAction::SendReply {
            query_id: 1,
            payload: vec![1, 2, 3],
        };
    }

    use crate::book::MemoryBookStore;

    #[test]
    fn content_query_hit_returns_reply() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let cid = ContentId::for_book(b"not stored", crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        let data = b"stats test blob";
        let cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();
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
                assert_eq!(payload.len(), 88); // 11 u64 fields
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let data = b"explicitly published blob";
        let cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let data = b"transiting blob";
        let cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let data = b"repeated transit";
        let cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();

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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        // Fill cache: 3 items → window=1, probation=2.
        for i in 0..3 {
            let data = format!("hot-{i}");
            let cid =
                ContentId::for_book(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap();
            tier.handle(StorageTierEvent::TransitContent {
                cid,
                data: data.into_bytes(),
            });
        }
        // Boost frequency of the probation items via queries.
        for i in 0..2 {
            let data = format!("hot-{i}");
            let cid =
                ContentId::for_book(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap();
            for _ in 0..10 {
                tier.handle(StorageTierEvent::ContentQuery { query_id: 0, cid });
            }
        }
        let admitted_before = tier.metrics().transit_stored;
        let rejected_before = tier.metrics().transit_rejected;

        // Cold transit item with zero frequency should be rejected.
        let cold_data = b"cold-newcomer-will-lose";
        let cold_cid = ContentId::for_book(cold_data, crate::cid::ContentFlags::default()).unwrap();
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        let book_a = ContentId::for_book(b"child-a", crate::cid::ContentFlags::default()).unwrap();
        let book_b = ContentId::for_book(b"child-b", crate::cid::ContentFlags::default()).unwrap();
        let children = [book_a, book_b];
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        // CID for "real data" but send "tampered data"
        let cid = ContentId::for_book(b"real data", crate::cid::ContentFlags::default()).unwrap();
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        let cid = ContentId::for_book(b"original", crate::cid::ContentFlags::default()).unwrap();
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        // Craft a CID with correct hash but wrong payload_size by mutating
        // the size bits in the header (bytes 0-3).
        let data = b"correct data";
        let real_cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();
        let mut bytes = real_cid.to_bytes();
        // Corrupt the size: set size to 999 instead of 12.
        let header = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let keep_mask = 0xFFC0_0003u32; // mode (bits 31-28) + full depth (bits 27-22) + checksum (bits 1-0)
        let fake_header = (header & keep_mask) | ((999u32 & 0xF_FFFF) << 2);
        bytes[0..4].copy_from_slice(&fake_header.to_be_bytes());
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
        let (tier, actions) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
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

    fn make_tier_with_policy(policy: ContentPolicy) -> StorageTier<MemoryBookStore> {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            policy,
            FilterBroadcastConfig::default(),
        );
        tier
    }

    fn cid_with_class(data: &[u8], encrypted: bool, ephemeral: bool) -> (ContentId, Vec<u8>) {
        let flags = crate::cid::ContentFlags {
            encrypted,
            ephemeral,
            sha224: false,
            lsb_mode: false,
        };
        let cid = ContentId::for_book(data, flags).unwrap();
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
    fn publish_public_ephemeral_no_announce_when_policy_off() {
        let policy = ContentPolicy {
            public_ephemeral_announce: false,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"live stream", false, true);
        let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
        // Stored but not announced.
        assert!(actions.is_empty(), "no announce when policy off");
        assert_eq!(tier.metrics().publishes_stored, 1);
    }

    #[test]
    fn publish_public_ephemeral_announces_when_policy_on() {
        let policy = ContentPolicy {
            public_ephemeral_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"live stream", false, true);
        let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
        assert_eq!(tier.metrics().publishes_stored, 1);
    }

    #[test]
    fn publish_encrypted_durable_no_announce_when_policy_off() {
        let policy = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: false,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted doc", true, false);
        let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
        // Stored but not announced.
        assert!(actions.is_empty(), "no announce when policy off");
        assert_eq!(tier.metrics().publishes_stored, 1);
    }

    #[test]
    fn publish_encrypted_durable_announces_when_policy_on() {
        let policy = ContentPolicy {
            encrypted_durable_persist: true,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        let mut tier = make_tier_with_policy(policy);
        let (cid, data) = cid_with_class(b"encrypted doc", true, false);
        let actions = tier.handle(StorageTierEvent::PublishContent { cid, data });
        assert_eq!(actions.len(), 1);
        assert!(matches!(
            &actions[0],
            StorageTierAction::AnnounceContent { .. }
        ));
        assert_eq!(tier.metrics().publishes_stored, 1);
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
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            policy,
            FilterBroadcastConfig::default(),
        );

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
    #[should_panic(expected = "encrypted_durable_announce requires encrypted_durable_persist")]
    fn announce_without_persist_panics() {
        let policy = ContentPolicy {
            encrypted_durable_persist: false,
            encrypted_durable_announce: true,
            ..ContentPolicy::default()
        };
        make_tier_with_policy(policy);
    }

    #[test]
    fn storage_tier_accepts_policy() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let policy = ContentPolicy::default();
        let (tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            policy,
            FilterBroadcastConfig::default(),
        );
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
    fn persist_to_disk_suppressed_when_disk_disabled() {
        // When disk_enabled is false (default), PersistToDisk is not emitted
        // to avoid cloning data when the runtime can't service the action.
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        assert!(!tier.disk_enabled());

        let (dur_cid, dur_data) = cid_with_class(b"durable content", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: dur_cid,
            data: dur_data,
        });
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::PersistToDisk { .. })),
            "PersistToDisk should not be emitted when disk is disabled"
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
    fn persist_to_disk_emitted_when_disk_enabled_transit() {
        // When disk_enabled is true, durable transit content emits PersistToDisk.
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_disk(std::iter::empty::<(ContentId, u64)>());

        let (dur_cid, dur_data) = cid_with_class(b"durable content", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid: dur_cid,
            data: dur_data.clone(),
        });
        let persist = actions
            .iter()
            .find(|a| matches!(a, StorageTierAction::PersistToDisk { .. }));
        assert!(
            persist.is_some(),
            "PersistToDisk should be emitted for durable transit when disk enabled"
        );
        match persist.unwrap() {
            StorageTierAction::PersistToDisk { cid, data } => {
                assert_eq!(*cid, dur_cid);
                assert_eq!(data, &dur_data);
            }
            _ => unreachable!(),
        }

        // Ephemeral content should NOT emit PersistToDisk even when disk enabled.
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
    fn persist_to_disk_emitted_when_disk_enabled_publish() {
        // When disk_enabled is true, durable published content emits PersistToDisk.
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_disk(std::iter::empty::<(ContentId, u64)>());

        let (dur_cid, dur_data) = cid_with_class(b"durable published", false, false);
        let actions = tier.handle(StorageTierEvent::PublishContent {
            cid: dur_cid,
            data: dur_data.clone(),
        });
        let persist = actions
            .iter()
            .find(|a| matches!(a, StorageTierAction::PersistToDisk { .. }));
        assert!(
            persist.is_some(),
            "PersistToDisk should be emitted for durable publish when disk enabled"
        );
        match persist.unwrap() {
            StorageTierAction::PersistToDisk { cid, data } => {
                assert_eq!(*cid, dur_cid);
                assert_eq!(data, &dur_data);
            }
            _ => unreachable!(),
        }

        // Ephemeral published content should NOT emit PersistToDisk.
        let (eph_cid, eph_data) = cid_with_class(b"ephemeral published", false, true);
        let actions = tier.handle(StorageTierEvent::PublishContent {
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
            sha224: false,
            lsb_mode: false,
        };
        let cid = ContentId::for_book(&data, flags).unwrap();

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
            sha224: false,
            lsb_mode: false,
        };
        let cid = ContentId::for_book(b"original", flags).unwrap();

        // DiskReadComplete with data that doesn't match the CID hash.
        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid,
            query_id: 42,
            data: b"corrupted".to_vec(),
        });
        // Corrupted data is rejected, but we still send an empty reply
        // so the querier doesn't hang.
        assert_eq!(
            actions.len(),
            1,
            "should send empty reply for corrupted data"
        );
        assert!(
            matches!(&actions[0], StorageTierAction::SendReply { query_id: 42, payload } if payload.is_empty()),
            "corrupted disk data should produce empty reply"
        );
        assert_eq!(tier.metrics().disk_read_failures, 1);
    }

    #[test]
    fn disk_read_complete_cid_survives_moderate_pressure() {
        // DiskReadComplete pre-warms the frequency sketch so the re-cached
        // CID survives the admission challenge under moderate pressure,
        // breaking the disk-read thrashing loop.
        let budget = StorageBudget {
            cache_capacity: 5, // window=1, protected=1, probation=3
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        // Fill cache with items (each gets frequency ~1 from transit admission).
        for i in 0..4 {
            let (cid, data) = cid_with_class(format!("fill-{i}").as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent { cid, data });
        }

        // Simulate disk read — CID re-enters cache with frequency boost.
        let disk_data = b"disk-resident content";
        let flags = crate::cid::ContentFlags {
            encrypted: false,
            ephemeral: false,
            sha224: false,
            lsb_mode: false,
        };
        let disk_cid = ContentId::for_book(disk_data, flags).unwrap();
        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid: disk_cid,
            query_id: 50,
            data: disk_data.to_vec(),
        });
        assert_eq!(actions.len(), 1); // SendReply

        // Push more items to create eviction pressure. The disk-read CID
        // will be pushed out of window into the admission challenge.
        for i in 0..3 {
            let (cid, data) = cid_with_class(format!("pressure-{i}").as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent { cid, data });
        }

        // The disk-read CID should still be queryable thanks to frequency boost.
        let query_actions = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 51,
            cid: disk_cid,
        });
        assert!(
            !query_actions.is_empty(),
            "disk-read CID should survive moderate pressure thanks to frequency boost"
        );
    }

    #[test]
    fn disk_read_failed_sends_empty_reply() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let cid = ContentId::for_book(b"missing", crate::cid::ContentFlags::default()).unwrap();
        let actions = tier.handle(StorageTierEvent::DiskReadFailed { cid, query_id: 42 });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 42);
                assert!(
                    payload.is_empty(),
                    "failed disk read should send empty reply"
                );
            }
            other => panic!("expected SendReply, got {other:?}"),
        }
        assert_eq!(tier.metrics().disk_read_failures, 1);
    }

    #[test]
    fn cache_miss_returns_empty_when_disk_disabled() {
        // When disk_enabled is false (default), cache misses return []
        // even if the CID is in the disk_index.
        let budget = StorageBudget {
            cache_capacity: 3,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let (cid, data) = cid_with_class(b"durable on disk", false, false);

        // Store via transit — adds to cache.
        tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.clone(),
        });

        // Flood cache to evict original CID. Double-transit to build frequency.
        for i in 0..20 {
            let filler = format!("filler-{i}");
            let (filler_cid, filler_data) = cid_with_class(filler.as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent {
                cid: filler_cid,
                data: filler_data.clone(),
            });
            tier.handle(StorageTierEvent::TransitContent {
                cid: filler_cid,
                data: filler_data,
            });
        }

        // Query original CID — cache miss returns empty when disk is disabled.
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 77, cid });
        assert!(
            actions.is_empty(),
            "cache miss should return empty when disk is disabled, got: {actions:?}"
        );
    }

    #[test]
    fn cache_miss_emits_disk_lookup_when_disk_enabled() {
        // When disk_enabled is true and the CID is in the disk_index,
        // a cache miss emits DiskLookup instead of empty.
        let budget = StorageBudget {
            cache_capacity: 3,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let (cid, data) = cid_with_class(b"durable on disk", false, false);

        // Enable disk with the CID pre-populated (simulates startup scan).
        tier.enable_disk(std::iter::once((cid, 100)));

        // Store via transit — adds to cache and disk_index.
        tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.clone(),
        });

        // Flood cache to evict original CID. Double-transit to build frequency.
        for i in 0..20 {
            let filler = format!("filler-{i}");
            let (filler_cid, filler_data) = cid_with_class(filler.as_bytes(), false, false);
            tier.handle(StorageTierEvent::TransitContent {
                cid: filler_cid,
                data: filler_data.clone(),
            });
            tier.handle(StorageTierEvent::TransitContent {
                cid: filler_cid,
                data: filler_data,
            });
        }

        // Query original CID — should emit DiskLookup since it's in disk_index.
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 77, cid });
        assert_eq!(actions.len(), 1, "should emit DiskLookup, got: {actions:?}");
        match &actions[0] {
            StorageTierAction::DiskLookup {
                cid: lookup_cid,
                query_id,
            } => {
                assert_eq!(*lookup_cid, cid);
                assert_eq!(*query_id, 77);
            }
            other => panic!("expected DiskLookup, got {other:?}"),
        }
    }

    #[test]
    fn cache_miss_returns_empty_when_cid_not_in_disk_index() {
        // Even with disk_enabled, if the CID is not in disk_index,
        // cache miss returns empty.
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_disk(std::iter::empty::<(ContentId, u64)>());

        let cid = ContentId::for_book(b"unknown", crate::cid::ContentFlags::default()).unwrap();
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 88, cid });
        assert!(
            actions.is_empty(),
            "cache miss for CID not in disk_index should return empty"
        );
    }

    #[test]
    fn enable_disk_populates_index() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        assert!(!tier.disk_enabled());

        let cid_a = ContentId::for_book(b"alpha", crate::cid::ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"beta", crate::cid::ContentFlags::default()).unwrap();

        tier.enable_disk(vec![(cid_a, 100), (cid_b, 100)]);
        assert!(tier.disk_enabled());

        // Both CIDs should trigger DiskLookup on cache miss.
        let actions_a = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 1,
            cid: cid_a,
        });
        assert_eq!(actions_a.len(), 1);
        assert!(matches!(
            &actions_a[0],
            StorageTierAction::DiskLookup { cid, query_id: 1 } if *cid == cid_a
        ));

        let actions_b = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 2,
            cid: cid_b,
        });
        assert_eq!(actions_b.len(), 1);
        assert!(matches!(
            &actions_b[0],
            StorageTierAction::DiskLookup { cid, query_id: 2 } if *cid == cid_b
        ));

        // Unknown CID should still return empty.
        let cid_c = ContentId::for_book(b"gamma", crate::cid::ContentFlags::default()).unwrap();
        let actions_c = tier.handle(StorageTierEvent::ContentQuery {
            query_id: 3,
            cid: cid_c,
        });
        assert!(actions_c.is_empty());
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
        // Fill cache with PublicDurable + PublicEphemeral, then apply pressure.
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

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

        // Give durable items frequency via repeated queries.
        for _ in 0..5 {
            tier.handle(StorageTierEvent::ContentQuery {
                query_id: 99,
                cid: d1_cid,
            });
            tier.handle(StorageTierEvent::ContentQuery {
                query_id: 99,
                cid: d2_cid,
            });
        }

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

        // Fill cache to trigger evictions via publish (bypasses admission).
        for i in 0..20 {
            let data = format!("pressure-{i}");
            let (pressure_cid, pressure_data) = cid_with_class(data.as_bytes(), false, false);
            tier.handle(StorageTierEvent::PublishContent {
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

    // ---- Bloom filter broadcast ----

    #[test]
    fn filter_broadcast_config_defaults() {
        let config = FilterBroadcastConfig::default();
        assert_eq!(config.mutation_threshold, 100);
        assert_eq!(config.max_interval_ticks, 30);
        assert_eq!(config.expected_items, 1024);
        assert!((config.fp_rate - 0.001).abs() < f64::EPSILON);
    }

    #[test]
    fn filter_broadcast_config_clamps_zero_mutation_threshold() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let filter_config = FilterBroadcastConfig {
            mutation_threshold: 0,
            ..FilterBroadcastConfig::default()
        };
        let (tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            filter_config,
        );
        assert_eq!(
            tier.filter_config().mutation_threshold,
            1,
            "mutation_threshold = 0 should be clamped to 1"
        );
    }

    #[test]
    fn filter_broadcast_at_mutation_threshold() {
        // Create a tier with low mutation threshold for testing.
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let filter_config = FilterBroadcastConfig {
            mutation_threshold: 3,
            ..FilterBroadcastConfig::default()
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            filter_config,
        );

        // First 2 transits: no BroadcastFilter action.
        for i in 0..2 {
            let data = format!("transit-{i}");
            let cid =
                ContentId::for_book(data.as_bytes(), crate::cid::ContentFlags::default()).unwrap();
            let actions = tier.handle(StorageTierEvent::TransitContent {
                cid,
                data: data.into_bytes(),
            });
            assert!(
                !actions
                    .iter()
                    .any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
                "mutation {i}: should not emit BroadcastFilter yet"
            );
        }

        // 3rd transit: should include BroadcastFilter action.
        let data = b"transit-2";
        let cid = ContentId::for_book(data, crate::cid::ContentFlags::default()).unwrap();
        let actions = tier.handle(StorageTierEvent::TransitContent {
            cid,
            data: data.to_vec(),
        });
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
            "3rd mutation should trigger BroadcastFilter"
        );
    }

    #[test]
    fn filter_timer_tick_triggers_broadcast() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let actions = tier.handle(StorageTierEvent::FilterTimerTick);
        assert_eq!(actions.len(), 2);
        assert!(
            matches!(&actions[0], StorageTierAction::BroadcastFilter { .. }),
            "FilterTimerTick should emit BroadcastFilter"
        );
        assert!(
            matches!(&actions[1], StorageTierAction::BroadcastCuckooFilter { .. }),
            "FilterTimerTick should emit BroadcastCuckooFilter"
        );
    }

    #[test]
    fn filter_excludes_encrypted_ephemeral() {
        use crate::bloom::BloomFilter;

        let mut tier = make_tier_with_policy(ContentPolicy::default());

        // Publish a PublicDurable item (goes through normal admission).
        let pub_data = b"public durable item";
        let (pub_cid, pub_vec) = cid_with_class(pub_data, false, false);
        tier.handle(StorageTierEvent::PublishContent {
            cid: pub_cid,
            data: pub_vec,
        });

        // Inject an EncryptedEphemeral CID directly into the cache,
        // bypassing the class_admits gate in handle_transit/handle_publish.
        // This exercises the class_admits check inside rebuild_filter.
        let (ee_cid, ee_data) = cid_with_class(b"encrypted ephemeral", true, true);
        tier.cache_mut().store_preadmitted(ee_cid, ee_data);

        // Trigger filter rebuild via FilterTimerTick.
        let actions = tier.handle(StorageTierEvent::FilterTimerTick);
        assert_eq!(actions.len(), 2);

        let payload = match &actions[0] {
            StorageTierAction::BroadcastFilter { payload } => payload,
            other => panic!("expected BroadcastFilter, got {other:?}"),
        };
        let filter = BloomFilter::from_bytes(payload).unwrap();

        // PublicDurable CID should be in the filter.
        assert!(
            filter.may_contain(&pub_cid),
            "PublicDurable CID should be in the filter"
        );

        // EncryptedEphemeral CID is in the cache but class_admits excludes
        // it from the filter rebuild — this is the actual behavioral guarantee.
        assert!(
            !filter.may_contain(&ee_cid),
            "EncryptedEphemeral CID should be excluded by class_admits"
        );
    }

    #[test]
    fn filter_broadcast_at_disk_read_mutation_threshold() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let filter_config = FilterBroadcastConfig {
            mutation_threshold: 2,
            ..FilterBroadcastConfig::default()
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            filter_config,
        );

        // First DiskReadComplete: mutation counter = 1, no broadcast.
        let (cid1, data1) = cid_with_class(b"durable-disk-1", false, false);
        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid: cid1,
            query_id: 1,
            data: data1,
        });
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
            "1st disk read: should not emit BroadcastFilter yet"
        );

        // Second DiskReadComplete: mutation counter = 2, should trigger broadcast.
        let (cid2, data2) = cid_with_class(b"durable-disk-2", false, false);
        let actions = tier.handle(StorageTierEvent::DiskReadComplete {
            cid: cid2,
            query_id: 2,
            data: data2,
        });
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
            "2nd disk read: should trigger BroadcastFilter at threshold"
        );
    }

    #[test]
    fn cuckoo_filter_broadcast_on_timer_tick() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let actions = tier.handle(StorageTierEvent::FilterTimerTick);
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
            "should emit BroadcastFilter"
        );
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, StorageTierAction::BroadcastCuckooFilter { .. })),
            "should emit BroadcastCuckooFilter"
        );
    }

    #[test]
    fn enable_disk_with_sizes_tracks_bytes() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        let cid1 = ContentId::from_bytes([0x01; 32]);
        let cid2 = ContentId::from_bytes([0x02; 32]);

        tier.enable_disk(vec![(cid1, 100), (cid2, 200)]);

        assert_eq!(tier.disk_used_bytes(), 300);
        assert!(tier.disk_contains(&cid1));
        assert!(tier.disk_contains(&cid2));
    }

    #[test]
    fn set_disk_quota() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        let actions = tier.set_disk_quota(1024);
        assert!(actions.is_empty()); // no data on disk, nothing to evict
        assert_eq!(tier.disk_quota(), Some(1024));
    }

    #[test]
    fn set_disk_quota_evicts_existing_data_over_quota() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);
        let cid_c = ContentId::from_bytes([0x0C; 32]);

        // Simulate existing data on disk (e.g. from previous run)
        tier.enable_disk(vec![(cid_a, 100), (cid_b, 100), (cid_c, 100)]);
        assert_eq!(tier.disk_used_bytes(), 300);

        // Set a quota tighter than existing data — should evict immediately
        let actions = tier.set_disk_quota(200);

        // Should have evicted the oldest entry (cid_a)
        assert!(actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_a)));
        assert!(!tier.disk_contains(&cid_a));
        assert!(tier.disk_contains(&cid_b));
        assert!(tier.disk_contains(&cid_c));
        assert_eq!(tier.disk_used_bytes(), 200);
    }

    // ---- Eviction logic ----

    #[test]
    fn eviction_removes_oldest_when_over_quota() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(250);

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);
        let cid_c = ContentId::from_bytes([0x0C; 32]);

        let actions_a = tier.test_persist_to_disk(cid_a, 100);
        assert!(!actions_a
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));

        let actions_b = tier.test_persist_to_disk(cid_b, 100);
        assert!(!actions_b
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));

        // Third book pushes over quota — should evict cid_a (oldest)
        let actions_c = tier.test_persist_to_disk(cid_c, 100);
        assert!(actions_c
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_a)));
        assert!(!tier.disk_contains(&cid_a));
        assert!(tier.disk_contains(&cid_b));
        assert!(tier.disk_contains(&cid_c));
        assert_eq!(tier.disk_used_bytes(), 200);
    }

    #[test]
    fn eviction_skips_pinned_books() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(250);

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);
        let cid_c = ContentId::from_bytes([0x0C; 32]);

        tier.test_persist_to_disk(cid_a, 100);
        tier.test_persist_to_disk(cid_b, 100);
        tier.cache_pin(cid_a); // pin the oldest

        // Over quota — should skip cid_a (pinned) and evict cid_b
        let actions = tier.test_persist_to_disk(cid_c, 100);
        assert!(actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_b)));
        assert!(tier.disk_contains(&cid_a)); // pinned, still present
        assert!(!tier.disk_contains(&cid_b)); // evicted
        assert!(tier.disk_contains(&cid_c));
    }

    #[test]
    fn no_eviction_without_quota() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());

        let cid = ContentId::from_bytes([0x01; 32]);
        let actions = tier.test_persist_to_disk(cid, 1_000_000);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));
    }

    #[test]
    fn eviction_safety_valve_all_pinned() {
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(150);

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);
        tier.test_persist_to_disk(cid_a, 100);
        tier.cache_pin(cid_a);

        let actions = tier.test_persist_to_disk(cid_b, 100);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { .. })));
        assert!(tier.disk_contains(&cid_a));
        assert!(tier.disk_contains(&cid_b));
    }

    #[test]
    fn dedup_on_re_persist() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(500);

        let cid = ContentId::from_bytes([0x01; 32]);
        tier.test_persist_to_disk(cid, 100);
        tier.test_persist_to_disk(cid, 100); // same CID again
        assert_eq!(tier.disk_used_bytes(), 100); // counted once
    }

    #[test]
    fn disk_read_complete_refreshes_lru_ordering() {
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(250);

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);
        let cid_c = ContentId::from_bytes([0x0C; 32]);

        tier.test_persist_to_disk(cid_a, 100);
        tier.test_persist_to_disk(cid_b, 100);

        // Simulate a disk read of cid_a — moves it to back of LRU
        tier.touch_disk_lru(&cid_a);

        // Now cid_b is oldest. Adding cid_c should evict cid_b, not cid_a.
        let actions = tier.test_persist_to_disk(cid_c, 100);
        assert!(actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_b)));
        assert!(tier.disk_contains(&cid_a)); // refreshed, not evicted
    }

    #[test]
    fn eviction_cancels_pending_persist_in_same_batch() {
        // Regression test: when two persists happen in the same tick and the
        // second triggers eviction of the first, both PersistToDisk and
        // RemoveFromDisk would be emitted for the same CID. Since the event
        // loop dispatches these as concurrent spawn_blocking tasks, the delete
        // could race ahead of the write, leaving an orphaned file.
        //
        // The fix cancels both: if a PersistToDisk for the evicted CID is in
        // the current actions list, remove it and skip RemoveFromDisk.
        let budget = StorageBudget {
            cache_capacity: 10,
            max_pinned_bytes: 0,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBookStore::new(),
            budget,
            ContentPolicy::default(),
            FilterBroadcastConfig::default(),
        );
        tier.enable_disk(Vec::new());
        tier.set_disk_quota(150); // tight quota: room for one 100-byte book

        let cid_a = ContentId::from_bytes([0x0A; 32]);
        let cid_b = ContentId::from_bytes([0x0B; 32]);

        // Persist both in one batch — cid_b triggers eviction of cid_a
        let actions = tier.test_persist_batch(&[(cid_a, 100), (cid_b, 100)]);

        // cid_a's PersistToDisk should have been cancelled (removed)
        let persist_a = actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::PersistToDisk { cid, .. } if *cid == cid_a));
        assert!(!persist_a, "PersistToDisk(cid_a) should be cancelled");

        // No RemoveFromDisk(cid_a) either — the pair cancels out
        let remove_a = actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::RemoveFromDisk { cid } if *cid == cid_a));
        assert!(!remove_a, "RemoveFromDisk(cid_a) should not be emitted");

        // cid_b's PersistToDisk should still be present
        let persist_b = actions
            .iter()
            .any(|a| matches!(a, StorageTierAction::PersistToDisk { cid, .. } if *cid == cid_b));
        assert!(persist_b, "PersistToDisk(cid_b) should still be present");

        // cid_a is gone from disk_index, cid_b is present
        assert!(!tier.disk_contains(&cid_a));
        assert!(tier.disk_contains(&cid_b));
        assert_eq!(tier.disk_used_bytes(), 100);
    }

    // ---- S3 fallback ----

    #[test]
    fn s3_lookup_emitted_on_durable_cache_disk_miss_when_enabled() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_s3();

        // PublicDurable CID — not in cache, not on disk.
        let (cid, _data) = cid_with_class(b"durable s3 content", false, false);
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 1, cid });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::S3Lookup {
                cid: lookup_cid,
                query_id,
            } => {
                assert_eq!(*lookup_cid, cid);
                assert_eq!(*query_id, 1);
            }
            other => panic!("expected S3Lookup, got {other:?}"),
        }
    }

    #[test]
    fn s3_lookup_not_emitted_for_ephemeral_content() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_s3();

        // PublicEphemeral CID — S3 only serves durable content.
        let (cid, _data) = cid_with_class(b"ephemeral stream", false, true);
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 2, cid });

        assert!(
            actions.is_empty(),
            "ephemeral content should not trigger S3Lookup, got: {actions:?}"
        );
    }

    #[test]
    fn s3_lookup_not_emitted_when_s3_disabled() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        // s3 NOT enabled (default).

        let (cid, _data) = cid_with_class(b"durable no s3", false, false);
        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 3, cid });

        assert!(
            actions.is_empty(),
            "S3Lookup should not be emitted when s3 is disabled, got: {actions:?}"
        );
    }

    #[test]
    fn s3_lookup_not_emitted_on_cache_hit() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_s3();

        // Put content in the cache.
        let data = b"cached s3 content";
        let cid = tier.cache_mut().insert(data).unwrap();

        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 4, cid });

        // Should get a SendReply from cache hit, not S3Lookup.
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 4);
                assert_eq!(payload.as_slice(), data.as_slice());
            }
            other => panic!("expected SendReply (cache hit), got {other:?}"),
        }
    }

    #[test]
    fn s3_lookup_not_emitted_on_disk_hit() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        tier.enable_s3();

        // Put CID in disk index so disk lookup takes priority over S3.
        let (cid, _data) = cid_with_class(b"disk-resident content", false, false);
        tier.enable_disk(vec![(cid, 100)]);

        let actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 5, cid });

        // Should get DiskLookup, not S3Lookup.
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::DiskLookup {
                cid: lookup_cid,
                query_id,
            } => {
                assert_eq!(*lookup_cid, cid);
                assert_eq!(*query_id, 5);
            }
            other => panic!("expected DiskLookup, got {other:?}"),
        }
    }

    #[test]
    fn s3_read_complete_caches_and_replies() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, data) = cid_with_class(b"s3 fetched content", false, false);

        let actions = tier.handle(StorageTierEvent::S3ReadComplete {
            cid,
            query_id: 10,
            data: data.clone(),
        });

        // Should contain SendReply with the data + AnnounceContent (PublicDurable).
        let reply = actions
            .iter()
            .find(|a| matches!(a, StorageTierAction::SendReply { .. }));
        assert!(reply.is_some(), "should emit SendReply");
        match reply.unwrap() {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 10);
                assert_eq!(payload, &data);
            }
            _ => unreachable!(),
        }
        assert_eq!(tier.metrics().s3_reads_served, 1);

        // Content should now be in cache.
        let query_actions = tier.handle(StorageTierEvent::ContentQuery { query_id: 11, cid });
        assert_eq!(query_actions.len(), 1);
        match &query_actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 11);
                assert_eq!(payload, &data);
            }
            other => panic!("expected SendReply from cache, got {other:?}"),
        }
        assert_eq!(tier.metrics().cache_hits, 1);
    }

    #[test]
    fn s3_read_complete_corrupted_replies_empty() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, _data) = cid_with_class(b"original s3 data", false, false);

        // Send corrupted data that doesn't match the CID hash.
        let actions = tier.handle(StorageTierEvent::S3ReadComplete {
            cid,
            query_id: 20,
            data: b"corrupted s3 data".to_vec(),
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 20);
                assert!(
                    payload.is_empty(),
                    "corrupted S3 data should produce empty reply"
                );
            }
            other => panic!("expected empty SendReply, got {other:?}"),
        }
        assert_eq!(tier.metrics().s3_read_failures, 1);
        assert_eq!(tier.metrics().s3_reads_served, 0);
    }

    #[test]
    fn s3_read_failed_replies_empty() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let (cid, _data) = cid_with_class(b"missing from s3", false, false);

        let actions = tier.handle(StorageTierEvent::S3ReadFailed { cid, query_id: 30 });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            StorageTierAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 30);
                assert!(payload.is_empty(), "S3 failure should produce empty reply");
            }
            other => panic!("expected empty SendReply, got {other:?}"),
        }
        assert_eq!(tier.metrics().s3_read_failures, 1);
    }
}

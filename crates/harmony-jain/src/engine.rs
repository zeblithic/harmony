//! Sans-I/O content lifecycle engine.

use alloc::vec::Vec;
use harmony_content::ContentId;
use harmony_roxy::catalog::ContentCategory;
use hashbrown::{HashMap, HashSet};

use crate::actions::*;
use crate::config::{FilterRuleSet, JainConfig};
use crate::error::JainError;
use crate::types::{ContentOrigin, ContentRecord, Sensitivity, SocialContext};

/// Sans-I/O content lifecycle engine.
///
/// `JainEngine` tracks content records on a node and provides pure-function
/// methods for ingestion, filtering, tick-based housekeeping, and reconciliation.
/// It emits [`JainAction`] variants for the caller to execute — it never performs
/// I/O itself.
pub struct JainEngine {
    records: HashMap<ContentId, ContentRecord>,
    pub(crate) config: JainConfig,
    filter_rules: FilterRuleSet,
    total_storage_bytes: u64,
    storage_capacity_bytes: u64,
}

impl JainEngine {
    /// Create a new engine with no tracked records.
    ///
    /// Returns an error if the configuration is invalid (e.g. zero half-life,
    /// inverted thresholds). See [`JainConfig::validate`] for the full list of
    /// checked invariants.
    ///
    /// # Storage capacity
    ///
    /// Pass `storage_capacity_bytes = 0` to disable storage-budget enforcement.
    /// When `0` is used, [`evaluate_ingest`](Self::evaluate_ingest) will never
    /// reject for `StorageBudgetExceeded` and [`tick`](Self::tick) will never
    /// emit `StorageNearFull` alerts.
    pub fn new(
        config: JainConfig,
        filter_rules: FilterRuleSet,
        storage_capacity_bytes: u64,
    ) -> Result<Self, JainError> {
        config.validate()?;
        Ok(Self {
            records: HashMap::new(),
            config,
            filter_rules,
            total_storage_bytes: 0,
            storage_capacity_bytes,
        })
    }

    /// Number of content records currently tracked.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Generate a health report summarizing the engine's current state.
    ///
    /// `now` is the current wall-clock timestamp, used to compute staleness
    /// scores for the `stale_count` field.
    pub fn health_report(&self, now: f64) -> HealthReport {
        let mut under_replicated_count = 0u32;
        let mut pinned_count = 0u32;
        let mut stale_count = 0u32;

        for record in self.records.values() {
            if record.replica_count < self.config.min_replica_count {
                under_replicated_count += 1;
            }
            if record.pinned {
                pinned_count += 1;
            }
            let score = crate::scoring::staleness_score(record, &self.config, now);
            if score.value() >= self.config.archive_threshold {
                stale_count += 1;
            }
        }

        let storage_used_percent_x100 = if self.storage_capacity_bytes > 0 {
            ((self.total_storage_bytes as f64 / self.storage_capacity_bytes as f64) * 10000.0)
                as u32
        } else {
            0
        };

        HealthReport {
            total_records: self.records.len() as u32,
            total_bytes: self.total_storage_bytes,
            storage_used_percent_x100,
            under_replicated_count,
            stale_count,
            pinned_count,
        }
    }

    /// Process a content event, updating internal state and returning any actions
    /// for the caller to execute.
    ///
    /// # Storage budget
    ///
    /// For [`ContentEvent::Stored`], the caller **must** first obtain
    /// [`IngestDecision::IndexAndStore`] or [`IngestDecision::StoreOnly`] from
    /// [`evaluate_ingest`](Self::evaluate_ingest) before dispatching this event.
    /// Skipping `evaluate_ingest` may cause `total_storage_bytes` to exceed
    /// `storage_capacity_bytes`, corrupting storage-usage accounting.
    pub fn handle_event(&mut self, event: ContentEvent) -> Vec<JainAction> {
        match event {
            ContentEvent::Stored {
                cid,
                size_bytes,
                content_type,
                origin,
                sensitivity,
                timestamp,
            } => {
                // Same-CID re-arrival: clear pending_local_repair if set
                // (local copy restored after reconcile detected it missing).
                // Otherwise ignore — cross-CID dedup is handled by Oluo.
                if let Some(existing) = self.records.get_mut(&cid) {
                    existing.pending_local_repair = false;
                    return Vec::new();
                }
                let record = ContentRecord {
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
                    pending_local_repair: false,
                };
                self.records.insert(cid, record);
                self.total_storage_bytes = self.total_storage_bytes.saturating_add(size_bytes);
                Vec::new()
            }
            ContentEvent::Accessed { cid, timestamp } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.last_accessed = record.last_accessed.max(timestamp);
                    record.access_count += 1;
                }
                Vec::new()
            }
            ContentEvent::Deleted { cid } => {
                if let Some(record) = self.records.remove(&cid) {
                    self.total_storage_bytes =
                        self.total_storage_bytes.saturating_sub(record.size_bytes);
                }
                Vec::new()
            }
            ContentEvent::Pinned { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.pinned = true;
                }
                Vec::new()
            }
            ContentEvent::Unpinned { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.pinned = false;
                }
                Vec::new()
            }
            ContentEvent::LicenseGranted { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.licensed = true;
                }
                Vec::new()
            }
            ContentEvent::LicenseExpired { cid } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.licensed = false;
                }
                Vec::new()
            }
            ContentEvent::ReplicaChanged { cid, new_count } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.replica_count = new_count;
                }
                Vec::new()
            }
        }
    }

    /// Evaluate whether a content candidate should be ingested.
    pub fn evaluate_ingest(&self, candidate: &IngestCandidate) -> IngestDecision {
        // 1. Duplicate check
        if self.records.contains_key(&candidate.cid) {
            return IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            };
        }

        // 2. Storage budget check (capacity of 0 means unlimited)
        if self.storage_capacity_bytes > 0
            && self.total_storage_bytes.saturating_add(candidate.size_bytes)
                > self.storage_capacity_bytes
        {
            return IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            };
        }

        // 3. Sensitivity check — intimate or above → store only (don't index in Oluo)
        if candidate.sensitivity >= Sensitivity::Intimate {
            return IngestDecision::StoreOnly;
        }

        // 4. Normal content
        IngestDecision::IndexAndStore
    }

    /// Evaluate whether content should be shared in a given social context.
    ///
    /// Iterates all filter rules and returns the most restrictive matching
    /// decision: Block > Confirm > Allow.
    pub fn filter_result(&self, cid: &ContentId, context: SocialContext) -> FilterDecision {
        // Disabled filter → allow everything
        if !self.filter_rules.enabled {
            return FilterDecision::Allow;
        }

        // Unknown CID → allow (safe default)
        let record = match self.records.get(cid) {
            Some(r) => r,
            None => return FilterDecision::Allow,
        };

        // Evaluate all rules and return the most restrictive decision.
        // Block > Confirm > Allow.
        let mut decision = FilterDecision::Allow;

        for rule in &self.filter_rules.rules {
            if record.sensitivity >= rule.min_sensitivity {
                if context > rule.max_context {
                    // Context is wider than allowed → Block
                    return FilterDecision::Block;
                }
                if context == rule.max_context && rule.require_confirmation {
                    // At the boundary with confirmation required → Confirm (upgradeable to Block)
                    decision = FilterDecision::Confirm;
                }
            }
        }

        decision
    }

    /// Advance the engine, computing staleness scores and emitting housekeeping
    /// actions.
    ///
    /// Call periodically with the current wall-clock timestamp.
    ///
    /// # Idempotency
    ///
    /// `tick` re-evaluates every record on each call and may emit the same
    /// `RepairNeeded` or `RecommendBurn` action across successive ticks until
    /// the underlying condition is resolved. Callers are responsible for
    /// deduplicating or rate-limiting these actions in their dispatch loop.
    pub fn tick(&mut self, now: f64) -> Vec<JainAction> {
        let mut actions = Vec::new();

        // Storage health check
        if self.storage_capacity_bytes > 0 {
            let used_pct = self.total_storage_bytes as f64 / self.storage_capacity_bytes as f64;
            if used_pct >= self.config.storage_alert_percent {
                actions.push(JainAction::HealthAlert {
                    alert: HealthAlertKind::StorageNearFull {
                        used_percent_x100: (used_pct * 10000.0) as u32,
                    },
                });
            }
        }

        // Per-record scoring
        for record in self.records.values() {
            // Under-replicated check
            if record.replica_count < self.config.min_replica_count {
                actions.push(JainAction::RepairNeeded {
                    cid: record.cid,
                    current_replicas: record.replica_count,
                    desired: self.config.min_replica_count,
                });
            }

            // Pinned or licensed content is exempt from cleanup — enforce
            // directly here rather than relying solely on staleness_score
            // returning FRESH.
            if record.pinned || record.licensed {
                continue;
            }

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

    /// Reconcile internal records against a disk snapshot.
    ///
    /// `now` is the current wall-clock timestamp, used to initialise `stored_at`
    /// and `last_accessed` for newly discovered records so they don't immediately
    /// appear maximally stale.
    ///
    /// # Snapshot completeness requirement
    ///
    /// `snapshot` **must** enumerate **every** piece of content currently on disk.
    /// Any tracked record absent from `snapshot` is treated as an orphan and
    /// permanently removed from the engine. Passing a partial or incremental
    /// snapshot will silently delete the corresponding records with no recovery
    /// path — ensure the snapshot is authoritative before calling `reconcile`.
    ///
    /// Handles four cases:
    /// 1. Snapshot entry exists on disk but is not tracked → add a default record
    /// 2. Tracked record has `exists_on_disk: false` → emit `FetchLocalCopy`,
    ///    set `pending_local_repair`
    /// 3. Tracked record exists on disk → clear `pending_local_repair`
    /// 4. Tracked record is not in the snapshot at all → orphan, remove it
    pub fn reconcile(&mut self, snapshot: &[SnapshotEntry], now: f64) -> Vec<JainAction> {
        let mut actions = Vec::new();
        let mut seen: HashSet<ContentId> = HashSet::new();

        for entry in snapshot {
            // Deduplicate: skip CIDs already processed in this snapshot pass.
            if !seen.insert(entry.cid) {
                continue;
            }

            if !entry.exists_on_disk {
                // Case 2: tracked CID whose backing data is missing.
                // Mark as pending repair so tick() won't recommend burn/archive
                // while the fetch is in progress.
                if let Some(record) = self.records.get_mut(&entry.cid) {
                    record.pending_local_repair = true;
                    actions.push(JainAction::FetchLocalCopy { cid: entry.cid });
                }
            } else if let Some(record) = self.records.get_mut(&entry.cid) {
                // Tracked and on disk: clear any in-progress repair flag.
                record.pending_local_repair = false;
            } else {
                // Case 1: on disk but not tracked → add default record.
                // Use Confidential as the safe default when the original
                // sensitivity is unknown — prevents accidental leakage of
                // content that may have been Intimate or Confidential before
                // a crash-recovery cycle.
                let record = ContentRecord {
                    cid: entry.cid,
                    size_bytes: entry.size_bytes,
                    content_type: ContentCategory::Bundle,
                    origin: ContentOrigin::CachedInTransit,
                    sensitivity: Sensitivity::Confidential,
                    stored_at: now,
                    last_accessed: now,
                    access_count: 0,
                    replica_count: 1,
                    pinned: false,
                    licensed: false,
                    pending_local_repair: false,
                };
                self.total_storage_bytes = self.total_storage_bytes.saturating_add(entry.size_bytes);
                self.records.insert(entry.cid, record);
            }
        }

        // Case 3: orphaned records — in engine but not in snapshot
        let orphaned: Vec<ContentId> = self
            .records
            .keys()
            .copied()
            .filter(|cid| !seen.contains(cid))
            .collect();

        if !orphaned.is_empty() {
            let orphan_count = orphaned.len() as u32;
            for cid in orphaned {
                if let Some(record) = self.records.remove(&cid) {
                    self.total_storage_bytes =
                        self.total_storage_bytes.saturating_sub(record.size_bytes);
                }
            }
            actions.push(JainAction::HealthAlert {
                alert: HealthAlertKind::StaleReconciliation {
                    records_without_backing: orphan_count,
                },
            });
        }

        actions
    }
}

impl Default for JainEngine {
    fn default() -> Self {
        Self::new(JainConfig::default(), FilterRuleSet::default(), 0)
            .expect("default JainConfig is always valid")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JainConfig;
    use crate::types::SocialContext;
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    fn default_engine() -> JainEngine {
        JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 10_000).unwrap()
    }

    // ── Task 8: Constructor, record_count, health_report ──

    #[test]
    fn new_engine_is_empty() {
        let engine = default_engine();
        assert_eq!(engine.record_count(), 0);
    }

    #[test]
    fn health_report_empty_engine() {
        let engine = default_engine();
        let report = engine.health_report(0.0);
        assert_eq!(report.total_records, 0);
        assert_eq!(report.total_bytes, 0);
        assert_eq!(report.storage_used_percent_x100, 0);
    }

    // ── Task 9: handle_event and evaluate_ingest ──

    #[test]
    fn handle_stored_event_creates_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-a");
        let actions = engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        assert!(actions.is_empty());
        assert_eq!(engine.record_count(), 1);
        let report = engine.health_report(0.0);
        assert_eq!(report.total_bytes, 100);
    }

    #[test]
    fn handle_stored_duplicate_is_ignored() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-dup");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 50,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        let actions = engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 50,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 2000.0,
        });
        // Same-CID re-arrival is silently ignored; cross-CID dedup is Oluo's job
        assert!(actions.is_empty());
        assert_eq!(engine.record_count(), 1);
        // Storage should not double-count
        assert_eq!(engine.health_report(0.0).total_bytes, 50);
    }

    #[test]
    fn handle_accessed_updates_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-acc");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        let actions = engine.handle_event(ContentEvent::Accessed {
            cid,
            timestamp: 2000.0,
        });
        assert!(actions.is_empty());
        assert_eq!(engine.record_count(), 1);
    }

    #[test]
    fn handle_deleted_removes_record() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-del");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 200,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::Deleted { cid });
        assert_eq!(engine.record_count(), 0);
        assert_eq!(engine.health_report(0.0).total_bytes, 0);
    }

    #[test]
    fn handle_pinned_and_unpinned() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-pin");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::Pinned { cid });
        assert_eq!(engine.health_report(0.0).pinned_count, 1);
        engine.handle_event(ContentEvent::Unpinned { cid });
        assert_eq!(engine.health_report(0.0).pinned_count, 0);
    }

    #[test]
    fn handle_license_granted_and_expired() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-lic");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::LicenseGranted { cid });
        assert_eq!(engine.record_count(), 1);
        engine.handle_event(ContentEvent::LicenseExpired { cid });
        assert_eq!(engine.record_count(), 1);
    }

    #[test]
    fn handle_replica_changed() {
        let mut engine = default_engine();
        let cid = make_cid(b"content-rep");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::PeerReplicated,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        assert_eq!(engine.health_report(0.0).under_replicated_count, 1);
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });
        assert_eq!(engine.health_report(0.0).under_replicated_count, 0);
    }

    #[test]
    fn evaluate_ingest_allows_normal_content() {
        let engine = default_engine();
        let candidate = IngestCandidate {
            cid: make_cid(b"new-content"),
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
        };
        assert_eq!(
            engine.evaluate_ingest(&candidate),
            IngestDecision::IndexAndStore
        );
    }

    #[test]
    fn evaluate_ingest_rejects_duplicate() {
        let mut engine = default_engine();
        let cid = make_cid(b"existing");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 50,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        let candidate = IngestCandidate {
            cid,
            size_bytes: 50,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
        };
        assert_eq!(
            engine.evaluate_ingest(&candidate),
            IngestDecision::Reject {
                reason: RejectReason::DuplicateContent,
            }
        );
    }

    #[test]
    fn evaluate_ingest_rejects_over_budget() {
        let engine = JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 50).unwrap();
        let candidate = IngestCandidate {
            cid: make_cid(b"big-content"),
            size_bytes: 100,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
        };
        assert_eq!(
            engine.evaluate_ingest(&candidate),
            IngestDecision::Reject {
                reason: RejectReason::StorageBudgetExceeded,
            }
        );
    }

    #[test]
    fn evaluate_ingest_stores_only_for_confidential() {
        let engine = default_engine();
        let candidate = IngestCandidate {
            cid: make_cid(b"secret-content"),
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Confidential,
        };
        assert_eq!(
            engine.evaluate_ingest(&candidate),
            IngestDecision::StoreOnly
        );
    }

    // ── Task 10: filter_result ──

    #[test]
    fn filter_allows_public_content_in_any_context() {
        let mut engine = default_engine();
        let cid = make_cid(b"public-content");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_blocks_intimate_in_professional() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-content");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Block
        );
    }

    #[test]
    fn filter_blocks_intimate_in_social() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-social");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Social),
            FilterDecision::Block
        );
    }

    #[test]
    fn filter_confirms_intimate_at_companion() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-companion");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Companion),
            FilterDecision::Confirm
        );
    }

    #[test]
    fn filter_allows_intimate_in_private() {
        let mut engine = default_engine();
        let cid = make_cid(b"intimate-private");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Private),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_blocks_confidential_outside_private() {
        let mut engine = default_engine();
        let cid = make_cid(b"confidential-content");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Confidential,
            timestamp: 1000.0,
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
        let rules = FilterRuleSet {
            enabled: false,
            ..FilterRuleSet::default()
        };
        let mut engine = JainEngine::new(JainConfig::default(), rules, 10_000).unwrap();
        let cid = make_cid(b"intimate-disabled");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 10,
            content_type: ContentCategory::Image,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Intimate,
            timestamp: 1000.0,
        });
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }

    #[test]
    fn filter_allows_unknown_cid() {
        let engine = default_engine();
        let cid = make_cid(b"unknown-content");
        assert_eq!(
            engine.filter_result(&cid, SocialContext::Professional),
            FilterDecision::Allow
        );
    }

    // ── Task 11: tick ──

    #[test]
    fn tick_recommends_burn_for_very_stale_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"stale-burn");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        let now = 10.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        let has_burn = actions
            .iter()
            .any(|a| matches!(a, JainAction::RecommendBurn { .. }));
        assert!(has_burn, "expected RecommendBurn action, got: {actions:?}");
    }

    #[test]
    fn tick_recommends_archive_for_moderately_stale_content() {
        let config = JainConfig {
            archive_threshold: 0.3,
            burn_threshold: 0.95,
            ..JainConfig::default()
        };
        let mut engine = JainEngine::new(config, FilterRuleSet::default(), 10_000).unwrap();
        let cid = make_cid(b"stale-archive");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        let now = 2.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        let has_archive = actions
            .iter()
            .any(|a| matches!(a, JainAction::RecommendArchive { .. }));
        assert!(
            has_archive,
            "expected RecommendArchive action, got: {actions:?}"
        );
    }

    #[test]
    fn tick_skips_pinned_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"pinned-old");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 500,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });
        engine.handle_event(ContentEvent::Pinned { cid });

        let now = 10.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(now);
        let has_cleanup = actions.iter().any(|a| {
            matches!(
                a,
                JainAction::RecommendBurn { .. } | JainAction::RecommendArchive { .. }
            )
        });
        assert!(
            !has_cleanup,
            "pinned content should not get cleanup recommendations, got: {actions:?}"
        );
    }

    #[test]
    fn tick_emits_repair_needed_for_under_replicated() {
        let mut engine = default_engine();
        let cid = make_cid(b"under-rep");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });

        let actions = engine.tick(1000.0);
        let has_repair = actions
            .iter()
            .any(|a| matches!(a, JainAction::RepairNeeded { .. }));
        assert!(has_repair, "expected RepairNeeded action, got: {actions:?}");
    }

    #[test]
    fn tick_emits_storage_near_full_alert() {
        let mut engine = JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 1000).unwrap();
        let cid = make_cid(b"big-content");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 900,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        let actions = engine.tick(1000.0);
        let has_storage_alert = actions.iter().any(|a| {
            matches!(
                a,
                JainAction::HealthAlert {
                    alert: HealthAlertKind::StorageNearFull { .. }
                }
            )
        });
        assert!(
            has_storage_alert,
            "expected StorageNearFull alert, got: {actions:?}"
        );
    }

    #[test]
    fn tick_no_actions_for_fresh_content() {
        let mut engine = default_engine();
        let cid = make_cid(b"fresh-content");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::SelfCreated,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });
        engine.handle_event(ContentEvent::Accessed {
            cid,
            timestamp: 1000.0,
        });

        let actions = engine.tick(1000.0);
        assert!(
            actions.is_empty(),
            "expected no actions for fresh content, got: {actions:?}"
        );
    }

    // ── Task 12: reconcile ──

    #[test]
    fn reconcile_adds_missing_records_with_confidential_default() {
        let mut engine = default_engine();
        let cid = make_cid(b"disk-only");
        let snapshot = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 256,
            exists_on_disk: true,
        }];
        engine.reconcile(&snapshot, 1000.0);
        assert_eq!(engine.record_count(), 1);
        assert_eq!(engine.health_report(0.0).total_bytes, 256);

        // Recovered records must default to Confidential (fail-closed) —
        // prevents accidental leakage of content whose original sensitivity
        // is unknown after crash recovery.
        let decision = engine.filter_result(&cid, SocialContext::Professional);
        assert_eq!(
            decision,
            FilterDecision::Block,
            "reconciled records should be Confidential and blocked in Professional context"
        );
    }

    #[test]
    fn reconcile_detects_missing_backing_data() {
        let mut engine = default_engine();
        let cid = make_cid(b"missing-backing");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        // Set replica count to 3 so we can verify it's reported correctly
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });
        let snapshot = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 100,
            exists_on_disk: false,
        }];
        let actions = engine.reconcile(&snapshot, 1000.0);
        let fetch = actions
            .iter()
            .any(|a| matches!(a, JainAction::FetchLocalCopy { .. }));
        assert!(
            fetch,
            "reconcile should emit FetchLocalCopy for missing backing data, got: {actions:?}"
        );
        // Record should be marked as pending repair (protected from burn).
        assert!(
            engine.records.get(&cid).unwrap().pending_local_repair,
            "record should be marked pending_local_repair"
        );
    }

    #[test]
    fn reconcile_detects_orphaned_records() {
        let mut engine = default_engine();
        let cid = make_cid(b"orphaned");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        assert_eq!(engine.record_count(), 1);

        // Empty snapshot → the tracked record is orphaned
        let snapshot: Vec<SnapshotEntry> = alloc::vec![];
        let actions = engine.reconcile(&snapshot, 1000.0);
        assert_eq!(engine.record_count(), 0);
        let has_stale_alert = actions.iter().any(|a| {
            matches!(
                a,
                JainAction::HealthAlert {
                    alert: HealthAlertKind::StaleReconciliation { .. }
                }
            )
        });
        assert!(
            has_stale_alert,
            "expected StaleReconciliation alert, got: {actions:?}"
        );
    }

    // ── Review feedback fixes ──

    #[test]
    fn default_engine_allows_ingest() {
        // Default engine has capacity=0, which means "unlimited"
        let engine = JainEngine::default();
        let candidate = IngestCandidate {
            cid: make_cid(b"default-ingest"),
            size_bytes: 1_000_000,
            content_type: ContentCategory::Video,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
        };
        assert_eq!(
            engine.evaluate_ingest(&candidate),
            IngestDecision::IndexAndStore,
        );
    }

    #[test]
    fn health_report_counts_stale_records() {
        let mut engine = default_engine();
        let cid = make_cid(b"stale-for-report");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        // At t=0, content is fresh
        assert_eq!(engine.health_report(0.0).stale_count, 0);

        // Far in the future, content is stale
        let far_future = 10.0 * engine.config.access_decay_half_life_secs;
        assert!(
            engine.health_report(far_future).stale_count > 0,
            "stale_count should be > 0 for old content"
        );
    }

    #[test]
    fn reconciled_records_do_not_immediately_score_stale() {
        let mut engine = default_engine();
        let cid = make_cid(b"reconciled-fresh");
        let now = 1_700_000_000.0; // realistic wall-clock timestamp
        let snapshot = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 256,
            exists_on_disk: true,
        }];
        engine.reconcile(&snapshot, now);
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        // Immediately after reconcile, content should not be recommended for burn
        let actions = engine.tick(now);
        let has_burn = actions
            .iter()
            .any(|a| matches!(a, JainAction::RecommendBurn { .. }));
        assert!(
            !has_burn,
            "reconciled record should not immediately get RecommendBurn, got: {actions:?}"
        );
    }

    #[test]
    fn pending_local_repair_prevents_burn() {
        let mut engine = default_engine();
        let cid = make_cid(b"repair-pending");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        // Reconcile detects missing local copy.
        let snapshot = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 100,
            exists_on_disk: false,
        }];
        engine.reconcile(&snapshot, 0.0);

        // Far future tick — content is old, but should NOT be recommended for
        // burn because it's pending local repair.
        let far_future = 10.0 * engine.config.access_decay_half_life_secs;
        let actions = engine.tick(far_future);
        let has_burn = actions
            .iter()
            .any(|a| matches!(a, JainAction::RecommendBurn { .. }));
        assert!(
            !has_burn,
            "content pending local repair must not get RecommendBurn, got: {actions:?}"
        );

        // Re-store clears the flag — now burn is possible.
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        assert!(
            !engine.records.get(&cid).unwrap().pending_local_repair,
            "re-store should clear pending_local_repair"
        );
    }

    #[test]
    fn reconcile_clears_pending_repair_when_back_on_disk() {
        let mut engine = default_engine();
        let cid = make_cid(b"repair-then-restored");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 200,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 0.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });

        // First reconcile: backing data missing → sets pending_local_repair.
        let missing = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 200,
            exists_on_disk: false,
        }];
        engine.reconcile(&missing, 0.0);
        assert!(
            engine.records.get(&cid).unwrap().pending_local_repair,
            "reconcile should set pending_local_repair when backing is missing"
        );

        // Second reconcile: backing data restored (e.g. by operator cp) → clears flag.
        let restored = alloc::vec![SnapshotEntry {
            cid,
            size_bytes: 200,
            exists_on_disk: true,
        }];
        engine.reconcile(&restored, 1.0);
        assert!(
            !engine.records.get(&cid).unwrap().pending_local_repair,
            "reconcile should clear pending_local_repair when backing is restored"
        );
    }

    #[test]
    fn new_rejects_invalid_config() {
        let config = JainConfig {
            access_decay_half_life_secs: 0.0,
            ..JainConfig::default()
        };
        assert!(JainEngine::new(config, FilterRuleSet::default(), 10_000).is_err());
    }

    #[test]
    fn reconcile_deduplicates_snapshot_entries() {
        let mut engine = default_engine();
        let cid = make_cid(b"dup-entry");
        engine.handle_event(ContentEvent::Stored {
            cid,
            size_bytes: 100,
            content_type: ContentCategory::Text,
            origin: ContentOrigin::Downloaded,
            sensitivity: Sensitivity::Public,
            timestamp: 1000.0,
        });
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 0 });
        // Same CID appears twice with exists_on_disk: false
        let snapshot = alloc::vec![
            SnapshotEntry {
                cid,
                size_bytes: 100,
                exists_on_disk: false,
            },
            SnapshotEntry {
                cid,
                size_bytes: 100,
                exists_on_disk: false,
            },
        ];
        let actions = engine.reconcile(&snapshot, 1000.0);
        let fetch_count = actions
            .iter()
            .filter(|a| matches!(a, JainAction::FetchLocalCopy { .. }))
            .count();
        assert_eq!(
            fetch_count, 1,
            "duplicate snapshot entries should produce exactly one FetchLocalCopy, got {fetch_count}"
        );
    }

    #[test]
    fn config_validate_catches_zero_replica_count() {
        let config = JainConfig {
            min_replica_count: 0,
            ..JainConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_catches_zero_half_life() {
        let config = JainConfig {
            access_decay_half_life_secs: 0.0,
            ..JainConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_catches_inverted_thresholds() {
        let config = JainConfig {
            archive_threshold: 0.9,
            burn_threshold: 0.5,
            ..JainConfig::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn config_validate_accepts_defaults() {
        assert!(JainConfig::default().validate().is_ok());
    }
}

//! Sans-I/O content lifecycle engine.

use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_content::ContentId;

use crate::actions::*;
use crate::config::{FilterRuleSet, JainConfig};
use crate::types::{ContentRecord, Sensitivity};

/// Sans-I/O content lifecycle engine.
///
/// `JainEngine` tracks content records on a node and provides pure-function
/// methods for ingestion, filtering, tick-based housekeeping, and reconciliation.
/// It emits [`JainAction`] variants for the caller to execute — it never performs
/// I/O itself.
pub struct JainEngine {
    records: HashMap<ContentId, ContentRecord>,
    pub(crate) config: JainConfig,
    #[allow(dead_code)]
    filter_rules: FilterRuleSet,
    total_storage_bytes: u64,
    storage_capacity_bytes: u64,
}

impl JainEngine {
    /// Create a new engine with no tracked records.
    pub fn new(
        config: JainConfig,
        filter_rules: FilterRuleSet,
        storage_capacity_bytes: u64,
    ) -> Self {
        Self {
            records: HashMap::new(),
            config,
            filter_rules,
            total_storage_bytes: 0,
            storage_capacity_bytes,
        }
    }

    /// Number of content records currently tracked.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }

    /// Generate a health report summarizing the engine's current state.
    pub fn health_report(&self) -> HealthReport {
        let mut under_replicated_count = 0u32;
        let mut pinned_count = 0u32;

        for record in self.records.values() {
            if record.replica_count < self.config.min_replica_count {
                under_replicated_count += 1;
            }
            if record.pinned {
                pinned_count += 1;
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
            stale_count: 0, // Computed by tick, not health_report
            pinned_count,
        }
    }

    /// Process a content event, updating internal state and returning any actions
    /// for the caller to execute.
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
                if self.records.contains_key(&cid) {
                    return alloc::vec![JainAction::RecommendDedup {
                        keep: cid,
                        burn: cid,
                    }];
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
                };
                self.records.insert(cid, record);
                self.total_storage_bytes += size_bytes;
                Vec::new()
            }
            ContentEvent::Accessed { cid, timestamp } => {
                if let Some(record) = self.records.get_mut(&cid) {
                    record.last_accessed = timestamp;
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

        // 2. Storage budget check
        if self.total_storage_bytes + candidate.size_bytes > self.storage_capacity_bytes {
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
}

impl Default for JainEngine {
    fn default() -> Self {
        Self::new(JainConfig::default(), FilterRuleSet::default(), 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::JainConfig;
    use crate::types::ContentOrigin;
    use harmony_content::ContentFlags;
    use harmony_roxy::catalog::ContentCategory;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    fn default_engine() -> JainEngine {
        JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 10_000)
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
        let report = engine.health_report();
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
        let report = engine.health_report();
        assert_eq!(report.total_bytes, 100);
    }

    #[test]
    fn handle_stored_duplicate_emits_dedup() {
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
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], JainAction::RecommendDedup { .. }));
        assert_eq!(engine.record_count(), 1);
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
        assert_eq!(engine.health_report().total_bytes, 0);
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
        assert_eq!(engine.health_report().pinned_count, 1);
        engine.handle_event(ContentEvent::Unpinned { cid });
        assert_eq!(engine.health_report().pinned_count, 0);
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
        assert_eq!(engine.health_report().under_replicated_count, 1);
        engine.handle_event(ContentEvent::ReplicaChanged { cid, new_count: 3 });
        assert_eq!(engine.health_report().under_replicated_count, 0);
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
        assert_eq!(engine.evaluate_ingest(&candidate), IngestDecision::IndexAndStore);
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
        let engine = JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 50);
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
        assert_eq!(engine.evaluate_ingest(&candidate), IngestDecision::StoreOnly);
    }
}

//! Sans-I/O content lifecycle engine.

#[allow(unused_imports)]
use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_content::ContentId;

use crate::actions::*;
use crate::config::{FilterRuleSet, JainConfig};
use crate::types::ContentRecord;

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
    #[allow(dead_code)]
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
    use harmony_content::ContentFlags;

    fn make_cid(data: &[u8]) -> ContentId {
        ContentId::for_blob(data, ContentFlags::default()).unwrap()
    }

    fn default_engine() -> JainEngine {
        JainEngine::new(JainConfig::default(), FilterRuleSet::default(), 10_000)
    }

    // Suppress unused-function warnings for helpers used in later commits.
    #[allow(dead_code)]
    fn _use_helpers() {
        let _ = make_cid(b"x");
        let _ = default_engine();
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
}

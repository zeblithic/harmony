//! Configuration for the content lifecycle engine.

use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

use crate::types::{Sensitivity, SocialContext};

/// Top-level configuration for the Jain content lifecycle engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JainConfig {
    /// Staleness score above which content is archived (moved to cold storage).
    pub archive_threshold: f64,
    /// Staleness score above which content is burned (permanently deleted).
    pub burn_threshold: f64,
    /// Minimum number of replicas to maintain on the network.
    pub min_replica_count: u8,
    /// Storage usage percentage at which alerts are triggered.
    pub storage_alert_percent: f64,
    /// Half-life in seconds for access-count decay.
    pub access_decay_half_life_secs: f64,
    /// Weight modifier for self-created content (reduces staleness).
    pub self_created_weight: f64,
}

impl Default for JainConfig {
    fn default() -> Self {
        Self {
            archive_threshold: 0.6,
            burn_threshold: 0.85,
            min_replica_count: 2,
            storage_alert_percent: 0.85,
            access_decay_half_life_secs: 30.0 * 24.0 * 3600.0, // 30 days
            self_created_weight: 0.3,
        }
    }
}

/// A single filter rule controlling content sharing based on sensitivity
/// and social context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRule {
    /// Minimum sensitivity level that triggers this rule.
    pub min_sensitivity: Sensitivity,
    /// Maximum social context in which content at this sensitivity may be shared.
    pub max_context: SocialContext,
    /// Whether the user must confirm before sharing.
    pub require_confirmation: bool,
}

/// A set of filter rules that govern content sharing decisions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FilterRuleSet {
    /// The individual rules in this set.
    pub rules: Vec<FilterRule>,
    /// Whether the filter rule set is active.
    pub enabled: bool,
}

impl Default for FilterRuleSet {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![
                // Intimate content: only share with companions, require confirmation.
                FilterRule {
                    min_sensitivity: Sensitivity::Intimate,
                    max_context: SocialContext::Companion,
                    require_confirmation: true,
                },
                // Confidential content: only share privately, no confirmation
                // (auto-block in wider contexts).
                FilterRule {
                    min_sensitivity: Sensitivity::Confidential,
                    max_context: SocialContext::Private,
                    require_confirmation: false,
                },
                // Private content: share up to social context, no confirmation.
                FilterRule {
                    min_sensitivity: Sensitivity::Private,
                    max_context: SocialContext::Social,
                    require_confirmation: false,
                },
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_sane_thresholds() {
        let config = JainConfig::default();
        assert!(config.archive_threshold < config.burn_threshold);
        assert!(config.burn_threshold <= 1.0);
        assert!(config.min_replica_count >= 1);
        assert!(config.storage_alert_percent > 0.0);
        assert!(config.storage_alert_percent <= 1.0);
        assert!(config.access_decay_half_life_secs > 0.0);
        assert!(config.self_created_weight >= 0.0);
        assert!(config.self_created_weight <= 1.0);
    }

    #[test]
    fn default_filter_rules_block_intimate_in_professional() {
        let rules = FilterRuleSet::default();
        assert!(rules.enabled);
        assert!(!rules.rules.is_empty());
    }

    #[test]
    fn filter_rule_set_serialization_round_trip() {
        let rules = FilterRuleSet::default();
        let bytes = postcard::to_allocvec(&rules).unwrap();
        let decoded: FilterRuleSet = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.enabled, rules.enabled);
        assert_eq!(decoded.rules.len(), rules.rules.len());
        for (original, restored) in rules.rules.iter().zip(decoded.rules.iter()) {
            assert_eq!(original.min_sensitivity, restored.min_sensitivity);
            assert_eq!(original.max_context, restored.max_context);
            assert_eq!(original.require_confirmation, restored.require_confirmation);
        }
    }

    #[test]
    fn config_serialization_round_trip() {
        let config = JainConfig::default();
        let bytes = postcard::to_allocvec(&config).unwrap();
        let decoded: JainConfig = postcard::from_bytes(&bytes).unwrap();
        assert!((decoded.archive_threshold - config.archive_threshold).abs() < f64::EPSILON);
        assert!((decoded.burn_threshold - config.burn_threshold).abs() < f64::EPSILON);
        assert_eq!(decoded.min_replica_count, config.min_replica_count);
        assert!(
            (decoded.storage_alert_percent - config.storage_alert_percent).abs() < f64::EPSILON
        );
        assert!(
            (decoded.access_decay_half_life_secs - config.access_decay_half_life_secs).abs()
                < f64::EPSILON
        );
        assert!(
            (decoded.self_created_weight - config.self_created_weight).abs() < f64::EPSILON
        );
    }
}

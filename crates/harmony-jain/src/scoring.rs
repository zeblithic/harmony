//! Staleness scoring — pure function from ContentRecord + JainConfig → StalenessScore.

use crate::config::JainConfig;
use crate::types::{ContentOrigin, ContentRecord, StalenessScore};

/// Compute the staleness score for a content record given the current configuration
/// and wall-clock time.
///
/// # Hard overrides
///
/// Content that is pinned, licensed, or under-replicated is always considered
/// completely fresh (`StalenessScore::FRESH`).
///
/// # Decay model
///
/// Staleness combines exponential time-decay of last-access recency (70% weight)
/// with a saturating logarithmic frequency factor (30% weight). Self-created
/// content receives a configurable bonus that reduces its effective staleness.
pub fn staleness_score(record: &ContentRecord, config: &JainConfig, now: f64) -> StalenessScore {
    // Hard overrides → FRESH (0.0)
    if record.pinned
        || record.licensed
        || record.replica_count < config.min_replica_count
        || record.pending_local_repair
    {
        return StalenessScore::FRESH;
    }

    // Time decay: exponential half-life
    // Guard: if half-life is non-positive (should be caught by config validation),
    // treat as if infinite half-lives have elapsed — everything is maximally stale.
    // This prevents 0.0/0.0 = NaN when elapsed is also zero.
    let elapsed = (now - record.last_accessed).max(0.0);
    let half_lives = if config.access_decay_half_life_secs > 0.0 {
        elapsed / config.access_decay_half_life_secs
    } else {
        f64::INFINITY
    };
    let recency = pow_f(0.5, half_lives);

    // Frequency factor: saturating logarithmic curve
    let ac = record.access_count as f64;
    let freq = ln_1p(ac) / ln_1p(10.0 + ac);

    // Combined freshness
    let freshness = 0.7 * recency + 0.3 * freq;

    // Origin bonus for self-created content
    let origin_bonus = if record.origin == ContentOrigin::SelfCreated {
        config.self_created_weight
    } else {
        0.0
    };

    // Final staleness
    let staleness = (1.0 - freshness) * (1.0 - origin_bonus);
    StalenessScore::new(staleness)
}

/// Compute base^exp using the appropriate math backend.
#[cfg(feature = "std")]
fn pow_f(base: f64, exp: f64) -> f64 {
    base.powf(exp)
}

/// Compute base^exp using libm for no_std environments.
#[cfg(not(feature = "std"))]
fn pow_f(base: f64, exp: f64) -> f64 {
    libm::pow(base, exp)
}

/// Compute ln(1 + x) using the appropriate math backend.
#[cfg(feature = "std")]
fn ln_1p(x: f64) -> f64 {
    x.ln_1p()
}

/// Compute ln(1 + x) using libm for no_std environments.
#[cfg(not(feature = "std"))]
fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Sensitivity;
    use harmony_content::{ContentFlags, ContentId};
    use harmony_roxy::catalog::ContentCategory;

    fn base_record() -> ContentRecord {
        ContentRecord {
            cid: ContentId::for_book(b"test-content", ContentFlags::default()).unwrap(),
            size_bytes: 1024,
            content_type: ContentCategory::Music,
            origin: ContentOrigin::CachedInTransit,
            sensitivity: Sensitivity::Public,
            stored_at: 0.0,
            last_accessed: 0.0,
            access_count: 1,
            replica_count: 3,
            pinned: false,
            licensed: false,
            pending_local_repair: false,
        }
    }

    #[test]
    fn recently_accessed_content_scores_low() {
        let mut record = base_record();
        record.last_accessed = 100.0;
        record.access_count = 50;
        let config = JainConfig::default();
        let score = staleness_score(&record, &config, 100.0);
        assert!(
            score.value() < 0.1,
            "recently accessed content should score < 0.1, got {}",
            score.value()
        );
    }

    #[test]
    fn old_untouched_content_scores_high() {
        let mut record = base_record();
        record.last_accessed = 0.0;
        let config = JainConfig::default();
        let now = 4.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert!(
            score.value() > 0.8,
            "old untouched content should score > 0.8, got {}",
            score.value()
        );
    }

    #[test]
    fn pinned_content_always_fresh() {
        let mut record = base_record();
        record.pinned = true;
        record.last_accessed = 0.0;
        let config = JainConfig::default();
        let now = 100.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert!(
            (score.value() - 0.0).abs() < f64::EPSILON,
            "pinned content should be FRESH, got {}",
            score.value()
        );
    }

    #[test]
    fn licensed_content_always_fresh() {
        let mut record = base_record();
        record.licensed = true;
        record.last_accessed = 0.0;
        let config = JainConfig::default();
        let now = 100.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert!(
            (score.value() - 0.0).abs() < f64::EPSILON,
            "licensed content should be FRESH, got {}",
            score.value()
        );
    }

    #[test]
    fn under_replicated_content_always_fresh() {
        let mut record = base_record();
        record.replica_count = 1;
        record.last_accessed = 0.0;
        let config = JainConfig {
            min_replica_count: 2,
            ..JainConfig::default()
        };
        let now = 100.0 * config.access_decay_half_life_secs;
        let score = staleness_score(&record, &config, now);
        assert!(
            (score.value() - 0.0).abs() < f64::EPSILON,
            "under-replicated content should be FRESH, got {}",
            score.value()
        );
    }

    #[test]
    fn self_created_content_scores_lower_than_transit() {
        let config = JainConfig::default();
        let now = 2.0 * config.access_decay_half_life_secs;

        let mut self_created = base_record();
        self_created.origin = ContentOrigin::SelfCreated;
        self_created.last_accessed = 0.0;
        let self_score = staleness_score(&self_created, &config, now);

        let mut transit = base_record();
        transit.origin = ContentOrigin::CachedInTransit;
        transit.last_accessed = 0.0;
        let transit_score = staleness_score(&transit, &config, now);

        assert!(
            self_score.value() < transit_score.value(),
            "self-created ({}) should score lower than transit ({})",
            self_score.value(),
            transit_score.value()
        );
    }

    #[test]
    fn frequently_accessed_content_scores_lower() {
        let config = JainConfig::default();
        let now = 2.0 * config.access_decay_half_life_secs;

        let mut frequent = base_record();
        frequent.access_count = 100;
        frequent.last_accessed = 0.0;
        let frequent_score = staleness_score(&frequent, &config, now);

        let mut infrequent = base_record();
        infrequent.access_count = 1;
        infrequent.last_accessed = 0.0;
        let infrequent_score = staleness_score(&infrequent, &config, now);

        assert!(
            frequent_score.value() < infrequent_score.value(),
            "frequently accessed ({}) should score lower than infrequent ({})",
            frequent_score.value(),
            infrequent_score.value()
        );
    }
}

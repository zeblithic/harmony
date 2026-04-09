//! MAC blacklisting for DoS mitigation.
//!
//! Tracks per-MAC violation counts across 4 categories within sliding time
//! windows. When any category crosses its threshold, the MAC is banned with
//! escalating backoff durations.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use tracing::warn;

/// Category of violation detected from a source MAC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViolationCategory {
    /// Frame failed validation (too short, wrong EtherType, truncated, invalid UTF-8).
    MalformedFrame = 0,
    /// Frames/sec exceeds threshold from a single MAC.
    RateFlood = 1,
    /// Rapid identity_hash cycling in Scout frames.
    IdentityChurn = 2,
    /// key_expr outside allowed namespace prefix.
    NamespaceViolation = 3,
}

/// Configuration for [`MacBlacklist`] thresholds and ban durations.
pub struct BlacklistConfig {
    /// Time window for counting violations (default: 10 seconds).
    pub window_duration: Duration,
    /// Threshold per category to trigger a ban (indexed by [`ViolationCategory`]).
    pub thresholds: [u32; 4],
    /// Base ban duration for first offense (default: 60 seconds).
    pub base_ban: Duration,
    /// Multiplier per subsequent offense (default: 4x).
    pub escalation_factor: u32,
    /// Maximum ban duration cap (default: 1 hour).
    pub max_ban: Duration,
}

impl Default for BlacklistConfig {
    fn default() -> Self {
        Self {
            window_duration: Duration::from_secs(10),
            thresholds: [
                20,  // MalformedFrame
                500, // RateFlood
                5,   // IdentityChurn
                10,  // NamespaceViolation
            ],
            base_ban: Duration::from_secs(60),
            escalation_factor: 4,
            max_ban: Duration::from_secs(3600),
        }
    }
}

struct ViolationRecord {
    /// Per-category violation counts within the current time window.
    counts: [u32; 4],
    /// Start of the current counting window.
    window_start: Instant,
    /// Number of times this MAC has been banned (drives escalation).
    offense_count: u32,
    /// If Some, this MAC is currently banned until this time.
    banned_until: Option<Instant>,
}

/// Per-MAC violation tracker with threshold-based banning and escalating backoff.
pub struct MacBlacklist {
    entries: HashMap<[u8; 6], ViolationRecord>,
    config: BlacklistConfig,
}

impl MacBlacklist {
    /// Create a new blacklist with the given configuration.
    pub fn new(config: BlacklistConfig) -> Self {
        Self {
            entries: HashMap::new(),
            config,
        }
    }

    /// Returns true if this MAC is currently banned.
    ///
    /// Expired bans return false (lazy expiry — no cleanup needed at check time).
    pub fn is_blocked(&self, mac: &[u8; 6], now: Instant) -> bool {
        match self.entries.get(mac) {
            Some(record) => match record.banned_until {
                Some(deadline) if now < deadline => true,
                _ => false,
            },
            None => false,
        }
    }

    /// Record a violation from a source MAC.
    ///
    /// If the MAC's count for this category crosses the threshold within the
    /// current window, the MAC is banned with escalating duration. Returns
    /// true if this violation triggered a new ban.
    pub fn record_violation(
        &mut self,
        mac: &[u8; 6],
        category: ViolationCategory,
        now: Instant,
    ) -> bool {
        let config = &self.config;
        let record = self.entries.entry(*mac).or_insert_with(|| ViolationRecord {
            counts: [0; 4],
            window_start: now,
            offense_count: 0,
            banned_until: None,
        });

        // Reset counts if the window has expired.
        if now.duration_since(record.window_start) > config.window_duration {
            record.counts = [0; 4];
            record.window_start = now;
        }

        let idx = category as usize;
        record.counts[idx] += 1;

        if record.counts[idx] >= config.thresholds[idx] {
            // Compute escalating ban duration.
            let multiplier = config
                .escalation_factor
                .checked_pow(record.offense_count)
                .unwrap_or(u32::MAX);
            let ban_duration = config
                .base_ban
                .saturating_mul(multiplier)
                .min(config.max_ban);

            record.banned_until = Some(now + ban_duration);
            record.offense_count += 1;
            record.counts = [0; 4]; // Reset — ban is the response.

            warn!(
                mac = hex::encode(mac),
                category = ?category,
                ban_secs = ban_duration.as_secs(),
                offense = record.offense_count,
                "MAC banned"
            );

            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAC_A: [u8; 6] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    const MAC_B: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

    #[test]
    fn below_threshold_no_ban() {
        let config = BlacklistConfig {
            thresholds: [5, 500, 5, 10],
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        for _ in 0..4 {
            let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
            assert!(!triggered);
        }
        assert!(!bl.is_blocked(&MAC_A, t0));
    }

    #[test]
    fn at_threshold_triggers_ban() {
        let config = BlacklistConfig {
            thresholds: [5, 500, 5, 10],
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        for i in 0..5 {
            let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
            if i < 4 {
                assert!(!triggered, "should not trigger before threshold");
            } else {
                assert!(triggered, "5th violation should trigger ban");
            }
        }
        assert!(bl.is_blocked(&MAC_A, t0));
    }

    #[test]
    fn window_reset_clears_counts() {
        let config = BlacklistConfig {
            thresholds: [5, 500, 5, 10],
            window_duration: Duration::from_secs(10),
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        for _ in 0..4 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }

        let t1 = t0 + Duration::from_secs(11);
        let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t1);
        assert!(!triggered, "counts should have reset with new window");
        assert!(!bl.is_blocked(&MAC_A, t1));
    }

    #[test]
    fn rate_flood_threshold() {
        let config = BlacklistConfig::default();
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        for _ in 0..499 {
            let triggered = bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
            assert!(!triggered);
        }
        assert!(!bl.is_blocked(&MAC_A, t0));

        let triggered = bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
        assert!(triggered);
        assert!(bl.is_blocked(&MAC_A, t0));
    }

    #[test]
    fn identity_churn_threshold() {
        let config = BlacklistConfig::default();
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        for _ in 0..4 {
            let triggered = bl.record_violation(&MAC_A, ViolationCategory::IdentityChurn, t0);
            assert!(!triggered);
        }

        let triggered = bl.record_violation(&MAC_A, ViolationCategory::IdentityChurn, t0);
        assert!(triggered);
        assert!(bl.is_blocked(&MAC_A, t0));
    }

    #[test]
    fn multiple_categories_independent() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
        assert!(!bl.is_blocked(&MAC_A, t0), "neither category at threshold");

        let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        assert!(triggered, "MalformedFrame hit threshold independently");
        assert!(bl.is_blocked(&MAC_A, t0));
    }

    // Note: MAC_B is defined but used in later tests (Task 2+). Suppress dead_code warning.
    #[test]
    fn different_macs_tracked_independently() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // Ban MAC_A.
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }
        assert!(bl.is_blocked(&MAC_A, t0));
        assert!(!bl.is_blocked(&MAC_B, t0), "MAC_B should not be affected");
    }
}

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
        self.entries
            .get(mac)
            .is_some_and(|r| matches!(r.banned_until, Some(deadline) if now < deadline))
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
        // Don't accumulate violations while already banned. Without this guard,
        // a burst of N > threshold frames in one recv_frames drain would trigger
        // multiple bans back-to-back, immediately escalating offense_count.
        if self.is_blocked(mac, now) {
            return false;
        }

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

    /// Remove entries whose ban has expired AND whose last activity is older
    /// than the window duration. Called periodically to bound memory growth.
    ///
    /// This discards `offense_count` — a MAC that went quiet for a full window
    /// gets a fresh start. Intentional: forgiveness for transient issues.
    pub fn purge_expired(&mut self, now: Instant) {
        let window = self.config.window_duration;
        self.entries.retain(|_, record| {
            let ban_active = matches!(record.banned_until, Some(deadline) if now < deadline);
            let window_active = now.duration_since(record.window_start) <= window;
            ban_active || window_active
        });
    }

    /// Returns the number of tracked MAC entries (for testing).
    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.entries.len()
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

    #[test]
    fn ban_expiry() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            base_ban: Duration::from_secs(60),
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // Trigger a ban.
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }
        assert!(bl.is_blocked(&MAC_A, t0));

        // Still banned at 59 seconds.
        assert!(bl.is_blocked(&MAC_A, t0 + Duration::from_secs(59)));

        // Expired at 60 seconds.
        assert!(!bl.is_blocked(&MAC_A, t0 + Duration::from_secs(60)));
    }

    #[test]
    fn escalating_bans() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            base_ban: Duration::from_secs(60),
            escalation_factor: 4,
            max_ban: Duration::from_secs(3600),
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // 1st offense: 60s ban.
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }
        assert!(bl.is_blocked(&MAC_A, t0));
        assert!(bl.is_blocked(&MAC_A, t0 + Duration::from_secs(59)));
        assert!(!bl.is_blocked(&MAC_A, t0 + Duration::from_secs(60)));

        // 2nd offense: 60 * 4 = 240s ban.
        let t1 = t0 + Duration::from_secs(61);
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t1);
        }
        assert!(bl.is_blocked(&MAC_A, t1));
        assert!(bl.is_blocked(&MAC_A, t1 + Duration::from_secs(239)));
        assert!(!bl.is_blocked(&MAC_A, t1 + Duration::from_secs(240)));

        // 3rd offense: 60 * 16 = 960s ban.
        let t2 = t1 + Duration::from_secs(241);
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t2);
        }
        assert!(bl.is_blocked(&MAC_A, t2));
        assert!(bl.is_blocked(&MAC_A, t2 + Duration::from_secs(959)));
        assert!(!bl.is_blocked(&MAC_A, t2 + Duration::from_secs(960)));

        // 4th offense: 60 * 64 = 3840s → capped at 3600s.
        let t3 = t2 + Duration::from_secs(961);
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t3);
        }
        assert!(bl.is_blocked(&MAC_A, t3));
        assert!(bl.is_blocked(&MAC_A, t3 + Duration::from_secs(3599)));
        assert!(!bl.is_blocked(&MAC_A, t3 + Duration::from_secs(3600)));
    }

    #[test]
    fn purge_cleans_stale_entries() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            window_duration: Duration::from_secs(10),
            base_ban: Duration::from_secs(60),
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // MAC_A: trigger a ban (60s).
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }

        // MAC_B: record some violations but no ban.
        bl.record_violation(&MAC_B, ViolationCategory::MalformedFrame, t0);

        assert_eq!(bl.entry_count(), 2);

        // At t0 + 15s: MAC_B's window has expired (10s), no ban → purgeable.
        // MAC_A's ban is still active (60s) → NOT purgeable.
        let t1 = t0 + Duration::from_secs(15);
        bl.purge_expired(t1);
        assert_eq!(bl.entry_count(), 1);
        assert!(bl.is_blocked(&MAC_A, t1), "active ban must survive purge");

        // At t0 + 70s: MAC_A's ban has expired (60s) and window has expired → purgeable.
        let t2 = t0 + Duration::from_secs(70);
        bl.purge_expired(t2);
        assert_eq!(bl.entry_count(), 0);
    }

    #[test]
    fn offense_count_resets_after_purge() {
        let config = BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            window_duration: Duration::from_secs(10),
            base_ban: Duration::from_secs(60),
            escalation_factor: 4,
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // 1st offense: 60s ban.
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }
        assert!(bl.is_blocked(&MAC_A, t0));

        // Let ban expire and window expire, then purge.
        let t1 = t0 + Duration::from_secs(70);
        bl.purge_expired(t1);

        // 2nd offense after purge: should be 60s again (not 240s).
        let t2 = t1 + Duration::from_secs(1);
        for _ in 0..3 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t2);
        }
        assert!(bl.is_blocked(&MAC_A, t2));
        // If escalation were preserved, ban would last 240s. After purge, it's 60s.
        assert!(bl.is_blocked(&MAC_A, t2 + Duration::from_secs(59)));
        assert!(!bl.is_blocked(&MAC_A, t2 + Duration::from_secs(60)));
    }

    #[test]
    fn no_double_escalation_from_single_burst() {
        // Regression: a burst of N > threshold violations in one drain must
        // produce exactly one ban at the current offense level, not escalate
        // multiple times.
        let config = BlacklistConfig {
            thresholds: [20, 500, 5, 10],
            base_ban: Duration::from_secs(60),
            escalation_factor: 4,
            ..Default::default()
        };
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // Simulate 1000 RateFlood violations in one burst (threshold is 500).
        for _ in 0..1000 {
            bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
        }

        // Should be first-offense ban (60s), NOT escalated to 240s.
        assert!(bl.is_blocked(&MAC_A, t0));
        assert!(bl.is_blocked(&MAC_A, t0 + Duration::from_secs(59)));
        assert!(
            !bl.is_blocked(&MAC_A, t0 + Duration::from_secs(60)),
            "ban should be 60s (first offense), not escalated"
        );
    }
}

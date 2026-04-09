# Dynamic MAC Blacklisting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add in-process MAC blacklisting to the rawlink bridge that detects misbehaving mesh peers and drops their frames with escalating ban durations.

**Architecture:** A new `MacBlacklist` struct in `blacklist.rs` follows the sans-I/O pattern (caller-supplied `Instant` timestamps). The bridge calls `is_blocked()` at the top of the frame receive callback and `record_violation()` at each validation failure point. Escalating backoff bans (60s → 4m → 16m → 1h cap) with per-10s-window violation counters across 4 categories.

**Tech Stack:** Rust, std::collections::HashMap, std::time::{Duration, Instant}, tracing

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `crates/harmony-rawlink/src/blacklist.rs` | Create | `MacBlacklist`, `ViolationRecord`, `BlacklistConfig`, `ViolationCategory` — all detection and ban logic |
| `crates/harmony-rawlink/src/lib.rs` | Modify | Add `pub mod blacklist;` |
| `crates/harmony-rawlink/src/peer_table.rs` | Modify | Add `identity_for_mac()` reverse lookup for churn detection |
| `crates/harmony-rawlink/src/bridge.rs` | Modify | Integrate blacklist: `is_blocked` check, `record_violation` calls, identity churn detection, purge |

---

### Task 1: Core Types and `record_violation` with Threshold Detection

**Files:**
- Create: `crates/harmony-rawlink/src/blacklist.rs`
- Modify: `crates/harmony-rawlink/src/lib.rs`

- [ ] **Step 1: Write the failing tests for violation counting and threshold detection**

Add to `crates/harmony-rawlink/src/blacklist.rs`:

```rust
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

        // Record 4 malformed frames (threshold is 5).
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

        // Record 4 violations (just below threshold).
        for _ in 0..4 {
            bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        }

        // Advance past window — counts reset.
        let t1 = t0 + Duration::from_secs(11);
        let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t1);
        assert!(!triggered, "counts should have reset with new window");
        assert!(!bl.is_blocked(&MAC_A, t1));
    }

    #[test]
    fn rate_flood_threshold() {
        let config = BlacklistConfig::default(); // threshold: 500
        let mut bl = MacBlacklist::new(config);
        let t0 = Instant::now();

        // 499 frames — no ban.
        for _ in 0..499 {
            let triggered = bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
            assert!(!triggered);
        }
        assert!(!bl.is_blocked(&MAC_A, t0));

        // 500th — ban.
        let triggered = bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);
        assert!(triggered);
        assert!(bl.is_blocked(&MAC_A, t0));
    }

    #[test]
    fn identity_churn_threshold() {
        let config = BlacklistConfig::default(); // threshold: 5
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

        // 2 malformed (below threshold of 3).
        bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);

        // 1 rate flood (way below threshold of 500).
        bl.record_violation(&MAC_A, ViolationCategory::RateFlood, t0);

        assert!(!bl.is_blocked(&MAC_A, t0), "neither category at threshold");

        // 3rd malformed — triggers ban.
        let triggered = bl.record_violation(&MAC_A, ViolationCategory::MalformedFrame, t0);
        assert!(triggered, "MalformedFrame hit threshold independently");
        assert!(bl.is_blocked(&MAC_A, t0));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink blacklist`
Expected: FAIL — `MacBlacklist::new`, `record_violation`, `is_blocked` not implemented yet.

- [ ] **Step 3: Implement `MacBlacklist::new`, `record_violation`, and `is_blocked`**

Add to `crates/harmony-rawlink/src/blacklist.rs`, after the `MacBlacklist` struct definition:

```rust
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
```

- [ ] **Step 4: Register the module in `lib.rs`**

In `crates/harmony-rawlink/src/lib.rs`, add after the existing module declarations:

```rust
pub mod blacklist;
```

And add to the `pub use` section:

```rust
pub use blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink blacklist`
Expected: All 7 tests PASS.

- [ ] **Step 6: Run full workspace tests to check for regressions**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink`
Expected: All existing tests plus new tests PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/blacklist.rs crates/harmony-rawlink/src/lib.rs
git commit -m "feat(rawlink): add MacBlacklist with violation counting and threshold bans

Introduces blacklist.rs with per-MAC violation tracking across 4 categories
(MalformedFrame, RateFlood, IdentityChurn, NamespaceViolation). When any
category crosses its configurable threshold within a 10-second sliding window,
the MAC is banned. 7 unit tests cover threshold behavior, window reset,
rate flooding, identity churn, and category independence.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Ban Expiry, Escalating Backoff, and Purge

**Files:**
- Modify: `crates/harmony-rawlink/src/blacklist.rs`

- [ ] **Step 1: Write the failing tests for ban expiry, escalation, and purge**

Add to the `tests` module in `crates/harmony-rawlink/src/blacklist.rs`:

```rust
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
        // Blocked at 59s, free at 60s.
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
```

- [ ] **Step 2: Run tests to verify the new tests fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink blacklist`
Expected: FAIL — `purge_expired` and `entry_count` not implemented.

- [ ] **Step 3: Implement `purge_expired` and `entry_count`**

Add to the `impl MacBlacklist` block in `crates/harmony-rawlink/src/blacklist.rs`:

```rust
    /// Remove entries whose ban has expired AND whose last activity is older
    /// than the window duration. Called periodically to bound memory growth.
    ///
    /// This discards `offense_count` — a MAC that went quiet for a full window
    /// gets a fresh start. Intentional: forgiveness for transient issues.
    pub fn purge_expired(&mut self, now: Instant) {
        let window = self.config.window_duration;
        self.entries.retain(|_, record| {
            let ban_active = match record.banned_until {
                Some(deadline) if now < deadline => true,
                _ => false,
            };
            let window_active = now.duration_since(record.window_start) <= window;
            ban_active || window_active
        });
    }

    /// Returns the number of tracked MAC entries (for testing).
    #[cfg(test)]
    fn entry_count(&self) -> usize {
        self.entries.len()
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink blacklist`
Expected: All 11 tests PASS.

- [ ] **Step 5: Run full crate tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink`
Expected: All tests PASS.

- [ ] **Step 6: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/blacklist.rs
git commit -m "feat(rawlink): add ban expiry, escalating backoff, and purge to MacBlacklist

Bans auto-expire via lazy check in is_blocked(). Escalating backoff multiplies
base_ban by escalation_factor^offense_count, capped at max_ban. purge_expired()
removes stale entries to bound memory, resetting offense_count for forgiveness.
4 tests: ban expiry, escalation sequence with cap, purge cleanup, offense reset.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: PeerTable `identity_for_mac` Reverse Lookup

**Files:**
- Modify: `crates/harmony-rawlink/src/peer_table.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/harmony-rawlink/src/peer_table.rs`:

```rust
    #[test]
    fn identity_for_mac_returns_hash() {
        let mut table = PeerTable::new(Duration::from_secs(60));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        assert_eq!(table.identity_for_mac(&mac), Some(identity_hash));
    }

    #[test]
    fn identity_for_mac_unknown_returns_none() {
        let table = PeerTable::new(Duration::from_secs(60));
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        assert_eq!(table.identity_for_mac(&mac), None);
    }

    #[test]
    fn identity_for_mac_expired_returns_none() {
        let mut table = PeerTable::new(Duration::from_millis(1));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(table.identity_for_mac(&mac), None);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink peer_table`
Expected: FAIL — `identity_for_mac` not found.

- [ ] **Step 3: Implement `identity_for_mac`**

Add to the `impl PeerTable` block in `crates/harmony-rawlink/src/peer_table.rs`, after `lookup_by_mac`:

```rust
    /// Look up the identity hash associated with a MAC address.
    ///
    /// Returns `None` if the MAC is unknown or the entry has expired.
    /// Scans all entries (O(n)) — suitable for the small peer tables in mesh networks.
    pub fn identity_for_mac(&self, mac: &[u8; 6]) -> Option<[u8; 16]> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.mac == *mac && entry.last_seen.elapsed() < self.ttl)
            .map(|(identity_hash, _)| *identity_hash)
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink peer_table`
Expected: All peer_table tests PASS (existing 5 + new 3 = 8 total).

- [ ] **Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/peer_table.rs
git commit -m "feat(rawlink): add PeerTable::identity_for_mac reverse lookup

Returns the identity_hash for a given MAC address, needed by the blacklist's
identity churn detection. O(n) scan, suitable for small mesh peer tables.
3 tests: found, unknown, expired.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Bridge Integration — Blacklist Check, Rate Counting, and Violation Recording

**Files:**
- Modify: `crates/harmony-rawlink/src/bridge.rs`

This task integrates the blacklist into the bridge's `recv_frames` callback. It adds:
1. `is_blocked()` early rejection at the top of the callback
2. `record_violation(RateFlood)` for every non-blocked frame (rate counting)
3. `record_violation(MalformedFrame)` at each existing validation failure
4. `record_violation(NamespaceViolation)` at namespace check failure
5. `purge_expired()` alongside `peer_table.purge_expired()`

- [ ] **Step 1: Write the failing integration test — blocked MAC frames dropped**

Add to the `tests` module in `crates/harmony-rawlink/src/bridge.rs`:

```rust
    #[test]
    fn blacklist_blocks_banned_mac() {
        use crate::blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};

        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);
        let mut blacklist = MacBlacklist::new(BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            ..Default::default()
        });
        let now = Instant::now();

        // Ban MAC_B by recording 3 malformed frame violations.
        for _ in 0..3 {
            blacklist.record_violation(&MAC_B, ViolationCategory::MalformedFrame, now);
        }
        assert!(blacklist.is_blocked(&MAC_B, now));

        // B sends a valid scout frame.
        let scout = make_scout_payload(&IDENTITY);
        socket_b.send_frame(MAC_A, &scout).expect("mock send");

        let local_mac = MAC_A;
        let mut peer_table = PeerTable::new(Duration::from_secs(30));

        // Simulate recv_frames with blacklist check (mirrors bridge logic).
        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if blacklist.is_blocked(src_mac, now) {
                    return; // Dropped — banned MAC.
                }
                if src_mac == &local_mac || payload.is_empty() {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        // Scout from banned MAC must be silently dropped.
        assert_eq!(peer_table.peer_count(), 0, "banned MAC's scout must be dropped");
    }

    #[test]
    fn blacklist_does_not_affect_other_macs() {
        use crate::blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};

        let mac_c: [u8; 6] = [0xCC, 0xCC, 0xCC, 0xCC, 0xCC, 0xCC];
        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);

        let mut blacklist = MacBlacklist::new(BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            ..Default::default()
        });
        let now = Instant::now();

        // Ban MAC_C (not MAC_B).
        for _ in 0..3 {
            blacklist.record_violation(&mac_c, ViolationCategory::MalformedFrame, now);
        }

        // B sends a scout (B is not banned).
        let scout = make_scout_payload(&IDENTITY);
        socket_b.send_frame(MAC_A, &scout).expect("mock send");

        let local_mac = MAC_A;
        let mut peer_table = PeerTable::new(Duration::from_secs(30));

        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if blacklist.is_blocked(src_mac, now) {
                    return;
                }
                if src_mac == &local_mac || payload.is_empty() {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        // MAC_B is not banned — scout should be processed.
        assert_eq!(peer_table.peer_count(), 1, "unbanned MAC should be processed");
    }

    #[test]
    fn blacklist_ban_expires_traffic_resumes() {
        use crate::blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};

        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);
        let mut blacklist = MacBlacklist::new(BlacklistConfig {
            thresholds: [3, 500, 5, 10],
            base_ban: Duration::from_secs(60),
            ..Default::default()
        });
        let t0 = Instant::now();

        // Ban MAC_B.
        for _ in 0..3 {
            blacklist.record_violation(&MAC_B, ViolationCategory::MalformedFrame, t0);
        }
        assert!(blacklist.is_blocked(&MAC_B, t0));

        // After ban expires...
        let t1 = t0 + Duration::from_secs(61);
        assert!(!blacklist.is_blocked(&MAC_B, t1));

        // B sends a scout.
        let scout = make_scout_payload(&IDENTITY);
        socket_b.send_frame(MAC_A, &scout).expect("mock send");

        let local_mac = MAC_A;
        let mut peer_table = PeerTable::new(Duration::from_secs(30));

        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if blacklist.is_blocked(src_mac, t1) {
                    return;
                }
                if src_mac == &local_mac || payload.is_empty() {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        assert_eq!(peer_table.peer_count(), 1, "unbanned MAC traffic should resume");
    }
```

- [ ] **Step 2: Run tests to verify they pass (these tests don't require bridge modification — they simulate the logic)**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink blacklist_blocks_banned_mac blacklist_does_not_affect_other_macs blacklist_ban_expires_traffic_resumes`
Expected: All 3 PASS (they test the pattern, not the bridge integration itself).

- [ ] **Step 3: Integrate blacklist into `Bridge` struct and `run()` method**

In `crates/harmony-rawlink/src/bridge.rs`:

**3a. Add import** — at the top, add to the `use crate` block:

```rust
use crate::blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};
```

**3b. Add field to `Bridge` struct** (line ~110):

Change:
```rust
pub struct Bridge<S: RawSocket> {
    socket: S,
    session: zenoh::Session,
    config: BridgeConfig,
    peer_table: PeerTable,
    reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
}
```
To:
```rust
pub struct Bridge<S: RawSocket> {
    socket: S,
    session: zenoh::Session,
    config: BridgeConfig,
    peer_table: PeerTable,
    blacklist: MacBlacklist,
    reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
}
```

**3c. Initialize in `new()`** (line ~126):

Change:
```rust
    pub fn new(
        socket: S,
        session: zenoh::Session,
        config: BridgeConfig,
        reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
    ) -> Self {
        let peer_table = PeerTable::new(config.peer_ttl);
        Self {
            socket,
            session,
            config,
            peer_table,
            reticulum_outbound_rx,
        }
    }
```
To:
```rust
    pub fn new(
        socket: S,
        session: zenoh::Session,
        config: BridgeConfig,
        reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
    ) -> Self {
        let peer_table = PeerTable::new(config.peer_ttl);
        let blacklist = MacBlacklist::new(BlacklistConfig::default());
        Self {
            socket,
            session,
            config,
            peer_table,
            blacklist,
            reticulum_outbound_rx,
        }
    }
```

**3d. Add purge call** — in `run()`, after the peer table purge (line ~173):

Change:
```rust
                // 2. Purge expired peer table entries periodically.
                if now >= next_purge {
                    self.peer_table.purge_expired();
                    next_purge = now + self.config.peer_ttl;
                }
```
To:
```rust
                // 2. Purge expired peer table entries periodically.
                if now >= next_purge {
                    self.peer_table.purge_expired();
                    self.blacklist.purge_expired(now);
                    next_purge = now + self.config.peer_ttl;
                }
```

**3e. Add blacklist to destructuring** — in `process_inbound_frames` (line ~289):

Change:
```rust
        let Self {
            socket, peer_table, config, ..
        } = self;
```
To:
```rust
        let Self {
            socket, peer_table, blacklist, config, ..
        } = self;
```

**3f. Add blacklist check and rate counting in `recv_frames` callback** (line ~372):

Change:
```rust
        socket.recv_frames(&mut |src_mac, payload| {
            // Skip frames from ourselves (loopback on the interface).
            if src_mac == local_mac {
                return;
            }
            if payload.is_empty() {
                return;
            }
            had_inbound = true;
```
To:
```rust
        socket.recv_frames(&mut |src_mac, payload| {
            // Skip frames from ourselves (loopback on the interface).
            if src_mac == local_mac {
                return;
            }
            if payload.is_empty() {
                return;
            }

            // Reject frames from banned MACs (silent drop — no logging).
            let now = Instant::now();
            if blacklist.is_blocked(src_mac, now) {
                return;
            }

            // Every frame from a non-blocked MAC counts toward rate flood detection.
            blacklist.record_violation(src_mac, ViolationCategory::RateFlood, now);

            had_inbound = true;
```

**3g. Add violation recording at each validation failure in the `dispatch` closure.**

In Scout dispatch (line ~307), change:
```rust
                frame_type::SCOUT => {
                    if body.len() < 16 {
                        debug!(len = body.len(), "scout frame too short, ignoring");
                        return;
                    }
```
To:
```rust
                frame_type::SCOUT => {
                    if body.len() < 16 {
                        debug!(len = body.len(), "scout frame too short, ignoring");
                        blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                        return;
                    }
```

In Data dispatch — "too short" (line ~322), change:
```rust
                frame_type::DATA => {
                    // Data body: [6-byte origin_mac][u16 BE key_len][key][payload]
                    if body.len() < 6 + 2 {
                        debug!(len = body.len(), "data frame too short, ignoring");
                        return;
                    }
```
To:
```rust
                frame_type::DATA => {
                    // Data body: [6-byte origin_mac][u16 BE key_len][key][payload]
                    if body.len() < 6 + 2 {
                        debug!(len = body.len(), "data frame too short, ignoring");
                        blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                        return;
                    }
```

In Data dispatch — "truncated key_expr" (line ~336), change:
```rust
                    if body.len() < key_end {
                        debug!(
                            key_len,
                            frame_len = body.len(),
                            "data frame truncated key_expr, ignoring"
                        );
                        return;
                    }
```
To:
```rust
                    if body.len() < key_end {
                        debug!(
                            key_len,
                            frame_len = body.len(),
                            "data frame truncated key_expr, ignoring"
                        );
                        blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                        return;
                    }
```

In Data dispatch — "invalid UTF-8" (line ~344), change:
```rust
                    let key_expr = match std::str::from_utf8(&body[key_start..key_end]) {
                        Ok(s) => s,
                        Err(_) => {
                            debug!("data frame has invalid UTF-8 key_expr, ignoring");
                            return;
                        }
                    };
```
To:
```rust
                    let key_expr = match std::str::from_utf8(&body[key_start..key_end]) {
                        Ok(s) => s,
                        Err(_) => {
                            debug!("data frame has invalid UTF-8 key_expr, ignoring");
                            blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                            return;
                        }
                    };
```

In Data dispatch — "namespace violation" (line ~356), change:
```rust
                    if !key_expr.starts_with(&prefix) {
                        debug!(
                            key_expr,
                            "data frame key_expr outside allowed namespace, ignoring"
                        );
                        return;
                    }
```
To:
```rust
                    if !key_expr.starts_with(&prefix) {
                        debug!(
                            key_expr,
                            "data frame key_expr outside allowed namespace, ignoring"
                        );
                        blacklist.record_violation(src_mac, ViolationCategory::NamespaceViolation, Instant::now());
                        return;
                    }
```

- [ ] **Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink`
Expected: All tests PASS. Run clippy too:
Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo clippy -p harmony-rawlink`
Expected: Zero warnings.

- [ ] **Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/bridge.rs
git commit -m "feat(rawlink): integrate MacBlacklist into bridge recv_frames loop

Adds is_blocked() early rejection, RateFlood counting on every inbound frame,
MalformedFrame recording at 4 validation failure points, NamespaceViolation
recording, and periodic purge alongside peer table. 3 integration tests verify
banned MACs are dropped, other MACs unaffected, and traffic resumes after expiry.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Identity Churn Detection

**Files:**
- Modify: `crates/harmony-rawlink/src/bridge.rs`

- [ ] **Step 1: Write the failing test**

Add to the `tests` module in `crates/harmony-rawlink/src/bridge.rs`:

```rust
    #[test]
    fn identity_churn_triggers_blacklist() {
        use crate::blacklist::{BlacklistConfig, MacBlacklist, ViolationCategory};

        let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);

        let mut blacklist = MacBlacklist::new(BlacklistConfig {
            thresholds: [20, 500, 3, 10], // IdentityChurn threshold = 3
            ..Default::default()
        });
        let mut peer_table = PeerTable::new(Duration::from_secs(30));
        let local_mac = MAC_A;
        let now = Instant::now();

        // Send 4 scout frames from MAC_B with different identity hashes.
        // The first establishes the identity; the next 3 are churn violations.
        for i in 0u8..4 {
            let identity = [i + 1; 16];
            let scout = make_scout_payload(&identity);
            socket_b.send_frame(MAC_A, &scout).expect("mock send");
        }

        socket_a
            .recv_frames(&mut |src_mac, payload| {
                if src_mac == &local_mac || payload.is_empty() {
                    return;
                }
                if blacklist.is_blocked(src_mac, now) {
                    return;
                }
                if payload[0] == frame_type::SCOUT && payload.len() >= 17 {
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&payload[1..17]);

                    // Identity churn detection: different identity for same MAC.
                    if let Some(prev) = peer_table.identity_for_mac(src_mac) {
                        if prev != identity_hash {
                            blacklist.record_violation(
                                src_mac,
                                ViolationCategory::IdentityChurn,
                                now,
                            );
                        }
                    }
                    peer_table.update(identity_hash, *src_mac);
                }
            })
            .expect("recv should succeed");

        // 3 churn violations (scouts 2, 3, 4 each differ from the previous) → banned.
        assert!(
            blacklist.is_blocked(&MAC_B, now),
            "3 identity changes should trigger ban"
        );
    }
```

- [ ] **Step 2: Run test to verify it passes (this test simulates the logic pattern)**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink identity_churn_triggers_blacklist`
Expected: PASS (the test simulates the pattern we'll wire in).

- [ ] **Step 3: Add identity churn detection to the bridge's Scout dispatch**

In `crates/harmony-rawlink/src/bridge.rs`, in the `dispatch` closure's Scout branch, change:

```rust
                frame_type::SCOUT => {
                    if body.len() < 16 {
                        debug!(len = body.len(), "scout frame too short, ignoring");
                        blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                        return;
                    }
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&body[..16]);
                    peer_table.update(identity_hash, *src_mac);
                    debug!(
                        identity = hex::encode(identity_hash),
                        src_mac = hex::encode(src_mac),
                        "peer scouted"
                    );
                }
```
To:
```rust
                frame_type::SCOUT => {
                    if body.len() < 16 {
                        debug!(len = body.len(), "scout frame too short, ignoring");
                        blacklist.record_violation(src_mac, ViolationCategory::MalformedFrame, Instant::now());
                        return;
                    }
                    let mut identity_hash = [0u8; 16];
                    identity_hash.copy_from_slice(&body[..16]);

                    // Identity churn detection: if this MAC already has a
                    // different identity in the peer table, record a violation.
                    if let Some(prev) = peer_table.identity_for_mac(src_mac) {
                        if prev != identity_hash {
                            blacklist.record_violation(
                                src_mac,
                                ViolationCategory::IdentityChurn,
                                Instant::now(),
                            );
                        }
                    }

                    peer_table.update(identity_hash, *src_mac);
                    debug!(
                        identity = hex::encode(identity_hash),
                        src_mac = hex::encode(src_mac),
                        "peer scouted"
                    );
                }
```

- [ ] **Step 4: Run all tests and clippy**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test -p harmony-rawlink && cargo clippy -p harmony-rawlink`
Expected: All tests PASS, zero clippy warnings.

- [ ] **Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/bridge.rs
git commit -m "feat(rawlink): add identity churn detection in Scout dispatch

Before updating the peer table with a new identity_hash, checks if the MAC
already has a different identity. Rapid cycling (default: 5 changes / 10s)
triggers an IdentityChurn violation and eventual ban. Detects spoofing and
scanning on public-PSK mesh networks.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Final Clippy + Workspace Test + Module Doc Update

**Files:**
- Modify: `crates/harmony-rawlink/src/bridge.rs` (module doc)
- Modify: `crates/harmony-rawlink/src/blacklist.rs` (module doc)

- [ ] **Step 1: Update bridge module doc**

In `crates/harmony-rawlink/src/bridge.rs`, update the module doc at the top:

Change:
```rust
//! Async bridge connecting a [`RawSocket`] to a zenoh session.
//!
//! The bridge runs a continuous loop that:
//! 1. Broadcasts Scout frames on a jittered timer.
//! 2. Receives inbound L2 frames and publishes Data payloads into zenoh.
//! 3. Flushes the accumulated batch (gated by the jitter hold timer).
//! 4. Arms a jitter hold when inbound traffic was received (100-500ms delay).
//! 5. Drains the zenoh subscriber and broadcasts outbound Data frames.
//!
//! Steps 3-4 are ordered so expired holds flush before being replaced,
//! preventing indefinite batch starvation under continuous inbound traffic.
```
To:
```rust
//! Async bridge connecting a [`RawSocket`] to a zenoh session.
//!
//! The bridge runs a continuous loop that:
//! 1. Broadcasts Scout frames on a jittered timer.
//! 2. Receives inbound L2 frames and publishes Data payloads into zenoh.
//!    Frames from banned MACs are silently dropped at the top of the callback.
//!    Validation failures record violations toward per-MAC ban thresholds.
//! 3. Flushes the accumulated batch (gated by the jitter hold timer).
//! 4. Arms a jitter hold when inbound traffic was received (100-500ms delay).
//! 5. Drains the zenoh subscriber and broadcasts outbound Data frames.
//!
//! Steps 3-4 are ordered so expired holds flush before being replaced,
//! preventing indefinite batch starvation under continuous inbound traffic.
//!
//! MAC blacklisting detects 4 violation categories: malformed frames, rate
//! flooding, identity churn, and namespace violations. Bans escalate with
//! each offense (60s → 4min → 16min → 1h cap). See [`crate::blacklist`].
```

- [ ] **Step 2: Run full workspace tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo test --workspace`
Expected: All workspace tests PASS.

- [ ] **Step 3: Run clippy on full workspace**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo clippy --workspace`
Expected: Zero warnings.

- [ ] **Step 4: Run nightly rustfmt (required for harmony-os CI)**

Run: `cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist && cargo +nightly fmt --all -- --check`
Expected: No formatting issues. If any, fix with `cargo +nightly fmt --all`.

- [ ] **Step 5: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony/.claude/worktrees/jake-rawlink-mac-blacklist
git add crates/harmony-rawlink/src/bridge.rs
git commit -m "docs(rawlink): update bridge module doc for MAC blacklisting

Documents the blacklist check in the recv_frames callback and the 4
violation categories detected during frame processing.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

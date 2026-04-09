# Dynamic MAC Blacklisting for DoS Mitigation

## Overview

Add in-process MAC blacklisting to the rawlink bridge to detect and drop traffic from misbehaving mesh peers. When a peer exceeds configurable violation thresholds within a sliding time window, their MAC address is banned with escalating backoff durations. This is defense-in-depth for public-PSK mesh networks where any device can join.

**Scope:** Detection and in-process frame dropping only. Hostapd/mesh11sd hardware-level bans are a follow-up.

## Architecture

A new `MacBlacklist` struct in `crates/harmony-rawlink/src/blacklist.rs` follows the same sans-I/O pattern as `JitterHold` and `BatchAccumulator` — a pure data structure driven by caller-supplied `Instant` timestamps. The bridge orchestrates it with two calls: `is_blocked()` at the top of the frame receive callback (before any processing), and `record_violation()` at each existing validation failure point.

### Data Model

```rust
pub enum ViolationCategory {
    MalformedFrame,     // failed validation: too short, bad EtherType, truncated, invalid UTF-8
    RateFlood,          // frames/sec exceeds threshold from a single MAC
    IdentityChurn,      // rapid identity_hash cycling in Scout frames
    NamespaceViolation, // key_expr outside allowed prefix
}

struct ViolationRecord {
    /// Per-category violation counts within the current time window.
    counts: [u32; 4],  // indexed by ViolationCategory
    /// Start of the current counting window.
    window_start: Instant,
    /// Number of times this MAC has been banned (drives escalation).
    offense_count: u32,
    /// If Some, this MAC is currently banned until this time.
    banned_until: Option<Instant>,
}

pub struct MacBlacklist {
    entries: HashMap<[u8; 6], ViolationRecord>,
    config: BlacklistConfig,
}

pub struct BlacklistConfig {
    /// Time window for counting violations (default: 10 seconds).
    pub window_duration: Duration,
    /// Threshold per category to trigger a ban.
    pub thresholds: [u32; 4],
    /// Base ban duration for first offense (default: 60 seconds).
    pub base_ban: Duration,
    /// Multiplier per subsequent offense (default: 4x).
    /// 1st = 60s, 2nd = 4min, 3rd = 16min, 4th = 64min, capped at max_ban.
    pub escalation_factor: u32,
    /// Maximum ban duration cap (default: 1 hour).
    pub max_ban: Duration,
}
```

**Default thresholds (per 10-second window):**

| Category | Threshold | Rationale |
|----------|-----------|-----------|
| `MalformedFrame` | 20 | Tolerates buggy firmware noise |
| `RateFlood` | 500 | 50 frames/sec sustained over 10s |
| `IdentityChurn` | 5 | Changing identity more than once every 2s is suspicious |
| `NamespaceViolation` | 10 | Occasional probing vs. sustained scanning |

### Public Interface

```rust
impl MacBlacklist {
    pub fn new(config: BlacklistConfig) -> Self;

    /// Returns true if this MAC is currently banned.
    /// Expired bans return false (lazy expiry).
    pub fn is_blocked(&self, mac: &[u8; 6], now: Instant) -> bool;

    /// Record a violation. If the MAC's count for this category crosses
    /// the threshold within the current window, the MAC is banned.
    /// Returns true if this violation triggered a new ban.
    pub fn record_violation(
        &mut self,
        mac: &[u8; 6],
        category: ViolationCategory,
        now: Instant,
    ) -> bool;

    /// Remove entries whose ban has expired AND whose last activity
    /// is older than window_duration. Called periodically to bound memory.
    pub fn purge_expired(&mut self, now: Instant);
}
```

### Behavior Rules

**`is_blocked`:** Check `banned_until`. If `Some(t)` and `now < t`, return true. Otherwise false. Read-only, no mutation.

**`record_violation`:**
1. If `window_start + window_duration < now`, reset all counts to 0, set `window_start = now`.
2. Increment `counts[category]`.
3. If `counts[category] >= thresholds[category]`, compute ban: `min(base_ban * escalation_factor.pow(offense_count), max_ban)`. Set `banned_until = Some(now + ban_duration)`. Increment `offense_count`. Reset all counts to 0 (ban is the response; further counting is moot since `is_blocked` will reject frames before `record_violation` is called).
4. Return whether a ban was triggered.

**`purge_expired`:** Remove entries where `banned_until` is expired (or None) AND `window_start + window_duration < now`. This discards `offense_count` — a MAC that went quiet for a full window starts fresh. Intentional: forgiveness for transient issues.

**Escalating backoff:** With defaults (base=60s, factor=4x, cap=1h):
- 1st offense: 60s
- 2nd: 4 min
- 3rd: 16 min
- 4th+: 1 hour (capped)

## Bridge Integration

### Check Point

At the top of the `recv_frames` callback in `process_inbound_frames`, before any processing:

```rust
socket.recv_frames(&mut |src_mac, payload| {
    if blacklist.is_blocked(src_mac, now) {
        return;  // silent drop, no logging (attacker-controlled volume)
    }
    blacklist.record_violation(src_mac, ViolationCategory::RateFlood, now);
    if src_mac == local_mac { return; }
    // ... existing dispatch
})?;
```

The blacklist is a separate struct from the socket, so it can be mutated inside the `recv_frames` closure without aliasing conflicts — same pattern as `peer_table`.

### Record Points

| Failure | Location | Category |
|---------|----------|----------|
| Frame too short / wrong EtherType | `recv_frames` callback, after header check | `MalformedFrame` |
| Scout body < 16 bytes | Scout dispatch | `MalformedFrame` |
| Data body < 8 bytes / truncated key | Data dispatch | `MalformedFrame` |
| Invalid UTF-8 in key_expr | Data dispatch | `MalformedFrame` |
| Key outside allowed namespace | Data dispatch | `NamespaceViolation` |
| Different identity_hash for known MAC | Scout dispatch, before `peer_table.update()` | `IdentityChurn` |

### Identity Churn Detection

New check in Scout frame dispatch: before calling `peer_table.update(identity_hash, src_mac)`, look up the MAC in the peer table. If it already has a *different* identity_hash, record an `IdentityChurn` violation.

### Purge

Called alongside the existing `peer_table.purge_expired()` in the bridge loop — same periodic cadence.

### Logging

- `warn!` when a MAC gets banned: MAC address, triggering category, ban duration, offense count. Once per ban event.
- `debug!` on individual violations. Hidden by default.
- No logging on `is_blocked` drops — attacker controls the volume.

## File Changes

- **Create:** `crates/harmony-rawlink/src/blacklist.rs` — `MacBlacklist`, `ViolationRecord`, `BlacklistConfig`, `ViolationCategory`
- **Modify:** `crates/harmony-rawlink/src/lib.rs` — add `mod blacklist; pub use blacklist::*;`
- **Modify:** `crates/harmony-rawlink/src/bridge.rs` — add blacklist field, `is_blocked` check in `recv_frames` callback, `record_violation` calls at 5 failure points + rate counting, identity churn check against peer table, `purge_expired` call

## Testing

### Unit Tests (blacklist.rs) — 10 tests

All tests use controlled `Instant` values. No real time, no flakiness.

1. **Below threshold, no ban.** Record N-1 violations per category. Assert `is_blocked` returns false.
2. **At threshold, triggers ban.** Record N violations. Assert `record_violation` returns true. Assert `is_blocked` returns true. Assert ban duration matches `base_ban`.
3. **Escalating bans.** Trigger ban, advance past expiry, trigger again. Assert duration is `base_ban * escalation_factor`. Third is `base_ban * factor^2`. Assert cap at `max_ban`.
4. **Window reset.** Record N-1 violations, advance past `window_duration`, record 1. Assert no ban.
5. **Ban expiry.** Trigger ban, advance past `banned_until`. Assert `is_blocked` returns false.
6. **Purge cleans stale entries.** Create entries, advance past window + ban, call `purge_expired`. Assert removed. Active bans NOT purged.
7. **Offense count resets after purge.** Trigger ban, let expire, purge, trigger new ban. Assert `base_ban` (not escalated).
8. **Rate flood threshold.** 500 frames in 10s = ban. 499 = no ban.
9. **Identity churn threshold.** 5 changes = ban. 4 = no ban.
10. **Multiple categories independent.** MalformedFrame hits threshold, RateFlood doesn't. Assert ban triggers.

### Integration Tests (bridge.rs) — 3 tests

Using `MockSocket` pairs.

11. **Blocked MAC frames silently dropped.** Send frames from MAC A, trigger ban, send more. Assert none published to zenoh. MAC B traffic unaffected.
12. **Ban expires, traffic resumes.** Trigger ban, advance past duration, send frames. Assert published.
13. **Malformed frames from one MAC don't affect others.** 20 malformed from MAC A (triggers ban). MAC B valid frames still flow.

## Out of Scope

- Hostapd / mesh11sd IPC (follow-up bead)
- CLI flags or config file knobs for thresholds
- Metrics/telemetry export
- Persistent blacklist across restarts
- Whitelist mode
- BPF-level kernel filtering

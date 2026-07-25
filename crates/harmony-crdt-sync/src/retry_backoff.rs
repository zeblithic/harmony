// SPDX-License-Identifier: Apache-2.0 OR MIT
//! The escalating retry-backoff *schedule*: when may the next attempt run?
//!
//! This is the smallest shareable piece of "an attempt failed; try again,
//! but not immediately and not forever at the same cadence". It owns the
//! delay arithmetic and the resulting instant, and nothing else — no
//! transport, no notion of what is being attempted, not even whether an
//! attempt is currently outstanding. Callers that need in-flight tracking
//! or request/reply semantics layer that on top; see [`BackfillLatch`] and
//! [`RootFetchLatch`], which do exactly that.
//!
//! [`BackfillLatch`]: crate::backfill_latch::BackfillLatch
//! [`RootFetchLatch`]: crate::backfill_latch::RootFetchLatch
//!
//! # Why this is its own type
//!
//! Two consumers already need the identical schedule for non-identical
//! reasons:
//!
//! - the **backfill latches**, where a request went unanswered because no
//!   holder was online, and
//! - a **publish retry**, where a snapshot publish failed against an
//!   unhealthy transport (ZEB-761).
//!
//! Those differ in everything except the delay rule. Restating the rule per
//! consumer is how two copies of one schedule drift apart, so the rule has
//! exactly one home and each consumer composes it.
//!
//! # The rule
//!
//! The first failure waits [`RETRY_BASE_MS`], and each *consecutive* failure
//! doubles the wait up to [`RETRY_CAP_MS`]. Reaching the cap is not
//! give-up: the schedule stays at the cap indefinitely, because a caller
//! that stops retrying has silently dropped the work. A success clears the
//! escalation, so an intermittent failure does not inherit the delay earned
//! by an earlier outage.
//!
//! ```
//! use harmony_crdt_sync::RetryBackoff;
//!
//! let mut backoff = RetryBackoff::new(30_000, 600_000);
//! assert_eq!(backoff.pending_at(), None); // nothing owed yet
//!
//! // Two consecutive failures: 30 s, then 60 s from the failure instant.
//! assert_eq!(backoff.on_failure(1_000), 31_000);
//! assert_eq!(backoff.on_failure(31_000), 91_000);
//! assert_eq!(backoff.pending_at(), Some(91_000));
//!
//! // A success drops the escalation; the next failure starts over at base.
//! backoff.clear(91_000);
//! assert_eq!(backoff.pending_at(), None);
//! assert_eq!(backoff.on_failure(100_000), 130_000);
//! ```

/// First retry delay after a failed attempt (30 s).
pub const RETRY_BASE_MS: u64 = 30_000;

/// Maximum delay between retry attempts (600 s). Doubling stops here; the
/// schedule stays at this cadence for as long as attempts keep failing.
pub const RETRY_CAP_MS: u64 = 600_000;

/// One escalation step: the first delay is `base` clamped to `cap` (a
/// misconfigured `base > cap` must not violate the cap), then doubles per
/// consecutive failure up to `cap`.
///
/// The doubling saturates: with a caller-injected `cap_ms` near `u64::MAX`
/// an unchecked `* 2` could overflow (a debug-build panic, or a release
/// wrap that clamps *below* the intended delay). `saturating_mul` keeps the
/// clamp-to-`cap` semantics intact at any magnitude.
fn escalate(current_delay_ms: u64, base_ms: u64, cap_ms: u64) -> u64 {
    if current_delay_ms == 0 {
        base_ms.min(cap_ms)
    } else {
        current_delay_ms.saturating_mul(2).min(cap_ms)
    }
}

/// An escalating retry-backoff schedule.
///
/// Feed it failures ([`on_failure`](Self::on_failure)) and successes
/// ([`clear`](Self::clear)); read back the instant the next attempt may run
/// ([`next_at`](Self::next_at) / [`pending_at`](Self::pending_at)).
///
/// Time is caller-supplied milliseconds on whatever monotonic scale the
/// driver already uses — the schedule never reads a clock itself, so it is
/// runtime-free and testable without I/O.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RetryBackoff {
    /// Current delay (ms); `0` = no consecutive failure outstanding.
    delay_ms: u64,
    /// Earliest instant (ms) the next attempt may run. Meaningful only
    /// while `delay_ms > 0`; see [`pending_at`](Self::pending_at).
    next_at: u64,
    /// First-delay after a failure (ms). Production = [`RETRY_BASE_MS`].
    base_ms: u64,
    /// Escalation ceiling (ms). Production = [`RETRY_CAP_MS`].
    cap_ms: u64,
}

impl RetryBackoff {
    /// A cleared schedule with the given base/cap, in milliseconds.
    ///
    /// Use [`RETRY_BASE_MS`] / [`RETRY_CAP_MS`] for the production
    /// schedule, or inject a compressed one in tests so a backoff
    /// assertion does not cost wall-clock seconds.
    pub fn new(base_ms: u64, cap_ms: u64) -> Self {
        Self {
            delay_ms: 0,
            next_at: 0,
            base_ms,
            cap_ms,
        }
    }

    /// Record a failed attempt at `now_ms`: escalate the delay and arm the
    /// next instant. Returns that instant, which is also readable later via
    /// [`next_at`](Self::next_at).
    ///
    /// The arming saturates, so a huge injected delay cannot wrap
    /// `next_at` into the past — which would defeat the backoff entirely
    /// and turn a retry loop into a spin.
    pub fn on_failure(&mut self, now_ms: u64) -> u64 {
        self.delay_ms = escalate(self.delay_ms, self.base_ms, self.cap_ms);
        self.next_at = now_ms.saturating_add(self.delay_ms);
        self.next_at
    }

    /// Drop the escalation: no retry is owed, and the next failure starts
    /// again at [`base_ms`](Self::new).
    ///
    /// `now_ms` becomes the earliest-next-attempt instant, so a caller that
    /// reads [`next_at`](Self::next_at) unconditionally sees "now" rather
    /// than a stale past instant.
    pub fn clear(&mut self, now_ms: u64) {
        self.delay_ms = 0;
        self.next_at = now_ms;
    }

    /// The earliest instant (ms) at which the next attempt may run,
    /// regardless of whether one is owed.
    ///
    /// Prefer [`pending_at`](Self::pending_at) when scheduling a wakeup —
    /// this accessor cannot distinguish "retry at T" from "nothing owed,
    /// and T is when we last cleared".
    pub fn next_at(&self) -> u64 {
        self.next_at
    }

    /// The instant a retry is owed at, or `None` when none is.
    ///
    /// This is the scheduling accessor: `None` means the caller should not
    /// arm a retry wakeup at all, which is what keeps an idle driver from
    /// waking on a schedule it has already satisfied.
    pub fn pending_at(&self) -> Option<u64> {
        (self.delay_ms > 0).then_some(self.next_at)
    }

    /// The current delay (ms); `0` when cleared. Exposed for observability
    /// and for tests asserting that the escalation actually escalates.
    pub fn delay_ms(&self) -> u64 {
        self.delay_ms
    }

    /// The configured first-failure delay (ms).
    ///
    /// Readable so a caller that rebuilds itself on a transport-recovery
    /// reset can carry its schedule across, rather than silently reverting
    /// an injected one to the production default.
    pub fn base_ms(&self) -> u64 {
        self.base_ms
    }

    /// The configured escalation ceiling (ms). See
    /// [`base_ms`](Self::base_ms).
    pub fn cap_ms(&self) -> u64 {
        self.cap_ms
    }
}

impl Default for RetryBackoff {
    /// The production schedule: [`RETRY_BASE_MS`] → [`RETRY_CAP_MS`].
    fn default() -> Self {
        Self::new(RETRY_BASE_MS, RETRY_CAP_MS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fresh_schedule_owes_nothing() {
        let b = RetryBackoff::new(30_000, 600_000);
        assert_eq!(b.pending_at(), None);
        assert_eq!(b.delay_ms(), 0);
        // `next_at` is readable but meaningless before the first failure.
        assert_eq!(b.next_at(), 0);
    }

    #[test]
    fn consecutive_failures_double_up_to_the_cap_then_hold() {
        let mut b = RetryBackoff::new(100, 400);
        // base, then doubling, then clamped at the cap forever.
        assert_eq!(b.on_failure(0), 100);
        assert_eq!(b.on_failure(0), 200);
        assert_eq!(b.on_failure(0), 400);
        assert_eq!(b.on_failure(0), 400);
        assert_eq!(b.on_failure(0), 400);
        // Reaching the cap is not give-up: a retry is still owed.
        assert_eq!(b.pending_at(), Some(400));
    }

    #[test]
    fn the_armed_instant_is_relative_to_the_failure_not_the_epoch() {
        let mut b = RetryBackoff::new(100, 400);
        assert_eq!(b.on_failure(5_000), 5_100);
        assert_eq!(b.on_failure(9_000), 9_200);
    }

    #[test]
    fn clear_drops_the_escalation_so_the_next_failure_restarts_at_base() {
        let mut b = RetryBackoff::new(100, 400);
        b.on_failure(0);
        b.on_failure(0);
        assert_eq!(b.delay_ms(), 200);

        b.clear(1_000);
        assert_eq!(b.pending_at(), None);
        assert_eq!(b.delay_ms(), 0);
        // `clear` parks the instant at `now`, never in the past.
        assert_eq!(b.next_at(), 1_000);

        // An intermittent failure must not inherit the delay an earlier
        // outage earned.
        assert_eq!(b.on_failure(1_000), 1_100);
    }

    #[test]
    fn a_base_above_the_cap_still_respects_the_cap() {
        // A misconfigured schedule must not out-wait its own ceiling.
        let mut b = RetryBackoff::new(10_000, 400);
        assert_eq!(b.on_failure(0), 400);
    }

    #[test]
    fn escalation_saturates_instead_of_wrapping_into_the_past() {
        // A wrapped `next_at` would land BEFORE `now`, so every poll would
        // fire immediately — the exact spin the backoff exists to prevent.
        let mut b = RetryBackoff::new(u64::MAX, u64::MAX);
        let armed = b.on_failure(10);
        assert_eq!(armed, u64::MAX);
        assert!(armed >= 10, "armed instant must never precede the failure");

        // And the doubling itself saturates rather than overflowing.
        let mut b = RetryBackoff::new(u64::MAX / 2 + 2, u64::MAX);
        b.on_failure(0);
        assert_eq!(b.on_failure(0), u64::MAX);
    }

    #[test]
    fn pending_at_tracks_owed_ness_not_merely_elapsed_time() {
        // The distinction `next_at` cannot make: after `clear`, the instant
        // is still readable but nothing is owed at it.
        let mut b = RetryBackoff::new(100, 400);
        b.on_failure(0);
        assert_eq!(b.pending_at(), Some(100));
        b.clear(100);
        assert_eq!(b.next_at(), 100);
        assert_eq!(b.pending_at(), None);
    }

    #[test]
    fn default_is_the_production_schedule() {
        let mut b = RetryBackoff::default();
        assert_eq!(b.on_failure(0), RETRY_BASE_MS);
        assert_eq!(b.on_failure(0), RETRY_BASE_MS * 2);
    }
}

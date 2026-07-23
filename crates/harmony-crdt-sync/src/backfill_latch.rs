// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Backfill/backoff latches: the pure decision cores for paginated
//! catch-up over an unreliable, best-effort request/reply transport.
//!
//! A device that has fallen behind (fresh join, reconnect after downtime)
//! asks online holders for the history it missed. The serving side caps
//! every reply page, and a query can go entirely unanswered when no holder
//! is online. These latches are the requesting side's *pure* state
//! machines: decisions go in (page outcomes, wall-clock milliseconds) and
//! actions come out ([`BackfillAction`] / [`RootFetchAction`]). They hold
//! no transport handle and no async runtime, so the full
//! paging/backoff/in-flight state space is testable without I/O. A caller
//! supplies an async *driver* that interprets the emitted actions —
//! issuing requests, sleeping until `WaitUntil`, and feeding outcomes back.
//!
//! Two shapes:
//!
//! - [`BackfillLatch<W>`] — paginated catch-up keyed by a watermark `W`
//!   (typically a hybrid logical clock). A completed short/empty page
//!   satisfies it; a full page that advances the watermark drives an
//!   immediate re-request (the paging loop); a full page that does not
//!   advance, or an unanswered query, arms an escalating backoff.
//! - [`RootFetchLatch`] — the page-less sibling for a single full-state
//!   pull, where any reply satisfies and zero replies means no responder.
//!
//! `W` is only ever cloned and equality-compared, never inspected, so a
//! domain watermark type keeps its own home (and any wire-format contract)
//! in the caller's crate; the latch stays generic and domain-free.
//!
//! Provenance: extracted from harmony-client's `channel_backfill.rs`
//! (spec D24 for the paging latch, D3/D6/D7 for the root-fetch latch),
//! byte-for-behavior preserving. The async drivers, transport-epoch
//! re-arm cooldown, and reconcile-mode selection stay caller-side.

/// First retry delay after an unanswered request (30 s).
pub const BACKFILL_RETRY_BASE_MS: u64 = 30_000;

/// Maximum delay between retry attempts (600 s). Doubling stops here; the
/// latch retries forever at this cadence until answered.
pub const BACKFILL_RETRY_CAP_MS: u64 = 600_000;

/// What a paginated backfill driver should do next, as decided by
/// [`BackfillLatch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackfillAction<W> {
    /// Send a backfill request for events strictly after `since`
    /// (`None` = from the beginning of history).
    Request { since: Option<W> },
    /// Nothing to do until the given wall-clock instant (ms); re-poll
    /// `next_action` at or after that time.
    WaitUntil(u64),
    /// The latch is satisfied — no further requests needed until
    /// [`BackfillLatch::reset`].
    Idle,
}

/// Result of a *completed* backfill reply page (a holder answered).
///
/// An unanswered query is NOT a `PageOutcome` — report that via
/// [`BackfillLatch::on_no_reply`] instead.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PageOutcome<W> {
    /// Number of events the page carried.
    pub events: usize,
    /// Maximum watermark among the page's events (`None` for an empty page).
    pub max_hlc_seen: Option<W>,
    /// The per-request limit the serving side was asked for (page cap).
    pub limit: usize,
}

/// Retry latch + paging state machine for one paginated backfill.
///
/// Drive it by polling [`next_action`](Self::next_action) with the current
/// wall clock and feeding outcomes back via
/// [`on_page_complete`](Self::on_page_complete) /
/// [`on_no_reply`](Self::on_no_reply). At most one request is outstanding
/// at a time (in-flight guard).
///
/// `W` is the watermark type (e.g. a hybrid logical clock). It is only
/// cloned and equality-compared, so `W: Clone + PartialEq` suffices.
#[derive(Debug, Clone)]
pub struct BackfillLatch<W> {
    /// Request events strictly after this watermark (`None` = from start).
    since: Option<W>,
    /// Set by a completed short/empty page; cleared by `reset`.
    satisfied: bool,
    /// A `Request` has been handed out and neither `on_page_complete`
    /// nor `on_no_reply` has been called yet.
    in_flight: bool,
    /// Earliest wall-clock ms at which the next request may be sent.
    next_retry_at: u64,
    /// Current backoff delay (ms); 0 = no consecutive no-reply yet.
    retry_delay_ms: u64,
    /// First-retry delay after an unanswered request (ms). Production =
    /// [`BACKFILL_RETRY_BASE_MS`]; tests inject smaller values via
    /// [`Self::new_with_backoff`].
    retry_base_ms: u64,
    /// Backoff ceiling (ms). Production = [`BACKFILL_RETRY_CAP_MS`].
    retry_cap_ms: u64,
}

impl<W: Clone + PartialEq> BackfillLatch<W> {
    /// New, unsatisfied latch requesting history after `watermark`, with
    /// the production (spec D24) backoff schedule.
    pub fn new(watermark: Option<W>) -> Self {
        Self::new_with_backoff(watermark, BACKFILL_RETRY_BASE_MS, BACKFILL_RETRY_CAP_MS)
    }

    /// New latch with an explicit backoff schedule (test-injectable; the
    /// spec D24 base/cap are the production values — see [`Self::new`]).
    pub fn new_with_backoff(watermark: Option<W>, base_ms: u64, cap_ms: u64) -> Self {
        Self {
            since: watermark,
            satisfied: false,
            in_flight: false,
            next_retry_at: 0,
            retry_delay_ms: 0,
            retry_base_ms: base_ms,
            retry_cap_ms: cap_ms,
        }
    }

    /// True once a completed short/empty page has answered the query.
    pub fn is_satisfied(&self) -> bool {
        self.satisfied
    }

    /// Decide the next driver action at wall-clock `now_ms`.
    pub fn next_action(&mut self, now_ms: u64) -> BackfillAction<W> {
        if self.satisfied {
            return BackfillAction::Idle;
        }
        if self.in_flight {
            // A request is already outstanding; nothing new until an
            // outcome lands. `next_retry_at` may sit in the past here
            // (it is only re-armed by `on_no_reply`), so clamp to `now`.
            return BackfillAction::WaitUntil(self.next_retry_at.max(now_ms));
        }
        if now_ms < self.next_retry_at {
            return BackfillAction::WaitUntil(self.next_retry_at);
        }
        self.in_flight = true;
        BackfillAction::Request {
            since: self.since.clone(),
        }
    }

    /// Record a completed reply page (clears in-flight).
    ///
    /// Short or empty page satisfies the latch (spec D24: a served
    /// "nothing" is an answer) and resets backoff. Full page (`events >=
    /// limit`, `limit > 0`) means more history may remain:
    ///
    /// - **Progress** (`max_hlc_seen` is `Some` and differs from the
    ///   current `since`): advance `since`, reset backoff, and leave the
    ///   latch unsatisfied so the next `next_action(now)` re-requests
    ///   immediately — the paging loop.
    /// - **No progress** (`max_hlc_seen` is `None` or equals the current
    ///   `since`): see the no-progress branch below.
    pub fn on_page_complete(&mut self, outcome: PageOutcome<W>, now_ms: u64) {
        self.in_flight = false;

        let full_page = outcome.limit > 0 && outcome.events >= outcome.limit;
        if !full_page {
            // Spec D24: a served "nothing more" is an answer.
            self.satisfied = true;
            self.retry_delay_ms = 0;
            self.next_retry_at = now_ms;
            return;
        }
        let progressed = outcome.max_hlc_seen.is_some() && outcome.max_hlc_seen != self.since;
        if progressed {
            // Paging loop: more history may remain behind the cap and the
            // verified watermark moved — re-request immediately from the
            // new window, resetting the no-reply backoff.
            self.since = outcome.max_hlc_seen;
            self.retry_delay_ms = 0;
            self.next_retry_at = now_ms;
        } else {
            // No-progress full page: the verified watermark did not move
            // past the window we asked for, so an immediate re-request
            // would replay the exact same window — a hostile holder
            // serving garbage that fails verification (or one that keeps
            // serving already-held duplicates) would otherwise drive a
            // tight zero-backoff request loop until shutdown. Arm the same
            // escalating backoff as [`Self::on_no_reply`] WITHOUT
            // satisfying the latch: history may genuinely remain, and the
            // holder set can change, so backing off (rather than declaring
            // done) keeps liveness.
            self.arm_backoff(now_ms);
        }
    }

    /// Record an unanswered request (clears in-flight, arms backoff).
    ///
    /// Delay schedule: `retry_base_ms` (production 30 s), doubling per
    /// consecutive no-reply, capped at `retry_cap_ms` (production 600 s);
    /// retries forever (the driver enforces shutdown).
    pub fn on_no_reply(&mut self, now_ms: u64) {
        self.in_flight = false;
        self.arm_backoff(now_ms);
    }

    /// Escalate the retry backoff and arm `next_retry_at`. Shared by
    /// [`Self::on_no_reply`] and the no-progress full-page branch of
    /// [`Self::on_page_complete`]. Delegates the step computation to
    /// `arm_backoff_step`.
    fn arm_backoff(&mut self, now_ms: u64) {
        self.retry_delay_ms =
            arm_backoff_step(self.retry_delay_ms, self.retry_base_ms, self.retry_cap_ms);
        self.next_retry_at = now_ms + self.retry_delay_ms;
    }

    /// Re-arm a satisfied latch with a new watermark (transport-recovery
    /// hook); clears in-flight and backoff state. Preserves the configured
    /// backoff schedule.
    pub fn reset(&mut self, watermark: Option<W>) {
        *self = Self::new_with_backoff(watermark, self.retry_base_ms, self.retry_cap_ms);
    }
}

/// One escalation step of the shared retry-backoff schedule: first retry
/// waits `base` clamped to `cap` (a misconfigured base > cap must not
/// violate the cap), then doubles per consecutive miss up to the cap.
/// Shared by [`BackfillLatch`] and [`RootFetchLatch`].
fn arm_backoff_step(current_delay_ms: u64, base_ms: u64, cap_ms: u64) -> u64 {
    if current_delay_ms == 0 {
        base_ms.min(cap_ms)
    } else {
        (current_delay_ms * 2).min(cap_ms)
    }
}

/// What a state-root fetch driver should do next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RootFetchAction {
    /// Send one state-root query (full-state exchange — no `since`).
    Request,
    /// Re-poll `next_action` at or after this wall-clock ms.
    WaitUntil(u64),
    /// Satisfied — park until `reset()` (transport-recovery re-arm).
    Idle,
}

/// Retry latch for a single full-state (root) pull.
///
/// Page-less sibling of [`BackfillLatch`]: a responder always has a root,
/// so ≥1 reply satisfies and zero replies means no responder. Shares the
/// spec backoff schedule (30 s base doubling to a 600 s cap, retrying
/// forever — the driver enforces shutdown).
#[derive(Debug, Clone)]
pub struct RootFetchLatch {
    satisfied: bool,
    in_flight: bool,
    next_retry_at: u64,
    retry_delay_ms: u64,
    retry_base_ms: u64,
    retry_cap_ms: u64,
}

impl RootFetchLatch {
    /// New, unsatisfied latch with the production (spec D3) backoff schedule.
    pub fn new() -> Self {
        Self::new_with_backoff(BACKFILL_RETRY_BASE_MS, BACKFILL_RETRY_CAP_MS)
    }

    /// New latch with an explicit backoff schedule (test-injectable;
    /// mirrors [`BackfillLatch::new_with_backoff`]).
    pub fn new_with_backoff(base_ms: u64, cap_ms: u64) -> Self {
        Self {
            satisfied: false,
            in_flight: false,
            next_retry_at: 0,
            retry_delay_ms: 0,
            retry_base_ms: base_ms,
            retry_cap_ms: cap_ms,
        }
    }

    /// True once at least one responder has replied.
    pub fn is_satisfied(&self) -> bool {
        self.satisfied
    }

    /// Decide the next driver action at wall-clock `now_ms`.
    pub fn next_action(&mut self, now_ms: u64) -> RootFetchAction {
        if self.satisfied {
            return RootFetchAction::Idle;
        }
        if self.in_flight {
            return RootFetchAction::WaitUntil(self.next_retry_at.max(now_ms));
        }
        if now_ms < self.next_retry_at {
            return RootFetchAction::WaitUntil(self.next_retry_at);
        }
        self.in_flight = true;
        RootFetchAction::Request
    }

    /// ≥1 responder replied: satisfied, backoff cleared.
    pub fn on_reply(&mut self, now_ms: u64) {
        self.in_flight = false;
        self.satisfied = true;
        self.retry_delay_ms = 0;
        self.next_retry_at = now_ms;
    }

    /// Zero responders: arm the escalating backoff (same schedule and
    /// clamp semantics as `BackfillLatch::arm_backoff`).
    pub fn on_no_reply(&mut self, now_ms: u64) {
        self.in_flight = false;
        self.retry_delay_ms =
            arm_backoff_step(self.retry_delay_ms, self.retry_base_ms, self.retry_cap_ms);
        self.next_retry_at = now_ms + self.retry_delay_ms;
    }

    /// Transport-recovery re-arm: unsatisfy, clear in-flight + backoff.
    pub fn reset(&mut self) {
        *self = Self::new_with_backoff(self.retry_base_ms, self.retry_cap_ms);
    }
}

impl Default for RootFetchLatch {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A small stand-in watermark: the latch only clones + equality-compares
    // it, so `u64` exercises the exact same decision logic the client's
    // `Hlc` watermark does.
    type W = u64;

    fn full(events: usize, max: Option<W>) -> PageOutcome<W> {
        PageOutcome {
            events,
            max_hlc_seen: max,
            limit: events,
        }
    }

    fn short(events: usize, limit: usize, max: Option<W>) -> PageOutcome<W> {
        PageOutcome {
            events,
            max_hlc_seen: max,
            limit,
        }
    }

    #[test]
    fn in_flight_guard_blocks_second_request() {
        let mut l = BackfillLatch::<W>::new(Some(0));
        assert_eq!(
            l.next_action(500),
            BackfillAction::Request { since: Some(0) }
        );
        // A request is outstanding: no new request, wait (clamped to now).
        assert_eq!(l.next_action(600), BackfillAction::WaitUntil(600));
    }

    #[test]
    fn short_page_satisfies_then_idle() {
        let mut l = BackfillLatch::<W>::new(Some(0));
        assert_eq!(l.next_action(0), BackfillAction::Request { since: Some(0) });
        l.on_page_complete(short(3, 1000, Some(2)), 100);
        assert!(l.is_satisfied());
        assert_eq!(l.next_action(100), BackfillAction::Idle);
    }

    #[test]
    fn reset_rearms_with_new_watermark() {
        let mut l = BackfillLatch::<W>::new(Some(0));
        l.next_action(0);
        l.on_page_complete(short(0, 1000, None), 100);
        assert_eq!(l.next_action(100), BackfillAction::Idle);
        l.reset(Some(9));
        assert_eq!(
            l.next_action(200),
            BackfillAction::Request { since: Some(9) }
        );
    }

    #[test]
    fn arm_backoff_step_clamps_and_doubles() {
        // base > cap must never violate the cap.
        assert_eq!(arm_backoff_step(0, 500, 400), 400);
        // first step from base.
        assert_eq!(arm_backoff_step(0, 100, 400), 100);
        // doubling.
        assert_eq!(arm_backoff_step(100, 100, 400), 200);
        assert_eq!(arm_backoff_step(200, 100, 400), 400);
        // capped.
        assert_eq!(arm_backoff_step(400, 100, 400), 400);
    }

    /// Golden decision-sequence: a fixed script of polls + outcomes must
    /// produce this exact action list. Any change to the paging loop,
    /// backoff schedule, cap clamp, satisfaction, or reset semantics breaks
    /// it. The harmony-client behavior tests pin the same logic against the
    /// production `Hlc` watermark; this is the domain-free anchor.
    #[test]
    fn golden_backfill_decision_sequence() {
        let mut l = BackfillLatch::<W>::new_with_backoff(Some(0), 100, 400);
        let mut actions = Vec::new();

        actions.push(l.next_action(1000)); // A: initial request
        l.on_page_complete(full(1000, Some(5)), 1000); // progress 0 -> 5
        actions.push(l.next_action(1000)); // B: paging loop, immediate
        l.on_page_complete(full(1000, Some(5)), 2000); // full but no progress
        actions.push(l.next_action(2000)); // C: no-progress backoff (100)
        actions.push(l.next_action(2100)); // D: retry
        l.on_no_reply(2100); // backoff 100 -> 200
        actions.push(l.next_action(2200)); // E: wait (200)
        actions.push(l.next_action(2300)); // F: retry
        l.on_no_reply(2300); // backoff 200 -> 400
        actions.push(l.next_action(2400)); // G: wait (400)
        actions.push(l.next_action(2700)); // H: retry
        l.on_no_reply(2700); // backoff 400 -> 400 (capped)
        actions.push(l.next_action(2800)); // I: wait, cap holds (400)
        actions.push(l.next_action(3100)); // J: retry
        l.on_page_complete(short(10, 1000, Some(7)), 3100); // satisfied
        actions.push(l.next_action(3100)); // K: idle
        l.reset(Some(7)); // transport-recovery re-arm
        actions.push(l.next_action(3200)); // L: request from new watermark

        assert_eq!(
            actions,
            vec![
                BackfillAction::Request { since: Some(0) }, // A
                BackfillAction::Request { since: Some(5) }, // B
                BackfillAction::WaitUntil(2100),            // C
                BackfillAction::Request { since: Some(5) }, // D
                BackfillAction::WaitUntil(2300),            // E
                BackfillAction::Request { since: Some(5) }, // F
                BackfillAction::WaitUntil(2700),            // G
                BackfillAction::Request { since: Some(5) }, // H
                BackfillAction::WaitUntil(3100),            // I
                BackfillAction::Request { since: Some(5) }, // J
                BackfillAction::Idle,                       // K
                BackfillAction::Request { since: Some(7) }, // L
            ]
        );
    }

    #[test]
    fn root_fetch_in_flight_guard() {
        let mut l = RootFetchLatch::new();
        assert_eq!(l.next_action(500), RootFetchAction::Request);
        assert_eq!(l.next_action(600), RootFetchAction::WaitUntil(600));
    }

    /// Golden decision-sequence for the page-less root-fetch latch.
    #[test]
    fn golden_root_fetch_decision_sequence() {
        let mut l = RootFetchLatch::new_with_backoff(100, 400);
        let mut actions = Vec::new();

        actions.push(l.next_action(1000)); // Request
        l.on_no_reply(1000); // backoff -> 100
        actions.push(l.next_action(1050)); // WaitUntil(1100)
        actions.push(l.next_action(1100)); // Request
        l.on_no_reply(1100); // backoff -> 200
        actions.push(l.next_action(1200)); // WaitUntil(1300)
        actions.push(l.next_action(1300)); // Request
        l.on_reply(1300); // satisfied
        actions.push(l.next_action(1300)); // Idle
        l.reset(); // re-arm
        actions.push(l.next_action(1400)); // Request

        assert_eq!(
            actions,
            vec![
                RootFetchAction::Request,
                RootFetchAction::WaitUntil(1100),
                RootFetchAction::Request,
                RootFetchAction::WaitUntil(1300),
                RootFetchAction::Request,
                RootFetchAction::Idle,
                RootFetchAction::Request,
            ]
        );
    }
}

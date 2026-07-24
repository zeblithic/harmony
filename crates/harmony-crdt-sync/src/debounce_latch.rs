// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Debounce latch: the pure decision core for "a local mutation happened —
//! when should we publish, and what happens if that publish fails?"
//!
//! A replicated dataset that published on every mutation would emit one
//! encrypt + store + broadcast per keystroke. Collapsing a burst into a
//! single publish needs a **sliding** debounce: each mutation pushes the
//! deadline out, so the publish lands once the writer pauses rather than on
//! a fixed cadence from the first edit.
//!
//! [`DebounceLatch`] owns that window and nothing else — no timer, no
//! transport, no clock. Milliseconds go in, a [`deadline`](DebounceLatch::deadline)
//! comes out, and the caller's driver sleeps until it and reports back. The
//! same latch also serves the two paths that bypass the window: an explicit
//! flush (always publishes, even idle) and shutdown (publishes only if there
//! is genuinely unpublished work).
//!
//! # The dirty signal belongs to the caller
//!
//! Notably the latch does **not** hold the dirty flag. In a real engine that
//! flag is written by every mutation site in the application — in the donor,
//! 144 of them — from arbitrary tasks, against an engine loop that is
//! meanwhile sleeping on the deadline. That makes it shared, concurrently
//! written state, typically an `AtomicBool` behind an `Arc`, and a kernel
//! cannot own state its callers must write without the kernel. A latch that
//! kept its own `bool` would force the caller to mirror the atomic into it,
//! and the two would disagree the moment a mutation landed between the two
//! clears — strictly worse than no kernel at all.
//!
//! So the caller owns the bit and the latch owns the *rule*: pass what the
//! claim took, get back whether to restore it.
//!
//! # Claim and settle
//!
//! The failure mode this discipline exists to prevent: a publish path that
//! clears the dirty flag *before* publishing, and then loses the publish. The
//! signal is consumed, nothing was replicated, and the state sits dirty with
//! no pending deadline — so the next shutdown flush sees a clean latch and
//! skips it. The mutation is persisted locally but never reaches a peer.
//!
//! Clearing the flag up front is not optional (a mutation arriving *during*
//! the publish must re-arm, not be swallowed by the completing publish), so
//! the latch makes the restore hard to skip instead: each publish path
//! returns a `#[must_use]` [`PublishClaim`] carrying the signal the caller
//! took, and [`PublishClaim::settle`] consumes it and returns a `#[must_use]`
//! [`DirtySignal`] saying whether to put the signal back. Both halves are
//! `#[must_use]`, so neither dropping the claim nor ignoring the verdict
//! passes review silently.
//!
//! Restoring the *signal* is not the same as scheduling a *retry*, and this
//! latch does only the former — see [`PublishClaim::settle`] for why
//! self-scheduling here would spin, and what to compose instead.
//!
//! ```
//! use harmony_crdt_sync::debounce_latch::{DebounceLatch, DirtySignal, PublishOutcome};
//! use core::sync::atomic::{AtomicBool, Ordering};
//!
//! // The caller's dirty flag: written by mutation sites on other tasks.
//! let dirty = AtomicBool::new(false);
//! let mut latch = DebounceLatch::new(250);
//! assert_eq!(latch.deadline(), None, "idle until a mutation arrives");
//!
//! // A mutation elsewhere sets the flag and pokes the engine.
//! dirty.store(true, Ordering::Release);
//! latch.mark_dirty(1_000);
//! assert_eq!(latch.deadline(), Some(1_250));
//!
//! // A second mutation slides the window rather than publishing twice.
//! latch.mark_dirty(1_100);
//! assert_eq!(latch.deadline(), Some(1_350));
//!
//! // The driver slept to the deadline. Take the signal, then publish.
//! let claim = latch.on_deadline(dirty.swap(false, Ordering::AcqRel));
//! assert!(claim.should_publish());
//!
//! // The publish failed, so the caller is told to restore the signal — but
//! // no deadline is re-armed: the retry rides the next mutation, flush, or
//! // shutdown rather than spinning on an instant that has already passed.
//! if claim.settle(PublishOutcome::Failed) == DirtySignal::Restore {
//!     dirty.store(true, Ordering::Release);
//! }
//! assert!(dirty.load(Ordering::Acquire));
//! assert_eq!(latch.deadline(), None);
//! ```
//!
//! Provenance: extracted from harmony-client's `fleet_sync::internal_task`
//! select loop (ZEB-571 item 6b), behavior-preserving; `community_state_sync`
//! runs the same loop and is the second adopter. The `Notify`/timer plumbing,
//! the dirty flag itself, the publish, and the persist that follows it all
//! stay caller-side.

/// How a publish attempt ended, reported back through
/// [`PublishClaim::settle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishOutcome {
    /// The publish reached the transport. The claimed signal is spent.
    Succeeded,
    /// The publish failed. The claimed signal is restored so a later
    /// deadline, flush, or shutdown retries it.
    Failed,
}

/// What [`PublishClaim::settle`] tells the caller to do with the dirty flag
/// it owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[must_use = "ignoring a DirtySignal::Restore drops the dirty flag of a FAILED publish, \
              so the mutation is persisted locally but never replicated"]
pub enum DirtySignal {
    /// Put the dirty flag back: the publish that took it failed, and the
    /// work must survive to be retried.
    Restore,
    /// Leave the flag clear: the publish succeeded, or there was no signal
    /// to begin with.
    Spent,
}

/// The dirty signal a publish path took from the caller, to be handed back
/// to [`settle`](Self::settle) with the outcome.
///
/// Dropping a claim instead of settling it leaves the caller's flag clear —
/// exactly the silent-loss shape this discipline exists to prevent, hence the
/// `#[must_use]`. Deliberately neither [`Clone`] nor [`Copy`], and
/// [`settle`](Self::settle) consumes it: a claim is one publish attempt's
/// hold on the signal, and settling it twice could restore a signal a later
/// claim already took.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a PublishClaim must be settled with `settle` — dropping it strands the \
              dirty flag as taken even if the publish failed, so the mutation is \
              persisted locally but never replicated"]
pub struct PublishClaim {
    should_publish: bool,
    took_dirty: bool,
}

impl PublishClaim {
    /// Whether this path should actually publish.
    ///
    /// Always true for [`on_deadline`](DebounceLatch::on_deadline) and
    /// [`on_flush`](DebounceLatch::on_flush); true for
    /// [`on_shutdown`](DebounceLatch::on_shutdown) only when there was
    /// unpublished work.
    #[must_use]
    pub fn should_publish(&self) -> bool {
        self.should_publish
    }

    /// Consume the claim with the publish outcome, reporting whether the
    /// caller must restore its dirty flag.
    ///
    /// Returns [`DirtySignal::Restore`] only when the publish **failed** and
    /// this claim actually took a signal — restoring one that was never
    /// there would manufacture unpublished work out of an idle flush.
    ///
    /// This says nothing about *when* to retry, and the latch deliberately
    /// never self-schedules one. Re-arming to the deadline that just fired
    /// would target an instant already in the past — that is *why* it fired —
    /// so a persistently failing publish (transport down, store unreachable)
    /// would fire, fail, re-arm to the same past instant, and spin without
    /// backoff. Retry *pacing* is a policy the driver owns: compose
    /// [`BackfillLatch`](crate::backfill_latch::BackfillLatch) from this crate
    /// when a failing publish should be retried autonomously under escalating
    /// backoff.
    pub fn settle(self, outcome: PublishOutcome) -> DirtySignal {
        if outcome == PublishOutcome::Failed && self.took_dirty {
            DirtySignal::Restore
        } else {
            DirtySignal::Spent
        }
    }
}

/// Sliding-window debounce state for a replicated dataset's publish path.
///
/// Holds the window only — see the [module docs](self) for why the dirty
/// flag stays with the caller, and for the claim/settle discipline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebounceLatch {
    debounce_ms: u64,
    deadline: Option<u64>,
}

impl DebounceLatch {
    /// A latch that collapses mutations arriving within `debounce_ms` of
    /// each other into one publish.
    #[must_use]
    pub fn new(debounce_ms: u64) -> Self {
        Self {
            debounce_ms,
            deadline: None,
        }
    }

    /// Slide the publish window out to `now_ms + debounce_ms`, arming it if
    /// it was idle.
    ///
    /// Call this when the caller's dirty flag is set. Idempotent in effect:
    /// repeated calls within a burst keep pushing the deadline, collapsing
    /// the burst into a single publish once the writer pauses.
    pub fn mark_dirty(&mut self, now_ms: u64) {
        self.deadline = Some(now_ms.saturating_add(self.debounce_ms));
    }

    /// The wall-clock millisecond to sleep until, or `None` when the window
    /// is idle and the driver should wait for a mutation instead.
    #[must_use]
    pub fn deadline(&self) -> Option<u64> {
        self.deadline
    }

    /// Whether a publish is currently scheduled.
    #[must_use]
    pub fn is_armed(&self) -> bool {
        self.deadline.is_some()
    }

    /// The debounce window elapsed: disarm and take the caller's signal.
    ///
    /// `was_dirty` is what the caller's flag held when it cleared it (an
    /// `AtomicBool::swap(false)`, typically). The claim always publishes —
    /// the deadline is only ever armed by [`mark_dirty`](Self::mark_dirty),
    /// so reaching it means there is something to send.
    pub fn on_deadline(&mut self, was_dirty: bool) -> PublishClaim {
        self.deadline = None;
        PublishClaim {
            should_publish: true,
            took_dirty: was_dirty,
        }
    }

    /// An explicit flush: publish **even if nothing is dirty**.
    ///
    /// Callers use this as a fence ("the current state is now on the wire"),
    /// so it must not be skipped for want of a dirty signal; the cost of a
    /// redundant publish is one store + broadcast round-trip.
    pub fn on_flush(&mut self, was_dirty: bool) -> PublishClaim {
        self.deadline = None;
        PublishClaim {
            should_publish: true,
            took_dirty: was_dirty,
        }
    }

    /// Shutdown: publish only if there is genuinely unpublished work.
    ///
    /// Unlike [`on_flush`](Self::on_flush) this does not force a publish — a
    /// clean flag at shutdown means everything already reached the transport.
    pub fn on_shutdown(&mut self, was_dirty: bool) -> PublishClaim {
        self.deadline = None;
        PublishClaim {
            should_publish: was_dirty,
            took_dirty: was_dirty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicBool, Ordering};

    /// The caller's half of the contract: the shared dirty flag that every
    /// mutation site writes, plus the window.
    ///
    /// Modelled exactly as the adopters hold it — an `AtomicBool` written
    /// from other tasks — so these tests exercise the real composition rather
    /// than a latch-only fiction.
    struct Driver {
        dirty: AtomicBool,
        latch: DebounceLatch,
    }

    impl Driver {
        fn new(debounce_ms: u64) -> Self {
            Self {
                dirty: AtomicBool::new(false),
                latch: DebounceLatch::new(debounce_ms),
            }
        }

        /// A mutation on some other task: set the flag, poke the engine.
        fn notify_dirty(&mut self, now_ms: u64) {
            self.dirty.store(true, Ordering::Release);
            self.latch.mark_dirty(now_ms);
        }

        fn take(&self) -> bool {
            self.dirty.swap(false, Ordering::AcqRel)
        }

        fn apply(&self, signal: DirtySignal) {
            if signal == DirtySignal::Restore {
                self.dirty.store(true, Ordering::Release);
            }
        }

        fn is_dirty(&self) -> bool {
            self.dirty.load(Ordering::Acquire)
        }
    }

    #[test]
    fn starts_idle_with_no_deadline() {
        let d = Driver::new(250);
        assert!(!d.is_dirty());
        assert!(!d.latch.is_armed());
        assert_eq!(d.latch.deadline(), None);
    }

    #[test]
    fn mark_dirty_arms_the_window() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);

        assert!(d.is_dirty());
        assert!(d.latch.is_armed());
        assert_eq!(d.latch.deadline(), Some(1_250));
    }

    // Sliding, not fixed: a burst collapses into ONE publish that lands after
    // the writer pauses, rather than firing on a cadence from the first edit.
    #[test]
    fn repeated_mutations_slide_the_window() {
        let mut d = Driver::new(250);

        d.notify_dirty(1_000);
        d.notify_dirty(1_100);
        d.notify_dirty(1_200);

        assert_eq!(d.latch.deadline(), Some(1_450));
    }

    #[test]
    fn deadline_claim_publishes_and_disarms() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);

        let claim = d.latch.on_deadline(d.take());
        assert!(claim.should_publish());
        assert_eq!(d.latch.deadline(), None);
        assert!(!d.is_dirty(), "the signal is taken, not left pending");

        d.apply(claim.settle(PublishOutcome::Succeeded));
        assert!(!d.is_dirty());
    }

    // The load-bearing property: a failed publish must NOT consume the dirty
    // signal, or the mutation is persisted locally and never replicated —
    // and the next shutdown sees a clean flag and skips its flush.
    #[test]
    fn failed_publish_restores_the_dirty_signal() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);

        let claim = d.latch.on_deadline(d.take());
        d.apply(claim.settle(PublishOutcome::Failed));

        assert!(d.is_dirty(), "retryable on the next opportunity");

        // Shutdown now sees the restored signal and flushes it.
        let retry = d.latch.on_shutdown(d.take());
        assert!(retry.should_publish());
        d.apply(retry.settle(PublishOutcome::Succeeded));
        assert!(!d.is_dirty());
    }

    // Pins the deliberate half of the above: settling a failure restores the
    // SIGNAL but arms no deadline, so the latch never self-schedules a retry.
    // Re-arming would target the instant that just fired — already in the past
    // — so a persistently failing publish would fire/fail/re-arm in a tight
    // loop with no backoff. Retry pacing belongs to the driver (compose
    // BackfillLatch); changing this should be a deliberate act, not a drift.
    #[test]
    fn failed_publish_restores_the_signal_but_arms_no_deadline() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);
        assert_eq!(d.latch.deadline(), Some(1_250));

        let claim = d.latch.on_deadline(d.take());
        d.apply(claim.settle(PublishOutcome::Failed));

        assert!(d.is_dirty(), "the work survives");
        assert_eq!(
            d.latch.deadline(),
            None,
            "but no deadline is armed — the driver decides when to retry"
        );

        // The next mutation is what re-arms, at a FUTURE instant.
        d.notify_dirty(5_000);
        assert_eq!(d.latch.deadline(), Some(5_250));
    }

    #[test]
    fn flush_publishes_even_when_idle() {
        let mut d = Driver::new(250);
        assert!(!d.is_dirty());

        let claim = d.latch.on_flush(d.take());
        assert!(
            claim.should_publish(),
            "an explicit flush is a fence — never skipped for want of a signal"
        );
        d.apply(claim.settle(PublishOutcome::Succeeded));
    }

    // An idle flush took no signal, so failing it must not invent one.
    #[test]
    fn failed_idle_flush_does_not_invent_a_dirty_signal() {
        let mut d = Driver::new(250);

        let claim = d.latch.on_flush(d.take());
        d.apply(claim.settle(PublishOutcome::Failed));

        assert!(!d.is_dirty());
    }

    #[test]
    fn shutdown_skips_the_publish_when_clean() {
        let mut d = Driver::new(250);

        let claim = d.latch.on_shutdown(d.take());
        assert!(
            !claim.should_publish(),
            "everything already reached the transport"
        );
        d.apply(claim.settle(PublishOutcome::Succeeded));
    }

    #[test]
    fn shutdown_publishes_when_dirty() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);

        let claim = d.latch.on_shutdown(d.take());
        assert!(claim.should_publish());
        d.apply(claim.settle(PublishOutcome::Succeeded));
        assert!(!d.is_dirty());
    }

    // A mutation landing while a publish is in flight must survive that
    // publish completing — it describes state the in-flight publish did not
    // carry. This is the case that forces the flag to be cleared UP FRONT,
    // which is what makes the settle discipline necessary in the first place.
    #[test]
    fn mutation_during_an_in_flight_publish_survives_settle() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);

        let claim = d.latch.on_deadline(d.take());
        // ... publish is in flight; a new mutation arrives on another task.
        d.notify_dirty(1_400);
        d.apply(claim.settle(PublishOutcome::Succeeded));

        assert!(d.is_dirty(), "the newer mutation is still pending");
        assert_eq!(d.latch.deadline(), Some(1_650));
    }

    // Settling is a verdict on the claim alone, never a mutation of the
    // window — which is what lets the caller keep the flag in an atomic that
    // other tasks write while a publish is in flight.
    #[test]
    fn settling_never_touches_the_window() {
        let mut d = Driver::new(250);
        d.notify_dirty(1_000);
        let claim = d.latch.on_deadline(d.take());
        d.notify_dirty(2_000);

        assert_eq!(
            claim.settle(PublishOutcome::Failed),
            DirtySignal::Restore,
            "the failed publish's signal comes back"
        );
        assert_eq!(
            d.latch.deadline(),
            Some(2_250),
            "and the window still belongs to the mutation that armed it"
        );
    }

    #[test]
    fn deadline_arithmetic_saturates() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(u64::MAX);

        assert_eq!(
            latch.deadline(),
            Some(u64::MAX),
            "no overflow panic at the end of the epoch"
        );
    }
}

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
//! [`DebounceLatch`] owns that state machine and nothing else — no timer, no
//! transport, no clock. Milliseconds go in, a [`deadline`](DebounceLatch::deadline)
//! comes out, and the caller's driver sleeps until it and reports back. The
//! same latch also serves the two paths that bypass the window: an explicit
//! flush (always publishes, even idle) and shutdown (publishes only if there
//! is genuinely unpublished work).
//!
//! # Claim and settle
//!
//! The failure mode this type exists to prevent: a publish path that clears
//! the dirty flag *before* publishing, and then loses the publish. The
//! signal is consumed, nothing was replicated, and the state sits dirty with
//! no pending deadline — so the next shutdown flush sees a clean latch and
//! skips it. The mutation is persisted locally but never reaches a peer.
//!
//! Clearing the flag up front is not optional (a mutation arriving *during*
//! the publish must re-arm, not be swallowed by the completing publish), so
//! the latch instead makes the restore mandatory-looking: each publish path
//! returns a `#[must_use]` [`PublishClaim`] carrying the signal it took, and
//! [`settle`](DebounceLatch::settle) hands it back with the outcome. A
//! failed publish restores the dirty state so the next deadline, flush, or
//! shutdown retries it.
//!
//! ```
//! use harmony_crdt_sync::debounce_latch::{DebounceLatch, PublishOutcome};
//!
//! let mut latch = DebounceLatch::new(250);
//! assert_eq!(latch.deadline(), None, "idle until a mutation arrives");
//!
//! latch.mark_dirty(1_000);
//! assert_eq!(latch.deadline(), Some(1_250));
//!
//! // A second mutation slides the window rather than publishing twice.
//! latch.mark_dirty(1_100);
//! assert_eq!(latch.deadline(), Some(1_350));
//!
//! // The driver slept to the deadline; claim the signal and publish.
//! let claim = latch.on_deadline();
//! assert!(claim.should_publish());
//!
//! // The publish failed — settling restores the signal so it is retried.
//! latch.settle(claim, PublishOutcome::Failed);
//! assert!(latch.is_dirty());
//! ```
//!
//! Provenance: extracted from harmony-client's `fleet_sync::internal_task`
//! select loop (ZEB-571 item 6b), behavior-preserving. The `Notify`/timer
//! plumbing, the publish itself, and the persist that follows it stay
//! caller-side.

/// How a publish attempt ended, reported back through
/// [`DebounceLatch::settle`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PublishOutcome {
    /// The publish reached the transport. The claimed signal is spent.
    Succeeded,
    /// The publish failed. The claimed signal is restored so a later
    /// deadline, flush, or shutdown retries it.
    Failed,
}

/// The dirty signal claimed by a publish path, to be handed back to
/// [`DebounceLatch::settle`] with the outcome.
///
/// Dropping a claim instead of settling it behaves like
/// [`PublishOutcome::Succeeded`] — the signal stays consumed — which is the
/// silent-loss shape the latch exists to prevent, hence the `#[must_use]`.
/// Deliberately neither [`Clone`] nor [`Copy`]: a claim represents one
/// publish attempt's hold on the dirty signal, and settling it twice would
/// restore a signal that a later claim already took.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a PublishClaim must be settled with `settle` — dropping it consumes \
              the dirty signal even if the publish failed, so the mutation is \
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
}

/// Sliding-window debounce state for a replicated dataset's publish path.
///
/// See the [module docs](self) for the claim/settle discipline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DebounceLatch {
    debounce_ms: u64,
    dirty: bool,
    deadline: Option<u64>,
}

impl DebounceLatch {
    /// A latch that collapses mutations arriving within `debounce_ms` of
    /// each other into one publish.
    #[must_use]
    pub fn new(debounce_ms: u64) -> Self {
        Self {
            debounce_ms,
            dirty: false,
            deadline: None,
        }
    }

    /// Record a local mutation at `now_ms` and slide the publish window out
    /// to `now_ms + debounce_ms`.
    ///
    /// Idempotent in effect: repeated calls within a burst keep pushing the
    /// deadline, collapsing the burst into a single publish once the writer
    /// pauses.
    pub fn mark_dirty(&mut self, now_ms: u64) {
        self.dirty = true;
        self.deadline = Some(now_ms.saturating_add(self.debounce_ms));
    }

    /// The wall-clock millisecond to sleep until, or `None` when the latch
    /// is idle and the driver should wait for a mutation instead.
    #[must_use]
    pub fn deadline(&self) -> Option<u64> {
        self.deadline
    }

    /// Whether there is unpublished local work.
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// The debounce window elapsed: disarm and claim the dirty signal.
    ///
    /// The claim always publishes — the deadline is only ever armed by
    /// [`mark_dirty`](DebounceLatch::mark_dirty), so reaching it means there
    /// is something to send.
    pub fn on_deadline(&mut self) -> PublishClaim {
        self.deadline = None;
        self.claim(true)
    }

    /// An explicit flush: publish **even if the latch is idle**.
    ///
    /// Callers use this as a fence ("the current state is now on the wire"),
    /// so it must not be skipped for want of a dirty signal; the cost of a
    /// redundant publish is one store + broadcast round-trip.
    pub fn on_flush(&mut self) -> PublishClaim {
        self.deadline = None;
        self.claim(true)
    }

    /// Shutdown: publish only if there is genuinely unpublished work.
    ///
    /// Unlike [`on_flush`](DebounceLatch::on_flush) this does not force a
    /// publish — a clean latch at shutdown means everything already reached
    /// the transport.
    pub fn on_shutdown(&mut self) -> PublishClaim {
        self.deadline = None;
        let dirty = self.dirty;
        self.claim(dirty)
    }

    /// Hand a claim back with the publish outcome.
    ///
    /// On [`PublishOutcome::Failed`] the claimed dirty signal is restored,
    /// so the next deadline, flush, or shutdown retries the publish instead
    /// of silently dropping it.
    pub fn settle(&mut self, claim: PublishClaim, outcome: PublishOutcome) {
        if outcome == PublishOutcome::Failed && claim.took_dirty {
            self.dirty = true;
        }
    }

    /// Take the dirty signal, recording whether there was one so
    /// [`settle`](DebounceLatch::settle) can put it back.
    fn claim(&mut self, should_publish: bool) -> PublishClaim {
        let took_dirty = self.dirty;
        self.dirty = false;
        PublishClaim {
            should_publish,
            took_dirty,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn starts_idle_with_no_deadline() {
        let latch = DebounceLatch::new(250);
        assert!(!latch.is_dirty());
        assert_eq!(latch.deadline(), None);
    }

    #[test]
    fn mark_dirty_arms_the_window() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(1_000);

        assert!(latch.is_dirty());
        assert_eq!(latch.deadline(), Some(1_250));
    }

    // Sliding, not fixed: a burst collapses into ONE publish that lands after
    // the writer pauses, rather than firing on a cadence from the first edit.
    #[test]
    fn repeated_mutations_slide_the_window() {
        let mut latch = DebounceLatch::new(250);

        latch.mark_dirty(1_000);
        latch.mark_dirty(1_100);
        latch.mark_dirty(1_200);

        assert_eq!(latch.deadline(), Some(1_450));
    }

    #[test]
    fn deadline_claim_publishes_and_disarms() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(1_000);

        let claim = latch.on_deadline();
        assert!(claim.should_publish());
        assert_eq!(latch.deadline(), None);
        assert!(!latch.is_dirty(), "the signal is claimed, not left pending");

        latch.settle(claim, PublishOutcome::Succeeded);
        assert!(!latch.is_dirty());
    }

    // The load-bearing property: a failed publish must NOT consume the dirty
    // signal, or the mutation is persisted locally and never replicated —
    // and the next shutdown sees a clean latch and skips its flush.
    #[test]
    fn failed_publish_restores_the_dirty_signal() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(1_000);

        let claim = latch.on_deadline();
        latch.settle(claim, PublishOutcome::Failed);

        assert!(latch.is_dirty(), "retryable on the next opportunity");

        // Shutdown now sees the restored signal and flushes it.
        let retry = latch.on_shutdown();
        assert!(retry.should_publish());
        latch.settle(retry, PublishOutcome::Succeeded);
        assert!(!latch.is_dirty());
    }

    #[test]
    fn flush_publishes_even_when_idle() {
        let mut latch = DebounceLatch::new(250);
        assert!(!latch.is_dirty());

        let claim = latch.on_flush();
        assert!(
            claim.should_publish(),
            "an explicit flush is a fence — never skipped for want of a signal"
        );
        latch.settle(claim, PublishOutcome::Succeeded);
    }

    // An idle flush took no signal, so failing it must not invent one.
    #[test]
    fn failed_idle_flush_does_not_invent_a_dirty_signal() {
        let mut latch = DebounceLatch::new(250);

        let claim = latch.on_flush();
        latch.settle(claim, PublishOutcome::Failed);

        assert!(!latch.is_dirty());
    }

    #[test]
    fn shutdown_skips_the_publish_when_clean() {
        let mut latch = DebounceLatch::new(250);

        let claim = latch.on_shutdown();
        assert!(
            !claim.should_publish(),
            "everything already reached the transport"
        );
        latch.settle(claim, PublishOutcome::Succeeded);
    }

    #[test]
    fn shutdown_publishes_when_dirty() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(1_000);

        let claim = latch.on_shutdown();
        assert!(claim.should_publish());
        latch.settle(claim, PublishOutcome::Succeeded);
        assert!(!latch.is_dirty());
    }

    // A mutation landing while a publish is in flight must survive that
    // publish completing — it describes state the in-flight publish did not
    // carry.
    #[test]
    fn mutation_during_an_in_flight_publish_survives_settle() {
        let mut latch = DebounceLatch::new(250);
        latch.mark_dirty(1_000);

        let claim = latch.on_deadline();
        // ... publish is in flight; a new mutation arrives.
        latch.mark_dirty(1_400);
        latch.settle(claim, PublishOutcome::Succeeded);

        assert!(latch.is_dirty(), "the newer mutation is still pending");
        assert_eq!(latch.deadline(), Some(1_650));
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

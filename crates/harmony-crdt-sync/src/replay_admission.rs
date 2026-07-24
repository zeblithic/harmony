// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Replay admission: the pure decision core for "may this inbound publish
//! be applied, and when may the source's watermark advance?"
//!
//! A device replicating a dataset over best-effort pubsub receives publishes
//! from its peers, each stamped with the publisher's clock. It must accept a
//! publish only if that stamp beats the last one it accepted from the same
//! publisher — otherwise a re-delivered, reordered, or maliciously replayed
//! frame re-applies stale state. [`ReplayTracker`] is that check, plus the
//! per-source watermark map it reads, with **no** transport, crypto, or
//! persistence: a receiver hands it a source and a clock, and gets one of
//! three verdicts back.
//!
//! # Apply-before-advance
//!
//! The subtle part is not the comparison — it is *when the watermark moves*.
//! Between admitting a publish and applying it, a receiver typically does
//! fallible work: fetch a content blob, decrypt it, decode it, merge it. If
//! the watermark advances **before** that work and any of it fails, the
//! publish is lost permanently: the peer's frame was never applied, yet a
//! re-delivery of it now fails the replay check as a duplicate. The peer has
//! no way to know, so it never re-sends. The dataset silently diverges.
//!
//! So the watermark must advance only *after* a successful apply. That rule
//! is easy to state and easy to get wrong, because both orderings compile
//! and the wrong one fails only under a fault. This module encodes it in the
//! types instead of a comment: [`admit`](ReplayTracker::admit) takes `&self`
//! and cannot advance anything, and the only way to advance is
//! [`commit`](ReplayTracker::commit), which **consumes a
//! [`CommitTicket`]** — and a ticket exists only as the payload of an
//! [`Admission::Accept`]. A receiver therefore cannot advance a watermark it
//! never admitted, and cannot advance before it holds the proof that it had
//! something to advance for. Dropping the ticket (the fallible work failed)
//! leaves the watermark untouched, which is exactly the retry-safe state.
//!
//! ```
//! use harmony_crdt_sync::replay_admission::{Admission, ReplayTracker};
//!
//! let mut tracker: ReplayTracker<&str, u64> = ReplayTracker::new("me");
//!
//! // Our own publish, reflected back by the transport.
//! assert!(matches!(tracker.admit(&"me", &7), Admission::Echo));
//!
//! // A peer's first publish is admissible.
//! let Admission::Accept(ticket) = tracker.admit(&"peer", &5) else {
//!     panic!("first publish from a source is always admissible")
//! };
//!
//! // ... fetch, decrypt, decode, merge — all fallible. Only on success:
//! assert!(tracker.commit(ticket));
//!
//! // Now a re-delivery of the same frame is rejected.
//! assert!(matches!(tracker.admit(&"peer", &5), Admission::Duplicate));
//! assert!(matches!(tracker.admit(&"peer", &6), Admission::Accept(_)));
//! ```
//!
//! Had the apply failed, the ticket would simply be dropped — and `&5` would
//! still be admissible on the peer's next delivery.
//!
//! # Watermarks
//!
//! `C` is only compared and cloned, never inspected, so the domain's clock
//! type (and any wire-format contract it carries) stays in the caller's
//! crate. Its [`Ord`] must be the domain's "strictly newer" relation —
//! [`HlcTick`](crate::hlc::HlcTick) is the usual building block.
//!
//! [`MonotoneMap`] is the underlying per-key "replace iff strictly newer"
//! fold, exposed on its own because receivers generally need a second one:
//! tracking each peer's acknowledgement of *us* (harvested from inbound
//! frames) is the same operation without the admission discipline.
//!
//! Provenance: extracted from harmony-client's `fleet_sync` inbound path
//! (ZEB-571 item 6b), behavior-preserving — steps 3/4 are [`admit`], step 9
//! is [`commit`], step 10 is a [`MonotoneMap`]. The decrypt, CAS fetch,
//! decode, and merge between them stay caller-side.
//!
//! [`admit`]: ReplayTracker::admit
//! [`commit`]: ReplayTracker::commit

use alloc::collections::BTreeMap;

/// A per-key watermark map under a "replace iff strictly newer" rule.
///
/// Every entry only ever moves forward under [`observe`](MonotoneMap::observe),
/// so out-of-order or re-delivered observations cannot walk a watermark
/// backwards.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonotoneMap<K, C> {
    entries: BTreeMap<K, C>,
}

impl<K: Ord, C: Ord> Default for MonotoneMap<K, C> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Ord, C: Ord> MonotoneMap<K, C> {
    /// An empty map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
        }
    }

    /// Rebuild from previously-persisted entries.
    ///
    /// The caller is trusted to supply entries it actually observed — this
    /// is the restore-from-disk seam, not an admission path.
    #[must_use]
    pub fn from_entries(entries: BTreeMap<K, C>) -> Self {
        Self { entries }
    }

    /// Record `clock` for `key` if it is strictly newer than the current
    /// entry (or there is none). Returns whether the watermark advanced.
    ///
    /// A `false` return means the observation was stale or a duplicate and
    /// the map is unchanged.
    pub fn observe(&mut self, key: K, clock: C) -> bool {
        match self.entries.get(&key) {
            Some(existing) if *existing >= clock => false,
            _ => {
                self.entries.insert(key, clock);
                true
            }
        }
    }

    /// The current watermark for `key`, if any.
    pub fn get(&self, key: &K) -> Option<&C> {
        self.entries.get(key)
    }

    /// The entries, for persistence or inspection.
    #[must_use]
    pub fn entries(&self) -> &BTreeMap<K, C> {
        &self.entries
    }

    /// Consume the map, yielding its entries.
    #[must_use]
    pub fn into_entries(self) -> BTreeMap<K, C> {
        self.entries
    }

    /// Number of keys with a watermark.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether no key has a watermark yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Proof that a specific `(source, clock)` pair passed
/// [`ReplayTracker::admit`], and the only key that opens
/// [`ReplayTracker::commit`].
///
/// Hold it across the fallible apply work; pass it to `commit` on success,
/// drop it on failure. Dropping is the correct, retry-safe response to a
/// failed apply — the source's watermark stays where it was, so the peer's
/// next delivery of the same frame is admitted again.
///
/// Deliberately not [`Clone`]: a ticket is the proof for one admitted
/// publish, and duplicating it would let a stale copy outlive the apply it
/// belongs to.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "dropping a CommitTicket leaves the source's watermark un-advanced \
              (correct after a FAILED apply, a silent bug after a successful one)"]
pub struct CommitTicket<K, C> {
    source: K,
    clock: C,
}

impl<K, C> CommitTicket<K, C> {
    /// The source this ticket admits.
    pub fn source(&self) -> &K {
        &self.source
    }

    /// The clock this ticket would advance the source's watermark to.
    pub fn clock(&self) -> &C {
        &self.clock
    }
}

/// The verdict on one inbound publish.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "an admission verdict decides whether the publish is applied"]
pub enum Admission<K, C> {
    /// Our own publish, reflected back to us by the transport. Not an
    /// error and not a replay — simply nothing to do.
    Echo,
    /// Not strictly newer than the watermark already recorded for this
    /// source: a re-delivery, a reorder, or a replay. Discard it.
    Duplicate,
    /// Admissible. Apply it, then [`commit`](ReplayTracker::commit) the
    /// ticket — in that order (see the module docs).
    Accept(CommitTicket<K, C>),
}

/// Per-source replay protection with the apply-before-advance discipline
/// enforced by the type system.
///
/// See the [module docs](self) for why the admit/commit split exists.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayTracker<K, C> {
    local: K,
    accepted: MonotoneMap<K, C>,
}

impl<K: Ord + Clone, C: Ord + Clone> ReplayTracker<K, C> {
    /// A tracker for a receiver whose own source id is `local` (used to
    /// recognise our own publishes as [`Admission::Echo`]).
    #[must_use]
    pub fn new(local: K) -> Self {
        Self {
            local,
            accepted: MonotoneMap::new(),
        }
    }

    /// Rebuild from persisted watermarks — the restore-from-disk seam.
    #[must_use]
    pub fn from_accepted(local: K, accepted: BTreeMap<K, C>) -> Self {
        Self {
            local,
            accepted: MonotoneMap::from_entries(accepted),
        }
    }

    /// Judge one inbound publish. **Read-only**: this cannot advance any
    /// watermark, by construction.
    pub fn admit(&self, source: &K, clock: &C) -> Admission<K, C> {
        if *source == self.local {
            return Admission::Echo;
        }
        match self.accepted.get(source) {
            Some(existing) if *existing >= *clock => Admission::Duplicate,
            _ => Admission::Accept(CommitTicket {
                source: source.clone(),
                clock: clock.clone(),
            }),
        }
    }

    /// Advance a source's watermark, consuming the ticket that admitted it.
    /// Call this **only after** the publish has been successfully applied.
    ///
    /// Returns whether the watermark advanced. A `false` return means the
    /// watermark had already moved past this ticket's clock between `admit`
    /// and `commit` — impossible on a single-threaded receiver, and
    /// harmlessly idempotent on a concurrent one, since the advance itself
    /// is monotone and can never regress.
    pub fn commit(&mut self, ticket: CommitTicket<K, C>) -> bool {
        self.accepted.observe(ticket.source, ticket.clock)
    }

    /// This receiver's own source id.
    pub fn local(&self) -> &K {
        &self.local
    }

    /// The accepted watermarks, for persistence or inspection.
    #[must_use]
    pub fn accepted(&self) -> &BTreeMap<K, C> {
        self.accepted.entries()
    }

    /// The watermark accepted from `source`, if any.
    pub fn accepted_from(&self, source: &K) -> Option<&C> {
        self.accepted.get(source)
    }

    /// Number of sources with an accepted watermark.
    #[must_use]
    pub fn len(&self) -> usize {
        self.accepted.len()
    }

    /// Whether nothing has been accepted from any source yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.accepted.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::{String, ToString};

    fn tracker() -> ReplayTracker<String, u64> {
        ReplayTracker::new("me".to_string())
    }

    fn accept(
        t: &ReplayTracker<String, u64>,
        source: &str,
        clock: u64,
    ) -> CommitTicket<String, u64> {
        match t.admit(&source.to_string(), &clock) {
            Admission::Accept(ticket) => ticket,
            other => panic!("expected Accept, got {other:?}"),
        }
    }

    #[test]
    fn our_own_publish_is_an_echo() {
        let t = tracker();
        assert!(matches!(t.admit(&"me".to_string(), &1), Admission::Echo));
    }

    #[test]
    fn first_publish_from_a_source_is_admitted() {
        let t = tracker();
        assert!(matches!(
            t.admit(&"peer".to_string(), &1),
            Admission::Accept(_)
        ));
    }

    #[test]
    fn admit_does_not_advance_the_watermark() {
        let t = tracker();
        let _ticket = accept(&t, "peer", 5);
        assert!(
            t.accepted_from(&"peer".to_string()).is_none(),
            "admit is read-only — only commit may advance"
        );
        // Un-committed, so the same clock is still admissible.
        assert!(matches!(
            t.admit(&"peer".to_string(), &5),
            Admission::Accept(_)
        ));
    }

    #[test]
    fn committed_clock_is_then_a_duplicate() {
        let mut t = tracker();
        let ticket = accept(&t, "peer", 5);
        assert!(t.commit(ticket));

        assert!(matches!(
            t.admit(&"peer".to_string(), &5),
            Admission::Duplicate
        ));
        assert!(matches!(
            t.admit(&"peer".to_string(), &4),
            Admission::Duplicate,
        ));
        assert!(matches!(
            t.admit(&"peer".to_string(), &6),
            Admission::Accept(_)
        ));
    }

    // The load-bearing property: a publish whose apply FAILED (ticket
    // dropped) must remain admissible, or the peer's frame is lost forever
    // — it never learns to re-send, and the datasets diverge silently.
    #[test]
    fn dropped_ticket_leaves_the_publish_retryable() {
        let mut t = tracker();

        let ticket = accept(&t, "peer", 5);
        drop(ticket); // the fetch/decrypt/decode/merge failed

        let retry = accept(&t, "peer", 5);
        assert!(t.commit(retry), "the retry applies and advances");
        assert_eq!(t.accepted_from(&"peer".to_string()), Some(&5));
    }

    #[test]
    fn watermarks_are_per_source() {
        let mut t = tracker();
        let a = accept(&t, "a", 9);
        assert!(t.commit(a));

        assert!(
            matches!(t.admit(&"b".to_string(), &1), Admission::Accept(_)),
            "a high watermark for one source must not gate another"
        );
        assert_eq!(t.len(), 1);
    }

    #[test]
    fn commit_is_monotone_and_cannot_regress_a_watermark() {
        let mut t = tracker();

        // Two tickets admitted against the same (empty) prior state — the
        // shape a concurrent receiver could produce.
        let older = accept(&t, "peer", 5);
        let newer = accept(&t, "peer", 9);

        assert!(t.commit(newer));
        assert!(
            !t.commit(older),
            "the stale commit is refused, not applied backwards"
        );
        assert_eq!(t.accepted_from(&"peer".to_string()), Some(&9));
    }

    #[test]
    fn round_trips_through_persisted_entries() {
        let mut t = tracker();
        let ticket = accept(&t, "peer", 5);
        t.commit(ticket);

        let restored = ReplayTracker::from_accepted("me".to_string(), t.accepted().clone());
        assert_eq!(restored, t);
        assert!(matches!(
            restored.admit(&"peer".to_string(), &5),
            Admission::Duplicate
        ));
    }

    #[test]
    fn monotone_map_replaces_only_on_strictly_newer() {
        let mut m: MonotoneMap<String, u64> = MonotoneMap::new();

        assert!(m.observe("peer".to_string(), 5), "first observation lands");
        assert!(!m.observe("peer".to_string(), 5), "equal is not newer");
        assert!(!m.observe("peer".to_string(), 4), "older is refused");
        assert!(m.observe("peer".to_string(), 6), "newer advances");

        assert_eq!(m.get(&"peer".to_string()), Some(&6));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn monotone_map_starts_empty() {
        let m: MonotoneMap<String, u64> = MonotoneMap::default();
        assert!(m.is_empty());
        assert_eq!(m.get(&"peer".to_string()), None);
    }
}

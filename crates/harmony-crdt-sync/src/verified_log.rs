// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Verified event-log engine: a pure, in-memory log of domain events that
//! dedups by a stable id, verifies each new event against the materialized
//! state of everything strictly prior to it, and re-materializes on demand.
//!
//! A [`LogPolicy`] supplies the domain: the event and id types, a total
//! order ([`cmp`](LogPolicy::cmp)) defining which events are "prior", a
//! [`verify`](LogPolicy::verify) predicate run against the prior state, and
//! a [`materialize`](LogPolicy::materialize) fold. [`VerifiedLog`] owns only
//! the verified event set and the insert/dedup/verify mechanics; it is
//! sans-I/O — no async, transport, persistence, or crypto — so a domain's
//! signature checks, encryption, on-disk format, and sync driver all stay in
//! the caller's crate. This is the event-log sibling of the snapshot engine
//! and the [`backfill_latch`](crate::backfill_latch) this crate already hosts
//! (ZEB-571 item 6).
//!
//! ```
//! use harmony_crdt_sync::verified_log::{LogPolicy, VerifiedLog, InsertOutcome};
//! use core::cmp::Ordering;
//!
//! struct P;
//! impl LogPolicy for P {
//!     type Event = (u32, u64); // (id, order)
//!     type EventId = u32;
//!     type State = usize;      // count of materialized events
//!     type Context = ();
//!     type Error = ();
//!     type SupersessionKey = (); // never supersedes (method left defaulted)
//!     fn event_id(e: &(u32, u64)) -> u32 { e.0 }
//!     fn cmp(a: &(u32, u64), b: &(u32, u64)) -> Ordering { a.1.cmp(&b.1).then(a.0.cmp(&b.0)) }
//!     fn verify(_e: &(u32, u64), _prior: &usize, _ctx: &()) -> Result<(), ()> { Ok(()) }
//!     fn materialize(events: &[&(u32, u64)], _ctx: &()) -> usize { events.len() }
//! }
//!
//! let mut log: VerifiedLog<P> = VerifiedLog::new();
//! assert_eq!(log.insert((1, 10), &()), InsertOutcome::Inserted);
//! assert_eq!(log.insert((1, 10), &()), InsertOutcome::AlreadyKnown);
//! assert_eq!(log.materialize(&()), 1);
//! ```

use alloc::collections::btree_map::Entry;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::cmp::Ordering;

/// The domain policy backing a [`VerifiedLog`].
///
/// Every method is an associated function (no `&self`): a policy is a
/// zero-sized type carrying only the domain's behavior, never state — all
/// state lives in the log's events and the caller-supplied
/// [`Context`](LogPolicy::Context).
pub trait LogPolicy {
    /// A domain event (e.g. a signed membership event). Stored by value.
    type Event;
    /// The dedup key. Two events with the same id are the same event; the
    /// second insert is [`InsertOutcome::AlreadyKnown`] without re-verifying.
    type EventId: Ord + Clone;
    /// The materialized view produced by folding an event set.
    type State;
    /// Per-call configuration threaded into both `verify` and `materialize` —
    /// e.g. a dataset's admin address, or a wall-clock floor. The caller may
    /// rebuild it for each [`insert`](VerifiedLog::insert) and may derive parts
    /// of it from the event being inserted (the engine passes the same `ctx`
    /// to the prior-state `materialize` and to `verify`). It is simply not
    /// carried *inside* the stored events.
    type Context;
    /// A verification failure reported via [`InsertOutcome::Rejected`].
    type Error;

    /// Extract an event's dedup id.
    fn event_id(e: &Self::Event) -> Self::EventId;

    /// A strict total order over events. Defines which stored events are
    /// "prior" to a candidate (those that compare [`Ordering::Less`]).
    /// Distinct events must never compare [`Ordering::Equal`].
    fn cmp(a: &Self::Event, b: &Self::Event) -> Ordering;

    /// Verify a candidate event against `prior` — the materialized state of
    /// exactly the events that precede it under [`cmp`](LogPolicy::cmp).
    fn verify(e: &Self::Event, prior: &Self::State, ctx: &Self::Context)
        -> Result<(), Self::Error>;

    /// Fold an event set into a materialized view. `events` is supplied in
    /// unspecified order; a policy that needs ordered input sorts internally.
    fn materialize(events: &[&Self::Event], ctx: &Self::Context) -> Self::State;

    /// The grouping key for deterministic supersession (ZEB-813). Events
    /// whose [`supersession_key`](LogPolicy::supersession_key) returns equal
    /// `Some` keys form one lineage, in which only the maximum under
    /// [`cmp`](LogPolicy::cmp) is retained. Policies that never supersede
    /// declare `()` and leave the method defaulted.
    type SupersessionKey: Ord;

    /// Extract an event's supersession-lineage key, or `None` for events
    /// that never supersede and are never superseded (ZEB-813).
    ///
    /// Contract (policy obligations; the log cannot check them):
    ///
    /// (a) pure — same input, same answer, on every replica;
    /// (b) keyed events must be MATERIALIZE-NEUTRAL: dropping a superseded
    ///     event must not change `materialize`'s output for any event
    ///     subset, because removal changes the strictly-prior set that
    ///     future `verify` calls see.
    ///
    /// Convergence is then by construction: the retained set is "the
    /// per-key maximum under `cmp` plus all `None`-keyed events" — a
    /// join-semilattice, identical on every replica regardless of arrival
    /// order or merge interleaving.
    ///
    /// Default: `None` for everything — nothing supersedes anything (the
    /// behavior of every adopter predating this seam).
    fn supersession_key(_e: &Self::Event) -> Option<Self::SupersessionKey> {
        None
    }
}

/// The result of [`VerifiedLog::insert`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome<E> {
    /// The event was new and verified; it is now in the log.
    Inserted,
    /// An event with this id was already present; nothing changed and
    /// `verify` was not run.
    AlreadyKnown,
    /// The event was new but a stored event with the same
    /// [`supersession_key`](LogPolicy::supersession_key) is greater under
    /// [`cmp`](LogPolicy::cmp); the candidate was dropped WITHOUT running
    /// `verify` (mirrors `AlreadyKnown`'s verify-skip: a stale record
    /// changes nothing, so proving it valid buys nothing). Callers treat
    /// this exactly like `AlreadyKnown`: no state change, no persistence,
    /// no dirty mark.
    ///
    /// MIGRATION (ZEB-813, breaking change): this variant is new, and
    /// `LogPolicy` gained a required `SupersessionKey` associated type.
    /// Downstream exhaustive matches must add an arm — map it to whatever
    /// their `AlreadyKnown` arm does — and policies that never supersede
    /// declare `type SupersessionKey = ();`. Such policies never produce
    /// this variant.
    Superseded,
    /// The event was new but failed verification; it was not stored.
    Rejected(E),
}

/// A verified, in-memory event log keyed by [`LogPolicy::EventId`].
pub struct VerifiedLog<P: LogPolicy> {
    events: BTreeMap<P::EventId, P::Event>,
}

impl<P: LogPolicy> Default for VerifiedLog<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: LogPolicy> VerifiedLog<P> {
    /// An empty log.
    pub fn new() -> Self {
        Self {
            events: BTreeMap::new(),
        }
    }

    /// Build a log from events whose provenance is already trusted — e.g.
    /// loaded from local persistence that verified them when they first
    /// arrived — **without** re-running [`LogPolicy::verify`]. Deduplicates by
    /// [`event_id`](LogPolicy::event_id) (a later duplicate replaces an
    /// earlier one).
    ///
    /// Use this only for trusted events; use [`insert`](Self::insert) for
    /// events arriving over the wire. Domains whose persistence does not
    /// re-verify on load (the common case) restore through here so a boot
    /// does not re-verify the whole history.
    pub fn from_verified_events(events: impl IntoIterator<Item = P::Event>) -> Self {
        let mut map = BTreeMap::new();
        for e in events {
            map.insert(P::event_id(&e), e);
        }
        // ZEB-813: deterministic supersession compaction at load. Group by
        // supersession key and keep only each group's maximum under `cmp` —
        // the semilattice maximum directly, so trusted restores land on the
        // same set an insert-ordered replica converges to, including
        // already-over-cap logs persisted by pre-supersession builds. One
        // key extraction per event and a map of per-group winners keep this
        // O(n log n); the restore path exists precisely for large logs.
        let winners: BTreeMap<P::SupersessionKey, P::EventId> = {
            let mut best: BTreeMap<P::SupersessionKey, &P::Event> = BTreeMap::new();
            for e in map.values() {
                if let Some(key) = P::supersession_key(e) {
                    match best.entry(key) {
                        Entry::Vacant(v) => {
                            v.insert(e);
                        }
                        Entry::Occupied(mut o) => {
                            if P::cmp(e, o.get()) == Ordering::Greater {
                                o.insert(e);
                            }
                        }
                    }
                }
            }
            best.into_iter().map(|(k, e)| (k, P::event_id(e))).collect()
        };
        let stale: Vec<P::EventId> = map
            .values()
            .filter(|e| {
                P::supersession_key(e).is_some_and(|key| winners.get(&key) != Some(&P::event_id(e)))
            })
            .map(|e| P::event_id(e))
            .collect();
        for stale_id in stale {
            map.remove(&stale_id);
        }
        Self { events: map }
    }

    /// Insert an event: dedup by id, then verify against the materialized
    /// state of all strictly-prior events (by [`LogPolicy::cmp`]). Stored
    /// only on `Ok`.
    ///
    /// Verification is **per-insert against the current strictly-prior set** —
    /// an event is checked once, when it lands. Inserting an event that sorts
    /// *before* events already present does **not** re-validate those later
    /// events against the now-changed prior state, and an event rejected
    /// because a dependency had not yet arrived is not automatically retried.
    /// A domain that accepts events out of order is responsible for eventual
    /// consistency at its sync layer — e.g. by re-delivering the complete
    /// event set so every event is (re-)attempted with its full prior present.
    /// This matches the reference adopter, community-membership's
    /// `insert_event`, whose whole-state-root sync provides exactly that
    /// re-delivery; adding re-validation here would diverge from it.
    pub fn insert(&mut self, event: P::Event, ctx: &P::Context) -> InsertOutcome<P::Error> {
        let id = P::event_id(&event);
        if self.events.contains_key(&id) {
            return InsertOutcome::AlreadyKnown;
        }
        if let Some(key) = P::supersession_key(&event) {
            if self.events.values().any(|existing| {
                P::supersession_key(existing).as_ref() == Some(&key)
                    && P::cmp(existing, &event) == Ordering::Greater
            }) {
                return InsertOutcome::Superseded;
            }
        }
        let prior_state = {
            let prior: Vec<&P::Event> = self
                .events
                .values()
                .filter(|existing| P::cmp(existing, &event) == Ordering::Less)
                .collect();
            P::materialize(&prior, ctx)
        };
        match P::verify(&event, &prior_state, ctx) {
            Ok(()) => {
                if let Some(key) = P::supersession_key(&event) {
                    // Every stored same-key event is strictly less under
                    // `cmp` here — a greater one would have returned
                    // `Superseded` above, and distinct events never compare
                    // `Equal` — so the whole lineage is stale.
                    let stale: Vec<P::EventId> = self
                        .events
                        .values()
                        .filter(|existing| P::supersession_key(existing).as_ref() == Some(&key))
                        .map(|existing| P::event_id(existing))
                        .collect();
                    for stale_id in stale {
                        self.events.remove(&stale_id);
                    }
                }
                self.events.insert(id, event);
                InsertOutcome::Inserted
            }
            Err(e) => InsertOutcome::Rejected(e),
        }
    }

    /// Whether an event with this id is present.
    pub fn contains(&self, id: &P::EventId) -> bool {
        self.events.contains_key(id)
    }

    /// The stored event with this id, if any.
    pub fn get(&self, id: &P::EventId) -> Option<&P::Event> {
        self.events.get(id)
    }

    /// Number of stored events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Iterate stored events in [`LogPolicy::EventId`] order.
    pub fn events(&self) -> impl Iterator<Item = &P::Event> {
        self.events.values()
    }

    /// Materialize the full view over all stored events. Uncached — the
    /// caller owns any caching policy.
    pub fn materialize(&self, ctx: &P::Context) -> P::State {
        let all: Vec<&P::Event> = self.events.values().collect();
        P::materialize(&all, ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BTreeSet;

    /// Toy policy: events carry an id, a total-order key, and whether they
    /// require at least one prior event. `verify` reads `prior` (a set of
    /// prior ids) so the tests exercise the strictly-prior contract.
    struct Toy;

    #[derive(Clone)]
    struct ToyEvent {
        id: u32,
        order: u64,
        needs_prior: bool,
    }

    #[derive(Debug, PartialEq, Eq)]
    enum ToyErr {
        NoPrior,
    }

    impl LogPolicy for Toy {
        type Event = ToyEvent;
        type EventId = u32;
        type State = BTreeSet<u32>;
        type Context = ();
        type Error = ToyErr;
        type SupersessionKey = ();

        fn event_id(e: &ToyEvent) -> u32 {
            e.id
        }
        fn cmp(a: &ToyEvent, b: &ToyEvent) -> Ordering {
            a.order.cmp(&b.order).then(a.id.cmp(&b.id))
        }
        fn verify(e: &ToyEvent, prior: &BTreeSet<u32>, _ctx: &()) -> Result<(), ToyErr> {
            if e.needs_prior && prior.is_empty() {
                Err(ToyErr::NoPrior)
            } else {
                Ok(())
            }
        }
        fn materialize(events: &[&ToyEvent], _ctx: &()) -> BTreeSet<u32> {
            events.iter().map(|e| e.id).collect()
        }
    }

    fn ev(id: u32, order: u64, needs_prior: bool) -> ToyEvent {
        ToyEvent {
            id,
            order,
            needs_prior,
        }
    }

    #[test]
    fn insert_new_event_is_inserted_and_materialized() {
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        assert_eq!(log.insert(ev(1, 10, false), &()), InsertOutcome::Inserted);
        assert_eq!(log.len(), 1);
        assert!(log.contains(&1));
        assert_eq!(log.materialize(&()), BTreeSet::from([1]));
    }

    #[test]
    fn duplicate_id_is_already_known_without_verifying() {
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        assert_eq!(log.insert(ev(1, 10, false), &()), InsertOutcome::Inserted);
        // A second event with the SAME id but needs_prior=true would be
        // Rejected if verify ran; AlreadyKnown proves dedup short-circuits it.
        assert_eq!(log.insert(ev(1, 5, true), &()), InsertOutcome::AlreadyKnown);
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn needs_prior_on_empty_log_is_rejected() {
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        assert_eq!(
            log.insert(ev(1, 10, true), &()),
            InsertOutcome::Rejected(ToyErr::NoPrior)
        );
        assert_eq!(log.len(), 0);
    }

    #[test]
    fn verify_sees_only_strictly_prior_events() {
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        // An event ordered LATER than the candidate is not "prior": insert a
        // late event first, then an early needs_prior event whose prior set
        // is empty -> Rejected.
        assert_eq!(log.insert(ev(2, 20, false), &()), InsertOutcome::Inserted);
        assert_eq!(
            log.insert(ev(1, 10, true), &()),
            InsertOutcome::Rejected(ToyErr::NoPrior)
        );
        // But an event ordered AFTER an existing one sees it as prior -> Inserted.
        assert_eq!(log.insert(ev(3, 30, true), &()), InsertOutcome::Inserted);
        assert_eq!(log.materialize(&()), BTreeSet::from([2, 3]));
    }

    #[test]
    fn from_verified_events_skips_verification_and_dedups() {
        // An event that `insert` WOULD reject (needs_prior on an empty prior)
        // is still loaded when trusted; a duplicate id collapses to one entry.
        let log: VerifiedLog<Toy> =
            VerifiedLog::from_verified_events([ev(1, 10, true), ev(2, 20, true), ev(1, 99, false)]);
        assert_eq!(log.len(), 2);
        assert!(log.contains(&1));
        assert!(log.contains(&2));
        // The later id-1 duplicate won.
        assert_eq!(log.get(&1).unwrap().order, 99);
    }

    #[test]
    fn events_iterate_in_event_id_order() {
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        log.insert(ev(3, 30, false), &());
        log.insert(ev(1, 10, false), &());
        log.insert(ev(2, 20, false), &());
        let ids: Vec<u32> = log.events().map(|e| e.id).collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    // ── ZEB-813 supersession ───────────────────────────────────────────

    /// Supersession toy: same `key`, greater order supersedes. `verify`
    /// counts invocations through the context so tests can prove
    /// `Superseded` skips it.
    struct SupToy;

    #[derive(Clone)]
    struct SupEvent {
        id: u32,
        order: u64,
        key: u32,
    }

    impl LogPolicy for SupToy {
        type Event = SupEvent;
        type EventId = u32;
        type State = BTreeSet<u32>;
        type Context = core::cell::Cell<u32>;
        type Error = ();
        type SupersessionKey = u32;

        fn event_id(e: &SupEvent) -> u32 {
            e.id
        }
        fn cmp(a: &SupEvent, b: &SupEvent) -> Ordering {
            a.order.cmp(&b.order).then(a.id.cmp(&b.id))
        }
        fn verify(
            _e: &SupEvent,
            _prior: &BTreeSet<u32>,
            ctx: &core::cell::Cell<u32>,
        ) -> Result<(), ()> {
            ctx.set(ctx.get() + 1);
            Ok(())
        }
        fn materialize(events: &[&SupEvent], _ctx: &core::cell::Cell<u32>) -> BTreeSet<u32> {
            events.iter().map(|e| e.id).collect()
        }
        fn supersession_key(e: &SupEvent) -> Option<u32> {
            Some(e.key)
        }
    }

    fn sup(id: u32, order: u64, key: u32) -> SupEvent {
        SupEvent { id, order, key }
    }

    #[test]
    fn supersession_converges_across_all_arrival_orders() {
        let evs = [sup(1, 10, 7), sup(2, 20, 7), sup(3, 30, 7)];
        let orders: [[usize; 3]; 6] = [
            [0, 1, 2],
            [0, 2, 1],
            [1, 0, 2],
            [1, 2, 0],
            [2, 0, 1],
            [2, 1, 0],
        ];
        for order in orders {
            let ctx = core::cell::Cell::new(0);
            let mut log: VerifiedLog<SupToy> = VerifiedLog::new();
            for i in order {
                log.insert(evs[i].clone(), &ctx);
            }
            let ids: Vec<u32> = log.events().map(|e| e.id).collect();
            assert_eq!(
                ids,
                vec![3],
                "arrival order {order:?} must retain only newest"
            );
        }
    }

    #[test]
    fn different_keys_never_supersede() {
        let ctx = core::cell::Cell::new(0);
        let mut log: VerifiedLog<SupToy> = VerifiedLog::new();
        assert_eq!(log.insert(sup(1, 10, 7), &ctx), InsertOutcome::Inserted);
        assert_eq!(log.insert(sup(2, 20, 8), &ctx), InsertOutcome::Inserted);
        assert_eq!(log.len(), 2);
    }

    #[test]
    fn superseded_on_arrival_skips_verify() {
        let ctx = core::cell::Cell::new(0);
        let mut log: VerifiedLog<SupToy> = VerifiedLog::new();
        assert_eq!(log.insert(sup(2, 20, 7), &ctx), InsertOutcome::Inserted);
        assert_eq!(ctx.get(), 1);
        assert_eq!(log.insert(sup(1, 10, 7), &ctx), InsertOutcome::Superseded);
        assert_eq!(ctx.get(), 1, "Superseded must not run verify");
        assert_eq!(log.len(), 1);
    }

    #[test]
    fn from_verified_events_compacts() {
        let log: VerifiedLog<SupToy> =
            VerifiedLog::from_verified_events([sup(1, 10, 7), sup(2, 20, 7), sup(3, 30, 9)]);
        let ids: Vec<u32> = log.events().map(|e| e.id).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn default_policy_never_supersedes() {
        // `Toy` has no `supersedes` impl — the default keeps every event
        // and never returns Superseded, identical to pre-ZEB-813 behavior.
        let mut log: VerifiedLog<Toy> = VerifiedLog::new();
        for id in [1u32, 2, 3] {
            assert_eq!(
                log.insert(ev(id, id as u64, false), &()),
                InsertOutcome::Inserted
            );
        }
        assert_eq!(log.len(), 3);
    }
}

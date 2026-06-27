//! Generic DHT-rendezvous resolve kernel.
//!
//! "Find a live serving peer for a topic via signed DHT slots derived from a
//! shared key." The subtle behaviors live here once — escalating concurrent
//! probe, first-responder-wins, hung-probe immunity, per-batch deadline, and
//! freshness re-sampled *after* the await — so every consumer (community
//! open-join, friend Case-D, …) shares one proven driver and supplies only its
//! own slot info-layout + payload decoder.
//!
//! The slot keying lives in [`crate::derive`]; this module is only the driver
//! on top, so a consumer migrating onto it changes **no key bytes**.

use crate::derive::{derive_ephemeral_key, PkarrCase};
use crate::epoch::epoch_tolerance_window;
use crate::resolver::PkarrResolver;
use std::sync::Arc;
use std::time::Duration;
use zeroize::Zeroizing;

/// Widening schedule + per-batch deadline for an escalating rendezvous resolve.
pub struct RendezvousResolveConfig {
    /// Widening curve of batch widths, e.g. `[1, 2, 4]`: probe slot 0, then
    /// slots 0..1, then slots 0..3. `[1]` is the degenerate single-slot resolve
    /// (the friend Case-D shape). Each width should be clamped to the consumer's
    /// slot count: the driver fans out one concurrent probe per (slot, epoch)
    /// pair across `0..width`, so realistic curves are a handful of slots, not
    /// thousands. The kernel additionally caps each width to the u16 slot space
    /// as a backstop against a wrapped cast.
    pub batch_curve: Vec<usize>,
    /// Per-batch resolve deadline: on it elapsing, widen to the next batch
    /// rather than hanging on a slow/stuck probe.
    pub per_batch_deadline: Duration,
}

impl Default for RendezvousResolveConfig {
    /// A reasonable escalating default; consumers with a known slot count build
    /// an explicit curve (and own any env-var parsing).
    fn default() -> Self {
        Self {
            batch_curve: vec![1, 2, 4],
            per_batch_deadline: Duration::from_millis(2_500),
        }
    }
}

/// Result of an escalating-batch resolve, carrying the instrumentation a
/// consumer's tuning needs: which slot answered, how long it took, and how many
/// widening batches were probed.
#[derive(Debug)]
pub struct RendezvousResolveOutcome<P> {
    pub payload: Option<P>,
    pub winning_slot: Option<u16>,
    pub elapsed_ms: u64,
    pub batches_tried: usize,
}

// Hand-impl (NOT derive): a derived Default would wrongly require `P: Default`.
impl<P> Default for RendezvousResolveOutcome<P> {
    fn default() -> Self {
        Self {
            payload: None,
            winning_slot: None,
            elapsed_ms: 0,
            batches_tried: 0,
        }
    }
}

/// Probe one rendezvous slot at one epoch. Returns `Some` only for a live,
/// freshness-valid record decoded into `P`. The production impl
/// ([`PkarrSlotResolver`]) derives the slot verifying-key and queries pkarr;
/// tests inject a deterministic stub.
///
/// `P: Send` is part of the contract: the `#[async_trait]` futures are
/// `Send`-boxed and the driver runs them in a `Send` task, so any payload type
/// must cross threads — declaring it here makes the requirement explicit instead
/// of surfacing as a confusing impl-site error.
///
/// Implementations should be **cancellation-safe**: the driver drops in-flight
/// `resolve_slot` futures when another slot wins the batch or when the per-batch
/// deadline elapses, so avoid non-idempotent side effects that can't be safely
/// abandoned mid-probe.
#[async_trait::async_trait]
pub trait SlotResolver<P: Send> {
    async fn resolve_slot(&self, slot_index: u16, epoch_id: u64) -> Option<P>;
}

/// Escalating-batch rendezvous resolve over any [`SlotResolver`] (`now_ms` is
/// supplied so the driver stays clock-free apart from the per-batch deadline).
/// For each width `w` in `cfg.batch_curve`, probe slots `0..w` across the
/// epoch-tolerance window CONCURRENTLY and return on the FIRST live record — the
/// first slot to respond wins (not strictly the lowest), so one hung/slow probe
/// can never stall discovery. Each batch is bounded by `cfg.per_batch_deadline`:
/// on the deadline elapsing OR all probes returning `None`, widen to the next
/// width. Returns an empty outcome (cold start) if no slot answers.
pub async fn resolve_rendezvous_with<P: Send, R: SlotResolver<P> + Sync>(
    resolver: &R,
    now_ms: u64,
    cfg: &RendezvousResolveConfig,
) -> RendezvousResolveOutcome<P> {
    use futures::stream::{FuturesUnordered, StreamExt};

    let started = std::time::Instant::now();
    let epoch_window = epoch_tolerance_window(now_ms);
    let mut outcome = RendezvousResolveOutcome::default();

    for &width in &cfg.batch_curve {
        outcome.batches_tried += 1;
        // Clamp the batch width into the u16 slot space before casting: a width
        // at/above 65_536 would wrap `width as u16` (65_536 -> 0) and silently
        // probe an empty range. Consumers are expected to clamp to their own slot
        // count, but the kernel defends itself against a misconfigured curve.
        let slot_count = width.min(usize::from(u16::MAX) + 1);
        // Probe every (slot, epoch) pair in this batch concurrently, draining
        // them as they complete so the FIRST live slot wins without waiting on
        // slower/hung probes. Bounded by the per-batch deadline.
        let mut probes: FuturesUnordered<_> = (0..slot_count)
            .flat_map(|slot| {
                let slot = slot as u16;
                epoch_window.iter().map(move |&epoch_id| async move {
                    resolver
                        .resolve_slot(slot, epoch_id)
                        .await
                        .map(|payload| (slot, payload))
                })
            })
            .collect();

        let winner = tokio::time::timeout(cfg.per_batch_deadline, async {
            while let Some(result) = probes.next().await {
                if let Some((slot, payload)) = result {
                    return Some((slot, payload));
                }
            }
            None
        })
        .await
        // On the batch deadline elapsing (Err), treat the batch as exhausted and
        // widen to the next width rather than hanging.
        .unwrap_or(None);

        if let Some((slot, payload)) = winner {
            outcome.winning_slot = Some(slot);
            outcome.payload = Some(payload);
            outcome.elapsed_ms = started.elapsed().as_millis() as u64;
            tracing::debug!(
                winning_slot = slot,
                elapsed_ms = outcome.elapsed_ms,
                batches_tried = outcome.batches_tried,
                "rendezvous resolved"
            );
            return outcome;
        }
    }

    outcome.elapsed_ms = started.elapsed().as_millis() as u64;
    tracing::debug!(
        elapsed_ms = outcome.elapsed_ms,
        batches_tried = outcome.batches_tried,
        "rendezvous resolve found no live slot (cold start)"
    );
    outcome
}

/// Production [`SlotResolver`]: derives the per-slot verifying-key from `ikm`
/// under `case` + the consumer's `info_for(slot, epoch)` layout, queries pkarr,
/// re-samples freshness AFTER the await, and decodes the routing blob into `P`.
///
/// [`PkarrResolver::resolve`] already verifies the outer BEP44 envelope (proving
/// the writer held the per-slot key derived from `ikm`) AND the record's inner
/// identity signature. What this layer does NOT do is bind that identity to a
/// *trusted* beacon: a rendezvous joiner may not know the beacon's identity in
/// advance, so the trust decision is deferred to the consumer's
/// handshake/admission layer.
///
/// `ikm` is held in a [`Zeroizing`] buffer — for shared-secret cases (e.g.
/// [`PkarrCase::Friend`]) it is sensitive key material that should not linger in
/// freed memory.
pub struct PkarrSlotResolver<P, F>
where
    F: Fn(&[u8]) -> Option<P>,
{
    pub pkarr: Arc<PkarrResolver>,
    pub case: PkarrCase,
    pub ikm: Zeroizing<Vec<u8>>,
    pub info_for: Arc<dyn Fn(u16, u64) -> Vec<u8> + Send + Sync>,
    pub decode: F,
}

#[async_trait::async_trait]
impl<P, F> SlotResolver<P> for PkarrSlotResolver<P, F>
where
    P: Send,
    F: Fn(&[u8]) -> Option<P> + Send + Sync,
{
    async fn resolve_slot(&self, slot_index: u16, epoch_id: u64) -> Option<P> {
        let info = (self.info_for)(slot_index, epoch_id);
        let vk = derive_ephemeral_key(self.case, &self.ikm, &info).verifying_key();
        // Both a hard backend error and a clean miss yield `None` (a failed probe
        // is "no beacon at this slot — widen/retry"), but log the error case so a
        // DHT/relay outage is diagnosable instead of silently looking like an
        // empty slot.
        let rec = match self.pkarr.resolve(&vk).await {
            Ok(Some(rec)) => rec,
            Ok(None) => return None,
            Err(e) => {
                tracing::debug!(slot = slot_index, error = ?e, "rendezvous slot probe errored — treating as a miss");
                return None;
            }
        };
        // Re-sample the wall clock AFTER the awaited resolve so freshness is
        // checked against "now", not a timestamp captured before a possibly long
        // network round-trip (the stale-clock bug fixed in PR#306).
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        rec.verify_freshness(now_ms).ok()?;
        (self.decode)(rec.routing_blob.as_slice())
    }
}

/// Deterministic slot claim: sort the advertiser set ascending, dedup, and
/// return the rank of `me` as its slot index — `None` if `me` is absent or ranks
/// at/beyond `cap`. Because the advertiser set is consumer-replicated (e.g. a
/// CRDT), every member computes the same ordering, so each slot has exactly one
/// writer.
pub fn slot_for_advertiser<A: Ord>(advertisers: &[A], me: &A, cap: usize) -> Option<u16> {
    // Borrow rather than copy so non-`Copy` ordered identifiers (e.g. `String`,
    // `Vec<u8>`) work, not just fixed-size keys.
    let mut sorted: Vec<&A> = advertisers.iter().collect();
    sorted.sort_unstable();
    sorted.dedup();
    let rank = sorted.iter().position(|a| *a == me)?;
    // Reject ranks at/beyond the cap, and any rank not representable as a u16
    // slot index — truncating `rank as u16` would alias two advertisers onto the
    // same slot, breaking the one-writer-per-slot guarantee.
    if rank >= cap || rank > usize::from(u16::MAX) {
        return None;
    }
    Some(rank as u16)
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- slot_for_advertiser (generic over an Ord+Copy address) ---

    #[test]
    fn slot_assignment_is_deterministic_across_members() {
        // Two members compute the SAME ordering from the same (unordered) set.
        let set_a = [3u32, 1, 2];
        let set_b = [2u32, 3, 1];
        for who in [1u32, 2, 3] {
            assert_eq!(
                slot_for_advertiser(&set_a, &who, 4),
                slot_for_advertiser(&set_b, &who, 4),
                "ordering disagreed for {who}"
            );
        }
        assert_eq!(slot_for_advertiser(&set_a, &1, 4), Some(0));
        assert_eq!(slot_for_advertiser(&set_a, &2, 4), Some(1));
        assert_eq!(slot_for_advertiser(&set_a, &3, 4), Some(2));
    }

    #[test]
    fn not_in_set_returns_none() {
        assert_eq!(slot_for_advertiser(&[1u32, 2], &9, 4), None);
    }

    #[test]
    fn rank_beyond_cap_returns_none() {
        // cap=4 fills slots 0..3; a 5th (highest) ranks 4 >= cap → no slot.
        let set = [1u32, 2, 3, 4, 5];
        assert_eq!(slot_for_advertiser(&set, &5, 4), None);
        assert_eq!(slot_for_advertiser(&set, &4, 4), Some(3));
    }

    #[test]
    fn duplicate_addresses_do_not_shift_ranks() {
        let set = [1u32, 2, 2, 3];
        assert_eq!(slot_for_advertiser(&set, &1, 4), Some(0));
        assert_eq!(slot_for_advertiser(&set, &2, 4), Some(1));
        assert_eq!(slot_for_advertiser(&set, &3, 4), Some(2));
    }

    // --- escalating-batch driver (generic over a payload P) ---

    #[derive(Clone, Debug, PartialEq)]
    struct Beacon(u8);

    /// Deterministic resolver: answers only for one configured live slot (or
    /// never). Ignores `epoch_id` — the escalating-batch logic is under test,
    /// not epoch derivation.
    struct StubResolver {
        live_slot: Option<u16>,
    }

    #[async_trait::async_trait]
    impl SlotResolver<Beacon> for StubResolver {
        async fn resolve_slot(&self, slot_index: u16, _epoch_id: u64) -> Option<Beacon> {
            (Some(slot_index) == self.live_slot).then_some(Beacon(slot_index as u8))
        }
    }

    fn community_curve() -> RendezvousResolveConfig {
        RendezvousResolveConfig {
            batch_curve: vec![1, 2, 4],
            per_batch_deadline: Duration::from_millis(2_500),
        }
    }

    #[tokio::test]
    async fn returns_slot0_without_widening_when_slot0_is_live() {
        let stub = StubResolver { live_slot: Some(0) };
        let out = resolve_rendezvous_with(&stub, 1_000_000, &community_curve()).await;
        assert_eq!(out.winning_slot, Some(0));
        assert_eq!(
            out.batches_tried, 1,
            "should not widen past the first batch"
        );
        assert_eq!(out.payload, Some(Beacon(0)));
    }

    #[tokio::test]
    async fn widens_to_find_a_live_slot_when_slot0_is_dead() {
        let stub = StubResolver { live_slot: Some(2) }; // only slot 2 answers
        let out = resolve_rendezvous_with(&stub, 1_000_000, &community_curve()).await;
        assert_eq!(out.winning_slot, Some(2));
        assert!(out.batches_tried >= 3, "had to widen to the full set");
        assert_eq!(out.payload, Some(Beacon(2)));
    }

    #[tokio::test]
    async fn cold_start_returns_none() {
        let stub = StubResolver { live_slot: None };
        let out = resolve_rendezvous_with(&stub, 1_000_000, &community_curve()).await;
        assert_eq!(out.payload, None);
        assert_eq!(out.winning_slot, None);
    }

    /// Resolver whose slot 0 NEVER completes (hangs) but whose slot 1 answers
    /// live — proves the resolve returns the first-responding slot and never
    /// blocks on the hung probe.
    struct HungSlot0Resolver;

    #[async_trait::async_trait]
    impl SlotResolver<Beacon> for HungSlot0Resolver {
        async fn resolve_slot(&self, slot_index: u16, _epoch_id: u64) -> Option<Beacon> {
            if slot_index == 0 {
                std::future::pending::<()>().await; // models a hung/dropped probe
                unreachable!("pending() never resolves");
            }
            (slot_index == 1).then_some(Beacon(1))
        }
    }

    #[tokio::test]
    async fn hung_probe_does_not_block_a_live_higher_slot() {
        let out = tokio::time::timeout(
            Duration::from_secs(10),
            resolve_rendezvous_with(&HungSlot0Resolver, 1_000_000, &community_curve()),
        )
        .await
        .expect("resolve must not hang on a stuck slot-0 probe");
        assert_eq!(out.winning_slot, Some(1));
        assert_eq!(out.payload, Some(Beacon(1)));
    }

    /// Friend Case-D shape: a single-slot (`[1]`) curve resolves slot 0 — proves
    /// the kernel fits the degenerate N=1 consumer, not just the community
    /// N-slot one, and never probes a slot outside the curve width.
    #[tokio::test]
    async fn friend_shape_single_slot_curve_resolves() {
        let cfg = RendezvousResolveConfig {
            batch_curve: vec![1],
            per_batch_deadline: Duration::from_millis(2_500),
        };
        let live = StubResolver { live_slot: Some(0) };
        let out = resolve_rendezvous_with(&live, 1_000_000, &cfg).await;
        assert_eq!(out.winning_slot, Some(0));
        assert_eq!(out.batches_tried, 1);
        assert_eq!(out.payload, Some(Beacon(0)));
        // A single-slot curve must NOT probe slot 1 even if slot 1 is "live".
        let only_slot1 = StubResolver { live_slot: Some(1) };
        let out2 = resolve_rendezvous_with(&only_slot1, 1_000_000, &cfg).await;
        assert_eq!(
            out2.payload, None,
            "single-slot curve must not probe slot 1"
        );
    }
}

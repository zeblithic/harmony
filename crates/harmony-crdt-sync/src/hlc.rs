// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Hybrid-logical-clock tick arithmetic: the pure rule for "what is the
//! next clock reading this writer may stamp?"
//!
//! A replicated writer stamps each publish with a clock that must be
//! **strictly newer** than its own previous stamp, because the receiving
//! side's replay protection accepts a publish only if it beats the last one
//! recorded for that source (see [`replay_admission`](crate::replay_admission)).
//! A bare wall clock cannot supply that: two publishes inside the same
//! millisecond tie, and a backward NTP correction goes *backwards*. The
//! hybrid rule pairs the wall reading with a logical counter that breaks
//! ties and absorbs backward steps.
//!
//! [`HlcTick`] is deliberately **identity-free** — it carries only
//! `(wall_ms, logical)`, never a device/writer id, and has no serde
//! derives. A domain's on-the-wire clock type (which typically appends the
//! writer id and carries a pinned encoding) stays in the caller's crate and
//! composes this for the arithmetic alone. That keeps the wire contract
//! where the domain owns it while the subtle monotonicity rule lives in one
//! audited place.
//!
//! Reading the wall clock is the caller's job: [`HlcTick::next`] takes
//! `wall_ms` as a parameter, so the whole rule — including its pathological
//! branches — is testable without touching a clock.
//!
//! ```
//! use harmony_crdt_sync::hlc::HlcTick;
//!
//! // First stamp from a fresh writer.
//! let t0 = HlcTick::next(None, 1_000);
//! assert_eq!(t0, HlcTick { wall_ms: 1_000, logical: 0 });
//!
//! // Second stamp in the SAME millisecond: the logical counter breaks the tie.
//! let t1 = HlcTick::next(Some(t0), 1_000);
//! assert_eq!(t1, HlcTick { wall_ms: 1_000, logical: 1 });
//! assert!(t1 > t0);
//!
//! // The wall clock jumps BACKWARD (NTP correction): the tick still advances.
//! let t2 = HlcTick::next(Some(t1), 900);
//! assert_eq!(t2, HlcTick { wall_ms: 1_000, logical: 2 });
//! assert!(t2 > t1);
//!
//! // A later millisecond resets the logical counter.
//! let t3 = HlcTick::next(Some(t2), 2_000);
//! assert_eq!(t3, HlcTick { wall_ms: 2_000, logical: 0 });
//! assert!(t3 > t2);
//! ```
//!
//! Provenance: extracted from harmony-client's `fleet_sync::compute_next_hlc`
//! (ZEB-571 item 6b), behavior-preserving. The clock read, the tracker
//! lock, and the wire clock type stay caller-side.

/// One hybrid-logical-clock reading: a wall-clock millisecond paired with a
/// logical counter that disambiguates stamps sharing a millisecond.
///
/// The derived [`Ord`] is lexicographic on `(wall_ms, logical)` — field
/// order is load-bearing, since that ordering is exactly the "strictly
/// newer" relation replay protection tests. A domain clock that appends a
/// writer id keeps the same shape by declaring that field last.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct HlcTick {
    /// Wall-clock milliseconds. Never decreases across successive
    /// [`next`](HlcTick::next) results for one writer, even when the
    /// underlying clock steps backward.
    pub wall_ms: u64,
    /// Tie-breaking counter within `wall_ms`, reset to 0 whenever the
    /// effective wall reading advances.
    pub logical: u32,
}

impl HlcTick {
    /// Compute the tick strictly after `prev`, given the writer's current
    /// wall-clock reading in milliseconds.
    ///
    /// The result is **always** strictly greater than `prev` (when `prev`
    /// is `Some`), which is the property the caller's replay protection
    /// depends on. Two cases advance the logical counter instead of the
    /// wall reading:
    ///
    /// 1. `wall_ms == prev.wall_ms` — a second stamp inside the same
    ///    millisecond, the common case under a burst of writes.
    /// 2. `wall_ms < prev.wall_ms` — the wall clock stepped backward (NTP
    ///    correction, VM snapshot restore). The previous wall reading is
    ///    kept, so the tick never regresses.
    ///
    /// `logical` uses a **saturating** add. Under a sustained backward
    /// correction or repeated clock faults case 2 repeats without `wall_ms`
    /// ever advancing; an unchecked `u32` add would eventually wrap and
    /// produce a tick *smaller* than its predecessor, silently breaking the
    /// strict-newer monotonicity replay protection rests on. Saturation
    /// instead pins the counter at [`u32::MAX`]: pathological but bounded.
    /// Further stamps from that writer on that wall reading then tie rather
    /// than advance, so the receiver rejects them as duplicates until the
    /// wall clock catches up — a stall, which is strictly preferable to
    /// admitting a replay.
    #[must_use]
    pub fn next(prev: Option<HlcTick>, wall_ms: u64) -> HlcTick {
        let (logical, prev_wall) = match prev {
            // Cases 1 and 2 above: the wall reading did not advance past
            // `prev`, so the logical counter carries the strict increase and
            // `prev.wall_ms` is retained as the floor.
            Some(p) if p.wall_ms >= wall_ms => (p.logical.saturating_add(1), p.wall_ms),
            // The wall reading advanced: it alone makes the tick strictly
            // newer, so the counter restarts.
            Some(p) => (0, p.wall_ms),
            None => (0, 0),
        };

        HlcTick {
            wall_ms: core::cmp::max(wall_ms, prev_wall),
            logical,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_tick_takes_the_wall_reading_with_zero_logical() {
        assert_eq!(
            HlcTick::next(None, 1_234),
            HlcTick {
                wall_ms: 1_234,
                logical: 0
            }
        );
    }

    #[test]
    fn same_millisecond_advances_the_logical_counter() {
        let a = HlcTick::next(None, 500);
        let b = HlcTick::next(Some(a), 500);
        let c = HlcTick::next(Some(b), 500);

        assert_eq!(b.wall_ms, 500);
        assert_eq!((a.logical, b.logical, c.logical), (0, 1, 2));
        assert!(b > a && c > b, "strictly newer within one millisecond");
    }

    #[test]
    fn advancing_wall_clock_resets_the_logical_counter() {
        let a = HlcTick {
            wall_ms: 500,
            logical: 7,
        };
        let b = HlcTick::next(Some(a), 501);

        assert_eq!(
            b,
            HlcTick {
                wall_ms: 501,
                logical: 0
            }
        );
        assert!(b > a);
    }

    // The load-bearing case: a backward wall-clock step must NOT produce a
    // tick that compares older, or the peer's replay check would reject every
    // subsequent publish from this writer as a duplicate.
    #[test]
    fn backward_wall_clock_step_still_advances() {
        let a = HlcTick {
            wall_ms: 10_000,
            logical: 0,
        };
        let b = HlcTick::next(Some(a), 9_000);
        let c = HlcTick::next(Some(b), 1);

        assert_eq!(b.wall_ms, 10_000, "wall reading is floored at the previous");
        assert_eq!(b.logical, 1);
        assert!(b > a, "strictly newer despite the clock going backwards");
        assert!(c > b, "and again on a further backward reading");
    }

    // Saturation is the safe failure: the tick STOPS advancing rather than
    // wrapping to a smaller value, which would admit a replay.
    #[test]
    fn logical_saturates_instead_of_wrapping() {
        let maxed = HlcTick {
            wall_ms: 10_000,
            logical: u32::MAX,
        };
        let next = HlcTick::next(Some(maxed), 9_000);

        assert_eq!(next, maxed, "pinned at the cap, never wrapped");
        assert!(
            !(next > maxed),
            "a saturated tick ties rather than regressing — the receiver \
             rejects it as a duplicate instead of accepting a replay"
        );

        // Recovery: once the wall clock passes the pinned reading, the
        // counter restarts and progress resumes.
        let recovered = HlcTick::next(Some(maxed), 10_001);
        assert_eq!(
            recovered,
            HlcTick {
                wall_ms: 10_001,
                logical: 0
            }
        );
        assert!(recovered > maxed);
    }

    #[test]
    fn ordering_is_lexicographic_on_wall_then_logical() {
        let earlier_wall = HlcTick {
            wall_ms: 1,
            logical: u32::MAX,
        };
        let later_wall = HlcTick {
            wall_ms: 2,
            logical: 0,
        };
        assert!(
            later_wall > earlier_wall,
            "wall_ms dominates the logical counter"
        );
    }
}

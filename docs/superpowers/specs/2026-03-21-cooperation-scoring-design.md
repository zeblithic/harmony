# Adjacent Peer Cooperation Scoring

**Bead:** harmony-1fu
**Date:** 2026-03-21
**Status:** Draft

## Problem

The Reticulum transport layer routes packets based solely on hop count and
announce recency. There is no feedback on whether a neighbor actually forwards
traffic — a silent-dropping neighbor is indistinguishable from a reliable one.
On an 802.11s mesh where multiple paths exist, the node has no way to prefer
reliable relays over unreliable ones.

## Solution

Add a `CooperationTable` to the Node state machine that tracks per-interface
cooperation scores using exponential moving averages (EMA). Two signals feed
the scores: announce forwarding (control plane) and proof delivery (data plane).
Scores influence broadcast routing via probabilistic weighted send — the
highest-scored interface always sends, lower-scored interfaces send
proportionally less, but never zero (0.05 floor for self-healing).

## Design Decisions

### Two signals: announce forwarding + proof delivery

- **Announce forwarding** (70% weight): When we hear our own announce come
  back via an interface, the neighbor on that interface forwarded it. Strongest
  signal — control plane cooperation is more fundamental than data plane.
- **Proof delivery** (30% weight): When a packet sent through an interface
  results in a delivery proof returning, the neighbor relayed the data. Covers
  the data plane.

Announce forwarding was chosen as the primary signal because it's harder to
game (announces are broadcast, not targeted) and provides a baseline even when
no data traffic flows.

### EMA scoring model

Each observation updates the score:
```
score = α * observation + (1 - α) * score
score = max(score, floor)
```

- `α = 0.1` — each observation accounts for 10% of the new score. ~10-15
  observations to converge meaningfully from the initial value.
- `floor = 0.05` — minimum score. Prevents complete starvation. Implements
  the prisoner's dilemma "always cooperate a little" dynamic.
- `initial = 0.5` — new/unknown peers start neutral. Converges within 2-3
  minutes of normal mesh traffic.
- `observation = 1.0` (success) or `0.0` (failure).

EMA was chosen over windowed counters or Bayesian models for simplicity:
constant memory per peer (two floats), single parameter, well-understood
decay properties, bounded 0.0–1.0.

### Staleness decay

If no observations arrive for an interface within a staleness window (default
300 seconds / 5 minutes), the score decays toward `initial` (0.5) at a rate
of 10% per window. This prevents permanently-departed peers from retaining
stale high scores and prevents permanently-bad-but-quiet peers from retaining
stale low scores. A peer that returns after absence gets a fair re-evaluation.

### Interface-level scoring (not multi-path)

Scores are per-interface, not per-destination-path. The PathTable continues
to store a single best path per destination. Scoring affects the broadcast
path — which interfaces receive rebroadcasted announces and broadcast packets.
This is where most mesh traffic flows today. Multi-path routing can be added
later by extending PathTable to store alternate routes.

### Weighted probabilistic send with guaranteed delivery

The Node emits `SendOnInterface` actions with a `weight` field (0.05–1.0)
derived from the cooperation score. The I/O layer (event loop) generates a
random float; if `rand < weight`, it sends. This keeps the sans-I/O boundary
clean — the Node computes weights, the I/O layer makes the random decision.

**Guaranteed minimum delivery:** The highest-scored interface in each
broadcast set always sends at weight 1.0, regardless of its actual score.
This prevents the pathological case where all interfaces are at floor (0.05)
and ~90% of broadcasts are silently dropped. At least one interface always
delivers every broadcast.

**Exception:** Directed packets (known path in PathTable) always send at
weight 1.0. Cooperation scoring only affects broadcast/rebroadcast.

**Single interface:** Weight is always 1.0 regardless of score. Don't suppress
the only available path.

## Architecture

### New file: `cooperation.rs` in `harmony-reticulum`

Uses conditional `HashMap` import for `no_std` compatibility, matching the
existing crate pattern (`hashbrown` when `no_std`, `std::collections` otherwise).

```rust
pub struct CooperationTable {
    scores: HashMap<Arc<str>, CooperationScore>,
    alpha: f32,              // default 0.1
    floor: f32,              // default 0.05
    initial: f32,            // default 0.5
    staleness_window: u64,   // default 300_000 ms (5 minutes)
}

pub struct CooperationScore {
    pub announce_score: f32,
    pub proof_score: f32,
    pub combined: f32,
    pub observation_count: u64,
    pub last_observed: u64,  // monotonic ms
}
```

**Combined score:** `combined = 0.7 * announce_score + 0.3 * proof_score`.

**Methods:**

- `new(alpha, floor, initial, staleness_window)` — construct with parameters
- `get_weight(interface: &str) -> f32` — returns combined score, or `initial`
  if interface unknown. Returns 1.0 if only one interface registered.
- `get_broadcast_weights() -> Vec<(Arc<str>, f32)>` — returns all interface
  weights with the highest-scored interface forced to 1.0.
- `observe_announce_forwarded(interface, now)` — positive announce signal
- `observe_announce_timeout(interface, now)` — negative announce signal
- `observe_proof_delivered(interface, now)` — positive proof signal
- `observe_proof_timeout(interface, now)` — negative proof signal
- `decay_stale(now)` — decay scores toward `initial` for interfaces with
  no observations within `staleness_window`. Called on timer tick.
- `remove_interface(interface)` — cleanup on deregistration
- `iter() -> impl Iterator<Item = (&str, &CooperationScore)>` — for diagnostics

### Observation events in Node

**Announce forwarding (positive):**

In `handle_announce()`: when processing a received announce, check if the
announce's destination hash matches one of our `announcing_destinations`.
If so, the interface it arrived on gets a positive announce observation.
This means our announce traversed the mesh and came back — the neighbor
forwarded it. The observation happens inside the Node state machine
(not from an external event).

Note: this detects echoes of our *originating* announces, not transport-mode
rebroadcasts. Transport node echo detection uses the separate
`check_rebroadcast_echo()` mechanism.

**Announce forwarding (negative):**

On each timer tick: for every interface where we sent an announce but haven't
heard an echo within 3 announce cycles (~22.5 seconds, computed as
`3 * filter_broadcast_interval_ticks * 250ms`), record a negative observation.
Tracked per interface+destination pair:
`pending_echoes: HashMap<(Arc<str>, DestinationHash), u64>` (key → tick sent).
Cleared when a matching echo arrives or when the timeout fires.

**Proof delivery (positive):**

In `route_proof()`: after successfully looking up the proof's packet hash in
the reverse table to find the outbound interface, record a positive proof
observation for that interface. The observation happens inside the Node
state machine during proof routing.

**Proof delivery (negative):**

In `expire_reverse_table()`: before removing expired entries, check each
for a `proof_received: bool` field. Entries that expire without a matching
proof generate a negative proof observation for their `outbound_interface`.
Requires adding a `proof_received: bool` field to `ReverseTableEntry`
(initialized to `false`, set to `true` in `route_proof()` when the proof
arrives).

### Routing integration

**SendOnInterface weight field:**

Rather than modifying the existing `SendOnInterface` variant (which would
require updating ~30+ match arms), add an optional weight to the existing
struct:

```rust
SendOnInterface {
    interface_name: Arc<str>,
    raw: Vec<u8>,
    weight: Option<f32>,  // None = always send (directed), Some = weighted (broadcast)
}
```

Existing code that constructs `SendOnInterface` passes `weight: None`
(directed sends, always delivered). Broadcast paths pass
`weight: Some(cooperation_score)`. This minimizes churn — existing match
arms that destructure `{ interface_name, raw, .. }` continue to work.

**Broadcast weighting in Node:**

When the Node emits broadcast `SendOnInterface` actions, it calls
`cooperation_table.get_broadcast_weights()` which returns all interface
weights with the highest forced to 1.0. Each `SendOnInterface` action
gets the corresponding weight.

**Event loop (harmony-node):**

```rust
RuntimeAction::SendOnInterface { interface_name, raw, weight } => {
    let should_send = match weight {
        None => true,  // directed: always send
        Some(w) if w >= 1.0 => true,  // best interface: always send
        Some(w) => rand::random::<f32>() < w,  // probabilistic
    };
    if should_send {
        // existing broadcast + unicast send logic
    }
}
```

### Edge cases

- **Single interface:** `get_weight()` returns 1.0 when only one interface
  is registered. Don't suppress the only path.
- **All interfaces at floor:** The highest-scored interface (even if at
  floor) always sends at 1.0. Others send at 0.05. At least one interface
  always delivers every broadcast.
- **Interface reappears after removal:** Starts at 0.5 (initial). Fresh
  start for rebooted neighbors.
- **Stale scores:** After 5 minutes without observations, scores decay
  toward 0.5 at 10% per window. Departed peers don't retain stale scores.
- **Announce echo from our own interface:** Filtered by comparing interface
  names. Only other interfaces get credit for forwarding.
- **Single→multi interface transition:** When a second interface is registered,
  the first interface's score is preserved (not reset). If it was accumulating
  negative observations while alone (which don't affect routing since weight
  was forced to 1.0), those low scores take effect once a second interface
  appears. This is correct — the scoring was still tracking the neighbor's
  behavior even while unable to act on it.

## Testing

**Unit tests for CooperationTable:**
- EMA convergence: 10 positive observations from 0.5 → above 0.8
- EMA decay: 10 negative observations from 0.5 → near floor
- Floor enforcement: score never below 0.05
- New interface: starts at 0.5
- Combined weighting: 70/30 split verified
- Remove interface: cleaned up
- Single interface: weight always 1.0
- Broadcast weights: highest always 1.0
- Staleness decay: no observations for 5 min → score moves toward 0.5

**Node integration tests:**
- Announce echo detection: emit announce, hear echo, verify score increase
- Announce timeout: emit announce, tick past timeout, verify score decrease
- Proof delivery: send packet, receive proof, verify score increase
- Proof timeout: reverse table entry expires without proof, verify decrease
- Weighted broadcast: statistical distribution test over many packets

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-reticulum/src/cooperation.rs` | New: CooperationTable, CooperationScore, EMA logic |
| `crates/harmony-reticulum/src/node.rs` | Node owns CooperationTable, feeds observations, emits weighted SendOnInterface |
| `crates/harmony-reticulum/src/lib.rs` | Add `pub mod cooperation;` |
| `crates/harmony-reticulum/src/path_table.rs` | Add `proof_received: bool` to ReverseTableEntry |
| `crates/harmony-node/src/event_loop.rs` | Handle `weight` field for probabilistic send |

## What is NOT in Scope

- No persistence of scores across restarts (fresh start at 0.5)
- No multi-path PathTable (single best path, scoring affects broadcast only)
- No trust integration (cooperation ≠ trust)
- No remote score reporting (local-only, no gossip)
- No per-destination scoring (per-interface only)

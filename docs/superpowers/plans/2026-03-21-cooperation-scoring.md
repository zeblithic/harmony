# Adjacent Peer Cooperation Scoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-interface cooperation scoring to the Reticulum Node, enabling weighted probabilistic broadcast that favors reliable relay neighbors.

**Architecture:** A new `cooperation.rs` module in `harmony-reticulum` implements `CooperationTable` with EMA-based scoring from two signals (announce forwarding + proof delivery). The Node owns the table alongside its PathTable, feeds observations during event processing, and tags broadcast `SendOnInterface` actions with cooperation weights. The event loop uses weights for probabilistic send decisions.

**Tech Stack:** Rust, harmony-reticulum (no_std compatible), sans-I/O state machine pattern

**Spec:** `docs/superpowers/specs/2026-03-21-cooperation-scoring-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-reticulum/src/cooperation.rs` | **New**: CooperationTable, CooperationScore, EMA logic, broadcast weight calculation |
| `crates/harmony-reticulum/src/node.rs` | Node owns CooperationTable, feeds observations, emits weighted SendOnInterface |
| `crates/harmony-reticulum/src/lib.rs` | Add `pub mod cooperation;` and re-exports |
| `crates/harmony-node/src/event_loop.rs` | Handle `weight` field on SendOnInterface for probabilistic send |

---

### Task 1: CooperationTable data structure and EMA logic (TDD)

**Files:**
- Create: `crates/harmony-reticulum/src/cooperation.rs`
- Modify: `crates/harmony-reticulum/src/lib.rs`

Pure data structure — no integration with Node yet. TDD: tests first, then implement.

- [ ] **Step 1: Add `pub mod cooperation;` to lib.rs**

In `crates/harmony-reticulum/src/lib.rs`, add after `pub mod node;`:

```rust
pub mod cooperation;
```

And add to the re-exports at the bottom:

```rust
pub use cooperation::{CooperationScore, CooperationTable};
```

- [ ] **Step 2: Create cooperation.rs with types and test stubs**

Create `crates/harmony-reticulum/src/cooperation.rs` with:

- Conditional HashMap import (`hashbrown` for no_std, `std::collections` for std)
- `CooperationTable` struct with `scores`, `alpha`, `floor`, `initial`, `staleness_window`
- `CooperationScore` struct with `announce_score`, `proof_score`, `combined`, `observation_count`, `last_observed`
- `todo!()` method stubs: `new`, `get_weight`, `get_broadcast_weights`, `observe_announce_forwarded`, `observe_announce_timeout`, `observe_proof_delivered`, `observe_proof_timeout`, `decay_stale`, `remove_interface`, `register_interface`, `interface_count`, `iter`
- Unit tests (all should fail initially with `todo!()` panics):
  1. `ema_converges_positive` — 10 positive announce observations from 0.5 → above 0.8
  2. `ema_converges_negative` — 10 negative announce observations from 0.5 → near floor
  3. `floor_enforced` — score never below 0.05
  4. `new_interface_starts_at_initial` — unknown interface returns 0.5
  5. `combined_weighting` — 70/30 announce/proof split verified
  6. `remove_interface_cleans_up` — removed interface gone from iter
  7. `single_interface_always_one` — get_weight returns 1.0 when one interface
  8. `broadcast_weights_highest_forced_one` — get_broadcast_weights forces highest to 1.0
  9. `staleness_decay` — no observations for > staleness_window → score moves toward initial

- [ ] **Step 3: Run tests — verify they fail**

Run: `cargo test -p harmony-reticulum cooperation::tests`
Expected: `todo!()` panics

- [ ] **Step 4: Implement CooperationTable**

Key implementation details:

```rust
use alloc::{sync::Arc, vec::Vec};
#[cfg(not(feature = "std"))]
use hashbrown::HashMap;
#[cfg(feature = "std")]
use std::collections::HashMap;

pub struct CooperationTable {
    scores: HashMap<Arc<str>, CooperationScore>,
    alpha: f32,
    floor: f32,
    initial: f32,
    staleness_window: u64,
}

#[derive(Debug, Clone)]
pub struct CooperationScore {
    pub announce_score: f32,
    pub proof_score: f32,
    pub combined: f32,
    pub observation_count: u64,
    pub last_observed: u64,
}
```

EMA update helper (private):
```rust
fn ema_update(&self, current: f32, success: bool) -> f32 {
    let observation = if success { 1.0 } else { 0.0 };
    let updated = self.alpha * observation + (1.0 - self.alpha) * current;
    updated.max(self.floor)
}
```

Combined score: `0.7 * announce_score + 0.3 * proof_score`

`get_broadcast_weights()`: collect all `(interface, combined)` pairs, find the max, force that one to 1.0, return all.

`decay_stale(now)`: for each score where `now - last_observed > staleness_window`, move 10% toward `initial`: `score = score + 0.1 * (initial - score)`.

`register_interface(name)`: insert with `initial` scores if not already present.

- [ ] **Step 5: Run tests — verify they pass**

Run: `cargo test -p harmony-reticulum cooperation::tests -v`
Expected: all 9 tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-reticulum/src/cooperation.rs crates/harmony-reticulum/src/lib.rs
git commit -m "feat(reticulum): add CooperationTable with EMA scoring

Per-interface cooperation scores with announce forwarding (70%) and
proof delivery (30%) signals. Floor at 0.05 for self-healing.
Broadcast weights force highest to 1.0 for guaranteed delivery.
9 unit tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Add weight to SendOnInterface and update Node

**Files:**
- Modify: `crates/harmony-reticulum/src/node.rs`

Add the `weight` field to `SendOnInterface` and wire the CooperationTable into the Node.

- [ ] **Step 1: Add `weight: Option<f32>` to SendOnInterface**

In the `NodeAction` enum (line ~244), change:

```rust
    SendOnInterface {
        interface_name: Arc<str>,
        raw: Vec<u8>,
    },
```

to:

```rust
    /// Send raw bytes on the named interface.
    /// `weight`: `None` = always send (directed), `Some(w)` = probabilistic (broadcast).
    SendOnInterface {
        interface_name: Arc<str>,
        raw: Vec<u8>,
        weight: Option<f32>,
    },
```

- [ ] **Step 2: Add CooperationTable to Node struct**

Add to the `Node` struct (line ~296):

```rust
    cooperation: CooperationTable,
```

Add `use crate::cooperation::CooperationTable;` at the top.

Initialize in `new()` and `new_transport()`:

```rust
    cooperation: CooperationTable::default(),
```

Implement `Default` for `CooperationTable` if not already done (alpha=0.1, floor=0.05, initial=0.5, staleness=300_000).

- [ ] **Step 3: Update all SendOnInterface construction sites**

Every place that constructs `NodeAction::SendOnInterface` needs `weight`:

- `send_on_interface()` helper (line ~628): this is for directed sends via the path table. Add `weight: None`.
- Broadcast paths (announce rebroadcast, unknown-destination broadcast): Add `weight: Some(cooperation.get_weight(&interface_name))`.

The key method is `send_on_interface()` — it's the single helper used for directed sends. All broadcast sends construct `SendOnInterface` inline. Search for all construction sites and categorize:

- **Directed** (weight: None): `send_on_interface()`, `relay_packet()`, `forward_link_request()`, `route_proof()`, `route_link_data()`
- **Broadcast** (weight: Some): announce rebroadcast on all interfaces, unknown-destination flood

Use `get_broadcast_weights()` for the broadcast case to ensure the highest is forced to 1.0.

- [ ] **Step 4: Update all match arms that destructure SendOnInterface**

Any test or code that pattern-matches `SendOnInterface { interface_name, raw }` needs updating to `SendOnInterface { interface_name, raw, .. }` (ignoring weight) or `SendOnInterface { interface_name, raw, weight }` (using weight).

Most test assertions can use `..` to ignore the new field.

- [ ] **Step 5: Register interfaces with CooperationTable**

In `register_interface()`, also call `self.cooperation.register_interface(name.clone())`.

- [ ] **Step 6: Run tests — verify all existing tests pass**

Run: `cargo test -p harmony-reticulum -v`
Expected: all existing tests pass (weight field is additive, defaults to None for directed sends)

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-reticulum/src/node.rs
git commit -m "feat(reticulum): wire CooperationTable into Node, weighted SendOnInterface

Node owns CooperationTable. SendOnInterface gains weight: Option<f32>.
Directed sends: None (always deliver). Broadcast sends: Some(score).
Highest-scored interface forced to 1.0 for guaranteed delivery.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Announce echo observation

**Files:**
- Modify: `crates/harmony-reticulum/src/node.rs`

Feed positive/negative announce forwarding observations into the CooperationTable.

- [ ] **Step 1: Add pending_echoes tracking**

Add to the `Node` struct:

```rust
    /// Tracks announce echoes we're waiting for: (interface, dest_hash) → tick_sent.
    pending_echoes: HashMap<(Arc<str>, DestinationHash), u64>,
```

Initialize to `HashMap::new()` in constructors.

- [ ] **Step 2: Record pending echoes when we send announces**

In the announce generation/send path: when the Node emits `SendOnInterface` actions for our own announces (in `announce()` or the announce scheduling path), record each `(interface_name, dest_hash)` in `pending_echoes` with the current tick timestamp.

- [ ] **Step 3: Detect announce echoes on receive**

In `handle_event()` / announce processing: when an announce arrives, check if its destination hash matches any of our `announcing_destinations`. If so:
- The interface it arrived on gets a positive observation: `cooperation.observe_announce_forwarded(&interface_name, now)`
- Remove the matching `(arriving_interface, dest_hash)` from `pending_echoes`
- Skip the arriving interface's own echo (filter: only credit *other* interfaces)

- [ ] **Step 4: Timeout negative observations on timer tick**

In the timer tick handler, after path expiry:
- Define timeout: `3 * announce_interval_ticks * 250` ms (configurable via the Node, but default ~22.5 seconds)
- Iterate `pending_echoes`, collect entries older than timeout
- For each expired entry: `cooperation.observe_announce_timeout(&interface_name, now)`
- Remove expired entries
- Also call `cooperation.decay_stale(now)` for staleness decay

- [ ] **Step 5: Add tests**

- `test_announce_echo_positive_observation` — register two interfaces, emit announce, simulate echo on second interface, verify score increased
- `test_announce_timeout_negative_observation` — register interface, emit announce, tick past timeout without echo, verify score decreased

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-reticulum -v`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-reticulum/src/node.rs
git commit -m "feat(reticulum): announce echo observation for cooperation scoring

Track pending announce echoes. Positive observation when our announce
comes back via another interface. Negative observation on timeout
(3 announce cycles). Staleness decay on timer tick.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Proof delivery observation

**Files:**
- Modify: `crates/harmony-reticulum/src/node.rs`

Feed proof delivery observations into the CooperationTable.

- [ ] **Step 1: Add `proof_received: bool` to ReverseTableEntry**

Change the `ReverseTableEntry` struct:

```rust
struct ReverseTableEntry {
    received_interface: Arc<str>,
    outbound_interface: Arc<str>,
    timestamp: u64,
    proof_received: bool,  // NEW: set to true when proof arrives
}
```

Initialize `proof_received: false` everywhere `ReverseTableEntry` is constructed.

- [ ] **Step 2: Mark proof received and observe**

In the proof routing path (`route_proof()` or wherever proofs are matched to reverse table entries): when a proof matches a reverse table entry, set `entry.proof_received = true` and call `cooperation.observe_proof_delivered(&entry.outbound_interface, now)`.

- [ ] **Step 3: Observe proof timeout on expiry**

In the reverse table expiry path (timer tick, where `reverse_table.retain()` is called): before removing expired entries, check `proof_received`. If `false`, call `cooperation.observe_proof_timeout(&entry.outbound_interface, now)`.

Note: `retain()` closure can't call methods on `self` (borrow conflict). Extract the entries to expire first, then process observations, then remove.

- [ ] **Step 4: Add tests**

- `test_proof_delivery_positive_observation` — send packet (creates reverse entry), simulate proof arrival, verify score increased
- `test_proof_timeout_negative_observation` — send packet, tick past REVERSE_TIMEOUT without proof, verify score decreased

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-reticulum -v`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-reticulum/src/node.rs
git commit -m "feat(reticulum): proof delivery observation for cooperation scoring

Track proof_received on reverse table entries. Positive observation
when proof arrives for a relayed packet. Negative observation when
reverse table entry expires without proof.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Event loop probabilistic send

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/runtime.rs`

Wire the weight field into the event loop's send path.

- [ ] **Step 1: Update RuntimeAction::SendOnInterface**

In `crates/harmony-node/src/runtime.rs`, the `RuntimeAction::SendOnInterface` mirrors the reticulum `NodeAction`. Add `weight: Option<f32>` to match.

Update the dispatch code that converts `NodeAction::SendOnInterface` to `RuntimeAction::SendOnInterface` to pass through the weight.

- [ ] **Step 2: Update dispatch_action in event_loop.rs**

In the `SendOnInterface` arm of `dispatch_action()`, add probabilistic send:

```rust
RuntimeAction::SendOnInterface { ref interface_name, ref raw, weight } => {
    let should_send = match weight {
        None => true,
        Some(w) if w >= 1.0 => true,
        Some(w) => rand::random::<f32>() < w,
    };
    if should_send {
        // existing send logic (broadcast + unicast fan-out)
    }
}
```

- [ ] **Step 3: Run all harmony-node tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs crates/harmony-node/src/runtime.rs
git commit -m "feat(harmony-node): probabilistic send from cooperation weights

Event loop respects weight field on SendOnInterface. None = always
send. Some(w) = send with probability w. Broadcast traffic weighted
by neighbor cooperation scores.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Full verification

**Files:** None (verification only)

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-reticulum -p harmony-node`
Expected: no new warnings from our changes

- [ ] **Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: our files are formatted correctly

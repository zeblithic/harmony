# ZEB-216 Phase 3a: Runtime Unicast IPC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `RuntimeEvent::SendUnicastToDevice` and `RuntimeAction::UnicastReceived` to harmony-runtime, wired through to existing Reticulum unicast plumbing. This is the harmony-side companion PR for harmony-client's ZEB-216 Sub-B (DM transport, Phases 2/3b consume these).

**Architecture:** Phase 3a is pure plumbing on existing surfaces. No new types — `[u8; 16]` (matching `IdentityHash` at `harmony-identity/src/identity.rs:37`) is the source identifier. `destination_hash` on `SendUnicastToDevice` is a Reticulum *destination* hash (not a raw device identity hash) — the client computes it from a device identity hash + a destination name. The inbound side activates a deliberately-stubbed handler at `runtime.rs:2350-2351` (the catch-all `_ => {}` after `SendOnInterface`/`PacketDropped` in `dispatch_router_actions`, where `NodeAction::DeliverLocally` currently falls through and is silently dropped). The outbound side adds a translation from the new `RuntimeEvent::SendUnicastToDevice` to the existing `RuntimeAction::SendOnInterface`, looking up the destination's interface via the inner Reticulum router's path table.

**Tech Stack:** Rust 1.94, tokio (async runtime is `tracing` for logs), postcard (workspace serialization choice — but note that `RuntimeEvent`/`RuntimeAction` are NOT serde-derived; they're internal Rust enums consumed by the embedding layer directly).

**Linear:** ZEB-226 (this ticket), parent ZEB-216, sibling ZEB-225 (Phase 2, parallelizable).

---

## File Structure

Primary work is in `crates/harmony-runtime/`, with a small companion change in `crates/harmony-reticulum/` for the `DeliverLocally.source` field (chosen during Task 3 investigation per Option C in the open questions below):

- **Modify:** `crates/harmony-runtime/src/runtime.rs`
  - Around line 195+ — add `RuntimeEvent::SendUnicastToDevice` variant (Tier 1)
  - Around line 341+ — add `RuntimeAction::UnicastReceived` variant (Tier 1)
  - Around line 2350-2351 (`dispatch_router_actions`) — replace silent drop of `NodeAction::DeliverLocally` with `UnicastReceived` emission
  - In the runtime tick loop (find by reading `pub fn tick` at line 2005 + grep for where `RuntimeEvent` variants get dispatched) — add a `SendUnicastToDevice` event handler that translates to `SendOnInterface`
  - Inline tests in the same file's `#[cfg(test)]` block (matches existing convention)

- **Modify (likely):** `crates/harmony-reticulum/src/node.rs` — add a `source: Option<[u8; 16]>` field to `NodeAction::DeliverLocally` and a test-only `path_table_mut_for_tests()` accessor on `Node`. (See Task 3 + Task 4 investigations.)

- **Modify (if needed):** `crates/harmony-runtime/src/lib.rs` or other re-export paths if the new variants need to surface through a `pub use`

No new files.

---

## Open implementation questions

These were flagged for the implementer to resolve by reading the actual code. Resolutions captured below.

1. **`destination_hash` → interface routing.** When the runtime processes `SendUnicastToDevice { destination_hash }`, how does it know which interface the destination is reachable on? Look at `PeerManager` (referenced at line 259 in runtime.rs) or the router's path table. The existing `InitiateTunnel` action shape (line 365+) carries `relay_url` per peer — likely there's a similar lookup for Reticulum interfaces. If the destination is unknown, the right behavior for Phase 3a is to log + drop (Phase 3b in harmony-client adds retry/expiration semantics on top). **Resolved during execution:** path-table lookup via `Node::path_table()`. Round-10 review tightened the drain semantics (defer-on-miss when router queue still has backlog; bound by `router_max_per_tick`).

2. **Source identity resolution on receive.** `NodeAction::DeliverLocally { destination_hash, packet, interface_name }` carries the local destination, not the remote source. The source identity needs to come from somewhere:
   - **Option A:** The packet payload (Reticulum link packets carry origin info in their header)
   - **Option B:** Link state — the router knows which link delivered the packet, and the link knows its remote identity
   - **Option C:** A new field added to `NodeAction::DeliverLocally` to carry source

   Read `harmony-reticulum/src/node.rs` around line 236 (where `DeliverLocally` is emitted) to determine the source's actual provenance. **Resolved during execution: Option C** — added `source: Option<[u8; 16]>` to `NodeAction::DeliverLocally`. The Option type was tightened in round-9 review to make the "unknown source" Phase 3a behavior unforgeable (sentinel hashes can be misinterpreted as real identities). The construction site emits `None` until link/identity binding lands in ZEB-227 / Phase 3b.

---

## Tasks

### Task 1: Add `RuntimeEvent::SendUnicastToDevice` variant

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs:195` (the `pub enum RuntimeEvent` block)
- Test: same file, `#[cfg(test)]` block

- [ ] **Step 1: Write the failing test**

Add to the test module in `runtime.rs`:

```rust
#[test]
fn runtime_event_send_unicast_to_device_constructs_and_compares() {
    use super::RuntimeEvent;
    let e = RuntimeEvent::SendUnicastToDevice {
        destination_hash: [0xaa; 16],
        packet: vec![0x01, 0x02, 0x03],
    };
    // Match-and-extract — proves variant is constructible and the fields
    // are accessible by name (catches accidental tuple-variant or wrong
    // field-name typos).
    match e {
        RuntimeEvent::SendUnicastToDevice { destination_hash, packet } => {
            assert_eq!(destination_hash, [0xaa; 16]);
            assert_eq!(packet, vec![0x01, 0x02, 0x03]);
        }
        _ => panic!("expected SendUnicastToDevice variant"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib runtime_event_send_unicast_to_device_constructs_and_compares`
Expected: FAIL with "no variant or associated item named `SendUnicastToDevice` found for enum `RuntimeEvent`"

- [ ] **Step 3: Add the enum variant**

In `crates/harmony-runtime/src/runtime.rs`, find the `pub enum RuntimeEvent` block at line 195. Add a new variant in the Tier 1 group (after `InboundPacket` and `TimerTick`, before the Tier 2 `QueryReceived`):

```rust
    /// Tier 1: Client requests a Reticulum unicast send to a specific
    /// destination. Translated to `RuntimeAction::SendOnInterface` during
    /// the next tick by looking up `destination_hash` in the inner
    /// router's path table. If the lookup misses, the request is logged
    /// and dropped — retry/expiration semantics are the client's concern
    /// (handled at the outbox layer in harmony-client per ZEB-216 Sub-B
    /// Phase 3b).
    ///
    /// **`destination_hash` is a Reticulum destination hash, NOT a raw
    /// device identity hash.** The path table is keyed by destination
    /// hashes which the runtime learns from announces. The harmony-client
    /// side (Phase 3b, ZEB-227) is responsible for computing this from a
    /// device identity hash + a destination name (e.g. DM inbox, voice
    /// channel, file sync — the runtime is generic plumbing and does not
    /// know what destination names exist).
    ///
    /// (ZEB-216 Sub-B Phase 3a — DM transport surface)
    SendUnicastToDevice {
        /// 16-byte Reticulum destination hash. Caller-provided; see the
        /// variant doc above for the derivation contract.
        destination_hash: [u8; 16],
        /// Opaque packet bytes — the runtime does not parse or validate
        /// the payload. The client owns encryption + framing.
        packet: Vec<u8>,
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib runtime_event_send_unicast_to_device_constructs_and_compares`
Expected: PASS

- [ ] **Step 5: Run gates**

```bash
cargo fmt --all -- --check && \
  cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings && \
  cargo check --manifest-path crates/harmony-runtime/Cargo.toml
```

Expected: all three exit 0.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-runtime/src/runtime.rs
git commit -m "feat(zeb-226): add RuntimeEvent::SendUnicastToDevice variant

Phase 3a, step 1 of 5. Adds the inbound side of the unicast IPC
contract that harmony-client's DM outbox (ZEB-216 Phase 3b) will
consume. Variant is wired in subsequent steps; this commit is just
the type addition + constructor test."
```

---

### Task 2: Add `RuntimeAction::UnicastReceived` variant

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs:341` (the `pub enum RuntimeAction` block)
- Test: same file, `#[cfg(test)]` block

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn runtime_action_unicast_received_constructs_and_compares() {
    use super::RuntimeAction;
    let a = RuntimeAction::UnicastReceived {
        destination_hash: [0xcc; 16], // local dest the packet was addressed to (round 10)
        source: Some([0xbb; 16]),     // Phase 3a uses None; round-9 tightened to Option
        packet: vec![0x10, 0x20],
    };
    let b = RuntimeAction::UnicastReceived {
        destination_hash: [0xcc; 16],
        source: Some([0xbb; 16]),
        packet: vec![0x10, 0x20],
    };
    // Variant must derive PartialEq + Clone (matches existing actions
    // — this is a regression guard so the new variant doesn't drop the
    // derives and break test infrastructure that depends on them).
    assert_eq!(a, b);
    let _cloned: RuntimeAction = a.clone();
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib runtime_action_unicast_received_constructs_and_compares`
Expected: FAIL — `UnicastReceived` not found.

- [ ] **Step 3: Add the enum variant**

In `runtime.rs`, find `pub enum RuntimeAction` at line 341. Add in the Tier 1 group (alongside `SendOnInterface`):

```rust
    /// Tier 1: Surface a received Reticulum unicast packet to the
    /// client. Emitted from `dispatch_router_actions` when the inner
    /// router produces `NodeAction::DeliverLocally` for a packet
    /// addressed to a locally-registered destination.
    ///
    /// `destination_hash` is the local Reticulum destination hash the
    /// packet was addressed to. (Added in round 10.) The client
    /// registered the destination, so it knows the mapping from hash
    /// to handler — needed because the client registers MULTIPLE
    /// destination types (DM inbox, voice channel, file sync, ...)
    /// and dispatches each inbound packet by destination kind.
    ///
    /// `source` is the remote device's identity hash, **when known**.
    /// `Some(hash)` once link/identity binding lands in ZEB-227 /
    /// Phase 3b; `None` in Phase 3a (the Node layer doesn't track
    /// terminal-link state yet, so the runtime can't surface a real
    /// source). Round 9 tightened this from `[u8; 16]` to
    /// `Option<[u8; 16]>` to make the "unknown source" case
    /// unforgeable — sentinel hashes can be misinterpreted as real
    /// identities. Consumers MUST handle the `None` case explicitly.
    ///
    /// `packet` is the raw payload — the client owns decryption +
    /// framing parsing.
    ///
    /// (ZEB-216 Sub-B Phase 3a — DM transport surface)
    UnicastReceived {
        /// 16-byte local Reticulum destination hash the packet was
        /// addressed to. (Round 10.)
        destination_hash: [u8; 16],
        /// 16-byte device identity hash of the remote sender, when
        /// known. `None` in Phase 3a; `Some(hash)` after ZEB-227.
        /// (Round 9 tightened from `[u8; 16]`.)
        source: Option<[u8; 16]>,
        /// Opaque packet bytes as received from the wire.
        packet: Vec<u8>,
    },
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib runtime_action_unicast_received_constructs_and_compares`
Expected: PASS

- [ ] **Step 5: Run gates**

```bash
cargo fmt --all -- --check && \
  cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings && \
  cargo check --manifest-path crates/harmony-runtime/Cargo.toml
```

Expected: all three exit 0.

- [ ] **Step 6: Commit**

```bash
git commit -am "feat(zeb-226): add RuntimeAction::UnicastReceived variant

Phase 3a, step 2 of 5. Adds the outbound side of the unicast IPC
contract — the runtime emits this from the next tick() when a
unicast packet arrives at a locally-registered destination. Variant
is wired in step 3 (currently DeliverLocally is silently dropped)."
```

---

### Task 3: Wire `NodeAction::DeliverLocally` → `RuntimeAction::UnicastReceived`

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs` around line 2350-2351 (`dispatch_router_actions`, the silent-drop at the catch-all)
- Test: inline test exercising the path with a constructed `NodeAction::DeliverLocally`

**Investigation step (BEFORE writing the test):** Read these spots and decide on the source-resolution approach:
1. `crates/harmony-reticulum/src/node.rs:236` — where `NodeAction::DeliverLocally` is emitted. Does the surrounding code have access to the source identity (from link state, packet header, or router state)? Capture it into the variant if needed.
2. If `DeliverLocally` doesn't carry source, you have two options:
   a. Add a `source: Option<[u8; 16]>` field to `NodeAction::DeliverLocally` (touches the Reticulum crate, but cleanest). The Option is required because Phase 3a can't resolve a real identity yet — round-9 review tightened from `[u8; 16]` to make the unknown case unforgeable (sentinels can be misinterpreted as real identities).
   b. Look up source from `self.peer_manager` or the router's link table at the dispatch site (no Reticulum-crate change)
3. Pick whichever is cleanest given the actual code shape. Document the choice in the commit message. **Resolved during execution: Option (a) — see Open Questions.**

If option (a) requires modifying `harmony-reticulum`, that's acceptable scope — both crates change in the same commit.

- [ ] **Step 1: Investigate source resolution (read-only, no test yet)**

Read `harmony-reticulum/src/node.rs:236` ± 30 lines and trace where `DeliverLocally` is constructed. Determine which of (a)/(b) above applies. Write a one-line note in the implementation comment explaining the chosen approach.

- [ ] **Step 2: Write the failing integration test**

```rust
#[test]
fn dispatch_router_actions_surfaces_deliver_locally_as_unicast_received() {
    // Construct a NodeRuntime with default config (or use the existing
    // test-harness builder — search for `fn make_runtime` or `fn test_runtime`
    // in this file and reuse).
    //
    // Construct a NodeAction::DeliverLocally with:
    //   destination_hash: <our local identity>
    //   packet: vec![0xde, 0xad, 0xbe, 0xef]
    //   interface_name: "test"
    //   source: <remote peer hash>  (if option (a) above)
    //
    // Feed it through dispatch_router_actions (the function that currently
    // drops it — find the right entry point; may be private, in which case
    // call via a higher-level path that produces it).
    //
    // Capture the resulting Vec<RuntimeAction> and assert exactly one
    // RuntimeAction::UnicastReceived with source=peer_hash, packet=<bytes>.
    //
    // The exact construction details depend on the resolution chosen in
    // Step 1 — write the test against the chosen path.
    todo!("see investigation comment for the right harness shape")
}
```

(The `todo!()` above is intentional — the actual test body depends on which test-harness pattern the existing tests use. Read existing tests around the `mod tests` block in runtime.rs first.)

- [ ] **Step 3: Run test to verify it fails**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib dispatch_router_actions_surfaces_deliver_locally_as_unicast_received`
Expected: FAIL — either `todo!()` panic, or assertion failure once the test body is fleshed out, or compile error if option (a) requires a field that doesn't exist yet.

- [ ] **Step 4: Implement the wiring**

In `runtime.rs` around line 2340-2352, replace the silent drop:

```rust
                            NodeAction::PacketDropped { reason, .. } => {
                                tracing::warn!(?reason, "Reticulum announce dropped");
                            }
                            // BEFORE:
                            // _ => {}
                            // AFTER (replace catch-all with explicit DeliverLocally arm):
                            NodeAction::DeliverLocally {
                                destination_hash: _,  // local; not surfaced
                                packet,
                                interface_name: _,    // diagnostic only at this layer
                                // source: ...  (if option (a))
                            } => {
                                // Resolve source identity per investigation in Step 1.
                                let source = /* link-state lookup OR field */;
                                out.push(RuntimeAction::UnicastReceived { source, packet });
                            }
                            _ => {} // genuinely diagnostic — keep dropping
```

The catch-all `_ => {}` MUST stay as a fallthrough for any other `NodeAction` variants the inner router produces — only `DeliverLocally` graduates from drop to surface.

If source resolution needs `&mut self.peer_manager` (or similar), make sure `dispatch_router_actions` has the right access; refactor the function signature minimally if needed.

- [ ] **Step 5: Flesh out the test, run it, verify it passes**

Replace the `todo!()` with the real test body using whatever helpers existing tests use. Run:
`cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib dispatch_router_actions_surfaces_deliver_locally_as_unicast_received`
Expected: PASS

- [ ] **Step 6: Run gates**

```bash
cargo fmt --all -- --check && \
  cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings && \
  cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib
```

Expected: all three exit 0. Full lib test suite passes.

If you needed to modify `harmony-reticulum` (option a), also run:

```bash
cargo clippy --manifest-path crates/harmony-reticulum/Cargo.toml --all-targets -- -D warnings && \
  cargo test --manifest-path crates/harmony-reticulum/Cargo.toml --lib
```

- [ ] **Step 7: Commit**

```bash
git commit -am "feat(zeb-226): surface DeliverLocally as UnicastReceived to client

Phase 3a, step 3 of 5. Replaces the silent drop at runtime.rs:2350
(\"Other router actions are diagnostics — drop for now\") with an
explicit DeliverLocally handler that emits RuntimeAction::UnicastReceived
in the next tick(). Source identity is resolved from <chosen approach
per investigation>.

This activates the inbound half of harmony-client's ZEB-216 Sub-B
DM transport. The drop comment was a deliberate placeholder waiting
for the client-facing consumer — Phase 3a is that consumer."
```

---

### Task 4: Wire `RuntimeEvent::SendUnicastToDevice` → `RuntimeAction::SendOnInterface`

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs` — find where `RuntimeEvent` variants are handled in the tick loop. Likely `pub fn tick` at line 2005 dispatches via a match against pending events. If events are queued elsewhere and processed during tick, find the dispatch site and add the `SendUnicastToDevice` arm.
- Test: inline test feeding a `SendUnicastToDevice` event and observing the resulting `SendOnInterface` action

**Investigation step:** Read these to figure out destination_hash → interface routing:
1. `crates/harmony-runtime/src/runtime.rs` around `PeerManager` references (line 259+) — does the peer manager track interface-per-peer?
2. Look at how existing `SendOnInterface` actions are emitted in the codebase (grep for `SendOnInterface {`) — what's `interface_name` typically derived from?
3. If the peer manager doesn't track interface-per-destination, look at the Reticulum router's path table or the link table for the same info.
4. Decide: lookup function exists already → call it. Doesn't exist → add a small helper.

- [ ] **Step 1: Investigate destination_hash → interface routing (read-only)**

Determine the lookup. Document the chosen approach in an implementation comment.

- [ ] **Step 2: Write the failing integration test**

```rust
#[test]
fn send_unicast_to_device_emits_send_on_interface_for_known_target() {
    // Construct a NodeRuntime with one known destination at a known interface.
    // (Reuse existing test-harness if there's one; if not, build a minimal
    // NodeConfig + InMemoryStore setup like other tests in this file.)
    //
    // Pre-populate the routing source (chosen during investigation) so
    // destination_hash=[0xaa; 16] resolves to interface_name="test-iface".
    //
    // Feed RuntimeEvent::SendUnicastToDevice {
    //     destination_hash: [0xaa; 16],
    //     packet: vec![0x01, 0x02, 0x03],
    // }
    //
    // Run runtime.tick(), capture Vec<RuntimeAction>.
    //
    // Assert exactly one RuntimeAction::SendOnInterface with:
    //   interface_name = "test-iface"
    //   raw = (whatever the unicast wrapping produces — may be the packet
    //          verbatim, may be wrapped in a Reticulum frame)
    //   weight = None  (directed send, not broadcast)
    todo!("see investigation comment for the right harness shape")
}

#[test]
fn send_unicast_to_device_drops_for_unknown_target() {
    // Same setup but with a destination_hash NOT in the routing source.
    // Expect the action vec to NOT contain any SendOnInterface
    // for the unknown destination. A WARN-level tracing log is acceptable.
    // Phase 3b in harmony-client owns retry/expiration semantics.
    todo!()
}
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib send_unicast_to_device`
Expected: both FAIL — the dispatch arm doesn't exist yet.

- [ ] **Step 4: Implement the dispatch arm**

In the runtime tick / event dispatch path, add:

```rust
    RuntimeEvent::SendUnicastToDevice { destination_hash, packet } => {
        // Resolve destination_hash's reachable interface via <chosen approach>.
        match self.<lookup>(&destination_hash) {
            Some(interface_name) => {
                // Wrap into Reticulum frame if needed (depends on existing
                // SendOnInterface conventions — match what other unicast
                // sends do at this layer).
                actions.push(RuntimeAction::SendOnInterface {
                    interface_name: Arc::from(interface_name.as_str()),
                    raw: packet,
                    weight: None, // directed send, not broadcast
                });
            }
            None => {
                tracing::warn!(
                    destination_hash_first_4 = ?&destination_hash[..4],
                    packet_len = packet.len(),
                    "SendUnicastToDevice destination_hash unknown — dropping. \
                     Client (harmony-client) owns retry/expiration via outbox."
                );
            }
        }
    }
```

(Adjust the exact match shape based on the existing dispatch pattern. Round-9 / round-10 review iterated on the drain semantics — the landed code routes via `pending_unicast_sends` queued in `push_event` and drained in `tick()` after the router queue, with `router_max_per_tick` budget + defer-on-miss when the router still has backlog.)

- [ ] **Step 5: Flesh out tests, run them, verify they pass**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib send_unicast_to_device`
Expected: both PASS.

- [ ] **Step 6: Run gates**

```bash
cargo fmt --all -- --check && \
  cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings && \
  cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib
```

Expected: all three exit 0.

- [ ] **Step 7: Commit**

```bash
git commit -am "feat(zeb-226): translate SendUnicastToDevice to SendOnInterface

Phase 3a, step 4 of 5. Wires the outbound half of the unicast IPC
contract: RuntimeEvent::SendUnicastToDevice (client request) is
processed during tick() and translated to RuntimeAction::SendOnInterface
with the destination_hash's resolved interface (via <chosen lookup>).

Unknown destinations are logged at WARN and dropped — retry/expiration
semantics live in harmony-client's outbox (ZEB-216 Sub-B Phase 3b)."
```

---

### Task 5: End-to-end round-trip integration test

**Files:**
- Modify: `crates/harmony-runtime/src/runtime.rs` — add one comprehensive test in the same `#[cfg(test)]` block
- (No production code changes — this task is test-only)

The single most valuable confidence test: a unicast packet emitted by node A's runtime appears as a `UnicastReceived` action in node B's runtime. If the harmony-runtime test suite already has a two-node helper (search `fn two_node` or similar), use it. Otherwise, simulate by manually feeding the SendOnInterface output of node A as an InboundPacket to node B.

- [ ] **Step 1: Look for an existing two-node test harness**

```bash
grep -n "two_node\|two_runtime\|peer_pair" crates/harmony-runtime/src/runtime.rs
grep -rn "two_node\|two_runtime" crates/harmony-node/tests/
```

If a helper exists: use it. If not: build a minimal one inline in the test (two `NodeRuntime` instances, manual A → B packet routing).

- [ ] **Step 2: Write the round-trip test**

```rust
#[test]
fn unicast_round_trip_a_to_b_surfaces_as_unicast_received() {
    // Setup: two NodeRuntime instances representing devices A and B.
    // Pick a destination hash (Reticulum DM destination), register it
    // as a local destination on B, seed A's path table to route it to
    // a shared interface name.
    let (mut a, mut b) = make_two_node_pair();  // or inline construction
    let dest = [0xb0u8; 16];                    // shared Reticulum dest hash
    b.router.register_destination(dest);
    let payload = b"hello unicast".to_vec();

    // A: queue the send via the new event.
    a.feed_event(RuntimeEvent::SendUnicastToDevice {
        destination_hash: dest,
        packet: payload.clone(),
    });

    // A: tick and grab the SendOnInterface action.
    let a_actions = a.tick();
    let send_on_iface = a_actions
        .iter()
        .find_map(|action| match action {
            RuntimeAction::SendOnInterface { interface_name, raw, weight } => {
                Some((interface_name.clone(), raw.clone(), *weight))
            }
            _ => None,
        })
        .expect("A should have emitted SendOnInterface for dest");

    // B: feed the wire bytes as InboundPacket.
    b.feed_event(RuntimeEvent::InboundPacket {
        interface_name: send_on_iface.0.to_string(),
        raw: send_on_iface.1.clone(),
        now: 1000,
    });

    // B: tick and look for UnicastReceived.
    let b_actions = b.tick();
    let received = b_actions
        .iter()
        .find_map(|action| match action {
            RuntimeAction::UnicastReceived {
                destination_hash,
                source,
                packet,
            } => Some((*destination_hash, *source, packet.clone())),
            _ => None,
        })
        .expect("B should have surfaced UnicastReceived");

    // destination_hash flows through (round 10).
    assert_eq!(received.0, dest, "destination_hash must match B's local dest");
    // source is None in Phase 3a (Option<[u8; 16]>; round 9 tightening) —
    // becomes Some(a.local_identity_hash()) after ZEB-227 lands.
    assert_eq!(received.1, None, "source is None until ZEB-227");
    assert_eq!(received.2, payload, "packet must round-trip unchanged");
}
```

(Adjust API names — `feed_event`, `local_identity_hash` — to whatever the actual harmony-runtime methods are. `local_identity_hash` exists per `pub fn` listing at line 1502.)

- [ ] **Step 3: Run the test**

Run: `cargo test --manifest-path crates/harmony-runtime/Cargo.toml --lib unicast_round_trip_a_to_b_surfaces_as_unicast_received`
Expected: PASS

When this fails, the most likely root causes are:
- The minimal two-node setup doesn't share a real interface (need to wire the `interface_name` through correctly)
- The Reticulum link layer requires handshake before unicast (need to either pre-establish the link in the test setup, or use a simpler non-link path if available)

Should the second cause apply ("real Reticulum link handshake required for round-trip"), that's acceptable scope-creep for a focused unit test only IF the existing test infrastructure makes link-establishment easy. Otherwise, defer the full round-trip test to harmony-client's Phase 3b integration tests and leave a comment in this test explaining the deferral.

- [ ] **Step 4: Run gates**

```bash
cargo fmt --all -- --check && \
  cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings && \
  cargo test --manifest-path crates/harmony-runtime/Cargo.toml
```

Note: this is the FINAL gate run, so include the full test suite (no `--lib` filter — we want integration tests too).

- [ ] **Step 5: Commit**

```bash
git commit -am "test(zeb-226): end-to-end unicast round-trip A → B

Phase 3a, step 5 of 5. Validates the full plumbing: A's
SendUnicastToDevice event produces a SendOnInterface action whose
bytes, when fed as B's InboundPacket, surface as a UnicastReceived
action with source=A's identity and packet bytes intact."
```

---

### Task 6 (process): Push branch + open PR

- [ ] **Step 1: Push the branch**

```bash
git push -u origin zeb-226-runtime-unicast-ipc
```

- [ ] **Step 2: Open the PR**

```bash
gh pr create --repo zeblithic/harmony \
  --title "feat(runtime): SendUnicastToDevice + UnicastReceived IPC kinds (ZEB-226)" \
  --body "$(cat <<'EOF'
## Summary

Adds the runtime IPC surface for ZEB-216 Sub-B Phase 3a (DM transport companion):

- New `RuntimeEvent::SendUnicastToDevice { destination_hash, packet }` — client requests a Reticulum unicast send to a specific destination. Translated to `RuntimeAction::SendOnInterface` during the next tick via path-table interface lookup.
- New `RuntimeAction::UnicastReceived { destination_hash, source, packet }` — runtime surfaces a received Reticulum unicast packet to the client. Activates the previously-stubbed `// Other router actions are diagnostics — drop for now` handler at runtime.rs:2350-2351. `source` is `Option<[u8; 16]>` (round-9 tightening — `None` until ZEB-227 wires link-identity binding); `destination_hash` is the local destination the packet was addressed to (round-10, lets the client dispatch by destination kind).
- End-to-end round-trip integration test (A's send → B's receive).

Pure plumbing — no new types, no new Reticulum protocol features. Uses `[u8; 16]` for both fields (matches `harmony_identity::IdentityHash` for the source).

## Linear

- ZEB-226 (this PR)
- Parent: ZEB-216 (DM transport)
- Sibling: ZEB-225 (Phase 2, harmony-client outbox + IPC, can land in parallel)
- Unblocks: ZEB-227 (Phase 3b, harmony-client real Reticulum delivery)

## Test plan

- [x] `cargo fmt --all -- --check` clean
- [x] `cargo clippy --manifest-path crates/harmony-runtime/Cargo.toml --all-targets -- -D warnings` clean
- [x] `cargo test --manifest-path crates/harmony-runtime/Cargo.toml` — all green
- [ ] CI passes
EOF
)"
```

---

## Self-review checklist (run mentally before dispatching)

1. **Spec coverage:** ZEB-216 Sub-B Phase 3a deliverables are (a) new IPC command for unicast send (Task 1+4), (b) new IPC event for unicast receive (Task 2+3), (c) tests covering both round-tripping (Task 5). ✓ All three covered.

2. **Placeholder scan:** Every step contains the actual code or actual command. The `todo!()` in Task 3 Step 2 and Task 4 Step 2 IS a placeholder — but it's an intentional one tied to a real investigation step (the test body depends on which lookup approach is chosen). Each `todo!()` has a "see investigation comment" pointer.

3. **Type consistency:** `RuntimeEvent::SendUnicastToDevice` uses `destination_hash: [u8; 16]` (Reticulum dest hash). `RuntimeAction::UnicastReceived` uses `destination_hash: [u8; 16]` (local dest, round-10) + `source: Option<[u8; 16]>` (remote sender; round-9 tightened from `[u8; 16]` so the unknown case is unforgeable). ✓

4. **Open questions are flagged, not hidden:** Two real investigation steps (destination_hash→interface routing in Task 4, source resolution in Task 3) are surfaced as explicit "Investigation step" sub-steps before the failing test. The implementer is told what to read and how to choose. ✓

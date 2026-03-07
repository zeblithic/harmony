# Ring 1 Unikernel Event Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire sans-I/O state machines into the unikernel event loop so two QEMU nodes discover each other via Reticulum announces and exchange periodic heartbeat messages.

**Architecture:** Extend `UnikernelRuntime` to absorb the two-phase announce pattern, manage a peer table, and schedule heartbeat data packets. Add a PIT timer for wall-clock milliseconds. Simplify `main.rs` to dispatch `RuntimeAction`s instead of raw `NodeAction`s.

**Tech Stack:** Rust (`no_std`), x86 PIT 8254 timer, harmony-reticulum (Node, Packet), harmony-platform (NetworkInterface, EntropySource), harmony-unikernel (UnikernelRuntime)

**Design doc:** `docs/plans/2026-03-07-event-loop-design.md`

---

### Task 1: RuntimeAction enum and PeerInfo struct

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs:1-10` (add imports and new types)
- Modify: `crates/harmony-unikernel/src/lib.rs:9-13` (add re-exports)
- Modify: `crates/harmony-unikernel/Cargo.toml` (no changes expected — alloc already available)

**Step 1: Define RuntimeAction and PeerInfo**

Add to the top of `event_loop.rs`, after existing imports:

```rust
use alloc::collections::BTreeMap;
use alloc::sync::Arc;

use harmony_reticulum::destination::DestinationName;
use harmony_reticulum::path_table::DestinationHash;

/// Runtime-level output actions. The caller dispatches these — no
/// protocol-internal actions like AnnounceNeeded leak through.
#[derive(Debug)]
pub enum RuntimeAction {
    /// Send raw bytes on a named interface.
    SendOnInterface { interface_name: Arc<str>, raw: Vec<u8> },
    /// A new peer was discovered via announce.
    PeerDiscovered { address_hash: [u8; 16], hops: u8 },
    /// A previously known peer has gone silent.
    PeerLost { address_hash: [u8; 16] },
    /// A heartbeat was received from a peer.
    HeartbeatReceived { address_hash: [u8; 16], uptime_ms: u64 },
    /// A non-heartbeat packet was delivered locally.
    DeliverLocally { destination_hash: [u8; 16], payload: Vec<u8> },
}

/// Tracked state for a discovered peer.
#[derive(Debug, Clone)]
pub struct PeerInfo {
    pub address_hash: [u8; 16],
    pub dest_hash: [u8; 16],
    pub last_seen_ms: u64,
    pub hops: u8,
    pub discovered_at_ms: u64,
}
```

**Step 2: Add re-exports to lib.rs**

Change `crates/harmony-unikernel/src/lib.rs` to:

```rust
pub use event_loop::{PeerInfo, RuntimeAction, UnikernelRuntime};
pub use harmony_reticulum::NodeAction;
pub use platform::entropy::KernelEntropy;
pub use platform::persistence::MemoryState;
pub use serial::SerialWriter;
```

**Step 3: Run tests to verify it compiles**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All 5 existing tests pass. No new tests yet — just verifying types compile.

**Step 4: Commit**

```bash
cd /Users/zeblith/work/zeblithic/harmony-os
git add crates/harmony-unikernel/src/event_loop.rs crates/harmony-unikernel/src/lib.rs
git commit -m "feat(unikernel): add RuntimeAction enum and PeerInfo struct"
```

---

### Task 2: Extend UnikernelRuntime with new fields

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs:11-29` (add fields to struct and constructor)

**Step 1: Write failing test**

Add to the `tests` module in `event_loop.rs`:

```rust
#[test]
fn runtime_has_no_peers_initially() {
    let runtime = make_runtime();
    assert_eq!(runtime.peer_count(), 0);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel -- runtime_has_no_peers_initially`
Expected: FAIL — `peer_count` method does not exist.

**Step 3: Add new fields and peer_count() method**

Update the struct definition:

```rust
pub struct UnikernelRuntime<E: EntropySource, P: PersistentState> {
    node: Node,
    identity: PrivateIdentity,
    entropy: E,
    persistence: P,
    tick_count: u64,
    // Announcing
    dest_name: Option<DestinationName>,
    dest_hash: Option<DestinationHash>,
    // Peer tracking
    peers: BTreeMap<[u8; 16], PeerInfo>,
    heartbeat_interval_ms: u64,
    peer_timeout_ms: u64,
    last_heartbeat_ms: u64,
    boot_time_ms: u64,
}
```

Update the constructor:

```rust
pub fn new(identity: PrivateIdentity, entropy: E, persistence: P) -> Self {
    let node = Node::new();
    UnikernelRuntime {
        node,
        identity,
        entropy,
        persistence,
        tick_count: 0,
        dest_name: None,
        dest_hash: None,
        peers: BTreeMap::new(),
        heartbeat_interval_ms: 5_000,
        peer_timeout_ms: 15_000,
        last_heartbeat_ms: 0,
        boot_time_ms: 0,
    }
}
```

Add accessor:

```rust
/// Number of currently tracked peers.
pub fn peer_count(&self) -> usize {
    self.peers.len()
}
```

**Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All 6 tests pass (5 existing + 1 new).

**Step 5: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "feat(unikernel): add peer table and heartbeat fields to UnikernelRuntime"
```

---

### Task 3: register_announcing_destination()

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs`

**Step 1: Write failing test**

```rust
#[test]
fn register_announcing_destination_sets_dest_hash() {
    let mut runtime = make_runtime();
    let dest_hash = runtime.register_announcing_destination("harmony", &["node"], 300_000, 0);
    assert_ne!(dest_hash, [0u8; 16]);
    assert_eq!(runtime.node.announcing_destination_count(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel -- register_announcing`
Expected: FAIL — method does not exist.

**Step 3: Implement**

```rust
/// Register this node's identity as an announcing destination.
///
/// Creates a `DestinationName` from `app_name` and `aspects`,
/// registers it on the inner `Node` for announce scheduling,
/// and stores the destination hash for heartbeat routing.
///
/// `announce_interval_ms` is in milliseconds; the `Node` scheduler
/// uses seconds internally, so we convert.
///
/// Returns the computed destination hash.
pub fn register_announcing_destination(
    &mut self,
    app_name: &str,
    aspects: &[&str],
    announce_interval_ms: u64,
    now: u64,
) -> DestinationHash {
    let dest_name = DestinationName::from_name(app_name, aspects)
        .expect("invalid destination name");
    let identity_clone = self.identity.clone();
    let announce_interval_secs = announce_interval_ms / 1000;
    let now_secs = now / 1000;
    let dest_hash = self.node.register_announcing_destination(
        identity_clone,
        dest_name.clone(),
        Vec::new(),
        Some(announce_interval_secs),
        now_secs,
    );
    self.dest_name = Some(dest_name);
    self.dest_hash = Some(dest_hash);
    self.boot_time_ms = now;
    dest_hash
}
```

**Important:** Check that `PrivateIdentity` implements `Clone`. If not, we need to reconstruct from bytes. The `register_announcing_destination` on `Node` takes ownership of the identity, so we need a copy. Check `harmony-identity` — the `PrivateIdentity` has `to_private_bytes()` / `from_private_bytes()` for round-tripping. Use that:

```rust
let identity_bytes = self.identity.to_private_bytes();
let identity_clone = PrivateIdentity::from_private_bytes(&identity_bytes)
    .expect("identity round-trip failed");
```

**Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All 7 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "feat(unikernel): add register_announcing_destination method"
```

---

### Task 4: Enhanced tick() — announce resolution and peer timeout

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs`

This is the core change: `tick()` returns `Vec<RuntimeAction>` instead of `Vec<NodeAction>`, absorbs `AnnounceNeeded` internally, and checks peer timeouts.

**Step 1: Write failing tests**

```rust
#[test]
fn tick_resolves_announce_needed_internally() {
    let mut runtime = make_runtime();
    runtime.register_interface("test0");
    // Register with announce_interval=1000ms (1 second).
    // Node scheduler uses seconds, so next_announce_at = now_secs.
    runtime.register_announcing_destination("harmony", &["node"], 1_000, 0);

    // Tick at time 1000ms (1 second) — should trigger announce.
    let actions = runtime.tick(1_000);

    // Should have SendOnInterface (the announce), not AnnounceNeeded.
    let has_send = actions.iter().any(|a| matches!(a, RuntimeAction::SendOnInterface { .. }));
    assert!(has_send, "tick should resolve announces into SendOnInterface");
}

#[test]
fn tick_emits_peer_lost_after_timeout() {
    let mut runtime = make_runtime();
    runtime.register_interface("test0");

    // Manually insert a peer.
    runtime.peers.insert([0xAA; 16], PeerInfo {
        address_hash: [0xAA; 16],
        last_seen_ms: 0,
        hops: 1,
        discovered_at_ms: 0,
    });
    assert_eq!(runtime.peer_count(), 1);

    // Tick at 16 seconds — peer_timeout_ms is 15_000.
    let actions = runtime.tick(16_000);
    assert_eq!(runtime.peer_count(), 0);
    let has_lost = actions.iter().any(|a| matches!(a, RuntimeAction::PeerLost { .. }));
    assert!(has_lost, "should emit PeerLost");
}
```

**Step 2: Run tests to verify they fail**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel -- tick_resolves tick_emits_peer_lost`
Expected: FAIL — `tick()` still returns `Vec<NodeAction>`.

**Step 3: Implement enhanced tick()**

Replace the existing `tick()` method:

```rust
/// Process a timer tick.
///
/// Internally resolves `AnnounceNeeded` (two-phase announce),
/// emits heartbeats, and checks peer timeouts. Returns only
/// `RuntimeAction` variants — no protocol-internal actions leak.
///
/// `now` is monotonic milliseconds from `PitTimer::now_ms()`.
pub fn tick(&mut self, now: u64) -> Vec<RuntimeAction> {
    self.tick_count += 1;
    let mut out = Vec::new();

    // Convert to seconds for the Node scheduler.
    let now_secs = now / 1000;
    let node_actions = self.node.handle_event(NodeEvent::TimerTick { now: now_secs });

    for action in node_actions {
        match action {
            NodeAction::AnnounceNeeded { dest_hash } => {
                // Resolve two-phase announce: call node.announce() with RNG.
                let announce_actions = self.node.announce(
                    &dest_hash,
                    &mut self.entropy,
                    now_secs,
                );
                for aa in announce_actions {
                    if let NodeAction::SendOnInterface { interface_name, raw } = aa {
                        out.push(RuntimeAction::SendOnInterface { interface_name, raw });
                    }
                }
            }
            NodeAction::SendOnInterface { interface_name, raw } => {
                out.push(RuntimeAction::SendOnInterface { interface_name, raw });
            }
            _ => {
                // PathsExpired, ReverseTableExpired, etc. — diagnostic, skip.
            }
        }
    }

    // Check peer timeouts.
    let timeout = self.peer_timeout_ms;
    let timed_out: Vec<[u8; 16]> = self.peers
        .iter()
        .filter(|(_, p)| now.saturating_sub(p.last_seen_ms) > timeout)
        .map(|(k, _)| *k)
        .collect();
    for addr in timed_out {
        self.peers.remove(&addr);
        out.push(RuntimeAction::PeerLost { address_hash: addr });
    }

    out
}
```

**Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All tests pass. Note: existing tests that called `tick()` and expected `Vec<NodeAction>` will need their type annotations updated since `tick()` now returns `Vec<RuntimeAction>`. Update:
- `tick_increments_counter` — just calls `tick()`, doesn't inspect return type, should be fine.
- `tick_without_interfaces_returns_empty_or_minimal_actions` — matches on `NodeAction::SendOnInterface`. Update to match `RuntimeAction::SendOnInterface`.

**Step 5: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "feat(unikernel): tick() resolves announces and checks peer timeouts"
```

---

### Task 5: Enhanced handle_packet() — peer discovery and heartbeat reception

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs`

**Step 1: Write failing tests**

```rust
#[test]
fn handle_packet_emits_peer_discovered_on_announce() {
    let mut runtime = make_runtime();
    runtime.register_interface("test0");
    runtime.register_announcing_destination("harmony", &["node"], 300_000, 0);

    // Build a valid announce from a second identity.
    let mut entropy2 = test_entropy();
    let peer_identity = PrivateIdentity::generate(&mut entropy2);
    let peer_addr = peer_identity.public_identity().address_hash;
    let dest_name = harmony_reticulum::destination::DestinationName::from_name(
        "harmony", &["node"],
    ).unwrap();

    let announce_packet = harmony_reticulum::announce::build_announce(
        &peer_identity, &dest_name, &mut entropy2, 0, &[], None,
    ).unwrap();
    let raw = announce_packet.to_bytes().unwrap();

    let actions = runtime.handle_packet("test0", raw, 1_000);
    let has_discovered = actions.iter().any(|a| matches!(
        a, RuntimeAction::PeerDiscovered { address_hash, .. } if *address_hash == peer_addr
    ));
    assert!(has_discovered, "should emit PeerDiscovered on valid announce");
    assert_eq!(runtime.peer_count(), 1);
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel -- handle_packet_emits_peer_discovered`
Expected: FAIL — `handle_packet()` still returns `Vec<NodeAction>`.

**Step 3: Implement enhanced handle_packet()**

Replace the existing `handle_packet()` method:

```rust
/// Feed an inbound packet into the node and translate results.
///
/// Intercepts `AnnounceReceived` to update the peer table and
/// `DeliverLocally` to parse heartbeat payloads.
///
/// `now` is monotonic milliseconds.
pub fn handle_packet(
    &mut self,
    interface_name: &str,
    data: Vec<u8>,
    now: u64,
) -> Vec<RuntimeAction> {
    let now_secs = now / 1000;
    let node_actions = self.node.handle_event(NodeEvent::InboundPacket {
        interface_name: String::from(interface_name),
        raw: data,
        now: now_secs,
    });

    let mut out = Vec::new();

    for action in node_actions {
        match action {
            NodeAction::AnnounceReceived {
                destination_hash,
                validated_announce,
                hops,
                ..
            } => {
                let addr = validated_announce.identity.address_hash;
                let is_new = !self.peers.contains_key(&addr);
                self.peers.insert(addr, PeerInfo {
                    address_hash: addr,
                    dest_hash: destination_hash,
                    last_seen_ms: now,
                    hops,
                    discovered_at_ms: if is_new { now } else {
                        self.peers.get(&addr)
                            .map(|p| p.discovered_at_ms)
                            .unwrap_or(now)
                    },
                });
                if is_new {
                    out.push(RuntimeAction::PeerDiscovered {
                        address_hash: addr,
                        hops,
                    });
                }
            }
            NodeAction::DeliverLocally { packet, destination_hash, .. } => {
                let payload: &[u8] = &packet.data;
                // Check for heartbeat magic: "HBT\x01"
                if payload.len() >= 28
                    && payload[0..4] == [0x48, 0x42, 0x54, 0x01]
                {
                    let mut sender = [0u8; 16];
                    sender.copy_from_slice(&payload[4..20]);
                    let uptime_ms = u64::from_be_bytes(
                        payload[20..28].try_into().unwrap()
                    );
                    // Update last_seen for this peer.
                    if let Some(peer) = self.peers.get_mut(&sender) {
                        peer.last_seen_ms = now;
                    }
                    out.push(RuntimeAction::HeartbeatReceived {
                        address_hash: sender,
                        uptime_ms,
                    });
                } else {
                    out.push(RuntimeAction::DeliverLocally {
                        destination_hash,
                        payload: payload.to_vec(),
                    });
                }
            }
            NodeAction::AnnounceNeeded { dest_hash } => {
                // Resolve inline (can happen on inbound too).
                let announce_actions = self.node.announce(
                    &dest_hash,
                    &mut self.entropy,
                    now_secs,
                );
                for aa in announce_actions {
                    if let NodeAction::SendOnInterface { interface_name, raw } = aa {
                        out.push(RuntimeAction::SendOnInterface { interface_name, raw });
                    }
                }
            }
            NodeAction::SendOnInterface { interface_name, raw } => {
                out.push(RuntimeAction::SendOnInterface { interface_name, raw });
            }
            _ => {}
        }
    }

    out
}
```

**Step 4: Add test imports**

The test needs access to `harmony_reticulum::announce::build_announce` and `Packet::to_bytes()`. Add to the test module's imports:

```rust
use harmony_reticulum::announce::build_announce;
use harmony_reticulum::destination::DestinationName;
```

**Step 5: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All tests pass.

**Step 6: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "feat(unikernel): handle_packet() tracks peers and parses heartbeats"
```

---

### Task 6: Heartbeat emission in tick()

**Files:**
- Modify: `crates/harmony-unikernel/src/event_loop.rs`

**Step 1: Write failing test**

```rust
#[test]
fn tick_emits_heartbeats_to_peers() {
    let mut runtime = make_runtime();
    runtime.register_interface("test0");
    runtime.register_announcing_destination("harmony", &["node"], 300_000, 0);

    // Insert a fake peer.
    runtime.peers.insert([0xBB; 16], PeerInfo {
        address_hash: [0xBB; 16],
        last_seen_ms: 0,
        hops: 1,
        discovered_at_ms: 0,
    });

    // Tick at 6 seconds — past heartbeat_interval_ms (5000).
    let actions = runtime.tick(6_000);

    // Should have at least one SendOnInterface for the heartbeat.
    let send_count = actions.iter().filter(|a| matches!(a, RuntimeAction::SendOnInterface { .. })).count();
    assert!(send_count > 0, "should emit heartbeat SendOnInterface");
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel -- tick_emits_heartbeats`
Expected: FAIL — heartbeat emission not implemented yet.

**Step 3: Add heartbeat building and emission to tick()**

Add a helper method for building heartbeat packets:

```rust
/// Build a 28-byte heartbeat payload.
fn build_heartbeat(&self, now: u64) -> [u8; 28] {
    let mut buf = [0u8; 28];
    // Magic: "HBT\x01"
    buf[0..4].copy_from_slice(&[0x48, 0x42, 0x54, 0x01]);
    // Sender address hash
    buf[4..20].copy_from_slice(&self.identity.public_identity().address_hash);
    // Uptime
    let uptime = now.saturating_sub(self.boot_time_ms);
    buf[20..28].copy_from_slice(&uptime.to_be_bytes());
    buf
}
```

Add a method for building a Reticulum data packet from a heartbeat payload:

```rust
use harmony_reticulum::packet::{
    Packet, PacketHeader, PacketFlags, HeaderType, PropagationType,
    DestinationType, PacketType,
};
use harmony_reticulum::context::PacketContext;

/// Build a Type1 broadcast data packet addressed to `dest_hash`
/// containing the given payload.
fn build_data_packet(dest_hash: &DestinationHash, payload: &[u8]) -> Option<Vec<u8>> {
    let packet = Packet {
        header: PacketHeader {
            flags: PacketFlags {
                ifac: false,
                header_type: HeaderType::Type1,
                context_flag: false,
                propagation: PropagationType::Broadcast,
                destination_type: DestinationType::Single,
                packet_type: PacketType::Data,
            },
            hops: 0,
            transport_id: None,
            destination_hash: *dest_hash,
            context: PacketContext::None,
        },
        data: Arc::from(payload),
    };
    packet.to_bytes().ok()
}
```

Then add heartbeat emission at the end of `tick()`, before the peer timeout check:

```rust
// Emit heartbeats if interval has elapsed and we have peers.
if !self.peers.is_empty()
    && now.saturating_sub(self.last_heartbeat_ms) >= self.heartbeat_interval_ms
{
    let hbt = self.build_heartbeat(now);
    // Route a heartbeat to each peer's destination hash so that
    // DeliverLocally fires on the receiving node.
    let peer_dest_hashes: Vec<[u8; 16]> =
        self.peers.values().map(|p| p.dest_hash).collect();
    for peer_dest in &peer_dest_hashes {
        if let Some(raw) = Self::build_data_packet(peer_dest, &hbt) {
            let send_actions = self.node.route_packet(peer_dest, raw);
            for sa in send_actions {
                if let NodeAction::SendOnInterface { interface_name, raw } = sa {
                    out.push(RuntimeAction::SendOnInterface { interface_name, raw });
                }
            }
        }
    }
    self.last_heartbeat_ms = now;
}
```

**Note on routing:** Heartbeats are addressed to each peer's destination hash (learned from their announce). This ensures `DeliverLocally` fires on the receiving node. On a broadcast LAN, the physical packets may overlap, but the Reticulum destination_hash in each packet header must match the receiver's registered destination for delivery to work.

**Step 4: Run tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo test -p harmony-unikernel`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-unikernel/src/event_loop.rs
git commit -m "feat(unikernel): emit broadcast heartbeat in tick()"
```

---

### Task 7: PIT timer driver

**Files:**
- Create: `crates/harmony-boot/src/pit.rs`
- Modify: `crates/harmony-boot/src/main.rs:12` (add `mod pit;`)

**Step 1: Implement PIT timer**

Create `crates/harmony-boot/src/pit.rs`:

```rust
// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal PIT (8254) Channel 2 timer for monotonic milliseconds.
//!
//! Uses the "read-back" method: load a full 16-bit countdown, then
//! read the current count to measure elapsed ticks. No interrupts.

use x86_64::instructions::port::Port;

/// PIT oscillator frequency in Hz.
const PIT_FREQUENCY: u64 = 1_193_182;

/// PIT I/O ports.
const CHANNEL_2: u16 = 0x42;
const COMMAND: u16 = 0x43;
/// Port 0x61 controls the Channel 2 gate.
const PORT_61: u16 = 0x61;

/// Monotonic millisecond timer using PIT Channel 2.
pub struct PitTimer {
    accumulated_ticks: u64,
    last_count: u16,
}

impl PitTimer {
    /// Initialize PIT Channel 2 in mode 0 (one-shot), counting down
    /// from 0xFFFF (~54.9ms at 1.193182 MHz).
    pub fn init() -> Self {
        unsafe {
            // Enable Channel 2 gate (bit 0 of port 0x61), disable speaker (bit 1).
            let mut port61: Port<u8> = Port::new(PORT_61);
            let val = port61.read();
            port61.write((val | 0x01) & !0x02);

            // Command: Channel 2, lobyte/hibyte, mode 0 (one-shot), binary.
            // 0b10_11_000_0 = 0xB0
            Port::new(COMMAND).write(0xB0u8);

            // Load count = 0xFFFF (maximum countdown).
            let mut ch2: Port<u8> = Port::new(CHANNEL_2);
            ch2.write(0xFFu8); // low byte
            ch2.write(0xFFu8); // high byte
        }

        let initial = Self::read_count();

        PitTimer {
            accumulated_ticks: 0,
            last_count: initial,
        }
    }

    /// Read the current 16-bit count from Channel 2.
    fn read_count() -> u16 {
        unsafe {
            // Latch Channel 2: command 0b10_00_000_0 = 0x80
            Port::new(COMMAND).write(0x80u8);
            let mut ch2: Port<u8> = Port::new(CHANNEL_2);
            let lo = ch2.read() as u16;
            let hi = ch2.read() as u16;
            (hi << 8) | lo
        }
    }

    /// Return monotonic milliseconds since `init()`.
    ///
    /// Must be called at least once per ~55ms to avoid missing a
    /// counter wraparound. The spin-loop event loop calls this every
    /// iteration, so this is easily satisfied.
    pub fn now_ms(&mut self) -> u64 {
        let current = Self::read_count();

        // The counter counts DOWN. Elapsed = last - current.
        // If current > last, the counter wrapped around (passed 0).
        let elapsed = if current <= self.last_count {
            self.last_count - current
        } else {
            // Wrapped: last_count ticks down to 0, then 0xFFFF down to current.
            self.last_count + (0xFFFF - current) + 1
        };

        self.accumulated_ticks += elapsed as u64;
        self.last_count = current;

        // Reload the counter if it's getting low, to avoid
        // reaching 0 and stopping (mode 0 stops at terminal count).
        if current < 1000 {
            unsafe {
                Port::new(COMMAND).write(0xB0u8);
                let mut ch2: Port<u8> = Port::new(CHANNEL_2);
                ch2.write(0xFFu8);
                ch2.write(0xFFu8);
            }
            self.last_count = Self::read_count();
        }

        // Convert accumulated ticks to milliseconds.
        self.accumulated_ticks * 1000 / PIT_FREQUENCY
    }
}
```

**Step 2: Add module declaration**

Add `mod pit;` to `crates/harmony-boot/src/main.rs` after `mod virtio;` (line 14).

**Step 3: Build to verify**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && just build-kernel`
Expected: Compiles without errors. (Can't unit test PIT — it uses I/O ports, only works on real x86 / QEMU.)

**Step 4: Commit**

```bash
git add crates/harmony-boot/src/pit.rs crates/harmony-boot/src/main.rs
git commit -m "feat(boot): add PIT Channel 2 timer for monotonic milliseconds"
```

---

### Task 8: Rewire main.rs event loop

**Files:**
- Modify: `crates/harmony-boot/src/main.rs`

This task integrates everything: PIT timer, announcing destination, and RuntimeAction dispatch.

**Step 1: Update imports**

Replace the harmony-unikernel imports at the top of main.rs:

```rust
use harmony_unikernel::serial::{hex_encode, SerialWriter};
use harmony_unikernel::{KernelEntropy, MemoryState, RuntimeAction, UnikernelRuntime};
```

Remove: `use harmony_platform::NetworkInterface;` — move it to the dispatch function's local use.
Remove: `use harmony_unikernel::NodeAction;` — no longer needed.

**Step 2: Add PIT init after serial init**

After `serial.log("BOOT", "Harmony unikernel v0.1.0");` (line 166), add:

```rust
let mut pit = pit::PitTimer::init();
serial.log("PIT", "timer initialized");
```

**Step 3: Update announcing destination registration**

After `runtime.register_interface("virtio0");` (line 263), add:

```rust
let now = pit.now_ms();
let dest_hash = runtime.register_announcing_destination("harmony", &["node"], 300_000, now);
let mut dest_hex = [0u8; 32];
hex_encode(&dest_hash, &mut dest_hex);
let dest_str = core::str::from_utf8(&dest_hex).unwrap_or("????????????????????????????????");
serial.log("DEST", dest_str);
```

**Step 4: Replace event loop**

Replace the entire event loop (from `let mut tick: u64 = 0;` to the end of the `loop`) with:

```rust
serial.log("READY", "entering event loop");

#[cfg(feature = "qemu-test")]
qemu_debug_exit(0x10);

loop {
    let now = pit.now_ms();

    // Poll network for inbound packets.
    if let Some(ref mut net) = virtio_net {
        while let Some(data) = harmony_platform::NetworkInterface::receive(net) {
            let actions = runtime.handle_packet("virtio0", data, now);
            dispatch_actions(&actions, &mut virtio_net, &mut serial);
        }
    }

    // Timer tick.
    let actions = runtime.tick(now);
    dispatch_actions(&actions, &mut virtio_net, &mut serial);

    core::hint::spin_loop();
}
```

**Step 5: Add dispatch_actions() function**

Add before `kernel_main()`:

```rust
/// Dispatch RuntimeActions: send packets and log events.
fn dispatch_actions(
    actions: &[RuntimeAction],
    virtio_net: &mut Option<virtio::net::VirtioNet>,
    serial: &mut SerialWriter<impl FnMut(u8)>,
) {
    use core::fmt::Write;
    use harmony_unikernel::serial::hex_encode;

    for action in actions {
        match action {
            RuntimeAction::SendOnInterface { interface_name, raw } => {
                if interface_name.as_ref() == "virtio0" {
                    if let Some(ref mut net) = virtio_net {
                        let _ = harmony_platform::NetworkInterface::send(net, raw);
                    }
                }
            }
            RuntimeAction::PeerDiscovered { address_hash, hops } => {
                let mut hex = [0u8; 32];
                hex_encode(address_hash, &mut hex);
                let s = core::str::from_utf8(&hex).unwrap_or("?");
                let _ = writeln!(serial, "[PEER+] {} ({} hops)", s, hops);
            }
            RuntimeAction::PeerLost { address_hash } => {
                let mut hex = [0u8; 32];
                hex_encode(address_hash, &mut hex);
                let s = core::str::from_utf8(&hex).unwrap_or("?");
                let _ = writeln!(serial, "[PEER-] {}", s);
            }
            RuntimeAction::HeartbeatReceived { address_hash, uptime_ms } => {
                let mut hex = [0u8; 32];
                hex_encode(address_hash, &mut hex);
                let s = core::str::from_utf8(&hex).unwrap_or("?");
                let _ = writeln!(serial, "[HBT] {} uptime={}ms", s, uptime_ms);
            }
            RuntimeAction::DeliverLocally { destination_hash, payload } => {
                let mut hex = [0u8; 32];
                hex_encode(destination_hash, &mut hex);
                let s = core::str::from_utf8(&hex).unwrap_or("?");
                let _ = writeln!(serial, "[RECV] {} {}B", s, payload.len());
            }
        }
    }
}
```

**Step 6: Build and smoke test**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && just build-kernel`
Expected: Compiles without errors.

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && just test-qemu`
Expected: QEMU boots, prints `[IDENTITY]`, `[PIT]`, `[DEST]`, `[READY]`, exits cleanly.

**Step 7: Commit**

```bash
git add crates/harmony-boot/src/main.rs
git commit -m "feat(boot): rewire event loop with PIT timer, announcing, and RuntimeAction dispatch"
```

---

### Task 9: Two-node smoke test

**Files:** None modified — manual verification.

**Step 1: Run two QEMU peers**

In terminal 1:
```bash
cd /Users/zeblith/work/zeblithic/harmony-os && just run
```

In terminal 2:
```bash
cd /Users/zeblith/work/zeblithic/harmony-os && just run-peer
```

**Step 2: Verify serial output**

Expected on each node after ~5-10 seconds:

```
[BOOT] Harmony unikernel v0.1.0
[PIT] timer initialized
[HEAP] 4194304
[ENTROPY] RDRAND available
[IDENTITY] <32 hex chars>
[VIRTIO] <mac address>
[DEST] <32 hex chars>
[READY] entering event loop
[PEER+] <other node's address> (1 hops)
[HBT] <other node's address> uptime=<N>ms
[HBT] <other node's address> uptime=<N>ms
...
```

**Step 3: Verify peer loss detection**

Kill one QEMU instance (Ctrl-C). The other node should print after ~15 seconds:

```
[PEER-] <dead node's address>
```

**Step 4: Document results**

If everything works, run full quality gates:

```bash
cd /Users/zeblith/work/zeblithic/harmony-os && just check
```

Expected: All tests pass, clippy clean, fmt clean.

**Step 5: Commit any final fixes**

If the smoke test revealed issues, fix and commit each fix individually with descriptive messages.

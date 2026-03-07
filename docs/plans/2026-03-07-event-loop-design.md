# Ring 1 Unikernel Event Loop Design

**Status:** Approved design
**Date:** 2026-03-07
**Bead:** harmony-44x
**Scope:** harmony-os (crates/harmony-boot, crates/harmony-unikernel)

## Goal

Wire the Harmony node runtime into the unikernel event loop so two
QEMU nodes discover each other via Reticulum announces and exchange
periodic heartbeat messages. This completes Ring 1: a bootable
single-purpose node that participates in the Harmony network.

## Milestone

Two instances of the unikernel running in QEMU on the same virtual
LAN (multicast 230.0.0.1:1234) should:

1. Boot, generate identity, initialize VirtIO-net
2. Register a "harmony.node" announcing destination
3. Broadcast Reticulum announces at 300-second intervals
4. Discover each other when announces are received
5. Send periodic heartbeat data packets (every 5 seconds)
6. Detect peer disappearance after 15 seconds of silence
7. Log all events (discovery, heartbeats, loss) to serial

## PIT Timer

A minimal PIT (8254) driver using Channel 2 in one-shot mode to
provide wall-clock milliseconds. No interrupts — purely polled.

PIT Channel 2 counts down from a loaded value at 1.193182 MHz. We
read the current count to measure elapsed time. On each call to
`now_ms()`, we accumulate elapsed ticks and convert to milliseconds.

### API

```rust
// crates/harmony-boot/src/pit.rs
pub struct PitTimer {
    accumulated_ticks: u64,
    last_count: u16,
}

impl PitTimer {
    /// Initialize PIT Channel 2 in one-shot mode.
    pub fn init() -> Self;

    /// Read current monotonic time in milliseconds.
    /// Must be called at least once per ~55ms to avoid
    /// counter wraparound (the PIT is 16-bit at ~1.19 MHz).
    pub fn now_ms(&mut self) -> u64;
}
```

### Integration

Constructed in `kernel_main()` after serial init. Every event loop
iteration calls `now_ms()` and passes the result to `runtime.tick(now)`
and `runtime.handle_packet(..., now)`.

### Counter reload safety

PIT Channel 2 in mode 0 counts down and stops at 0 (it does not
wrap). The timer reloads the counter when it drops below 1000 ticks
(~0.84ms), well before it reaches terminal count. The spin loop
iterates far faster than 55ms, so the reload always fires in time.
Up to ~1000 ticks may be lost per reload — acceptable for Ring 1's
5s/15s protocol intervals.

## UnikernelRuntime Extensions

The runtime absorbs the two-phase announce pattern, peer tracking, and
heartbeat scheduling. Main.rs never sees `AnnounceNeeded` or raw
`NodeAction` — the runtime resolves everything internally and returns
clean `RuntimeAction`s.

### New fields

```rust
pub struct UnikernelRuntime<E, P> {
    // existing
    node: Node,
    identity: PrivateIdentity,
    entropy: E,
    persistence: P,
    tick_count: u64,

    // new
    peers: BTreeMap<[u8; 16], PeerInfo>,
    heartbeat_interval_ms: u64,       // default: 5000
    peer_timeout_ms: u64,             // default: 15000 (3x heartbeat)
    last_heartbeat_ms: u64,
}

pub struct PeerInfo {
    pub address_hash: [u8; 16],
    pub dest_hash: [u8; 16],
    pub last_seen_ms: u64,
    pub hops: u8,
    pub discovered_at_ms: u64,
}
```

### New methods

- `register_announcing_destination(name, aspects, announce_interval_ms, now)`
  — Registers the node's identity as an announcing destination on the
  `Node`. Returns the destination hash for logging.

- `tick(now) -> Vec<RuntimeAction>` — Enhanced: after calling
  `node.handle_event(TimerTick)`, internally resolves any
  `AnnounceNeeded` by calling `node.announce(rng)`. Then checks
  heartbeat timer and emits heartbeats to all known peers. Checks peer
  timeouts and emits `PeerLost` for silent peers.

- `handle_packet(interface_name, data, now) -> Vec<RuntimeAction>` —
  Enhanced: after calling node, intercepts `AnnounceReceived` to update
  peer table (emitting `PeerDiscovered` for new peers), and intercepts
  `DeliverLocally` to parse heartbeat payloads and update `last_seen_ms`.

### RuntimeAction enum

```rust
pub enum RuntimeAction {
    /// Send raw bytes on a named interface.
    SendOnInterface { interface_name: Arc<str>, raw: Vec<u8> },
    /// A new peer was discovered via announce.
    PeerDiscovered { address_hash: [u8; 16], hops: u8 },
    /// A peer hasn't been heard from in peer_timeout_ms.
    PeerLost { address_hash: [u8; 16] },
    /// A heartbeat was received from a peer.
    HeartbeatReceived { address_hash: [u8; 16], uptime_ms: u64 },
    /// A non-heartbeat packet was delivered locally.
    DeliverLocally { destination_hash: [u8; 16], payload: Vec<u8> },
}
```

## Heartbeat Protocol

A simple binary heartbeat message, sent as a Reticulum data packet to
each discovered peer's destination hash.

### Wire format (28 bytes)

```
Offset  Size   Field
------  -----  -------------------------
  0       4    Magic: 0x48 0x42 0x54 0x01 ("HBT" + version 1)
  4      16    Sender's address_hash
 20       8    Sender's uptime_ms (u64 big-endian)
------  -----
 28     total
```

### Sending

On each `tick()`, if `now - last_heartbeat_ms >= heartbeat_interval_ms`,
the runtime iterates over known peers, builds a heartbeat packet for
each, and calls `node.route_packet(peer_dest_hash, heartbeat_bytes)`.
Updates `last_heartbeat_ms`.

### Receiving

When `DeliverLocally` arrives, the runtime checks if the payload starts
with the HBT magic. If so, it parses the sender address hash, updates
`peer.last_seen_ms`, and emits `RuntimeAction::HeartbeatReceived`.
Non-heartbeat payloads pass through as `RuntimeAction::DeliverLocally`.

### Peer timeout

On each `tick()`, after heartbeat sending, the runtime scans all peers.
Any peer with `now - last_seen_ms > peer_timeout_ms` is removed and
emitted as `RuntimeAction::PeerLost`. An announce from a previously-lost
peer re-discovers them (`PeerDiscovered` fires again).

### Default intervals

- Heartbeat: 5,000 ms (5 seconds)
- Timeout: 15,000 ms (3x heartbeat — tolerates 2 missed beats)
- Announce: 300,000 ms (5 minutes, standard Reticulum interval)

## main.rs Event Loop Changes

### Additions

- PIT timer init after serial init
- `runtime.register_announcing_destination("harmony", &["node"], 300_000, now)`
- Replace bare `tick` counter with `pit.now_ms()`

### Simplified dispatch

```rust
loop {
    let now = pit.now_ms();

    if let Some(ref mut net) = virtio_net {
        while let Some(data) = NetworkInterface::receive(net) {
            let actions = runtime.handle_packet("virtio0", data, now);
            dispatch(&mut serial, &mut virtio_net, &actions);
        }
    }

    let actions = runtime.tick(now);
    dispatch(&mut serial, &mut virtio_net, &actions);

    core::hint::spin_loop();
}
```

### dispatch() helper

- `SendOnInterface` — call `net.send(&raw)`
- `PeerDiscovered` — log `[PEER+] <hex_address> (N hops)`
- `PeerLost` — log `[PEER-] <hex_address>`
- `HeartbeatReceived` — log `[HBT] <hex_address> uptime=Nms`
- `DeliverLocally` — log `[RECV] <hex_address> N bytes`

## Testing Strategy

### Unit tests (harmony-unikernel, host-side)

The runtime is fully sans-I/O, so all logic is testable without QEMU:

- **Announce resolution** — `tick()` with enough elapsed time returns
  `SendOnInterface` (not `AnnounceNeeded`)
- **Peer discovery** — feed valid announce via `handle_packet()`, verify
  `PeerDiscovered` emitted and peer table populated
- **Heartbeat emission** — advance time past interval, verify heartbeat
  `SendOnInterface` actions for each known peer
- **Heartbeat reception** — feed heartbeat data packet, verify
  `HeartbeatReceived` emitted and `last_seen_ms` updated
- **Peer timeout** — advance time past timeout with no heartbeats,
  verify `PeerLost` emitted and peer removed
- **Peer re-discovery** — after `PeerLost`, feed announce, verify
  `PeerDiscovered` fires again

### Integration test (QEMU)

Existing `just test-qemu` verifies single-node boot. Two-node discovery
is validated manually via `just run` + `just run-peer`.

## File Inventory

### Create

- `crates/harmony-boot/src/pit.rs` — PIT Channel 2 timer (~50 lines)

### Modify

- `crates/harmony-unikernel/src/event_loop.rs` — RuntimeAction, peer
  table, heartbeat, announce absorption, tests
- `crates/harmony-unikernel/src/lib.rs` — re-export RuntimeAction,
  PeerInfo
- `crates/harmony-boot/src/main.rs` — PIT init, announcing destination,
  RuntimeAction dispatch

### Unchanged

- All virtio module files, pci.rs, serial.rs, harmony-platform traits

## Non-Goals (YAGNI)

- No Reticulum link establishment (follow-up bead)
- No encryption of heartbeat packets (comes with links)
- No LXMF-compatible destination naming (follow-up)
- No HPET/LAPIC/TSC timer (PIT suffices for Ring 1)
- No interrupt-driven I/O (stay in poll mode)
- No persistent identity (MemoryState is fine)
- No multi-interface support beyond virtio0

## Future Work

- **Reticulum links** — upgrade heartbeats to run over encrypted link
  channels. The liveness detection and peer tracking built here carries
  forward. Links use ~0.44 bits/sec per peer for keepalive, enabling
  thousands of simultaneous peer connections.
- **LXMF messaging** — add an "lxmf.delivery" announcing destination
  for Reticulum-compatible messaging.
- **Timer upgrade** — HPET or LAPIC timer with interrupt-driven wakeup
  to replace spin-loop with `hlt`.

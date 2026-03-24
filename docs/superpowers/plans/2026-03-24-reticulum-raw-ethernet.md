# Reticulum Raw Ethernet Transport Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add raw L2 Ethernet transport for Reticulum packets via the existing rawlink bridge, enabling IP-free mesh networking.

**Architecture:** Extend `harmony-rawlink::Bridge` with Reticulum (0x00) frame handling. Two mpsc channels connect the bridge to the event loop: inbound (L2 → Node) and outbound (Node → L2). The Reticulum Node state machine is unchanged — it processes `InboundPacket` events regardless of transport.

**Tech Stack:** Rust, harmony-rawlink (AF_PACKET bridge), harmony-reticulum (Node state machine), tokio mpsc channels

**Spec:** `docs/superpowers/specs/2026-03-24-reticulum-raw-ethernet-design.md`

---

### Task 1: Add Reticulum channel support to Bridge

Extend `BridgeConfig` and `Bridge` to accept channel pairs for Reticulum frame routing. Handle `frame_type::RETICULUM` (0x00) on inbound. Drain outbound channel each iteration.

**Files:**
- Modify: `crates/harmony-rawlink/src/bridge.rs` — add fields, inbound arm, outbound drain
- Modify: `crates/harmony-rawlink/src/lib.rs` — re-export channel types if needed

**Context:**
- `BridgeConfig` is at bridge.rs:24-45 with fields: identity_hash, subscribe_pattern, scout_interval, peer_ttl
- `Bridge` struct is at bridge.rs:47-53 with fields: socket, session, config, peer_table
- `Bridge::new()` at bridge.rs:60 takes `(socket, session, config)`
- Frame type match is at bridge.rs:189-253 with arms: SCOUT (190), DATA (204), other (250)
- Bridge loop body is at bridge.rs:86-151 with error recovery
- `frame_type::RETICULUM = 0x00` defined in lib.rs:23

- [ ] **Step 1: Add channel fields to BridgeConfig and Bridge**

In `BridgeConfig` (bridge.rs:24-45), add:
```rust
/// Channel for sending inbound Reticulum packets to the event loop.
/// None if Reticulum L2 transport is not configured.
pub reticulum_inbound_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
```

In `Bridge` struct (bridge.rs:47-53), add:
```rust
/// Channel for receiving outbound Reticulum packets from the event loop.
reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
```

Update `BridgeConfig::default()` to set `reticulum_inbound_tx: None`.

Update `Bridge::new()` to accept an optional outbound receiver:
```rust
pub fn new(
    socket: S,
    session: zenoh::Session,
    config: BridgeConfig,
    reticulum_outbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>>,
) -> Self {
    let peer_table = PeerTable::new(config.peer_ttl);
    Self {
        socket,
        session,
        config,
        peer_table,
        reticulum_outbound_rx,
    }
}
```

Update ALL callers of `Bridge::new()` — there's one in `event_loop.rs` (~line 252) and possibly in tests. Pass `None` for existing callers that don't use Reticulum.

- [ ] **Step 2: Add RETICULUM frame handling in recv_frames callback**

In the frame type match (bridge.rs:189-253), add a new arm BEFORE the `DATA` arm (the match goes SCOUT → RETICULUM → DATA → other):

```rust
frame_type::RETICULUM => {
    // Raw Reticulum packet — strip the 1-byte type tag and forward
    // to the event loop via the inbound channel.
    if payload.len() > 1 {
        if let Some(ref reticulum_tx) = reticulum_tx {
            let packet = payload[1..].to_vec();
            let _ = reticulum_tx.try_send(packet);
        }
    }
}
```

The `reticulum_tx` variable needs to be available inside the `recv_frames` callback. Add `config` to the existing destructure pattern (bridge.rs:~176-178):

```rust
let Self { socket, peer_table, config, .. } = self;
let reticulum_tx = &config.reticulum_inbound_tx;
```

This works because the destructure splits `self` into disjoint borrows — `socket` (mut), `peer_table` (mut), and `config` (shared ref). The closure captures `reticulum_tx` by reference.

- [ ] **Step 3: Add outbound Reticulum drain in bridge loop**

In the bridge loop body (bridge.rs:86-151), add a new step AFTER the outbound zenoh samples drain (after step 4, before step 5 yield). This goes inside the `async { ... }` block that catches errors:

```rust
// 5. Drain outbound Reticulum packets → broadcast as L2 frames.
if let Some(ref mut rx) = self.reticulum_outbound_rx {
    while let Ok(packet) = rx.try_recv() {
        let mut frame_payload = Vec::with_capacity(1 + packet.len());
        frame_payload.push(frame_type::RETICULUM);
        frame_payload.extend_from_slice(&packet);
        self.socket.send_frame(frame::BROADCAST_MAC, &frame_payload)?;
        trace!(packet_len = packet.len(), "outbound Reticulum frame sent");
    }
}
```

**Placement:** Insert this drain INSIDE the `async { ... }` error-catching block (bridge.rs:~90-121), after the zenoh outbound drain while-loop (step 4) and before `Ok(())`. It goes at approximately bridge.rs line 117-118. This avoids borrow conflicts because `process_inbound_frames` has already returned by this point.

- [ ] **Step 4: Write tests**

Add tests to the existing test module in bridge.rs:

```rust
#[test]
fn reticulum_frame_routed_to_inbound_channel() {
    let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);
    let (tx, rx) = std::sync::mpsc::channel();

    // Build a Reticulum frame payload: [0x00 tag][fake reticulum packet]
    let reticulum_packet = vec![0xAA; 100]; // fake 100-byte Reticulum packet
    let mut frame_payload = vec![frame_type::RETICULUM];
    frame_payload.extend_from_slice(&reticulum_packet);

    socket_b.send_frame(MAC_A, &frame_payload).expect("mock send");

    // Simulate what the bridge does for RETICULUM frames
    socket_a
        .recv_frames(&mut |_src_mac, payload| {
            if !payload.is_empty() && payload[0] == frame_type::RETICULUM && payload.len() > 1 {
                let packet = payload[1..].to_vec();
                let _ = tx.send(packet);
            }
        })
        .expect("recv should succeed");

    let received = rx.try_recv().expect("should have received a packet");
    assert_eq!(received, reticulum_packet);
}

#[test]
fn reticulum_outbound_encoding() {
    // Verify outbound Reticulum packet gets frame_type::RETICULUM prefix
    let reticulum_packet = vec![0xBB; 200];
    let mut frame_payload = Vec::with_capacity(1 + reticulum_packet.len());
    frame_payload.push(frame_type::RETICULUM);
    frame_payload.extend_from_slice(&reticulum_packet);

    assert_eq!(frame_payload[0], 0x00); // RETICULUM tag
    assert_eq!(&frame_payload[1..], &reticulum_packet[..]);
}

#[test]
fn interleaved_frame_types_routed_correctly() {
    let (mut socket_a, mut socket_b) = MockSocket::pair(MAC_A, MAC_B);
    let (ret_tx, ret_rx) = std::sync::mpsc::channel();

    // Send Scout, then Reticulum, then unknown
    let scout = make_scout_payload(&IDENTITY);
    let mut ret_frame = vec![frame_type::RETICULUM];
    ret_frame.extend_from_slice(&[0xCC; 50]);
    let unknown = vec![0xFF, 0x01, 0x02];

    socket_b.send_frame(MAC_A, &scout).unwrap();
    socket_b.send_frame(MAC_A, &ret_frame).unwrap();
    socket_b.send_frame(MAC_A, &unknown).unwrap();

    let mut peer_table = PeerTable::new(Duration::from_secs(30));
    let mut unknown_count = 0usize;

    socket_a
        .recv_frames(&mut |src_mac, payload| {
            if payload.is_empty() { return; }
            match payload[0] {
                frame_type::SCOUT if payload.len() >= 17 => {
                    let mut hash = [0u8; 16];
                    hash.copy_from_slice(&payload[1..17]);
                    peer_table.update(hash, *src_mac);
                }
                frame_type::RETICULUM if payload.len() > 1 => {
                    let _ = ret_tx.send(payload[1..].to_vec());
                }
                _ => { unknown_count += 1; }
            }
        })
        .unwrap();

    assert_eq!(peer_table.peer_count(), 1, "scout should update peer table");
    assert!(ret_rx.try_recv().is_ok(), "reticulum packet should be received");
    assert_eq!(unknown_count, 1, "unknown frame type should be counted");
}
```

- [ ] **Step 5: Run tests and commit**

```bash
cargo test -p harmony-rawlink
```
Expected: all tests pass (existing 20 + 3 new).

```bash
git add crates/harmony-rawlink/
git commit -m "feat(rawlink): add Reticulum frame handling with inbound/outbound channels"
```

---

### Task 2: Wire Reticulum channels into event loop

Create the channel pairs, pass them to the bridge, add a select loop arm for inbound packets, extend dispatch for outbound, and register the L2 interface with the runtime.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs` — channels, select arm, dispatch, interface registration

**Context:**
- Bridge spawning at event_loop.rs:240-269 (cfg-gated `#[cfg(all(target_os = "linux", feature = "rawlink"))]`)
- Select loop arms at event_loop.rs:455-701
- `dispatch_action()` at event_loop.rs:1056-1065
- `SendOnInterface` dispatch at event_loop.rs:1072-1102 (tunnel- prefix check at line 1083)
- UDP InboundPacket push at event_loop.rs:463-467
- Interface registration example in runtime.rs:1162-1166

- [ ] **Step 1: Create channels and pass to bridge**

In the rawlink bridge spawning block (event_loop.rs:240-269), before `Bridge::new()`:

```rust
// Create Reticulum L2 channels
let (ret_inbound_tx, ret_inbound_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
let (ret_outbound_tx, ret_outbound_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
```

Update `BridgeConfig` construction to include `reticulum_inbound_tx`:
```rust
let bridge_config = harmony_rawlink::BridgeConfig {
    identity_hash: runtime.local_pq_identity_hash(),
    reticulum_inbound_tx: Some(ret_inbound_tx),
    ..harmony_rawlink::BridgeConfig::default()
};
```

Update `Bridge::new()` call to pass the outbound receiver:
```rust
let mut bridge = harmony_rawlink::Bridge::new(
    socket,
    session.clone(),
    bridge_config,
    Some(ret_outbound_rx),
);
```

Store `ret_outbound_tx` and `ret_inbound_rx` for use in the select loop and dispatch. They need to be accessible after the cfg-gated block. Declare them before the block with `Option` wrappers:

```rust
// Before the cfg block:
let mut ret_inbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>> = None;
let mut ret_outbound_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>> = None;
let mut rawlink_iface_name: Option<String> = None;

// Inside the cfg block, after successful bridge setup:
ret_inbound_rx = Some(/* the rx from channel creation */);
ret_outbound_tx = Some(/* the tx from channel creation */);
rawlink_iface_name = Some(format!("l2:{}", iface));
```

On non-Linux or without the `rawlink` feature, these remain `None` — all rawlink paths are skipped.

- [ ] **Step 2: Register L2 interface with the runtime**

Inside the cfg-gated block, after spawning the bridge:
```rust
let iface_name = format!("l2:{}", iface);
runtime.push_event(RuntimeEvent::InboundPacket {
    interface_name: iface_name.clone(),
    raw: vec![], // empty packet — just to trigger interface awareness
    now: now_ms(),
});
```

The router's `register_interface` is called inside `push_event` when handling `TunnelHandshakeComplete` (runtime.rs:1162-1166). The `router` field is private. Follow the same pattern — add a new `RuntimeEvent` variant:

```rust
// In RuntimeEvent enum:
L2InterfaceReady { interface_name: String },
```

Handle it in `push_event`:
```rust
RuntimeEvent::L2InterfaceReady { interface_name } => {
    self.router.register_interface(
        interface_name,
        harmony_reticulum::InterfaceMode::Full,
        None, // no IFAC
    );
}
```

Then in the event loop's bridge setup block:
```rust
let iface_name = format!("l2:{}", iface);
runtime.push_event(RuntimeEvent::L2InterfaceReady {
    interface_name: iface_name.clone(),
});
```

This follows the existing `TunnelHandshakeComplete` pattern and avoids exposing new public API on NodeRuntime.

- [ ] **Step 3: Add select loop arm for inbound Reticulum packets**

In the select loop (event_loop.rs:455-701), add a new arm. The pattern for optional receivers:

```rust
// Add after the existing arms (e.g., after Arm 5 tunnel bridge):
result = async {
    match ret_inbound_rx.as_mut() {
        Some(rx) => rx.recv().await,
        None => std::future::pending().await,
    }
} => {
    if let Some(packet) = result {
        if let Some(ref iface_name) = rawlink_iface_name {
            runtime.push_event(RuntimeEvent::InboundPacket {
                interface_name: iface_name.clone(),
                raw: packet,
                now: now_ms(),
            });
        }
    }
}
```

The `std::future::pending()` pattern is used elsewhere in the event loop (e.g., for the iroh endpoint accept loop at line ~702) when a component is optional.

- [ ] **Step 4: Extend dispatch_action for L2 outbound**

In `dispatch_action()` (event_loop.rs:1056-1065), add `ret_outbound_tx: &Option<tokio::sync::mpsc::Sender<Vec<u8>>>` as a new parameter.

In the `SendOnInterface` handler (event_loop.rs:1072-1102), add the `l2:` prefix check BEFORE the `tunnel-` check:

```rust
RuntimeAction::SendOnInterface { interface_name, raw, weight } => {
    // ... existing weight/probabilistic check ...

    if interface_name.starts_with("l2:") {
        // Route to rawlink bridge for L2 broadcast
        if let Some(ref tx) = ret_outbound_tx {
            let _ = tx.try_send(raw);
        }
    } else if interface_name.starts_with("tunnel-") {
        // existing tunnel path
    } else {
        // existing UDP broadcast path
    }
}
```

Update BOTH call sites of `dispatch_action()` to pass the new parameter:
1. Line ~367 (startup actions loop)
2. Line ~895 (tick actions loop)

Search for `dispatch_action(` in event_loop.rs to confirm.

**Cross-platform note:** All new event_loop code is inside `#[cfg(all(target_os = "linux", feature = "rawlink"))]` blocks or uses `Option` wrappers. macOS CI will compile with `ret_inbound_rx: Option<...> = None` — verify the non-Linux path compiles correctly.

- [ ] **Step 5: Compile and test**

```bash
cargo test -p harmony-node
cargo test -p harmony-rawlink
cargo clippy -p harmony-node
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/ crates/harmony-rawlink/
git commit -m "feat(node): wire Reticulum L2 transport into event loop

Create mpsc channel pair connecting rawlink bridge to event loop.
Inbound: L2 RETICULUM frames → InboundPacket events.
Outbound: SendOnInterface with l2: prefix → bridge → AF_PACKET broadcast.
Register L2 interface with the Reticulum router at startup."
```

---

### Task 3: Final verification and cleanup

Run full workspace tests, verify no regressions, clean up any warnings.

**Files:**
- Possibly modify: any files with clippy warnings from Tasks 1-2

- [ ] **Step 1: Run full workspace tests**

```bash
cargo test --workspace
```
Expected: all tests pass across all crates.

- [ ] **Step 2: Run clippy**

```bash
cargo clippy --workspace
```
Fix any new warnings.

- [ ] **Step 3: Commit any cleanup**

```bash
git add .
git commit -m "chore: fix clippy warnings from Reticulum L2 transport"
```

(Skip this commit if clippy is clean.)

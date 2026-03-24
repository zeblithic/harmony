# Reticulum Raw Ethernet Transport Design

## Goal

Add raw L2 Ethernet transport for Reticulum packets, bypassing UDP/IP entirely. Combined with the existing Zenoh AF_PACKET transport (harmony-rawlink), this enables a fully IP-free Harmony mesh on 802.11s networks.

## Architecture

Extend the existing rawlink bridge to handle `frame_type::RETICULUM` (0x00) frames. Two mpsc channels connect the bridge to the event loop: one for inbound packets (L2 → Node) and one for outbound (Node → L2). The Reticulum Node state machine is unchanged — it receives `InboundPacket` events and emits `SendOnInterface` actions regardless of whether the transport is UDP, QUIC tunnel, or raw Ethernet.

## Frame Format

Uses the shared Harmony EtherType 0x88B5 with frame type discriminator 0x00:

```
[6 dst_mac][6 src_mac][2 EtherType=0x88B5][1 frame_type=0x00][≤500 bytes raw Reticulum packet]
```

Total overhead: 15 bytes. Reticulum's 500-byte MTU fits well within the 1485-byte Ethernet payload limit. No additional headers — the raw Reticulum packet follows the type tag directly.

All outbound Reticulum frames are broadcast (`ff:ff:ff:ff:ff:ff`), matching Reticulum's shared-medium broadcast semantics. The Node's `PacketHashlist` deduplicates packets received from multiple paths.

## Bridge Changes (harmony-rawlink)

### New BridgeConfig fields

```rust
pub reticulum_inbound_tx: Option<mpsc::Sender<Vec<u8>>>,
```

### New Bridge field

```rust
reticulum_outbound_rx: Option<mpsc::Receiver<Vec<u8>>>,
```

### Inbound handling

New arm in the `recv_frames` match block:
```rust
frame_type::RETICULUM => {
    if payload.len() > 1 {
        let packet = payload[1..].to_vec();  // strip type tag
        if let Some(ref tx) = reticulum_tx {
            let _ = tx.try_send(packet);
        }
    }
}
```

### Outbound handling

In the bridge loop, drain the outbound channel each iteration:
```rust
while let Ok(packet) = reticulum_outbound_rx.try_recv() {
    let mut frame_payload = Vec::with_capacity(1 + packet.len());
    frame_payload.push(frame_type::RETICULUM);
    frame_payload.extend_from_slice(&packet);
    self.socket.send_frame(BROADCAST_MAC, &frame_payload)?;
}
```

### No echo prevention needed

The Node's `PacketHashlist` already deduplicates Reticulum packets by content hash. A packet received from both L2 and UDP is processed once and dropped on the second arrival. No bridge-level dedup required.

## Event Loop Changes (harmony-node)

### Channel setup at startup

```rust
let (ret_inbound_tx, mut ret_inbound_rx) = mpsc::channel(256);
let (ret_outbound_tx, ret_outbound_rx) = mpsc::channel(256);
```

Pass `ret_inbound_tx` to `BridgeConfig::reticulum_inbound_tx`. Pass `ret_outbound_rx` to the bridge. Store `ret_outbound_tx` for the dispatch path.

### Interface registration

Register the L2 interface with the runtime so the Node routes packets to it:
```rust
runtime.register_interface("l2:mesh0", InterfaceMode::Full, ...);
```

The interface name uses the `l2:` prefix, consistent with `tunnel-` for QUIC tunnels and `udp0` for UDP broadcast.

### Select loop — inbound

New arm to drain the Reticulum inbound channel:
```rust
Some(packet) = ret_inbound_rx.recv() => {
    runtime.push_event(RuntimeEvent::InboundPacket {
        interface_name: "l2:mesh0".to_string(),
        raw: packet,
        now: now_ms(),
    });
}
```

### Dispatch — outbound

Extend `SendOnInterface` handler with `l2:` prefix matching:
```rust
if interface_name.starts_with("l2:") {
    if let Some(ref tx) = ret_outbound_tx {
        let _ = tx.try_send(raw);
    }
} else if interface_name.starts_with("tunnel-") {
    // existing tunnel path
} else {
    // existing UDP broadcast path
}
```

The `l2:` check goes before `tunnel-` since prefix matching is order-dependent.

## Interface Naming

- `"l2:{interface}"` — e.g., `"l2:mesh0"`, `"l2:wlan0"`
- Derived from the `rawlink_interface` config value
- Prefix-based routing in the dispatch logic, consistent with `"tunnel-"` pattern

## What Does NOT Change

- **harmony-reticulum crate** — Node state machine, packet format, routing, announces, IFAC, path tables. Completely unchanged.
- **UDP transport** — Continues to work alongside L2. A node can have both `udp0` and `l2:mesh0` active simultaneously.
- **Tunnel transport** — QUIC tunnels via iroh are unaffected.
- **Packet deduplication** — The Node's `PacketHashlist` handles packets arriving from multiple interfaces.

## Testing Strategy

### Unit tests (harmony-rawlink, MockSocket)

- Reticulum frame inbound round-trip: encode → MockSocket → bridge → inbound channel
- Outbound encoding: outbound channel → bridge → MockSocket with correct frame type
- Interleaved frame types: Scout + Data + Reticulum in sequence, verify correct routing

### Integration tests (harmony-node)

- Interface registration with `"l2:mesh0"` name
- `SendOnInterface` dispatch routes `l2:` prefix to outbound channel
- Inbound channel pushes `InboundPacket` with correct interface name

### No harmony-reticulum test changes

The Node is transport-agnostic. Existing tests cover packet processing regardless of interface name.

## Scope Exclusions

- **Unicast Reticulum packets** — all outbound is broadcast (Reticulum's default)
- **IFAC (Interface Access Codes)** — can be configured on the registered interface but not required for this first pass
- **Multiple L2 interfaces** — one `rawlink_interface` per node for now
- **Performance tuning** — channel sizes, batch processing, etc. are future work

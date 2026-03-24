# Zenoh AF_PACKET Transport (harmony-rawlink) Design

## Goal

Bypass the Linux IP stack entirely for Zenoh traffic on 802.11s mesh networks by bridging AF_PACKET raw Ethernet sockets to the zenoh session via shared memory. Eliminates 42-62 bytes of IP/UDP overhead per frame and avoids Netfilter, conntrack, and routing table processing.

## Architecture

A new `harmony-rawlink` crate implements:
- **TPACKET_V3 ring buffers** for zero-copy NIC→userspace frame ingestion
- **Zenoh SHM** for zero-copy bridge→zenoh-consumers delivery
- **L2 scouting** via broadcast Ethernet frames for peer discovery without IP
- **Application-level bridge** between the raw socket and the zenoh session (no zenoh internal transport APIs)

The bridge runs as an independent tokio task, communicating with the rest of harmony purely through zenoh pub/sub. No changes to the harmony-zenoh crate.

## Platform Abstraction

All bridge logic operates through a `RawSocket` trait:

```rust
pub trait RawSocket: Send {
    fn send_frame(&mut self, dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError>;
    fn recv_frames(&mut self, callback: &mut dyn FnMut(&[u8; 6], &[u8])) -> Result<(), RawLinkError>;
    fn local_mac(&self) -> [u8; 6];
}
```

Two implementations:
- **`AfPacketSocket`** (`#[cfg(target_os = "linux")]`) — Real AF_PACKET with TPACKET_V3 ring buffers.
- **`MockSocket`** (`#[cfg(test)]`) — Channel-backed mock for cross-platform testing.

The bridge, frame encoding, scouting, and SHM integration are platform-independent.

## Ethernet Frame Format

EtherType **0x88B5** (IEEE 802.1 Local Experimental), consistent with harmony-os's virtio-net driver.

```
[6 dst_mac][6 src_mac][2 EtherType=0x88B5][1 frame_type][payload...]
```

### Frame Types

The first byte after the EtherType is a shared discriminator across all Harmony L2 protocols. This allows rawlink, harmony-os, and future L2 transports to coexist on the same 802.11s mesh segment without misinterpreting each other's frames.

| Tag | Name | Addressing | Purpose |
|-----|------|-----------|---------|
| `0x00` | Reticulum | Per Reticulum routing | Raw Reticulum packet (harmony-os, harmony-kuw) |
| `0x01` | Scout | Broadcast (`ff:ff:ff:ff:ff:ff`) | Announces presence, carries identity_hash |
| `0x02` | Data | Broadcast (see note) | Carries zenoh key expression + payload |

**Note on Data addressing:** All outbound Data frames are broadcast in this first implementation. Unicast optimization requires application-level destination addressing (e.g., identity_hash embedded in the zenoh payload or key expression) which is future work. Scouting still populates the peer table for future unicast use.

### Scout Payload

```
[16 bytes identity_hash][variable: postcard-serialized zenoh locator info]
```

Broadcast every ~5 seconds (with jitter). On receipt, the bridge learns `identity_hash → MAC address` for subsequent unicast.

### Data Payload

```
[LEB128 key_expr_len][key_expr bytes][payload bytes]
```

Sent unicast to the learned MAC. Falls back to broadcast if destination MAC is unknown.

### MTU

802.11s standard 1500-byte Ethernet MTU. With 15-byte overhead (14 Ethernet header + 1 frame type tag), 1485 bytes of usable payload. Zenoh handles fragmentation above this layer.

## TPACKET_V3 Ring Buffers

Linux-only, inside `AfPacketSocket`.

### Setup

1. `socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))`
2. Bind to named interface via `sockaddr_ll`
3. Set `PACKET_VERSION` to `TPACKET_V3`
4. Configure `PACKET_RX_RING`: 1MB block size, 4 blocks (4MB total), 2048-byte frame slots
5. Configure `PACKET_TX_RING`: same parameters
6. `mmap()` both rings into userspace
7. Attach BPF filter for EtherType 0x88B5

### BPF Filter

Three instructions — only accept Harmony frames:
```
ldh [12]              ; load EtherType at offset 12
jeq #0x88B5, accept   ; match → accept
ret #0                ; no match → reject
accept: ret #65535
```

### Receive Path

1. `poll()` on socket fd with timeout
2. Walk current RX block's linked list of `tpacket3_hdr` entries
3. Extract src MAC + payload, invoke callback
4. Mark block as `TP_STATUS_KERNEL` (return to kernel)

### Send Path

1. Find next TX block slot (`TP_STATUS_AVAILABLE`)
2. Write complete Ethernet frame into slot
3. Mark as `TP_STATUS_SEND_REQUEST`
4. `sendto()` to flush

## Bridge and Zenoh SHM Integration

The bridge runs as a tokio task, connecting `RawSocket` to the zenoh session.

### Inbound (L2 → zenoh)

1. `recv_frames` delivers `(src_mac, payload)` from ring buffer
2. Parse frame type:
   - **Scout (0x01):** Update peer table (`identity_hash → MAC`). No zenoh publication.
   - **Data (0x02):** Decode key expression + payload. Allocate zenoh SHM buffer, memcpy payload into it, publish to zenoh.
3. SHM buffer is zero-copy to all zenoh subscribers in the same process.

### Outbound (zenoh → L2)

1. Bridge subscribes to configurable key expressions (e.g., `harmony/**`)
2. On zenoh sample: broadcast the frame (all outbound Data frames are broadcast in this version — unicast optimization is future work)
3. Encode as Data frame (0x02), send via `send_frame` with destination `ff:ff:ff:ff:ff:ff`

### Peer Table

```rust
struct PeerTable {
    by_identity: HashMap<[u8; 16], PeerEntry>,
}
struct PeerEntry {
    mac: [u8; 6],
    last_seen: Instant,
}
```

Populated from Scout frames. The bridge sends its own Scout every 5-10 seconds (uniform jitter). Entries expire after 30 seconds (3× the 10s max scout interval).

### SHM Configuration

`ShmProvider` with configurable pool size (default 8MB — ~5000 in-flight 1500-byte frames). Allocation failure falls back to regular `ZBytes` with memcpy rather than dropping frames.

## Event Loop Integration

### Configuration

Add to `ConfigFile`:
```rust
pub rawlink_interface: Option<String>,  // e.g., "mesh0"
```

If `None`, no AF_PACKET bridge starts. If `Some`, bind to the interface at startup.

### Startup

After zenoh session creation:
```rust
if let Some(ref iface) = rawlink_interface {
    let socket = harmony_rawlink::AfPacketSocket::new(iface)?;
    let bridge = harmony_rawlink::Bridge::new(socket, session.clone())?;
    tokio::spawn(bridge.run());
}
```

### Capability Check

`AfPacketSocket::new()` returns a clear error if `CAP_NET_RAW` is missing: `"AF_PACKET requires CAP_NET_RAW capability"`. The node continues without the bridge — it still works over IP.

## Testing Strategy

### Unit Tests (all platforms, via MockSocket)

- Frame encoding/decoding round-trips (Scout and Data)
- Peer table: Scout updates, expiry, unknown-peer broadcast fallback
- Bridge inbound: inject Scout → peer table populated; inject Data → zenoh publish called
- Bridge outbound: zenoh sample → Data frame emitted with correct MAC
- BPF filter byte generation

### Linux Integration Tests (`#[cfg(target_os = "linux")]`, feature `rawlink-integration`)

- AfPacketSocket creation on loopback (requires `CAP_NET_RAW`)
- Ring buffer mmap setup
- Send/recv round-trip on loopback with BPF filter

### No zenoh SHM tests

SHM allocation tested indirectly via bridge logic. We test our usage, not zenoh internals.

## Scope Exclusions

- **Custom multicast MACs** — broadcast scouting is sufficient for 802.11s. Multicast optimization is future work.
- **Zenoh transport plugin (deep integration)** — application-level bridge avoids coupling to zenoh internals.
- **Frame fragmentation at L2** — zenoh handles fragmentation above the bridge.
- **Multiple interfaces** — one `rawlink_interface` per node. Multi-interface is future work.
- **Encryption at L2** — frames are unencrypted on the wire. Zenoh session encryption and tunnel encryption handle confidentiality at higher layers.

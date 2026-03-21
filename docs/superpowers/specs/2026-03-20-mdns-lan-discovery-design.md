# mDNS/DNS-SD LAN Peer Discovery for harmony-node

**Bead:** harmony-k4g
**Date:** 2026-03-20
**Status:** Draft

## Problem

harmony-node currently sends all outbound Reticulum packets to `255.255.255.255:4242`
(LAN broadcast). There is no peer discovery mechanism — peers are only learned reactively
when they broadcast Zenoh filters. This wastes airtime on 802.11s mesh networks (broadcast
frames transmit at the lowest basic rate) and provides no visibility into which nodes are
actually present on the LAN.

## Solution

Add mDNS (RFC 6762) / DNS-SD (RFC 6763) service discovery using the `mdns-sd` crate.
Each node advertises `_harmony._udp.local.` with its Reticulum address and port. Nodes
browse for the same service type, building a local peer table. Outbound packets are
always broadcast (preserving compatibility with non-mDNS nodes), and additionally
unicast to each known peer for reliable delivery. Reticulum's PacketHashlist
deduplicates the double-delivery at the receiver.

## Design Decisions

### Crate choice: mdns-sd

- Pure Rust, no system library bindings (musl-friendly)
- Both advertise and browse in a single crate
- Internal threading model with flume channel for async bridging
- Minimal dependencies (6-7 crates, no tokio requirement)
- Actively maintained (0.18.x, March 2026)

### Unicast fan-out (not per-peer Reticulum interfaces)

The Node state machine continues to see a single `"udp0"` interface in `Full` mode.
The event loop maintains a `PeerTable` internally and fans out outbound packets to all
known peer addresses via unicast UDP. This preserves the sans-I/O boundary — the Node
doesn't know or care about individual LAN peers. They're all one hop away; per-peer
routing decisions aren't useful at this layer.

Broadcast is always sent (preserving compatibility with non-mDNS peers, e.g., stock
Python Reticulum nodes). Additionally, unicast is sent to each known mDNS peer. This
means mDNS-aware peers receive the packet twice — once via broadcast, once via unicast.
Reticulum's `PacketHashlist` deduplicates at the receiver, so the double-delivery has
no protocol-level effect. The unicast path ensures reliable delivery even when broadcast
is unreliable (common on WiFi with rate-limited multicast).

### Passive liveness (no active probing)

Peers are marked as "seen" whenever any UDP packet arrives from their address. If no
packet is received within a configurable timeout (default 60 seconds), the peer is
evicted from the table. This works because Harmony nodes produce constant traffic:
Reticulum announces, Zenoh filter broadcasts (every 7.5s), and normal data packets.
A 60-second silence genuinely means the peer is gone.

Zero extra traffic overhead compared to active ping-based liveness.

### No Avahi dependency

`mdns-sd` binds its own multicast socket with `SO_REUSEADDR`/`SO_REUSEPORT`. It
coexists with Avahi or any other mDNS responder. The OpenWRT package does not require
`avahi-daemon` or `umdns`.

## Architecture

### New file: `discovery.rs`

```rust
pub struct PeerTable {
    peers: HashMap<SocketAddr, PeerInfo>,
    // Reverse index: Reticulum address → set of SocketAddrs.
    // A single mDNS peer may resolve to multiple addresses (IPv4 + IPv6,
    // or multiple interfaces). ServiceRemoved provides a fullname (which
    // encodes the Reticulum address), not a SocketAddr, so this index is
    // needed for removal.
    addr_to_sockets: HashMap<[u8; 16], HashSet<SocketAddr>>,
    our_addr: [u8; 16],      // for self-discovery filtering
    stale_timeout: Duration,  // default 60s
}

pub struct PeerInfo {
    pub reticulum_addr: [u8; 16],  // from TXT record
    pub proto_version: u8,          // from TXT record
    pub last_seen: Instant,         // updated on ANY inbound UDP
    pub discovered_at: Instant,
}
```

**Construction:** `PeerTable::new(our_addr: [u8; 16], stale_timeout: Duration)`. The
`our_addr` bytes come directly from `identity.ed25519.public_identity().address_hash`
in `main.rs`, passed as a new parameter to `event_loop::run()`. This avoids depending
on the hex string `node_addr`, which is moved into `NodeConfig`.

**Methods:**

- `add_peer(addr, reticulum_addr, proto_version)` — insert or update in both `peers`
  and `addr_to_sockets`. Skips if `reticulum_addr == our_addr` (self-discovery
  filtering). Idempotent. A single `ServiceResolved` event may yield multiple
  addresses (IPv4 + IPv6); the caller iterates `get_addresses()` and calls
  `add_peer()` for each.
- `remove_by_reticulum_addr(reticulum_addr)` — remove all `SocketAddr` entries for
  this Reticulum address. Used on `ServiceRemoved`, which provides the fullname
  (containing the hex address) but not a `SocketAddr`. Returns removed count for
  logging.
- `mark_seen(src_addr)` — update `last_seen`. Called on every inbound UDP packet.
  No-op if `src_addr` is not in the table (unknown sender, e.g. broadcast from
  undiscovered node).
- `evict_stale() -> Vec<SocketAddr>` — remove peers not seen within `stale_timeout`.
  Also cleans up `addr_to_sockets`. Returns evicted addresses for logging.
- `peer_addrs() -> impl Iterator<Item = &SocketAddr>` — for unicast fan-out.

**mDNS setup function:**

```rust
pub fn start_mdns(
    listen_port: u16,
    reticulum_addr: &[u8; 16],
) -> Result<(ServiceDaemon, Receiver<ServiceEvent>)>
```

- Registers `_harmony._udp.local.` service
- Instance name: full 32 hex chars of Reticulum address (fits DNS-SD's 63-byte label limit)
- TXT records: `addr={full 32-char hex}`, `proto=1`
- Browses for `_harmony._udp.local.`
- Returns daemon handle + flume Receiver

### Event loop changes (`event_loop.rs`)

**5th select! arm** for mDNS events:

```rust
Ok(event) = mdns_rx.recv_async() => {
    match event {
        ServiceEvent::ServiceResolved(info) => {
            // Parse TXT records for addr + proto
            let reticulum_addr = parse_txt_addr(info.get_properties());
            let proto = parse_txt_proto(info.get_properties());
            // ServiceResolved may return multiple addresses (IPv4 + IPv6)
            for ip in info.get_addresses() {
                let socket_addr = SocketAddr::new(ip.to_ip_addr(), info.get_port());
                peer_table.add_peer(socket_addr, reticulum_addr, proto);
            }
        }
        ServiceEvent::ServiceRemoved(_service_type, fullname) => {
            // fullname encodes the instance name which contains our hex addr prefix.
            // Parse the Reticulum address from the instance name and remove by it.
            if let Some(reticulum_addr) = parse_instance_addr(&fullname) {
                peer_table.remove_by_reticulum_addr(&reticulum_addr);
            }
        }
        _ => {} // ServiceFound (triggers resolve), SearchStarted, SearchStopped
    }
}
```

**Inbound UDP** (Arm 1): After `recv_from()`, call `peer_table.mark_seen(src_addr)`.

**Outbound UDP** (`SendOnInterface` in `dispatch_action`):

The `dispatch_action` function receives `&PeerTable` as an additional parameter
(passed as `Option<&PeerTable>` — `None` when `--no-mdns` is set or during startup
action dispatch before mDNS is initialized).

```rust
// Always broadcast (preserves compatibility with non-mDNS nodes)
if let Err(e) = udp.send_to(&raw, broadcast_addr).await {
    tracing::warn!("broadcast send failed: {e}");
}
// Additionally unicast to each known mDNS peer
if let Some(peers) = peer_table {
    for addr in peers.peer_addrs() {
        if let Err(e) = udp.send_to(&raw, addr).await {
            tracing::warn!(peer = %addr, "unicast send failed: {e}");
        }
    }
}
```

Note: errors are logged and continued, not propagated with `?`. A transient send
failure to one peer must not crash the event loop.

**Timer tick** (Arm 2): After pushing `TimerTick`, call `peer_table.evict_stale()`.
Log evicted peers at `info` level.

**Shutdown** (Arm 4): Call `mdns_daemon.shutdown()` to send mDNS goodbye.

### CLI flags (`main.rs`)

| Flag | Default | Description |
|------|---------|-------------|
| `--no-mdns` | false | Disable mDNS discovery (broadcast-only mode) |
| `--mdns-stale-timeout` | 60 | Seconds before evicting a silent peer |

When `--no-mdns` is set, skip `start_mdns()`, don't add the 5th select! arm, and use
broadcast-only send path. The node works exactly as it does today.

### UCI config (harmony-openwrt)

Two new options in `harmony-node.init` and `harmony-node.conf`:

```
option no_mdns '0'
option mdns_stale_timeout '60'
```

## Edge Cases

**mDNS bind failure:** Log warning, proceed without discovery. Falls back to
broadcast-only — identical to current behavior. No crash.

**Self-discovery:** Filtered by comparing discovered `addr` TXT field against our own
Reticulum address in `PeerTable::add_peer()`.

**Duplicate discoveries:** `mdns-sd` may emit multiple events for the same peer (refresh,
multiple interfaces). `add_peer()` is idempotent — HashMap upsert.

**Multiple addresses per peer:** `ServiceResolved` may return multiple IPs (IPv4 + IPv6,
multiple interfaces). Each becomes a separate `PeerTable` entry keyed by `SocketAddr`,
linked via `addr_to_sockets` reverse index. `remove_by_reticulum_addr()` removes all of
them atomically.

**Proto version mismatch:** Log at `warn`, still add peer. Reticulum's packet validation
rejects incompatible packets at the protocol layer.

**Multiple network interfaces:** `mdns-sd` discovers on all interfaces automatically.
The UDP socket is bound to `0.0.0.0`, so unicast replies work to any source address.

**Non-mDNS peers (mixed networks):** Broadcast is always sent regardless of peer table
state. A stock Python Reticulum node (no mDNS) continues to receive packets via
broadcast. mDNS-aware peers receive packets twice (broadcast + unicast); Reticulum's
`PacketHashlist` deduplicates at the receiver.

**ServiceRemoved unreliability:** mDNS goodbye packets are best-effort UDP multicast.
If a peer crashes without sending a goodbye, `ServiceRemoved` never fires. Passive
liveness eviction (60s timeout) is the **authoritative** removal mechanism.
`ServiceRemoved` is an optimization for faster removal when the goodbye is received —
not the primary removal path.

**tokio current_thread + mdns-sd thread:** `mdns-sd` spawns its own OS thread and
communicates via flume channels. `flume::Receiver::recv_async()` is runtime-agnostic
and works correctly on tokio's `current_thread` flavor. The `Receiver` is `Send + Sync`,
so holding it in the select! loop is safe despite `NodeRuntime` being `!Send`.

## Testing

**Unit tests** for `PeerTable`:
- add/remove/mark_seen/evict_stale lifecycle
- self-filtering (add_peer with our own address is no-op)
- idempotent add_peer (duplicate insert doesn't panic or duplicate)
- evict_stale respects timeout boundary

**Integration test**: Start two `mdns-sd` daemons in-process, verify mutual discovery
via the flume channel. No network required (localhost multicast).

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/Cargo.toml` | Add `mdns-sd = { version = "0.18", features = ["async"] }` |
| `crates/harmony-node/src/discovery.rs` | New: PeerTable, start_mdns, TXT parsing |
| `crates/harmony-node/src/event_loop.rs` | 5th select! arm, unicast fan-out, mark_seen, stale eviction, shutdown |
| `crates/harmony-node/src/main.rs` | `--no-mdns`, `--mdns-stale-timeout` flags, conditional startup |
| `crates/harmony-node/src/main.rs` | Add `mod discovery;` |
| `harmony-openwrt: harmony-node.init` | Two new UCI options |
| `harmony-openwrt: harmony-node.conf` | Defaults for new options |

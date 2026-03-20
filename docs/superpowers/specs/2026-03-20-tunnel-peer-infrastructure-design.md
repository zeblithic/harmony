# Tunnel Peer Infrastructure Design

**Date:** 2026-03-20
**Status:** Draft
**Scope:** `harmony-tunnel` crate, `harmony-node` iroh-net integration, `iroh.q8.fyi` relay, `harmony-peers` tunnel lifecycle, `harmony-discovery` tunnel routing hints

## Overview

Harmony's network has two kinds of peers: **adjacent peers** (reachable via LAN broadcast, serial, LoRa — physical next-hop) and **tunnel peers** (reachable via IP tunnels across the internet — social-graph peers). This design establishes infrastructure for discovering, connecting, and maintaining high-efficiency direct tunnels between tunnel peers.

The design follows Harmony's sans-I/O paradigm: all protocol logic lives in pure state machines. The async event loop in `harmony-node` bridges real I/O (iroh-net, UDP, Zenoh) to the state machines.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Relay model | Iroh-inspired (DERP-like) | Battle-tested, automatic upgrade to direct, relay is a dumb pipe |
| Transport | iroh-net (wraps Quinn/QUIC) | Handles hole-punching, relay, connection migration; already in dep tree |
| Crypto | PQ-first (ML-KEM-768 + ML-DSA-65) | Ed25519 is legacy/Reticulum-compat only; new infra is PQ-native from day 0 |
| Peer treatment | Dual-path (Reticulum control + Zenoh data) | 500B MTU fine for control plane; data plane needs native QUIC throughput |
| Relay identity | BLAKE3(ML-DSA-65 pub)[:32] as NodeId | Stock iroh-relay, no fork needed; relay routes by opaque key hash |
| Handshake | ML-KEM encaps + ML-DSA mutual auth → HKDF-BLAKE3 → ChaCha20-Poly1305 | Simple, PQ-native, no legacy compat needed for new protocol |
| Relay access | Open baseline + reputation bonus | Anyone can bootstrap; trust scores unlock higher limits |
| Relay domain | `iroh.q8.fyi` | User-owned domain, dedicated to Harmony IP-bridge infrastructure |

## Section 1: `harmony-tunnel` Crate — PQ Tunnel State Machine

### Purpose

Sans-I/O state machine managing authenticated, encrypted tunnels between remote peers. No network I/O — the caller feeds bytes in, gets bytes out.

### PQ Handshake (2 messages)

```
Initiator                              Responder
    |                                      |
    |-- TunnelInit ----------------------->|
    |   ML-KEM-768 encaps ciphertext       |
    |   ML-DSA-65 public key               |
    |   ML-DSA-65 signature over above     |
    |   Nonce (32 bytes)                   |
    |                                      |
    |<-------------- TunnelAccept ---------|
    |   ML-DSA-65 public key               |
    |   ML-DSA-65 signature over           |
    |     (shared_secret || init_nonce)     |
    |   Nonce (32 bytes)                   |
    |                                      |
    |== Session established ===============|
    |   Key = HKDF-BLAKE3(shared_secret,   |
    |         init_nonce || accept_nonce)   |
    |   Cipher = ChaCha20-Poly1305         |
```

The handshake is purpose-built for tunnel establishment. It does not replace the Reticulum link handshake or Zenoh session handshake — those run on top of the established tunnel for their respective protocol layers.

### Dual-Path Multiplexing

After handshake, the tunnel carries two frame types over the same encrypted session:

- **Reticulum frames** (tag `0x01`): Raw Reticulum packets (up to 500 bytes). Fed into the `Node` state machine as `InboundPacket` events on a virtual "tunnel interface."
- **Zenoh frames** (tag `0x02`): Zenoh protocol messages (arbitrary size, QUIC handles fragmentation). Fed into the `Session` state machine.

**Frame format:**
```
[1 byte tag] [2 bytes length (big-endian)] [payload]
     ↑              ↑                          ↑
  0x01/0x02    up to 65535              encrypted content
```

All frames are encrypted with the session's ChaCha20-Poly1305 key using an incrementing nonce counter.

### State Machine

```
Idle → Initiating → HandshakeWait → Active → Closing → Closed
                                       ↑         |
                                       └─────────┘ (reconnect)
```

**Events consumed:**
- `InboundBytes(Vec<u8>)` — raw bytes from iroh-net connection
- `SendReticulum(packet)` — Reticulum packet to send through tunnel
- `SendZenoh(message)` — Zenoh message to send through tunnel
- `Tick` — periodic timer for keepalive
- `Close` — graceful shutdown

**Actions emitted:**
- `OutboundBytes(Vec<u8>)` — encrypted bytes to write to iroh-net connection
- `ReticulumReceived(packet)` — decrypted Reticulum packet for Node state machine
- `ZenohReceived(message)` — decrypted Zenoh message for Session state machine
- `HandshakeComplete(peer_identity)` — PQ identity of authenticated peer
- `Error(reason)` — handshake failure, decryption error, protocol violation
- `Closed` — tunnel torn down

### Security Properties

- **Forward secrecy:** ML-KEM encapsulation generates a fresh shared secret per session. Compromise of long-term ML-DSA signing keys does not reveal past session keys.
- **Mutual authentication:** Both peers prove possession of their ML-DSA-65 private key. The responder signs over the shared secret to prove it decapsulated correctly.
- **Replay protection:** Nonces from both sides are mixed into the session key derivation. ChaCha20-Poly1305 nonce counter prevents replayed frames.
- **Zeroization:** All ephemeral key material (ML-KEM shared secret, session keys) is zeroized on drop via the `zeroize` crate.

## Section 2: iroh-net Integration in `harmony-node`

### Purpose

Wire iroh-net into the existing tokio `select!` event loop, mapping iroh-net connections to `harmony-tunnel` state machine instances.

### Event Loop Extension

The current event loop has three `select!` arms (UDP, timer, Zenoh bridge). A fourth arm is added:

```
select! {
    // existing:
    udp_recv => RuntimeEvent::InboundPacket,
    timer    => RuntimeEvent::TimerTick,
    zenoh    => RuntimeEvent::ZenohBridge(...),
    // new:
    iroh     => RuntimeEvent::TunnelEvent(peer_id, bytes),
}
```

### iroh-net Endpoint Setup

At node startup (if tunnel support is configured):

- Create an `iroh::Endpoint` with:
  - `SecretKey` derived from PQ identity: `BLAKE3(ml_dsa_65_private_key)[:32]` — used **only** for iroh-net internal addressing, **not** for Harmony authentication
  - Relay URL: `https://iroh.q8.fyi` (configurable via CLI `--relay-url` or TOML config)
  - ALPN: `b"harmony-tunnel/1"`
- If no relay URL and no tunnel contacts are configured, iroh-net is not started (zero overhead)

### Connection Management

**Outbound:** When `PeerManager` emits `InitiateTunnel(node_id, relay_url, direct_addrs)`:
1. Event loop calls `endpoint.connect(node_id, ALPN)`
2. Spawns a task that drives the `TunnelSession` initiator-side handshake
3. Bidirectional `mpsc` channel connects the task back to the main event loop

**Inbound:** `endpoint.accept()` loop:
1. Spawns a task per incoming connection
2. Creates `TunnelSession` in responder mode
3. Runs PQ handshake, then channels back to main event loop

### Interface Registration

On `HandshakeComplete(peer_identity)`, the event loop:

1. Registers a Reticulum `Interface` with the `Node` state machine:
   - Mode: `PointToPoint`
   - Name: `tunnel-{short_node_id}` (first 8 hex chars of NodeId)
   - MTU: 500 (Reticulum standard, applied to the Reticulum frame path only)
2. Opens a Zenoh `Session` with the remote peer using the established tunnel's Zenoh frame path
3. Both layers now treat this tunnel peer identically to a LAN peer

### Graceful Degradation

- iroh-net not configured → not started, zero overhead
- Relay unreachable → iroh-net still attempts direct connections via known addresses
- Tunnel drops → PeerManager handles reconnection with exponential backoff
- iroh-net reports direct address upgrade → connection silently migrates off relay

## Section 3: Relay Server at `iroh.q8.fyi`

### Purpose

Public rendezvous/relay for Harmony peers behind different NATs. Runs stock `iroh-relay` with rate limiting.

### Deployment

- **Binary:** Stock `iroh-relay` — no Harmony-specific server code
- **TLS:** Let's Encrypt via the `iroh-relay` built-in ACME support
- **Privacy:** Relay never sees plaintext (PQ encryption is end-to-end)
- **Hosting:** Minimal resource requirements (relay is CPU-cheap, bandwidth is the main cost)

### Rate Limiting (Hybrid Baseline + Reputation)

| Tier | Criteria | Concurrent sessions | Bandwidth | Conn attempts/hr |
|------|----------|--------------------:|----------:|------------------:|
| Baseline | Any NodeId | 3 | 100 KB/s | 30 |
| Trusted | Valid UCAN capability token | 10 | 1 MB/s | 120 |
| Reputation | Trust score ≥ threshold | 25 | 5 MB/s | 300 |

- Rate limiting via a lightweight proxy (nginx or Rust sidecar) in front of `iroh-relay`
- Proxy extracts NodeId from connection, validates optional UCAN bearer token, maps to tier
- Trust score lookups are best-effort/cached; unavailable → baseline tier
- Per-NodeId tracking prevents Sybil abuse at baseline tier

### Node-Side Configuration

- `--relay-url https://iroh.q8.fyi` (CLI flag or TOML config)
- `--relay-token <ucan>` (optional, for authenticated tier)
- Default: no relay (LAN-only). Explicit opt-in for WAN connectivity.

### Deployment Order

This is the last piece to deploy. All protocol work can be tested on a local subnet without any relay. Deploy `iroh.q8.fyi` when cross-NAT connectivity is needed.

## Section 4: Tunnel Peer Lifecycle in `harmony-peers`

### Purpose

Extend the PeerManager to handle tunnel peer connections alongside adjacent peers, with unified state transitions but transport-aware connection logic.

### Contact Store Additions (`harmony-contacts`)

`ContactAddress` enum gains a new variant:

```
ContactAddress::Tunnel {
    node_id: [u8; 32],              // BLAKE3(ML-DSA-65 pub key)
    relay_url: Option<Url>,          // preferred relay
    direct_addrs: Vec<SocketAddr>,   // known direct addresses
}
```

Existing `ContactAddress::Reticulum { destination_hash }` is unchanged. A contact can have **both** — tunnel for direct QUIC connectivity, Reticulum for mesh-routed fallback.

### PeerManager State Machine Extension

Existing states remain: `Searching → Connecting → Connected → Disabled`

Transport-aware behavior in the `Connecting` state:
- Tunnel contacts: emit `InitiateTunnel(node_id, relay_url, direct_addrs)` action
- Reticulum contacts: emit `InitiateLink(destination_hash)` action (existing)
- Dual-address contacts: try tunnel first (lower latency), fall back to Reticulum if tunnel fails

New events consumed:
- `TunnelEstablished(node_id, peer_identity)` → transition to Connected
- `TunnelFailed(node_id, reason)` → backoff and retry, or fall back to Reticulum
- `TunnelDropped(node_id)` → transition to Searching, begin reconnection

### Connection Quality Tracking

```
ConnectionQuality {
    rtt_ms: u32,
    transport: Transport,
    uptime: Duration,
}

enum Transport {
    Lan,
    Tunnel { relayed: bool },
}
```

- Tunnel peers report RTT from handshake and keepalive
- Adjacent peers report RTT from Reticulum link establishment
- Higher layers can query `connection_quality(peer_id)` for routing decisions

### Reconnection Policy

- Same exponential backoff as existing design (30s / 120s / 600s caps by priority)
- On tunnel drop, if peer has a Reticulum address: immediately attempt Reticulum link as fallback (no backoff for fallback path)
- On relay → direct upgrade: update `direct_addrs` in contact store for faster future reconnection

## Section 5: Tunnel Discovery — Relay Routing Hints

### Purpose

Extend `harmony-discovery` so peers can advertise their tunnel reachability via AnnounceRecords.

### RoutingHint Extension

Existing variants:
- `RoutingHint::ReticulumDestination(destination_hash)`
- `RoutingHint::ZenohLocator(locator)` (future)

New variant:
```
RoutingHint::Tunnel {
    node_id: [u8; 32],              // BLAKE3(ML-DSA-65 pub key)
    relay_url: Option<String>,       // e.g., "https://iroh.q8.fyi"
    direct_addrs: Vec<SocketAddr>,   // publicly reachable addrs
}
```

An AnnounceRecord can carry **multiple** routing hints — a peer might be reachable via both Reticulum and tunnel. Consumers pick the best path.

### Publishing Tunnel Hints

1. When iroh-net Endpoint starts, it discovers its NodeId and any direct addresses via STUN
2. DiscoveryManager emits `UpdateLocalRoutingHints(Vec<RoutingHint>)` action
3. Event loop feeds back the tunnel hint
4. Node includes the hint in periodic AnnounceRecords on `harmony/identity/{address_hex}/announce`
5. Direct addresses refresh when iroh-net detects address changes

### Consuming Tunnel Hints

1. Node receives AnnounceRecord with `RoutingHint::Tunnel`
2. DiscoveryManager emits `PeerDiscovered(peer_identity, routing_hints)`
3. PeerManager (or user via contacts) decides whether to initiate tunnel connection
4. Automatic peering: if peer is in contact store and tunnel hint arrives, PeerManager attempts connection

### Hint Freshness

- Tunnel hints carry timestamps from AnnounceRecord's `created_at`
- Direct addresses are ephemeral — stale threshold: 1 hour (configurable)
- Relay URL is stable — only changes if peer switches relay providers

### Privacy

- Publishing direct addresses reveals public IP — **opt-in** behavior
- Nodes can advertise relay-only hints (`direct_addrs: []`) for privacy
- Ties into `harmony-d63` (peering privacy) for future onion-style hint obfuscation

## Bead Decomposition

| # | Bead Title | Priority | Dependencies | Scope |
|---|-----------|----------|-------------|-------|
| 1 | `harmony-tunnel` crate: PQ tunnel state machine | P1 | None | Handshake, encryption, dual-path mux, state machine |
| 2 | iroh-net integration in `harmony-node` event loop | P1 | #1 | Endpoint setup, connection mgmt, interface registration |
| 3 | Tunnel peer lifecycle in `harmony-peers` | P2 | #1, #2 | ContactAddress::Tunnel, PeerManager extension, quality tracking |
| 4 | Tunnel routing hints in `harmony-discovery` | P2 | #1 | RoutingHint::Tunnel, publish/consume hints, freshness |
| 5 | Relay server deployment at `iroh.q8.fyi` | P3 | #2 | Stock iroh-relay, rate limiting proxy, UCAN validation |

## Implementation Order

1. **`harmony-tunnel` crate** — protocol core, fully testable without network
2. **iroh-net in `harmony-node`** — wire transport to state machine
3. **`harmony-peers` tunnel lifecycle** — peer management for tunnels
4. **`harmony-discovery` tunnel hints** — advertise/discover tunnel reachability
5. **`iroh.q8.fyi` relay** — deploy when cross-NAT needed (LAN works without it)

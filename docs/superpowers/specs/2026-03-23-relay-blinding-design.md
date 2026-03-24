# MASQUE-Style Relay Blinding Design

## Goal

Prevent the iroh relay operator from building a NodeId-to-NodeId social graph by replacing deterministic NodeIds with ephemeral identities at the transport layer, while preserving ML-DSA-based authentication at the application layer inside the encrypted QUIC stream.

## Architecture

Two identity layers, cleanly separated:

- **Transport identity (ephemeral):** A random `iroh::SecretKey` generated at startup (responder) or per-connection (initiator). This is what the relay sees. It has no cryptographic relationship to the node's real identity. The relay cannot correlate connections across restarts or link initiator to responder.
- **Application identity (permanent):** The existing ML-DSA-65 keypair, used for tunnel handshake signatures, frame AEAD AAD, and peer authentication. Exchanged inside the encrypted QUIC stream. The relay never sees it.

No relay modifications are required — the stock iroh-relay binary works as-is. The privacy improvement is entirely client-side.

## Ephemeral Identity Lifecycle

### Responder (long-lived, per-startup)

1. At node startup, generate `iroh::SecretKey::generate()` (random, not derived from ML-DSA identity).
2. Create `iroh::Endpoint` with this ephemeral key, configured with the relay map and `HARMONY_TUNNEL_ALPN`.
3. The Endpoint registers with the relay under its ephemeral NodeId.
4. Spawn an accept loop: `endpoint.accept().await` in a tokio task, running `tunnel_task::run_responder` for each incoming connection.
5. Emit `RuntimeEvent::LocalTunnelInfo` carrying the ephemeral NodeId so announce records can include it in `RoutingHint::Tunnel { node_id, relay_url, direct_addrs }`.
6. The Endpoint lives for the node's lifetime. Identity rotates on restart.

### Initiator (transient, per-dial)

1. When `RuntimeAction::InitiateTunnel` fires (after stochastic delay), generate a fresh `iroh::SecretKey::generate()`.
2. Create a transient `iroh::Endpoint` with this key and the same relay map.
3. Call `endpoint.connect(NodeAddr::new(target_ephemeral_node_id).with_relay_url(relay_url), HARMONY_TUNNEL_ALPN).await`.
4. Run `tunnel_task::run_initiator` over the resulting QUIC connection.
5. On success, spawn `TunnelBridge` and send `HandshakeComplete` through the channel.
6. Store the transient Endpoint alongside the bridge — it must stay alive as long as the QUIC connection is active (iroh uses it for relay forwarding and hole-punching).
7. On tunnel close, drop both the bridge and the Endpoint.

## Discovery Integration

The responder's ephemeral NodeId is advertised via the existing Zenoh announce mechanism. `RoutingHint::Tunnel { node_id, relay_url, direct_addrs }` carries the ephemeral NodeId instead of the deterministic BLAKE3 hash. The `identity_hash` in the `AnnounceRecord` remains the stable peer identifier — the ephemeral NodeId is purely a transport-layer address.

When a responder restarts, it generates a new ephemeral NodeId and re-announces. Peers holding the old NodeId will fail to connect and re-resolve via Zenoh discovery.

## Event Loop Changes

### Responder Endpoint Setup

Replace the current deterministic `SecretKey` derivation (event_loop.rs:262-300) with:
- `iroh::SecretKey::generate()` for a random ephemeral key
- Same Endpoint builder configuration (relay map, ALPN, bind)
- Spawn an accept loop task that processes incoming connections
- Emit `RuntimeEvent::LocalTunnelInfo` with the ephemeral NodeId

### Initiator Dial (currently stubbed)

Replace the TODO stub at event_loop.rs:841-847 with:
- Generate `iroh::SecretKey::generate()` per dial
- Build a transient `iroh::Endpoint`
- `endpoint.connect(target_addr, ALPN).await`
- Run `tunnel_task::run_initiator` on the QUIC connection
- On success, spawn bridge, store Endpoint alongside it
- Send `HandshakeComplete` event

### Logging

Remove the deterministic NodeId from `tracing::info!` at line 841-845. Log the ephemeral NodeId instead (or omit it, since it's meaningless for debugging identity).

### Tunnel Identity Map

`tunnel_identities: HashMap<String, [u8; 32]>` continues to map interface names to the permanent NodeId (BLAKE3 of ML-DSA pubkey), derived during the tunnel handshake. Unchanged.

## What Does NOT Change

- **harmony-tunnel crate:** The handshake (`TunnelInit`/`TunnelAccept`), frame encryption, session management, and keepalive logic are application-layer — they operate inside the QUIC stream and don't interact with iroh identities. No changes needed.
- **Frame AAD:** Continues to use the permanent NodeId (BLAKE3 of ML-DSA pubkey) exchanged during the handshake. Not the ephemeral transport NodeId.
- **Keepalive jitter seeding:** Uses the permanent NodeId. Unchanged.
- **tunnel_task.rs function signatures:** `run_initiator()` and `run_responder()` take an `iroh::Connection`. They don't care how the connection was established.

## Privacy Properties

| Observer | Before | After |
|----------|--------|-------|
| Relay operator | Sees deterministic NodeId-A → NodeId-B, builds social graph | Sees random ephemeral-X → ephemeral-Y, unlinkable across connections or restarts |
| Zenoh subscriber | Sees `identity_hash` + deterministic `node_id` in announces | Sees `identity_hash` + ephemeral `node_id` (rotates on restart) |
| LAN/ISP passive | Sees QUIC connection to relay IP | Same (no change at IP layer) |

The relay blinding specifically addresses the "relay operator" threat from the peering privacy design doc. Zenoh-layer discovery privacy is a separate concern (harmony-0kq).

## Testing Strategy

### Unit Tests (harmony-tunnel)

No changes needed — existing tests are identity-layer-agnostic.

### Integration Tests (harmony-node)

- **Ephemeral NodeId uniqueness:** Two `iroh::SecretKey::generate()` calls produce different NodeIds.
- **Responder uses ephemeral key:** Responder Endpoint's NodeId differs from BLAKE3(ML-DSA signing key).
- **Initiator uses fresh key per dial:** Each `DeferredDial` creates a new Endpoint with a unique random SecretKey.
- **Announce carries ephemeral NodeId:** `LocalTunnelInfo` event carries the ephemeral NodeId, not the deterministic one.

### No End-to-End Relay Test

QUIC connection establishment is iroh's responsibility. Our tests verify we provide the right ephemeral identities to iroh's API.

## Scope Exclusions

- **Periodic identity rotation** — restart is sufficient for now.
- **Zenoh discovery privacy** — separate bead (harmony-0kq).
- **Custom relay protocol** — stock iroh-relay works as-is.
- **Direct peer-to-peer connections** (hole-punching without relay) — ephemeral identities work the same way; iroh handles this transparently.

# Session State Machine (sans-I/O) Design

**Bead:** harmony-0p6.3
**Date:** 2026-03-02
**Status:** Approved

## Summary

Add a sans-I/O session state machine to harmony-zenoh that manages a single
peer-to-peer encrypted connection. The session handles identity-based handshake,
resource ID mapping (key expression ↔ ExprId), keepalive/stale detection, and
graceful close — all driven by caller-provided events, returning actions for the
caller to execute.

## Decision Record

- **Thin session coordinator** over full session stack: Session manages lifecycle
  and resource declarations only. It does not own SubscriptionTable or
  HarmonyEnvelope — the caller composes those. Single responsibility, testable
  independently.

- **Two-party sessions** over multi-peer hub: Each Session represents exactly one
  peer-to-peer connection, matching the E2EE envelope's sender/recipient model.
  Multi-peer coordination lives in a higher layer.

- **Identity-based handshake** over capability negotiation or immediate open:
  Both sides exchange Ed25519 proofs during Init, confirming they're talking to
  the expected peer. Reuses harmony-identity primitives, no new crypto.

- **Caller-provided timestamps** over std::time::Instant: Same pattern as
  harmony-reticulum's Node. Fully deterministic, testable without a real clock.

- **Graceful close message** over unilateral cleanup: Active side sends Close,
  peer responds with CloseAck. Timeout fallback for ungraceful disconnect via
  keepalive staleness.

- **Four lifecycle states** (Init, Active, Closing, Closed) over five: Open and
  Active collapsed — the handshake proof exchange is sufficient to consider the
  session active. Keepalive timeout handles idle cleanup.

## Lifecycle State Machine

```
Init ──(handshake verified + handshake sent)──► Active
Active ──(close initiated)────────────────────► Closing
Closing ──(CloseAck received or timeout)──────► Closed
Any state ──(keepalive timeout)───────────────► Closed
```

- **Init**: Awaiting mutual identity handshake. Emits `SendHandshake` on
  creation. Transitions to Active when both sides have sent and verified proofs.
- **Active**: Authenticated, resources can be declared/undeclared, keepalive
  timer running.
- **Closing**: Close message sent, waiting for ack. Timeout forces Closed.
- **Closed**: Terminal. No further events accepted.

## Core Types

```rust
pub type ExprId = u64;

pub struct Session { ... }

pub enum SessionState { Init, Active, Closing, Closed }

pub struct SessionConfig {
    pub keepalive_interval_ms: u64,  // default: 30_000
    pub stale_timeout_ms: u64,       // default: 90_000
    pub close_timeout_ms: u64,       // default: 5_000
}

pub enum SessionEvent {
    HandshakeReceived { proof: Vec<u8> },
    ResourceDeclared { expr_id: ExprId, key_expr: String },
    ResourceUndeclared { expr_id: ExprId },
    KeepaliveReceived,
    CloseReceived,
    CloseAckReceived,
    TimerTick { now_ms: u64 },
}

pub enum SessionAction {
    SendHandshake { proof: Vec<u8> },
    SendKeepalive,
    SendClose,
    SendCloseAck,
    ResourceAdded { expr_id: ExprId, key_expr: String },
    ResourceRemoved { expr_id: ExprId },
    PeerStale,
    SessionOpened,
    SessionClosed,
}
```

Entry point: `session.handle_event(event) -> Result<Vec<SessionAction>, ZenohError>`

## Handshake Protocol

1. Session created with `(local: &PrivateIdentity, remote: &Identity, now_ms)`.
   Starts in Init, emits `SendHandshake`.
2. Handshake payload: `sign(local_private_key, "harmony-session-v1" || remote_address_hash)`.
   Signing the remote's address prevents replay to a different peer.
3. On `HandshakeReceived { proof }`:
   `verify(remote_public_key, "harmony-session-v1" || local_address_hash, proof)`.
4. Session tracks `handshake_sent` and `handshake_verified`. Transitions to
   Active only when both are true. Emits `SessionOpened`.
5. Verification failure → Closed, emits `SessionClosed`.

## Resource Declaration Table

Two independent maps, one per direction:

- **Local resources** (`HashMap<ExprId, String>`): Declared by this side.
  `session.declare_resource(key_expr)` allocates next ExprId (monotonic from 1),
  stores mapping, returns `SendResourceDeclare` action.
- **Remote resources** (`HashMap<ExprId, String>`): Populated from
  `ResourceDeclared` events. Validated for no ExprId collision. Emits
  `ResourceAdded` so caller can update subscription routing.

Undeclare removes from the relevant map and emits `ResourceRemoved`. On session
close, all resources are bulk-undeclared (one `ResourceRemoved` per resource).

ExprId namespaces are independent per side — each peer assigns its own IDs.

## Keepalive & Stale Detection

Tracks `last_received_ms` (updated on any inbound event) and `last_sent_ms`.

On `TimerTick { now_ms }`:
- If `now_ms - last_sent_ms >= keepalive_interval_ms`: emit `SendKeepalive`.
- If `now_ms - last_received_ms >= stale_timeout_ms`: emit `PeerStale`,
  transition to Closed.

Defaults: 30s keepalive interval, 90s stale timeout (3× interval).

## Close Handshake

**Initiated locally:**
1. `session.initiate_close(now_ms)` → Closing state, emit `SendClose`.
2. `CloseAckReceived` → Closed, emit `SessionClosed`, bulk-undeclare.
3. `TimerTick` with `now_ms - close_initiated_ms >= close_timeout_ms` →
   force-Closed.

**Initiated remotely:**
1. `CloseReceived` → emit `SendCloseAck`, transition to Closed, bulk-undeclare.

## Error Variants

New additions to `ZenohError`:

- `SessionNotActive` — event in wrong state
- `HandshakeFailed(String)` — signature verification failed
- `DuplicateExprId(u64)` — peer reused an ExprId
- `UnknownExprId(u64)` — undeclare for nonexistent ID

## Module Structure

```
crates/harmony-zenoh/src/
  lib.rs            — add pub mod session, re-export public types
  session.rs        — Session, SessionState, SessionEvent, SessionAction,
                      SessionConfig, ExprId, handle_event logic
  envelope.rs       — unchanged
  keyspace.rs       — unchanged
  subscription.rs   — unchanged
  error.rs          — add 4 new variants
```

No new external dependencies. harmony-identity (already a dep) provides
Ed25519 sign/verify for the handshake.

## Tests

1. Handshake roundtrip — both sides exchange proofs, both reach Active
2. Handshake with wrong identity fails
3. Handshake replay rejected (proof signed for different peer)
4. Resource declare/undeclare lifecycle
5. Duplicate ExprId rejected
6. Unknown ExprId undeclare rejected
7. Local ExprId allocation is monotonic
8. Keepalive emitted after interval
9. Stale peer detected after timeout
10. Graceful close handshake (Close → CloseAck → Closed)
11. Close timeout forces Closed
12. Remote-initiated close (CloseReceived → SendCloseAck → Closed)

# Selective Discovery Opt-In

**Date:** 2026-03-22
**Status:** Draft
**Scope:** `harmony-identity` (CapabilityType), `harmony-zenoh` (namespace), `harmony-node` (runtime, event_loop)
**Bead:** harmony-0kq

## Problem

Announces published to `harmony/reticulum/announce/{dest_hash}` are visible
to all subscribers. When an announce includes `RoutingHint::Tunnel` (with
NodeId, relay URL, and direct socket addresses), any listener can map the
identity to its network topology. This enables passive surveillance: an
observer subscribes to the wildcard namespace and builds an identity-to-IP
mapping without any interaction.

## Solution

Split announcement into two layers:

1. **Public broadcast** — Announces published to the broadcast namespace
   contain only Reticulum and Zenoh routing hints. No tunnel information
   is included. Strangers can see that an identity exists and is reachable
   via Reticulum (which is already anonymous — Reticulum routes don't carry
   IP addresses). If a node has no Reticulum hints, the public record has
   zero routing hints — the identity is discoverable but not directly
   reachable without authentication.

2. **Authenticated queryable** — A Zenoh queryable on
   `harmony/discover/{identity_hash}` serves full routing hints (including
   tunnel) only to peers who present a valid PQ UCAN token with
   `CapabilityType::Discovery`. Unauthorized queries receive the public
   version (Reticulum-only hints).

Note: `harmony/discover/` is distinct from `harmony/identity/` (which is
for DID document resolution). Discovery serves routing hints for tunnel
connectivity; identity resolution serves cryptographic key material and
DID documents.

## Design Decisions

### Two separately-signed records (not post-hoc filtering)

Routing hints are part of `SignablePayload`. Stripping tunnel hints after
signing would invalidate the signature. Instead, the runtime builds two
records at announce time using `AnnounceBuilder`:

- **Public record** — built with Reticulum/Zenoh hints only, properly signed
- **Full record** — built with all hints (including tunnel), properly signed

Both are stored on NodeRuntime as pre-serialized bytes. The public record
is broadcast and served to unauthorized queries; the full record is served
from the authenticated queryable to authorized queries.

### New CapabilityType::Discovery (not reuse of Identity)

Adding `Discovery = 7` gives explicit semantics. The `resource` field holds
the 16-byte identity hash of the announcer ("I authorize you to discover
*me*"). Overloading `Identity = 4` would create ambiguity as more identity
operations are added.

### UCAN token validation (not contact-store gating)

UCAN tokens are self-certifying — the queryable validates the token
cryptographically without shared state. This enables:
- Fine-grained control (per-peer, time-bounded authorization)
- Delegation via UCAN proof chains (future)
- Decoupling authorization from the social graph

Contact-store gating would conflate "known peer" with "authorized for
discovery" and couldn't support delegation.

### Graceful degradation on invalid tokens

Invalid or missing tokens receive the public announce (Reticulum-only hints),
not silence. Reticulum hints are already publicly broadcast, so returning
them from the queryable leaks no additional information. This gives the
caller a useful fallback path (Reticulum connectivity) and avoids a slow
timeout failure.

### Token verified with local key (not pubkey_cache)

Discovery tokens are issued by the local node (the announcer). The
`issuer` field in the token is our own identity hash. We verify the
ML-DSA signature using our own public key — NOT `pubkey_cache`, which
stores remote peers' keys. The local node always knows its own key.

### Unix timestamp via `last_unix_now` field (not modifying QueryReceived)

The discover query handler needs Unix epoch seconds for token time-bound
validation. Rather than adding `unix_now` to `RuntimeEvent::QueryReceived`
(which would touch 20+ test call sites), we add a `last_unix_now: u64`
field to `NodeRuntime`, updated from `SystemTime::now()` at the event loop
boundary on each `TimerTick`. Timer ticks arrive every 250ms, so the value
is at most 250ms stale — well within the 60-second clock skew tolerance
used by `verify_announce()`.

## Architecture

### Public Broadcast (outbound)

When `DiscoveryAction::PublishAnnounce` fires, the runtime publishes the
**public record** (Reticulum-only hints) to `harmony/reticulum/announce/`.

### Authenticated Queryable (inbound)

At startup, declare a queryable on `harmony/discover/**`. Store the
queryable ID as `discover_queryable_id`.

Query handler (`handle_discover_query`):

1. Parse `identity_hash` from key expression (`harmony/discover/{hex}`)
2. Verify this is our identity (compare against `local_identity_hash`)
3. If not our identity → no reply
4. If payload is empty → respond with public record
5. Parse payload as `PqUcanToken::from_bytes()`
6. Validate token:
   - `capability == Discovery`
   - `resource` == local identity hash (16 bytes)
   - Time bounds: `not_before <= last_unix_now < expires_at`
   - ML-DSA signature verified with the local node's own public key
7. Valid → respond with full record (including tunnel hints)
8. Invalid → respond with public record

Response format: serialized `AnnounceRecord` bytes (same format as
broadcast).

### Record Building

Both records are built when routing information changes. Triggers:

- `LocalTunnelInfo` event (iroh endpoint binds, relay URL changes)
- Any change to Reticulum routing hints

The runtime holds:

```rust
local_public_announce: Option<Vec<u8>>,   // Serialized, Reticulum-only
local_full_announce: Option<Vec<u8>>,     // Serialized, all hints
```

Stored as serialized bytes to avoid re-serializing on every query response.

The `DiscoveryManager` receives the full record via `set_local_record()`
for its cache and liveliness tracking. The two-record split is the
runtime's responsibility at the I/O boundary — the manager doesn't need
to know about public vs. full records.

Building requires the local identity's private key for signing. The full
signing path is deferred to whenever outbound announce publishing is wired
(existing TODO in runtime.rs). For this bead, records are injected via
test helpers and the queryable serves whatever is stored.

### Token Semantics

| Field | Value |
|-------|-------|
| `issuer` | Announcer's identity hash (local node) |
| `audience` | Authorized peer's identity hash |
| `capability` | `Discovery` (7) |
| `resource` | Announcer's identity hash (16 bytes) |
| `not_before` | Earliest valid timestamp |
| `expires_at` | Token expiry |
| `signature` | ML-DSA-65 by the issuer (local node's own key) |

Token issuance and query-side construction are out of scope. This bead
implements the validation and response side only.

## Changes

### harmony-identity

Add to `CapabilityType`:

```rust
Discovery = 7,
```

Update the `TryFrom<u8>` implementation to handle `7 => Ok(Self::Discovery)`.

### harmony-zenoh

Add `discover` namespace module:

```rust
pub mod discover {
    pub const PREFIX: &str = "harmony/discover";
    pub const SUB: &str = "harmony/discover/**";

    pub fn key(identity_hash_hex: &str) -> String {
        format!("{PREFIX}/{identity_hash_hex}")
    }
}
```

### harmony-node (runtime.rs)

New fields:

```rust
discover_queryable_id: QueryableId,
local_public_announce: Option<Vec<u8>>,   // Serialized, Reticulum-only
local_full_announce: Option<Vec<u8>>,     // Serialized, all hints
last_unix_now: u64,                       // Unix epoch seconds, updated on TimerTick
```

New in constructor: declare discover queryable, initialize fields.

New dispatch in `route_query()`: `handle_discover_query()` for
`discover_queryable_id`.

New method `handle_discover_query()`: parses identity from key, validates
token with local key, responds with full or public record.

Update `TimerTick` handling: store `last_unix_now` from event loop.

### harmony-node (event_loop.rs)

Inject `unix_now` into `TimerTick` events — add a `unix_now: u64` field
to `RuntimeEvent::TimerTick`, populated from `SystemTime::now()` at the
event loop boundary.

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-identity/src/ucan.rs` | Add `Discovery = 7`, update `TryFrom<u8>` |
| `crates/harmony-zenoh/src/namespace.rs` | Add `discover` module |
| `crates/harmony-node/src/runtime.rs` | Declare discover queryable, `handle_discover_query()`, local announce fields, `last_unix_now` field, TimerTick update |
| `crates/harmony-node/src/event_loop.rs` | Inject `unix_now` into `TimerTick` events |

## Testing

- `discovery_capability_type_roundtrip` — `Discovery = 7` byte roundtrip
- `discover_query_valid_token_returns_full_hints` — authorized query gets tunnel hints
- `discover_query_invalid_token_returns_public_hints` — bad token gets Reticulum-only
- `discover_query_missing_token_returns_public_hints` — empty payload gets Reticulum-only
- `discover_query_expired_token_returns_public_hints` — expired token gets public
- `discover_query_wrong_capability_returns_public_hints` — Content token rejected
- `discover_query_wrong_identity_no_reply` — query for unknown identity gets no reply
- `public_announce_broadcast_excludes_tunnel_hints` — outbound publish stripped
- `runtime_builds_public_and_full_records` — LocalTunnelInfo produces two records with different hint sets

## What is NOT in Scope

- Token issuance (programmatic creation deferred to future bead)
- Token distribution (out-of-band for V1)
- Query-side construction (client sends token as query payload — straightforward, not specced)
- Delegation chains (UCAN `proof` field supports this, but V1 validates root tokens only)
- Outbound announce publishing wiring (existing TODO, separate concern)
- Ed25519 discovery tokens (ML-DSA-65 only for now)
- Changes to `harmony-discovery` crate (record building and filtering handled at runtime layer)

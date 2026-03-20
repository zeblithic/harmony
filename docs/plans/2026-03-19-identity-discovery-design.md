# Identity Namespace & Discovery Design

**Date:** 2026-03-19
**Status:** Approved
**Scope:** New `harmony-discovery` crate + additive `harmony-zenoh` namespace patterns
**Bead:** harmony-tq4

## Overview

Harmony needs Zenoh-native identity discovery to complement Reticulum's
local/mesh announce mechanism. Reticulum handles Tier 1 transport-level
discovery (ad-hoc, subnet, direct peer), but navigating a vast hierarchy
of identities across the broader network requires Zenoh's scalable
pub/sub and queryable infrastructure.

This design introduces a `harmony-discovery` crate that provides:

1. **Announce records** — signed, self-contained identity announcements
   published via Zenoh pub/sub
2. **Queryable resolution** — request/response lookup of identity
   records for on-demand discovery
3. **Liveliness tracking** — Zenoh liveliness tokens for binary
   online/offline presence
4. **Offline resolution trait** — interface for DHT/content-layer
   fallback (implementation deferred)

## Design Philosophy

### Zenoh Is the Discovery Backbone

Reticulum is transport. Zenoh is discovery. Reticulum announces work
at the packet level for local mesh discovery, but Zenoh's hierarchical
key expressions, wildcard matching, and queryable pattern provide the
scalable backbone for global identity resolution. Reticulum announces
can feed into the Zenoh discovery layer via runtime bridging, but the
two are independent systems.

### Sans-I/O

Following the established Harmony pattern, the `DiscoveryManager` is a
pure state machine. It consumes events (announce received, query
received, liveliness change, tick) and emits actions (publish, respond,
notify). No Zenoh runtime dependency, no async, no I/O. The caller maps
actions to actual Zenoh operations.

### Separation from Profiles

Identity discovery (`harmony/identity/`) handles machine identity:
crypto keys, routing hints, presence. User-facing profiles (display
name, bio, avatar) remain in the reserved `harmony/profile/` namespace
and are a separate concern (bead harmony-3xn). Different crates,
different update cadence, different privacy requirements.

## Namespace

All identity discovery operates under `harmony/identity/`:

```
harmony/identity/{address_hex}/announce    — pub/sub announce channel
harmony/identity/{address_hex}/resolve     — queryable endpoint
harmony/identity/{address_hex}/alive       — liveliness token
harmony/identity/**/announce               — subscribe to all announces
harmony/identity/**/alive                  — subscribe to all presence
```

Where `{address_hex}` is the 32-char lowercase hex encoding of the
16-byte `IdentityHash`. Hash-only addressing (no `CryptoSuite` byte
in the key expression) — consistent with content addressing and keeps
keys shorter. The crypto suite is carried inside the announce record
payload.

### Announce (Pub/Sub)

Identities publish `AnnounceRecord` when they come online or their
metadata changes. Subscribers matching `harmony/identity/**/announce`
receive all announces; subscribers matching a specific address receive
updates for that identity only.

### Resolve (Queryable)

When a node needs identity data for a specific `IdentityHash`, it
queries `harmony/identity/{address_hex}/resolve`. The identity owner
(or any caching relay with the record) responds with the
`AnnounceRecord`. This is the "pull" complement to the "push" announce.

### Alive (Liveliness)

Binary presence. An identity declares a liveliness token at
`harmony/identity/{address_hex}/alive` while online. Zenoh
automatically removes it when the session drops. Subscribers get
join/leave events. No capability or tier information — just
online/offline.

## AnnounceRecord

The core data structure for identity discovery:

```rust
/// A signed identity announcement for Zenoh-based discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnnounceRecord {
    pub identity_ref: IdentityRef,          // 17B: hash + suite
    pub public_key: Vec<u8>,                // verifying key bytes
    pub routing_hints: Vec<RoutingHint>,     // how to reach this identity
    pub published_at: u64,                  // unix timestamp
    pub expires_at: u64,                    // prevents stale caching
    pub nonce: [u8; 16],                    // replay protection
    pub signature: Vec<u8>,                 // self-signed
}
```

### Field Rationale

- **`identity_ref`** — `IdentityRef` carries both address hash and
  `CryptoSuite` tag. The suite tells the verifier which signature
  algorithm to use.

- **`public_key`** — the verifying key (Ed25519: 32B, ML-DSA-65:
  1952B). Included so a receiver can verify the signature without a
  prior relationship. Same convention as `CredentialKeyResolver`.

- **`routing_hints`** — how to reach this identity. A vec because an
  identity might be reachable via multiple transports:

  ```rust
  #[derive(Debug, Clone, Serialize, Deserialize)]
  pub enum RoutingHint {
      /// Reticulum destination hash (for Tier 1 link establishment)
      Reticulum { destination_hash: [u8; 16] },
      /// Zenoh locator (for direct Zenoh session)
      Zenoh { locator: Vec<u8> },
  }
  ```

- **`published_at`** — when this record was created. Used for
  freshness comparison when multiple records exist for the same
  identity.

- **`expires_at`** — prevents indefinite caching of stale records.
  Short-lived for mobile identities, longer for stable infrastructure.

- **`nonce`** — 16 random bytes for replay protection and correlation
  resistance.

- **`signature`** — self-signed by the identity's private key. Unlike
  credentials (which are signed by an issuer), announces are always
  self-signed. The `public_key` in the record is used for verification.

### Serialization

Postcard with format version byte prefix (same pattern as `Credential`,
`KeyEventLog`, and `TrustStore`).

### Self-Signed Verification

Announce verification differs from credential verification: the public
key is in the record itself, not resolved externally. The verifier:

1. Checks `expires_at > now` (not expired)
2. Re-derives the address hash from `public_key` and compares with
   `identity_ref.hash` (prevents spoofing)
3. Verifies the signature using `public_key` and `identity_ref.suite`

No external key resolver needed. This makes announces self-contained
and verifiable by any node without prior state.

### Address Derivation Check

For Ed25519 identities, the address is `SHA256(X25519_pub ||
Ed25519_pub)[:16]`, which requires the encryption key — not available
in the announce record (which only carries the verifying key). Two
options:

1. Include the encryption key in the record (adds 32B for Ed25519,
   ~1.2KB for ML-KEM-768)
2. Skip the address derivation check for now and rely on signature
   validity alone

Option 2 is acceptable for V1: if the signature verifies against
`public_key`, the announcer demonstrably controls the private key
corresponding to that public key. The address hash is a convenient
routing identifier but not a security boundary — the signature is.
A future enhancement can add the encryption key for full address
re-derivation if needed.

## AnnounceBuilder

Sans-I/O builder pattern, consistent with `CredentialBuilder`:

```rust
pub struct AnnounceBuilder {
    identity_ref: IdentityRef,
    public_key: Vec<u8>,
    routing_hints: Vec<RoutingHint>,
    published_at: u64,
    expires_at: u64,
    nonce: [u8; 16],
}

impl AnnounceBuilder {
    pub fn new(
        identity_ref: IdentityRef,
        public_key: Vec<u8>,
        published_at: u64,
        expires_at: u64,
        nonce: [u8; 16],
    ) -> Self { ... }

    pub fn add_routing_hint(&mut self, hint: RoutingHint) -> &mut Self { ... }
    pub fn signable_payload(&self) -> Vec<u8> { ... }
    pub fn build(self, signature: Vec<u8>) -> AnnounceRecord { ... }
}
```

`expires_at` must be > `published_at` (constructor panics otherwise,
same as `CredentialBuilder`). `nonce` is required to prevent accidental
omission.

## DiscoveryManager

Sans-I/O state machine following the `PeerManager` pattern:

```rust
pub struct DiscoveryManager {
    /// Our own published announce record (if announcing)
    local_record: Option<AnnounceRecord>,
    /// Cache of verified remote announce records
    known_identities: HashMap<IdentityHash, AnnounceRecord>,
    /// Which identities are currently online (liveliness)
    online: HashSet<IdentityHash>,
}
```

### Events

```rust
pub enum DiscoveryEvent {
    /// Received a published announce from the network
    AnnounceReceived { record: AnnounceRecord },
    /// Someone queried us for an identity's record
    QueryReceived { address: IdentityHash, query_id: u64 },
    /// Liveliness change from Zenoh
    LivelinessChange { address: IdentityHash, alive: bool },
    /// Periodic maintenance
    Tick { now: u64 },
}
```

### Actions

```rust
pub enum DiscoveryAction {
    /// Publish our announce record to the network
    PublishAnnounce { record: AnnounceRecord },
    /// Respond to a resolve query
    RespondToQuery { query_id: u64, record: Option<AnnounceRecord> },
    /// Declare or undeclare our liveliness token
    SetLiveliness { alive: bool },
    /// Notify the application of a discovered identity
    IdentityDiscovered { record: AnnounceRecord },
    /// Notify that an identity went offline
    IdentityOffline { address: IdentityHash },
    /// An expired record was evicted from cache
    RecordExpired { address: IdentityHash },
}
```

### Behavior

`on_event(event) -> Vec<DiscoveryAction>` — pure function.

- **`AnnounceReceived`**: verify signature, check expiry, check
  freshness (only accept if newer than cached), update cache, emit
  `IdentityDiscovered`.
- **`QueryReceived`**: look up cache by address, respond with record
  if known (`Some`) or `None`. This enables any node to act as a
  caching relay.
- **`LivelinessChange(alive=true)`**: add to online set, emit
  `IdentityDiscovered` if we have a cached record for this address.
- **`LivelinessChange(alive=false)`**: remove from online set, emit
  `IdentityOffline`.
- **`Tick`**: evict expired records from cache, emit `RecordExpired`
  for each.

### Local Identity Management

```rust
impl DiscoveryManager {
    /// Set our local announce record for publishing.
    pub fn set_local_record(&mut self, record: AnnounceRecord);

    /// Start announcing — emits PublishAnnounce + SetLiveliness.
    pub fn start_announcing(&mut self) -> Vec<DiscoveryAction>;

    /// Stop announcing — emits SetLiveliness(false).
    pub fn stop_announcing(&mut self) -> Vec<DiscoveryAction>;

    /// Check if an identity is currently online.
    pub fn is_online(&self, address: &IdentityHash) -> bool;

    /// Get a cached announce record.
    pub fn get_record(&self, address: &IdentityHash) -> Option<&AnnounceRecord>;
}
```

## Offline Resolution

A trait interface for DHT/content-layer fallback. Designed now,
implementation deferred to a future bead:

```rust
/// Resolve an identity's announce record from persistent storage.
///
/// Used when no live Queryable answers. The content layer can
/// implement this by storing AnnounceRecords as content objects.
pub trait OfflineResolver {
    fn resolve(&self, address: &IdentityHash) -> Option<AnnounceRecord>;
}
```

The `DiscoveryManager` does not call this directly — the caller
(runtime integration layer) falls back to offline resolution when a
query yields no cached result. This keeps the state machine pure and
avoids coupling to the content layer.

## Error Handling

```rust
pub enum DiscoveryError {
    Expired,
    SignatureInvalid,
    AddressMismatch,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}
```

Implements `Display` and `std::error::Error` (under `std` feature),
following the pattern in `CredentialError` and `KelError`.

## Integration Points

### harmony-zenoh (additive change)

Key expression builders added to `namespace.rs` for the
`harmony/identity/` namespace:

```rust
pub fn identity_announce_key(address_hex: &str) -> String { ... }
pub fn identity_resolve_key(address_hex: &str) -> String { ... }
pub fn identity_alive_key(address_hex: &str) -> String { ... }
pub fn identity_all_announces() -> &'static str { ... }
pub fn identity_all_alive() -> &'static str { ... }
```

No logic changes — just new key expression patterns alongside the
existing tiers.

### harmony-peers

`DiscoveryAction::IdentityDiscovered` maps to
`PeerEvent::AnnounceReceived` at the runtime layer. The runtime
bridges discovery events into the peer manager's state machine.

### harmony-contacts

When a discovered identity matches a contact, the runtime can call
`ContactStore::update_last_seen`. No direct coupling.

### harmony-reticulum

`RoutingHint::Reticulum` carries the destination hash needed for
Tier 1 link establishment. Reticulum announces can feed into
`DiscoveryEvent::AnnounceReceived` via runtime bridging, making
Reticulum announces visible through the Zenoh discovery layer.

## Crate Structure

```
harmony-discovery/
  Cargo.toml
  src/
    lib.rs              — no_std setup, re-exports
    record.rs           — AnnounceRecord, RoutingHint, AnnounceBuilder, serialization
    verify.rs           — verify_announce(), signature dispatch
    manager.rs          — DiscoveryManager, DiscoveryEvent, DiscoveryAction
    resolve.rs          — OfflineResolver trait
    error.rs            — DiscoveryError enum
```

### Dependencies

```toml
[dependencies]
ed25519-dalek = { workspace = true }
harmony-crypto = { workspace = true, features = ["serde"] }
harmony-identity = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
postcard = { workspace = true }

[dev-dependencies]
rand = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = ["harmony-identity/std", "harmony-crypto/std", "postcard/use-std", "serde/std"]
test-utils = ["hashbrown"]

[dependencies.hashbrown]
workspace = true
optional = true
```

No dependency on `harmony-zenoh` — the key expression builders live
in `harmony-zenoh` but `harmony-discovery` doesn't import them. The
runtime integration layer uses both crates together.

### Re-exports (lib.rs)

```rust
pub use record::{AnnounceRecord, AnnounceBuilder, RoutingHint};
pub use verify::verify_announce;
pub use manager::{DiscoveryManager, DiscoveryEvent, DiscoveryAction};
pub use resolve::OfflineResolver;
pub use error::DiscoveryError;
```

## Changes to Existing Crates

**harmony-zenoh:** Additive only — new key expression builder functions
in `namespace.rs` for the `harmony/identity/` namespace. No logic
changes, no breaking changes.

No other crate modifications. All integration happens at the runtime
layer.

## Testing Strategy

### AnnounceRecord tests
- Builder produces valid record with correct fields
- `signable_payload()` is deterministic
- Serialization round-trip preserves all fields
- Builder panics if `expires_at <= published_at`

### Verification tests
- Valid self-signed announce passes
- Expired announce rejected
- Tampered signature rejected
- Address mismatch rejected (wrong public key for hash)
- Both Ed25519 and ML-DSA-65 announce paths

### DiscoveryManager tests
- `AnnounceReceived` with valid record: caches and emits `IdentityDiscovered`
- `AnnounceReceived` with invalid signature: ignored, no action emitted
- `AnnounceReceived` with expired record: ignored
- Fresher record replaces older cached record
- Stale record does not replace fresher cached record
- `QueryReceived` for known identity: responds with record
- `QueryReceived` for unknown identity: responds with `None`
- `LivelinessChange(alive=true)`: adds to online set, emits `IdentityDiscovered`
- `LivelinessChange(alive=false)`: removes from online set, emits `IdentityOffline`
- `Tick` evicts expired records, emits `RecordExpired`
- `start_announcing` emits `PublishAnnounce` + `SetLiveliness(true)`
- `stop_announcing` emits `SetLiveliness(false)`

### Key expression tests (in harmony-zenoh)
- Builder functions produce correct key expressions
- Wildcard patterns match expected addresses

### Integration tests
- Full flow: build announce → publish → receive → verify → cache → query → respond
- Ed25519 and ML-DSA-65 issuer paths

## Future Beads

- **Offline resolution backend** — content-layer integration for
  persistent announce record storage and DHT-style discovery
- **Capability advertisement** — `harmony/services/` namespace for
  advertising node capabilities (relay, storage, compute)
- **Announce record encryption** — optional encryption of routing
  hints for privacy-sensitive identities

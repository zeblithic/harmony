# Peer Model Design: Contacts Store & Peering Lifecycle

**Date:** 2026-03-19
**Status:** Approved
**Scope:** `harmony-contacts` crate + `harmony-peers` crate

## Overview

Harmony nodes interact with two classes of peers:

- **Adjacent peers** — every device within direct physical communication range
  (WiFi, Bluetooth, LoRa, etc.). Transient, discovered via announces. No trust
  required. Purpose: routing, relay, mesh density. Managed by the existing
  transport layer (path table, announce machinery in `harmony-reticulum`).

- **Tunnel peers** — intentional, persistent relationships with known identities
  regardless of physical location. Purpose: social connectivity, backup,
  content precaching. Maintained by the new peering lifecycle.

This design introduces two new crates that together enable tunnel peer
management:

1. **`harmony-contacts`** — pure data crate for the contacts store
2. **`harmony-peers`** — sans-I/O state machine for the peering lifecycle

## Prerequisites

### `IdentityHash` type alias

The `harmony-identity` crate currently uses raw `[u8; ADDRESS_HASH_LENGTH]`
fields. This design requires a named type alias:

```rust
// In harmony-identity
pub type IdentityHash = [u8; ADDRESS_HASH_LENGTH]; // [u8; 16]
```

This is a backward-compatible addition (type alias, not a new struct).

### Identity-to-Destination Translation

The Reticulum layer speaks in `DestinationHash` (derived from identity
address + destination name), while the contacts/peers layer speaks in
`IdentityHash` (the identity's address hash). The integration layer must
translate between these:

- **Identity → Destination:** The caller looks up the contact's full
  `Identity` (from announce cache or identity store) and combines it with a
  `DestinationName` to derive the `DestinationHash` for Link initiation
  and path requests.
- **Destination → Identity:** When the Node emits `AnnounceReceived`, the
  caller extracts `validated_announce.identity.address_hash` to identify
  which contact the announce belongs to.

The `PeerManager` operates purely in identity-hash space. The caller is
responsible for the translation, using the path table's `ValidatedAnnounce`
(which carries the full `Identity`) or a local identity cache.

## Architecture

```
harmony-identity
    ├── harmony-contacts   (pure data: ContactStore, Contact, PeeringPolicy)
    └── harmony-reticulum
        └── harmony-peers  (lifecycle: PeerManager state machine)
```

### Separation of Concerns

| Concern | Owner | NOT owned here |
|---------|-------|----------------|
| "Who are my intentional peers?" | harmony-contacts | Trust scores (separate trust store) |
| "Maintain links to those peers" | harmony-peers | Actual I/O (caller executes actions) |
| "Route packets to destinations" | harmony-reticulum | Contact/social concepts |
| "Who is this identity?" | harmony-identity | Peering policies |
| "How much do I trust them?" | Future trust store | Embedded in contacts |

Trust, identity, and contacts are separate stores with clear boundaries.
A contact references an identity by hash. Trust is queried from a separate
store when needed. The contacts store does not embed trust scores — they are
different concerns that overlap but remain independent.

## Crate 1: `harmony-contacts`

### Purpose

Pure data crate for managing intentional peer relationships. No I/O, no
networking dependencies. Depends only on `harmony-identity` for `IdentityHash`.

### `no_std` Compatibility

Both `harmony-contacts` and `harmony-peers` follow the workspace convention:
`#![cfg_attr(not(feature = "std"), no_std)]` with `hashbrown` for HashMap
(no_std-compatible, already a workspace dependency). This ensures usability
in `harmony-os` (Ring 1/2 environments without std).

### Serialization

Uses `postcard` (already a workspace dependency) for compact binary
serialization. The serialized format is versioned: the first byte is a
format version number (starting at `1`) to support forward-compatible
schema evolution.

### Core Types

```rust
/// A persistent, intentional relationship with another identity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    /// Reference into the identity store.
    pub identity_hash: IdentityHash,
    /// Local human-assigned label (not from the identity's profile).
    pub display_name: Option<String>,
    /// Controls persistent link behavior.
    pub peering: PeeringPolicy,
    /// When this contact was added (unix timestamp seconds).
    pub added_at: u64,
    /// Last time we had an active link to this peer (updated by peering lifecycle).
    pub last_seen: Option<u64>,
    /// Freeform user annotation.
    pub notes: Option<String>,
}

/// Controls whether and how aggressively the peering lifecycle
/// maintains a persistent Link to this contact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeeringPolicy {
    /// Whether to maintain a persistent link at all.
    pub enabled: bool,
    /// How aggressively to reconnect when the link drops.
    pub priority: PeeringPriority,
}

/// Reconnection aggressiveness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PeeringPriority {
    /// Announce-driven only. Reconnect when we passively hear them.
    Low,
    /// Moderate active probing. Path request every ~120s.
    Normal,
    /// Aggressive probing. Path request every ~30s, multiple interfaces.
    High,
}
```

### ContactStore

```rust
/// Persistent store of intentional peer relationships.
pub struct ContactStore {
    contacts: HashMap<IdentityHash, Contact>,
}

impl ContactStore {
    pub fn new() -> Self;

    // CRUD
    pub fn add(&mut self, contact: Contact) -> Result<(), ContactError>;
    pub fn remove(&mut self, id: &IdentityHash) -> Option<Contact>;
    pub fn get(&self, id: &IdentityHash) -> Option<&Contact>;
    pub fn get_mut(&mut self, id: &IdentityHash) -> Option<&mut Contact>;
    pub fn contains(&self, id: &IdentityHash) -> bool;

    // Lifecycle updates
    pub fn update_last_seen(&mut self, id: &IdentityHash, timestamp: u64);

    // Queries used by peering lifecycle
    pub fn peers_with_peering_enabled(&self) -> impl Iterator<Item = &Contact>;
    pub fn peers_by_priority(&self, p: PeeringPriority) -> impl Iterator<Item = &Contact>;

    // Enumeration
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    pub fn iter(&self) -> impl Iterator<Item = (&IdentityHash, &Contact)>;

    // Persistence
    pub fn serialize(&self) -> Vec<u8>;
    pub fn deserialize(data: &[u8]) -> Result<Self, ContactError>;
}
```

### Error Types

```rust
pub enum ContactError {
    /// Contact with this identity hash already exists.
    AlreadyExists(IdentityHash),
    /// Deserialization failed.
    DeserializeError(&'static str),
}
```

### What This Crate Does NOT Do

- No trust scores (separate trust store, separate bead)
- No identity resolution or key management (harmony-identity's job)
- No networking, no Link establishment
- No replication policy (future bead will add `ReplicationPolicy` to Contact)
- No adjacent/discovered peer tracking (transport layer concern)

### Testing Strategy

Pure data crate — tests are straightforward:
- CRUD: add, get, remove, update, duplicate detection
- Serialization: round-trip, corrupt data handling
- Query filtering: by policy enabled, by priority
- Edge cases: empty store, remove nonexistent, iterate during modification

## Crate 2: `harmony-peers`

### Purpose

Sans-I/O state machine that maintains persistent Links to contacts with
peering enabled. Follows the same pattern as `Node` in harmony-reticulum:
the caller provides events, the PeerManager returns actions, the caller
executes the actual I/O.

### Event/Action Model

```rust
/// Events fed into the PeerManager by the caller/runtime.
#[derive(Debug, Clone)]
pub enum PeerEvent {
    /// A contact was added or its peering policy changed.
    ContactChanged { identity_hash: IdentityHash },
    /// A contact was removed from the store.
    ContactRemoved { identity_hash: IdentityHash },
    /// An announce for this identity appeared in the path table.
    AnnounceReceived { identity_hash: IdentityHash },
    /// A Link to this peer reached Active state.
    LinkEstablished { identity_hash: IdentityHash, now: u64 },
    /// A Link to this peer closed (timeout, error, remote close).
    LinkClosed { identity_hash: IdentityHash },
    /// Periodic timer tick. Caller decides interval (~1s recommended).
    Tick { now: u64 },
}

/// Actions the PeerManager asks the caller to perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAction {
    /// Initiate a Link to this destination (path is known).
    InitiateLink { identity_hash: IdentityHash },
    /// Send a path request for this destination (path unknown).
    SendPathRequest { identity_hash: IdentityHash },
    /// Drop/close an active Link (contact removed or policy disabled).
    CloseLink { identity_hash: IdentityHash },
    /// Update last_seen timestamp on the contact.
    UpdateLastSeen { identity_hash: IdentityHash, timestamp: u64 },
}
```

### PeerManager

```rust
pub struct PeerManager {
    /// Tracked state for each contact with peering enabled.
    peers: HashMap<IdentityHash, PeerState>,
}

/// Per-peer connection lifecycle state.
struct PeerState {
    status: PeerStatus,
    priority: PeeringPriority,
    /// When we last sent a path request (for probe interval calculation).
    last_probe: Option<u64>,
    /// When we last had an active link.
    last_seen: Option<u64>,
    /// When the current Connecting attempt started (for timeout).
    connecting_since: Option<u64>,
    /// Consecutive failed connection attempts (drives backoff).
    retry_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PeerStatus {
    /// No known path. Waiting for announce or actively probing.
    Searching,
    /// Path known, Link initiation requested but not yet established.
    Connecting,
    /// Link is Active.
    Connected,
    /// Peering policy disabled but contact still exists.
    Disabled,
}
```

### Core Logic

```rust
impl PeerManager {
    pub fn new() -> Self;

    /// Process an event. Returns zero or more actions for the caller to execute.
    pub fn on_event(
        &mut self,
        event: PeerEvent,
        contacts: &ContactStore,
    ) -> Vec<PeerAction>;
}
```

#### State Transitions

| Current State | Event | New State | Action |
|---------------|-------|-----------|--------|
| (none) | ContactChanged (enabled) | Searching | (begin probing on next Tick) |
| (none) | ContactChanged (disabled) | Disabled | — |
| Searching | AnnounceReceived | Connecting | InitiateLink |
| Searching | Tick (probe interval elapsed, Normal/High) | Searching | SendPathRequest |
| Searching | Tick (Low priority) | Searching | — (announce-driven only) |
| Connecting | LinkEstablished | Connected | UpdateLastSeen |
| Connecting | LinkClosed | Searching | (increment retry_count) |
| Connected | LinkClosed | Searching | (increment retry_count) |
| Connected | Tick | Connected | — (monitor only) |
| Searching/Connecting/Connected | ContactChanged (disabled) | Disabled | CloseLink (if was Connected) |
| Searching/Connecting/Connected | ContactChanged (priority change) | (same) | (update priority) |
| Any | ContactRemoved | (removed) | CloseLink (if was Connected/Connecting) |
| Disabled | ContactChanged (enabled) | Searching | (begin probing on next Tick) |
| Connecting | Tick (link timeout) | Searching | (increment retry_count) |

**Connecting timeout:** The Reticulum layer handles Link handshake timeouts
and emits a `LinkClosed` event, which drives the Connecting → Searching
transition. The `Connecting | Tick` row is a fallback safety net — if no
`LinkClosed` arrives within 60 seconds, PeerManager assumes the attempt
failed and transitions to Searching itself.

#### Probe Intervals & Backoff

For `Normal` and `High` priority peers in `Searching` state:

- **Base interval:** High = 30s, Normal = 120s
- **Backoff:** `min(base_interval * 2^retry_count, 600s)` — caps at 10 minutes
- **Reset:** retry_count resets to 0 on `LinkEstablished`
- **Low priority:** No active probing. Only transitions to `Connecting` via
  `AnnounceReceived`.

This means a High-priority peer that's been offline for a while gets probed
at 30s → 60s → 120s → 240s → 600s → 600s → ... until they come back online,
at which point probing resets to 30s.

### Integration Point

The runtime (harmony-node or an async executor) wires everything together:

```rust
// Simplified integration loop — the caller translates between
// Reticulum's DestinationHash world and the PeerManager's IdentityHash world.
loop {
    let node_actions = node.on_event(raw_event);

    for action in node_actions {
        match action {
            NodeAction::AnnounceReceived { validated_announce, .. } => {
                // Extract the identity's address hash from the announce
                let id_hash = validated_announce.identity.address_hash;
                let actions = peer_manager.on_event(
                    PeerEvent::AnnounceReceived { identity_hash: id_hash },
                    &contact_store,
                );
                // When executing InitiateLink, the caller uses the
                // ValidatedAnnounce's full Identity + DestinationName
                // to call Link::initiate() via the Node.
                execute_actions(actions, &mut node, &identity_cache);
            }
            // ... translate LinkEstablished, LinkClosed similarly
        }
    }

    // Timer tick
    let actions = peer_manager.on_event(
        PeerEvent::Tick { now: current_time() },
        &contact_store,
    );
    execute_actions(actions, &mut node, &identity_cache);
}
```

### What This Crate Does NOT Do

- No actual I/O — the caller executes all actions
- No content routing or replication (future beads)
- No adjacent peer tracking (transport layer)
- No trust evaluation (peering is purely policy-driven from contacts store)
- No Link encryption or handshake logic (harmony-reticulum's job)

### Testing Strategy

Sans-I/O enables fully deterministic tests with no mocks:

- **Contact lifecycle:** Add contact → Tick → assert SendPathRequest for High/Normal
- **Announce discovery:** Feed AnnounceReceived → assert InitiateLink
- **Link established:** Feed LinkEstablished → assert Connected, no further actions
- **Reconnection:** Feed LinkClosed → assert Searching, backoff on next probe
- **Policy change:** Disable peering → assert CloseLink if was connected
- **Low priority:** Add Low contact → Tick → assert NO probing action
- **Backoff:** Close link repeatedly → assert probe intervals increase
- **Backoff cap:** Many failures → assert interval never exceeds 600s
- **Recovery:** LinkEstablished after failures → assert retry_count resets

## Future Beads (Out of Scope)

The following are tracked as separate beads, each requiring their own design
cycle. Notes from brainstorming are captured here for context.

### Trust Store

Persistent store for directional 8-bit trust scores (Identity / Compliance /
Association / Endorsement, 2 bits each). Separate from contacts. Queryable
by any identity hash.

- Transitive trust via EigenTrust-style propagation
- Published via Zenoh PubSub: `/trust/<truster-id>/<trustee-id>`
- Scores are public by default — unpublished scores are presumed `0x00`
- Users may choose not to publish, but operate in a "void" with no
  transitive trust benefit

### Identity Model Update

Post-quantum cryptography from day 0. Namespace structure:
`/id/sha256/<hash>`, `/id/sha224/<hash>`.

- PQC is the native/trusted encryption; Curve25519 is supported for backward
  compatibility with existing Reticulum networks but treated as untrusted
  ("essentially plaintext") — no downgrade attack risk because Harmony
  already doesn't trust it
- Hierarchical identity clustering: certain identities rise to "root" status
  (government IDs, verifiable credentials, AI agent licenses)
- Existing PQC work in harmony and harmony-os repos provides foundation
- Flat identity set → organic clustering based on credential types

### Profiles & Endorsements

Public information published about/by identities. Separate from identity itself.

- Profiles are published by the owning identity, signed by its private key
- Network accepts profiles for publishing based on valid signature
- Endorsements: "Identity A endorses Identity B" — signed statements that
  feed into transitive trust (trusting A increases trust in B)
- Accusations with cryptographic proof: truth is an absolute defense;
  whistleblowing is valuable to the community
- Endorsing a bad actor may impact the endorser's trust score

### Replication Policy

Which tunnel peers store copies of your data. Asymmetric — Alex backs up to
you, you don't back up to Alex.

- Adds `ReplicationPolicy` field to Contact
- Questions for future design: data granularity (all vs selective), encrypted
  at rest requirements, revocation mechanics, storage quotas

### Social Content Routing

Interest-driven data flow along the social graph. Precaching based on
trust/interest edges.

- The social graph has intrinsic routing/logistical value — if you like
  something, your friends are the first people you share it with
- Integration with Zenoh pub/sub for interest propagation
- Questions for future design: push vs pull, bandwidth budgets per peer,
  interest taxonomy

### Peering Privacy

Metadata protection for the social graph.

- Who can deduce your peer relationships from network observation?
- Potential mitigations: onion routing for link establishment, cover traffic
- Requires dedicated analysis — separate bead

### Adjacent Peer Reliability

Transport-layer tracking of physical neighbors' cooperation and reliability.

- Even untrusted adjacent peers add value if they faithfully relay traffic
- History-based scoring: peers that cooperate earn better link prioritization
- Long-term prisoner's dilemma dynamics: always try a little, self-healing
  network
- Scoring algorithm, decay rate, routing integration — all future design
  questions

# Peer Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement contacts store and peering lifecycle as two new crates (`harmony-contacts`, `harmony-peers`).

**Architecture:** Pure-data contacts store (`harmony-contacts`) depends only on `harmony-identity` for `IdentityHash`. Sans-I/O peering lifecycle (`harmony-peers`) consumes contacts + events from the Reticulum node and produces actions. Both crates are `no_std`-compatible, using `hashbrown` for maps and `postcard` for serialization.

**Tech Stack:** Rust 1.85+, no_std + alloc, hashbrown, postcard, TDD

**Spec:** `docs/plans/2026-03-19-peer-model-design.md`

---

## File Structure

### New files

```
crates/harmony-identity/src/identity.rs       — add IdentityHash type alias (modify)
crates/harmony-contacts/Cargo.toml            — crate manifest
crates/harmony-contacts/src/lib.rs            — re-exports
crates/harmony-contacts/src/contact.rs        — Contact, PeeringPolicy, PeeringPriority types
crates/harmony-contacts/src/store.rs          — ContactStore: CRUD, queries, serialization
crates/harmony-contacts/src/error.rs          — ContactError enum
crates/harmony-peers/Cargo.toml               — crate manifest
crates/harmony-peers/src/lib.rs               — re-exports
crates/harmony-peers/src/event.rs             — PeerEvent, PeerAction enums
crates/harmony-peers/src/state.rs             — PeerState, PeerStatus
crates/harmony-peers/src/manager.rs           — PeerManager: on_event, state transitions, probing
Cargo.toml                                    — add workspace members + deps (modify)
```

---

### Task 1: Add `IdentityHash` type alias to `harmony-identity`

**Files:**
- Modify: `crates/harmony-identity/src/identity.rs:28`
- Modify: `crates/harmony-identity/src/lib.rs`

- [ ] **Step 1: Add type alias**

In `crates/harmony-identity/src/identity.rs`, after line 28 (`pub const ADDRESS_HASH_LENGTH`), add:

```rust
/// A 128-bit identity address hash: `SHA256(X25519_pub || Ed25519_pub)[:16]`.
/// Used as the canonical key for referencing identities across Harmony.
pub type IdentityHash = [u8; ADDRESS_HASH_LENGTH];
```

- [ ] **Step 2: Re-export from lib.rs**

In `crates/harmony-identity/src/lib.rs`, add to the `pub use identity::` line:

```rust
pub use identity::{Identity, IdentityHash, PrivateIdentity};
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-identity`
Expected: All existing tests pass (type alias is backward-compatible).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-identity/src/identity.rs crates/harmony-identity/src/lib.rs
git commit -m "feat(identity): add IdentityHash type alias"
```

---

### Task 2: Scaffold `harmony-contacts` crate

**Files:**
- Create: `crates/harmony-contacts/Cargo.toml`
- Create: `crates/harmony-contacts/src/lib.rs`
- Create: `crates/harmony-contacts/src/error.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-contacts"
description = "Contact store for intentional peer relationships in the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-identity = { workspace = true }
hashbrown = { workspace = true, features = ["serde"] }
postcard = { workspace = true }
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }

[features]
default = ["std"]
std = [
    "harmony-identity/std",
    "postcard/use-std",
    "serde/std",
]

[dev-dependencies]
```

- [ ] **Step 2: Create error.rs**

```rust
use harmony_identity::IdentityHash;

/// Errors returned by ContactStore operations.
#[derive(Debug)]
pub enum ContactError {
    /// A contact with this identity hash already exists.
    AlreadyExists(IdentityHash),
    /// Deserialization of persisted data failed.
    DeserializeError(&'static str),
}

impl core::fmt::Display for ContactError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyExists(h) => write!(f, "contact already exists: {:02x?}", &h[..4]),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}
```

- [ ] **Step 3: Create lib.rs**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod error;

pub use error::ContactError;
```

- [ ] **Step 4: Add workspace entries**

In root `Cargo.toml`, add `"crates/harmony-contacts"` to the `members` list and add:

```toml
harmony-contacts = { path = "crates/harmony-contacts", default-features = false }
```

to the `[workspace.dependencies]` section.

- [ ] **Step 5: Build**

Run: `cargo check -p harmony-contacts`
Expected: Compiles successfully.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-contacts/ Cargo.toml
git commit -m "feat(contacts): scaffold harmony-contacts crate"
```

---

### Task 3: Implement `Contact` and `PeeringPolicy` types

**Files:**
- Create: `crates/harmony-contacts/src/contact.rs`
- Modify: `crates/harmony-contacts/src/lib.rs`

- [ ] **Step 1: Write tests first**

Add to the bottom of `contact.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_peering_policy_is_disabled_normal() {
        let policy = PeeringPolicy::default();
        assert!(!policy.enabled);
        assert_eq!(policy.priority, PeeringPriority::Normal);
    }

    #[test]
    fn peering_priority_ordering() {
        assert!(PeeringPriority::Low < PeeringPriority::Normal);
        assert!(PeeringPriority::Normal < PeeringPriority::High);
    }

    #[test]
    fn contact_creation() {
        let contact = Contact {
            identity_hash: [0xAB; 16],
            display_name: Some("Alice".into()),
            peering: PeeringPolicy { enabled: true, priority: PeeringPriority::High },
            added_at: 1710000000,
            last_seen: None,
            notes: None,
        };
        assert_eq!(contact.identity_hash, [0xAB; 16]);
        assert!(contact.peering.enabled);
    }

    #[test]
    fn contact_serde_round_trip() {
        let contact = Contact {
            identity_hash: [0x42; 16],
            display_name: Some("Bob".into()),
            peering: PeeringPolicy { enabled: true, priority: PeeringPriority::Normal },
            added_at: 1710000000,
            last_seen: Some(1710001000),
            notes: Some("My friend".into()),
        };
        let bytes = postcard::to_allocvec(&contact).unwrap();
        let decoded: Contact = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.identity_hash, contact.identity_hash);
        assert_eq!(decoded.display_name, contact.display_name);
        assert_eq!(decoded.peering.priority, PeeringPriority::Normal);
        assert_eq!(decoded.last_seen, Some(1710001000));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-contacts`
Expected: FAIL — `Contact` type not defined yet.

- [ ] **Step 3: Implement types**

```rust
use alloc::string::String;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

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
    /// Last time we had an active link to this peer.
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

impl Default for PeeringPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: PeeringPriority::Normal,
        }
    }
}

/// Reconnection aggressiveness levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeeringPriority {
    /// Announce-driven only. Reconnect when we passively hear them.
    Low,
    /// Moderate active probing. Path request every ~120s.
    Normal,
    /// Aggressive probing. Path request every ~30s.
    High,
}
```

- [ ] **Step 4: Update lib.rs**

```rust
pub mod contact;
pub use contact::{Contact, PeeringPolicy, PeeringPriority};
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-contacts`
Expected: All 4 tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-contacts/
git commit -m "feat(contacts): add Contact, PeeringPolicy, PeeringPriority types"
```

---

### Task 4: Implement `ContactStore`

**Files:**
- Create: `crates/harmony-contacts/src/store.rs`
- Modify: `crates/harmony-contacts/src/lib.rs`

- [ ] **Step 1: Write tests first**

Add to the bottom of `store.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::contact::{PeeringPolicy, PeeringPriority};

    fn make_contact(id_byte: u8, enabled: bool, priority: PeeringPriority) -> Contact {
        Contact {
            identity_hash: [id_byte; 16],
            display_name: None,
            peering: PeeringPolicy { enabled, priority },
            added_at: 1710000000,
            last_seen: None,
            notes: None,
        }
    }

    #[test]
    fn add_and_get() {
        let mut store = ContactStore::new();
        let contact = make_contact(0xAA, true, PeeringPriority::Normal);
        store.add(contact.clone()).unwrap();
        assert_eq!(store.len(), 1);
        assert!(store.contains(&[0xAA; 16]));
        assert_eq!(store.get(&[0xAA; 16]).unwrap().identity_hash, [0xAA; 16]);
    }

    #[test]
    fn add_duplicate_fails() {
        let mut store = ContactStore::new();
        store.add(make_contact(0xBB, true, PeeringPriority::Normal)).unwrap();
        let result = store.add(make_contact(0xBB, false, PeeringPriority::Low));
        assert!(matches!(result, Err(ContactError::AlreadyExists(_))));
    }

    #[test]
    fn remove_returns_contact() {
        let mut store = ContactStore::new();
        store.add(make_contact(0xCC, true, PeeringPriority::High)).unwrap();
        let removed = store.remove(&[0xCC; 16]);
        assert!(removed.is_some());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut store = ContactStore::new();
        assert!(store.remove(&[0xFF; 16]).is_none());
    }

    #[test]
    fn update_last_seen() {
        let mut store = ContactStore::new();
        store.add(make_contact(0xDD, true, PeeringPriority::Normal)).unwrap();
        store.update_last_seen(&[0xDD; 16], 999);
        assert_eq!(store.get(&[0xDD; 16]).unwrap().last_seen, Some(999));
    }

    #[test]
    fn peers_with_peering_enabled() {
        let mut store = ContactStore::new();
        store.add(make_contact(0x01, true, PeeringPriority::Normal)).unwrap();
        store.add(make_contact(0x02, false, PeeringPriority::Normal)).unwrap();
        store.add(make_contact(0x03, true, PeeringPriority::High)).unwrap();
        let enabled: Vec<_> = store.peers_with_peering_enabled().collect();
        assert_eq!(enabled.len(), 2);
    }

    #[test]
    fn peers_by_priority() {
        let mut store = ContactStore::new();
        store.add(make_contact(0x01, true, PeeringPriority::Low)).unwrap();
        store.add(make_contact(0x02, true, PeeringPriority::High)).unwrap();
        store.add(make_contact(0x03, true, PeeringPriority::High)).unwrap();
        let high: Vec<_> = store.peers_by_priority(PeeringPriority::High).collect();
        assert_eq!(high.len(), 2);
        let low: Vec<_> = store.peers_by_priority(PeeringPriority::Low).collect();
        assert_eq!(low.len(), 1);
    }

    #[test]
    fn empty_store() {
        let store = ContactStore::new();
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        assert!(store.get(&[0x00; 16]).is_none());
    }

    #[test]
    fn serialize_round_trip() {
        let mut store = ContactStore::new();
        store.add(make_contact(0xAA, true, PeeringPriority::High)).unwrap();
        store.add(make_contact(0xBB, false, PeeringPriority::Low)).unwrap();

        let bytes = store.serialize();
        let restored = ContactStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.len(), 2);
        assert!(restored.contains(&[0xAA; 16]));
        assert!(restored.contains(&[0xBB; 16]));
    }

    #[test]
    fn deserialize_bad_data() {
        let result = ContactStore::deserialize(&[0xFF, 0xFF, 0xFF]);
        assert!(matches!(result, Err(ContactError::DeserializeError(_))));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-contacts`
Expected: FAIL — `ContactStore` not defined.

- [ ] **Step 3: Implement ContactStore**

```rust
use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

use crate::contact::{Contact, PeeringPriority};
use crate::error::ContactError;

/// Serialization format version. First byte of serialized data.
const FORMAT_VERSION: u8 = 1;

/// Persistent store of intentional peer relationships.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContactStore {
    contacts: HashMap<IdentityHash, Contact>,
}

impl ContactStore {
    pub fn new() -> Self {
        Self {
            contacts: HashMap::new(),
        }
    }

    pub fn add(&mut self, contact: Contact) -> Result<(), ContactError> {
        if self.contacts.contains_key(&contact.identity_hash) {
            return Err(ContactError::AlreadyExists(contact.identity_hash));
        }
        self.contacts.insert(contact.identity_hash, contact);
        Ok(())
    }

    pub fn remove(&mut self, id: &IdentityHash) -> Option<Contact> {
        self.contacts.remove(id)
    }

    pub fn get(&self, id: &IdentityHash) -> Option<&Contact> {
        self.contacts.get(id)
    }

    pub fn get_mut(&mut self, id: &IdentityHash) -> Option<&mut Contact> {
        self.contacts.get_mut(id)
    }

    pub fn contains(&self, id: &IdentityHash) -> bool {
        self.contacts.contains_key(id)
    }

    pub fn update_last_seen(&mut self, id: &IdentityHash, timestamp: u64) {
        if let Some(contact) = self.contacts.get_mut(id) {
            contact.last_seen = Some(timestamp);
        }
    }

    pub fn peers_with_peering_enabled(&self) -> impl Iterator<Item = &Contact> {
        self.contacts.values().filter(|c| c.peering.enabled)
    }

    pub fn peers_by_priority(&self, priority: PeeringPriority) -> impl Iterator<Item = &Contact> {
        self.contacts
            .values()
            .filter(move |c| c.peering.enabled && c.peering.priority == priority)
    }

    pub fn len(&self) -> usize {
        self.contacts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.contacts.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&IdentityHash, &Contact)> {
        self.contacts.iter()
    }

    pub fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        buf.extend_from_slice(&postcard::to_allocvec(self).expect("serialize"));
        buf
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, ContactError> {
        if data.is_empty() {
            return Err(ContactError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(ContactError::DeserializeError("unsupported format version"));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| ContactError::DeserializeError("postcard decode failed"))
    }
}

impl Default for ContactStore {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Update lib.rs**

Add to `lib.rs`:

```rust
pub mod store;
pub use store::ContactStore;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-contacts`
Expected: All 14 tests pass (4 from Task 3 + 10 from Task 4).

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-contacts/
git commit -m "feat(contacts): implement ContactStore with CRUD, queries, persistence"
```

---

### Task 5: Scaffold `harmony-peers` crate with event/action types

**Files:**
- Create: `crates/harmony-peers/Cargo.toml`
- Create: `crates/harmony-peers/src/lib.rs`
- Create: `crates/harmony-peers/src/event.rs`
- Create: `crates/harmony-peers/src/state.rs`
- Modify: `Cargo.toml` (workspace)

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "harmony-peers"
description = "Sans-I/O peering lifecycle manager for the Harmony network"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[dependencies]
harmony-identity = { workspace = true }
harmony-contacts = { workspace = true }
hashbrown = { workspace = true }

[features]
default = ["std"]
std = [
    "harmony-identity/std",
    "harmony-contacts/std",
]

[dev-dependencies]
```

- [ ] **Step 2: Create event.rs**

```rust
use harmony_identity::IdentityHash;

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
    /// Initiate a Link to this identity (path is known).
    InitiateLink { identity_hash: IdentityHash },
    /// Send a path request for this identity's destination (path unknown).
    SendPathRequest { identity_hash: IdentityHash },
    /// Drop/close an active Link.
    CloseLink { identity_hash: IdentityHash },
    /// Update last_seen timestamp on the contact.
    UpdateLastSeen { identity_hash: IdentityHash, timestamp: u64 },
}
```

- [ ] **Step 3: Create state.rs**

```rust
use harmony_contacts::PeeringPriority;

/// Per-peer connection lifecycle state.
#[derive(Debug, Clone)]
pub struct PeerState {
    pub status: PeerStatus,
    pub priority: PeeringPriority,
    /// When we last sent a path request (unix timestamp seconds).
    pub last_probe: Option<u64>,
    /// When we last had an active link.
    pub last_seen: Option<u64>,
    /// When the current Connecting attempt started (for timeout).
    pub connecting_since: Option<u64>,
    /// Consecutive failed connection attempts (drives backoff).
    pub retry_count: u32,
}

impl PeerState {
    pub fn new(priority: PeeringPriority) -> Self {
        Self {
            status: PeerStatus::Searching,
            priority,
            last_probe: None,
            last_seen: None,
            connecting_since: None,
            retry_count: 0,
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            status: PeerStatus::Disabled,
            priority: PeeringPriority::Normal,
            last_probe: None,
            last_seen: None,
            connecting_since: None,
            retry_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerStatus {
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

- [ ] **Step 4: Create lib.rs**

```rust
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

pub mod event;
pub mod state;

pub use event::{PeerAction, PeerEvent};
pub use state::{PeerState, PeerStatus};
```

- [ ] **Step 5: Add workspace entries**

In root `Cargo.toml`, add `"crates/harmony-peers"` to the `members` list and add:

```toml
harmony-peers = { path = "crates/harmony-peers", default-features = false }
```

- [ ] **Step 6: Build**

Run: `cargo check -p harmony-peers`
Expected: Compiles successfully.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-peers/ Cargo.toml
git commit -m "feat(peers): scaffold harmony-peers with event/action/state types"
```

---

### Task 6: Implement `PeerManager` core state machine

**Files:**
- Create: `crates/harmony-peers/src/manager.rs`
- Modify: `crates/harmony-peers/src/lib.rs`

- [ ] **Step 1: Write tests first**

Add to the bottom of `manager.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_contacts::{Contact, ContactStore, PeeringPolicy, PeeringPriority};

    fn make_store_with_contact(
        id_byte: u8,
        enabled: bool,
        priority: PeeringPriority,
    ) -> ContactStore {
        let mut store = ContactStore::new();
        store
            .add(Contact {
                identity_hash: [id_byte; 16],
                display_name: None,
                peering: PeeringPolicy { enabled, priority },
                added_at: 1000,
                last_seen: None,
                notes: None,
            })
            .unwrap();
        store
    }

    #[test]
    fn contact_added_high_priority_probes_on_tick() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xAA, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xAA; 16] },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xAA; 16],
        }));
    }

    #[test]
    fn contact_added_low_priority_no_probe() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xBB, true, PeeringPriority::Low);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xBB; 16] },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(!actions.iter().any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
    }

    #[test]
    fn announce_received_triggers_link_initiation() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xCC, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xCC; 16] },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::AnnounceReceived { identity_hash: [0xCC; 16] },
            &store,
        );
        assert!(actions.contains(&PeerAction::InitiateLink {
            identity_hash: [0xCC; 16],
        }));
    }

    #[test]
    fn announce_for_unknown_contact_ignored() {
        let mut mgr = PeerManager::new();
        let store = ContactStore::new();
        let actions = mgr.on_event(
            PeerEvent::AnnounceReceived { identity_hash: [0xFF; 16] },
            &store,
        );
        assert!(actions.is_empty());
    }

    #[test]
    fn contact_added_normal_priority_probes_on_tick() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xA1, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xA1; 16] },
            &store,
        );
        // First tick — should probe (no last_probe yet)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA1; 16],
        }));
        // 60s later — too soon for Normal (120s base)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1060 }, &store);
        assert!(!actions.iter().any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
        // 121s later — past 120s interval
        let actions = mgr.on_event(PeerEvent::Tick { now: 1121 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA1; 16],
        }));
    }

    #[test]
    fn link_established_transitions_to_connected() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xDD, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xDD; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived { identity_hash: [0xDD; 16] },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0xDD; 16], now: 5000 },
            &store,
        );
        assert!(actions.contains(&PeerAction::UpdateLastSeen {
            identity_hash: [0xDD; 16],
            timestamp: 5000,
        }));
        assert_eq!(mgr.peers.get(&[0xDD; 16]).unwrap().status, PeerStatus::Connected);
    }

    #[test]
    fn link_closed_transitions_to_searching() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xEE, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xEE; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0xEE; 16], now: 5000 },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkClosed { identity_hash: [0xEE; 16] },
            &store,
        );
        let peer = mgr.peers.get(&[0xEE; 16]).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
    }

    #[test]
    fn contact_removed_closes_link() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x11, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x11; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0x11; 16], now: 5000 },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::ContactRemoved { identity_hash: [0x11; 16] },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0x11; 16],
        }));
        assert!(!mgr.peers.contains_key(&[0x11; 16]));
    }

    #[test]
    fn disabled_policy_closes_active_link() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0x22, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x22; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0x22; 16], now: 5000 },
            &store,
        );
        // Disable peering
        store.get_mut(&[0x22; 16]).unwrap().peering.enabled = false;
        let actions = mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x22; 16] },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0x22; 16],
        }));
        assert_eq!(mgr.peers.get(&[0x22; 16]).unwrap().status, PeerStatus::Disabled);
    }

    #[test]
    fn priority_change_updates_probe_interval() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0xA2, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xA2; 16] },
            &store,
        );
        // Probe at t=1000
        mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        // Change to High priority
        store.get_mut(&[0xA2; 16]).unwrap().peering.priority = PeeringPriority::High;
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0xA2; 16] },
            &store,
        );
        // At t=1031 — past High interval (30s) but not Normal (120s)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1031 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA2; 16],
        }));
    }

    #[test]
    fn re_enable_policy_starts_searching() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0x33, false, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x33; 16] },
            &store,
        );
        assert_eq!(mgr.peers.get(&[0x33; 16]).unwrap().status, PeerStatus::Disabled);

        store.get_mut(&[0x33; 16]).unwrap().peering.enabled = true;
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x33; 16] },
            &store,
        );
        assert_eq!(mgr.peers.get(&[0x33; 16]).unwrap().status, PeerStatus::Searching);
    }

    #[test]
    fn connected_peer_no_probing() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x44, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x44; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0x44; 16], now: 5000 },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 9999 }, &store);
        assert!(!actions.iter().any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
    }

    #[test]
    fn probe_interval_respected() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x55, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x55; 16] },
            &store,
        );
        // First tick at t=1000 — should probe (no last_probe yet)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x55; 16],
        }));

        // Tick at t=1010 — too soon for High (30s base)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1010 }, &store);
        assert!(!actions.iter().any(|a| matches!(a, PeerAction::SendPathRequest { .. })));

        // Tick at t=1031 — past 30s interval
        let actions = mgr.on_event(PeerEvent::Tick { now: 1031 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x55; 16],
        }));
    }

    #[test]
    fn backoff_increases_probe_interval() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x66, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x66; 16] },
            &store,
        );

        // Simulate 3 failed connection attempts
        for _ in 0..3 {
            mgr.on_event(
                PeerEvent::AnnounceReceived { identity_hash: [0x66; 16] },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed { identity_hash: [0x66; 16] },
                &store,
            );
        }
        assert_eq!(mgr.peers.get(&[0x66; 16]).unwrap().retry_count, 3);

        // After 3 retries, interval = min(30 * 2^3, 600) = 240s
        mgr.on_event(PeerEvent::Tick { now: 10000 }, &store);
        let actions = mgr.on_event(PeerEvent::Tick { now: 10100 }, &store);
        // 100s later — should NOT probe (240s interval)
        assert!(!actions.iter().any(|a| matches!(a, PeerAction::SendPathRequest { .. })));

        let actions = mgr.on_event(PeerEvent::Tick { now: 10241 }, &store);
        // 241s later — should probe
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x66; 16],
        }));
    }

    #[test]
    fn backoff_caps_at_600s() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x77, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x77; 16] },
            &store,
        );

        // Simulate 10 failed attempts — 30 * 2^10 = 30720, capped at 600
        for _ in 0..10 {
            mgr.on_event(
                PeerEvent::AnnounceReceived { identity_hash: [0x77; 16] },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed { identity_hash: [0x77; 16] },
                &store,
            );
        }

        // Probe at t=0
        mgr.on_event(PeerEvent::Tick { now: 50000 }, &store);
        // At t+601 should probe (600s cap)
        let actions = mgr.on_event(PeerEvent::Tick { now: 50601 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x77; 16],
        }));
    }

    #[test]
    fn link_established_resets_retry_count() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x88, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x88; 16] },
            &store,
        );
        // Fail 5 times
        for _ in 0..5 {
            mgr.on_event(
                PeerEvent::AnnounceReceived { identity_hash: [0x88; 16] },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed { identity_hash: [0x88; 16] },
                &store,
            );
        }
        assert_eq!(mgr.peers.get(&[0x88; 16]).unwrap().retry_count, 5);

        // Succeed
        mgr.on_event(
            PeerEvent::AnnounceReceived { identity_hash: [0x88; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished { identity_hash: [0x88; 16], now: 90000 },
            &store,
        );
        assert_eq!(mgr.peers.get(&[0x88; 16]).unwrap().retry_count, 0);
    }

    #[test]
    fn connecting_timeout_transitions_to_searching() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x99, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged { identity_hash: [0x99; 16] },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived { identity_hash: [0x99; 16] },
            &store,
        );
        assert_eq!(mgr.peers.get(&[0x99; 16]).unwrap().status, PeerStatus::Connecting);

        // First tick stamps connecting_since
        mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert_eq!(mgr.peers.get(&[0x99; 16]).unwrap().status, PeerStatus::Connecting);

        // Tick 61s later — should timeout
        mgr.on_event(PeerEvent::Tick { now: 1061 }, &store);
        let peer = mgr.peers.get(&[0x99; 16]).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-peers`
Expected: FAIL — `PeerManager` not defined.

- [ ] **Step 3: Implement PeerManager**

```rust
use alloc::vec::Vec;
use hashbrown::HashMap;
use harmony_contacts::{ContactStore, PeeringPriority};
use harmony_identity::IdentityHash;

use crate::event::{PeerAction, PeerEvent};
use crate::state::{PeerState, PeerStatus};

/// Base probe interval in seconds for High priority peers.
const PROBE_INTERVAL_HIGH: u64 = 30;
/// Base probe interval in seconds for Normal priority peers.
const PROBE_INTERVAL_NORMAL: u64 = 120;
/// Maximum probe interval in seconds (backoff cap).
const PROBE_INTERVAL_MAX: u64 = 600;
/// Timeout for Connecting state (safety net, seconds).
const CONNECTING_TIMEOUT: u64 = 60;

/// Sans-I/O state machine that maintains persistent Links to contacts
/// with peering enabled.
pub struct PeerManager {
    pub(crate) peers: HashMap<IdentityHash, PeerState>,
}

impl PeerManager {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    /// Process an event. Returns zero or more actions for the caller to execute.
    pub fn on_event(
        &mut self,
        event: PeerEvent,
        contacts: &ContactStore,
    ) -> Vec<PeerAction> {
        let mut actions = Vec::new();

        match event {
            PeerEvent::ContactChanged { identity_hash } => {
                self.handle_contact_changed(identity_hash, contacts, &mut actions);
            }
            PeerEvent::ContactRemoved { identity_hash } => {
                if let Some(peer) = self.peers.remove(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        actions.push(PeerAction::CloseLink { identity_hash });
                    }
                }
            }
            PeerEvent::AnnounceReceived { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if peer.status == PeerStatus::Searching {
                        peer.status = PeerStatus::Connecting;
                        peer.connecting_since = None; // stamped on first Tick
                        actions.push(PeerAction::InitiateLink { identity_hash });
                    }
                }
            }
            PeerEvent::LinkEstablished { identity_hash, now } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    peer.status = PeerStatus::Connected;
                    peer.retry_count = 0;
                    peer.last_seen = Some(now);
                    peer.connecting_since = None;
                    actions.push(PeerAction::UpdateLastSeen {
                        identity_hash,
                        timestamp: now,
                    });
                }
            }
            PeerEvent::LinkClosed { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = peer.retry_count.saturating_add(1);
                    }
                }
            }
            PeerEvent::Tick { now } => {
                self.handle_tick(now, &mut actions);
            }
        }

        actions
    }

    fn handle_contact_changed(
        &mut self,
        identity_hash: IdentityHash,
        contacts: &ContactStore,
        actions: &mut Vec<PeerAction>,
    ) {
        let contact = match contacts.get(&identity_hash) {
            Some(c) => c,
            None => return,
        };

        if contact.peering.enabled {
            match self.peers.get_mut(&identity_hash) {
                Some(peer) => {
                    // Update priority; if was Disabled, transition to Searching
                    peer.priority = contact.peering.priority;
                    if peer.status == PeerStatus::Disabled {
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = 0;
                    }
                }
                None => {
                    self.peers.insert(
                        identity_hash,
                        PeerState::new(contact.peering.priority),
                    );
                }
            }
        } else {
            // Disabled
            match self.peers.get_mut(&identity_hash) {
                Some(peer) => {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        actions.push(PeerAction::CloseLink { identity_hash });
                    }
                    peer.status = PeerStatus::Disabled;
                }
                None => {
                    self.peers.insert(identity_hash, PeerState::new_disabled());
                }
            }
        }
    }

    fn handle_tick(&mut self, now: u64, actions: &mut Vec<PeerAction>) {
        for (id, peer) in self.peers.iter_mut() {
            match peer.status {
                PeerStatus::Searching => {
                    let interval = Self::probe_interval(peer.priority, peer.retry_count);
                    if interval == 0 {
                        continue; // Low priority — no active probing
                    }
                    let should_probe = match peer.last_probe {
                        None => true,
                        Some(last) => now.saturating_sub(last) >= interval,
                    };
                    if should_probe {
                        peer.last_probe = Some(now);
                        actions.push(PeerAction::SendPathRequest { identity_hash: *id });
                    }
                }
                PeerStatus::Connecting => {
                    // Safety net: timeout if no LinkEstablished/LinkClosed arrives.
                    // Stamp connecting_since on first observation.
                    match peer.connecting_since {
                        None => {
                            peer.connecting_since = Some(now);
                        }
                        Some(since) => {
                            if now.saturating_sub(since) >= CONNECTING_TIMEOUT {
                                peer.status = PeerStatus::Searching;
                                peer.retry_count = peer.retry_count.saturating_add(1);
                                peer.connecting_since = None;
                            }
                        }
                    }
                }
                PeerStatus::Connected | PeerStatus::Disabled => {}
            }
        }
    }

    fn probe_interval(priority: PeeringPriority, retry_count: u32) -> u64 {
        let base = match priority {
            PeeringPriority::Low => return 0, // No active probing
            PeeringPriority::Normal => PROBE_INTERVAL_NORMAL,
            PeeringPriority::High => PROBE_INTERVAL_HIGH,
        };
        let backoff = base.saturating_mul(1u64 << retry_count.min(20));
        backoff.min(PROBE_INTERVAL_MAX)
    }
}

impl Default for PeerManager {
    fn default() -> Self {
        Self::new()
    }
}
```

- [ ] **Step 4: Update lib.rs**

Add to `lib.rs`:

```rust
pub mod manager;
pub use manager::PeerManager;
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-peers`
Expected: All 14 tests pass.

- [ ] **Step 6: Run full workspace tests**

Run: `cargo test --workspace`
Expected: All tests pass across all crates.

- [ ] **Step 7: Run clippy**

Run: `cargo clippy --workspace`
Expected: No warnings.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-peers/
git commit -m "feat(peers): implement PeerManager state machine with probing and backoff"
```

---

### Task 7: Final quality gate

- [ ] **Step 1: Full workspace test**

Run: `cargo test --workspace`
Expected: All tests pass (365+ existing + ~28 new).

- [ ] **Step 2: Format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

- [ ] **Step 3: Clippy**

Run: `cargo clippy --workspace`
Expected: Clean.

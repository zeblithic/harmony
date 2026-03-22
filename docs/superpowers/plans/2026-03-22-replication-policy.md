# Replication Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-u18`.

**Goal:** Enable tunnel peers to exchange symmetric storage quotas for each other's encrypted-durable content, with push-based replication and self-retrieval for disaster recovery.

**Architecture:** Four layers: (1) `ReplicationPolicy` on `Contact` defines the quota, (2) `ReplicaStore` trait stores encrypted books for other peers, (3) tunnel frame tag `0x03` carries replication messages (PUSH/PULL/STATUS), (4) runtime scheduling pushes unreplicated content on a timer. Public content is excluded — the network handles it naturally.

**Tech Stack:** Rust, `harmony-contacts` (ReplicationPolicy), `harmony-content` (ReplicaStore trait), `harmony-tunnel` (frame tag + event/action), `harmony-node` (scheduling)

**Spec:** `docs/superpowers/specs/2026-03-22-replication-policy-design.md`

---

## File Structure

```
crates/harmony-contacts/src/
└── contact.rs           — Add ReplicationPolicy, replication field on Contact

crates/harmony-content/src/
├── replica.rs           — NEW: ReplicaStore trait + MemoryReplicaStore
└── lib.rs               — Export replica module

crates/harmony-tunnel/src/
├── frame.rs             — Add FrameTag::Replication (0x03)
├── event.rs             — Add SendReplication / ReplicationReceived
├── session.rs           — Handle replication frames
└── replication.rs       — NEW: ReplicationMessage encode/decode

crates/harmony-node/src/
├── runtime.rs           — Add ReplicaPush/ReplicaReceived events, scheduling
└── event_loop.rs        — Route replication actions to tunnel senders
```

---

### Task 1: ReplicationPolicy on Contact

**Files:**
- Modify: `crates/harmony-contacts/src/contact.rs`
- Modify: `crates/harmony-contacts/src/store.rs` (bump FORMAT_VERSION)
- Modify: `crates/harmony-contacts/src/lib.rs` (re-export)

- [ ] **Step 1: Write test for ReplicationPolicy serialization**

```rust
#[test]
fn replication_policy_roundtrip() {
    let policy = ReplicationPolicy { quota_bytes: 50 * 1024 * 1024 * 1024 };
    let bytes = postcard::to_allocvec(&policy).unwrap();
    let decoded: ReplicationPolicy = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(decoded.quota_bytes, 50 * 1024 * 1024 * 1024);
}
```

- [ ] **Step 2: Implement ReplicationPolicy**

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplicationPolicy {
    /// Symmetric storage quota in bytes. Both peers provision this
    /// amount for each other's encrypted-durable content.
    pub quota_bytes: u64,
}
```

Add to `Contact`:
```rust
    /// Replication policy: encrypted backup delegation.
    /// None = no replication. Some = active with specified quota.
    pub replication: Option<ReplicationPolicy>,
```

Update ALL test helpers that construct `Contact` to include `replication: None`.

- [ ] **Step 3: Bump FORMAT_VERSION in store.rs**

Bump from current value to next (adding a field changes postcard layout).

- [ ] **Step 4: Re-export from lib.rs**

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-contacts -p harmony-peers`
Expected: All pass (peers crate also constructs Contacts in tests).

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(contacts): add ReplicationPolicy for encrypted backup delegation"
```

---

### Task 2: ReplicaStore trait and MemoryReplicaStore

**Files:**
- Create: `crates/harmony-content/src/replica.rs`
- Modify: `crates/harmony-content/src/lib.rs`

- [ ] **Step 1: Write tests for replica store**

```rust
#[test]
fn store_and_retrieve() {
    let mut store = MemoryReplicaStore::new();
    let peer = [0xAA; 16];
    let cid = [0xBB; 32];
    let data = vec![1, 2, 3, 4, 5];
    store.store(peer, cid, data.clone(), 1024).unwrap();
    assert_eq!(store.retrieve(&peer, &cid), Some(data));
    assert_eq!(store.usage(&peer), 5);
}

#[test]
fn quota_evicts_oldest() {
    let mut store = MemoryReplicaStore::new();
    let peer = [0xAA; 16];
    // Store 3 books of 100 bytes each
    for i in 0..3u8 {
        let mut cid = [0u8; 32];
        cid[0] = i;
        store.store(peer, cid, vec![i; 100], 300).unwrap();
    }
    assert_eq!(store.usage(&peer), 300);

    // Store a 4th book — should evict the oldest (cid[0]=0)
    let mut cid4 = [0u8; 32];
    cid4[0] = 3;
    store.store(peer, cid4, vec![3; 100], 300).unwrap();
    assert_eq!(store.usage(&peer), 300);
    // Oldest (cid[0]=0) should be evicted
    let mut cid0 = [0u8; 32];
    cid0[0] = 0;
    assert_eq!(store.retrieve(&peer, &cid0), None);
    // Newest should still be there
    assert_eq!(store.retrieve(&peer, &cid4), Some(vec![3; 100]));
}

#[test]
fn book_exceeding_quota_rejected() {
    let mut store = MemoryReplicaStore::new();
    let peer = [0xAA; 16];
    let cid = [0xBB; 32];
    let result = store.store(peer, cid, vec![0; 1000], 500);
    assert!(result.is_err());
}

#[test]
fn idempotent_store() {
    let mut store = MemoryReplicaStore::new();
    let peer = [0xAA; 16];
    let cid = [0xBB; 32];
    store.store(peer, cid, vec![1; 100], 1024).unwrap();
    store.store(peer, cid, vec![1; 100], 1024).unwrap(); // same CID
    assert_eq!(store.usage(&peer), 100); // not doubled
}
```

- [ ] **Step 2: Implement ReplicaStore trait and MemoryReplicaStore**

```rust
use alloc::vec::Vec;
use harmony_identity::IdentityHash;

/// Error from replica storage operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplicaError {
    /// A single book exceeds the entire quota.
    ExceedsQuota { book_size: usize, quota: u64 },
}

/// Stores encrypted books on behalf of other peers.
///
/// Keyed by (peer_identity, content_id). Separate from the node's
/// own content store — replicated data doesn't compete with local cache.
pub trait ReplicaStore {
    fn store(
        &mut self,
        peer: IdentityHash,
        cid: [u8; 32],
        data: Vec<u8>,
        quota: u64,
    ) -> Result<(), ReplicaError>;

    fn retrieve(&self, peer: &IdentityHash, cid: &[u8; 32]) -> Option<Vec<u8>>;

    fn usage(&self, peer: &IdentityHash) -> u64;

    fn evict_to(&mut self, peer: &IdentityHash, target_bytes: u64);
}
```

`MemoryReplicaStore` uses `HashMap<IdentityHash, BTreeMap<u64, ([u8; 32], Vec<u8>)>>` where the `u64` key is an insertion counter (for oldest-first eviction). Or simpler: `HashMap<IdentityHash, Vec<ReplicaEntry>>` where entries are ordered by insertion time.

- [ ] **Step 3: Export from lib.rs**

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-content`

- [ ] **Step 5: Commit**

```bash
git commit -m "feat(content): ReplicaStore trait for encrypted backup delegation"
```

---

### Task 3: Replication frame tag and message types

**Files:**
- Modify: `crates/harmony-tunnel/src/frame.rs` (add FrameTag::Replication)
- Create: `crates/harmony-tunnel/src/replication.rs` (message encode/decode)
- Modify: `crates/harmony-tunnel/src/event.rs` (add SendReplication / ReplicationReceived)
- Modify: `crates/harmony-tunnel/src/session.rs` (handle replication frames)
- Modify: `crates/harmony-tunnel/src/lib.rs` (export)

- [ ] **Step 1: Add FrameTag::Replication**

In `frame.rs`, add `Replication = 0x03` to the enum and `from_byte` match.

- [ ] **Step 2: Create replication.rs with message types**

```rust
use alloc::vec::Vec;
use crate::error::TunnelError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ReplicationOp {
    Push = 0x01,
    Pull = 0x02,
    PullResponse = 0x03,
    Status = 0x04,
    StatusResponse = 0x05,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplicationMessage {
    pub op: ReplicationOp,
    pub cid: [u8; 32],
    pub payload: Vec<u8>,
}

impl ReplicationMessage {
    pub fn encode(&self) -> Vec<u8> { ... }
    pub fn decode(data: &[u8]) -> Result<Self, TunnelError> { ... }

    // Convenience constructors
    pub fn push(cid: [u8; 32], data: Vec<u8>) -> Self { ... }
    pub fn pull(cid: [u8; 32]) -> Self { ... }
    pub fn pull_response(cid: [u8; 32], data: Vec<u8>) -> Self { ... }
    pub fn status() -> Self { ... }
    pub fn status_response(used: u64, quota: u64) -> Self { ... }
}
```

Add tests for encode/decode roundtrip of each op.

- [ ] **Step 3: Add TunnelEvent/TunnelAction variants**

In `event.rs`:
```rust
// TunnelEvent:
    SendReplication { message: Vec<u8>, now_ms: u64 },

// TunnelAction:
    ReplicationReceived { message: Vec<u8> },
```

The message is the encoded `ReplicationMessage` bytes. Keeping it as `Vec<u8>` preserves the tunnel's transport-agnosticism — it doesn't parse replication semantics, just encrypts/decrypts.

- [ ] **Step 4: Handle replication frames in session.rs**

In `handle_encrypted_frame`, add:
```rust
FrameTag::Replication => Ok(vec![TunnelAction::ReplicationReceived {
    message: frame.payload,
}]),
```

In `handle_send`, add a `handle_send_replication` path (or extend `handle_send` with a `FrameTag::Replication` variant).

In `handle_event`, add:
```rust
TunnelEvent::SendReplication { message, now_ms } => {
    self.last_sent_ms = now_ms;
    self.handle_send(FrameTag::Replication, message)
}
```

- [ ] **Step 5: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-tunnel`

- [ ] **Step 6: Commit**

```bash
git commit -m "feat(tunnel): replication frame tag (0x03) and message protocol"
```

---

### Task 4: Runtime replication scheduling

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml` (if needed)

- [ ] **Step 1: Add RuntimeAction/RuntimeEvent variants**

```rust
// RuntimeAction:
    ReplicaPush {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        data: Vec<u8>,
    },

// RuntimeEvent:
    ReplicaReceived {
        peer_identity: [u8; 16],
        cid: [u8; 32],
        data: Vec<u8>,
    },
```

- [ ] **Step 2: Add ReplicaStore and tracking to NodeRuntime**

Add fields:
```rust
    replica_store: harmony_content::replica::MemoryReplicaStore,
    replicated_set: HashSet<([u8; 16], [u8; 32])>, // (peer, cid) pairs already pushed
    ticks_since_replica_scan: u32,
```

- [ ] **Step 3: Handle ReplicaReceived in push_event**

When a PUSH arrives from a peer:
```rust
RuntimeEvent::ReplicaReceived { peer_identity, cid, data } => {
    let quota = self.contact_store.get(&peer_identity)
        .and_then(|c| c.replication.as_ref())
        .map(|r| r.quota_bytes)
        .unwrap_or(0);
    if quota > 0 {
        let _ = self.replica_store.store(peer_identity, cid, data, quota);
    }
}
```

- [ ] **Step 4: Add replication scan in tick()**

Every 240 ticks (~60s at 250ms tick interval):
```rust
self.ticks_since_replica_scan += 1;
if self.ticks_since_replica_scan >= 240 {
    self.ticks_since_replica_scan = 0;
    // For each contact with replication enabled and active tunnel:
    // Find one encrypted-durable book not in replicated_set
    // Emit ReplicaPush action
}
```

The scan iterates the local BookStore for encrypted-durable CIDs (check `ContentId::flags().encrypted && !ContentId::flags().ephemeral`), skips any in `replicated_set` for this peer, and emits one `ReplicaPush` per peer per scan cycle.

- [ ] **Step 5: Route ReplicaPush in event loop**

In `event_loop.rs`, handle `RuntimeAction::ReplicaPush`:
```rust
RuntimeAction::ReplicaPush { peer_identity, cid, data } => {
    // Find the tunnel sender for this peer (lookup by identity hash)
    // Send ReplicationMessage::push(cid, data) as a replication frame
}
```

This requires the `tunnel_identities` mapping (interface_name → identity_hash) to be reversible, or a direct `identity_hash → TunnelSender` lookup.

- [ ] **Step 6: Write tests**

```rust
#[test]
fn replica_received_stores_within_quota() {
    let (mut rt, _) = make_runtime();
    // Add contact with replication policy
    // Push ReplicaReceived event
    // Verify replica store has the data
}

#[test]
fn replica_received_rejected_without_policy() {
    // Push ReplicaReceived for a peer without replication
    // Verify nothing stored
}
```

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(node): replication scheduling and replica store integration"
```

---

### Task 5: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-contacts -p harmony-content -p harmony-tunnel -p harmony-node`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: clippy fixes for replication policy"
```

---

## Summary

| Task | Description | Crate | Key Output |
|------|-------------|-------|------------|
| 1 | ReplicationPolicy on Contact | harmony-contacts | `quota_bytes` field, FORMAT_VERSION bump |
| 2 | ReplicaStore trait + MemoryReplicaStore | harmony-content | Store/retrieve/evict encrypted books for peers |
| 3 | Replication frame tag + message types | harmony-tunnel | FrameTag::Replication (0x03), PUSH/PULL/STATUS messages |
| 4 | Runtime scheduling + event handling | harmony-node | Push unreplicated content on timer, store incoming replicas |
| 5 | Cleanup | all | Clippy clean, workspace tests pass |

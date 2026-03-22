# Replication Policy: Encrypted Backup Delegation to Tunnel Peers

**Date:** 2026-03-22
**Status:** Draft
**Scope:** `harmony-contacts` (ReplicationPolicy), `harmony-content` (ReplicaStore trait), `harmony-tunnel` (replication frame tag), `harmony-node` (scheduling + event handling)
**Bead:** harmony-u18

## Overview

Tunnel peers are more than routing nodes — they're trusted storage delegates. Alice and Bob can provision symmetric storage quotas for each other's **encrypted-durable** content. Alice pushes encrypted books to Bob over their existing tunnel; Bob stores them opaquely and serves them back on request.

**Public content is not part of this design.** Public-durable content flows naturally through the content-addressed network — it's cached by demand and persisted in public libraries. Replication policy applies exclusively to private/encrypted data that wouldn't otherwise survive a device failure.

### Key Properties

- **Symmetric quota:** Both peers provision the same amount of space. Alice gives Bob 50GB, Bob gives Alice 50GB. Fair trade, self-regulating.
- **Opaque storage:** Bob never decrypts Alice's content. He stores ciphertext and serves it on valid request.
- **Newest-first eviction:** When quota is exceeded, oldest replicated books are evicted. Most recent data is most valuable for disaster recovery.
- **Idempotent push:** Content is addressed by CID. Pushing the same book twice is a no-op.
- **No sync protocol:** One-directional push, not bidirectional sync. No conflict resolution needed — CID-addressed content is immutable.

### What this design does NOT cover (deferred)

- **Token-gated serving** (harmony-5zt): Allowing authorized third parties to retrieve Alice's encrypted content from Bob by presenting a self-certifying bearer token. Deferred to a follow-up bead.
- **Quota negotiation protocol:** v1 sets quota on both sides manually. Dynamic negotiation is future work.
- **Selective replication by tag/namespace:** v1 replicates all encrypted-durable content. Per-item selection is future work.

## Section 1: ReplicationPolicy on Contact

**Where:** `crates/harmony-contacts/src/contact.rs`

Add `ReplicationPolicy` to the contact model:

```rust
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplicationPolicy {
    /// Symmetric storage quota in bytes. Both peers provision this
    /// amount for each other's encrypted-durable content.
    /// Example: 50 * 1024 * 1024 * 1024 = 50 GB.
    pub quota_bytes: u64,
}
```

Add to `Contact`:

```rust
pub struct Contact {
    // ... existing fields ...
    /// Replication policy: encrypted backup delegation.
    /// None = no replication. Some = active with specified quota.
    pub replication: Option<ReplicationPolicy>,
}
```

`None` means no replication relationship. `Some(policy)` with `quota_bytes > 0` means active replication. The quota is symmetric by convention — both sides set the same value.

Bump `ContactStore::FORMAT_VERSION` if adding a new field to Contact (postcard wire format change).

## Section 2: ReplicaStore

**Where:** New `ReplicaStore` trait in `crates/harmony-content/` (or a new crate if cleaner)

Stores encrypted books on behalf of other peers, keyed by `(identity_hash, content_id)`. Separate from the node's own content store — replicated data doesn't compete with local cache.

```rust
pub trait ReplicaStore {
    /// Store a book for a peer, respecting quota.
    /// Returns Err if the book exceeds the quota even after eviction.
    fn store(&mut self, peer: IdentityHash, cid: [u8; 32], data: Vec<u8>, quota: u64)
        -> Result<(), ReplicaError>;

    /// Retrieve a book stored for a peer.
    fn retrieve(&self, peer: IdentityHash, cid: &[u8; 32]) -> Option<Vec<u8>>;

    /// Current bytes stored for a peer.
    fn usage(&self, peer: &IdentityHash) -> u64;

    /// Evict oldest books for a peer until usage <= target_bytes.
    fn evict_to(&mut self, peer: &IdentityHash, target_bytes: u64);
}
```

**Quota enforcement:** On `store()`, if `usage(peer) + data.len() > quota`, evict oldest books until there's room. If a single book exceeds the entire quota, reject it.

**Implementations:**
- `MemoryReplicaStore` — in-memory `HashMap` for testing
- Future: disk-backed implementation for production

## Section 3: Replication Protocol over Tunnel

**Where:** `crates/harmony-tunnel/src/frame.rs` (new tag), new replication message types

Add frame tag `0x03 — Replication` to the tunnel frame format:

```
0x00 — Keepalive
0x01 — Reticulum
0x02 — Zenoh
0x03 — Replication
```

### Replication message format

Inside the `0x03` frame payload:

```
[1 byte op][32 bytes CID][variable payload]
```

| Op | Name | Direction | Payload |
|----|------|-----------|---------|
| `0x01` | PUSH | Alice → Bob | Encrypted book data |
| `0x02` | PULL | Alice → Bob | Empty (request own book back) |
| `0x03` | PULL_RESPONSE | Bob → Alice | Encrypted book data |
| `0x04` | STATUS | Either → Either | Empty |
| `0x05` | STATUS_RESPONSE | Either → Either | `used_bytes: u64 LE, quota_bytes: u64 LE` |

### Why a dedicated frame tag?

Replication is a bilateral concern between two tunnel peers — it doesn't need pub/sub semantics or namespace routing. A dedicated frame tag keeps it simple, avoids polluting the Zenoh namespace, and allows the tunnel task to route replication frames to a dedicated handler.

### TunnelSession changes

Add `SendReplication` to `TunnelEvent` and `ReplicationReceived` to `TunnelAction`, following the existing pattern for Reticulum/Zenoh frames. The session encrypts/decrypts replication frames identically to other frame types.

## Section 4: Replication Scheduling in the Runtime

**Where:** `crates/harmony-node/src/runtime.rs`

### Push scheduling

Every N ticks (default ~60 seconds), the runtime scans for unreplicated encrypted-durable content:

1. For each contact with `replication.quota_bytes > 0` and an active tunnel:
2. Find encrypted-durable books in the local store not yet pushed to this peer
3. Push one book per tick cycle (rate-limited to avoid bandwidth spikes)

### Replication tracking

`HashSet<(IdentityHash, ContentId)>` tracks which books have been pushed to which peers. Ephemeral — lost on restart. On restart, Alice queries `STATUS` from each delegate and reconciles, or re-pushes (idempotent by CID).

### New RuntimeAction/RuntimeEvent

```rust
RuntimeAction::ReplicaPush {
    peer_identity: IdentityHash,
    cid: [u8; 32],
    data: Vec<u8>,
}

RuntimeEvent::ReplicaReceived {
    peer_identity: IdentityHash,
    cid: [u8; 32],
    data: Vec<u8>,
}
```

The event loop translates `ReplicaPush` into a replication frame on the appropriate tunnel. `ReplicaReceived` fires when a PUSH frame arrives; the runtime stores it in the `ReplicaStore` after quota check.

### No complex sync

One-directional push, not bidirectional sync. Alice decides what to push. CID-addressed content is immutable — "push if not present" is the only operation needed. No conflict resolution, no vector clocks, no causal ordering.

## Implementation Order

1. **ReplicationPolicy on Contact** — data model change (small, no deps)
2. **ReplicaStore trait + MemoryReplicaStore** — storage abstraction
3. **Replication frame tag + message types** — tunnel protocol extension
4. **TunnelSession replication frame handling** — encrypt/decrypt
5. **Runtime scheduling + event handling** — push/receive orchestration
6. **CLI: `--replica-quota`** — configure quota per peer for testing

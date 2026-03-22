# Token-Gated Serving Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-5zt`.

**Goal:** Allow authorized third parties to retrieve encrypted content from replication delegates by presenting a self-certifying PQ UCAN bearer token signed by the content owner.

**Architecture:** Three changes: (1) new replication op `PullWithToken` (0x06) in harmony-tunnel, (2) public key cache in NodeRuntime populated from discovery/handshake events, (3) token validation logic that checks capability, resource, expiry, owner, and ML-DSA signature before serving.

**Tech Stack:** Rust, `harmony-tunnel` (replication.rs), `harmony-identity` (PqUcanToken, MlDsaPublicKey), `harmony-node` (runtime.rs, event_loop.rs)

**Spec:** `docs/superpowers/specs/2026-03-22-token-gated-serving-design.md`

---

## File Structure

```
crates/harmony-tunnel/src/
└── replication.rs     — Add PullWithToken op (0x06)

crates/harmony-node/src/
├── runtime.rs         — Public key cache, token validation, handle PullWithToken
└── event_loop.rs      — Populate pubkey cache from HandshakeComplete + discovery
```

---

### Task 1: Add PullWithToken replication op

**Files:**
- Modify: `crates/harmony-tunnel/src/replication.rs`

- [ ] **Step 1: Write test for PullWithToken encode/decode**

```rust
#[test]
fn pull_with_token_roundtrip() {
    let cid = [0xAA; 32];
    let token_bytes = vec![1, 2, 3, 4, 5]; // dummy token
    let msg = ReplicationMessage::pull_with_token(cid, token_bytes.clone());
    assert_eq!(msg.op, ReplicationOp::PullWithToken);
    assert_eq!(msg.cid, cid);
    assert_eq!(msg.payload, token_bytes);

    let encoded = msg.encode();
    let decoded = ReplicationMessage::decode(&encoded).unwrap();
    assert_eq!(decoded.op, ReplicationOp::PullWithToken);
    assert_eq!(decoded.cid, cid);
    assert_eq!(decoded.payload, token_bytes);
}
```

- [ ] **Step 2: Add PullWithToken to ReplicationOp**

```rust
pub enum ReplicationOp {
    Push = 0x01,
    Pull = 0x02,
    PullResponse = 0x03,
    Status = 0x04,
    StatusResponse = 0x05,
    PullWithToken = 0x06,
}
```

Update `from_byte()` to handle `0x06`.

Add convenience constructor:
```rust
pub fn pull_with_token(cid: [u8; 32], token: Vec<u8>) -> Self {
    Self { op: ReplicationOp::PullWithToken, cid, payload: token }
}
```

- [ ] **Step 3: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-tunnel pull_with_token`

- [ ] **Step 4: Commit**

```bash
git commit -m "feat(tunnel): add PullWithToken replication op (0x06)"
```

---

### Task 2: Public key cache in NodeRuntime

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/src/event_loop.rs`

Bob needs the issuer's ML-DSA public key to verify token signatures. Build a cache populated from two sources: tunnel HandshakeComplete events and discovery AnnounceRecords.

- [ ] **Step 1: Add pubkey cache to NodeRuntime**

In `runtime.rs`, add to `NodeRuntime`:

```rust
    /// Cache of ML-DSA public keys by identity hash.
    /// Populated from HandshakeComplete and AnnounceRecord events.
    pubkey_cache: HashMap<[u8; 16], Vec<u8>>,
```

Initialize as `HashMap::new()` in the constructor.

- [ ] **Step 2: Populate from HandshakeComplete**

In `push_event`, when handling `RuntimeEvent::TunnelHandshakeComplete`, the event currently doesn't carry the public key. We need to either:
- Add `peer_dsa_pubkey: Vec<u8>` to `RuntimeEvent::TunnelHandshakeComplete`, OR
- Add a new `RuntimeEvent::PeerPublicKeyLearned { identity_hash, pubkey }`

The simplest: add a new event. In the event loop, when `TunnelBridgeEvent::HandshakeComplete` fires (which carries `peer_dsa_pubkey`), push this new event to the runtime.

```rust
RuntimeEvent::PeerPublicKeyLearned {
    identity_hash: [u8; 16],
    dsa_pubkey: Vec<u8>,
}
```

Handler:
```rust
RuntimeEvent::PeerPublicKeyLearned { identity_hash, dsa_pubkey } => {
    self.pubkey_cache.insert(identity_hash, dsa_pubkey);
}
```

- [ ] **Step 3: Populate from discovery announces**

In `process_discovered_tunnel_hints`, the `AnnounceRecord` contains `public_key: Vec<u8>`. After extracting tunnel hints, also cache the public key:

```rust
self.pubkey_cache.insert(identity_hash, record.public_key.clone());
```

- [ ] **Step 4: Wire HandshakeComplete pubkey in event_loop.rs**

In the `TunnelBridgeEvent::HandshakeComplete` arm, after the existing `is_current` guard:

```rust
// Cache the peer's ML-DSA public key for token verification.
if let Some(contact) = runtime.contact_store().find_by_tunnel_node_id(&peer_node_id) {
    runtime.push_event(RuntimeEvent::PeerPublicKeyLearned {
        identity_hash: contact.identity_hash,
        dsa_pubkey: peer_dsa_pubkey.clone(),
    });
}
```

Note: `peer_dsa_pubkey` is already available in the `HandshakeComplete` destructuring — check that it's not being discarded with `_`.

- [ ] **Step 5: Add accessor**

```rust
pub fn get_peer_pubkey(&self, identity_hash: &[u8; 16]) -> Option<&[u8]> {
    self.pubkey_cache.get(identity_hash).map(|v| v.as_slice())
}
```

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-node`

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(node): public key cache for token signature verification"
```

---

### Task 3: Token validation and serving logic

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/Cargo.toml` (if harmony-identity not already a dep)

The core feature: validate a PqUcanToken and serve the replicated content if authorized.

- [ ] **Step 1: Add RuntimeAction::ReplicaPullResponse**

```rust
RuntimeAction::ReplicaPullResponse {
    peer_identity: [u8; 16],
    cid: [u8; 32],
    data: Vec<u8>,
}
```

- [ ] **Step 2: Implement token validation method**

```rust
/// Validate a PullWithToken request and emit a PullResponse if authorized.
fn handle_pull_with_token(
    &mut self,
    peer_identity: [u8; 16],
    cid: [u8; 32],
    token_bytes: Vec<u8>,
) -> Option<RuntimeAction> {
    let unix_now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);

    // 1. Deserialize token
    let token = match harmony_identity::PqUcanToken::from_bytes(&token_bytes) {
        Ok(t) => t,
        Err(e) => {
            tracing::debug!("token parse failed: {e:?}");
            return None;
        }
    };

    // 2. Check capability
    if token.capability != harmony_identity::CapabilityType::Content {
        tracing::debug!("token capability is not Content");
        return None;
    }

    // 3. Check resource matches CID
    if token.resource.len() != 32 || token.resource[..] != cid[..] {
        tracing::debug!("token resource doesn't match requested CID");
        return None;
    }

    // 4. Check expiry
    if token.expires_at != 0 && token.expires_at <= unix_now {
        tracing::debug!("token expired");
        return None;
    }

    // 5. Check not-before
    if token.not_before > unix_now {
        tracing::debug!("token not yet valid");
        return None;
    }

    // 6. Check owner — the replica must have been pushed by the token issuer
    let owner_match = self.replica_store.retrieve(&token.issuer, &cid).is_some();
    if !owner_match {
        tracing::debug!("no replica from token issuer for this CID");
        return None;
    }

    // 7. Verify ML-DSA signature
    let pubkey_bytes = match self.pubkey_cache.get(&token.issuer) {
        Some(pk) => pk,
        None => {
            tracing::debug!("issuer public key not cached");
            return None;
        }
    };
    let pubkey = match harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(pubkey_bytes) {
        Ok(pk) => pk,
        Err(_) => {
            tracing::debug!("cached pubkey invalid");
            return None;
        }
    };
    if token.verify_signature(&pubkey).is_err() {
        tracing::debug!("token signature verification failed");
        return None;
    }

    // All checks passed — serve the content
    let data = self.replica_store.retrieve(&token.issuer, &cid)?;
    Some(RuntimeAction::ReplicaPullResponse {
        peer_identity,
        cid,
        data,
    })
}
```

- [ ] **Step 3: Handle PullWithToken in ReplicaReceived**

In the `ReplicaReceived` handler, after the existing `Push` handling, add:

```rust
// If this is a PullWithToken, validate and respond
if rep_op == ReplicationOp::PullWithToken {
    if let Some(action) = self.handle_pull_with_token(peer_identity, cid, payload) {
        // Buffer the response action
    }
}
```

Wait — `ReplicaReceived` currently only handles Push ops. The event loop parses the replication message and only forwards Push. We need to also forward PullWithToken.

Update the `ReplicationReceived` handler in `event_loop.rs` to forward PullWithToken alongside Push:

```rust
if rep_msg.op == ReplicationOp::Push || rep_msg.op == ReplicationOp::PullWithToken {
    // ... existing lookup and push_event ...
}
```

Then in `runtime.rs`, the `ReplicaReceived` handler needs to know the op. Add `op: u8` to `RuntimeEvent::ReplicaReceived`:

```rust
RuntimeEvent::ReplicaReceived {
    peer_identity: [u8; 16],
    op: u8,  // ReplicationOp discriminant
    cid: [u8; 32],
    data: Vec<u8>,
}
```

Route based on op:
```rust
match op {
    0x01 => { /* existing Push handling */ }
    0x06 => {
        if let Some(action) = self.handle_pull_with_token(peer_identity, cid, data) {
            self.pending_direct_actions.push(action);
        }
    }
    _ => {}
}
```

- [ ] **Step 4: Handle ReplicaPullResponse in event loop**

```rust
RuntimeAction::ReplicaPullResponse { peer_identity, cid, data } => {
    // Encode as PullResponse replication message and send via tunnel
    let msg = harmony_tunnel::replication::ReplicationMessage::pull_response(cid, data);
    // Look up tunnel sender for this peer and send
    tracing::debug!(
        identity = %hex::encode(peer_identity),
        cid = %hex::encode(&cid[..8]),
        "serving replicated content via token"
    );
    // TODO: route to tunnel sender (same pattern as ReplicaPush)
}
```

- [ ] **Step 5: Write tests**

```rust
#[test]
fn pull_with_token_valid_serves_content() {
    // Setup: create runtime, add contact, store replica, cache pubkey
    // Create a valid PqUcanToken signed by the content owner
    // Call handle_pull_with_token
    // Verify it returns Some(ReplicaPullResponse)
}

#[test]
fn pull_with_token_expired_rejected() {
    // Same setup but token.expires_at is in the past
    // Verify None returned
}

#[test]
fn pull_with_token_wrong_cid_rejected() {
    // Token resource doesn't match requested CID
    // Verify None returned
}

#[test]
fn pull_with_token_unknown_issuer_rejected() {
    // Token issuer not in pubkey cache
    // Verify None returned
}

#[test]
fn pull_with_token_no_replica_rejected() {
    // Token is valid but no replica exists for the issuer+CID
    // Verify None returned
}
```

Note: Creating valid `PqUcanToken` in tests requires ML-DSA key generation and signing. Use `harmony_crypto::ml_dsa::generate(&mut OsRng)` and `PqPrivateIdentity::issue_pq_root_token()`. Tests need `RUST_MIN_STACK=8388608` for PQ key gen.

- [ ] **Step 6: Run tests**

Run: `RUST_MIN_STACK=8388608 cargo test -p harmony-node pull_with_token`

- [ ] **Step 7: Commit**

```bash
git commit -m "feat(node): token-gated serving of replicated encrypted content"
```

---

### Task 4: Cleanup and verification

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-tunnel -p harmony-node`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Commit**

```bash
git commit -m "chore: clippy fixes for token-gated serving"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | PullWithToken op (0x06) | New replication message type |
| 2 | Public key cache | HashMap populated from handshake + discovery |
| 3 | Token validation + serving | 7-step validation, ReplicaPullResponse action |
| 4 | Cleanup | Clippy clean, workspace tests pass |

**End-to-end flow after this bead:**
1. Alice issues `PqUcanToken { capability: Content, resource: cid, audience: carol }` and signs with ML-DSA
2. Carol sends `PullWithToken(cid, token_bytes)` to Bob via tunnel
3. Bob validates: capability, resource, expiry, owner, signature
4. Bob serves `PullResponse(cid, encrypted_data)` back to Carol
5. Carol decrypts with the key Alice shared out-of-band

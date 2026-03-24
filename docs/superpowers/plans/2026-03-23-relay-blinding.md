# MASQUE-Style Relay Blinding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace deterministic iroh NodeIds with ephemeral identities so the relay operator cannot build a social graph, while wiring up the previously-stubbed initiator dial.

**Architecture:** Two identity layers — ephemeral `iroh::SecretKey` at transport (relay sees random, unlinkable NodeIds) and permanent ML-DSA-65 at application (exchanged inside encrypted QUIC stream). Responder gets one ephemeral identity per startup; initiator gets a fresh one per dial.

**Tech Stack:** Rust, iroh 0.91 (Endpoint, SecretKey, Connection), harmony-tunnel (TunnelSession, ML-KEM/ML-DSA handshake), tokio

**Spec:** `docs/superpowers/specs/2026-03-23-relay-blinding-design.md`

---

### Task 1: Replace deterministic SecretKey with ephemeral (responder)

Replace the BLAKE3(signing_key) derivation with a random ephemeral SecretKey. The responder Endpoint's NodeId becomes random and unlinkable to the ML-DSA identity. The existing accept loop and LocalTunnelInfo emission already handle the rest.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs:263-272` — replace SecretKey derivation

**Context:**
- Current derivation (lines 263-272):
  ```rust
  let mut sk_bytes = config.local_identity.signing_key().as_bytes();
  let mut hash = harmony_crypto::hash::blake3_hash(&sk_bytes);
  sk_bytes.zeroize();
  let secret_key = iroh::SecretKey::from(hash);
  hash.zeroize();
  ```
- The `LocalTunnelInfo` emission at line 306-309 already uses `ep.node_id()`, so it automatically carries the ephemeral NodeId into announce records.
- The accept loop (lines 696-767) and ready connection handler (lines 769-791) are already wired and work with any Endpoint — no changes needed.

- [ ] **Step 1: Replace SecretKey derivation**

Replace lines 263-272 with:
```rust
let secret_key = iroh::SecretKey::generate(&mut rand::rngs::OsRng);
```

Remove the `sk_bytes`, `hash`, and `zeroize()` lines. The `zeroize` import for these specific variables may become unused — check and clean up.

- [ ] **Step 2: Write test for ephemeral NodeId**

Add to the test module in event_loop.rs (or a new test if no module exists — check the file's test structure):
```rust
#[test]
fn ephemeral_secret_key_differs_from_deterministic() {
    // Verify that iroh::SecretKey::generate() produces a different
    // NodeId than the old BLAKE3(signing_key) derivation.
    let identity = harmony_identity::PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
    let deterministic_hash = harmony_crypto::hash::blake3_hash(&identity.signing_key().as_bytes());
    let deterministic_key = iroh::SecretKey::from(deterministic_hash);

    let ephemeral_key = iroh::SecretKey::generate(&mut rand::rngs::OsRng);

    assert_ne!(
        deterministic_key.public().as_bytes(),
        ephemeral_key.public().as_bytes(),
        "ephemeral key must differ from deterministic derivation"
    );
}
```

Note: This test may need to be in a different location depending on how harmony-node tests are structured (it's a binary crate). Check the existing test module structure (the exploration found tests at `tests::` in main.rs). Adapt the test to use available imports. The key assertion is that a random SecretKey produces a different NodeId than the old deterministic derivation.

- [ ] **Step 3: Run tests**

```bash
cargo test -p harmony-node
```
Expected: all 211 tests pass.

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/event_loop.rs
git commit -m "feat(node): replace deterministic iroh SecretKey with ephemeral

The responder's iroh Endpoint now uses a random SecretKey generated at
startup instead of BLAKE3(ML-DSA signing key). This prevents the relay
operator from linking the transport identity to the permanent ML-DSA
identity. The ephemeral NodeId rotates on restart."
```

---

### Task 2: Extend InitiateTunnel action to carry PQ public keys

The initiator dial needs the peer's PQ public keys (ML-DSA + ML-KEM) to run the tunnel handshake. Currently `RuntimeAction::InitiateTunnel` only carries `identity_hash`, `node_id`, and `relay_url`. Add the pubkey fields so the event loop has everything it needs to construct a `PqIdentity` and call `run_initiator()`.

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs:264-268` — extend InitiateTunnel variant
- Modify: `crates/harmony-node/src/event_loop.rs:147-152` — extend DeferredDial struct
- Modify: `crates/harmony-node/src/event_loop.rs:1055-1062` — update DeferredDial construction
- Modify: wherever `RuntimeAction::InitiateTunnel` is emitted in the runtime — add keys from announce/contact

**Context:**
- `RuntimeAction::InitiateTunnel` is at runtime.rs:264-268
- `DeferredDial` struct is at event_loop.rs:147-152
- DeferredDial is constructed from the action at event_loop.rs:1055-1062
- The runtime emits InitiateTunnel when the PeerManager decides to dial a peer. Search for `InitiateTunnel` in runtime.rs to find the emission site(s). The announce record (`AnnounceRecord`) has `public_key: Vec<u8>` (ML-DSA) and `encryption_key: Vec<u8>` (ML-KEM).

- [ ] **Step 1: Add fields to RuntimeAction::InitiateTunnel**

In `runtime.rs`, extend the variant:
```rust
InitiateTunnel {
    identity_hash: [u8; 16],
    node_id: [u8; 32],
    relay_url: Option<String>,
    /// ML-DSA-65 public key from the peer's announce record.
    peer_dsa_pubkey: Vec<u8>,
    /// ML-KEM-768 public key from the peer's announce record.
    peer_kem_pubkey: Vec<u8>,
},
```

- [ ] **Step 2: Update DeferredDial struct**

In `event_loop.rs`, add to DeferredDial:
```rust
#[derive(Debug)]
struct DeferredDial {
    fire_at_ms: u64,
    identity_hash: [u8; 16],
    node_id: [u8; 32],
    relay_url: Option<String>,
    peer_dsa_pubkey: Vec<u8>,
    peer_kem_pubkey: Vec<u8>,
}
```

Note: `DeferredDial` implements `PartialOrd`/`Ord` for the BinaryHeap (ordering by `fire_at_ms`). The new `Vec<u8>` fields don't participate in ordering — verify the Ord impl only compares `fire_at_ms`. If it derives `Ord`, the Vec fields will change sort order — fix by using a manual Ord impl that only compares `fire_at_ms`.

- [ ] **Step 3: Update DeferredDial construction from InitiateTunnel action**

At event_loop.rs:1055-1062, update:
```rust
RuntimeAction::InitiateTunnel {
    identity_hash,
    node_id,
    relay_url,
    peer_dsa_pubkey,
    peer_kem_pubkey,
} => {
    let delay_ms = 500 + (rand::random::<u64>() % 3500);
    let fire_at = tunnel_task::millis_since_start() + delay_ms;
    deferred_dials.push(Reverse(DeferredDial {
        fire_at_ms: fire_at,
        identity_hash,
        node_id,
        relay_url,
        peer_dsa_pubkey,
        peer_kem_pubkey,
    }));
}
```

- [ ] **Step 4: Update InitiateTunnel emission in the runtime's translation layer**

The emission site is `translate_peer_actions_out()` in `runtime.rs` (~line 2063-2080). It translates from `PeerAction::InitiateTunnel` (in the `harmony-peers` crate) to `RuntimeAction::InitiateTunnel`. The `PeerAction` variant only carries `identity_hash`, `node_id`, `relay_url` — do NOT modify the `harmony-peers` crate.

Instead, look up the keys from the runtime's `DiscoveryManager` during translation:
```rust
// In translate_peer_actions_out(), at the InitiateTunnel arm:
let (peer_dsa_pubkey, peer_kem_pubkey) = self
    .discovery
    .get_record(&identity_hash, self.last_now)
    .map(|record| (record.public_key.clone(), record.encryption_key.clone()))
    .unwrap_or_default();
```

The `DiscoveryManager` stores full `AnnounceRecord`s which contain:
- `public_key: Vec<u8>` → `peer_dsa_pubkey` (ML-DSA, 1952 bytes)
- `encryption_key: Vec<u8>` → `peer_kem_pubkey` (ML-KEM, 1184 bytes)

If the record is not found (edge case — peer action without a prior announce), the empty vecs will cause PqIdentity construction to fail at dial time, which is handled gracefully in Task 3.

- [ ] **Step 5: Compile and run tests**

```bash
cargo test -p harmony-node
```

The new fields may cause compile errors where `InitiateTunnel` is pattern-matched — fix all match arms. Expected: all tests pass after fixing.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): extend InitiateTunnel action with PQ public keys

Add peer_dsa_pubkey and peer_kem_pubkey to InitiateTunnel and
DeferredDial so the event loop can construct PqIdentity for the
tunnel handshake without a separate lookup."
```

---

### Task 3: Wire up initiator dial with ephemeral identity and Endpoint storage

Replace the TODO stub at event_loop.rs:835-850 with actual iroh Endpoint creation + connection + handshake. Each dial uses a fresh ephemeral SecretKey. Store the transient Endpoint alongside the TunnelSender so it stays alive for the QUIC connection's lifetime.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs:835-850` — replace TODO stub with dial logic
- Modify: `crates/harmony-node/src/event_loop.rs` — add `initiator_endpoints` HashMap, clean up on TunnelClosed

**Context:**
- The responder path (lines 769-791) shows how to spawn `run_responder` and store `TunnelSender`. The initiator path mirrors this.
- `run_initiator()` signature (tunnel_task.rs:34-42):
  ```rust
  pub async fn run_initiator(
      conn: Connection,
      local_identity: &PqPrivateIdentity,
      remote_identity: &PqIdentity,
      bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
      cmd_rx: mpsc::Receiver<TunnelCommand>,
      interface_name: String,
      connection_id: u64,
  )
  ```
- `PqIdentity` must be constructed from the DeferredDial's pubkey bytes. The correct approach:
  ```rust
  use harmony_crypto::ml_dsa::MlDsaPublicKey;
  use harmony_crypto::ml_kem::MlKemPublicKey;

  let dsa_key = MlDsaPublicKey::from_bytes(&dial.peer_dsa_pubkey)?;  // 1952 bytes
  let kem_key = MlKemPublicKey::from_bytes(&dial.peer_kem_pubkey)?;  // 1184 bytes
  let remote_id = PqIdentity::from_public_keys(kem_key, dsa_key);
  ```
  Verify exact method names by reading `harmony_identity::PqIdentity` — the order may be `(kem, dsa)` or `(dsa, kem)`. Get this right or the handshake will fail.
- `HARMONY_TUNNEL_ALPN` = `b"harmony-tunnel/1"` (tunnel_task.rs:23)
- Interface naming convention: `"tunnel-{8hex}"` from `connection.remote_node_id()` (event_loop.rs:735). For initiator dials, use the same convention with the target's ephemeral NodeId.
- Connection ID: auto-incrementing `u64` at event_loop.rs:240 (`let mut next_connection_id: u64 = 1;`)
- The relay URL and relay map: Currently built at event_loop.rs:276-289. For transient Endpoints, use the same relay map. Extract the relay map construction into a variable that can be reused, or store the `relay_url` config for reuse.
- Tunnel limit: 64 concurrent (event_loop.rs:240 `const MAX_TUNNELS: usize = 64;`)

- [ ] **Step 1: Add initiator_endpoints storage**

Near the `tunnel_senders` declaration (line 242), add:
```rust
// Stores the transient iroh::Endpoint for initiator-originated tunnels.
// Must stay alive as long as the QUIC connection is active (iroh uses it
// for relay forwarding). Dropped on TunnelClosed.
let mut initiator_endpoints: HashMap<String, iroh::Endpoint> = HashMap::new();
```

- [ ] **Step 2: Store relay map for reuse**

The relay map is currently built inline during responder Endpoint setup (lines 276-289). Extract it so it can be reused for transient initiator Endpoints. Before the Endpoint builder:
```rust
let relay_map = config.relay_url.as_ref().map(|url| {
    let relay_url: iroh::RelayUrl = url.parse().expect("relay URL already validated");
    iroh::RelayMap::from_iter([relay_url])
});
```

Then use `relay_map.clone()` in both the responder builder and initiator dials.

- [ ] **Step 3: Replace the TODO stub with dial logic**

Replace lines 835-850. The key insight: reuse the existing `ReadyConnection` → `conn_rx` pipeline. The dial task binds an ephemeral Endpoint, connects, then sends a `ReadyConnection` through `conn_tx`. The event loop's ready connection handler (arm 7) stores the TunnelSender and spawns either `run_initiator` or `run_responder` based on whether `remote_pq_identity` is present.

```rust
let current_ms = now_ms();
while let Some(Reverse(front)) = deferred_dials.peek() {
    if front.fire_at_ms <= current_ms {
        let Reverse(dial) = deferred_dials.pop().unwrap();

        // Check tunnel limit
        if tunnel_senders.len() >= MAX_TUNNELS {
            tracing::warn!(
                identity = %hex::encode(dial.identity_hash),
                "tunnel limit reached, skipping dial"
            );
            continue;
        }

        // Construct PqIdentity for the handshake
        let remote_pq_identity = match construct_pq_identity_from_bytes(
            &dial.peer_dsa_pubkey,
            &dial.peer_kem_pubkey,
        ) {
            Ok(id) => id,
            Err(e) => {
                tracing::warn!(
                    identity = %hex::encode(dial.identity_hash),
                    err = ?e,
                    "failed to construct PqIdentity for tunnel dial"
                );
                continue;
            }
        };

        let connection_id = next_connection_id;
        next_connection_id += 1;

        // Build target NodeAddr from ephemeral NodeId in announce
        let target_node_id = iroh::NodeId::from_bytes(&dial.node_id)
            .expect("node_id is always 32 bytes");
        let mut node_addr = iroh::NodeAddr::new(target_node_id);
        if let Some(ref url) = dial.relay_url {
            if let Ok(relay_url) = url.parse::<iroh::RelayUrl>() {
                node_addr = node_addr.with_relay_url(relay_url);
            }
        }

        let interface_name = format!("tunnel-{}", &hex::encode(&dial.node_id[..4]));
        let conn_tx_clone = conn_tx.clone();
        let relay_map_clone = relay_map.clone();

        // Spawn async dial task — sends ReadyConnection through conn_tx on success
        tokio::spawn(async move {
            // Generate ephemeral identity for this dial
            let ephemeral_key = iroh::SecretKey::generate(&mut rand::rngs::OsRng);
            let mut ep_builder = iroh::Endpoint::builder()
                .alpns(vec![tunnel_task::HARMONY_TUNNEL_ALPN.to_vec()])
                .secret_key(ephemeral_key);
            if let Some(ref rm) = relay_map_clone {
                ep_builder = ep_builder.relay_mode(iroh::RelayMode::Custom(rm.clone()));
            } else {
                ep_builder = ep_builder.relay_mode(iroh::RelayMode::Disabled);
            }

            let ep = match ep_builder.bind().await {
                Ok(ep) => ep,
                Err(e) => {
                    tracing::warn!(err = %e, "ephemeral endpoint bind failed");
                    return;
                }
            };

            let conn = match ep.connect(node_addr, tunnel_task::HARMONY_TUNNEL_ALPN).await {
                Ok(conn) => conn,
                Err(e) => {
                    tracing::warn!(err = %e, "tunnel dial failed");
                    return;
                }
            };

            tracing::info!(%interface_name, "tunnel dial connected");

            // Send ReadyConnection back to event loop for TunnelSender setup
            let _ = conn_tx_clone.send(ReadyConnection {
                connection: conn,
                connection_id,
                interface_name,
                remote_pq_identity: Some(remote_pq_identity),
                initiator_endpoint: Some(ep),
            }).await;
        });
    } else {
        break;
    }
}
```

**Helper function** — add near the top of event_loop.rs or in tunnel_bridge.rs:
```rust
fn construct_pq_identity_from_bytes(
    dsa_bytes: &[u8],
    kem_bytes: &[u8],
) -> Result<harmony_identity::PqIdentity, Box<dyn std::error::Error>> {
    let dsa_key = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(dsa_bytes)?;
    let kem_key = harmony_crypto::ml_kem::MlKemPublicKey::from_bytes(kem_bytes)?;
    Ok(harmony_identity::PqIdentity::from_public_keys(kem_key, dsa_key))
}
```
Verify the exact constructor signature and argument order by reading `PqIdentity`.

To support this, extend `ReadyConnection` (tunnel_bridge.rs:73-77) with an optional field:
```rust
pub struct ReadyConnection {
    pub connection: iroh::endpoint::Connection,
    pub connection_id: u64,
    pub interface_name: String,
    /// For initiator connections: the peer's PQ identity for the handshake.
    /// None for responder connections (identity learned during handshake).
    pub remote_pq_identity: Option<PqIdentity>,
    /// For initiator connections: the transient iroh Endpoint (must stay alive).
    pub initiator_endpoint: Option<iroh::Endpoint>,
}
```

Then the spawned dial task sends a `ReadyConnection` through `conn_tx`, and the event loop's handler at arm 7 checks `remote_pq_identity` to decide whether to call `run_initiator` or `run_responder`.

Adapt the ready connection handler (lines 769-791) to handle both cases.

- [ ] **Step 4: Update ReadyConnection struct**

In `tunnel_bridge.rs`, extend:
```rust
pub struct ReadyConnection {
    pub connection: iroh::endpoint::Connection,
    pub connection_id: u64,
    pub interface_name: String,
    pub remote_pq_identity: Option<harmony_identity::PqIdentity>,
    pub initiator_endpoint: Option<iroh::Endpoint>,
}
```

Update all existing construction sites of `ReadyConnection` (in the accept loop at event_loop.rs:696-767) to set `remote_pq_identity: None` and `initiator_endpoint: None`.

- [ ] **Step 5: Update ready connection handler for both initiator and responder**

At event_loop.rs:769-791, update to handle both cases:
```rust
ready = conn_rx.recv() => {
    let ReadyConnection {
        connection,
        connection_id,
        interface_name,
        remote_pq_identity,
        initiator_endpoint,
    } = ready;

    let (cmd_tx, cmd_rx) = mpsc::channel(32);
    tunnel_senders.insert(interface_name.clone(), TunnelSender::new(cmd_tx, connection_id));

    // Store initiator endpoint if present
    if let Some(ep) = initiator_endpoint {
        initiator_endpoints.insert(interface_name.clone(), ep);
    }

    let identity = tunnel_config.local_identity.clone();
    let tx = bridge_tx.clone();

    if let Some(remote_id) = remote_pq_identity {
        // Initiator path
        tokio::spawn(async move {
            tunnel_task::run_initiator(
                connection, &identity, &remote_id,
                tx, cmd_rx, interface_name, connection_id,
            ).await;
        });
    } else {
        // Responder path (existing)
        tokio::spawn(async move {
            tunnel_task::run_responder(
                connection, &identity,
                tx, cmd_rx, interface_name, connection_id,
            ).await;
        });
    }
}
```

- [ ] **Step 6: Clean up initiator endpoints on TunnelClosed**

At the TunnelClosed handler (event_loop.rs:673-692), add:
```rust
initiator_endpoints.remove(&interface_name);
```

- [ ] **Step 7: Compile and run tests**

```bash
cargo test -p harmony-node
cargo clippy -p harmony-node
```

Fix any compilation errors. Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): wire up initiator dial with ephemeral identity

Replace the InitiateTunnel TODO stub with actual iroh Endpoint
creation, QUIC connection, and handshake spawning. Each dial uses
a fresh random SecretKey. Transient Endpoints are stored alongside
TunnelSenders and cleaned up on TunnelClosed."
```

---

### Task 4: Tests and cleanup

Verify the full implementation works correctly. Add targeted tests and run the full workspace suite.

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs` — add/verify tests
- Modify: `crates/harmony-node/src/runtime.rs` — fix any test breakage from Task 2

- [ ] **Step 1: Verify existing tests pass**

```bash
cargo test -p harmony-node
```

Expected: all 211+ tests pass. If any tests broke from the InitiateTunnel field additions (Task 2), fix them — the tests may construct InitiateTunnel actions directly and need the new fields added.

- [ ] **Step 2: Add test for ephemeral NodeId in LocalTunnelInfo**

The runtime processes `LocalTunnelInfo { node_id, relay_url }` and stores it as a `RoutingHint::Tunnel`. Verify the runtime test coverage confirms this path works. If there's an existing test for `LocalTunnelInfo`, verify it still passes. If not, add one in `runtime.rs` tests:

```rust
#[test]
fn local_tunnel_info_stores_hint() {
    // This test should already exist — verify it passes with ephemeral NodeId
    // The key assertion: the node_id in the routing hint matches what was emitted
}
```

- [ ] **Step 3: Add test for DeferredDial ordering**

Verify the BinaryHeap ordering still works correctly after adding Vec fields to DeferredDial:
```rust
#[test]
fn deferred_dial_ordering_ignores_pubkey_fields() {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let dial1 = DeferredDial {
        fire_at_ms: 100,
        identity_hash: [0; 16],
        node_id: [0; 32],
        relay_url: None,
        peer_dsa_pubkey: vec![1, 2, 3],
        peer_kem_pubkey: vec![4, 5, 6],
    };
    let dial2 = DeferredDial {
        fire_at_ms: 50,
        identity_hash: [0; 16],
        node_id: [0; 32],
        relay_url: None,
        peer_dsa_pubkey: vec![7, 8, 9],
        peer_kem_pubkey: vec![10, 11, 12],
    };

    let mut heap = BinaryHeap::new();
    heap.push(Reverse(dial1));
    heap.push(Reverse(dial2));

    // Min-heap: earlier fire_at_ms should come out first
    let first = heap.pop().unwrap().0;
    assert_eq!(first.fire_at_ms, 50);
}
```

- [ ] **Step 4: Run full workspace tests**

```bash
cargo test --workspace
```

Expected: all tests pass across all crates. The harmony-tunnel crate tests should pass unchanged (no changes to that crate).

- [ ] **Step 5: Run clippy**

```bash
cargo clippy --workspace
```

Fix any warnings.

- [ ] **Step 6: Commit**

```bash
git add .
git commit -m "test(node): add tests for ephemeral identity and deferred dial ordering"
```

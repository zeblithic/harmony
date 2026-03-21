# Tunnel Peer Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-h6k`.

**Goal:** Extend PeerManager and ContactStore for tunnel peer connections, wire them into NodeRuntime so the full lifecycle (discover → connect → maintain → reconnect) works end-to-end with the iroh tunnel infrastructure.

**Architecture:** Three layers of changes: (1) `harmony-contacts` gets a `ContactAddress` enum so contacts can specify tunnel addressing (NodeId + relay URL), (2) `harmony-peers` gets tunnel-aware events/actions (`InitiateTunnel`, `TunnelEstablished`, `TunnelFailed`, `TunnelDropped`) and `ConnectionQuality` tracking, (3) `harmony-node` wires PeerManager into the runtime, translating between `PeerAction`s and the existing tunnel bridge/task infrastructure.

**Tech Stack:** Rust, `harmony-contacts`, `harmony-peers`, `harmony-node` (runtime + event loop), `harmony-tunnel` (TunnelSession via existing tunnel_task.rs)

**Spec:** `docs/superpowers/specs/2026-03-20-tunnel-peer-infrastructure-design.md` — Section 4

**Scope:** This plan covers **Bead harmony-h6k** only. Discovery-driven address population comes in Bead harmony-lbv.

**Prerequisites:** PR #84 (iroh-net integration) must be merged. The event loop already has `tunnel_task.rs`, `tunnel_bridge.rs`, iroh Endpoint setup, accept loop, connection management, and `TunnelConfig`. This plan builds on that infrastructure.

**no_std note:** `harmony-contacts` and `harmony-peers` are `no_std`-compatible. `ContactAddress` uses `String` for `relay_url` and `direct_addrs` (not `Url`/`SocketAddr`) because the `url` crate and `core::net::SocketAddr` are not `no_std`-friendly. The spec calls for `Url` but this is a pragmatic deviation documented here. Conversion from `String` to typed `Url`/`SocketAddr` happens at the event loop boundary.

---

## File Structure

```
crates/harmony-contacts/src/
├── contact.rs        — Add ContactAddress enum, addresses field on Contact
├── store.rs          — Add lookup-by-node-id helper
└── lib.rs            — Re-export ContactAddress

crates/harmony-peers/src/
├── event.rs          — Add tunnel event/action variants
├── state.rs          — Add ConnectionQuality, transport tracking
├── manager.rs        — Tunnel-aware state transitions, dual-address strategy
└── lib.rs            — Re-export new types

crates/harmony-node/src/
├── main.rs           — Add --add-tunnel-peer CLI command
├── runtime.rs        — Add PeerManager + ContactStore fields, wire events
└── event_loop.rs     — Route PeerActions to tunnel infrastructure
```

---

### Task 1: Add ContactAddress to harmony-contacts

**Files:**
- Modify: `crates/harmony-contacts/src/contact.rs`
- Modify: `crates/harmony-contacts/src/store.rs`
- Modify: `crates/harmony-contacts/src/lib.rs`

This task adds the `ContactAddress` enum and an `addresses` field on `Contact` so contacts can specify how to reach them (Reticulum destination hash, tunnel NodeId + relay URL, or both).

- [ ] **Step 1: Write test for ContactAddress serialization roundtrip**

In `crates/harmony-contacts/src/contact.rs`, add to tests:

```rust
#[test]
fn contact_address_tunnel_roundtrip() {
    let addr = ContactAddress::Tunnel {
        node_id: [0xAB; 32],
        relay_url: Some("https://iroh.q8.fyi".into()),
        direct_addrs: vec![],
    };
    let serialized = postcard::to_allocvec(&addr).unwrap();
    let deserialized: ContactAddress = postcard::from_bytes(&serialized).unwrap();
    match deserialized {
        ContactAddress::Tunnel { node_id, relay_url, .. } => {
            assert_eq!(node_id, [0xAB; 32]);
            assert_eq!(relay_url.as_deref(), Some("https://iroh.q8.fyi"));
        }
        _ => panic!("expected Tunnel variant"),
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-contacts contact_address_tunnel_roundtrip`
Expected: FAIL — `ContactAddress` not defined.

- [ ] **Step 3: Implement ContactAddress enum**

Add to `contact.rs`:

```rust
/// How to reach a contact. A contact may have multiple addresses
/// for different transport paths.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContactAddress {
    /// Reachable via Reticulum path table (LAN/mesh).
    Reticulum {
        destination_hash: [u8; 16],
    },
    /// Reachable via iroh-net QUIC tunnel.
    Tunnel {
        /// BLAKE3(ML-DSA-65 public key) — the iroh NodeId seed.
        node_id: [u8; 32],
        /// Preferred relay server URL (e.g., "https://iroh.q8.fyi").
        relay_url: Option<alloc::string::String>,
        /// Known direct socket addresses (ephemeral, may be stale).
        direct_addrs: alloc::vec::Vec<alloc::string::String>,
    },
}
```

Add an `addresses` field to the `Contact` struct:

```rust
pub struct Contact {
    // ... existing fields ...
    /// Transport addresses for reaching this contact.
    pub addresses: alloc::vec::Vec<ContactAddress>,
}
```

Update `Contact` construction in existing tests to include `addresses: vec![]`. This includes:
- Test helpers in `crates/harmony-contacts/src/store.rs` (e.g., `make_contact`)
- Test helpers in `crates/harmony-peers/src/manager.rs` (e.g., `make_store_with_contact`)
- Any other test that constructs `Contact` directly across the workspace

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-contacts`
Expected: All tests pass.

- [ ] **Step 5: Add store helper to look up contacts by tunnel NodeId**

In `store.rs`, add:

```rust
/// Find a contact that has a matching tunnel NodeId.
pub fn find_by_tunnel_node_id(&self, node_id: &[u8; 32]) -> Option<&Contact> {
    self.contacts.values().find(|c| {
        c.addresses.iter().any(|addr| matches!(
            addr,
            ContactAddress::Tunnel { node_id: id, .. } if id == node_id
        ))
    })
}
```

- [ ] **Step 6: Re-export ContactAddress from lib.rs**

Add `ContactAddress` to the `pub use` line in `lib.rs`.

- [ ] **Step 7: Run all contact tests**

Run: `cargo test -p harmony-contacts`
Expected: All tests pass.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-contacts/
git commit -m "feat(contacts): add ContactAddress enum with Reticulum and Tunnel variants"
```

---

### Task 2: Add tunnel events and actions to harmony-peers

**Files:**
- Modify: `crates/harmony-peers/src/event.rs`
- Modify: `crates/harmony-peers/src/state.rs`
- Modify: `crates/harmony-peers/Cargo.toml` (if needed)
- Modify: `crates/harmony-peers/src/lib.rs`

- [ ] **Step 1: Add tunnel event variants to PeerEvent**

In `event.rs`, add to `PeerEvent`:

```rust
    /// A tunnel connection was established to this peer.
    TunnelEstablished {
        identity_hash: IdentityHash,
        node_id: [u8; 32],
        now: u64,
    },
    /// A tunnel connection attempt failed.
    TunnelFailed {
        identity_hash: IdentityHash,
    },
    /// An established tunnel was dropped.
    TunnelDropped {
        identity_hash: IdentityHash,
    },
```

- [ ] **Step 2: Add tunnel action variant to PeerAction**

In `event.rs`, add to `PeerAction`:

```rust
    /// Initiate a tunnel connection to a peer via iroh-net.
    InitiateTunnel {
        identity_hash: IdentityHash,
        node_id: [u8; 32],
        relay_url: Option<String>,
    },
```

- [ ] **Step 3: Add ConnectionQuality and Transport to state.rs**

```rust
/// Transport type for an active peer connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// Connected via LAN/mesh Reticulum link.
    Lan,
    /// Connected via iroh-net QUIC tunnel.
    Tunnel { relayed: bool },
}

/// Quality metrics for an active peer connection.
#[derive(Debug, Clone)]
pub struct ConnectionQuality {
    pub rtt_ms: Option<u32>,
    pub transport: Transport,
    pub connected_since: u64,
}
```

Add transport tracking to `PeerState`:

```rust
pub struct PeerState {
    // ... existing fields ...
    /// Active connection quality (only set when Connected).
    pub connection_quality: Option<ConnectionQuality>,
}
```

- [ ] **Step 4: Re-export new types from lib.rs**

Add `ConnectionQuality`, `Transport`, `ContactAddress` to the public API.

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-peers`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-peers/
git commit -m "feat(peers): add tunnel event/action variants and ConnectionQuality tracking"
```

---

### Task 3: Tunnel-aware PeerManager state transitions

**Files:**
- Modify: `crates/harmony-peers/src/manager.rs`

The core logic: when PeerManager decides to connect to a peer, it checks the contact's addresses. If a `ContactAddress::Tunnel` exists, emit `InitiateTunnel` instead of `InitiateLink`. Handle `TunnelEstablished`/`TunnelFailed`/`TunnelDropped` events with the same state machine transitions as the Reticulum link equivalents.

- [ ] **Step 1: Write test for tunnel-first connection strategy**

```rust
#[test]
fn tunnel_address_emits_initiate_tunnel() {
    let mut store = ContactStore::new();
    store.add(Contact {
        identity_hash: [1u8; 16],
        display_name: None,
        peering: PeeringPolicy { enabled: true, priority: PeeringPriority::High },
        added_at: 0,
        last_seen: None,
        notes: None,
        addresses: vec![ContactAddress::Tunnel {
            node_id: [0xAA; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
            direct_addrs: vec![],
        }],
    }).unwrap();

    let mut mgr = PeerManager::new();
    let actions = mgr.on_event(
        PeerEvent::ContactChanged { identity_hash: [1u8; 16] },
        &store,
    );
    // Should be Searching now, waiting for announce or probe

    // Simulate announce received
    let actions = mgr.on_event(
        PeerEvent::AnnounceReceived { identity_hash: [1u8; 16] },
        &store,
    );

    // Should emit InitiateTunnel (not InitiateLink) because contact has tunnel address
    assert!(actions.iter().any(|a| matches!(a, PeerAction::InitiateTunnel { .. })));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-peers tunnel_address_emits_initiate_tunnel`

- [ ] **Step 3: Implement tunnel-aware connection logic**

In `manager.rs`, modify `handle_announce_received` to check the contact's addresses:

```rust
fn handle_announce_received(
    &mut self,
    identity_hash: IdentityHash,
    contacts: &ContactStore,
    actions: &mut Vec<PeerAction>,
) {
    // ... existing state lookup ...

    // Check if contact has a tunnel address — prefer tunnel over Reticulum link
    if let Some(contact) = contacts.get(&identity_hash) {
        if let Some(tunnel_addr) = contact.addresses.iter().find_map(|a| match a {
            ContactAddress::Tunnel { node_id, relay_url, .. } => {
                Some((*node_id, relay_url.clone()))
            }
            _ => None,
        }) {
            actions.push(PeerAction::InitiateTunnel {
                identity_hash,
                node_id: tunnel_addr.0,
                relay_url: tunnel_addr.1,
            });
            // Update state to Connecting
            if let Some(peer) = self.peers.get_mut(&identity_hash) {
                peer.status = PeerStatus::Connecting;
                peer.connecting_since = Some(/* now from event */);
            }
            return;
        }
    }

    // Fallback: existing InitiateLink behavior
    actions.push(PeerAction::InitiateLink { identity_hash });
    // ... existing state transition ...
}
```

- [ ] **Step 4: Handle TunnelEstablished event**

Add to `on_event` match:

```rust
PeerEvent::TunnelEstablished { identity_hash, node_id, now } => {
    if let Some(peer) = self.peers.get_mut(&identity_hash) {
        peer.status = PeerStatus::Connected;
        peer.last_seen = Some(now);
        peer.retry_count = 0;
        peer.connecting_since = None;
        peer.connection_quality = Some(ConnectionQuality {
            rtt_ms: None,
            transport: Transport::Tunnel { relayed: false },
            connected_since: now,
        });
        actions.push(PeerAction::UpdateLastSeen {
            identity_hash,
            timestamp: now,
        });
    }
}
```

- [ ] **Step 5: Handle TunnelFailed and TunnelDropped events**

```rust
PeerEvent::TunnelFailed { identity_hash } => {
    if let Some(peer) = self.peers.get_mut(&identity_hash) {
        peer.status = PeerStatus::Searching;
        peer.retry_count += 1;
        peer.connecting_since = None;
        peer.connection_quality = None;
    }
}
PeerEvent::TunnelDropped { identity_hash } => {
    if let Some(peer) = self.peers.get_mut(&identity_hash) {
        peer.status = PeerStatus::Searching;
        peer.retry_count += 1;
        peer.connection_quality = None;
        // If peer also has a Reticulum address, try that immediately
        // (no backoff for fallback path)
        if let Some(contact) = contacts.get(&identity_hash) {
            if contact.addresses.iter().any(|a| matches!(a, ContactAddress::Reticulum { .. })) {
                actions.push(PeerAction::InitiateLink { identity_hash });
                peer.status = PeerStatus::Connecting;
            }
        }
    }
}
```

Note: `TunnelDropped` needs access to `contacts` — ensure the `on_event` signature passes `&ContactStore`.

- [ ] **Step 6: Write test for tunnel established → connected**

```rust
#[test]
fn tunnel_established_transitions_to_connected() {
    let (mut mgr, store) = setup_tunnel_peer(); // helper that creates manager + store with tunnel contact
    mgr.on_event(PeerEvent::ContactChanged { identity_hash: [1u8; 16] }, &store);
    mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: [1u8; 16] }, &store);

    let actions = mgr.on_event(
        PeerEvent::TunnelEstablished {
            identity_hash: [1u8; 16],
            node_id: [0xAA; 32],
            now: 1000,
        },
        &store,
    );

    assert!(mgr.peers[&[1u8; 16]].status == PeerStatus::Connected);
    assert!(actions.iter().any(|a| matches!(a, PeerAction::UpdateLastSeen { .. })));
}
```

- [ ] **Step 7: Write test for tunnel dropped → Reticulum fallback**

```rust
#[test]
fn tunnel_dropped_falls_back_to_reticulum() {
    let mut store = ContactStore::new();
    store.add(Contact {
        identity_hash: [1u8; 16],
        // ... policy ...
        addresses: vec![
            ContactAddress::Tunnel { node_id: [0xAA; 32], relay_url: None, direct_addrs: vec![] },
            ContactAddress::Reticulum { destination_hash: [0xBB; 16] },
        ],
    }).unwrap();

    let mut mgr = PeerManager::new();
    mgr.on_event(PeerEvent::ContactChanged { identity_hash: [1u8; 16] }, &store);
    mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: [1u8; 16] }, &store);
    mgr.on_event(PeerEvent::TunnelEstablished { identity_hash: [1u8; 16], node_id: [0xAA; 32], now: 100 }, &store);

    // Tunnel drops
    let actions = mgr.on_event(PeerEvent::TunnelDropped { identity_hash: [1u8; 16] }, &store);

    // Should immediately try Reticulum fallback (no backoff)
    assert!(actions.iter().any(|a| matches!(a, PeerAction::InitiateLink { .. })));
}
```

- [ ] **Step 8: Run all peer tests**

Run: `cargo test -p harmony-peers`
Expected: All tests pass.

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-peers/
git commit -m "feat(peers): tunnel-aware state transitions with dual-address fallback"
```

---

### Task 4: Wire PeerManager into NodeRuntime

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml`

Add `PeerManager` and `ContactStore` as fields on `NodeRuntime`. Feed PeerManager events from the router (AnnounceReceived) and timer (Tick). Translate PeerActions into RuntimeActions.

- [ ] **Step 1: Add harmony-peers and harmony-contacts deps**

In `crates/harmony-node/Cargo.toml`, add:

```toml
harmony-contacts = { workspace = true, features = ["std"] }
harmony-peers = { workspace = true, features = ["std"] }
```

- [ ] **Step 2: Add fields to NodeRuntime**

In `runtime.rs`, add to the `NodeRuntime` struct:

```rust
    peer_manager: harmony_peers::PeerManager,
    contact_store: harmony_contacts::ContactStore,
```

Initialize them in `NodeRuntime::new()`:

```rust
    peer_manager: harmony_peers::PeerManager::new(),
    contact_store: harmony_contacts::ContactStore::new(),
```

- [ ] **Step 3: Add RuntimeAction variants for tunnel peer lifecycle**

```rust
    /// Initiate a tunnel connection to a peer.
    InitiateTunnel {
        identity_hash: [u8; 16],
        node_id: [u8; 32],
        relay_url: Option<String>,
    },
    /// Send a Reticulum path request for a peer.
    SendPathRequest {
        identity_hash: [u8; 16],
    },
```

- [ ] **Step 4: Add RuntimeEvent variants for peer lifecycle**

```rust
    /// A tunnel peer connection was established.
    TunnelPeerEstablished {
        identity_hash: [u8; 16],
        node_id: [u8; 32],
        now: u64,
    },
    /// A tunnel peer connection failed.
    TunnelPeerFailed {
        identity_hash: [u8; 16],
    },
    /// An established tunnel peer connection was dropped.
    TunnelPeerDropped {
        identity_hash: [u8; 16],
    },
    /// A contact was added or updated in the store.
    ContactChanged {
        identity_hash: [u8; 16],
    },
```

- [ ] **Step 5: Wire PeerManager in push_event and tick**

In `push_event`, handle the new events by forwarding to PeerManager:

```rust
RuntimeEvent::TunnelPeerEstablished { identity_hash, node_id, now } => {
    let actions = self.peer_manager.on_event(
        harmony_peers::PeerEvent::TunnelEstablished { identity_hash, node_id, now },
        &self.contact_store,
    );
    // Translate peer actions and append to the runtime actions returned by tick().
    // PeerManager actions are processed in the same tick cycle as router actions.
    let runtime_actions = self.translate_peer_actions(actions);
    // These will be returned from tick() alongside router/storage/compute actions.
}
// ... similar for TunnelPeerFailed, TunnelPeerDropped, ContactChanged
```

In `tick`, after processing router events, translate `NodeAction::AnnounceReceived` into `PeerEvent::AnnounceReceived`:

```rust
// After processing router actions:
for action in &router_actions {
    if let NodeAction::AnnounceReceived { validated_announce, .. } = action {
        let id_hash = validated_announce.identity.address_hash;
        let peer_actions = self.peer_manager.on_event(
            harmony_peers::PeerEvent::AnnounceReceived { identity_hash: id_hash },
            &self.contact_store,
        );
        // Translate PeerActions → RuntimeActions
    }
}
```

Feed PeerManager tick from TimerTick:

```rust
RuntimeEvent::TimerTick { now } => {
    // ... existing timer handling ...
    let peer_actions = self.peer_manager.on_event(
        harmony_peers::PeerEvent::Tick { now },
        &self.contact_store,
    );
    // Translate PeerActions → RuntimeActions
}
```

- [ ] **Step 6: Translate PeerActions to RuntimeActions**

Add a helper method:

```rust
fn translate_peer_actions(&mut self, actions: Vec<harmony_peers::PeerAction>) -> Vec<RuntimeAction> {
    actions.into_iter().filter_map(|action| match action {
        PeerAction::InitiateTunnel { identity_hash, node_id, relay_url } => {
            Some(RuntimeAction::InitiateTunnel { identity_hash, node_id, relay_url })
        }
        PeerAction::InitiateLink { identity_hash } => {
            // TODO: translate to Reticulum link request
            None
        }
        PeerAction::SendPathRequest { identity_hash } => {
            Some(RuntimeAction::SendPathRequest { identity_hash })
        }
        PeerAction::UpdateLastSeen { identity_hash, timestamp } => {
            self.contact_store.update_last_seen(&identity_hash, timestamp);
            None
        }
        PeerAction::CloseLink { .. } => {
            // TODO: close tunnel or link
            None
        }
    }).collect()
}
```

- [ ] **Step 7: Expose contact_store mutably for CLI additions**

Add a public method:

```rust
pub fn contact_store_mut(&mut self) -> &mut harmony_contacts::ContactStore {
    &mut self.contact_store
}
```

- [ ] **Step 8: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): wire PeerManager and ContactStore into NodeRuntime"
```

---

### Task 5: Route PeerActions in the event loop

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`

Connect `RuntimeAction::InitiateTunnel` to the existing tunnel task infrastructure. When the runtime emits `InitiateTunnel`, the event loop creates an iroh connection and spawns a tunnel task, the same way the accept arm does for inbound connections.

- [ ] **Step 1: Handle InitiateTunnel in dispatch_action**

In the `dispatch_action` function, add a match arm:

```rust
RuntimeAction::InitiateTunnel { identity_hash, node_id, relay_url } => {
    if let Some(ref ep) = iroh_endpoint {
        let node_id_iroh = iroh::PublicKey::from_bytes(&node_id)
            .expect("valid 32-byte public key");
        let addr = iroh::NodeAddr::from(node_id_iroh);
        // TODO: add relay_url and direct_addrs to NodeAddr when available

        let conn_id = next_connection_id;
        next_connection_id += 1;
        let conn_tx = conn_tx.clone();
        inflight_handshakes += 1;

        let ep = ep.clone();
        tokio::spawn(async move {
            match ep.connect(addr, tunnel_task::HARMONY_TUNNEL_ALPN).await {
                Ok(connection) => {
                    let iface = format!("tunnel-{}", hex::encode(&node_id[..8]));
                    let _ = conn_tx.send(Some(ReadyConnection {
                        connection,
                        connection_id: conn_id,
                        interface_name: iface,
                    })).await;
                }
                Err(e) => {
                    tracing::warn!(err = %e, "outbound tunnel connection failed");
                    let _ = conn_tx.send(None).await;
                }
            }
        });
    }
}
```

Handle `InitiateTunnel` directly in the action dispatch loop in the select body (where `tick()` results are processed), NOT inside `dispatch_action`. This keeps `dispatch_action` focused on Reticulum/Zenoh I/O while tunnel connection management stays in the main loop where `iroh_endpoint`, `conn_tx`, `next_connection_id`, and `inflight_handshakes` are in scope. Pattern: match on `InitiateTunnel` before calling `dispatch_action` for other variants.

- [ ] **Step 2: Feed TunnelBridgeEvent::HandshakeComplete back to PeerManager**

In the tunnel bridge arm (Arm 4), when `HandshakeComplete` arrives, also push a `TunnelPeerEstablished` event to the runtime:

```rust
TunnelBridgeEvent::HandshakeComplete {
    interface_name,
    peer_node_id,
    peer_dsa_pubkey: _,
    connection_id,
} => {
    // ... existing connection_id guard ...
    if is_current {
        runtime.push_event(RuntimeEvent::TunnelHandshakeComplete {
            interface_name: interface_name.clone(),
            peer_node_id,
        });
        // Also notify PeerManager
        // Look up identity_hash from node_id via contact store
        if let Some(contact) = runtime.contact_store().find_by_tunnel_node_id(&peer_node_id) {
            runtime.push_event(RuntimeEvent::TunnelPeerEstablished {
                identity_hash: contact.identity_hash,
                node_id: peer_node_id,
                now: now_ms(),
            });
        }
    }
}
```

- [ ] **Step 3: Feed TunnelClosed back to PeerManager**

In the `TunnelClosed` arm, notify PeerManager:

```rust
if is_current {
    // ... existing cleanup ...
    // Notify PeerManager of tunnel drop
    if let Some(contact) = runtime.contact_store().find_by_tunnel_node_id(/* need node_id */) {
        runtime.push_event(RuntimeEvent::TunnelPeerDropped {
            identity_hash: contact.identity_hash,
        });
    }
}
```

The event loop already maintains `tunnel_senders: HashMap<String, TunnelSender>`. Add a parallel `tunnel_identities: HashMap<String, [u8; 16]>` mapping `interface_name → identity_hash`. Populate it when `HandshakeComplete` arrives (after looking up identity_hash from node_id via contact store). Use it in the `TunnelClosed` handler to find the identity_hash for the PeerManager notification.

- [ ] **Step 4: Handle SendPathRequest**

```rust
RuntimeAction::SendPathRequest { identity_hash } => {
    // Path requests are Reticulum-level — send a path request packet
    // via the router. For now, log it — full path request support
    // requires Node::send_path_request() which may not exist yet.
    tracing::debug!(?identity_hash, "path request for peer (not yet implemented)");
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): route PeerActions to tunnel infrastructure in event loop"
```

---

### Task 6: CLI command to add tunnel peers

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

Add a `--add-tunnel-peer` CLI arg that adds a contact with a tunnel address before starting the event loop. Format: `<identity_hash_hex>:<node_id_hex>[@relay_url]`.

- [ ] **Step 1: Add CLI arg**

```rust
        /// Add a tunnel peer contact before starting.
        /// Format: <identity_hash_hex>:<node_id_hex>[@relay_url]
        /// Example: aabbccdd...:<node_id>@https://iroh.q8.fyi
        #[arg(long, value_name = "PEER_SPEC")]
        add_tunnel_peer: Vec<String>,
```

- [ ] **Step 2: Parse and add contacts before event loop starts**

After creating the runtime but before calling `event_loop::run()`:

```rust
for spec in &add_tunnel_peer {
    let (id_and_node, relay) = if let Some(idx) = spec.find('@') {
        (&spec[..idx], Some(spec[idx + 1..].to_string()))
    } else {
        (spec.as_str(), None)
    };

    let parts: Vec<&str> = id_and_node.split(':').collect();
    if parts.len() != 2 {
        return Err(format!("invalid --add-tunnel-peer format: {spec}").into());
    }

    let id_bytes = hex::decode(parts[0])
        .map_err(|e| format!("invalid identity_hash hex: {e}"))?;
    let node_bytes = hex::decode(parts[1])
        .map_err(|e| format!("invalid node_id hex: {e}"))?;

    if id_bytes.len() != 16 || node_bytes.len() != 32 {
        return Err("identity_hash must be 16 bytes, node_id must be 32 bytes".into());
    }

    let mut identity_hash = [0u8; 16];
    identity_hash.copy_from_slice(&id_bytes);
    let mut node_id = [0u8; 32];
    node_id.copy_from_slice(&node_bytes);

    let contact = harmony_contacts::Contact {
        identity_hash,
        display_name: None,
        peering: harmony_contacts::PeeringPolicy {
            enabled: true,
            priority: harmony_contacts::PeeringPriority::High,
        },
        added_at: 0,
        last_seen: None,
        notes: None,
        addresses: vec![harmony_contacts::ContactAddress::Tunnel {
            node_id,
            relay_url: relay,
            direct_addrs: vec![],
        }],
    };

    rt.contact_store_mut().add(contact)
        .map_err(|e| format!("failed to add tunnel peer: {e}"))?;

    tracing::info!(
        identity = %hex::encode(identity_hash),
        node_id = %hex::encode(&node_id[..8]),
        "added tunnel peer contact"
    );
}
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(node): add --add-tunnel-peer CLI for manual tunnel peer registration"
```

---

### Task 7: Integration tests and cleanup

**Files:**
- Modify: various (clippy fixes)

- [ ] **Step 1: Run clippy across affected crates**

Run: `cargo clippy -p harmony-contacts -p harmony-peers -p harmony-node`
Fix any warnings.

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`
Verify no regressions.

- [ ] **Step 3: Commit cleanup**

```bash
git add -A
git commit -m "chore: clippy fixes and cleanup for tunnel peer lifecycle"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | ContactAddress enum in harmony-contacts | `Tunnel { node_id, relay_url, direct_addrs }` |
| 2 | Tunnel events/actions in harmony-peers | `InitiateTunnel`, `TunnelEstablished/Failed/Dropped`, `ConnectionQuality` |
| 3 | Tunnel-aware PeerManager state transitions | Tunnel-first strategy, Reticulum fallback on drop |
| 4 | Wire PeerManager into NodeRuntime | Fields + event routing + action translation |
| 5 | Route PeerActions in event loop | `InitiateTunnel` → iroh connect, HandshakeComplete → PeerManager |
| 6 | CLI command for manual tunnel peers | `--add-tunnel-peer <id>:<node_id>[@relay]` |
| 7 | Integration tests and cleanup | Clippy clean, workspace tests pass |

**Notes:**
- Outbound initiator connections now work via PeerManager → event loop → iroh connect → tunnel task. The full lifecycle: add contact → PeerManager searches → announce received → InitiateTunnel → iroh connect → HandshakeComplete → PeerEstablished → Connected.
- Discovery-driven address population (Bead harmony-lbv) will later auto-populate ContactAddress::Tunnel from AnnounceRecords, eliminating the need for manual `--add-tunnel-peer`.
- Reticulum path request support (`SendPathRequest`) is stubbed — full implementation requires `Node::send_path_request()` which is a separate concern.

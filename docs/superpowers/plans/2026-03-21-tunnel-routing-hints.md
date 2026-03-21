# Tunnel Routing Hints Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Note:** The `- [ ]` checkboxes below are execution tracking markers for the agentic worker, not persistent TODO items. Task tracking uses `bd` (beads) — see bead `harmony-lbv`.

**Goal:** Extend `harmony-discovery` with `RoutingHint::Tunnel` so peers can advertise their iroh-net tunnel reachability via AnnounceRecords, and wire DiscoveryManager into the runtime so tunnel hints auto-populate the contact store.

**Architecture:** Three layers: (1) add `RoutingHint::Tunnel` variant to harmony-discovery (the only library change — everything else already supports arbitrary routing hints), (2) wire DiscoveryManager into NodeRuntime so `IdentityDiscovered` actions are processed, (3) when a discovered peer has tunnel hints, auto-populate or update the contact store's `ContactAddress::Tunnel`, which triggers PeerManager to initiate connections.

**Tech Stack:** Rust, `harmony-discovery` (DiscoveryManager, AnnounceRecord, RoutingHint), `harmony-contacts` (ContactStore, ContactAddress), `harmony-node` (NodeRuntime)

**Spec:** `docs/superpowers/specs/2026-03-20-tunnel-peer-infrastructure-design.md` — Section 5

**Scope:** This plan covers **Bead harmony-lbv** only — the last code bead in the tunnel infrastructure chain.

**Prerequisites:** PRs #82, #84, #89 merged (harmony-tunnel, iroh-net integration, peer lifecycle).

---

## File Structure

```
crates/harmony-discovery/src/
├── record.rs       — Add RoutingHint::Tunnel variant
└── manager.rs      — No changes needed (IdentityDiscovered already emits full records)

crates/harmony-node/src/
└── runtime.rs      — Wire DiscoveryManager, process IdentityDiscovered with tunnel hints
```

---

### Task 1: Add RoutingHint::Tunnel variant

**Files:**
- Modify: `crates/harmony-discovery/src/record.rs`

The only library change. Add the `Tunnel` variant to `RoutingHint`. The `AnnounceRecord`, `AnnounceBuilder`, and `DiscoveryManager` already handle `Vec<RoutingHint>` generically — no changes needed elsewhere.

- [ ] **Step 1: Write test for Tunnel routing hint serialization**

In `crates/harmony-discovery/src/record.rs`, add to the test module:

```rust
#[test]
fn tunnel_routing_hint_roundtrip() {
    let hint = RoutingHint::Tunnel {
        node_id: [0xAA; 32],
        relay_url: Some("https://iroh.q8.fyi".into()),
        direct_addrs: vec!["192.168.1.10:4242".into()],
    };
    let bytes = postcard::to_allocvec(&hint).unwrap();
    let decoded: RoutingHint = postcard::from_bytes(&bytes).unwrap();
    assert_eq!(hint, decoded);
}

#[test]
fn announce_with_tunnel_hint_roundtrip() {
    // Build an AnnounceRecord with a Tunnel routing hint and verify
    // the full record serializes/deserializes correctly.
    let hint = RoutingHint::Tunnel {
        node_id: [0xBB; 32],
        relay_url: None,
        direct_addrs: vec![],
    };
    let builder = AnnounceBuilder::new(
        IdentityRef { hash: [0x01; 16], suite: CryptoSuite::Ed25519 },
        vec![0u8; 32], // dummy public key
        1000,
        2000,
        [0u8; 16],
    );
    let mut builder = builder;
    builder.add_routing_hint(hint.clone());
    let payload = builder.signable_payload();
    // Sign with dummy signature for test purposes
    let record = builder.build(vec![0u8; 64]);

    // Verify the hint survives serialization
    assert_eq!(record.routing_hints.len(), 1);
    assert_eq!(record.routing_hints[0], hint);

    let serialized = record.serialize().unwrap();
    let deserialized = AnnounceRecord::deserialize(&serialized).unwrap();
    assert_eq!(deserialized.routing_hints[0], hint);
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-discovery tunnel_routing_hint`
Expected: FAIL — `RoutingHint::Tunnel` variant not defined.

- [ ] **Step 3: Add the Tunnel variant**

In `record.rs`, add to `RoutingHint`:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RoutingHint {
    /// Reticulum destination hash (16 bytes).
    Reticulum { destination_hash: [u8; 16] },
    /// Zenoh locator (e.g. `"tcp/192.168.1.1:7447"`).
    Zenoh { locator: alloc::string::String },
    /// iroh-net tunnel: NodeId + optional relay + optional direct addresses.
    Tunnel {
        /// BLAKE3(ML-DSA-65 public key) — the iroh NodeId seed.
        node_id: [u8; 32],
        /// Preferred relay server URL (e.g., "https://iroh.q8.fyi").
        relay_url: Option<alloc::string::String>,
        /// Known direct socket addresses (ephemeral, may be stale).
        /// Wire format uses strings; parsed to SocketAddr at boundaries.
        direct_addrs: alloc::vec::Vec<alloc::string::String>,
    },
}
```

- [ ] **Step 4: Bump FORMAT_VERSION**

Adding a new enum variant changes the postcard wire format. Old deserializers seeing variant index 2 (Tunnel) would fail at the postcard level. Bump the record FORMAT_VERSION so old code rejects the entire record cleanly:

```rust
const FORMAT_VERSION: u8 = 2;
```

Update the deserialize check to accept both v1 and v2 (v1 records simply won't contain Tunnel hints — they'll deserialize fine since existing variants are unchanged):

```rust
if data[0] != FORMAT_VERSION && data[0] != 1 {
    return Err(DiscoveryError::DeserializeError("unsupported format version"));
}
```

Note: Unlike the ContactStore case, postcard can deserialize old v1 records because the `RoutingHint` variants haven't changed — they're enum variants with the same indices (0 = Reticulum, 1 = Zenoh). Only new records with `Tunnel` (index 2) would fail on old code, which is expected.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-discovery`
Expected: All tests pass.

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-discovery/
git commit -m "feat(discovery): add RoutingHint::Tunnel for iroh-net tunnel advertising"
```

---

### Task 2: Wire DiscoveryManager into NodeRuntime

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`
- Modify: `crates/harmony-node/Cargo.toml` (add harmony-discovery dep if not present)

Add `DiscoveryManager` as a field on `NodeRuntime`. Feed discovery events from Zenoh subscriptions and timer ticks. Process `DiscoveryAction::IdentityDiscovered` to extract tunnel hints and auto-populate the contact store.

- [ ] **Step 1: Add harmony-discovery dependency**

In `crates/harmony-node/Cargo.toml`, add if not already present:

```toml
harmony-discovery = { workspace = true, features = ["std"] }
```

- [ ] **Step 2: Add DiscoveryManager field to NodeRuntime**

```rust
discovery: harmony_discovery::DiscoveryManager,
```

Initialize in `NodeRuntime::new()`:

```rust
discovery: harmony_discovery::DiscoveryManager::new(),
```

- [ ] **Step 3: Add RuntimeEvent variant for discovery**

```rust
/// A signed announce record was received from the network.
DiscoveryAnnounceReceived {
    record_bytes: Vec<u8>,
    now: u64,
},
```

- [ ] **Step 4: Handle DiscoveryAnnounceReceived in push_event**

```rust
RuntimeEvent::DiscoveryAnnounceReceived { record_bytes, now } => {
    // Deserialize and verify the announce record.
    let record = match harmony_discovery::AnnounceRecord::deserialize(&record_bytes) {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("invalid announce record: {e:?}");
            return;
        }
    };
    if let Err(e) = harmony_discovery::verify_announce(&record, now / 1000) {
        tracing::debug!("announce verification failed: {e:?}");
        return;
    }

    // Feed to DiscoveryManager.
    let actions = self.discovery.on_event(
        harmony_discovery::DiscoveryEvent::AnnounceReceived {
            record,
            now: now / 1000, // DiscoveryManager uses seconds
        },
    );
    self.dispatch_discovery_actions(actions);
}
```

- [ ] **Step 5: Implement dispatch_discovery_actions**

```rust
fn dispatch_discovery_actions(&mut self, actions: Vec<harmony_discovery::DiscoveryAction>) {
    use harmony_discovery::DiscoveryAction;
    for action in actions {
        match action {
            DiscoveryAction::IdentityDiscovered { record } => {
                self.process_discovered_tunnel_hints(&record);
            }
            // Other actions (PublishAnnounce, RespondToQuery, etc.) are
            // handled when the full Zenoh discovery namespace is wired.
            // For now, only IdentityDiscovered matters for tunnel hints.
            _ => {}
        }
    }
}
```

- [ ] **Step 6: Implement process_discovered_tunnel_hints**

This is the core logic: extract `RoutingHint::Tunnel` from the discovered record and auto-populate `ContactAddress::Tunnel` in the contact store.

```rust
fn process_discovered_tunnel_hints(&mut self, record: &harmony_discovery::AnnounceRecord) {
    use harmony_contacts::{Contact, ContactAddress, PeeringPolicy, PeeringPriority};
    use harmony_discovery::RoutingHint;

    let identity_hash = record.identity_ref.hash;

    // Extract tunnel hints from the announce record.
    let tunnel_hints: Vec<_> = record.routing_hints.iter().filter_map(|hint| {
        if let RoutingHint::Tunnel { node_id, relay_url, direct_addrs } = hint {
            Some(ContactAddress::Tunnel {
                node_id: *node_id,
                relay_url: relay_url.clone(),
                direct_addrs: direct_addrs.clone(),
            })
        } else {
            None
        }
    }).collect();

    if tunnel_hints.is_empty() {
        return; // No tunnel hints — nothing to do.
    }

    // Update existing contact or create a new one with discovered addresses.
    if let Some(contact) = self.contact_store.get_mut(&identity_hash) {
        // Update tunnel addresses on existing contact (replace stale ones).
        contact.addresses.retain(|a| !matches!(a, ContactAddress::Tunnel { .. }));
        contact.addresses.extend(tunnel_hints);
        // Notify PeerManager that contact addresses changed.
        let peer_actions = self.peer_manager.on_event(
            harmony_peers::PeerEvent::ContactChanged { identity_hash },
            &self.contact_store,
        );
        self.translate_peer_actions(peer_actions);
    } else {
        // Auto-create contact for discovered peer with tunnel addresses.
        // Peering is enabled at Normal priority — the peer announced themselves
        // on the network, so they're a legitimate discovery target.
        let unix_now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let contact = Contact {
            identity_hash,
            display_name: None,
            peering: PeeringPolicy {
                enabled: true,
                priority: PeeringPriority::Normal,
            },
            added_at: unix_now,
            last_seen: None,
            notes: None,
            addresses: tunnel_hints,
        };
        if self.contact_store.add(contact).is_ok() {
            let peer_actions = self.peer_manager.on_event(
                harmony_peers::PeerEvent::ContactChanged { identity_hash },
                &self.contact_store,
            );
            self.translate_peer_actions(peer_actions);
        }
    }
}
```

- [ ] **Step 7: Feed DiscoveryManager tick from TimerTick**

In the `TimerTick` handler, add (after the existing PeerManager tick):

```rust
// Discovery tick — expire stale records.
let disc_actions = self.discovery.on_event(
    harmony_discovery::DiscoveryEvent::Tick { now: self.last_now / 1000 },
);
self.dispatch_discovery_actions(disc_actions);
```

- [ ] **Step 8: Write test for tunnel hint auto-population**

```rust
#[test]
fn discovery_announce_with_tunnel_hint_creates_contact() {
    let (mut rt, _) = make_runtime();

    // Build a fake announce record with a tunnel hint.
    // We can't easily sign it, so test process_discovered_tunnel_hints directly.
    let record = harmony_discovery::AnnounceRecord {
        identity_ref: harmony_identity::IdentityRef {
            hash: [0xDD; 16],
            suite: harmony_identity::CryptoSuite::Ed25519,
        },
        public_key: vec![0u8; 32],
        routing_hints: vec![harmony_discovery::RoutingHint::Tunnel {
            node_id: [0xEE; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
            direct_addrs: vec![],
        }],
        published_at: 1000,
        expires_at: 2000,
        nonce: [0u8; 16],
        signature: vec![0u8; 64],
    };

    rt.process_discovered_tunnel_hints(&record);

    // Contact should have been auto-created with tunnel address.
    let contact = rt.contact_store().get(&[0xDD; 16]).unwrap();
    assert_eq!(contact.addresses.len(), 1);
    assert!(matches!(
        &contact.addresses[0],
        harmony_contacts::ContactAddress::Tunnel { node_id, relay_url, .. }
        if *node_id == [0xEE; 32] && relay_url.as_deref() == Some("https://iroh.q8.fyi")
    ));
}

#[test]
fn discovery_tunnel_hint_updates_existing_contact() {
    let (mut rt, _) = make_runtime();

    // Pre-add a contact with an old tunnel address.
    let contact = harmony_contacts::Contact {
        identity_hash: [0xFF; 16],
        display_name: None,
        peering: harmony_contacts::PeeringPolicy {
            enabled: true,
            priority: harmony_contacts::PeeringPriority::High,
        },
        added_at: 1000,
        last_seen: None,
        notes: None,
        addresses: vec![harmony_contacts::ContactAddress::Tunnel {
            node_id: [0x11; 32],
            relay_url: None,
            direct_addrs: vec![],
        }],
    };
    rt.contact_store_mut().add(contact).unwrap();

    // Discovery announces a new tunnel address for the same peer.
    let record = harmony_discovery::AnnounceRecord {
        identity_ref: harmony_identity::IdentityRef {
            hash: [0xFF; 16],
            suite: harmony_identity::CryptoSuite::Ed25519,
        },
        public_key: vec![0u8; 32],
        routing_hints: vec![harmony_discovery::RoutingHint::Tunnel {
            node_id: [0x22; 32], // new node_id
            relay_url: Some("https://new-relay.example.com".into()),
            direct_addrs: vec!["1.2.3.4:4242".into()],
        }],
        published_at: 2000,
        expires_at: 3000,
        nonce: [0u8; 16],
        signature: vec![0u8; 64],
    };

    rt.process_discovered_tunnel_hints(&record);

    // Old tunnel address should be replaced, not duplicated.
    let contact = rt.contact_store().get(&[0xFF; 16]).unwrap();
    assert_eq!(contact.addresses.len(), 1);
    assert!(matches!(
        &contact.addresses[0],
        harmony_contacts::ContactAddress::Tunnel { node_id, .. }
        if *node_id == [0x22; 32]
    ));
}
```

- [ ] **Step 9: Verify tests pass**

Run: `cargo test -p harmony-node discovery_announce`

- [ ] **Step 10: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): wire DiscoveryManager, auto-populate contacts from tunnel hints"
```

---

### Task 3: Publish local tunnel hints in announces

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

When the node has an iroh Endpoint, include the tunnel `RoutingHint` in the local AnnounceRecord so other peers can discover this node's tunnel reachability.

- [ ] **Step 1: Add RuntimeEvent for local tunnel info**

```rust
/// The iroh Endpoint's addressing info has been determined.
/// Used to include tunnel routing hints in our AnnounceRecords.
LocalTunnelInfo {
    node_id: [u8; 32],
    relay_url: Option<String>,
},
```

- [ ] **Step 2: Store local tunnel hint on NodeRuntime**

Add a field:
```rust
local_tunnel_hint: Option<harmony_discovery::RoutingHint>,
```

Handle the event:
```rust
RuntimeEvent::LocalTunnelInfo { node_id, relay_url } => {
    self.local_tunnel_hint = Some(harmony_discovery::RoutingHint::Tunnel {
        node_id,
        relay_url,
        direct_addrs: vec![],
    });
}
```

- [ ] **Step 3: Include tunnel hint when building local AnnounceRecord**

Find where the local AnnounceRecord is built (or add a method to build it). When `local_tunnel_hint` is `Some`, add it to the record's routing hints via `builder.add_routing_hint()`.

Note: The full announce publishing flow (sign → publish via Zenoh) may not be fully wired yet. If so, just ensure the `DiscoveryManager::set_local_record()` path includes the tunnel hint. The actual Zenoh publishing is handled by `DiscoveryAction::PublishAnnounce` which the event loop will dispatch when Zenoh integration is complete.

- [ ] **Step 4: Feed LocalTunnelInfo from event loop**

In `event_loop.rs`, after the iroh Endpoint is bound, push the local tunnel info:

```rust
if let Some(ref ep) = iroh_endpoint {
    let node_id_bytes = ep.node_id().as_bytes();
    let relay_url = tunnel_config.as_ref()
        .and_then(|c| c.relay_url.clone());
    runtime.push_event(RuntimeEvent::LocalTunnelInfo {
        node_id: *node_id_bytes,
        relay_url,
    });
}
```

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): include tunnel routing hints in local AnnounceRecords"
```

---

### Task 4: Integration tests and cleanup

**Files:**
- Various (clippy fixes)

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-discovery -p harmony-node`

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`

- [ ] **Step 3: Commit cleanup**

```bash
git add -A
git commit -m "chore: clippy fixes for tunnel routing hints"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Add `RoutingHint::Tunnel` to harmony-discovery | New variant with node_id, relay_url, direct_addrs |
| 2 | Wire DiscoveryManager into NodeRuntime | Auto-populate contacts from tunnel hints on IdentityDiscovered |
| 3 | Publish local tunnel hints | Include iroh NodeId in outgoing AnnounceRecords |
| 4 | Integration tests and cleanup | Clippy clean, workspace tests pass |

**Completion note:** This is the last code bead in the tunnel infrastructure chain. After this, peers can:
1. Advertise their tunnel reachability via signed AnnounceRecords
2. Discover other peers' tunnel addresses automatically
3. Auto-populate the contact store, triggering PeerManager to initiate connections
4. Establish PQ-authenticated encrypted tunnels via iroh-net

The only remaining bead (`harmony-84h`) is the relay server deployment at `iroh.q8.fyi` — an ops task, not code.

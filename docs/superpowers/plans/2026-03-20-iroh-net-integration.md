# iroh-net Integration in `harmony-node` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire iroh-net into the harmony-node async event loop so tunnel peers can connect via QUIC, with automatic relay fallback and hole-punching.

**Architecture:** The iroh `Endpoint` runs alongside the existing UDP socket and Zenoh session. A bridge channel (like the existing Zenoh bridge) forwards tunnel events from spawned per-connection tasks to the main `select!` loop. Each tunnel connection spawns a task that drives a `TunnelSession` state machine, converting between iroh `Connection` streams and `TunnelEvent`/`TunnelAction`. When a tunnel completes its PQ handshake, the event loop registers a virtual Reticulum interface (`PointToPoint` mode) and a Zenoh session for the peer.

**Tech Stack:** `iroh` 0.91 (Endpoint, Connection, SecretKey, RelayMode — 0.96 has ml-dsa digest-chain conflicts), `harmony-tunnel` (TunnelSession), tokio (select!, mpsc, spawn_local)

**Spec:** `docs/superpowers/specs/2026-03-20-tunnel-peer-infrastructure-design.md` — Section 2

**Scope:** This plan covers **Bead harmony-dgb** only. Beads #3-5 (peer lifecycle, discovery hints, relay deployment) have separate plans.

**MSRV note:** iroh 0.91 requires MSRV 1.82 / edition 2021. Our workspace MSRV is 1.85 / edition 2021. This works because iroh is a *dependency*, not a workspace member — Cargo applies the dependency's own edition/MSRV rules when compiling it. The workspace `rust-version = "1.85"` in `Cargo.toml` applies only to our own crates.

---

## File Structure

```
crates/harmony-node/
├── Cargo.toml                  — Add iroh + harmony-tunnel deps
└── src/
    ├── main.rs                 — Add --relay-url and --tunnel-peer CLI args
    ├── event_loop.rs           — Add iroh Endpoint setup + 4th select! arm
    ├── tunnel_bridge.rs        — NEW: TunnelBridge (mpsc channel + TunnelEvent enum)
    ├── tunnel_task.rs          — NEW: per-connection async task (iroh streams ↔ TunnelSession)
    └── runtime.rs              — Add RuntimeEvent/RuntimeAction variants for tunnels
```

---

### Task 1: Add iroh dependency and tunnel bridge types

**Files:**
- Modify: `Cargo.toml` (workspace root — add iroh to workspace deps)
- Modify: `crates/harmony-node/Cargo.toml` (add iroh + harmony-tunnel deps)
- Create: `crates/harmony-node/src/tunnel_bridge.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod tunnel_bridge;`)

- [ ] **Step 1: Add iroh to workspace dependencies**

In `Cargo.toml` (workspace root), under `[workspace.dependencies]` in the `# Networking` section:

```toml
iroh = "0.91"
```

- [ ] **Step 2: Add iroh + harmony-tunnel to harmony-node deps**

In `crates/harmony-node/Cargo.toml`, add to `[dependencies]`:

```toml
harmony-tunnel = { workspace = true, features = ["std"] }
iroh = { workspace = true }
```

- [ ] **Step 3: Create `tunnel_bridge.rs`**

This module defines the bridge types between spawned tunnel tasks and the main select loop, following the same pattern as the existing `ZenohEvent` bridge.

```rust
//! Bridge between spawned iroh tunnel tasks and the main event loop.
//!
//! Each active tunnel connection runs in its own spawned task. Events flow
//! back to the select loop via an mpsc channel, matching the pattern used
//! for Zenoh task bridging.

use tokio::sync::mpsc;

/// Events sent from tunnel tasks to the main select loop.
#[derive(Debug)]
pub enum TunnelBridgeEvent {
    /// A tunnel handshake completed — peer is authenticated.
    HandshakeComplete {
        /// Short hex identifier for the tunnel interface name.
        interface_name: String,
        /// The remote peer's NodeId (BLAKE3 of ML-DSA pubkey), for routing.
        peer_node_id: [u8; 32],
        /// The remote peer's ML-DSA public key bytes.
        peer_dsa_pubkey: Vec<u8>,
    },
    /// A decrypted Reticulum packet arrived from a tunnel peer.
    ReticulumReceived {
        interface_name: String,
        packet: Vec<u8>,
    },
    /// A decrypted Zenoh message arrived from a tunnel peer.
    ZenohReceived {
        interface_name: String,
        message: Vec<u8>,
    },
    /// A tunnel connection was closed or errored.
    TunnelClosed {
        interface_name: String,
        reason: String,
    },
}

/// Handle for sending Reticulum packets into a tunnel (held by the event loop).
///
/// When the event loop needs to send a packet on a tunnel interface
/// (RuntimeAction::SendOnInterface with a tunnel interface name),
/// it looks up the corresponding TunnelSender and forwards the packet.
#[derive(Debug, Clone)]
pub struct TunnelSender {
    tx: mpsc::Sender<TunnelCommand>,
}

/// Commands sent from the event loop to a tunnel task.
#[derive(Debug)]
pub enum TunnelCommand {
    /// Send a Reticulum packet through this tunnel.
    SendReticulum { packet: Vec<u8> },
    /// Send a Zenoh message through this tunnel.
    SendZenoh { message: Vec<u8> },
    /// Close this tunnel.
    Close,
}

impl TunnelSender {
    pub fn new(tx: mpsc::Sender<TunnelCommand>) -> Self {
        Self { tx }
    }

    pub async fn send_reticulum(&self, packet: Vec<u8>) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::SendReticulum { packet }).await
    }

    pub async fn send_zenoh(&self, message: Vec<u8>) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::SendZenoh { message }).await
    }

    pub async fn close(&self) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::Close).await
    }
}
```

- [ ] **Step 4: Register module in `main.rs`**

Add `mod tunnel_bridge;` and `mod tunnel_task;` (task module created in Task 2) to the module declarations in `crates/harmony-node/src/main.rs`.

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-node`
Expected: Compiles (warnings about unused are OK — `tunnel_task` module may need a stub).

Create a temporary stub for `tunnel_task.rs`:
```rust
// Per-connection tunnel task — implemented in Task 2.
```

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/ Cargo.toml
git commit -m "feat(node): add iroh dependency and tunnel bridge types"
```

---

### Task 2: Per-connection tunnel task

**Files:**
- Create: `crates/harmony-node/src/tunnel_task.rs` (replace stub)

This is the async task that runs per iroh Connection. It bridges between iroh QUIC streams and the `TunnelSession` sans-I/O state machine, forwarding events to the main loop via the bridge channel.

- [ ] **Step 1: Implement the tunnel task**

Replace `crates/harmony-node/src/tunnel_task.rs`:

```rust
//! Per-connection async task that bridges iroh QUIC streams to TunnelSession.
//!
//! One task is spawned per active tunnel connection (inbound or outbound).
//! It owns a TunnelSession state machine and drives it by:
//! 1. Reading bytes from the iroh RecvStream → TunnelEvent::InboundBytes
//! 2. Processing TunnelActions: OutboundBytes → iroh SendStream
//! 3. Forwarding ReticulumReceived/ZenohReceived → TunnelBridgeEvent via mpsc
//! 4. Running a keepalive timer → TunnelEvent::Tick

use std::time::{Duration, Instant};

use harmony_identity::PqPrivateIdentity;
use harmony_tunnel::session::TunnelSession;
use harmony_tunnel::{TunnelAction, TunnelEvent};
use iroh::endpoint::Connection;
use tokio::sync::mpsc;

use crate::tunnel_bridge::{TunnelBridgeEvent, TunnelCommand, TunnelSender};

/// ALPN protocol identifier for harmony tunnels.
pub const HARMONY_TUNNEL_ALPN: &[u8] = b"harmony-tunnel/1";

/// Run the initiator side of a tunnel connection.
///
/// Creates a TunnelSession, sends TunnelInit over the stream, waits for
/// TunnelAccept, then enters the main read/write loop.
pub async fn run_initiator(
    conn: Connection,
    local_identity: &PqPrivateIdentity,
    remote_identity: &harmony_identity::PqIdentity,
    bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
    cmd_rx: mpsc::Receiver<TunnelCommand>,
) {
    let now_ms = millis_since_start();
    let interface_name = format!("tunnel-{}", hex::encode(&conn.remote_id().as_bytes()[..4]));

    // Create TunnelSession as initiator
    let mut rng = rand::rngs::OsRng;
    let (mut session, init_actions) = match TunnelSession::new_initiator(
        &mut rng,
        local_identity,
        remote_identity,
        now_ms,
    ) {
        Ok(result) => result,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("handshake init failed: {e}"),
            }).await;
            return;
        }
    };

    // Open a bidirectional stream for the tunnel
    let (mut send_stream, mut recv_stream) = match conn.open_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("failed to open bi stream: {e}"),
            }).await;
            return;
        }
    };

    // Send TunnelInit
    for action in init_actions {
        if let TunnelAction::OutboundBytes { data } = action {
            if let Err(e) = write_length_prefixed(&mut send_stream, &data).await {
                let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("failed to send TunnelInit: {e}"),
                }).await;
                return;
            }
        }
    }

    // Read TunnelAccept
    let accept_bytes = match read_length_prefixed(&mut recv_stream).await {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("failed to read TunnelAccept: {e}"),
            }).await;
            return;
        }
    };

    // Process TunnelAccept
    let now_ms = millis_since_start();
    let actions = match session.handle_event(TunnelEvent::InboundBytes {
        data: accept_bytes,
        now_ms,
    }) {
        Ok(actions) => actions,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("handshake failed: {e}"),
            }).await;
            return;
        }
    };

    // Process handshake completion actions
    for action in &actions {
        if let TunnelAction::HandshakeComplete {
            peer_dsa_pubkey,
            peer_node_id,
        } = action
        {
            let _ = bridge_tx.send(TunnelBridgeEvent::HandshakeComplete {
                interface_name: interface_name.clone(),
                peer_node_id: *peer_node_id,
                peer_dsa_pubkey: peer_dsa_pubkey.clone(),
            }).await;
        }
    }

    // Enter main loop
    run_tunnel_loop(session, send_stream, recv_stream, bridge_tx, cmd_rx, interface_name).await;
}

/// Run the responder side of a tunnel connection.
///
/// Reads TunnelInit from the stream, creates TunnelSession as responder,
/// sends TunnelAccept, then enters the main read/write loop.
pub async fn run_responder(
    conn: Connection,
    local_identity: &PqPrivateIdentity,
    bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
    cmd_rx: mpsc::Receiver<TunnelCommand>,
) {
    let interface_name = format!("tunnel-{}", hex::encode(&conn.remote_id().as_bytes()[..4]));

    // Accept the bidirectional stream
    let (mut send_stream, mut recv_stream) = match conn.accept_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("failed to accept bi stream: {e}"),
            }).await;
            return;
        }
    };

    // Read TunnelInit
    let init_bytes = match read_length_prefixed(&mut recv_stream).await {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("failed to read TunnelInit: {e}"),
            }).await;
            return;
        }
    };

    // Create TunnelSession as responder
    let mut rng = rand::rngs::OsRng;
    let now_ms = millis_since_start();
    let (session, accept_actions) = match TunnelSession::new_responder(
        &mut rng,
        local_identity,
        &init_bytes,
        now_ms,
    ) {
        Ok(result) => result,
        Err(e) => {
            let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                interface_name,
                reason: format!("responder handshake failed: {e}"),
            }).await;
            return;
        }
    };

    // Process accept actions (send TunnelAccept + emit HandshakeComplete)
    for action in &accept_actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                if let Err(e) = write_length_prefixed(&mut send_stream, data).await {
                    let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                        interface_name,
                        reason: format!("failed to send TunnelAccept: {e}"),
                    }).await;
                    return;
                }
            }
            TunnelAction::HandshakeComplete {
                peer_dsa_pubkey,
                peer_node_id,
            } => {
                let _ = bridge_tx.send(TunnelBridgeEvent::HandshakeComplete {
                    interface_name: interface_name.clone(),
                    peer_node_id: *peer_node_id,
                    peer_dsa_pubkey: peer_dsa_pubkey.clone(),
                }).await;
            }
            _ => {}
        }
    }

    // Enter main loop
    run_tunnel_loop(session, send_stream, recv_stream, bridge_tx, cmd_rx, interface_name).await;
}

/// Main tunnel read/write/keepalive loop.
///
/// Runs until the tunnel is closed, errors, or the command channel drops.
async fn run_tunnel_loop(
    mut session: TunnelSession,
    mut send_stream: iroh::endpoint::SendStream,
    mut recv_stream: iroh::endpoint::RecvStream,
    bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
    mut cmd_rx: mpsc::Receiver<TunnelCommand>,
    interface_name: String,
) {
    // Timer fires at 10s intervals — more frequent than the 30s keepalive
    // interval, but the TunnelSession state machine decides when to actually
    // emit a keepalive frame (every 30s) and when to declare dead peer (90s).
    // The 10s tick ensures responsive timeout detection.
    let mut keepalive_timer = tokio::time::interval(Duration::from_secs(10));
    keepalive_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            // Read from iroh stream
            result = read_length_prefixed(&mut recv_stream) => {
                match result {
                    Ok(data) => {
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::InboundBytes { data, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, &bridge_tx, &interface_name,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                                    interface_name: interface_name.clone(),
                                    reason: format!("tunnel error: {e}"),
                                }).await;
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.clone(),
                            reason: format!("stream read error: {e}"),
                        }).await;
                        break;
                    }
                }
            }

            // Commands from the event loop
            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(TunnelCommand::SendReticulum { packet }) => {
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::SendReticulum { packet, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, &bridge_tx, &interface_name,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("[{interface_name}] send reticulum error: {e}");
                            }
                        }
                    }
                    Some(TunnelCommand::SendZenoh { message }) => {
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::SendZenoh { message, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, &bridge_tx, &interface_name,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("[{interface_name}] send zenoh error: {e}");
                            }
                        }
                    }
                    Some(TunnelCommand::Close) | None => {
                        let _ = session.handle_event(TunnelEvent::Close);
                        let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.clone(),
                            reason: "closed by event loop".to_string(),
                        }).await;
                        break;
                    }
                }
            }

            // Keepalive timer
            _ = keepalive_timer.tick() => {
                let now_ms = millis_since_start();
                match session.handle_event(TunnelEvent::Tick { now_ms }) {
                    Ok(actions) => {
                        if !dispatch_tunnel_actions(
                            &actions, &mut send_stream, &bridge_tx, &interface_name,
                        ).await {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.clone(),
                            reason: format!("tick error: {e}"),
                        }).await;
                        break;
                    }
                }
            }
        }
    }
}

/// Process TunnelActions: send outbound bytes, forward bridge events.
/// Returns false if the tunnel should be closed (Closed action received).
async fn dispatch_tunnel_actions(
    actions: &[TunnelAction],
    send_stream: &mut iroh::endpoint::SendStream,
    bridge_tx: &mpsc::Sender<TunnelBridgeEvent>,
    interface_name: &str,
) -> bool {
    for action in actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                if let Err(e) = write_length_prefixed(send_stream, data).await {
                    let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                        interface_name: interface_name.to_string(),
                        reason: format!("write error: {e}"),
                    }).await;
                    return false;
                }
            }
            TunnelAction::ReticulumReceived { packet } => {
                let _ = bridge_tx.send(TunnelBridgeEvent::ReticulumReceived {
                    interface_name: interface_name.to_string(),
                    packet: packet.clone(),
                }).await;
            }
            TunnelAction::ZenohReceived { message } => {
                let _ = bridge_tx.send(TunnelBridgeEvent::ZenohReceived {
                    interface_name: interface_name.to_string(),
                    message: message.clone(),
                }).await;
            }
            TunnelAction::Error { reason } => {
                let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                    interface_name: interface_name.to_string(),
                    reason: reason.clone(),
                }).await;
                return false;
            }
            TunnelAction::Closed => {
                let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                    interface_name: interface_name.to_string(),
                    reason: "session closed".to_string(),
                }).await;
                return false;
            }
            TunnelAction::HandshakeComplete { .. } => {
                // Already handled during handshake phase
            }
        }
    }
    true
}

// ── Wire helpers ────────────────────────────────────────────────────────────

/// Write a length-prefixed message: [4 bytes big-endian length][payload].
async fn write_length_prefixed(
    stream: &mut iroh::endpoint::SendStream,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::AsyncWriteExt;
    let len = (data.len() as u32).to_be_bytes();
    stream.write_all(&len).await?;
    stream.write_all(data).await?;
    Ok(())
}

/// Read a length-prefixed message: [4 bytes big-endian length][payload].
async fn read_length_prefixed(
    stream: &mut iroh::endpoint::RecvStream,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    use tokio::io::AsyncReadExt;
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    // Sanity check: reject messages larger than 16 MiB
    if len > 16 * 1024 * 1024 {
        return Err(format!("message too large: {len} bytes").into());
    }

    let mut buf = vec![0u8; len];
    stream.read_exact(&mut buf).await?;
    Ok(buf)
}

/// Monotonic milliseconds since process start (for TunnelSession timestamps).
fn millis_since_start() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis() as u64
}
```

- [ ] **Step 2: Verify it compiles**

Run: `cargo check -p harmony-node`
Expected: Compiles (warnings about unused functions are OK).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-node/src/tunnel_task.rs
git commit -m "feat(node): per-connection tunnel task bridging iroh streams to TunnelSession"
```

---

### Task 3: Add RuntimeEvent/RuntimeAction variants for tunnels

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

Add the event/action variants the event loop needs to interact with the runtime when tunnel events arrive.

- [ ] **Step 1: Add RuntimeEvent variants**

In `runtime.rs`, add to the `RuntimeEvent` enum:

```rust
    /// A tunnel handshake completed — register interface and session.
    TunnelHandshakeComplete {
        interface_name: String,
        peer_node_id: [u8; 32],
    },
    /// A Reticulum packet arrived via a tunnel interface.
    TunnelReticulumReceived {
        interface_name: String,
        packet: Vec<u8>,
    },
    /// A tunnel was closed.
    TunnelClosed {
        interface_name: String,
    },
```

- [ ] **Step 2: Handle new events in `push_event` / `tick`**

In the `push_event` method, add match arms for the new variants:

- `TunnelHandshakeComplete`: Register a new `InterfaceMode::PointToPoint` interface with the Reticulum Node.
- `TunnelReticulumReceived`: Feed the packet into the Reticulum Node as an `InboundPacket` on the tunnel interface.
- `TunnelClosed`: Log and clean up (no Reticulum deregister needed — path entries expire naturally).

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p harmony-node`

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): add tunnel RuntimeEvent variants for interface registration"
```

---

### Task 4: Wire iroh Endpoint into the event loop

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/main.rs` (CLI args)

This is the main integration task. Add the iroh Endpoint setup, the accept loop, and the fourth `select!` arm.

- [ ] **Step 1: Add CLI args for tunnel configuration**

In `main.rs`, add to the `Commands::Run` struct:

```rust
        /// Relay server URL for tunnel connections (e.g., https://iroh.q8.fyi).
        /// If not set, tunnel support is disabled.
        #[arg(long, value_name = "URL")]
        relay_url: Option<String>,
```

- [ ] **Step 2: Pass tunnel config to the event loop**

Update the `event_loop::run()` signature to accept optional tunnel config:

```rust
pub struct TunnelConfig {
    pub relay_url: Option<String>,
    pub local_identity: harmony_identity::PqPrivateIdentity,
}
```

Update the call in `main.rs` to pass the config when `--relay-url` is provided.

- [ ] **Step 3: Set up iroh Endpoint in the event loop**

In `event_loop.rs`, after the Zenoh session setup, conditionally create the iroh Endpoint:

```rust
    // ── iroh tunnel endpoint (optional) ────────────────────────────────────
    let (tunnel_tx, mut tunnel_rx) = mpsc::channel::<TunnelBridgeEvent>(256);
    let mut tunnel_senders: std::collections::HashMap<String, TunnelSender> = Default::default();

    let iroh_endpoint = if let Some(ref tunnel_config) = tunnel_config {
        let secret_key = derive_iroh_secret_key(&tunnel_config.local_identity);

        // Use empty_builder to avoid including N0's default relay servers.
        // We only want our own relay (iroh.q8.fyi) or no relay at all.
        let relay_mode = if tunnel_config.relay_url.is_some() {
            iroh::RelayMode::Disabled // Overridden below with Custom
        } else {
            iroh::RelayMode::Disabled
        };
        let mut builder = iroh::Endpoint::empty_builder(relay_mode)
            .alpns(vec![HARMONY_TUNNEL_ALPN.to_vec()])
            .secret_key(secret_key);

        if let Some(ref url) = tunnel_config.relay_url {
            let relay_url: iroh::RelayUrl = url.parse()
                .expect("invalid relay URL");
            let relay_map = iroh::RelayMap::from(relay_url);
            builder = builder.relay_mode(iroh::RelayMode::Custom(relay_map));
        }

        Some(builder.bind().await?)
    } else {
        None
    };
```

Where `derive_iroh_secret_key` derives a 32-byte Ed25519-shaped key from the PQ identity:

```rust
fn derive_iroh_secret_key(identity: &PqPrivateIdentity) -> iroh::SecretKey {
    let pub_id = identity.public_identity();
    let hash = harmony_crypto::hash::blake3_hash(&pub_id.verifying_key.as_bytes());
    // Use the hash as seed for iroh's SecretKey (Ed25519)
    iroh::SecretKey::from_bytes(&hash)
}
```

- [ ] **Step 4: Add the tunnel select! arm**

Add a fourth arm to the `select!` block for inbound tunnel bridge events, and a fifth arm for accepting new iroh connections:

```rust
    // Arm 4: Tunnel bridge events
    event = tunnel_rx.recv() => {
        if let Some(event) = event {
            match event {
                TunnelBridgeEvent::HandshakeComplete {
                    interface_name, peer_node_id, ..
                } => {
                    runtime.push_event(RuntimeEvent::TunnelHandshakeComplete {
                        interface_name,
                        peer_node_id,
                    });
                }
                TunnelBridgeEvent::ReticulumReceived {
                    interface_name, packet,
                } => {
                    runtime.push_event(RuntimeEvent::TunnelReticulumReceived {
                        interface_name,
                        packet,
                    });
                }
                TunnelBridgeEvent::ZenohReceived { .. } => {
                    // TODO: Route to Zenoh session (Bead #3)
                }
                TunnelBridgeEvent::TunnelClosed {
                    interface_name, reason,
                } => {
                    eprintln!("[{interface_name}] tunnel closed: {reason}");
                    tunnel_senders.remove(&interface_name);
                    runtime.push_event(RuntimeEvent::TunnelClosed { interface_name });
                }
            }
        }
    }

    // Arm 5: Accept incoming iroh connections
    conn = async {
        if let Some(ref ep) = iroh_endpoint {
            ep.accept().await
        } else {
            std::future::pending().await
        }
    } => {
        if let Some(incoming) = conn {
            match incoming.await {
                Ok(connection) => {
                    let (cmd_tx, cmd_rx) = mpsc::channel::<TunnelCommand>(64);
                    let iface = format!("tunnel-{}", hex::encode(&connection.remote_id().as_bytes()[..4]));
                    tunnel_senders.insert(iface.clone(), TunnelSender::new(cmd_tx));

                    let tx = tunnel_tx.clone();
                    let identity = tunnel_config.as_ref().unwrap().local_identity.clone();
                    tokio::spawn(async move {
                        tunnel_task::run_responder(connection, &identity, tx, cmd_rx).await;
                    });
                }
                Err(e) => {
                    eprintln!("iroh accept error: {e}");
                }
            }
        }
    }
```

- [ ] **Step 5: Route SendOnInterface to tunnel senders**

In `dispatch_action`, update the `SendOnInterface` handler to check if the interface name starts with `"tunnel-"`. If so, look up the `TunnelSender` and send the packet instead of UDP:

```rust
RuntimeAction::SendOnInterface { interface_name, raw } => {
    if interface_name.starts_with("tunnel-") {
        if let Some(sender) = tunnel_senders.get(interface_name.as_ref()) {
            let _ = sender.send_reticulum(raw).await;
        }
    } else {
        // Existing UDP send logic
        // ...
    }
}
```

- [ ] **Step 6: Verify it compiles**

Run: `cargo check -p harmony-node`

Note: This will likely require adjusting type signatures, lifetimes, and imports. The iroh `Accept` type, `Connection` type, and `SendStream`/`RecvStream` types come from `iroh::endpoint`. The `PqPrivateIdentity` may need `Clone` — check if it already implements it, and if not, find an alternative (e.g., `Arc<PqPrivateIdentity>`).

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): wire iroh Endpoint into event loop with accept and bridge"
```

---

### Task 5: Add outbound tunnel connection support

**Files:**
- Modify: `crates/harmony-node/src/event_loop.rs`
- Modify: `crates/harmony-node/src/main.rs`

Add the ability to connect to a known tunnel peer by EndpointAddr (for testing). Full peer-manager-driven connections come in Bead #3.

- [ ] **Step 1: Add `--tunnel-peer` CLI arg**

In `main.rs`, add to `Commands::Run`:

```rust
        /// Connect to a tunnel peer by iroh EndpointAddr (for testing).
        /// Format: the peer's EndpointAddr string.
        #[arg(long, value_name = "ADDR")]
        tunnel_peer: Option<String>,
```

- [ ] **Step 2: Initiate outbound connection after endpoint starts**

After the iroh Endpoint is bound, if `--tunnel-peer` is specified, spawn an outbound connection task:

```rust
if let (Some(ref ep), Some(ref peer_addr_str)) = (&iroh_endpoint, &tunnel_peer) {
    let addr: iroh::EndpointAddr = peer_addr_str.parse()
        .expect("invalid tunnel peer address");
    let conn = ep.connect(addr, HARMONY_TUNNEL_ALPN).await?;

    let (cmd_tx, cmd_rx) = mpsc::channel::<TunnelCommand>(64);
    let iface = format!("tunnel-{}", hex::encode(&conn.remote_id().as_bytes()[..4]));
    tunnel_senders.insert(iface, TunnelSender::new(cmd_tx));

    let tx = tunnel_tx.clone();
    let identity = tunnel_config.as_ref().unwrap().local_identity.clone();
    let remote_pub = /* need the remote PqIdentity — for now, skip MITM check */;
    tokio::spawn(async move {
        tunnel_task::run_initiator(conn, &identity, &remote_pub, tx, cmd_rx).await;
    });
}
```

Note: Full initiator support requires knowing the remote peer's PqIdentity (ML-KEM public key for encapsulation). This comes from the contact store (Bead #3) or discovery hints (Bead #4). For now, document this as a TODO — the responder path is fully functional for testing.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): add --tunnel-peer CLI arg for outbound tunnel connections"
```

---

### Task 6: Integration smoke test and cleanup

**Files:**
- Modify: `crates/harmony-node/src/main.rs` (cleanup)
- Modify: `crates/harmony-node/src/event_loop.rs` (cleanup)

- [ ] **Step 1: Run clippy**

Run: `cargo clippy -p harmony-node`
Fix any warnings.

- [ ] **Step 2: Run workspace tests**

Run: `RUST_MIN_STACK=8388608 cargo test --workspace`
Verify no regressions.

- [ ] **Step 3: Add iroh endpoint close on shutdown**

In the shutdown arm of the select! loop, close the iroh endpoint:

```rust
if let Some(ref ep) = iroh_endpoint {
    ep.close().await;
}
```

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-node/
git commit -m "feat(node): cleanup and graceful iroh endpoint shutdown"
```

---

## Summary

| Task | Description | Key Output |
|------|-------------|------------|
| 1 | Add iroh dep + tunnel bridge types | `tunnel_bridge.rs`, Cargo.toml changes |
| 2 | Per-connection tunnel task | `tunnel_task.rs` — iroh streams ↔ TunnelSession |
| 3 | RuntimeEvent/Action tunnel variants | Interface registration on handshake complete |
| 4 | Wire iroh Endpoint into event loop | 4th+5th select! arms, accept loop, SendOnInterface routing |
| 5 | Outbound tunnel connection support | `--tunnel-peer` CLI arg |
| 6 | Integration smoke test + cleanup | Clippy clean, workspace tests pass |

**Notes:**
- This plan focuses on the transport plumbing. The responder path is fully functional (any iroh peer can connect and exchange encrypted Reticulum packets). The initiator path requires the remote peer's PqIdentity, which comes from the contact store (Bead #3 — `harmony-h6k`) or discovery hints (Bead #4 — `harmony-lbv`). For now, outbound connections are stubbed with a TODO.
- **Zenoh session over tunnel:** The spec (Section 2, point 2 of Interface Registration) calls for opening a Zenoh Session over the tunnel's Zenoh frame path. This plan registers only the Reticulum interface on handshake. Zenoh session integration over tunnels is deferred to Bead #3 (`harmony-h6k`) where the peer lifecycle manages both protocol layers. The `ZenohReceived` bridge event is plumbed but marked TODO.
- **N0 relays excluded:** The plan uses `Endpoint::empty_builder()` to avoid including N0's default relay servers. Only the user-configured relay (e.g., `iroh.q8.fyi`) is used, or no relay if `--relay-url` is omitted.

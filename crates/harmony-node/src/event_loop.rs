//! Async event loop — drives `NodeRuntime` via tokio select! with UDP, timer, Zenoh, and iroh tunnels.
//!
//! `NodeRuntime` is `!Send`; it lives entirely on the select loop task.
//! Zenoh objects that need spawning are cloned (Session is internally Arc'd).

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use harmony_content::book::MemoryBookStore;
use harmony_identity::PqPrivateIdentity;
use zeroize::Zeroize;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;

use crate::runtime::{NodeRuntime, RuntimeAction, RuntimeEvent};
use crate::tunnel_bridge::{ReadyConnection, TunnelBridgeEvent, TunnelSender};
use crate::tunnel_task;

/// Configuration for iroh tunnel connectivity.
pub struct TunnelConfig {
    /// Optional relay URL for NAT traversal.
    pub relay_url: Option<String>,
    /// Node identity for ML-KEM tunnel handshakes.
    pub local_identity: Arc<PqPrivateIdentity>,
}

/// Internal bridge events from spawned Zenoh tasks to the select loop.
enum ZenohEvent {
    /// Inbound Zenoh query (non-compute).
    Query {
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Inbound Zenoh query on a compute key expression.
    ComputeQuery {
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Inbound Zenoh subscription sample.
    Subscription {
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Response to a FetchContent / FetchModule get() call.
    FetchResponse {
        cid: [u8; 32],
        is_module: bool,
        result: Result<Vec<u8>, String>,
    },
}

/// Run the async event loop.
///
/// # Parameters
/// - `runtime`: the sans-I/O node state machine (`!Send` — stays on this task).
/// - `startup_actions`: actions returned by `NodeRuntime::new`, executed before the loop.
/// - `listen_addr`: UDP socket address to bind (broadcast also sent here).
/// - `tunnel_config`: optional iroh tunnel configuration (enables tunnel accept/connect).
pub async fn run(
    mut runtime: NodeRuntime<MemoryBookStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
    tunnel_config: Option<TunnelConfig>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── UDP socket ────────────────────────────────────────────────────────────
    let udp = UdpSocket::bind(listen_addr).await?;
    udp.set_broadcast(true)?;
    let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", listen_addr.port())
        .parse()
        .expect("static broadcast addr");

    // ── Zenoh session ─────────────────────────────────────────────────────────
    let session = zenoh::open(zenoh::Config::default()).await?;

    // ── mpsc channel: Zenoh tasks → select loop ───────────────────────────────
    let (zenoh_tx, mut zenoh_rx) = mpsc::channel::<ZenohEvent>(256);

    // ── Tunnel bridge: spawned tunnel tasks → select loop ───────────────────
    const MAX_TUNNEL_CONNECTIONS: usize = 64;
    let (tunnel_tx, mut tunnel_rx) = mpsc::channel::<TunnelBridgeEvent>(256);
    let mut tunnel_senders: HashMap<String, TunnelSender> = HashMap::new();
    let mut next_connection_id: u64 = 0;

    // ── Ready-connection channel: QUIC handshake tasks → select loop ──────
    // Sends Some(ReadyConnection) on success, None on handshake failure.
    // None lets the event loop decrement inflight_handshakes even when the
    // spawned task never produces a usable connection.
    let (conn_tx, mut conn_rx) = mpsc::channel::<Option<ReadyConnection>>(64);
    // Counts QUIC handshakes that have been spawned but have not yet delivered
    // a result on conn_rx. Added to tunnel_senders.len() for the connection cap
    // so a burst of simultaneous connects cannot all slip under the limit.
    let mut inflight_handshakes: usize = 0;

    // ── iroh Endpoint (optional, gated on --relay-url) ─────────────────────
    let mut iroh_endpoint = if let Some(ref config) = tunnel_config {
        // Derive iroh SecretKey from PQ identity's ML-DSA *private* signing key.
        // This gives a deterministic mapping: same PQ identity → same iroh NodeId,
        // without leaking the secret — the verifying key is public, so deriving
        // from it would let anyone impersonate this node's iroh transport identity.
        let mut sk_bytes = config.local_identity.signing_key().as_bytes();
        let mut hash = harmony_crypto::hash::blake3_hash(&sk_bytes);
        sk_bytes.zeroize();
        let secret_key = iroh::SecretKey::from(hash);
        hash.zeroize(); // [u8; 32] is Copy — SecretKey::from copies it, zeroize the original

        let mut builder = iroh::Endpoint::builder()
            .alpns(vec![tunnel_task::HARMONY_TUNNEL_ALPN.to_vec()])
            .secret_key(secret_key);

        if let Some(ref url) = config.relay_url {
            let relay_url: iroh::RelayUrl = url
                .parse()
                .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                    format!("invalid relay URL '{url}': {e}").into()
                })?;
            let relay_map = iroh::RelayMap::from_iter([relay_url]);
            builder = builder.relay_mode(iroh::RelayMode::Custom(relay_map));
        } else {
            builder = builder.relay_mode(iroh::RelayMode::Disabled);
        }

        let ep = builder.bind().await.map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            format!("iroh endpoint bind failed: {e}").into()
        })?;
        tracing::info!(node_id = %ep.node_id(), "iroh tunnel endpoint ready");
        Some(ep)
    } else {
        None
    };

    // ── Execute startup actions (declare queryables + subscribers) ────────────
    for action in startup_actions {
        dispatch_action(
            action,
            &session,
            &zenoh_tx,
            &udp,
            &broadcast_addr,
            &tunnel_senders,
        )
        .await;
    }

    // ── Monotonic epoch ─────────────────────────────────────────────────────
    // Shared with tunnel_task::millis_since_start() via OnceLock — same epoch.
    let now_ms = crate::tunnel_task::millis_since_start;

    // ── Timer (250 ms tick) ───────────────────────────────────────────────────
    let mut timer = time::interval(Duration::from_millis(250));
    timer.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

    // ── Reusable UDP receive buffer ───────────────────────────────────────────
    // Max UDP datagram size — Reticulum MTU is 500 (default) or 1024 (medium
    // interfaces). Use 65535 to never silently truncate.
    let mut udp_buf = vec![0u8; 65535];

    // ── Monotonic query-id counter ────────────────────────────────────────────
    let mut next_query_id: u64 = 1;

    // ── Shutdown signal ───────────────────────────────────────────────────────
    // Register SIGTERM outside the async block so registration failures
    // propagate via ? instead of panicking inside the event loop.
    #[cfg(unix)]
    let mut sigterm = {
        use tokio::signal::unix::{signal, SignalKind};
        signal(SignalKind::terminate())
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("failed to register SIGTERM handler: {e}").into()
            })?
    };

    let shutdown = async {
        #[cfg(unix)]
        {
            tokio::select! {
                // Some(_) pattern: if recv() returns None (stream exhausted),
                // the arm is disabled rather than triggering a spurious shutdown.
                Some(_) = sigterm.recv() => tracing::info!("SIGTERM received — shutting down"),
                result = tokio::signal::ctrl_c() => match result {
                    Ok(()) => tracing::info!("Ctrl+C received — shutting down"),
                    Err(e) => tracing::warn!(err = %e, "SIGINT handler failed — shutting down"),
                },
            }
        }
        #[cfg(not(unix))]
        {
            match tokio::signal::ctrl_c().await {
                Ok(()) => tracing::info!("Ctrl+C received — shutting down"),
                Err(e) => tracing::warn!(err = %e, "SIGINT handler failed — shutting down"),
            }
        }
    };
    tokio::pin!(shutdown);

    // ── Select loop ──────────────────────────────────────────────────────────
    //
    // Events from UDP and Zenoh are buffered via push_event(). tick() is
    // called ONLY when the 250ms timer fires — this ensures tick_count,
    // ticks_since_filter_broadcast, and PeerFilterTable eviction all
    // advance at wall-clock rate, not at event rate. Under high traffic,
    // events queue up and are drained on the next tick.
    loop {
        let mut should_tick = false;

        tokio::select! {
            // Arm 1: UDP packet received — buffer only, tick on timer.
            result = udp.recv_from(&mut udp_buf) => {
                match result {
                    Ok((len, _src)) => {
                        runtime.push_event(RuntimeEvent::InboundPacket {
                            interface_name: "udp0".to_string(),
                            raw: udp_buf[..len].to_vec(),
                            now: now_ms(),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(err = %e, "UDP recv error");
                        use std::io::ErrorKind::*;
                        match e.kind() {
                            WouldBlock | Interrupted => {}
                            _ => return Err(e.into()),
                        }
                    }
                }
            }

            // Arm 2: 250 ms timer tick — push TimerTick AND trigger tick().
            _ = timer.tick() => {
                runtime.push_event(RuntimeEvent::TimerTick { now: now_ms() });
                should_tick = true;
            }

            // Arm 3: Zenoh bridge event — buffer only, tick on timer.
            // Note: the None arm is currently unreachable because zenoh_tx lives
            // alongside this loop. It exists for defense-in-depth.
            maybe = zenoh_rx.recv() => {
                match maybe {
                    None => {
                        tracing::warn!("zenoh channel closed — exiting");
                        break;
                    }
                    Some(ev) => match ev {
                        ZenohEvent::Query { key_expr, payload } => {
                            let query_id = next_query_id;
                            next_query_id += 1;
                            runtime.push_event(RuntimeEvent::QueryReceived { query_id, key_expr, payload });
                        }
                        ZenohEvent::ComputeQuery { key_expr, payload } => {
                            let query_id = next_query_id;
                            next_query_id += 1;
                            runtime.push_event(RuntimeEvent::ComputeQuery { query_id, key_expr, payload });
                        }
                        ZenohEvent::Subscription { key_expr, payload } => {
                            runtime.push_event(RuntimeEvent::SubscriptionMessage { key_expr, payload });
                        }
                        ZenohEvent::FetchResponse { cid, is_module, result } => {
                            if is_module {
                                runtime.push_event(RuntimeEvent::ModuleFetchResponse { cid, result });
                            } else {
                                runtime.push_event(RuntimeEvent::ContentFetchResponse { cid, result });
                            }
                        }
                    }
                }
            }

            // Arm 4: Tunnel bridge event — buffer only, tick on timer.
            maybe = tunnel_rx.recv() => {
                let event = match maybe {
                    None => {
                        tracing::warn!("tunnel bridge channel closed");
                        break;
                    }
                    Some(e) => e,
                };
                match event {
                    TunnelBridgeEvent::HandshakeComplete {
                        interface_name,
                        peer_node_id,
                        peer_dsa_pubkey: _, // TODO(harmony-h6k): store in contact registry
                        connection_id,
                    } => {
                        // Guard against stale handshake completions: only register the
                        // interface if the connection_id still matches the current sender.
                        // Without this, a delayed handshake from a previous connection
                        // could double-register the router interface.
                        let is_current = tunnel_senders
                            .get(&interface_name)
                            .map(|s| s.connection_id == connection_id)
                            .unwrap_or(false);
                        if is_current {
                            runtime.push_event(RuntimeEvent::TunnelHandshakeComplete {
                                interface_name,
                                peer_node_id,
                            });
                        }
                    }
                    TunnelBridgeEvent::ReticulumReceived {
                        interface_name,
                        packet,
                        connection_id,
                    } => {
                        let is_current = tunnel_senders
                            .get(&interface_name)
                            .map(|s| s.connection_id == connection_id)
                            .unwrap_or(false);
                        if is_current {
                            runtime.push_event(RuntimeEvent::TunnelReticulumReceived {
                                interface_name,
                                packet,
                                now: now_ms(),
                            });
                        }
                    }
                    TunnelBridgeEvent::ZenohReceived { connection_id, .. } => {
                        // TODO(harmony-h6k): Zenoh over tunnel
                        let _ = connection_id; // will be guarded like ReticulumReceived
                    }
                    TunnelBridgeEvent::TunnelClosed {
                        interface_name,
                        reason,
                        connection_id,
                    } => {
                        tracing::info!(%interface_name, %reason, "tunnel closed");
                        // Only remove the sender AND unregister the router interface
                        // if the connection_id matches. A stale close from a previous
                        // connection must not tear down a reconnected peer's interface.
                        let is_current = tunnel_senders
                            .get(&interface_name)
                            .map(|s| s.connection_id == connection_id)
                            .unwrap_or(false);
                        if is_current {
                            tunnel_senders.remove(&interface_name);
                            runtime.push_event(RuntimeEvent::TunnelClosed { interface_name });
                        }
                        // else: stale close from old connection — ignore
                    }
                }
            }

            // Arm 5: Accept incoming iroh connections.
            //
            // The QUIC handshake (connecting.await) is spawned in a separate
            // task to avoid blocking the event loop for 100-500ms+. Completed
            // connections arrive on conn_rx (Arm 6).
            incoming = async {
                if let Some(ref ep) = iroh_endpoint {
                    ep.accept().await
                } else {
                    // No iroh endpoint — pend forever (arm disabled).
                    std::future::pending::<Option<iroh::endpoint::Incoming>>().await
                }
            } => {
                if incoming.is_none() {
                    // ep.accept() returned None — endpoint is closed.
                    // Disable this arm to prevent busy-spinning.
                    tracing::warn!("iroh endpoint closed — disabling tunnel accept");
                    iroh_endpoint = None;
                }
                if let Some(incoming) = incoming {
                    // Check connection limit BEFORE spawning the handshake task.
                    // Include inflight handshakes so a burst of simultaneous
                    // connects cannot all pass before any completes.
                    if tunnel_senders.len() + inflight_handshakes >= MAX_TUNNEL_CONNECTIONS {
                        tracing::warn!(limit = MAX_TUNNEL_CONNECTIONS, "tunnel connection limit reached — rejecting");
                        drop(incoming); // sends QUIC RESET
                    } else {
                        let conn_id = next_connection_id;
                        next_connection_id += 1;
                        let conn_tx = conn_tx.clone();
                        inflight_handshakes += 1;

                        match incoming.accept() {
                            Ok(connecting) => {
                                tokio::spawn(async move {
                                    match connecting.await {
                                        Ok(connection) => {
                                            let iface = match connection.remote_node_id() {
                                                Ok(id) => format!(
                                                    "tunnel-{}",
                                                    hex::encode(&id.as_bytes()[..8])
                                                ),
                                                Err(_) => format!(
                                                    "tunnel-{:08x}",
                                                    rand::random::<u32>()
                                                ),
                                            };
                                            let _ = conn_tx.send(Some(ReadyConnection {
                                                connection,
                                                connection_id: conn_id,
                                                interface_name: iface,
                                            })).await;
                                        }
                                        Err(e) => {
                                            tracing::warn!(err = %e, "QUIC handshake failed");
                                            // Signal failure so the event loop can decrement
                                            // inflight_handshakes even though no connection arrived.
                                            let _ = conn_tx.send(None).await;
                                        }
                                    }
                                });
                            }
                            Err(e) => {
                                tracing::warn!(err = %e, "iroh accept error");
                                // accept() failed synchronously — no task was spawned, so
                                // decrement the counter we just incremented.
                                inflight_handshakes = inflight_handshakes.saturating_sub(1);
                            }
                        }
                    }
                }
            }

            // Arm 6: Ready connection from QUIC handshake task — spawn tunnel.
            ready = conn_rx.recv() => {
                match ready {
                    Some(Some(ReadyConnection { connection, connection_id, interface_name })) => {
                        inflight_handshakes = inflight_handshakes.saturating_sub(1);
                        let (cmd_tx, cmd_rx) = mpsc::channel(64);
                        tunnel_senders.insert(
                            interface_name.clone(),
                            TunnelSender::new(cmd_tx, connection_id),
                        );

                        let tx = tunnel_tx.clone();
                        let identity =
                            tunnel_config.as_ref().expect("conn_rx received a connection but tunnel_config is None").local_identity.clone();
                        let iface_clone = interface_name.clone();
                        tokio::spawn(async move {
                            tunnel_task::run_responder(
                                connection, &identity, tx, cmd_rx, iface_clone, connection_id,
                            )
                            .await;
                        });
                        tracing::info!(%interface_name, "accepted incoming tunnel");
                    }
                    Some(None) => {
                        // Handshake failed — decrement the inflight counter so future
                        // connections are not incorrectly blocked by the ghost slot.
                        inflight_handshakes = inflight_handshakes.saturating_sub(1);
                    }
                    None => {
                        // Channel closed — should not happen while the loop is live.
                        tracing::warn!("ready-connection channel closed");
                        break;
                    }
                }
            }

            // Arm 7: Graceful shutdown (SIGTERM from procd, Ctrl+C).
            _ = &mut shutdown => {
                break;
            }
        }

        // Only tick on timer — counters (tick_count, ticks_since_filter_broadcast,
        // peer filter eviction) must advance at wall-clock rate, not event rate.
        // Events queue up between ticks (max ~250ms). At worst case 1 Gbps with
        // 500-byte Reticulum frames, ~62K events queue per tick (~31 MB) — bounded
        // by the timer interval and acceptable for 1-2 GB RAM routers.
        // A drain-without-counter-increment API would allow bounded queues under
        // flood conditions, but requires changing NodeRuntime's interface — deferred.
        if should_tick {
            let actions = runtime.tick();
            for action in actions {
                dispatch_action(action, &session, &zenoh_tx, &udp, &broadcast_addr, &tunnel_senders).await;
            }
        }
    }

    // ── Graceful iroh shutdown ────────────────────────────────────────────
    if let Some(ref ep) = iroh_endpoint {
        tracing::info!("closing iroh endpoint");
        ep.close().await;
    }

    Ok(())
}

/// Dispatch a single `RuntimeAction` to the appropriate I/O mechanism.
async fn dispatch_action(
    action: RuntimeAction,
    session: &zenoh::Session,
    zenoh_tx: &mpsc::Sender<ZenohEvent>,
    udp: &UdpSocket,
    broadcast_addr: &SocketAddr,
    tunnel_senders: &HashMap<String, TunnelSender>,
) {
    match action {
        // ── Tier 1: Send on interface (UDP broadcast or tunnel) ──────────────
        // Awaited inline (not spawned) because tokio::net::UdpSocket is not
        // Clone. For Reticulum's small MTU (≤1024 bytes), send_to is
        // effectively non-blocking on a UDP socket with default buffer sizes.
        // Tunnel interfaces route through TunnelSender (non-blocking try_send).
        RuntimeAction::SendOnInterface {
            ref interface_name,
            ref raw,
        } => {
            if interface_name.starts_with("tunnel-") {
                if let Some(sender) = tunnel_senders.get(interface_name.as_ref()) {
                    if sender.try_send_reticulum(raw.clone()).is_err() {
                        tracing::warn!(%interface_name, "tunnel send queue full — dropping packet");
                    }
                }
            } else if let Err(e) = udp.send_to(raw, broadcast_addr).await {
                tracing::warn!(err = %e, "UDP send error");
            }
        }

        // ── Tier 2: Zenoh publish (spawned to avoid blocking select loop) ────
        RuntimeAction::Publish { key_expr, payload } => {
            let session = session.clone();
            tokio::spawn(async move {
                if let Err(e) = session.put(&key_expr, payload).await {
                    tracing::warn!(%key_expr, err = %e, "zenoh put error");
                }
            });
        }

        // ── Tier 2: Reply to query — stub for v1 ─────────────────────────────
        // Query objects cannot be passed through channels in zenoh 1.x without
        // restructuring (the query handle must stay alive on the spawned task).
        // Deferred to a follow-up bead.
        RuntimeAction::SendReply { query_id, .. } => {
            tracing::debug!(query_id, "SendReply not yet implemented");
        }

        // ── Tier 3: Fetch content via Zenoh get() ─────────────────────────────
        RuntimeAction::FetchContent { cid } => {
            let cid_hex = hex::encode(cid);
            let key_expr = harmony_zenoh::namespace::content::fetch_key(&cid_hex);
            let tx = zenoh_tx.clone();
            let session = session.clone();
            tokio::spawn(async move {
                let result = fetch_via_zenoh(&session, &key_expr).await;
                let _ = tx
                    .send(ZenohEvent::FetchResponse { cid, is_module: false, result })
                    .await;
            });
        }

        // ── Tier 3: Fetch WASM module via Zenoh get() ────────────────────────
        // Modules are CID-addressed blobs stored in the content namespace —
        // same key scheme as FetchContent. Only the response routing differs
        // (ModuleFetchResponse vs ContentFetchResponse).
        RuntimeAction::FetchModule { cid } => {
            let cid_hex = hex::encode(cid);
            let key_expr = harmony_zenoh::namespace::content::fetch_key(&cid_hex);
            let tx = zenoh_tx.clone();
            let session = session.clone();
            tokio::spawn(async move {
                let result = fetch_via_zenoh(&session, &key_expr).await;
                let _ = tx
                    .send(ZenohEvent::FetchResponse { cid, is_module: true, result })
                    .await;
            });
        }

        // ── Setup: Declare queryable ──────────────────────────────────────────
        // Awaited inline — these only fire during startup (before the loop).
        // Dynamic runtime declarations are not yet wired; when they are,
        // spawning these to avoid blocking the select loop is warranted.
        RuntimeAction::DeclareQueryable { key_expr } => {
            let is_compute = key_expr.starts_with("harmony/compute/");
            let tx = zenoh_tx.clone();
            match session.declare_queryable(&key_expr).await {
                Ok(qbl) => {
                    tokio::spawn(async move {
                        while let Ok(query) = qbl.recv_async().await {
                            let qkey = query.key_expr().to_string();
                            let payload = query
                                .payload()
                                .map(|p| p.to_bytes().to_vec())
                                .unwrap_or_default();
                            let ev = if is_compute {
                                ZenohEvent::ComputeQuery { key_expr: qkey, payload }
                            } else {
                                ZenohEvent::Query { key_expr: qkey, payload }
                            };
                            if tx.send(ev).await.is_err() {
                                break;
                            }
                        }
                    });
                }
                Err(e) => {
                    tracing::error!(%key_expr, err = %e, "declare_queryable failed");
                }
            }
        }

        // ── Setup: Declare subscriber ─────────────────────────────────────────
        RuntimeAction::Subscribe { key_expr } => {
            let tx = zenoh_tx.clone();
            match session.declare_subscriber(&key_expr).await {
                Ok(sub) => {
                    tokio::spawn(async move {
                        while let Ok(sample) = sub.recv_async().await {
                            let skey = sample.key_expr().to_string();
                            let payload = sample.payload().to_bytes().to_vec();
                            if tx
                                .send(ZenohEvent::Subscription { key_expr: skey, payload })
                                .await
                                .is_err()
                            {
                                break;
                            }
                        }
                    });
                }
                Err(e) => {
                    tracing::error!(%key_expr, err = %e, "declare_subscriber failed");
                }
            }
        }
    }
}

/// Issue a Zenoh `get()` for the given key expression and return the first reply's payload.
///
/// The entire fetch (including all reply iterations) is bounded by a 30-second
/// wall-clock deadline. This prevents spawned tasks from accumulating
/// indefinitely — even if a peer sends repeated error replies.
async fn fetch_via_zenoh(
    session: &zenoh::Session,
    key_expr: &str,
) -> Result<Vec<u8>, String> {
    let replies = session
        .get(key_expr)
        .await
        .map_err(|e| format!("zenoh get error: {e}"))?;

    let deadline = Duration::from_secs(30);
    tokio::time::timeout(deadline, async {
        while let Ok(reply) = replies.recv_async().await {
            match reply.result() {
                Ok(sample) => {
                    return Ok(sample.payload().to_bytes().to_vec());
                }
                Err(err) => {
                    let msg = String::from_utf8_lossy(&err.payload().to_bytes()).into_owned();
                    tracing::warn!(%key_expr, err = %msg, "zenoh get reply error");
                }
            }
        }
        Err(format!("no successful reply for '{key_expr}'"))
    })
    .await
    .unwrap_or_else(|_| Err(format!("no successful reply for '{key_expr}' (timed out after 30s)")))
}

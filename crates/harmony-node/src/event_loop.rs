//! Async event loop — drives `NodeRuntime` via tokio select! with UDP, timer, and Zenoh.
//!
//! `NodeRuntime` is `!Send`; it lives entirely on the select loop task.
//! Zenoh objects that need spawning are cloned (Session is internally Arc'd).

use std::net::SocketAddr;
use std::time::{Duration, Instant};

use harmony_content::book::MemoryBookStore;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;

use crate::discovery::{self, PeerTable};
use crate::runtime::{NodeRuntime, RuntimeAction, RuntimeEvent};

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
pub async fn run(
    mut runtime: NodeRuntime<MemoryBookStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
    mdns_addr: Option<[u8; 16]>,
    mdns_stale_timeout: Duration,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── UDP socket ────────────────────────────────────────────────────────────
    let udp = UdpSocket::bind(listen_addr).await?;
    udp.set_broadcast(true)?;
    let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", listen_addr.port())
        .parse()
        .expect("static broadcast addr");

    // ── mDNS peer discovery (optional) ──────────────────────────────────────
    let mut peer_table = PeerTable::new(
        mdns_addr.unwrap_or([0; 16]),
        mdns_stale_timeout,
    );
    let mut mdns_state = match mdns_addr {
        Some(addr) => match discovery::start_mdns(listen_addr.port(), &addr) {
            Ok((daemon, rx)) => {
                tracing::info!("mDNS discovery started on _harmony._udp.local.");
                Some((daemon, rx))
            }
            Err(e) => {
                tracing::warn!(err = %e, "mDNS discovery failed to start — broadcast only");
                None
            }
        },
        None => {
            tracing::info!("mDNS discovery disabled (--no-mdns)");
            None
        }
    };

    // ── Zenoh session ─────────────────────────────────────────────────────────
    let session = zenoh::open(zenoh::Config::default()).await?;

    // ── mpsc channel: Zenoh tasks → select loop ───────────────────────────────
    let (zenoh_tx, mut zenoh_rx) = mpsc::channel::<ZenohEvent>(256);

    // ── Execute startup actions (declare queryables + subscribers) ────────────
    for action in startup_actions {
        dispatch_action(
            action,
            &session,
            &zenoh_tx,
            &udp,
            &broadcast_addr,
            None,
        )
        .await;
    }

    // ── Monotonic epoch ─────────────────────────────────────────────────────
    let epoch = Instant::now();
    let now_ms = || epoch.elapsed().as_millis() as u64;

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
                    Ok((len, src)) => {
                        // Invariant: src port matches the peer's mDNS-announced listen port
                        // because each node sends from its bound --listen-address socket.
                        peer_table.mark_seen(&src);
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
                for addr in peer_table.evict_stale() {
                    tracing::info!(peer = %addr, "evicted stale mDNS peer");
                }
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

            // Arm 4: mDNS discovery events (when enabled).
            result = async {
                match &mdns_state {
                    Some((_, rx)) => Some(rx.recv_async().await),
                    None => std::future::pending::<Option<Result<mdns_sd::ServiceEvent, _>>>().await,
                }
            } => {
                match result {
                    Some(Ok(event)) => match event {
                        mdns_sd::ServiceEvent::ServiceResolved(info) => {
                            let properties = info.get_properties();
                            if let Some(reticulum_addr) = discovery::parse_txt_addr(properties) {
                                let proto = discovery::parse_txt_proto(properties);
                                for ip in info.get_addresses() {
                                    let ip_addr = ip.to_ip_addr();
                                    // Only track addresses matching our socket family
                                    let matches = match listen_addr {
                                        SocketAddr::V4(_) => ip_addr.is_ipv4(),
                                        SocketAddr::V6(_) => ip_addr.is_ipv6(),
                                    };
                                    if !matches {
                                        continue;
                                    }
                                    let socket_addr = SocketAddr::new(ip_addr, info.get_port());
                                    if peer_table.add_peer(socket_addr, reticulum_addr, proto) {
                                        tracing::info!(
                                            peer = %socket_addr,
                                            addr = %hex::encode(reticulum_addr),
                                            proto,
                                            "mDNS peer discovered"
                                        );
                                    }
                                }
                            }
                        }
                        mdns_sd::ServiceEvent::ServiceRemoved(_service_type, fullname) => {
                            if let Some(reticulum_addr) = discovery::parse_instance_addr(&fullname) {
                                let removed = peer_table.remove_by_reticulum_addr(&reticulum_addr);
                                if removed > 0 {
                                    tracing::info!(
                                        addr = %hex::encode(reticulum_addr),
                                        removed,
                                        "mDNS peer removed (goodbye)"
                                    );
                                }
                            }
                        }
                        _ => {}
                    },
                    Some(Err(_)) => {
                        tracing::warn!("mDNS channel disconnected — discovery disabled");
                        mdns_state = None;
                    }
                    None => unreachable!("pending future cannot resolve to None"),
                }
            }

            // Arm 5: Graceful shutdown (SIGTERM from procd, Ctrl+C).
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
                dispatch_action(action, &session, &zenoh_tx, &udp, &broadcast_addr, Some(&peer_table)).await;
            }
        }
    }

    if let Some((daemon, _)) = mdns_state {
        if let Err(e) = daemon.shutdown() {
            tracing::warn!(err = %e, "mDNS shutdown error");
        }
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
    peer_table: Option<&PeerTable>,
) {
    match action {
        // ── Tier 1: UDP broadcast + unicast fan-out ───────────────────────────
        // Awaited inline (not spawned) because tokio::net::UdpSocket is not
        // Clone. For Reticulum's small MTU (≤1024 bytes), send_to is
        // effectively non-blocking on a UDP socket with default buffer sizes.
        RuntimeAction::SendOnInterface { raw, .. } => {
            if let Err(e) = udp.send_to(&raw, broadcast_addr).await {
                tracing::warn!(err = %e, "UDP broadcast send error");
            }
            if let Some(peers) = peer_table {
                for addr in peers.peer_addrs() {
                    if let Err(e) = udp.send_to(&raw, addr).await {
                        tracing::warn!(peer = %addr, err = %e, "UDP unicast send error");
                    }
                }
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

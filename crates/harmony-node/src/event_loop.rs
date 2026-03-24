//! Async event loop — drives `NodeRuntime` via tokio select! with UDP, timer, Zenoh, and iroh tunnels.
//!
//! `NodeRuntime` is `!Send`; it lives entirely on the select loop task.
//! Zenoh objects that need spawning are cloned (Session is internally Arc'd).

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use harmony_content::book::MemoryBookStore;
use harmony_identity::PqPrivateIdentity;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;

use crate::discovery::{self, PeerTable};
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

/// Tracks tunnel peers from the config file for persistent reconnection.
/// Fields are populated now but actual reconnection is deferred to Bead harmony-h6k.
#[allow(dead_code)]
struct ConfigTunnelPeers {
    peers: Vec<ConfigTunnelPeer>,
}

#[allow(dead_code)]
struct ConfigTunnelPeer {
    node_id: String,
    name: Option<String>,
    interface_name: String,
    next_retry: Option<tokio::time::Instant>,
    backoff: std::time::Duration,
    connected: bool,
}

#[allow(dead_code)]
const INITIAL_BACKOFF: std::time::Duration = std::time::Duration::from_secs(1);
#[allow(dead_code)]
const MAX_BACKOFF: std::time::Duration = std::time::Duration::from_secs(60);

#[allow(dead_code)]
impl ConfigTunnelPeers {
    fn new(entries: Vec<crate::config::TunnelEntry>) -> Self {
        Self {
            peers: entries
                .into_iter()
                .map(|entry| ConfigTunnelPeer {
                    interface_name: entry
                        .name
                        .clone()
                        .map(|n| {
                            // Clamp to IFNAMSIZ (15 bytes). Find the last valid
                            // char boundary at or before byte 15 to avoid panic
                            // on multi-byte UTF-8.
                            if n.len() <= 15 {
                                n
                            } else {
                                let mut end = 15;
                                while end > 0 && !n.is_char_boundary(end) {
                                    end -= 1;
                                }
                                n[..end].to_string()
                            }
                        })
                        .unwrap_or_else(|| {
                            // "tc-" + 8 hex chars = 11 chars, within IFNAMSIZ (15)
                            format!("tc-{}", &entry.node_id[..8.min(entry.node_id.len())])
                        }),
                    node_id: entry.node_id,
                    name: entry.name,
                    next_retry: Some(tokio::time::Instant::now()), // connect immediately
                    backoff: INITIAL_BACKOFF,
                    connected: false,
                })
                .collect(),
        }
    }

    fn mark_disconnected(&mut self, interface_name: &str) {
        if let Some(peer) = self
            .peers
            .iter_mut()
            .find(|p| p.interface_name == interface_name)
        {
            peer.connected = false;
            // Schedule retry at current backoff, THEN double for next time.
            let retry_delay = peer.backoff;
            peer.next_retry = Some(tokio::time::Instant::now() + retry_delay);
            peer.backoff = (peer.backoff * 2).min(MAX_BACKOFF);
            tracing::info!(
                %interface_name,
                retry_in_secs = retry_delay.as_secs(),
                "config tunnel disconnected — scheduling reconnect"
            );
        }
    }

    fn mark_connected(&mut self, interface_name: &str) {
        if let Some(peer) = self
            .peers
            .iter_mut()
            .find(|p| p.interface_name == interface_name)
        {
            peer.connected = true;
            peer.backoff = INITIAL_BACKOFF;
            peer.next_retry = None;
        }
    }
}

/// Internal bridge events from spawned Zenoh tasks to the select loop.
enum ZenohEvent {
    /// Inbound Zenoh query (non-compute).
    Query { key_expr: String, payload: Vec<u8> },
    /// Inbound Zenoh query on a compute key expression.
    ComputeQuery { key_expr: String, payload: Vec<u8> },
    /// Inbound Zenoh subscription sample.
    Subscription { key_expr: String, payload: Vec<u8> },
    /// Response to a FetchContent / FetchModule get() call.
    FetchResponse {
        cid: [u8; 32],
        is_module: bool,
        result: Result<Vec<u8>, String>,
    },
}

/// A tunnel dial request waiting for its scheduled fire time.
///
/// Introduces a random 500-4000ms delay between the PeerManager emitting
/// `InitiateTunnel` and the event loop actually dialing the relay, breaking
/// the timing correlation between Zenoh discovery queries and QUIC
/// connection attempts.
#[derive(Debug)]
struct DeferredDial {
    fire_at_ms: u64,
    identity_hash: [u8; 16],
    node_id: [u8; 32],
    relay_url: Option<String>,
    peer_dsa_pubkey: Vec<u8>,
    peer_kem_pubkey: Vec<u8>,
}

impl Ord for DeferredDial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.fire_at_ms.cmp(&other.fire_at_ms)
    }
}

impl PartialOrd for DeferredDial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for DeferredDial {
    fn eq(&self, other: &Self) -> bool {
        self.fire_at_ms == other.fire_at_ms
    }
}

impl Eq for DeferredDial {}

/// Run the async event loop.
///
/// # Parameters
/// - `runtime`: the sans-I/O node state machine (`!Send` — stays on this task).
/// - `startup_actions`: actions returned by `NodeRuntime::new`, executed before the loop.
/// - `listen_addr`: UDP socket address to bind (broadcast also sent here).
/// - `mdns_addr`: optional 16-byte Reticulum address for mDNS discovery (None = disabled).
/// - `mdns_stale_timeout`: duration after which silent mDNS peers are evicted.
/// - `tunnel_config`: optional iroh tunnel configuration (enables tunnel accept/connect).
/// - `bootstrap_peers`: static peers from the config `[peers]` section; added to PeerTable
///   at startup so they receive unicast traffic even when not reachable via mDNS.
/// - `tunnel_entries`: config-sourced tunnel peers to track for reconnection.
/// - `did_web_cache_ttl`: TTL in seconds for the did:web gateway cache.
pub async fn run(
    mut runtime: NodeRuntime<MemoryBookStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
    mdns_addr: Option<[u8; 16]>,
    mdns_stale_timeout: Duration,
    tunnel_config: Option<TunnelConfig>,
    bootstrap_peers: Vec<SocketAddr>,
    tunnel_entries: Vec<crate::config::TunnelEntry>,
    did_web_cache_ttl: u64,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // ── UDP socket ────────────────────────────────────────────────────────────
    let udp = UdpSocket::bind(listen_addr).await?;
    udp.set_broadcast(true)?;
    let broadcast_addr: SocketAddr = format!("255.255.255.255:{}", listen_addr.port())
        .parse()
        .expect("static broadcast addr");

    // ── mDNS peer discovery (optional) ──────────────────────────────────────
    let mut peer_table = PeerTable::new(mdns_addr.unwrap_or([0; 16]), mdns_stale_timeout);
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

    // ── Bootstrap peers from config file ────────────────────────────────────
    // Add each as a pinned peer — exempt from stale eviction so cross-subnet
    // peers that never send us traffic are retained indefinitely. Each gets a
    // unique placeholder Reticulum address derived from its socket address.
    for peer_addr in &bootstrap_peers {
        tracing::info!(peer = %peer_addr, "adding pinned bootstrap peer");
        peer_table.add_pinned_peer(*peer_addr);
    }

    // ── Zenoh session ─────────────────────────────────────────────────────────
    let session = zenoh::open(zenoh::Config::default()).await?;

    // ── mpsc channel: Zenoh tasks → select loop ───────────────────────────────
    let (zenoh_tx, mut zenoh_rx) = mpsc::channel::<ZenohEvent>(256);

    // ── Tunnel bridge: spawned tunnel tasks → select loop ───────────────────
    const MAX_TUNNEL_CONNECTIONS: usize = 64;
    let (tunnel_tx, mut tunnel_rx) = mpsc::channel::<TunnelBridgeEvent>(256);
    let mut tunnel_senders: HashMap<String, TunnelSender> = HashMap::new();
    // Maps interface_name → peer identity hash (first 16 bytes of peer_node_id,
    // which is BLAKE3(ML-DSA pubkey)). Used to route replication frames to the
    // correct peer in RuntimeEvent::ReplicaReceived. Populated at HandshakeComplete,
    // removed at TunnelClosed.
    // Maps interface_name → full 32-byte peer_node_id (BLAKE3 of ML-DSA pubkey).
    // Used to resolve contacts via find_by_tunnel_node_id() for replication routing.
    let mut tunnel_identities: HashMap<String, [u8; 32]> = HashMap::new();
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
        // Use a fresh random SecretKey so the relay cannot link the Endpoint's
        // NodeId to the node's permanent ML-DSA identity.
        let secret_key = iroh::SecretKey::generate(&mut rand::rngs::OsRng);

        let mut builder = iroh::Endpoint::builder()
            .alpns(vec![tunnel_task::HARMONY_TUNNEL_ALPN.to_vec()])
            .secret_key(secret_key);

        if let Some(ref url) = config.relay_url {
            let relay_url: iroh::RelayUrl =
                url.parse()
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("invalid relay URL '{url}': {e}").into()
                    })?;
            let relay_map = iroh::RelayMap::from_iter([relay_url]);
            builder = builder.relay_mode(iroh::RelayMode::Custom(relay_map));
        } else {
            builder = builder.relay_mode(iroh::RelayMode::Disabled);
        }

        let ep = builder
            .bind()
            .await
            .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("iroh endpoint bind failed: {e}").into()
            })?;
        tracing::info!(node_id = %ep.node_id(), "iroh tunnel endpoint ready");
        Some(ep)
    } else {
        None
    };

    // ── Push local tunnel info into the runtime ──────────────────────────────
    if let Some(ref ep) = iroh_endpoint {
        let node_id_bytes = *ep.node_id().as_bytes();
        let relay_url = tunnel_config.as_ref().and_then(|c| c.relay_url.clone());
        runtime.push_event(RuntimeEvent::LocalTunnelInfo {
            node_id: node_id_bytes,
            relay_url,
        });
    }

    // ConfigTunnelPeers tracks config-sourced tunnels and drives backoff reconnection.
    // Actual outbound connection initiation is deferred to Bead harmony-h6k.
    #[allow(unused_variables, unused_mut)]
    let mut config_tunnels = ConfigTunnelPeers::new(tunnel_entries);

    // ── Deferred dial queue (stochastic delay for tunnel privacy) ───────────
    // BinaryHeap<Reverse<...>> is a min-heap by fire_at_ms, ensuring earlier
    // fire times are always drained first regardless of insertion order.
    let mut deferred_dials: BinaryHeap<Reverse<DeferredDial>> = BinaryHeap::new();

    // ── Execute startup actions (declare queryables + subscribers) ────────────
    for action in startup_actions {
        dispatch_action(
            action,
            &session,
            &zenoh_tx,
            &udp,
            &broadcast_addr,
            None,
            &tunnel_senders,
            &mut deferred_dials,
        )
        .await;
    }

    // ── did:web gateway queryable ────────────────────────────────────────────
    match session
        .declare_queryable(harmony_zenoh::namespace::identity::web::ALL)
        .await
    {
        Ok(qbl) => {
            tokio::spawn(crate::did_web_gateway::run(qbl, did_web_cache_ttl));
        }
        Err(e) => {
            tracing::error!(err = %e, "failed to declare did:web gateway queryable");
        }
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
        signal(SignalKind::terminate()).map_err(
            |e| -> Box<dyn std::error::Error + Send + Sync> {
                format!("failed to register SIGTERM handler: {e}").into()
            },
        )?
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
                let unix_now = SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                runtime.push_event(RuntimeEvent::TimerTick { now: now_ms(), unix_now });
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
                        if let Some((daemon, _)) = mdns_state.take() {
                            if let Err(e) = daemon.shutdown() {
                                tracing::warn!(err = %e, "mDNS shutdown error");
                            }
                        }
                    }
                    None => unreachable!("pending future cannot resolve to None"),
                }
            }

            // Arm 5: Tunnel bridge event — buffer only, tick on timer.
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
                        peer_dsa_pubkey,
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
                            // Store full 32-byte peer_node_id for replication routing.
                            // Used with find_by_tunnel_node_id() to resolve the contact's
                            // canonical identity_hash for quota lookup.
                            tunnel_identities.insert(interface_name.clone(), peer_node_id);

                            // Cache the peer's ML-DSA public key for token verification.
                            // If the contact isn't registered yet, the pubkey is intentionally
                            // dropped — it will be cached via the discovery announce path
                            // (process_discovered_tunnel_hints) when the contact is created.
                            if let Some(contact) = runtime.contact_store().find_by_tunnel_node_id(&peer_node_id) {
                                runtime.push_event(RuntimeEvent::PeerPublicKeyLearned {
                                    identity_hash: contact.identity_hash,
                                    dsa_pubkey: peer_dsa_pubkey,
                                });
                            }

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
                    TunnelBridgeEvent::ReplicationReceived {
                        interface_name,
                        message,
                        connection_id,
                    } => {
                        let is_current = tunnel_senders
                            .get(&interface_name)
                            .map(|s| s.connection_id == connection_id)
                            .unwrap_or(false);
                        if is_current {
                            // Parse the replication message to extract CID and data.
                            if let Ok(rep_msg) =
                                harmony_tunnel::replication::ReplicationMessage::decode(&message)
                            {
                                use harmony_tunnel::replication::ReplicationOp;
                                // Resolve the contact's canonical identity_hash.
                                if let Some(node_id) = tunnel_identities.get(&interface_name) {
                                    if let Some(contact) = runtime.contact_store().find_by_tunnel_node_id(node_id) {
                                        let peer_id = contact.identity_hash;
                                        match rep_msg.op {
                                            ReplicationOp::Push => {
                                                runtime.push_event(RuntimeEvent::ReplicaPushReceived {
                                                    peer_identity: peer_id,
                                                    cid: rep_msg.cid,
                                                    data: rep_msg.payload,
                                                });
                                            }
                                            ReplicationOp::PullWithToken => {
                                                // Fail-closed: reject if system clock is broken
                                                // rather than passing unix_now=0 which bypasses
                                                // expiry checks in the runtime.
                                                let unix_now = match std::time::SystemTime::now()
                                                    .duration_since(std::time::UNIX_EPOCH)
                                                {
                                                    Ok(d) => d.as_secs(),
                                                    Err(_) => {
                                                        tracing::warn!("system clock error; rejecting PullWithToken");
                                                        continue;
                                                    }
                                                };
                                                runtime.push_event(RuntimeEvent::ReplicaPullWithTokenReceived {
                                                    peer_identity: peer_id,
                                                    cid: rep_msg.cid,
                                                    token_bytes: rep_msg.payload,
                                                    unix_now,
                                                });
                                            }
                                            _ => {
                                                tracing::trace!(
                                                    op = ?rep_msg.op,
                                                    cid = %hex::encode(&rep_msg.cid[..8]),
                                                    "unhandled replication op — ignored"
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
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
                            tunnel_identities.remove(&interface_name);
                            runtime.push_event(RuntimeEvent::TunnelClosed { interface_name });
                        }
                        // else: stale close from old connection — ignore
                    }
                }
            }

            // Arm 6: Accept incoming iroh connections.
            //
            // The QUIC handshake (connecting.await) is spawned in a separate
            // task to avoid blocking the event loop for 100-500ms+. Completed
            // connections arrive on conn_rx (Arm 7).
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

            // Arm 7: Ready connection from QUIC handshake task — spawn tunnel.
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

            // Arm 8: Graceful shutdown (SIGTERM from procd, Ctrl+C).
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
                dispatch_action(
                    action,
                    &session,
                    &zenoh_tx,
                    &udp,
                    &broadcast_addr,
                    Some(&peer_table),
                    &tunnel_senders,
                    &mut deferred_dials,
                )
                .await;
            }

            // Drain deferred dials that have reached their fire time.
            // Because the heap is ordered by fire_at_ms (min-heap via Reverse),
            // we stop as soon as the earliest entry hasn't fired yet.
            let current_ms = now_ms();
            while let Some(Reverse(front)) = deferred_dials.peek() {
                if front.fire_at_ms <= current_ms {
                    let Reverse(dial) = deferred_dials.pop().unwrap();
                    tracing::info!(
                        identity = %hex::encode(dial.identity_hash),
                        node_id = %hex::encode(&dial.node_id[..8]),
                        relay = ?dial.relay_url,
                        "InitiateTunnel fired (stub — iroh-net not yet wired)"
                    );
                    // TODO(harmony-h6k): execute actual iroh Endpoint.connect()
                    // using dial.node_id and dial.relay_url here.
                } else {
                    break;
                }
            }
        }
    }

    // ── Graceful mDNS shutdown ────────────────────────────────────────────
    if let Some((daemon, _)) = mdns_state {
        if let Err(e) = daemon.shutdown() {
            tracing::warn!(err = %e, "mDNS shutdown error");
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
#[allow(clippy::too_many_arguments)]
async fn dispatch_action(
    action: RuntimeAction,
    session: &zenoh::Session,
    zenoh_tx: &mpsc::Sender<ZenohEvent>,
    udp: &UdpSocket,
    broadcast_addr: &SocketAddr,
    peer_table: Option<&PeerTable>,
    tunnel_senders: &HashMap<String, TunnelSender>,
    deferred_dials: &mut BinaryHeap<Reverse<DeferredDial>>,
) {
    match action {
        // ── Tier 1: Send on interface (UDP broadcast + unicast, or tunnel) ────
        // Awaited inline (not spawned) because tokio::net::UdpSocket is not
        // Clone. For Reticulum's small MTU (≤1024 bytes), send_to is
        // effectively non-blocking on a UDP socket with default buffer sizes.
        // Tunnel interfaces route through TunnelSender (non-blocking try_send).
        RuntimeAction::SendOnInterface {
            ref interface_name,
            ref raw,
            weight,
        } => {
            let should_send = match weight {
                None => true,                         // directed: always send
                Some(w) if w >= 1.0 => true,          // best interface: always send
                Some(w) => rand::random::<f32>() < w, // probabilistic
            };
            if should_send {
                if interface_name.starts_with("tunnel-") {
                    if let Some(sender) = tunnel_senders.get(interface_name.as_ref()) {
                        if sender.try_send_reticulum(raw.clone()).is_err() {
                            tracing::warn!(%interface_name, "tunnel send queue full — dropping packet");
                        }
                    }
                } else {
                    if let Err(e) = udp.send_to(raw, broadcast_addr).await {
                        tracing::warn!(err = %e, "UDP broadcast send error");
                    }
                    if let Some(peers) = peer_table {
                        for addr in peers.peer_addrs() {
                            if let Err(e) = udp.send_to(raw, addr).await {
                                tracing::warn!(peer = %addr, err = %e, "UDP unicast send error");
                            }
                        }
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
                    .send(ZenohEvent::FetchResponse {
                        cid,
                        is_module: false,
                        result,
                    })
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
                    .send(ZenohEvent::FetchResponse {
                        cid,
                        is_module: true,
                        result,
                    })
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
                                ZenohEvent::ComputeQuery {
                                    key_expr: qkey,
                                    payload,
                                }
                            } else {
                                ZenohEvent::Query {
                                    key_expr: qkey,
                                    payload,
                                }
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
                                .send(ZenohEvent::Subscription {
                                    key_expr: skey,
                                    payload,
                                })
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

        // ── Peer lifecycle: Tunnel initiation ─────────────────────────────────
        // Deferred by a random 500-4000ms delay for privacy — breaks timing
        // correlation between Zenoh discovery queries and QUIC connection attempts.
        // The actual dial (stub) fires when the deferred queue drains.
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
            tracing::debug!(
                identity = %hex::encode(identity_hash),
                delay_ms,
                "tunnel dial deferred for privacy"
            );
        }

        // ── Peer lifecycle: Path request ──────────────────────────────────────
        // Stub — announce probing will be wired when path request packets are
        // implemented in the Reticulum router.
        RuntimeAction::SendPathRequest { identity_hash } => {
            tracing::debug!(
                identity = %hex::encode(identity_hash),
                "SendPathRequest (stub)"
            );
        }
        RuntimeAction::CloseTunnel { identity_hash } => {
            // TODO: look up tunnel_sender by identity_hash (via tunnel_identities map)
            // and send TunnelCommand::Close. Requires the tunnel_identities map to be
            // accessible here, which will be wired when the full tunnel lifecycle is
            // integrated with the event loop's tunnel infrastructure.
            tracing::debug!(
                identity = %hex::encode(identity_hash),
                "CloseTunnel (stub)"
            );
        }
        RuntimeAction::ReplicaPush {
            peer_identity,
            cid,
            data,
        } => {
            // TODO: look up tunnel_sender by peer_identity (via tunnel_identities map)
            // and send a ReplicationMessage::push(cid, data) as a FrameTag::Replication
            // frame. Requires identity_hash → TunnelSender mapping.
            tracing::debug!(
                peer = %hex::encode(peer_identity),
                cid = %hex::encode(cid),
                size = data.len(),
                "ReplicaPush (stub — tunnel routing not yet wired)"
            );
        }
        RuntimeAction::ReplicaPullResponse {
            peer_identity,
            cid,
            data,
        } => {
            // TODO: route to tunnel sender (same pattern as ReplicaPush)
            tracing::debug!(
                identity = %hex::encode(peer_identity),
                cid = %hex::encode(&cid[..8]),
                size = data.len(),
                "serving replicated content via token (stub — tunnel routing not yet wired)"
            );
        }
        RuntimeAction::QueryMemo { key_expr } => {
            // TODO: implement in Task 3 — issue a Zenoh session.get() for the memo key
            // and feed each reply back as RuntimeEvent::MemoFetchResponse.
            tracing::debug!(%key_expr, "QueryMemo (stub — not yet wired)");
        }
    }
}

/// Issue a Zenoh `get()` for the given key expression and return the first reply's payload.
///
/// The entire fetch (including all reply iterations) is bounded by a 30-second
/// wall-clock deadline. This prevents spawned tasks from accumulating
/// indefinitely — even if a peer sends repeated error replies.
async fn fetch_via_zenoh(session: &zenoh::Session, key_expr: &str) -> Result<Vec<u8>, String> {
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
    .unwrap_or_else(|_| {
        Err(format!(
            "no successful reply for '{key_expr}' (timed out after 30s)"
        ))
    })
}

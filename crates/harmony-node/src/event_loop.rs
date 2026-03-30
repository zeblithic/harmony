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
use harmony_content::cid::ContentId;
use harmony_identity::PqPrivateIdentity;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;

use crate::discovery::{self, PeerTable};
use crate::runtime::{NodeRuntime, RuntimeAction, RuntimeEvent};
use crate::tunnel_bridge::{ReadyConnection, TunnelBridgeEvent, TunnelSender};
use crate::tunnel_task;

/// Result of an async disk I/O operation, sent back to the event loop via channel.
enum DiskIoResult {
    /// Book data successfully read from disk.
    ReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
    /// Book not found or read error.
    ReadFailed { cid: ContentId, query_id: u64 },
}

/// Result of an async S3 fetch, sent back to the event loop via channel.
#[allow(dead_code)] // Variants constructed only when `archivist` feature is enabled.
enum S3IoResult {
    /// Book data successfully fetched from S3.
    ReadComplete {
        cid: ContentId,
        query_id: u64,
        data: Vec<u8>,
    },
    /// Book not found in S3 or fetch error.
    ReadFailed { cid: ContentId, query_id: u64 },
}

/// Results from a streaming inference task.
#[cfg(feature = "inference")]
enum InferenceResult {
    /// A streaming token to publish to Zenoh.
    Chunk {
        task_id: String,
        sequence: u32,
        token_text: String,
        token_id: Option<u32>,
        final_chunk: bool,
    },
    /// Inference completed — send query reply and return engine.
    Complete {
        query_id: u64,
        task_id: String,
        output: crate::inference::InferenceOutput,
        engine: harmony_inference::QwenEngine,
    },
    /// Inference failed — send error reply and return engine.
    Failed {
        query_id: u64,
        task_id: String,
        error: String,
        engine: harmony_inference::QwenEngine,
    },
    /// Inference task panicked — engine is lost, need to recreate.
    Panicked { query_id: u64, task_id: String },
}

/// Type alias for the S3 read library reference.
///
/// When the `archivist` feature is enabled, this holds an `Arc<S3Library>` for
/// cloning into spawned async tasks. Without the feature, it is a no-op `()`
/// placeholder so `dispatch_action`'s signature stays uniform.
#[cfg(feature = "archivist")]
type S3ReadLibrary = Option<Arc<harmony_s3::S3Library>>;
#[cfg(not(feature = "archivist"))]
type S3ReadLibrary = Option<()>;

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
    /// DSD: verify response from target node.
    VerifyResponse { payload: Vec<u8> },
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
/// - `rawlink_interface`: optional network interface name for the AF_PACKET L2 bridge
///   (Linux only, requires the `rawlink` feature and `CAP_NET_RAW`).
/// - `archivist_config`: optional S3 archivist configuration (requires the `archivist` feature).
/// - `data_dir`: optional directory for persistent CAS book storage. When set, durable books
///   are written to disk via `spawn_blocking` and reloaded on restart.
#[allow(clippy::too_many_arguments, unused_variables)]
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
    rawlink_interface: Option<String>,
    archivist_config: Option<crate::config::ArchivistConfig>,
    data_dir: Option<std::path::PathBuf>,
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

    // ── S3 library (single init, shared between archivist + read fallback) ───
    #[cfg(feature = "archivist")]
    let s3_read_library: S3ReadLibrary = if let Some(ref archivist) = archivist_config {
        match harmony_s3::S3Library::new(
            archivist.bucket.clone(),
            archivist.prefix.clone(),
            archivist.region.clone(),
        )
        .await
        {
            Ok(s3) => {
                // Clone for the archivist write path (consumes its copy).
                let archivist_s3 = s3.clone();
                let archivist_session = session.clone();
                let _archivist_handle = tokio::spawn(async move {
                    harmony_s3::archivist::run(archivist_s3, archivist_session).await;
                    tracing::error!("S3 archivist task exited — archival is no longer active");
                });
                tracing::info!(
                    bucket = %archivist.bucket,
                    prefix = %archivist.prefix,
                    "S3 archivist + read fallback enabled"
                );
                // Wrap in Arc for the read fallback path.
                Some(Arc::new(s3))
            }
            Err(e) => {
                tracing::warn!(err = %e, "S3 failed to init — archivist and read fallback disabled");
                // Disable s3 in the runtime so StorageTier stops emitting S3Lookup.
                runtime.disable_s3();
                None
            }
        }
    } else {
        None
    };

    #[cfg(not(feature = "archivist"))]
    let s3_read_library: S3ReadLibrary = None;

    #[cfg(not(feature = "archivist"))]
    if archivist_config.is_some() {
        tracing::warn!(
            "archivist config is present but this binary was compiled without \
             the `archivist` feature — archival is disabled"
        );
        runtime.disable_s3();
    }

    // ── L2 Reticulum channel endpoints (populated when rawlink bridge starts) ─
    // On non-Linux / non-rawlink builds these stay None; allow_unused_mut avoids
    // warnings while keeping the cfg-gated assignment path simple.
    #[allow(unused_mut)]
    let mut ret_inbound_rx: Option<tokio::sync::mpsc::Receiver<Vec<u8>>> = None;
    #[allow(unused_mut)]
    let mut ret_outbound_tx: Option<tokio::sync::mpsc::Sender<Vec<u8>>> = None;
    #[allow(unused_mut)]
    let mut rawlink_iface_name: Option<String> = None;

    // ── AF_PACKET rawlink bridge (Linux + rawlink feature only) ──────────────
    #[cfg(all(target_os = "linux", feature = "rawlink"))]
    if let Some(ref iface) = rawlink_interface {
        match harmony_rawlink::af_packet::AfPacketSocket::new(iface) {
            Ok(socket) => {
                let (ri_tx, ri_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);
                let (ro_tx, ro_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(256);

                let bridge_config = harmony_rawlink::BridgeConfig {
                    identity_hash: runtime.local_pq_identity_hash(),
                    reticulum_inbound_tx: Some(ri_tx),
                    ..harmony_rawlink::BridgeConfig::default()
                };
                let mut bridge = harmony_rawlink::Bridge::new(
                    socket,
                    session.clone(),
                    bridge_config,
                    Some(ro_rx),
                );

                // Register L2 interface with the Reticulum router
                let iface_name = format!("l2:{}", iface);
                runtime.push_event(RuntimeEvent::L2InterfaceReady {
                    interface_name: iface_name.clone(),
                });

                ret_inbound_rx = Some(ri_rx);
                ret_outbound_tx = Some(ro_tx);
                rawlink_iface_name = Some(iface_name);

                tokio::spawn(async move {
                    if let Err(e) = bridge.run().await {
                        tracing::warn!(err = %e, "rawlink bridge stopped");
                    }
                });
                tracing::info!(%iface, "rawlink AF_PACKET bridge started");
            }
            Err(e) => {
                tracing::warn!(
                    interface = %iface,
                    err = %e,
                    "rawlink bridge failed to start — continuing without L2 transport"
                );
            }
        }
    }

    // ── mpsc channel: Zenoh tasks → select loop ───────────────────────────────
    let (zenoh_tx, mut zenoh_rx) = mpsc::channel::<ZenohEvent>(256);

    // ── Tunnel bridge: spawned tunnel tasks → select loop ───────────────────
    const MAX_TUNNEL_CONNECTIONS: usize = 64;
    let (tunnel_tx, mut tunnel_rx) = mpsc::channel::<TunnelBridgeEvent>(256);
    let mut tunnel_senders: HashMap<String, TunnelSender> = HashMap::new();
    // Keeps initiator-side iroh Endpoints alive for the lifetime of the connection.
    // When the initiator dials, it creates a transient Endpoint that owns the QUIC
    // state; dropping it would tear down the connection. Removed on TunnelClosed.
    let mut initiator_endpoints: HashMap<String, iroh::Endpoint> = HashMap::new();
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
    // Pre-compute the relay map so it can be cloned for initiator dials.
    let relay_map: Option<iroh::RelayMap> = if let Some(ref config) = tunnel_config {
        if let Some(ref url) = config.relay_url {
            let relay_url: iroh::RelayUrl =
                url.parse()
                    .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
                        format!("invalid relay URL '{url}': {e}").into()
                    })?;
            Some(iroh::RelayMap::from_iter([relay_url]))
        } else {
            None
        }
    } else {
        None
    };

    let mut iroh_endpoint = if tunnel_config.is_some() {
        // Use a fresh random SecretKey so the relay cannot link the Endpoint's
        // NodeId to the node's permanent ML-DSA identity.
        let secret_key = iroh::SecretKey::generate(&mut rand::rngs::OsRng);

        let mut builder = iroh::Endpoint::builder()
            .alpns(vec![tunnel_task::HARMONY_TUNNEL_ALPN.to_vec()])
            .secret_key(secret_key);

        if let Some(ref rm) = relay_map {
            builder = builder.relay_mode(iroh::RelayMode::Custom(rm.clone()));
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

    // ── Disk I/O completion channel ─────────────────────────────────────────
    // spawn_blocking tasks send results back here; the select loop feeds them
    // into the runtime as DiskReadComplete / DiskReadFailed events.
    // Write completions are fire-and-forget (errors logged in the spawned task).
    let (disk_tx, mut disk_rx) = mpsc::channel::<DiskIoResult>(64);

    // ── S3 I/O completion channel ────────────────────────────────────────────
    // Async S3 fetch tasks send results back here; the select loop feeds them
    // into the runtime as S3ReadComplete / S3ReadFailed events.
    let (s3_tx, mut s3_rx) = mpsc::channel::<S3IoResult>(64);

    // ── Inference streaming completion channel ─────────────────────────────────
    // spawn_blocking inference tasks stream tokens and final results back here.
    #[cfg(feature = "inference")]
    let (inference_tx, mut inference_rx) = mpsc::channel::<InferenceResult>(64);

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
            &ret_outbound_tx,
            &data_dir,
            &disk_tx,
            &s3_read_library,
            &s3_tx,
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

    // ── Load persisted memos into NodeRuntime's store ───────────────────
    #[allow(unused_variables)]
    let mut memo_disk_bytes: u64 = 0;

    if let Some(ref dir) = data_dir {
        let dir_clone = dir.clone();
        let memo_entries = tokio::task::spawn_blocking(move || {
            crate::memo_io::scan_memos(&dir_clone)
        })
        .await
        .unwrap_or_else(|e| {
            tracing::warn!("memo scan task panicked: {}", e);
            Vec::new()
        });

        let store = runtime.memo_store_mut();
        for (memo, size) in memo_entries {
            if store.insert(memo) {
                memo_disk_bytes += size;
            }
        }
        tracing::info!("Loaded {} memos from disk ({} bytes)", store.len(), memo_disk_bytes);

        // Load LFU counts if available.
        let lfu_path = dir.join("memo_lfu.bin");
        if lfu_path.exists() {
            match std::fs::read(&lfu_path) {
                Ok(bytes) => match store.load_lfu_counts(&bytes) {
                    Ok(n) => tracing::info!("Loaded {} LFU counters from disk", n),
                    Err(e) => tracing::warn!("Failed to parse memo_lfu.bin: {} — starting fresh", e),
                },
                Err(e) => tracing::warn!("Failed to read memo_lfu.bin: {} — starting fresh", e),
            }
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

    // ── Peer keepalive counter ──────────────────────────────────────────────
    // Refresh mDNS peer timestamps every ~20s (80 ticks × 250ms) to prevent
    // false stale eviction. The mDNS daemon handles actual peer removal via
    // ServiceRemoved when a service's DNS TTL expires without refresh.
    let mut ticks_since_peer_refresh: u64 = 0;
    const PEER_REFRESH_INTERVAL_TICKS: u64 = 80; // ~20 seconds

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

                // Periodically refresh mDNS peer timestamps to prevent false
                // stale eviction. Actual peer removal is handled by mDNS
                // ServiceRemoved events (DNS TTL expiry / goodbye packets).
                ticks_since_peer_refresh += 1;
                if mdns_state.is_some() && ticks_since_peer_refresh >= PEER_REFRESH_INTERVAL_TICKS {
                    ticks_since_peer_refresh = 0;
                    peer_table.refresh_mdns_peers();
                }

                for addr in peer_table.evict_stale() {
                    tracing::info!(peer = %addr, "evicted stale mDNS peer");
                }

                // ── Flush memo LFU counters periodically (~5 min) ──────────
                {
                    static MEMO_FLUSH_COUNTER: std::sync::atomic::AtomicU32 =
                        std::sync::atomic::AtomicU32::new(0);
                    let tick = MEMO_FLUSH_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // Timer is 250ms; 1200 ticks × 250ms = 5 minutes.
                    if tick % 1200 == 0 && tick > 0 {
                        if let Some(ref dir) = data_dir {
                            match runtime.memo_store_mut().serialize_lfu_counts() {
                                Ok(bytes) => {
                                    let path = dir.join("memo_lfu.bin");
                                    tokio::task::spawn_blocking(move || {
                                        let tmp = path.with_extension("tmp");
                                        let result = (|| -> std::io::Result<()> {
                                            let mut f = std::fs::File::create(&tmp)?;
                                            std::io::Write::write_all(&mut f, &bytes)?;
                                            f.sync_all()?;
                                            std::fs::rename(&tmp, &path)?;
                                            Ok(())
                                        })();
                                        if let Err(e) = result {
                                            tracing::warn!("Failed to flush memo_lfu.bin: {}", e);
                                        }
                                    });
                                }
                                Err(e) => tracing::warn!("Failed to serialize LFU counts: {}", e),
                            }
                        }
                    }
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
                        ZenohEvent::VerifyResponse { payload } => {
                            runtime.push_event(RuntimeEvent::VerifyResponse { payload });
                        }
                    }
                }
            }

            // Arm 4: Disk I/O completion — feed results back into runtime.
            Some(disk_result) = disk_rx.recv() => {
                match disk_result {
                    DiskIoResult::ReadComplete { cid, query_id, data } => {
                        runtime.push_event(RuntimeEvent::DiskReadComplete { cid, query_id, data });
                    }
                    DiskIoResult::ReadFailed { cid, query_id } => {
                        runtime.push_event(RuntimeEvent::DiskReadFailed { cid, query_id });
                    }
                }
            }

            // Arm 4b: S3 I/O completion — feed results back into runtime.
            Some(s3_result) = s3_rx.recv() => {
                match s3_result {
                    S3IoResult::ReadComplete { cid, query_id, data } => {
                        runtime.push_event(RuntimeEvent::S3ReadComplete { cid, query_id, data });
                    }
                    S3IoResult::ReadFailed { cid, query_id } => {
                        runtime.push_event(RuntimeEvent::S3ReadFailed { cid, query_id });
                    }
                }
            }

            // Arm 4c: Inference streaming completion — publish chunks and handle results.
            // Uses the async-block-with-pending pattern (like rawlink Arm 9) so the
            // arm compiles even when the `inference` feature is disabled.
            Some(_inference_result) = async {
                #[cfg(feature = "inference")]
                { inference_rx.recv().await }
                #[cfg(not(feature = "inference"))]
                { std::future::pending::<Option<()>>().await }
            } => {
                // The type of `inference_result` differs by feature, but the body
                // is unreachable when inference is disabled (pending never resolves).
                #[cfg(feature = "inference")]
                handle_inference_result(
                    inference_result,
                    &mut runtime,
                    &session,
                    &zenoh_tx,
                    &udp,
                    &broadcast_addr,
                    &peer_table,
                    &tunnel_senders,
                    &mut deferred_dials,
                    &ret_outbound_tx,
                    &data_dir,
                    &disk_tx,
                    &s3_read_library,
                    &s3_tx,
                )
                .await;
            }

            // Arm 5: mDNS discovery events (when enabled).
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

            // Arm 6: Tunnel bridge event — buffer only, tick on timer.
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
                            initiator_endpoints.remove(&interface_name);
                            runtime.push_event(RuntimeEvent::TunnelClosed { interface_name });
                        }
                        // else: stale close from old connection — ignore
                    }
                }
            }

            // Arm 7: Accept incoming iroh connections.
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
                                                remote_pq_identity: None,
                                                initiator_endpoint: None,
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

            // Arm 8: Ready connection from QUIC handshake task — spawn tunnel.
            ready = conn_rx.recv() => {
                match ready {
                    Some(Some(ready_conn)) => {
                        inflight_handshakes = inflight_handshakes.saturating_sub(1);
                        let ReadyConnection {
                            connection,
                            connection_id,
                            interface_name,
                            remote_pq_identity,
                            initiator_endpoint,
                        } = ready_conn;

                        // Store initiator endpoint if present — keeps the QUIC state alive.
                        if let Some(ep) = initiator_endpoint {
                            initiator_endpoints.insert(interface_name.clone(), ep);
                        }

                        let (cmd_tx, cmd_rx) = mpsc::channel(64);
                        tunnel_senders.insert(
                            interface_name.clone(),
                            TunnelSender::new(cmd_tx, connection_id),
                        );

                        let identity =
                            tunnel_config.as_ref().expect("conn_rx received a connection but tunnel_config is None").local_identity.clone();

                        if let Some(remote_id) = remote_pq_identity {
                            // Initiator path — we know the peer's identity
                            let tx = tunnel_tx.clone();
                            let iface_clone = interface_name.clone();
                            tokio::spawn(async move {
                                tunnel_task::run_initiator(
                                    connection, &identity, &remote_id, tx, cmd_rx, iface_clone, connection_id,
                                )
                                .await;
                            });
                            tracing::info!(%interface_name, "initiated outbound tunnel");
                        } else {
                            // Responder path — peer identity learned during handshake
                            let tx = tunnel_tx.clone();
                            let iface_clone = interface_name.clone();
                            tokio::spawn(async move {
                                tunnel_task::run_responder(
                                    connection, &identity, tx, cmd_rx, iface_clone, connection_id,
                                )
                                .await;
                            });
                            tracing::info!(%interface_name, "accepted incoming tunnel");
                        }
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

            // Arm 9: Inbound Reticulum packet from L2 rawlink bridge.
            result = async {
                match ret_inbound_rx.as_mut() {
                    Some(rx) => rx.recv().await,
                    None => std::future::pending().await,
                }
            } => {
                match result {
                    Some(packet) => {
                        if let Some(ref iface_name) = rawlink_iface_name {
                            runtime.push_event(RuntimeEvent::InboundPacket {
                                interface_name: iface_name.clone(),
                                raw: packet,
                                now: now_ms(),
                            });
                        }
                    }
                    None => {
                        // Bridge task exited — sender dropped. Clean up fully:
                        // clear channels, unregister the interface from the router.
                        tracing::warn!("L2 rawlink bridge inbound channel closed — disabling L2 transport");
                        ret_inbound_rx = None;
                        ret_outbound_tx = None;
                        if let Some(ref iface_name) = rawlink_iface_name {
                            runtime.push_event(RuntimeEvent::L2InterfaceClosed {
                                interface_name: iface_name.clone(),
                            });
                        }
                        rawlink_iface_name = None;
                    }
                }
            }

            // Arm 10: Graceful shutdown (SIGTERM from procd, Ctrl+C).
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
                // Intercept RunInference before dispatch — we need `&mut runtime`
                // to take/return the engine, which dispatch_action doesn't have.
                #[cfg(feature = "inference")]
                if let RuntimeAction::RunInference {
                    query_id,
                    ref task_id,
                    ref input,
                    sampling_params_raw,
                } = action
                {
                    // Step 1: Tokenize BEFORE taking engine — borrow via as_ref().
                    use harmony_inference::InferenceEngine;
                    let (tokens, is_token_mode) = match input {
                        crate::inference::InferenceInput::Text(prompt) => {
                            match runtime.inference_engine_ref() {
                                Some(engine_ref) => match engine_ref.tokenize(prompt) {
                                    Ok(t) => (t, false),
                                    Err(e) => {
                                        let result = harmony_agent::AgentResult {
                                            task_id: task_id.clone(),
                                            status: harmony_agent::TaskStatus::Rejected,
                                            output: None,
                                            error: Some(format!("tokenize failed: {e}")),
                                        };
                                        let error_payload = harmony_agent::encode_result(&result)
                                            .unwrap_or_else(|_| b"tokenize failed".to_vec());
                                        dispatch_action(
                                            RuntimeAction::SendReply {
                                                query_id,
                                                payload: error_payload,
                                            },
                                            &session,
                                            &zenoh_tx,
                                            &udp,
                                            &broadcast_addr,
                                            Some(&peer_table),
                                            &tunnel_senders,
                                            &mut deferred_dials,
                                            &ret_outbound_tx,
                                            &data_dir,
                                            &disk_tx,
                                            &s3_read_library,
                                            &s3_tx,
                                        )
                                        .await;
                                        continue;
                                    }
                                },
                                None => {
                                    let result = harmony_agent::AgentResult {
                                        task_id: task_id.clone(),
                                        status: harmony_agent::TaskStatus::Rejected,
                                        output: None,
                                        error: Some("inference engine busy or not loaded".into()),
                                    };
                                    let error_payload = harmony_agent::encode_result(&result)
                                        .unwrap_or_else(|_| b"engine busy or not loaded".to_vec());
                                    dispatch_action(
                                        RuntimeAction::SendReply {
                                            query_id,
                                            payload: error_payload,
                                        },
                                        &session,
                                        &zenoh_tx,
                                        &udp,
                                        &broadcast_addr,
                                        Some(&peer_table),
                                        &tunnel_senders,
                                        &mut deferred_dials,
                                        &ret_outbound_tx,
                                        &data_dir,
                                        &disk_tx,
                                        &s3_read_library,
                                        &s3_tx,
                                    )
                                    .await;
                                    continue;
                                }
                            }
                        }
                        crate::inference::InferenceInput::TokenIds(ids) => (ids.clone(), true),
                    };

                    // Step 2: Take engine after successful tokenization.
                    // NOTE: In the Engram branch, the engine is held idle during
                    // async shard fetches (up to 30s per shard via try_join_all).
                    // Concurrent requests are rejected as "engine busy" during this
                    // window. Deferring take_inference_engine() into the async block
                    // would require a channel-based re-acquisition mechanism.
                    // The proper fix is prefetch pipelining (harmony-geef) where
                    // shards are pre-fetched before the inference request arrives.
                    if let Some(engine) = runtime.take_inference_engine() {
                        let tx = inference_tx.clone();
                        let max_tokens = crate::inference::DEFAULT_MAX_INFERENCE_TOKENS;
                        let query_id = query_id;
                        let task_id = task_id.clone();
                        let params = crate::inference::decode_sampling_params(&sampling_params_raw);

                        // Step 3: Prepare Engram (sync) — pure computation, no I/O.
                        let engram_prep = runtime
                            .engram_client()
                            .and_then(|client| {
                                runtime.engram_module().map(|module| (client, module))
                            })
                            .and_then(|(client, module)| {
                                match harmony_inference::engram_bridge::prepare_engram_request(
                                    client, &tokens,
                                ) {
                                    Ok(request) if !request.required_shards.is_empty() => {
                                        Some((client.clone(), module.clone(), request))
                                    }
                                    Ok(_) => {
                                        tracing::debug!("engram: no shards needed (short sequence)");
                                        None
                                    }
                                    Err(e) => {
                                        tracing::warn!(err = %e, "engram prepare failed — skipping");
                                        None
                                    }
                                }
                            });

                        if let Some((client, module, request)) = engram_prep {
                            // Step 4a: Engram branch — fetch shards async, then spawn blocking.
                            let injection_layers = runtime.engram_injection_layers().to_vec();
                            let session_clone = session.clone();
                            let panic_tx = inference_tx.clone();
                            let panic_query_id = query_id;
                            let panic_task_id = task_id.clone();
                            let outer_panic_tx = inference_tx.clone();
                            let outer_panic_qid = query_id;
                            let outer_panic_tid = task_id.clone();

                            let outer_handle = tokio::spawn(async move {
                                // Fetch all required shards in parallel via Zenoh.
                                // try_join_all short-circuits
                                // on the first failure, avoiding a 30s wait per slow shard
                                // while the engine is held.
                                let fetch_futures: Vec<_> = request
                                    .required_shards
                                    .iter()
                                    .map(|shard_req| {
                                        let cid_hex = hex::encode(shard_req.cid);
                                        let key_expr =
                                            harmony_zenoh::namespace::content::fetch_key(&cid_hex);
                                        let session_ref = &session_clone;
                                        let shard_index = shard_req.shard_index;
                                        async move {
                                            let data =
                                                fetch_via_zenoh(session_ref, &key_expr).await?;
                                            Ok::<_, String>((shard_index, data))
                                        }
                                    })
                                    .collect();

                                let engram_prefill = match futures_util::future::try_join_all(
                                    fetch_futures,
                                )
                                .await
                                {
                                    Ok(pairs) => {
                                        let shard_data: HashMap<u64, Vec<u8>> =
                                            pairs.into_iter().collect();
                                        Some(EngramPrefill {
                                            client,
                                            module,
                                            request,
                                            shard_data,
                                            injection_layers,
                                        })
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            err = %e,
                                            "engram shard fetch failed — falling back to non-Engram"
                                        );
                                        None
                                    }
                                };

                                let handle = tokio::task::spawn_blocking(move || {
                                    run_inference_loop(
                                        engine,
                                        tx,
                                        query_id,
                                        task_id,
                                        tokens,
                                        is_token_mode,
                                        params,
                                        max_tokens,
                                        engram_prefill,
                                    );
                                });
                                if let Err(e) = handle.await {
                                    tracing::error!(
                                        err = %e,
                                        "inference task panicked — engine lost"
                                    );
                                    let _ = panic_tx
                                        .send(InferenceResult::Panicked {
                                            query_id: panic_query_id,
                                            task_id: panic_task_id,
                                        })
                                        .await;
                                }
                            });
                            // Monitor the outer async task for panics during shard
                            // fetching (before spawn_blocking). If the async portion
                            // panics, the engine is dropped — send Panicked signal
                            // so inference_running is cleared.
                            tokio::spawn(async move {
                                if let Err(e) = outer_handle.await {
                                    tracing::error!(
                                        err = %e,
                                        "engram async task panicked — engine lost"
                                    );
                                    let _ = outer_panic_tx
                                        .send(InferenceResult::Panicked {
                                            query_id: outer_panic_qid,
                                            task_id: outer_panic_tid,
                                        })
                                        .await;
                                }
                            });
                        } else {
                            // Step 4b: Non-Engram branch — spawn blocking directly.
                            let panic_tx = inference_tx.clone();
                            let panic_query_id = query_id;
                            let panic_task_id = task_id.clone();
                            let handle = tokio::task::spawn_blocking(move || {
                                run_inference_loop(
                                    engine,
                                    tx,
                                    query_id,
                                    task_id,
                                    tokens,
                                    is_token_mode,
                                    params,
                                    max_tokens,
                                    None,
                                );
                            });
                            tokio::spawn(async move {
                                if let Err(e) = handle.await {
                                    tracing::error!(
                                        err = %e,
                                        "inference task panicked — engine lost"
                                    );
                                    let _ = panic_tx
                                        .send(InferenceResult::Panicked {
                                            query_id: panic_query_id,
                                            task_id: panic_task_id,
                                        })
                                        .await;
                                }
                            });
                        }
                    } else {
                        // Engine busy or not loaded — reject with proper AgentResult wire format.
                        let result = harmony_agent::AgentResult {
                            task_id: task_id.clone(),
                            status: harmony_agent::TaskStatus::Rejected,
                            output: None,
                            error: Some("inference engine busy or not loaded".into()),
                        };
                        let error_payload = harmony_agent::encode_result(&result)
                            .unwrap_or_else(|_| b"inference engine busy".to_vec());
                        dispatch_action(
                            RuntimeAction::SendReply {
                                query_id,
                                payload: error_payload,
                            },
                            &session,
                            &zenoh_tx,
                            &udp,
                            &broadcast_addr,
                            Some(&peer_table),
                            &tunnel_senders,
                            &mut deferred_dials,
                            &ret_outbound_tx,
                            &data_dir,
                            &disk_tx,
                            &s3_read_library,
                            &s3_tx,
                        )
                        .await;
                    }
                    continue;
                }

                dispatch_action(
                    action,
                    &session,
                    &zenoh_tx,
                    &udp,
                    &broadcast_addr,
                    Some(&peer_table),
                    &tunnel_senders,
                    &mut deferred_dials,
                    &ret_outbound_tx,
                    &data_dir,
                    &disk_tx,
                    &s3_read_library,
                    &s3_tx,
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

                    // Check tunnel limit (include inflight handshakes)
                    if tunnel_senders.len() + inflight_handshakes >= MAX_TUNNEL_CONNECTIONS {
                        tracing::warn!(
                            identity = %hex::encode(dial.identity_hash),
                            "tunnel limit reached — dropping deferred dial"
                        );
                        continue;
                    }

                    // Construct PqIdentity from the announce's public key bytes
                    let remote_pq_identity =
                        match construct_pq_identity(&dial.peer_dsa_pubkey, &dial.peer_kem_pubkey) {
                            Ok(id) => id,
                            Err(e) => {
                                tracing::warn!(
                                    identity = %hex::encode(dial.identity_hash),
                                    err = %e,
                                    "PqIdentity construction failed — dropping dial"
                                );
                                continue;
                            }
                        };

                    let connection_id = next_connection_id;
                    next_connection_id += 1;
                    inflight_handshakes += 1;

                    // Build target NodeAddr
                    let target_node_id = match iroh::NodeId::from_bytes(&dial.node_id) {
                        Ok(id) => id,
                        Err(e) => {
                            tracing::warn!(
                                identity = %hex::encode(dial.identity_hash),
                                err = %e,
                                "invalid NodeId in announce record — dropping dial"
                            );
                            inflight_handshakes -= 1;
                            continue;
                        }
                    };
                    let mut node_addr = iroh::NodeAddr::new(target_node_id);
                    if let Some(ref url) = dial.relay_url {
                        if let Ok(relay_url) = url.parse::<iroh::RelayUrl>() {
                            node_addr = node_addr.with_relay_url(relay_url);
                        }
                    }

                    let interface_name = format!("tunnel-{}", &hex::encode(&dial.node_id[..8]));
                    let conn_tx_clone = conn_tx.clone();
                    let relay_map_clone = relay_map.clone();

                    tracing::info!(
                        identity = %hex::encode(dial.identity_hash),
                        node_id = %hex::encode(&dial.node_id[..8]),
                        relay = ?dial.relay_url,
                        "initiating tunnel dial"
                    );

                    // Spawn async dial — sends ReadyConnection on success
                    tokio::spawn(async move {
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
                                // Signal failure so inflight_handshakes is decremented
                                let _ = conn_tx_clone.send(None).await;
                                return;
                            }
                        };

                        let conn = match ep
                            .connect(node_addr, tunnel_task::HARMONY_TUNNEL_ALPN)
                            .await
                        {
                            Ok(conn) => conn,
                            Err(e) => {
                                tracing::warn!(err = %e, "tunnel dial failed");
                                ep.close().await;
                                let _ = conn_tx_clone.send(None).await;
                                return;
                            }
                        };

                        tracing::info!(%interface_name, "tunnel dial connected");
                        let _ = conn_tx_clone
                            .send(Some(ReadyConnection {
                                connection: conn,
                                connection_id,
                                interface_name,
                                remote_pq_identity: Some(remote_pq_identity),
                                initiator_endpoint: Some(ep),
                            }))
                            .await;
                    });
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
    for (iface, ep) in initiator_endpoints.drain() {
        tracing::debug!(%iface, "closing initiator endpoint");
        ep.close().await;
    }
    if let Some(ref ep) = iroh_endpoint {
        tracing::info!("closing iroh endpoint");
        ep.close().await;
    }

    Ok(())
}

/// Dispatch a single `RuntimeAction` to the appropriate I/O mechanism.
#[allow(clippy::too_many_arguments, unused_variables)]
async fn dispatch_action(
    action: RuntimeAction,
    session: &zenoh::Session,
    zenoh_tx: &mpsc::Sender<ZenohEvent>,
    udp: &UdpSocket,
    broadcast_addr: &SocketAddr,
    peer_table: Option<&PeerTable>,
    tunnel_senders: &HashMap<String, TunnelSender>,
    deferred_dials: &mut BinaryHeap<Reverse<DeferredDial>>,
    ret_outbound_tx: &Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
    data_dir: &Option<std::path::PathBuf>,
    disk_tx: &mpsc::Sender<DiskIoResult>,
    s3_read_library: &S3ReadLibrary,
    s3_tx: &mpsc::Sender<S3IoResult>,
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
                if interface_name.starts_with("l2:") {
                    if let Some(ref tx) = ret_outbound_tx {
                        if tx.try_send(raw.clone()).is_err() {
                            tracing::warn!(%interface_name, "L2 send queue full — dropping packet");
                        }
                    }
                } else if interface_name.starts_with("tunnel-") {
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
        RuntimeAction::SendVerifyQuery { key_expr, payload } => {
            // DSD: send a Zenoh query to the target's verify queryable and feed
            // the reply back as RuntimeEvent::VerifyResponse.
            // IMPORTANT: Every code path must send a VerifyResponse (success or
            // error) so the edge always clears dsd_session.
            let tx = zenoh_tx.clone();
            let session = session.clone();
            tokio::spawn(async move {
                let deadline = Duration::from_secs(30);
                let result = tokio::time::timeout(deadline, async {
                    let replies = match session.get(&key_expr).payload(payload).await {
                        Ok(r) => r,
                        Err(e) => {
                            return Err(format!("zenoh get failed: {e}"));
                        }
                    };
                    while let Ok(reply) = replies.recv_async().await {
                        if let Ok(sample) = reply.into_result() {
                            let resp_payload = sample.payload().to_bytes().to_vec();
                            let _ = tx
                                .send(ZenohEvent::VerifyResponse {
                                    payload: resp_payload,
                                })
                                .await;
                            return Ok(());
                        }
                    }
                    Err("no valid reply from target".to_string())
                })
                .await;

                // Send error VerifyResponse for timeout, get failure, or no reply.
                let err_msg = match result {
                    Ok(Ok(())) => return, // Success — response already sent above
                    Ok(Err(e)) => e,
                    Err(_) => "verify query timed out after 30s".to_string(),
                };
                tracing::warn!(%key_expr, err = %err_msg, "DSD verify query failed");
                let err_payload = harmony_speculative::VerifyResponse::serialize_error(&err_msg);
                let _ = tx
                    .send(ZenohEvent::VerifyResponse {
                        payload: err_payload,
                    })
                    .await;
            });
        }
        RuntimeAction::PersistToDisk { cid, data } => {
            if let Some(ref dir) = data_dir {
                let dir = dir.clone();
                tokio::task::spawn_blocking(move || {
                    if let Err(e) = crate::disk_io::write_book(&dir, &cid, &data) {
                        tracing::error!(
                            cid = %hex::encode(&cid.to_bytes()[..8]),
                            err = %e,
                            "disk write failed"
                        );
                    }
                });
            }
        }
        RuntimeAction::DiskLookup { cid, query_id } => {
            if let Some(ref dir) = data_dir {
                let dir = dir.clone();
                let tx = disk_tx.clone();
                tokio::task::spawn_blocking(move || match crate::disk_io::read_book(&dir, &cid) {
                    Ok(data) => {
                        let _ = tx.blocking_send(DiskIoResult::ReadComplete {
                            cid,
                            query_id,
                            data,
                        });
                    }
                    Err(_) => {
                        let _ = tx.blocking_send(DiskIoResult::ReadFailed { cid, query_id });
                    }
                });
            } else {
                tracing::error!(
                    cid = %hex::encode(&cid.to_bytes()[..8]),
                    "DiskLookup dispatched but data_dir not configured"
                );
                let _ = disk_tx.try_send(DiskIoResult::ReadFailed { cid, query_id });
            }
        }
        RuntimeAction::RemoveFromDisk { cid } => {
            if let Some(ref dir) = data_dir {
                let dir = dir.clone();
                tokio::task::spawn_blocking(move || {
                    if let Err(e) = crate::disk_io::delete_book(&dir, &cid) {
                        tracing::warn!(?cid, error = %e, "failed to delete book from disk");
                    }
                });
            }
        }
        RuntimeAction::S3Lookup { cid, query_id } => {
            #[cfg(feature = "archivist")]
            {
                if let Some(ref s3) = s3_read_library {
                    let s3 = s3.clone();
                    let tx = s3_tx.clone();
                    tokio::spawn(async move {
                        match s3.get_book(&cid.to_bytes()).await {
                            Ok(Some(data)) => {
                                let _ = tx
                                    .send(S3IoResult::ReadComplete {
                                        cid,
                                        query_id,
                                        data,
                                    })
                                    .await;
                            }
                            Ok(None) => {
                                tracing::debug!(cid = %hex::encode(&cid.to_bytes()[..8]), "S3 book not found");
                                let _ = tx.send(S3IoResult::ReadFailed { cid, query_id }).await;
                            }
                            Err(e) => {
                                tracing::warn!(cid = %hex::encode(&cid.to_bytes()[..8]), err = %e, "S3 fetch error");
                                let _ = tx.send(S3IoResult::ReadFailed { cid, query_id }).await;
                            }
                        }
                    });
                } else {
                    // S3 library unavailable (init failed at startup) — resolve
                    // the query immediately so it doesn't hang.
                    tracing::warn!(
                        cid = %hex::encode(&cid.to_bytes()[..8]),
                        "S3Lookup dispatched but s3_read_library not initialised"
                    );
                    let _ = s3_tx.try_send(S3IoResult::ReadFailed { cid, query_id });
                }
            }
            // When archivist feature is disabled, resolve the query immediately.
            #[cfg(not(feature = "archivist"))]
            {
                tracing::debug!("S3Lookup ignored — archivist feature not enabled");
                let _ = s3_tx.try_send(S3IoResult::ReadFailed { cid, query_id });
            }
        }
        // RunInference is intercepted in the action loop before dispatch_action
        // is called. If it reaches here, it's a logic error — log and ignore.
        #[cfg(feature = "inference")]
        RuntimeAction::RunInference { .. } => {
            tracing::error!("RunInference reached dispatch_action — should be intercepted");
        }
    }
}

/// Process a single inference result from the streaming channel.
///
/// Called from the select loop's inference arm. Publishes stream chunks to Zenoh,
/// returns the engine to the runtime on completion/failure, and dispatches a
/// `SendReply` for the final query response.
#[cfg(feature = "inference")]
#[allow(clippy::too_many_arguments)]
async fn handle_inference_result(
    inference_result: InferenceResult,
    runtime: &mut NodeRuntime<MemoryBookStore>,
    session: &zenoh::Session,
    zenoh_tx: &mpsc::Sender<ZenohEvent>,
    udp: &UdpSocket,
    broadcast_addr: &SocketAddr,
    peer_table: &PeerTable,
    tunnel_senders: &HashMap<String, TunnelSender>,
    deferred_dials: &mut BinaryHeap<Reverse<DeferredDial>>,
    ret_outbound_tx: &Option<tokio::sync::mpsc::Sender<Vec<u8>>>,
    data_dir: &Option<std::path::PathBuf>,
    disk_tx: &mpsc::Sender<DiskIoResult>,
    s3_read_library: &S3ReadLibrary,
    s3_tx: &mpsc::Sender<S3IoResult>,
) {
    match inference_result {
        InferenceResult::Chunk {
            task_id,
            sequence,
            token_text,
            token_id,
            final_chunk,
        } => {
            // Token mode uses token_id (Some): {"token_id": N} for content,
            // {"final": true} for the termination sentinel.
            // Text mode uses token_id (None): {"token": "text"} always,
            // preserving 0x02 backwards compatibility.
            let payload_value = if let Some(id) = token_id {
                if final_chunk {
                    serde_json::json!({"final": true})
                } else {
                    serde_json::json!({"token_id": id})
                }
            } else {
                serde_json::json!({"token": token_text})
            };
            let chunk = harmony_agent::StreamChunk {
                task_id: task_id.clone(),
                sequence,
                payload: payload_value,
                final_chunk,
            };
            if let Ok(payload) = harmony_agent::encode_chunk(&chunk) {
                let node_addr_hex = hex::encode(runtime.local_pq_identity_hash());
                let key_expr =
                    harmony_zenoh::namespace::agent::stream_key(&node_addr_hex, &task_id);
                if final_chunk {
                    // Await final chunk directly so it completes before the
                    // Complete/Failed handler dispatches the query reply.
                    if let Err(e) = session.put(&key_expr, payload).await {
                        tracing::warn!(%key_expr, err = %e, "final stream chunk publish error");
                    }
                } else {
                    let session_clone = session.clone();
                    tokio::spawn(async move {
                        if let Err(e) = session_clone.put(&key_expr, payload).await {
                            tracing::warn!(%key_expr, err = %e, "stream chunk publish error");
                        }
                    });
                }
            }
        }
        InferenceResult::Complete {
            query_id,
            task_id,
            output,
            engine,
        } => {
            runtime.return_inference_engine(engine);
            let output_json = match &output {
                crate::inference::InferenceOutput::Text(text) => {
                    serde_json::json!({"text": text})
                }
                crate::inference::InferenceOutput::TokenIds(ids) => {
                    serde_json::json!({"token_ids": ids})
                }
            };
            let result = harmony_agent::AgentResult {
                task_id,
                status: harmony_agent::TaskStatus::Success,
                output: Some(output_json),
                error: None,
            };
            match harmony_agent::encode_result(&result) {
                Ok(payload) => {
                    dispatch_action(
                        RuntimeAction::SendReply { query_id, payload },
                        session,
                        zenoh_tx,
                        udp,
                        broadcast_addr,
                        Some(peer_table),
                        tunnel_senders,
                        deferred_dials,
                        ret_outbound_tx,
                        data_dir,
                        disk_tx,
                        s3_read_library,
                        s3_tx,
                    )
                    .await;
                }
                Err(e) => {
                    tracing::error!(%query_id, err = %e, "encode_result failed; query will not be answered");
                }
            }
        }
        InferenceResult::Failed {
            query_id,
            task_id,
            error,
            engine,
        } => {
            runtime.return_inference_engine(engine);
            let result = harmony_agent::AgentResult {
                task_id,
                status: harmony_agent::TaskStatus::Failed,
                output: None,
                error: Some(error),
            };
            match harmony_agent::encode_result(&result) {
                Ok(payload) => {
                    dispatch_action(
                        RuntimeAction::SendReply { query_id, payload },
                        session,
                        zenoh_tx,
                        udp,
                        broadcast_addr,
                        Some(peer_table),
                        tunnel_senders,
                        deferred_dials,
                        ret_outbound_tx,
                        data_dir,
                        disk_tx,
                        s3_read_library,
                        s3_tx,
                    )
                    .await;
                }
                Err(e) => {
                    tracing::error!(%query_id, err = %e, "encode_result failed; query will not be answered");
                }
            }
        }
        InferenceResult::Panicked { query_id, task_id } => {
            // Engine is gone (consumed by panic). Reset inference_running so future
            // requests can reload the engine from cached GGUF/tokenizer data.
            runtime.reset_inference_after_panic();
            let result = harmony_agent::AgentResult {
                task_id,
                status: harmony_agent::TaskStatus::Failed,
                output: None,
                error: Some("inference engine panicked".into()),
            };
            match harmony_agent::encode_result(&result) {
                Ok(payload) => {
                    dispatch_action(
                        RuntimeAction::SendReply { query_id, payload },
                        session,
                        zenoh_tx,
                        udp,
                        broadcast_addr,
                        Some(peer_table),
                        tunnel_senders,
                        deferred_dials,
                        ret_outbound_tx,
                        data_dir,
                        disk_tx,
                        s3_read_library,
                        s3_tx,
                    )
                    .await;
                }
                Err(e) => {
                    tracing::error!(%query_id, err = %e, "encode_result failed after panic");
                }
            }
        }
    }
}

/// Pre-fetched Engram data for the blocking inference task.
#[cfg(feature = "inference")]
struct EngramPrefill {
    client: harmony_engram::EngramClient,
    module: harmony_inference::EngramGatedResidual,
    request: harmony_inference::engram_bridge::EngramRequest,
    shard_data: std::collections::HashMap<u64, Vec<u8>>,
    injection_layers: Vec<usize>,
}

/// Run the autoregressive token loop on a blocking thread.
///
/// Streams token chunks via `tx` and returns the engine (with output or error)
/// when complete. Supports both text mode (detokenize + accumulate text) and
/// token mode (accumulate raw token IDs). The event loop receives results on
/// `inference_rx` and routes them.
///
/// Tokens are pre-tokenized by the caller — no `InferenceInput` match needed.
/// When `engram` is `Some`, the prefill forward uses Engram-augmented inference.
#[cfg(feature = "inference")]
fn run_inference_loop(
    engine: harmony_inference::QwenEngine,
    tx: mpsc::Sender<InferenceResult>,
    query_id: u64,
    task_id: String,
    tokens: Vec<u32>,
    is_token_mode: bool,
    sampling_params: harmony_inference::SamplingParams,
    max_tokens: u32,
    engram: Option<EngramPrefill>,
) {
    use harmony_inference::InferenceEngine;

    let mut cache = match engine.new_cache() {
        Ok(c) => c,
        Err(e) => {
            let _ = tx.blocking_send(InferenceResult::Failed {
                query_id,
                task_id,
                error: format!("cache init failed: {e}"),
                engine,
            });
            return;
        }
    };

    let mut history: Vec<u32> = Vec::new();
    history.extend_from_slice(&tokens);

    // Prefill forward: use Engram-augmented path if available.
    let mut logits = match engram {
        Some(ref ep) => {
            match harmony_inference::engram_bridge::resolve_engram_embeddings(
                &ep.client,
                &ep.request,
                &ep.shard_data,
                engine.device(),
            ) {
                Ok(embeddings) => {
                    let ctx = harmony_inference::EngramContext {
                        module: &ep.module,
                        embeddings,
                        injection_layers: &ep.injection_layers,
                    };
                    match engine.forward_with_engram(&tokens, &mut cache, &ctx) {
                        Ok(l) => l,
                        Err(e) => {
                            // forward_with_engram may have partially updated the
                            // KV cache (layers 0..k got entries but position wasn't
                            // advanced). A fallback with the same cache would produce
                            // duplicate KV entries. Create a fresh cache instead.
                            tracing::warn!(err = %e, "engram forward failed — creating fresh cache for fallback");
                            cache = match engine.new_cache() {
                                Ok(c) => c,
                                Err(e2) => {
                                    let _ = tx.blocking_send(InferenceResult::Failed {
                                        query_id,
                                        task_id,
                                        error: format!(
                                            "cache re-init failed after engram error: {e2}"
                                        ),
                                        engine,
                                    });
                                    return;
                                }
                            };
                            match engine.forward(&tokens, &mut cache) {
                                Ok(l) => l,
                                Err(e2) => {
                                    let _ = tx.blocking_send(InferenceResult::Failed {
                                        query_id,
                                        task_id,
                                        error: format!("forward failed: {e2}"),
                                        engine,
                                    });
                                    return;
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(err = %e, "engram resolve failed — falling back to plain forward");
                    match engine.forward(&tokens, &mut cache) {
                        Ok(l) => l,
                        Err(e2) => {
                            let _ = tx.blocking_send(InferenceResult::Failed {
                                query_id,
                                task_id,
                                error: format!("forward failed: {e2}"),
                                engine,
                            });
                            return;
                        }
                    }
                }
            }
        }
        None => match engine.forward(&tokens, &mut cache) {
            Ok(l) => l,
            Err(e) => {
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("forward failed: {e}"),
                    engine,
                });
                return;
            }
        },
    };

    let mut full_text = String::new();
    let mut generated_ids: Vec<u32> = Vec::new();
    let mut sequence = 0u32;
    let eos = engine.eos_token_id();

    loop {
        let next_token = match engine.sample(&logits, &sampling_params, &history) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.blocking_send(InferenceResult::Failed {
                    query_id,
                    task_id,
                    error: format!("sample failed: {e}"),
                    engine,
                });
                return;
            }
        };

        if eos == Some(next_token) || sequence >= max_tokens {
            // Token mode: set token_id to trigger the {"final": true} path.
            // Text mode: token_id None preserves {"token": ""} (backwards compat).
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: String::new(),
                token_id: if is_token_mode { Some(0) } else { None },
                final_chunk: true,
            });
            break;
        }

        history.push(next_token);

        if is_token_mode {
            generated_ids.push(next_token);
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: String::new(),
                token_id: Some(next_token),
                final_chunk: false,
            });
        } else {
            let text = engine.detokenize(&[next_token]).unwrap_or_default();
            full_text.push_str(&text);
            let _ = tx.blocking_send(InferenceResult::Chunk {
                task_id: task_id.clone(),
                sequence,
                token_text: text,
                token_id: None,
                final_chunk: false,
            });
        }

        sequence += 1;

        // Opportunistic decode-step Engram injection: reuse cached shards from
        // prefill. Falls back to plain forward on any miss or error.
        logits = 'decode_fwd: {
            if let Some(ref ep) = engram {
                if history.len() >= 2 {
                    let window = &history[history.len().saturating_sub(3)..];
                    if let Ok(req) =
                        harmony_inference::engram_bridge::prepare_engram_request(&ep.client, window)
                    {
                        let all_cached = req
                            .required_shards
                            .iter()
                            .all(|s| ep.shard_data.contains_key(&s.shard_index));
                        if all_cached {
                            if let Ok(embeddings) =
                                harmony_inference::engram_bridge::resolve_engram_embeddings(
                                    &ep.client,
                                    &req,
                                    &ep.shard_data,
                                    engine.device(),
                                )
                            {
                                if let Ok(slice) =
                                    embeddings.narrow(1, req.seq_len.saturating_sub(1), 1)
                                {
                                    let ctx = harmony_inference::EngramContext {
                                        module: &ep.module,
                                        embeddings: slice,
                                        injection_layers: &ep.injection_layers,
                                    };
                                    match engine
                                        .forward_with_engram(&[next_token], &mut cache, &ctx)
                                    {
                                        Ok(l) => break 'decode_fwd l,
                                        Err(e) => {
                                            // Warn (not trace) — this is an unexpected
                                            // engine error, not a normal cache miss.
                                            // Note: KV cache may have partial entries
                                            // from layers 0..k. For seq_len=1 decode
                                            // this is minor (1 token of duplicates).
                                            tracing::warn!(
                                                err = %e,
                                                "decode engram forward failed — falling back"
                                            );
                                        }
                                    }
                                }
                            }
                        }
                    }
                    tracing::trace!("engram decode miss — using plain forward");
                }
            }
            match engine.forward(&[next_token], &mut cache) {
                Ok(l) => l,
                Err(e) => {
                    let _ = tx.blocking_send(InferenceResult::Failed {
                        query_id,
                        task_id,
                        error: format!("forward failed: {e}"),
                        engine,
                    });
                    return;
                }
            }
        };
    }

    let output = if is_token_mode {
        crate::inference::InferenceOutput::TokenIds(generated_ids)
    } else {
        crate::inference::InferenceOutput::Text(full_text)
    };
    let _ = tx.blocking_send(InferenceResult::Complete {
        query_id,
        task_id,
        output,
        engine,
    });
}

/// Construct a `PqIdentity` from raw ML-DSA verifying key and ML-KEM encapsulation key bytes.
///
/// The announce record stores: `public_key` = ML-DSA (1952 bytes),
/// `encryption_key` = ML-KEM (1184 bytes). `PqIdentity::from_public_keys` takes
/// `(MlKemPublicKey, MlDsaPublicKey)` — note the order.
fn construct_pq_identity(
    dsa_bytes: &[u8],
    kem_bytes: &[u8],
) -> Result<harmony_identity::PqIdentity, String> {
    let verifying_key = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(dsa_bytes)
        .map_err(|e| format!("invalid ML-DSA public key: {e}"))?;
    let encryption_key = harmony_crypto::ml_kem::MlKemPublicKey::from_bytes(kem_bytes)
        .map_err(|e| format!("invalid ML-KEM public key: {e}"))?;
    Ok(harmony_identity::PqIdentity::from_public_keys(
        encryption_key,
        verifying_key,
    ))
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

/// Persist a memo to disk if data_dir is configured.
#[allow(dead_code)]
fn persist_memo_to_disk(
    data_dir: &Option<std::path::PathBuf>,
    memo: &harmony_memo::Memo,
) {
    if let Some(ref dir) = data_dir {
        let dir = dir.clone();
        let memo = memo.clone();
        tokio::task::spawn_blocking(move || {
            if let Err(e) = crate::memo_io::write_memo(&dir, &memo) {
                tracing::warn!("Failed to persist memo: {}", e);
            }
        });
    }
}

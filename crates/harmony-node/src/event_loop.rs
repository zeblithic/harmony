//! Async event loop — drives `NodeRuntime` via tokio select! with UDP, timer, and Zenoh.
//!
//! `NodeRuntime` is `!Send`; it lives entirely on the select loop task.
//! Zenoh objects that need spawning are cloned (Session is internally Arc'd).

use std::net::SocketAddr;
use std::time::Duration;

use harmony_content::blob::MemoryBlobStore;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tokio::time;

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
    mut runtime: NodeRuntime<MemoryBlobStore>,
    startup_actions: Vec<RuntimeAction>,
    listen_addr: SocketAddr,
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

    // ── Execute startup actions (declare queryables + subscribers) ────────────
    for action in startup_actions {
        dispatch_action(
            action,
            &session,
            &zenoh_tx,
            &udp,
            &broadcast_addr,
        )
        .await;
    }

    // ── Timer (250 ms tick) ───────────────────────────────────────────────────
    let mut timer = time::interval(Duration::from_millis(250));
    timer.set_missed_tick_behavior(time::MissedTickBehavior::Skip);

    // ── Reusable UDP receive buffer ───────────────────────────────────────────
    // Max UDP datagram size — Reticulum MTU is 500 (default) or 1024 (medium
    // interfaces). Use 65535 to never silently truncate.
    let mut udp_buf = vec![0u8; 65535];

    // ── Monotonic query-id counter ────────────────────────────────────────────
    let mut next_query_id: u64 = 1;

    // ── Select loop ──────────────────────────────────────────────────────────
    loop {
        let runtime_events: Vec<RuntimeEvent> = tokio::select! {
            // Arm 1: UDP packet received
            result = udp.recv_from(&mut udp_buf) => {
                match result {
                    Ok((len, _src)) => {
                        let interface_name = "udp0".to_string();
                        let raw = udp_buf[..len].to_vec();
                        let now = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_millis() as u64;
                        vec![RuntimeEvent::InboundPacket { interface_name, raw, now }]
                    }
                    Err(e) => {
                        eprintln!("[event_loop] UDP recv error: {e}");
                        // Fatal socket errors should exit — transient ones continue.
                        use std::io::ErrorKind::*;
                        match e.kind() {
                            WouldBlock | Interrupted => {}
                            _ => return Err(e.into()),
                        }
                        vec![]
                    }
                }
            }

            // Arm 2: 250 ms timer tick
            _ = timer.tick() => {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64;
                vec![RuntimeEvent::TimerTick { now }]
            }

            // Arm 3: Zenoh bridge event
            maybe = zenoh_rx.recv() => {
                match maybe {
                    None => {
                        eprintln!("[event_loop] Zenoh channel closed — exiting");
                        break;
                    }
                    Some(ev) => match ev {
                        ZenohEvent::Query { key_expr, payload } => {
                            let query_id = next_query_id;
                            next_query_id += 1;
                            vec![RuntimeEvent::QueryReceived { query_id, key_expr, payload }]
                        }
                        ZenohEvent::ComputeQuery { key_expr, payload } => {
                            let query_id = next_query_id;
                            next_query_id += 1;
                            vec![RuntimeEvent::ComputeQuery { query_id, key_expr, payload }]
                        }
                        ZenohEvent::Subscription { key_expr, payload } => {
                            vec![RuntimeEvent::SubscriptionMessage { key_expr, payload }]
                        }
                        ZenohEvent::FetchResponse { cid, is_module, result } => {
                            if is_module {
                                vec![RuntimeEvent::ModuleFetchResponse { cid, result }]
                            } else {
                                vec![RuntimeEvent::ContentFetchResponse { cid, result }]
                            }
                        }
                    }
                }
            }
        };

        // Push all events from this select arm into the runtime.
        for event in runtime_events {
            runtime.push_event(event);
        }

        // Run one runtime tick and dispatch resulting actions.
        let actions = runtime.tick();
        for action in actions {
            dispatch_action(action, &session, &zenoh_tx, &udp, &broadcast_addr).await;
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
) {
    match action {
        // ── Tier 1: UDP broadcast ─────────────────────────────────────────────
        RuntimeAction::SendOnInterface { raw, .. } => {
            if let Err(e) = udp.send_to(&raw, broadcast_addr).await {
                eprintln!("[event_loop] UDP send error: {e}");
            }
        }

        // ── Tier 2: Zenoh publish ─────────────────────────────────────────────
        RuntimeAction::Publish { key_expr, payload } => {
            if let Err(e) = session.put(&key_expr, payload).await {
                eprintln!("[event_loop] Zenoh put error on '{key_expr}': {e}");
            }
        }

        // ── Tier 2: Reply to query — stub for v1 ─────────────────────────────
        // Query objects cannot be passed through channels in zenoh 1.x without
        // restructuring (the query handle must stay alive on the spawned task).
        // Deferred to a follow-up bead.
        RuntimeAction::SendReply { query_id, .. } => {
            eprintln!("[event_loop] SendReply query_id={query_id}: reply passthrough not yet implemented");
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
                    eprintln!("[event_loop] declare_queryable '{key_expr}' failed: {e}");
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
                    eprintln!("[event_loop] declare_subscriber '{key_expr}' failed: {e}");
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
                    eprintln!("[event_loop] get reply error: {msg}");
                }
            }
        }
        Err(format!("no successful reply for '{key_expr}'"))
    })
    .await
    .unwrap_or_else(|_| Err(format!("no successful reply for '{key_expr}' (timed out after 30s)")))
}

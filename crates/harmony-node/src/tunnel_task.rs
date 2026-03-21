//! Per-connection async task that bridges iroh QUIC streams to TunnelSession.
//!
//! One task is spawned per active tunnel connection (inbound or outbound).
//! It owns a TunnelSession state machine and drives it by:
//! 1. Reading bytes from the iroh RecvStream -> TunnelEvent::InboundBytes
//! 2. Processing TunnelActions: OutboundBytes -> iroh SendStream
//! 3. Forwarding ReticulumReceived/ZenohReceived -> TunnelBridgeEvent via mpsc
//! 4. Running a keepalive timer -> TunnelEvent::Tick

use std::time::{Duration, Instant};

use harmony_identity::{PqIdentity, PqPrivateIdentity};
use harmony_tunnel::session::TunnelSession;
use harmony_tunnel::{TunnelAction, TunnelEvent};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use tokio::sync::mpsc;

use crate::tunnel_bridge::{TunnelBridgeEvent, TunnelCommand};

/// ALPN protocol identifier for harmony tunnels.
pub const HARMONY_TUNNEL_ALPN: &[u8] = b"harmony-tunnel/1";

/// Maximum time allowed for the handshake phase (stream open/accept,
/// length-prefixed message exchange, TunnelSession creation). The main loop
/// has its own keepalive/dead-peer timeout and does not need this.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// Run the initiator side of a tunnel connection.
///
/// Creates a TunnelSession, sends TunnelInit over the stream, waits for
/// TunnelAccept, then enters the main read/write loop.
pub async fn run_initiator(
    conn: Connection,
    local_identity: &PqPrivateIdentity,
    remote_identity: &PqIdentity,
    bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
    cmd_rx: mpsc::Receiver<TunnelCommand>,
    interface_name: String,
    connection_id: u64,
) {
    // Wrap handshake phase in timeout
    let handshake_result = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        initiator_handshake(&conn, local_identity, remote_identity, &bridge_tx, &interface_name, connection_id),
    )
    .await;

    let (session, send_stream, recv_stream) = match handshake_result {
        Ok(Ok(result)) => result,
        Ok(Err(reason)) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason,
                    connection_id,
                })
                .await;
            return;
        }
        Err(_timeout) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: "handshake timed out".to_string(),
                    connection_id,
                })
                .await;
            return;
        }
    };

    // Enter main loop
    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        bridge_tx,
        cmd_rx,
        interface_name,
        connection_id,
    )
    .await;
}

/// Handshake phase for the initiator: open bi stream, send TunnelInit,
/// read TunnelAccept, process handshake completion actions.
async fn initiator_handshake(
    conn: &Connection,
    local_identity: &PqPrivateIdentity,
    remote_identity: &PqIdentity,
    bridge_tx: &mpsc::Sender<TunnelBridgeEvent>,
    interface_name: &str,
    connection_id: u64,
) -> Result<(TunnelSession, SendStream, RecvStream), String> {
    let now_ms = millis_since_start();

    // Create TunnelSession as initiator
    let mut rng = rand::rngs::OsRng;
    let (mut session, init_actions) = TunnelSession::new_initiator(
        &mut rng,
        local_identity,
        remote_identity,
        now_ms,
    )
    .map_err(|e| format!("handshake init failed: {e}"))?;

    // Open a bidirectional stream for the tunnel
    let (mut send_stream, mut recv_stream) = conn
        .open_bi()
        .await
        .map_err(|e| format!("failed to open bi stream: {e}"))?;

    // Send TunnelInit (OutboundBytes from init_actions)
    for action in init_actions {
        if let TunnelAction::OutboundBytes { data } = action {
            write_length_prefixed(&mut send_stream, &data)
                .await
                .map_err(|e| format!("failed to send TunnelInit: {e}"))?;
        }
    }

    // Read TunnelAccept
    let accept_bytes = read_length_prefixed(&mut recv_stream, HANDSHAKE_MAX_MESSAGE)
        .await
        .map_err(|e| format!("failed to read TunnelAccept: {e}"))?;

    // Process TunnelAccept through the state machine
    let now_ms = millis_since_start();
    let actions = session
        .handle_event(TunnelEvent::InboundBytes {
            data: accept_bytes,
            now_ms,
        })
        .map_err(|e| format!("handshake failed: {e}"))?;

    // Dispatch handshake completion actions (HandshakeComplete + any OutboundBytes)
    if !dispatch_tunnel_actions(&actions, &mut send_stream, bridge_tx, interface_name, connection_id).await {
        return Err("handshake dispatch failed".to_string());
    }

    Ok((session, send_stream, recv_stream))
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
    interface_name: String,
    connection_id: u64,
) {
    // Wrap handshake phase in timeout
    let handshake_result = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        responder_handshake(&conn, local_identity, &bridge_tx, &interface_name, connection_id),
    )
    .await;

    let (session, send_stream, recv_stream) = match handshake_result {
        Ok(Ok(result)) => result,
        Ok(Err(reason)) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason,
                    connection_id,
                })
                .await;
            return;
        }
        Err(_timeout) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: "handshake timed out".to_string(),
                    connection_id,
                })
                .await;
            return;
        }
    };

    // Enter main loop
    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        bridge_tx,
        cmd_rx,
        interface_name,
        connection_id,
    )
    .await;
}

/// Handshake phase for the responder: accept bi stream, read TunnelInit,
/// create TunnelSession, send TunnelAccept.
async fn responder_handshake(
    conn: &Connection,
    local_identity: &PqPrivateIdentity,
    bridge_tx: &mpsc::Sender<TunnelBridgeEvent>,
    interface_name: &str,
    connection_id: u64,
) -> Result<(TunnelSession, SendStream, RecvStream), String> {
    // Accept the bidirectional stream
    let (mut send_stream, mut recv_stream) = conn
        .accept_bi()
        .await
        .map_err(|e| format!("failed to accept bi stream: {e}"))?;

    // Read TunnelInit
    let init_bytes = read_length_prefixed(&mut recv_stream, HANDSHAKE_MAX_MESSAGE)
        .await
        .map_err(|e| format!("failed to read TunnelInit: {e}"))?;

    // Create TunnelSession as responder
    let mut rng = rand::rngs::OsRng;
    let now_ms = millis_since_start();
    let (session, accept_actions) = TunnelSession::new_responder(
        &mut rng,
        local_identity,
        &init_bytes,
        now_ms,
    )
    .map_err(|e| format!("responder handshake failed: {e}"))?;

    // Dispatch accept actions (sends TunnelAccept + emits HandshakeComplete)
    if !dispatch_tunnel_actions(
        &accept_actions,
        &mut send_stream,
        bridge_tx,
        interface_name,
        connection_id,
    )
    .await
    {
        return Err("handshake dispatch failed".to_string());
    }

    Ok((session, send_stream, recv_stream))
}

/// Main tunnel read/write/keepalive loop.
///
/// Runs until the tunnel is closed, errors, or the command channel drops.
async fn run_tunnel_loop(
    mut session: TunnelSession,
    mut send_stream: SendStream,
    mut recv_stream: RecvStream,
    bridge_tx: mpsc::Sender<TunnelBridgeEvent>,
    mut cmd_rx: mpsc::Receiver<TunnelCommand>,
    interface_name: String,
    connection_id: u64,
) {
    // Timer fires at 10s intervals -- more frequent than the 30s keepalive
    // interval, but the TunnelSession state machine decides when to actually
    // emit a keepalive frame (every 30s) and when to declare dead peer (90s).
    // The 10s tick ensures responsive timeout detection.
    let mut keepalive_timer = tokio::time::interval(Duration::from_secs(10));
    keepalive_timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            // Read from iroh stream
            result = read_length_prefixed(&mut recv_stream, DATA_MAX_MESSAGE) => {
                match result {
                    Ok(data) => {
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::InboundBytes { data, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, &bridge_tx, &interface_name, connection_id,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                                    interface_name: interface_name.clone(),
                                    reason: format!("tunnel error: {e}"),
                                    connection_id,
                                }).await;
                                break;
                            }
                        }
                    }
                    Err(e) => {
                        let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.clone(),
                            reason: format!("stream read error: {e}"),
                            connection_id,
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
                                    &actions, &mut send_stream, &bridge_tx, &interface_name, connection_id,
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
                                    &actions, &mut send_stream, &bridge_tx, &interface_name, connection_id,
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
                            connection_id,
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
                            &actions, &mut send_stream, &bridge_tx, &interface_name, connection_id,
                        ).await {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = bridge_tx.send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.clone(),
                            reason: format!("tick error: {e}"),
                            connection_id,
                        }).await;
                        break;
                    }
                }
            }
        }
    }
}

/// Process TunnelActions: send outbound bytes, forward bridge events.
///
/// Returns `false` if the tunnel should be closed (Error or Closed action
/// received), `true` to continue.
async fn dispatch_tunnel_actions(
    actions: &[TunnelAction],
    send_stream: &mut SendStream,
    bridge_tx: &mpsc::Sender<TunnelBridgeEvent>,
    interface_name: &str,
    connection_id: u64,
) -> bool {
    for action in actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                if let Err(e) = write_length_prefixed(send_stream, data).await {
                    let _ = bridge_tx
                        .send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.to_string(),
                            reason: format!("write error: {e}"),
                            connection_id,
                        })
                        .await;
                    return false;
                }
            }
            TunnelAction::ReticulumReceived { packet } => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::ReticulumReceived {
                        interface_name: interface_name.to_string(),
                        packet: packet.clone(),
                    })
                    .await;
            }
            TunnelAction::ZenohReceived { message } => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::ZenohReceived {
                        interface_name: interface_name.to_string(),
                        message: message.clone(),
                    })
                    .await;
            }
            TunnelAction::HandshakeComplete {
                peer_dsa_pubkey,
                peer_node_id,
            } => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::HandshakeComplete {
                        interface_name: interface_name.to_string(),
                        peer_node_id: *peer_node_id,
                        peer_dsa_pubkey: peer_dsa_pubkey.clone(),
                        connection_id,
                    })
                    .await;
            }
            TunnelAction::Error { reason } => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::TunnelClosed {
                        interface_name: interface_name.to_string(),
                        reason: reason.clone(),
                        connection_id,
                    })
                    .await;
                return false;
            }
            TunnelAction::Closed => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::TunnelClosed {
                        interface_name: interface_name.to_string(),
                        reason: "session closed".to_string(),
                        connection_id,
                    })
                    .await;
                return false;
            }
        }
    }
    true
}

// -- Wire helpers -----------------------------------------------------------

/// Write a length-prefixed message: [4 bytes big-endian length][payload].
async fn write_length_prefixed(
    stream: &mut SendStream,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let len = (data.len() as u32).to_be_bytes();
    stream.write_all(&len).await?;
    stream.write_all(data).await?;
    Ok(())
}

/// Maximum message size during handshake (pre-authentication).
///
/// TunnelInit is ~6381 bytes (1088 CT + 1952 DSA pk + 32 nonce + 3309 sig).
/// TunnelAccept is ~5293 bytes (1952 DSA pk + 32 nonce + 3309 sig).
/// 8 KiB gives comfortable headroom. This caps pre-auth allocation to
/// MAX_TUNNEL_CONNECTIONS × 8 KiB = 512 KiB total.
const HANDSHAKE_MAX_MESSAGE: usize = 8 * 1024;

/// Maximum message size during the authenticated data phase.
/// Reticulum MTU is 500–1024 bytes; the higher cap here covers Zenoh
/// content-item frames routed over the tunnel (Bead #3). Capped at 2 MiB
/// to limit per-peer allocation (64 peers × 2 MiB = 128 MiB worst case).
const DATA_MAX_MESSAGE: usize = 2 * 1024 * 1024;

/// Read a length-prefixed message: [4 bytes big-endian length][payload].
///
/// `max_bytes` caps the allocation to prevent unauthenticated peers from
/// triggering large allocations during the handshake phase.
async fn read_length_prefixed(
    stream: &mut RecvStream,
    max_bytes: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).await?;
    let len = u32::from_be_bytes(len_buf) as usize;

    if len > max_bytes {
        return Err(format!("message too large: {len} bytes (max {max_bytes})").into());
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

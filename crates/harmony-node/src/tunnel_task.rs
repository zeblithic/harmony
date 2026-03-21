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
) {
    let interface_name = match conn.remote_node_id() {
        Ok(node_id) => format!("tunnel-{}", hex::encode(&node_id.as_bytes()[..4])),
        Err(_) => format!("tunnel-{:08x}", rand::random::<u32>()),
    };

    let now_ms = millis_since_start();

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
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("handshake init failed: {e}"),
                })
                .await;
            return;
        }
    };

    // Open a bidirectional stream for the tunnel
    let (mut send_stream, mut recv_stream) = match conn.open_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("failed to open bi stream: {e}"),
                })
                .await;
            return;
        }
    };

    // Send TunnelInit (OutboundBytes from init_actions)
    for action in init_actions {
        if let TunnelAction::OutboundBytes { data } = action {
            if let Err(e) = write_length_prefixed(&mut send_stream, &data).await {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::TunnelClosed {
                        interface_name,
                        reason: format!("failed to send TunnelInit: {e}"),
                    })
                    .await;
                return;
            }
        }
    }

    // Read TunnelAccept
    let accept_bytes = match read_length_prefixed(&mut recv_stream).await {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("failed to read TunnelAccept: {e}"),
                })
                .await;
            return;
        }
    };

    // Process TunnelAccept through the state machine
    let now_ms = millis_since_start();
    let actions = match session.handle_event(TunnelEvent::InboundBytes {
        data: accept_bytes,
        now_ms,
    }) {
        Ok(actions) => actions,
        Err(e) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("handshake failed: {e}"),
                })
                .await;
            return;
        }
    };

    // Dispatch handshake completion actions (HandshakeComplete + any OutboundBytes)
    if !dispatch_tunnel_actions(&actions, &mut send_stream, &bridge_tx, &interface_name).await {
        return;
    }

    // Enter main loop
    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        bridge_tx,
        cmd_rx,
        interface_name,
    )
    .await;
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
    let interface_name = match conn.remote_node_id() {
        Ok(node_id) => format!("tunnel-{}", hex::encode(&node_id.as_bytes()[..4])),
        Err(_) => format!("tunnel-{:08x}", rand::random::<u32>()),
    };

    // Accept the bidirectional stream
    let (mut send_stream, mut recv_stream) = match conn.accept_bi().await {
        Ok(streams) => streams,
        Err(e) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("failed to accept bi stream: {e}"),
                })
                .await;
            return;
        }
    };

    // Read TunnelInit
    let init_bytes = match read_length_prefixed(&mut recv_stream).await {
        Ok(bytes) => bytes,
        Err(e) => {
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("failed to read TunnelInit: {e}"),
                })
                .await;
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
            let _ = bridge_tx
                .send(TunnelBridgeEvent::TunnelClosed {
                    interface_name,
                    reason: format!("responder handshake failed: {e}"),
                })
                .await;
            return;
        }
    };

    // Dispatch accept actions (sends TunnelAccept + emits HandshakeComplete)
    if !dispatch_tunnel_actions(
        &accept_actions,
        &mut send_stream,
        &bridge_tx,
        &interface_name,
    )
    .await
    {
        return;
    }

    // Enter main loop
    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        bridge_tx,
        cmd_rx,
        interface_name,
    )
    .await;
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
///
/// Returns `false` if the tunnel should be closed (Error or Closed action
/// received), `true` to continue.
async fn dispatch_tunnel_actions(
    actions: &[TunnelAction],
    send_stream: &mut SendStream,
    bridge_tx: &mpsc::Sender<TunnelBridgeEvent>,
    interface_name: &str,
) -> bool {
    for action in actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                if let Err(e) = write_length_prefixed(send_stream, data).await {
                    let _ = bridge_tx
                        .send(TunnelBridgeEvent::TunnelClosed {
                            interface_name: interface_name.to_string(),
                            reason: format!("write error: {e}"),
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
                    })
                    .await;
            }
            TunnelAction::Error { reason } => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::TunnelClosed {
                        interface_name: interface_name.to_string(),
                        reason: reason.clone(),
                    })
                    .await;
                return false;
            }
            TunnelAction::Closed => {
                let _ = bridge_tx
                    .send(TunnelBridgeEvent::TunnelClosed {
                        interface_name: interface_name.to_string(),
                        reason: "session closed".to_string(),
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

/// Read a length-prefixed message: [4 bytes big-endian length][payload].
async fn read_length_prefixed(
    stream: &mut RecvStream,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
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

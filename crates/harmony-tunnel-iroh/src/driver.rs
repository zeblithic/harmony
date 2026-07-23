//! ZEB-473 (DM-over-iroh, Move 1a): per-connection async driver bridging an
//! iroh QUIC bi-stream to a `harmony_tunnel::TunnelSession`.
//!
//! One task runs per active tunnel connection (inbound responder or outbound
//! initiator). It owns the sans-I/O `TunnelSession` state machine and drives it
//! by:
//!   1. reading length-prefixed frames off the iroh `RecvStream` →
//!      `TunnelEvent::InboundBytes`,
//!   2. servicing a command channel (`TunnelCommand::SendDm` / `Close`) →
//!      `TunnelEvent::SendDm` / `Close`,
//!   3. a ~10s keepalive `interval` → `TunnelEvent::Tick`.
//!
//! Every batch of `TunnelAction`s is applied through a two-pass dispatch:
//! ALL `OutboundBytes` are written to the bi-stream FIRST (so e.g. a
//! `TunnelAccept` is on the wire before `HandshakeComplete` registers the
//! session), THEN `DmReceived` is pushed onto the ingest seam,
//! `HandshakeComplete` flips the manager handle to Active + flushes the pending
//! queue, and `Closed`/`Error` exit the loop.
//!
//! Provenance: this driver originated in harmony-node
//! (`crates/harmony-node/src/tunnel_task.rs`), was adapted in harmony-client for
//! the client's persistent-endpoint DM path, and is now extracted to this core
//! crate (ZEB-739) as the shared home for both. The deliberate divergence from
//! the original node template: the initiator dials over a PERSISTENT
//! [`IrohEndpoint`] (`IrohEndpoint::inner()`), not an ephemeral one, so the peer
//! can dial us back by our stable `EndpointId` and the collision-dedup in
//! [`TunnelManager`] can converge on a single survivor.

use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use harmony_identity::{PqIdentity, PqPrivateIdentity};
use harmony_iroh::endpoint::IrohEndpoint;
use harmony_tunnel::session::{TunnelSession, TunnelState};
use harmony_tunnel::{TunnelAction, TunnelEvent};
use iroh::endpoint::{Connection, RecvStream, SendStream};
use tokio::sync::mpsc;
use tokio_util::codec::{FramedRead, LengthDelimitedCodec};

use crate::framing::{read_len_prefixed, write_len_prefixed, Endian, FramingError};
use crate::manager::{HandshakeOutcome, InboundDm, TunnelCommand, TunnelManager};
use crate::peer::TunnelPeer;
use crate::versioning::{
    alpn, check_hello_compatible, decode_hello, encode_hello, TunnelHello, TUNNEL_HELLO_MAX,
};

/// Maximum time allowed for the handshake phase (stream open/accept +
/// length-prefixed `TunnelInit`/`TunnelAccept` exchange + state-machine
/// creation). The main loop has its own keepalive dead-peer timeout and does
/// not need this.
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(15);

/// Maximum message size during the pre-authentication handshake.
///
/// `TunnelInit` is ~6381 bytes (1088 CT + 1952 DSA pk + 32 nonce + 3309 sig);
/// `TunnelAccept` is ~5293 bytes. 8 KiB gives comfortable headroom and caps
/// pre-auth allocation per connection.
const HANDSHAKE_MAX_MESSAGE: usize = 8 * 1024;

/// Maximum message size during the authenticated data phase. A DM packet is a
/// sealed+signed CidNotify (single-digit KB); 2 MiB is a generous cap that
/// bounds per-peer allocation against a misbehaving authenticated peer.
const DATA_MAX_MESSAGE: usize = 2 * 1024 * 1024;

/// Keepalive tick cadence. The `TunnelSession` state machine decides when to
/// actually emit a keepalive frame (jittered 25–35s) and when to declare a dead
/// peer (110s); the 10s tick just gives it responsive timeout detection.
const KEEPALIVE_TICK: Duration = Duration::from_secs(10);

/// Why a tunnel handshake failed (ZEB-623). Splitting the failure lets the
/// caller record a *protocol-incompatible* peer loudly via the
/// [`CompatSink`](crate::manager::CompatSink) — surfaced in the app's network
/// health view — while every pre-existing dial/stream/crypto failure stays an
/// opaque `Other` that just drops the attempt (the always-deposit rung covers
/// durability).
enum HandshakeFailure {
    /// The peer's tunnel hello advertised a `protocol_version` below our
    /// minimum. BOTH sides record `reason` via the [`CompatSink`], keyed by the
    /// peer's authenticated iroh EndpointId: the initiator knows it up front
    /// (`addr.id`); the responder reads it from `conn.remote_id()`. iroh
    /// authenticates the remote endpoint key in its TLS handshake, so this id is
    /// trustworthy even before the PQ tunnel handshake completes — unlike the
    /// tunnel NodeId (`blake3(ML-DSA pubkey)`), which is not authenticated until
    /// then.
    ///
    /// [`CompatSink`]: crate::manager::CompatSink
    Incompatible { reason: String },
    /// Any other handshake failure (connect, stream, decode, crypto, timeout).
    Other(String),
}

impl std::fmt::Display for HandshakeFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HandshakeFailure::Incompatible { reason } => {
                write!(f, "incompatible protocol: {reason}")
            }
            HandshakeFailure::Other(reason) => write!(f, "{reason}"),
        }
    }
}

/// Run the **responder** side of an inbound tunnel connection.
///
/// `accept_bi()` → read `TunnelInit` → `TunnelSession::new_responder` → write
/// `TunnelAccept` → register the live session into [`TunnelManager`] (so our
/// outbound DMs to this peer reuse the bidirectional tunnel) → `run_tunnel_loop`.
pub async fn run_tunnel_responder(
    conn: Connection,
    local_pq: Arc<PqPrivateIdentity>,
    mgr: Arc<TunnelManager>,
    ingest_tx: mpsc::Sender<InboundDm>,
) {
    let handshake =
        tokio::time::timeout(HANDSHAKE_TIMEOUT, responder_handshake(&conn, &local_pq)).await;

    let (session, send_stream, recv_stream, peer_node_id) = match handshake {
        Ok(Ok(v)) => v,
        Ok(Err(failure)) => {
            // ZEB-623: record a *protocol-incompatible* inbound peer via the
            // CompatSink, keyed by the peer's iroh EndpointId. iroh authenticates
            // the remote endpoint key in its TLS handshake, so `conn.remote_id()`
            // is trustworthy even though the PQ tunnel handshake never completed.
            // ONLY the Incompatible arm records: `Other` covers transient dial/
            // stream/decode/crypto/timeout failures, which are NOT protocol
            // incompatibility, and a v1-generation inbound carries no hello so it
            // can never produce Incompatible (keeping the N-1 window unflagged).
            if let HandshakeFailure::Incompatible { reason } = &failure {
                let remote_id = conn.remote_id();
                mgr.compat().record_handshake_outcome(
                    *remote_id.as_bytes(),
                    HandshakeOutcome::Incompatible {
                        reason: reason.clone(),
                    },
                );
            }
            tracing::debug!(%failure, "ZEB-473: inbound tunnel handshake failed");
            // Explicitly close before dropping (mirrors the accept loop's
            // pre-install close) so the peer sees a prompt application close
            // rather than waiting out an idle timeout.
            conn.close(0u32.into(), b"tunnel-handshake-failed");
            return;
        }
        Err(_) => {
            tracing::debug!("ZEB-473: inbound tunnel handshake timed out");
            conn.close(0u32.into(), b"tunnel-handshake-failed");
            return;
        }
    };

    // ZEB-623 round-2: clear any stale incompatibility record for this peer — a
    // successful INBOUND handshake (over v1 or v2) proves we can now speak a
    // compatible protocol. Keyed by the peer's IROH EndpointId
    // (`conn.remote_id()`), the SAME key the failure arm above records under, so
    // we clear symmetrically; otherwise a previously-flagged peer that reconnects
    // INBOUND stays flagged until we happen to dial it outbound.
    mgr.compat().record_handshake_outcome(
        *conn.remote_id().as_bytes(),
        HandshakeOutcome::Compatible,
    );

    // Register the live responder session so our outbound DMs to this peer
    // reuse the bidirectional tunnel. The manager applies lower-NodeId collision
    // dedup: if it already holds a session for this peer, the loser is closed
    // and this `cmd_rx` may be dropped immediately — `run_tunnel_loop` then
    // exits cleanly on the first `recv()` returning `None`. The returned `epoch`
    // identifies THIS session so loop-exit evicts only our own entry (CR12).
    let (cmd_rx, epoch) = mgr.register_inbound(peer_node_id);

    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        peer_node_id,
        Arc::clone(&mgr),
        epoch,
        ingest_tx,
        cmd_rx,
    )
    .await;
}

/// Responder handshake: accept the bi-stream, (v2) read + gate the peer hello,
/// read `TunnelInit`, build the session, (v2) write our own hello, then write
/// `TunnelAccept`. ZEB-623: the generation is read from the negotiated ALPN
/// (`conn.alpn()`); a `/v1` connection carries no hello (unchanged wire
/// format). Returns the session + streams + the peer's authenticated NodeId.
async fn responder_handshake(
    conn: &Connection,
    local_pq: &PqPrivateIdentity,
) -> Result<(TunnelSession, SendStream, RecvStream, [u8; 32]), HandshakeFailure> {
    // ZEB-623: the negotiated ALPN tells us the generation.
    let is_v2 = conn.alpn() == alpn::HARMONY_TUNNEL_V2;

    let (mut send_stream, mut recv_stream) = conn
        .accept_bi()
        .await
        .map_err(|e| HandshakeFailure::Other(format!("accept_bi: {e}")))?;

    // v2: the peer's capabilities hello precedes its TunnelInit on the wire.
    if is_v2 {
        let peer_hello_bytes = read_length_prefixed(&mut recv_stream, TUNNEL_HELLO_MAX)
            .await
            .map_err(|e| HandshakeFailure::Other(format!("read hello: {e}")))?;
        let peer_hello = decode_hello(&peer_hello_bytes)
            .map_err(|e| HandshakeFailure::Other(format!("decode hello: {e}")))?;
        if let Err(reason) = check_hello_compatible(&peer_hello) {
            // The peer's *tunnel* NodeId (`blake3(ML-DSA pubkey)`) isn't
            // authenticated until the PQ handshake completes — but iroh HAS
            // authenticated the remote *endpoint* key in its TLS handshake, so the
            // caller (`run_tunnel_responder`) DOES record this incompatibility via
            // the CompatSink, keyed by `conn.remote_id()`. Here we just log LOUDLY
            // with that id and reject; the sink write happens on the returned
            // Incompatible.
            tracing::warn!(
                remote = %conn.remote_id(),
                %reason,
                "ZEB-623: inbound tunnel peer speaks an incompatible protocol hello; rejecting"
            );
            return Err(HandshakeFailure::Incompatible { reason });
        }
    }

    let init_bytes = read_length_prefixed(&mut recv_stream, HANDSHAKE_MAX_MESSAGE)
        .await
        .map_err(|e| HandshakeFailure::Other(format!("read TunnelInit: {e}")))?;

    let mut rng = rand::rngs::OsRng;
    let now_ms = millis_since_start();
    let (session, actions) = TunnelSession::new_responder(&mut rng, local_pq, &init_bytes, now_ms)
        .map_err(|e| HandshakeFailure::Other(format!("new_responder: {e}")))?;

    // v2: write our own hello BEFORE the TunnelAccept (the initiator reads the
    // peer hello first, mirroring our read order above).
    if is_v2 {
        let hello = encode_hello(&TunnelHello::current())
            .map_err(|e| HandshakeFailure::Other(format!("encode hello: {e}")))?;
        write_length_prefixed(&mut send_stream, &hello)
            .await
            .map_err(|e| HandshakeFailure::Other(format!("write hello: {e}")))?;
    }

    // Extract the authenticated peer NodeId from the HandshakeComplete action,
    // and write the TunnelAccept before returning (so the bytes are on the wire
    // before we register the session).
    let mut peer_node_id = None;
    for action in &actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                write_length_prefixed(&mut send_stream, data)
                    .await
                    .map_err(|e| HandshakeFailure::Other(format!("write TunnelAccept: {e}")))?;
            }
            TunnelAction::HandshakeComplete {
                peer_node_id: id, ..
            } => {
                peer_node_id = Some(*id);
            }
            _ => {}
        }
    }

    let peer_node_id = peer_node_id.ok_or_else(|| {
        HandshakeFailure::Other("responder handshake produced no peer NodeId".to_string())
    })?;
    Ok((session, send_stream, recv_stream, peer_node_id))
}

/// Run the **initiator** side of an outbound tunnel connection.
///
/// Dials the peer over the PERSISTENT iroh endpoint, opens a bi-stream,
/// completes the PQ handshake, then enters `run_tunnel_loop`. On a successful
/// handshake the manager handle for `peer_node_id` is flipped to Active and its
/// `pending` queue flushed (driven inside `run_tunnel_loop`'s dispatch).
#[allow(clippy::too_many_arguments)]
pub async fn run_tunnel_initiator(
    endpoint: Arc<IrohEndpoint>,
    contact: TunnelPeer,
    local_pq: Arc<PqPrivateIdentity>,
    peer_node_id: [u8; 32],
    mgr: Arc<TunnelManager>,
    epoch: u64,
    ingest_tx: mpsc::Sender<InboundDm>,
    cmd_rx: mpsc::Receiver<TunnelCommand>,
) {
    // Resolve the dial target from the peer. A malformed node id is a hard
    // failure: drop the Dialing handle so the pending DMs fall back to the
    // always-deposit durability path. The peer's PQ identity is already
    // reconstructed on the `TunnelPeer` (Seam A moved that parse to the mapping
    // boundary), so it can no longer fail here.
    let addr = match dial_addr(&contact) {
        Ok(a) => a,
        Err(reason) => {
            tracing::debug!(%reason, "ZEB-473: outbound tunnel dial-addr build failed");
            mgr.note_dial_failed(peer_node_id, epoch);
            return;
        }
    };
    let peer_pq = peer_pq_identity(&contact);
    run_tunnel_initiator_inner(
        endpoint,
        addr,
        peer_pq,
        local_pq,
        peer_node_id,
        mgr,
        epoch,
        ingest_tx,
        cmd_rx,
    )
    .await;
}

/// Inner initiator driver over a pre-resolved dial `addr` + `peer_pq` (the
/// production entry [`run_tunnel_initiator`] derives both from the peer).
/// Split out (ZEB-623) so hermetic tests can inject a connectable loopback
/// `addr` — a node-id-only `dial_addr` can't resolve over loopback in iroh
/// 1.0's discovery-only dial — and thereby exercise the real v2/v1 negotiation
/// plus the compat-sink wiring end to end.
#[allow(clippy::too_many_arguments)]
async fn run_tunnel_initiator_inner(
    endpoint: Arc<IrohEndpoint>,
    addr: iroh::EndpointAddr,
    peer_pq: PqIdentity,
    local_pq: Arc<PqPrivateIdentity>,
    peer_node_id: [u8; 32],
    mgr: Arc<TunnelManager>,
    epoch: u64,
    ingest_tx: mpsc::Sender<InboundDm>,
    cmd_rx: mpsc::Receiver<TunnelCommand>,
) {
    // ZEB-623: capture the peer's IROH EndpointId (the CompatSink key) before
    // `addr` is moved into the handshake. The sink is keyed by THIS id — not the
    // tunnel `peer_node_id` (`blake3(ML-DSA pubkey)`) — so the incompat/compat
    // records line up with the app's network-health reader.
    let iroh_join_key = *addr.id.as_bytes();
    let handshake = tokio::time::timeout(
        HANDSHAKE_TIMEOUT,
        initiator_handshake(&endpoint, addr, &peer_pq, &local_pq),
    )
    .await;

    let (session, send_stream, recv_stream) = match handshake {
        Ok(Ok(v)) => v,
        Ok(Err(HandshakeFailure::Incompatible { reason })) => {
            // ZEB-623: the peer speaks a tunnel protocol below our minimum.
            // Record it LOUDLY via the CompatSink keyed by the peer's IROH
            // EndpointId, then drop the Dialing handle so DMs fall back to
            // always-deposit durability. The dial-failure bookkeeping stays on
            // the tunnel `peer_node_id` (the session-map key); pass our epoch so
            // a newer session that replaced us isn't evicted.
            mgr.compat()
                .record_handshake_outcome(iroh_join_key, HandshakeOutcome::Incompatible { reason });
            mgr.note_dial_failed(peer_node_id, epoch);
            return;
        }
        Ok(Err(HandshakeFailure::Other(reason))) => {
            tracing::debug!(%reason, "ZEB-473: outbound tunnel handshake failed");
            mgr.note_dial_failed(peer_node_id, epoch);
            return;
        }
        Err(_) => {
            tracing::debug!("ZEB-473: outbound tunnel handshake timed out");
            mgr.note_dial_failed(peer_node_id, epoch);
            return;
        }
    };

    // Handshake reached Active. ZEB-623: clear any stale incompatibility record
    // for this peer — a successful handshake, over v1 or v2, proves we can speak
    // a compatible protocol. Keyed by the peer's IROH EndpointId. Then flip the
    // manager handle to Active (keyed by the tunnel `peer_node_id`) and flush any
    // DMs buffered while we were dialing (applies the lower-NodeId dedup if an
    // inbound session for this peer raced in).
    mgr.compat()
        .record_handshake_outcome(iroh_join_key, HandshakeOutcome::Compatible);
    mgr.note_active(peer_node_id);

    run_tunnel_loop(
        session,
        send_stream,
        recv_stream,
        peer_node_id,
        Arc::clone(&mgr),
        epoch,
        ingest_tx,
        cmd_rx,
    )
    .await;
}

/// Initiator handshake over a pre-resolved `addr`. ZEB-623: dials the newest
/// tunnel generation (`/v2`) first and falls back to `/v1` on ANY connect error
/// (a peer that only registered the `/v1` ALPN rejects the `/v2` negotiation).
/// On a `/v2` connection the first stream frame each side writes is the
/// versioned [`TunnelHello`]: the initiator pipelines `[hello][TunnelInit]`,
/// then reads the peer's `[hello][TunnelAccept]` and gates the peer hello
/// through [`check_hello_compatible`] before driving the state machine to
/// Active. `/v1` connections carry no hello (unchanged wire format). Returns the
/// active session + streams.
async fn initiator_handshake(
    endpoint: &IrohEndpoint,
    addr: iroh::EndpointAddr,
    peer_pq: &PqIdentity,
    local_pq: &PqPrivateIdentity,
) -> Result<(TunnelSession, SendStream, RecvStream), HandshakeFailure> {
    // Try v2 first; fall back to v1 on ANY connect error. `EndpointAddr` is
    // Clone, so the v2 attempt keeps a copy for the possible v1 retry.
    let (conn, gen2) = match endpoint
        .inner()
        .connect(addr.clone(), alpn::HARMONY_TUNNEL_V2)
        .await
    {
        Ok(c) => (c, true),
        Err(e2) => {
            tracing::debug!(err = %e2, "ZEB-623: tunnel v2 connect failed; falling back to v1");
            match endpoint.inner().connect(addr, alpn::HARMONY_TUNNEL_V1).await {
                Ok(c) => (c, false),
                Err(e1) => {
                    return Err(HandshakeFailure::Other(format!(
                        "connect v2: {e2}; v1: {e1}"
                    )))
                }
            }
        }
    };
    let (mut send_stream, mut recv_stream) = conn
        .open_bi()
        .await
        .map_err(|e| HandshakeFailure::Other(format!("open_bi: {e}")))?;

    // v2: our capabilities hello precedes the TunnelInit on the wire.
    if gen2 {
        let hello = encode_hello(&TunnelHello::current())
            .map_err(|e| HandshakeFailure::Other(format!("encode hello: {e}")))?;
        write_length_prefixed(&mut send_stream, &hello)
            .await
            .map_err(|e| HandshakeFailure::Other(format!("write hello: {e}")))?;
    }

    let mut rng = rand::rngs::OsRng;
    let now_ms = millis_since_start();
    let (mut session, init_actions) =
        TunnelSession::new_initiator(&mut rng, local_pq, peer_pq, now_ms)
            .map_err(|e| HandshakeFailure::Other(format!("new_initiator: {e}")))?;

    for action in init_actions {
        if let TunnelAction::OutboundBytes { data } = action {
            write_length_prefixed(&mut send_stream, &data)
                .await
                .map_err(|e| HandshakeFailure::Other(format!("write TunnelInit: {e}")))?;
        }
    }

    // v2: read + gate the peer's hello BEFORE the TunnelAccept (the responder
    // pipelines them in the same order).
    if gen2 {
        let peer_hello_bytes = read_length_prefixed(&mut recv_stream, TUNNEL_HELLO_MAX)
            .await
            .map_err(|e| HandshakeFailure::Other(format!("read hello: {e}")))?;
        let peer_hello = decode_hello(&peer_hello_bytes)
            .map_err(|e| HandshakeFailure::Other(format!("decode hello: {e}")))?;
        if let Err(reason) = check_hello_compatible(&peer_hello) {
            // ZEB-623: explicitly close before returning so the peer sees a prompt
            // application close instead of an idle timeout. The caller
            // (`run_tunnel_initiator_inner`) records the incompatibility via the
            // CompatSink keyed by the peer's iroh EndpointId.
            conn.close(0u32.into(), b"tunnel-protocol-incompatible");
            return Err(HandshakeFailure::Incompatible { reason });
        }
    }

    let accept_bytes = read_length_prefixed(&mut recv_stream, HANDSHAKE_MAX_MESSAGE)
        .await
        .map_err(|e| HandshakeFailure::Other(format!("read TunnelAccept: {e}")))?;
    let now_ms = millis_since_start();
    let actions = session
        .handle_event(TunnelEvent::InboundBytes {
            data: accept_bytes,
            now_ms,
        })
        .map_err(|e| HandshakeFailure::Other(format!("handle TunnelAccept: {e}")))?;

    // The accept-processing emits HandshakeComplete (and possibly nothing else);
    // there are no outbound bytes here. Confirm we reached Active.
    if session.state() != TunnelState::Active {
        return Err(HandshakeFailure::Other(
            "initiator did not reach Active after TunnelAccept".to_string(),
        ));
    }
    debug_assert!(actions
        .iter()
        .any(|a| matches!(a, TunnelAction::HandshakeComplete { .. })));

    Ok((session, send_stream, recv_stream))
}

/// Build the iroh `EndpointAddr` to dial from a [`TunnelPeer`].
///
/// `pub(crate)` so the driver's unit tests can assert the constructed address
/// carries the right node id + relay.
pub(crate) fn dial_addr(contact: &TunnelPeer) -> Result<iroh::EndpointAddr, String> {
    let ep_id = iroh::EndpointId::from_bytes(&contact.node_id)
        .map_err(|e| format!("tunnel endpoint id: {e}"))?;
    let mut addr = iroh::EndpointAddr::new(ep_id);
    if let Some(relay) = contact.home_relay.clone() {
        addr = addr.with_relay_url(relay);
    }
    Ok(addr)
}

/// The peer's PQ identity for the handshake. Seam A: the [`TunnelPeer`] carries
/// the already-reconstructed `PqIdentity`, so this is now an infallible clone
/// (the client rebuilt it from raw KEM+DSA pubkey bytes here — that parse moved
/// to the app's mapping boundary).
pub(crate) fn peer_pq_identity(contact: &TunnelPeer) -> PqIdentity {
    contact.pq_identity.clone()
}

/// Main read/write/keepalive loop. Runs until the tunnel closes, errors, or the
/// command channel drops.
///
/// `FramedRead` + `LengthDelimitedCodec` make the read arm cancel-safe: the
/// codec buffers partial reads internally, so dropping the read future mid-frame
/// (when `select!` fires another arm) does not discard consumed bytes — the next
/// `.next()` resumes from the buffered position.
#[allow(clippy::too_many_arguments)]
async fn run_tunnel_loop(
    mut session: TunnelSession,
    mut send_stream: SendStream,
    recv_stream: RecvStream,
    peer_node_id: [u8; 32],
    mgr: Arc<TunnelManager>,
    epoch: u64,
    ingest_tx: mpsc::Sender<InboundDm>,
    mut cmd_rx: mpsc::Receiver<TunnelCommand>,
) {
    let codec = LengthDelimitedCodec::builder()
        .length_field_length(4)
        .big_endian()
        .max_frame_length(DATA_MAX_MESSAGE)
        .new_codec();
    let mut framed = FramedRead::new(recv_stream, codec);

    let mut keepalive = tokio::time::interval(KEEPALIVE_TICK);
    keepalive.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        tokio::select! {
            frame = framed.next() => {
                match frame {
                    Some(Ok(bytes)) => {
                        let data = bytes.to_vec();
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::InboundBytes { data, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, peer_node_id, &ingest_tx,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::debug!(err = %e, "ZEB-473: tunnel inbound error; closing");
                                break;
                            }
                        }
                    }
                    Some(Err(e)) => {
                        tracing::debug!(err = %e, "ZEB-473: tunnel stream read error; closing");
                        break;
                    }
                    None => {
                        tracing::debug!("ZEB-473: tunnel stream closed by peer");
                        break;
                    }
                }
            }

            cmd = cmd_rx.recv() => {
                match cmd {
                    Some(TunnelCommand::SendDm(payload)) => {
                        let now_ms = millis_since_start();
                        match session.handle_event(TunnelEvent::SendDm { payload, now_ms }) {
                            Ok(actions) => {
                                if !dispatch_tunnel_actions(
                                    &actions, &mut send_stream, peer_node_id, &ingest_tx,
                                ).await {
                                    break;
                                }
                            }
                            Err(e) => {
                                tracing::warn!(err = %e, "ZEB-473: tunnel SendDm error");
                            }
                        }
                    }
                    Some(TunnelCommand::Close) | None => {
                        if let Ok(actions) = session.handle_event(TunnelEvent::Close) {
                            let _ = dispatch_tunnel_actions(
                                &actions, &mut send_stream, peer_node_id, &ingest_tx,
                            ).await;
                        }
                        break;
                    }
                }
            }

            _ = keepalive.tick() => {
                let now_ms = millis_since_start();
                match session.handle_event(TunnelEvent::Tick { now_ms }) {
                    Ok(actions) => {
                        if !dispatch_tunnel_actions(
                            &actions, &mut send_stream, peer_node_id, &ingest_tx,
                        ).await {
                            break;
                        }
                    }
                    Err(e) => {
                        tracing::debug!(err = %e, "ZEB-473: tunnel tick error; closing");
                        break;
                    }
                }
            }
        }
    }

    // Loop exit = session over. Best-effort: the send half is finished here; the
    // recv half drops with `framed`.
    let _ = send_stream.finish();
    // CR12: evict THIS session from the manager so dead entries don't accumulate
    // under peer churn. `note_closed` is epoch-guarded: it removes only if the
    // current entry is still this same (now-dead) session, never an ABA
    // replacement that a redial/dedup installed for the same peer.
    mgr.note_closed(peer_node_id, epoch);
}

/// Two-pass dispatch of `TunnelAction`s.
///
/// Pass 1 writes every `OutboundBytes` to the bi-stream (length-prefixed) and
/// RECORDS (without acting on) the first terminal `Error`/`Closed`. Pass 2
/// forwards EVERY `DmReceived` payload to the ingest seam — even when this
/// batch also carries a terminal action — and only THEN does the recorded
/// terminal decision take effect. Returns `true` to continue the loop, `false`
/// to exit.
///
/// G2 (ZEB-473): the terminal bail must NOT gate `DmReceived` forwarding. The
/// `TunnelSession` state machine can legitimately emit `[DmReceived(x), Error]`
/// in a single `poll` (the DM's bytes were already processed); dropping `x`
/// because an `Error` followed it in the same batch would silently lose a
/// fully-received DM. So we drain all `DmReceived` first, then honor the bail.
///
/// `TunnelSession` guarantees `OutboundBytes` precede `Error`/`Closed`, so a
/// terminal action never strands a trailing frame. An outbound WRITE error,
/// however, is itself a hard stop: there is no point forwarding further bytes
/// onto a broken stream, but we still drain any already-received DMs first.
async fn dispatch_tunnel_actions(
    actions: &[TunnelAction],
    send_stream: &mut SendStream,
    peer_node_id: [u8; 32],
    ingest_tx: &mpsc::Sender<InboundDm>,
) -> bool {
    // Pass 1: write all outbound bytes; RECORD (don't act on) terminal actions
    // and write failures so Pass 2 can still drain inbound DMs first.
    let mut terminal = false;
    for action in actions {
        match action {
            TunnelAction::OutboundBytes { data } => {
                if let Err(e) = write_length_prefixed(send_stream, data).await {
                    tracing::debug!(err = %e, "ZEB-473: tunnel write error; closing");
                    terminal = true;
                    break;
                }
            }
            TunnelAction::Error { reason } => {
                tracing::debug!(%reason, "ZEB-473: tunnel session error; closing");
                terminal = true;
            }
            TunnelAction::Closed => {
                tracing::debug!("ZEB-473: tunnel session closed");
                terminal = true;
            }
            _ => {}
        }
    }

    // Pass 2: forward DM payloads to the ingest seam — ALWAYS, even on a
    // terminal batch, so a DmReceived that arrived alongside an Error isn't
    // dropped. HandshakeComplete on the initiator/responder is already handled
    // before entering the loop (responder registers in the manager; initiator
    // reaches Active in the handshake fn); we ignore it here. Zenoh/Replication
    // frames are not expected on a DM tunnel and are dropped (logged).
    for action in actions {
        match action {
            TunnelAction::DmReceived { payload } => {
                if ingest_tx
                    .send(InboundDm {
                        peer_node_id,
                        payload: payload.clone(),
                    })
                    .await
                    .is_err()
                {
                    tracing::debug!("ZEB-473: DM ingest channel closed; dropping inbound DM");
                }
            }
            TunnelAction::ZenohReceived { .. } | TunnelAction::ReplicationReceived { .. } => {
                tracing::debug!("ZEB-473: unexpected non-DM frame on DM tunnel; dropping");
            }
            _ => {}
        }
    }

    // Now honor the recorded terminal decision (after inbound DMs were drained).
    !terminal
}

// ── Wire helpers (4-byte big-endian length prefix) ──────────────────────────

/// Write a length-prefixed message: `[4 bytes big-endian length][payload]`.
///
/// Prefix and payload are written from the audited framing codec so a partial
/// write can't leave the peer's `LengthDelimitedCodec` mid-frame.
async fn write_length_prefixed(
    stream: &mut SendStream,
    data: &[u8],
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // The write side is caller-bounded (handshake messages), so cap at the
    // wire-representable max (u32::MAX): preserves the historical "always
    // write" behavior while keeping the prefix u32-safe. BE + allow-empty
    // matches this protocol's shipped wire format exactly.
    write_len_prefixed(stream, data, u32::MAX as usize, Endian::Be, true)
        .await
        .map_err(Into::into)
}

/// Read a length-prefixed message: `[4 bytes big-endian length][payload]`.
///
/// `max_bytes` caps the allocation so an unauthenticated peer can't trigger a
/// huge allocation during the handshake phase. Used only in the handshake (the
/// data phase reads through `FramedRead`).
async fn read_length_prefixed(
    stream: &mut RecvStream,
    max_bytes: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error + Send + Sync>> {
    // BE + allow-empty: this protocol accepts zero-length frames (the cap is
    // read-side only). Preserve the exact "message too large" error text.
    read_len_prefixed(stream, max_bytes, Endian::Be, true)
        .await
        .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
            match e {
                FramingError::OutOfBounds(f) => {
                    format!("message too large: {} bytes (max {})", f.len, f.max).into()
                }
                FramingError::Io(io) => Box::new(io),
            }
        })
}

/// Monotonic milliseconds since first call (process-global epoch). Used for
/// `TunnelSession` timestamps.
pub(crate) fn millis_since_start() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    let start = START.get_or_init(Instant::now);
    start.elapsed().as_millis() as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manager::node_id_from_dsa_pubkey;
    use crate::peer::TunnelPeer;
    use crate::testsupport::{
        loopback_endpoint, loopback_endpoint_v1_only, tunnel_alpns, warm_up_iroh_global_init, wrap,
        NoopCompatSink, RecordingCompatSink,
    };
    use iroh::{EndpointAddr, TransportAddr};

    fn gen_pq() -> Arc<PqPrivateIdentity> {
        Arc::new(PqPrivateIdentity::generate(&mut rand::rngs::OsRng))
    }

    /// Build a `TunnelPeer` from a real PQ identity + a real iroh EndpointId.
    fn peer_from(pq: &PqPrivateIdentity, iroh_id: [u8; 32], relay: Option<iroh::RelayUrl>) -> TunnelPeer {
        TunnelPeer {
            node_id: iroh_id,
            pq_identity: pq.public_identity().clone(),
            home_relay: relay,
        }
    }

    #[test]
    fn dial_addr_carries_node_id_and_relay() {
        // A valid EndpointId is any ed25519 public key; generate one via iroh.
        let sk = iroh::SecretKey::generate();
        let node_id = *sk.public().as_bytes();
        let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let relay: iroh::RelayUrl = "https://relay.example/".parse().expect("relay url");
        let peer = peer_from(&pq, node_id, Some(relay));
        let addr = dial_addr(&peer).expect("dial addr");
        assert_eq!(
            *addr.id.as_bytes(),
            node_id,
            "addr must carry the peer node id"
        );
        assert!(
            addr.relay_urls()
                .any(|u| u.to_string().contains("relay.example")),
            "addr must carry the peer relay url"
        );
    }

    #[test]
    fn dial_addr_tolerates_missing_relay() {
        let sk = iroh::SecretKey::generate();
        let node_id = *sk.public().as_bytes();
        let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let peer = peer_from(&pq, node_id, None);
        let addr = dial_addr(&peer).expect("dial addr");
        assert_eq!(*addr.id.as_bytes(), node_id);
        assert_eq!(addr.relay_urls().count(), 0);
    }

    #[test]
    fn peer_pq_identity_returns_the_peer_identity() {
        // Seam A: `peer_pq_identity` now clones the identity carried on the peer;
        // its address hash must match the source identity (so the initiator's
        // handshake authenticates the right peer).
        let sk = iroh::SecretKey::generate();
        let id = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let pubid = id.public_identity();
        let peer = peer_from(&id, *sk.public().as_bytes(), None);
        let rebuilt = peer_pq_identity(&peer);
        assert_eq!(
            rebuilt.address_hash, pubid.address_hash,
            "peer_pq_identity must return the peer's own identity"
        );
    }

    #[test]
    fn millis_since_start_is_monotonic() {
        let a = millis_since_start();
        let b = millis_since_start();
        assert!(b >= a);
    }

    /// Build a `TunnelManager` over a loopback endpoint with a caller-supplied
    /// compat sink. Never dials on its own.
    fn manager_with_compat(
        endpoint: iroh::endpoint::Endpoint,
        local_pq: Arc<PqPrivateIdentity>,
        ingest_tx: mpsc::Sender<InboundDm>,
        compat: Arc<dyn crate::manager::CompatSink>,
    ) -> Arc<TunnelManager> {
        Arc::new(TunnelManager::new(wrap(endpoint), local_pq, ingest_tx, compat))
    }

    // ── ZEB-473 must-have: in-process two-session round-trip (v1) ────────────
    //
    // Drives the REAL responder path (`run_tunnel_responder` over a connected
    // iroh loopback pair) against an initiator driven through the same
    // `TunnelSession` + `run_tunnel_loop` the production initiator uses. One
    // `Dm` frame round-trips each direction; we assert the responder's ingest
    // channel receives the BYTE-IDENTICAL payload, then responder→initiator.

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn in_process_handshake_round_trips_dm_both_directions() {
        warm_up_iroh_global_init().await;
        tokio::time::timeout(std::time::Duration::from_secs(30), handshake_inner())
            .await
            .expect("in-process tunnel handshake must complete within 30s");
    }

    async fn handshake_inner() {
        let initiator_pq = gen_pq();
        let responder_pq = gen_pq();

        // Loopback endpoints: ep_a dials (initiator), ep_b accepts (responder).
        let ep_a = loopback_endpoint(tunnel_alpns()).await;
        let ep_b = loopback_endpoint(tunnel_alpns()).await;
        let ep_b_addr = EndpointAddr::from_parts(
            ep_b.id(),
            ep_b.bound_sockets().into_iter().map(TransportAddr::Ip),
        );

        // Responder-side TunnelManager + ingest channel.
        let (resp_ingest_tx, mut resp_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let resp_mgr = manager_with_compat(
            ep_b.clone(),
            Arc::clone(&responder_pq),
            resp_ingest_tx.clone(),
            Arc::new(NoopCompatSink),
        );

        // Spawn the production responder driver on the accepted connection.
        let resp_pq = Arc::clone(&responder_pq);
        let resp_mgr_task = Arc::clone(&resp_mgr);
        let resp_ingest_task = resp_ingest_tx.clone();
        let ep_b_accept = ep_b.clone();
        let responder = tokio::spawn(async move {
            let incoming = ep_b_accept
                .accept()
                .await
                .expect("incoming")
                .await
                .expect("connection established");
            run_tunnel_responder(incoming, resp_pq, resp_mgr_task, resp_ingest_task).await;
        });

        // Initiator side: dial v1 explicitly, handshake, then run the production
        // loop. Dial the loopback EndpointAddr directly (the `dial_addr` helper
        // omits direct IPs, which loopback-with-relays-disabled needs).
        let conn = ep_a
            .connect(ep_b_addr, alpn::HARMONY_TUNNEL_V1)
            .await
            .expect("connect");
        let (mut send_stream, mut recv_stream) = conn.open_bi().await.expect("open_bi");

        let mut rng = rand::rngs::OsRng;
        let (mut init_session, init_actions) = TunnelSession::new_initiator(
            &mut rng,
            &initiator_pq,
            responder_pq.public_identity(),
            millis_since_start(),
        )
        .expect("new_initiator");
        for action in init_actions {
            if let TunnelAction::OutboundBytes { data } = action {
                write_length_prefixed(&mut send_stream, &data)
                    .await
                    .expect("write TunnelInit");
            }
        }
        let accept_bytes = read_length_prefixed(&mut recv_stream, HANDSHAKE_MAX_MESSAGE)
            .await
            .expect("read TunnelAccept");
        init_session
            .handle_event(TunnelEvent::InboundBytes {
                data: accept_bytes,
                now_ms: millis_since_start(),
            })
            .expect("handle TunnelAccept");
        assert_eq!(init_session.state(), TunnelState::Active);

        let init_node_id =
            node_id_from_dsa_pubkey(&initiator_pq.public_identity().verifying_key.as_bytes());

        // Drive the initiator loop via a command channel + its own ingest seam.
        let (init_cmd_tx, init_cmd_rx) = mpsc::channel::<TunnelCommand>(8);
        let (init_ingest_tx, mut init_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let resp_node_id =
            node_id_from_dsa_pubkey(&responder_pq.public_identity().verifying_key.as_bytes());
        let init_mgr = manager_with_compat(
            ep_a.clone(),
            Arc::clone(&initiator_pq),
            init_ingest_tx.clone(),
            Arc::new(NoopCompatSink),
        );
        let initiator_loop = tokio::spawn(async move {
            run_tunnel_loop(
                init_session,
                send_stream,
                recv_stream,
                resp_node_id,
                init_mgr,
                0,
                init_ingest_tx,
                init_cmd_rx,
            )
            .await;
        });

        // (1) initiator → responder: send a Dm; assert byte-identical receipt.
        let dm_payload = b"sealed+signed-dm-bytes-A".to_vec();
        init_cmd_tx
            .send(TunnelCommand::SendDm(dm_payload.clone()))
            .await
            .expect("queue SendDm");
        let received =
            tokio::time::timeout(std::time::Duration::from_secs(10), resp_ingest_rx.recv())
                .await
                .expect("responder ingest within 10s")
                .expect("responder ingest payload");
        assert_eq!(
            received.payload, dm_payload,
            "responder must receive the byte-identical DM payload"
        );
        assert_eq!(
            received.peer_node_id, init_node_id,
            "responder must attribute the DM to the initiator's NodeId"
        );

        // (2) responder → initiator: the responder loop registered the session in
        // `resp_mgr` under the initiator's NodeId; send a DM back over it.
        let dm_back = b"sealed+signed-dm-bytes-B".to_vec();
        resp_mgr.test_send_over_handle(init_node_id, dm_back.clone());
        let back = tokio::time::timeout(std::time::Duration::from_secs(10), init_ingest_rx.recv())
            .await
            .expect("initiator ingest within 10s")
            .expect("initiator ingest payload");
        assert_eq!(
            back.payload, dm_back,
            "initiator must receive the byte-identical reply DM payload"
        );

        // Tear down: closing the initiator command channel ends its loop; the
        // responder loop ends when the stream closes.
        drop(init_cmd_tx);
        let _ = initiator_loop.await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), responder).await;
    }

    // ── ZEB-623 / Seam B: tunnel/v2 hello negotiation + CompatSink ───────────

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn v2_dialer_to_v2_acceptor_exchanges_hello_and_reaches_active() {
        warm_up_iroh_global_init().await;
        tokio::time::timeout(std::time::Duration::from_secs(30), v2_to_v2_inner())
            .await
            .expect("v2↔v2 hello negotiation must complete within 30s");
    }

    async fn v2_to_v2_inner() {
        let initiator_pq = gen_pq();
        let responder_pq = gen_pq();

        // Both endpoints bind v1+v2: the dialer's v2 connect succeeds.
        let ep_a = loopback_endpoint(tunnel_alpns()).await;
        let ep_b = loopback_endpoint(tunnel_alpns()).await;
        let ep_b_addr = EndpointAddr::from_parts(
            ep_b.id(),
            ep_b.bound_sockets().into_iter().map(TransportAddr::Ip),
        );

        // Responder side: pre-seed the ACCEPTOR-side sink with a STALE
        // incompatibility keyed by the dialer's iroh EndpointId, simulating a peer
        // that was previously flagged and has since upgraded. A successful INBOUND
        // handshake must clear it symmetrically (mirrors the initiator path).
        let resp_compat = Arc::new(RecordingCompatSink::default());
        resp_compat.note_incompatible_direct(*ep_a.id().as_bytes(), "stale".into());
        let (resp_ingest_tx, mut resp_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let resp_mgr = manager_with_compat(
            ep_b.clone(),
            Arc::clone(&responder_pq),
            resp_ingest_tx.clone(),
            Arc::clone(&resp_compat) as Arc<dyn crate::manager::CompatSink>,
        );
        let resp_pq = Arc::clone(&responder_pq);
        let ep_b_accept = ep_b.clone();
        let responder = tokio::spawn(async move {
            let incoming = ep_b_accept
                .accept()
                .await
                .expect("incoming")
                .await
                .expect("connection established");
            run_tunnel_responder(incoming, resp_pq, resp_mgr, resp_ingest_tx).await;
        });

        // Initiator side: a FRESH compat sink we assert on.
        let compat = Arc::new(RecordingCompatSink::default());
        let (init_ingest_tx, _init_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let init_mgr = manager_with_compat(
            ep_a.clone(),
            Arc::clone(&initiator_pq),
            init_ingest_tx.clone(),
            Arc::clone(&compat) as Arc<dyn crate::manager::CompatSink>,
        );

        let peer_node_id =
            node_id_from_dsa_pubkey(&responder_pq.public_identity().verifying_key.as_bytes());
        let init_node_id =
            node_id_from_dsa_pubkey(&initiator_pq.public_identity().verifying_key.as_bytes());
        let iroh_join_key = *ep_b_addr.id.as_bytes();
        let (init_cmd_tx, init_cmd_rx) = mpsc::channel::<TunnelCommand>(8);

        let endpoint = wrap(ep_a.clone());
        let peer_pq = responder_pq.public_identity().clone();
        let init_mgr_task = Arc::clone(&init_mgr);
        let initiator = tokio::spawn(async move {
            run_tunnel_initiator_inner(
                endpoint,
                ep_b_addr,
                peer_pq,
                initiator_pq,
                peer_node_id,
                init_mgr_task,
                0,
                init_ingest_tx,
                init_cmd_rx,
            )
            .await;
        });

        // Prove Active by round-tripping a DM initiator→responder.
        let dm_payload = b"v2-negotiated-dm".to_vec();
        init_cmd_tx
            .send(TunnelCommand::SendDm(dm_payload.clone()))
            .await
            .expect("queue SendDm");
        let received =
            tokio::time::timeout(std::time::Duration::from_secs(10), resp_ingest_rx.recv())
                .await
                .expect("responder ingest within 10s")
                .expect("responder ingest payload");
        assert_eq!(received.payload, dm_payload);
        assert_eq!(received.peer_node_id, init_node_id);

        // A compatible v2 handshake records NO incompatibility for the peer (the
        // initiator keys its clear by the peer's iroh EndpointId).
        assert!(
            compat.incompat_reason(&iroh_join_key).is_none(),
            "a compatible v2 handshake must leave the compat sink clean"
        );

        // The successful INBOUND handshake must have CLEARED the pre-seeded stale
        // incompatibility, keyed by the dialer's iroh EndpointId.
        assert!(
            resp_compat.incompat_reason(ep_a.id().as_bytes()).is_none(),
            "a successful inbound handshake must clear a stale incompatibility \
             for the peer's iroh EndpointId on the acceptor side"
        );

        drop(init_cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), initiator).await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), responder).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn incompatible_hello_is_rejected_and_recorded() {
        warm_up_iroh_global_init().await;
        tokio::time::timeout(
            std::time::Duration::from_secs(30),
            incompatible_hello_inner(),
        )
        .await
        .expect("incompatible-hello rejection must complete within 30s");
    }

    async fn incompatible_hello_inner() {
        let initiator_pq = gen_pq();
        // A valid peer identity so `new_initiator` builds a TunnelInit; the fake
        // acceptor never completes the PQ handshake (the initiator bails on the
        // low hello first).
        let responder_pq = gen_pq();

        let ep_a = loopback_endpoint(tunnel_alpns()).await;
        let ep_b = loopback_endpoint(tunnel_alpns()).await;
        let ep_b_addr = EndpointAddr::from_parts(
            ep_b.id(),
            ep_b.bound_sockets().into_iter().map(TransportAddr::Ip),
        );

        // Fake acceptor: accept the v2 connection, drain the initiator's hello,
        // then reply with a hello advertising protocol_version 0 (below MIN).
        let ep_b_accept = ep_b.clone();
        let fake_responder = tokio::spawn(async move {
            let incoming = ep_b_accept.accept().await.expect("incoming");
            let conn = incoming.await.expect("connection established");
            let (mut send, mut recv) = conn.accept_bi().await.expect("accept_bi");
            // Drain the initiator's hello frame (it pipelines [hello][TunnelInit]).
            let _ = read_length_prefixed(&mut recv, TUNNEL_HELLO_MAX).await;
            let low_hello = encode_hello(&TunnelHello {
                protocol_version: 0,
                capabilities: 0,
            })
            .expect("encode low hello");
            write_length_prefixed(&mut send, &low_hello)
                .await
                .expect("write low hello");
            // Hold the connection open until the initiator hangs up.
            tokio::time::sleep(std::time::Duration::from_secs(3)).await;
        });

        let compat = Arc::new(RecordingCompatSink::default());
        let (init_ingest_tx, _init_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let init_mgr = manager_with_compat(
            ep_a.clone(),
            Arc::clone(&initiator_pq),
            init_ingest_tx.clone(),
            Arc::clone(&compat) as Arc<dyn crate::manager::CompatSink>,
        );

        let peer_node_id =
            node_id_from_dsa_pubkey(&responder_pq.public_identity().verifying_key.as_bytes());
        let (_init_cmd_tx, init_cmd_rx) = mpsc::channel::<TunnelCommand>(8);
        let endpoint = wrap(ep_a.clone());
        let peer_pq = responder_pq.public_identity().clone();

        // The compat sink is keyed by the peer's IROH EndpointId, DISTINCT from
        // the tunnel `peer_node_id` (`blake3(ML-DSA pubkey)`). Snapshot the join
        // key before `ep_b_addr` is moved; the two ids must genuinely differ or
        // the assertion below is vacuous.
        let iroh_join_key = *ep_b_addr.id.as_bytes();
        assert_ne!(
            iroh_join_key, peer_node_id,
            "test fixture must derive distinct iroh EndpointId vs tunnel node id"
        );

        // Drive the real initiator: it must fail Incompatible and record it.
        run_tunnel_initiator_inner(
            endpoint,
            ep_b_addr,
            peer_pq,
            initiator_pq,
            peer_node_id,
            Arc::clone(&init_mgr),
            0,
            init_ingest_tx,
            init_cmd_rx,
        )
        .await;

        // The incompat record MUST land under the peer's IROH EndpointId (the
        // network-health join key), NOT the tunnel node id.
        assert!(
            compat.incompat_reason(&iroh_join_key).is_some(),
            "an incompatible peer hello must be recorded under the peer's IROH EndpointId"
        );
        assert!(
            compat.incompat_reason(&peer_node_id).is_none(),
            "the incompat record must NOT be keyed by the tunnel node id"
        );

        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), fake_responder).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn v2_dialer_falls_back_to_v1_only_acceptor() {
        warm_up_iroh_global_init().await;
        tokio::time::timeout(std::time::Duration::from_secs(30), v2_fallback_inner())
            .await
            .expect("v2→v1 fallback must complete within 30s");
    }

    async fn v2_fallback_inner() {
        let initiator_pq = gen_pq();
        let responder_pq = gen_pq();

        // Dialer binds v1+v2; acceptor binds ONLY v1 → the v2 connect fails and
        // the dialer must fall back to v1.
        let ep_a = loopback_endpoint(tunnel_alpns()).await;
        let ep_b = loopback_endpoint_v1_only().await;
        let ep_b_addr = EndpointAddr::from_parts(
            ep_b.id(),
            ep_b.bound_sockets().into_iter().map(TransportAddr::Ip),
        );

        let (resp_ingest_tx, mut resp_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let resp_mgr = manager_with_compat(
            ep_b.clone(),
            Arc::clone(&responder_pq),
            resp_ingest_tx.clone(),
            Arc::new(NoopCompatSink),
        );
        let resp_pq = Arc::clone(&responder_pq);
        let ep_b_accept = ep_b.clone();
        let responder = tokio::spawn(async move {
            // The dialer's v2 attempt lands FIRST and fails ALPN negotiation on
            // this v1-only endpoint. Mirror the production accept loop: skip the
            // failed incoming and wait for the v1 fallback dial.
            let conn = loop {
                let incoming = ep_b_accept.accept().await.expect("incoming");
                match incoming.await {
                    Ok(conn) => break conn,
                    Err(e) => {
                        tracing::debug!(err = %e, "skipping failed v2 incoming on v1-only acceptor");
                        continue;
                    }
                }
            };
            run_tunnel_responder(conn, resp_pq, resp_mgr, resp_ingest_tx).await;
        });

        let (init_ingest_tx, _init_ingest_rx) = mpsc::channel::<InboundDm>(8);
        let init_mgr = manager_with_compat(
            ep_a.clone(),
            Arc::clone(&initiator_pq),
            init_ingest_tx.clone(),
            Arc::new(NoopCompatSink),
        );

        let peer_node_id =
            node_id_from_dsa_pubkey(&responder_pq.public_identity().verifying_key.as_bytes());
        let init_node_id =
            node_id_from_dsa_pubkey(&initiator_pq.public_identity().verifying_key.as_bytes());
        let (init_cmd_tx, init_cmd_rx) = mpsc::channel::<TunnelCommand>(8);

        let endpoint = wrap(ep_a.clone());
        let peer_pq = responder_pq.public_identity().clone();
        let initiator = tokio::spawn(async move {
            run_tunnel_initiator_inner(
                endpoint,
                ep_b_addr,
                peer_pq,
                initiator_pq,
                peer_node_id,
                init_mgr,
                0,
                init_ingest_tx,
                init_cmd_rx,
            )
            .await;
        });

        // Fallback still reaches Active: a DM round-trips over the v1 tunnel.
        let dm_payload = b"v1-fallback-dm".to_vec();
        init_cmd_tx
            .send(TunnelCommand::SendDm(dm_payload.clone()))
            .await
            .expect("queue SendDm");
        let received =
            tokio::time::timeout(std::time::Duration::from_secs(10), resp_ingest_rx.recv())
                .await
                .expect("responder ingest within 10s")
                .expect("responder ingest payload");
        assert_eq!(received.payload, dm_payload);
        assert_eq!(received.peer_node_id, init_node_id);

        drop(init_cmd_tx);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), initiator).await;
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), responder).await;
    }

    /// G2 (ZEB-473): a `DmReceived` that arrives in the SAME batch as a terminal
    /// `Error` must STILL be forwarded to the ingest seam before the dispatch
    /// returns `false` (loop-exit).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dm_received_before_error_in_same_batch_is_still_forwarded() {
        warm_up_iroh_global_init().await;

        // A real loopback bi-stream gives us a live `SendStream` to satisfy the
        // signature; this test only drives the dispatch's action handling.
        let ep_a = loopback_endpoint(tunnel_alpns()).await;
        let ep_b = loopback_endpoint(tunnel_alpns()).await;
        let ep_b_addr = EndpointAddr::from_parts(
            ep_b.id(),
            ep_b.bound_sockets().into_iter().map(TransportAddr::Ip),
        );
        // Keep the responder side alive so the stream stays open.
        let ep_b_accept = ep_b.clone();
        let acceptor = tokio::spawn(async move {
            if let Some(incoming) = ep_b_accept.accept().await {
                let _conn = incoming.await;
                tokio::time::sleep(std::time::Duration::from_secs(5)).await;
            }
        });
        let conn = ep_a
            .connect(ep_b_addr, alpn::HARMONY_TUNNEL_V1)
            .await
            .expect("connect");
        let (mut send_stream, _recv_stream) = conn.open_bi().await.expect("open_bi");

        let peer_node_id = [0xAB; 32];
        let (ingest_tx, mut ingest_rx) = mpsc::channel::<InboundDm>(8);

        // The load-bearing batch: a received DM FOLLOWED by a terminal Error.
        let dm_payload = b"the-dm-that-must-not-be-dropped".to_vec();
        let actions = vec![
            TunnelAction::DmReceived {
                payload: dm_payload.clone(),
            },
            TunnelAction::Error {
                reason: "peer closed right after delivering a DM".to_string(),
            },
        ];

        let keep_going =
            dispatch_tunnel_actions(&actions, &mut send_stream, peer_node_id, &ingest_tx).await;

        assert!(
            !keep_going,
            "a terminal Error in the batch must still exit the loop"
        );
        let got = ingest_rx
            .try_recv()
            .expect("the DmReceived must reach ingest even though an Error followed it");
        assert_eq!(got.peer_node_id, peer_node_id);
        assert_eq!(got.payload, dm_payload);
        assert!(
            ingest_rx.try_recv().is_err(),
            "exactly one DM should have been forwarded"
        );

        drop(conn);
        let _ = tokio::time::timeout(std::time::Duration::from_secs(5), acceptor).await;
    }
}

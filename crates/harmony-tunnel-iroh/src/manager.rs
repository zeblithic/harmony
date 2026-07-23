//! ZEB-473 (DM-over-iroh, Move 1a): the per-peer PQ tunnel session map.
//! Ported to core (ZEB-739) from harmony-client `tunnel_manager.rs` with two
//! app-coupling seams cut — [`TunnelPeer`](crate::peer::TunnelPeer) replaces the
//! owner-state `DeviceTunnelContact`, and [`CompatSink`] replaces the concrete
//! `ProtocolCompatRegistry`.
//!
//! `TunnelManager` owns one `TunnelHandle` per peer device NodeId. It lazily
//! dials an outbound tunnel on the first DM to a peer, reuses a bidirectional
//! tunnel a peer opened to us, buffers DMs sent before a dial completes, and
//! resolves simultaneous-dial collisions deterministically (lower-NodeId
//! initiator wins, applied identically on both sides → converges on one
//! survivor).
//!
//! The NodeId key is `blake3(peer ML-DSA-65 pubkey)` (32 bytes) — the same
//! derivation `harmony_tunnel` uses internally, so the key the responder
//! authenticates (via `TunnelAction::HandshakeComplete`) matches the key the
//! dialer computes from the peer's `pq_identity`.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::sync::Mutex;

use harmony_identity::PqPrivateIdentity;
use harmony_iroh::endpoint::IrohEndpoint;
use tokio::sync::mpsc;

use crate::peer::TunnelPeer;

/// Capacity of a per-tunnel command channel. DMs are infrequent and the loop
/// drains promptly; a small buffer absorbs a burst without unbounded growth.
const CMD_CHANNEL_CAP: usize = 64;

/// Max DMs buffered while a dial is in flight, per peer. Beyond this the oldest
/// pending DM is dropped (it still went out over the always-deposit durability
/// path, so dropping the live attempt is graceful).
const MAX_PENDING_PER_PEER: usize = 64;

/// ZEB-485: how long the higher-NodeId peer waits to ACCEPT the lower peer's
/// inbound dial before dialing itself. The lower peer dials immediately, so an
/// inbound normally lands in tens of ms; this only fires when the lower peer
/// has nothing to send (so isn't dialing). Short enough for responsive
/// liveness, long enough that a normal inbound cancels it. Durability is
/// covered by the always-deposit rung during the wait.
const FALLBACK_DIAL_DELAY: std::time::Duration = std::time::Duration::from_secs(1);

/// The outcome the tunnel driver reports to a [`CompatSink`] for one peer.
///
/// Ported from the client's `HandshakeFailure`/registry split (ZEB-623): the
/// driver records exactly two things about a peer's protocol compatibility — a
/// successful (compatible) handshake, or an incompatible-hello rejection with a
/// reason. Transient dial/stream/crypto/timeout failures are NOT reported here
/// (they just drop the attempt), so they are absent from this enum by design.
#[derive(Debug, Clone)]
pub enum HandshakeOutcome {
    /// A successful handshake (over `/v1` or `/v2`) — proves we can speak a
    /// compatible protocol with this peer; clears any stale incompatibility.
    Compatible,
    /// The peer's tunnel hello advertised a `protocol_version` below our
    /// minimum. The `reason` is the human-readable incompatibility string.
    Incompatible { reason: String },
}

/// Seam B (ZEB-739): narrow sink for per-peer handshake/version outcomes,
/// replacing the app-coupled `ProtocolCompatRegistry` the client threaded in.
///
/// The tunnel driver calls exactly one method — [`Self::record_handshake_outcome`]
/// — at the two call sites where the client called
/// `ProtocolCompatRegistry::note_incompatible` / `note_compatible`: on an
/// incompatible-hello rejection (both initiator and responder) and on a
/// successful handshake (both sides clear). The `peer` key is the peer's
/// **iroh EndpointId** bytes (the key the client's Network Health surface joins
/// on) — NOT the tunnel node id (`blake3(ML-DSA pubkey)`). The app supplies an
/// impl that surfaces incompatibility (the client's registry does); harmony-node
/// can use a no-op.
pub trait CompatSink: Send + Sync {
    /// Record the compatibility `outcome` observed for the peer identified by
    /// its iroh EndpointId bytes `peer`.
    fn record_handshake_outcome(&self, peer: [u8; 32], outcome: HandshakeOutcome);
}

/// A decrypted inbound DM payload handed off to the ingest seam.
///
/// The tunnel loops push one of these per `TunnelAction::DmReceived`. The app
/// owns the receiver end of the ingest channel and runs the verify/decrypt/
/// apply/emit pipeline over it.
#[derive(Debug)]
pub struct InboundDm {
    /// The authenticated peer device NodeId the DM arrived from
    /// (`blake3(peer ML-DSA pubkey)`).
    pub peer_node_id: [u8; 32],
    /// The opaque sealed+signed DM packet bytes (the caller verifies/unseals).
    pub payload: Vec<u8>,
}

/// A command sent into a per-tunnel loop over its `cmd_tx`.
#[derive(Debug)]
pub enum TunnelCommand {
    /// Carry an opaque sealed+signed DM packet over the tunnel.
    SendDm(Vec<u8>),
    /// Gracefully close the tunnel.
    Close,
}

/// Which side opened the tunnel (load-bearing for collision dedup).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TunnelRole {
    Initiator,
    Responder,
}

/// Lifecycle state of a tunnel handle as the manager sees it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TunnelHandleState {
    /// Outbound dial in flight; DMs queue in `pending`.
    Dialing,
    /// Handshake complete; DMs go straight over `cmd_tx`.
    Active,
    /// Being torn down (dedup loser / explicit close).
    Closing,
    /// ZEB-485: higher-NodeId peer is buffering DMs and waiting to ACCEPT the
    /// lower peer's inbound dial (the single-dialer rule). A fallback timer
    /// promotes this to `Dialing` if no inbound arrives within
    /// `FALLBACK_DIAL_DELAY`. Like `Dialing`, DMs queue in `pending`.
    AwaitingInbound,
}

/// The manager's handle to one live (or dialing) tunnel.
struct TunnelHandle {
    /// Command sink driving the per-tunnel loop.
    cmd_tx: mpsc::Sender<TunnelCommand>,
    state: TunnelHandleState,
    role: TunnelRole,
    /// Monotonic session identity. A new dial/inbound-register that replaces an
    /// entry for the same peer gets a FRESH epoch, so a dead loop can evict ONLY
    /// its own session (`note_closed` matches on this), never an ABA replacement.
    epoch: u64,
    /// DMs buffered while `Dialing`; flushed in order on `Active`.
    pending: VecDeque<Vec<u8>>,
}

/// Per-peer PQ tunnel session map + lifecycle.
pub struct TunnelManager {
    sessions: Mutex<HashMap<[u8; 32], TunnelHandle>>,
    /// Monotonic source of per-session [`TunnelHandle::epoch`] values.
    next_epoch: std::sync::atomic::AtomicU64,
    endpoint: Arc<IrohEndpoint>,
    local_pq: Arc<PqPrivateIdentity>,
    ingest_tx: mpsc::Sender<InboundDm>,
    /// `blake3(our own ML-DSA pubkey)` — the LHS of the lower-NodeId dedup
    /// comparison. When we initiated and `self_node_id < peer_node_id`, our dial
    /// is the survivor; both peers apply this identical rule.
    self_node_id: [u8; 32],
    /// ZEB-623 / Seam B: per-peer protocol-compatibility sink. The initiator/
    /// responder records an incompatible peer here on a rejected tunnel hello
    /// and clears it on a later compatible handshake. Keyed by the peer's IROH
    /// EndpointId (see [`CompatSink`]).
    compat: Arc<dyn CompatSink>,
}

/// Derive the 32-byte tunnel NodeId from an ML-DSA-65 public key, matching
/// `harmony_tunnel`'s internal `blake3(dsa_pubkey)` derivation.
pub fn node_id_from_dsa_pubkey(dsa_pubkey: &[u8]) -> [u8; 32] {
    *blake3::hash(dsa_pubkey).as_bytes()
}

impl TunnelManager {
    /// Construct a manager over `endpoint` (used only for OUTBOUND dials),
    /// authenticating as `local_pq`, delivering inbound DMs to `ingest_tx`, and
    /// reporting protocol-compat outcomes to `compat`.
    pub fn new(
        endpoint: Arc<IrohEndpoint>,
        local_pq: Arc<PqPrivateIdentity>,
        ingest_tx: mpsc::Sender<InboundDm>,
        compat: Arc<dyn CompatSink>,
    ) -> Self {
        let self_node_id =
            node_id_from_dsa_pubkey(&local_pq.public_identity().verifying_key.as_bytes());
        Self {
            sessions: Mutex::new(HashMap::new()),
            next_epoch: std::sync::atomic::AtomicU64::new(0),
            endpoint,
            local_pq,
            ingest_tx,
            self_node_id,
            compat,
        }
    }

    /// Test-only constructor that pins `self_node_id` to a chosen value. The
    /// dial-vs-await election (and `keep_new`) turn on the numeric ordering of
    /// `self_node_id` against the peer's, so pinning it lets tests place
    /// themselves deterministically above/below a sentinel peer — instead of
    /// leaning on the 2^-256 improbability that a randomly hashed self collides
    /// with an all-0x00/all-0xFF peer.
    #[cfg(test)]
    pub(crate) fn new_with_self_node_id(
        endpoint: Arc<IrohEndpoint>,
        local_pq: Arc<PqPrivateIdentity>,
        ingest_tx: mpsc::Sender<InboundDm>,
        self_node_id: [u8; 32],
    ) -> Self {
        Self {
            self_node_id,
            ..Self::new(
                endpoint,
                local_pq,
                ingest_tx,
                Arc::new(crate::testsupport::NoopCompatSink),
            )
        }
    }

    /// Allocate a fresh monotonic session epoch (see [`TunnelHandle::epoch`]).
    fn alloc_epoch(&self) -> u64 {
        self.next_epoch
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Our own tunnel NodeId (`blake3(our ML-DSA pubkey)`).
    pub fn self_node_id(&self) -> [u8; 32] {
        self.self_node_id
    }

    /// Gracefully close the underlying iroh endpoint. This tears down every live
    /// tunnel and ends the inbound accept loop (its `accept()` returns `None`).
    /// Idempotent — `IrohEndpoint::shutdown` is safe to call more than once.
    pub async fn shutdown(&self) {
        self.endpoint.shutdown().await;
    }

    /// Seam B: the per-peer protocol-compatibility sink this manager reports to.
    /// The initiator/responder loops record an incompatible peer here (rejected
    /// tunnel hello) and clear it on a later compatible handshake.
    pub(crate) fn compat(&self) -> &Arc<dyn CompatSink> {
        &self.compat
    }

    /// Send (or queue) a sealed+signed DM packet to `peer_node_id` over a PQ
    /// tunnel, lazily dialing if no session exists. One call per recipient
    /// device with a [`TunnelPeer`].
    pub fn send_dm(
        self: &Arc<Self>,
        peer_node_id: [u8; 32],
        contact: &TunnelPeer,
        packet: Vec<u8>,
    ) {
        let mut sessions = self
            .sessions
            .lock()
            .expect("tunnel sessions mutex poisoned");
        match sessions.get_mut(&peer_node_id) {
            Some(handle) => match handle.state {
                TunnelHandleState::Active => {
                    // Try the live tunnel. Distinguish the two try_send errors:
                    //   * `Full` — the loop is alive but its command channel is
                    //     backpressured. KEEP the active session; do NOT redial.
                    //     Dropping this single best-effort live attempt is
                    //     acceptable (the deposit rung covers durability), and
                    //     tearing down a healthy tunnel on transient backpressure
                    //     would be far worse.
                    //   * `Closed` — the loop has gone away (rx dropped). Remove
                    //     the dead session and re-dial, SEEDING the failed packet
                    //     into the new dial's pending so it isn't lost.
                    if let Err(e) = handle.cmd_tx.try_send(TunnelCommand::SendDm(packet)) {
                        match e {
                            mpsc::error::TrySendError::Full(_) => {
                                // Live tunnel, just backpressured — keep it.
                            }
                            mpsc::error::TrySendError::Closed(TunnelCommand::SendDm(packet)) => {
                                sessions.remove(&peer_node_id);
                                drop(sessions);
                                self.dial_or_await(peer_node_id, contact, vec![packet]);
                            }
                            mpsc::error::TrySendError::Closed(_) => {
                                // G3 (ZEB-473): invariant violation — we only
                                // ever `try_send(SendDm)` here, so the inner
                                // command can only be `SendDm`. Surface it
                                // LOUDLY rather than silently dropping the
                                // packet; still redial (without a seed) rather
                                // than panic on a networking hot path if the
                                // command variant ever changes.
                                tracing::error!(
                                    "ZEB-473: tunnel send_dm hit a non-SendDm Closed variant — \
                                     bookkeeping invariant violated; redialing without seed"
                                );
                                sessions.remove(&peer_node_id);
                                drop(sessions);
                                self.dial_or_await(peer_node_id, contact, vec![]);
                            }
                        }
                    }
                }
                TunnelHandleState::Dialing | TunnelHandleState::AwaitingInbound => {
                    push_pending(&mut handle.pending, packet);
                }
                TunnelHandleState::Closing => {
                    // The current session is tearing down (dedup loser); start a
                    // fresh dial keyed under the same peer once it's gone. Drop
                    // the old entry and re-dial.
                    sessions.remove(&peer_node_id);
                    drop(sessions);
                    self.dial_or_await(peer_node_id, contact, vec![packet]);
                }
            },
            None => {
                drop(sessions);
                self.dial_or_await(peer_node_id, contact, vec![packet]);
            }
        }
    }

    /// Proactively establish a PQ tunnel to `peer_node_id` WITHOUT sending a DM.
    ///
    /// ZEB-739 task 5: harmony-node pumps no DMs of its own but pre-establishes
    /// tunnels to peers it discovers (its old forked transport dialed them
    /// directly). This is the DM-less analog of [`Self::send_dm`]: if no session
    /// (or in-flight dial) exists for the peer, it initiates one through the
    /// SAME single-dialer election ([`Self::dial_or_await`]) with an EMPTY
    /// pending queue, so the tunnel is brought up (handshake + keepalive) but
    /// carries no application frame until something is sent. Routing through the
    /// election is load-bearing: it prevents a concurrent inbound + outbound to
    /// the same peer from landing a `Responder` handle under our key while our
    /// own dial completes (which would trip `note_active`'s invariant assert).
    ///
    /// Idempotent: a no-op when a `Dialing`/`Active`/`AwaitingInbound`/`Closing`
    /// handle already exists for the peer.
    pub fn ensure_tunnel(self: &Arc<Self>, peer_node_id: [u8; 32], contact: &TunnelPeer) {
        {
            let sessions = self
                .sessions
                .lock()
                .expect("tunnel sessions mutex poisoned");
            if sessions.contains_key(&peer_node_id) {
                return;
            }
        }
        // No session yet — initiate through the single-dialer election. The
        // spawn_dial / spawn_await_inbound paths re-check under the lock for a
        // racing `register_inbound`, so this drop-then-dial can't double-install.
        self.dial_or_await(peer_node_id, contact, Vec::new());
    }

    /// ZEB-485 single-dialer gate. The LOWER NodeId is the sole dialer; the
    /// higher NodeId buffers `seed_pending` and waits to accept the lower
    /// peer's inbound dial. Routes EVERY fresh dial-initiation (initial dial +
    /// both redial paths) so a redial can't re-create the simultaneous-dial
    /// collision either.
    fn dial_or_await(
        self: &Arc<Self>,
        peer_node_id: [u8; 32],
        contact: &TunnelPeer,
        seed_pending: Vec<Vec<u8>>,
    ) {
        if self.self_node_id < peer_node_id {
            self.spawn_dial(peer_node_id, contact, seed_pending);
        } else {
            self.spawn_await_inbound(peer_node_id, contact, seed_pending);
        }
    }

    /// Insert an `AwaitingInbound` handle holding `seed_pending`, wait for the
    /// lower peer's inbound dial, and arm a fallback dial: if no inbound has
    /// registered within `FALLBACK_DIAL_DELAY`, promote to a real dial (the
    /// lower peer isn't dialing). The retained `cmd_rx` is handed to
    /// `run_tunnel_initiator` only if the fallback fires.
    fn spawn_await_inbound(
        self: &Arc<Self>,
        peer_node_id: [u8; 32],
        contact: &TunnelPeer,
        seed_pending: Vec<Vec<u8>>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
        for p in seed_pending {
            push_pending(&mut pending, p);
        }
        let epoch = self.alloc_epoch();
        {
            let mut sessions = self
                .sessions
                .lock()
                .expect("tunnel sessions mutex poisoned");
            // Double-check: an inbound dial may have raced in while we computed.
            // Redirect our seed DMs onto that survivor, like `drain_pending_into`.
            if let Some(existing) = sessions.get_mut(&peer_node_id) {
                // A live `Active` survivor takes the seeds over its cmd channel;
                // `Full` is an acceptable best-effort drop (matches `send_dm`'s
                // Active path — tearing a healthy tunnel down on transient
                // backpressure is worse), but a `Closed` survivor means its loop
                // already died — keep those packets and re-dial rather than
                // silently lose the live attempt (CodeAnt). Non-Active survivors
                // just buffer on their own pending.
                let redial_seed: Vec<Vec<u8>> = if existing.state == TunnelHandleState::Active {
                    let mut unsent = Vec::new();
                    for p in pending.drain(..) {
                        if let Err(mpsc::error::TrySendError::Closed(TunnelCommand::SendDm(p))) =
                            existing.cmd_tx.try_send(TunnelCommand::SendDm(p))
                        {
                            unsent.push(p);
                        }
                    }
                    unsent
                } else {
                    for p in pending.drain(..) {
                        push_pending(&mut existing.pending, p);
                    }
                    Vec::new()
                };
                if !redial_seed.is_empty() {
                    // Survivor's loop was dead — drop it and re-dial with the
                    // unsent seeds (the gate re-applies, so this can't recreate a
                    // simultaneous-dial collision).
                    sessions.remove(&peer_node_id);
                    drop(sessions);
                    self.dial_or_await(peer_node_id, contact, redial_seed);
                }
                return;
            }
            sessions.insert(
                peer_node_id,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::AwaitingInbound,
                    role: TunnelRole::Initiator,
                    epoch,
                    pending,
                },
            );
        }

        // Arm the fallback dial.
        let mgr = Arc::clone(self);
        let endpoint = Arc::clone(&self.endpoint);
        let local_pq = Arc::clone(&self.local_pq);
        let ingest_tx = self.ingest_tx.clone();
        let contact = contact.clone();
        tokio::spawn(async move {
            tokio::time::sleep(FALLBACK_DIAL_DELAY).await;
            // Promote to a real dial ONLY if still awaiting under our epoch (an
            // inbound that landed first replaced the handle / bumped the epoch).
            {
                let mut sessions = mgr.sessions.lock().expect("tunnel sessions mutex poisoned");
                match sessions.get_mut(&peer_node_id) {
                    Some(h)
                        if h.state == TunnelHandleState::AwaitingInbound && h.epoch == epoch =>
                    {
                        h.state = TunnelHandleState::Dialing;
                    }
                    _ => return, // inbound arrived / evicted / replaced — done.
                }
            }
            crate::driver::run_tunnel_initiator(
                endpoint,
                contact,
                local_pq,
                peer_node_id,
                mgr,
                epoch,
                ingest_tx,
                cmd_rx,
            )
            .await;
        });
    }

    /// Insert a `Dialing` handle and spawn the initiator loop. `seed_pending`
    /// pre-loads the buffered DMs (the one that triggered the dial, plus any
    /// redirected from a closed session).
    fn spawn_dial(
        self: &Arc<Self>,
        peer_node_id: [u8; 32],
        contact: &TunnelPeer,
        seed_pending: Vec<Vec<u8>>,
    ) {
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        let mut pending: VecDeque<Vec<u8>> = VecDeque::new();
        for p in seed_pending {
            push_pending(&mut pending, p);
        }
        let epoch = self.alloc_epoch();
        {
            let mut sessions = self
                .sessions
                .lock()
                .expect("tunnel sessions mutex poisoned");
            // Double-checked: a concurrent register_inbound may have raced in.
            if sessions.contains_key(&peer_node_id) {
                // Re-route the seeds through the now-existing handle instead of
                // overwriting it. Falls back to send-over-cmd_tx where possible.
                if let Some(existing) = sessions.get_mut(&peer_node_id) {
                    for p in pending.drain(..) {
                        match existing.state {
                            TunnelHandleState::Active => {
                                let _ = existing.cmd_tx.try_send(TunnelCommand::SendDm(p));
                            }
                            _ => push_pending(&mut existing.pending, p),
                        }
                    }
                }
                return;
            }
            sessions.insert(
                peer_node_id,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Dialing,
                    role: TunnelRole::Initiator,
                    epoch,
                    pending,
                },
            );
        }

        let mgr = Arc::clone(self);
        let endpoint = Arc::clone(&self.endpoint);
        let local_pq = Arc::clone(&self.local_pq);
        let ingest_tx = self.ingest_tx.clone();
        let contact = contact.clone();
        tokio::spawn(async move {
            crate::driver::run_tunnel_initiator(
                endpoint,
                contact,
                local_pq,
                peer_node_id,
                mgr,
                epoch,
                ingest_tx,
                cmd_rx,
            )
            .await;
        });
    }

    /// Called by the responder once an inbound handshake completes. Registers
    /// the (born-Active) session and returns the `cmd_rx` the loop drains.
    ///
    /// Applies lower-NodeId collision dedup: if a session already exists for
    /// this peer, the survivor is the tunnel whose INITIATOR NodeId is
    /// numerically lower. The loser's `cmd_tx`/handle is closed; any `pending`
    /// is redirected to the survivor.
    pub fn register_inbound(&self, peer_node_id: [u8; 32]) -> (mpsc::Receiver<TunnelCommand>, u64) {
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        let epoch = self.alloc_epoch();
        let new_handle = TunnelHandle {
            cmd_tx,
            state: TunnelHandleState::Active,
            role: TunnelRole::Responder,
            epoch,
            pending: VecDeque::new(),
        };

        let mut sessions = self
            .sessions
            .lock()
            .expect("tunnel sessions mutex poisoned");
        match sessions.remove(&peer_node_id) {
            None => {
                sessions.insert(peer_node_id, new_handle);
            }
            Some(existing) => {
                // Collision. The inbound session we just accepted was INITIATED
                // by the peer (peer_node_id). The existing handle's initiator is
                // ours if role==Initiator, else the peer.
                let new_initiator = peer_node_id; // responder side: peer initiated.
                let existing_initiator = match existing.role {
                    TunnelRole::Initiator => self.self_node_id,
                    TunnelRole::Responder => peer_node_id,
                };
                if keep_new(new_initiator, existing_initiator) {
                    // Keep the inbound (new) session. Redirect the loser's
                    // pending onto the survivor, close the loser.
                    let mut survivor = new_handle;
                    drain_pending_into(&existing, &mut survivor);
                    close_handle(existing);
                    sessions.insert(peer_node_id, survivor);
                } else {
                    // Keep the existing session; drop the inbound one. The
                    // responder loop whose cmd_rx we'd have returned will exit on
                    // the first recv() == None (we drop cmd_rx by not returning a
                    // matched survivor) — but we MUST still return a receiver, so
                    // return a closed one (its sender is dropped immediately).
                    sessions.insert(peer_node_id, existing);
                    // `cmd_rx` here belongs to `new_handle.cmd_tx`, already
                    // dropped (new_handle was moved into the survivor branch
                    // only on the keep_new path). On this path new_handle is
                    // dropped, so cmd_rx returns None immediately → the inbound
                    // responder loop exits, closing its connection.
                }
            }
        }
        (cmd_rx, epoch)
    }

    /// Called by a per-tunnel loop at EVERY exit path. Evicts the session for
    /// `peer_node_id` ONLY when the current entry is still THIS dead session
    /// (matched by `epoch`), so a newer session that replaced it (dedup / redial
    /// for the same peer) is never clobbered — the ABA guard. Without this, dead
    /// entries accumulate in `sessions` under peer churn (unbounded retention).
    /// CR12 (ZEB-473).
    pub fn note_closed(&self, peer_node_id: [u8; 32], epoch: u64) {
        let mut sessions = self
            .sessions
            .lock()
            .expect("tunnel sessions mutex poisoned");
        if let Some(handle) = sessions.get(&peer_node_id) {
            if handle.epoch == epoch {
                sessions.remove(&peer_node_id);
            }
        }
    }

    /// Called by the initiator loop once its handshake reaches Active. Flips the
    /// `Dialing` handle to `Active` and flushes any `pending` DMs over `cmd_tx`,
    /// in order. Applies the same lower-NodeId dedup if an inbound session for
    /// this peer raced in while we were dialing.
    pub fn note_active(&self, peer_node_id: [u8; 32]) {
        let mut sessions = self
            .sessions
            .lock()
            .expect("tunnel sessions mutex poisoned");
        let Some(handle) = sessions.get_mut(&peer_node_id) else {
            return;
        };

        // If an inbound (Responder) session raced in and currently holds the
        // slot, resolve the collision. The inbound one's initiator is the peer;
        // our completing dial's initiator is us.
        if handle.role == TunnelRole::Responder {
            let keep_ours = keep_new(self.self_node_id, peer_node_id);
            if !keep_ours {
                // The inbound responder session wins; our just-completed dial is
                // the loser. Leave the responder handle in place. Our initiator
                // loop will find no flush target and naturally idle out / the
                // caller can drop it. Nothing to do here.
                return;
            }
            // G1 (ZEB-473): "our dial wins over a responder handle in our slot"
            // is UNREACHABLE today — the initiator never calls `register_inbound`,
            // so a Responder handle never sits under our key when our own dial
            // completes. The old code silently `return`ed here, which would have
            // ORPHANED our just-completed initiator task (its loop would run with
            // no manager handle) if the invariant ever broke. Make it loud so a
            // future regression that lands a responder under our key is caught
            // immediately rather than leaking a running loop.
            unreachable!(
                "note_active: a Responder handle won dedup against our own completing dial — \
                 the initiator never registers an inbound session, so this is an invariant break"
            );
        }

        // Normal path: our own Dialing initiator handle. Flip to Active + flush.
        //
        // ZEB-482: flush WITHOUT draining `pending` (clone each packet, leave the
        // queue intact). In a simultaneous-dial collision (both peers `send_dm`
        // and dial each other at once — exactly what the DM-Space invite does, as
        // it fires the moment the friend handshake lands), an inbound responder
        // session can win the lower-NodeId dedup AFTER our dial completes. When
        // that happens, `register_inbound` redirects the loser's `pending` onto
        // the surviving session (`drain_pending_into`). If we had drained pending
        // here, the redirect would find nothing and the just-flushed packet —
        // already transmitted over the LOSING initiator connection the peer
        // discards — would be lost. Retaining pending lets the redirect re-deliver
        // it over the winning session. The DM packets (`apply_invite` idempotent
        // on `space_id`; `apply_inbox` idempotent on `(space_id, message_cid)`)
        // dedup a duplicate harmlessly, so the at-most-once-extra send is safe.
        // After Active, new sends go via `cmd_tx` directly (not `pending`), so the
        // retained queue is never re-flushed by another `note_active`.
        //
        // Greptile P2: enforce that at-most-once flush IN-FUNCTION rather than by
        // call-site discipline alone. We reach here only with role == Initiator
        // (the Responder branch above returned or panicked); a second
        // `note_active` on an already-Active initiator handle would re-clone and
        // re-send the RETAINED `pending` queue. Guard against it so a future
        // refactor that calls `note_active` twice can't silently double-send.
        if handle.state == TunnelHandleState::Active {
            return;
        }
        handle.state = TunnelHandleState::Active;
        for packet in handle.pending.iter().cloned() {
            if handle
                .cmd_tx
                .try_send(TunnelCommand::SendDm(packet))
                .is_err()
            {
                // Loop gone already; stop flushing.
                break;
            }
        }
    }

    /// Called by the initiator loop when a dial fails (handshake error/timeout).
    /// Removes our `Dialing` handle so a later DM re-dials — but ONLY when the
    /// current entry is still THIS dial (matched by `epoch`), exactly like
    /// [`note_closed`](Self::note_closed)'s ABA guard. A newer session that
    /// replaced it (dedup / redial) is never clobbered — including a fresh
    /// `AwaitingInbound` handle, which is tagged `role=Initiator, state≠Active`
    /// and so matched the old role/state heuristic, letting a stale dial task
    /// silently evict a valid new session (Greptile, ZEB-485). Pending DMs are
    /// dropped (the always-deposit path covers durability).
    pub fn note_dial_failed(&self, peer_node_id: [u8; 32], epoch: u64) {
        let mut sessions = self
            .sessions
            .lock()
            .expect("tunnel sessions mutex poisoned");
        if let Some(handle) = sessions.get(&peer_node_id) {
            if handle.epoch == epoch {
                sessions.remove(&peer_node_id);
            }
        }
    }

    #[cfg(test)]
    fn handle_snapshot(
        &self,
        peer_node_id: &[u8; 32],
    ) -> Option<(TunnelHandleState, TunnelRole, usize)> {
        let sessions = self.sessions.lock().expect("poisoned");
        sessions
            .get(peer_node_id)
            .map(|h| (h.state, h.role, h.pending.len()))
    }

    /// Test-only: send a DM packet over the live handle registered for
    /// `peer_node_id` (used by the cross-module in-process handshake test to
    /// drive the responder→initiator direction through a registered session).
    #[cfg(test)]
    pub(crate) fn test_send_over_handle(&self, peer_node_id: [u8; 32], packet: Vec<u8>) {
        let sessions = self.sessions.lock().expect("poisoned");
        if let Some(handle) = sessions.get(&peer_node_id) {
            let _ = handle.cmd_tx.try_send(TunnelCommand::SendDm(packet));
        }
    }
}

/// Lower-NodeId-wins comparison. The survivor is the tunnel whose INITIATOR
/// NodeId is numerically lower. Returns `true` when the NEW session's initiator
/// is the lower (so the new session is kept). A tie (same initiator, i.e. the
/// peer dialed us twice) keeps the new one.
fn keep_new(new_initiator: [u8; 32], existing_initiator: [u8; 32]) -> bool {
    new_initiator <= existing_initiator
}

/// Bound the pending queue: drop-oldest past the cap.
fn push_pending(pending: &mut VecDeque<Vec<u8>>, packet: Vec<u8>) {
    if pending.len() >= MAX_PENDING_PER_PEER {
        pending.pop_front();
    }
    pending.push_back(packet);
}

/// Redirect a (loser) handle's pending DMs onto the survivor. Active survivors
/// take them straight over `cmd_tx`; otherwise they queue.
fn drain_pending_into(loser: &TunnelHandle, survivor: &mut TunnelHandle) {
    for packet in loser.pending.iter().cloned() {
        match survivor.state {
            TunnelHandleState::Active => {
                let _ = survivor.cmd_tx.try_send(TunnelCommand::SendDm(packet));
            }
            _ => push_pending(&mut survivor.pending, packet),
        }
    }
}

/// Close a losing handle: best-effort `Close` command so its loop tears the
/// connection down (the cmd_tx then drops with the handle).
fn close_handle(mut handle: TunnelHandle) {
    handle.state = TunnelHandleState::Closing;
    let _ = handle.cmd_tx.try_send(TunnelCommand::Close);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testsupport::{test_manager, test_manager_with_self, test_peer};

    fn fixed_node_id(b: u8) -> [u8; 32] {
        [b; 32]
    }

    #[test]
    fn keep_new_lower_initiator_wins() {
        let low = fixed_node_id(0x01);
        let high = fixed_node_id(0xFE);
        // New session initiated by the lower NodeId is kept.
        assert!(keep_new(low, high));
        // New session initiated by the higher NodeId is dropped.
        assert!(!keep_new(high, low));
        // Tie keeps new (peer re-dialed).
        assert!(keep_new(low, low));
    }

    #[tokio::test(flavor = "current_thread")]
    async fn register_inbound_keeps_lower_initiator_on_collision() {
        let (mgr, _ingest_rx) = test_manager().await;
        let peer = fixed_node_id(0x05);

        // First inbound register: no collision, born Active/Responder.
        let _rx1 = mgr.register_inbound(peer);
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Active, TunnelRole::Responder))
        );

        // Simulate a SECOND inbound register for the same peer (peer re-dialed).
        // Tie on the initiator (both = peer) keeps the new one; still Active.
        let _rx2 = mgr.register_inbound(peer);
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Active, TunnelRole::Responder)),
            "a single survivor remains after a colliding inbound register"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn note_active_flushes_pending_in_order() {
        let (mgr, _ingest_rx) = test_manager().await;
        let peer = fixed_node_id(0x07);

        // Manually install a Dialing handle with a captured cmd_rx so we can
        // observe the flush (bypass spawn_dial's real iroh dial).
        let (cmd_tx, mut cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Dialing,
                    role: TunnelRole::Initiator,
                    epoch: 0,
                    pending: VecDeque::from(vec![
                        b"dm-1".to_vec(),
                        b"dm-2".to_vec(),
                        b"dm-3".to_vec(),
                    ]),
                },
            );
        }

        mgr.note_active(peer);

        // The handle is now Active. ZEB-482: `pending` is RETAINED (not drained)
        // after the flush so a later simultaneous-dial dedup-loss can redirect the
        // same (idempotent) DMs onto the winning session via `register_inbound`'s
        // `drain_pending_into` — see `note_active`'s flush comment. So pending
        // still holds the 3 buffered DMs.
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::Active, 3))
        );

        // The three buffered DMs flushed over cmd_tx, in order.
        let mut got = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            if let TunnelCommand::SendDm(p) = cmd {
                got.push(p);
            }
        }
        assert_eq!(
            got,
            vec![b"dm-1".to_vec(), b"dm-2".to_vec(), b"dm-3".to_vec()],
            "pending DMs must flush in FIFO order on Active"
        );

        // Greptile P2 idempotency: a SECOND note_active on the now-Active handle
        // must NOT re-flush the RETAINED pending queue (the in-function guard
        // returns early). Without the guard, the clone-flush would re-send all 3.
        mgr.note_active(peer);
        let mut extra = Vec::new();
        while let Ok(cmd) = cmd_rx.try_recv() {
            if let TunnelCommand::SendDm(p) = cmd {
                extra.push(p);
            }
        }
        assert!(
            extra.is_empty(),
            "a second note_active must not re-flush retained pending (got {extra:?})"
        );
        // Pending is still retained (the early-return doesn't touch it).
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::Active, 3))
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn send_dm_buffers_while_dialing() {
        let (mgr, _ingest_rx) = test_manager().await;
        let peer = fixed_node_id(0x09);

        // Pre-install a Dialing handle (avoid a real dial). Its cmd_rx is held so
        // try_send won't fail.
        let (cmd_tx, _cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Dialing,
                    role: TunnelRole::Initiator,
                    epoch: 0,
                    pending: VecDeque::new(),
                },
            );
        }

        let contact = test_peer(0x09);
        mgr.send_dm(peer, &contact, b"queued".to_vec());

        // A DM sent while Dialing is buffered (pending len == 1), not sent.
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::Dialing, 1))
        );
    }

    /// CodeAnt/Qodo F2 (packet loss on reconnect): when an Active handle's
    /// `cmd_tx` is CLOSED (the loop died), `send_dm` must remove the dead
    /// session and re-dial WITH the failed packet seeded into the new dial's
    /// pending — not drop it.
    #[tokio::test(flavor = "current_thread")]
    async fn send_dm_active_closed_seeds_failed_packet_into_redial() {
        // ZEB-485: self pinned to 0x80 and peer 0xFF > self, so we are the LOWER
        // NodeId and the redial takes the dial branch (not the await branch).
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0xFF);

        // Install an Active handle whose cmd_rx is dropped → try_send → Closed.
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        drop(cmd_rx);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Active,
                    role: TunnelRole::Initiator,
                    epoch: 0,
                    pending: VecDeque::new(),
                },
            );
        }

        let contact = test_peer(0xFF);
        mgr.send_dm(peer, &contact, b"must-not-lose".to_vec());

        // The dead session was replaced by a fresh Dialing handle that carries
        // the failed packet in its pending queue (background dial never
        // connects — the test contact is unreachable — so it stays buffered).
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, _)| s),
            Some(TunnelHandleState::Dialing),
            "a closed Active handle must be replaced by a fresh dial"
        );
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(_, _, p)| p),
            Some(1),
            "the failed packet must be SEEDED into the redial, not dropped"
        );
    }

    /// CodeAnt/Qodo F2 companion: when an Active handle's `cmd_tx` is FULL (the
    /// loop is alive, just backpressured), `send_dm` must KEEP the active
    /// session and NOT re-dial.
    #[tokio::test(flavor = "current_thread")]
    async fn send_dm_active_full_keeps_session_no_redial() {
        let (mgr, _ingest_rx) = test_manager().await;
        let peer = fixed_node_id(0x12);

        // Install an Active handle, then SATURATE its cmd_tx so the next
        // try_send returns Full (keep the rx alive so it's Full, not Closed).
        let (cmd_tx, _cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        for _ in 0..CMD_CHANNEL_CAP {
            cmd_tx
                .try_send(TunnelCommand::SendDm(b"fill".to_vec()))
                .expect("prime the channel to capacity");
        }
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Active,
                    role: TunnelRole::Initiator,
                    epoch: 0,
                    pending: VecDeque::new(),
                },
            );
        }

        let contact = test_peer(0x12);
        mgr.send_dm(peer, &contact, b"backpressured".to_vec());

        // Session stays Active (not torn down, not re-dialed) and its pending
        // queue is untouched (the Full attempt is dropped, not buffered/redialed).
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::Active, 0)),
            "a Full (backpressured) live tunnel must be kept, not redialed"
        );
    }

    #[test]
    fn node_id_derivation_is_blake3_of_dsa_pubkey() {
        let id = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let dsa = id.public_identity().verifying_key.as_bytes();
        let derived = node_id_from_dsa_pubkey(&dsa);
        assert_eq!(derived, *blake3::hash(&dsa).as_bytes());
    }

    /// CR12 (ZEB-473): a loop's `note_closed` evicts its OWN session but must
    /// NOT evict a replacement session installed for the same peer (the ABA
    /// case). The epoch guard is what distinguishes them.
    #[tokio::test(flavor = "current_thread")]
    async fn note_closed_evicts_own_session_but_not_a_replacement() {
        let (mgr, _ingest_rx) = test_manager().await;
        let peer = fixed_node_id(0x21);

        // Register an inbound session → epoch1. Its session is present.
        let (_rx1, epoch1) = mgr.register_inbound(peer);
        assert!(
            mgr.handle_snapshot(&peer).is_some(),
            "the registered session must be present"
        );

        // note_closed with the matching epoch evicts it.
        mgr.note_closed(peer, epoch1);
        assert!(
            mgr.handle_snapshot(&peer).is_none(),
            "a loop must evict its own (epoch-matched) session on exit"
        );

        // ABA: a NEW session for the same peer (epoch2) replaces the slot, then
        // the OLD loop (epoch1) belatedly calls note_closed. The replacement
        // must survive.
        let (_rx2, epoch2) = mgr.register_inbound(peer);
        assert_ne!(epoch1, epoch2, "a fresh session must get a fresh epoch");
        mgr.note_closed(peer, epoch1);
        assert!(
            mgr.handle_snapshot(&peer).is_some(),
            "a stale loop's note_closed must NOT evict the replacement session (ABA guard)"
        );

        // The replacement's own note_closed (epoch2) does evict it.
        mgr.note_closed(peer, epoch2);
        assert!(
            mgr.handle_snapshot(&peer).is_none(),
            "the replacement evicts itself on its own exit"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn send_dm_lower_self_dials_immediately() {
        // ZEB-485: when WE are the lower NodeId, send_dm to an unknown peer dials
        // right away (a Dialing/Initiator handle appears synchronously — the
        // spawned dial task can't run before this sync assert on current_thread).
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0xFF); // peer 0xFF > self 0x80 => we are lower.
        let contact = test_peer(0xFF);
        mgr.send_dm(peer, &contact, b"hi".to_vec());
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Dialing, TunnelRole::Initiator)),
            "lower-NodeId self must dial immediately"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn send_dm_higher_self_awaits_inbound() {
        // ZEB-485: when WE are the higher NodeId, send_dm buffers in an
        // AwaitingInbound handle instead of dialing.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00); // peer 0x00 < self 0x80 => we are higher.
        let contact = test_peer(0x00);
        mgr.send_dm(peer, &contact, b"hi".to_vec());
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::AwaitingInbound, 1)),
            "higher-NodeId self must buffer + await the inbound dial"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn await_inbound_falls_back_to_dial_after_delay() {
        // ZEB-485: if no inbound arrives, the higher peer dials after the delay.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00); // peer 0x00 < self 0x80 => we await first.
        let contact = test_peer(0x00);
        mgr.send_dm(peer, &contact, b"hi".to_vec());
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, _)| s),
            Some(TunnelHandleState::AwaitingInbound),
            "starts in AwaitingInbound"
        );

        // Before the delay elapses, still awaiting.
        tokio::time::advance(FALLBACK_DIAL_DELAY / 2).await;
        tokio::task::yield_now().await;
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, _)| s),
            Some(TunnelHandleState::AwaitingInbound),
            "still awaiting before the fallback delay"
        );

        // After the delay, the fallback fired: the handle is no longer awaiting
        // (it promoted to Dialing; the background dial to the bogus contact may
        // then fail and evict it — either way it left AwaitingInbound).
        tokio::time::advance(FALLBACK_DIAL_DELAY).await;
        tokio::task::yield_now().await;
        assert_ne!(
            mgr.handle_snapshot(&peer).map(|(s, _, _)| s),
            Some(TunnelHandleState::AwaitingInbound),
            "fallback dial must have fired after the delay"
        );
    }

    #[tokio::test(flavor = "current_thread", start_paused = true)]
    async fn await_inbound_fallback_is_noop_when_inbound_arrives_first() {
        // ZEB-485: an inbound dial that lands before the delay cancels the fallback.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00); // peer 0x00 < self 0x80 => we await.
        let contact = test_peer(0x00);
        mgr.send_dm(peer, &contact, b"hi".to_vec());

        // The lower peer's inbound dial lands: keep_new(peer<=self)=true keeps the
        // inbound Responder session and drains our buffered DM onto it.
        let _rx = mgr.register_inbound(peer);
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Active, TunnelRole::Responder)),
            "inbound replaced the awaiting handle"
        );

        // Advancing past the delay must NOT promote anything (no second dial).
        tokio::time::advance(FALLBACK_DIAL_DELAY * 2).await;
        tokio::task::yield_now().await;
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Active, TunnelRole::Responder)),
            "fallback must be a no-op once an inbound session exists"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn register_inbound_drains_awaiting_inbound_pending() {
        // ZEB-485: when the lower peer's inbound dial lands while we (higher) are
        // AwaitingInbound, register_inbound keeps the inbound and redirects our
        // buffered DMs onto it. AwaitingInbound is tagged role=Initiator, so the
        // existing keep_new/drain_pending_into path covers it with NO change.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00); // peer 0x00 <= self 0x80 => keep_new(peer, self) == true.

        // Install an AwaitingInbound handle holding two buffered DMs. Its cmd_tx is
        // dead (rx dropped) — the drain targets the SURVIVOR (the new inbound), not
        // this handle, so that is fine.
        let (cmd_tx, _dead_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::AwaitingInbound,
                    role: TunnelRole::Initiator,
                    epoch: 0,
                    pending: VecDeque::from(vec![b"p1".to_vec(), b"p2".to_vec()]),
                },
            );
        }

        let (mut cmd_rx, _epoch) = mgr.register_inbound(peer);
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, r, _)| (s, r)),
            Some((TunnelHandleState::Active, TunnelRole::Responder)),
            "the inbound survivor replaces the awaiting handle"
        );

        let mut drained = Vec::new();
        while let Ok(TunnelCommand::SendDm(p)) = cmd_rx.try_recv() {
            drained.push(p);
        }
        assert_eq!(
            drained,
            vec![b"p1".to_vec(), b"p2".to_vec()],
            "the awaiting handle's pending DMs are redirected onto the inbound survivor"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn await_inbound_double_check_redials_when_racing_survivor_is_closed() {
        // ZEB-485 (CodeAnt): if an inbound dial raced in but its loop is already
        // Closed (rx dropped), spawn_await_inbound's double-check must NOT
        // silently drop the seeded DM — it re-dials, seeding the unsent packet.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00); // higher self => the await branch.

        // Pre-insert a DEAD Active handle (cmd_rx dropped => try_send => Closed).
        let (cmd_tx, cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        drop(cmd_rx);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::Active,
                    role: TunnelRole::Responder,
                    epoch: 0,
                    pending: VecDeque::new(),
                },
            );
        }

        let contact = test_peer(0x00);
        // Drive the double-check directly with a seed.
        mgr.spawn_await_inbound(peer, &contact, vec![b"seed".to_vec()]);

        // The dead survivor was evicted and the seed re-dialed: we are higher,
        // so the re-dial routes back to a fresh AwaitingInbound still carrying it.
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, p)| (s, p)),
            Some((TunnelHandleState::AwaitingInbound, 1)),
            "a Closed racing survivor must re-dial+seed, not silently drop"
        );
    }

    #[tokio::test(flavor = "current_thread")]
    async fn note_dial_failed_only_evicts_its_own_epoch() {
        // ZEB-485 (Greptile): a STALE dial task's note_dial_failed must not evict a
        // newer session that replaced it for the same peer — including a fresh
        // AwaitingInbound (role=Initiator, state!=Active, which matched the old
        // role/state heuristic). Same ABA guard as note_closed.
        let (mgr, _ingest_rx) = test_manager_with_self(fixed_node_id(0x80)).await;
        let peer = fixed_node_id(0x00);

        // The "new" valid session: a fresh AwaitingInbound at epoch 7.
        let (cmd_tx, _cmd_rx) = mpsc::channel(CMD_CHANNEL_CAP);
        {
            let mut sessions = mgr.sessions.lock().unwrap();
            sessions.insert(
                peer,
                TunnelHandle {
                    cmd_tx,
                    state: TunnelHandleState::AwaitingInbound,
                    role: TunnelRole::Initiator,
                    epoch: 7,
                    pending: VecDeque::new(),
                },
            );
        }

        // A stale dial task (epoch 3) reports failure — must be a no-op.
        mgr.note_dial_failed(peer, 3);
        assert_eq!(
            mgr.handle_snapshot(&peer).map(|(s, _, _)| s),
            Some(TunnelHandleState::AwaitingInbound),
            "stale-epoch note_dial_failed must not evict the newer session"
        );

        // The matching-epoch dial failure DOES evict its own handle.
        mgr.note_dial_failed(peer, 7);
        assert!(
            mgr.handle_snapshot(&peer).is_none(),
            "matching-epoch note_dial_failed evicts its own handle"
        );
    }

    /// Focused Seam-B unit test: the `CompatSink` records an incompatible peer
    /// and then clears it on a subsequent compatible outcome (the exact two
    /// transitions the driver drives).
    #[test]
    fn compat_sink_records_then_clears_incompatibility() {
        let sink = crate::testsupport::RecordingCompatSink::default();
        let peer = fixed_node_id(0x33);
        assert_eq!(sink.incompat_reason(&peer), None);

        sink.record_handshake_outcome(
            peer,
            HandshakeOutcome::Incompatible {
                reason: "tunnel hello v0 < min 1".into(),
            },
        );
        assert_eq!(
            sink.incompat_reason(&peer).as_deref(),
            Some("tunnel hello v0 < min 1"),
            "an Incompatible outcome must be recorded with its reason"
        );

        sink.record_handshake_outcome(peer, HandshakeOutcome::Compatible);
        assert_eq!(
            sink.incompat_reason(&peer),
            None,
            "a Compatible outcome must clear the recorded incompatibility"
        );
    }
}

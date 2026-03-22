use alloc::{boxed::Box, string::String, sync::Arc, vec, vec::Vec};
use core::fmt;
#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

use harmony_crypto::hash;
use harmony_identity::identity::PrivateIdentity;
use rand_core::CryptoRngCore;

use crate::announce::{build_announce, validate_announce, ValidatedAnnounce};
use crate::context::PacketContext;
use crate::cooperation::CooperationTable;
use crate::destination::DestinationName;
use crate::ifac::IfacAuthenticator;
use crate::interface::InterfaceMode;
use crate::link;
use crate::packet::{
    DestinationType, HeaderType, Packet, PacketFlags, PacketHeader, PacketType, PropagationType,
};
use crate::packet_hashlist::PacketHashlist;
use crate::path_table::{DestinationHash, PathTable, PathUpdateResult};

// ── Constants (transport-mode announce propagation) ──────────────────────

/// Maximum retransmit retries for announce propagation (Python: `PATHFINDER_R`).
const PATHFINDER_R: u8 = 1;

/// Base retry delay in seconds for announce retransmits (Python: `PATHFINDER_G`).
const PATHFINDER_G: u64 = 5;

/// Maximum local rebroadcast echoes before considering propagation done.
const LOCAL_REBROADCASTS_MAX: u8 = 2;

/// Announce rate table entries unseen for this many seconds are evicted.
/// Matches the shortest path expiry (Roaming = 6 hours).
const RATE_TABLE_EXPIRY: u64 = 21_600;

/// Milliseconds to wait for an announce echo before recording a negative
/// cooperation observation. Conservative at 90s (3x a typical 30s interval).
/// Announce echo timeout in seconds. If we don't hear our own announce
/// echo back within this window, record a negative observation.
/// 90 seconds ≈ 3× a typical 30-second announce interval.
const ECHO_TIMEOUT: u64 = 90;

/// Reverse table entries older than this are expired (8 minutes).
/// Matches Python `Transport.REVERSE_TIMEOUT`.
const REVERSE_TIMEOUT: u64 = 480;

/// Validated link table entries expire after 10 minutes of inactivity.
/// Matches Python `Transport.STALE_TIME * 1.25`.
const LINK_TIMEOUT: u64 = 600;

/// Seconds per hop to allow for link proof arrival (unvalidated entries).
const ESTABLISHMENT_TIMEOUT_PER_HOP: u64 = 6;

// ── Types ───────────────────────────────────────────────────────────────

/// Per-interface announce rate limiting configuration.
#[derive(Debug, Clone, Copy)]
pub struct AnnounceRateConfig {
    /// Minimum seconds between announces for the same destination.
    pub target: u64,
    /// Number of violations allowed before blocking.
    pub grace: u32,
    /// Additional seconds added to target when blocked.
    pub penalty: u64,
}

/// Configuration for a registered interface.
pub struct InterfaceConfig {
    pub mode: InterfaceMode,
    pub ifac: Option<IfacAuthenticator>,
    pub announce_rate: Option<AnnounceRateConfig>,
}

impl fmt::Debug for InterfaceConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("InterfaceConfig")
            .field("mode", &self.mode)
            .field("ifac", &self.ifac.is_some())
            .field("announce_rate", &self.announce_rate.is_some())
            .finish()
    }
}

/// A locally owned destination that can generate announces.
struct AnnouncingDestination {
    identity: PrivateIdentity,
    name: DestinationName,
    app_data: Vec<u8>,
    announce_interval: Option<u64>,
    next_announce_at: u64,
}

impl fmt::Debug for AnnouncingDestination {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AnnouncingDestination")
            .field("name", &self.name)
            .field("app_data_len", &self.app_data.len())
            .field("announce_interval", &self.announce_interval)
            .field("next_announce_at", &self.next_announce_at)
            .finish()
    }
}

/// Per-destination rate limiting state for announce propagation.
struct AnnounceRateEntry {
    last_checked: u64,
    rate_violations: u32,
    blocked_until: u64,
}

/// An announce queued for transport retransmission.
struct AnnounceTableEntry {
    #[allow(dead_code)] // Retained for diagnostics
    received_at: u64,
    retransmit_at: u64,
    retries: u8,
    local_rebroadcasts: u8,
    source_interface: Arc<str>,
    hops: u8,
    destination_hash: DestinationHash,
    packet_data: Arc<[u8]>,
}

/// A reverse table entry mapping a relayed packet hash back to the interface
/// it arrived on. Used for routing proofs back along the original data path.
struct ReverseTableEntry {
    /// Interface the data packet was received on (proof goes back here).
    received_interface: Arc<str>,
    /// Interface the data packet was forwarded to.
    outbound_interface: Arc<str>,
    /// Monotonic timestamp when this entry was created.
    timestamp: u64,
    /// Whether a proof was received for this relayed packet.
    proof_received: bool,
}

/// A link table entry tracking a link request passing through this transport node.
/// Used for routing link proofs back and link data bidirectionally.
struct LinkTableEntry {
    /// Monotonic timestamp when created or last data activity.
    timestamp: u64,
    /// Next transport toward the destination (from path table).
    next_hop: DestinationHash,
    /// Interface to forward toward the destination.
    outbound_interface: Arc<str>,
    /// Hops from us to destination (from path table).
    remaining_hops: u8,
    /// Interface the link request arrived on.
    received_interface: Arc<str>,
    /// Packet hops at time of receipt (post-increment).
    taken_hops: u8,
    /// Original destination of the link request.
    destination_hash: DestinationHash,
    /// False until link proof received; controls expiry mode.
    validated: bool,
    /// Absolute deadline for proof arrival (unvalidated entries).
    proof_timeout: u64,
}

/// Input events from the caller.
#[derive(Debug)]
pub enum NodeEvent {
    /// An inbound packet arrived on the named interface.
    InboundPacket {
        interface_name: String,
        raw: Vec<u8>,
        now: u64,
    },
    /// Periodic timer tick for maintenance tasks.
    TimerTick { now: u64 },
}

/// Reason a packet was dropped (diagnostic).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DropReason {
    /// Packet arrived on an unregistered interface.
    UnknownInterface,
    /// IFAC unmasking failed (wrong credentials or corrupted).
    IfacFailed,
    /// IFAC flag/config mismatch (flag set but no IFAC, or vice versa).
    IfacMismatch,
    /// Packet parsing failed.
    ParseFailed,
    /// PLAIN/GROUP packet with hops > 1 (multi-hop broadcast).
    MultiHopBroadcast,
    /// Duplicate packet detected in hashlist.
    DuplicatePacket,
    /// Announce signature or destination hash verification failed.
    AnnounceInvalid,
    /// Announce with PLAIN or GROUP destination type (invalid).
    InvalidAnnounceDestType,
    /// No local destination registered for this packet.
    NoLocalDestination,
    /// announce() called for an unregistered destination.
    UnknownDestination,
    /// build_announce() returned an error.
    AnnounceBuildFailed,
    /// packet.to_bytes() failed on outbound.
    OutboundSerializeFailed,
    /// IFAC mask() failed on outbound.
    OutboundIfacFailed,
    /// Announce dropped by per-destination rate limiter.
    AnnounceRateLimited,
    /// Announce could not be re-serialized as HEADER_2 (likely MTU overflow).
    AnnounceRebroadcastSerializeFailed,
    /// Type2 data packet addressed to us but no path to the destination.
    NoRouteForTransport,
    /// Forwarded packet failed to_bytes() (e.g., MTU exceeded).
    RelaySerializeFailed,
    /// Proof arrived but no reverse table entry found for routing.
    ProofNoReverseEntry,
    /// LinkRequest for destination not in path table.
    NoRouteForLinkRequest,
    /// LRPROOF arrived but no link table entry found.
    LinkProofNoEntry,
    /// LRPROOF hops don't match expected remaining_hops.
    LinkProofHopsMismatch,
    /// LRPROOF arrived on wrong interface.
    LinkProofWrongInterface,
}

/// Output actions for the caller.
#[derive(Debug)]
pub enum NodeAction {
    /// A valid announce was received and the path table was updated.
    AnnounceReceived {
        destination_hash: DestinationHash,
        validated_announce: Box<ValidatedAnnounce>,
        path_update: PathUpdateResult,
        interface_name: Arc<str>,
        hops: u8,
    },
    /// A packet should be delivered to a locally registered destination.
    DeliverLocally {
        destination_hash: DestinationHash,
        packet: Packet,
        interface_name: Arc<str>,
    },
    /// Expired paths were removed from the path table.
    PathsExpired { count: usize },
    /// A packet was dropped for the given reason.
    PacketDropped {
        reason: DropReason,
        interface_name: Arc<str>,
    },
    /// A destination's scheduled announce is due. Caller should call node.announce().
    AnnounceNeeded { dest_hash: DestinationHash },
    /// Send raw bytes on the named interface (outbound).
    SendOnInterface {
        interface_name: Arc<str>,
        raw: Vec<u8>,
        /// Cooperation weight for probabilistic broadcast selection.
        /// `None` = directed send (always deliver).
        /// `Some(score)` = broadcast send (caller may drop if random > score).
        weight: Option<f32>,
    },
    /// Transport node rebroadcast an announce on interfaces (excluding source).
    AnnounceRebroadcast {
        destination_hash: DestinationHash,
        hops: u8,
    },
    /// Transport node relayed a data packet toward its destination.
    PacketRelayed {
        destination_hash: DestinationHash,
        next_hop: DestinationHash,
        interface_name: Arc<str>,
    },
    /// Transport node routed a proof back toward the packet originator.
    ProofRelayed {
        proof_destination: DestinationHash,
        interface_name: Arc<str>,
    },
    /// Reverse table entries expired during timer tick.
    ReverseTableExpired { count: usize },
    /// Transport node forwarded a link request toward its destination.
    LinkRequestForwarded {
        link_id: DestinationHash,
        destination_hash: DestinationHash,
        next_hop: DestinationHash,
        interface_name: Arc<str>,
    },
    /// Transport node routed a link proof back toward the initiator.
    LinkProofRouted {
        link_id: DestinationHash,
        interface_name: Arc<str>,
    },
    /// Transport node routed a link data packet.
    LinkDataRouted {
        link_id: DestinationHash,
        interface_name: Arc<str>,
    },
    /// Link table entries expired during timer tick.
    LinkTableExpired { count: usize },
}

// ── Node ────────────────────────────────────────────────────────────────

/// The node coordinator state machine.
///
/// Ties interfaces, path table, packet hashlist, and IFAC together. In leaf
/// mode, the node receives packets, processes announces to learn paths,
/// deduplicates, and delivers packets to locally registered destinations.
/// In transport mode (created via `new_transport`), it also re-broadcasts
/// received announces as HEADER_2 packets with rate limiting.
pub struct Node {
    path_table: PathTable,
    packet_hashlist: PacketHashlist,
    interfaces: HashMap<Arc<str>, InterfaceConfig>,
    local_destinations: HashSet<DestinationHash>,
    announcing_destinations: HashMap<DestinationHash, AnnouncingDestination>,
    transport_identity: Option<PrivateIdentity>,
    announce_table: HashMap<DestinationHash, AnnounceTableEntry>,
    announce_rate_table: HashMap<DestinationHash, AnnounceRateEntry>,
    reverse_table: HashMap<DestinationHash, ReverseTableEntry>,
    link_table: HashMap<DestinationHash, LinkTableEntry>,
    cooperation: CooperationTable,
    /// Tracks announce echoes: (interface, dest_hash) -> tick when announce was sent.
    /// Used to detect whether neighbors forwarded our announces.
    pending_echoes: HashMap<(Arc<str>, DestinationHash), u64>,
}

impl Node {
    /// Create a leaf-mode node with no interfaces or destinations.
    pub fn new() -> Self {
        Self {
            path_table: PathTable::new(),
            packet_hashlist: PacketHashlist::new(),
            interfaces: HashMap::new(),
            local_destinations: HashSet::new(),
            announcing_destinations: HashMap::new(),
            transport_identity: None,
            announce_table: HashMap::new(),
            announce_rate_table: HashMap::new(),
            reverse_table: HashMap::new(),
            link_table: HashMap::new(),
            cooperation: CooperationTable::default(),
            pending_echoes: HashMap::new(),
        }
    }

    /// Create a transport-mode node. Stores identity for HEADER_2 construction.
    pub fn new_transport(identity: PrivateIdentity) -> Self {
        Self {
            path_table: PathTable::new(),
            packet_hashlist: PacketHashlist::new(),
            interfaces: HashMap::new(),
            local_destinations: HashSet::new(),
            announcing_destinations: HashMap::new(),
            transport_identity: Some(identity),
            announce_table: HashMap::new(),
            announce_rate_table: HashMap::new(),
            reverse_table: HashMap::new(),
            link_table: HashMap::new(),
            cooperation: CooperationTable::default(),
            pending_echoes: HashMap::new(),
        }
    }

    /// Whether the node is in transport mode (has a transport identity).
    pub fn is_transport(&self) -> bool {
        self.transport_identity.is_some()
    }

    /// The 16-byte transport identity hash, if in transport mode.
    pub fn transport_identity_hash(&self) -> Option<[u8; 16]> {
        self.transport_identity
            .as_ref()
            .map(|id| id.public_identity().address_hash)
    }

    /// Set rate limiting config on an existing interface. Returns false if not found.
    pub fn set_announce_rate(&mut self, name: &str, config: AnnounceRateConfig) -> bool {
        match self.interfaces.get_mut(name) {
            Some(iface) => {
                iface.announce_rate = Some(config);
                true
            }
            None => false,
        }
    }

    /// Number of entries in the announce propagation table.
    pub fn announce_table_len(&self) -> usize {
        self.announce_table.len()
    }

    /// Number of entries in the reverse table.
    pub fn reverse_table_len(&self) -> usize {
        self.reverse_table.len()
    }

    /// Number of entries in the link table.
    pub fn link_table_len(&self) -> usize {
        self.link_table.len()
    }

    /// Number of pending announce echo entries.
    pub fn pending_echoes_len(&self) -> usize {
        self.pending_echoes.len()
    }

    /// Read-only access to the cooperation table for diagnostics and testing.
    pub fn cooperation(&self) -> &CooperationTable {
        &self.cooperation
    }

    /// Register an interface by name and configuration.
    ///
    /// Takes ownership of the optional IFAC authenticator.
    pub fn register_interface(
        &mut self,
        name: String,
        mode: InterfaceMode,
        ifac: Option<IfacAuthenticator>,
    ) {
        let arc_name: Arc<str> = Arc::from(name);
        self.cooperation.register_interface(Arc::clone(&arc_name));
        self.interfaces.insert(
            arc_name,
            InterfaceConfig {
                mode,
                ifac,
                announce_rate: None,
            },
        );
    }

    /// Unregister an interface. Returns `true` if it was registered.
    pub fn unregister_interface(&mut self, name: &str) -> bool {
        self.interfaces.remove(name).is_some()
    }

    /// Register a local destination hash for delivery.
    pub fn register_destination(&mut self, dest_hash: DestinationHash) {
        self.local_destinations.insert(dest_hash);
    }

    /// Unregister a local destination. Returns `true` if it was registered.
    ///
    /// Also removes the destination from announcing destinations if present.
    pub fn unregister_destination(&mut self, dest_hash: &DestinationHash) -> bool {
        self.announcing_destinations.remove(dest_hash);
        self.local_destinations.remove(dest_hash)
    }

    /// Register a destination that can generate announces and receive packets.
    ///
    /// Takes ownership of the `PrivateIdentity`. Also registers for local delivery.
    /// Returns the computed `DestinationHash`.
    pub fn register_announcing_destination(
        &mut self,
        identity: PrivateIdentity,
        name: DestinationName,
        app_data: Vec<u8>,
        announce_interval: Option<u64>,
        now: u64,
    ) -> DestinationHash {
        let dest_hash = name.destination_hash(&identity.public_identity().address_hash);
        self.local_destinations.insert(dest_hash);
        self.announcing_destinations.insert(
            dest_hash,
            AnnouncingDestination {
                identity,
                name,
                app_data,
                announce_interval,
                next_announce_at: now,
            },
        );
        dest_hash
    }

    /// Unregister an announcing destination. Returns `true` if it existed.
    ///
    /// Also removes from local_destinations.
    pub fn unregister_announcing_destination(&mut self, dest_hash: &DestinationHash) -> bool {
        self.local_destinations.remove(dest_hash);
        self.announcing_destinations.remove(dest_hash).is_some()
    }

    /// Build and route an announce for a registered announcing destination.
    ///
    /// Two-phase design: `handle_event(TimerTick)` emits `AnnounceNeeded`, then
    /// the caller invokes this with their own RNG. This keeps the scheduler
    /// deterministic and the node fully sans-I/O.
    ///
    /// Records pending echo entries for each interface so that echoed announces
    /// from neighbors can be detected and credited via the cooperation table.
    pub fn announce(
        &mut self,
        dest_hash: &DestinationHash,
        rng: &mut impl CryptoRngCore,
        now: u64,
    ) -> Vec<NodeAction> {
        // Build announce packet — scoped borrow of announcing_destinations.
        let raw = {
            let ad = match self.announcing_destinations.get(dest_hash) {
                Some(ad) => ad,
                None => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::UnknownDestination,
                        interface_name: Arc::from(""),
                    }];
                }
            };
            let packet = match build_announce(&ad.identity, &ad.name, rng, now, &ad.app_data, None)
            {
                Ok(p) => p,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::AnnounceBuildFailed,
                        interface_name: Arc::from(""),
                    }];
                }
            };
            match packet.to_bytes() {
                Ok(bytes) => bytes,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::OutboundSerializeFailed,
                        interface_name: Arc::from(""),
                    }];
                }
            }
        };

        // Broadcast on all interfaces — separate borrow of self.interfaces.
        let actions = self.broadcast_on_all_interfaces(&raw);

        // Record pending echo entries for each interface we successfully sent on.
        // When we later hear this announce come back via a *different* interface,
        // we credit the echoing interface as a positive cooperation signal.
        let dest = *dest_hash;
        for action in &actions {
            if let NodeAction::SendOnInterface { interface_name, .. } = action {
                self.pending_echoes
                    .insert((Arc::clone(interface_name), dest), now);
            }
        }

        actions
    }

    /// Route a pre-built raw packet to the appropriate interface(s).
    ///
    /// Known path → send on the path's interface. No path → broadcast all.
    pub fn route_packet(
        &self,
        destination_hash: &DestinationHash,
        raw: Vec<u8>,
    ) -> Vec<NodeAction> {
        if let Some(entry) = self.path_table.get(destination_hash) {
            self.send_on_interface(&entry.interface_name, &raw)
        } else {
            self.broadcast_on_all_interfaces(&raw)
        }
    }

    /// Number of announcing destinations.
    pub fn announcing_destination_count(&self) -> usize {
        self.announcing_destinations.len()
    }

    /// Process an event and return any resulting actions.
    ///
    /// This is the core entry point. The caller feeds events (inbound packets,
    /// timer ticks) and receives actions (deliveries, drops, path updates).
    pub fn handle_event(&mut self, event: NodeEvent) -> Vec<NodeAction> {
        match event {
            NodeEvent::InboundPacket {
                interface_name,
                raw,
                now,
            } => self.process_inbound(interface_name, raw, now),
            NodeEvent::TimerTick { now } => {
                let mut actions = Vec::new();

                let count = self.path_table.expire(now);
                if count > 0 {
                    actions.push(NodeAction::PathsExpired { count });
                }

                for (dest_hash, ad) in &mut self.announcing_destinations {
                    if let Some(interval) = ad.announce_interval {
                        if now >= ad.next_announce_at {
                            actions.push(NodeAction::AnnounceNeeded {
                                dest_hash: *dest_hash,
                            });
                            let jitter = (dest_hash[0] as u64) % 60;
                            ad.next_announce_at = now + interval + jitter;
                        }
                    }
                }

                if self.is_transport() {
                    actions.extend(self.process_announce_table(now));
                    self.expire_rate_table(now);

                    // Expire stale reverse table entries and emit negative
                    // cooperation signals for entries that never received a proof.
                    let expired_entries: Vec<_> = self
                        .reverse_table
                        .iter()
                        .filter(|(_, e)| now.saturating_sub(e.timestamp) >= REVERSE_TIMEOUT)
                        .map(|(hash, e)| (*hash, e.outbound_interface.clone(), e.proof_received))
                        .collect();
                    for (_hash, outbound, proof_received) in &expired_entries {
                        if !proof_received {
                            self.cooperation.observe_proof_timeout(outbound, now);
                        }
                    }
                    let expired_count = expired_entries.len();
                    self.reverse_table
                        .retain(|hash, _| !expired_entries.iter().any(|(h, _, _)| h == hash));
                    if expired_count > 0 {
                        actions.push(NodeAction::ReverseTableExpired {
                            count: expired_count,
                        });
                    }

                    // Expire stale link table entries
                    let link_before = self.link_table.len();
                    self.link_table.retain(|_, entry| {
                        if entry.validated {
                            // Validated links: expire after LINK_TIMEOUT of inactivity
                            now.saturating_sub(entry.timestamp) < LINK_TIMEOUT
                        } else {
                            // Unvalidated links: expire after proof_timeout
                            now < entry.proof_timeout
                        }
                    });
                    let link_expired = link_before - self.link_table.len();
                    if link_expired > 0 {
                        actions.push(NodeAction::LinkTableExpired {
                            count: link_expired,
                        });
                    }
                }

                // Expire pending echo entries that timed out without an echo.
                // Each expired entry is a negative cooperation signal: the
                // interface's neighbor did not forward our announce.
                let expired_echoes: Vec<(Arc<str>, DestinationHash)> = self
                    .pending_echoes
                    .iter()
                    .filter(|(_, &ts)| now.saturating_sub(ts) > ECHO_TIMEOUT)
                    .map(|(key, _)| key.clone())
                    .collect();
                for (iface, dest) in &expired_echoes {
                    self.cooperation.observe_announce_timeout(iface, now);
                    self.pending_echoes.remove(&(Arc::clone(iface), *dest));
                }

                // Decay scores for interfaces that haven't been observed recently.
                self.cooperation.decay_stale(now);

                actions
            }
        }
    }

    /// Read-only access to the path table.
    pub fn path_table(&self) -> &PathTable {
        &self.path_table
    }

    /// Number of registered interfaces.
    pub fn interface_count(&self) -> usize {
        self.interfaces.len()
    }

    /// Number of registered local destinations.
    pub fn destination_count(&self) -> usize {
        self.local_destinations.len()
    }

    // ── Private helpers (outbound) ─────────────────────────────────────

    /// IFAC-mask (if configured) and emit `SendOnInterface` for one interface.
    fn send_on_interface(&self, interface_name: &Arc<str>, raw: &[u8]) -> Vec<NodeAction> {
        let iface = match self.interfaces.get(&**interface_name) {
            Some(c) => c,
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::UnknownInterface,
                    interface_name: Arc::clone(interface_name),
                }];
            }
        };
        let outbound = if let Some(ref auth) = iface.ifac {
            match auth.mask(raw) {
                Ok(masked) => masked,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::OutboundIfacFailed,
                        interface_name: Arc::clone(interface_name),
                    }];
                }
            }
        } else {
            raw.to_vec()
        };
        vec![NodeAction::SendOnInterface {
            interface_name: Arc::clone(interface_name),
            raw: outbound,
            weight: None,
        }]
    }

    /// Broadcast raw bytes on all registered interfaces, IFAC-masking each.
    ///
    /// Each interface gets a cooperation weight via `get_broadcast_weights()`.
    /// The highest-scored interface is forced to 1.0 to guarantee delivery.
    fn broadcast_on_all_interfaces(&self, raw: &[u8]) -> Vec<NodeAction> {
        let weights = self.cooperation.get_broadcast_weights();
        let mut actions = Vec::with_capacity(self.interfaces.len());
        for (name, weight) in &weights {
            if let Some(iface) = self.interfaces.get(&**name) {
                let outbound = if let Some(ref auth) = iface.ifac {
                    match auth.mask(raw) {
                        Ok(masked) => masked,
                        Err(_) => {
                            actions.push(NodeAction::PacketDropped {
                                reason: DropReason::OutboundIfacFailed,
                                interface_name: Arc::clone(name),
                            });
                            continue;
                        }
                    }
                } else {
                    raw.to_vec()
                };
                actions.push(NodeAction::SendOnInterface {
                    interface_name: Arc::clone(name),
                    raw: outbound,
                    weight: Some(*weight),
                });
            }
        }
        // Include any interfaces not yet in the cooperation table (shouldn't
        // happen in normal use, but keeps the invariant that all interfaces
        // participate in broadcasts).
        for name in self.interfaces.keys() {
            if !weights.iter().any(|(n, _)| n == name) {
                actions.extend(self.send_on_interface(name, raw));
            }
        }
        actions
    }

    /// Broadcast on all interfaces except the named one (source exclusion).
    ///
    /// Computes cooperation weights AFTER excluding the source interface, so the
    /// "force highest to 1.0" guarantee applies to the actual sending set.
    fn broadcast_except(&self, exclude: &str, raw: &[u8]) -> Vec<NodeAction> {
        let all_weights = self.cooperation.get_broadcast_weights();
        // Filter out excluded interface, then re-apply "highest = 1.0" invariant.
        let filtered: Vec<_> = all_weights
            .into_iter()
            .filter(|(n, _)| n.as_ref() != exclude)
            .collect();
        let max_w = filtered.iter().map(|(_, w)| *w).fold(f32::NEG_INFINITY, f32::max);
        let weights: Vec<(Arc<str>, f32)> = filtered
            .into_iter()
            .map(|(n, w)| if w >= max_w { (n, 1.0) } else { (n, w) })
            .collect();
        let mut actions = Vec::with_capacity(self.interfaces.len());
        for (name, weight) in &weights {
            if let Some(iface) = self.interfaces.get(&**name) {
                let outbound = if let Some(ref auth) = iface.ifac {
                    match auth.mask(raw) {
                        Ok(masked) => masked,
                        Err(_) => {
                            actions.push(NodeAction::PacketDropped {
                                reason: DropReason::OutboundIfacFailed,
                                interface_name: Arc::clone(name),
                            });
                            continue;
                        }
                    }
                } else {
                    raw.to_vec()
                };
                actions.push(NodeAction::SendOnInterface {
                    interface_name: Arc::clone(name),
                    raw: outbound,
                    weight: Some(*weight),
                });
            }
        }
        // Include any interfaces not yet in the cooperation table.
        for name in self.interfaces.keys() {
            if &**name != exclude && !weights.iter().any(|(n, _)| n == name) {
                actions.extend(self.send_on_interface(name, raw));
            }
        }
        actions
    }

    /// Check if an announce for dest_hash is rate-limited. Mutates rate table.
    fn is_rate_limited(
        &mut self,
        dest: DestinationHash,
        config: &AnnounceRateConfig,
        now: u64,
    ) -> bool {
        let entry = self
            .announce_rate_table
            .entry(dest)
            .or_insert_with(|| AnnounceRateEntry {
                last_checked: 0,
                rate_violations: 0,
                blocked_until: 0,
            });

        // If currently blocked, check if penalty has expired
        if now < entry.blocked_until {
            return true;
        }

        // Check interval since last announce
        let elapsed = now.saturating_sub(entry.last_checked);
        if entry.last_checked > 0 && elapsed < config.target {
            entry.rate_violations += 1;
            if entry.rate_violations > config.grace {
                entry.blocked_until = now + config.target + config.penalty;
                // Reset violations so grace is renewable after penalty expires
                entry.rate_violations = 0;
                return true;
            }
        } else {
            // Properly spaced — reset violations
            entry.rate_violations = 0;
        }

        entry.last_checked = now;
        false
    }

    /// Evict stale entries from the announce rate table.
    ///
    /// An entry is stale when it hasn't been checked recently and isn't
    /// actively blocking a destination. Called from TimerTick.
    fn expire_rate_table(&mut self, now: u64) {
        self.announce_rate_table.retain(|_, entry| {
            let age = now.saturating_sub(entry.last_checked);
            // Keep if: still within expiry window, OR actively blocking
            age < RATE_TABLE_EXPIRY || now < entry.blocked_until
        });
    }

    /// Detect local rebroadcast echo or downstream forward.
    /// Returns `true` if the packet matched an announce table entry (echo or forward).
    fn check_rebroadcast_echo(&mut self, dest: DestinationHash, incoming_hops: u8) -> bool {
        let (matched, should_remove) = if let Some(entry) = self.announce_table.get_mut(&dest) {
            // incoming_hops is post-increment (our pipeline already did +1)
            // entry.hops is stored post-increment too
            // Echo: we hear our own rebroadcast back (same hop count)
            if incoming_hops.saturating_sub(1) == entry.hops {
                entry.local_rebroadcasts += 1;
                let remove =
                    entry.retries > 0 && entry.local_rebroadcasts >= LOCAL_REBROADCASTS_MAX;
                (true, remove)
            }
            // Downstream forward: another transport picked it up (hops+1)
            else if incoming_hops.saturating_sub(1) == entry.hops.saturating_add(1) {
                (true, entry.retries > 0)
            } else {
                // Entry exists but hops don't match echo/forward pattern
                (true, false)
            }
        } else {
            (false, false)
        };

        if should_remove {
            self.announce_table.remove(&dest);
        }

        matched
    }

    /// Process announce table: remove completed entries, retransmit due entries.
    fn process_announce_table(&mut self, now: u64) -> Vec<NodeAction> {
        let transport_hash = match self.transport_identity_hash() {
            Some(h) => h,
            None => return Vec::new(),
        };

        // First pass: collect completed and due entries
        let mut completed = Vec::new();
        #[allow(clippy::type_complexity)]
        let mut due: Vec<(DestinationHash, Option<Vec<u8>>, Arc<str>, u8)> = Vec::new();

        for (dest, entry) in &self.announce_table {
            // Completed: exceeded max retries, or retries>0 with enough rebroadcasts
            if entry.retries > PATHFINDER_R
                || (entry.retries > 0 && entry.local_rebroadcasts >= LOCAL_REBROADCASTS_MAX)
            {
                completed.push(*dest);
                continue;
            }

            // Due for retransmit
            if now >= entry.retransmit_at {
                // Build HEADER_2 packet
                let packet = Packet {
                    header: PacketHeader {
                        flags: PacketFlags {
                            ifac: false,
                            header_type: HeaderType::Type2,
                            context_flag: false,
                            propagation: PropagationType::Transport,
                            destination_type: DestinationType::Single,
                            packet_type: PacketType::Announce,
                        },
                        hops: entry.hops,
                        transport_id: Some(transport_hash),
                        destination_hash: entry.destination_hash,
                        context: PacketContext::None,
                    },
                    data: entry.packet_data.clone(),
                };

                // Always mark as due (even on serialization failure) so
                // retries increments and the entry can eventually be cleaned up.
                let raw = packet.to_bytes().ok();
                due.push((*dest, raw, entry.source_interface.clone(), entry.hops));
            }
        }

        // Remove completed entries
        for dest in &completed {
            self.announce_table.remove(dest);
        }

        // Process due retransmits
        let mut actions = Vec::new();
        for (dest, raw, source_iface, hops) in due {
            if let Some(entry) = self.announce_table.get_mut(&dest) {
                entry.retries += 1;
                entry.retransmit_at = now + PATHFINDER_G + (dest[2] as u64 % 3);
            }
            match raw {
                Some(bytes) => {
                    actions.extend(self.broadcast_except(&source_iface, &bytes));
                    actions.push(NodeAction::AnnounceRebroadcast {
                        destination_hash: dest,
                        hops,
                    });
                }
                None => {
                    actions.push(NodeAction::PacketDropped {
                        reason: DropReason::AnnounceRebroadcastSerializeFailed,
                        interface_name: source_iface,
                    });
                }
            }
        }

        actions
    }

    // ── Private pipeline (inbound) ──────────────────────────────────────

    fn process_inbound(
        &mut self,
        interface_name: String,
        raw: Vec<u8>,
        now: u64,
    ) -> Vec<NodeAction> {
        // 1. Interface lookup + 2. IFAC handling
        // Confined to a block so the immutable borrow of self.interfaces
        // is dropped before we mutate self.packet_hashlist / self.path_table.
        // Also recovers the interned Arc<str> from the HashMap key.
        let (interface_name, processed_raw, interface_mode) = {
            let (arc_name, iface_config) =
                match self.interfaces.get_key_value(interface_name.as_str()) {
                    Some((k, v)) => (Arc::clone(k), v),
                    None => {
                        return vec![NodeAction::PacketDropped {
                            reason: DropReason::UnknownInterface,
                            interface_name: Arc::from(interface_name),
                        }];
                    }
                };

            let has_ifac_flag = raw.first().is_some_and(|&b| b & 0x80 != 0);

            let processed = match (iface_config.ifac.is_some(), has_ifac_flag) {
                (true, true) => match iface_config.ifac.as_ref().unwrap().unmask(&raw) {
                    Ok(unmasked) => unmasked,
                    Err(_) => {
                        return vec![NodeAction::PacketDropped {
                            reason: DropReason::IfacFailed,
                            interface_name: arc_name,
                        }];
                    }
                },
                (true, false) | (false, true) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::IfacMismatch,
                        interface_name: arc_name,
                    }];
                }
                (false, false) => raw,
            };

            (arc_name, processed, iface_config.mode)
        };
        // From here, interface_name is Arc<str> (shadows the original String).

        // 3. Parse
        let mut packet = match Packet::from_bytes(&processed_raw) {
            Ok(p) => p,
            Err(_) => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::ParseFailed,
                    interface_name,
                }];
            }
        };

        // 4. Hops increment
        packet.header.hops = packet.header.hops.saturating_add(1);

        // 5. Broadcast filter
        let dest_type = packet.header.flags.destination_type;
        let pkt_type = packet.header.flags.packet_type;
        let is_broadcast_type =
            matches!(dest_type, DestinationType::Plain | DestinationType::Group);

        if is_broadcast_type {
            if pkt_type == PacketType::Announce {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::InvalidAnnounceDestType,
                    interface_name,
                }];
            }
            if packet.header.hops > 1 {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::MultiHopBroadcast,
                    interface_name,
                }];
            }
        }

        // 6. Dedup
        let hashable = packet
            .hashable_part()
            .expect("hashable_part is infallible for parsed packets");
        let full_packet_hash = hash::full_hash(&hashable);
        let is_announce = pkt_type == PacketType::Announce;
        let is_lrproof =
            pkt_type == PacketType::Proof && packet.header.context == PacketContext::LrProof;

        // Duplicate non-announces are dropped; duplicate announces still pass
        // through for path evaluation (multiple paths to same destination).
        // LRPROOF packets are also exempt: link proofs must pass through
        // transport nodes for link establishment.
        if !self.packet_hashlist.insert(full_packet_hash) && !is_announce && !is_lrproof {
            return vec![NodeAction::PacketDropped {
                reason: DropReason::DuplicatePacket,
                interface_name,
            }];
        }

        // 7. Dispatch by packet type
        if is_announce {
            self.process_announce(
                packet,
                &full_packet_hash,
                interface_name,
                interface_mode,
                now,
            )
        } else {
            self.process_data_packet(packet, interface_name, now)
        }
    }

    fn process_announce(
        &mut self,
        packet: Packet,
        full_packet_hash: &[u8; 32],
        interface_name: Arc<str>,
        interface_mode: InterfaceMode,
        now: u64,
    ) -> Vec<NodeAction> {
        let validated = match validate_announce(&packet) {
            Ok(v) => v,
            Err(_) => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::AnnounceInvalid,
                    interface_name,
                }];
            }
        };

        let destination_hash = validated.destination_hash;
        let random_hash = validated.random_hash;
        let hops = packet.header.hops;

        // Rebroadcast detection runs BEFORE rate limiting so echoes of our
        // own rebroadcasts don't consume grace points. Echoes are expected
        // traffic, not abuse — they shouldn't count toward rate violations.
        let was_echo = if self.is_transport() {
            self.check_rebroadcast_echo(destination_hash, hops)
        } else {
            false
        };

        // Rate limiting (transport mode only).
        // Skip if this packet matched an announce table entry (echo/forward of
        // our own rebroadcast) — even if the entry was just removed by echo
        // detection, the packet is expected traffic, not abuse.
        if self.is_transport() && !was_echo {
            let rate_config = self
                .interfaces
                .get(&interface_name)
                .and_then(|c| c.announce_rate);
            if let Some(config) = rate_config {
                if self.is_rate_limited(destination_hash, &config, now) {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::AnnounceRateLimited,
                        interface_name,
                    }];
                }
            }
        }

        // Announce packet hash: truncated (first 16 bytes) of the full hash
        let mut announce_packet_hash: DestinationHash = [0u8; 16];
        announce_packet_hash.copy_from_slice(&full_packet_hash[..16]);

        // If the announce arrived via a transport node (Type2 header),
        // next_hop is the transport_id so relay knows to chain through it.
        // For direct (Type1) announces, next_hop is the destination itself.
        let next_hop = packet.header.transport_id.unwrap_or(destination_hash);

        let path_update = self.path_table.update(
            destination_hash,
            next_hop,
            hops,
            interface_name.clone(),
            announce_packet_hash,
            random_hash,
            interface_mode,
            now,
        );

        // Announce echo detection: if this announce is for a destination we own,
        // a neighbor forwarded our announce back to us. Credit the *receiving*
        // interface — the neighbor on that interface cooperated by forwarding.
        //
        // We broadcast announces on ALL interfaces, so any echo arriving on any
        // interface means that interface's neighbor forwarded it. No need to check
        // which interface we sent on — we sent on all of them.
        if self.announcing_destinations.contains_key(&destination_hash) {
            // Confirm we actually originated an announce recently (pending echo exists)
            let has_pending = self
                .pending_echoes
                .keys()
                .any(|(_, dh)| *dh == destination_hash);
            if has_pending {
                self.cooperation
                    .observe_announce_forwarded(&interface_name, now);
                // Only remove the echoing interface's entry — leave other interfaces'
                // entries so they can independently earn credit or timeout. This prevents
                // first-echo-wins bias where the fastest neighbor monopolizes positive
                // observations while equally-cooperative neighbors get nothing.
                self.pending_echoes
                    .remove(&(Arc::clone(&interface_name), destination_hash));
            }
        }

        // Announce table insertion (transport mode only)
        if self.is_transport()
            && !self.local_destinations.contains(&destination_hash)
            && !self.announce_table.contains_key(&destination_hash)
            && matches!(
                path_update,
                PathUpdateResult::Inserted | PathUpdateResult::Updated | PathUpdateResult::Kept
            )
        {
            let initial_delay = (destination_hash[1] as u64) % 5;
            self.announce_table.insert(
                destination_hash,
                AnnounceTableEntry {
                    received_at: now,
                    retransmit_at: now + initial_delay,
                    retries: 0,
                    local_rebroadcasts: 0,
                    source_interface: interface_name.clone(),
                    hops,
                    destination_hash,
                    packet_data: packet.data,
                },
            );
        }

        vec![NodeAction::AnnounceReceived {
            destination_hash,
            validated_announce: Box::new(validated),
            path_update,
            interface_name,
            hops,
        }]
    }

    fn process_data_packet(
        &mut self,
        packet: Packet,
        interface_name: Arc<str>,
        now: u64,
    ) -> Vec<NodeAction> {
        let destination_hash = packet.header.destination_hash;

        // 1. Local delivery takes priority
        if self.local_destinations.contains(&destination_hash) {
            return vec![NodeAction::DeliverLocally {
                destination_hash,
                packet,
                interface_name,
            }];
        }

        // 2. Non-transport nodes cannot relay
        let our_transport_hash = match self.transport_identity_hash() {
            Some(h) => h,
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::NoLocalDestination,
                    interface_name,
                }];
            }
        };

        // 3. LRPROOF routing via link table (before transport_id check —
        //    link proofs travel as Type1/Broadcast with no transport_id)
        if packet.header.flags.packet_type == PacketType::Proof
            && packet.header.context == PacketContext::LrProof
        {
            return self.route_link_proof(&packet, &interface_name, now);
        }

        // 4. Regular proof routing via reverse table
        if packet.header.flags.packet_type == PacketType::Proof {
            return self.route_proof(&packet, &interface_name, now);
        }

        // 5. Only relay Type2 packets addressed to our transport_id
        if packet.header.transport_id != Some(our_transport_hash) {
            return vec![NodeAction::PacketDropped {
                reason: DropReason::NoLocalDestination,
                interface_name,
            }];
        }

        // 6. LinkRequest → create link table entry + forward
        if packet.header.flags.packet_type == PacketType::LinkRequest {
            return self.forward_link_request(packet, interface_name, now);
        }

        // 7. Link data routing via link table
        if self.link_table.contains_key(&destination_hash) {
            return self.route_link_data(packet, interface_name, now);
        }

        // 8. Look up destination in path table — extract only what relay needs
        let (next_hop, path_iface) = match self.path_table.get(&destination_hash) {
            Some(entry) => (entry.next_hop, Arc::clone(&entry.interface_name)),
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::NoRouteForTransport,
                    interface_name,
                }];
            }
        };

        // 9. Compute reverse key before relay (needs packet data pre-move)
        let reverse_key = hash::truncated_hash(
            &packet
                .hashable_part()
                .expect("hashable_part is infallible for parsed packets"),
        );

        // 10. Relay the packet
        let actions = self.relay_packet(packet, next_hop, &path_iface);

        // 11. Only store reverse table entry if relay succeeded
        if !actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
        {
            self.reverse_table.insert(
                reverse_key,
                ReverseTableEntry {
                    received_interface: interface_name,
                    outbound_interface: Arc::clone(&path_iface),
                    timestamp: now,
                    proof_received: false,
                },
            );
        }

        actions
    }

    /// Relay a data packet to the next hop.
    ///
    /// Final hop (next_hop == destination): convert Type2→Type1, Broadcast propagation.
    /// Intermediate hop: keep Type2, replace transport_id with next transport.
    fn relay_packet(
        &self,
        packet: Packet,
        next_hop: DestinationHash,
        out_iface: &Arc<str>,
    ) -> Vec<NodeAction> {
        let destination_hash = packet.header.destination_hash;
        let is_final_hop = next_hop == destination_hash;

        // Hops were already incremented in the inbound pipeline (process_incoming),
        // so we forward with the current value — matching Python Reticulum behavior.
        let forwarded = if is_final_hop {
            // Final hop: convert Type2 → Type1 (direct delivery)
            Packet {
                header: PacketHeader {
                    flags: PacketFlags {
                        header_type: HeaderType::Type1,
                        propagation: PropagationType::Broadcast,
                        ..packet.header.flags
                    },
                    hops: packet.header.hops,
                    transport_id: None,
                    destination_hash,
                    context: packet.header.context,
                },
                data: packet.data,
            }
        } else {
            // Intermediate hop: keep Type2, replace transport_id
            Packet {
                header: PacketHeader {
                    flags: packet.header.flags,
                    hops: packet.header.hops,
                    transport_id: Some(next_hop),
                    destination_hash,
                    context: packet.header.context,
                },
                data: packet.data,
            }
        };

        match forwarded.to_bytes() {
            Ok(raw) => {
                let mut actions = self.send_on_interface(out_iface, &raw);
                if !actions
                    .iter()
                    .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
                {
                    actions.push(NodeAction::PacketRelayed {
                        destination_hash,
                        next_hop,
                        interface_name: Arc::clone(out_iface),
                    });
                }
                actions
            }
            Err(_) => {
                vec![NodeAction::PacketDropped {
                    reason: DropReason::RelaySerializeFailed,
                    interface_name: Arc::clone(out_iface),
                }]
            }
        }
    }

    /// Forward a LinkRequest packet toward its destination and create a link table entry.
    fn forward_link_request(
        &mut self,
        packet: Packet,
        interface_name: Arc<str>,
        now: u64,
    ) -> Vec<NodeAction> {
        let destination_hash = packet.header.destination_hash;

        // Look up destination in path table — extract only needed fields
        let (next_hop, path_iface, path_hops) = match self.path_table.get(&destination_hash) {
            Some(entry) => (
                entry.next_hop,
                Arc::clone(&entry.interface_name),
                entry.hops,
            ),
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::NoRouteForLinkRequest,
                    interface_name,
                }];
            }
        };

        // Compute link_id from the request packet
        let link_id = match link::link_id_from_request(&packet) {
            Ok(id) => id,
            Err(_) => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::NoRouteForLinkRequest,
                    interface_name,
                }];
            }
        };

        let taken_hops = packet.header.hops;
        let proof_timeout = now + ESTABLISHMENT_TIMEOUT_PER_HOP * (path_hops as u64).max(1);

        // Forward via relay_packet (same Type2→Type1/Type2 conversion as data relay)
        let actions = self.relay_packet(packet, next_hop, &path_iface);

        // Only create link table entry if relay succeeded
        if !actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
        {
            self.link_table.insert(
                link_id,
                LinkTableEntry {
                    timestamp: now,
                    next_hop,
                    outbound_interface: Arc::clone(&path_iface),
                    remaining_hops: path_hops,
                    received_interface: interface_name,
                    taken_hops,
                    destination_hash,
                    validated: false,
                    proof_timeout,
                },
            );

            let mut result = actions;
            result.push(NodeAction::LinkRequestForwarded {
                link_id,
                destination_hash,
                next_hop,
                interface_name: path_iface,
            });
            result
        } else {
            actions
        }
    }

    /// Route an LRPROOF (link proof) back toward the link initiator via the link table.
    fn route_link_proof(
        &mut self,
        packet: &Packet,
        interface_name: &Arc<str>,
        now: u64,
    ) -> Vec<NodeAction> {
        let link_id = packet.header.destination_hash;

        let entry = match self.link_table.get_mut(&link_id) {
            Some(e) => e,
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::LinkProofNoEntry,
                    interface_name: Arc::clone(interface_name),
                }];
            }
        };

        // Validate hop count matches expected remaining_hops
        if packet.header.hops != entry.remaining_hops {
            return vec![NodeAction::PacketDropped {
                reason: DropReason::LinkProofHopsMismatch,
                interface_name: Arc::clone(interface_name),
            }];
        }

        // Validate proof arrived on the outbound interface (from destination side)
        if **interface_name != *entry.outbound_interface {
            return vec![NodeAction::PacketDropped {
                reason: DropReason::LinkProofWrongInterface,
                interface_name: Arc::clone(interface_name),
            }];
        }

        // Mark entry as validated and update timestamp
        entry.validated = true;
        entry.timestamp = now;
        let target_interface = Arc::clone(&entry.received_interface);

        // Construct Type1/Broadcast proof packet for the initiator side
        let forwarded = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    header_type: HeaderType::Type1,
                    propagation: PropagationType::Broadcast,
                    ..packet.header.flags
                },
                hops: packet.header.hops,
                transport_id: None,
                destination_hash: link_id,
                context: packet.header.context,
            },
            data: packet.data.clone(),
        };

        match forwarded.to_bytes() {
            Ok(raw) => {
                let mut actions = self.send_on_interface(&target_interface, &raw);
                if !actions
                    .iter()
                    .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
                {
                    actions.push(NodeAction::LinkProofRouted {
                        link_id,
                        interface_name: target_interface,
                    });
                }
                actions
            }
            Err(_) => {
                vec![NodeAction::PacketDropped {
                    reason: DropReason::RelaySerializeFailed,
                    interface_name: target_interface,
                }]
            }
        }
    }

    /// Route link data bidirectionally through the link table.
    ///
    /// Unlike raw relay, this rewrites transport headers for the next hop:
    /// - Final hop (directly reachable recipient): Type2 → Type1/Broadcast
    /// - Intermediate hop toward destination: transport_id = next transport node
    /// - Intermediate hop toward initiator: best-effort (multi-hop reverse path
    ///   is a known limitation — deferred to a future bead)
    fn route_link_data(
        &mut self,
        packet: Packet,
        interface_name: Arc<str>,
        now: u64,
    ) -> Vec<NodeAction> {
        let link_id = packet.header.destination_hash;

        // Determine the target interface and direction based on which side
        // the packet arrived from. `toward_destination` is true when forwarding
        // from the initiator side toward the link destination.
        let (target_interface, toward_destination) = {
            let entry = self.link_table.get(&link_id).unwrap();

            if entry.outbound_interface == entry.received_interface {
                // Same-interface case: use hops to distinguish direction
                if packet.header.hops == entry.taken_hops {
                    (Some(Arc::clone(&entry.outbound_interface)), true)
                } else if packet.header.hops == entry.remaining_hops {
                    (Some(Arc::clone(&entry.outbound_interface)), false)
                } else {
                    (None, false)
                }
            } else if *interface_name == *entry.outbound_interface
                && packet.header.hops == entry.remaining_hops
            {
                // From destination side → forward to initiator
                (Some(Arc::clone(&entry.received_interface)), false)
            } else if *interface_name == *entry.received_interface
                && packet.header.hops == entry.taken_hops
            {
                // From initiator side → forward to destination
                (Some(Arc::clone(&entry.outbound_interface)), true)
            } else {
                (None, false)
            }
        };

        let target = match target_interface {
            Some(t) => t,
            None => {
                // No matching direction — silently drop
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::NoLocalDestination,
                    interface_name,
                }];
            }
        };

        // Rewrite transport headers for the next hop.
        let entry = self.link_table.get(&link_id).unwrap();
        let same_interface_equidistant = entry.outbound_interface == entry.received_interface
            && entry.taken_hops == entry.remaining_hops;
        let is_final_hop = if same_interface_equidistant {
            // Both endpoints on the same broadcast medium at equal distance:
            // direction is ambiguous but both are directly reachable, so
            // always convert to Type1/Broadcast.
            true
        } else if toward_destination {
            // Final hop: next_hop IS the actual destination
            entry.next_hop == entry.destination_hash
        } else {
            // Final hop toward initiator: request came directly (single hop)
            entry.taken_hops <= 1
        };
        let next_hop = entry.next_hop;

        let forwarded = if is_final_hop {
            // Final hop: convert Type2 → Type1 (direct delivery)
            Packet {
                header: PacketHeader {
                    flags: PacketFlags {
                        header_type: HeaderType::Type1,
                        propagation: PropagationType::Broadcast,
                        ..packet.header.flags
                    },
                    hops: packet.header.hops,
                    transport_id: None,
                    destination_hash: link_id,
                    context: packet.header.context,
                },
                data: packet.data,
            }
        } else if toward_destination {
            // Intermediate hop toward destination: address to next transport
            Packet {
                header: PacketHeader {
                    flags: packet.header.flags,
                    hops: packet.header.hops,
                    transport_id: Some(next_hop),
                    destination_hash: link_id,
                    context: packet.header.context,
                },
                data: packet.data,
            }
        } else {
            // Intermediate hop toward initiator: multi-hop reverse path is
            // unknown (we don't store the previous transport identity).
            // Preserve existing headers as best-effort. Full multi-transport-hop
            // reverse routing is deferred to a future bead.
            packet
        };

        match forwarded.to_bytes() {
            Ok(raw) => {
                let mut actions = self.send_on_interface(&target, &raw);
                if !actions
                    .iter()
                    .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
                {
                    // Update timestamp to keep validated links alive
                    if let Some(entry) = self.link_table.get_mut(&link_id) {
                        entry.timestamp = now;
                    }
                    actions.push(NodeAction::LinkDataRouted {
                        link_id,
                        interface_name: target,
                    });
                }
                actions
            }
            Err(_) => {
                vec![NodeAction::PacketDropped {
                    reason: DropReason::RelaySerializeFailed,
                    interface_name: target,
                }]
            }
        }
    }

    /// Route a proof packet back toward the originator via the reverse table.
    fn route_proof(
        &mut self,
        packet: &Packet,
        interface_name: &Arc<str>,
        now: u64,
    ) -> Vec<NodeAction> {
        let proof_dest = packet.header.destination_hash;

        let entry = match self.reverse_table.get_mut(&proof_dest) {
            Some(e) => e,
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::ProofNoReverseEntry,
                    interface_name: Arc::clone(interface_name),
                }];
            }
        };

        // Mark proof received and record positive cooperation observation.
        // Only credit the first proof — duplicate proofs from redundant mesh paths
        // should not inflate the score.
        let already_received = entry.proof_received;
        entry.proof_received = true;
        let outbound_iface = entry.outbound_interface.clone();
        let received_iface = entry.received_interface.clone();

        if !already_received {
            self.cooperation.observe_proof_delivered(&outbound_iface, now);
        }

        // Forward proof with current hops (already incremented in inbound pipeline)
        let forwarded = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    header_type: HeaderType::Type1,
                    propagation: PropagationType::Broadcast,
                    ..packet.header.flags
                },
                hops: packet.header.hops,
                transport_id: None,
                destination_hash: proof_dest,
                context: packet.header.context,
            },
            data: packet.data.clone(),
        };

        match forwarded.to_bytes() {
            Ok(raw) => {
                let mut actions = self.send_on_interface(&received_iface, &raw);
                if !actions
                    .iter()
                    .any(|a| matches!(a, NodeAction::PacketDropped { .. }))
                {
                    actions.push(NodeAction::ProofRelayed {
                        proof_destination: proof_dest,
                        interface_name: received_iface,
                    });
                }
                actions
            }
            Err(_) => {
                vec![NodeAction::PacketDropped {
                    reason: DropReason::RelaySerializeFailed,
                    interface_name: received_iface,
                }]
            }
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::announce::build_announce;
    use crate::context::PacketContext;
    use crate::destination::DestinationName;
    use crate::packet::{HeaderType, PacketFlags, PacketHeader, PropagationType};
    use crate::path_table::PathEntry;
    use harmony_identity::identity::PrivateIdentity;
    use rand::rngs::OsRng;

    fn dest(id: u8) -> DestinationHash {
        let mut h = [0u8; 16];
        h[0] = id;
        h
    }

    fn make_node_with_interface() -> Node {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node
    }

    /// Build a Single-destination data packet as raw bytes.
    fn make_data_packet_raw(dest_hash: DestinationHash, hops: u8) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops,
                transport_id: None,
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0xDE, 0xAD, 0xBE, 0xEF].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a Plain-destination data packet as raw bytes.
    fn make_plain_data_packet_raw(dest_hash: DestinationHash, hops: u8) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Plain,
                    packet_type: PacketType::Data,
                },
                hops,
                transport_id: None,
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0x01, 0x02, 0x03].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a Plain-destination announce packet as raw bytes (invalid combo).
    fn make_plain_announce_raw(dest_hash: DestinationHash) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Plain,
                    packet_type: PacketType::Announce,
                },
                hops: 0,
                transport_id: None,
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0; 200].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a valid cryptographic announce and return (raw_bytes, dest_hash).
    fn make_valid_announce() -> (Vec<u8>, DestinationHash) {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest_name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let packet =
            build_announce(&identity, &dest_name, &mut OsRng, 1_700_000_000, b"", None).unwrap();
        let dest_hash = packet.header.destination_hash;
        (packet.to_bytes().unwrap(), dest_hash)
    }

    // ── 1. Registration ────────────────────────────────────────────────

    #[test]
    fn register_and_unregister_interface() {
        let mut node = Node::new();
        assert_eq!(node.interface_count(), 0);

        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        assert_eq!(node.interface_count(), 1);

        assert!(node.unregister_interface("eth0"));
        assert_eq!(node.interface_count(), 0);
        assert!(!node.unregister_interface("eth0"));
    }

    #[test]
    fn register_and_unregister_destination() {
        let mut node = Node::new();
        assert_eq!(node.destination_count(), 0);

        node.register_destination(dest(1));
        assert_eq!(node.destination_count(), 1);

        assert!(node.unregister_destination(&dest(1)));
        assert_eq!(node.destination_count(), 0);
        assert!(!node.unregister_destination(&dest(1)));
    }

    #[test]
    fn unknown_interface_drops_packet() {
        let mut node = Node::new();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth99".into(),
            raw: vec![0x00; 50],
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped {
                reason,
                interface_name,
            } => {
                assert_eq!(*reason, DropReason::UnknownInterface);
                assert_eq!(&**interface_name, "eth99");
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    // ── 2. IFAC ────────────────────────────────────────────────────────

    #[test]
    fn ifac_unmask_success() {
        let mut node = Node::new();
        let auth_for_mask = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        let auth_for_node = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth_for_node));

        let dh = dest(1);
        node.register_destination(dh);

        let raw = make_data_packet_raw(dh, 0);
        let masked = auth_for_mask.mask(&raw).unwrap();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: masked,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], NodeAction::DeliverLocally { .. }));
    }

    #[test]
    fn ifac_flag_set_but_no_ifac_configured_drops() {
        let mut node = make_node_with_interface();

        let mut raw = make_data_packet_raw(dest(1), 0);
        raw[0] |= 0x80; // Set IFAC flag manually

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::IfacMismatch);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn ifac_expected_but_flag_missing_drops() {
        let mut node = Node::new();
        let auth = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth));

        let raw = make_data_packet_raw(dest(1), 0);
        assert_eq!(raw[0] & 0x80, 0); // Verify no IFAC flag

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::IfacMismatch);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn ifac_wrong_credentials_drops() {
        let mut node = Node::new();
        let auth_node = IfacAuthenticator::new(Some("network_a"), None, 8).unwrap();
        let auth_other = IfacAuthenticator::new(Some("network_b"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth_node));

        let raw = make_data_packet_raw(dest(1), 0);
        let masked = auth_other.mask(&raw).unwrap();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: masked,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::IfacFailed);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    // ── 3. Parse errors ────────────────────────────────────────────────

    #[test]
    fn too_short_packet_drops() {
        let mut node = make_node_with_interface();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: vec![0x00; 5],
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::ParseFailed);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn empty_raw_bytes_drops() {
        let mut node = make_node_with_interface();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: vec![],
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::ParseFailed);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    // ── 4. Broadcast filter ────────────────────────────────────────────

    #[test]
    fn plain_hops_1_accepted() {
        let mut node = make_node_with_interface();
        let dh = dest(1);
        node.register_destination(dh);

        // hops=0 on wire → incremented to 1 → passes broadcast filter
        let raw = make_plain_data_packet_raw(dh, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], NodeAction::DeliverLocally { .. }));
    }

    #[test]
    fn plain_hops_gt1_dropped() {
        let mut node = make_node_with_interface();
        node.register_destination(dest(1));

        // hops=1 on wire → incremented to 2 → dropped
        let raw = make_plain_data_packet_raw(dest(1), 1);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::MultiHopBroadcast);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn plain_announce_dropped() {
        let mut node = make_node_with_interface();
        let raw = make_plain_announce_raw(dest(1));

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::InvalidAnnounceDestType);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    // ── 5. Dedup ───────────────────────────────────────────────────────

    #[test]
    fn duplicate_data_dropped() {
        let mut node = make_node_with_interface();
        let dh = dest(1);
        node.register_destination(dh);

        let raw = make_data_packet_raw(dh, 0);

        // First — delivered
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], NodeAction::DeliverLocally { .. }));

        // Second — duplicate dropped
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1001,
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::DuplicatePacket);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn duplicate_announce_passes() {
        let mut node = make_node_with_interface();
        let (raw, _dest_hash) = make_valid_announce();

        // First announce — inserted
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], NodeAction::AnnounceReceived { .. }));

        // Second announce (same packet) — still processes, path table sees duplicate blob
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1001,
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::AnnounceReceived { path_update, .. } => {
                assert_eq!(*path_update, PathUpdateResult::DuplicateBlob);
            }
            _ => panic!("expected AnnounceReceived"),
        }
    }

    // ── 6. Announce processing ─────────────────────────────────────────

    #[test]
    fn valid_announce_updates_path() {
        let mut node = make_node_with_interface();
        let (raw, dest_hash) = make_valid_announce();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::AnnounceReceived {
                destination_hash: dh,
                path_update,
                interface_name,
                hops,
                ..
            } => {
                assert_eq!(*dh, dest_hash);
                assert_eq!(*path_update, PathUpdateResult::Inserted);
                assert_eq!(&**interface_name, "eth0");
                assert_eq!(*hops, 1); // incremented from 0
            }
            _ => panic!("expected AnnounceReceived"),
        }

        // Verify path table
        let entry = node.path_table().get(&dest_hash).unwrap();
        assert_eq!(entry.hops, 1);
        assert_eq!(&*entry.interface_name, "eth0");
        assert_eq!(entry.next_hop, dest_hash); // leaf mode: next_hop = dest
    }

    #[test]
    fn invalid_announce_signature_dropped() {
        let mut node = make_node_with_interface();
        let (mut raw, _) = make_valid_announce();

        // Tamper with the signature (last byte of payload)
        let last = raw.len() - 1;
        raw[last] ^= 0xFF;

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::AnnounceInvalid);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn announce_hops_incremented() {
        let mut node = make_node_with_interface();
        let (raw, _) = make_valid_announce();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        match &actions[0] {
            NodeAction::AnnounceReceived { hops, .. } => {
                assert_eq!(*hops, 1); // 0 + 1
            }
            _ => panic!("expected AnnounceReceived"),
        }
    }

    // ── 7. Local delivery ──────────────────────────────────────────────

    #[test]
    fn data_to_registered_dest_delivered() {
        let mut node = make_node_with_interface();
        let dh = dest(42);
        node.register_destination(dh);

        let raw = make_data_packet_raw(dh, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::DeliverLocally {
                destination_hash,
                interface_name,
                ..
            } => {
                assert_eq!(*destination_hash, dh);
                assert_eq!(&**interface_name, "eth0");
            }
            _ => panic!("expected DeliverLocally"),
        }
    }

    #[test]
    fn data_to_unknown_dest_dropped() {
        let mut node = make_node_with_interface();
        // dest(42) not registered

        let raw = make_data_packet_raw(dest(42), 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::NoLocalDestination);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    // ── 8. Timer ───────────────────────────────────────────────────────

    #[test]
    fn timer_expires_old_paths() {
        let mut node = make_node_with_interface();

        let (raw, _) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });
        assert!(!node.path_table().is_empty());

        // Timer tick past default expiry (7 days = 604800s)
        let actions = node.handle_event(NodeEvent::TimerTick {
            now: 1000 + 604_800 + 1,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PathsExpired { count } => {
                assert_eq!(*count, 1);
            }
            _ => panic!("expected PathsExpired"),
        }
        assert!(node.path_table().is_empty());
    }

    #[test]
    fn timer_no_expiry_returns_empty() {
        let mut node = make_node_with_interface();

        let (raw, _) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        let actions = node.handle_event(NodeEvent::TimerTick { now: 1001 });
        assert!(actions.is_empty());
    }

    // ── 9. Edge case ───────────────────────────────────────────────────

    #[test]
    fn hops_255_saturating_add() {
        let mut node = make_node_with_interface();
        let dh = dest(1);
        node.register_destination(dh);

        let raw = make_data_packet_raw(dh, 255);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::DeliverLocally { packet, .. } => {
                assert_eq!(packet.header.hops, 255); // saturated, not overflowed
            }
            _ => panic!("expected DeliverLocally"),
        }
    }

    // ── Announcing destination helpers ───────────────────────────────

    fn make_announcing_node() -> (Node, DestinationHash) {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(identity, name, vec![], Some(300), 1000);
        (node, dh)
    }

    fn make_announcing_node_with_ifac() -> (Node, DestinationHash) {
        let mut node = Node::new();
        let auth = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth));

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(identity, name, vec![], Some(300), 1000);
        (node, dh)
    }

    // ── 10. Announcing destination registration ─────────────────────

    #[test]
    fn register_announcing_dest_returns_correct_hash() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("myapp", &["svc"]).unwrap();
        let expected = name.destination_hash(&identity.public_identity().address_hash);

        let dh = node.register_announcing_destination(identity, name, vec![], None, 0);
        assert_eq!(dh, expected);
        assert_eq!(node.announcing_destination_count(), 1);
        assert_eq!(node.destination_count(), 1); // also registered for local delivery
    }

    #[test]
    fn unregister_announcing_dest_removes_from_both() {
        let (mut node, dh) = make_announcing_node();
        assert_eq!(node.announcing_destination_count(), 1);
        assert_eq!(node.destination_count(), 1);

        assert!(node.unregister_announcing_destination(&dh));
        assert_eq!(node.announcing_destination_count(), 0);
        assert_eq!(node.destination_count(), 0);

        // Second call returns false
        assert!(!node.unregister_announcing_destination(&dh));
    }

    #[test]
    fn announcing_dest_receives_local_delivery() {
        let (mut node, dh) = make_announcing_node();
        let raw = make_data_packet_raw(dh, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], NodeAction::DeliverLocally { .. }));
    }

    // ── 11. Announce generation ─────────────────────────────────────

    #[test]
    fn announce_emits_send_for_each_interface() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(identity, name, vec![], None, 0);

        let actions = node.announce(&dh, &mut OsRng, 1_700_000_000);
        assert_eq!(actions.len(), 2);

        let mut iface_names: Vec<&str> = actions
            .iter()
            .map(|a| match a {
                NodeAction::SendOnInterface { interface_name, .. } => &**interface_name,
                _ => panic!("expected SendOnInterface"),
            })
            .collect();
        iface_names.sort();
        assert_eq!(iface_names, &["eth0", "wlan0"]);
    }

    #[test]
    fn announce_unknown_dest_returns_drop() {
        let mut node = Node::new();
        let actions = node.announce(&dest(99), &mut OsRng, 1_700_000_000);

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::UnknownDestination);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn announce_with_ifac_produces_masked_output() {
        let (mut node, dh) = make_announcing_node_with_ifac();

        let actions = node.announce(&dh, &mut OsRng, 1_700_000_000);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::SendOnInterface { raw, .. } => {
                // Masked packets have the IFAC flag set (bit 7 of first byte)
                assert_ne!(raw[0] & 0x80, 0, "IFAC flag should be set on masked output");
            }
            _ => panic!("expected SendOnInterface"),
        }
    }

    #[test]
    fn announce_mixed_ifac_interfaces() {
        let mut node = Node::new();
        let auth = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth));
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(identity, name, vec![], None, 0);

        let actions = node.announce(&dh, &mut OsRng, 1_700_000_000);
        assert_eq!(actions.len(), 2);

        for action in &actions {
            match action {
                NodeAction::SendOnInterface {
                    interface_name,
                    raw,
                    ..
                } => {
                    if &**interface_name == "eth0" {
                        // IFAC interface → flag set
                        assert_ne!(raw[0] & 0x80, 0);
                    } else {
                        // No IFAC → flag clear
                        assert_eq!(raw[0] & 0x80, 0);
                    }
                }
                _ => panic!("expected SendOnInterface"),
            }
        }
    }

    // ── 12. Auto-announce scheduling ────────────────────────────────

    #[test]
    fn timer_emits_announce_needed_when_due() {
        let (mut node, dh) = make_announcing_node();
        // next_announce_at = 1000 (the `now` passed to register)

        let actions = node.handle_event(NodeEvent::TimerTick { now: 1000 });
        let announce_needed: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, NodeAction::AnnounceNeeded { .. }))
            .collect();
        assert_eq!(announce_needed.len(), 1);
        match announce_needed[0] {
            NodeAction::AnnounceNeeded { dest_hash } => assert_eq!(*dest_hash, dh),
            _ => unreachable!(),
        }
    }

    #[test]
    fn timer_no_announce_before_interval() {
        let (mut node, _dh) = make_announcing_node();
        // After first tick at now=1000, next_announce_at = 1000 + 300 + jitter

        // Fire first tick to advance the schedule
        node.handle_event(NodeEvent::TimerTick { now: 1000 });

        // Fire again at now=1100 — well before the 300s interval
        let actions = node.handle_event(NodeEvent::TimerTick { now: 1100 });
        let announce_needed: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, NodeAction::AnnounceNeeded { .. }))
            .collect();
        assert!(announce_needed.is_empty());
    }

    #[test]
    fn manual_only_dest_never_auto_announces() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        // announce_interval = None → manual only
        node.register_announcing_destination(identity, name, vec![], None, 0);

        // Even at far future, no auto-announce
        let actions = node.handle_event(NodeEvent::TimerTick { now: 999_999 });
        let announce_needed: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, NodeAction::AnnounceNeeded { .. }))
            .collect();
        assert!(announce_needed.is_empty());
    }

    // ── 13. Outbound routing ────────────────────────────────────────

    #[test]
    fn route_packet_known_path_sends_on_correct_interface() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        // Inject a path via inbound announce on eth0
        let (raw, dh) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        let data = vec![0xDE, 0xAD];
        let actions = node.route_packet(&dh, data.clone());
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::SendOnInterface {
                interface_name,
                raw,
                ..
            } => {
                assert_eq!(&**interface_name, "eth0");
                assert_eq!(*raw, data);
            }
            _ => panic!("expected SendOnInterface"),
        }
    }

    #[test]
    fn route_packet_no_path_broadcasts_all() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        let data = vec![0xBE, 0xEF];
        let actions = node.route_packet(&dest(99), data);
        assert_eq!(actions.len(), 2);

        let mut iface_names: Vec<&str> = actions
            .iter()
            .map(|a| match a {
                NodeAction::SendOnInterface { interface_name, .. } => &**interface_name,
                _ => panic!("expected SendOnInterface"),
            })
            .collect();
        iface_names.sort();
        assert_eq!(iface_names, &["eth0", "wlan0"]);
    }

    #[test]
    fn route_packet_with_ifac_masks_outbound() {
        let mut node = Node::new();
        let auth = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth));

        // Route some raw data — needs to be long enough for IFAC mask (min packet size)
        let raw = make_data_packet_raw(dest(1), 0);
        let actions = node.route_packet(&dest(99), raw.clone());
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::SendOnInterface { raw: masked, .. } => {
                assert_ne!(*masked, raw, "output should be masked");
                assert_ne!(masked[0] & 0x80, 0, "IFAC flag should be set");
            }
            _ => panic!("expected SendOnInterface"),
        }
    }

    #[test]
    fn route_packet_broadcast_mixed_ifac() {
        let mut node = Node::new();
        let auth = IfacAuthenticator::new(Some("testnet"), None, 8).unwrap();
        node.register_interface("eth0".into(), InterfaceMode::Full, Some(auth));
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        let raw = make_data_packet_raw(dest(1), 0);
        let actions = node.route_packet(&dest(99), raw);
        assert_eq!(actions.len(), 2);

        for action in &actions {
            match action {
                NodeAction::SendOnInterface {
                    interface_name,
                    raw,
                    ..
                } => {
                    if &**interface_name == "eth0" {
                        assert_ne!(raw[0] & 0x80, 0, "eth0 should have IFAC");
                    } else {
                        assert_eq!(raw[0] & 0x80, 0, "wlan0 should not have IFAC");
                    }
                }
                _ => panic!("expected SendOnInterface"),
            }
        }
    }

    #[test]
    fn route_packet_stale_path_reports_unknown_interface() {
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        // Learn a path on eth0
        let (raw, dh) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Unregister eth0 — path entry now references a stale interface
        node.unregister_interface("eth0");

        let actions = node.route_packet(&dh, vec![0xDE, 0xAD]);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped {
                reason,
                interface_name,
            } => {
                assert_eq!(*reason, DropReason::UnknownInterface);
                assert_eq!(&**interface_name, "eth0");
            }
            _ => panic!("expected PacketDropped for stale interface"),
        }
    }

    // ── 14. Edge cases ──────────────────────────────────────────────

    #[test]
    fn unregister_then_announce_fails() {
        let (mut node, dh) = make_announcing_node();
        node.unregister_announcing_destination(&dh);

        let actions = node.announce(&dh, &mut OsRng, 1_700_000_000);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::UnknownDestination);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn unregister_destination_also_removes_announcing() {
        let (mut node, dh) = make_announcing_node();
        assert_eq!(node.announcing_destination_count(), 1);

        // Use the generic unregister_destination (not unregister_announcing_destination)
        node.unregister_destination(&dh);

        assert_eq!(node.announcing_destination_count(), 0);
        assert_eq!(node.destination_count(), 0);
    }

    #[test]
    fn jitter_derived_from_dest_hash() {
        // Verify jitter = dest_hash[0] % 60 and next_announce_at = now + interval + jitter
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("app1", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(identity, name, vec![], Some(300), 0);

        // Fire timer to trigger announce and advance the schedule
        node.handle_event(NodeEvent::TimerTick { now: 0 });

        let expected_jitter = (dh[0] as u64) % 60;
        let ad = &node.announcing_destinations[&dh];
        assert_eq!(ad.next_announce_at, 300 + expected_jitter);
        assert!(expected_jitter < 60);
    }

    // ── Transport-mode test helpers ──────────────────────────────────

    /// Create a transport node with "eth0" and "wlan0". Returns (node, transport_hash).
    fn make_transport_node() -> (Node, [u8; 16]) {
        let transport_id = PrivateIdentity::generate(&mut OsRng);
        let transport_hash = transport_id.public_identity().address_hash;
        let mut node = Node::new_transport(transport_id);
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);
        (node, transport_hash)
    }

    /// Create a transport node with rate limiting on "eth0".
    fn make_transport_node_with_rate(target: u64, grace: u32, penalty: u64) -> (Node, [u8; 16]) {
        let (mut node, th) = make_transport_node();
        node.set_announce_rate(
            "eth0",
            AnnounceRateConfig {
                target,
                grace,
                penalty,
            },
        );
        (node, th)
    }

    // ── 15. Transport mode ───────────────────────────────────────────

    #[test]
    fn transport_node_created_with_identity() {
        let (node, th) = make_transport_node();
        assert!(node.is_transport());
        assert_eq!(node.transport_identity_hash(), Some(th));
    }

    #[test]
    fn leaf_node_is_not_transport() {
        let node = Node::new();
        assert!(!node.is_transport());
        assert_eq!(node.transport_identity_hash(), None);
    }

    // ── 16. Announce table insertion ─────────────────────────────────

    #[test]
    fn transport_queues_announce_for_forwarding() {
        let (mut node, _th) = make_transport_node();
        let (raw, _dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(node.announce_table_len(), 1);
    }

    #[test]
    fn leaf_does_not_populate_announce_table() {
        let mut node = make_node_with_interface();
        let (raw, _dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(node.announce_table_len(), 0);
    }

    #[test]
    fn local_dest_announce_not_forwarded() {
        let (mut node, _th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        // Register the destination locally — transport should not queue it
        node.register_destination(dh);

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        assert_eq!(node.announce_table_len(), 0);
    }

    // ── 17. Retransmit scheduling ────────────────────────────────────

    #[test]
    fn timer_retransmits_due_announce_as_header2() {
        let (mut node, th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);

        // Tick far enough ahead for the initial delay to pass
        let actions = node.handle_event(NodeEvent::TimerTick { now: 1010 });

        // Should have SendOnInterface actions (on wlan0 only, excluding source eth0)
        // plus an AnnounceRebroadcast action
        let sends: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                NodeAction::SendOnInterface {
                    interface_name,
                    raw,
                    ..
                } => Some((&**interface_name, raw.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            sends.len(),
            1,
            "should broadcast on 1 interface (excluding source)"
        );
        assert_eq!(sends[0].0, "wlan0");

        // Verify the rebroadcast bytes are a valid HEADER_2 packet
        let pkt = Packet::from_bytes(&sends[0].1).unwrap();
        assert_eq!(pkt.header.flags.header_type, HeaderType::Type2);
        assert_eq!(pkt.header.flags.propagation, PropagationType::Transport);
        assert_eq!(pkt.header.flags.packet_type, PacketType::Announce);
        assert_eq!(pkt.header.transport_id, Some(th));
        assert_eq!(pkt.header.destination_hash, dh);

        // AnnounceRebroadcast action present
        let rebroadcasts: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, NodeAction::AnnounceRebroadcast { .. }))
            .collect();
        assert_eq!(rebroadcasts.len(), 1);
    }

    #[test]
    fn timer_retransmit_excludes_source_interface() {
        let transport_id = PrivateIdentity::generate(&mut OsRng);
        let mut node = Node::new_transport(transport_id);
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);
        node.register_interface("lora0".into(), InterfaceMode::Full, None);

        let (raw, _dh) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        let actions = node.handle_event(NodeEvent::TimerTick { now: 1010 });
        let sends: Vec<_> = actions
            .iter()
            .filter_map(|a| match a {
                NodeAction::SendOnInterface { interface_name, .. } => Some(&**interface_name),
                _ => None,
            })
            .collect();
        // Should send on wlan0 and lora0, but NOT eth0
        assert_eq!(sends.len(), 2);
        assert!(!sends.contains(&"eth0"));
    }

    #[test]
    fn timer_removes_after_max_retries() {
        let (mut node, _th) = make_transport_node();
        let (raw, _dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);

        // Tick 1: retries 0→1 (first transmit, retries == PATHFINDER_R)
        node.handle_event(NodeEvent::TimerTick { now: 1010 });
        assert_eq!(node.announce_table_len(), 1);

        // Tick 2: retries 1→2 (completion check is retries > PATHFINDER_R=1,
        //         which is false at start of tick; retransmit bumps to 2)
        node.handle_event(NodeEvent::TimerTick { now: 1020 });
        assert_eq!(node.announce_table_len(), 1);

        // Tick 3: completion check sees retries=2 > PATHFINDER_R=1 → removed
        node.handle_event(NodeEvent::TimerTick { now: 1030 });
        assert_eq!(node.announce_table_len(), 0);
    }

    #[test]
    fn timer_retransmit_deterministic_jitter() {
        let (mut node, _th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Initial retransmit_at = 1000 + (dh[1] % 5)
        let expected_initial = 1000 + (dh[1] as u64) % 5;
        let entry = &node.announce_table[&dh];
        assert_eq!(entry.retransmit_at, expected_initial);

        // After first retransmit, next retransmit_at = tick_time + PATHFINDER_G + (dh[2] % 3)
        let tick_time = expected_initial;
        node.handle_event(NodeEvent::TimerTick { now: tick_time });

        let expected_next = tick_time + PATHFINDER_G + (dh[2] as u64 % 3);
        let entry = &node.announce_table[&dh];
        assert_eq!(entry.retransmit_at, expected_next);
    }

    // ── 18. Rebroadcast detection ────────────────────────────────────

    #[test]
    fn echo_increments_local_rebroadcasts() {
        let (mut node, _th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        // Receive on eth0, queued with hops=1 (incremented from 0)
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);
        assert_eq!(node.announce_table[&dh].local_rebroadcasts, 0);

        // Retransmit so retries > 0
        node.handle_event(NodeEvent::TimerTick { now: 1010 });
        assert!(node.announce_table[&dh].retries > 0);

        // Echo: our rebroadcast (hops=1) arrives back on wire with hops=1,
        // our pipeline increments to 2. So incoming_hops=2, and
        // incoming_hops-1 = 1 == stored_hops(1) → echo detected.
        node.check_rebroadcast_echo(dh, 2);
        assert_eq!(node.announce_table[&dh].local_rebroadcasts, 1);
    }

    #[test]
    fn echo_at_max_removes_entry() {
        let (mut node, _th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Trigger a retransmit so retries > 0
        node.handle_event(NodeEvent::TimerTick { now: 1010 });

        // Simulate LOCAL_REBROADCASTS_MAX echoes (incoming_hops=2 for echo)
        for _ in 0..LOCAL_REBROADCASTS_MAX {
            node.check_rebroadcast_echo(dh, 2);
        }

        // Entry should be removed after reaching max
        assert_eq!(node.announce_table_len(), 0);
    }

    #[test]
    fn downstream_forward_removes_entry() {
        let (mut node, _th) = make_transport_node();
        let (raw, dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Retransmit so retries > 0
        node.handle_event(NodeEvent::TimerTick { now: 1010 });

        // Downstream forward: another transport picked it up (hops = stored+1)
        // stored hops = 1, so downstream forward has incoming_hops such that
        // incoming_hops - 1 == stored_hops + 1 → incoming_hops = stored_hops + 2 = 3
        let stored_hops = node.announce_table[&dh].hops;
        node.check_rebroadcast_echo(dh, stored_hops + 2);

        assert_eq!(node.announce_table_len(), 0);
    }

    // ── 19. Rate limiting ────────────────────────────────────────────

    #[test]
    fn rate_first_announce_passes() {
        let (mut node, _th) = make_transport_node_with_rate(60, 2, 120);
        let (raw, _dh) = make_valid_announce();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Should not be rate limited
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::AnnounceReceived { .. })));
    }

    #[test]
    fn rate_under_target_triggers_violation() {
        let (mut node, _th) = make_transport_node_with_rate(60, 2, 120);

        // First announce
        let (raw1, dh1) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw1,
            now: 1000,
        });

        // Second announce for same dest within target — need fresh announce
        // but same destination. Since make_valid_announce generates new identities,
        // we test rate limiting at the is_rate_limited level.
        assert!(!node.is_rate_limited(
            dh1,
            &AnnounceRateConfig {
                target: 60,
                grace: 2,
                penalty: 120,
            },
            1010,
        ));
        // This one is under target (10s < 60s)
        let rate_entry = node.announce_rate_table.get(&dh1).unwrap();
        assert_eq!(rate_entry.rate_violations, 1);
    }

    #[test]
    fn rate_grace_exhausted_blocks() {
        let (mut node, _th) = make_transport_node_with_rate(60, 1, 120);
        let dh = dest(1);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 1,
            penalty: 120,
        };

        // First — allowed, records last_checked
        assert!(!node.is_rate_limited(dh, &config, 1000));
        // Second — violation 1 (under grace of 1)
        assert!(!node.is_rate_limited(dh, &config, 1010));
        // Third — violation 2 > grace 1 → blocked
        assert!(node.is_rate_limited(dh, &config, 1020));
    }

    #[test]
    fn rate_block_expires_after_penalty() {
        let (mut node, _th) = make_transport_node_with_rate(60, 0, 120);
        let dh = dest(2);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 0,
            penalty: 120,
        };

        // First — allowed
        assert!(!node.is_rate_limited(dh, &config, 1000));
        // Second — violation 1 > grace 0 → blocked, blocked_until = 1010 + 60 + 120 = 1190
        assert!(node.is_rate_limited(dh, &config, 1010));
        // Still blocked at 1100
        assert!(node.is_rate_limited(dh, &config, 1100));
        // Unblocked at 1190 (blocked_until = 1010 + 60 + 120)
        assert!(!node.is_rate_limited(dh, &config, 1190));
    }

    #[test]
    fn rate_no_config_no_limiting() {
        let (mut node, _th) = make_transport_node();
        // No rate config on any interface

        let (raw, dh) = make_valid_announce();
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Should pass through without rate limiting
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::AnnounceReceived { .. })));
        assert!(node.announce_rate_table.is_empty());
        // But announce table should still be populated (transport mode)
        assert_eq!(node.announce_table_len(), 1);
        assert!(node.announce_table.contains_key(&dh));
    }

    // ── 20. Edge cases ───────────────────────────────────────────────

    #[test]
    fn duplicate_blob_not_forwarded() {
        let (mut node, _th) = make_transport_node();
        let (raw, _dh) = make_valid_announce();

        // First — inserted, queued for forwarding
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);

        // Same announce again — DuplicateBlob path_update, should not re-queue
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1001,
        });
        match &actions[0] {
            NodeAction::AnnounceReceived { path_update, .. } => {
                assert_eq!(*path_update, PathUpdateResult::DuplicateBlob);
            }
            _ => panic!("expected AnnounceReceived"),
        }
        // Still only 1 entry (not re-queued, and already existed so not inserted again)
        assert_eq!(node.announce_table_len(), 1);
    }

    #[test]
    fn announce_table_count_accessor() {
        let (mut node, _th) = make_transport_node();
        assert_eq!(node.announce_table_len(), 0);

        let (raw, _dh) = make_valid_announce();
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);
    }

    #[test]
    fn set_announce_rate_returns_false_for_unknown() {
        let (mut node, _th) = make_transport_node();
        assert!(!node.set_announce_rate(
            "nonexistent",
            AnnounceRateConfig {
                target: 60,
                grace: 2,
                penalty: 120,
            },
        ));
    }

    // ── 21. Regression tests (PR #8 review feedback) ────────────────

    #[test]
    fn rate_violations_reset_when_properly_spaced() {
        // Regression: violations previously never reset, causing permanent blocking.
        let (mut node, _th) = make_transport_node();
        let dh = dest(7);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 2,
            penalty: 120,
        };

        // First announce — always passes, sets last_checked
        assert!(!node.is_rate_limited(dh, &config, 1000));

        // Two rapid announces — accumulate 2 violations (at grace limit)
        assert!(!node.is_rate_limited(dh, &config, 1010)); // violation 1
        assert!(!node.is_rate_limited(dh, &config, 1020)); // violation 2

        // Now wait properly (>= target) — violations should reset to 0
        assert!(!node.is_rate_limited(dh, &config, 1100)); // elapsed=80 >= 60

        // Two more rapid announces should be tolerated (fresh grace budget)
        assert!(!node.is_rate_limited(dh, &config, 1110)); // violation 1 (fresh)
        assert!(!node.is_rate_limited(dh, &config, 1120)); // violation 2 (fresh)

        // Third rapid announce exceeds grace → blocked
        assert!(node.is_rate_limited(dh, &config, 1130));
    }

    #[test]
    fn rate_violations_reset_after_penalty_expiry() {
        // Regression: after penalty expired, a single violation would re-trigger
        // blocking because accumulated violations were never cleared.
        let (mut node, _th) = make_transport_node();
        let dh = dest(8);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 1,
            penalty: 120,
        };

        // Build up to block: announce, rapid, rapid → blocked
        assert!(!node.is_rate_limited(dh, &config, 1000));
        assert!(!node.is_rate_limited(dh, &config, 1010)); // violation 1 (at grace)
        assert!(node.is_rate_limited(dh, &config, 1020)); // violation 2 > grace → blocked
                                                          // blocked_until = 1020 + 60 + 120 = 1200

        // After penalty expires, should get a fresh grace budget
        assert!(!node.is_rate_limited(dh, &config, 1200)); // unblocked, resets
        assert!(!node.is_rate_limited(dh, &config, 1210)); // violation 1 (fresh grace)
                                                           // Should NOT be immediately blocked — grace is renewed
        assert!(node.is_rate_limited(dh, &config, 1220)); // violation 2 > grace → blocked again
    }

    #[test]
    fn serialization_failure_does_not_trap_entry() {
        // Regression: if to_bytes() failed, the entry stayed forever because
        // retries was never incremented.
        let (mut node, _transport_hash) = make_transport_node();

        // Craft an announce table entry with data that will exceed MTU as Type2.
        // Type1 max data = 500 - 19 = 481, Type2 max data = 500 - 35 = 465.
        // 470 bytes of data is valid as Type1 but overflows as Type2.
        let dh = dest(9);
        node.announce_table.insert(
            dh,
            AnnounceTableEntry {
                received_at: 1000,
                retransmit_at: 1000,
                retries: 0,
                local_rebroadcasts: 0,
                source_interface: "eth0".into(),
                hops: 1,
                destination_hash: dh,
                packet_data: vec![0u8; 470].into(), // will overflow Type2 MTU
            },
        );
        assert_eq!(node.announce_table_len(), 1);

        // Tick 1: retransmit_at <= now, serialization fails → retries becomes 1
        let actions = node.handle_event(NodeEvent::TimerTick { now: 1000 });
        // Should emit a serialize-failed drop
        assert!(actions.iter().any(|a| matches!(
            a,
            NodeAction::PacketDropped {
                reason: DropReason::AnnounceRebroadcastSerializeFailed,
                ..
            }
        )));
        // Entry still present (retries=1, not > PATHFINDER_R=1 yet)
        assert_eq!(node.announce_table_len(), 1);

        // Tick 2: retries becomes 2 > PATHFINDER_R=1 → entry removed on next tick
        let _actions = node.handle_event(NodeEvent::TimerTick { now: 1010 });

        // Tick 3: completion check fires (retries=2 > 1) → entry removed
        let _ = node.handle_event(NodeEvent::TimerTick { now: 1020 });
        assert_eq!(
            node.announce_table_len(),
            0,
            "entry should be cleaned up after retries exceed PATHFINDER_R"
        );
    }

    #[test]
    fn near_mtu_announce_emits_diagnostic_on_type2_overflow() {
        // Regression: near-MTU announces silently vanished during Type1→Type2
        // conversion. Now they emit AnnounceRebroadcastSerializeFailed.
        let (mut node, _th) = make_transport_node();
        let dh = dest(10);

        // 466 bytes of data: valid as Type1 (19+466=485 < 500) but
        // overflows Type2 (35+466=501 > 500).
        node.announce_table.insert(
            dh,
            AnnounceTableEntry {
                received_at: 1000,
                retransmit_at: 1000,
                retries: 0,
                local_rebroadcasts: 0,
                source_interface: "eth0".into(),
                hops: 1,
                destination_hash: dh,
                packet_data: vec![0xAA; 466].into(),
            },
        );

        let actions = node.handle_event(NodeEvent::TimerTick { now: 1000 });

        // Must have a diagnostic drop, not silence
        let has_drop = actions.iter().any(|a| {
            matches!(
                a,
                NodeAction::PacketDropped {
                    reason: DropReason::AnnounceRebroadcastSerializeFailed,
                    ..
                }
            )
        });
        assert!(
            has_drop,
            "expected AnnounceRebroadcastSerializeFailed action"
        );

        // Must NOT have a rebroadcast action (serialization failed)
        let has_rebroadcast = actions
            .iter()
            .any(|a| matches!(a, NodeAction::AnnounceRebroadcast { .. }));
        assert!(
            !has_rebroadcast,
            "should not rebroadcast on serialize failure"
        );
    }

    #[test]
    fn rate_table_entries_evicted_when_stale() {
        let (mut node, _th) = make_transport_node();
        let dh = dest(11);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 2,
            penalty: 120,
        };

        // Seed a rate entry at t=1000
        assert!(!node.is_rate_limited(dh, &config, 1000));
        assert_eq!(node.announce_rate_table.len(), 1);

        // Tick well before expiry — entry retained
        node.handle_event(NodeEvent::TimerTick {
            now: 1000 + RATE_TABLE_EXPIRY - 1,
        });
        assert_eq!(node.announce_rate_table.len(), 1);

        // Tick at expiry boundary — entry evicted (age == RATE_TABLE_EXPIRY, not <)
        node.handle_event(NodeEvent::TimerTick {
            now: 1000 + RATE_TABLE_EXPIRY,
        });
        assert_eq!(node.announce_rate_table.len(), 0);
    }

    #[test]
    fn rate_table_blocked_entry_retained_until_penalty_expires() {
        let (mut node, _th) = make_transport_node();
        let dh = dest(12);
        let config = AnnounceRateConfig {
            target: 60,
            grace: 0,
            penalty: RATE_TABLE_EXPIRY + 1000, // penalty outlasts expiry window
        };

        // Trigger a block: first pass, then immediate violation
        assert!(!node.is_rate_limited(dh, &config, 1000));
        assert!(node.is_rate_limited(dh, &config, 1010));
        // blocked_until = 1010 + 60 + (RATE_TABLE_EXPIRY + 1000)

        let blocked_until = 1010 + 60 + RATE_TABLE_EXPIRY + 1000;

        // Tick past RATE_TABLE_EXPIRY but before blocked_until — entry retained
        node.handle_event(NodeEvent::TimerTick {
            now: 1000 + RATE_TABLE_EXPIRY + 500,
        });
        assert_eq!(
            node.announce_rate_table.len(),
            1,
            "blocked entry must survive past age expiry"
        );

        // Tick after blocked_until AND past expiry — now evictable
        node.handle_event(NodeEvent::TimerTick {
            now: blocked_until + RATE_TABLE_EXPIRY,
        });
        assert_eq!(node.announce_rate_table.len(), 0);
    }

    #[test]
    fn echo_not_blocked_by_rate_limiter() {
        // Regression: rate limiting ran before echo detection, so echoes of our
        // own rebroadcasts consumed grace points. With grace=0, echoes were
        // blocked entirely, preventing echo-based completion.
        let (mut node, _th) = make_transport_node_with_rate(60, 0, 120);
        let (raw, dh) = make_valid_announce();

        // Receive announce on rate-limited interface at t=1000
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::AnnounceReceived { .. })));
        assert_eq!(node.announce_table_len(), 1);
        assert_eq!(node.announce_table[&dh].local_rebroadcasts, 0);
        // Stored hops = 1 (wire hops 0, pipeline incremented)

        // Retransmit so retries > 0
        node.handle_event(NodeEvent::TimerTick { now: 1010 });
        assert!(node.announce_table[&dh].retries > 0);

        // Build an echo: our rebroadcast (hops=1 on wire) heard back by a
        // neighbor who sends it back to us. Set wire hops=1 so our pipeline
        // increments to 2, giving incoming_hops-1 = 1 == stored_hops(1).
        let mut echo = raw.clone();
        echo[1] = 1; // hops byte on wire

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: echo,
            now: 1012,
        });

        // Should NOT be rate-limited (no AnnounceRateLimited drop)
        let was_rate_limited = actions.iter().any(|a| {
            matches!(
                a,
                NodeAction::PacketDropped {
                    reason: DropReason::AnnounceRateLimited,
                    ..
                }
            )
        });
        assert!(
            !was_rate_limited,
            "echoes of tracked announces must bypass rate limiting"
        );

        // Echo detection should have incremented local_rebroadcasts
        assert_eq!(
            node.announce_table[&dh].local_rebroadcasts, 1,
            "echo should have been detected and counted"
        );
    }

    #[test]
    fn echo_removal_does_not_trigger_rate_limiting() {
        // Regression: check_rebroadcast_echo could remove the announce table
        // entry, then the rate limiter guard (checking announce_table.contains_key)
        // would see the entry gone and rate-limit the very packet that was an echo.
        let (mut node, _th) = make_transport_node_with_rate(60, 0, 120);
        let (raw, dh) = make_valid_announce();

        // Receive original announce
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: raw.clone(),
            now: 1000,
        });
        assert_eq!(node.announce_table_len(), 1);

        // Retransmit so retries > 0
        node.handle_event(NodeEvent::TimerTick { now: 1010 });
        assert!(node.announce_table[&dh].retries > 0);

        // Send LOCAL_REBROADCASTS_MAX echoes through the full pipeline.
        // The final echo removes the entry. With grace=0, if rate limiting
        // ran after removal, that echo would get blocked.
        for i in 0..LOCAL_REBROADCASTS_MAX {
            let mut echo = raw.clone();
            echo[1] = 1; // wire hops=1 → pipeline to 2 → echo match
            let actions = node.handle_event(NodeEvent::InboundPacket {
                interface_name: "eth0".into(),
                raw: echo,
                now: 1012 + i as u64,
            });

            // No echo should ever be rate-limited
            let was_rate_limited = actions.iter().any(|a| {
                matches!(
                    a,
                    NodeAction::PacketDropped {
                        reason: DropReason::AnnounceRateLimited,
                        ..
                    }
                )
            });
            assert!(
                !was_rate_limited,
                "echo #{} should not be rate-limited (even if entry was just removed)",
                i + 1
            );
        }

        // Entry should be removed after max echoes
        assert_eq!(node.announce_table_len(), 0);
        // Rate table has exactly 1 entry — from the original announce at t=1000.
        // The echoes must not have added or modified any additional entries.
        assert_eq!(
            node.announce_rate_table.len(),
            1,
            "only the original announce should touch the rate table, not echoes"
        );
        // The entry should still show last_checked=1000 (untouched by echoes)
        let rate_entry = &node.announce_rate_table[&dh];
        assert_eq!(rate_entry.last_checked, 1000);
        assert_eq!(rate_entry.rate_violations, 0);
    }

    // ── Test Helpers (relay) ─────────────────────────────────────────

    /// Create a transport node with a path table entry for `remote_dest`,
    /// reachable via `next_hop` on "wlan0".
    fn make_relay_node(
        remote_dest: DestinationHash,
        next_hop: DestinationHash,
        hops: u8,
    ) -> (Node, [u8; 16]) {
        let (mut node, th) = make_transport_node();
        let blob = crate::path_table::timestamp_from_random_blob;
        let _ = blob; // just checking import; we build a blob inline
        let mut random_blob = [0u8; crate::path_table::RANDOM_BLOB_LENGTH];
        random_blob[0] = 1;
        let ts: u64 = 1000;
        let ts_bytes = ts.to_be_bytes();
        random_blob[5..10].copy_from_slice(&ts_bytes[3..8]);
        node.path_table.update(
            remote_dest,
            next_hop,
            hops,
            "wlan0".into(),
            dest(99),
            random_blob,
            InterfaceMode::Full,
            1000,
        );
        (node, th)
    }

    /// Build a Type2 data packet (raw bytes) addressed to a specific transport_id.
    fn make_type2_data_raw(
        transport_id: [u8; 16],
        dest_hash: DestinationHash,
        hops: u8,
    ) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops,
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0xDE, 0xAD].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a Type1 proof packet (raw bytes) with destination_hash set to
    /// the truncated hash of the original packet's hashable_part.
    fn make_proof_raw(proof_dest: DestinationHash, hops: u8) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Proof,
                },
                hops,
                transport_id: None,
                destination_hash: proof_dest,
                context: PacketContext::None,
            },
            data: vec![0xAA, 0x00, 0xFF].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Compute the reverse table key for a Type2 data packet.
    fn reverse_key_for(transport_id: [u8; 16], dest_hash: DestinationHash) -> DestinationHash {
        let pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0, // doesn't matter — hashable_part strips hops
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: vec![0xDE, 0xAD].into(),
        };
        hash::truncated_hash(&pkt.hashable_part().unwrap())
    }

    // ── 16. Path table next_hop fix ─────────────────────────────────

    #[test]
    fn path_table_stores_transport_id_as_next_hop() {
        // Simulate receiving a Type2 announce: transport_id present
        let (mut node, th) = make_transport_node();
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest_name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let announce_pkt =
            build_announce(&identity, &dest_name, &mut OsRng, 1_700_000_000, b"", None).unwrap();
        let dh = announce_pkt.header.destination_hash;

        // Wrap as Type2 with our transport_id
        let type2 = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    header_type: HeaderType::Type2,
                    propagation: PropagationType::Transport,
                    ..announce_pkt.header.flags
                },
                hops: 1,
                transport_id: Some(th),
                destination_hash: dh,
                context: announce_pkt.header.context,
            },
            data: announce_pkt.data,
        };
        let raw = type2.to_bytes().unwrap();

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Should produce AnnounceReceived
        let received = actions
            .iter()
            .find(|a| matches!(a, NodeAction::AnnounceReceived { .. }));
        assert!(received.is_some(), "expected AnnounceReceived");

        // Check path entry: next_hop should be transport_hash, not dest_hash
        let entry = node.path_table().get(&dh).unwrap();
        assert_eq!(
            entry.next_hop, th,
            "next_hop should be transport_id for Type2 announce"
        );
    }

    #[test]
    fn path_table_stores_dest_as_next_hop_for_type1() {
        let mut node = make_node_with_interface();
        let (raw, dh) = make_valid_announce();

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        let entry = node.path_table().get(&dh).unwrap();
        assert_eq!(
            entry.next_hop, dh,
            "next_hop should be dest_hash for Type1 announce"
        );
    }

    // ── 17. Relay basics ────────────────────────────────────────────

    #[test]
    fn transport_relays_type2_packet_addressed_to_it() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let has_send = actions
            .iter()
            .any(|a| matches!(a, NodeAction::SendOnInterface { .. }));
        let has_relay = actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketRelayed { .. }));
        assert!(has_send, "expected SendOnInterface");
        assert!(has_relay, "expected PacketRelayed");
    }

    #[test]
    fn transport_ignores_type2_not_addressed_to_it() {
        let remote = dest(42);
        let (mut node, _th) = make_relay_node(remote, remote, 3);
        let other_transport = dest(99);
        let raw = make_type2_data_raw(other_transport, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let dropped = actions
            .iter()
            .find(|a| matches!(a, NodeAction::PacketDropped { .. }));
        assert!(
            dropped.is_some(),
            "expected PacketDropped for wrong transport_id"
        );
    }

    #[test]
    fn leaf_does_not_relay() {
        let mut node = make_node_with_interface();
        let remote = dest(42);
        // Type1 data packet for unregistered dest
        let raw = make_data_packet_raw(remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        match &actions[0] {
            NodeAction::PacketDropped { reason, .. } => {
                assert_eq!(*reason, DropReason::NoLocalDestination);
            }
            _ => panic!("expected PacketDropped"),
        }
    }

    #[test]
    fn relay_local_delivery_takes_priority() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        node.register_destination(remote);

        let raw = make_type2_data_raw(th, remote, 0);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let delivered = actions
            .iter()
            .any(|a| matches!(a, NodeAction::DeliverLocally { .. }));
        assert!(delivered, "local delivery should take priority over relay");
    }

    // ── 18. Type2→Type1 conversion at final hop ─────────────────────

    #[test]
    fn final_hop_converts_to_type1() {
        let remote = dest(42);
        // next_hop == remote → final hop
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        assert!(send.is_some());
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        assert_eq!(forwarded.header.flags.header_type, HeaderType::Type1);
        assert_eq!(
            forwarded.header.flags.propagation,
            PropagationType::Broadcast
        );
    }

    #[test]
    fn final_hop_increments_hops() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 5);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        // Inbound pipeline increments by 1; relay preserves (matching Python)
        assert_eq!(forwarded.header.hops, 5 + 1);
    }

    #[test]
    fn final_hop_strips_transport_id() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        assert!(forwarded.header.transport_id.is_none());
    }

    // ── 19. Type2→Type2 intermediate relay ──────────────────────────

    #[test]
    fn intermediate_hop_keeps_type2() {
        let remote = dest(42);
        let next_transport = dest(77);
        // next_hop != remote → intermediate hop
        let (mut node, th) = make_relay_node(remote, next_transport, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        assert_eq!(forwarded.header.flags.header_type, HeaderType::Type2);
    }

    #[test]
    fn intermediate_hop_replaces_transport_id() {
        let remote = dest(42);
        let next_transport = dest(77);
        let (mut node, th) = make_relay_node(remote, next_transport, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        assert_eq!(
            forwarded.header.transport_id,
            Some(next_transport),
            "transport_id should be replaced with next transport"
        );
    }

    #[test]
    fn intermediate_hop_increments_hops() {
        let remote = dest(42);
        let next_transport = dest(77);
        let (mut node, th) = make_relay_node(remote, next_transport, 3);
        let raw = make_type2_data_raw(th, remote, 3);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        assert_eq!(forwarded.header.hops, 3 + 1); // inbound +1, relay preserves
    }

    // ── 20. Reverse table ───────────────────────────────────────────

    #[test]
    fn relay_creates_reverse_table_entry() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        assert_eq!(node.reverse_table_len(), 0);

        let raw = make_type2_data_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        assert_eq!(node.reverse_table_len(), 1);
    }

    #[test]
    fn reverse_table_entry_records_interfaces() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let key = reverse_key_for(th, remote);
        let entry = node
            .reverse_table
            .get(&key)
            .expect("reverse entry should exist");
        assert_eq!(&*entry.received_interface, "eth0");
        assert_eq!(&*entry.outbound_interface, "wlan0"); // path entry points to wlan0
    }

    #[test]
    fn reverse_table_expired_by_timer() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });
        assert_eq!(node.reverse_table_len(), 1);

        // Advance past REVERSE_TIMEOUT
        let actions = node.handle_event(NodeEvent::TimerTick {
            now: 1000 + REVERSE_TIMEOUT + 1,
        });

        assert_eq!(node.reverse_table_len(), 0);
        let expired = actions
            .iter()
            .any(|a| matches!(a, NodeAction::ReverseTableExpired { count: 1 }));
        assert!(expired, "expected ReverseTableExpired action");
    }

    #[test]
    fn reverse_table_fresh_entry_retained() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);
        let raw = make_type2_data_raw(th, remote, 0);

        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 1000,
        });

        // Advance to just before expiry
        node.handle_event(NodeEvent::TimerTick {
            now: 1000 + REVERSE_TIMEOUT - 1,
        });

        assert_eq!(
            node.reverse_table_len(),
            1,
            "fresh entry should be retained"
        );
    }

    #[test]
    fn failed_relay_does_not_create_reverse_entry() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);

        // Remove the outbound interface so relay will fail
        node.interfaces.remove("wlan0");
        assert_eq!(node.reverse_table_len(), 0);

        let raw = make_type2_data_raw(th, remote, 0);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        // Relay should fail (unknown outbound interface)
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. })));
        // Reverse table should remain empty
        assert_eq!(node.reverse_table_len(), 0);
    }

    // ── 21. Proof routing ───────────────────────────────────────────

    #[test]
    fn proof_routed_via_reverse_table() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);

        // First relay a data packet to create reverse entry
        let data_raw = make_type2_data_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: data_raw,
            now: 2000,
        });

        // Now send a proof with dest = reverse_key
        let key = reverse_key_for(th, remote);
        let proof_raw = make_proof_raw(key, 0);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        let relayed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::ProofRelayed { .. }));
        assert!(relayed, "expected ProofRelayed");

        // Should be sent on "eth0" (the received_interface of the reverse entry)
        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { interface_name, .. } => Some(&**interface_name),
            _ => None,
        });
        assert_eq!(
            send,
            Some("eth0"),
            "proof should go back on received interface"
        );
    }

    #[test]
    fn proof_no_reverse_entry_dropped() {
        let (mut node, _th) = make_transport_node();
        let unknown_key = dest(88);
        let proof_raw = make_proof_raw(unknown_key, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: proof_raw,
            now: 2000,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::ProofNoReverseEntry));
    }

    #[test]
    fn proof_increments_hops() {
        let remote = dest(42);
        let (mut node, th) = make_relay_node(remote, remote, 3);

        let data_raw = make_type2_data_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: data_raw,
            now: 2000,
        });

        let key = reverse_key_for(th, remote);
        let proof_raw = make_proof_raw(key, 2);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw),
            _ => None,
        });
        let forwarded = Packet::from_bytes(send.unwrap()).unwrap();
        // Inbound pipeline +1, route_proof preserves = original(2) + 1 = 3
        assert_eq!(forwarded.header.hops, 2 + 1);
    }

    // ── 22. Error cases ─────────────────────────────────────────────

    #[test]
    fn relay_no_path_drops_packet() {
        let (mut node, th) = make_transport_node();
        let unknown_dest = dest(50);
        // No path table entry for dest(50)
        let raw = make_type2_data_raw(th, unknown_dest, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::NoRouteForTransport));
    }

    #[test]
    fn relay_serialize_failure_drops() {
        let remote = dest(42);
        let next_transport = dest(77);
        // Intermediate relay: Type2→Type2 (keeps 35-byte header)
        let (node, _th) = make_relay_node(remote, next_transport, 3);

        // 35 + 470 = 505 > MTU(500) — will fail to_bytes()
        let oversized_pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: Some(next_transport),
                destination_hash: remote,
                context: PacketContext::None,
            },
            data: vec![0u8; 470].into(),
        };
        let path_entry = node.path_table().get(&remote).unwrap().clone();
        let eth0: Arc<str> = Arc::from("eth0");
        let actions = node.relay_packet(oversized_pkt, path_entry.next_hop, &eth0);

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::RelaySerializeFailed));
    }

    #[test]
    fn relay_send_failure_suppresses_relayed_action() {
        let remote = dest(42);
        let (node, _th) = make_relay_node(remote, remote, 3); // final hop

        let pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Data,
                },
                hops: 0,
                transport_id: Some(dest(99)),
                destination_hash: remote,
                context: PacketContext::None,
            },
            data: vec![0xBB; 10].into(),
        };
        // Use a path entry pointing to a non-existent interface
        let bad_entry = PathEntry {
            next_hop: remote,
            hops: 3,
            interface_name: "no_such_iface".into(),
            learned_at: 1000,
            expires_at: 2000,
            announce_packet_hash: dest(0),
            announce_timestamp: 1000,
            random_blobs: vec![],
        };
        let actions = node.relay_packet(pkt, bad_entry.next_hop, &bad_entry.interface_name);

        // Should get PacketDropped but NOT PacketRelayed
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. })));
        assert!(!actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketRelayed { .. })));
    }

    #[test]
    fn proof_send_failure_suppresses_relayed_action() {
        let (mut node, _th) = make_transport_node();
        let proof_key = dest(55);

        // Insert reverse table entry pointing to a non-existent interface
        node.reverse_table.insert(
            proof_key,
            ReverseTableEntry {
                received_interface: "no_such_iface".into(),
                outbound_interface: "wlan0".into(),
                timestamp: 1000,
                proof_received: false,
            },
        );

        let proof_pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Proof,
                },
                hops: 0,
                transport_id: None,
                destination_hash: proof_key,
                context: PacketContext::None,
            },
            data: vec![0xCC; 10].into(),
        };
        let eth0: Arc<str> = Arc::from("eth0");
        let actions = node.route_proof(&proof_pkt, &eth0, 2000);

        // Should get PacketDropped but NOT ProofRelayed
        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. })));
        assert!(!actions
            .iter()
            .any(|a| matches!(a, NodeAction::ProofRelayed { .. })));
    }

    #[test]
    fn relay_reverse_table_len_accessor() {
        let mut node = Node::new();
        assert_eq!(node.reverse_table_len(), 0);

        node.reverse_table.insert(
            dest(1),
            ReverseTableEntry {
                received_interface: "eth0".into(),
                outbound_interface: "wlan0".into(),
                timestamp: 1000,
                proof_received: false,
            },
        );
        assert_eq!(node.reverse_table_len(), 1);
    }

    // ── Link table test helpers ──────────────────────────────────────

    /// Create a transport node with "eth0" and "wlan0", with a path table
    /// entry for `remote_dest` reachable via next_hop on "wlan0" at given hops.
    /// Returns (node, transport_hash, dest_hash).
    fn make_link_relay_node(
        remote_dest: DestinationHash,
        next_hop: DestinationHash,
        hops: u8,
    ) -> (Node, [u8; 16]) {
        let (mut node, th) = make_transport_node();
        let mut random_blob = [0u8; crate::path_table::RANDOM_BLOB_LENGTH];
        random_blob[0] = 1;
        let ts: u64 = 1000;
        let ts_bytes = ts.to_be_bytes();
        random_blob[5..10].copy_from_slice(&ts_bytes[3..8]);
        node.path_table.update(
            remote_dest,
            next_hop,
            hops,
            "wlan0".into(),
            dest(99),
            random_blob,
            InterfaceMode::Full,
            1000,
        );
        (node, th)
    }

    /// Build a Type2 LinkRequest packet (raw bytes) addressed to a transport.
    fn make_link_request_raw(
        transport_id: [u8; 16],
        dest_hash: DestinationHash,
        hops: u8,
    ) -> Vec<u8> {
        // LinkRequest data: [x25519_pub:32][ed25519_pub:32][signalling:3] = 67 bytes
        let mut data = vec![0u8; 67];
        // Fill with deterministic but non-zero data so hashable_part is meaningful
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(dest_hash[0]);
        }
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops,
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: data.into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a Type1 LRPROOF packet (raw bytes).
    fn make_lrproof_raw(link_id: DestinationHash, hops: u8) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: true,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Proof,
                },
                hops,
                transport_id: None,
                destination_hash: link_id,
                context: PacketContext::LrProof,
            },
            data: vec![0xBB; 99].into(), // Proof data (signature + x25519 + signalling)
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a Type2 link data packet (raw bytes) with given dest_hash (link_id).
    fn make_link_data_raw(transport_id: [u8; 16], link_id: DestinationHash, hops: u8) -> Vec<u8> {
        Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Link,
                    packet_type: PacketType::Data,
                },
                hops,
                transport_id: Some(transport_id),
                destination_hash: link_id,
                context: PacketContext::None,
            },
            data: vec![0xCC; 20].into(),
        }
        .to_bytes()
        .unwrap()
    }

    /// Compute the link_id for a LinkRequest going through a transport node.
    fn link_id_for_request(
        transport_id: [u8; 16],
        dest_hash: DestinationHash,
        hops: u8,
    ) -> DestinationHash {
        let mut data = vec![0u8; 67];
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i as u8).wrapping_add(dest_hash[0]);
        }
        let pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type2,
                    context_flag: false,
                    propagation: PropagationType::Transport,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::LinkRequest,
                },
                hops,
                transport_id: Some(transport_id),
                destination_hash: dest_hash,
                context: PacketContext::None,
            },
            data: data.into(),
        };
        crate::link::link_id_from_request(&pkt).unwrap()
    }

    // ── Link ID helper ───────────────────────────────────────────────

    #[test]
    fn link_id_from_request_matches_initiate() {
        let responder_priv = PrivateIdentity::generate(&mut OsRng);
        let responder_pub = responder_priv.public_identity();
        let dest_name = DestinationName::from_name("testapp", &["link"]).unwrap();

        let (_link, request_packet) =
            crate::link::Link::initiate(&mut OsRng, responder_pub, &dest_name).unwrap();

        let helper_id = crate::link::link_id_from_request(&request_packet).unwrap();
        assert_eq!(helper_id, *_link.link_id());
    }

    // ── LinkRequest forwarding ───────────────────────────────────────

    #[test]
    fn transport_forwards_link_request() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);
        let raw = make_link_request_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let has_send = actions
            .iter()
            .any(|a| matches!(a, NodeAction::SendOnInterface { .. }));
        let has_forwarded = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkRequestForwarded { .. }));
        assert!(has_send, "expected SendOnInterface");
        assert!(has_forwarded, "expected LinkRequestForwarded");
    }

    #[test]
    fn link_request_creates_link_table_entry() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);
        assert_eq!(node.link_table_len(), 0);

        let raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        assert_eq!(node.link_table_len(), 1);

        let link_id = link_id_for_request(th, remote, 0);
        let entry = node.link_table.get(&link_id).expect("entry should exist");
        assert_eq!(&*entry.outbound_interface, "wlan0");
        assert_eq!(&*entry.received_interface, "eth0");
        assert_eq!(entry.remaining_hops, 3);
        assert_eq!(entry.taken_hops, 1); // hops 0 on wire, pipeline increments to 1
        assert!(!entry.validated);
    }

    #[test]
    fn link_request_no_route_drops() {
        let (mut node, th) = make_transport_node();
        // No path table entry for dest(50)
        let raw = make_link_request_raw(th, dest(50), 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::NoRouteForLinkRequest));
        assert_eq!(node.link_table_len(), 0);
    }

    #[test]
    fn link_request_failed_relay_no_entry() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        // Remove the outbound interface so relay will fail
        node.interfaces.remove("wlan0");

        let raw = make_link_request_raw(th, remote, 0);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        assert!(actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketDropped { .. })));
        assert_eq!(
            node.link_table_len(),
            0,
            "failed relay should not create link table entry"
        );
    }

    // ── Link proof routing ───────────────────────────────────────────

    #[test]
    fn link_proof_routes_via_link_table() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        // Forward a link request to create the entry
        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });
        assert_eq!(node.link_table_len(), 1);

        // Send LRPROOF with matching link_id and correct hops
        let link_id = link_id_for_request(th, remote, 0);
        let entry = node.link_table.get(&link_id).unwrap();
        let remaining_hops = entry.remaining_hops;

        // LRPROOF arrives on outbound_interface with remaining_hops
        // Wire hops = remaining_hops - 1 (pipeline will increment)
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        let routed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkProofRouted { .. }));
        assert!(routed, "expected LinkProofRouted");

        // Should be sent on "eth0" (the received_interface)
        let send = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { interface_name, .. } => Some(&**interface_name),
            _ => None,
        });
        assert_eq!(
            send,
            Some("eth0"),
            "proof should route back on received interface"
        );
    }

    #[test]
    fn link_proof_marks_validated() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;
        assert!(!node.link_table.get(&link_id).unwrap().validated);

        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        assert!(node.link_table.get(&link_id).unwrap().validated);
    }

    #[test]
    fn link_proof_no_entry_dropped() {
        let (mut node, _th) = make_transport_node();
        let unknown_link_id = dest(88);
        let proof_raw = make_lrproof_raw(unknown_link_id, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2000,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::LinkProofNoEntry));
    }

    #[test]
    fn link_proof_hops_mismatch_dropped() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);

        // Send proof with wrong hops (0 on wire → 1 after increment, but remaining=3)
        let proof_raw = make_lrproof_raw(link_id, 0);
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::LinkProofHopsMismatch));
    }

    #[test]
    fn link_proof_wrong_interface_dropped() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;

        // Send proof on "eth0" instead of "wlan0" (the outbound_interface)
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: proof_raw,
            now: 2001,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::LinkProofWrongInterface));
    }

    // ── Link data routing ────────────────────────────────────────────

    #[test]
    fn link_data_routes_initiator_to_destination() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        // Forward LinkRequest to create link table entry
        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let entry = node.link_table.get(&link_id).unwrap();
        // Mark validated so we know it's a live link
        let taken_hops = entry.taken_hops;

        // Validate the link (route a proof through)
        let remaining_hops = entry.remaining_hops;
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        // Send link data from initiator side (received_interface="eth0", taken_hops)
        // Wire hops = taken_hops - 1 (pipeline will increment)
        let data_raw = make_link_data_raw(th, link_id, taken_hops.saturating_sub(1));
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: data_raw,
            now: 2002,
        });

        let routed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkDataRouted { .. }));
        assert!(routed, "expected LinkDataRouted");

        // Should be sent on "wlan0" (toward destination)
        let send_action = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface {
                interface_name,
                raw,
                ..
            } => Some((&**interface_name, raw.clone())),
            _ => None,
        });
        let (iface, raw) = send_action.unwrap();
        assert_eq!(iface, "wlan0");

        // Final hop (next_hop == destination): should be converted to Type1
        assert_eq!(
            raw[0] & 0x40,
            0,
            "expected Type1 header (final hop toward destination)"
        );
    }

    #[test]
    fn link_data_routes_destination_to_initiator() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;

        // Validate the link
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        // Send link data from destination side (outbound_interface="wlan0", remaining_hops)
        // Wire hops = remaining_hops - 1 (pipeline will increment)
        let data_raw = make_link_data_raw(th, link_id, remaining_hops.saturating_sub(1));
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: data_raw,
            now: 2002,
        });

        let routed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkDataRouted { .. }));
        assert!(routed, "expected LinkDataRouted");

        // Should be sent on "eth0" (toward initiator)
        let send_action = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface {
                interface_name,
                raw,
                ..
            } => Some((&**interface_name, raw.clone())),
            _ => None,
        });
        let (iface, raw) = send_action.unwrap();
        assert_eq!(iface, "eth0");

        // Single-hop initiator (taken_hops==1): should be converted to Type1
        assert_eq!(
            raw[0] & 0x40,
            0,
            "expected Type1 header (final hop toward initiator)"
        );
    }

    #[test]
    fn link_data_updates_timestamp() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let entry = node.link_table.get(&link_id).unwrap();
        let remaining_hops = entry.remaining_hops;
        assert_eq!(entry.timestamp, 2000);

        // Validate the link
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        // Send link data
        let data_raw = make_link_data_raw(th, link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: data_raw,
            now: 3000,
        });

        assert_eq!(node.link_table.get(&link_id).unwrap().timestamp, 3000);
    }

    #[test]
    fn link_data_unknown_link_falls_through() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);
        // No link table entry — this is a normal data packet
        let raw = make_type2_data_raw(th, remote, 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        // Should fall through to normal path relay
        let relayed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::PacketRelayed { .. }));
        assert!(relayed, "should fall through to normal path relay");
    }

    #[test]
    fn link_data_equidistant_same_interface_routes_correctly() {
        // When taken_hops == remaining_hops on the same interface, direction
        // is ambiguous. Both endpoints are on the same broadcast medium, so
        // the transport should produce Type1/Broadcast regardless of direction.
        let remote = dest(42);
        let (mut node, th) = make_transport_node();

        // Path entry on "eth0" with hops=2 (same interface + same hops as request)
        let mut random_blob = [0u8; crate::path_table::RANDOM_BLOB_LENGTH];
        random_blob[0] = 1;
        let ts: u64 = 1000;
        let ts_bytes = ts.to_be_bytes();
        random_blob[5..10].copy_from_slice(&ts_bytes[3..8]);
        node.path_table.update(
            remote,
            remote, // next_hop == destination (final hop toward dest)
            2,      // remaining_hops = 2
            "eth0".into(),
            dest(99),
            random_blob,
            InterfaceMode::Full,
            1000,
        );

        // LinkRequest arrives on "eth0" with wire hops=1 (→ taken_hops=2 after increment)
        // This gives: outbound_interface="eth0", received_interface="eth0", taken_hops=2, remaining_hops=2
        let req_raw = make_link_request_raw(th, remote, 1);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 1);
        let entry = node.link_table.get(&link_id).unwrap();
        assert_eq!(
            entry.taken_hops, entry.remaining_hops,
            "precondition: equidistant"
        );
        assert_eq!(
            entry.outbound_interface, entry.received_interface,
            "precondition: same interface"
        );
        let equidistant_hops = entry.taken_hops;

        // Validate the link (proof arrives on "eth0" = outbound_interface)
        let proof_raw = make_lrproof_raw(link_id, equidistant_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: proof_raw,
            now: 2001,
        });

        // Send link data with matching hops (wire = equidistant_hops - 1, becomes equidistant_hops)
        let data_raw = make_link_data_raw(th, link_id, equidistant_hops.saturating_sub(1));
        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: data_raw,
            now: 2002,
        });

        let routed = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkDataRouted { .. }));
        assert!(routed, "equidistant link data should be routed");

        // Must produce Type1 header (final hop on shared medium)
        let send_action = actions.iter().find_map(|a| match a {
            NodeAction::SendOnInterface { raw, .. } => Some(raw.clone()),
            _ => None,
        });
        let raw = send_action.unwrap();
        assert_eq!(
            raw[0] & 0x40,
            0,
            "equidistant same-interface must produce Type1 (not Type2)"
        );
    }

    // ── Dedup exemption ──────────────────────────────────────────────

    #[test]
    fn lrproof_not_deduplicated() {
        let (mut node, th) = make_link_relay_node(dest(42), dest(42), 3);

        // Forward a link request first to create the entry
        let req_raw = make_link_request_raw(th, dest(42), 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, dest(42), 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;

        // Send same LRPROOF twice
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));

        let actions1 = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw.clone(),
            now: 2001,
        });
        let routed1 = actions1
            .iter()
            .any(|a| matches!(a, NodeAction::LinkProofRouted { .. }));
        assert!(routed1, "first LRPROOF should route");

        // Second identical LRPROOF — should NOT be deduplicated
        let actions2 = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2002,
        });
        let was_deduplicated = actions2.iter().any(|a| {
            matches!(
                a,
                NodeAction::PacketDropped {
                    reason: DropReason::DuplicatePacket,
                    ..
                }
            )
        });
        assert!(!was_deduplicated, "LRPROOF should not be deduplicated");
    }

    // ── Link table expiry ────────────────────────────────────────────

    #[test]
    fn validated_link_expires_after_timeout() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;

        // Validate the link
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });
        assert!(node.link_table.get(&link_id).unwrap().validated);

        // Timer tick past LINK_TIMEOUT
        let actions = node.handle_event(NodeEvent::TimerTick {
            now: 2001 + LINK_TIMEOUT + 1,
        });

        assert_eq!(node.link_table_len(), 0);
        let expired = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkTableExpired { count: 1 }));
        assert!(expired, "expected LinkTableExpired action");
    }

    #[test]
    fn unvalidated_link_expires_after_proof_timeout() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });
        assert_eq!(node.link_table_len(), 1);

        let link_id = link_id_for_request(th, remote, 0);
        let entry = node.link_table.get(&link_id).unwrap();
        assert!(!entry.validated);
        let proof_timeout = entry.proof_timeout;

        // Timer tick past proof_timeout
        let actions = node.handle_event(NodeEvent::TimerTick { now: proof_timeout });

        assert_eq!(node.link_table_len(), 0);
        let expired = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkTableExpired { count: 1 }));
        assert!(expired, "expected LinkTableExpired action");
    }

    #[test]
    fn active_link_kept_by_data_traffic() {
        let remote = dest(42);
        let (mut node, th) = make_link_relay_node(remote, remote, 3);

        let req_raw = make_link_request_raw(th, remote, 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });

        let link_id = link_id_for_request(th, remote, 0);
        let remaining_hops = node.link_table.get(&link_id).unwrap().remaining_hops;

        // Validate
        let proof_raw = make_lrproof_raw(link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: proof_raw,
            now: 2001,
        });

        // Send data traffic just before timeout would expire
        let refresh_time = 2001 + LINK_TIMEOUT - 10;
        let data_raw = make_link_data_raw(th, link_id, remaining_hops.saturating_sub(1));
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw: data_raw,
            now: refresh_time,
        });

        // Timer tick past original timeout but within refreshed window
        node.handle_event(NodeEvent::TimerTick {
            now: 2001 + LINK_TIMEOUT + 1,
        });
        assert_eq!(
            node.link_table_len(),
            1,
            "data traffic should keep link alive"
        );

        // Timer tick past refreshed timeout
        let actions = node.handle_event(NodeEvent::TimerTick {
            now: refresh_time + LINK_TIMEOUT + 1,
        });
        assert_eq!(
            node.link_table_len(),
            0,
            "link should expire after inactivity"
        );
        let expired = actions
            .iter()
            .any(|a| matches!(a, NodeAction::LinkTableExpired { .. }));
        assert!(expired);
    }

    // ── Edge cases ───────────────────────────────────────────────────

    #[test]
    fn leaf_node_drops_link_request() {
        let mut node = make_node_with_interface();
        // Build a Type2 LinkRequest addressed to some transport — but node is leaf
        let raw = make_link_request_raw(dest(99), dest(42), 0);

        let actions = node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw,
            now: 2000,
        });

        let dropped = actions.iter().find_map(|a| match a {
            NodeAction::PacketDropped { reason, .. } => Some(reason),
            _ => None,
        });
        assert_eq!(dropped, Some(&DropReason::NoLocalDestination));
    }

    #[test]
    fn link_table_len_accessor() {
        let (mut node, th) = make_link_relay_node(dest(42), dest(42), 3);
        assert_eq!(node.link_table_len(), 0);

        let req_raw = make_link_request_raw(th, dest(42), 0);
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "eth0".into(),
            raw: req_raw,
            now: 2000,
        });
        assert_eq!(node.link_table_len(), 1);
    }

    // ── Announce echo observation ───────────────────────────────────

    #[test]
    fn test_announce_echo_positive_observation() {
        // Set up: Node with 1 interface (eth0) and an announcing destination.
        // We announce (only on eth0), then add a second interface (wlan0).
        // When the echoed announce arrives on wlan0, wlan0 should get a
        // positive cooperation score because we did NOT send on wlan0.
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(
            identity,
            name,
            vec![],
            Some(300),
            1000,
        );

        // Emit announce — only eth0 is registered, so pending_echoes only has (eth0, dh).
        let actions = node.announce(&dh, &mut OsRng, 1_700_000_000);
        assert_eq!(actions.len(), 1, "should send on eth0 only");
        assert_eq!(node.pending_echoes_len(), 1);

        // Capture the raw announce bytes from the SendOnInterface action.
        let raw = match &actions[0] {
            NodeAction::SendOnInterface { raw, .. } => raw.clone(),
            _ => panic!("expected SendOnInterface"),
        };

        // Now register wlan0 (simulating a second interface appearing, or
        // equivalently, a neighbor forwarding our announce to a different segment).
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        // Verify initial cooperation score for wlan0 is 0.5 (default initial).
        // Note: with 2 interfaces, get_weight returns actual score, not forced 1.0.
        let initial_weight = node.cooperation().get_weight("wlan0");
        assert!(
            (initial_weight - 0.5).abs() < 1e-6,
            "wlan0 initial weight should be 0.5, got {initial_weight}"
        );

        // Feed the echoed announce on wlan0 — a neighbor forwarded our announce back.
        node.handle_event(NodeEvent::InboundPacket {
            interface_name: "wlan0".into(),
            raw,
            now: 1_700_000_001,
        });

        // Verify: wlan0's cooperation score increased above the initial 0.5.
        let weight_after = node.cooperation().get_weight("wlan0");
        assert!(
            weight_after > 0.5,
            "wlan0 cooperation score should have increased above 0.5 after \
             positive echo observation, got {weight_after}"
        );

        // Only the echoing interface's entry is removed (wlan0 had no entry since
        // it was registered after the announce). eth0's entry remains for independent
        // timeout tracking.
        assert_eq!(
            node.pending_echoes_len(),
            1,
            "eth0's pending echo should remain (wlan0 had no entry to remove)"
        );
    }

    #[test]
    fn test_announce_echo_timeout_negative_observation() {
        // Set up: Node with 1 interface (eth0) and an announcing destination.
        // We send an announce but no echo arrives. After the timeout, the
        // cooperation score for eth0 should decrease below the initial 0.5.
        let mut node = Node::new();
        node.register_interface("eth0".into(), InterfaceMode::Full, None);
        // Register a second interface so get_weight returns actual scores
        // (single-interface always returns 1.0).
        node.register_interface("wlan0".into(), InterfaceMode::Full, None);

        let identity = PrivateIdentity::generate(&mut OsRng);
        let name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let dh = node.register_announcing_destination(
            identity,
            name,
            vec![],
            Some(300),
            1000,
        );

        let now = 1_700_000_000_u64;

        // Emit announce — broadcasts on both eth0 and wlan0.
        let actions = node.announce(&dh, &mut OsRng, now);
        assert_eq!(actions.len(), 2, "should send on both interfaces");
        assert_eq!(node.pending_echoes_len(), 2);

        // Verify initial scores.
        let eth0_before = node.cooperation().get_weight("eth0");
        let wlan0_before = node.cooperation().get_weight("wlan0");
        assert!(
            (eth0_before - 0.5).abs() < 1e-6,
            "eth0 initial weight should be 0.5"
        );
        assert!(
            (wlan0_before - 0.5).abs() < 1e-6,
            "wlan0 initial weight should be 0.5"
        );

        // Tick past the echo timeout without any echo arriving.
        let timeout_now = now + ECHO_TIMEOUT + 1;
        node.handle_event(NodeEvent::TimerTick { now: timeout_now });

        // Pending echoes should be expired.
        assert_eq!(
            node.pending_echoes_len(),
            0,
            "pending echoes should be cleared after timeout"
        );

        // Both interfaces should have decreased cooperation scores.
        let eth0_after = node.cooperation().get_weight("eth0");
        let wlan0_after = node.cooperation().get_weight("wlan0");
        assert!(
            eth0_after < 0.5,
            "eth0 cooperation score should have decreased below 0.5 after \
             timeout, got {eth0_after}"
        );
        assert!(
            wlan0_after < 0.5,
            "wlan0 cooperation score should have decreased below 0.5 after \
             timeout, got {wlan0_after}"
        );
    }

    // ── Proof delivery cooperation observation tests ────────────────

    #[test]
    fn proof_delivered_increases_cooperation_score() {
        // Build a transport node with two interfaces so get_weight
        // returns actual scores (not the forced 1.0 for single-interface).
        let (mut node, _th) = make_transport_node();
        let now = 1000_u64;

        let initial = node.cooperation().get_weight("wlan0");
        assert!(
            (initial - 0.5).abs() < f32::EPSILON,
            "initial wlan0 weight should be 0.5, got {initial}"
        );

        // Simulate a proof arriving for a packet we forwarded on wlan0.
        // Insert a reverse table entry, then route a proof through it.
        let proof_key = dest(99);
        node.reverse_table.insert(
            proof_key,
            ReverseTableEntry {
                received_interface: "eth0".into(),
                outbound_interface: "wlan0".into(),
                timestamp: now,
                proof_received: false,
            },
        );

        let proof_pkt = Packet {
            header: PacketHeader {
                flags: PacketFlags {
                    ifac: false,
                    header_type: HeaderType::Type1,
                    context_flag: false,
                    propagation: PropagationType::Broadcast,
                    destination_type: DestinationType::Single,
                    packet_type: PacketType::Proof,
                },
                hops: 0,
                transport_id: None,
                destination_hash: proof_key,
                context: PacketContext::None,
            },
            data: vec![0xAA; 10].into(),
        };

        let eth0: Arc<str> = Arc::from("eth0");
        let _actions = node.route_proof(&proof_pkt, &eth0, now + 100);

        // wlan0 should have an increased cooperation score.
        let after = node.cooperation().get_weight("wlan0");
        assert!(
            after > 0.5,
            "wlan0 cooperation score should have increased above 0.5 after \
             proof delivery, got {after}"
        );

        // The reverse table entry should be marked as proof_received.
        assert!(
            node.reverse_table
                .get(&proof_key)
                .expect("entry should still exist")
                .proof_received,
            "proof_received flag should be set to true"
        );
    }

    #[test]
    fn proof_timeout_decreases_cooperation_score() {
        // Build a transport node with two interfaces.
        let (mut node, _th) = make_transport_node();
        let now = 1000_u64;

        let initial = node.cooperation().get_weight("wlan0");
        assert!(
            (initial - 0.5).abs() < f32::EPSILON,
            "initial wlan0 weight should be 0.5, got {initial}"
        );

        // Insert a reverse table entry for a packet forwarded on wlan0,
        // but never deliver a proof.
        node.reverse_table.insert(
            dest(88),
            ReverseTableEntry {
                received_interface: "eth0".into(),
                outbound_interface: "wlan0".into(),
                timestamp: now,
                proof_received: false,
            },
        );

        // Advance past REVERSE_TIMEOUT to trigger expiry.
        let timeout_now = now + REVERSE_TIMEOUT + 1;
        node.handle_event(NodeEvent::TimerTick { now: timeout_now });

        // The entry should be expired.
        assert_eq!(
            node.reverse_table_len(),
            0,
            "reverse table should be empty after timeout"
        );

        // wlan0 should have a decreased cooperation score.
        let after = node.cooperation().get_weight("wlan0");
        assert!(
            after < 0.5,
            "wlan0 cooperation score should have decreased below 0.5 after \
             proof timeout, got {after}"
        );
    }
}

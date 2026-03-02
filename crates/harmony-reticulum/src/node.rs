use std::collections::{HashMap, HashSet};
use std::fmt;

use harmony_crypto::hash;
use harmony_identity::identity::PrivateIdentity;
use rand_core::CryptoRngCore;

use crate::announce::{build_announce, validate_announce, ValidatedAnnounce};
use crate::context::PacketContext;
use crate::destination::DestinationName;
use crate::ifac::IfacAuthenticator;
use crate::interface::InterfaceMode;
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

/// Maximum timestamps tracked per destination for rate limiting.
const MAX_RATE_TIMESTAMPS: usize = 16;

// ── Types ───────────────────────────────────────────────────────────────

/// Per-interface announce rate limiting configuration.
#[derive(Debug, Clone)]
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
    timestamps: Vec<u64>,
}

/// An announce queued for transport retransmission.
struct AnnounceTableEntry {
    #[allow(dead_code)] // Retained for diagnostics
    received_at: u64,
    retransmit_at: u64,
    retries: u8,
    local_rebroadcasts: u8,
    source_interface: String,
    hops: u8,
    destination_hash: DestinationHash,
    packet_data: Vec<u8>,
    context_flag: bool,
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
}

/// Output actions for the caller.
#[derive(Debug)]
pub enum NodeAction {
    /// A valid announce was received and the path table was updated.
    AnnounceReceived {
        destination_hash: DestinationHash,
        validated_announce: Box<ValidatedAnnounce>,
        path_update: PathUpdateResult,
        interface_name: String,
        hops: u8,
    },
    /// A packet should be delivered to a locally registered destination.
    DeliverLocally {
        destination_hash: DestinationHash,
        packet: Packet,
        interface_name: String,
    },
    /// Expired paths were removed from the path table.
    PathsExpired { count: usize },
    /// A packet was dropped for the given reason.
    PacketDropped {
        reason: DropReason,
        interface_name: String,
    },
    /// A destination's scheduled announce is due. Caller should call node.announce().
    AnnounceNeeded {
        dest_hash: DestinationHash,
    },
    /// Send raw bytes on the named interface (outbound).
    SendOnInterface {
        interface_name: String,
        raw: Vec<u8>,
    },
    /// Transport node rebroadcast an announce on interfaces (excluding source).
    AnnounceRebroadcast {
        destination_hash: DestinationHash,
        hops: u8,
    },
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
    interfaces: HashMap<String, InterfaceConfig>,
    local_destinations: HashSet<DestinationHash>,
    announcing_destinations: HashMap<DestinationHash, AnnouncingDestination>,
    transport_identity: Option<PrivateIdentity>,
    announce_table: HashMap<DestinationHash, AnnounceTableEntry>,
    announce_rate_table: HashMap<DestinationHash, AnnounceRateEntry>,
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

    /// Register an interface by name and configuration.
    ///
    /// Takes ownership of the optional IFAC authenticator.
    pub fn register_interface(
        &mut self,
        name: String,
        mode: InterfaceMode,
        ifac: Option<IfacAuthenticator>,
    ) {
        self.interfaces.insert(
            name,
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
    pub fn announce(
        &self,
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
                        interface_name: String::new(),
                    }];
                }
            };
            let packet = match build_announce(
                &ad.identity,
                &ad.name,
                rng,
                now,
                &ad.app_data,
                None,
            ) {
                Ok(p) => p,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::AnnounceBuildFailed,
                        interface_name: String::new(),
                    }];
                }
            };
            match packet.to_bytes() {
                Ok(bytes) => bytes,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::OutboundSerializeFailed,
                        interface_name: String::new(),
                    }];
                }
            }
        };

        // Broadcast on all interfaces — separate borrow of self.interfaces.
        self.broadcast_on_all_interfaces(&raw)
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
                }

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
    fn send_on_interface(&self, interface_name: &str, raw: &[u8]) -> Vec<NodeAction> {
        let iface = match self.interfaces.get(interface_name) {
            Some(c) => c,
            None => {
                return vec![NodeAction::PacketDropped {
                    reason: DropReason::UnknownInterface,
                    interface_name: interface_name.to_owned(),
                }];
            }
        };
        let outbound = if let Some(ref auth) = iface.ifac {
            match auth.mask(raw) {
                Ok(masked) => masked,
                Err(_) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::OutboundIfacFailed,
                        interface_name: interface_name.to_owned(),
                    }];
                }
            }
        } else {
            raw.to_vec()
        };
        vec![NodeAction::SendOnInterface {
            interface_name: interface_name.to_owned(),
            raw: outbound,
        }]
    }

    /// Broadcast raw bytes on all registered interfaces, IFAC-masking each.
    fn broadcast_on_all_interfaces(&self, raw: &[u8]) -> Vec<NodeAction> {
        let mut actions = Vec::with_capacity(self.interfaces.len());
        for name in self.interfaces.keys() {
            actions.extend(self.send_on_interface(name, raw));
        }
        actions
    }

    /// Broadcast on all interfaces except the named one (source exclusion).
    fn broadcast_except(&self, exclude: &str, raw: &[u8]) -> Vec<NodeAction> {
        let mut actions = Vec::with_capacity(self.interfaces.len());
        for name in self.interfaces.keys() {
            if name != exclude {
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
        let entry = self.announce_rate_table.entry(dest).or_insert_with(|| {
            AnnounceRateEntry {
                last_checked: 0,
                rate_violations: 0,
                blocked_until: 0,
                timestamps: Vec::new(),
            }
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
                return true;
            }
        }

        // Record this timestamp
        entry.last_checked = now;
        if entry.timestamps.len() >= MAX_RATE_TIMESTAMPS {
            entry.timestamps.remove(0);
        }
        entry.timestamps.push(now);

        false
    }

    /// Detect local rebroadcast echo or downstream forward.
    fn check_rebroadcast_echo(&mut self, dest: DestinationHash, incoming_hops: u8) {
        let should_remove = if let Some(entry) = self.announce_table.get_mut(&dest) {
            // incoming_hops is post-increment (our pipeline already did +1)
            // entry.hops is stored post-increment too
            // Echo: we hear our own rebroadcast back (same hop count)
            if incoming_hops.saturating_sub(1) == entry.hops {
                entry.local_rebroadcasts += 1;
                entry.retries > 0
                    && entry.local_rebroadcasts >= LOCAL_REBROADCASTS_MAX
            }
            // Downstream forward: another transport picked it up (hops+1)
            else if incoming_hops.saturating_sub(1) == entry.hops.saturating_add(1) {
                entry.retries > 0
            } else {
                false
            }
        } else {
            false
        };

        if should_remove {
            self.announce_table.remove(&dest);
        }
    }

    /// Process announce table: remove completed entries, retransmit due entries.
    fn process_announce_table(&mut self, now: u64) -> Vec<NodeAction> {
        let transport_hash = match self.transport_identity_hash() {
            Some(h) => h,
            None => return Vec::new(),
        };

        // First pass: collect completed and due entries
        let mut completed = Vec::new();
        let mut due: Vec<(DestinationHash, Vec<u8>, String, u8)> = Vec::new();

        for (dest, entry) in &self.announce_table {
            // Completed: exceeded max retries, or retries>0 with enough rebroadcasts
            if entry.retries > PATHFINDER_R
                || (entry.retries > 0
                    && entry.local_rebroadcasts >= LOCAL_REBROADCASTS_MAX)
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
                            context_flag: entry.context_flag,
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

                if let Ok(raw) = packet.to_bytes() {
                    due.push((*dest, raw, entry.source_interface.clone(), entry.hops));
                }
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
                entry.retransmit_at =
                    now + PATHFINDER_G + (dest[2] as u64 % 3);
            }
            actions.extend(self.broadcast_except(&source_iface, &raw));
            actions.push(NodeAction::AnnounceRebroadcast {
                destination_hash: dest,
                hops,
            });
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
        let (processed_raw, interface_mode) = {
            let iface_config = match self.interfaces.get(&interface_name) {
                Some(config) => config,
                None => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::UnknownInterface,
                        interface_name,
                    }];
                }
            };

            let has_ifac_flag = raw.first().is_some_and(|&b| b & 0x80 != 0);

            let processed = match (iface_config.ifac.is_some(), has_ifac_flag) {
                (true, true) => {
                    match iface_config.ifac.as_ref().unwrap().unmask(&raw) {
                        Ok(unmasked) => unmasked,
                        Err(_) => {
                            return vec![NodeAction::PacketDropped {
                                reason: DropReason::IfacFailed,
                                interface_name,
                            }];
                        }
                    }
                }
                (true, false) | (false, true) => {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::IfacMismatch,
                        interface_name,
                    }];
                }
                (false, false) => raw,
            };

            (processed, iface_config.mode)
        };

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

        // Duplicate non-announces are dropped; duplicate announces still pass
        // through for path evaluation (multiple paths to same destination).
        if !self.packet_hashlist.insert(full_packet_hash.to_vec())
            && !is_announce
        {
            return vec![NodeAction::PacketDropped {
                reason: DropReason::DuplicatePacket,
                interface_name,
            }];
        }

        // 7. Dispatch by packet type
        if is_announce {
            self.process_announce(packet, &full_packet_hash, interface_name, interface_mode, now)
        } else {
            self.process_data_packet(packet, interface_name)
        }
    }

    fn process_announce(
        &mut self,
        packet: Packet,
        full_packet_hash: &[u8; 32],
        interface_name: String,
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
        let context_flag = packet.header.flags.context_flag;

        // Rate limiting (transport mode only)
        if self.is_transport() {
            // Clone the rate config (if any) to avoid holding a borrow on self.interfaces
            let rate_config = self
                .interfaces
                .get(&interface_name)
                .and_then(|c| c.announce_rate.clone());
            if let Some(config) = rate_config {
                if self.is_rate_limited(destination_hash, &config, now) {
                    return vec![NodeAction::PacketDropped {
                        reason: DropReason::AnnounceRateLimited,
                        interface_name,
                    }];
                }
            }
        }

        // Rebroadcast detection (transport mode only)
        if self.is_transport() {
            self.check_rebroadcast_echo(destination_hash, hops);
        }

        // Announce packet hash: truncated (first 16 bytes) of the full hash
        let mut announce_packet_hash: DestinationHash = [0u8; 16];
        announce_packet_hash.copy_from_slice(&full_packet_hash[..16]);

        // In leaf mode, next_hop is the destination itself
        let next_hop = destination_hash;

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

        // Announce table insertion (transport mode only)
        if self.is_transport()
            && !self.local_destinations.contains(&destination_hash)
            && !self.announce_table.contains_key(&destination_hash)
            && matches!(
                path_update,
                PathUpdateResult::Inserted
                    | PathUpdateResult::Updated
                    | PathUpdateResult::Kept
            )
        {
            let initial_delay = (destination_hash[1] as u64) % 2;
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
                    context_flag,
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
        &self,
        packet: Packet,
        interface_name: String,
    ) -> Vec<NodeAction> {
        let destination_hash = packet.header.destination_hash;

        if self.local_destinations.contains(&destination_hash) {
            vec![NodeAction::DeliverLocally {
                destination_hash,
                packet,
                interface_name,
            }]
        } else {
            vec![NodeAction::PacketDropped {
                reason: DropReason::NoLocalDestination,
                interface_name,
            }]
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
            data: vec![0xDE, 0xAD, 0xBE, 0xEF],
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
            data: vec![0x01, 0x02, 0x03],
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
            data: vec![0; 200],
        }
        .to_bytes()
        .unwrap()
    }

    /// Build a valid cryptographic announce and return (raw_bytes, dest_hash).
    fn make_valid_announce() -> (Vec<u8>, DestinationHash) {
        let identity = PrivateIdentity::generate(&mut OsRng);
        let dest_name = DestinationName::from_name("testapp", &["svc"]).unwrap();
        let packet = build_announce(
            &identity,
            &dest_name,
            &mut OsRng,
            1_700_000_000,
            b"",
            None,
        )
        .unwrap();
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
                assert_eq!(interface_name, "eth99");
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
                assert_eq!(interface_name, "eth0");
                assert_eq!(*hops, 1); // incremented from 0
            }
            _ => panic!("expected AnnounceReceived"),
        }

        // Verify path table
        let entry = node.path_table().get(&dest_hash).unwrap();
        assert_eq!(entry.hops, 1);
        assert_eq!(entry.interface_name, "eth0");
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
                assert_eq!(interface_name, "eth0");
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
                NodeAction::SendOnInterface { interface_name, .. } => interface_name.as_str(),
                _ => panic!("expected SendOnInterface"),
            })
            .collect();
        iface_names.sort();
        assert_eq!(iface_names, &["eth0", "wlan0"]);
    }

    #[test]
    fn announce_unknown_dest_returns_drop() {
        let node = Node::new();
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
        let (node, dh) = make_announcing_node_with_ifac();

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
                } => {
                    if interface_name == "eth0" {
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
            } => {
                assert_eq!(interface_name, "eth0");
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
                NodeAction::SendOnInterface { interface_name, .. } => interface_name.as_str(),
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
                } => {
                    if interface_name == "eth0" {
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
            NodeAction::PacketDropped { reason, interface_name } => {
                assert_eq!(*reason, DropReason::UnknownInterface);
                assert_eq!(interface_name, "eth0");
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
    fn make_transport_node_with_rate(
        target: u64,
        grace: u32,
        penalty: u64,
    ) -> (Node, [u8; 16]) {
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
                NodeAction::SendOnInterface { interface_name, raw } => {
                    Some((interface_name.as_str(), raw.clone()))
                }
                _ => None,
            })
            .collect();
        assert_eq!(sends.len(), 1, "should broadcast on 1 interface (excluding source)");
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
                NodeAction::SendOnInterface { interface_name, .. } => {
                    Some(interface_name.as_str())
                }
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

        // Initial retransmit_at = 1000 + (dh[1] % 2)
        let expected_initial = 1000 + (dh[1] as u64) % 2;
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
}

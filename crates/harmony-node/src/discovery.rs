use std::collections::{HashMap, HashSet};
use std::net::SocketAddr;
use std::time::{Duration, Instant};

use mdns_sd::{ServiceDaemon, ServiceEvent, ServiceInfo};

const SERVICE_TYPE: &str = "_harmony._udp.local.";

/// Information about a discovered LAN peer.
pub struct PeerInfo {
    pub reticulum_addr: [u8; 16],
    pub proto_version: u8,
    pub last_seen: Instant,
    /// Pinned peers (from config `[peers]`) are never evicted by stale timeout.
    pub pinned: bool,
}

/// Tracks LAN peers discovered via mDNS/DNS-SD.
///
/// Keyed by `SocketAddr` (IP + port) in the primary map; a reverse index
/// maps each Reticulum address to all socket addresses that announced it.
/// Self-announcements are silently dropped. Capped at `MAX_PEERS` to bound
/// send amplification from mDNS flooding.
pub struct PeerTable {
    pub(crate) peers: HashMap<SocketAddr, PeerInfo>,
    pub(crate) addr_to_sockets: HashMap<[u8; 16], HashSet<SocketAddr>>,
    our_addr: [u8; 16],
    stale_timeout: Duration,
}

/// Maximum number of tracked peers. Prevents O(N) send amplification if a
/// malicious actor floods mDNS with fake service advertisements. 128 is far
/// more than any realistic LAN mesh; real deployments are 5-30 nodes.
const MAX_PEERS: usize = 128;

impl PeerTable {
    pub fn new(our_addr: [u8; 16], stale_timeout: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            addr_to_sockets: HashMap::new(),
            our_addr,
            stale_timeout,
        }
    }

    /// Add or refresh a peer. No-op when `reticulum_addr == our_addr`.
    /// Returns `true` if the peer was newly inserted, `false` if an existing entry was updated.
    pub fn add_peer(
        &mut self,
        addr: SocketAddr,
        reticulum_addr: [u8; 16],
        proto_version: u8,
    ) -> bool {
        if reticulum_addr == self.our_addr {
            return false;
        }
        let is_new = !self.peers.contains_key(&addr);
        // Reject new peers beyond the cap to bound send amplification.
        // Existing peers (upsert) are always allowed through.
        if is_new && self.peers.len() >= MAX_PEERS {
            return false;
        }
        let now = Instant::now();
        let old_reticulum_addr = self.peers.get(&addr).map(|p| p.reticulum_addr);
        self.peers
            .entry(addr)
            .and_modify(|p| {
                p.last_seen = now;
                p.proto_version = proto_version;
                p.reticulum_addr = reticulum_addr;
            })
            .or_insert(PeerInfo {
                reticulum_addr,
                proto_version,
                last_seen: now,
                pinned: false,
            });
        // Clean up the old reverse-index entry if the Reticulum address changed.
        if let Some(old_ra) = old_reticulum_addr {
            if old_ra != reticulum_addr {
                if let Some(sockets) = self.addr_to_sockets.get_mut(&old_ra) {
                    sockets.remove(&addr);
                    if sockets.is_empty() {
                        self.addr_to_sockets.remove(&old_ra);
                    }
                }
            }
        }
        self.addr_to_sockets
            .entry(reticulum_addr)
            .or_default()
            .insert(addr);
        is_new
    }

    /// Add a pinned peer that is never evicted by stale timeout.
    /// Uses a unique placeholder Reticulum address derived from the socket address
    /// to avoid reverse-index collisions between bootstrap peers.
    pub fn add_pinned_peer(&mut self, addr: SocketAddr) {
        // Derive a unique placeholder via hash of the full SocketAddr.
        // Uniform for both IPv4 and IPv6 — no byte-overlap collisions.
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        addr.hash(&mut hasher);
        let hash = hasher.finish().to_le_bytes();
        let mut placeholder = [0xFEu8; 16]; // 0xFE prefix distinguishes from real addresses
        placeholder[..8].copy_from_slice(&hash);
        if placeholder == self.our_addr {
            tracing::warn!(peer = %addr, "bootstrap peer placeholder collides with our address — skipping");
            return;
        }
        if !self.peers.contains_key(&addr) && self.peers.len() >= MAX_PEERS {
            tracing::warn!(peer = %addr, "bootstrap peer not added: MAX_PEERS cap reached");
            return;
        }
        // Only update reverse index if we actually insert (not if already present).
        if self.peers.contains_key(&addr) {
            return;
        }
        let now = Instant::now();
        self.peers.insert(
            addr,
            PeerInfo {
                reticulum_addr: placeholder,
                proto_version: 0,
                last_seen: now,
                pinned: true,
            },
        );
        self.addr_to_sockets
            .entry(placeholder)
            .or_default()
            .insert(addr);
    }

    /// Remove all socket addrs associated with `reticulum_addr`.
    /// Returns the number of socket entries removed.
    pub fn remove_by_reticulum_addr(&mut self, reticulum_addr: &[u8; 16]) -> usize {
        let Some(sockets) = self.addr_to_sockets.remove(reticulum_addr) else {
            return 0;
        };
        let count = sockets.len();
        for sa in &sockets {
            self.peers.remove(sa);
        }
        count
    }

    /// Update `last_seen` for a known socket address. No-op for unknown addrs.
    pub fn mark_seen(&mut self, src_addr: &SocketAddr) {
        if let Some(info) = self.peers.get_mut(src_addr) {
            info.last_seen = Instant::now();
        }
    }

    /// Refresh `last_seen` for all non-pinned peers. Called periodically while
    /// mDNS is active — the mDNS daemon tracks service TTLs and will emit
    /// `ServiceRemoved` if a peer actually goes down, so stale eviction is a
    /// secondary safety net, not the primary removal mechanism.
    pub fn refresh_mdns_peers(&mut self) {
        let now = Instant::now();
        for info in self.peers.values_mut() {
            if !info.pinned {
                info.last_seen = now;
            }
        }
    }

    /// Remove peers whose `last_seen` is older than `stale_timeout`.
    /// Cleans the reverse index as well. Returns the evicted socket addresses.
    pub fn evict_stale(&mut self) -> Vec<SocketAddr> {
        let now = Instant::now();
        let timeout = self.stale_timeout;

        // Collect (socket_addr, reticulum_addr) pairs that are stale so we can
        // clean both maps without borrow-checker conflicts.
        let stale: Vec<(SocketAddr, [u8; 16])> = self
            .peers
            .iter()
            .filter(|(_, info)| !info.pinned && now.duration_since(info.last_seen) > timeout)
            .map(|(sa, info)| (*sa, info.reticulum_addr))
            .collect();

        let evicted: Vec<SocketAddr> = stale.iter().map(|(sa, _)| *sa).collect();

        for (sa, ra) in &stale {
            self.peers.remove(sa);
            if let Some(sockets) = self.addr_to_sockets.get_mut(ra) {
                sockets.remove(sa);
                if sockets.is_empty() {
                    self.addr_to_sockets.remove(ra);
                }
            }
        }

        evicted
    }

    /// Iterate all known peer socket addresses.
    pub fn peer_addrs(&self) -> impl Iterator<Item = &SocketAddr> {
        self.peers.keys()
    }

    /// Total number of tracked socket addresses.
    pub fn peer_count(&self) -> usize {
        self.peers.len()
    }
}

// ── mDNS/DNS-SD helpers ─────────────────────────────────────────────────────

/// Parse the Reticulum address from a DNS-SD TXT record property set.
///
/// Expects a property `addr=<32 hex chars>`. Returns `None` if missing or malformed.
pub fn parse_txt_addr(properties: &mdns_sd::TxtProperties) -> Option<[u8; 16]> {
    let val_str = properties.get_property_val_str("addr")?;
    let bytes = hex::decode(val_str).ok()?;
    if bytes.len() != 16 {
        return None;
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    Some(arr)
}

/// Parse the protocol version from a DNS-SD TXT record property set.
///
/// Expects a property `proto=<u8>`. Returns 0 if missing or malformed.
pub fn parse_txt_proto(properties: &mdns_sd::TxtProperties) -> u8 {
    properties
        .get_property_val_str("proto")
        .and_then(|s| s.parse::<u8>().ok())
        .unwrap_or(0)
}

/// Parse the Reticulum address from an mDNS instance fullname.
///
/// The fullname has the format `<32 hex chars>._harmony._udp.local.`
/// Returns `None` if the prefix is not valid hex or wrong length.
pub fn parse_instance_addr(fullname: &str) -> Option<[u8; 16]> {
    let prefix = fullname.split('.').next()?;
    if prefix.len() != 32 {
        return None;
    }
    let bytes = hex::decode(prefix).ok()?;
    if bytes.len() != 16 {
        return None;
    }
    let mut arr = [0u8; 16];
    arr.copy_from_slice(&bytes);
    Some(arr)
}

/// Start the mDNS daemon: register our service and browse for peers.
///
/// Returns the daemon handle (for shutdown) and the Receiver for events.
/// On failure (e.g., can't bind port 5353), returns `Err` — caller should log
/// and proceed without discovery.
pub fn start_mdns(
    listen_port: u16,
    reticulum_addr: &[u8; 16],
) -> Result<
    (ServiceDaemon, mdns_sd::Receiver<ServiceEvent>),
    Box<dyn std::error::Error + Send + Sync>,
> {
    let daemon = ServiceDaemon::new()?;
    let instance_name = hex::encode(reticulum_addr);
    let host = format!("{instance_name}.local.");

    let properties: [(&str, &str); 2] = [("addr", &instance_name), ("proto", "1")];

    let service = ServiceInfo::new(
        SERVICE_TYPE,
        &instance_name,
        &host,
        "",
        listen_port,
        &properties[..],
    )?
    .enable_addr_auto();

    daemon.register(service)?;
    let receiver = daemon.browse(SERVICE_TYPE)?;
    Ok((daemon, receiver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{Ipv4Addr, SocketAddr};
    use std::time::{Duration, Instant};

    fn make_addr(port: u16) -> SocketAddr {
        SocketAddr::new(std::net::IpAddr::V4(Ipv4Addr::new(192, 168, 1, 100)), port)
    }

    fn make_reticulum_addr(tag: u8) -> [u8; 16] {
        [tag; 16]
    }

    fn table_with_timeout(secs: u64) -> PeerTable {
        PeerTable::new([0u8; 16], Duration::from_secs(secs))
    }

    // ── 1. add_and_iterate_peers ──────────────────────────────────────────────
    #[test]
    fn add_and_iterate_peers() {
        let mut t = table_with_timeout(60);
        let sa = make_addr(4001);
        let ra = make_reticulum_addr(1);

        t.add_peer(sa, ra, 1);

        assert_eq!(t.peer_count(), 1);
        assert!(t.peer_addrs().any(|a| *a == sa));
        let info = t.peers.get(&sa).expect("peer should be in map");
        assert_eq!(info.reticulum_addr, ra);
        assert_eq!(info.proto_version, 1);
    }

    // ── 2. self_discovery_filtered ────────────────────────────────────────────
    #[test]
    fn self_discovery_filtered() {
        let our_addr = make_reticulum_addr(99);
        let mut t = PeerTable::new(our_addr, Duration::from_secs(60));
        let sa = make_addr(4002);

        t.add_peer(sa, our_addr, 1);

        assert_eq!(t.peer_count(), 0, "self-announcement must be dropped");
        assert!(t.addr_to_sockets.get(&our_addr).is_none());
    }

    // ── 3. idempotent_add ─────────────────────────────────────────────────────
    #[test]
    fn idempotent_add() {
        let mut t = table_with_timeout(60);
        let sa = make_addr(4003);
        let ra = make_reticulum_addr(2);

        t.add_peer(sa, ra, 1);
        t.add_peer(sa, ra, 2); // same socket, should upsert (not duplicate)

        assert_eq!(t.peer_count(), 1);
        let info = t.peers.get(&sa).unwrap();
        assert_eq!(
            info.proto_version, 2,
            "proto_version should be updated on upsert"
        );
        assert_eq!(
            t.addr_to_sockets.get(&ra).unwrap().len(),
            1,
            "reverse index should still have one entry"
        );
    }

    #[test]
    fn identity_rotation_cleans_reverse_index() {
        let mut t = table_with_timeout(60);
        let sa = make_addr(4004);
        let ra_old = make_reticulum_addr(10);
        let ra_new = make_reticulum_addr(11);

        t.add_peer(sa, ra_old, 1);
        assert_eq!(t.peer_count(), 1);
        assert!(t.addr_to_sockets.get(&ra_old).unwrap().contains(&sa));

        // Same socket, new Reticulum address (identity rotation)
        t.add_peer(sa, ra_new, 1);
        assert_eq!(t.peer_count(), 1);
        // Old reverse-index entry must be gone
        assert!(
            t.addr_to_sockets.get(&ra_old).is_none(),
            "old reticulum addr must be removed from reverse index"
        );
        // New reverse-index entry must exist
        assert!(t.addr_to_sockets.get(&ra_new).unwrap().contains(&sa));
        // PeerInfo must reflect the new address
        assert_eq!(t.peers.get(&sa).unwrap().reticulum_addr, ra_new);
    }

    // ── 4. multiple_addrs_per_reticulum_peer ──────────────────────────────────
    #[test]
    fn multiple_addrs_per_reticulum_peer() {
        let mut t = table_with_timeout(60);
        let ra = make_reticulum_addr(3);
        let sa1 = make_addr(4010);
        let sa2 = make_addr(4011);

        t.add_peer(sa1, ra, 1);
        t.add_peer(sa2, ra, 1);

        assert_eq!(t.peer_count(), 2);
        let sockets = t.addr_to_sockets.get(&ra).unwrap();
        assert!(sockets.contains(&sa1));
        assert!(sockets.contains(&sa2));
    }

    // ── 5. remove_by_reticulum_addr_removes_all_sockets ──────────────────────
    #[test]
    fn remove_by_reticulum_addr_removes_all_sockets() {
        let mut t = table_with_timeout(60);
        let ra = make_reticulum_addr(4);
        let sa1 = make_addr(4020);
        let sa2 = make_addr(4021);

        t.add_peer(sa1, ra, 1);
        t.add_peer(sa2, ra, 1);

        let removed = t.remove_by_reticulum_addr(&ra);

        assert_eq!(removed, 2);
        assert_eq!(t.peer_count(), 0);
        assert!(t.addr_to_sockets.get(&ra).is_none());
    }

    // ── 6. remove_by_reticulum_addr_unknown_is_zero ───────────────────────────
    #[test]
    fn remove_by_reticulum_addr_unknown_is_zero() {
        let mut t = table_with_timeout(60);
        let ra = make_reticulum_addr(5);

        let removed = t.remove_by_reticulum_addr(&ra);

        assert_eq!(removed, 0);
    }

    // ── 7. mark_seen_updates_timestamp ───────────────────────────────────────
    #[test]
    fn mark_seen_updates_timestamp() {
        let mut t = table_with_timeout(60);
        let sa = make_addr(4030);
        let ra = make_reticulum_addr(6);

        t.add_peer(sa, ra, 1);
        let before = t.peers.get(&sa).unwrap().last_seen;

        // Spin until at least one nanosecond has elapsed to guarantee Instant advances.
        let start = Instant::now();
        while Instant::now().duration_since(start) < Duration::from_nanos(1) {}

        t.mark_seen(&sa);
        let after = t.peers.get(&sa).unwrap().last_seen;

        assert!(
            after >= before,
            "last_seen should be non-decreasing after mark_seen"
        );
    }

    // ── 8. mark_seen_unknown_addr_is_noop ────────────────────────────────────
    #[test]
    fn mark_seen_unknown_addr_is_noop() {
        let mut t = table_with_timeout(60);
        let unknown = make_addr(9999);

        // Must not panic or insert.
        t.mark_seen(&unknown);
        assert_eq!(t.peer_count(), 0);
    }

    // ── 9. evict_stale_removes_old_peers ─────────────────────────────────────
    #[test]
    fn evict_stale_removes_old_peers() {
        // Use zero timeout so every peer is immediately stale.
        let mut t = PeerTable::new([0u8; 16], Duration::ZERO);
        let sa = make_addr(4040);
        let ra = make_reticulum_addr(7);

        t.add_peer(sa, ra, 1);

        // Tiny spin to ensure some time has elapsed.
        let start = Instant::now();
        while Instant::now().duration_since(start) < Duration::from_nanos(1) {}

        let evicted = t.evict_stale();

        assert!(evicted.contains(&sa));
        assert_eq!(t.peer_count(), 0);
        assert!(
            t.addr_to_sockets.get(&ra).is_none(),
            "reverse index must be cleaned"
        );
    }

    // ── 10. evict_stale_keeps_fresh_peers ────────────────────────────────────
    #[test]
    fn evict_stale_keeps_fresh_peers() {
        let mut t = PeerTable::new([0u8; 16], Duration::from_secs(3600));
        let sa = make_addr(4050);
        let ra = make_reticulum_addr(8);

        t.add_peer(sa, ra, 1);
        let evicted = t.evict_stale();

        assert!(evicted.is_empty(), "fresh peer must not be evicted");
        assert_eq!(t.peer_count(), 1);
    }

    // ── 11. evict_stale_cleans_reverse_index ─────────────────────────────────
    #[test]
    fn evict_stale_cleans_reverse_index() {
        // Zero timeout: both sockets for the same reticulum addr go stale.
        let mut t = PeerTable::new([0u8; 16], Duration::ZERO);
        let ra = make_reticulum_addr(9);
        let sa1 = make_addr(4060);
        let sa2 = make_addr(4061);

        t.add_peer(sa1, ra, 1);
        t.add_peer(sa2, ra, 1);

        // Spin to let time advance past zero timeout.
        let start = Instant::now();
        while Instant::now().duration_since(start) < Duration::from_nanos(1) {}

        let evicted = t.evict_stale();

        assert_eq!(evicted.len(), 2);
        assert_eq!(t.peer_count(), 0);
        // The reverse-index entry for ra must be gone entirely.
        assert!(
            t.addr_to_sockets.get(&ra).is_none(),
            "addr_to_sockets must be empty after all sockets for a reticulum addr evict"
        );
    }

    #[test]
    fn max_peers_cap_rejects_new_peers() {
        let our = [0xFF; 16]; // won't collide with generated addrs
        let mut t = PeerTable::new(our, Duration::from_secs(60));
        // Fill to MAX_PEERS
        for i in 0..super::MAX_PEERS {
            let sa = make_addr(5000 + i as u16);
            let mut ra = [0u8; 16];
            // Ensure ra != our_addr by setting byte 2 to a non-0xFF value
            ra[0] = (i >> 8) as u8;
            ra[1] = (i & 0xFF) as u8;
            ra[2] = 0x01;
            assert!(t.add_peer(sa, ra, 1), "peer {i} should be accepted");
        }
        assert_eq!(t.peer_count(), super::MAX_PEERS);

        // Next new peer should be rejected
        let extra = make_addr(9000);
        assert!(!t.add_peer(extra, make_reticulum_addr(200), 1));
        assert_eq!(t.peer_count(), super::MAX_PEERS);

        // But upsert of existing peer still works
        let existing = make_addr(5000);
        let mut ra0 = [0u8; 16];
        ra0[2] = 0x01;
        assert!(!t.add_peer(existing, ra0, 2)); // returns false (not new), but updates
        assert_eq!(t.peers.get(&existing).unwrap().proto_version, 2);
    }

    // ── pinned peer tests ─────────────────────────────────────────────────

    #[test]
    fn pinned_peer_survives_eviction() {
        let mut t = PeerTable::new([0u8; 16], Duration::ZERO);
        let sa = make_addr(6000);
        t.add_pinned_peer(sa);
        assert_eq!(t.peer_count(), 1);

        // Spin past zero timeout
        let start = Instant::now();
        while Instant::now().duration_since(start) < Duration::from_nanos(1) {}

        let evicted = t.evict_stale();
        assert!(evicted.is_empty(), "pinned peer must not be evicted");
        assert_eq!(t.peer_count(), 1);
    }

    #[test]
    fn pinned_peers_have_unique_placeholders() {
        let mut t = PeerTable::new([0u8; 16], Duration::from_secs(60));
        let sa1 = make_addr(6001);
        let sa2 = make_addr(6002);
        t.add_pinned_peer(sa1);
        t.add_pinned_peer(sa2);
        assert_eq!(t.peer_count(), 2);
        // Each should be in a separate reverse-index bucket
        assert_eq!(t.addr_to_sockets.len(), 2);
    }

    // ── parse_instance_addr tests ────────────────────────────────────────────

    #[test]
    fn parse_instance_addr_valid() {
        let hex_addr = "aabbccdd11223344aabbccdd11223344";
        let fullname = format!("{hex_addr}._harmony._udp.local.");
        let result = parse_instance_addr(&fullname);
        assert!(result.is_some());
        assert_eq!(hex::encode(result.unwrap()), hex_addr);
    }

    #[test]
    fn parse_instance_addr_invalid_hex() {
        let fullname = "not_valid_hex_here_0000000000000._harmony._udp.local.";
        assert!(parse_instance_addr(fullname).is_none());
    }

    #[test]
    fn parse_instance_addr_ignores_suffix() {
        let fullname = "aabbccdd11223344aabbccdd11223344._other._tcp.local.";
        assert!(parse_instance_addr(fullname).is_some());
    }

    #[test]
    fn parse_instance_addr_too_short() {
        let fullname = "aabb._harmony._udp.local.";
        assert!(parse_instance_addr(fullname).is_none());
    }
}

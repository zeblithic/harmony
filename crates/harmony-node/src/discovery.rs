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
    pub discovered_at: Instant,
}

/// Tracks LAN peers discovered via mDNS/DNS-SD.
///
/// Keyed by `SocketAddr` (IP + port) in the primary map; a reverse index
/// maps each Reticulum address to all socket addresses that announced it.
/// Self-announcements are silently dropped.
pub struct PeerTable {
    pub(crate) peers: HashMap<SocketAddr, PeerInfo>,
    pub(crate) addr_to_sockets: HashMap<[u8; 16], HashSet<SocketAddr>>,
    our_addr: [u8; 16],
    stale_timeout: Duration,
}

impl PeerTable {
    pub fn new(our_addr: [u8; 16], stale_timeout: Duration) -> Self {
        Self {
            peers: HashMap::new(),
            addr_to_sockets: HashMap::new(),
            our_addr,
            stale_timeout,
        }
    }

    /// Add or refresh a peer. Returns `true` if newly inserted, `false` if updated.
    /// No-op (returns `false`) when `reticulum_addr == our_addr`.
    pub fn add_peer(
        &mut self,
        addr: SocketAddr,
        reticulum_addr: [u8; 16],
        proto_version: u8,
    ) -> bool {
        if reticulum_addr == self.our_addr {
            return false;
        }
        let now = Instant::now();
        let is_new = !self.peers.contains_key(&addr);
        self.peers
            .entry(addr)
            .and_modify(|p| {
                p.last_seen = now;
                p.proto_version = proto_version;
            })
            .or_insert(PeerInfo {
                reticulum_addr,
                proto_version,
                last_seen: now,
                discovered_at: now,
            });
        self.addr_to_sockets
            .entry(reticulum_addr)
            .or_default()
            .insert(addr);
        is_new
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

    /// Remove peers whose `last_seen` is older than `stale_timeout`.
    /// Cleans the reverse index as well. Returns the evicted socket addresses.
    pub fn evict_stale(&mut self) -> Vec<SocketAddr> {
        let now = Instant::now();
        let timeout = self.stale_timeout;

        let stale: Vec<(SocketAddr, [u8; 16])> = self
            .peers
            .iter()
            .filter(|(_, info)| now.duration_since(info.last_seen) > timeout)
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

    let hex_addr = hex::encode(reticulum_addr);
    let properties: [(&str, &str); 2] = [("addr", &hex_addr), ("proto", "1")];

    let service = ServiceInfo::new(SERVICE_TYPE, &instance_name, &host, "", listen_port, &properties[..])?
        .enable_addr_auto();

    daemon.register(service)?;
    let receiver = daemon.browse(SERVICE_TYPE)?;
    Ok((daemon, receiver))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::{IpAddr, Ipv4Addr};

    fn our_addr() -> [u8; 16] {
        [0xAA; 16]
    }

    fn peer_addr_a() -> [u8; 16] {
        [0xBB; 16]
    }

    fn peer_addr_b() -> [u8; 16] {
        [0xCC; 16]
    }

    fn socket(port: u16) -> SocketAddr {
        SocketAddr::new(IpAddr::V4(Ipv4Addr::new(192, 168, 1, port as u8)), 4242)
    }

    // ── PeerTable tests ─────────────────────────────────────────────────────

    #[test]
    fn add_and_iterate_peers() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        table.add_peer(socket(11), peer_addr_b(), 1);
        assert_eq!(table.peer_count(), 2);
        let addrs: Vec<_> = table.peer_addrs().collect();
        assert!(addrs.contains(&&socket(10)));
        assert!(addrs.contains(&&socket(11)));
    }

    #[test]
    fn self_discovery_filtered() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        assert!(!table.add_peer(socket(10), our_addr(), 1));
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn idempotent_add() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        assert!(table.add_peer(socket(10), peer_addr_a(), 1));
        assert!(!table.add_peer(socket(10), peer_addr_a(), 1));
        assert_eq!(table.peer_count(), 1);
    }

    #[test]
    fn multiple_addrs_per_reticulum_peer() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let ipv6_addr = SocketAddr::new(IpAddr::V6("::1".parse().unwrap()), 4242);
        table.add_peer(ipv6_addr, peer_addr_a(), 1);
        assert_eq!(table.peer_count(), 2);
    }

    #[test]
    fn remove_by_reticulum_addr_removes_all_sockets() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let ipv6_addr = SocketAddr::new(IpAddr::V6("::1".parse().unwrap()), 4242);
        table.add_peer(ipv6_addr, peer_addr_a(), 1);
        assert_eq!(table.peer_count(), 2);

        let removed = table.remove_by_reticulum_addr(&peer_addr_a());
        assert_eq!(removed, 2);
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn remove_by_reticulum_addr_unknown_is_zero() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        let removed = table.remove_by_reticulum_addr(&peer_addr_a());
        assert_eq!(removed, 0);
    }

    #[test]
    fn mark_seen_updates_timestamp() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let before = table.peers.get(&socket(10)).unwrap().last_seen;
        std::thread::sleep(Duration::from_millis(2));
        table.mark_seen(&socket(10));
        let after = table.peers.get(&socket(10)).unwrap().last_seen;
        assert!(after > before);
    }

    #[test]
    fn mark_seen_unknown_addr_is_noop() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.mark_seen(&socket(99));
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn evict_stale_removes_old_peers() {
        let mut table = PeerTable::new(our_addr(), Duration::from_millis(10));
        table.add_peer(socket(10), peer_addr_a(), 1);
        std::thread::sleep(Duration::from_millis(20));
        let evicted = table.evict_stale();
        assert_eq!(evicted.len(), 1);
        assert_eq!(evicted[0], socket(10));
        assert_eq!(table.peer_count(), 0);
    }

    #[test]
    fn evict_stale_keeps_fresh_peers() {
        let mut table = PeerTable::new(our_addr(), Duration::from_secs(60));
        table.add_peer(socket(10), peer_addr_a(), 1);
        let evicted = table.evict_stale();
        assert!(evicted.is_empty());
        assert_eq!(table.peer_count(), 1);
    }

    #[test]
    fn evict_stale_cleans_reverse_index() {
        let mut table = PeerTable::new(our_addr(), Duration::from_millis(10));
        table.add_peer(socket(10), peer_addr_a(), 1);
        std::thread::sleep(Duration::from_millis(20));
        table.evict_stale();
        assert!(table.addr_to_sockets.is_empty());
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
    fn parse_instance_addr_wrong_suffix() {
        let fullname = "aabbccdd11223344aabbccdd11223344._other._tcp.local.";
        assert!(parse_instance_addr(fullname).is_some());
    }

    #[test]
    fn parse_instance_addr_too_short() {
        let fullname = "aabb._harmony._udp.local.";
        assert!(parse_instance_addr(fullname).is_none());
    }
}

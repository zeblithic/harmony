use std::collections::HashMap;
use std::time::{Duration, Instant};

/// A peer table mapping identity_hash to MAC address with TTL-based expiry.
pub struct PeerTable {
    entries: HashMap<[u8; 16], PeerEntry>,
    ttl: Duration,
}

struct PeerEntry {
    mac: [u8; 6],
    last_seen: Instant,
}

impl PeerTable {
    /// Create a new peer table with the specified TTL.
    ///
    /// # Arguments
    /// * `ttl` - Time-to-live for entries before they expire
    pub fn new(ttl: Duration) -> Self {
        PeerTable {
            entries: HashMap::new(),
            ttl,
        }
    }

    /// Insert or refresh a peer entry.
    ///
    /// # Arguments
    /// * `identity_hash` - 128-bit identity hash
    /// * `mac` - 48-bit MAC address
    pub fn update(&mut self, identity_hash: [u8; 16], mac: [u8; 6]) {
        self.entries.insert(
            identity_hash,
            PeerEntry {
                mac,
                last_seen: Instant::now(),
            },
        );
    }

    /// Look up a peer's MAC address.
    ///
    /// Returns `None` if the peer is unknown or if the entry has expired.
    ///
    /// # Arguments
    /// * `identity_hash` - 128-bit identity hash
    pub fn lookup(&self, identity_hash: &[u8; 16]) -> Option<[u8; 6]> {
        self.entries.get(identity_hash).and_then(|entry| {
            if entry.last_seen.elapsed() < self.ttl {
                Some(entry.mac)
            } else {
                None
            }
        })
    }

    /// Get the number of valid (non-expired) peers in the table.
    pub fn peer_count(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.last_seen.elapsed() < self.ttl)
            .count()
    }

    /// Check if a MAC address belongs to any known (non-expired) peer.
    pub fn lookup_by_mac(&self, mac: &[u8; 6]) -> bool {
        self.entries
            .values()
            .any(|entry| entry.mac == *mac && entry.last_seen.elapsed() < self.ttl)
    }

    /// Look up the identity hash associated with a MAC address.
    ///
    /// Returns `None` if the MAC is unknown or the entry has expired.
    /// Scans all entries (O(n)) — suitable for the small peer tables in mesh networks.
    pub fn identity_for_mac(&self, mac: &[u8; 6]) -> Option<[u8; 16]> {
        self.entries
            .iter()
            .find(|(_, entry)| entry.mac == *mac && entry.last_seen.elapsed() < self.ttl)
            .map(|(identity_hash, _)| *identity_hash)
    }

    /// Remove all expired entries from the table.
    pub fn purge_expired(&mut self) {
        let ttl = self.ttl;
        self.entries
            .retain(|_, entry| entry.last_seen.elapsed() < ttl);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn insert_and_lookup() {
        let mut table = PeerTable::new(Duration::from_secs(60));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        assert_eq!(table.lookup(&identity_hash), Some(mac));
    }

    #[test]
    fn unknown_returns_none() {
        let table = PeerTable::new(Duration::from_secs(60));
        let identity_hash = [1u8; 16];

        assert_eq!(table.lookup(&identity_hash), None);
    }

    #[test]
    fn expired_entry_returns_none() {
        let mut table = PeerTable::new(Duration::from_millis(1));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        thread::sleep(Duration::from_millis(10));

        assert_eq!(table.lookup(&identity_hash), None);
    }

    #[test]
    fn update_refreshes_expiry() {
        let mut table = PeerTable::new(Duration::from_millis(50));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        thread::sleep(Duration::from_millis(30));

        // Refresh the entry
        table.update(identity_hash, mac);
        thread::sleep(Duration::from_millis(30));

        // Should still be valid since we updated at 30ms mark
        assert_eq!(table.lookup(&identity_hash), Some(mac));
    }

    #[test]
    fn peer_count() {
        let mut table = PeerTable::new(Duration::from_secs(60));
        let identity_hash1 = [1u8; 16];
        let identity_hash2 = [2u8; 16];
        let mac1 = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x01];
        let mac2 = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0x02];

        assert_eq!(table.peer_count(), 0);

        table.update(identity_hash1, mac1);
        assert_eq!(table.peer_count(), 1);

        table.update(identity_hash2, mac2);
        assert_eq!(table.peer_count(), 2);
    }

    #[test]
    fn identity_for_mac_returns_hash() {
        let mut table = PeerTable::new(Duration::from_secs(60));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        assert_eq!(table.identity_for_mac(&mac), Some(identity_hash));
    }

    #[test]
    fn identity_for_mac_unknown_returns_none() {
        let table = PeerTable::new(Duration::from_secs(60));
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
        assert_eq!(table.identity_for_mac(&mac), None);
    }

    #[test]
    fn identity_for_mac_expired_returns_none() {
        let mut table = PeerTable::new(Duration::from_millis(1));
        let identity_hash = [1u8; 16];
        let mac = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

        table.update(identity_hash, mac);
        thread::sleep(Duration::from_millis(10));
        assert_eq!(table.identity_for_mac(&mac), None);
    }
}

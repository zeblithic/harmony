use alloc::{sync::Arc, vec, vec::Vec};
use hashbrown::HashMap;

use harmony_crypto::hash::TRUNCATED_HASH_LENGTH;

use crate::interface::InterfaceMode;

// ── Constants (from Python Transport.py) ────────────────────────────────

/// Maximum number of hops a path can traverse (Python: `PATHFINDER_M`).
pub const MAX_HOPS: u8 = 128;

/// Default path expiry in seconds (7 days).
pub const PATH_EXPIRY_DEFAULT: u64 = 604_800;

/// Path expiry for AccessPoint interfaces (1 day).
pub const PATH_EXPIRY_ACCESS_POINT: u64 = 86_400;

/// Path expiry for Roaming interfaces (6 hours).
pub const PATH_EXPIRY_ROAMING: u64 = 21_600;

/// Maximum announce random blobs retained per destination for replay detection.
pub const MAX_RANDOM_BLOBS: usize = 64;

/// Length of a single random blob in bytes.
pub const RANDOM_BLOB_LENGTH: usize = 10;

// ── Types ───────────────────────────────────────────────────────────────

/// A 16-byte destination hash (truncated SHA-256).
pub type DestinationHash = [u8; TRUNCATED_HASH_LENGTH];

/// A single routing entry in the path table.
#[derive(Debug, Clone)]
pub struct PathEntry {
    /// Monotonic timestamp (seconds) when this path was learned.
    pub learned_at: u64,
    /// Destination hash of the next hop toward this destination.
    pub next_hop: DestinationHash,
    /// Number of hops to reach this destination.
    pub hops: u8,
    /// Absolute expiry time in monotonic seconds.
    pub expires_at: u64,
    /// Name of the interface this path was learned on.
    pub interface_name: Arc<str>,
    /// Hash of the announce packet that created this entry.
    pub announce_packet_hash: DestinationHash,
    /// Announce timestamp of the current winning route (extracted from its random blob).
    /// Used for "same hops, newer timestamp" comparisons independently of the blob FIFO.
    pub announce_timestamp: u64,
    /// FIFO list of random blobs from announces, for replay detection.
    pub random_blobs: Vec<[u8; RANDOM_BLOB_LENGTH]>,
}

/// Result of a path table update operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PathUpdateResult {
    /// New destination inserted.
    Inserted,
    /// Existing entry replaced with a better route.
    Updated,
    /// Existing route is equal or better; no change.
    Kept,
    /// Random blob already seen — announce is a replay.
    DuplicateBlob,
    /// Hop count exceeds `MAX_HOPS`.
    ExceedsMaxHops,
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Return the path expiry duration (seconds) for a given interface mode.
pub fn path_expiry_for_mode(mode: InterfaceMode) -> u64 {
    match mode {
        InterfaceMode::AccessPoint => PATH_EXPIRY_ACCESS_POINT,
        InterfaceMode::Roaming => PATH_EXPIRY_ROAMING,
        _ => PATH_EXPIRY_DEFAULT,
    }
}

/// Extract the announce timestamp from the last 5 bytes of a random blob.
///
/// The random blob format (from `build_random_hash`) is:
///   `SHA256(random)[:5] || timestamp.to_be_bytes()[3..8]`
///
/// We reconstruct the u64 timestamp by zero-extending the 5 bytes.
pub fn timestamp_from_random_blob(blob: &[u8; RANDOM_BLOB_LENGTH]) -> u64 {
    let mut buf = [0u8; 8];
    buf[3..8].copy_from_slice(&blob[5..10]);
    u64::from_be_bytes(buf)
}

// ── PathTable ───────────────────────────────────────────────────────────

/// Routing knowledge: maps destination hashes to path entries.
pub struct PathTable {
    entries: HashMap<DestinationHash, PathEntry>,
}

impl PathTable {
    /// Create an empty path table.
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Look up a path entry by destination hash.
    pub fn get(&self, dest: &DestinationHash) -> Option<&PathEntry> {
        self.entries.get(dest)
    }

    /// Remove a path entry. Returns `true` if it existed.
    pub fn remove(&mut self, dest: &DestinationHash) -> bool {
        self.entries.remove(dest).is_some()
    }

    /// Number of entries in the table.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all (destination_hash, entry) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&DestinationHash, &PathEntry)> {
        self.entries.iter()
    }

    /// Remove all expired entries. Returns the count removed.
    pub fn expire(&mut self, now: u64) -> usize {
        let before = self.entries.len();
        self.entries.retain(|_, entry| entry.expires_at > now);
        before - self.entries.len()
    }

    /// Update the path table with information from an announce.
    ///
    /// Decision rules (matching Python `Transport.py`):
    /// 1. Hop count > `MAX_HOPS` → reject
    /// 2. Random blob already seen in existing entry → reject as replay
    /// 3. No existing entry → insert
    /// 4. Existing entry expired → replace
    /// 5. Fewer hops → replace
    /// 6. Same hops, newer timestamp → replace
    /// 7. Otherwise → keep existing
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        dest: DestinationHash,
        next_hop: DestinationHash,
        hops: u8,
        interface_name: Arc<str>,
        announce_packet_hash: DestinationHash,
        random_blob: [u8; RANDOM_BLOB_LENGTH],
        interface_mode: InterfaceMode,
        now: u64,
    ) -> PathUpdateResult {
        if hops > MAX_HOPS {
            return PathUpdateResult::ExceedsMaxHops;
        }

        let expiry = now + path_expiry_for_mode(interface_mode);

        match self.entries.get_mut(&dest) {
            Some(existing) => {
                // Check for duplicate random blob (replay detection)
                if existing.random_blobs.contains(&random_blob) {
                    return PathUpdateResult::DuplicateBlob;
                }

                let new_timestamp = timestamp_from_random_blob(&random_blob);
                let should_replace = existing.expires_at <= now
                    || hops < existing.hops
                    || (hops == existing.hops && new_timestamp > existing.announce_timestamp);

                // Always record the blob, even if we don't replace the route
                push_blob(&mut existing.random_blobs, random_blob);

                if should_replace {
                    existing.learned_at = now;
                    existing.next_hop = next_hop;
                    existing.hops = hops;
                    existing.expires_at = expiry;
                    existing.interface_name = interface_name;
                    existing.announce_packet_hash = announce_packet_hash;
                    existing.announce_timestamp = new_timestamp;
                    PathUpdateResult::Updated
                } else {
                    PathUpdateResult::Kept
                }
            }
            None => {
                let announce_timestamp = timestamp_from_random_blob(&random_blob);
                self.entries.insert(
                    dest,
                    PathEntry {
                        learned_at: now,
                        next_hop,
                        hops,
                        expires_at: expiry,
                        interface_name,
                        announce_packet_hash,
                        announce_timestamp,
                        random_blobs: vec![random_blob],
                    },
                );
                PathUpdateResult::Inserted
            }
        }
    }
}

impl Default for PathTable {
    fn default() -> Self {
        Self::new()
    }
}

/// Push a random blob onto the FIFO list, evicting the oldest if at capacity.
fn push_blob(blobs: &mut Vec<[u8; RANDOM_BLOB_LENGTH]>, blob: [u8; RANDOM_BLOB_LENGTH]) {
    if blobs.len() >= MAX_RANDOM_BLOBS {
        blobs.remove(0);
    }
    blobs.push(blob);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dest(id: u8) -> DestinationHash {
        let mut h = [0u8; 16];
        h[0] = id;
        h
    }

    fn blob(id: u8, timestamp: u64) -> [u8; RANDOM_BLOB_LENGTH] {
        let mut b = [0u8; RANDOM_BLOB_LENGTH];
        b[0] = id; // unique random portion
        let ts = timestamp.to_be_bytes();
        b[5..10].copy_from_slice(&ts[3..8]);
        b
    }

    // ── Insert ──────────────────────────────────────────────────────────

    #[test]
    fn insert_new_destination() {
        let mut table = PathTable::new();
        let result = table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        assert_eq!(result, PathUpdateResult::Inserted);
        assert_eq!(table.len(), 1);

        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(2));
        assert_eq!(entry.hops, 3);
        assert_eq!(&*entry.interface_name, "eth0");
        assert_eq!(entry.learned_at, 1000);
        assert_eq!(entry.expires_at, 1000 + PATH_EXPIRY_DEFAULT);
        assert_eq!(entry.random_blobs.len(), 1);
    }

    // ── Update (fewer hops) ─────────────────────────────────────────────

    #[test]
    fn update_with_fewer_hops() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            5,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        let result = table.update(
            dest(1),
            dest(3),
            2,
            "eth1".into(),
            dest(11),
            blob(2, 1001),
            InterfaceMode::Full,
            1001,
        );

        assert_eq!(result, PathUpdateResult::Updated);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(3));
        assert_eq!(entry.hops, 2);
        assert_eq!(&*entry.interface_name, "eth1");
    }

    // ── Update (same hops, newer timestamp) ─────────────────────────────

    #[test]
    fn update_same_hops_newer_timestamp() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        let result = table.update(
            dest(1),
            dest(3),
            3,
            "eth1".into(),
            dest(11),
            blob(2, 2000),
            InterfaceMode::Full,
            2000,
        );

        assert_eq!(result, PathUpdateResult::Updated);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(3));
    }

    // ── Keep (same hops, older timestamp) ────────────────────────────────

    #[test]
    fn keep_same_hops_older_timestamp() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 2000),
            InterfaceMode::Full,
            2000,
        );

        let result = table.update(
            dest(1),
            dest(3),
            3,
            "eth1".into(),
            dest(11),
            blob(2, 1000),
            InterfaceMode::Full,
            2001,
        );

        assert_eq!(result, PathUpdateResult::Kept);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(2)); // unchanged
    }

    // ── Keep (more hops) ────────────────────────────────────────────────

    #[test]
    fn keep_more_hops() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        let result = table.update(
            dest(1),
            dest(3),
            5,
            "eth1".into(),
            dest(11),
            blob(2, 2000),
            InterfaceMode::Full,
            2000,
        );

        assert_eq!(result, PathUpdateResult::Kept);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.hops, 3); // unchanged
    }

    // ── Update expired entry ────────────────────────────────────────────

    #[test]
    fn update_expired_entry() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        // Advance time past expiry
        let far_future = 1000 + PATH_EXPIRY_DEFAULT + 1;
        let result = table.update(
            dest(1),
            dest(3),
            10, // worse hops, but entry is expired
            "eth1".into(),
            dest(11),
            blob(2, far_future),
            InterfaceMode::Full,
            far_future,
        );

        assert_eq!(result, PathUpdateResult::Updated);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.hops, 10);
    }

    // ── Duplicate blob (replay) ─────────────────────────────────────────

    #[test]
    fn duplicate_blob_rejected() {
        let mut table = PathTable::new();
        let b = blob(1, 1000);
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            b,
            InterfaceMode::Full,
            1000,
        );

        let result = table.update(
            dest(1),
            dest(3),
            1, // better hops, but same blob
            "eth1".into(),
            dest(11),
            b, // same blob
            InterfaceMode::Full,
            1001,
        );

        assert_eq!(result, PathUpdateResult::DuplicateBlob);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(2)); // unchanged
    }

    // ── Exceeds max hops ────────────────────────────────────────────────

    #[test]
    fn exceeds_max_hops_rejected() {
        let mut table = PathTable::new();
        let result = table.update(
            dest(1),
            dest(2),
            MAX_HOPS + 1,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        assert_eq!(result, PathUpdateResult::ExceedsMaxHops);
        assert!(table.is_empty());
    }

    #[test]
    fn max_hops_exactly_accepted() {
        let mut table = PathTable::new();
        let result = table.update(
            dest(1),
            dest(2),
            MAX_HOPS,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        assert_eq!(result, PathUpdateResult::Inserted);
    }

    // ── Expiry sweep ────────────────────────────────────────────────────

    #[test]
    fn expire_removes_old_entries() {
        let mut table = PathTable::new();
        // Entry expiring at t=2000
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Roaming, // 6 hour expiry
            1000,
        );
        // Entry expiring much later
        table.update(
            dest(2),
            dest(3),
            2,
            "eth0".into(),
            dest(11),
            blob(2, 1000),
            InterfaceMode::Full, // 7 day expiry
            1000,
        );

        assert_eq!(table.len(), 2);

        // Advance past roaming expiry but not full expiry
        let removed = table.expire(1000 + PATH_EXPIRY_ROAMING + 1);
        assert_eq!(removed, 1);
        assert_eq!(table.len(), 1);
        assert!(table.get(&dest(1)).is_none());
        assert!(table.get(&dest(2)).is_some());
    }

    // ── Remove ──────────────────────────────────────────────────────────

    #[test]
    fn remove_existing() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        assert!(table.remove(&dest(1)));
        assert!(table.is_empty());
        assert!(!table.remove(&dest(1)));
    }

    // ── Mode-dependent TTLs ─────────────────────────────────────────────

    #[test]
    fn mode_dependent_expiry() {
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::Full),
            PATH_EXPIRY_DEFAULT
        );
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::PointToPoint),
            PATH_EXPIRY_DEFAULT
        );
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::AccessPoint),
            PATH_EXPIRY_ACCESS_POINT
        );
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::Roaming),
            PATH_EXPIRY_ROAMING
        );
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::Boundary),
            PATH_EXPIRY_DEFAULT
        );
        assert_eq!(
            path_expiry_for_mode(InterfaceMode::Gateway),
            PATH_EXPIRY_DEFAULT
        );
    }

    // ── Timestamp extraction ────────────────────────────────────────────

    #[test]
    fn timestamp_extraction() {
        let ts: u64 = 1_700_000_000;
        let b = blob(42, ts);
        assert_eq!(timestamp_from_random_blob(&b), ts);
    }

    #[test]
    fn timestamp_extraction_zero() {
        let b = blob(0, 0);
        assert_eq!(timestamp_from_random_blob(&b), 0);
    }

    // ── Random blob FIFO cap ────────────────────────────────────────────

    #[test]
    fn random_blob_fifo_cap() {
        let mut table = PathTable::new();

        // Insert initial entry
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(0, 1000),
            InterfaceMode::Full,
            1000,
        );

        // Add MAX_RANDOM_BLOBS more unique blobs (some will be Kept, some Updated)
        for i in 1..=MAX_RANDOM_BLOBS as u8 {
            table.update(
                dest(1),
                dest(2),
                3,
                "eth0".into(),
                dest(10),
                blob(i, 1000 + i as u64),
                InterfaceMode::Full,
                1000 + i as u64,
            );
        }

        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.random_blobs.len(), MAX_RANDOM_BLOBS);

        // First blob (id=0) should have been evicted
        assert!(!entry.random_blobs.iter().any(|b| b[0] == 0));
        // Last blob should be present
        assert!(entry
            .random_blobs
            .iter()
            .any(|b| b[0] == MAX_RANDOM_BLOBS as u8));
    }

    // ── Iter ────────────────────────────────────────────────────────────

    #[test]
    fn iter_yields_all_entries() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );
        table.update(
            dest(2),
            dest(3),
            2,
            "eth0".into(),
            dest(11),
            blob(2, 1000),
            InterfaceMode::Full,
            1000,
        );

        let keys: Vec<_> = table.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&dest(1)));
        assert!(keys.contains(&dest(2)));
    }

    // ── Kept blob doesn't corrupt timestamp comparison ────────────────

    #[test]
    fn kept_blob_does_not_corrupt_winning_timestamp() {
        let mut table = PathTable::new();

        // Establish 3-hop route at ts=1000
        table.update(
            dest(1),
            dest(2),
            3,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        // Worse 5-hop route at ts=500 → Kept, but blob is recorded
        let result = table.update(
            dest(1),
            dest(3),
            5,
            "eth1".into(),
            dest(11),
            blob(2, 500),
            InterfaceMode::Full,
            1001,
        );
        assert_eq!(result, PathUpdateResult::Kept);

        // Same 3-hop route at ts=800 — should NOT replace ts=1000 route
        let result = table.update(
            dest(1),
            dest(4),
            3,
            "eth2".into(),
            dest(12),
            blob(3, 800),
            InterfaceMode::Full,
            1002,
        );
        assert_eq!(result, PathUpdateResult::Kept);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(2)); // original route preserved
        assert_eq!(entry.announce_timestamp, 1000);
    }

    #[test]
    fn kept_blob_with_higher_timestamp_does_not_cause_false_upgrade() {
        let mut table = PathTable::new();

        // Establish 2-hop route at ts=1000
        table.update(
            dest(1),
            dest(2),
            2,
            "eth0".into(),
            dest(10),
            blob(1, 1000),
            InterfaceMode::Full,
            1000,
        );

        // Worse 5-hop route at ts=5000 → Kept (more hops)
        let result = table.update(
            dest(1),
            dest(3),
            5,
            "eth1".into(),
            dest(11),
            blob(2, 5000),
            InterfaceMode::Full,
            1001,
        );
        assert_eq!(result, PathUpdateResult::Kept);

        // Same 2-hop route at ts=1500 — should replace ts=1000 (genuinely newer)
        let result = table.update(
            dest(1),
            dest(4),
            2,
            "eth2".into(),
            dest(12),
            blob(3, 1500),
            InterfaceMode::Full,
            1002,
        );
        assert_eq!(result, PathUpdateResult::Updated);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.next_hop, dest(4));
        assert_eq!(entry.announce_timestamp, 1500);
    }

    // ── Blob recorded even on Kept ──────────────────────────────────────

    #[test]
    fn blob_recorded_on_kept() {
        let mut table = PathTable::new();
        table.update(
            dest(1),
            dest(2),
            2,
            "eth0".into(),
            dest(10),
            blob(1, 2000),
            InterfaceMode::Full,
            2000,
        );

        // Worse route — will be Kept, but blob should still be recorded
        let result = table.update(
            dest(1),
            dest(3),
            5,
            "eth1".into(),
            dest(11),
            blob(2, 1000),
            InterfaceMode::Full,
            2001,
        );

        assert_eq!(result, PathUpdateResult::Kept);
        let entry = table.get(&dest(1)).unwrap();
        assert_eq!(entry.random_blobs.len(), 2);
    }
}

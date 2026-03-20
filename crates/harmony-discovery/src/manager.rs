use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};
use harmony_identity::IdentityHash;

use crate::record::AnnounceRecord;
use crate::verify::verify_announce;

#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    AnnounceReceived { record: AnnounceRecord, now: u64 },
    QueryReceived { address: IdentityHash, query_id: u64, now: u64 },
    LivelinessChange { address: IdentityHash, alive: bool, now: u64 },
    Tick { now: u64 },
}

#[derive(Debug, Clone)]
pub enum DiscoveryAction {
    PublishAnnounce { record: AnnounceRecord },
    RespondToQuery { query_id: u64, record: Option<AnnounceRecord> },
    SetLiveliness { alive: bool },
    IdentityDiscovered { record: AnnounceRecord },
    IdentityOffline { address: IdentityHash },
    /// A record's `expires_at` has passed; evicted by `Tick`.
    RecordExpired { address: IdentityHash },
    /// A valid record was dropped because the cache was full.
    CacheEvicted { address: IdentityHash },
}

/// Default maximum number of cached identity records.
const DEFAULT_MAX_CACHE_SIZE: usize = 4096;

#[derive(Debug)]
pub struct DiscoveryManager {
    local_record: Option<AnnounceRecord>,
    known_identities: HashMap<IdentityHash, AnnounceRecord>,
    online: HashSet<IdentityHash>,
    max_cache_size: usize,
}

impl Default for DiscoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoveryManager {
    pub fn new() -> Self {
        Self::with_cache_size(DEFAULT_MAX_CACHE_SIZE)
    }

    /// Create a manager with a custom cache size limit.
    ///
    /// When the cache reaches this limit, the oldest record (by
    /// `published_at`) is evicted to make room for new entries.
    ///
    /// # Panics
    ///
    /// Panics if `max_cache_size` is 0.
    pub fn with_cache_size(max_cache_size: usize) -> Self {
        assert!(max_cache_size > 0, "max_cache_size must be at least 1");
        Self {
            local_record: None,
            known_identities: HashMap::new(),
            online: HashSet::new(),
            max_cache_size,
        }
    }

    pub fn set_local_record(&mut self, record: AnnounceRecord) {
        self.local_record = Some(record);
    }

    /// Start announcing this identity on the network.
    ///
    /// Emits `PublishAnnounce` (if a local record has been set via
    /// [`set_local_record`](Self::set_local_record)) and
    /// `SetLiveliness { alive: true }`.
    ///
    /// Call `set_local_record` before this method. Calling without a
    /// local record will advertise presence but serve `None` to all
    /// resolve queries.
    ///
    /// Safe to call multiple times — emitted actions are idempotent
    /// (Zenoh liveliness re-registration is a no-op), though callers
    /// that queue `PublishAnnounce` for transmission will send a
    /// redundant announce.
    #[must_use]
    pub fn start_announcing(&mut self) -> Vec<DiscoveryAction> {
        debug_assert!(
            self.local_record.is_some(),
            "start_announcing called without a local record"
        );
        let mut actions = Vec::new();
        if let Some(record) = self.local_record.clone() {
            actions.push(DiscoveryAction::PublishAnnounce { record });
        }
        actions.push(DiscoveryAction::SetLiveliness { alive: true });
        actions
    }

    /// Stop announcing. Safe to call multiple times.
    #[must_use]
    pub fn stop_announcing(&mut self) -> Vec<DiscoveryAction> {
        alloc::vec![DiscoveryAction::SetLiveliness { alive: false }]
    }

    pub fn is_online(&self, address: &IdentityHash) -> bool {
        self.online.contains(address)
    }

    /// Get an announce record for an address, filtered by expiry.
    ///
    /// Prefers `local_record` for the local identity (always
    /// authoritative), falls back to cached remote records.
    /// Returns `None` if no valid record exists.
    pub fn get_record(&self, address: &IdentityHash, now: u64) -> Option<&AnnounceRecord> {
        self.local_record
            .as_ref()
            .filter(|r| r.identity_ref.hash == *address && now < r.expires_at)
            .or_else(|| {
                self.known_identities
                    .get(address)
                    .filter(|r| now < r.expires_at)
            })
    }

    #[must_use]
    pub fn on_event(&mut self, event: DiscoveryEvent) -> Vec<DiscoveryAction> {
        let mut actions = Vec::new();
        match event {
            DiscoveryEvent::AnnounceReceived { record, now } => {
                self.handle_announce(record, now, &mut actions);
            }
            DiscoveryEvent::QueryReceived { address, query_id, now } => {
                // Prefer local_record for our own identity — always authoritative.
                let record = self
                    .local_record
                    .as_ref()
                    .filter(|r| r.identity_ref.hash == address && now < r.expires_at)
                    .cloned()
                    .or_else(|| {
                        self.known_identities
                            .get(&address)
                            .filter(|r| now < r.expires_at)
                            .cloned()
                    });
                actions.push(DiscoveryAction::RespondToQuery { query_id, record });
            }
            DiscoveryEvent::LivelinessChange { address, alive, now } => {
                if alive {
                    if self.online.insert(address) {
                        // Newly online — emit IdentityDiscovered if we have a valid record.
                        if let Some(record) = self
                            .known_identities
                            .get(&address)
                            .filter(|r| now < r.expires_at)
                            .cloned()
                        {
                            actions.push(DiscoveryAction::IdentityDiscovered { record });
                        }
                    }
                } else if self.online.remove(&address) {
                    actions.push(DiscoveryAction::IdentityOffline { address });
                }
            }
            DiscoveryEvent::Tick { now } => {
                self.evict_expired(now, &mut actions);
            }
        }
        actions
    }

    fn handle_announce(
        &mut self,
        record: AnnounceRecord,
        now: u64,
        actions: &mut Vec<DiscoveryAction>,
    ) {
        if verify_announce(&record, now).is_err() {
            return;
        }
        let addr = record.identity_ref.hash;
        if let Some(existing) = self.known_identities.get(&addr) {
            if record.published_at <= existing.published_at {
                return;
            }
        }

        // Evict oldest record before inserting if cache is full (and this
        // is a new address, not an update to an existing entry).
        if !self.known_identities.contains_key(&addr)
            && self.known_identities.len() >= self.max_cache_size
        {
            if let Some((&oldest_addr, oldest_rec)) = self
                .known_identities
                .iter()
                .min_by_key(|(_, r)| r.published_at)
            {
                if oldest_rec.published_at < record.published_at {
                    let evicted = oldest_addr;
                    self.known_identities.remove(&evicted);
                    actions.push(DiscoveryAction::CacheEvicted { address: evicted });
                } else {
                    // New entry is older than everything cached — reject it.
                    return;
                }
            }
        }

        self.known_identities.insert(addr, record.clone());

        // Only emit IdentityDiscovered if the peer is already online;
        // otherwise LivelinessChange will emit it when the token arrives.
        // This prevents double-emission in the normal announce→liveliness flow.
        if self.online.contains(&addr) {
            actions.push(DiscoveryAction::IdentityDiscovered { record });
        }
    }

    fn evict_expired(&mut self, now: u64, actions: &mut Vec<DiscoveryAction>) {
        let expired: Vec<IdentityHash> = self
            .known_identities
            .iter()
            .filter(|(_, record)| now >= record.expires_at)
            .map(|(addr, _)| *addr)
            .collect();
        for addr in expired {
            self.known_identities.remove(&addr);
            // NOTE: Do NOT remove from self.online here. The online set tracks
            // Zenoh liveliness tokens, which are independent of record expiry.
            // A peer can be alive (liveliness active) while their record has
            // expired. When they re-announce, handle_announce will cache the
            // new record and emit IdentityDiscovered (since they're online).
            actions.push(DiscoveryAction::RecordExpired { address: addr });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{AnnounceBuilder, RoutingHint};
    use harmony_identity::{CryptoSuite, IdentityRef};
    use rand::rngs::OsRng;

    fn build_valid_record(published_at: u64, expires_at: u64) -> AnnounceRecord {
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let mut builder = AnnounceBuilder::new(
            identity_ref,
            identity.verifying_key.to_bytes().to_vec(),
            published_at,
            expires_at,
            [0x01; 16],
        );
        builder.add_routing_hint(RoutingHint::Reticulum {
            destination_hash: [0xCC; 16],
        });
        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        builder.build(signature.to_vec())
    }

    fn build_invalid_record() -> AnnounceRecord {
        let identity_ref = IdentityRef::new([0xAA; 16], CryptoSuite::Ed25519);
        let builder = AnnounceBuilder::new(
            identity_ref,
            alloc::vec![0x01; 32],
            1000,
            2000,
            [0x01; 16],
        );
        let _payload = builder.signable_payload();
        builder.build(alloc::vec![0xFF; 64])
    }

    #[test]
    fn announce_received_valid_caches_silently() {
        // Announce caches the record but does NOT emit IdentityDiscovered
        // unless the peer is already online — liveliness join handles that.
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record.clone(), now: 1500 });
        assert!(mgr.get_record(&addr, 1500).is_some());
        assert!(actions.is_empty());
    }

    #[test]
    fn announce_for_online_peer_emits_discovered() {
        // If the peer is already online, a new/updated announce emits IdentityDiscovered.
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 5000);
        let addr = record.identity_ref.hash;

        // Peer comes online first (no record yet, so no emission)
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true, now: 1500 });

        // Now announce arrives — peer is already online, so emit
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn announce_received_invalid_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_invalid_record();
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        assert!(mgr.get_record(&addr, 1500).is_none());
        assert!(actions.is_empty());
    }

    #[test]
    fn announce_received_expired_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 3000 });
        assert!(mgr.get_record(&addr, 1500).is_none());
        assert!(actions.is_empty());
    }

    #[test]
    fn fresher_record_replaces_older() {
        let mut mgr = DiscoveryManager::new();
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let builder1 = AnnounceBuilder::new(identity_ref, pk.clone(), 1000, 3000, [0x01; 16]);
        let payload1 = builder1.signable_payload();
        let record1 = builder1.build(private.sign(&payload1).to_vec());
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record1, now: 1500 });

        let builder2 = AnnounceBuilder::new(identity_ref, pk.clone(), 2000, 4000, [0x02; 16]);
        let payload2 = builder2.signable_payload();
        let record2 = builder2.build(private.sign(&payload2).to_vec());
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record2, now: 2500 });

        let cached = mgr.get_record(&identity_ref.hash, 2500).unwrap();
        assert_eq!(cached.published_at, 2000);
    }

    #[test]
    fn stale_record_does_not_replace_fresher() {
        let mut mgr = DiscoveryManager::new();
        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let builder1 = AnnounceBuilder::new(identity_ref, pk.clone(), 2000, 4000, [0x01; 16]);
        let payload1 = builder1.signable_payload();
        let record1 = builder1.build(private.sign(&payload1).to_vec());
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record1, now: 2500 });

        let builder2 = AnnounceBuilder::new(identity_ref, pk.clone(), 1000, 3000, [0x02; 16]);
        let payload2 = builder2.signable_payload();
        let record2 = builder2.build(private.sign(&payload2).to_vec());
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record2, now: 2500 });

        let cached = mgr.get_record(&identity_ref.hash, 2500).unwrap();
        assert_eq!(cached.published_at, 2000);
        assert!(actions.is_empty());
    }

    #[test]
    fn query_received_known_identity() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });

        let actions = mgr.on_event(DiscoveryEvent::QueryReceived { address: addr, query_id: 42, now: 1500 });
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RespondToQuery { query_id: 42, record: Some(_) })));
    }

    #[test]
    fn query_received_unknown_identity() {
        let mut mgr = DiscoveryManager::new();
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived { address: [0xFF; 16], query_id: 99, now: 1500 });
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RespondToQuery { query_id: 99, record: None })));
    }

    #[test]
    fn liveliness_join_without_cached_record() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];
        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true, now: 1500 });
        assert!(mgr.is_online(&addr));
        assert!(actions.is_empty());
    }

    #[test]
    fn liveliness_join_with_cached_record_emits_discovered() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 5000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true, now: 1500 });
        assert!(mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn liveliness_join_with_expired_cached_record_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });

        // Liveliness join after record expired — no IdentityDiscovered
        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: addr, alive: true, now: 3000,
        });
        assert!(mgr.is_online(&addr));
        assert!(actions.is_empty());
    }

    #[test]
    fn liveliness_leave() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true, now: 1500 });

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: false, now: 1500 });
        assert!(!mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
    }

    #[test]
    fn cache_evicts_oldest_when_full() {
        let mut mgr = DiscoveryManager::with_cache_size(2);

        let r1 = build_valid_record(1000, 5000);
        let addr1 = r1.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r1, now: 1500 });

        let r2 = build_valid_record(2000, 5000);
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r2, now: 2500 });

        // Cache is full (2 entries). Adding a third evicts the oldest (r1).
        let r3 = build_valid_record(3000, 6000);
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r3, now: 3500 });

        assert!(mgr.get_record(&addr1, 3500).is_none()); // r1 evicted
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::CacheEvicted { .. })));
    }

    #[test]
    fn cache_rejects_older_than_all_cached() {
        let mut mgr = DiscoveryManager::with_cache_size(2);

        let r1 = build_valid_record(2000, 5000);
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r1, now: 2500 });

        let r2 = build_valid_record(3000, 6000);
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r2, now: 3500 });

        // Cache full. New record is older than everything cached — rejected.
        let r3 = build_valid_record(1000, 5000);
        let addr3 = r3.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: r3, now: 3500 });

        assert!(mgr.get_record(&addr3, 3500).is_none()); // not inserted
        assert!(actions.is_empty()); // no eviction, no discovery
    }

    #[test]
    #[should_panic(expected = "max_cache_size")]
    fn zero_cache_size_panics() {
        DiscoveryManager::with_cache_size(0);
    }

    #[test]
    fn liveliness_leave_for_never_online_peer_is_silent() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];
        // Leave without prior join — no spurious IdentityOffline
        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: false, now: 1500 });
        assert!(actions.is_empty());
    }

    #[test]
    fn tick_evicts_expired_records() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        assert!(mgr.get_record(&addr, 1500).is_some());

        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 3000 });
        assert!(mgr.get_record(&addr, 1500).is_none());
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
    }

    #[test]
    fn tick_preserves_online_state_when_record_expires() {
        // Liveliness tokens are independent of record expiry. A peer can
        // be online (liveliness active) while their record has expired.
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true, now: 1500 });
        assert!(mgr.is_online(&addr));

        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 3000 });
        // Record evicted, but peer stays online
        assert!(mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
        assert!(!actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
    }

    #[test]
    fn query_filters_expired_records() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });

        // Query after expiry but before tick
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: addr, query_id: 1, now: 3000,
        });
        assert!(actions.iter().any(|a| matches!(
            a, DiscoveryAction::RespondToQuery { query_id: 1, record: None }
        )));
    }

    #[test]
    fn query_filters_expired_local_record() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        mgr.set_local_record(record);

        // Query after local record has expired
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: addr, query_id: 1, now: 3000,
        });
        assert!(actions.iter().any(|a| matches!(
            a, DiscoveryAction::RespondToQuery { query_id: 1, record: None }
        )));
    }

    #[test]
    fn query_prefers_local_record_over_cache() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        // Cache an older record
        let builder1 = AnnounceBuilder::new(identity_ref, pk.clone(), 1000, 5000, [0x01; 16]);
        let payload1 = builder1.signable_payload();
        let record1 = builder1.build(private.sign(&payload1).to_vec());

        // Set local_record with a fresher version
        let mut builder2 = AnnounceBuilder::new(identity_ref, pk.clone(), 2000, 6000, [0x02; 16]);
        builder2.add_routing_hint(RoutingHint::Reticulum { destination_hash: [0xDD; 16] });
        let payload2 = builder2.signable_payload();
        let record2 = builder2.build(private.sign(&payload2).to_vec());

        // Simulate: peer is online, announce cached
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: identity_ref.hash, alive: true, now: 1500 });
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record1, now: 1500 });

        // Now update local_record without re-announcing
        mgr.set_local_record(record2);

        // Query should return the fresher local_record, not the stale cache
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: identity_ref.hash, query_id: 1, now: 2500,
        });
        match &actions[0] {
            DiscoveryAction::RespondToQuery { query_id: 1, record: Some(r) } => {
                assert_eq!(r.published_at, 2000); // local_record's timestamp
                assert_eq!(r.routing_hints.len(), 1); // local_record's hints
            }
            other => panic!("expected RespondToQuery with Some, got {:?}", other),
        }
    }

    #[test]
    fn query_returns_local_record() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 5000);
        let addr = record.identity_ref.hash;
        mgr.set_local_record(record);

        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: addr, query_id: 1, now: 2000,
        });
        assert!(actions.iter().any(|a| matches!(
            a, DiscoveryAction::RespondToQuery { query_id: 1, record: Some(_) }
        )));
    }

    #[test]
    fn start_and_stop_announcing() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        mgr.set_local_record(record);

        let actions = mgr.start_announcing();
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::PublishAnnounce { .. })));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::SetLiveliness { alive: true })));

        let actions = mgr.stop_announcing();
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::SetLiveliness { alive: false })));
    }

    #[test]
    fn full_flow_announce_cache_query() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let mut builder =
            AnnounceBuilder::new(identity_ref, pk, 1000, 5000, [0x10; 16]);
        builder
            .add_routing_hint(RoutingHint::Reticulum {
                destination_hash: [0xAA; 16],
            })
            .add_routing_hint(RoutingHint::Zenoh {
                locator: alloc::string::String::from("tcp/10.0.0.1:7447"),
            });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        // 2. Receive the announce (peer not online yet — cached silently)
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record.clone(),
            now: 2000,
        });
        assert!(actions.is_empty());
        assert!(mgr.get_record(&identity_ref.hash, 2000).is_some());

        // 3. Query for it
        let actions = mgr.on_event(DiscoveryEvent::QueryReceived {
            address: identity_ref.hash,
            query_id: 1,
            now: 2000,
        });
        match &actions[0] {
            DiscoveryAction::RespondToQuery {
                query_id: 1,
                record: Some(r),
            } => {
                assert_eq!(r.identity_ref, identity_ref);
                assert_eq!(r.routing_hints.len(), 2);
            }
            other => panic!("expected RespondToQuery, got {:?}", other),
        }

        // 4. Liveliness tracks online state
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange {
            address: identity_ref.hash,
            alive: true,
            now: 2000,
        });
        assert!(mgr.is_online(&identity_ref.hash));

        // 5. Tick evicts record but preserves online state (liveliness independent)
        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 6000 });
        assert!(mgr.get_record(&identity_ref.hash, 6000).is_none());
        assert!(mgr.is_online(&identity_ref.hash)); // still online — liveliness independent
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
        assert!(!actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
    }

    #[test]
    fn announce_serde_round_trip_through_manager() {
        let mut mgr = DiscoveryManager::new();

        let private = harmony_identity::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let identity_ref = IdentityRef::from(identity);
        let pk = identity.verifying_key.to_bytes().to_vec();

        let builder = AnnounceBuilder::new(identity_ref, pk, 1000, 5000, [0x20; 16]);
        let payload = builder.signable_payload();
        let record = builder.build(private.sign(&payload).to_vec());

        // Serialize, deserialize, then feed to manager
        let bytes = record.serialize().unwrap();
        let restored = AnnounceRecord::deserialize(&bytes).unwrap();

        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: restored,
            now: 2000,
        });
        // Peer not online, so announce caches silently
        assert!(actions.is_empty());
        assert!(mgr.get_record(&identity_ref.hash, 2000).is_some());
    }
}

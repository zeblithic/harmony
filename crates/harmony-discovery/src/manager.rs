use alloc::vec::Vec;
use hashbrown::{HashMap, HashSet};
use harmony_identity::IdentityHash;

use crate::record::AnnounceRecord;
use crate::verify::verify_announce;

#[derive(Debug, Clone)]
pub enum DiscoveryEvent {
    AnnounceReceived { record: AnnounceRecord, now: u64 },
    QueryReceived { address: IdentityHash, query_id: u64, now: u64 },
    LivelinessChange { address: IdentityHash, alive: bool },
    Tick { now: u64 },
}

#[derive(Debug, Clone)]
pub enum DiscoveryAction {
    PublishAnnounce { record: AnnounceRecord },
    RespondToQuery { query_id: u64, record: Option<AnnounceRecord> },
    SetLiveliness { alive: bool },
    IdentityDiscovered { record: AnnounceRecord },
    IdentityOffline { address: IdentityHash },
    RecordExpired { address: IdentityHash },
}

pub struct DiscoveryManager {
    local_record: Option<AnnounceRecord>,
    known_identities: HashMap<IdentityHash, AnnounceRecord>,
    online: HashSet<IdentityHash>,
}

impl Default for DiscoveryManager {
    fn default() -> Self {
        Self::new()
    }
}

impl DiscoveryManager {
    pub fn new() -> Self {
        Self {
            local_record: None,
            known_identities: HashMap::new(),
            online: HashSet::new(),
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

    #[must_use]
    pub fn stop_announcing(&mut self) -> Vec<DiscoveryAction> {
        alloc::vec![DiscoveryAction::SetLiveliness { alive: false }]
    }

    pub fn is_online(&self, address: &IdentityHash) -> bool {
        self.online.contains(address)
    }

    pub fn get_record(&self, address: &IdentityHash) -> Option<&AnnounceRecord> {
        self.known_identities.get(address)
    }

    #[must_use]
    pub fn on_event(&mut self, event: DiscoveryEvent) -> Vec<DiscoveryAction> {
        let mut actions = Vec::new();
        match event {
            DiscoveryEvent::AnnounceReceived { record, now } => {
                self.handle_announce(record, now, &mut actions);
            }
            DiscoveryEvent::QueryReceived { address, query_id, now } => {
                let record = self
                    .known_identities
                    .get(&address)
                    .filter(|r| now < r.expires_at)
                    .cloned()
                    .or_else(|| {
                        self.local_record
                            .as_ref()
                            .filter(|r| r.identity_ref.hash == address && now < r.expires_at)
                            .cloned()
                    });
                actions.push(DiscoveryAction::RespondToQuery { query_id, record });
            }
            DiscoveryEvent::LivelinessChange { address, alive } => {
                if alive {
                    self.online.insert(address);
                    if let Some(record) = self.known_identities.get(&address).cloned() {
                        actions.push(DiscoveryAction::IdentityDiscovered { record });
                    }
                } else {
                    self.online.remove(&address);
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
        self.known_identities.insert(addr, record.clone());
        actions.push(DiscoveryAction::IdentityDiscovered { record });
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
            if self.online.remove(&addr) {
                actions.push(DiscoveryAction::IdentityOffline { address: addr });
            }
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
    fn announce_received_valid_caches_and_emits() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record: record.clone(), now: 1500 });
        assert!(mgr.get_record(&addr).is_some());
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn announce_received_invalid_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_invalid_record();
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        assert!(mgr.get_record(&addr).is_none());
        assert!(actions.is_empty());
    }

    #[test]
    fn announce_received_expired_ignored() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 3000 });
        assert!(mgr.get_record(&addr).is_none());
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

        let cached = mgr.get_record(&identity_ref.hash).unwrap();
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

        let cached = mgr.get_record(&identity_ref.hash).unwrap();
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
        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true });
        assert!(mgr.is_online(&addr));
        assert!(actions.is_empty());
    }

    #[test]
    fn liveliness_join_with_cached_record_emits_discovered() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 5000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true });
        assert!(mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
    }

    #[test]
    fn liveliness_leave() {
        let mut mgr = DiscoveryManager::new();
        let addr = [0xAA; 16];
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true });

        let actions = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: false });
        assert!(!mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
    }

    #[test]
    fn tick_evicts_expired_records() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;
        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        assert!(mgr.get_record(&addr).is_some());

        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 3000 });
        assert!(mgr.get_record(&addr).is_none());
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
    }

    #[test]
    fn tick_evicts_online_state_for_expired_records() {
        let mut mgr = DiscoveryManager::new();
        let record = build_valid_record(1000, 2000);
        let addr = record.identity_ref.hash;

        let _ = mgr.on_event(DiscoveryEvent::AnnounceReceived { record, now: 1500 });
        let _ = mgr.on_event(DiscoveryEvent::LivelinessChange { address: addr, alive: true });
        assert!(mgr.is_online(&addr));

        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 3000 });
        assert!(!mgr.is_online(&addr));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
        assert!(actions.iter().any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
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
                locator: alloc::vec![0xBB; 4],
            });

        let payload = builder.signable_payload();
        let signature = private.sign(&payload);
        let record = builder.build(signature.to_vec());

        // 2. Receive the announce
        let actions = mgr.on_event(DiscoveryEvent::AnnounceReceived {
            record: record.clone(),
            now: 2000,
        });
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));

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
        });
        assert!(mgr.is_online(&identity_ref.hash));

        // 5. Tick evicts after expiry (including online state)
        let actions = mgr.on_event(DiscoveryEvent::Tick { now: 6000 });
        assert!(mgr.get_record(&identity_ref.hash).is_none());
        assert!(!mgr.is_online(&identity_ref.hash));
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityOffline { .. })));
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::RecordExpired { .. })));
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
        assert!(actions
            .iter()
            .any(|a| matches!(a, DiscoveryAction::IdentityDiscovered { .. })));
        assert!(mgr.get_record(&identity_ref.hash).is_some());
    }
}

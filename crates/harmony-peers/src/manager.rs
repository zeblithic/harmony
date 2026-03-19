use alloc::vec::Vec;
use harmony_contacts::{ContactStore, PeeringPriority};
use harmony_identity::IdentityHash;
use hashbrown::HashMap;

use crate::event::{PeerAction, PeerEvent};
use crate::state::{PeerState, PeerStatus};

const PROBE_INTERVAL_HIGH: u64 = 30;
const PROBE_INTERVAL_NORMAL: u64 = 120;
const PROBE_INTERVAL_MAX: u64 = 600;
const CONNECTING_TIMEOUT: u64 = 60;

pub struct PeerManager {
    pub(crate) peers: HashMap<IdentityHash, PeerState>,
}

impl PeerManager {
    pub fn new() -> Self {
        Self {
            peers: HashMap::new(),
        }
    }

    pub fn on_event(&mut self, event: PeerEvent, contacts: &ContactStore) -> Vec<PeerAction> {
        let mut actions = Vec::new();
        match event {
            PeerEvent::ContactChanged { identity_hash } => {
                self.handle_contact_changed(identity_hash, contacts, &mut actions);
            }
            PeerEvent::ContactRemoved { identity_hash } => {
                if let Some(peer) = self.peers.remove(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        actions.push(PeerAction::CloseLink { identity_hash });
                    }
                }
            }
            PeerEvent::AnnounceReceived { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if peer.status == PeerStatus::Searching {
                        peer.status = PeerStatus::Connecting;
                        peer.connecting_since = None; // stamped on first Tick
                        actions.push(PeerAction::InitiateLink { identity_hash });
                    }
                }
            }
            PeerEvent::LinkEstablished { identity_hash, now } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    peer.status = PeerStatus::Connected;
                    peer.retry_count = 0;
                    peer.last_seen = Some(now);
                    peer.connecting_since = None;
                    actions.push(PeerAction::UpdateLastSeen {
                        identity_hash,
                        timestamp: now,
                    });
                }
            }
            PeerEvent::LinkClosed { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = peer.retry_count.saturating_add(1);
                    }
                }
            }
            PeerEvent::Tick { now } => {
                self.handle_tick(now, &mut actions);
            }
        }
        actions
    }

    fn handle_contact_changed(
        &mut self,
        identity_hash: IdentityHash,
        contacts: &ContactStore,
        actions: &mut Vec<PeerAction>,
    ) {
        let contact = match contacts.get(&identity_hash) {
            Some(c) => c,
            None => return,
        };
        if contact.peering.enabled {
            match self.peers.get_mut(&identity_hash) {
                Some(peer) => {
                    peer.priority = contact.peering.priority;
                    if peer.status == PeerStatus::Disabled {
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = 0;
                    }
                }
                None => {
                    self.peers
                        .insert(identity_hash, PeerState::new(contact.peering.priority));
                }
            }
        } else {
            match self.peers.get_mut(&identity_hash) {
                Some(peer) => {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        actions.push(PeerAction::CloseLink { identity_hash });
                    }
                    peer.status = PeerStatus::Disabled;
                }
                None => {
                    self.peers.insert(identity_hash, PeerState::new_disabled());
                }
            }
        }
    }

    fn handle_tick(&mut self, now: u64, actions: &mut Vec<PeerAction>) {
        for (id, peer) in self.peers.iter_mut() {
            match peer.status {
                PeerStatus::Searching => {
                    let interval = Self::probe_interval(peer.priority, peer.retry_count);
                    if interval == 0 {
                        continue;
                    }
                    let should_probe = match peer.last_probe {
                        None => true,
                        Some(last) => now.saturating_sub(last) >= interval,
                    };
                    if should_probe {
                        peer.last_probe = Some(now);
                        actions.push(PeerAction::SendPathRequest { identity_hash: *id });
                    }
                }
                PeerStatus::Connecting => match peer.connecting_since {
                    None => {
                        peer.connecting_since = Some(now);
                    }
                    Some(since) => {
                        if now.saturating_sub(since) >= CONNECTING_TIMEOUT {
                            peer.status = PeerStatus::Searching;
                            peer.retry_count = peer.retry_count.saturating_add(1);
                            peer.connecting_since = None;
                        }
                    }
                },
                PeerStatus::Connected | PeerStatus::Disabled => {}
            }
        }
    }

    fn probe_interval(priority: PeeringPriority, retry_count: u32) -> u64 {
        let base = match priority {
            PeeringPriority::Low => return 0,
            PeeringPriority::Normal => PROBE_INTERVAL_NORMAL,
            PeeringPriority::High => PROBE_INTERVAL_HIGH,
        };
        let backoff = base.saturating_mul(1u64 << retry_count.min(20));
        backoff.min(PROBE_INTERVAL_MAX)
    }
}

impl Default for PeerManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_contacts::{Contact, ContactStore, PeeringPolicy, PeeringPriority};

    fn make_store_with_contact(
        id_byte: u8,
        enabled: bool,
        priority: PeeringPriority,
    ) -> ContactStore {
        let mut store = ContactStore::new();
        store
            .add(Contact {
                identity_hash: [id_byte; 16],
                display_name: None,
                peering: PeeringPolicy { enabled, priority },
                added_at: 1000,
                last_seen: None,
                notes: None,
            })
            .unwrap();
        store
    }

    #[test]
    fn contact_added_high_priority_probes_on_tick() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xAA, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xAA; 16],
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xAA; 16]
        }));
    }

    #[test]
    fn contact_added_normal_priority_probes_on_tick() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xA1, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xA1; 16],
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA1; 16]
        }));
        // 60s later — too soon for Normal (120s base)
        let actions = mgr.on_event(PeerEvent::Tick { now: 1060 }, &store);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
        // 121s later — past 120s interval
        let actions = mgr.on_event(PeerEvent::Tick { now: 1121 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA1; 16]
        }));
    }

    #[test]
    fn contact_added_low_priority_no_probe() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xBB, true, PeeringPriority::Low);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xBB; 16],
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
    }

    #[test]
    fn announce_received_triggers_link_initiation() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xCC, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xCC; 16],
            },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xCC; 16],
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::InitiateLink {
            identity_hash: [0xCC; 16]
        }));
    }

    #[test]
    fn announce_for_unknown_contact_ignored() {
        let mut mgr = PeerManager::new();
        let store = ContactStore::new();
        let actions = mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xFF; 16],
            },
            &store,
        );
        assert!(actions.is_empty());
    }

    #[test]
    fn link_established_transitions_to_connected() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xDD, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xDD; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xDD; 16],
            },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0xDD; 16],
                now: 5000,
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::UpdateLastSeen {
            identity_hash: [0xDD; 16],
            timestamp: 5000
        }));
        assert_eq!(
            mgr.peers.get(&[0xDD; 16]).unwrap().status,
            PeerStatus::Connected
        );
    }

    #[test]
    fn link_closed_transitions_to_searching() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xEE, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xEE; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0xEE; 16],
                now: 5000,
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkClosed {
                identity_hash: [0xEE; 16],
            },
            &store,
        );
        let peer = mgr.peers.get(&[0xEE; 16]).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
    }

    #[test]
    fn contact_removed_closes_link() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x11, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x11; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0x11; 16],
                now: 5000,
            },
            &store,
        );
        let actions = mgr.on_event(
            PeerEvent::ContactRemoved {
                identity_hash: [0x11; 16],
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0x11; 16]
        }));
        assert!(!mgr.peers.contains_key(&[0x11; 16]));
    }

    #[test]
    fn disabled_policy_closes_active_link() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0x22, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x22; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0x22; 16],
                now: 5000,
            },
            &store,
        );
        store.get_mut(&[0x22; 16]).unwrap().peering.enabled = false;
        let actions = mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x22; 16],
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0x22; 16]
        }));
        assert_eq!(
            mgr.peers.get(&[0x22; 16]).unwrap().status,
            PeerStatus::Disabled
        );
    }

    #[test]
    fn priority_change_updates_probe_interval() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0xA2, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xA2; 16],
            },
            &store,
        );
        mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        store.get_mut(&[0xA2; 16]).unwrap().peering.priority = PeeringPriority::High;
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xA2; 16],
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1031 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0xA2; 16]
        }));
    }

    #[test]
    fn re_enable_policy_starts_searching() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0x33, false, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x33; 16],
            },
            &store,
        );
        assert_eq!(
            mgr.peers.get(&[0x33; 16]).unwrap().status,
            PeerStatus::Disabled
        );
        store.get_mut(&[0x33; 16]).unwrap().peering.enabled = true;
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x33; 16],
            },
            &store,
        );
        assert_eq!(
            mgr.peers.get(&[0x33; 16]).unwrap().status,
            PeerStatus::Searching
        );
    }

    #[test]
    fn connected_peer_no_probing() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x44, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x44; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0x44; 16],
                now: 5000,
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 9999 }, &store);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
    }

    #[test]
    fn probe_interval_respected() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x55, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x55; 16],
            },
            &store,
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x55; 16]
        }));
        let actions = mgr.on_event(PeerEvent::Tick { now: 1010 }, &store);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
        let actions = mgr.on_event(PeerEvent::Tick { now: 1031 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x55; 16]
        }));
    }

    #[test]
    fn backoff_increases_probe_interval() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x66, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x66; 16],
            },
            &store,
        );
        for _ in 0..3 {
            mgr.on_event(
                PeerEvent::AnnounceReceived {
                    identity_hash: [0x66; 16],
                },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed {
                    identity_hash: [0x66; 16],
                },
                &store,
            );
        }
        assert_eq!(mgr.peers.get(&[0x66; 16]).unwrap().retry_count, 3);
        // interval = min(30 * 2^3, 600) = 240s
        mgr.on_event(PeerEvent::Tick { now: 10000 }, &store);
        let actions = mgr.on_event(PeerEvent::Tick { now: 10100 }, &store);
        assert!(!actions
            .iter()
            .any(|a| matches!(a, PeerAction::SendPathRequest { .. })));
        let actions = mgr.on_event(PeerEvent::Tick { now: 10241 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x66; 16]
        }));
    }

    #[test]
    fn backoff_caps_at_600s() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x77, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x77; 16],
            },
            &store,
        );
        for _ in 0..10 {
            mgr.on_event(
                PeerEvent::AnnounceReceived {
                    identity_hash: [0x77; 16],
                },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed {
                    identity_hash: [0x77; 16],
                },
                &store,
            );
        }
        mgr.on_event(PeerEvent::Tick { now: 50000 }, &store);
        let actions = mgr.on_event(PeerEvent::Tick { now: 50601 }, &store);
        assert!(actions.contains(&PeerAction::SendPathRequest {
            identity_hash: [0x77; 16]
        }));
    }

    #[test]
    fn link_established_resets_retry_count() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x88, true, PeeringPriority::Normal);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x88; 16],
            },
            &store,
        );
        for _ in 0..5 {
            mgr.on_event(
                PeerEvent::AnnounceReceived {
                    identity_hash: [0x88; 16],
                },
                &store,
            );
            mgr.on_event(
                PeerEvent::LinkClosed {
                    identity_hash: [0x88; 16],
                },
                &store,
            );
        }
        assert_eq!(mgr.peers.get(&[0x88; 16]).unwrap().retry_count, 5);
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0x88; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0x88; 16],
                now: 90000,
            },
            &store,
        );
        assert_eq!(mgr.peers.get(&[0x88; 16]).unwrap().retry_count, 0);
    }

    #[test]
    fn connecting_timeout_transitions_to_searching() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0x99, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0x99; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0x99; 16],
            },
            &store,
        );
        assert_eq!(
            mgr.peers.get(&[0x99; 16]).unwrap().status,
            PeerStatus::Connecting
        );
        // First tick stamps connecting_since
        mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        assert_eq!(
            mgr.peers.get(&[0x99; 16]).unwrap().status,
            PeerStatus::Connecting
        );
        // Tick 61s later — should timeout
        mgr.on_event(PeerEvent::Tick { now: 1061 }, &store);
        let peer = mgr.peers.get(&[0x99; 16]).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
    }
}

use alloc::vec::Vec;
use harmony_contacts::{ContactAddress, ContactStore, PeeringPriority};
use harmony_identity::IdentityHash;
use hashbrown::HashMap;

use crate::event::{PeerAction, PeerEvent};
use crate::state::{ConnectionQuality, PeerState, PeerStatus, Transport};

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

    #[must_use]
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

                        // Check whether the contact has a Tunnel address — if so, prefer it.
                        let tunnel_action =
                            contacts.get(&identity_hash).and_then(|contact| {
                                contact.addresses.iter().find_map(|addr| {
                                    if let ContactAddress::Tunnel { node_id, relay_url, .. } = addr
                                    {
                                        Some(PeerAction::InitiateTunnel {
                                            identity_hash,
                                            node_id: *node_id,
                                            relay_url: relay_url.clone(),
                                        })
                                    } else {
                                        None
                                    }
                                })
                            });

                        if let Some(action) = tunnel_action {
                            actions.push(action);
                        } else {
                            actions.push(PeerAction::InitiateLink { identity_hash });
                        }
                    }
                }
            }
            PeerEvent::TunnelEstablished { identity_hash, node_id, now } => {
                match self.peers.get_mut(&identity_hash) {
                    Some(peer) => match peer.status {
                        PeerStatus::Connecting | PeerStatus::Searching => {
                            let relayed = contacts
                                .get(&identity_hash)
                                .and_then(|c| {
                                    c.addresses.iter().find_map(|addr| {
                                        if let ContactAddress::Tunnel {
                                            node_id: addr_node_id,
                                            relay_url,
                                            ..
                                        } = addr
                                        {
                                            if *addr_node_id == node_id {
                                                // TODO: This infers relayed from the stored
                                                // config, not the actual connection path. Once
                                                // TunnelBridgeEvent carries the real relay status
                                                // from iroh-net, pass it through TunnelEstablished.
                                                Some(relay_url.is_some())
                                            } else {
                                                None
                                            }
                                        } else {
                                            None
                                        }
                                    })
                                })
                                .unwrap_or(false);

                            peer.status = PeerStatus::Connected;
                            peer.retry_count = 0;
                            peer.last_seen = Some(now);
                            peer.connecting_since = None;
                            peer.connection_quality = Some(ConnectionQuality {
                                rtt_ms: None,
                                transport: Transport::Tunnel { relayed },
                                connected_since: now,
                            });
                            actions.push(PeerAction::UpdateLastSeen {
                                identity_hash,
                                timestamp: now,
                            });
                        }
                        PeerStatus::Disabled => {
                            // Race: CloseLink/tunnel teardown in flight.
                        }
                        PeerStatus::Connected => {}
                    },
                    None => {
                        // Contact was removed; ignore stale tunnel completion.
                    }
                }
            }
            PeerEvent::TunnelFailed { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    // Only act on Connecting — a stale TunnelFailed arriving after
                    // the Tick timeout already moved the peer to Searching should
                    // not double-increment retry_count or corrupt last_probe.
                    if matches!(peer.status, PeerStatus::Connecting) {
                        peer.last_probe =
                            peer.connecting_since.or(peer.last_seen).or(peer.last_probe);
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = peer.retry_count.saturating_add(1);
                        peer.connecting_since = None;
                        peer.connection_quality = None;
                    }
                }
            }
            PeerEvent::TunnelDropped { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        peer.last_probe =
                            peer.connecting_since.or(peer.last_seen).or(peer.last_probe);
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = peer.retry_count.saturating_add(1);
                        peer.connecting_since = None;
                        peer.connection_quality = None;

                        // If the contact also has a Reticulum address, immediately fall back
                        // to a Reticulum link — no backoff for a fallback path.
                        let has_reticulum = contacts
                            .get(&identity_hash)
                            .map(|c| {
                                c.addresses.iter().any(|addr| {
                                    matches!(addr, ContactAddress::Reticulum { .. })
                                })
                            })
                            .unwrap_or(false);

                        if has_reticulum {
                            actions.push(PeerAction::InitiateLink { identity_hash });
                            peer.status = PeerStatus::Connecting;
                        }
                    }
                }
            }
            PeerEvent::LinkEstablished { identity_hash, now } => {
                match self.peers.get_mut(&identity_hash) {
                    Some(peer) => match peer.status {
                        PeerStatus::Connecting | PeerStatus::Searching => {
                            peer.status = PeerStatus::Connected;
                            peer.retry_count = 0;
                            peer.last_seen = Some(now);
                            peer.connecting_since = None;
                            actions.push(PeerAction::UpdateLastSeen {
                                identity_hash,
                                timestamp: now,
                            });
                        }
                        PeerStatus::Disabled => {
                            // Race: CloseLink was emitted but link completed first.
                            actions.push(PeerAction::CloseLink { identity_hash });
                        }
                        PeerStatus::Connected => {}
                    },
                    None => {
                        // Race: contact was removed but link completed first.
                        actions.push(PeerAction::CloseLink { identity_hash });
                    }
                }
            }
            PeerEvent::LinkClosed { identity_hash } => {
                if let Some(peer) = self.peers.get_mut(&identity_hash) {
                    if matches!(peer.status, PeerStatus::Connected | PeerStatus::Connecting) {
                        // Reset last_probe so backoff starts from the failure,
                        // not from the original probe that led to this attempt.
                        peer.last_probe =
                            peer.connecting_since.or(peer.last_seen).or(peer.last_probe);
                        peer.status = PeerStatus::Searching;
                        peer.retry_count = peer.retry_count.saturating_add(1);
                        peer.connecting_since = None;
                        peer.connection_quality = None;
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
                            // Start backoff from now, not from the original probe.
                            peer.last_probe = Some(now);
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
    use harmony_contacts::{Contact, ContactAddress, ContactStore, PeeringPolicy, PeeringPriority};

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
                addresses: vec![],
            })
            .unwrap();
        store
    }

    fn make_tunnel_contact(
        identity_hash: IdentityHash,
        with_reticulum: bool,
    ) -> (ContactStore, Contact) {
        let mut addrs = vec![ContactAddress::Tunnel {
            node_id: [0xAA; 32],
            relay_url: Some("https://iroh.q8.fyi".into()),
            direct_addrs: vec![],
        }];
        if with_reticulum {
            addrs.push(ContactAddress::Reticulum {
                destination_hash: [0xBB; 16],
            });
        }
        let contact = Contact {
            identity_hash,
            display_name: None,
            peering: PeeringPolicy {
                enabled: true,
                priority: PeeringPriority::Normal,
            },
            added_at: 1000,
            last_seen: None,
            notes: None,
            addresses: addrs,
        };
        let mut store = ContactStore::new();
        store.add(contact.clone()).unwrap();
        (store, contact)
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

    #[test]
    fn backoff_starts_from_failure_not_original_probe() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xE1, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xE1; 16],
            },
            &store,
        );
        // Probe at t=1000
        mgr.on_event(PeerEvent::Tick { now: 1000 }, &store);
        // Announce arrives, enter Connecting
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xE1; 16],
            },
            &store,
        );
        // Tick stamps connecting_since at t=1020
        mgr.on_event(PeerEvent::Tick { now: 1020 }, &store);
        // Connection fails at t=1050
        mgr.on_event(
            PeerEvent::LinkClosed {
                identity_hash: [0xE1; 16],
            },
            &store,
        );
        // retry_count=1, interval = min(30*2^1, 600) = 60s
        // Backoff should be from connecting_since (t=1020), not original probe (t=1000)
        // So next probe should be at t=1020+60 = t=1080
        let actions = mgr.on_event(PeerEvent::Tick { now: 1055 }, &store);
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, PeerAction::SendPathRequest { .. })),
            "Should NOT probe at t=1055 — backoff from t=1020 requires 60s"
        );
        let actions = mgr.on_event(PeerEvent::Tick { now: 1081 }, &store);
        assert!(
            actions.contains(&PeerAction::SendPathRequest {
                identity_hash: [0xE1; 16]
            }),
            "Should probe at t=1081 — past 60s backoff from t=1020"
        );
    }

    #[test]
    fn link_established_after_contact_removed_emits_close() {
        let mut mgr = PeerManager::new();
        let store = make_store_with_contact(0xF0, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xF0; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xF0; 16],
            },
            &store,
        );
        // Remove contact — emits CloseLink, removes peer entry
        let actions = mgr.on_event(
            PeerEvent::ContactRemoved {
                identity_hash: [0xF0; 16],
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0xF0; 16],
        }));
        assert!(!mgr.peers.contains_key(&[0xF0; 16]));

        // Race: link completes after removal
        let actions = mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0xF0; 16],
                now: 5000,
            },
            &store,
        );
        // Should emit CloseLink for the dangling link
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0xF0; 16],
        }));
    }

    #[test]
    fn link_established_while_disabled_emits_close() {
        let mut mgr = PeerManager::new();
        let mut store = make_store_with_contact(0xF1, true, PeeringPriority::High);
        mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xF1; 16],
            },
            &store,
        );
        mgr.on_event(
            PeerEvent::AnnounceReceived {
                identity_hash: [0xF1; 16],
            },
            &store,
        );
        // Disable peering while Connecting — emits CloseLink
        store.get_mut(&[0xF1; 16]).unwrap().peering.enabled = false;
        let actions = mgr.on_event(
            PeerEvent::ContactChanged {
                identity_hash: [0xF1; 16],
            },
            &store,
        );
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0xF1; 16],
        }));
        assert_eq!(
            mgr.peers.get(&[0xF1; 16]).unwrap().status,
            PeerStatus::Disabled
        );

        // Race: link completes before CloseLink is processed
        let actions = mgr.on_event(
            PeerEvent::LinkEstablished {
                identity_hash: [0xF1; 16],
                now: 5000,
            },
            &store,
        );
        // Should re-emit CloseLink, NOT transition to Connected
        assert!(actions.contains(&PeerAction::CloseLink {
            identity_hash: [0xF1; 16],
        }));
        assert_eq!(
            mgr.peers.get(&[0xF1; 16]).unwrap().status,
            PeerStatus::Disabled
        );
    }

    #[test]
    fn tunnel_address_emits_initiate_tunnel() {
        let id: IdentityHash = [0xD0; 16];
        let (store, _) = make_tunnel_contact(id, false);
        let mut mgr = PeerManager::new();
        mgr.on_event(PeerEvent::ContactChanged { identity_hash: id }, &store);
        let actions = mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: id }, &store);
        // Must emit InitiateTunnel, not InitiateLink
        assert!(
            actions.contains(&PeerAction::InitiateTunnel {
                identity_hash: id,
                node_id: [0xAA; 32],
                relay_url: Some("https://iroh.q8.fyi".into()),
            }),
            "expected InitiateTunnel, got: {actions:?}"
        );
        assert!(
            !actions.iter().any(|a| matches!(a, PeerAction::InitiateLink { .. })),
            "should NOT emit InitiateLink for tunnel contact"
        );
    }

    #[test]
    fn tunnel_established_transitions_to_connected() {
        let id: IdentityHash = [0xD1; 16];
        let (store, _) = make_tunnel_contact(id, false);
        let mut mgr = PeerManager::new();
        mgr.on_event(PeerEvent::ContactChanged { identity_hash: id }, &store);
        mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: id }, &store);

        let actions = mgr.on_event(
            PeerEvent::TunnelEstablished { identity_hash: id, node_id: [0xAA; 32], now: 7000 },
            &store,
        );
        let peer = mgr.peers.get(&id).unwrap();
        assert_eq!(peer.status, PeerStatus::Connected);
        assert_eq!(peer.retry_count, 0);
        assert!(peer.connection_quality.is_some());
        let cq = peer.connection_quality.as_ref().unwrap();
        assert_eq!(cq.connected_since, 7000);
        assert!(matches!(cq.transport, Transport::Tunnel { relayed: true }));
        assert!(actions.contains(&PeerAction::UpdateLastSeen {
            identity_hash: id,
            timestamp: 7000,
        }));
    }

    #[test]
    fn tunnel_failed_transitions_to_searching() {
        let id: IdentityHash = [0xD2; 16];
        let (store, _) = make_tunnel_contact(id, false);
        let mut mgr = PeerManager::new();
        mgr.on_event(PeerEvent::ContactChanged { identity_hash: id }, &store);
        mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: id }, &store);
        assert_eq!(mgr.peers.get(&id).unwrap().status, PeerStatus::Connecting);

        mgr.on_event(PeerEvent::TunnelFailed { identity_hash: id }, &store);
        let peer = mgr.peers.get(&id).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
        assert!(peer.connection_quality.is_none());
    }

    #[test]
    fn tunnel_dropped_falls_back_to_reticulum() {
        let id: IdentityHash = [0xD3; 16];
        let (store, _) = make_tunnel_contact(id, true); // Tunnel + Reticulum
        let mut mgr = PeerManager::new();
        mgr.on_event(PeerEvent::ContactChanged { identity_hash: id }, &store);
        mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: id }, &store);
        mgr.on_event(
            PeerEvent::TunnelEstablished { identity_hash: id, node_id: [0xAA; 32], now: 8000 },
            &store,
        );
        assert_eq!(mgr.peers.get(&id).unwrap().status, PeerStatus::Connected);

        let actions =
            mgr.on_event(PeerEvent::TunnelDropped { identity_hash: id }, &store);
        let peer = mgr.peers.get(&id).unwrap();
        // Falls back to Reticulum immediately — status is Connecting, not Searching.
        assert_eq!(peer.status, PeerStatus::Connecting);
        assert_eq!(peer.retry_count, 1);
        assert!(peer.connection_quality.is_none());
        // Should immediately emit InitiateLink as Reticulum fallback
        assert!(
            actions.contains(&PeerAction::InitiateLink { identity_hash: id }),
            "expected InitiateLink fallback, got: {actions:?}"
        );
    }

    #[test]
    fn tunnel_dropped_no_fallback_without_reticulum_address() {
        let id: IdentityHash = [0xD4; 16];
        let (store, _) = make_tunnel_contact(id, false); // Tunnel only
        let mut mgr = PeerManager::new();
        mgr.on_event(PeerEvent::ContactChanged { identity_hash: id }, &store);
        mgr.on_event(PeerEvent::AnnounceReceived { identity_hash: id }, &store);
        mgr.on_event(
            PeerEvent::TunnelEstablished { identity_hash: id, node_id: [0xAA; 32], now: 8000 },
            &store,
        );
        assert_eq!(mgr.peers.get(&id).unwrap().status, PeerStatus::Connected);

        let actions =
            mgr.on_event(PeerEvent::TunnelDropped { identity_hash: id }, &store);
        let peer = mgr.peers.get(&id).unwrap();
        assert_eq!(peer.status, PeerStatus::Searching);
        assert_eq!(peer.retry_count, 1);
        // No Reticulum address → no InitiateLink
        assert!(
            !actions.iter().any(|a| matches!(a, PeerAction::InitiateLink { .. })),
            "should NOT emit InitiateLink when there is no Reticulum address"
        );
    }
}

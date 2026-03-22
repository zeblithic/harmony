use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ContactAddress {
    Reticulum { destination_hash: [u8; 16] },
    Tunnel {
        node_id: [u8; 32],
        relay_url: Option<String>,
        direct_addrs: Vec<String>,
    },
}

/// Replication policy: symmetric encrypted backup delegation.
///
/// When set on a `Contact`, both peers provision the specified storage quota
/// for each other's encrypted-durable content. The quota is enforced by the
/// local `ReplicaStore` — content exceeding it is evicted oldest-first.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ReplicationPolicy {
    /// Symmetric storage quota in bytes. Both peers provision this
    /// amount for each other's encrypted-durable content.
    pub quota_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub identity_hash: IdentityHash,
    pub display_name: Option<String>,
    pub peering: PeeringPolicy,
    pub added_at: u64,
    pub last_seen: Option<u64>,
    pub notes: Option<String>,
    pub addresses: Vec<ContactAddress>,
    /// Replication policy: encrypted backup delegation.
    /// None = no replication. Some = active with specified quota.
    pub replication: Option<ReplicationPolicy>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeeringPolicy {
    pub enabled: bool,
    pub priority: PeeringPriority,
}

impl Default for PeeringPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            priority: PeeringPriority::Normal,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PeeringPriority {
    Low,
    Normal,
    High,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_peering_policy_is_disabled_normal() {
        let policy = PeeringPolicy::default();
        assert!(!policy.enabled);
        assert_eq!(policy.priority, PeeringPriority::Normal);
    }

    #[test]
    fn peering_priority_ordering() {
        assert!(PeeringPriority::Low < PeeringPriority::Normal);
        assert!(PeeringPriority::Normal < PeeringPriority::High);
    }

    #[test]
    fn contact_creation() {
        let contact = Contact {
            identity_hash: [0xAB; 16],
            display_name: Some("Alice".into()),
            peering: PeeringPolicy {
                enabled: true,
                priority: PeeringPriority::High,
            },
            added_at: 1710000000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: None,
        };
        assert_eq!(contact.identity_hash, [0xAB; 16]);
        assert!(contact.peering.enabled);
    }

    #[test]
    fn contact_serde_round_trip() {
        let contact = Contact {
            identity_hash: [0x42; 16],
            display_name: Some("Bob".into()),
            peering: PeeringPolicy {
                enabled: true,
                priority: PeeringPriority::Normal,
            },
            added_at: 1710000000,
            last_seen: Some(1710001000),
            notes: Some("My friend".into()),
            addresses: vec![],
            replication: None,
        };
        let bytes = postcard::to_allocvec(&contact).unwrap();
        let decoded: Contact = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.identity_hash, contact.identity_hash);
        assert_eq!(decoded.display_name, contact.display_name);
        assert_eq!(decoded.peering.priority, PeeringPriority::Normal);
        assert_eq!(decoded.last_seen, Some(1710001000));
    }

    #[test]
    fn contact_with_tunnel_address() {
        let contact = Contact {
            identity_hash: [0xCC; 16],
            display_name: None,
            peering: PeeringPolicy::default(),
            added_at: 1710000000,
            last_seen: None,
            notes: None,
            addresses: vec![ContactAddress::Tunnel {
                node_id: [0xEE; 32],
                relay_url: Some("https://iroh.q8.fyi".into()),
                direct_addrs: vec!["192.168.1.10:4242".into()],
            }],
            replication: None,
        };
        assert_eq!(contact.addresses.len(), 1);
        assert!(matches!(
            &contact.addresses[0],
            ContactAddress::Tunnel { node_id, .. } if *node_id == [0xEE; 32]
        ));
    }

    #[test]
    fn contact_address_serde_round_trip() {
        let reticulum_addr = ContactAddress::Reticulum {
            destination_hash: [0x11; 16],
        };
        let bytes = postcard::to_allocvec(&reticulum_addr).unwrap();
        let decoded: ContactAddress = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, reticulum_addr);

        let tunnel_addr = ContactAddress::Tunnel {
            node_id: [0x22; 32],
            relay_url: Some("https://relay.example.com".into()),
            direct_addrs: vec!["192.168.1.1:1234".into(), "10.0.0.1:5678".into()],
        };
        let bytes = postcard::to_allocvec(&tunnel_addr).unwrap();
        let decoded: ContactAddress = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, tunnel_addr);

        let tunnel_addr_no_relay = ContactAddress::Tunnel {
            node_id: [0x33; 32],
            relay_url: None,
            direct_addrs: vec![],
        };
        let bytes = postcard::to_allocvec(&tunnel_addr_no_relay).unwrap();
        let decoded: ContactAddress = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, tunnel_addr_no_relay);
    }

    #[test]
    fn contact_with_addresses_serde_round_trip() {
        let contact = Contact {
            identity_hash: [0x55; 16],
            display_name: Some("Charlie".into()),
            peering: PeeringPolicy {
                enabled: true,
                priority: PeeringPriority::High,
            },
            added_at: 1710000000,
            last_seen: None,
            notes: None,
            addresses: vec![
                ContactAddress::Reticulum {
                    destination_hash: [0xAA; 16],
                },
                ContactAddress::Tunnel {
                    node_id: [0xBB; 32],
                    relay_url: Some("https://relay.example.com".into()),
                    direct_addrs: vec!["127.0.0.1:1234".into()],
                },
            ],
            replication: None,
        };
        let bytes = postcard::to_allocvec(&contact).unwrap();
        let decoded: Contact = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.addresses.len(), 2);
        assert_eq!(
            decoded.addresses[0],
            ContactAddress::Reticulum {
                destination_hash: [0xAA; 16]
            }
        );
    }

    #[test]
    fn replication_policy_roundtrip() {
        let policy = ReplicationPolicy {
            quota_bytes: 50 * 1024 * 1024 * 1024,
        };
        let bytes = postcard::to_allocvec(&policy).unwrap();
        let decoded: ReplicationPolicy = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.quota_bytes, 50 * 1024 * 1024 * 1024);
    }

    #[test]
    fn contact_with_replication_serde_round_trip() {
        let contact = Contact {
            identity_hash: [0x77; 16],
            display_name: Some("Replicated".into()),
            peering: PeeringPolicy::default(),
            added_at: 1710000000,
            last_seen: None,
            notes: None,
            addresses: vec![],
            replication: Some(ReplicationPolicy {
                quota_bytes: 10 * 1024 * 1024 * 1024,
            }),
        };
        let bytes = postcard::to_allocvec(&contact).unwrap();
        let decoded: Contact = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.replication, Some(ReplicationPolicy { quota_bytes: 10 * 1024 * 1024 * 1024 }));
    }
}

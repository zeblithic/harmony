use alloc::string::String;
use alloc::vec::Vec;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

/// A reachable address for a contact.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContactAddress {
    /// iroh-net tunnel: NodeId + optional relay + optional direct addresses.
    Tunnel {
        /// iroh NodeId (32 bytes).
        node_id: [u8; 32],
        /// Preferred relay server URL (e.g., "https://iroh.q8.fyi").
        relay_url: Option<String>,
        /// Known direct socket addresses (ephemeral, may be stale).
        direct_addrs: Vec<String>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub identity_hash: IdentityHash,
    pub display_name: Option<String>,
    pub peering: PeeringPolicy,
    pub added_at: u64,
    pub last_seen: Option<u64>,
    pub notes: Option<String>,
    /// Network addresses where this contact can be reached.
    /// Note: `#[serde(default)]` is ineffective with postcard (non-self-describing).
    /// Backward compat is handled by FORMAT_VERSION gating in ContactStore.
    pub addresses: Vec<ContactAddress>,
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
        };
        assert_eq!(contact.addresses.len(), 1);
        assert!(matches!(
            &contact.addresses[0],
            ContactAddress::Tunnel { node_id, .. } if *node_id == [0xEE; 32]
        ));
    }
}

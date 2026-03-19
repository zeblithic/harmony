use alloc::string::String;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Contact {
    pub identity_hash: IdentityHash,
    pub display_name: Option<String>,
    pub peering: PeeringPolicy,
    pub added_at: u64,
    pub last_seen: Option<u64>,
    pub notes: Option<String>,
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
        };
        let bytes = postcard::to_allocvec(&contact).unwrap();
        let decoded: Contact = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded.identity_hash, contact.identity_hash);
        assert_eq!(decoded.display_name, contact.display_name);
        assert_eq!(decoded.peering.priority, PeeringPriority::Normal);
        assert_eq!(decoded.last_seen, Some(1710001000));
    }
}

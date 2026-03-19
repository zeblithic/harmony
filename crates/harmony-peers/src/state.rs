use harmony_contacts::PeeringPriority;

#[derive(Debug, Clone)]
pub struct PeerState {
    pub status: PeerStatus,
    pub priority: PeeringPriority,
    pub last_probe: Option<u64>,
    pub last_seen: Option<u64>,
    pub connecting_since: Option<u64>,
    pub retry_count: u32,
}

impl PeerState {
    pub fn new(priority: PeeringPriority) -> Self {
        Self {
            status: PeerStatus::Searching,
            priority,
            last_probe: None,
            last_seen: None,
            connecting_since: None,
            retry_count: 0,
        }
    }

    pub fn new_disabled() -> Self {
        Self {
            status: PeerStatus::Disabled,
            priority: PeeringPriority::Normal,
            last_probe: None,
            last_seen: None,
            connecting_since: None,
            retry_count: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeerStatus {
    Searching,
    Connecting,
    Connected,
    Disabled,
}

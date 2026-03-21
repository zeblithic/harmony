use harmony_contacts::PeeringPriority;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transport {
    /// Connected via LAN/mesh Reticulum link.
    /// Not yet constructed — reserved for harmony-k4g (LAN peer discovery).
    #[allow(dead_code)]
    Lan,
    /// Connected via iroh-net QUIC tunnel.
    Tunnel { relayed: bool },
}

#[derive(Debug, Clone)]
pub struct ConnectionQuality {
    pub rtt_ms: Option<u32>,
    pub transport: Transport,
    pub connected_since: u64,
}

#[derive(Debug, Clone)]
pub struct PeerState {
    pub status: PeerStatus,
    pub priority: PeeringPriority,
    pub last_probe: Option<u64>,
    pub last_seen: Option<u64>,
    pub connecting_since: Option<u64>,
    pub retry_count: u32,
    pub connection_quality: Option<ConnectionQuality>,
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
            connection_quality: None,
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
            connection_quality: None,
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

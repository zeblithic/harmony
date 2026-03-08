use crate::error::ReticulumError;
use alloc::vec::Vec;

/// Interface mode (matches Python Interface.MODE_*).
///
/// Determines how a node handles traffic on this interface:
/// routing, path discovery, and announce propagation behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum InterfaceMode {
    /// Full routing node: participates in all transport activities.
    Full = 0x01,
    /// Point-to-point: direct link between two nodes.
    PointToPoint = 0x02,
    /// Access point: provides network access to roaming nodes.
    AccessPoint = 0x03,
    /// Roaming: mobile node connecting through access points.
    Roaming = 0x04,
    /// Boundary: connects two separate network segments.
    Boundary = 0x05,
    /// Gateway: bridge to external networks or transports.
    Gateway = 0x06,
}

impl InterfaceMode {
    /// Whether a Transport node should actively discover paths for
    /// this interface mode. Matches Python `DISCOVER_PATHS_FOR`.
    pub fn discovers_paths(&self) -> bool {
        matches!(self, Self::AccessPoint | Self::Gateway | Self::Roaming)
    }
}

/// Direction capabilities of an interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InterfaceDirection {
    /// Can transmit outbound packets (OUT).
    pub can_send: bool,
    /// Can receive inbound packets (IN).
    pub can_receive: bool,
    /// Can forward packets for other nodes (FWD).
    pub can_forward: bool,
}

impl InterfaceDirection {
    /// Full bidirectional with forwarding.
    pub fn bidirectional() -> Self {
        Self {
            can_send: true,
            can_receive: true,
            can_forward: true,
        }
    }

    /// Receive-only interface (e.g., a passive monitor).
    pub fn inbound_only() -> Self {
        Self {
            can_send: false,
            can_receive: true,
            can_forward: false,
        }
    }

    /// Transmit-only interface (e.g., a beacon).
    pub fn outbound_only() -> Self {
        Self {
            can_send: true,
            can_receive: false,
            can_forward: false,
        }
    }
}

/// Interface traffic statistics.
#[derive(Debug, Clone, Default)]
pub struct InterfaceStats {
    pub bytes_sent: u64,
    pub bytes_received: u64,
}

/// Transport-agnostic interface trait.
///
/// This is the I/O boundary in the sans-I/O architecture. Protocol logic
/// above this trait is pure computation; concrete implementations provide
/// actual send/receive over TCP, UDP, serial, LoRa, Zenoh, etc.
pub trait Interface {
    /// Human-readable interface name.
    fn name(&self) -> &str;

    /// Maximum transmission unit in bytes.
    fn mtu(&self) -> usize;

    /// Whether the interface is currently online.
    fn online(&self) -> bool;

    /// Nominal bitrate in bits per second.
    fn bitrate(&self) -> u64;

    /// Interface operating mode.
    fn mode(&self) -> InterfaceMode;

    /// Direction capabilities.
    fn direction(&self) -> InterfaceDirection;

    /// Traffic statistics.
    fn stats(&self) -> &InterfaceStats;

    /// Queue raw bytes for outbound transmission.
    fn send(&mut self, data: &[u8]) -> Result<(), ReticulumError>;

    /// Dequeue a received packet, if any.
    fn receive(&mut self) -> Option<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_values_match_python() {
        assert_eq!(InterfaceMode::Full as u8, 0x01);
        assert_eq!(InterfaceMode::PointToPoint as u8, 0x02);
        assert_eq!(InterfaceMode::AccessPoint as u8, 0x03);
        assert_eq!(InterfaceMode::Roaming as u8, 0x04);
        assert_eq!(InterfaceMode::Boundary as u8, 0x05);
        assert_eq!(InterfaceMode::Gateway as u8, 0x06);
    }

    #[test]
    fn discovers_paths_matches_python() {
        // Python DISCOVER_PATHS_FOR = [MODE_ACCESS_POINT, MODE_GATEWAY, MODE_ROAMING]
        assert!(!InterfaceMode::Full.discovers_paths());
        assert!(!InterfaceMode::PointToPoint.discovers_paths());
        assert!(InterfaceMode::AccessPoint.discovers_paths());
        assert!(InterfaceMode::Roaming.discovers_paths());
        assert!(!InterfaceMode::Boundary.discovers_paths());
        assert!(InterfaceMode::Gateway.discovers_paths());
    }

    #[test]
    fn bidirectional_direction() {
        let d = InterfaceDirection::bidirectional();
        assert!(d.can_send);
        assert!(d.can_receive);
        assert!(d.can_forward);
    }

    #[test]
    fn inbound_only_direction() {
        let d = InterfaceDirection::inbound_only();
        assert!(!d.can_send);
        assert!(d.can_receive);
        assert!(!d.can_forward);
    }

    #[test]
    fn outbound_only_direction() {
        let d = InterfaceDirection::outbound_only();
        assert!(d.can_send);
        assert!(!d.can_receive);
        assert!(!d.can_forward);
    }

    #[test]
    fn stats_default_zero() {
        let stats = InterfaceStats::default();
        assert_eq!(stats.bytes_sent, 0);
        assert_eq!(stats.bytes_received, 0);
    }
}

use alloc::{
    collections::VecDeque,
    string::{String, ToString},
    vec::Vec,
};

use crate::error::ReticulumError;
use crate::interface::{Interface, InterfaceDirection, InterfaceMode, InterfaceStats};
use crate::packet::MTU;

/// In-memory interface for testing.
///
/// Sans-I/O design: the caller explicitly transfers packets between
/// interfaces using `deliver_to()`. No shared state or threading needed.
pub struct LoopbackInterface {
    name: String,
    outbox: VecDeque<Vec<u8>>,
    inbox: VecDeque<Vec<u8>>,
    stats: InterfaceStats,
    online: bool,
    mode: InterfaceMode,
    bitrate: u64,
}

impl LoopbackInterface {
    /// Create a new loopback interface with the given name.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            outbox: VecDeque::new(),
            inbox: VecDeque::new(),
            stats: InterfaceStats::default(),
            online: true,
            mode: InterfaceMode::Full,
            bitrate: 1_000_000_000, // 1 Gbps virtual
        }
    }

    /// Transfer all outbox packets to the peer's inbox.
    ///
    /// This simulates a physical link: everything sent from `self` arrives
    /// at `peer`. Call this in both directions for bidirectional communication.
    pub fn deliver_to(&mut self, peer: &mut Self) {
        while let Some(packet) = self.outbox.pop_front() {
            peer.stats.bytes_received += packet.len() as u64;
            peer.inbox.push_back(packet);
        }
    }

    /// Set online/offline state.
    pub fn set_online(&mut self, online: bool) {
        self.online = online;
    }
}

impl Interface for LoopbackInterface {
    fn name(&self) -> &str {
        &self.name
    }

    fn mtu(&self) -> usize {
        MTU
    }

    fn online(&self) -> bool {
        self.online
    }

    fn bitrate(&self) -> u64 {
        self.bitrate
    }

    fn mode(&self) -> InterfaceMode {
        self.mode
    }

    fn direction(&self) -> InterfaceDirection {
        InterfaceDirection::bidirectional()
    }

    fn stats(&self) -> &InterfaceStats {
        &self.stats
    }

    fn send(&mut self, data: &[u8]) -> Result<(), ReticulumError> {
        if !self.online {
            return Err(ReticulumError::InterfaceOffline);
        }
        self.stats.bytes_sent += data.len() as u64;
        self.outbox.push_back(data.to_vec());
        Ok(())
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        self.inbox.pop_front()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn send_receive_roundtrip_via_deliver_to() {
        let mut alice = LoopbackInterface::new("alice");
        let mut bob = LoopbackInterface::new("bob");

        let payload = vec![0x01, 0x02, 0x03, 0x04];
        alice.send(&payload).unwrap();
        alice.deliver_to(&mut bob);

        let received = bob.receive().unwrap();
        assert_eq!(received, payload);
    }

    #[test]
    fn send_while_offline_rejected() {
        let mut iface = LoopbackInterface::new("offline");
        iface.set_online(false);

        let result = iface.send(&[0x01]);
        assert!(matches!(result, Err(ReticulumError::InterfaceOffline)));
    }

    #[test]
    fn receive_returns_none_when_empty() {
        let mut iface = LoopbackInterface::new("empty");
        assert!(iface.receive().is_none());
    }

    #[test]
    fn multiple_packets_delivered_in_order() {
        let mut sender = LoopbackInterface::new("sender");
        let mut receiver = LoopbackInterface::new("receiver");

        sender.send(&[1]).unwrap();
        sender.send(&[2]).unwrap();
        sender.send(&[3]).unwrap();
        sender.deliver_to(&mut receiver);

        assert_eq!(receiver.receive().unwrap(), vec![1]);
        assert_eq!(receiver.receive().unwrap(), vec![2]);
        assert_eq!(receiver.receive().unwrap(), vec![3]);
        assert!(receiver.receive().is_none());
    }

    #[test]
    fn stats_updated_on_send_and_receive() {
        let mut alice = LoopbackInterface::new("alice");
        let mut bob = LoopbackInterface::new("bob");

        alice.send(&[0u8; 100]).unwrap();
        assert_eq!(alice.stats().bytes_sent, 100);

        alice.deliver_to(&mut bob);
        assert_eq!(bob.stats().bytes_received, 100);

        bob.receive();
        // bytes_received is updated on delivery, not on receive()
        assert_eq!(bob.stats().bytes_received, 100);
    }

    #[test]
    fn bidirectional_communication() {
        let mut alice = LoopbackInterface::new("alice");
        let mut bob = LoopbackInterface::new("bob");

        alice.send(b"hello bob").unwrap();
        alice.deliver_to(&mut bob);
        assert_eq!(bob.receive().unwrap(), b"hello bob");

        bob.send(b"hello alice").unwrap();
        bob.deliver_to(&mut alice);
        assert_eq!(alice.receive().unwrap(), b"hello alice");
    }

    #[test]
    fn default_properties() {
        let iface = LoopbackInterface::new("test");
        assert_eq!(iface.name(), "test");
        assert_eq!(iface.mtu(), MTU);
        assert!(iface.online());
        assert_eq!(iface.bitrate(), 1_000_000_000);
        assert_eq!(iface.mode(), InterfaceMode::Full);
        assert_eq!(iface.direction(), InterfaceDirection::bidirectional());
    }

    #[test]
    fn set_online_toggles_state() {
        let mut iface = LoopbackInterface::new("toggle");
        assert!(iface.online());

        iface.set_online(false);
        assert!(!iface.online());

        iface.set_online(true);
        assert!(iface.online());
    }
}

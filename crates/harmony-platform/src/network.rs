use alloc::vec::Vec;

use crate::error::PlatformError;

/// Platform-provided network I/O for the event loop.
///
/// The event loop polls [`NetworkInterface::receive`] for inbound bytes and
/// calls [`NetworkInterface::send`] to actuate `SendOnInterface` actions from
/// the sans-I/O state machines. This bridges real I/O to the event/action model.
///
/// The existing `Interface` trait in `harmony-reticulum` carries
/// protocol-specific metadata (mode, bitrate, stats). `NetworkInterface`
/// is the platform bridge — focused purely on byte I/O.
pub trait NetworkInterface {
    /// Human-readable interface name (e.g., `"eth0"`, `"lora0"`).
    fn name(&self) -> &str;

    /// Maximum transmission unit in bytes.
    fn mtu(&self) -> usize;

    /// Send raw bytes on this interface.
    fn send(&mut self, data: &[u8]) -> Result<(), PlatformError>;

    /// Poll for one inbound packet. Returns `None` if no data is available.
    fn receive(&mut self) -> Option<Vec<u8>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::VecDeque;
    use alloc::string::String;

    struct TestLoopback {
        name: String,
        mtu: usize,
        inbox: VecDeque<Vec<u8>>,
        sent: Vec<Vec<u8>>,
    }

    impl TestLoopback {
        fn new(name: &str, mtu: usize) -> Self {
            TestLoopback {
                name: name.into(),
                mtu,
                inbox: VecDeque::new(),
                sent: Vec::new(),
            }
        }

        fn inject(&mut self, data: &[u8]) {
            self.inbox.push_back(data.to_vec());
        }
    }

    impl NetworkInterface for TestLoopback {
        fn name(&self) -> &str {
            &self.name
        }

        fn mtu(&self) -> usize {
            self.mtu
        }

        fn send(&mut self, data: &[u8]) -> Result<(), PlatformError> {
            if data.len() > self.mtu {
                return Err(PlatformError::SendFailed);
            }
            self.sent.push(data.to_vec());
            Ok(())
        }

        fn receive(&mut self) -> Option<Vec<u8>> {
            self.inbox.pop_front()
        }
    }

    #[test]
    fn send_and_receive_round_trip() {
        let mut iface = TestLoopback::new("lo0", 500);
        iface.inject(b"hello mesh");
        let received = iface.receive().unwrap();
        assert_eq!(received, b"hello mesh");
        assert!(iface.receive().is_none());
    }

    #[test]
    fn send_records_outbound_data() {
        let mut iface = TestLoopback::new("eth0", 1500);
        iface.send(b"packet-one").unwrap();
        iface.send(b"packet-two").unwrap();
        assert_eq!(iface.sent.len(), 2);
        assert_eq!(iface.sent[0], b"packet-one");
    }

    #[test]
    fn send_exceeding_mtu_fails() {
        let mut iface = TestLoopback::new("lora0", 4);
        let result = iface.send(b"too-long");
        assert!(result.is_err());
    }

    #[test]
    fn name_and_mtu_accessors() {
        let iface = TestLoopback::new("wlan0", 1400);
        assert_eq!(iface.name(), "wlan0");
        assert_eq!(iface.mtu(), 1400);
    }

    #[test]
    fn works_as_trait_object() {
        let mut iface = TestLoopback::new("dyn0", 500);
        let net: &mut dyn NetworkInterface = &mut iface;
        assert_eq!(net.name(), "dyn0");
        net.send(b"via-dyn").unwrap();
    }
}

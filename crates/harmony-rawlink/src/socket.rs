//! Platform abstraction trait for raw Ethernet sockets and a channel-backed mock.

use crate::error::RawLinkError;

/// Abstraction over a raw Ethernet socket.
///
/// `send_frame` accepts a destination MAC and the payload bytes that follow the
/// Ethernet header (EtherType + frame type tag + body). The implementation is
/// responsible for prepending the full Ethernet header on the wire.
///
/// `recv_frames` drains all available incoming frames and invokes `callback`
/// once per frame with `(src_mac, payload_after_ethernet_header)`.
pub trait RawSocket: Send {
    /// Send a frame to `dst_mac` with the given payload.
    fn send_frame(&mut self, dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError>;

    /// Drain incoming frames, calling `callback` for each one.
    fn recv_frames(&mut self, callback: &mut dyn FnMut(&[u8; 6], &[u8])) -> Result<(), RawLinkError>;

    /// Returns the local MAC address associated with this socket.
    fn local_mac(&self) -> [u8; 6];
}

/// A loopback pair of mock sockets backed by `std::sync::mpsc` channels.
///
/// Only compiled in `#[cfg(test)]` so it never ships in production builds.
#[cfg(test)]
pub struct MockSocket {
    mac: [u8; 6],
    tx: std::sync::mpsc::Sender<([u8; 6], Vec<u8>)>,
    rx: std::sync::mpsc::Receiver<([u8; 6], Vec<u8>)>,
}

#[cfg(test)]
impl MockSocket {
    /// Creates a connected pair of mock sockets.
    ///
    /// Frames sent on `a` are received on `b`, and vice-versa.
    pub fn pair(mac_a: [u8; 6], mac_b: [u8; 6]) -> (Self, Self) {
        let (tx_a, rx_b) = std::sync::mpsc::channel();
        let (tx_b, rx_a) = std::sync::mpsc::channel();

        let a = MockSocket { mac: mac_a, tx: tx_a, rx: rx_a };
        let b = MockSocket { mac: mac_b, tx: tx_b, rx: rx_b };
        (a, b)
    }
}

#[cfg(test)]
impl RawSocket for MockSocket {
    fn send_frame(&mut self, _dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError> {
        self.tx
            .send((self.mac, payload.to_vec()))
            .map_err(|e| RawLinkError::SocketError(format!("mock send failed: {e}")))?;
        Ok(())
    }

    fn recv_frames(
        &mut self,
        callback: &mut dyn FnMut(&[u8; 6], &[u8]),
    ) -> Result<(), RawLinkError> {
        // Drain all currently available messages without blocking.
        loop {
            match self.rx.try_recv() {
                Ok((src_mac, payload)) => callback(&src_mac, &payload),
                Err(std::sync::mpsc::TryRecvError::Empty) => break,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Err(RawLinkError::SocketError(
                        "mock channel disconnected".into(),
                    ));
                }
            }
        }
        Ok(())
    }

    fn local_mac(&self) -> [u8; 6] {
        self.mac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MAC_A: [u8; 6] = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
    const MAC_B: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];

    #[test]
    fn mock_socket_local_mac() {
        let (a, b) = MockSocket::pair(MAC_A, MAC_B);
        assert_eq!(a.local_mac(), MAC_A);
        assert_eq!(b.local_mac(), MAC_B);
    }

    #[test]
    fn mock_socket_send_recv_round_trip() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);
        let payload = b"hello from A";

        // A sends to B
        a.send_frame(MAC_B, payload).expect("send should succeed");

        let mut received: Vec<([u8; 6], Vec<u8>)> = Vec::new();
        b.recv_frames(&mut |src, data| {
            received.push((*src, data.to_vec()));
        })
        .expect("recv should succeed");

        assert_eq!(received.len(), 1);
        assert_eq!(received[0].0, MAC_A);
        assert_eq!(received[0].1, payload);
    }

    #[test]
    fn mock_socket_empty_recv() {
        let (_a, mut b) = MockSocket::pair(MAC_A, MAC_B);

        let mut count = 0usize;
        b.recv_frames(&mut |_, _| count += 1)
            .expect("empty recv should not error");

        assert_eq!(count, 0, "no frames were sent, callback must not fire");
    }
}

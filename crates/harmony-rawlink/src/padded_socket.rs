//! Socket wrapper that pads outgoing frames to uniform block sizes.
//!
//! Prevents traffic analysis based on frame size — a 31-byte scout frame and
//! a 500-byte Reticulum announce look identical on the wire (both 1024 bytes).

use crate::error::RawLinkError;
use crate::socket::RawSocket;
use crate::ETH_HEADER_LEN;

/// Default padding block size. All frames are padded to the nearest multiple
/// of this value (accounting for the Ethernet header that the socket prepends).
pub const DEFAULT_PAD_BLOCK: usize = 1024;

/// Wraps a `RawSocket` to pad outgoing payloads to uniform block sizes.
///
/// The total frame on the wire (Ethernet header + payload) is rounded up to
/// the nearest multiple of `pad_block`. Padding bytes are zero-filled.
///
/// Receiving is unaffected — callers are expected to know the true payload
/// length from protocol-level framing (length fields, type tags, etc.).
pub struct PaddedSocket<S> {
    inner: S,
    pad_block: usize,
}

impl<S> PaddedSocket<S> {
    pub fn new(inner: S, pad_block: usize) -> Self {
        assert!(pad_block > ETH_HEADER_LEN, "pad_block must exceed Ethernet header");
        Self { inner, pad_block }
    }
}

impl<S: RawSocket> RawSocket for PaddedSocket<S> {
    fn send_frame(&mut self, dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError> {
        let wire_len = ETH_HEADER_LEN + payload.len();
        let padded_wire_len = wire_len.next_multiple_of(self.pad_block);
        let padded_payload_len = padded_wire_len - ETH_HEADER_LEN;

        if padded_payload_len == payload.len() {
            // Already aligned — no allocation needed.
            return self.inner.send_frame(dst_mac, payload);
        }

        let mut padded = vec![0u8; padded_payload_len];
        padded[..payload.len()].copy_from_slice(payload);
        self.inner.send_frame(dst_mac, &padded)
    }

    fn recv_frames(
        &mut self,
        callback: &mut dyn FnMut(&[u8; 6], &[u8]),
    ) -> Result<(), RawLinkError> {
        self.inner.recv_frames(callback)
    }

    fn local_mac(&self) -> [u8; 6] {
        self.inner.local_mac()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::socket::MockSocket;

    const MAC_A: [u8; 6] = [0x00; 6];
    const MAC_B: [u8; 6] = [0xFF; 6];

    #[test]
    fn pads_small_payload_to_block_boundary() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);
        // Wrap sender only.
        let mut padded = PaddedSocket::new(a, 1024);

        // 17-byte payload (scout frame) + 14-byte ETH header = 31 bytes.
        // Should pad to 1024 total → payload becomes 1024 - 14 = 1010 bytes.
        let payload = vec![0x01; 17];
        padded.send_frame(MAC_B, &payload).unwrap();

        let mut received = Vec::new();
        b.recv_frames(&mut |_src, data| received.push(data.to_vec()))
            .unwrap();

        assert_eq!(received.len(), 1);
        assert_eq!(received[0].len(), 1024 - ETH_HEADER_LEN);
        // Original payload preserved at the start.
        assert_eq!(&received[0][..17], &payload[..]);
        // Padding is zero-filled.
        assert!(received[0][17..].iter().all(|&b| b == 0));
    }

    #[test]
    fn pads_medium_payload() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);
        let mut padded = PaddedSocket::new(a, 1024);

        // 500-byte Reticulum packet + 1-byte type tag = 501 bytes payload.
        // + 14 ETH header = 515 bytes → pad to 1024.
        let payload = vec![0xAB; 501];
        padded.send_frame(MAC_B, &payload).unwrap();

        let mut received = Vec::new();
        b.recv_frames(&mut |_src, data| received.push(data.to_vec()))
            .unwrap();

        assert_eq!(received[0].len(), 1024 - ETH_HEADER_LEN);
        assert_eq!(&received[0][..501], &payload[..]);
    }

    #[test]
    fn large_payload_pads_to_next_block() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);
        let mut padded = PaddedSocket::new(a, 1024);

        // 1100-byte payload + 14 ETH = 1114 bytes → pad to 2048.
        let payload = vec![0xCD; 1100];
        padded.send_frame(MAC_B, &payload).unwrap();

        let mut received = Vec::new();
        b.recv_frames(&mut |_src, data| received.push(data.to_vec()))
            .unwrap();

        assert_eq!(received[0].len(), 2048 - ETH_HEADER_LEN);
        assert_eq!(&received[0][..1100], &payload[..]);
    }

    #[test]
    fn exactly_aligned_payload_not_re_padded() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);
        let mut padded = PaddedSocket::new(a, 1024);

        // payload = 1024 - 14 = 1010 bytes → wire = 1024, already aligned.
        let payload = vec![0xEE; 1024 - ETH_HEADER_LEN];
        padded.send_frame(MAC_B, &payload).unwrap();

        let mut received = Vec::new();
        b.recv_frames(&mut |_src, data| received.push(data.to_vec()))
            .unwrap();

        assert_eq!(received[0].len(), 1024 - ETH_HEADER_LEN);
    }

    #[test]
    fn recv_passes_through_unchanged() {
        let (mut a, mut b) = MockSocket::pair(MAC_A, MAC_B);

        // Send unpadded from raw socket A.
        a.send_frame(MAC_B, &[1, 2, 3]).unwrap();

        // Receive through padded wrapper on B.
        let mut padded_b = PaddedSocket::new(b, 1024);
        let mut received = Vec::new();
        padded_b
            .recv_frames(&mut |_src, data| received.push(data.to_vec()))
            .unwrap();

        // Received data is exactly what was sent — no padding on receive.
        assert_eq!(received[0], &[1, 2, 3]);
    }

    #[test]
    fn local_mac_delegates() {
        let (a, _b) = MockSocket::pair(MAC_A, MAC_B);
        let padded = PaddedSocket::new(a, 1024);
        assert_eq!(padded.local_mac(), MAC_A);
    }
}

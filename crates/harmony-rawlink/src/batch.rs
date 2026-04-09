//! Batch frame encoding and decoding for the Harmony L2 protocol.
//!
//! Packs multiple Data/Reticulum sub-frames into a single Ethernet broadcast
//! to reduce PHY preamble overhead on WiFi mesh (where broadcast frames cannot
//! use A-MSDU/A-MPDU aggregation).
//!
//! Wire format:
//! ```text
//! [0x03]                                ← BATCH frame type
//! [1 sub_type][2 sub_len BE][payload…]  ← sub-frame 1
//! [1 sub_type][2 sub_len BE][payload…]  ← sub-frame 2
//! …
//! ```

/// Overhead per sub-frame entry: 1 byte type + 2 bytes length (big-endian).
const SUB_FRAME_HEADER: usize = 3;

/// Accumulates outgoing sub-frames into a single BATCH frame payload.
///
/// The bridge drains outbound channels into the accumulator each poll cycle,
/// then calls [`flush`] to get a complete batch payload for `send_frame()`.
pub struct BatchAccumulator {
    buf: Vec<u8>,
    max_payload: usize,
}

impl BatchAccumulator {
    /// Creates a new accumulator for the given Ethernet MTU.
    ///
    /// `max_payload` is computed as `mtu - 14 (ETH_HEADER_LEN) - 1 (batch type byte)`.
    /// For standard 1500-byte MTU, this is 1485.
    pub fn new(mtu: usize) -> Self {
        let max_payload = mtu.saturating_sub(14 + 1);
        Self {
            buf: Vec::new(),
            max_payload,
        }
    }

    /// Returns true if no sub-frames have been pushed since the last flush.
    pub fn is_empty(&self) -> bool {
        self.buf.is_empty()
    }

    /// Appends a sub-frame to the batch.
    ///
    /// Returns `Some(batch)` if the current batch was full and had to be flushed
    /// to make room. The returned batch is ready to pass to `send_frame()`.
    /// Returns `None` if the frame fit in the current batch.
    pub fn push(&mut self, frame_type: u8, payload: &[u8]) -> Option<Vec<u8>> {
        let entry_size = SUB_FRAME_HEADER + payload.len();

        // If the buffer is empty, start a new batch with the type byte.
        if self.buf.is_empty() {
            self.buf.reserve(1 + entry_size);
            self.buf.push(crate::frame_type::BATCH);
        }

        // Check if this entry fits in the current batch.
        // buf already contains the 0x03 prefix + previous entries.
        if self.buf.len() + entry_size > 1 + self.max_payload {
            // Auto-flush: take current batch, start fresh with this entry.
            let completed = std::mem::take(&mut self.buf);
            self.buf.reserve(1 + entry_size);
            self.buf.push(crate::frame_type::BATCH);
            self.append_entry(frame_type, payload);
            Some(completed)
        } else {
            self.append_entry(frame_type, payload);
            None
        }
    }

    /// Returns the completed batch if non-empty, resetting the buffer.
    ///
    /// Returns `None` if no sub-frames have been pushed since the last flush.
    pub fn flush(&mut self) -> Option<Vec<u8>> {
        if self.buf.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.buf))
        }
    }

    /// Appends a sub-frame entry `[type][len BE][payload]` to the buffer.
    fn append_entry(&mut self, frame_type: u8, payload: &[u8]) {
        self.buf.push(frame_type);
        self.buf
            .extend_from_slice(&(payload.len() as u16).to_be_bytes());
        self.buf.extend_from_slice(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_accumulator_is_empty() {
        let acc = BatchAccumulator::new(1500);
        assert!(acc.is_empty());
        assert_eq!(acc.max_payload, 1485);
    }

    #[test]
    fn push_single_frame_and_flush() {
        let mut acc = BatchAccumulator::new(1500);
        let payload = b"hello";
        let auto_flush = acc.push(frame_type::DATA, payload);
        assert!(auto_flush.is_none(), "single small frame should not auto-flush");
        assert!(!acc.is_empty());

        let batch = acc.flush().expect("should produce a batch");
        assert!(acc.is_empty(), "flush should reset");

        // Verify wire format: [0x03][0x02][0x00][0x05][hello]
        assert_eq!(batch[0], frame_type::BATCH);
        assert_eq!(batch[1], frame_type::DATA);
        assert_eq!(u16::from_be_bytes([batch[2], batch[3]]), 5);
        assert_eq!(&batch[4..9], b"hello");
        assert_eq!(batch.len(), 1 + 3 + 5); // batch_type + sub_header + payload
    }

    #[test]
    fn push_multiple_frames_and_flush() {
        let mut acc = BatchAccumulator::new(1500);

        let ret_payload = vec![0xAA; 100];
        assert!(acc.push(frame_type::RETICULUM, &ret_payload).is_none());

        let data_payload = b"harmony/test";
        assert!(acc.push(frame_type::DATA, data_payload).is_none());

        let batch = acc.flush().expect("should produce a batch");

        // [0x03] [0x00][0x00][0x64][100 bytes] [0x02][0x00][0x0C][12 bytes]
        assert_eq!(batch[0], frame_type::BATCH);

        // Sub-frame 1: RETICULUM, len=100
        assert_eq!(batch[1], frame_type::RETICULUM);
        assert_eq!(u16::from_be_bytes([batch[2], batch[3]]), 100);
        assert_eq!(&batch[4..104], &ret_payload[..]);

        // Sub-frame 2: DATA, len=12
        assert_eq!(batch[104], frame_type::DATA);
        assert_eq!(u16::from_be_bytes([batch[105], batch[106]]), 12);
        assert_eq!(&batch[107..119], data_payload);

        assert_eq!(batch.len(), 1 + (3 + 100) + (3 + 12));
    }

    #[test]
    fn auto_flush_on_overflow() {
        // Use a tiny MTU to force overflow quickly.
        // MTU=50 → max_payload = 50 - 14 - 1 = 35
        // Batch buf can hold: 1 (type) + 35 = 36 bytes total.
        // First push: 1 (type) + 3 (header) + 20 (payload) = 24 bytes — fits.
        // Second push: 24 + 3 + 20 = 47 > 36 — triggers auto-flush.
        let mut acc = BatchAccumulator::new(50);
        let payload = vec![0xBB; 20];

        assert!(acc.push(frame_type::DATA, &payload).is_none());

        let flushed = acc.push(frame_type::DATA, &payload);
        assert!(flushed.is_some(), "second push should auto-flush");

        let batch1 = flushed.unwrap();
        // First batch contains only the first sub-frame.
        assert_eq!(batch1[0], frame_type::BATCH);
        assert_eq!(batch1.len(), 1 + 3 + 20);

        // Second sub-frame is in the accumulator, waiting for flush.
        let batch2 = acc.flush().expect("should have second batch");
        assert_eq!(batch2[0], frame_type::BATCH);
        assert_eq!(batch2.len(), 1 + 3 + 20);
    }

    #[test]
    fn empty_flush_returns_none() {
        let mut acc = BatchAccumulator::new(1500);
        assert!(acc.flush().is_none());
    }

    #[test]
    fn double_flush_returns_none() {
        let mut acc = BatchAccumulator::new(1500);
        acc.push(frame_type::DATA, b"hello");
        acc.flush().expect("first flush should produce batch");
        assert!(acc.flush().is_none(), "second flush should be empty");
    }

    #[test]
    fn max_size_sub_frame() {
        let acc = BatchAccumulator::new(1500);
        // Max sub-frame payload: max_payload - SUB_FRAME_HEADER = 1485 - 3 = 1482
        let max_sub_payload = acc.max_payload - SUB_FRAME_HEADER;
        assert_eq!(max_sub_payload, 1482);

        let mut acc = BatchAccumulator::new(1500);
        let payload = vec![0xCC; 1482];
        assert!(acc.push(frame_type::DATA, &payload).is_none());

        let batch = acc.flush().expect("should produce batch");
        // 1 (batch type) + 3 (sub header) + 1482 (payload) = 1486
        assert_eq!(batch.len(), 1 + 3 + 1482);
    }
}

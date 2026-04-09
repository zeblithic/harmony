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
        let max_payload = mtu.saturating_sub(crate::ETH_HEADER_LEN + 1);
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
        debug_assert_ne!(frame_type, crate::frame_type::BATCH, "batch frames cannot be nested");
        let entry_size = SUB_FRAME_HEADER + payload.len();

        // If the buffer is empty, start a new batch with the type byte.
        if self.buf.is_empty() {
            self.buf.reserve(1 + entry_size);
            self.buf.push(crate::frame_type::BATCH);
        }

        // Check if this entry fits in the current batch.
        // buf already contains the 0x03 prefix + previous entries.
        // Only auto-flush if there are already sub-frames in the batch (buf.len() > 1);
        // an oversized first entry is placed in a solo batch and will be dropped by the
        // kernel — matching existing behavior for oversized standalone frames.
        if self.buf.len() > 1 && self.buf.len() + entry_size > 1 + self.max_payload {
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

/// Iterator over sub-frames in a BATCH payload.
///
/// Yields `(frame_type, sub_payload)` for each sub-frame. Stops when the
/// payload is exhausted or a sub-frame header is truncated.
pub struct BatchIter<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> Iterator for BatchIter<'a> {
    type Item = (u8, &'a [u8]);

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.data.len() - self.pos;
        if remaining < SUB_FRAME_HEADER {
            return None;
        }

        let frame_type = self.data[self.pos];
        let len =
            u16::from_be_bytes([self.data[self.pos + 1], self.data[self.pos + 2]]) as usize;
        let payload_start = self.pos + SUB_FRAME_HEADER;
        let payload_end = payload_start + len;

        if payload_end > self.data.len() {
            // Truncated sub-frame — stop iteration.
            return None;
        }

        self.pos = payload_end;
        Some((frame_type, &self.data[payload_start..payload_end]))
    }
}

impl std::iter::FusedIterator for BatchIter<'_> {}

/// Decodes a BATCH frame payload into an iterator of sub-frames.
///
/// The input `payload` must include the leading `0x03` batch type byte
/// (as produced by [`BatchAccumulator::flush`]). The iterator skips the
/// first byte and yields `(frame_type, sub_payload)` for each sub-frame.
///
/// Stops cleanly on truncated data — already-yielded sub-frames are valid.
pub fn decode_batch(payload: &[u8]) -> BatchIter<'_> {
    debug_assert!(
        payload.is_empty() || payload.first() == Some(&crate::frame_type::BATCH),
        "decode_batch expects BATCH frame (0x03 prefix)"
    );
    // Skip the 0x03 batch type byte.
    let start = if payload.first() == Some(&crate::frame_type::BATCH) {
        1
    } else {
        0
    };
    BatchIter {
        data: payload,
        pos: start,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame_type;

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

    #[test]
    fn oversized_sub_frame_produces_solo_batch() {
        // A sub-frame larger than max_payload is placed in a solo batch.
        // The kernel will drop the oversized Ethernet frame — this matches
        // existing behavior for oversized standalone frames.
        let mut acc = BatchAccumulator::new(1500);
        let oversized = vec![0xDD; 1500]; // way bigger than 1482 max
        assert!(acc.push(frame_type::DATA, &oversized).is_none());

        let batch = acc.flush().expect("should produce oversized batch");
        assert_eq!(batch[0], frame_type::BATCH);
        // 1 (batch) + 3 (header) + 1500 (payload) = 1504, exceeds MTU
        assert_eq!(batch.len(), 1 + 3 + 1500);
    }

    #[test]
    fn decode_roundtrip() {
        let mut acc = BatchAccumulator::new(1500);
        acc.push(frame_type::RETICULUM, &[0xAA; 50]);
        acc.push(frame_type::DATA, b"harmony/topic/data-payload-here");
        acc.push(frame_type::RETICULUM, &[0xBB; 10]);
        let batch = acc.flush().unwrap();

        let subs: Vec<(u8, Vec<u8>)> = decode_batch(&batch)
            .map(|(t, p)| (t, p.to_vec()))
            .collect();

        assert_eq!(subs.len(), 3);
        assert_eq!(subs[0].0, frame_type::RETICULUM);
        assert_eq!(subs[0].1, vec![0xAA; 50]);
        assert_eq!(subs[1].0, frame_type::DATA);
        assert_eq!(subs[1].1, b"harmony/topic/data-payload-here");
        assert_eq!(subs[2].0, frame_type::RETICULUM);
        assert_eq!(subs[2].1, vec![0xBB; 10]);
    }

    #[test]
    fn decode_truncated_stops_cleanly() {
        let mut acc = BatchAccumulator::new(1500);
        acc.push(frame_type::DATA, b"valid");
        acc.push(frame_type::RETICULUM, &[0xCC; 40]);
        let mut batch = acc.flush().unwrap();

        // Truncate the second sub-frame's payload (chop 10 bytes off the end).
        batch.truncate(batch.len() - 10);

        let subs: Vec<(u8, Vec<u8>)> = decode_batch(&batch)
            .map(|(t, p)| (t, p.to_vec()))
            .collect();

        // Only the first sub-frame should be yielded.
        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].0, frame_type::DATA);
        assert_eq!(subs[0].1, b"valid");
    }

    #[test]
    fn decode_unknown_type_skipped_by_length() {
        // Manually build a batch with an unknown type (0xFF) between two valid ones.
        let mut batch = vec![frame_type::BATCH];
        // Sub-frame 1: RETICULUM, 3 bytes
        batch.push(frame_type::RETICULUM);
        batch.extend_from_slice(&3u16.to_be_bytes());
        batch.extend_from_slice(&[0xAA; 3]);
        // Sub-frame 2: unknown type 0xFF, 5 bytes
        batch.push(0xFF);
        batch.extend_from_slice(&5u16.to_be_bytes());
        batch.extend_from_slice(&[0x00; 5]);
        // Sub-frame 3: DATA, 2 bytes
        batch.push(frame_type::DATA);
        batch.extend_from_slice(&2u16.to_be_bytes());
        batch.extend_from_slice(&[0xBB; 2]);

        let subs: Vec<(u8, Vec<u8>)> = decode_batch(&batch)
            .map(|(t, p)| (t, p.to_vec()))
            .collect();

        // All three should be yielded — the caller decides what to do with unknown types.
        assert_eq!(subs.len(), 3);
        assert_eq!(subs[0].0, frame_type::RETICULUM);
        assert_eq!(subs[1].0, 0xFF);
        assert_eq!(subs[2].0, frame_type::DATA);
        assert_eq!(subs[2].1, vec![0xBB; 2]);
    }

    #[test]
    fn decode_empty_batch() {
        // Just the batch type byte, no sub-frames.
        let batch = vec![frame_type::BATCH];
        let subs: Vec<_> = decode_batch(&batch).collect();
        assert!(subs.is_empty());
    }

    #[test]
    fn decode_completely_empty_slice() {
        let subs: Vec<_> = decode_batch(&[]).collect();
        assert!(subs.is_empty());
    }

    #[test]
    fn decode_trailing_header_fragment_ignored() {
        // A batch with a valid sub-frame followed by only 2 bytes (incomplete header).
        let mut batch = vec![frame_type::BATCH];
        // Valid sub-frame: DATA, 4 bytes
        batch.push(frame_type::DATA);
        batch.extend_from_slice(&4u16.to_be_bytes());
        batch.extend_from_slice(&[0xAA; 4]);
        // Trailing fragment: only 2 bytes (need 3 for a header)
        batch.extend_from_slice(&[0xFF, 0x00]);

        let subs: Vec<(u8, Vec<u8>)> = decode_batch(&batch)
            .map(|(t, p)| (t, p.to_vec()))
            .collect();

        assert_eq!(subs.len(), 1);
        assert_eq!(subs[0].0, frame_type::DATA);
        assert_eq!(subs[0].1, vec![0xAA; 4]);
    }
}

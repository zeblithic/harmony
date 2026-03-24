//! Scout and Data frame encoding/decoding for the Harmony L2 protocol.
//!
//! All frames share the layout:
//! `[6 dst_mac][6 src_mac][2 EtherType=0x88B5][1 frame_type][payload...]`

use crate::{error::RawLinkError, frame_type, ETH_HEADER_LEN, FRAME_OVERHEAD, HARMONY_ETHERTYPE};

/// Broadcast MAC address (FF:FF:FF:FF:FF:FF).
pub const BROADCAST_MAC: [u8; 6] = [0xFF; 6];

/// Validates the Ethernet header and frame type tag.
///
/// Checks that:
/// - The frame is long enough to contain a full header.
/// - Bytes 12–13 are the `HARMONY_ETHERTYPE`.
/// - Byte 14 matches `expected_type`.
pub fn validate_header(frame: &[u8], expected_type: u8) -> Result<(), RawLinkError> {
    if frame.len() < FRAME_OVERHEAD {
        return Err(RawLinkError::FrameError(format!(
            "frame too short: {} < {}",
            frame.len(),
            FRAME_OVERHEAD,
        )));
    }

    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    if ethertype != HARMONY_ETHERTYPE {
        return Err(RawLinkError::FrameError(format!(
            "wrong EtherType: 0x{ethertype:04X} (expected 0x{HARMONY_ETHERTYPE:04X})",
        )));
    }

    let frame_type = frame[ETH_HEADER_LEN];
    if frame_type != expected_type {
        return Err(RawLinkError::FrameError(format!(
            "wrong frame type: 0x{frame_type:02X} (expected 0x{expected_type:02X})",
        )));
    }

    Ok(())
}

/// Encodes a Scout frame.
///
/// Layout after FRAME_OVERHEAD: `[16 identity_hash]`
/// Destination is always the broadcast MAC.
pub fn encode_scout_frame(src_mac: [u8; 6], identity_hash: &[u8; 16]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(FRAME_OVERHEAD + 16);

    // dst_mac (broadcast), src_mac
    frame.extend_from_slice(&BROADCAST_MAC);
    frame.extend_from_slice(&src_mac);

    // EtherType
    frame.extend_from_slice(&HARMONY_ETHERTYPE.to_be_bytes());

    // Frame type
    frame.push(frame_type::SCOUT);

    // Payload: 16-byte identity hash
    frame.extend_from_slice(identity_hash);

    frame
}

/// Decodes a Scout frame.
///
/// Returns `(src_mac, identity_hash)`.
pub fn decode_scout_frame(frame: &[u8]) -> Result<([u8; 6], [u8; 16]), RawLinkError> {
    validate_header(frame, frame_type::SCOUT)?;

    // Minimum length: FRAME_OVERHEAD + 16 (identity_hash)
    let needed = FRAME_OVERHEAD + 16;
    if frame.len() < needed {
        return Err(RawLinkError::FrameError(format!(
            "scout frame too short: {} < {needed}",
            frame.len(),
        )));
    }

    let mut src_mac = [0u8; 6];
    src_mac.copy_from_slice(&frame[6..12]);

    let mut identity_hash = [0u8; 16];
    identity_hash.copy_from_slice(&frame[FRAME_OVERHEAD..FRAME_OVERHEAD + 16]);

    Ok((src_mac, identity_hash))
}

/// Encodes a Data frame.
///
/// Layout after FRAME_OVERHEAD: `[2 key_expr_len (big-endian)][key_expr bytes][payload...]`
///
/// Returns `FrameError` if `key_expr` exceeds 65535 bytes.
pub fn encode_data_frame(
    src_mac: [u8; 6],
    dst_mac: [u8; 6],
    key_expr: &str,
    payload: &[u8],
) -> Result<Vec<u8>, RawLinkError> {
    let key_bytes = key_expr.as_bytes();
    let key_len = key_bytes.len();
    if key_len > u16::MAX as usize {
        return Err(RawLinkError::FrameError(format!(
            "key_expr too long: {key_len} bytes (max {})",
            u16::MAX
        )));
    }

    let total = FRAME_OVERHEAD + 2 + key_len + payload.len();
    let mut frame = Vec::with_capacity(total);

    // dst_mac, src_mac
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);

    // EtherType
    frame.extend_from_slice(&HARMONY_ETHERTYPE.to_be_bytes());

    // Frame type
    frame.push(frame_type::DATA);

    // key_expr length as u16 big-endian, then key_expr bytes
    frame.extend_from_slice(&(key_len as u16).to_be_bytes());
    frame.extend_from_slice(key_bytes);

    // Payload
    frame.extend_from_slice(payload);

    Ok(frame)
}

/// Decodes a Data frame.
///
/// Returns `(src_mac, key_expr, payload)`.
pub fn decode_data_frame(frame: &[u8]) -> Result<([u8; 6], String, Vec<u8>), RawLinkError> {
    validate_header(frame, frame_type::DATA)?;

    // Need at least FRAME_OVERHEAD + 2 (key_expr_len field)
    let min_len = FRAME_OVERHEAD + 2;
    if frame.len() < min_len {
        return Err(RawLinkError::FrameError(format!(
            "data frame too short for key_expr length field: {} < {min_len}",
            frame.len(),
        )));
    }

    let mut src_mac = [0u8; 6];
    src_mac.copy_from_slice(&frame[6..12]);

    let key_len = u16::from_be_bytes([frame[FRAME_OVERHEAD], frame[FRAME_OVERHEAD + 1]]) as usize;

    let key_start = FRAME_OVERHEAD + 2;
    let key_end = key_start + key_len;

    if frame.len() < key_end {
        return Err(RawLinkError::FrameError(format!(
            "data frame too short for key_expr body: {} < {key_end}",
            frame.len(),
        )));
    }

    let key_expr = std::str::from_utf8(&frame[key_start..key_end])
        .map_err(|e| RawLinkError::FrameError(format!("invalid UTF-8 in key_expr: {e}")))?
        .to_string();

    let payload = frame[key_end..].to_vec();

    Ok((src_mac, key_expr, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SRC: [u8; 6] = [0x11, 0x22, 0x33, 0x44, 0x55, 0x66];
    const DST: [u8; 6] = [0xAA, 0xBB, 0xCC, 0xDD, 0xEE, 0xFF];
    const HASH: [u8; 16] = [
        0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F,
        0x10,
    ];

    #[test]
    fn scout_frame_round_trip() {
        let frame = encode_scout_frame(SRC, &HASH);
        let (got_src, got_hash) = decode_scout_frame(&frame).expect("decode should succeed");
        assert_eq!(got_src, SRC);
        assert_eq!(got_hash, HASH);
    }

    #[test]
    fn scout_frame_broadcast_dst() {
        let frame = encode_scout_frame(SRC, &HASH);
        // First 6 bytes must be all 0xFF (broadcast)
        assert_eq!(&frame[0..6], &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn data_frame_round_trip() {
        let key = "harmony/test/topic";
        let payload = b"hello, world!";
        let frame = encode_data_frame(SRC, DST, key, payload).expect("encode should succeed");
        let (got_src, got_key, got_payload) =
            decode_data_frame(&frame).expect("decode should succeed");
        assert_eq!(got_src, SRC);
        assert_eq!(got_key, key);
        assert_eq!(got_payload, payload);
    }

    #[test]
    fn data_frame_too_short_rejected() {
        // A frame shorter than FRAME_OVERHEAD must be rejected
        let short = vec![0u8; FRAME_OVERHEAD - 1];
        let err = decode_data_frame(&short).unwrap_err();
        assert!(
            matches!(err, RawLinkError::FrameError(_)),
            "expected FrameError, got {err:?}"
        );
    }

    #[test]
    fn wrong_ethertype_rejected() {
        let mut frame = encode_scout_frame(SRC, &HASH);
        // Corrupt EtherType bytes
        frame[12] = 0x08;
        frame[13] = 0x00; // IPv4
        let err = decode_scout_frame(&frame).unwrap_err();
        assert!(
            matches!(err, RawLinkError::FrameError(_)),
            "expected FrameError, got {err:?}"
        );
    }

    #[test]
    fn wrong_frame_type_rejected() {
        let mut frame = encode_scout_frame(SRC, &HASH);
        // Change frame type byte to DATA
        frame[ETH_HEADER_LEN] = frame_type::DATA;
        // Try to decode as Scout — should fail
        let err = decode_scout_frame(&frame).unwrap_err();
        assert!(
            matches!(err, RawLinkError::FrameError(_)),
            "expected FrameError, got {err:?}"
        );
    }
}

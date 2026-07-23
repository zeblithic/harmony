//! The single audited length-prefixed framing codec for iroh bi-streams
//! (ported verbatim from harmony-client `iroh_framing.rs`, ZEB-572).
//!
//! A frame is `[u32 length prefix][body]`. This is the one audited
//! implementation of the cap-before-alloc DoS guard the tunnel driver frames
//! its handshake messages with.
//!
//! Two layers:
//!  * [`encode_len_prefix`] / [`decode_len_prefix`] — the pure, sync security
//!    boundary (the cap-before-alloc guard).
//!  * [`write_len_prefixed`] / [`read_len_prefixed`] — async convenience
//!    wrappers used by the tunnel driver's handshake reads/writes.
//!
//! Endianness is a parameter so every shipped wire format is preserved. The
//! tunnel driver uses [`Endian::Be`] (network order) exclusively — the 4-byte
//! big-endian length prefix is LOAD-BEARING and preserved byte-for-byte.

use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Byte order of the 4-byte `u32` length prefix. A parameter (not a constant)
/// because shipped protocols disagree and their on-the-wire bytes must be
/// preserved.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endian {
    /// Network order. The convention for all NEW framing (and the tunnel).
    Be,
    /// Little-endian. Backward-compat only. The tunnel driver never uses it;
    /// carried for parity with the audited source codec (and exercised by the
    /// codec's own tests), so `#[allow(dead_code)]` for the lib target.
    #[allow(dead_code)]
    Le,
}

impl Endian {
    fn encode(self, len: u32) -> [u8; 4] {
        match self {
            Endian::Be => len.to_be_bytes(),
            Endian::Le => len.to_le_bytes(),
        }
    }

    fn decode(self, buf: [u8; 4]) -> u32 {
        match self {
            Endian::Be => u32::from_be_bytes(buf),
            Endian::Le => u32::from_le_bytes(buf),
        }
    }
}

/// A length prefix that is zero (when empty is disallowed) or exceeds the cap.
/// Raised BEFORE any body byte is read or allocated, so an attacker-supplied
/// prefix can never drive an allocation past `max`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[error("frame length {len} out of bounds (max {max})")]
pub struct FrameLenError {
    pub len: usize,
    pub max: usize,
}

/// Encode a body length into its 4-byte prefix. Rejects an empty body (unless
/// `allow_empty`) or one exceeding `max`, before producing any bytes.
pub fn encode_len_prefix(
    body_len: usize,
    max: usize,
    endian: Endian,
    allow_empty: bool,
) -> Result<[u8; 4], FrameLenError> {
    if (!allow_empty && body_len == 0) || body_len > max {
        return Err(FrameLenError { len: body_len, max });
    }
    // The wire prefix is a u32: reject (never silently truncate) a body that
    // can't be represented, so the audited boundary stays safe even when a
    // caller passes a `max` at/above u32::MAX (the no-cap write sites).
    let len = u32::try_from(body_len).map_err(|_| FrameLenError { len: body_len, max })?;
    Ok(endian.encode(len))
}

/// Decode a 4-byte length prefix into the body length to read next. Rejects a
/// zero length (unless `allow_empty`) or one exceeding `max`, before the body
/// is read or allocated.
pub fn decode_len_prefix(
    buf: [u8; 4],
    max: usize,
    endian: Endian,
    allow_empty: bool,
) -> Result<usize, FrameLenError> {
    let len = endian.decode(buf) as usize;
    if (!allow_empty && len == 0) || len > max {
        return Err(FrameLenError { len, max });
    }
    Ok(len)
}

/// An error from the async framing wrappers: either the body length was out of
/// bounds (the cap guard fired, before any I/O on the body) or the underlying
/// stream returned an I/O error.
#[derive(Debug, thiserror::Error)]
pub enum FramingError {
    #[error(transparent)]
    OutOfBounds(#[from] FrameLenError),
    #[error("frame I/O: {0}")]
    Io(#[from] std::io::Error),
}

/// Write a `[u32 prefix][body]` frame. The bound check ([`encode_len_prefix`])
/// runs first, so an out-of-bounds body is rejected before anything is written.
/// Prefix then body are written as two `write_all`s (each all-or-error).
pub async fn write_len_prefixed<W: AsyncWrite + Unpin>(
    w: &mut W,
    body: &[u8],
    max: usize,
    endian: Endian,
    allow_empty: bool,
) -> Result<(), FramingError> {
    let prefix = encode_len_prefix(body.len(), max, endian, allow_empty)?;
    w.write_all(&prefix).await?;
    w.write_all(body).await?;
    Ok(())
}

/// Read a `[u32 prefix][body]` frame. The prefix is decoded and bound-checked
/// ([`decode_len_prefix`]) before the body is allocated or read, so an
/// attacker-supplied prefix never drives an allocation past `max`.
pub async fn read_len_prefixed<R: AsyncRead + Unpin>(
    r: &mut R,
    max: usize,
    endian: Endian,
    allow_empty: bool,
) -> Result<Vec<u8>, FramingError> {
    let mut len_buf = [0u8; 4];
    r.read_exact(&mut len_buf).await?;
    let len = decode_len_prefix(len_buf, max, endian, allow_empty)?;
    let mut body = vec![0u8; len];
    r.read_exact(&mut body).await?;
    Ok(body)
}

#[cfg(test)]
mod core_tests {
    use super::*;

    const MAX: usize = 256 * 1024;

    #[test]
    fn encode_rejects_empty_when_disallowed() {
        assert_eq!(
            encode_len_prefix(0, MAX, Endian::Le, false),
            Err(FrameLenError { len: 0, max: MAX })
        );
    }

    #[test]
    fn encode_accepts_empty_when_allowed() {
        assert_eq!(
            encode_len_prefix(0, MAX, Endian::Be, true),
            Ok([0, 0, 0, 0])
        );
    }

    #[test]
    fn encode_rejects_oversize() {
        assert_eq!(
            encode_len_prefix(MAX + 1, MAX, Endian::Le, false),
            Err(FrameLenError {
                len: MAX + 1,
                max: MAX
            })
        );
    }

    #[test]
    fn encode_accepts_at_cap() {
        assert!(encode_len_prefix(MAX, MAX, Endian::Le, false).is_ok());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn encode_rejects_body_exceeding_u32_even_with_huge_max() {
        let too_big = u32::MAX as usize + 1;
        assert_eq!(
            encode_len_prefix(too_big, usize::MAX, Endian::Le, false),
            Err(FrameLenError {
                len: too_big,
                max: usize::MAX
            })
        );
    }

    #[test]
    fn decode_rejects_zero_when_disallowed() {
        assert_eq!(
            decode_len_prefix([0, 0, 0, 0], MAX, Endian::Le, false),
            Err(FrameLenError { len: 0, max: MAX })
        );
    }

    #[test]
    fn decode_accepts_zero_when_allowed() {
        assert_eq!(
            decode_len_prefix([0, 0, 0, 0], MAX, Endian::Be, true),
            Ok(0)
        );
    }

    #[test]
    fn be_and_le_differ_and_each_round_trips() {
        let len = 0x0102_0304usize;
        let be = encode_len_prefix(len, usize::MAX, Endian::Be, false).unwrap();
        let le = encode_len_prefix(len, usize::MAX, Endian::Le, false).unwrap();
        assert_eq!(be, [0x01, 0x02, 0x03, 0x04]);
        assert_eq!(le, [0x04, 0x03, 0x02, 0x01]);
        assert_ne!(be, le);
        assert_eq!(
            decode_len_prefix(be, usize::MAX, Endian::Be, false).unwrap(),
            len
        );
    }
}

#[cfg(test)]
mod wrapper_tests {
    use super::*;

    const MAX: usize = 1024;

    /// Framing round-trip (big-endian, the tunnel's wire order): a non-empty
    /// body writes `[4-byte BE len][body]` and reads back byte-identical.
    #[tokio::test]
    async fn round_trip_be_nonempty() {
        let body = b"hello tunnel frame".to_vec();
        let mut buf = Vec::new();
        write_len_prefixed(&mut buf, &body, MAX, Endian::Be, false)
            .await
            .unwrap();
        assert_eq!(&buf[..4], &(body.len() as u32).to_be_bytes());
        let mut reader = buf.as_slice();
        let got = read_len_prefixed(&mut reader, MAX, Endian::Be, false)
            .await
            .unwrap();
        assert_eq!(got, body);
    }

    /// Framing round-trip for a zero-length frame with `allow_empty` (the
    /// tunnel driver's write side allows empty).
    #[tokio::test]
    async fn round_trip_be_allow_empty_zero_length() {
        let mut buf = Vec::new();
        write_len_prefixed(&mut buf, &[], MAX, Endian::Be, true)
            .await
            .unwrap();
        assert_eq!(buf, vec![0, 0, 0, 0]);
        let mut reader = buf.as_slice();
        let got = read_len_prefixed(&mut reader, MAX, Endian::Be, true)
            .await
            .unwrap();
        assert!(got.is_empty());
    }

    #[tokio::test]
    async fn read_rejects_oversize_before_body() {
        let prefix_only = ((MAX + 1) as u32).to_be_bytes().to_vec();
        let mut reader = prefix_only.as_slice();
        let err = read_len_prefixed(&mut reader, MAX, Endian::Be, false)
            .await
            .expect_err("oversize prefix must be rejected");
        assert!(
            matches!(err, FramingError::OutOfBounds(FrameLenError { len, max }) if len == MAX + 1 && max == MAX),
            "expected OutOfBounds rejected before body read, got {err:?}"
        );
    }

    #[tokio::test]
    async fn write_rejects_oversize_writes_nothing() {
        let body = vec![0u8; MAX + 1];
        let mut buf = Vec::new();
        let err = write_len_prefixed(&mut buf, &body, MAX, Endian::Be, false)
            .await
            .expect_err("oversize body must be rejected");
        assert!(matches!(err, FramingError::OutOfBounds(_)));
        assert!(buf.is_empty(), "nothing should be written on rejection");
    }
}

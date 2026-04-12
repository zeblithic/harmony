//! SMTP line codec for tokio I/O.
//!
//! Implements [`tokio_util::codec::Decoder`] with two modes:
//! - **Command mode**: extracts CRLF-delimited lines (SMTP commands).
//! - **Data mode**: collects message body until the terminating `.\r\n` on a
//!   line by itself, performing dot-unstuffing per RFC 5321 §4.5.2.
//!
//! The session driver toggles between modes when the sans-I/O state machine
//! enters or leaves the `DataReceiving` state.

use bytes::{Buf, BytesMut};
use std::io;
use tokio_util::codec::Decoder;

/// Maximum length of a single SMTP command line (including CRLF).
/// RFC 5321 §4.5.3.1.4: 512 octets including CRLF.
const MAX_LINE_LEN: usize = 512;

/// Frames produced by [`SmtpCodec`].
#[derive(Debug, PartialEq, Eq)]
pub enum SmtpFrame {
    /// A single SMTP command line (CRLF stripped).
    Line(String),
    /// Complete message body after DATA, dot-unstuffed.
    Data(Vec<u8>),
}

/// Codec operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Command,
    Data,
}

/// SMTP protocol codec with command and data modes.
///
/// In command mode, decodes CRLF-terminated lines (up to 512 bytes per RFC 5321).
/// In data mode, accumulates bytes until a lone `.\r\n` terminates the body,
/// then yields the dot-unstuffed content.
#[derive(Debug)]
pub struct SmtpCodec {
    mode: Mode,
    max_message_size: usize,
    data_buf: Vec<u8>,
}

impl SmtpCodec {
    /// Create a new codec in command mode.
    pub fn new(max_message_size: usize) -> Self {
        Self {
            mode: Mode::Command,
            max_message_size,
            data_buf: Vec::new(),
        }
    }

    /// Switch to data mode (call when state machine enters `DataReceiving`).
    pub fn enter_data_mode(&mut self) {
        self.mode = Mode::Data;
        self.data_buf.clear();
    }

    /// Switch back to command mode (call after `SmtpFrame::Data` is yielded).
    pub fn enter_command_mode(&mut self) {
        self.mode = Mode::Command;
        self.data_buf.clear();
    }

    /// Decode a single CRLF-terminated command line.
    fn decode_line(&self, src: &mut BytesMut) -> Result<Option<SmtpFrame>, io::Error> {
        // Search for \r\n
        let crlf_pos = src.windows(2).position(|w| w == b"\r\n");

        match crlf_pos {
            Some(pos) => {
                // Enforce RFC 5321 §4.5.3.1.4: 512 octets including CRLF
                if pos + 2 > MAX_LINE_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("SMTP command line exceeds {} bytes", MAX_LINE_LEN),
                    ));
                }

                // Extract line bytes (not including CRLF)
                let line_bytes = src.split_to(pos);
                // Consume the CRLF
                src.advance(2);

                let line = String::from_utf8(line_bytes.to_vec()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in SMTP command")
                })?;

                Ok(Some(SmtpFrame::Line(line)))
            }
            None => {
                // No complete line yet — check if we're over the limit
                if src.len() > MAX_LINE_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("SMTP command line exceeds {} bytes", MAX_LINE_LEN),
                    ));
                }
                Ok(None)
            }
        }
    }

    /// Decode data mode: accumulate lines, dot-unstuff, detect terminator.
    fn decode_data(&mut self, src: &mut BytesMut) -> Result<Option<SmtpFrame>, io::Error> {
        loop {
            // Find next CRLF
            let crlf_pos = src.windows(2).position(|w| w == b"\r\n");

            match crlf_pos {
                Some(pos) => {
                    let line = &src[..pos];

                    // Check for terminator: lone dot
                    if line == b"." {
                        src.advance(pos + 2); // consume ".\r\n"
                        let body = std::mem::take(&mut self.data_buf);
                        return Ok(Some(SmtpFrame::Data(body)));
                    }

                    // Dot-unstuffing per RFC 5321 §4.5.2: "If the first
                    // character is a period and there are other characters
                    // on the line, the first character is deleted."
                    let content = if line.starts_with(b".") {
                        &line[1..]
                    } else {
                        line
                    };

                    // Check size limit before appending
                    let new_size = self.data_buf.len() + content.len() + 2; // +2 for CRLF
                    if new_size > self.max_message_size {
                        self.data_buf.clear();
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "message exceeds maximum size of {} bytes",
                                self.max_message_size
                            ),
                        ));
                    }

                    // Append line content + CRLF to body buffer
                    self.data_buf.extend_from_slice(content);
                    self.data_buf.extend_from_slice(b"\r\n");

                    // Consume from source
                    src.advance(pos + 2);
                }
                None => {
                    // No complete line yet — wait for more data.
                    // Check accumulated + pending doesn't exceed limit.
                    if self.data_buf.len() + src.len() > self.max_message_size {
                        self.data_buf.clear();
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "message exceeds maximum size of {} bytes",
                                self.max_message_size
                            ),
                        ));
                    }
                    return Ok(None);
                }
            }
        }
    }
}

impl Decoder for SmtpCodec {
    type Item = SmtpFrame;
    type Error = io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.mode {
            Mode::Command => self.decode_line(src),
            Mode::Data => self.decode_data(src),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> SmtpCodec {
        SmtpCodec::new(10 * 1024 * 1024) // 10 MB
    }

    fn small_codec() -> SmtpCodec {
        SmtpCodec::new(100) // 100 bytes for size limit tests
    }

    // ── Command mode tests ──────────────────────────────────────────

    #[test]
    fn decode_simple_command() {
        let mut c = codec();
        let mut buf = BytesMut::from("EHLO example.com\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Line("EHLO example.com".to_string())));
        assert!(buf.is_empty());
    }

    #[test]
    fn decode_partial_command_then_complete() {
        let mut c = codec();
        let mut buf = BytesMut::from("EHLO exam");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        buf.extend_from_slice(b"ple.com\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Line("EHLO example.com".to_string())));
    }

    #[test]
    fn decode_multiple_commands() {
        let mut c = codec();
        let mut buf = BytesMut::from("EHLO x.com\r\nMAIL FROM:<a@b.com>\r\n");

        let f1 = c.decode(&mut buf).unwrap();
        assert_eq!(f1, Some(SmtpFrame::Line("EHLO x.com".to_string())));

        let f2 = c.decode(&mut buf).unwrap();
        assert_eq!(
            f2,
            Some(SmtpFrame::Line("MAIL FROM:<a@b.com>".to_string()))
        );

        assert!(buf.is_empty());
    }

    #[test]
    fn decode_empty_line() {
        let mut c = codec();
        let mut buf = BytesMut::from("\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Line(String::new())));
    }

    #[test]
    fn reject_oversized_command_line() {
        let mut c = codec();
        let long_line = "X".repeat(MAX_LINE_LEN + 1);
        let mut buf = BytesMut::from(long_line.as_bytes());
        let err = c.decode(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn accept_max_length_command_line() {
        let mut c = codec();
        // 510 chars + \r\n = 512 bytes = MAX_LINE_LEN
        let line = "X".repeat(MAX_LINE_LEN - 2);
        let mut buf = BytesMut::from(format!("{}\r\n", line).as_bytes());
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Line(line)));
    }

    // ── Data mode tests ─────────────────────────────────────────────

    #[test]
    fn decode_simple_data() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from("Hello world\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(b"Hello world\r\n".to_vec())));
        assert!(buf.is_empty());
    }

    #[test]
    fn decode_multiline_data() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from("Line 1\r\nLine 2\r\nLine 3\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(SmtpFrame::Data(b"Line 1\r\nLine 2\r\nLine 3\r\n".to_vec()))
        );
    }

    #[test]
    fn decode_data_dot_unstuffing() {
        let mut c = codec();
        c.enter_data_mode();
        // "..Hello" should become ".Hello" after unstuffing
        let mut buf = BytesMut::from("..Hello\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(b".Hello\r\n".to_vec())));
    }

    #[test]
    fn decode_data_multiple_dot_unstuffing() {
        let mut c = codec();
        c.enter_data_mode();
        // "...test" -> "..test" (only first dot removed)
        let mut buf = BytesMut::from("...test\r\n..single\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(SmtpFrame::Data(b"..test\r\n.single\r\n".to_vec()))
        );
    }

    #[test]
    fn decode_data_empty_body() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from(".\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(Vec::new())));
    }

    #[test]
    fn decode_data_partial_then_complete() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from("Hello ");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        buf.extend_from_slice(b"world\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(b"Hello world\r\n".to_vec())));
    }

    #[test]
    fn decode_data_size_limit_exceeded() {
        let mut c = small_codec(); // 100 byte limit
        c.enter_data_mode();
        let long_line = "X".repeat(110);
        let mut buf = BytesMut::from(format!("{}\r\n.\r\n", long_line).as_bytes());
        let err = c.decode(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("exceeds maximum size"));
    }

    #[test]
    fn decode_data_preserves_blank_lines() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from("Line 1\r\n\r\nLine 3\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(SmtpFrame::Data(b"Line 1\r\n\r\nLine 3\r\n".to_vec()))
        );
    }

    // ── Mode switching tests ────────────────────────────────────────

    #[test]
    fn mode_switch_command_to_data_and_back() {
        let mut c = codec();

        // Command mode: read EHLO
        let mut buf = BytesMut::from("EHLO x.com\r\n");
        let f1 = c.decode(&mut buf).unwrap();
        assert_eq!(f1, Some(SmtpFrame::Line("EHLO x.com".to_string())));

        // Switch to data mode
        c.enter_data_mode();
        buf.extend_from_slice(b"Body text\r\n.\r\n");
        let f2 = c.decode(&mut buf).unwrap();
        assert_eq!(f2, Some(SmtpFrame::Data(b"Body text\r\n".to_vec())));

        // Switch back to command mode
        c.enter_command_mode();
        buf.extend_from_slice(b"QUIT\r\n");
        let f3 = c.decode(&mut buf).unwrap();
        assert_eq!(f3, Some(SmtpFrame::Line("QUIT".to_string())));
    }

    #[test]
    fn data_after_terminator_stays_in_buffer() {
        let mut c = codec();
        c.enter_data_mode();
        let mut buf = BytesMut::from("body\r\n.\r\nQUIT\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(b"body\r\n".to_vec())));
        // QUIT\r\n should remain in buffer for next command-mode decode
        assert_eq!(&buf[..], b"QUIT\r\n");
    }

    #[test]
    fn data_dot_on_line_alone_is_terminator() {
        let mut c = codec();
        c.enter_data_mode();
        // A line with just "." followed by CRLF is the terminator.
        // ".text" is dot-unstuffed to "text" per RFC 5321 §4.5.2
        // (the sender would have transmitted "..text" to represent ".text")
        let mut buf = BytesMut::from(".text\r\n.\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(SmtpFrame::Data(b"text\r\n".to_vec())));
    }

    #[test]
    fn reject_oversized_command_with_crlf_in_buffer() {
        let mut c = codec();
        let long_line = "X".repeat(MAX_LINE_LEN); // 512 Xs + \r\n = 514 bytes
        let mut buf = BytesMut::from(format!("{}\r\n", long_line).as_bytes());
        let err = c.decode(&mut buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }
}

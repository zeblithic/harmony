//! IMAP protocol codec for tokio I/O.
//!
//! Implements [`tokio_util::codec::Decoder`] with three modes:
//! - **Command mode**: extracts CRLF-delimited lines, detects literal requests.
//! - **Literal mode**: consumes N raw bytes (after `{N}\r\n` or `{N+}\r\n`),
//!   appends to the partial command buffer, then returns to command mode.
//! - **Idle mode**: only accepts `DONE\r\n`.
//!
//! The session driver toggles idle mode when IDLE is entered/exited.

use bytes::{Buf, BytesMut};
use std::io;
use tokio_util::codec::Decoder;

/// Maximum length of a single IMAP command line (before literal expansion).
/// IMAP doesn't have a strict line limit like SMTP's 512, but we enforce
/// a generous limit to prevent memory exhaustion.
const MAX_LINE_LEN: usize = 8192;

/// Maximum literal size we'll accept (10 MB).
const MAX_LITERAL_SIZE: usize = 10 * 1024 * 1024;

/// Frames produced by [`ImapCodec`].
#[derive(Debug, PartialEq, Eq)]
pub enum ImapFrame {
    /// A complete command line with all literals resolved.
    CommandLine(String),
    /// The codec detected a synchronizing literal `{N}\r\n` — the I/O driver
    /// must send `+ Ready\r\n` before the client will send the literal data.
    /// After sending the continuation, call `acknowledge_continuation()`.
    NeedsContinuation { size: usize },
    /// Client sent `DONE\r\n` during IDLE mode.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Command,
    /// Awaiting N bytes of literal data, with synchronizing flag.
    Literal {
        remaining: usize,
    },
    /// Waiting for I/O driver to send continuation response.
    /// Transitions to Literal once acknowledged.
    AwaitingContinuation {
        size: usize,
    },
    Idle,
}

/// IMAP protocol codec.
#[derive(Debug)]
pub struct ImapCodec {
    mode: Mode,
    /// Partial command being built across multiple lines + literals.
    command_buf: String,
}

impl Default for ImapCodec {
    fn default() -> Self {
        Self::new()
    }
}

impl ImapCodec {
    /// Create a new codec in command mode.
    pub fn new() -> Self {
        Self {
            mode: Mode::Command,
            command_buf: String::new(),
        }
    }

    /// Switch to idle mode (after IDLE command processed).
    pub fn enter_idle_mode(&mut self) {
        self.mode = Mode::Idle;
        self.command_buf.clear();
    }

    /// Switch back to command mode (after DONE received or IDLE cancelled).
    pub fn exit_idle_mode(&mut self) {
        self.mode = Mode::Command;
        self.command_buf.clear();
    }

    /// Acknowledge that the continuation response (`+ Ready\r\n`) has been sent.
    /// Transitions from AwaitingContinuation to Literal mode.
    pub fn acknowledge_continuation(&mut self) {
        if let Mode::AwaitingContinuation { size } = self.mode {
            self.mode = Mode::Literal { remaining: size };
        }
    }

    /// Decode in command mode: read CRLF-terminated lines, detect literals.
    fn decode_command(&mut self, src: &mut BytesMut) -> Result<Option<ImapFrame>, io::Error> {
        let crlf_pos = src.windows(2).position(|w| w == b"\r\n");

        match crlf_pos {
            Some(pos) => {
                if pos > MAX_LINE_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("IMAP command line exceeds {MAX_LINE_LEN} bytes"),
                    ));
                }

                let line_bytes = src.split_to(pos);
                src.advance(2); // consume CRLF

                let line = String::from_utf8(line_bytes.to_vec()).map_err(|_| {
                    io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in IMAP command")
                })?;

                // Check for literal request at end of line: {N} or {N+}
                if let Some(literal_info) = detect_literal(&line) {
                    if literal_info.size > MAX_LITERAL_SIZE {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "literal size {} exceeds maximum {MAX_LITERAL_SIZE}",
                                literal_info.size
                            ),
                        ));
                    }

                    // Strip the literal marker from the line and append to command buffer
                    let prefix = &line[..literal_info.marker_start];
                    self.command_buf.push_str(prefix);

                    if literal_info.non_sync {
                        // Non-synchronizing literal: go straight to literal mode
                        self.mode = Mode::Literal {
                            remaining: literal_info.size,
                        };
                        // Try to decode more from the buffer
                        return self.decode_literal(src);
                    } else {
                        // Synchronizing literal: need continuation response first
                        self.mode = Mode::AwaitingContinuation {
                            size: literal_info.size,
                        };
                        return Ok(Some(ImapFrame::NeedsContinuation {
                            size: literal_info.size,
                        }));
                    }
                }

                // No literal — complete command line
                if self.command_buf.is_empty() {
                    Ok(Some(ImapFrame::CommandLine(line)))
                } else {
                    self.command_buf.push_str(&line);
                    let complete = std::mem::take(&mut self.command_buf);
                    Ok(Some(ImapFrame::CommandLine(complete)))
                }
            }
            None => {
                if src.len() > MAX_LINE_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("IMAP command line exceeds {MAX_LINE_LEN} bytes"),
                    ));
                }
                Ok(None) // need more data
            }
        }
    }

    /// Decode in literal mode: consume exactly N bytes.
    fn decode_literal(&mut self, src: &mut BytesMut) -> Result<Option<ImapFrame>, io::Error> {
        let remaining = match self.mode {
            Mode::Literal { remaining } => remaining,
            _ => return Ok(None),
        };

        if src.len() < remaining {
            return Ok(None); // need more data
        }

        // Consume the literal bytes
        let literal_bytes = src.split_to(remaining);
        let literal_str = String::from_utf8(literal_bytes.to_vec()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8 in IMAP literal")
        })?;

        self.command_buf.push_str(&literal_str);
        self.mode = Mode::Command;

        // Continue decoding — there may be more of the command line after the literal
        self.decode_command(src)
    }

    /// Decode in idle mode: only accept `DONE\r\n`.
    fn decode_idle(&mut self, src: &mut BytesMut) -> Result<Option<ImapFrame>, io::Error> {
        let crlf_pos = src.windows(2).position(|w| w == b"\r\n");

        match crlf_pos {
            Some(pos) => {
                let line = &src[..pos];
                let line_str = std::str::from_utf8(line)
                    .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "invalid UTF-8"))?;

                if line_str.trim().eq_ignore_ascii_case("DONE") {
                    src.advance(pos + 2);
                    Ok(Some(ImapFrame::Done))
                } else {
                    // In IDLE mode, only DONE is valid — ignore other input
                    src.advance(pos + 2);
                    Ok(None)
                }
            }
            None => {
                if src.len() > MAX_LINE_LEN {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        "oversized input during IDLE",
                    ));
                }
                Ok(None)
            }
        }
    }
}

impl Decoder for ImapCodec {
    type Item = ImapFrame;
    type Error = io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> Result<Option<Self::Item>, Self::Error> {
        match self.mode {
            Mode::Command => self.decode_command(src),
            Mode::Literal { .. } => self.decode_literal(src),
            Mode::AwaitingContinuation { .. } => {
                // Waiting for driver to call acknowledge_continuation()
                Ok(None)
            }
            Mode::Idle => self.decode_idle(src),
        }
    }
}

// ── Literal detection ───────────────────────────────────────────────

struct LiteralInfo {
    size: usize,
    non_sync: bool,
    marker_start: usize,
}

/// Detect a literal marker at the end of a line: `{N}` or `{N+}`.
fn detect_literal(line: &str) -> Option<LiteralInfo> {
    let trimmed = line.trim_end();
    if !trimmed.ends_with('}') {
        return None;
    }

    let brace_open = trimmed.rfind('{')?;
    let inner = &trimmed[brace_open + 1..trimmed.len() - 1];

    let (num_str, non_sync) = if let Some(n) = inner.strip_suffix('+') {
        (n, true)
    } else {
        (inner, false)
    };

    let size = num_str.parse::<usize>().ok()?;
    Some(LiteralInfo {
        size,
        non_sync,
        marker_start: brace_open,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn codec() -> ImapCodec {
        ImapCodec::new()
    }

    // ── Command mode tests ──────────────────────────────────────────

    #[test]
    fn decode_simple_command() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 CAPABILITY\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine("A001 CAPABILITY".to_string()))
        );
        assert!(buf.is_empty());
    }

    #[test]
    fn decode_partial_command_then_complete() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 CAP");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        buf.extend_from_slice(b"ABILITY\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine("A001 CAPABILITY".to_string()))
        );
    }

    #[test]
    fn decode_multiple_commands() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 NOOP\r\nA002 LOGOUT\r\n");

        let f1 = c.decode(&mut buf).unwrap();
        assert_eq!(f1, Some(ImapFrame::CommandLine("A001 NOOP".to_string())));

        let f2 = c.decode(&mut buf).unwrap();
        assert_eq!(f2, Some(ImapFrame::CommandLine("A002 LOGOUT".to_string())));

        assert!(buf.is_empty());
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
    fn accept_max_length_command() {
        let mut c = codec();
        let line = "X".repeat(MAX_LINE_LEN - 2); // fits with \r\n
        let mut buf = BytesMut::from(format!("{line}\r\n").as_bytes());
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::CommandLine(line)));
    }

    // ── Literal tests (synchronizing) ───────────────────────────────

    #[test]
    fn decode_synchronizing_literal() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 LOGIN {5}\r\n");

        // Should yield NeedsContinuation
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::NeedsContinuation { size: 5 }));
        assert!(buf.is_empty());

        // Driver sends "+ Ready\r\n", then acknowledges
        c.acknowledge_continuation();

        // Client sends literal data + rest of command
        buf.extend_from_slice(b"alice password\r\n");

        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine(
                "A001 LOGIN alice password".to_string()
            ))
        );
    }

    #[test]
    fn decode_synchronizing_literal_partial_data() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 LOGIN {5}\r\n");

        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::NeedsContinuation { size: 5 }));

        c.acknowledge_continuation();

        // Only 3 of 5 bytes arrived
        buf.extend_from_slice(b"ali");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        // Rest arrives
        buf.extend_from_slice(b"ce pass\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine("A001 LOGIN alice pass".to_string()))
        );
    }

    // ── Literal tests (non-synchronizing) ───────────────────────────

    #[test]
    fn decode_non_sync_literal() {
        let mut c = codec();
        // {5+} means non-synchronizing — no continuation needed
        let mut buf = BytesMut::from("A001 LOGIN {5+}\r\nalice password\r\n");

        let frame = c.decode(&mut buf).unwrap();
        // Should directly yield the complete command (no NeedsContinuation)
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine(
                "A001 LOGIN alice password".to_string()
            ))
        );
    }

    #[test]
    fn decode_non_sync_literal_partial() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 LOGIN {5+}\r\nali");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        buf.extend_from_slice(b"ce password\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine(
                "A001 LOGIN alice password".to_string()
            ))
        );
    }

    // ── Literal detection ───────────────────────────────────────────

    #[test]
    fn detect_literal_sync() {
        let info = detect_literal("A001 LOGIN {5}").unwrap();
        assert_eq!(info.size, 5);
        assert!(!info.non_sync);
    }

    #[test]
    fn detect_literal_non_sync() {
        let info = detect_literal("A001 LOGIN {42+}").unwrap();
        assert_eq!(info.size, 42);
        assert!(info.non_sync);
    }

    #[test]
    fn detect_literal_none() {
        assert!(detect_literal("A001 CAPABILITY").is_none());
        assert!(detect_literal("A001 FETCH 1 FLAGS").is_none());
    }

    #[test]
    fn detect_literal_zero_size() {
        let info = detect_literal("A001 APPEND INBOX {0}").unwrap();
        assert_eq!(info.size, 0);
    }

    #[test]
    fn reject_oversized_literal() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 LOGIN {999999999}\r\n");
        let err = c.decode(&mut buf).unwrap_err();
        assert!(err.to_string().contains("exceeds maximum"));
    }

    // ── Idle mode tests ─────────────────────────────────────────────

    #[test]
    fn decode_idle_done() {
        let mut c = codec();
        c.enter_idle_mode();

        let mut buf = BytesMut::from("DONE\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::Done));
    }

    #[test]
    fn decode_idle_done_case_insensitive() {
        let mut c = codec();
        c.enter_idle_mode();

        let mut buf = BytesMut::from("done\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::Done));
    }

    #[test]
    fn decode_idle_ignores_non_done() {
        let mut c = codec();
        c.enter_idle_mode();

        let mut buf = BytesMut::from("NOOP\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, None); // ignored
        assert!(buf.is_empty()); // consumed
    }

    #[test]
    fn decode_idle_partial_then_done() {
        let mut c = codec();
        c.enter_idle_mode();

        let mut buf = BytesMut::from("DON");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        buf.extend_from_slice(b"E\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::Done));
    }

    // ── Mode switching tests ────────────────────────────────────────

    #[test]
    fn mode_switch_command_to_idle_and_back() {
        let mut c = codec();

        // Command mode
        let mut buf = BytesMut::from("A001 IDLE\r\n");
        let f1 = c.decode(&mut buf).unwrap();
        assert_eq!(f1, Some(ImapFrame::CommandLine("A001 IDLE".to_string())));

        // Switch to idle
        c.enter_idle_mode();
        buf.extend_from_slice(b"DONE\r\n");
        let f2 = c.decode(&mut buf).unwrap();
        assert_eq!(f2, Some(ImapFrame::Done));

        // Back to command
        c.exit_idle_mode();
        buf.extend_from_slice(b"A002 NOOP\r\n");
        let f3 = c.decode(&mut buf).unwrap();
        assert_eq!(f3, Some(ImapFrame::CommandLine("A002 NOOP".to_string())));
    }

    #[test]
    fn awaiting_continuation_blocks_decode() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 LOGIN {5}\r\n");

        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(frame, Some(ImapFrame::NeedsContinuation { size: 5 }));

        // Before acknowledging, decode returns None
        buf.extend_from_slice(b"alice");
        assert_eq!(c.decode(&mut buf).unwrap(), None);

        // After acknowledging, literal data is consumed
        c.acknowledge_continuation();
        buf.extend_from_slice(b" pass\r\n");
        let frame = c.decode(&mut buf).unwrap();
        assert_eq!(
            frame,
            Some(ImapFrame::CommandLine("A001 LOGIN alice pass".to_string()))
        );
    }

    #[test]
    fn data_after_command_stays_in_buffer() {
        let mut c = codec();
        let mut buf = BytesMut::from("A001 NOOP\r\nA002 LOGOUT\r\n");

        let f1 = c.decode(&mut buf).unwrap();
        assert_eq!(f1, Some(ImapFrame::CommandLine("A001 NOOP".to_string())));
        assert_eq!(&buf[..], b"A002 LOGOUT\r\n");
    }
}

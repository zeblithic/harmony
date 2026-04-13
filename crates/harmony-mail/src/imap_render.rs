//! IMAP response rendering.
//!
//! Converts HarmonyMessages to IMAP FETCH response formats: ENVELOPE,
//! BODYSTRUCTURE, BODY[], RFC822, and FLAGS. RFC 5322 is rendered on demand
//! using `mail-builder` — the gateway never stores RFC 5322.

use crate::imap_parse::FetchAttribute;
use crate::message::{HarmonyMessage, RecipientType, ADDRESS_HASH_LEN};

// ── ENVELOPE ────────────────────────────────────────────────────────

/// Build an IMAP ENVELOPE response string from a HarmonyMessage.
///
/// Format per RFC 9051 §7.5.2:
/// `(date subject from sender reply-to to cc bcc in-reply-to message-id)`
/// Each address is `(personal NIL mailbox host)`.
pub fn build_envelope(msg: &HarmonyMessage, domain: &str) -> String {
    let date = format_imap_date(msg.timestamp);
    let subject = quote_nstring(&msg.subject);
    let sender_addr = format_address_hash(&msg.sender_address, domain);
    let from = format_address_list(std::slice::from_ref(&sender_addr));
    let sender = from.clone(); // sender = from for our purposes
    let reply_to = from.clone(); // reply-to defaults to from

    let mut to_addrs = Vec::new();
    let mut cc_addrs = Vec::new();
    let mut bcc_addrs = Vec::new();
    for recip in &msg.recipients {
        let addr = format_address_hash(&recip.address_hash, domain);
        match recip.recipient_type {
            RecipientType::To => to_addrs.push(addr),
            RecipientType::Cc => cc_addrs.push(addr),
            RecipientType::Bcc => bcc_addrs.push(addr),
        }
    }

    let to = format_address_list(&to_addrs);
    let cc = format_address_list(&cc_addrs);
    let bcc = format_address_list(&bcc_addrs);

    let mid_hex = hex::encode(msg.message_id);
    let message_id = format!("\"<{mid_hex}@{domain}>\"");

    let in_reply_to = match msg.in_reply_to {
        Some(ref id) => {
            let rid_hex = hex::encode(id);
            format!("\"<{rid_hex}@{domain}>\"")
        }
        None => "NIL".to_string(),
    };

    format!(
        "({date} {subject} {from} {sender} {reply_to} {to} {cc} {bcc} {in_reply_to} {message_id})"
    )
}

/// Format a Harmony address hash as an IMAP address structure.
/// Returns `(NIL NIL "user-XXXX" "domain")`.
fn format_address_hash(hash: &[u8; ADDRESS_HASH_LEN], domain: &str) -> String {
    let hex = hex::encode(&hash[..4]);
    format!("(NIL NIL \"user-{hex}\" \"{domain}\")")
}

fn format_address_list(addrs: &[String]) -> String {
    if addrs.is_empty() {
        "NIL".to_string()
    } else {
        let joined = addrs.join("");
        format!("({joined})")
    }
}

// ── BODYSTRUCTURE ───────────────────────────────────────────────────

/// Build a BODYSTRUCTURE response for a HarmonyMessage.
///
/// For v1.1, all messages are treated as single-part text/plain.
/// Multipart rendering is deferred to when attachment content is available.
pub fn build_bodystructure(msg: &HarmonyMessage) -> String {
    // Compute size as it will appear on the wire (LF → CRLF normalization)
    let crlf_body = msg.body.replace('\n', "\r\n");
    let body_size = crlf_body.len();
    let lines = crlf_body.matches("\r\n").count() + 1;
    // UTF-8 can contain non-ASCII bytes → "8BIT" per RFC 2045
    format!("(\"TEXT\" \"PLAIN\" (\"CHARSET\" \"UTF-8\") NIL NIL \"8BIT\" {body_size} {lines})")
}

// ── RFC 5322 rendering ──────────────────────────────────────────────

/// Render a HarmonyMessage to RFC 5322 bytes.
///
/// Delegates to `outbound::build_rfc5322()` for the core rendering,
/// using the gateway domain for From/Message-ID headers.
pub fn render_rfc5322(msg: &HarmonyMessage, domain: &str) -> Result<Vec<u8>, RenderError> {
    let sender_display = format_sender_display(&msg.sender_address, domain);
    crate::outbound::build_rfc5322(msg, domain, &sender_display)
        .map_err(|e| RenderError::Build(e.to_string()))
}

fn format_sender_display(hash: &[u8; ADDRESS_HASH_LEN], domain: &str) -> String {
    let hex = hex::encode(&hash[..4]);
    format!("user-{hex}@{domain}")
}

// ── Section extraction ──────────────────────────────────────────────

/// Extract a section from rendered RFC 5322 bytes.
///
/// Supported sections:
/// - `None` (empty `[]`): entire message
/// - `"HEADER"`: headers only (up to blank line)
/// - `"TEXT"`: body only (after blank line)
/// - `"HEADER.FIELDS (field-list)"`: specific headers
pub fn render_section(rfc5322: &[u8], section: Option<&str>) -> Vec<u8> {
    match section {
        None => rfc5322.to_vec(),
        Some(s) => {
            let upper = s.to_uppercase();
            if upper == "HEADER" {
                extract_header(rfc5322)
            } else if upper == "TEXT" {
                extract_text(rfc5322)
            } else if upper.starts_with("HEADER.FIELDS") {
                // Extract specific headers
                let fields = parse_header_fields(s);
                extract_specific_headers(rfc5322, &fields)
            } else {
                // For numbered sections, return the whole body for now
                extract_text(rfc5322)
            }
        }
    }
}

fn extract_header(rfc5322: &[u8]) -> Vec<u8> {
    // Headers end at first \r\n\r\n (or \n\n)
    if let Some(pos) = find_header_end(rfc5322) {
        rfc5322[..pos + 4].to_vec() // include the blank line
    } else {
        rfc5322.to_vec()
    }
}

fn extract_text(rfc5322: &[u8]) -> Vec<u8> {
    if let Some(pos) = find_header_end(rfc5322) {
        rfc5322[pos + 4..].to_vec()
    } else {
        Vec::new()
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == b"\r\n\r\n")
}

fn parse_header_fields(section: &str) -> Vec<String> {
    // "HEADER.FIELDS (Subject From To)" -> ["SUBJECT", "FROM", "TO"]
    if let Some(start) = section.find('(') {
        if let Some(end) = section.find(')') {
            return section[start + 1..end]
                .split_whitespace()
                .map(|s| s.to_uppercase())
                .collect();
        }
    }
    Vec::new()
}

fn extract_specific_headers(rfc5322: &[u8], fields: &[String]) -> Vec<u8> {
    let header_bytes = extract_header(rfc5322);
    let header_str = String::from_utf8_lossy(&header_bytes);

    let mut result = String::new();
    let mut current_header = String::new();
    let mut include = false;

    for line in header_str.split("\r\n") {
        if line.is_empty() {
            // End of headers — flush the last header if it matched
            if include {
                result.push_str(&current_header);
            }
            break;
        }

        if !line.starts_with(' ') && !line.starts_with('\t') {
            // New header — flush the previous one if it matched
            if include {
                result.push_str(&current_header);
            }
            current_header = format!("{line}\r\n");
            let field_name = line.split(':').next().unwrap_or("").trim().to_uppercase();
            include = fields.iter().any(|f| f == &field_name);
        } else {
            // Continuation of current header
            current_header.push_str(line);
            current_header.push_str("\r\n");
        }
    }
    // Note: no post-loop flush — the blank-line break already handles
    // the last matching header. If there's no blank line (malformed input),
    // we intentionally don't emit partial headers.
    result.push_str("\r\n"); // trailing blank line
    result.into_bytes()
}

// ── FLAGS formatting ────────────────────────────────────────────────

/// Format flags for IMAP response: `(\Seen \Flagged)`
pub fn format_flags(flags: &[String]) -> String {
    if flags.is_empty() {
        "()".to_string()
    } else {
        format!("({})", flags.join(" "))
    }
}

// ── FETCH response building ─────────────────────────────────────────

/// Build a complete `* N FETCH (...)` response for one message.
///
/// Returns a list of response lines. For BODY[] with literal data,
/// the literal prefix `{N}\r\n` is included inline — the caller
/// writes each line followed by CRLF.
pub fn build_fetch_response(
    seqnum: u32,
    uid: u32,
    attrs: &[FetchAttribute],
    msg: &HarmonyMessage,
    flags: &[String],
    rfc822_size: u32,
    domain: &str,
) -> Result<FetchResponse, RenderError> {
    let mut items = Vec::new();
    let mut literal_data: Option<Vec<u8>> = None;

    // Reject requests with multiple literal-producing attributes — the response
    // model only supports one literal payload per FETCH response.
    let literal_attr_count = attrs
        .iter()
        .filter(|a| {
            matches!(
                a,
                FetchAttribute::BodySection { .. }
                    | FetchAttribute::Rfc822
                    | FetchAttribute::Rfc822Header
                    | FetchAttribute::Rfc822Text
            )
        })
        .count();
    if literal_attr_count > 1 {
        return Err(RenderError::Build(
            "multiple literal FETCH attributes in a single request are not yet supported"
                .to_string(),
        ));
    }

    for attr in attrs {
        match attr {
            FetchAttribute::Flags => {
                items.push(format!("FLAGS {}", format_flags(flags)));
            }
            FetchAttribute::Uid => {
                items.push(format!("UID {uid}"));
            }
            FetchAttribute::InternalDate => {
                let date = format_internal_date(msg.timestamp);
                items.push(format!("INTERNALDATE \"{date}\""));
            }
            FetchAttribute::Rfc822Size => {
                items.push(format!("RFC822.SIZE {rfc822_size}"));
            }
            FetchAttribute::Envelope => {
                items.push(format!("ENVELOPE {}", build_envelope(msg, domain)));
            }
            FetchAttribute::Body => {
                items.push(format!("BODY {}", build_bodystructure(msg)));
            }
            FetchAttribute::BodyStructure => {
                items.push(format!("BODYSTRUCTURE {}", build_bodystructure(msg)));
            }
            FetchAttribute::BodySection {
                section,
                partial,
                peek: _,
            } => {
                let rfc5322 = render_rfc5322(msg, domain)?;
                let data = render_section(&rfc5322, section.as_deref());
                let data = apply_partial(&data, *partial);

                let section_str = section.as_deref().unwrap_or("");
                let tag = "BODY"; // BODY.PEEK response uses "BODY" per RFC 9051 §7.4.2
                let section_display = if section_str.is_empty() {
                    "[]".to_string()
                } else {
                    format!("[{section_str}]")
                };
                items.push(format!("{tag}{section_display} {{{}}}", data.len()));
                literal_data = Some(data);
            }
            FetchAttribute::Rfc822 => {
                let rfc5322 = render_rfc5322(msg, domain)?;
                items.push(format!("RFC822 {{{}}}", rfc5322.len()));
                literal_data = Some(rfc5322);
            }
            FetchAttribute::Rfc822Header => {
                let rfc5322 = render_rfc5322(msg, domain)?;
                let header = extract_header(&rfc5322);
                items.push(format!("RFC822.HEADER {{{}}}", header.len()));
                literal_data = Some(header);
            }
            FetchAttribute::Rfc822Text => {
                let rfc5322 = render_rfc5322(msg, domain)?;
                let text = extract_text(&rfc5322);
                items.push(format!("RFC822.TEXT {{{}}}", text.len()));
                literal_data = Some(text);
            }
        }
    }

    Ok(FetchResponse {
        seqnum,
        items,
        literal_data,
    })
}

/// A rendered FETCH response ready for transmission.
#[derive(Debug)]
pub struct FetchResponse {
    /// Sequence number for the `* N FETCH` prefix.
    pub seqnum: u32,
    /// FETCH data items (e.g., `FLAGS (\Seen)`, `UID 5`).
    pub items: Vec<String>,
    /// Optional literal body data (for BODY[], RFC822, etc.).
    pub literal_data: Option<Vec<u8>>,
}

impl FetchResponse {
    /// Format the complete response for transmission.
    /// Returns bytes ready to write to the client.
    ///
    /// For literal data, the IMAP format is:
    /// `* 1 FETCH (... BODY[] {N}\r\n<data>)\r\n`
    /// The literal size marker `{N}` is already in the items list; we insert
    /// CRLF + raw data + closing paren after it.
    pub fn to_bytes(&self) -> Vec<u8> {
        let items_str = self.items.join(" ");
        let mut result = format!("* {} FETCH ({}", self.seqnum, items_str);
        if let Some(ref data) = self.literal_data {
            // Items already contain "{N}" — append CRLF, literal data, then close
            result.push_str("\r\n");
            let mut bytes = result.into_bytes();
            bytes.extend_from_slice(data);
            bytes.extend_from_slice(b")\r\n");
            bytes
        } else {
            result.push_str(")\r\n");
            result.into_bytes()
        }
    }
}

fn apply_partial(data: &[u8], partial: Option<(u32, u32)>) -> Vec<u8> {
    match partial {
        Some((offset, count)) => {
            let start = offset as usize;
            let end = std::cmp::min(start + count as usize, data.len());
            if start >= data.len() {
                Vec::new()
            } else {
                data[start..end].to_vec()
            }
        }
        None => data.to_vec(),
    }
}

// ── Date formatting ─────────────────────────────────────────────────

/// Format a unix timestamp as an IMAP date string: `"11-Apr-2026 00:00:00 +0000"`
pub(crate) fn format_internal_date(timestamp: u64) -> String {
    // Simple UTC formatting without chrono dependency
    let secs = timestamp;
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    let (year, month, day) = days_to_date(days);
    let month_str = MONTHS[month as usize - 1];

    format!("{day:02}-{month_str}-{year:04} {hours:02}:{minutes:02}:{seconds:02} +0000")
}

/// Format a unix timestamp for ENVELOPE date field.
fn format_imap_date(timestamp: u64) -> String {
    format!("\"{}\"", format_internal_date(timestamp))
}

/// Quote a string as an IMAP nstring (NIL if empty-ish, quoted otherwise).
fn quote_nstring(s: &str) -> String {
    if s.is_empty() {
        "NIL".to_string()
    } else {
        let escaped = s.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{escaped}\"")
    }
}

/// Convert days since Unix epoch to (year, month, day).
fn days_to_date(days: u64) -> (u32, u32, u32) {
    // Algorithm from Howard Hinnant's chrono-compatible date algorithms
    let z = days as i64 + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = (z - era * 146097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    (y as u32, m, d)
}

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

#[derive(Debug, thiserror::Error)]
pub enum RenderError {
    #[error("RFC 5322 build error: {0}")]
    Build(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{MailMessageType, MessageFlags, Recipient, VERSION};

    fn sample_message() -> HarmonyMessage {
        HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Email,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1744358400, // 2025-04-11T00:00:00Z
            message_id: [0xAA; 16],
            in_reply_to: None,
            sender_address: [0xBB; 16],
            recipients: vec![Recipient {
                address_hash: [0xCC; 16],
                recipient_type: RecipientType::To,
            }],
            subject: "Test subject".to_string(),
            body: "Hello from Harmony!".to_string(),
            attachments: Vec::new(),
        }
    }

    #[test]
    fn envelope_basic() {
        let msg = sample_message();
        let env = build_envelope(&msg, "q8.fyi");

        assert!(env.starts_with('('));
        assert!(env.ends_with(')'));
        assert!(env.contains("\"Test subject\""), "envelope: {env}");
        assert!(env.contains("q8.fyi"), "envelope: {env}");
        assert!(
            env.contains("NIL"),
            "should have NIL for missing in-reply-to: {env}"
        );
    }

    #[test]
    fn envelope_with_reply() {
        let mut msg = sample_message();
        msg.in_reply_to = Some([0xDD; 16]);
        let env = build_envelope(&msg, "q8.fyi");

        assert!(
            !env.contains("NIL NIL NIL NIL NIL NIL)"),
            "in-reply-to should not be NIL: {env}"
        );
        assert!(
            env.contains("dddddddddddddddddddddddddddddddd@q8.fyi"),
            "env: {env}"
        );
    }

    #[test]
    fn bodystructure_plain_text() {
        let msg = sample_message();
        let bs = build_bodystructure(&msg);
        assert!(bs.contains("\"TEXT\""));
        assert!(bs.contains("\"PLAIN\""));
        assert!(bs.contains("\"UTF-8\""));
    }

    #[test]
    fn render_rfc5322_produces_bytes() {
        let msg = sample_message();
        let bytes = render_rfc5322(&msg, "q8.fyi").unwrap();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Subject: Test subject"), "text: {text}");
        assert!(text.contains("Hello from Harmony!"), "text: {text}");
    }

    #[test]
    fn section_header_extraction() {
        let rfc5322 = b"Subject: Test\r\nFrom: a@b.com\r\n\r\nBody here";
        let header = render_section(rfc5322, Some("HEADER"));
        let text = String::from_utf8_lossy(&header);
        assert!(text.contains("Subject: Test"));
        assert!(text.contains("From: a@b.com"));
        assert!(!text.contains("Body here"));
    }

    #[test]
    fn section_text_extraction() {
        let rfc5322 = b"Subject: Test\r\n\r\nBody here";
        let text = render_section(rfc5322, Some("TEXT"));
        assert_eq!(text, b"Body here");
    }

    #[test]
    fn section_empty_is_whole_message() {
        let rfc5322 = b"Subject: Test\r\n\r\nBody";
        let whole = render_section(rfc5322, None);
        assert_eq!(whole, rfc5322);
    }

    #[test]
    fn format_flags_empty() {
        assert_eq!(format_flags(&[]), "()");
    }

    #[test]
    fn format_flags_multiple() {
        let flags = vec!["\\Seen".to_string(), "\\Flagged".to_string()];
        assert_eq!(format_flags(&flags), "(\\Seen \\Flagged)");
    }

    #[test]
    fn fetch_response_flags_and_uid() {
        let msg = sample_message();
        let resp = build_fetch_response(
            1,
            1,
            &[FetchAttribute::Flags, FetchAttribute::Uid],
            &msg,
            &["\\Seen".to_string()],
            500,
            "q8.fyi",
        )
        .unwrap();

        assert_eq!(resp.seqnum, 1);
        assert!(resp.items.iter().any(|i| i.contains("FLAGS")));
        assert!(resp.items.iter().any(|i| i.contains("UID 1")));
        assert!(resp.literal_data.is_none());
    }

    #[test]
    fn fetch_response_body_section_produces_literal() {
        let msg = sample_message();
        let resp = build_fetch_response(
            1,
            1,
            &[FetchAttribute::BodySection {
                section: None,
                partial: None,
                peek: false,
            }],
            &msg,
            &[],
            500,
            "q8.fyi",
        )
        .unwrap();

        assert!(resp.literal_data.is_some());
        let data = resp.literal_data.unwrap();
        let text = String::from_utf8_lossy(&data);
        assert!(text.contains("Hello from Harmony!"));
    }

    #[test]
    fn internal_date_formatting() {
        let date = format_internal_date(1744358400);
        // 2025-04-11 08:00:00 UTC
        assert!(date.contains("Apr"), "date: {date}");
        assert!(date.contains("2025"), "date: {date}");
        assert!(date.contains("08:00:00"), "date: {date}");
        assert!(date.contains("+0000"), "date: {date}");
    }

    #[test]
    fn partial_fetch() {
        let data = b"Hello, World!";
        assert_eq!(apply_partial(data, Some((0, 5))), b"Hello");
        assert_eq!(apply_partial(data, Some((7, 100))), b"World!");
        assert_eq!(apply_partial(data, Some((100, 5))), b"");
        assert_eq!(apply_partial(data, None), b"Hello, World!");
    }

    #[test]
    fn header_fields_extraction() {
        let rfc5322 = b"Subject: Test\r\nFrom: a@b.com\r\nTo: c@d.com\r\n\r\nBody";
        let result = render_section(rfc5322, Some("HEADER.FIELDS (Subject To)"));
        let text = String::from_utf8_lossy(&result);
        assert!(text.contains("Subject: Test"), "text: {text}");
        assert!(text.contains("To: c@d.com"), "text: {text}");
        assert!(!text.contains("From:"), "should not include From: {text}");
    }

    #[test]
    fn days_to_date_epoch() {
        let (y, m, d) = days_to_date(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }

    #[test]
    fn days_to_date_2025() {
        // 2025-04-11 = day 20189 since epoch
        let (y, m, d) = days_to_date(20189);
        assert_eq!((y, m, d), (2025, 4, 11));
    }

    #[test]
    fn quote_nstring_empty() {
        assert_eq!(quote_nstring(""), "NIL");
    }

    #[test]
    fn quote_nstring_special_chars() {
        assert_eq!(quote_nstring("hello \"world\""), "\"hello \\\"world\\\"\"");
    }
}

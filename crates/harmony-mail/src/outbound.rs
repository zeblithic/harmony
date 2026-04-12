//! Outbound SMTP: HarmonyMessage → RFC 5322 translation, sending via lettre,
//! and persistent queue with retry schedule.

use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::message::{self, HarmonyMessage, MailMessageType, RecipientType, ADDRESS_HASH_LEN};

// ── RFC 5322 message building ───────────────────────────────────────

/// Build an RFC 5322 message from a HarmonyMessage.
///
/// `domain` is the gateway's domain (used in From and Message-ID headers).
/// `sender_display` is the human-readable sender name/address for the From header.
///
/// Returns the message as a byte vector ready for SMTP transmission.
pub fn build_rfc5322(
    msg: &HarmonyMessage,
    domain: &str,
    sender_display: &str,
) -> Result<Vec<u8>, QueueError> {
    let mut builder = mail_builder::MessageBuilder::new();

    // From header
    builder = builder.from(sender_display.to_string());

    // To / Cc headers
    for recip in &msg.recipients {
        let addr = format_address_hash(&recip.address_hash, domain);
        match recip.recipient_type {
            RecipientType::To => {
                builder = builder.to(addr);
            }
            RecipientType::Cc => {
                builder = builder.cc(addr);
            }
            RecipientType::Bcc => {
                // BCC recipients get individual deliveries; not included in headers
            }
        }
    }

    // Subject
    builder = builder.subject(&msg.subject);

    // Date (from unix timestamp)
    let date = mail_builder::headers::date::Date::new(msg.timestamp as i64);
    builder = builder.date(date);

    // Message-ID
    let mid = hex::encode(msg.message_id);
    builder = builder.message_id(format!("{mid}@{domain}"));

    // In-Reply-To
    if let Some(ref reply_id) = msg.in_reply_to {
        let rid = hex::encode(reply_id);
        builder = builder.in_reply_to(format!("{rid}@{domain}"));
    }

    // Body (text/plain)
    builder = builder.text_body(&msg.body);

    // Note: attachment blob fetch from harmony-content is not yet implemented.
    // AttachmentRefs are included as X-Harmony-Attachment headers for now.
    for att in &msg.attachments {
        let cid_hex = hex::encode(att.cid);
        // Sanitize filename and mime_type to prevent header injection
        let safe_filename = sanitize_header_value(&att.filename);
        let safe_mime = sanitize_header_value(&att.mime_type);
        builder = builder.header(
            "X-Harmony-Attachment",
            mail_builder::headers::raw::Raw::new(format!(
                "cid={};filename=\"{}\";type={};size={}",
                cid_hex, safe_filename, safe_mime, att.size
            )),
        );
    }

    builder
        .write_to_vec()
        .map_err(|e| QueueError::Serialize(format!("RFC 5322 build failed: {e}")))
}

/// Sanitize a string for safe inclusion in header values.
/// Strips CR, LF, and escapes double quotes.
fn sanitize_header_value(s: &str) -> String {
    s.chars()
        .filter(|c| *c != '\r' && *c != '\n')
        .map(|c| if c == '"' { '\'' } else { c })
        .collect()
}

/// Format a Harmony address hash as an email address for outbound headers.
fn format_address_hash(hash: &[u8; ADDRESS_HASH_LEN], domain: &str) -> String {
    let hex = hex::encode(&hash[..4]); // first 4 bytes = 8 hex chars
    format!("user-{hex}@{domain}")
}

// ── Outbound Queue ──────────────────────────────────────────────────

/// A queued outbound message.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct QueuedMessage {
    /// Unique queue entry ID.
    pub id: u64,
    /// RFC 5322 message bytes.
    pub rfc5322: Vec<u8>,
    /// Destination email address (for SMTP RCPT TO).
    pub recipient: String,
    /// Number of delivery attempts so far.
    pub attempts: usize,
    /// Unix timestamp of when this was first queued.
    pub queued_at: u64,
    /// Unix timestamp of next retry.
    pub next_retry: u64,
}

/// Persistent outbound queue backed by a JSONL file.
pub struct OutboundQueue {
    path: PathBuf,
    retry_schedule: Vec<u64>,
    max_retries: usize,
}

impl OutboundQueue {
    /// Create or open an outbound queue at the given path.
    pub fn new(queue_path: &Path, retry_schedule: Vec<u64>, max_retries: usize) -> Self {
        Self {
            path: queue_path.to_path_buf(),
            retry_schedule,
            max_retries,
        }
    }

    /// Add a message to the outbound queue.
    pub fn enqueue(&self, rfc5322: Vec<u8>, recipient: String) -> Result<(), QueueError> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);

        let now = now_secs();
        let seq = COUNTER.fetch_add(1, Ordering::Relaxed);
        let pid = std::process::id() as u64;
        // Combine timestamp + counter + PID to avoid collisions across restarts
        let mut hasher = blake3::Hasher::new();
        hasher.update(&now.to_le_bytes());
        hasher.update(&seq.to_le_bytes());
        hasher.update(&pid.to_le_bytes());
        let hash = hasher.finalize();
        let id = u64::from_le_bytes(hash.as_bytes()[..8].try_into().unwrap());
        let entry = QueuedMessage {
            id,
            rfc5322,
            recipient,
            attempts: 0,
            queued_at: now,
            next_retry: now, // immediate first attempt
        };

        let mut line =
            serde_json::to_string(&entry).map_err(|e| QueueError::Serialize(e.to_string()))?;
        line.push('\n');

        std::fs::create_dir_all(self.path.parent().unwrap_or(Path::new(".")))
            .map_err(|e| QueueError::Io(e.to_string()))?;

        use std::io::Write;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| QueueError::Io(e.to_string()))?;
        file.write_all(line.as_bytes())
            .map_err(|e| QueueError::Io(e.to_string()))?;

        Ok(())
    }

    /// Read all pending queue entries.
    pub fn read_pending(&self) -> Result<Vec<QueuedMessage>, QueueError> {
        let contents = match std::fs::read_to_string(&self.path) {
            Ok(c) => c,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(QueueError::Io(e.to_string())),
        };

        let mut entries = Vec::new();
        for line in contents.lines() {
            if line.trim().is_empty() {
                continue;
            }
            let entry: QueuedMessage =
                serde_json::from_str(line).map_err(|e| QueueError::Serialize(e.to_string()))?;
            entries.push(entry);
        }
        Ok(entries)
    }

    /// Rewrite the queue file with updated entries (after processing).
    /// Uses write-then-rename for crash safety — a crash mid-write leaves
    /// the original file intact instead of truncated/empty.
    pub fn write_all(&self, entries: &[QueuedMessage]) -> Result<(), QueueError> {
        let mut content = String::new();
        for entry in entries {
            let line =
                serde_json::to_string(entry).map_err(|e| QueueError::Serialize(e.to_string()))?;
            content.push_str(&line);
            content.push('\n');
        }
        let tmp_path = self.path.with_extension("tmp");
        std::fs::write(&tmp_path, content.as_bytes()).map_err(|e| QueueError::Io(e.to_string()))?;
        std::fs::rename(&tmp_path, &self.path).map_err(|e| QueueError::Io(e.to_string()))?;
        Ok(())
    }

    /// Get the retry delay for the given attempt number.
    pub fn retry_delay(&self, attempt: usize) -> Option<Duration> {
        if attempt >= self.max_retries {
            return None; // No more retries
        }
        let secs = self
            .retry_schedule
            .get(attempt)
            .copied()
            .unwrap_or(*self.retry_schedule.last().unwrap_or(&259200));
        Some(Duration::from_secs(secs))
    }
}

/// Build a bounce HarmonyMessage for a permanently failed delivery.
pub fn build_bounce(original: &HarmonyMessage, reason: &str) -> HarmonyMessage {
    use crate::message::{MessageFlags, VERSION};

    HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Bounce,
        flags: MessageFlags::new(false, true, false),
        timestamp: now_secs(),
        message_id: message::unique_message_id(),
        in_reply_to: Some(original.message_id),
        sender_address: [0u8; ADDRESS_HASH_LEN], // system sender
        recipients: vec![crate::message::Recipient {
            address_hash: original.sender_address,
            recipient_type: RecipientType::To,
        }],
        subject: format!("Delivery failure: {}", original.subject),
        body: format!(
            "Your message could not be delivered after multiple attempts.\n\n\
             Original subject: {}\n\
             Reason: {}\n",
            original.subject, reason
        ),
        attachments: Vec::new(),
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

/// Queue-related errors.
#[derive(Debug, thiserror::Error)]
pub enum QueueError {
    #[error("I/O error: {0}")]
    Io(String),
    #[error("serialization error: {0}")]
    Serialize(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{MessageFlags, Recipient, VERSION};

    fn sample_message() -> HarmonyMessage {
        HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Email,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1744358400, // 2025-04-11T00:00:00Z
            message_id: [0xAA; 16],
            in_reply_to: None,
            sender_address: [0xBB; ADDRESS_HASH_LEN],
            recipients: vec![Recipient {
                address_hash: [0xCC; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            }],
            subject: "Test subject".to_string(),
            body: "Hello from Harmony!".to_string(),
            attachments: Vec::new(),
        }
    }

    #[test]
    fn build_rfc5322_basic() {
        let msg = sample_message();
        let rfc5322 = build_rfc5322(&msg, "q8.fyi", "sender@q8.fyi").unwrap();
        let text = String::from_utf8_lossy(&rfc5322);

        eprintln!("RFC5322 output:\n{text}");

        assert!(text.contains("sender@q8.fyi"), "missing From address");
        assert!(text.contains("Subject: Test subject"), "missing Subject");
        assert!(text.contains("Hello from Harmony!"), "missing body");
        assert!(text.contains("@q8.fyi"), "missing Message-ID domain");
    }

    #[test]
    fn build_rfc5322_with_reply() {
        let mut msg = sample_message();
        msg.in_reply_to = Some([0xDD; 16]);
        msg.flags = MessageFlags::new(false, true, false);

        let rfc5322 = build_rfc5322(&msg, "q8.fyi", "sender@q8.fyi").unwrap();
        let text = String::from_utf8_lossy(&rfc5322);

        assert!(text.contains("In-Reply-To:"), "missing In-Reply-To header");
    }

    #[test]
    fn build_rfc5322_multiple_recipients() {
        let mut msg = sample_message();
        msg.recipients.push(Recipient {
            address_hash: [0xDD; ADDRESS_HASH_LEN],
            recipient_type: RecipientType::Cc,
        });

        let rfc5322 = build_rfc5322(&msg, "q8.fyi", "sender@q8.fyi").unwrap();
        let text = String::from_utf8_lossy(&rfc5322);

        assert!(text.contains("To:"), "missing To header");
        assert!(text.contains("Cc:"), "missing Cc header");
    }

    #[test]
    fn format_address_hash_uses_first_4_bytes() {
        let hash = [
            0xA1, 0xB2, 0xC3, 0xD4, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
            0x00, 0x00,
        ];
        let formatted = format_address_hash(&hash, "example.com");
        assert_eq!(formatted, "user-a1b2c3d4@example.com");
    }

    #[test]
    fn queue_enqueue_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let queue_path = dir.path().join("queue.jsonl");
        let queue = OutboundQueue::new(&queue_path, vec![300, 900], 3);

        queue
            .enqueue(b"RFC5322 content".to_vec(), "test@example.com".to_string())
            .unwrap();
        queue
            .enqueue(b"Second message".to_vec(), "other@example.com".to_string())
            .unwrap();

        let entries = queue.read_pending().unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].recipient, "test@example.com");
        assert_eq!(entries[1].recipient, "other@example.com");
    }

    #[test]
    fn queue_read_empty() {
        let dir = tempfile::tempdir().unwrap();
        let queue_path = dir.path().join("nonexistent.jsonl");
        let queue = OutboundQueue::new(&queue_path, vec![300], 3);

        let entries = queue.read_pending().unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn queue_retry_schedule() {
        let queue = OutboundQueue::new(Path::new("/tmp/test"), vec![300, 900, 1800], 3);

        assert_eq!(queue.retry_delay(0), Some(Duration::from_secs(300)));
        assert_eq!(queue.retry_delay(1), Some(Duration::from_secs(900)));
        assert_eq!(queue.retry_delay(2), Some(Duration::from_secs(1800)));
        assert_eq!(queue.retry_delay(3), None); // max retries exceeded
    }

    #[test]
    fn build_bounce_message() {
        let original = sample_message();
        let bounce = build_bounce(&original, "Mailbox not found");

        assert_eq!(bounce.message_type, MailMessageType::Bounce);
        assert_eq!(bounce.in_reply_to, Some(original.message_id));
        assert_eq!(bounce.recipients[0].address_hash, original.sender_address);
        assert!(bounce.subject.contains("Delivery failure"));
        assert!(bounce.body.contains("Mailbox not found"));
    }

    #[test]
    fn queue_write_all_replaces_contents() {
        let dir = tempfile::tempdir().unwrap();
        let queue_path = dir.path().join("queue.jsonl");
        let queue = OutboundQueue::new(&queue_path, vec![300], 3);

        // Enqueue 3 messages
        queue
            .enqueue(b"msg1".to_vec(), "a@b.com".to_string())
            .unwrap();
        queue
            .enqueue(b"msg2".to_vec(), "c@d.com".to_string())
            .unwrap();
        queue
            .enqueue(b"msg3".to_vec(), "e@f.com".to_string())
            .unwrap();

        // Simulate processing: remove the first, keep the rest
        let mut entries = queue.read_pending().unwrap();
        entries.remove(0);
        queue.write_all(&entries).unwrap();

        let remaining = queue.read_pending().unwrap();
        assert_eq!(remaining.len(), 2);
        assert_eq!(remaining[0].recipient, "c@d.com");
    }
}

//! Inbound format translation: RFC 5322 -> HarmonyMessage.
//!
//! Parses raw RFC 5322 email bytes using `mail-parser` and translates
//! to the Harmony-native message format. Attachments are extracted and
//! hashed with BLAKE3 to produce CID references.

use mail_parser::MimeHeaders;

use crate::message::{
    AttachmentRef, HarmonyMessage, MailMessageType, MessageFlags, Recipient, RecipientType,
    ADDRESS_HASH_LEN, CID_LEN, MESSAGE_ID_LEN, VERSION,
};

/// Result of translating an inbound RFC 5322 message.
#[derive(Debug)]
pub struct TranslatedMessage {
    /// The translated HarmonyMessage.
    pub message: HarmonyMessage,
    /// Extracted attachment data, paired with their AttachmentRef.
    /// The AttachmentRef in this vec mirrors the one in `message.attachments`.
    pub attachment_data: Vec<Vec<u8>>,
}

/// Translate a raw RFC 5322 email into a HarmonyMessage.
///
/// For now, recipient address hashes are derived by hashing the email address
/// with BLAKE3 and truncating to 16 bytes. In production, these would be
/// resolved via the Harmony announce cache.
pub fn translate_inbound(raw: &[u8]) -> Result<TranslatedMessage, TranslateError> {
    let parsed = mail_parser::MessageParser::default()
        .parse(raw)
        .ok_or(TranslateError::ParseFailed)?;

    // ── Sender ──────────────────────────────────────────────────────
    let sender_address = parsed
        .from()
        .and_then(|addrs| addrs.first())
        .and_then(|addr| addr.address())
        .map(hash_email_to_address)
        .unwrap_or([0u8; ADDRESS_HASH_LEN]);

    // ── Recipients ──────────────────────────────────────────────────
    let mut recipients = Vec::new();

    if let Some(to_list) = parsed.to() {
        for addr in to_list.iter() {
            if let Some(email) = addr.address() {
                recipients.push(Recipient {
                    address_hash: hash_email_to_address(email),
                    recipient_type: RecipientType::To,
                });
            }
        }
    }

    if let Some(cc_list) = parsed.cc() {
        for addr in cc_list.iter() {
            if let Some(email) = addr.address() {
                recipients.push(Recipient {
                    address_hash: hash_email_to_address(email),
                    recipient_type: RecipientType::Cc,
                });
            }
        }
    }

    // BCC is typically not present in received messages, but handle it
    if let Some(bcc_list) = parsed.bcc() {
        for addr in bcc_list.iter() {
            if let Some(email) = addr.address() {
                recipients.push(Recipient {
                    address_hash: hash_email_to_address(email),
                    recipient_type: RecipientType::Bcc,
                });
            }
        }
    }

    // ── Subject ─────────────────────────────────────────────────────
    let subject = parsed
        .subject()
        .unwrap_or("")
        .to_string();

    // ── Timestamp ───────────────────────────────────────────────────
    let timestamp = parsed
        .date()
        .map(|dt| dt.to_timestamp() as u64)
        .unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs()
        });

    // ── Message ID ──────────────────────────────────────────────────
    let message_id = parsed
        .message_id()
        .map(|id| hash_to_message_id(id.as_bytes()))
        .unwrap_or_else(random_message_id);

    // ── In-Reply-To ─────────────────────────────────────────────────
    let in_reply_to = parsed
        .in_reply_to()
        .as_text()
        .map(|id| hash_to_message_id(id.as_bytes()));

    // ── Body ────────────────────────────────────────────────────────
    // Prefer text/plain; fall back to stripping HTML tags
    let body = if let Some(text) = parsed.body_text(0) {
        text.to_string()
    } else if let Some(html) = parsed.body_html(0) {
        strip_html_tags(&html)
    } else {
        String::new()
    };

    // ── Flags (is_reply, is_forward) ────────────────────────────────
    let is_reply = in_reply_to.is_some();
    let is_forward = subject.starts_with("Fwd:") || subject.starts_with("FW:");

    // ── Attachments ─────────────────────────────────────────────────
    let mut attachments = Vec::new();
    let mut attachment_data = Vec::new();

    for part in parsed.attachments() {
        let data = part.contents();
        let cid = blake3_cid(data);

        let filename = part
            .attachment_name()
            .unwrap_or("attachment")
            .to_string();

        let mime_type = part
            .content_type()
            .map(|ct| {
                let main = ct.ctype();
                match ct.subtype() {
                    Some(sub) => format!("{main}/{sub}"),
                    None => main.to_string(),
                }
            })
            .unwrap_or_else(|| "application/octet-stream".to_string());

        attachments.push(AttachmentRef {
            cid,
            filename,
            mime_type,
            size: data.len() as u64,
        });
        attachment_data.push(data.to_vec());
    }

    let has_attachments = !attachments.is_empty();
    let flags = MessageFlags::new(has_attachments, is_reply, is_forward);

    let message = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Email,
        flags,
        timestamp,
        message_id,
        in_reply_to,
        sender_address,
        recipients,
        subject,
        body,
        attachments,
    };

    Ok(TranslatedMessage {
        message,
        attachment_data,
    })
}

/// Hash an email address to a 16-byte Harmony address hash.
/// In production, this would be replaced by announce cache lookup.
fn hash_email_to_address(email: &str) -> [u8; ADDRESS_HASH_LEN] {
    let hash = blake3::hash(email.as_bytes());
    let mut addr = [0u8; ADDRESS_HASH_LEN];
    addr.copy_from_slice(&hash.as_bytes()[..ADDRESS_HASH_LEN]);
    addr
}

/// Hash arbitrary bytes to a 16-byte message ID.
fn hash_to_message_id(data: &[u8]) -> [u8; MESSAGE_ID_LEN] {
    let hash = blake3::hash(data);
    let mut id = [0u8; MESSAGE_ID_LEN];
    id.copy_from_slice(&hash.as_bytes()[..MESSAGE_ID_LEN]);
    id
}

/// Generate a random 16-byte message ID.
fn random_message_id() -> [u8; MESSAGE_ID_LEN] {
    let hash = blake3::hash(&std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos()
        .to_le_bytes());
    let mut id = [0u8; MESSAGE_ID_LEN];
    id.copy_from_slice(&hash.as_bytes()[..MESSAGE_ID_LEN]);
    id
}

/// Compute a BLAKE3 CID (32 bytes) for attachment content.
fn blake3_cid(data: &[u8]) -> [u8; CID_LEN] {
    *blake3::hash(data).as_bytes()
}

/// Basic HTML tag stripping. Not a full HTML parser — strips angle-bracket tags
/// and decodes common entities. Good enough for email body extraction.
fn strip_html_tags(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;

    for ch in html.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }

    // Decode common HTML entities
    result
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&nbsp;", " ")
}

/// Errors during inbound translation.
#[derive(Debug, thiserror::Error)]
pub enum TranslateError {
    #[error("failed to parse RFC 5322 message")]
    ParseFailed,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal but valid RFC 5322 plain-text email.
    const PLAIN_EMAIL: &[u8] = b"From: sender@example.com\r\n\
To: recipient@harmony.test\r\n\
Subject: Hello from SMTP\r\n\
Date: Fri, 11 Apr 2026 12:00:00 +0000\r\n\
Message-ID: <test123@example.com>\r\n\
\r\n\
This is the body text.\r\n";

    #[test]
    fn translate_plain_text_email() {
        let result = translate_inbound(PLAIN_EMAIL).unwrap();
        let msg = &result.message;

        assert_eq!(msg.version, VERSION);
        assert_eq!(msg.message_type, MailMessageType::Email);
        assert_eq!(msg.subject, "Hello from SMTP");
        assert_eq!(msg.body, "This is the body text.\r\n");
        assert_eq!(msg.recipients.len(), 1);
        assert_eq!(msg.recipients[0].recipient_type, RecipientType::To);
        assert!(!msg.flags.has_attachments());
        assert!(!msg.flags.is_reply());
        assert!(!msg.flags.is_forward());
        assert!(result.attachment_data.is_empty());

        // Sender should be a hash of the email address
        let expected_sender = hash_email_to_address("sender@example.com");
        assert_eq!(msg.sender_address, expected_sender);

        // Message-ID should be a hash of the raw ID
        let expected_id = hash_to_message_id(b"test123@example.com");
        assert_eq!(msg.message_id, expected_id);
    }

    #[test]
    fn translate_reply_sets_flag() {
        let email = b"From: a@b.com\r\n\
To: c@d.com\r\n\
Subject: Re: something\r\n\
In-Reply-To: <orig@b.com>\r\n\
\r\n\
reply body\r\n";

        let result = translate_inbound(email).unwrap();
        assert!(result.message.flags.is_reply());
        assert!(result.message.in_reply_to.is_some());
    }

    #[test]
    fn translate_forward_sets_flag() {
        let email = b"From: a@b.com\r\n\
To: c@d.com\r\n\
Subject: Fwd: important\r\n\
\r\n\
forwarded body\r\n";

        let result = translate_inbound(email).unwrap();
        assert!(result.message.flags.is_forward());
    }

    #[test]
    fn translate_multiple_recipients() {
        let email = b"From: sender@test.com\r\n\
To: alice@test.com, bob@test.com\r\n\
Cc: charlie@test.com\r\n\
Subject: Group mail\r\n\
\r\n\
Hi everyone\r\n";

        let result = translate_inbound(email).unwrap();
        let msg = &result.message;
        assert_eq!(msg.recipients.len(), 3);

        let to_count = msg.recipients.iter()
            .filter(|r| r.recipient_type == RecipientType::To)
            .count();
        let cc_count = msg.recipients.iter()
            .filter(|r| r.recipient_type == RecipientType::Cc)
            .count();
        assert_eq!(to_count, 2);
        assert_eq!(cc_count, 1);
    }

    #[test]
    fn translate_html_only_strips_tags() {
        let email = b"From: a@b.com\r\n\
To: c@d.com\r\n\
Subject: HTML\r\n\
Content-Type: text/html\r\n\
\r\n\
<html><body><p>Hello &amp; welcome</p></body></html>\r\n";

        let result = translate_inbound(email).unwrap();
        assert!(result.message.body.contains("Hello & welcome"));
        assert!(!result.message.body.contains("<p>"));
    }

    #[test]
    fn translate_with_attachment() {
        // Multipart MIME message with a text body and a binary attachment
        let email = b"From: a@b.com\r\n\
To: c@d.com\r\n\
Subject: With file\r\n\
MIME-Version: 1.0\r\n\
Content-Type: multipart/mixed; boundary=\"boundary42\"\r\n\
\r\n\
--boundary42\r\n\
Content-Type: text/plain\r\n\
\r\n\
See attached.\r\n\
--boundary42\r\n\
Content-Type: application/pdf\r\n\
Content-Disposition: attachment; filename=\"doc.pdf\"\r\n\
Content-Transfer-Encoding: base64\r\n\
\r\n\
JVBERi0xLjAKMSAwIG9iago8PAovVHlwZSAvQ2F0YWxvZwovUGFnZXMgMiAwIFIKPj4KZW5k\r\n\
b2JqCg==\r\n\
--boundary42--\r\n";

        let result = translate_inbound(email).unwrap();
        let msg = &result.message;

        assert!(msg.flags.has_attachments());
        assert_eq!(msg.attachments.len(), 1);
        assert_eq!(msg.attachments[0].filename, "doc.pdf");
        assert_eq!(msg.attachments[0].mime_type, "application/pdf");
        assert!(msg.attachments[0].size > 0);

        // Attachment data should match
        assert_eq!(result.attachment_data.len(), 1);
        assert_eq!(result.attachment_data[0].len(), msg.attachments[0].size as usize);

        // CID should be BLAKE3 of the decoded content
        let expected_cid = blake3_cid(&result.attachment_data[0]);
        assert_eq!(msg.attachments[0].cid, expected_cid);
    }

    #[test]
    fn translate_missing_sender_uses_zero_hash() {
        let email = b"To: c@d.com\r\n\
Subject: No from\r\n\
\r\n\
body\r\n";

        let result = translate_inbound(email).unwrap();
        assert_eq!(result.message.sender_address, [0u8; ADDRESS_HASH_LEN]);
    }

    #[test]
    fn strip_html_handles_entities() {
        assert_eq!(
            strip_html_tags("<b>bold &amp; &lt;escaped&gt;</b>"),
            "bold & <escaped>"
        );
    }

    #[test]
    fn blake3_cid_is_deterministic() {
        let data = b"hello world";
        let cid1 = blake3_cid(data);
        let cid2 = blake3_cid(data);
        assert_eq!(cid1, cid2);
        assert_ne!(cid1, [0u8; CID_LEN]);
    }
}

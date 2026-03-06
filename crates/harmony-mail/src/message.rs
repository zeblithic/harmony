//! Harmony-native email message format.
//!
//! Binary wire format for HarmonyMessage — the internal representation used
//! after SMTP ingress and before Reticulum/Zenoh transport.
//!
//! See docs/plans/2026-03-05-harmony-mail-design.md for wire format spec.

use crate::error::MailError;

// ── Constants ──────────────────────────────────────────────────────────

/// Current wire format version.
pub const VERSION: u8 = 0x01;

/// Maximum subject length in bytes (RFC 2822 line limit).
pub const MAX_SUBJECT_LEN: usize = 998;

/// Maximum body length: 16 MiB.
pub const MAX_BODY_LEN: usize = 16 * 1024 * 1024;

/// Maximum number of recipients per message.
pub const MAX_RECIPIENTS: usize = 100;

/// Maximum number of attachment references per message.
pub const MAX_ATTACHMENTS: usize = 100;

/// Length of a Harmony address hash in bytes.
pub const ADDRESS_HASH_LEN: usize = 16;

/// Length of a content identifier (CID) in bytes.
pub const CID_LEN: usize = 32;

/// Length of a message identifier in bytes.
pub const MESSAGE_ID_LEN: usize = 16;

// ── Minimum wire size ──────────────────────────────────────────────────
// version(1) + type(1) + flags(1) + timestamp(8) + message_id(16)
// + in_reply_to_flag(1) + sender_address(16) + recipient_count(1)
// + subject_len(2) + body_len(4) + attachment_count(1)
const MIN_HEADER_SIZE: usize = 1 + 1 + 1 + 8 + 16 + 1 + 16 + 1 + 2 + 4 + 1;

// ── Enums ──────────────────────────────────────────────────────────────

/// The type of mail message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailMessageType {
    /// Standard email message.
    Email = 0x00,
    /// Delivery/read receipt.
    Receipt = 0x01,
    /// Bounce notification.
    Bounce = 0x02,
}

impl MailMessageType {
    /// Decode from a single byte.
    pub fn from_u8(val: u8) -> Result<Self, MailError> {
        match val {
            0x00 => Ok(Self::Email),
            0x01 => Ok(Self::Receipt),
            0x02 => Ok(Self::Bounce),
            other => Err(MailError::InvalidMessageType(other)),
        }
    }
}

/// Recipient role within a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipientType {
    /// Primary recipient.
    To = 0x00,
    /// Carbon-copy recipient.
    Cc = 0x01,
    /// Blind carbon-copy recipient (stripped before delivery).
    Bcc = 0x02,
}

impl RecipientType {
    /// Decode from a single byte.
    pub fn from_u8(val: u8) -> Result<Self, MailError> {
        match val {
            0x00 => Ok(Self::To),
            0x01 => Ok(Self::Cc),
            0x02 => Ok(Self::Bcc),
            other => Err(MailError::InvalidRecipientType(other)),
        }
    }
}

// ── MessageFlags ───────────────────────────────────────────────────────

/// Bitfield flags for a message.
///
/// - Bit 0: has attachments
/// - Bit 1: is reply
/// - Bit 2: is forward
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageFlags(u8);

impl MessageFlags {
    /// Construct flags from individual booleans.
    pub fn new(has_attachments: bool, is_reply: bool, is_forward: bool) -> Self {
        let mut bits = 0u8;
        if has_attachments {
            bits |= 1 << 0;
        }
        if is_reply {
            bits |= 1 << 1;
        }
        if is_forward {
            bits |= 1 << 2;
        }
        Self(bits)
    }

    /// Construct from raw bits.
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    /// Return the raw bits.
    pub fn bits(self) -> u8 {
        self.0
    }

    /// Whether the message has attachments.
    pub fn has_attachments(self) -> bool {
        self.0 & (1 << 0) != 0
    }

    /// Whether the message is a reply.
    pub fn is_reply(self) -> bool {
        self.0 & (1 << 1) != 0
    }

    /// Whether the message is a forward.
    pub fn is_forward(self) -> bool {
        self.0 & (1 << 2) != 0
    }
}

// ── Recipient ──────────────────────────────────────────────────────────

/// A message recipient identified by address hash and role.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipient {
    /// 128-bit Harmony address hash.
    pub address_hash: [u8; ADDRESS_HASH_LEN],
    /// Role of this recipient (To/Cc/Bcc).
    pub recipient_type: RecipientType,
}

// ── AttachmentRef ──────────────────────────────────────────────────────

/// Reference to an attachment stored in the content-addressed layer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentRef {
    /// Content identifier (32-byte hash).
    pub cid: [u8; CID_LEN],
    /// Original filename.
    pub filename: String,
    /// MIME type (e.g. "application/pdf").
    pub mime_type: String,
    /// Size in bytes.
    pub size: u64,
}

// ── HarmonyMessage ─────────────────────────────────────────────────────

/// A fully-decoded Harmony mail message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyMessage {
    /// Wire format version (currently 0x01).
    pub version: u8,
    /// Message type (Email, Receipt, Bounce).
    pub message_type: MailMessageType,
    /// Bitfield flags.
    pub flags: MessageFlags,
    /// Unix timestamp in seconds.
    pub timestamp: u64,
    /// Unique message identifier.
    pub message_id: [u8; MESSAGE_ID_LEN],
    /// If this is a reply, the ID of the original message.
    pub in_reply_to: Option<[u8; MESSAGE_ID_LEN]>,
    /// Sender's Harmony address hash.
    pub sender_address: [u8; ADDRESS_HASH_LEN],
    /// List of recipients.
    pub recipients: Vec<Recipient>,
    /// Subject line (UTF-8).
    pub subject: String,
    /// Message body (UTF-8).
    pub body: String,
    /// Attachment references.
    pub attachments: Vec<AttachmentRef>,
}

impl HarmonyMessage {
    /// Serialize the message to its binary wire format.
    ///
    /// # Errors
    ///
    /// Returns `MailError` if any field exceeds its size limit.
    pub fn to_bytes(&self) -> Result<Vec<u8>, MailError> {
        // Validate limits
        if self.subject.len() > MAX_SUBJECT_LEN {
            return Err(MailError::SubjectTooLong {
                len: self.subject.len(),
                max: MAX_SUBJECT_LEN,
            });
        }
        if self.body.len() > MAX_BODY_LEN {
            return Err(MailError::BodyTooLong {
                len: self.body.len(),
                max: MAX_BODY_LEN,
            });
        }
        if self.recipients.len() > MAX_RECIPIENTS {
            return Err(MailError::TooManyRecipients {
                count: self.recipients.len(),
                max: MAX_RECIPIENTS,
            });
        }
        if self.attachments.len() > MAX_ATTACHMENTS {
            return Err(MailError::TooManyAttachments {
                count: self.attachments.len(),
                max: MAX_ATTACHMENTS,
            });
        }

        // Validate attachment string lengths (must fit in u8)
        for att in &self.attachments {
            if att.filename.len() > 255 {
                return Err(MailError::FilenameTooLong {
                    len: att.filename.len(),
                });
            }
            if att.mime_type.len() > 255 {
                return Err(MailError::MimeTypeTooLong {
                    len: att.mime_type.len(),
                });
            }
        }

        let mut buf = Vec::with_capacity(MIN_HEADER_SIZE);

        // version: 1 byte
        buf.push(self.version);
        // message_type: 1 byte
        buf.push(self.message_type as u8);
        // flags: 1 byte
        buf.push(self.flags.bits());
        // timestamp: 8 bytes big-endian
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        // message_id: 16 bytes
        buf.extend_from_slice(&self.message_id);
        // in_reply_to: 1 byte flag + optional 16 bytes
        match &self.in_reply_to {
            None => buf.push(0x00),
            Some(id) => {
                buf.push(0x01);
                buf.extend_from_slice(id);
            }
        }
        // sender_address: 16 bytes
        buf.extend_from_slice(&self.sender_address);
        // recipient_count: 1 byte
        buf.push(self.recipients.len() as u8);
        // each recipient: 16 bytes address_hash + 1 byte type
        for r in &self.recipients {
            buf.extend_from_slice(&r.address_hash);
            buf.push(r.recipient_type as u8);
        }
        // subject_len: 2 bytes big-endian u16
        buf.extend_from_slice(&(self.subject.len() as u16).to_be_bytes());
        // subject: variable UTF-8
        buf.extend_from_slice(self.subject.as_bytes());
        // body_len: 4 bytes big-endian u32
        buf.extend_from_slice(&(self.body.len() as u32).to_be_bytes());
        // body: variable UTF-8
        buf.extend_from_slice(self.body.as_bytes());
        // attachment_count: 1 byte
        buf.push(self.attachments.len() as u8);
        // each attachment: 32 bytes cid + 1 byte filename_len + filename
        //                  + 1 byte mime_type_len + mime_type + 8 bytes size
        for att in &self.attachments {
            buf.extend_from_slice(&att.cid);
            buf.push(att.filename.len() as u8);
            buf.extend_from_slice(att.filename.as_bytes());
            buf.push(att.mime_type.len() as u8);
            buf.extend_from_slice(att.mime_type.as_bytes());
            buf.extend_from_slice(&att.size.to_be_bytes());
        }

        Ok(buf)
    }

    /// Deserialize a message from its binary wire format.
    ///
    /// # Errors
    ///
    /// Returns `MailError` for malformed, truncated, or oversized input.
    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < MIN_HEADER_SIZE {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: MIN_HEADER_SIZE,
            });
        }

        let mut pos = 0;

        // version: 1 byte
        let version = data[pos];
        if version != VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }
        pos += 1;

        // message_type: 1 byte
        let message_type = MailMessageType::from_u8(data[pos])?;
        pos += 1;

        // flags: 1 byte
        let flags = MessageFlags::from_bits(data[pos]);
        pos += 1;

        // timestamp: 8 bytes big-endian
        let timestamp = read_u64_be(data, &mut pos)?;

        // message_id: 16 bytes
        let message_id = read_fixed::<MESSAGE_ID_LEN>(data, &mut pos)?;

        // in_reply_to: 1 byte flag + optional 16 bytes
        let in_reply_to_flag = read_u8(data, &mut pos)?;
        let in_reply_to = match in_reply_to_flag {
            0x00 => None,
            0x01 => Some(read_fixed::<MESSAGE_ID_LEN>(data, &mut pos)?),
            other => return Err(MailError::InvalidInReplyToFlag(other)),
        };

        // sender_address: 16 bytes
        let sender_address = read_fixed::<ADDRESS_HASH_LEN>(data, &mut pos)?;

        // recipient_count: 1 byte
        let recipient_count = read_u8(data, &mut pos)? as usize;
        if recipient_count > MAX_RECIPIENTS {
            return Err(MailError::TooManyRecipients {
                count: recipient_count,
                max: MAX_RECIPIENTS,
            });
        }

        // each recipient: 16 bytes address_hash + 1 byte type
        let mut recipients = Vec::with_capacity(recipient_count);
        for _ in 0..recipient_count {
            let address_hash = read_fixed::<ADDRESS_HASH_LEN>(data, &mut pos)?;
            let rtype = RecipientType::from_u8(read_u8(data, &mut pos)?)?;
            recipients.push(Recipient {
                address_hash,
                recipient_type: rtype,
            });
        }

        // subject_len: 2 bytes big-endian u16
        let subject_len = read_u16_be(data, &mut pos)? as usize;
        if subject_len > MAX_SUBJECT_LEN {
            return Err(MailError::SubjectTooLong {
                len: subject_len,
                max: MAX_SUBJECT_LEN,
            });
        }

        // subject: variable UTF-8
        let subject = read_utf8(data, &mut pos, subject_len, "subject")?;

        // body_len: 4 bytes big-endian u32
        let body_len = read_u32_be(data, &mut pos)? as usize;
        if body_len > MAX_BODY_LEN {
            return Err(MailError::BodyTooLong {
                len: body_len,
                max: MAX_BODY_LEN,
            });
        }

        // body: variable UTF-8
        let body = read_utf8(data, &mut pos, body_len, "body")?;

        // attachment_count: 1 byte
        let attachment_count = read_u8(data, &mut pos)? as usize;
        if attachment_count > MAX_ATTACHMENTS {
            return Err(MailError::TooManyAttachments {
                count: attachment_count,
                max: MAX_ATTACHMENTS,
            });
        }

        // each attachment
        let mut attachments = Vec::with_capacity(attachment_count);
        for _ in 0..attachment_count {
            let cid = read_fixed::<CID_LEN>(data, &mut pos)?;
            let filename_len = read_u8(data, &mut pos)? as usize;
            let filename = read_utf8(data, &mut pos, filename_len, "filename")?;
            let mime_len = read_u8(data, &mut pos)? as usize;
            let mime_type = read_utf8(data, &mut pos, mime_len, "mime_type")?;
            let size = read_u64_be(data, &mut pos)?;
            attachments.push(AttachmentRef {
                cid,
                filename,
                mime_type,
                size,
            });
        }

        if pos != data.len() {
            return Err(MailError::TrailingBytes {
                count: data.len() - pos,
            });
        }

        Ok(HarmonyMessage {
            version,
            message_type,
            flags,
            timestamp,
            message_id,
            in_reply_to,
            sender_address,
            recipients,
            subject,
            body,
            attachments,
        })
    }
}

// ── Deserialization helpers ────────────────────────────────────────────

fn read_u8(data: &[u8], pos: &mut usize) -> Result<u8, MailError> {
    if *pos >= data.len() {
        return Err(MailError::Truncated { expected: 1 });
    }
    let val = data[*pos];
    *pos += 1;
    Ok(val)
}

fn read_u16_be(data: &[u8], pos: &mut usize) -> Result<u16, MailError> {
    let end = *pos + 2;
    if end > data.len() {
        return Err(MailError::Truncated {
            expected: end - data.len(),
        });
    }
    let val = u16::from_be_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(val)
}

fn read_u32_be(data: &[u8], pos: &mut usize) -> Result<u32, MailError> {
    let end = *pos + 4;
    if end > data.len() {
        return Err(MailError::Truncated {
            expected: end - data.len(),
        });
    }
    let val = u32::from_be_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(val)
}

fn read_u64_be(data: &[u8], pos: &mut usize) -> Result<u64, MailError> {
    let end = *pos + 8;
    if end > data.len() {
        return Err(MailError::Truncated {
            expected: end - data.len(),
        });
    }
    let val = u64::from_be_bytes(data[*pos..end].try_into().unwrap());
    *pos = end;
    Ok(val)
}

fn read_fixed<const N: usize>(data: &[u8], pos: &mut usize) -> Result<[u8; N], MailError> {
    let end = *pos + N;
    if end > data.len() {
        return Err(MailError::Truncated {
            expected: end - data.len(),
        });
    }
    let mut arr = [0u8; N];
    arr.copy_from_slice(&data[*pos..end]);
    *pos = end;
    Ok(arr)
}

fn read_utf8(
    data: &[u8],
    pos: &mut usize,
    len: usize,
    field: &'static str,
) -> Result<String, MailError> {
    let end = *pos + len;
    if end > data.len() {
        return Err(MailError::Truncated {
            expected: end - data.len(),
        });
    }
    let s = std::str::from_utf8(&data[*pos..end])
        .map_err(|_| MailError::InvalidUtf8 { field })?;
    *pos = end;
    Ok(s.to_owned())
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_type_roundtrip() {
        assert_eq!(MailMessageType::from_u8(0x00).unwrap(), MailMessageType::Email);
        assert_eq!(MailMessageType::from_u8(0x01).unwrap(), MailMessageType::Receipt);
        assert_eq!(MailMessageType::from_u8(0x02).unwrap(), MailMessageType::Bounce);
        assert!(MailMessageType::from_u8(0x03).is_err());
        assert!(MailMessageType::from_u8(0xFF).is_err());
    }

    #[test]
    fn recipient_type_roundtrip() {
        assert_eq!(RecipientType::from_u8(0x00).unwrap(), RecipientType::To);
        assert_eq!(RecipientType::from_u8(0x01).unwrap(), RecipientType::Cc);
        assert_eq!(RecipientType::from_u8(0x02).unwrap(), RecipientType::Bcc);
        assert!(RecipientType::from_u8(0x03).is_err());
        assert!(RecipientType::from_u8(0xFF).is_err());
    }

    #[test]
    fn flags_bitfield() {
        // All false
        let f = MessageFlags::new(false, false, false);
        assert_eq!(f.bits(), 0);
        assert!(!f.has_attachments());
        assert!(!f.is_reply());
        assert!(!f.is_forward());

        // Only has_attachments
        let f = MessageFlags::new(true, false, false);
        assert_eq!(f.bits(), 0b001);
        assert!(f.has_attachments());
        assert!(!f.is_reply());
        assert!(!f.is_forward());

        // Only is_reply
        let f = MessageFlags::new(false, true, false);
        assert_eq!(f.bits(), 0b010);
        assert!(!f.has_attachments());
        assert!(f.is_reply());
        assert!(!f.is_forward());

        // Only is_forward
        let f = MessageFlags::new(false, false, true);
        assert_eq!(f.bits(), 0b100);
        assert!(!f.has_attachments());
        assert!(!f.is_reply());
        assert!(f.is_forward());

        // All true
        let f = MessageFlags::new(true, true, true);
        assert_eq!(f.bits(), 0b111);
        assert!(f.has_attachments());
        assert!(f.is_reply());
        assert!(f.is_forward());

        // from_bits roundtrip
        for bits in 0..=0b111u8 {
            let f = MessageFlags::from_bits(bits);
            assert_eq!(f.bits(), bits);
        }
    }

    /// Helper to build a simple email message for tests.
    fn simple_message() -> HarmonyMessage {
        HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Email,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1_709_654_400, // 2024-03-05T12:00:00Z
            message_id: [0x01; MESSAGE_ID_LEN],
            in_reply_to: None,
            sender_address: [0xAA; ADDRESS_HASH_LEN],
            recipients: vec![Recipient {
                address_hash: [0xBB; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            }],
            subject: "Hello".to_string(),
            body: "Hi there".to_string(),
            attachments: vec![],
        }
    }

    #[test]
    fn simple_message_roundtrip() {
        let msg = simple_message();
        let bytes = msg.to_bytes().unwrap();
        let decoded = HarmonyMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn message_with_reply_and_attachments() {
        let msg = HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Email,
            flags: MessageFlags::new(true, true, false),
            timestamp: 1_709_654_400,
            message_id: [0x02; MESSAGE_ID_LEN],
            in_reply_to: Some([0x01; MESSAGE_ID_LEN]),
            sender_address: [0xAA; ADDRESS_HASH_LEN],
            recipients: vec![
                Recipient {
                    address_hash: [0xBB; ADDRESS_HASH_LEN],
                    recipient_type: RecipientType::To,
                },
                Recipient {
                    address_hash: [0xCC; ADDRESS_HASH_LEN],
                    recipient_type: RecipientType::Cc,
                },
            ],
            subject: "Re: Hello".to_string(),
            body: "Thanks for the message".to_string(),
            attachments: vec![AttachmentRef {
                cid: [0xDD; CID_LEN],
                filename: "doc.pdf".to_string(),
                mime_type: "application/pdf".to_string(),
                size: 1024,
            }],
        };

        let bytes = msg.to_bytes().unwrap();
        let decoded = HarmonyMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, msg);
        assert!(decoded.flags.has_attachments());
        assert!(decoded.flags.is_reply());
        assert!(!decoded.flags.is_forward());
        assert_eq!(decoded.in_reply_to, Some([0x01; MESSAGE_ID_LEN]));
        assert_eq!(decoded.recipients.len(), 2);
        assert_eq!(decoded.attachments.len(), 1);
        assert_eq!(decoded.attachments[0].filename, "doc.pdf");
        assert_eq!(decoded.attachments[0].size, 1024);
    }

    #[test]
    fn receipt_message_roundtrip() {
        let msg = HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Receipt,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1_709_654_400,
            message_id: [0x03; MESSAGE_ID_LEN],
            in_reply_to: Some([0x01; MESSAGE_ID_LEN]),
            sender_address: [0xBB; ADDRESS_HASH_LEN],
            recipients: vec![Recipient {
                address_hash: [0xAA; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            }],
            subject: String::new(),
            body: String::new(),
            attachments: vec![],
        };

        let bytes = msg.to_bytes().unwrap();
        let decoded = HarmonyMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, msg);
        assert_eq!(decoded.message_type, MailMessageType::Receipt);
    }

    #[test]
    fn bounce_message_roundtrip() {
        let msg = HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Bounce,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1_709_654_400,
            message_id: [0x04; MESSAGE_ID_LEN],
            in_reply_to: Some([0x01; MESSAGE_ID_LEN]),
            sender_address: [0x00; ADDRESS_HASH_LEN], // system sender
            recipients: vec![Recipient {
                address_hash: [0xAA; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            }],
            subject: "Undeliverable".to_string(),
            body: "Recipient not found on any reachable node".to_string(),
            attachments: vec![],
        };

        let bytes = msg.to_bytes().unwrap();
        let decoded = HarmonyMessage::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, msg);
        assert_eq!(decoded.message_type, MailMessageType::Bounce);
        assert_eq!(decoded.body, "Recipient not found on any reachable node");
    }

    #[test]
    fn simple_message_size() {
        let msg = simple_message();
        let bytes = msg.to_bytes().unwrap();
        // Fixed header: version(1) + type(1) + flags(1) + timestamp(8) + message_id(16)
        //   + in_reply_to_flag(1) + sender_address(16) + recipient_count(1)
        //   = 45
        // 1 recipient: 16 + 1 = 17
        // subject_len(2) + "Hello"(5) = 7
        // body_len(4) + "Hi there"(8) = 12
        // attachment_count(1) = 1
        // Total = 45 + 17 + 7 + 12 + 1 = 82
        assert!(
            bytes.len() < 150,
            "simple message should be under 150 bytes, got {}",
            bytes.len()
        );
    }
}

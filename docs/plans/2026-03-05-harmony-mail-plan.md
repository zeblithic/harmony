# Harmony Mail Gateway Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `harmony-mail`, a single-binary SMTP-to-Harmony bridge gateway that any domain operator can deploy.

**Architecture:** Sans-I/O SMTP state machine + Harmony bridge layer in a new workspace crate. External crates (`mail-parser`, `mail-auth`, `mail-builder`, `lettre`) handle RFC compliance. Harmony core crates handle identity, crypto, content, and messaging. The binary listens on SMTP ports, translates between RFC 5322 and a lean Harmony-native message format, and delivers via Harmony E2EE envelopes.

**Tech Stack:** Rust, tokio, rustls, mail-parser, mail-auth, mail-builder, lettre, instant-acme, harmony-* crates

**Design doc:** `docs/plans/2026-03-05-harmony-mail-design.md`

---

## Task 1: Crate Scaffold

**Files:**
- Create: `crates/harmony-mail/Cargo.toml`
- Create: `crates/harmony-mail/src/lib.rs`
- Create: `crates/harmony-mail/src/main.rs`
- Modify: `Cargo.toml` (workspace root — add member + workspace deps)

**Step 1: Add workspace dependencies for mail crates**

In root `Cargo.toml`, add to `[workspace.members]` and `[workspace.dependencies]`:

```toml
# In [workspace] members array, add:
"crates/harmony-mail",

# In [workspace.dependencies], add:
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
toml = "0.8"
clap = { version = "4", features = ["derive"] }
```

**Step 2: Create `crates/harmony-mail/Cargo.toml`**

```toml
[package]
name = "harmony-mail"
description = "SMTP-to-Harmony bridge gateway for decentralized email"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[[bin]]
name = "harmony-mail"
path = "src/main.rs"

[dependencies]
harmony-crypto.workspace = true
harmony-identity.workspace = true
harmony-content.workspace = true
harmony-reticulum.workspace = true
harmony-zenoh.workspace = true
thiserror.workspace = true
rand.workspace = true
rand_core.workspace = true

[dev-dependencies]
hex.workspace = true
```

Note: We start with only internal deps. External crates (mail-parser, mail-auth, etc.) are added in later tasks when their modules are built, to keep the dependency tree minimal at each step.

**Step 3: Create `crates/harmony-mail/src/lib.rs`**

```rust
pub mod error;
pub mod message;
```

**Step 4: Create `crates/harmony-mail/src/error.rs`**

```rust
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MailError {
    #[error("message too short: {len} bytes, minimum {min}")]
    MessageTooShort { len: usize, min: usize },

    #[error("unsupported message version: {0}")]
    UnsupportedVersion(u8),

    #[error("invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("invalid recipient type: {0}")]
    InvalidRecipientType(u8),

    #[error("subject too long: {len} bytes, maximum {max}")]
    SubjectTooLong { len: usize, max: usize },

    #[error("body too long: {len} bytes, maximum {max}")]
    BodyTooLong { len: usize, max: usize },

    #[error("too many recipients: {count}, maximum {max}")]
    TooManyRecipients { count: usize, max: usize },

    #[error("too many attachments: {count}, maximum {max}")]
    TooManyAttachments { count: usize, max: usize },

    #[error("truncated message: expected {expected} more bytes")]
    Truncated { expected: usize },
}
```

**Step 5: Create `crates/harmony-mail/src/main.rs`**

```rust
fn main() {
    println!("harmony-mail: not yet implemented");
}
```

**Step 6: Create placeholder `crates/harmony-mail/src/message.rs`**

```rust
//! Harmony-native email message format.
//!
//! See docs/plans/2026-03-05-harmony-mail-design.md for wire format spec.
```

**Step 7: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors

**Step 8: Commit**

```bash
git add crates/harmony-mail/ Cargo.toml
git commit -m "feat(mail): scaffold harmony-mail crate"
```

---

## Task 2: HarmonyMessage Wire Format — Types

**Files:**
- Create: `crates/harmony-mail/src/message.rs` (replace placeholder)

**Step 1: Write failing test for message type round-trip**

Add to bottom of `message.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_type_roundtrip() {
        assert_eq!(MailMessageType::from_u8(0x00).unwrap(), MailMessageType::Email);
        assert_eq!(MailMessageType::from_u8(0x01).unwrap(), MailMessageType::Receipt);
        assert_eq!(MailMessageType::from_u8(0x02).unwrap(), MailMessageType::Bounce);
        assert!(MailMessageType::from_u8(0x03).is_err());
    }

    #[test]
    fn recipient_type_roundtrip() {
        assert_eq!(RecipientType::from_u8(0x00).unwrap(), RecipientType::To);
        assert_eq!(RecipientType::from_u8(0x01).unwrap(), RecipientType::Cc);
        assert_eq!(RecipientType::from_u8(0x02).unwrap(), RecipientType::Bcc);
        assert!(RecipientType::from_u8(0x03).is_err());
    }

    #[test]
    fn flags_bitfield() {
        let flags = MessageFlags::from_bits(0b0000_0111);
        assert!(flags.has_attachments());
        assert!(flags.is_reply());
        assert!(flags.is_forward());

        let flags = MessageFlags::from_bits(0b0000_0000);
        assert!(!flags.has_attachments());
        assert!(!flags.is_reply());
        assert!(!flags.is_forward());
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail message::tests --no-run 2>&1 | head -20`
Expected: compilation errors (types don't exist yet)

**Step 3: Implement types**

```rust
//! Harmony-native email message format.
//!
//! Binary wire format optimized for E2EE transport over Harmony.
//! See docs/plans/2026-03-05-harmony-mail-design.md for full spec.

use crate::error::MailError;

/// Current message format version.
pub const VERSION: u8 = 0x01;

/// Maximum subject length in bytes.
pub const MAX_SUBJECT_LEN: usize = 998; // RFC 5322 line length limit, generous

/// Maximum body length in bytes (16 MB).
pub const MAX_BODY_LEN: usize = 16 * 1024 * 1024;

/// Maximum recipients per message.
pub const MAX_RECIPIENTS: usize = 100;

/// Maximum attachments per message.
pub const MAX_ATTACHMENTS: usize = 100;

/// Address hash length (matches harmony-identity).
pub const ADDRESS_HASH_LEN: usize = 16;

/// Content ID length (BLAKE3 hash).
pub const CID_LEN: usize = 32;

/// Message ID length.
pub const MESSAGE_ID_LEN: usize = 16;

/// Message type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MailMessageType {
    /// Normal email message.
    Email = 0x00,
    /// Delivery receipt.
    Receipt = 0x01,
    /// Bounce notification.
    Bounce = 0x02,
}

impl MailMessageType {
    pub fn from_u8(val: u8) -> Result<Self, MailError> {
        match val {
            0x00 => Ok(Self::Email),
            0x01 => Ok(Self::Receipt),
            0x02 => Ok(Self::Bounce),
            other => Err(MailError::InvalidMessageType(other)),
        }
    }
}

/// Recipient type discriminant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecipientType {
    To = 0x00,
    Cc = 0x01,
    Bcc = 0x02,
}

impl RecipientType {
    pub fn from_u8(val: u8) -> Result<Self, MailError> {
        match val {
            0x00 => Ok(Self::To),
            0x01 => Ok(Self::Cc),
            0x02 => Ok(Self::Bcc),
            other => Err(MailError::InvalidRecipientType(other)),
        }
    }
}

/// Message flags bitfield.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MessageFlags(u8);

impl MessageFlags {
    pub fn from_bits(bits: u8) -> Self {
        Self(bits)
    }

    pub fn bits(self) -> u8 {
        self.0
    }

    pub fn has_attachments(self) -> bool {
        self.0 & 0b0000_0001 != 0
    }

    pub fn is_reply(self) -> bool {
        self.0 & 0b0000_0010 != 0
    }

    pub fn is_forward(self) -> bool {
        self.0 & 0b0000_0100 != 0
    }

    pub fn new(has_attachments: bool, is_reply: bool, is_forward: bool) -> Self {
        let mut bits = 0u8;
        if has_attachments {
            bits |= 0b0000_0001;
        }
        if is_reply {
            bits |= 0b0000_0010;
        }
        if is_forward {
            bits |= 0b0000_0100;
        }
        Self(bits)
    }
}

/// A single recipient in the message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recipient {
    pub address_hash: [u8; ADDRESS_HASH_LEN],
    pub recipient_type: RecipientType,
}

/// Reference to an attachment stored as a Harmony content blob.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttachmentRef {
    pub cid: [u8; CID_LEN],
    pub filename: String,
    pub mime_type: String,
    pub size: u64,
}

/// A Harmony-native email message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HarmonyMessage {
    pub version: u8,
    pub message_type: MailMessageType,
    pub flags: MessageFlags,
    pub timestamp: u64,
    pub message_id: [u8; MESSAGE_ID_LEN],
    pub in_reply_to: Option<[u8; MESSAGE_ID_LEN]>,
    pub sender_address: [u8; ADDRESS_HASH_LEN],
    pub recipients: Vec<Recipient>,
    pub subject: String,
    pub body: String,
    pub attachments: Vec<AttachmentRef>,
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-mail`
Expected: all 3 tests pass

**Step 5: Commit**

```bash
git add crates/harmony-mail/src/message.rs
git commit -m "feat(mail): add HarmonyMessage types and wire format enums"
```

---

## Task 3: HarmonyMessage Serialization

**Files:**
- Modify: `crates/harmony-mail/src/message.rs`

**Step 1: Write failing test for serialize/deserialize round-trip**

Add to the `tests` module:

```rust
#[test]
fn simple_message_roundtrip() {
    let msg = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Email,
        flags: MessageFlags::new(false, false, false),
        timestamp: 1709654400000,
        message_id: [0xAA; MESSAGE_ID_LEN],
        in_reply_to: None,
        sender_address: [0xBB; ADDRESS_HASH_LEN],
        recipients: vec![Recipient {
            address_hash: [0xCC; ADDRESS_HASH_LEN],
            recipient_type: RecipientType::To,
        }],
        subject: "Meeting at 3pm".to_string(),
        body: "Hey, meeting at 3pm tomorrow?".to_string(),
        attachments: vec![],
    };

    let bytes = msg.to_bytes();
    let parsed = HarmonyMessage::from_bytes(&bytes).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn message_with_reply_and_attachments() {
    let msg = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Email,
        flags: MessageFlags::new(true, true, false),
        timestamp: 1709654400000,
        message_id: [0x11; MESSAGE_ID_LEN],
        in_reply_to: Some([0x22; MESSAGE_ID_LEN]),
        sender_address: [0x33; ADDRESS_HASH_LEN],
        recipients: vec![
            Recipient {
                address_hash: [0x44; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            },
            Recipient {
                address_hash: [0x55; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::Cc,
            },
        ],
        subject: "Re: Project update".to_string(),
        body: "Looks good, thanks!".to_string(),
        attachments: vec![AttachmentRef {
            cid: [0xFF; CID_LEN],
            filename: "report.pdf".to_string(),
            mime_type: "application/pdf".to_string(),
            size: 1024 * 1024,
        }],
    };

    let bytes = msg.to_bytes();
    let parsed = HarmonyMessage::from_bytes(&bytes).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn receipt_message_roundtrip() {
    let msg = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Receipt,
        flags: MessageFlags::new(false, false, false),
        timestamp: 1709654400000,
        message_id: [0xDD; MESSAGE_ID_LEN],
        in_reply_to: Some([0xEE; MESSAGE_ID_LEN]),
        sender_address: [0xFF; ADDRESS_HASH_LEN],
        recipients: vec![Recipient {
            address_hash: [0x00; ADDRESS_HASH_LEN],
            recipient_type: RecipientType::To,
        }],
        subject: String::new(),
        body: String::new(),
        attachments: vec![],
    };

    let bytes = msg.to_bytes();
    let parsed = HarmonyMessage::from_bytes(&bytes).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn bounce_message_roundtrip() {
    let msg = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Bounce,
        flags: MessageFlags::new(false, false, false),
        timestamp: 1709654400000,
        message_id: [0x99; MESSAGE_ID_LEN],
        in_reply_to: Some([0x88; MESSAGE_ID_LEN]),
        sender_address: [0x77; ADDRESS_HASH_LEN],
        recipients: vec![Recipient {
            address_hash: [0x66; ADDRESS_HASH_LEN],
            recipient_type: RecipientType::To,
        }],
        subject: String::new(),
        body: "550 5.1.1 Recipient not found".to_string(),
        attachments: vec![],
    };

    let bytes = msg.to_bytes();
    let parsed = HarmonyMessage::from_bytes(&bytes).unwrap();
    assert_eq!(msg, parsed);
}

#[test]
fn simple_message_size() {
    let msg = HarmonyMessage {
        version: VERSION,
        message_type: MailMessageType::Email,
        flags: MessageFlags::new(false, false, false),
        timestamp: 1709654400000,
        message_id: [0xAA; MESSAGE_ID_LEN],
        in_reply_to: None,
        sender_address: [0xBB; ADDRESS_HASH_LEN],
        recipients: vec![Recipient {
            address_hash: [0xCC; ADDRESS_HASH_LEN],
            recipient_type: RecipientType::To,
        }],
        subject: "Meeting at 3pm".to_string(),
        body: "Hey, meeting at 3pm tomorrow?".to_string(),
        attachments: vec![],
    };

    let bytes = msg.to_bytes();
    // Should be under 150 bytes for a simple message
    assert!(bytes.len() < 150, "simple message was {} bytes", bytes.len());
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail 2>&1 | head -5`
Expected: compilation error (to_bytes/from_bytes don't exist)

**Step 3: Implement serialization**

Add `impl HarmonyMessage` block with `to_bytes` and `from_bytes` methods. The wire format follows the design doc exactly:

```rust
impl HarmonyMessage {
    /// Fixed header size: version(1) + type(1) + flags(1) + timestamp(8) +
    /// message_id(16) + in_reply_to_flag(1) + sender_address(16) + recipient_count(1) = 45
    /// Plus optional in_reply_to(16).
    const FIXED_HEADER_MIN: usize = 1 + 1 + 1 + 8 + MESSAGE_ID_LEN + 1 + ADDRESS_HASH_LEN + 1;

    /// Serialize to binary wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(128);

        buf.push(self.version);
        buf.push(self.message_type as u8);
        buf.push(self.flags.bits());
        buf.extend_from_slice(&self.timestamp.to_be_bytes());
        buf.extend_from_slice(&self.message_id);

        // in_reply_to: 1 byte flag + optional 16 bytes
        match &self.in_reply_to {
            Some(id) => {
                buf.push(0x01);
                buf.extend_from_slice(id);
            }
            None => {
                buf.push(0x00);
            }
        }

        buf.extend_from_slice(&self.sender_address);

        // Recipients
        buf.push(self.recipients.len() as u8);
        for r in &self.recipients {
            buf.extend_from_slice(&r.address_hash);
            buf.push(r.recipient_type as u8);
        }

        // Subject
        let subject_bytes = self.subject.as_bytes();
        buf.extend_from_slice(&(subject_bytes.len() as u16).to_be_bytes());
        buf.extend_from_slice(subject_bytes);

        // Body
        let body_bytes = self.body.as_bytes();
        buf.extend_from_slice(&(body_bytes.len() as u32).to_be_bytes());
        buf.extend_from_slice(body_bytes);

        // Attachments
        buf.push(self.attachments.len() as u8);
        for a in &self.attachments {
            buf.extend_from_slice(&a.cid);
            let fname = a.filename.as_bytes();
            buf.push(fname.len() as u8);
            buf.extend_from_slice(fname);
            let mime = a.mime_type.as_bytes();
            buf.push(mime.len() as u8);
            buf.extend_from_slice(mime);
            buf.extend_from_slice(&a.size.to_be_bytes());
        }

        buf
    }

    /// Deserialize from binary wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, MailError> {
        if data.len() < Self::FIXED_HEADER_MIN {
            return Err(MailError::MessageTooShort {
                len: data.len(),
                min: Self::FIXED_HEADER_MIN,
            });
        }

        let mut pos = 0;

        let version = data[pos];
        pos += 1;
        if version != VERSION {
            return Err(MailError::UnsupportedVersion(version));
        }

        let message_type = MailMessageType::from_u8(data[pos])?;
        pos += 1;

        let flags = MessageFlags::from_bits(data[pos]);
        pos += 1;

        let timestamp = u64::from_be_bytes(
            data[pos..pos + 8]
                .try_into()
                .map_err(|_| MailError::Truncated { expected: 8 })?,
        );
        pos += 8;

        let mut message_id = [0u8; MESSAGE_ID_LEN];
        if pos + MESSAGE_ID_LEN > data.len() {
            return Err(MailError::Truncated {
                expected: MESSAGE_ID_LEN,
            });
        }
        message_id.copy_from_slice(&data[pos..pos + MESSAGE_ID_LEN]);
        pos += MESSAGE_ID_LEN;

        // in_reply_to flag
        if pos >= data.len() {
            return Err(MailError::Truncated { expected: 1 });
        }
        let has_reply = data[pos];
        pos += 1;
        let in_reply_to = if has_reply == 0x01 {
            if pos + MESSAGE_ID_LEN > data.len() {
                return Err(MailError::Truncated {
                    expected: MESSAGE_ID_LEN,
                });
            }
            let mut id = [0u8; MESSAGE_ID_LEN];
            id.copy_from_slice(&data[pos..pos + MESSAGE_ID_LEN]);
            pos += MESSAGE_ID_LEN;
            Some(id)
        } else {
            None
        };

        // Sender address
        if pos + ADDRESS_HASH_LEN > data.len() {
            return Err(MailError::Truncated {
                expected: ADDRESS_HASH_LEN,
            });
        }
        let mut sender_address = [0u8; ADDRESS_HASH_LEN];
        sender_address.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LEN]);
        pos += ADDRESS_HASH_LEN;

        // Recipients
        if pos >= data.len() {
            return Err(MailError::Truncated { expected: 1 });
        }
        let recipient_count = data[pos] as usize;
        pos += 1;
        if recipient_count > MAX_RECIPIENTS {
            return Err(MailError::TooManyRecipients {
                count: recipient_count,
                max: MAX_RECIPIENTS,
            });
        }

        let mut recipients = Vec::with_capacity(recipient_count);
        for _ in 0..recipient_count {
            if pos + ADDRESS_HASH_LEN + 1 > data.len() {
                return Err(MailError::Truncated {
                    expected: ADDRESS_HASH_LEN + 1,
                });
            }
            let mut address_hash = [0u8; ADDRESS_HASH_LEN];
            address_hash.copy_from_slice(&data[pos..pos + ADDRESS_HASH_LEN]);
            pos += ADDRESS_HASH_LEN;
            let recipient_type = RecipientType::from_u8(data[pos])?;
            pos += 1;
            recipients.push(Recipient {
                address_hash,
                recipient_type,
            });
        }

        // Subject
        if pos + 2 > data.len() {
            return Err(MailError::Truncated { expected: 2 });
        }
        let subject_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        pos += 2;
        if subject_len > MAX_SUBJECT_LEN {
            return Err(MailError::SubjectTooLong {
                len: subject_len,
                max: MAX_SUBJECT_LEN,
            });
        }
        if pos + subject_len > data.len() {
            return Err(MailError::Truncated {
                expected: subject_len,
            });
        }
        let subject = String::from_utf8_lossy(&data[pos..pos + subject_len]).into_owned();
        pos += subject_len;

        // Body
        if pos + 4 > data.len() {
            return Err(MailError::Truncated { expected: 4 });
        }
        let body_len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
            as usize;
        pos += 4;
        if body_len > MAX_BODY_LEN {
            return Err(MailError::BodyTooLong {
                len: body_len,
                max: MAX_BODY_LEN,
            });
        }
        if pos + body_len > data.len() {
            return Err(MailError::Truncated {
                expected: body_len,
            });
        }
        let body = String::from_utf8_lossy(&data[pos..pos + body_len]).into_owned();
        pos += body_len;

        // Attachments
        if pos >= data.len() {
            return Err(MailError::Truncated { expected: 1 });
        }
        let attachment_count = data[pos] as usize;
        pos += 1;
        if attachment_count > MAX_ATTACHMENTS {
            return Err(MailError::TooManyAttachments {
                count: attachment_count,
                max: MAX_ATTACHMENTS,
            });
        }

        let mut attachments = Vec::with_capacity(attachment_count);
        for _ in 0..attachment_count {
            if pos + CID_LEN > data.len() {
                return Err(MailError::Truncated { expected: CID_LEN });
            }
            let mut cid = [0u8; CID_LEN];
            cid.copy_from_slice(&data[pos..pos + CID_LEN]);
            pos += CID_LEN;

            if pos >= data.len() {
                return Err(MailError::Truncated { expected: 1 });
            }
            let fname_len = data[pos] as usize;
            pos += 1;
            if pos + fname_len > data.len() {
                return Err(MailError::Truncated {
                    expected: fname_len,
                });
            }
            let filename = String::from_utf8_lossy(&data[pos..pos + fname_len]).into_owned();
            pos += fname_len;

            if pos >= data.len() {
                return Err(MailError::Truncated { expected: 1 });
            }
            let mime_len = data[pos] as usize;
            pos += 1;
            if pos + mime_len > data.len() {
                return Err(MailError::Truncated {
                    expected: mime_len,
                });
            }
            let mime_type = String::from_utf8_lossy(&data[pos..pos + mime_len]).into_owned();
            pos += mime_len;

            if pos + 8 > data.len() {
                return Err(MailError::Truncated { expected: 8 });
            }
            let size = u64::from_be_bytes(
                data[pos..pos + 8]
                    .try_into()
                    .map_err(|_| MailError::Truncated { expected: 8 })?,
            );
            pos += 8;

            attachments.push(AttachmentRef {
                cid,
                filename,
                mime_type,
                size,
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
```

**Step 4: Run tests**

Run: `cargo test -p harmony-mail`
Expected: all 8 tests pass (3 from task 2 + 5 new)

**Step 5: Commit**

```bash
git add crates/harmony-mail/src/message.rs
git commit -m "feat(mail): implement HarmonyMessage binary serialization"
```

---

## Task 4: Address Resolution

**Files:**
- Create: `crates/harmony-mail/src/address.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

Implements the three-tier address resolution: hex identity, named identity (`name_namespace`), and vanity alias.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hex_address() {
        let result = parse_local_part("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6");
        assert_eq!(
            result,
            LocalPart::Hex([
                0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8, 0xc9, 0xd0, 0xe1, 0xf2, 0xa3,
                0xb4, 0xc5, 0xd6
            ])
        );
    }

    #[test]
    fn parse_named_address() {
        let result = parse_local_part("jake_z");
        assert_eq!(
            result,
            LocalPart::Named {
                name: "jake".to_string(),
                namespace: "z".to_string(),
            }
        );
    }

    #[test]
    fn parse_named_address_longer_namespace() {
        let result = parse_local_part("victoria_vrk");
        assert_eq!(
            result,
            LocalPart::Named {
                name: "victoria".to_string(),
                namespace: "vrk".to_string(),
            }
        );
    }

    #[test]
    fn parse_vanity_alias() {
        let result = parse_local_part("support");
        assert_eq!(
            result,
            LocalPart::Alias("support".to_string())
        );
    }

    #[test]
    fn parse_vanity_alias_with_dots() {
        let result = parse_local_part("first.last");
        assert_eq!(
            result,
            LocalPart::Alias("first.last".to_string())
        );
    }

    #[test]
    fn hex_address_case_insensitive() {
        let lower = parse_local_part("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6");
        let upper = parse_local_part("A1B2C3D4E5F6A7B8C9D0E1F2A3B4C5D6");
        assert_eq!(lower, upper);
    }

    #[test]
    fn hex_address_wrong_length_is_alias() {
        // 30 hex chars (15 bytes) — not valid as 16-byte address, treat as alias
        let result = parse_local_part("a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5");
        assert!(matches!(result, LocalPart::Alias(_)));
    }

    #[test]
    fn format_outbound_from_named() {
        let addr = format_outbound_sender(
            Some(("jake", "z")),
            &[0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8,
              0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6],
            "q8.fyi",
        );
        assert_eq!(addr, "jake_z@q8.fyi");
    }

    #[test]
    fn format_outbound_from_hex_uses_prefix() {
        let addr = format_outbound_sender(
            None,
            &[0xa1, 0xb2, 0xc3, 0xd4, 0xe5, 0xf6, 0xa7, 0xb8,
              0xc9, 0xd0, 0xe1, 0xf2, 0xa3, 0xb4, 0xc5, 0xd6],
            "q8.fyi",
        );
        assert_eq!(addr, "user-a1b2c3d4@q8.fyi");
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail address::tests --no-run 2>&1 | head -5`
Expected: compilation error

**Step 3: Implement address parsing**

```rust
//! Email address parsing and formatting for the Harmony mail gateway.
//!
//! Three-tier address resolution:
//! 1. Hex identity: 32 hex chars = 16-byte address hash
//! 2. Named identity: name_namespace format
//! 3. Vanity alias: anything else (operator-defined)

use crate::message::ADDRESS_HASH_LEN;

/// Parsed local part of an email address (the part before @).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocalPart {
    /// 32 hex characters decoded to a 16-byte Harmony address hash.
    Hex([u8; ADDRESS_HASH_LEN]),
    /// name_namespace format (e.g., "jake_z" -> name="jake", namespace="z").
    Named { name: String, namespace: String },
    /// Anything else — resolved via operator-defined alias table.
    Alias(String),
}

/// Parse an email local part into one of the three address types.
///
/// Resolution order:
/// 1. If exactly 32 hex characters -> Hex address
/// 2. If contains exactly one underscore with non-empty parts -> Named
/// 3. Otherwise -> Alias
pub fn parse_local_part(local: &str) -> LocalPart {
    // Try hex first: exactly 32 hex chars = 16 bytes
    let normalized = local.to_ascii_lowercase();
    if normalized.len() == ADDRESS_HASH_LEN * 2
        && normalized.chars().all(|c| c.is_ascii_hexdigit())
    {
        let mut bytes = [0u8; ADDRESS_HASH_LEN];
        for i in 0..ADDRESS_HASH_LEN {
            bytes[i] = u8::from_str_radix(&normalized[i * 2..i * 2 + 2], 16).unwrap();
        }
        return LocalPart::Hex(bytes);
    }

    // Try named: exactly one underscore with non-empty parts on both sides
    // Last underscore is the separator (name can contain underscores, namespace cannot)
    if let Some(sep_pos) = normalized.rfind('_') {
        let name_part = &normalized[..sep_pos];
        let namespace_part = &normalized[sep_pos + 1..];
        if !name_part.is_empty() && !namespace_part.is_empty() && !namespace_part.contains('_') {
            return LocalPart::Named {
                name: name_part.to_string(),
                namespace: namespace_part.to_string(),
            };
        }
    }

    // Fallback: vanity alias
    LocalPart::Alias(normalized)
}

/// Format an outbound From: address for SMTP.
///
/// Uses human-readable name if registered, otherwise truncated hex with "user-" prefix
/// to avoid SpamAssassin FROM_LOCAL_HEX penalty.
pub fn format_outbound_sender(
    registered_name: Option<(&str, &str)>,
    address_hash: &[u8; ADDRESS_HASH_LEN],
    domain: &str,
) -> String {
    match registered_name {
        Some((name, namespace)) => format!("{name}_{namespace}@{domain}"),
        None => {
            // Truncated hex with prefix to avoid spam scoring
            let short_hex: String = address_hash[..4]
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect();
            format!("user-{short_hex}@{domain}")
        }
    }
}
```

**Step 4: Update lib.rs**

```rust
pub mod address;
pub mod error;
pub mod message;
```

**Step 5: Run tests**

Run: `cargo test -p harmony-mail`
Expected: all tests pass

**Step 6: Commit**

```bash
git add crates/harmony-mail/src/address.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add three-tier email address parsing"
```

---

## Task 5: SMTP State Machine — Types and Core Transitions

**Files:**
- Create: `crates/harmony-mail/src/smtp.rs`
- Modify: `crates/harmony-mail/src/lib.rs`
- Modify: `crates/harmony-mail/src/error.rs`

This is the sans-I/O SMTP state machine. Pure `(State, Event) -> (State, Vec<Action>)`.

**Step 1: Write failing tests for the SMTP greeting and EHLO flow**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::net::Ipv4Addr;

    fn test_config() -> SmtpConfig {
        SmtpConfig {
            domain: "q8.fyi".to_string(),
            mx_host: "mail.q8.fyi".to_string(),
            max_message_size: 25 * 1024 * 1024,
            max_recipients: 100,
        }
    }

    #[test]
    fn connected_sends_greeting() {
        let config = test_config();
        let mut session = SmtpSession::new(config);

        let actions = session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: false,
        });

        assert_eq!(session.state, SmtpState::GreetingSent);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], SmtpAction::SendResponse(220, msg) if msg.contains("q8.fyi")));
    }

    #[test]
    fn ehlo_after_greeting() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: false,
        });

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));

        assert_eq!(session.state, SmtpState::Ready);
        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(250, _))));
    }

    #[test]
    fn mail_from_after_ready() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: true,
        });
        session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "sender@example.com".to_string(),
        }));

        assert_eq!(session.state, SmtpState::MailFromReceived);
        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(250, _))));
        assert!(actions.iter().any(|a| matches!(a, SmtpAction::CheckSpf { .. })));
    }

    #[test]
    fn rcpt_to_triggers_resolve() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: true,
        });
        session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "sender@example.com".to_string(),
        }));

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "jake_z@q8.fyi".to_string(),
        }));

        assert!(actions.iter().any(|a| matches!(a, SmtpAction::ResolveHarmonyAddress { .. })));
    }

    #[test]
    fn rcpt_to_unknown_recipient_rejects() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: true,
        });
        session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "sender@example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "jake_z@q8.fyi".to_string(),
        }));

        // Harmony resolver returns: not found
        let actions = session.handle(SmtpEvent::HarmonyResolved {
            local_part: "jake_z".to_string(),
            identity: None,
        });

        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(550, _))));
    }

    #[test]
    fn data_command_accepted_after_rcpt() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: true,
        });
        session.handle(SmtpEvent::Command(SmtpCommand::Ehlo {
            domain: "sender.example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::MailFrom {
            address: "sender@example.com".to_string(),
        }));
        session.handle(SmtpEvent::Command(SmtpCommand::RcptTo {
            address: "jake_z@q8.fyi".to_string(),
        }));
        session.handle(SmtpEvent::HarmonyResolved {
            local_part: "jake_z".to_string(),
            identity: Some([0xAA; 16]),
        });

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Data));

        assert_eq!(session.state, SmtpState::DataReceiving);
        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(354, _))));
    }

    #[test]
    fn quit_from_any_state() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: false,
        });

        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Quit));

        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(221, _))));
        assert!(actions.iter().any(|a| matches!(a, SmtpAction::Close)));
    }

    #[test]
    fn command_out_of_order_rejected() {
        let config = test_config();
        let mut session = SmtpSession::new(config);
        session.handle(SmtpEvent::Connected {
            peer_ip: IpAddr::V4(Ipv4Addr::new(192, 168, 1, 1)),
            tls: false,
        });

        // DATA before MAIL FROM -> error
        let actions = session.handle(SmtpEvent::Command(SmtpCommand::Data));

        assert!(actions.iter().any(|a| matches!(a, SmtpAction::SendResponse(503, _))));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail smtp::tests --no-run 2>&1 | head -5`
Expected: compilation error

**Step 3: Implement SMTP state machine**

Create `crates/harmony-mail/src/smtp.rs` with the state machine types and `handle` method. The full implementation should include:

- `SmtpState` enum matching the design doc state diagram
- `SmtpEvent` enum for all inputs
- `SmtpAction` enum for all outputs
- `SmtpCommand` enum for parsed SMTP commands
- `SmtpConfig` for gateway configuration
- `SmtpSession` struct holding state, envelope data, and resolved recipients
- `SmtpSession::handle(&mut self, event: SmtpEvent) -> Vec<SmtpAction>` as the pure transition function

Key states: `Connected`, `GreetingSent`, `Ready`, `MailFromReceived`, `RcptToReceived`, `DataReceiving`, `MessageComplete`, `Closed`.

The session tracks: `peer_ip`, `tls` status, `ehlo_domain`, `mail_from`, `recipients` (vec of resolved address hashes), `data_buffer`.

Use `crate::address::parse_local_part` for RCPT TO resolution.

**Step 4: Run tests**

Run: `cargo test -p harmony-mail`
Expected: all tests pass

**Step 5: Commit**

```bash
git add crates/harmony-mail/src/smtp.rs crates/harmony-mail/src/lib.rs crates/harmony-mail/src/error.rs
git commit -m "feat(mail): implement sans-I/O SMTP state machine"
```

---

## Task 6: SMTP Command Parser

**Files:**
- Create: `crates/harmony-mail/src/smtp_parse.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

Parses raw SMTP command lines (bytes) into `SmtpCommand` variants.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ehlo() {
        let cmd = parse_command(b"EHLO sender.example.com\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "sender.example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_helo() {
        let cmd = parse_command(b"HELO sender.example.com\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "sender.example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_mail_from() {
        let cmd = parse_command(b"MAIL FROM:<sender@example.com>\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: "sender@example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_mail_from_with_size() {
        let cmd = parse_command(b"MAIL FROM:<sender@example.com> SIZE=1024\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::MailFrom {
                address: "sender@example.com".to_string()
            }
        );
    }

    #[test]
    fn parse_rcpt_to() {
        let cmd = parse_command(b"RCPT TO:<jake_z@q8.fyi>\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::RcptTo {
                address: "jake_z@q8.fyi".to_string()
            }
        );
    }

    #[test]
    fn parse_data() {
        let cmd = parse_command(b"DATA\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Data);
    }

    #[test]
    fn parse_quit() {
        let cmd = parse_command(b"QUIT\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Quit);
    }

    #[test]
    fn parse_rset() {
        let cmd = parse_command(b"RSET\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Rset);
    }

    #[test]
    fn parse_noop() {
        let cmd = parse_command(b"NOOP\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::Noop);
    }

    #[test]
    fn parse_starttls() {
        let cmd = parse_command(b"STARTTLS\r\n").unwrap();
        assert_eq!(cmd, SmtpCommand::StartTls);
    }

    #[test]
    fn parse_case_insensitive() {
        let cmd = parse_command(b"ehlo EXAMPLE.COM\r\n").unwrap();
        assert_eq!(
            cmd,
            SmtpCommand::Ehlo {
                domain: "EXAMPLE.COM".to_string()
            }
        );
    }

    #[test]
    fn parse_unknown_command() {
        let result = parse_command(b"VRFY user\r\n");
        assert!(result.is_err());
    }
}
```

**Step 2: Implement parser**

Parse raw byte lines into `SmtpCommand`. Handle case-insensitive command verbs, angle-bracket address extraction for MAIL FROM/RCPT TO, ESMTP parameter stripping.

**Step 3: Run tests and commit**

Run: `cargo test -p harmony-mail`

```bash
git add crates/harmony-mail/src/smtp_parse.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add SMTP command line parser"
```

---

## Task 7: Spam Scoring Engine

**Files:**
- Create: `crates/harmony-mail/src/spam.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

Pure scoring function: takes authentication results + heuristic signals, returns a numeric score and verdict.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_message_passes() {
        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::Pass,
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: false,
            url_count: 2,
            empty_subject: false,
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: false,
        };

        let verdict = score(&signals);
        assert!(verdict.score <= 0);
        assert_eq!(verdict.action, SpamAction::Deliver);
    }

    #[test]
    fn dnsbl_listed_rejects() {
        let signals = SpamSignals {
            dnsbl_listed: true,
            fcrdns_pass: true,
            spf_result: SpfResult::Pass,
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: false,
        };

        let verdict = score(&signals);
        assert!(verdict.score >= 5);
        assert_eq!(verdict.action, SpamAction::Reject);
    }

    #[test]
    fn harmony_sender_gets_trust_bonus() {
        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: false, // -3 for FCrDNS fail
            spf_result: SpfResult::None,  // +1
            dkim_result: DkimResult::Missing, // +1
            dmarc_result: DmarcResult::None,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: true, // -3 trust bonus
            gateway_trust: Some(0.9),   // -2 high trust
            first_contact: false,
        };

        let verdict = score(&signals);
        assert!(verdict.score <= 0, "score was {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Deliver);
    }

    #[test]
    fn executable_attachment_rejects() {
        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::Pass,
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: true, // +5
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: false,
        };

        let verdict = score(&signals);
        assert!(verdict.score >= 5);
        assert_eq!(verdict.action, SpamAction::Reject);
    }

    #[test]
    fn borderline_message_delivers_with_header() {
        let signals = SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::SoftFail, // +1
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: true, // +1
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: true, // +1
        };

        let verdict = score(&signals);
        assert!(verdict.score >= 1 && verdict.score <= 4);
        assert_eq!(verdict.action, SpamAction::DeliverWithScore);
    }
}
```

**Step 2: Implement scoring engine**

Implement `SpamSignals`, `SpfResult`, `DkimResult`, `DmarcResult`, `SpamAction`, `SpamVerdict`, and the `score()` function matching the design doc's scoring table.

**Step 3: Run tests and commit**

Run: `cargo test -p harmony-mail`

```bash
git add crates/harmony-mail/src/spam.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add layered spam scoring engine"
```

---

## Task 8: Name Registration Types

**Files:**
- Create: `crates/harmony-mail/src/registry.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

The dual-signed name registration struct and verification logic.

**Step 1: Write failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    #[test]
    fn name_registration_sign_verify_roundtrip() {
        let mut rng = OsRng;
        let user = PrivateIdentity::generate(&mut rng);
        let gateway = PrivateIdentity::generate(&mut rng);

        let reg = NameRegistration::new(
            "jake",
            "z",
            "q8.fyi",
            user.public_identity().clone(),
            1709654400,
            &user,
            &gateway,
        );

        assert!(reg.verify_user_signature());
        assert!(reg.verify_domain_signature(gateway.public_identity()));
    }

    #[test]
    fn tampered_name_fails_verification() {
        let mut rng = OsRng;
        let user = PrivateIdentity::generate(&mut rng);
        let gateway = PrivateIdentity::generate(&mut rng);

        let mut reg = NameRegistration::new(
            "jake",
            "z",
            "q8.fyi",
            user.public_identity().clone(),
            1709654400,
            &user,
            &gateway,
        );

        reg.name = "evil".to_string();
        assert!(!reg.verify_user_signature());
    }

    #[test]
    fn serialization_roundtrip() {
        let mut rng = OsRng;
        let user = PrivateIdentity::generate(&mut rng);
        let gateway = PrivateIdentity::generate(&mut rng);

        let reg = NameRegistration::new(
            "jake",
            "z",
            "q8.fyi",
            user.public_identity().clone(),
            1709654400,
            &user,
            &gateway,
        );

        let bytes = reg.to_bytes();
        let parsed = NameRegistration::from_bytes(&bytes).unwrap();
        assert_eq!(reg.name, parsed.name);
        assert_eq!(reg.namespace, parsed.namespace);
        assert_eq!(reg.domain, parsed.domain);
        assert!(parsed.verify_user_signature());
        assert!(parsed.verify_domain_signature(gateway.public_identity()));
    }
}
```

**Step 2: Implement NameRegistration**

Struct with `name`, `namespace`, `domain`, `identity`, `registered_at`, `user_signature`, `domain_signature`. Sign/verify methods using Ed25519 from harmony-identity. Binary serialization for announce payloads.

**Step 3: Run tests and commit**

Run: `cargo test -p harmony-mail`

```bash
git add crates/harmony-mail/src/registry.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add dual-signed name registration"
```

---

## Task 9: Gateway Trust Types

**Files:**
- Create: `crates/harmony-mail/src/trust.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

The TrustReport and TrustMetrics types, plus aggregation logic.

**Step 1: Write failing tests**

Test that TrustReports can be signed/verified, and that weighted median aggregation works correctly with multiple reporters.

**Step 2: Implement trust types**

`GatewayAnnounce`, `TrustReport`, `TrustMetrics`, `TrustAggregator` with weighted median computation.

**Step 3: Run tests and commit**

```bash
git add crates/harmony-mail/src/trust.rs crates/harmony-mail/src/lib.rs
git commit -m "feat(mail): add cross-gateway trust coordination types"
```

---

## Task 10: Configuration and CLI

**Files:**
- Modify: `crates/harmony-mail/Cargo.toml` (add serde, toml, clap)
- Create: `crates/harmony-mail/src/config.rs`
- Modify: `crates/harmony-mail/src/main.rs`
- Modify: `crates/harmony-mail/src/lib.rs`

**Step 1: Write failing tests for config parsing**

Test that a TOML string deserializes to the correct `Config` struct matching the design doc's config format.

**Step 2: Implement Config struct and CLI**

`Config` with serde derives, `clap` CLI with subcommands: `init`, `run`, `user add`, `alias add`, `tls`, `warm`.

**Step 3: Run tests and commit**

```bash
git add crates/harmony-mail/
git commit -m "feat(mail): add config parsing and CLI scaffold"
```

---

## Checkpoint: Core Library Complete

At this point, run the full workspace test suite:

```bash
cargo test --workspace
cargo clippy --workspace
cargo fmt --all -- --check
```

All existing tests (365+) must still pass. The harmony-mail crate should have ~40+ tests covering message format, address parsing, SMTP state machine, spam scoring, name registration, trust types, and config parsing.

**This is a good point to create a bead, claim it, push, and get review** before proceeding to the networking/I/O tasks below.

---

## Task 11: SMTP Line Reader (Tokio I/O Layer)

**Files:**
- Create: `crates/harmony-mail/src/io.rs`
- Modify: `crates/harmony-mail/Cargo.toml` (add tokio)

The async I/O layer that reads SMTP command lines from a TCP stream and feeds them to the sans-I/O state machine.

Implements: line-buffered reading (CRLF-delimited), DATA dot-stuffing, connection timeouts, message size limits. Uses `tokio::io::AsyncBufRead`.

**Step 1: Write tests using in-memory tokio streams**

Test line extraction, dot-stuffing removal, timeout behavior.

**Step 2: Implement**

**Step 3: Run tests and commit**

---

## Task 12: SMTP Listener (Tokio TCP)

**Files:**
- Create: `crates/harmony-mail/src/server.rs`
- Modify: `crates/harmony-mail/src/main.rs`

Binds TCP listener, accepts connections, spawns per-connection tasks that wire the I/O layer to the SMTP state machine. Implements connection rate limiting (per-IP counters).

**Step 1: Write integration test that connects to the server and completes an SMTP handshake**

**Step 2: Implement**

**Step 3: Run tests and commit**

---

## Task 13: TLS Integration

**Files:**
- Create: `crates/harmony-mail/src/tls.rs`
- Modify: `crates/harmony-mail/Cargo.toml` (add rustls, rustls-pemfile, instant-acme)

STARTTLS upgrade for port 25/587, implicit TLS for port 465. Certificate loading from files. ACME support is v1 stretch — for now, manual cert paths.

**Step 1: Write tests with self-signed certs**

**Step 2: Implement**

**Step 3: Run tests and commit**

---

## Task 14: Outbound SMTP (lettre Integration)

**Files:**
- Create: `crates/harmony-mail/src/outbound.rs`
- Modify: `crates/harmony-mail/Cargo.toml` (add lettre, mail-builder)

Translates HarmonyMessage -> RFC 5322, DKIM-signs, sends via lettre. Includes the persistent outbound queue with retry schedule.

**Step 1: Write tests for HarmonyMessage -> RFC 5322 conversion**

**Step 2: Implement outbound path and queue**

**Step 3: Run tests and commit**

---

## Task 15: Inbound Format Translation

**Files:**
- Create: `crates/harmony-mail/src/translate.rs`
- Modify: `crates/harmony-mail/Cargo.toml` (add mail-parser, mail-auth)

Translates RFC 5322 -> HarmonyMessage. Extracts headers, body (prefers text/plain, converts HTML to plaintext), MIME attachments -> AttachmentRef CIDs.

**Step 1: Write tests with sample RFC 5322 messages**

Construct realistic test emails (plain text, HTML, multipart/mixed with attachment) and verify correct HarmonyMessage output.

**Step 2: Implement**

**Step 3: Run tests and commit**

---

## Task 16: DKIM/SPF/DMARC Integration

**Files:**
- Modify: `crates/harmony-mail/src/smtp.rs` (wire auth results into state machine)
- Modify: `crates/harmony-mail/src/outbound.rs` (DKIM signing)

Wire `mail-auth` crate into the inbound verification path (SMTP state machine's post-DATA phase) and the outbound signing path.

**Step 1: Write tests with known-good DKIM-signed messages**

**Step 2: Implement**

**Step 3: Run tests and commit**

---

## Final Integration

After all tasks complete:

```bash
cargo test --workspace
cargo clippy --workspace
cargo fmt --all -- --check
```

Verify the binary runs:

```bash
cargo run -p harmony-mail -- --help
cargo run -p harmony-mail -- init --domain test.example.com --admin-email admin@test.example.com
```

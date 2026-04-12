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

    #[error("invalid UTF-8 in {field}")]
    InvalidUtf8 { field: &'static str },

    #[error("invalid in_reply_to flag: {0}")]
    InvalidInReplyToFlag(u8),

    #[error("filename too long: {len} bytes, maximum 255")]
    FilenameTooLong { len: usize },

    #[error("mime type too long: {len} bytes, maximum 255")]
    MimeTypeTooLong { len: usize },

    #[error("unknown SMTP command: {0}")]
    UnknownCommand(String),

    #[error("invalid identity bytes in registration")]
    InvalidIdentity,

    #[error("trailing bytes after message: {count} extra bytes")]
    TrailingBytes { count: usize },

    #[error("{field} too long for u16 length prefix: {len} bytes (max 65535)")]
    StringTooLong { field: &'static str, len: usize },

    #[error("invalid magic bytes: expected {expected:?}, found {found:?}")]
    InvalidMagic { expected: [u8; 4], found: [u8; 4] },

    #[error("invalid flag value in {field}: {value:#04x}")]
    InvalidFlag { field: &'static str, value: u8 },

    #[error("too many entries: {count}, maximum {max}")]
    TooManyEntries { count: usize, max: usize },

    #[error("{field} too long: {len} bytes, maximum {max}")]
    FieldTooLong {
        field: &'static str,
        len: usize,
        max: usize,
    },
}

/// Validate that a string's length fits in a u16 for length-prefixed encoding.
pub(crate) fn check_u16_len(s: &str, field: &'static str) -> Result<u16, MailError> {
    u16::try_from(s.len()).map_err(|_| MailError::StringTooLong {
        field,
        len: s.len(),
    })
}

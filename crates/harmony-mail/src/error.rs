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
}

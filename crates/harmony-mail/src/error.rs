use harmony_mailbox::MailboxError;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum MailError {
    #[error(transparent)]
    Mailbox(#[from] MailboxError),

    #[error("unknown SMTP command: {0}")]
    UnknownCommand(String),

    #[error("invalid UTF-8 in SMTP command line")]
    InvalidSmtpUtf8,

    #[error("invalid identity bytes in registration")]
    InvalidIdentity,
}

/// Validate that a string's length fits in a u16 for length-prefixed encoding.
pub(crate) fn check_u16_len(s: &str, field: &'static str) -> Result<u16, MailError> {
    harmony_mailbox::error::check_u16_len(s, field).map_err(MailError::from)
}

/// Errors produced by content-addressing operations.
#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error("payload too large: {size} bytes exceeds maximum {max}")]
    PayloadTooLarge { size: usize, max: usize },

    #[error("invalid CID: checksum mismatch")]
    ChecksumMismatch,

    #[error("invalid CID: child depth {child} must be less than parent depth {parent}")]
    DepthViolation { child: u8, parent: u8 },

    #[error("not an inline metadata CID")]
    NotInlineMetadata,
}

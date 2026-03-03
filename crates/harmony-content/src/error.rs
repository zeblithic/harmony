use crate::cid::ContentId;

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

    #[error("invalid bundle length: {len} is not a multiple of 32")]
    InvalidBundleLength { len: usize },

    #[error("cannot build an empty bundle")]
    EmptyBundle,

    #[error("invalid chunker config: {reason}")]
    InvalidChunkerConfig { reason: &'static str },

    #[error("cannot ingest empty data")]
    EmptyData,

    #[error("content not found in store: {cid}")]
    MissingContent { cid: ContentId },

    #[error("invalid delta: {reason}")]
    InvalidDelta { reason: &'static str },
}

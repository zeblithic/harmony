/// Errors produced by the Harmony-Zenoh layer.
#[derive(Debug, thiserror::Error)]
pub enum ZenohError {
    #[error("invalid key expression: {0}")]
    InvalidKeyExpr(String),

    #[error("subscription not found: {0}")]
    SubscriptionNotFound(u64),

    #[error("envelope too short: {0} bytes, minimum {1}")]
    EnvelopeTooShort(usize, usize),

    #[error("unsupported envelope version: {0}")]
    UnsupportedVersion(u8),

    #[error("invalid message type: {0}")]
    InvalidMessageType(u8),

    #[error("envelope seal failed: {0}")]
    SealFailed(String),

    #[error("envelope open failed: {0}")]
    OpenFailed(String),
}

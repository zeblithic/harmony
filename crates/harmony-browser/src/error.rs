use thiserror::Error;

#[derive(Debug, Error)]
pub enum BrowserError {
    #[error("content error: {0}")]
    Content(#[from] harmony_content::error::ContentError),

    #[error("invalid CID hex: {0}")]
    InvalidCidHex(String),

    #[error("subscription not found: {0}")]
    SubscriptionNotFound(u64),
}

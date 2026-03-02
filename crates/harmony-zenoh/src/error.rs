/// Errors produced by the Harmony-Zenoh layer.
#[derive(Debug, thiserror::Error)]
pub enum ZenohError {
    #[error("invalid key expression: {0}")]
    InvalidKeyExpr(String),

    #[error("subscription not found: {0}")]
    SubscriptionNotFound(u64),
}

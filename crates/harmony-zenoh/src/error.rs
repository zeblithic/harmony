/// Errors produced by the Harmony-Zenoh layer.
#[derive(Debug, thiserror::Error)]
pub enum ZenohError {
    #[error("invalid key expression: {0}")]
    InvalidKeyExpr(String),

    #[error("missing format field: {0}")]
    MissingField(String),

    #[error("subscription not found: {0}")]
    SubscriptionNotFound(u64),
}

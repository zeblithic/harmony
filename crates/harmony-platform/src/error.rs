/// Errors from platform operations.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("network send failed")]
    SendFailed,

    #[error("persistent storage operation failed")]
    StorageFailed,
}

/// Errors from platform operations.
#[derive(Debug, thiserror::Error)]
pub enum PlatformError {
    #[error("network send failed")]
    SendFailed,

    #[error("persistent storage operation failed")]
    StorageFailed,
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn display_send_failed() {
        let err = PlatformError::SendFailed;
        assert_eq!(format!("{err}"), "network send failed");
    }

    #[test]
    fn display_storage_failed() {
        let err = PlatformError::StorageFailed;
        assert_eq!(format!("{err}"), "persistent storage operation failed");
    }

    #[test]
    fn debug_impl_exists() {
        let err = PlatformError::SendFailed;
        let dbg = format!("{err:?}");
        assert!(dbg.contains("SendFailed"));
    }
}

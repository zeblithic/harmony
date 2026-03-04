/// Errors from compute operations.
#[derive(Debug, thiserror::Error)]
pub enum ComputeError {
    #[error("invalid WASM module: {reason}")]
    InvalidModule { reason: String },

    #[error("execution trapped: {reason}")]
    Trap { reason: String },

    #[error("no pending execution to resume")]
    NoPendingExecution,

    #[error("export not found: {name}")]
    ExportNotFound { name: String },

    #[error("memory too small: need {need} bytes, have {have}")]
    MemoryTooSmall { need: usize, have: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_invalid_module() {
        let err = ComputeError::InvalidModule {
            reason: "bad magic".into(),
        };
        assert_eq!(err.to_string(), "invalid WASM module: bad magic");
    }

    #[test]
    fn error_display_trap() {
        let err = ComputeError::Trap {
            reason: "unreachable".into(),
        };
        assert_eq!(err.to_string(), "execution trapped: unreachable");
    }

    #[test]
    fn error_display_no_pending() {
        let err = ComputeError::NoPendingExecution;
        assert_eq!(err.to_string(), "no pending execution to resume");
    }

    #[test]
    fn error_display_export_not_found() {
        let err = ComputeError::ExportNotFound {
            name: "compute".into(),
        };
        assert_eq!(err.to_string(), "export not found: compute");
    }

    #[test]
    fn error_display_memory_too_small() {
        let err = ComputeError::MemoryTooSmall {
            need: 1024,
            have: 512,
        };
        assert_eq!(
            err.to_string(),
            "memory too small: need 1024 bytes, have 512"
        );
    }
}

/// Errors from workflow operations.
#[derive(Debug, thiserror::Error)]
pub enum WorkflowError {
    #[error("workflow not found: {id}")]
    NotFound { id: String },

    #[error("workflow already exists: {id}")]
    AlreadyExists { id: String },

    #[error("workflow not in expected state: expected {expected}, got {actual}")]
    InvalidState { expected: String, actual: String },

    #[error("compute error: {0}")]
    Compute(#[from] harmony_compute::ComputeError),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_not_found() {
        let err = WorkflowError::NotFound {
            id: "abc123".into(),
        };
        assert_eq!(err.to_string(), "workflow not found: abc123");
    }

    #[test]
    fn error_display_already_exists() {
        let err = WorkflowError::AlreadyExists {
            id: "abc123".into(),
        };
        assert_eq!(err.to_string(), "workflow already exists: abc123");
    }

    #[test]
    fn error_display_invalid_state() {
        let err = WorkflowError::InvalidState {
            expected: "Executing".into(),
            actual: "Complete".into(),
        };
        assert_eq!(
            err.to_string(),
            "workflow not in expected state: expected Executing, got Complete"
        );
    }
}

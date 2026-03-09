// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Error types for Kitri workflows.

use alloc::string::String;
use core::fmt;

/// Errors surfaced to Kitri workflow authors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KitriError {
    /// Workflow logic failed (panic, assertion, unrecoverable).
    WorkflowFailed { reason: String },
    /// I/O operation denied — capability not granted.
    Unauthorized { resource: String },
    /// Resource budget exceeded after max retries.
    ResourceExhausted { detail: String },
    /// Lyll/Nakaiah detected memory corruption.
    IntegrityViolation,
    /// Content not found in the mesh.
    ContentNotFound { cid: [u8; 32] },
    /// AI model invocation failed.
    InferFailed { reason: String },
    /// Checkpoint serialization/deserialization failed.
    CheckpointFailed { reason: String },
    /// Manifest parsing error.
    ManifestInvalid { reason: String },
    /// DAG validation error (cycles, missing steps).
    DagInvalid { reason: String },
    /// Compensation (saga rollback) failed.
    CompensationFailed { reason: String },
}

impl fmt::Display for KitriError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorkflowFailed { reason } => write!(f, "workflow failed: {reason}"),
            Self::Unauthorized { resource } => write!(f, "unauthorized: {resource}"),
            Self::ResourceExhausted { detail } => write!(f, "resource exhausted: {detail}"),
            Self::IntegrityViolation => write!(f, "integrity violation detected"),
            Self::ContentNotFound { cid } => write!(f, "content not found: {cid:02x?}"),
            Self::InferFailed { reason } => write!(f, "inference failed: {reason}"),
            Self::CheckpointFailed { reason } => write!(f, "checkpoint failed: {reason}"),
            Self::ManifestInvalid { reason } => write!(f, "invalid manifest: {reason}"),
            Self::DagInvalid { reason } => write!(f, "invalid DAG: {reason}"),
            Self::CompensationFailed { reason } => write!(f, "compensation failed: {reason}"),
        }
    }
}

/// Convenience alias for Kitri workflow results.
pub type KitriResult<T> = Result<T, KitriError>;

#[cfg(test)]
mod tests {
    use alloc::string::ToString;

    use super::*;

    #[test]
    fn kitri_error_variants_exist() {
        let e1 = KitriError::WorkflowFailed {
            reason: "boom".into(),
        };
        assert!(matches!(e1, KitriError::WorkflowFailed { .. }));

        let e2 = KitriError::Unauthorized {
            resource: "topic/foo".into(),
        };
        assert!(matches!(e2, KitriError::Unauthorized { .. }));

        let e3 = KitriError::ResourceExhausted {
            detail: "fuel".into(),
        };
        assert!(matches!(e3, KitriError::ResourceExhausted { .. }));

        let e4 = KitriError::IntegrityViolation;
        assert!(matches!(e4, KitriError::IntegrityViolation));

        let e5 = KitriError::ContentNotFound { cid: [0xAA; 32] };
        assert!(matches!(e5, KitriError::ContentNotFound { .. }));

        let e6 = KitriError::InferFailed {
            reason: "model unavailable".into(),
        };
        assert!(matches!(e6, KitriError::InferFailed { .. }));
    }

    #[test]
    fn kitri_result_alias_works() {
        let ok: KitriResult<u32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);

        let err: KitriResult<u32> = Err(KitriError::IntegrityViolation);
        assert!(err.is_err());
    }

    #[test]
    fn kitri_error_display() {
        let e = KitriError::WorkflowFailed {
            reason: "boom".into(),
        };
        assert_eq!(e.to_string(), "workflow failed: boom");

        let e = KitriError::IntegrityViolation;
        assert_eq!(e.to_string(), "integrity violation detected");

        let e = KitriError::ContentNotFound { cid: [0xAA; 32] };
        let display = e.to_string();
        assert!(display.starts_with("content not found:"));
        assert!(display.contains("aa"));

        let e = KitriError::Unauthorized {
            resource: "topic/secret".into(),
        };
        assert_eq!(e.to_string(), "unauthorized: topic/secret");

        let e = KitriError::ResourceExhausted {
            detail: "out of fuel".into(),
        };
        assert_eq!(e.to_string(), "resource exhausted: out of fuel");

        let e = KitriError::InferFailed {
            reason: "model timeout".into(),
        };
        assert_eq!(e.to_string(), "inference failed: model timeout");

        let e = KitriError::CheckpointFailed {
            reason: "corrupt".into(),
        };
        assert_eq!(e.to_string(), "checkpoint failed: corrupt");

        let e = KitriError::ManifestInvalid {
            reason: "missing field".into(),
        };
        assert_eq!(e.to_string(), "invalid manifest: missing field");

        let e = KitriError::DagInvalid {
            reason: "cycle detected".into(),
        };
        assert_eq!(e.to_string(), "invalid DAG: cycle detected");

        let e = KitriError::CompensationFailed {
            reason: "rollback error".into(),
        };
        assert_eq!(e.to_string(), "compensation failed: rollback error");
    }
}

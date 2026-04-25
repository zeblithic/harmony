use thiserror::Error;

#[derive(Debug, Error)]
pub enum OwnerError {
    #[error("invalid signature on {cert_type} cert")]
    InvalidSignature { cert_type: &'static str },

    #[error("unknown wire format version: {0}")]
    UnknownVersion(u8),

    #[error("CBOR encode/decode failure: {0}")]
    Cbor(String),

    #[error("identity hash does not match contained public keys")]
    IdentityHashMismatch,

    #[error("device {device:?} is not enrolled under owner {owner:?}")]
    NotEnrolled { owner: [u8; 16], device: [u8; 16] },

    #[error("device {device:?} is revoked")]
    Revoked { device: [u8; 16] },

    #[error("quorum requires at least {min} signers, got {got}")]
    InsufficientQuorum { min: usize, got: usize },

    #[error("trust state snapshot is older than freshness window ({age_days}d > 30d)")]
    StaleTrustState { age_days: u64 },

    #[error("identity in contested state — competing master/sibling assertions")]
    Contested,

    #[error("reclamation refuted by predecessor liveness")]
    ReclamationRefuted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_context() {
        let e = OwnerError::InvalidSignature { cert_type: "Enrollment" };
        assert!(format!("{e}").contains("Enrollment"));

        let e = OwnerError::InsufficientQuorum { min: 2, got: 1 };
        assert!(format!("{e}").contains("at least 2"));
    }
}

//! Error types for harmony-pkarr operations.
//!
//! Mirrors harmony-discovery's manual Display + std::error::Error pattern
//! (no thiserror dep at the harmony-core layer).

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PkarrError {
    /// BEP44 outer signature failed verification. (RPK1)
    OuterSignatureInvalid,
    /// Inner identity signature failed verification. (RPK2)
    InnerSignatureInvalid,
    /// `harmony_identity_pub` does not match the expected identity.
    /// (RPK3)
    IdentityMismatch,
    /// `announced_at_ms` outside ±30 min skew window. (RPK4)
    StaleOrSkewed,
    /// `routing_blob` could not be decoded by the caller's parser. (RPK5)
    /// Returned only by callers that wrap harmony-pkarr; harmony-pkarr itself
    /// treats `routing_blob` as opaque bytes.
    RoutingBlobInvalid,
    /// Relay returned a non-success HTTP status.
    RelayHttpError(u16),
    /// All configured relays are on cooldown or unreachable.
    NoRelaysAvailable,
    /// Relay returned a malformed BEP44 envelope.
    RelayResponseInvalid,
    /// CBOR serialize/deserialize failure on PkarrRoutingRecord payload.
    SerializeError(&'static str),
    DeserializeError(&'static str),
    /// Record structurally invalid (e.g., identity_pub wrong length).
    InvalidRecord,
    /// Record exceeds the pkarr SignedPacket size budget (MAX_BYTES = 1104).
    RecordTooLarge,
}

impl core::fmt::Display for PkarrError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OuterSignatureInvalid => write!(f, "BEP44 outer signature invalid"),
            Self::InnerSignatureInvalid => write!(f, "inner identity signature invalid"),
            Self::IdentityMismatch => {
                write!(f, "harmony_identity_pub does not match expected identity")
            }
            Self::StaleOrSkewed => write!(f, "announced_at_ms outside ±30min skew"),
            Self::RoutingBlobInvalid => write!(f, "routing_blob could not be decoded"),
            Self::RelayHttpError(status) => write!(f, "relay returned HTTP {status}"),
            Self::NoRelaysAvailable => {
                write!(f, "no relays available (all on cooldown or unreachable)")
            }
            Self::RelayResponseInvalid => write!(f, "relay returned malformed response"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
            Self::InvalidRecord => write!(f, "record structurally invalid"),
            Self::RecordTooLarge => write!(f, "record too large for pkarr packet"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PkarrError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display_includes_status_code() {
        let e = PkarrError::RelayHttpError(429);
        let s = format!("{}", e);
        assert!(s.contains("429"));
    }

    #[test]
    fn errors_are_comparable() {
        // PartialEq is load-bearing for #[derive(PartialEq)] on downstream types
        // that include PkarrError; verify the derive.
        assert_eq!(
            PkarrError::OuterSignatureInvalid,
            PkarrError::OuterSignatureInvalid
        );
        assert_ne!(
            PkarrError::OuterSignatureInvalid,
            PkarrError::InnerSignatureInvalid
        );
    }

    #[test]
    fn record_too_large_displays() {
        let s = format!("{}", PkarrError::RecordTooLarge);
        assert!(s.contains("too large"));
    }
}

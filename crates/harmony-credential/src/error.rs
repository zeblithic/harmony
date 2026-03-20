/// Errors returned by credential operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CredentialError {
    NotYetValid,
    Expired,
    IssuerNotFound,
    SignatureInvalid,
    Revoked,
    StatusListNotFound,
    DisclosureMismatch,
    DuplicateDisclosure,
    IndexOutOfBounds,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for CredentialError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotYetValid => write!(f, "credential not yet valid"),
            Self::Expired => write!(f, "credential expired"),
            Self::IssuerNotFound => write!(f, "issuer not found"),
            Self::SignatureInvalid => write!(f, "invalid credential signature"),
            Self::Revoked => write!(f, "credential revoked"),
            Self::StatusListNotFound => write!(f, "status list not found for issuer"),
            Self::DisclosureMismatch => write!(f, "disclosed claim does not match any digest"),
            Self::DuplicateDisclosure => write!(f, "duplicate disclosed claim"),
            Self::IndexOutOfBounds => write!(f, "status list index out of bounds"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CredentialError {}

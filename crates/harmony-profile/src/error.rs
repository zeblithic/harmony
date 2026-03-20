/// Errors returned by profile and endorsement operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProfileError {
    Expired,
    FutureTimestamp,
    InvalidRecord,
    SignatureInvalid,
    KeyNotFound,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for ProfileError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Expired => write!(f, "record expired"),
            Self::FutureTimestamp => write!(f, "published_at is too far in the future"),
            Self::InvalidRecord => write!(f, "record structurally invalid"),
            Self::SignatureInvalid => write!(f, "invalid signature"),
            Self::KeyNotFound => write!(f, "signing key not found"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for ProfileError {}

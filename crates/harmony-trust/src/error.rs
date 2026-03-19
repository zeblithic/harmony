/// Errors returned by TrustStore operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustError {
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for TrustError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TrustError {}

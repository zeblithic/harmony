/// Errors returned by discovery operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiscoveryError {
    Expired,
    SignatureInvalid,
    /// V2: will be returned when public_key→IdentityHash re-derivation
    /// check is implemented. Currently unused — see `verify_announce` docs.
    #[allow(dead_code)]
    AddressMismatch,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Expired => write!(f, "announce record expired"),
            Self::SignatureInvalid => write!(f, "invalid announce signature"),
            Self::AddressMismatch => write!(f, "public key does not match identity address"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for DiscoveryError {}

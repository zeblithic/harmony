use alloc::string::String;

#[non_exhaustive]
#[derive(Debug)]
pub enum SdJwtError {
    EmptyInput,
    MalformedCompact,
    Base64Error,
    JsonError(String),
    MissingAlgorithm,
    UnsupportedAlgorithm(String),
    InvalidDisclosure,
    /// The `typ` header is missing or not `"sd+jwt"` (RFC 9901 §3.3).
    WrongTokenType,
    /// A disclosure's hash does not appear in the signed `_sd` list.
    #[cfg(feature = "credential")]
    DisclosureHashMismatch,
    /// The same disclosure was presented more than once.
    #[cfg(feature = "credential")]
    DuplicateDisclosure,
    /// Decoded salt is shorter than 16 bytes (RFC 9901 §5.2.1).
    #[cfg(feature = "credential")]
    SaltTooShort,
    SignatureInvalid(harmony_identity::IdentityError),
}

impl core::fmt::Display for SdJwtError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "empty input"),
            Self::MalformedCompact => write!(f, "malformed JWS compact serialization"),
            Self::Base64Error => write!(f, "base64url decoding failed"),
            Self::JsonError(msg) => write!(f, "JSON parse error: {msg}"),
            Self::MissingAlgorithm => write!(f, "JWS header missing 'alg' field"),
            Self::UnsupportedAlgorithm(alg) => write!(f, "unsupported JWS algorithm: {alg}"),
            Self::InvalidDisclosure => {
                write!(
                    f,
                    "disclosure is not a valid [salt, name?, value] array"
                )
            }
            Self::WrongTokenType => write!(f, "typ header must be \"sd+jwt\" (RFC 9901 §3.3)"),
            #[cfg(feature = "credential")]
            Self::DisclosureHashMismatch => {
                write!(f, "disclosure hash not found in signed _sd list")
            }
            #[cfg(feature = "credential")]
            Self::DuplicateDisclosure => write!(f, "same disclosure presented more than once"),
            #[cfg(feature = "credential")]
            Self::SaltTooShort => write!(f, "decoded salt shorter than 16 bytes (RFC 9901 §5.2.1)"),
            Self::SignatureInvalid(e) => write!(f, "signature verification failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SdJwtError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SignatureInvalid(e) => Some(e),
            _ => None,
        }
    }
}

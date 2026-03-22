use alloc::string::String;

#[derive(Debug)]
pub enum SdJwtError {
    EmptyInput,
    MalformedCompact,
    Base64Error,
    JsonError(String),
    MissingAlgorithm,
    UnsupportedAlgorithm(String),
    InvalidDisclosure,
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
            Self::SignatureInvalid(e) => write!(f, "signature verification failed: {e}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SdJwtError {}

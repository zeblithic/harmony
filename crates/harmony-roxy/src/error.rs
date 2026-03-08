/// Errors produced by content-licensing operations.
#[derive(Debug, thiserror::Error)]
pub enum RoxyError {
    #[error("serialization error: {0}")]
    Serialization(#[from] postcard::Error),

    #[error("crypto error: {0}")]
    Crypto(#[from] harmony_crypto::CryptoError),

    #[error("identity error: {0}")]
    Identity(#[from] harmony_identity::IdentityError),

    #[error("content error: {0}")]
    Content(#[from] harmony_content::error::ContentError),

    #[error("manifest signature verification failed")]
    InvalidSignature,

    #[error("manifest creator does not match signer")]
    CreatorMismatch,

    #[error("malformed UCAN resource bytes")]
    InvalidResource,

    #[error("license has expired")]
    LicenseExpired,
}

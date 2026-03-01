/// Errors produced by identity operations.
#[derive(Debug, thiserror::Error)]
pub enum IdentityError {
    #[error("invalid public key length: expected 64 bytes, got {0}")]
    InvalidPublicKeyLength(usize),

    #[error("invalid private key length: expected 64 bytes, got {0}")]
    InvalidPrivateKeyLength(usize),

    #[error("invalid Ed25519 verifying key")]
    InvalidVerifyingKey,

    #[error("signature verification failed")]
    SignatureInvalid,

    #[error("decryption failed")]
    DecryptionFailed,

    #[error(transparent)]
    Crypto(#[from] harmony_crypto::CryptoError),
}

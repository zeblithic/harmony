/// Errors produced by cryptographic operations.
#[derive(Debug, thiserror::Error)]
pub enum CryptoError {
    #[error("ciphertext too short")]
    CiphertextTooShort,

    #[error("HMAC verification failed")]
    HmacMismatch,

    #[error("decryption failed (bad padding or corrupted data)")]
    DecryptionFailed,

    #[error("invalid key length: expected {expected}, got {got}")]
    InvalidKeyLength { expected: usize, got: usize },

    #[error("AEAD encryption failed")]
    AeadEncryptFailed,

    #[error("AEAD decryption failed (authentication tag mismatch)")]
    AeadDecryptFailed,

    #[error("HKDF output length {requested} exceeds maximum ({max})")]
    HkdfLengthExceeded { requested: usize, max: usize },
}

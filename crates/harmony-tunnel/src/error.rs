/// Errors that can occur during tunnel operations.
#[derive(Debug, thiserror::Error)]
pub enum TunnelError {
    #[error("tunnel is not in the expected state for this operation")]
    InvalidState,

    #[error("handshake signature verification failed")]
    SignatureVerificationFailed,

    #[error("ML-KEM decapsulation produced an invalid shared secret")]
    DecapsulationFailed,

    #[error("frame decryption failed (authentication tag mismatch)")]
    DecryptionFailed,

    #[error("frame too short: expected at least {expected} bytes, got {got}")]
    FrameTooShort { expected: usize, got: usize },

    #[error("unknown frame tag: 0x{tag:02x}")]
    UnknownFrameTag { tag: u8 },

    #[error("handshake message malformed: {reason}")]
    MalformedHandshake { reason: &'static str },

    #[error("replication message malformed: {reason}")]
    MalformedReplication { reason: &'static str },

    #[error("keepalive timeout: peer unresponsive")]
    KeepaliveTimeout,

    #[error("cryptographic operation failed: {0}")]
    Crypto(#[from] harmony_crypto::CryptoError),
}

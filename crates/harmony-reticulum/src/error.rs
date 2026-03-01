use harmony_identity::IdentityError;

/// Errors produced by Reticulum packet and announce operations.
#[derive(Debug, thiserror::Error)]
pub enum ReticulumError {
    #[error("packet too short: minimum {minimum} bytes, got {actual}")]
    PacketTooShort { minimum: usize, actual: usize },

    #[error("packet exceeds MTU: {size} bytes > {mtu} byte limit")]
    PacketExceedsMtu { size: usize, mtu: usize },

    #[error("invalid packet type: {0:#04x}")]
    InvalidPacketType(u8),

    #[error("announce too short: minimum {minimum} bytes, got {actual}")]
    AnnounceTooShort { minimum: usize, actual: usize },

    #[error("announce signature invalid")]
    AnnounceSignatureInvalid,

    #[error("announce destination mismatch: expected {expected}, actual {actual}")]
    AnnounceDestinationMismatch { expected: String, actual: String },

    #[error("transport header requires transport_id")]
    MissingTransportId,

    #[error("Type1 header must not have transport_id")]
    UnexpectedTransportId,

    #[error("invalid destination name: dots are not allowed in individual components")]
    InvalidDestinationName,

    #[error("app_data too large for MTU")]
    AppDataTooLarge,

    #[error("link request too short: minimum {minimum} bytes, got {actual}")]
    LinkRequestTooShort { minimum: usize, actual: usize },

    #[error("link proof too short: minimum {minimum} bytes, got {actual}")]
    LinkProofTooShort { minimum: usize, actual: usize },

    #[error("link proof signature invalid")]
    LinkProofSignatureInvalid,

    #[error("link identification invalid")]
    LinkIdentificationInvalid,

    #[error("link not active")]
    LinkNotActive,

    #[error("link already closed")]
    LinkAlreadyClosed,

    #[error(transparent)]
    Identity(#[from] IdentityError),
}

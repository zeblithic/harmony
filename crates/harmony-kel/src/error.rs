/// Errors returned by Key Event Log operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KelError {
    InvalidInceptionSignature,
    PreRotationMismatch,
    HashChainBroken,
    SequenceViolation,
    InvalidSignature,
    DuplicateInception,
    EmptyLog,
    SerializeError(&'static str),
    DeserializeError(&'static str),
}

impl core::fmt::Display for KelError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidInceptionSignature => write!(f, "invalid inception signature"),
            Self::PreRotationMismatch => write!(f, "pre-rotation commitment mismatch"),
            Self::HashChainBroken => write!(f, "hash chain broken"),
            Self::SequenceViolation => write!(f, "sequence number violation"),
            Self::InvalidSignature => write!(f, "invalid event signature"),
            Self::DuplicateInception => write!(f, "duplicate inception event"),
            Self::EmptyLog => write!(f, "empty key event log"),
            Self::SerializeError(msg) => write!(f, "serialize error: {msg}"),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for KelError {}

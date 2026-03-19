use harmony_identity::IdentityHash;

/// Errors returned by ContactStore operations.
#[derive(Debug)]
pub enum ContactError {
    /// A contact with this identity hash already exists.
    AlreadyExists(IdentityHash),
    /// Deserialization of persisted data failed.
    DeserializeError(&'static str),
}

impl core::fmt::Display for ContactError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::AlreadyExists(h) => write!(f, "contact already exists: {:02x?}", &h[..4]),
            Self::DeserializeError(msg) => write!(f, "deserialize error: {msg}"),
        }
    }
}

//! Binary E2EE message envelope for Harmony.
//!
//! Wire format (33-byte fixed header + variable ciphertext):
//!
//! ```text
//! [1B ver|type][16B sender_addr][12B nonce][4B sequence][N bytes ciphertext+tag]
//! ```
//!
//! The 33-byte header is passed as AAD to ChaCha20-Poly1305, cryptographically
//! binding the routing metadata to the encrypted payload.

use harmony_crypto::aead;

use crate::ZenohError;

/// Current envelope format version.
pub const VERSION: u8 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 1 + 16 + 12 + 4; // 33

/// Minimum envelope size: header + Poly1305 tag (16 bytes for empty plaintext).
pub const MIN_ENVELOPE_SIZE: usize = HEADER_SIZE + aead::TAG_LENGTH;

/// Message type carried in the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Data message (publish).
    Put = 0,
    /// Tombstone (delete).
    Del = 1,
}

impl MessageType {
    /// Decode from the low 4 bits of the version|type byte.
    pub fn from_u8(val: u8) -> Result<Self, ZenohError> {
        match val {
            0 => Ok(Self::Put),
            1 => Ok(Self::Del),
            other => Err(ZenohError::InvalidMessageType(other)),
        }
    }
}

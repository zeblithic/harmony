use alloc::vec::Vec;

use harmony_crypto::aead::{self, KEY_LENGTH, NONCE_LENGTH};

use crate::error::TunnelError;

/// Frame type tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum FrameTag {
    Keepalive = 0x00,
    Reticulum = 0x01,
    Zenoh = 0x02,
}

impl FrameTag {
    pub fn from_byte(b: u8) -> Result<Self, TunnelError> {
        match b {
            0x00 => Ok(Self::Keepalive),
            0x01 => Ok(Self::Reticulum),
            0x02 => Ok(Self::Zenoh),
            other => Err(TunnelError::UnknownFrameTag { tag: other }),
        }
    }
}

/// A plaintext frame before encryption / after decryption.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    pub tag: FrameTag,
    pub payload: Vec<u8>,
}

/// Frame header size: 1 byte tag + 4 bytes length.
const FRAME_HEADER_LEN: usize = 5;

impl Frame {
    /// Create a keepalive frame (empty payload, for deterministic tests).
    pub fn keepalive() -> Self {
        Self {
            tag: FrameTag::Keepalive,
            payload: Vec::new(),
        }
    }

    /// Encode the frame to wire format (tag + big-endian length + payload).
    pub fn encode(&self) -> Vec<u8> {
        let len = self.payload.len() as u32;
        let mut buf = Vec::with_capacity(FRAME_HEADER_LEN + self.payload.len());
        buf.push(self.tag as u8);
        buf.extend_from_slice(&len.to_be_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    /// Decode a frame from wire format.
    pub fn decode(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < FRAME_HEADER_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: FRAME_HEADER_LEN,
                got: data.len(),
            });
        }

        let tag = FrameTag::from_byte(data[0])?;
        let len = u32::from_be_bytes([data[1], data[2], data[3], data[4]]) as usize;

        if data.len() < FRAME_HEADER_LEN + len {
            return Err(TunnelError::FrameTooShort {
                expected: FRAME_HEADER_LEN + len,
                got: data.len(),
            });
        }

        let payload = data[FRAME_HEADER_LEN..FRAME_HEADER_LEN + len].to_vec();

        Ok(Self { tag, payload })
    }
}

/// Build a 12-byte AEAD nonce from a 64-bit counter.
///
/// Format: [4 bytes zero padding][8 bytes big-endian counter]
fn counter_to_nonce(counter: u64) -> [u8; NONCE_LENGTH] {
    let mut nonce = [0u8; NONCE_LENGTH];
    nonce[4..].copy_from_slice(&counter.to_be_bytes());
    nonce
}

/// Encrypt a frame as a single AEAD ciphertext.
///
/// The entire plaintext frame (tag + length + payload) is encrypted.
/// The nonce counter is incremented after each call.
/// AAD should be the remote peer's NodeId (32 bytes).
pub fn encrypt_frame(
    frame: &Frame,
    key: &[u8; KEY_LENGTH],
    aad: &[u8],
    nonce_counter: &mut u64,
) -> Result<Vec<u8>, TunnelError> {
    let plaintext = frame.encode();
    let nonce = counter_to_nonce(*nonce_counter);
    *nonce_counter += 1;

    aead::encrypt(key, &nonce, &plaintext, aad).map_err(TunnelError::Crypto)
}

/// Decrypt a ciphertext back into a frame.
///
/// Verifies the AEAD tag, then decodes the plaintext frame.
/// The nonce counter is incremented after each call.
pub fn decrypt_frame(
    ciphertext: &[u8],
    key: &[u8; KEY_LENGTH],
    aad: &[u8],
    nonce_counter: &mut u64,
) -> Result<Frame, TunnelError> {
    let nonce = counter_to_nonce(*nonce_counter);
    *nonce_counter += 1;

    let plaintext =
        aead::decrypt(key, &nonce, ciphertext, aad).map_err(|_| TunnelError::DecryptionFailed)?;

    Frame::decode(&plaintext)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_crypto::aead::KEY_LENGTH;

    #[test]
    fn frame_encode_decode_roundtrip() {
        let frame = Frame {
            tag: FrameTag::Reticulum,
            payload: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };

        let bytes = frame.encode();
        assert_eq!(bytes.len(), 1 + 4 + 4); // tag + length + payload

        let decoded = Frame::decode(&bytes).unwrap();
        assert_eq!(decoded.tag, FrameTag::Reticulum);
        assert_eq!(decoded.payload, vec![0xDE, 0xAD, 0xBE, 0xEF]);
    }

    #[test]
    fn keepalive_frame_has_empty_payload() {
        let frame = Frame::keepalive();
        let bytes = frame.encode();
        assert_eq!(bytes.len(), 5); // tag + length(0)
        assert_eq!(bytes[0], 0x00);
        assert_eq!(&bytes[1..5], &[0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn encrypted_frame_roundtrip() {
        let key = [0x42u8; KEY_LENGTH];
        let aad = [0xABu8; 32]; // NodeId as AAD

        let frame = Frame {
            tag: FrameTag::Zenoh,
            payload: b"hello zenoh".to_vec(),
        };

        let mut nonce_counter: u64 = 0;
        let encrypted = encrypt_frame(&frame, &key, &aad, &mut nonce_counter).unwrap();
        assert_eq!(nonce_counter, 1); // Counter incremented

        let mut decrypt_counter: u64 = 0;
        let decrypted = decrypt_frame(&encrypted, &key, &aad, &mut decrypt_counter).unwrap();
        assert_eq!(decrypt_counter, 1);

        assert_eq!(decrypted.tag, FrameTag::Zenoh);
        assert_eq!(decrypted.payload, b"hello zenoh");
    }

    #[test]
    fn wrong_aad_fails_decryption() {
        let key = [0x42u8; KEY_LENGTH];
        let aad = [0xABu8; 32];
        let wrong_aad = [0xCDu8; 32];

        let frame = Frame {
            tag: FrameTag::Reticulum,
            payload: b"secret".to_vec(),
        };

        let mut enc_counter: u64 = 0;
        let encrypted = encrypt_frame(&frame, &key, &aad, &mut enc_counter).unwrap();

        let mut dec_counter: u64 = 0;
        let result = decrypt_frame(&encrypted, &key, &wrong_aad, &mut dec_counter);
        assert!(result.is_err());
    }
}

//! ChaCha20-Poly1305 AEAD encryption for Harmony-native E2EE.
//!
//! This is the preferred cipher for Harmony traffic (vs Fernet for Reticulum compat).
//! Single-pass authenticated encryption with associated data.

use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    ChaCha20Poly1305, Nonce,
};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::CryptoError;

/// ChaCha20-Poly1305 key length (32 bytes).
pub const KEY_LENGTH: usize = 32;

/// ChaCha20-Poly1305 nonce length (12 bytes).
pub const NONCE_LENGTH: usize = 12;

/// Poly1305 authentication tag length (16 bytes).
pub const TAG_LENGTH: usize = 16;

/// Encrypt plaintext with ChaCha20-Poly1305.
///
/// - `key`: 32-byte symmetric key
/// - `nonce`: 12-byte nonce (must never be reused with the same key)
/// - `plaintext`: data to encrypt
/// - `aad`: additional authenticated data (authenticated but not encrypted)
///
/// Returns ciphertext with appended 16-byte Poly1305 tag.
pub fn encrypt(
    key: &[u8; KEY_LENGTH],
    nonce: &[u8; NONCE_LENGTH],
    plaintext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = ChaCha20Poly1305::new(key.into());
    let payload = Payload {
        msg: plaintext,
        aad,
    };
    cipher
        .encrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| CryptoError::AeadEncryptFailed)
}

/// Decrypt ciphertext with ChaCha20-Poly1305.
///
/// - `key`: 32-byte symmetric key
/// - `nonce`: 12-byte nonce used during encryption
/// - `ciphertext`: data to decrypt (includes appended 16-byte tag)
/// - `aad`: additional authenticated data (must match encryption)
pub fn decrypt(
    key: &[u8; KEY_LENGTH],
    nonce: &[u8; NONCE_LENGTH],
    ciphertext: &[u8],
    aad: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    let cipher = ChaCha20Poly1305::new(key.into());
    let payload = Payload {
        msg: ciphertext,
        aad,
    };
    cipher
        .decrypt(Nonce::from_slice(nonce), payload)
        .map_err(|_| CryptoError::AeadDecryptFailed)
}

/// Generate a random 12-byte nonce.
pub fn generate_nonce(rng: &mut impl CryptoRngCore) -> [u8; NONCE_LENGTH] {
    let mut nonce = [0u8; NONCE_LENGTH];
    rng.fill_bytes(&mut nonce);
    nonce
}

/// A 32-byte AEAD key that zeroizes on drop.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct AeadKey {
    bytes: [u8; KEY_LENGTH],
}

impl AeadKey {
    pub fn generate(rng: &mut impl CryptoRngCore) -> Self {
        let mut bytes = [0u8; KEY_LENGTH];
        rng.fill_bytes(&mut bytes);
        Self { bytes }
    }

    pub fn from_bytes(bytes: [u8; KEY_LENGTH]) -> Self {
        Self { bytes }
    }

    pub fn as_bytes(&self) -> &[u8; KEY_LENGTH] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn roundtrip_basic() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let plaintext = b"hello harmony";
        let aad = b"";

        let ciphertext = encrypt(key.as_bytes(), &nonce, plaintext, aad).unwrap();
        let decrypted = decrypt(key.as_bytes(), &nonce, &ciphertext, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn roundtrip_with_aad() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let plaintext = b"secret message";
        let aad = b"channel:harmony/server/123/msg";

        let ciphertext = encrypt(key.as_bytes(), &nonce, plaintext, aad).unwrap();
        let decrypted = decrypt(key.as_bytes(), &nonce, &ciphertext, aad).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_aad_fails() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let ciphertext = encrypt(key.as_bytes(), &nonce, b"data", b"correct_aad").unwrap();

        let result = decrypt(key.as_bytes(), &nonce, &ciphertext, b"wrong_aad");
        assert!(matches!(result, Err(CryptoError::AeadDecryptFailed)));
    }

    #[test]
    fn wrong_key_fails() {
        let key1 = AeadKey::generate(&mut OsRng);
        let key2 = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let ciphertext = encrypt(key1.as_bytes(), &nonce, b"data", b"").unwrap();

        let result = decrypt(key2.as_bytes(), &nonce, &ciphertext, b"");
        assert!(matches!(result, Err(CryptoError::AeadDecryptFailed)));
    }

    #[test]
    fn wrong_nonce_fails() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce1 = generate_nonce(&mut OsRng);
        let nonce2 = generate_nonce(&mut OsRng);
        let ciphertext = encrypt(key.as_bytes(), &nonce1, b"data", b"").unwrap();

        let result = decrypt(key.as_bytes(), &nonce2, &ciphertext, b"");
        assert!(matches!(result, Err(CryptoError::AeadDecryptFailed)));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let mut ciphertext = encrypt(key.as_bytes(), &nonce, b"data", b"").unwrap();
        ciphertext[0] ^= 0x01;

        let result = decrypt(key.as_bytes(), &nonce, &ciphertext, b"");
        assert!(matches!(result, Err(CryptoError::AeadDecryptFailed)));
    }

    #[test]
    fn ciphertext_length() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let plaintext = b"12345"; // 5 bytes
        let ciphertext = encrypt(key.as_bytes(), &nonce, plaintext, b"").unwrap();
        // ChaCha20-Poly1305: ciphertext = plaintext_len + 16-byte tag
        assert_eq!(ciphertext.len(), 5 + TAG_LENGTH);
    }

    #[test]
    fn roundtrip_empty_plaintext() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let ciphertext = encrypt(key.as_bytes(), &nonce, b"", b"").unwrap();
        let decrypted = decrypt(key.as_bytes(), &nonce, &ciphertext, b"").unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn roundtrip_large_payload() {
        let key = AeadKey::generate(&mut OsRng);
        let nonce = generate_nonce(&mut OsRng);
        let plaintext = vec![0xCDu8; 65536];
        let ciphertext = encrypt(key.as_bytes(), &nonce, &plaintext, b"").unwrap();
        let decrypted = decrypt(key.as_bytes(), &nonce, &ciphertext, b"").unwrap();
        assert_eq!(decrypted, plaintext);
    }
}

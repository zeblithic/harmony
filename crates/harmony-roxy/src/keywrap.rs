//! Per-consumer key wrapping via ECDH.
//!
//! Artists generate a symmetric content key (ChaCha20-Poly1305 `AeadKey`)
//! and encrypt their content with it. To grant access, the content key is
//! "wrapped" (encrypted) for a specific consumer using their public identity.
//! The consumer unwraps with their private identity to recover the key.

use alloc::vec::Vec;
use harmony_crypto::aead::AeadKey;
use harmony_identity::{Identity, PrivateIdentity};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::error::RoxyError;

/// Wrap a content key for a specific consumer.
///
/// Uses ECDH (ephemeral X25519 -> consumer's public key) + HKDF + Fernet
/// internally via `Identity::encrypt()`. The wrapped output can only be
/// unwrapped by the consumer's private identity.
///
/// Returns opaque wrapped bytes. Different on each call (ephemeral key).
pub fn wrap_key(
    rng: &mut impl CryptoRngCore,
    content_key: &AeadKey,
    recipient: &Identity,
) -> Result<Vec<u8>, RoxyError> {
    let wrapped = recipient.encrypt(rng, content_key.as_bytes())?;
    Ok(wrapped)
}

/// Unwrap a content key using the consumer's private identity.
///
/// Returns the recovered `AeadKey` for decrypting content.
pub fn unwrap_key(consumer: &PrivateIdentity, wrapped: &[u8]) -> Result<AeadKey, RoxyError> {
    let mut plaintext = consumer.decrypt(wrapped)?;
    if plaintext.len() != 32 {
        plaintext.zeroize();
        return Err(RoxyError::Crypto(
            harmony_crypto::CryptoError::InvalidKeyLength {
                expected: 32,
                got: plaintext.len(),
            },
        ));
    }
    let mut key_bytes = [0u8; 32];
    key_bytes.copy_from_slice(&plaintext);
    plaintext.zeroize();
    let key = AeadKey::from_bytes(key_bytes);
    key_bytes.zeroize();
    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_crypto::aead::AeadKey;
    use harmony_identity::PrivateIdentity;
    use rand::rngs::OsRng;

    #[test]
    fn wrap_and_unwrap_round_trip() {
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let unwrapped = unwrap_key(&consumer, &wrapped).unwrap();

        assert_eq!(content_key.as_bytes(), unwrapped.as_bytes());
    }

    #[test]
    fn wrong_consumer_cannot_unwrap() {
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let wrong_consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let result = unwrap_key(&wrong_consumer, &wrapped);

        assert!(result.is_err());
    }

    #[test]
    fn wrapped_key_is_different_each_time() {
        let consumer = PrivateIdentity::generate(&mut OsRng);
        let content_key = AeadKey::generate(&mut OsRng);

        let wrapped1 = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();
        let wrapped2 = wrap_key(&mut OsRng, &content_key, consumer.public_identity()).unwrap();

        // Ephemeral key differs each time, so wrapped output differs.
        assert_ne!(wrapped1, wrapped2);

        // But both unwrap to the same content key.
        let k1 = unwrap_key(&consumer, &wrapped1).unwrap();
        let k2 = unwrap_key(&consumer, &wrapped2).unwrap();
        assert_eq!(k1.as_bytes(), k2.as_bytes());
    }
}

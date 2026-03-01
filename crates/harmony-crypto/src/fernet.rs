//! Reticulum-compatible Fernet token encryption.
//!
//! Token format (no version byte or timestamp, matching Reticulum):
//! ```text
//! [16 bytes IV] [N bytes AES-256-CBC ciphertext (PKCS7 padded)] [32 bytes HMAC-SHA256]
//! ```
//!
//! Key format (64 bytes total for AES-256):
//! - First 32 bytes: HMAC-SHA256 signing key
//! - Last 32 bytes: AES-256 encryption key

use aes::Aes256;
use cbc::cipher::{block_padding::Pkcs7, BlockDecryptMut, BlockEncryptMut, KeyIvInit};
use hmac::{Hmac, Mac};
use rand_core::CryptoRngCore;
use sha2::Sha256;
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use crate::CryptoError;

type Aes256CbcEnc = cbc::Encryptor<Aes256>;
type Aes256CbcDec = cbc::Decryptor<Aes256>;
type HmacSha256 = Hmac<Sha256>;

const IV_LENGTH: usize = 16;
const HMAC_LENGTH: usize = 32;
const AES256_KEY_LENGTH: usize = 32;

/// Minimum Fernet token size in bytes.
///
/// AES-CBC with PKCS7 always produces at least one 16-byte block of ciphertext
/// (even for empty plaintext, padding fills an entire block), so the minimum
/// valid token is: 16 (IV) + 16 (one AES block) + 32 (HMAC) = 64 bytes.
pub const FERNET_TOKEN_MIN: usize = IV_LENGTH + 16 + HMAC_LENGTH;

/// Encrypt plaintext using Reticulum-compatible Fernet.
///
/// `key` must be 64 bytes: `[32B signing key][32B encryption key]`.
/// Returns `[16B IV][ciphertext with PKCS7 padding][32B HMAC]`.
pub fn encrypt(
    rng: &mut impl CryptoRngCore,
    key: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    validate_key_length(key)?;
    let (signing_key, encryption_key) = split_key(key);

    // Generate random IV
    let mut iv = [0u8; IV_LENGTH];
    rng.fill_bytes(&mut iv);

    // AES-256-CBC encrypt with PKCS7 padding
    let cipher = Aes256CbcEnc::new(encryption_key.into(), &iv.into());
    let ciphertext = cipher.encrypt_padded_vec_mut::<Pkcs7>(plaintext);

    // Assemble token: IV + ciphertext
    let mut token = Vec::with_capacity(IV_LENGTH + ciphertext.len() + HMAC_LENGTH);
    token.extend_from_slice(&iv);
    token.extend_from_slice(&ciphertext);

    // HMAC-SHA256 over IV + ciphertext
    let mut mac =
        HmacSha256::new_from_slice(signing_key).expect("HMAC accepts any key length");
    mac.update(&token);
    let tag = mac.finalize().into_bytes();
    token.extend_from_slice(&tag);

    Ok(token)
}

/// Decrypt a Reticulum-compatible Fernet token.
///
/// `key` must be 64 bytes: `[32B signing key][32B encryption key]`.
/// Verifies HMAC in constant time before decrypting.
pub fn decrypt(key: &[u8], token: &[u8]) -> Result<Vec<u8>, CryptoError> {
    validate_key_length(key)?;
    let (signing_key, encryption_key) = split_key(key);

    if token.len() < FERNET_TOKEN_MIN {
        return Err(CryptoError::CiphertextTooShort);
    }

    let tag_offset = token.len() - HMAC_LENGTH;
    let data = &token[..tag_offset];
    let expected_tag = &token[tag_offset..];

    // Verify HMAC in constant time
    let mut mac =
        HmacSha256::new_from_slice(signing_key).expect("HMAC accepts any key length");
    mac.update(data);
    let computed_tag = mac.finalize().into_bytes();

    if computed_tag.as_slice().ct_eq(expected_tag).into() {
        // HMAC valid — decrypt
        let iv = &data[..IV_LENGTH];
        let ciphertext = &data[IV_LENGTH..];

        let cipher = Aes256CbcDec::new(encryption_key.into(), iv.into());
        let mut plaintext = ciphertext.to_vec();
        let result = cipher
            .decrypt_padded_mut::<Pkcs7>(&mut plaintext)
            .map_err(|_| CryptoError::DecryptionFailed)?;
        let len = result.len();
        plaintext.truncate(len);
        Ok(plaintext)
    } else {
        Err(CryptoError::HmacMismatch)
    }
}

fn validate_key_length(key: &[u8]) -> Result<(), CryptoError> {
    if key.len() != AES256_KEY_LENGTH * 2 {
        return Err(CryptoError::InvalidKeyLength {
            expected: AES256_KEY_LENGTH * 2,
            got: key.len(),
        });
    }
    Ok(())
}

fn split_key(key: &[u8]) -> (&[u8], &[u8]) {
    (&key[..AES256_KEY_LENGTH], &key[AES256_KEY_LENGTH..])
}

/// Generate a random 64-byte Fernet key.
pub fn generate_key(rng: &mut impl CryptoRngCore) -> FernetKey {
    let mut bytes = [0u8; 64];
    rng.fill_bytes(&mut bytes);
    FernetKey { bytes }
}

/// A 64-byte Fernet key that zeroizes on drop.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct FernetKey {
    bytes: [u8; 64],
}

impl FernetKey {
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn roundtrip_basic() {
        let key = generate_key(&mut OsRng);
        let plaintext = b"hello harmony network";
        let token = encrypt(&mut OsRng, key.as_bytes(), plaintext).unwrap();
        let decrypted = decrypt(key.as_bytes(), &token).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn roundtrip_empty_plaintext() {
        let key = generate_key(&mut OsRng);
        let token = encrypt(&mut OsRng, key.as_bytes(), b"").unwrap();
        let decrypted = decrypt(key.as_bytes(), &token).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn roundtrip_exact_block_size() {
        // 16 bytes = exactly one AES block, padding adds a full block
        let key = generate_key(&mut OsRng);
        let plaintext = b"0123456789abcdef";
        let token = encrypt(&mut OsRng, key.as_bytes(), plaintext).unwrap();
        let decrypted = decrypt(key.as_bytes(), &token).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn roundtrip_large_payload() {
        let key = generate_key(&mut OsRng);
        let plaintext = vec![0xABu8; 4096];
        let token = encrypt(&mut OsRng, key.as_bytes(), &plaintext).unwrap();
        let decrypted = decrypt(key.as_bytes(), &token).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn token_format_structure() {
        let key = generate_key(&mut OsRng);
        let plaintext = b"test";
        let token = encrypt(&mut OsRng, key.as_bytes(), plaintext).unwrap();

        // Token should be: 16 (IV) + 16 (one padded block for 4 bytes) + 32 (HMAC) = 64
        assert_eq!(token.len(), 64);
    }

    #[test]
    fn wrong_key_fails_hmac() {
        let key1 = generate_key(&mut OsRng);
        let key2 = generate_key(&mut OsRng);
        let token = encrypt(&mut OsRng, key1.as_bytes(), b"secret").unwrap();

        let result = decrypt(key2.as_bytes(), &token);
        assert!(matches!(result, Err(CryptoError::HmacMismatch)));
    }

    #[test]
    fn tampered_ciphertext_fails_hmac() {
        let key = generate_key(&mut OsRng);
        let mut token = encrypt(&mut OsRng, key.as_bytes(), b"secret").unwrap();

        // Flip a bit in the ciphertext (after IV, before HMAC)
        token[20] ^= 0x01;

        let result = decrypt(key.as_bytes(), &token);
        assert!(matches!(result, Err(CryptoError::HmacMismatch)));
    }

    #[test]
    fn tampered_hmac_fails() {
        let key = generate_key(&mut OsRng);
        let mut token = encrypt(&mut OsRng, key.as_bytes(), b"secret").unwrap();

        // Flip a bit in the HMAC tag
        let last = token.len() - 1;
        token[last] ^= 0x01;

        let result = decrypt(key.as_bytes(), &token);
        assert!(matches!(result, Err(CryptoError::HmacMismatch)));
    }

    #[test]
    fn truncated_token_fails() {
        let key = generate_key(&mut OsRng);
        let token = encrypt(&mut OsRng, key.as_bytes(), b"hello").unwrap();

        // Truncate to less than minimum
        let result = decrypt(key.as_bytes(), &token[..40]);
        assert!(matches!(result, Err(CryptoError::CiphertextTooShort)));
    }

    #[test]
    fn tokens_under_64_bytes_rejected() {
        // Minimum valid: 16 (IV) + 16 (one AES block) + 32 (HMAC) = 64.
        // Tokens of 49-63 bytes can never be valid AES-CBC output.
        let key = generate_key(&mut OsRng);
        for len in [49, 55, 60, 63] {
            let fake_token = vec![0u8; len];
            let result = decrypt(key.as_bytes(), &fake_token);
            assert!(
                matches!(result, Err(CryptoError::CiphertextTooShort)),
                "token of {len} bytes should be rejected as too short"
            );
        }
    }

    #[test]
    fn invalid_key_length_rejected() {
        let result = encrypt(&mut OsRng, &[0u8; 32], b"test");
        assert!(matches!(
            result,
            Err(CryptoError::InvalidKeyLength {
                expected: 64,
                got: 32
            })
        ));
    }

    #[test]
    fn different_encryptions_produce_different_tokens() {
        let key = generate_key(&mut OsRng);
        let plaintext = b"same input";
        let t1 = encrypt(&mut OsRng, key.as_bytes(), plaintext).unwrap();
        let t2 = encrypt(&mut OsRng, key.as_bytes(), plaintext).unwrap();
        // Different random IVs should produce different tokens
        assert_ne!(t1, t2);
        // But both decrypt to the same plaintext
        assert_eq!(
            decrypt(key.as_bytes(), &t1).unwrap(),
            decrypt(key.as_bytes(), &t2).unwrap()
        );
    }
}

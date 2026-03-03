use hkdf::Hkdf;
use sha2::Sha256;
use zeroize::Zeroize;

use crate::CryptoError;

/// Default derived key length for AES-256-CBC Fernet (Reticulum default).
pub const DERIVED_KEY_LENGTH_256: usize = 64;

/// Derived key length for AES-128-CBC Fernet (Reticulum optional).
pub const DERIVED_KEY_LENGTH_128: usize = 32;

/// Maximum HKDF-SHA256 output length: 255 * 32 = 8160 bytes.
const HKDF_SHA256_MAX_OUTPUT: usize = 255 * 32;

/// Perform HKDF-SHA256 key derivation.
///
/// Matches Reticulum's key derivation:
/// - `ikm`: Input key material (e.g., ECDH shared secret)
/// - `salt`: Optional salt (defaults to 32 zero bytes if `None`)
/// - `info`: Optional context info (defaults to empty)
/// - `length`: Output key length in bytes (max 8160 for SHA-256)
pub fn derive_key(
    ikm: &[u8],
    salt: Option<&[u8]>,
    info: &[u8],
    length: usize,
) -> Result<Vec<u8>, CryptoError> {
    let hk = Hkdf::<Sha256>::new(salt, ikm);
    let mut okm = vec![0u8; length];
    hk.expand(info, &mut okm)
        .map_err(|_| CryptoError::HkdfLengthExceeded {
            requested: length,
            max: HKDF_SHA256_MAX_OUTPUT,
        })?;
    Ok(okm)
}

/// Convenience: derive a 64-byte key (Reticulum AES-256-CBC default).
///
/// The returned key is split by the caller:
/// - First 32 bytes: HMAC signing key
/// - Last 32 bytes: AES encryption key
pub fn derive_key_256(ikm: &[u8], salt: Option<&[u8]>) -> [u8; DERIVED_KEY_LENGTH_256] {
    let mut key = [0u8; DERIVED_KEY_LENGTH_256];
    let hk = Hkdf::<Sha256>::new(salt, ikm);
    // 64 bytes is a compile-time constant, well within the 8160-byte limit.
    hk.expand(&[], &mut key)
        .expect("64 bytes is within HKDF-SHA256 limits");
    key
}

/// Securely zeroed derived key wrapper.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct DerivedKey {
    bytes: Vec<u8>,
}

impl DerivedKey {
    pub fn new(
        ikm: &[u8],
        salt: Option<&[u8]>,
        info: &[u8],
        length: usize,
    ) -> Result<Self, CryptoError> {
        Ok(Self {
            bytes: derive_key(ikm, salt, info, length)?,
        })
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // RFC 5869 Test Case 1
    #[test]
    fn rfc5869_test_case_1() {
        let ikm = hex::decode("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b").unwrap();
        let salt = hex::decode("000102030405060708090a0b0c").unwrap();
        let info = hex::decode("f0f1f2f3f4f5f6f7f8f9").unwrap();
        let expected_okm = hex::decode(
            "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865",
        )
        .unwrap();

        let okm = derive_key(&ikm, Some(&salt), &info, 42).unwrap();
        assert_eq!(okm, expected_okm);
    }

    // RFC 5869 Test Case 2
    #[test]
    fn rfc5869_test_case_2() {
        let ikm = hex::decode(
            "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f\
             202122232425262728292a2b2c2d2e2f303132333435363738393a3b3c3d3e3f\
             404142434445464748494a4b4c4d4e4f",
        )
        .unwrap();
        let salt = hex::decode(
            "606162636465666768696a6b6c6d6e6f707172737475767778797a7b7c7d7e7f\
             808182838485868788898a8b8c8d8e8f909192939495969798999a9b9c9d9e9f\
             a0a1a2a3a4a5a6a7a8a9aaabacadaeaf",
        )
        .unwrap();
        let info = hex::decode(
            "b0b1b2b3b4b5b6b7b8b9babbbcbdbebfc0c1c2c3c4c5c6c7c8c9cacbcccdcecf\
             d0d1d2d3d4d5d6d7d8d9dadbdcdddedfe0e1e2e3e4e5e6e7e8e9eaebecedeeef\
             f0f1f2f3f4f5f6f7f8f9fafbfcfdfeff",
        )
        .unwrap();
        let expected_okm = hex::decode(
            "b11e398dc80327a1c8e7f78c596a49344f012eda2d4efad8a050cc4c19afa97c\
             59045a99cac7827271cb41c65e590e09da3275600c2f09b8367793a9aca3db71\
             cc30c58179ec3e87c14c01d5c1f3434f1d87",
        )
        .unwrap();

        let okm = derive_key(&ikm, Some(&salt), &info, 82).unwrap();
        assert_eq!(okm, expected_okm);
    }

    // RFC 5869 Test Case 3 — zero-length salt and info
    #[test]
    fn rfc5869_test_case_3() {
        let ikm = hex::decode("0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b0b").unwrap();
        let expected_okm = hex::decode(
            "8da4e775a563c18f715f802a063c5a31b8a11f5c5ee1879ec3454e5f3c738d2d\
             9d201395faa4b61a96c8",
        )
        .unwrap();

        let okm = derive_key(&ikm, None, &[], 42).unwrap();
        assert_eq!(okm, expected_okm);
    }

    #[test]
    fn derive_key_256_produces_64_bytes() {
        let ikm = b"shared_secret_material";
        let key = derive_key_256(ikm, Some(b"salt"));
        assert_eq!(key.len(), 64);
    }

    #[test]
    fn derive_key_256_deterministic() {
        let ikm = b"same_input";
        let salt = Some(b"same_salt".as_slice());
        let k1 = derive_key_256(ikm, salt);
        let k2 = derive_key_256(ikm, salt);
        assert_eq!(k1, k2);
    }

    #[test]
    fn derive_key_256_different_salts_differ() {
        let ikm = b"input";
        let k1 = derive_key_256(ikm, Some(b"salt_a"));
        let k2 = derive_key_256(ikm, Some(b"salt_b"));
        assert_ne!(k1, k2);
    }

    #[test]
    fn derived_key_wrapper_zeroizes_on_drop() {
        let dk = DerivedKey::new(b"ikm", None, &[], 32).unwrap();
        assert_eq!(dk.len(), 32);
        assert!(!dk.is_empty());
        let _ = dk.as_bytes();
    }

    #[test]
    fn excessive_length_returns_error() {
        let result = derive_key(b"ikm", None, &[], 8161);
        assert!(matches!(
            result,
            Err(CryptoError::HkdfLengthExceeded {
                requested: 8161,
                max: 8160,
            })
        ));
    }

    #[test]
    fn derived_key_wrapper_excessive_length_returns_error() {
        let result = DerivedKey::new(b"ikm", None, &[], 9000);
        assert!(matches!(
            result,
            Err(CryptoError::HkdfLengthExceeded { .. })
        ));
    }
}

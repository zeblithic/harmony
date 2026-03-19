//! ML-DSA-65 (FIPS 204) post-quantum digital signature scheme.
//!
//! Wraps the `ml-dsa` crate to provide a Harmony-idiomatic API with:
//! - `CryptoError` integration
//! - `zeroize` for secret material
//! - Fixed-size byte accessors

use alloc::vec::Vec;
use core::convert::TryFrom;

use ml_dsa::{
    signature::{Signer, Verifier},
    EncodedVerifyingKey, KeyGen, MlDsa65,
};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::CryptoError;

/// ML-DSA-65 verifying (public) key length in bytes.
pub const PK_LENGTH: usize = 1952;

/// ML-DSA-65 signing (secret) key seed length in bytes.
///
/// This is the 32-byte seed form, which is the preferred serialization
/// for signing keys in ML-DSA (FIPS 204).
pub const SK_LENGTH: usize = 32;

/// ML-DSA-65 signature length in bytes.
pub const SIG_LENGTH: usize = 3309;

/// ML-DSA-65 verifying (public) key.
#[derive(Clone)]
pub struct MlDsaPublicKey {
    inner: ml_dsa::VerifyingKey<MlDsa65>,
}

impl MlDsaPublicKey {
    /// Serialize to bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.encode().to_vec()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != PK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = EncodedVerifyingKey::<MlDsa65>::try_from(bytes).map_err(|_| {
            CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            }
        })?;
        let inner = ml_dsa::VerifyingKey::<MlDsa65>::decode(&arr);
        Ok(Self { inner })
    }
}

impl core::fmt::Debug for MlDsaPublicKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        write!(f, "MlDsaPublicKey({:02x}{:02x}..)", bytes[0], bytes[1])
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for MlDsaPublicKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.as_bytes())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MlDsaPublicKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: alloc::vec::Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

/// ML-DSA-65 signing (secret) key.
///
/// Stores the 32-byte seed from which the full signing key is derived.
/// The seed is zeroized on drop to protect key material.
pub struct MlDsaSecretKey {
    seed: [u8; SK_LENGTH],
}

impl Drop for MlDsaSecretKey {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

impl MlDsaSecretKey {
    /// Serialize to bytes (32-byte seed form).
    pub fn as_bytes(&self) -> Vec<u8> {
        self.seed.to_vec()
    }

    /// Deserialize from bytes (32-byte seed form).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            });
        }
        let mut seed = [0u8; SK_LENGTH];
        seed.copy_from_slice(bytes);
        Ok(Self { seed })
    }

    /// Derive the full `ml_dsa::SigningKey` from the stored seed.
    fn to_signing_key(&self) -> ml_dsa::SigningKey<MlDsa65> {
        let mut seed_arr = ml_dsa::B32::from(self.seed);
        let sk = ml_dsa::SigningKey::<MlDsa65>::from_seed(&seed_arr);
        seed_arr.zeroize();
        sk
    }

    /// Derive the corresponding public (verifying) key.
    pub fn public_key(&self) -> MlDsaPublicKey {
        let signing_key = self.to_signing_key();
        MlDsaPublicKey {
            inner: signing_key.verifying_key().clone(),
        }
    }
}

/// ML-DSA-65 signature.
#[derive(Clone)]
pub struct MlDsaSignature {
    bytes: Vec<u8>,
}

impl MlDsaSignature {
    /// Access signature bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SIG_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SIG_LENGTH,
                got: bytes.len(),
            });
        }
        // Validate by attempting to decode
        let _sig = ml_dsa::Signature::<MlDsa65>::try_from(bytes)
            .map_err(|_| CryptoError::MlDsaVerifyFailed)?;
        Ok(Self {
            bytes: bytes.to_vec(),
        })
    }
}

impl core::fmt::Debug for MlDsaSignature {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "MlDsaSignature({}B)", self.bytes.len())
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for MlDsaSignature {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.bytes)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MlDsaSignature {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes: alloc::vec::Vec<u8> = serde::Deserialize::deserialize(deserializer)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

/// Generate an ML-DSA-65 keypair.
///
/// Uses the provided RNG to generate a 32-byte seed, then derives
/// the keypair deterministically from it.
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlDsaPublicKey, MlDsaSecretKey) {
    let mut seed_bytes = [0u8; SK_LENGTH];
    rng.fill_bytes(&mut seed_bytes);
    let mut seed_arr = ml_dsa::B32::from(seed_bytes);
    let kp = MlDsa65::from_seed(&seed_arr);
    let pk = MlDsaPublicKey {
        inner: kp.verifying_key().clone(),
    };
    let sk = MlDsaSecretKey { seed: seed_bytes };
    seed_arr.zeroize();
    seed_bytes.zeroize();
    (pk, sk)
}

/// Sign a message using the given secret key.
///
/// Uses the deterministic variant of ML-DSA signing with an empty context string.
pub fn sign(sk: &MlDsaSecretKey, message: &[u8]) -> Result<MlDsaSignature, CryptoError> {
    let signing_key = sk.to_signing_key();
    let sig = signing_key
        .try_sign(message)
        .map_err(|_| CryptoError::MlDsaSignFailed)?;
    Ok(MlDsaSignature {
        bytes: sig.encode().to_vec(),
    })
}

/// Verify a signature against a message and public key.
pub fn verify(
    pk: &MlDsaPublicKey,
    message: &[u8],
    signature: &MlDsaSignature,
) -> Result<(), CryptoError> {
    let sig = ml_dsa::Signature::<MlDsa65>::try_from(signature.bytes.as_slice())
        .map_err(|_| CryptoError::MlDsaVerifyFailed)?;
    pk.inner
        .verify(message, &sig)
        .map_err(|_| CryptoError::MlDsaVerifyFailed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn keygen_produces_valid_keys() {
        let (pk, sk) = generate(&mut OsRng);
        assert_eq!(pk.as_bytes().len(), PK_LENGTH);
        assert_eq!(sk.as_bytes().len(), SK_LENGTH);
    }

    #[test]
    fn sign_verify_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let msg = b"hello post-quantum world";
        let sig = sign(&sk, msg).unwrap();
        assert_eq!(sig.as_bytes().len(), SIG_LENGTH);
        assert!(verify(&pk, msg, &sig).is_ok());
    }

    #[test]
    fn verify_wrong_message_fails() {
        let (pk, sk) = generate(&mut OsRng);
        let sig = sign(&sk, b"correct message").unwrap();
        assert!(verify(&pk, b"wrong message", &sig).is_err());
    }

    #[test]
    fn verify_wrong_key_fails() {
        let (_pk1, sk1) = generate(&mut OsRng);
        let (pk2, _sk2) = generate(&mut OsRng);
        let sig = sign(&sk1, b"message").unwrap();
        assert!(verify(&pk2, b"message", &sig).is_err());
    }

    #[test]
    fn public_key_serialization_roundtrip() {
        let (pk, _sk) = generate(&mut OsRng);
        let bytes = pk.as_bytes();
        let pk2 = MlDsaPublicKey::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), pk2.as_bytes());
    }

    #[test]
    fn secret_key_serialization_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let sk_bytes = sk.as_bytes();
        let sk2 = MlDsaSecretKey::from_bytes(&sk_bytes).unwrap();
        let sig = sign(&sk2, b"test").unwrap();
        assert!(verify(&pk, b"test", &sig).is_ok());
    }

    #[test]
    fn signature_serialization_roundtrip() {
        let (_pk, sk) = generate(&mut OsRng);
        let sig = sign(&sk, b"data").unwrap();
        let sig_bytes = sig.as_bytes();
        let sig2 = MlDsaSignature::from_bytes(sig_bytes).unwrap();
        assert_eq!(sig.as_bytes(), sig2.as_bytes());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn public_key_serde_round_trip() {
        let (pk, _sk) = generate(&mut rand::rngs::OsRng);
        let bytes = postcard::to_allocvec(&pk).unwrap();
        let decoded: MlDsaPublicKey = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), decoded.as_bytes());
    }

    #[cfg(feature = "serde")]
    #[test]
    fn signature_serde_round_trip() {
        let (pk, sk) = generate(&mut rand::rngs::OsRng);
        let sig = sign(&sk, b"test message").unwrap();
        let bytes = postcard::to_allocvec(&sig).unwrap();
        let decoded: MlDsaSignature = postcard::from_bytes(&bytes).unwrap();
        assert!(verify(&pk, b"test message", &decoded).is_ok());
    }
}

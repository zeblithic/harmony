//! ML-KEM-768 (FIPS 203) post-quantum key encapsulation mechanism.
//!
//! Wraps the `ml-kem` crate to provide a Harmony-idiomatic API with:
//! - `CryptoError` integration
//! - `zeroize` for secret material
//! - Fixed-size byte accessors

use alloc::vec::Vec;

use ml_kem::{array::Array, Decapsulate, KeyExport, MlKem768};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::CryptoError;

/// ML-KEM-768 encapsulation key (public key) length in bytes.
pub const PK_LENGTH: usize = 1184;

/// ML-KEM-768 decapsulation key (secret key) seed length in bytes.
///
/// This is the 64-byte seed form, which is the preferred serialization
/// for decapsulation keys in ML-KEM (FIPS 203).
pub const SK_LENGTH: usize = 64;

/// ML-KEM-768 ciphertext length in bytes.
pub const CT_LENGTH: usize = 1088;

/// ML-KEM-768 shared secret length in bytes.
pub const SS_LENGTH: usize = 32;

/// ML-KEM-768 encapsulation (public) key.
#[derive(Clone)]
pub struct MlKemPublicKey {
    inner: ml_kem::EncapsulationKey<MlKem768>,
}

impl MlKemPublicKey {
    /// Serialize to bytes.
    pub fn as_bytes(&self) -> Vec<u8> {
        self.inner.to_bytes().to_vec()
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != PK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: PK_LENGTH,
                got: bytes.len(),
            });
        }
        let arr = Array::try_from(bytes).map_err(|_| CryptoError::InvalidKeyLength {
            expected: PK_LENGTH,
            got: bytes.len(),
        })?;
        let inner = ml_kem::EncapsulationKey::<MlKem768>::new(&arr)
            .map_err(|_| CryptoError::MlKemEncapsulationFailed)?;
        Ok(Self { inner })
    }
}

impl core::fmt::Debug for MlKemPublicKey {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let bytes = self.as_bytes();
        write!(f, "MlKemPublicKey({:02x}{:02x}..)", bytes[0], bytes[1])
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for MlKemPublicKey {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_bytes(&self.as_bytes())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for MlKemPublicKey {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let bytes = deserializer.deserialize_byte_buf(crate::serde_helpers::BytesOrSeqVisitor)?;
        Self::from_bytes(&bytes).map_err(serde::de::Error::custom)
    }
}

/// ML-KEM-768 decapsulation (secret) key.
///
/// The inner `DecapsulationKey` implements `ZeroizeOnDrop` when the
/// `zeroize` feature is enabled (which it is in this workspace).
pub struct MlKemSecretKey {
    inner: ml_kem::DecapsulationKey<MlKem768>,
}

impl MlKemSecretKey {
    /// Serialize to bytes (64-byte seed form).
    pub fn as_bytes(&self) -> Vec<u8> {
        let mut arr = self.inner.to_bytes();
        let v = arr.to_vec();
        arr.zeroize();
        v
    }

    /// Deserialize from bytes (64-byte seed form).
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != SK_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: SK_LENGTH,
                got: bytes.len(),
            });
        }
        let mut seed = Array::try_from(bytes).map_err(|_| CryptoError::InvalidKeyLength {
            expected: SK_LENGTH,
            got: bytes.len(),
        })?;
        let inner = ml_kem::DecapsulationKey::<MlKem768>::from_seed(seed);
        seed.zeroize();
        Ok(Self { inner })
    }

    /// Derive the corresponding public (encapsulation) key.
    pub fn public_key(&self) -> MlKemPublicKey {
        MlKemPublicKey {
            inner: self.inner.encapsulation_key().clone(),
        }
    }
}

/// ML-KEM-768 ciphertext.
pub struct MlKemCiphertext {
    bytes: Vec<u8>,
}

impl MlKemCiphertext {
    /// Access ciphertext bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Create from raw bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CryptoError> {
        if bytes.len() != CT_LENGTH {
            return Err(CryptoError::InvalidKeyLength {
                expected: CT_LENGTH,
                got: bytes.len(),
            });
        }
        Ok(Self {
            bytes: bytes.to_vec(),
        })
    }
}

/// ML-KEM-768 shared secret (32 bytes).
///
/// Zeroized on drop to protect key material.
#[derive(Zeroize)]
#[zeroize(drop)]
pub struct MlKemSharedSecret {
    bytes: [u8; SS_LENGTH],
}

impl MlKemSharedSecret {
    /// Access shared secret bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Generate an ML-KEM-768 keypair.
///
/// Uses the provided RNG to generate a 64-byte seed, then derives
/// the keypair deterministically from it.
pub fn generate(rng: &mut impl CryptoRngCore) -> (MlKemPublicKey, MlKemSecretKey) {
    let mut seed_bytes = [0u8; SK_LENGTH];
    rng.fill_bytes(&mut seed_bytes);
    let mut seed = Array::from(seed_bytes);
    let dk = ml_kem::DecapsulationKey::<MlKem768>::from_seed(seed);
    let ek = dk.encapsulation_key().clone();
    seed_bytes.zeroize();
    seed.zeroize();
    (MlKemPublicKey { inner: ek }, MlKemSecretKey { inner: dk })
}

/// Encapsulate a shared secret using the given public key.
///
/// Returns the ciphertext and the shared secret. Uses the provided RNG
/// to generate the encapsulation randomness.
pub fn encapsulate(
    rng: &mut impl CryptoRngCore,
    pk: &MlKemPublicKey,
) -> Result<(MlKemCiphertext, MlKemSharedSecret), CryptoError> {
    let mut m = ml_kem::B32::default();
    rng.fill_bytes(m.as_mut_slice());
    let (ct, ss) = pk.inner.encapsulate_deterministic(&m);
    m.zeroize();
    let ct = MlKemCiphertext {
        bytes: ct.as_slice().to_vec(),
    };
    let mut ss_bytes = [0u8; SS_LENGTH];
    ss_bytes.copy_from_slice(ss.as_slice());
    Ok((ct, MlKemSharedSecret { bytes: ss_bytes }))
}

/// Decapsulate a shared secret using the given secret key and ciphertext.
///
/// ML-KEM uses implicit rejection: decapsulating with the wrong key produces
/// a valid-looking but incorrect shared secret rather than returning an error.
pub fn decapsulate(
    sk: &MlKemSecretKey,
    ct: &MlKemCiphertext,
) -> Result<MlKemSharedSecret, CryptoError> {
    let ct_arr =
        Array::try_from(ct.bytes.as_slice()).map_err(|_| CryptoError::MlKemDecapsulationFailed)?;
    let ss = sk.inner.decapsulate(&ct_arr);
    let mut ss_bytes = [0u8; SS_LENGTH];
    ss_bytes.copy_from_slice(ss.as_slice());
    Ok(MlKemSharedSecret { bytes: ss_bytes })
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
    fn encapsulate_decapsulate_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let (ct, ss_sender) = encapsulate(&mut OsRng, &pk).unwrap();
        let ss_receiver = decapsulate(&sk, &ct).unwrap();
        assert_eq!(ss_sender.as_bytes(), ss_receiver.as_bytes());
    }

    #[test]
    fn wrong_secret_key_fails_decapsulation() {
        let (pk, _sk1) = generate(&mut OsRng);
        let (_pk2, sk2) = generate(&mut OsRng);
        let (ct, ss_sender) = encapsulate(&mut OsRng, &pk).unwrap();
        // ML-KEM decapsulation with wrong key produces a different shared secret
        // (implicit rejection), it does not return an error
        let ss_wrong = decapsulate(&sk2, &ct).unwrap();
        assert_ne!(ss_sender.as_bytes(), ss_wrong.as_bytes());
    }

    #[test]
    fn public_key_serialization_roundtrip() {
        let (pk, _sk) = generate(&mut OsRng);
        let bytes = pk.as_bytes();
        let pk2 = MlKemPublicKey::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), pk2.as_bytes());
    }

    #[test]
    fn secret_key_serialization_roundtrip() {
        let (pk, sk) = generate(&mut OsRng);
        let sk_bytes = sk.as_bytes();
        let sk2 = MlKemSecretKey::from_bytes(&sk_bytes).unwrap();
        // Verify round-trip by encapsulating with original pk, decapsulating with restored sk
        let (ct, ss1) = encapsulate(&mut OsRng, &pk).unwrap();
        let ss2 = decapsulate(&sk2, &ct).unwrap();
        assert_eq!(ss1.as_bytes(), ss2.as_bytes());
    }

    #[test]
    fn shared_secret_is_32_bytes() {
        let (pk, sk) = generate(&mut OsRng);
        let (ct, ss) = encapsulate(&mut OsRng, &pk).unwrap();
        assert_eq!(ss.as_bytes().len(), 32);
        let ss2 = decapsulate(&sk, &ct).unwrap();
        assert_eq!(ss2.as_bytes().len(), 32);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn public_key_serde_round_trip() {
        let (pk, _sk) = generate(&mut rand::rngs::OsRng);
        let bytes = postcard::to_allocvec(&pk).unwrap();
        let decoded: MlKemPublicKey = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(pk.as_bytes(), decoded.as_bytes());
    }
}

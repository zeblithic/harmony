//! Post-quantum identity types using ML-KEM-768 + ML-DSA-65.
//!
//! These types parallel the classical `Identity`/`PrivateIdentity` (which use
//! X25519 + Ed25519) but use post-quantum algorithms from FIPS 203 and FIPS 204.
//!
//! The address format is identical: `SHA256(enc_pub || sign_pub)[:16]` — the
//! same 128-bit truncated hash, just with different input key material.
//!
//! The classical types remain unchanged; this module adds PQC alternatives.

use alloc::vec::Vec;
use core::fmt;

use harmony_crypto::hash;
use harmony_crypto::ml_dsa::{self, MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature};
use harmony_crypto::ml_kem::{self, MlKemPublicKey, MlKemSecretKey};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::IdentityError;

/// Length of the address hash in bytes (same as classical identity).
pub const ADDRESS_HASH_LENGTH: usize = hash::TRUNCATED_HASH_LENGTH; // 16

/// Length of the combined PQ public key in bytes:
/// ML-KEM-768 encapsulation key (1184) + ML-DSA-65 verifying key (1952).
pub const PQ_PUBLIC_KEY_LENGTH: usize = ml_kem::PK_LENGTH + ml_dsa::PK_LENGTH; // 3136

/// Length of the combined PQ private key in bytes (seed forms):
/// ML-KEM-768 decapsulation key seed (64) + ML-DSA-65 signing key seed (32).
pub const PQ_PRIVATE_KEY_LENGTH: usize = ml_kem::SK_LENGTH + ml_dsa::SK_LENGTH; // 96

/// A post-quantum public identity: ML-KEM-768 encapsulation key + ML-DSA-65 verifying key.
///
/// This is the information shared with peers. The address hash derived from
/// these keys serves as the node's network address.
#[derive(Clone)]
pub struct PqIdentity {
    /// ML-KEM-768 encapsulation (public) key for key encapsulation.
    pub encryption_key: MlKemPublicKey,
    /// ML-DSA-65 verifying (public) key for signature verification.
    pub verifying_key: MlDsaPublicKey,
    /// Truncated SHA-256 hash of the combined public keys (16 bytes).
    pub address_hash: [u8; ADDRESS_HASH_LENGTH],
}

impl fmt::Debug for PqIdentity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PqIdentity")
            .field("address_hash", &self.address_hash)
            .finish_non_exhaustive()
    }
}

impl PqIdentity {
    /// Construct a PQ identity from public keys, deriving the address hash.
    pub fn from_public_keys(encryption_key: MlKemPublicKey, verifying_key: MlDsaPublicKey) -> Self {
        let mut combined = Vec::with_capacity(PQ_PUBLIC_KEY_LENGTH);
        combined.extend_from_slice(&encryption_key.as_bytes());
        combined.extend_from_slice(&verifying_key.as_bytes());
        let address_hash = hash::truncated_hash(&combined);

        Self {
            encryption_key,
            verifying_key,
            address_hash,
        }
    }

    /// Serialize to bytes: `[ML-KEM-768 pub (1184)][ML-DSA-65 pub (1952)]`.
    pub fn to_public_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PQ_PUBLIC_KEY_LENGTH);
        bytes.extend_from_slice(&self.encryption_key.as_bytes());
        bytes.extend_from_slice(&self.verifying_key.as_bytes());
        bytes
    }

    /// Deserialize from bytes: `[ML-KEM-768 pub (1184)][ML-DSA-65 pub (1952)]`.
    pub fn from_public_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        if bytes.len() != PQ_PUBLIC_KEY_LENGTH {
            return Err(IdentityError::InvalidPqPublicKeyLength(bytes.len()));
        }

        let kem_bytes = &bytes[..ml_kem::PK_LENGTH];
        let dsa_bytes = &bytes[ml_kem::PK_LENGTH..];

        let encryption_key =
            MlKemPublicKey::from_bytes(kem_bytes).map_err(IdentityError::Crypto)?;
        let verifying_key = MlDsaPublicKey::from_bytes(dsa_bytes).map_err(IdentityError::Crypto)?;

        Ok(Self::from_public_keys(encryption_key, verifying_key))
    }

    /// Verify an ML-DSA-65 signature against this identity.
    pub fn verify(&self, message: &[u8], signature: &[u8]) -> Result<(), IdentityError> {
        let sig = MlDsaSignature::from_bytes(signature).map_err(IdentityError::Crypto)?;
        ml_dsa::verify(&self.verifying_key, message, &sig).map_err(IdentityError::Crypto)
    }

    /// Encrypt plaintext to this identity using ML-KEM-768 + ChaCha20-Poly1305.
    ///
    /// Wire format: `[1088B ML-KEM ciphertext][12B nonce][encrypted data + 16B tag]`
    pub fn encrypt(
        &self,
        rng: &mut impl CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, IdentityError> {
        // 1. ML-KEM encapsulate → shared secret
        let (ct, ss) =
            harmony_crypto::ml_kem::encapsulate(rng, &self.encryption_key).map_err(IdentityError::Crypto)?;

        // 2. HKDF-SHA256 → symmetric key
        let key_bytes = harmony_crypto::hkdf::derive_key(
            ss.as_bytes(),
            Some(&self.address_hash),
            b"harmony-pq-encrypt-v1",
            32,
        )
        .map_err(IdentityError::Crypto)?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // 3. ChaCha20-Poly1305 encrypt
        let nonce = harmony_crypto::aead::generate_nonce(rng);
        let encrypted =
            harmony_crypto::aead::encrypt(&key, &nonce, plaintext, &[]).map_err(IdentityError::Crypto)?;

        // Zeroize key material
        key.zeroize();

        // 4. Assemble wire format: [ML-KEM ct][nonce][encrypted+tag]
        let mut result =
            Vec::with_capacity(harmony_crypto::ml_kem::CT_LENGTH + 12 + encrypted.len());
        result.extend_from_slice(ct.as_bytes());
        result.extend_from_slice(&nonce);
        result.extend_from_slice(&encrypted);
        Ok(result)
    }
}

impl PartialEq for PqIdentity {
    fn eq(&self, other: &Self) -> bool {
        self.address_hash == other.address_hash
    }
}

impl Eq for PqIdentity {}

/// A full post-quantum identity with private keys, capable of signing.
///
/// Secret key material is zeroized on drop via the underlying
/// `MlKemSecretKey` and `MlDsaSecretKey` types.
pub struct PqPrivateIdentity {
    /// The public portion of this identity.
    identity: PqIdentity,
    /// ML-KEM-768 decapsulation (secret) key — zeroized on drop.
    encryption_secret: MlKemSecretKey,
    /// ML-DSA-65 signing (secret) key — zeroized on drop.
    signing_key: MlDsaSecretKey,
}

impl PqPrivateIdentity {
    /// Generate a new random PQ identity.
    pub fn generate(rng: &mut impl CryptoRngCore) -> Self {
        let (encryption_key, encryption_secret) = ml_kem::generate(rng);
        let (verifying_key, signing_key) = ml_dsa::generate(rng);

        let identity = PqIdentity::from_public_keys(encryption_key, verifying_key);

        Self {
            identity,
            encryption_secret,
            signing_key,
        }
    }

    /// Get the public identity.
    pub fn public_identity(&self) -> &PqIdentity {
        &self.identity
    }

    /// Sign a message with ML-DSA-65.
    pub fn sign(&self, message: &[u8]) -> Result<Vec<u8>, IdentityError> {
        let sig = ml_dsa::sign(&self.signing_key, message).map_err(IdentityError::Crypto)?;
        Ok(sig.as_bytes().to_vec())
    }

    /// Serialize the private key seeds to bytes:
    /// `[ML-KEM-768 seed (64)][ML-DSA-65 seed (32)]`.
    pub fn to_private_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PQ_PRIVATE_KEY_LENGTH);
        bytes.extend_from_slice(&self.encryption_secret.as_bytes());
        bytes.extend_from_slice(&self.signing_key.as_bytes());
        bytes
    }

    /// Deserialize from private key seed bytes:
    /// `[ML-KEM-768 seed (64)][ML-DSA-65 seed (32)]`.
    pub fn from_private_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        if bytes.len() != PQ_PRIVATE_KEY_LENGTH {
            return Err(IdentityError::InvalidPqPrivateKeyLength(bytes.len()));
        }

        let kem_seed = &bytes[..ml_kem::SK_LENGTH];
        let dsa_seed = &bytes[ml_kem::SK_LENGTH..];

        let encryption_secret =
            MlKemSecretKey::from_bytes(kem_seed).map_err(IdentityError::Crypto)?;
        let signing_key = MlDsaSecretKey::from_bytes(dsa_seed).map_err(IdentityError::Crypto)?;

        // Derive public keys from secret key seeds.
        let encryption_key = encryption_secret.public_key();
        let verifying_key = signing_key.public_key();

        let identity = PqIdentity::from_public_keys(encryption_key, verifying_key);

        Ok(Self {
            identity,
            encryption_secret,
            signing_key,
        })
    }

    /// Access the ML-KEM-768 secret key (for encryption/decryption operations).
    pub fn encryption_secret(&self) -> &MlKemSecretKey {
        &self.encryption_secret
    }

    /// Decrypt ciphertext encrypted to this identity.
    ///
    /// Expects wire format: `[1088B ML-KEM ciphertext][12B nonce][encrypted data + 16B tag]`
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, IdentityError> {
        let min_len = harmony_crypto::ml_kem::CT_LENGTH + 12 + 16; // ct + nonce + tag
        if ciphertext.len() < min_len {
            return Err(IdentityError::DecryptionFailed);
        }

        // 1. Extract ML-KEM ciphertext
        let ct = harmony_crypto::ml_kem::MlKemCiphertext::from_bytes(
            &ciphertext[..harmony_crypto::ml_kem::CT_LENGTH],
        )
        .map_err(IdentityError::Crypto)?;

        // 2. Decapsulate → shared secret
        let ss = harmony_crypto::ml_kem::decapsulate(&self.encryption_secret, &ct)
            .map_err(IdentityError::Crypto)?;

        // 3. HKDF-SHA256 → symmetric key
        let key_bytes = harmony_crypto::hkdf::derive_key(
            ss.as_bytes(),
            Some(&self.identity.address_hash),
            b"harmony-pq-encrypt-v1",
            32,
        )
        .map_err(IdentityError::Crypto)?;

        let mut key = [0u8; 32];
        key.copy_from_slice(&key_bytes);

        // 4. Extract nonce and decrypt
        let nonce_start = harmony_crypto::ml_kem::CT_LENGTH;
        let mut nonce = [0u8; 12];
        nonce.copy_from_slice(&ciphertext[nonce_start..nonce_start + 12]);

        let encrypted_data = &ciphertext[nonce_start + 12..];
        let result = harmony_crypto::aead::decrypt(&key, &nonce, encrypted_data, &[])
            .map_err(|_| IdentityError::DecryptionFailed);

        // Zeroize key material
        key.zeroize();

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn address_is_16_bytes() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let identity = priv_id.public_identity();
        assert_eq!(identity.address_hash.len(), ADDRESS_HASH_LENGTH);
    }

    #[test]
    fn address_derived_from_public_keys() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let id = priv_id.public_identity();
        // Re-derive address manually
        let mut combined = Vec::new();
        combined.extend_from_slice(&id.encryption_key.as_bytes());
        combined.extend_from_slice(&id.verifying_key.as_bytes());
        let expected = harmony_crypto::hash::truncated_hash(&combined);
        assert_eq!(id.address_hash, expected);
    }

    #[test]
    fn public_bytes_roundtrip() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let id = priv_id.public_identity();
        let bytes = id.to_public_bytes();
        assert_eq!(bytes.len(), PQ_PUBLIC_KEY_LENGTH);
        let id2 = PqIdentity::from_public_bytes(&bytes).unwrap();
        assert_eq!(id.address_hash, id2.address_hash);
    }

    #[test]
    fn different_keypairs_produce_different_addresses() {
        let id1 = PqPrivateIdentity::generate(&mut OsRng);
        let id2 = PqPrivateIdentity::generate(&mut OsRng);
        assert_ne!(
            id1.public_identity().address_hash,
            id2.public_identity().address_hash
        );
    }

    #[test]
    fn sign_verify_roundtrip() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let msg = b"post-quantum identity test";
        let sig = priv_id.sign(msg).unwrap();
        assert!(priv_id.public_identity().verify(msg, &sig).is_ok());
    }

    #[test]
    fn verify_wrong_message_fails() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let sig = priv_id.sign(b"correct").unwrap();
        assert!(priv_id.public_identity().verify(b"wrong", &sig).is_err());
    }

    #[test]
    fn private_bytes_roundtrip() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let priv_bytes = priv_id.to_private_bytes();
        assert_eq!(priv_bytes.len(), PQ_PRIVATE_KEY_LENGTH);

        let restored = PqPrivateIdentity::from_private_bytes(&priv_bytes).unwrap();
        assert_eq!(
            restored.public_identity().address_hash,
            priv_id.public_identity().address_hash
        );
        assert_eq!(
            restored.public_identity().to_public_bytes(),
            priv_id.public_identity().to_public_bytes()
        );
    }

    #[test]
    fn private_bytes_roundtrip_sign_verify() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let priv_bytes = priv_id.to_private_bytes();
        let restored = PqPrivateIdentity::from_private_bytes(&priv_bytes).unwrap();

        // Sign with restored key, verify with original public identity
        let msg = b"roundtrip signing test";
        let sig = restored.sign(msg).unwrap();
        assert!(priv_id.public_identity().verify(msg, &sig).is_ok());
    }

    #[test]
    fn invalid_public_key_length_rejected() {
        assert!(matches!(
            PqIdentity::from_public_bytes(&[0u8; 100]),
            Err(IdentityError::InvalidPqPublicKeyLength(100))
        ));
    }

    #[test]
    fn invalid_private_key_length_rejected() {
        assert!(matches!(
            PqPrivateIdentity::from_private_bytes(&[0u8; 50]),
            Err(IdentityError::InvalidPqPrivateKeyLength(50))
        ));
    }

    #[test]
    fn identity_equality_based_on_address_hash() {
        let priv_id = PqPrivateIdentity::generate(&mut OsRng);
        let pub_bytes = priv_id.public_identity().to_public_bytes();
        let restored = PqIdentity::from_public_bytes(&pub_bytes).unwrap();
        assert_eq!(priv_id.public_identity(), &restored);
    }

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let recipient = PqPrivateIdentity::generate(&mut OsRng);
        let plaintext = b"secret post-quantum message";

        let ciphertext = recipient
            .public_identity()
            .encrypt(&mut OsRng, plaintext)
            .unwrap();
        let decrypted = recipient.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn wrong_recipient_cannot_decrypt() {
        let recipient = PqPrivateIdentity::generate(&mut OsRng);
        let wrong = PqPrivateIdentity::generate(&mut OsRng);
        let ciphertext = recipient
            .public_identity()
            .encrypt(&mut OsRng, b"secret")
            .unwrap();
        // Decryption with wrong key should fail (AEAD tag verification fails)
        assert!(wrong.decrypt(&ciphertext).is_err());
    }

    #[test]
    fn ciphertext_format() {
        let recipient = PqPrivateIdentity::generate(&mut OsRng);
        let ciphertext = recipient
            .public_identity()
            .encrypt(&mut OsRng, b"test")
            .unwrap();
        // Format: [1088B ML-KEM ct][12B nonce][ciphertext + 16B tag]
        assert!(ciphertext.len() >= harmony_crypto::ml_kem::CT_LENGTH + 12 + 16);
    }
}

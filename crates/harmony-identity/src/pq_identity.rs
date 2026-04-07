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

use crate::ucan::{CapabilityType, PqUcanToken, UcanError, MAX_RESOURCE_SIZE};
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
        let (ct, ss) = harmony_crypto::ml_kem::encapsulate(rng, &self.encryption_key)
            .map_err(IdentityError::Crypto)?;

        // 2. HKDF-SHA256 → symmetric key
        let mut key_bytes = harmony_crypto::hkdf::derive_key(
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
        let result = harmony_crypto::aead::encrypt(&key, &nonce, plaintext, &[])
            .map_err(IdentityError::Crypto);

        // Zeroize key material before any return (including error path)
        key.zeroize();
        key_bytes.zeroize();

        let encrypted = result?;

        // 4. Assemble wire format: [ML-KEM ct][nonce][encrypted+tag]
        let mut wire = Vec::with_capacity(harmony_crypto::ml_kem::CT_LENGTH + 12 + encrypted.len());
        wire.extend_from_slice(ct.as_bytes());
        wire.extend_from_slice(&nonce);
        wire.extend_from_slice(&encrypted);
        Ok(wire)
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

    /// Generate a new random PQ identity with a proof-of-work puzzle.
    ///
    /// Wraps [`generate()`](Self::generate) and then solves an Argon2id
    /// Hashcash puzzle bound to the public key material. At production
    /// difficulty this takes ~6.4 seconds.
    pub fn generate_with_proof(
        rng: &mut impl CryptoRngCore,
        params: &crate::puzzle::PuzzleParams,
    ) -> (Self, crate::puzzle::IdentityProof) {
        let identity = Self::generate(rng);
        let pub_bytes = identity.identity.to_public_bytes();
        let proof = crate::puzzle::solve(&pub_bytes, params, rng);
        (identity, proof)
    }

    /// Verify that a proof-of-work is valid for this identity.
    pub fn verify_proof(
        &self,
        proof: &crate::puzzle::IdentityProof,
        params: &crate::puzzle::PuzzleParams,
    ) -> bool {
        let pub_bytes = self.identity.to_public_bytes();
        crate::puzzle::verify(&pub_bytes, proof, params)
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

        let mut kem_seed = self.encryption_secret.as_bytes();
        bytes.extend_from_slice(&kem_seed);
        kem_seed.zeroize();

        let mut dsa_seed = self.signing_key.as_bytes();
        bytes.extend_from_slice(&dsa_seed);
        dsa_seed.zeroize();

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

    /// Returns a reference to the ML-DSA-65 signing key.
    pub fn signing_key(&self) -> &harmony_crypto::ml_dsa::MlDsaSecretKey {
        &self.signing_key
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
        let mut key_bytes = harmony_crypto::hkdf::derive_key(
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
        key_bytes.zeroize();

        result
    }

    /// Issue a post-quantum root UCAN token — claims direct ownership of a resource.
    ///
    /// Root tokens have no `proof` field. The issuer is asserting they own
    /// the resource. The token is signed with ML-DSA-65.
    ///
    /// The `rng` parameter is used to generate the 16-byte nonce; ML-DSA-65
    /// signing itself is deterministic from the key.
    pub fn issue_pq_root_token(
        &self,
        rng: &mut impl CryptoRngCore,
        audience: &[u8; ADDRESS_HASH_LENGTH],
        capability: CapabilityType,
        resource: &[u8],
        not_before: u64,
        expires_at: u64,
    ) -> Result<PqUcanToken, UcanError> {
        if resource.len() > MAX_RESOURCE_SIZE {
            return Err(UcanError::ResourceTooLarge);
        }
        if expires_at != 0 && not_before > expires_at {
            return Err(UcanError::InvalidTimeWindow);
        }

        let mut nonce = [0u8; 16];
        rng.fill_bytes(&mut nonce);

        let mut token = PqUcanToken {
            issuer: self.identity.address_hash,
            audience: *audience,
            capability,
            resource: resource.to_vec(),
            not_before,
            expires_at,
            nonce,
            proof: None,
            signature: Vec::new(), // placeholder for signable_bytes()
        };

        let signable = token.signable_bytes();
        let sig = ml_dsa::sign(&self.signing_key, &signable)
            .map_err(|e| UcanError::Identity(IdentityError::Crypto(e)))?;
        token.signature = sig.as_bytes().to_vec();

        Ok(token)
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

    // ── PQ UCAN token tests ─────────────────────────────────────────

    #[test]
    fn pq_ucan_roundtrip() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let audience_addr = [0xBB; 16];
        let token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &audience_addr,
                CapabilityType::Content,
                b"cid:abc",
                0,
                9999999999,
            )
            .unwrap();

        let bytes = token.to_bytes();
        assert_eq!(bytes[0], 0x01); // crypto_suite = PQ

        let token2 = PqUcanToken::from_bytes(&bytes).unwrap();
        assert_eq!(token.issuer, token2.issuer);
        assert_eq!(token.audience, token2.audience);
        assert_eq!(token.capability, token2.capability);
        assert_eq!(token.resource, token2.resource);
        assert_eq!(token.not_before, token2.not_before);
        assert_eq!(token.expires_at, token2.expires_at);
        assert_eq!(token.nonce, token2.nonce);
        assert_eq!(token.proof, token2.proof);
        assert_eq!(token.signature, token2.signature);
    }

    #[test]
    fn pq_ucan_verify() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let audience_addr = [0xBB; 16];
        let token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &audience_addr,
                CapabilityType::Content,
                b"cid:abc",
                0,
                9999999999,
            )
            .unwrap();

        // Verify with issuer's public key
        assert!(token
            .verify_signature(&issuer.public_identity().verifying_key)
            .is_ok());
    }

    #[test]
    fn pq_ucan_wrong_key_fails_verify() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let wrong = PqPrivateIdentity::generate(&mut OsRng);
        let audience_addr = [0xBB; 16];
        let token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &audience_addr,
                CapabilityType::Content,
                b"cid:abc",
                0,
                9999999999,
            )
            .unwrap();

        assert!(token
            .verify_signature(&wrong.public_identity().verifying_key)
            .is_err());
    }

    #[test]
    fn pq_ucan_signature_length() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let token = issuer
            .issue_pq_root_token(&mut OsRng, &[0xBB; 16], CapabilityType::Memory, &[], 0, 0)
            .unwrap();

        assert_eq!(token.signature.len(), 3309); // ML-DSA-65 signature length
    }

    #[test]
    fn pq_ucan_issuer_is_address_hash() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &[0xBB; 16],
                CapabilityType::Content,
                b"test",
                0,
                0,
            )
            .unwrap();

        assert_eq!(token.issuer, issuer.public_identity().address_hash);
    }

    #[test]
    fn pq_ucan_nonce_is_random() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let t1 = issuer
            .issue_pq_root_token(&mut OsRng, &[0xBB; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();
        let t2 = issuer
            .issue_pq_root_token(&mut OsRng, &[0xBB; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        assert_ne!(t1.nonce, t2.nonce);
    }

    #[test]
    fn pq_ucan_resource_too_large_rejected() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let result = issuer.issue_pq_root_token(
            &mut OsRng,
            &[0xBB; 16],
            CapabilityType::Memory,
            &[0u8; 257],
            0,
            0,
        );
        assert!(matches!(result, Err(UcanError::ResourceTooLarge)));
    }

    #[test]
    fn pq_ucan_invalid_time_window_rejected() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let result = issuer.issue_pq_root_token(
            &mut OsRng,
            &[0xBB; 16],
            CapabilityType::Memory,
            &[],
            200,
            100,
        );
        assert!(matches!(result, Err(UcanError::InvalidTimeWindow)));
    }

    #[test]
    fn pq_ucan_tampered_signature_fails_verify() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let mut token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &[0xBB; 16],
                CapabilityType::Content,
                b"test",
                0,
                0,
            )
            .unwrap();

        token.signature[0] ^= 0xFF;
        assert!(token
            .verify_signature(&issuer.public_identity().verifying_key)
            .is_err());
    }

    #[test]
    fn pq_ucan_from_bytes_rejects_wrong_crypto_suite() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let token = issuer
            .issue_pq_root_token(&mut OsRng, &[0xBB; 16], CapabilityType::Content, &[], 0, 0)
            .unwrap();

        let mut bytes = token.to_bytes();
        bytes[0] = 0x00; // Change to Ed25519 suite
        assert!(PqUcanToken::from_bytes(&bytes).is_err());
    }

    #[test]
    fn pq_ucan_from_bytes_rejects_truncated() {
        assert!(PqUcanToken::from_bytes(&[0x01; 10]).is_err());
    }

    #[test]
    fn pq_ucan_content_hash_deterministic() {
        let issuer = PqPrivateIdentity::generate(&mut OsRng);
        let token = issuer
            .issue_pq_root_token(
                &mut OsRng,
                &[0xBB; 16],
                CapabilityType::Content,
                b"test",
                0,
                0,
            )
            .unwrap();

        let h1 = token.content_hash();
        let h2 = token.content_hash();
        assert_eq!(h1, h2);
    }

    // ── Proof-of-Work ──────────────────────────────────────────────────

    #[test]
    fn pq_generate_with_proof_produces_valid_proof() {
        let (id, proof) = PqPrivateIdentity::generate_with_proof(
            &mut OsRng,
            &crate::puzzle::PuzzleParams::TEST,
        );
        assert!(id.verify_proof(&proof, &crate::puzzle::PuzzleParams::TEST));
    }

    #[test]
    fn pq_proof_does_not_verify_for_different_identity() {
        // d=8 → 1/256 false-pass probability, safe for CI.
        let params = crate::puzzle::PuzzleParams {
            difficulty: 8,
            ..crate::puzzle::PuzzleParams::TEST
        };
        let (_id1, proof) = PqPrivateIdentity::generate_with_proof(&mut OsRng, &params);
        let id2 = PqPrivateIdentity::generate(&mut OsRng);
        assert!(!id2.verify_proof(&proof, &params));
    }

    #[test]
    fn pq_generate_with_proof_identity_is_functional() {
        let (id, _proof) = PqPrivateIdentity::generate_with_proof(
            &mut OsRng,
            &crate::puzzle::PuzzleParams::TEST,
        );
        let sig = id.sign(b"test").unwrap();
        assert!(id.public_identity().verify(b"test", &sig).is_ok());
    }
}

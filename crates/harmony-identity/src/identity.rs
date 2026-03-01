//! Self-sovereign Ed25519/X25519 identity management.
//!
//! An identity in Harmony (and Reticulum) consists of:
//! - An X25519 keypair for Diffie-Hellman key exchange (encryption)
//! - An Ed25519 keypair for digital signatures
//!
//! The public identity is the concatenation: `[32B X25519 pub][32B Ed25519 pub]`
//! The address hash is: `SHA256(public_identity)[:16]` (128-bit truncated hash)

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand_core::CryptoRngCore;
use x25519_dalek::{EphemeralSecret, PublicKey as X25519PublicKey, StaticSecret};
use zeroize::{Zeroize, ZeroizeOnDrop};

use harmony_crypto::hash;
use harmony_crypto::hkdf;

use crate::IdentityError;

/// Length of the combined public key in bytes.
pub const PUBLIC_KEY_LENGTH: usize = 64;

/// Length of the combined private key in bytes.
pub const PRIVATE_KEY_LENGTH: usize = 64;

/// Length of the address hash in bytes.
pub const ADDRESS_HASH_LENGTH: usize = hash::TRUNCATED_HASH_LENGTH; // 16

/// Length of an Ed25519 signature in bytes.
pub const SIGNATURE_LENGTH: usize = 64;

/// A public identity: X25519 public key + Ed25519 verifying key.
///
/// This is the information shared with peers. The address hash derived from
/// these keys serves as the node's network address.
#[derive(Clone, Debug)]
pub struct Identity {
    /// X25519 public key for Diffie-Hellman key exchange.
    pub encryption_key: X25519PublicKey,
    /// Ed25519 verifying key for signature verification.
    pub verifying_key: VerifyingKey,
    /// Truncated SHA-256 hash of the combined public keys (16 bytes).
    pub address_hash: [u8; ADDRESS_HASH_LENGTH],
}

impl Identity {
    /// Construct an identity from raw X25519 and Ed25519 public key bytes.
    pub fn from_public_keys(
        x25519_pub: &[u8; 32],
        ed25519_pub: &[u8; 32],
    ) -> Result<Self, IdentityError> {
        let encryption_key = X25519PublicKey::from(*x25519_pub);
        let verifying_key = VerifyingKey::from_bytes(ed25519_pub)
            .map_err(|_| IdentityError::InvalidVerifyingKey)?;

        let mut combined = [0u8; PUBLIC_KEY_LENGTH];
        combined[..32].copy_from_slice(x25519_pub);
        combined[32..].copy_from_slice(ed25519_pub);
        let address_hash = hash::truncated_hash(&combined);

        Ok(Self {
            encryption_key,
            verifying_key,
            address_hash,
        })
    }

    /// Deserialize from a 64-byte combined public key: `[32B X25519][32B Ed25519]`.
    pub fn from_public_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        if bytes.len() != PUBLIC_KEY_LENGTH {
            return Err(IdentityError::InvalidPublicKeyLength(bytes.len()));
        }
        let x25519_pub: [u8; 32] = bytes[..32].try_into().unwrap();
        let ed25519_pub: [u8; 32] = bytes[32..].try_into().unwrap();
        Self::from_public_keys(&x25519_pub, &ed25519_pub)
    }

    /// Serialize to 64 bytes: `[32B X25519 pub][32B Ed25519 pub]`.
    pub fn to_public_bytes(&self) -> [u8; PUBLIC_KEY_LENGTH] {
        let mut bytes = [0u8; PUBLIC_KEY_LENGTH];
        bytes[..32].copy_from_slice(self.encryption_key.as_bytes());
        bytes[32..].copy_from_slice(self.verifying_key.as_bytes());
        bytes
    }

    /// Verify an Ed25519 signature against this identity.
    pub fn verify(&self, message: &[u8], signature: &[u8; SIGNATURE_LENGTH]) -> Result<(), IdentityError> {
        let sig = Signature::from_bytes(signature);
        self.verifying_key
            .verify(message, &sig)
            .map_err(|_| IdentityError::SignatureInvalid)
    }

    /// Encrypt data to this identity using ephemeral ECDH + HKDF + Fernet.
    ///
    /// Returns: `[32B ephemeral X25519 pub][Fernet token]`
    ///
    /// This matches Reticulum's encryption scheme:
    /// 1. Generate ephemeral X25519 keypair
    /// 2. ECDH with recipient's X25519 public key → shared secret
    /// 3. HKDF-SHA256(shared_secret, salt=address_hash) → 64-byte derived key
    /// 4. Fernet encrypt(derived_key, plaintext) → token
    pub fn encrypt(
        &self,
        rng: &mut impl CryptoRngCore,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, IdentityError> {
        let ephemeral_secret = EphemeralSecret::random_from_rng(&mut *rng);
        let ephemeral_public = X25519PublicKey::from(&ephemeral_secret);

        // ECDH: ephemeral_secret × recipient_public → shared_secret
        let shared_secret = ephemeral_secret.diffie_hellman(&self.encryption_key);

        // HKDF: derive 64-byte Fernet key from shared secret
        let mut derived = hkdf::derive_key_256(
            shared_secret.as_bytes(),
            Some(&self.address_hash),
        );

        // Fernet encrypt
        let token = harmony_crypto::fernet::encrypt(rng, &derived, plaintext)?;

        // Zeroize derived key
        derived.zeroize();

        // Prepend ephemeral public key
        let mut result = Vec::with_capacity(32 + token.len());
        result.extend_from_slice(ephemeral_public.as_bytes());
        result.extend_from_slice(&token);

        Ok(result)
    }
}

impl PartialEq for Identity {
    fn eq(&self, other: &Self) -> bool {
        self.address_hash == other.address_hash
    }
}

impl Eq for Identity {}

/// A full identity with private keys, capable of signing and decrypting.
#[derive(ZeroizeOnDrop)]
pub struct PrivateIdentity {
    /// The public portion of this identity.
    #[zeroize(skip)]
    pub identity: Identity,
    /// X25519 static secret for decryption.
    encryption_secret: StaticSecret,
    /// Ed25519 signing key.
    signing_key: SigningKey,
}

impl PrivateIdentity {
    /// Generate a new random identity.
    pub fn generate(rng: &mut impl CryptoRngCore) -> Self {
        let signing_key = SigningKey::generate(&mut *rng);
        let encryption_secret = StaticSecret::random_from_rng(&mut *rng);

        let encryption_key = X25519PublicKey::from(&encryption_secret);
        let verifying_key = signing_key.verifying_key();

        let mut combined = [0u8; PUBLIC_KEY_LENGTH];
        combined[..32].copy_from_slice(encryption_key.as_bytes());
        combined[32..].copy_from_slice(verifying_key.as_bytes());
        let address_hash = hash::truncated_hash(&combined);

        let identity = Identity {
            encryption_key,
            verifying_key,
            address_hash,
        };

        Self {
            identity,
            encryption_secret,
            signing_key,
        }
    }

    /// Sign a message with Ed25519, returning a 64-byte signature.
    pub fn sign(&self, message: &[u8]) -> [u8; SIGNATURE_LENGTH] {
        self.signing_key.sign(message).to_bytes()
    }

    /// Decrypt data encrypted to this identity.
    ///
    /// Expects: `[32B ephemeral X25519 pub][Fernet token]`
    pub fn decrypt(&self, ciphertext: &[u8]) -> Result<Vec<u8>, IdentityError> {
        if ciphertext.len() < 33 {
            return Err(IdentityError::DecryptionFailed);
        }

        // Extract ephemeral public key
        let ephemeral_pub_bytes: [u8; 32] = ciphertext[..32].try_into().unwrap();
        let ephemeral_pub = X25519PublicKey::from(ephemeral_pub_bytes);
        let token = &ciphertext[32..];

        // ECDH: our_secret × ephemeral_public → shared_secret
        let shared_secret = self.encryption_secret.diffie_hellman(&ephemeral_pub);

        // HKDF: derive 64-byte Fernet key
        let mut derived = hkdf::derive_key_256(
            shared_secret.as_bytes(),
            Some(&self.identity.address_hash),
        );

        // Fernet decrypt
        let result = harmony_crypto::fernet::decrypt(&derived, token)
            .map_err(|_| IdentityError::DecryptionFailed);

        // Zeroize derived key
        derived.zeroize();

        result
    }

    /// Get the public identity.
    pub fn public_identity(&self) -> &Identity {
        &self.identity
    }

    /// Serialize the private key to 64 bytes: `[32B X25519 secret][32B Ed25519 secret]`.
    pub fn to_private_bytes(&self) -> [u8; PRIVATE_KEY_LENGTH] {
        let mut bytes = [0u8; PRIVATE_KEY_LENGTH];
        bytes[..32].copy_from_slice(self.encryption_secret.as_bytes());
        bytes[32..].copy_from_slice(self.signing_key.as_bytes());
        bytes
    }

    /// Deserialize from 64 private bytes: `[32B X25519 secret][32B Ed25519 secret]`.
    pub fn from_private_bytes(bytes: &[u8]) -> Result<Self, IdentityError> {
        if bytes.len() != PRIVATE_KEY_LENGTH {
            return Err(IdentityError::InvalidPrivateKeyLength(bytes.len()));
        }

        let x25519_secret_bytes: [u8; 32] = bytes[..32].try_into().unwrap();
        let ed25519_secret_bytes: [u8; 32] = bytes[32..].try_into().unwrap();

        let encryption_secret = StaticSecret::from(x25519_secret_bytes);
        let signing_key = SigningKey::from_bytes(&ed25519_secret_bytes);

        let encryption_key = X25519PublicKey::from(&encryption_secret);
        let verifying_key = signing_key.verifying_key();

        let mut combined = [0u8; PUBLIC_KEY_LENGTH];
        combined[..32].copy_from_slice(encryption_key.as_bytes());
        combined[32..].copy_from_slice(verifying_key.as_bytes());
        let address_hash = hash::truncated_hash(&combined);

        let identity = Identity {
            encryption_key,
            verifying_key,
            address_hash,
        };

        Ok(Self {
            identity,
            encryption_secret,
            signing_key,
        })
    }

    /// Perform raw X25519 ECDH key exchange with a peer's public key.
    ///
    /// Returns the 32-byte shared secret. The caller is responsible for
    /// deriving a symmetric key from this (e.g., via HKDF).
    pub fn ecdh(&self, peer_public: &X25519PublicKey) -> x25519_dalek::SharedSecret {
        self.encryption_secret.diffie_hellman(peer_public)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    // ── Key Generation & Address Derivation ─────────────────────────────

    #[test]
    fn generate_produces_valid_identity() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let pub_id = id.public_identity();

        // Address hash should be 16 bytes
        assert_eq!(pub_id.address_hash.len(), 16);

        // Public key serialization should be 64 bytes
        let pub_bytes = pub_id.to_public_bytes();
        assert_eq!(pub_bytes.len(), 64);
    }

    #[test]
    fn address_hash_matches_spec() {
        // Address = SHA256(X25519_pub || Ed25519_pub)[:16]
        let id = PrivateIdentity::generate(&mut OsRng);
        let pub_bytes = id.public_identity().to_public_bytes();

        let expected_hash = hash::truncated_hash(&pub_bytes);
        assert_eq!(id.identity.address_hash, expected_hash);
    }

    #[test]
    fn address_hash_deterministic_from_keys() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let pub_bytes = id.public_identity().to_public_bytes();

        // Reconstruct from public bytes and verify same hash
        let reconstructed = Identity::from_public_bytes(&pub_bytes).unwrap();
        assert_eq!(reconstructed.address_hash, id.identity.address_hash);
    }

    #[test]
    fn different_identities_have_different_addresses() {
        let id1 = PrivateIdentity::generate(&mut OsRng);
        let id2 = PrivateIdentity::generate(&mut OsRng);
        assert_ne!(id1.identity.address_hash, id2.identity.address_hash);
    }

    // ── Serialization Roundtrip ─────────────────────────────────────────

    #[test]
    fn public_key_roundtrip() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let pub_bytes = id.public_identity().to_public_bytes();
        let restored = Identity::from_public_bytes(&pub_bytes).unwrap();

        assert_eq!(restored.address_hash, id.identity.address_hash);
        assert_eq!(
            restored.encryption_key.as_bytes(),
            id.identity.encryption_key.as_bytes()
        );
        assert_eq!(
            restored.verifying_key.as_bytes(),
            id.identity.verifying_key.as_bytes()
        );
    }

    #[test]
    fn private_key_roundtrip() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let priv_bytes = id.to_private_bytes();
        assert_eq!(priv_bytes.len(), 64);

        let restored = PrivateIdentity::from_private_bytes(&priv_bytes).unwrap();
        assert_eq!(
            restored.identity.address_hash,
            id.identity.address_hash
        );
        assert_eq!(
            restored.public_identity().to_public_bytes(),
            id.public_identity().to_public_bytes()
        );
    }

    #[test]
    fn invalid_public_key_length_rejected() {
        assert!(matches!(
            Identity::from_public_bytes(&[0u8; 63]),
            Err(IdentityError::InvalidPublicKeyLength(63))
        ));
        assert!(matches!(
            Identity::from_public_bytes(&[0u8; 65]),
            Err(IdentityError::InvalidPublicKeyLength(65))
        ));
    }

    #[test]
    fn invalid_private_key_length_rejected() {
        assert!(matches!(
            PrivateIdentity::from_private_bytes(&[0u8; 32]),
            Err(IdentityError::InvalidPrivateKeyLength(32))
        ));
    }

    // ── Signing & Verification ──────────────────────────────────────────

    #[test]
    fn sign_verify_roundtrip() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let message = b"harmony is the way";
        let signature = id.sign(message);

        assert_eq!(signature.len(), 64);
        assert!(id.public_identity().verify(message, &signature).is_ok());
    }

    #[test]
    fn verify_wrong_message_fails() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let signature = id.sign(b"correct message");
        let result = id.public_identity().verify(b"wrong message", &signature);
        assert!(matches!(result, Err(IdentityError::SignatureInvalid)));
    }

    #[test]
    fn verify_wrong_key_fails() {
        let id1 = PrivateIdentity::generate(&mut OsRng);
        let id2 = PrivateIdentity::generate(&mut OsRng);
        let signature = id1.sign(b"message");
        let result = id2.public_identity().verify(b"message", &signature);
        assert!(matches!(result, Err(IdentityError::SignatureInvalid)));
    }

    #[test]
    fn tampered_signature_fails() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let mut signature = id.sign(b"message");
        signature[0] ^= 0x01;
        let result = id.public_identity().verify(b"message", &signature);
        assert!(matches!(result, Err(IdentityError::SignatureInvalid)));
    }

    #[test]
    fn sign_deterministic_for_same_key_and_message() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let s1 = id.sign(b"same message");
        let s2 = id.sign(b"same message");
        // Ed25519 is deterministic (RFC 8032)
        assert_eq!(s1, s2);
    }

    // ── Encryption & Decryption ─────────────────────────────────────────

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let alice = PrivateIdentity::generate(&mut OsRng);
        let plaintext = b"hello from alice";

        // Bob encrypts to Alice's public identity
        let ciphertext = alice.public_identity().encrypt(&mut OsRng, plaintext).unwrap();

        // Alice decrypts with her private key
        let decrypted = alice.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn encrypt_decrypt_empty_plaintext() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let ciphertext = id.public_identity().encrypt(&mut OsRng, b"").unwrap();
        let decrypted = id.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, b"");
    }

    #[test]
    fn encrypt_decrypt_large_payload() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let plaintext = vec![0xABu8; 8192];
        let ciphertext = id.public_identity().encrypt(&mut OsRng, &plaintext).unwrap();
        let decrypted = id.decrypt(&ciphertext).unwrap();
        assert_eq!(decrypted, plaintext);
    }

    #[test]
    fn ciphertext_starts_with_32_byte_ephemeral_key() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let ciphertext = id.public_identity().encrypt(&mut OsRng, b"test").unwrap();
        // First 32 bytes are the ephemeral X25519 public key
        assert!(ciphertext.len() > 32);
    }

    #[test]
    fn different_encryptions_produce_different_ciphertext() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let c1 = id.public_identity().encrypt(&mut OsRng, b"same").unwrap();
        let c2 = id.public_identity().encrypt(&mut OsRng, b"same").unwrap();
        // Different ephemeral keys → different ciphertext
        assert_ne!(c1, c2);
        // Both decrypt correctly
        assert_eq!(id.decrypt(&c1).unwrap(), id.decrypt(&c2).unwrap());
    }

    #[test]
    fn wrong_identity_cannot_decrypt() {
        let alice = PrivateIdentity::generate(&mut OsRng);
        let bob = PrivateIdentity::generate(&mut OsRng);

        let ciphertext = alice.public_identity().encrypt(&mut OsRng, b"for alice only").unwrap();
        let result = bob.decrypt(&ciphertext);
        assert!(result.is_err());
    }

    #[test]
    fn truncated_ciphertext_fails() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let result = id.decrypt(&[0u8; 20]);
        assert!(matches!(result, Err(IdentityError::DecryptionFailed)));
    }

    // ── ECDH Key Exchange ───────────────────────────────────────────────

    #[test]
    fn ecdh_shared_secret_matches() {
        let alice = PrivateIdentity::generate(&mut OsRng);
        let bob = PrivateIdentity::generate(&mut OsRng);

        let secret_ab = alice.ecdh(&bob.identity.encryption_key);
        let secret_ba = bob.ecdh(&alice.identity.encryption_key);

        assert_eq!(secret_ab.as_bytes(), secret_ba.as_bytes());
    }

    #[test]
    fn ecdh_different_peers_different_secrets() {
        let alice = PrivateIdentity::generate(&mut OsRng);
        let bob = PrivateIdentity::generate(&mut OsRng);
        let carol = PrivateIdentity::generate(&mut OsRng);

        let secret_ab = alice.ecdh(&bob.identity.encryption_key);
        let secret_ac = alice.ecdh(&carol.identity.encryption_key);

        assert_ne!(secret_ab.as_bytes(), secret_ac.as_bytes());
    }

    // ── Identity Equality ───────────────────────────────────────────────

    #[test]
    fn identity_equality_based_on_address_hash() {
        let id = PrivateIdentity::generate(&mut OsRng);
        let pub_bytes = id.public_identity().to_public_bytes();
        let restored = Identity::from_public_bytes(&pub_bytes).unwrap();
        assert_eq!(id.public_identity(), &restored);
    }

    #[test]
    fn different_identities_not_equal() {
        let id1 = PrivateIdentity::generate(&mut OsRng);
        let id2 = PrivateIdentity::generate(&mut OsRng);
        assert_ne!(id1.public_identity(), id2.public_identity());
    }
}

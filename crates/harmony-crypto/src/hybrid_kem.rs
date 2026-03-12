//! Hybrid KEM: X25519 + ML-KEM-768.
//!
//! Combines a classical X25519 key exchange with ML-KEM-768 key encapsulation.
//! The two shared secrets are combined via HKDF-SHA256 to produce a single
//! 32-byte symmetric key. This provides defense-in-depth: the result is secure
//! if either algorithm holds.

use rand_core::CryptoRngCore;
use x25519_dalek::{PublicKey as X25519PublicKey, StaticSecret as X25519Secret};
use zeroize::Zeroize;

use crate::ml_kem::{MlKemCiphertext, MlKemPublicKey, MlKemSecretKey};
use crate::CryptoError;

/// Perform hybrid key encapsulation using X25519 + ML-KEM-768.
///
/// 1. ML-KEM-768 encapsulate with `ml_pk` to get (ciphertext, ml_shared_secret).
/// 2. Generate an ephemeral X25519 keypair and perform DH with `x_pk`.
/// 3. Combine both shared secrets via HKDF-SHA256: `ikm = x_ss || ml_ss`,
///    `info = context`, producing a 32-byte key.
///
/// Returns `(ml_kem_ciphertext, x25519_ephemeral_public_key, shared_key)`.
pub fn hybrid_encapsulate(
    rng: &mut impl CryptoRngCore,
    ml_pk: &MlKemPublicKey,
    x_pk: &X25519PublicKey,
    context: &[u8],
) -> Result<(MlKemCiphertext, X25519PublicKey, [u8; 32]), CryptoError> {
    // Step 1: ML-KEM-768 encapsulate
    let (ct, ml_ss) = crate::ml_kem::encapsulate(rng, ml_pk)?;

    // Step 2: X25519 ephemeral key exchange
    let x_eph_secret = X25519Secret::random_from_rng(&mut *rng);
    let x_eph_public = X25519PublicKey::from(&x_eph_secret);
    let x_ss = x_eph_secret.diffie_hellman(x_pk);

    // Step 3: Combine via HKDF-SHA256
    //   ikm = x25519_shared_secret || ml_kem_shared_secret
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(x_ss.as_bytes());
    ikm[32..].copy_from_slice(ml_ss.as_bytes());

    let derived = crate::hkdf::derive_key(&ikm, None, context, 32)?;
    let mut key = [0u8; 32];
    key.copy_from_slice(&derived);

    // Zeroize intermediate key material
    ikm.zeroize();

    Ok((ct, x_eph_public, key))
}

/// Perform hybrid key decapsulation using X25519 + ML-KEM-768.
///
/// 1. ML-KEM-768 decapsulate with `ml_sk` and `ct` to recover ml_shared_secret.
/// 2. X25519 DH with `x_sk` and `x_eph_pk` to recover x_shared_secret.
/// 3. Combine both shared secrets via HKDF-SHA256 with the same parameters
///    as encapsulation to recover the 32-byte key.
pub fn hybrid_decapsulate(
    ml_sk: &MlKemSecretKey,
    ct: &MlKemCiphertext,
    x_sk: &X25519Secret,
    x_eph_pk: &X25519PublicKey,
    context: &[u8],
) -> Result<[u8; 32], CryptoError> {
    // Step 1: ML-KEM-768 decapsulate
    let ml_ss = crate::ml_kem::decapsulate(ml_sk, ct)?;

    // Step 2: X25519 DH
    let x_ss = x_sk.diffie_hellman(x_eph_pk);

    // Step 3: Combine via HKDF-SHA256 (same order: x_ss || ml_ss)
    let mut ikm = [0u8; 64];
    ikm[..32].copy_from_slice(x_ss.as_bytes());
    ikm[32..].copy_from_slice(ml_ss.as_bytes());

    let derived = crate::hkdf::derive_key(&ikm, None, context, 32)?;
    let mut key = [0u8; 32];
    key.copy_from_slice(&derived);

    // Zeroize intermediate key material
    ikm.zeroize();

    Ok(key)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn hybrid_kem_roundtrip() {
        let (ml_pk, ml_sk) = crate::ml_kem::generate(&mut OsRng);
        let x_sk = x25519_dalek::StaticSecret::random_from_rng(&mut OsRng);
        let x_pk = x25519_dalek::PublicKey::from(&x_sk);

        let (ct, x_eph_pk, shared_key_sender) =
            hybrid_encapsulate(&mut OsRng, &ml_pk, &x_pk, b"test-context").unwrap();
        let shared_key_receiver =
            hybrid_decapsulate(&ml_sk, &ct, &x_sk, &x_eph_pk, b"test-context").unwrap();

        assert_eq!(shared_key_sender, shared_key_receiver);
        assert_eq!(shared_key_sender.len(), 32);
    }

    #[test]
    fn different_contexts_produce_different_keys() {
        let (ml_pk, ml_sk) = crate::ml_kem::generate(&mut OsRng);
        let x_sk = x25519_dalek::StaticSecret::random_from_rng(&mut OsRng);
        let x_pk = x25519_dalek::PublicKey::from(&x_sk);

        let (ct, x_eph_pk, key1) =
            hybrid_encapsulate(&mut OsRng, &ml_pk, &x_pk, b"context-a").unwrap();
        // Decapsulate with different context should produce different key
        let key2 =
            hybrid_decapsulate(&ml_sk, &ct, &x_sk, &x_eph_pk, b"context-b").unwrap();

        assert_ne!(key1, key2);
    }
}

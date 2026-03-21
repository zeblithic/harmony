//! Shared signature verification dispatch for all Harmony crates.
//!
//! Dispatches to Ed25519 or ML-DSA-65 based on [`CryptoSuite`],
//! eliminating duplicated verification logic across harmony-credential,
//! harmony-discovery, and harmony-profile.

use crate::crypto_suite::CryptoSuite;
use crate::error::IdentityError;

/// Verify a signature against raw public key bytes, dispatching to
/// the appropriate algorithm based on `suite`.
///
/// # Key formats
///
/// - **Ed25519:** 32-byte verifying key
/// - **ML-DSA-65 / ML-DSA-65 Rotatable:** 1952-byte signing public key
///
/// # Errors
///
/// Returns [`IdentityError::SignatureInvalid`] if:
/// - Key bytes are the wrong length for the suite
/// - Key bytes don't represent a valid public key
/// - Signature bytes are malformed
/// - Signature does not verify against the message
///
/// # Note on MlDsa65Rotatable
///
/// Currently treats `MlDsa65Rotatable` the same as `MlDsa65` — verifies
/// against the provided key without rotation awareness. V2 will require
/// KEL chain walk to validate that the key was authorised by the
/// inception key at signing time.
pub fn verify_signature(
    suite: CryptoSuite,
    key_bytes: &[u8],
    message: &[u8],
    signature: &[u8],
) -> Result<(), IdentityError> {
    match suite {
        CryptoSuite::Ed25519 => {
            use ed25519_dalek::Verifier;
            let key_array: [u8; 32] = key_bytes
                .try_into()
                .map_err(|_| IdentityError::SignatureInvalid)?;
            let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&key_array)
                .map_err(|_| IdentityError::SignatureInvalid)?;
            let sig = ed25519_dalek::Signature::from_slice(signature)
                .map_err(|_| IdentityError::SignatureInvalid)?;
            verifying_key
                .verify(message, &sig)
                .map_err(|_| IdentityError::SignatureInvalid)
        }
        // TODO(v2): MlDsa65Rotatable needs rotation-aware verification —
        // the current key may differ from the one used to derive
        // identity_ref.hash. A rotation certificate or KEL chain walk
        // will be required to prove the new public_key was authorised
        // by the inception key.
        CryptoSuite::MlDsa65 | CryptoSuite::MlDsa65Rotatable => {
            let pk = harmony_crypto::ml_dsa::MlDsaPublicKey::from_bytes(key_bytes)
                .map_err(|_| IdentityError::SignatureInvalid)?;
            let sig = harmony_crypto::ml_dsa::MlDsaSignature::from_bytes(signature)
                .map_err(|_| IdentityError::SignatureInvalid)?;
            harmony_crypto::ml_dsa::verify(&pk, message, &sig)
                .map_err(|_| IdentityError::SignatureInvalid)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn ed25519_valid_signature() {
        let private = crate::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();
        let message = b"hello harmony";
        let signature = private.sign(message);

        assert!(verify_signature(
            CryptoSuite::Ed25519,
            &identity.verifying_key.to_bytes(),
            message,
            &signature,
        )
        .is_ok());
    }

    #[test]
    fn ed25519_wrong_signature_rejected() {
        let private = crate::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();

        let err = verify_signature(
            CryptoSuite::Ed25519,
            &identity.verifying_key.to_bytes(),
            b"hello",
            &[0xFF; 64],
        )
        .unwrap_err();
        assert!(matches!(err, IdentityError::SignatureInvalid));
    }

    #[test]
    fn ed25519_wrong_key_length_rejected() {
        let err =
            verify_signature(CryptoSuite::Ed25519, &[0x00; 16], b"msg", &[0x00; 64]).unwrap_err();
        assert!(matches!(err, IdentityError::SignatureInvalid));
    }

    #[test]
    fn ed25519_wrong_signature_length_rejected() {
        let private = crate::PrivateIdentity::generate(&mut OsRng);
        let identity = private.public_identity();

        let err = verify_signature(
            CryptoSuite::Ed25519,
            &identity.verifying_key.to_bytes(),
            b"msg",
            &[0x00; 32], // wrong length
        )
        .unwrap_err();
        assert!(matches!(err, IdentityError::SignatureInvalid));
    }

    #[test]
    fn ml_dsa65_valid_signature() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let message = b"hello post-quantum";
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, message).unwrap();

                assert!(verify_signature(
                    CryptoSuite::MlDsa65,
                    &sign_pk.as_bytes(),
                    message,
                    signature.as_bytes(),
                )
                .is_ok());
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn ml_dsa65_rotatable_uses_same_path() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let message = b"rotatable identity";
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, message).unwrap();

                assert!(verify_signature(
                    CryptoSuite::MlDsa65Rotatable,
                    &sign_pk.as_bytes(),
                    message,
                    signature.as_bytes(),
                )
                .is_ok());
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn ml_dsa65_wrong_key_rejected() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (_sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (other_pk, _) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let message = b"wrong key test";
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, message).unwrap();

                let err = verify_signature(
                    CryptoSuite::MlDsa65,
                    &other_pk.as_bytes(),
                    message,
                    signature.as_bytes(),
                )
                .unwrap_err();
                assert!(matches!(err, IdentityError::SignatureInvalid));
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn ml_dsa65_tampered_message_rejected() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let signature =
                    harmony_crypto::ml_dsa::sign(&sign_sk, b"original message").unwrap();

                let err = verify_signature(
                    CryptoSuite::MlDsa65,
                    &sign_pk.as_bytes(),
                    b"tampered message",
                    signature.as_bytes(),
                )
                .unwrap_err();
                assert!(matches!(err, IdentityError::SignatureInvalid));
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn ml_dsa65_rotatable_wrong_key_rejected() {
        std::thread::Builder::new()
            .stack_size(8 * 1024 * 1024)
            .spawn(|| {
                let (_sign_pk, sign_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let (other_pk, _) = harmony_crypto::ml_dsa::generate(&mut OsRng);
                let message = b"rotatable wrong key";
                let signature = harmony_crypto::ml_dsa::sign(&sign_sk, message).unwrap();

                let err = verify_signature(
                    CryptoSuite::MlDsa65Rotatable,
                    &other_pk.as_bytes(),
                    message,
                    signature.as_bytes(),
                )
                .unwrap_err();
                assert!(matches!(err, IdentityError::SignatureInvalid));
            })
            .expect("spawn")
            .join()
            .expect("join");
    }

    #[test]
    fn ml_dsa65_invalid_key_length_rejected() {
        let err =
            verify_signature(CryptoSuite::MlDsa65, &[0x00; 32], b"msg", &[0x00; 64]).unwrap_err();
        assert!(matches!(err, IdentityError::SignatureInvalid));
    }

    #[test]
    fn ml_dsa65_invalid_signature_length_rejected() {
        let err = verify_signature(
            CryptoSuite::MlDsa65,
            &[0x00; 1952], // valid-length key placeholder
            b"msg",
            &[0x00; 64], // ML-DSA-65 signatures are 3309 bytes
        )
        .unwrap_err();
        assert!(matches!(err, IdentityError::SignatureInvalid));
    }
}

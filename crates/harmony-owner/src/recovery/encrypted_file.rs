//! Argon2id + XChaCha20-Poly1305 encrypted-file encode/decode for the
//! 32-byte recovery seed plus optional metadata. Format spec lives in
//! `crate::recovery::wire`.

use serde::{Deserialize, Serialize};

use crate::recovery::wire::{
    serialize_header, HEADER_LEN, KDF_M_KIB, KDF_OUT_LEN, KDF_P, KDF_T, NONCE_LEN, SALT_LEN,
};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
use secrecy::{ExposeSecret, SecretString};
use zeroize::Zeroizing;

pub const FORMAT_STRING: &str = "harmony-owner-recovery-v1";
pub const MAX_COMMENT_LEN: usize = 256;

/// CBOR-encoded plaintext payload inside the AEAD ciphertext. The `format`
/// string is defense-in-depth: even though Poly1305 already proves the
/// payload was produced by someone with the passphrase, validating the
/// format string after decryption protects against future format
/// bifurcations being silently accepted by older parsers.
///
/// `pub` (not `pub(crate)`) because `encrypt_with_params_for_test` accepts
/// `&RecoveryFileBody` as a parameter, and that helper is callable from
/// integration tests when the `test-fixtures` feature is enabled. The
/// struct is only visible at all when the `recovery` feature is on
/// (the entire module is gated).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecoveryFileBody {
    pub format: String,
    #[serde(with = "serde_bytes")]
    pub seed: [u8; 32],
    pub mint_at: Option<u64>,
    pub comment: Option<String>,
}

/// Internal encrypt core. Takes already-built plaintext + already-chosen
/// salt + nonce. Both `to_encrypted_file` (production, random salt+nonce)
/// and `encrypt_with_params_for_test` (deterministic for fixtures) call
/// this.
fn encrypt_core(
    passphrase: &SecretString,
    plaintext: &[u8],
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
) -> Vec<u8> {
    let header = serialize_header();
    let key = derive_key_argon2id(passphrase, salt);
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let ct = cipher
        .encrypt(
            XNonce::from_slice(nonce),
            Payload { msg: plaintext, aad: &header },
        )
        .expect("XChaCha20-Poly1305 encryption is infallible for in-bounds inputs");
    let mut out = Vec::with_capacity(HEADER_LEN + SALT_LEN + NONCE_LEN + ct.len());
    out.extend_from_slice(&header);
    out.extend_from_slice(salt);
    out.extend_from_slice(nonce);
    out.extend_from_slice(&ct);
    out
}

fn derive_key_argon2id(
    passphrase: &SecretString,
    salt: &[u8; SALT_LEN],
) -> Zeroizing<[u8; KDF_OUT_LEN]> {
    use argon2::{Algorithm, Argon2, Params, Version};
    let params = Params::new(KDF_M_KIB, KDF_T as u32, KDF_P as u32, Some(KDF_OUT_LEN))
        .expect("Argon2 params are constants known to validate");
    let kdf = Argon2::new(Algorithm::Argon2id, Version::V0x13, params);
    let mut out: Zeroizing<[u8; KDF_OUT_LEN]> = Zeroizing::new([0u8; KDF_OUT_LEN]);
    kdf.hash_password_into(passphrase.expose_secret().as_bytes(), salt, out.as_mut())
        .expect("Argon2 derivation is infallible for valid params");
    out
}

/// Test-only deterministic encrypt: caller supplies salt and nonce so the
/// output is byte-stable for fixture pinning. Behind `test-fixtures` so it
/// is NOT part of the production API surface.
#[cfg(feature = "test-fixtures")]
pub fn encrypt_with_params_for_test(
    passphrase: &SecretString,
    body: &RecoveryFileBody,
    salt: &[u8; SALT_LEN],
    nonce: &[u8; NONCE_LEN],
) -> Vec<u8> {
    let plaintext = crate::cbor::to_canonical(body).expect("body always encodes");
    encrypt_core(passphrase, &plaintext, salt, nonce)
}

#[cfg(all(test, feature = "test-fixtures"))]
mod fixture_helper_tests {
    use super::*;
    use crate::recovery::wire::TAG_LEN;

    #[test]
    fn deterministic_with_fixed_inputs() {
        let pass = SecretString::from("test-passphrase".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [7u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("fixture".into()),
        };
        let salt = [0xAB; SALT_LEN];
        let nonce = [0xCD; NONCE_LEN];
        let a = encrypt_with_params_for_test(&pass, &body, &salt, &nonce);
        let b = encrypt_with_params_for_test(&pass, &body, &salt, &nonce);
        assert_eq!(a, b, "deterministic inputs must produce identical output");
        assert!(a.len() >= MIN_FILE_LEN_USE);
    }
    const MIN_FILE_LEN_USE: usize = HEADER_LEN + SALT_LEN + NONCE_LEN + TAG_LEN;
}

#[cfg(test)]
mod body_tests {
    use super::*;

    #[test]
    fn cbor_round_trip_minimal() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: None,
            comment: None,
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }

    #[test]
    fn cbor_round_trip_full() {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [42u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("primary owner".into()),
        };
        let bytes = crate::cbor::to_canonical(&body).unwrap();
        let back: RecoveryFileBody = ciborium::de::from_reader(&bytes[..]).unwrap();
        assert_eq!(back, body);
    }
}

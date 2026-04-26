//! Argon2id + XChaCha20-Poly1305 encrypted-file encode/decode for the
//! 32-byte recovery seed plus optional metadata. Format spec lives in
//! `crate::recovery::wire`.

use serde::{Deserialize, Serialize};

use crate::recovery::error::RecoveryError;
use crate::recovery::wire::{
    parse_header, serialize_header, HEADER_LEN, KDF_M_KIB, KDF_OUT_LEN, KDF_P, KDF_T, MAX_FILE_LEN,
    MIN_FILE_LEN, NONCE_LEN, SALT_LEN,
};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
use rand_core::{OsRng, RngCore};
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
            Payload {
                msg: plaintext,
                aad: &header,
            },
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
    // Wrap in Zeroizing: the CBOR-encoded plaintext contains the 32-byte
    // seed bytes. Without this wrapper the Vec drops unzeroized after
    // encrypt_core returns. This pattern propagates to the production
    // to_encrypted_file path landing in Task 11.
    let plaintext: Zeroizing<Vec<u8>> =
        Zeroizing::new(crate::cbor::to_canonical(body).expect("body always encodes"));
    encrypt_core(passphrase, &plaintext, salt, nonce)
}

#[cfg(all(test, feature = "test-fixtures"))]
mod fixture_helper_tests {
    use super::*;

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
        assert!(a.len() >= MIN_FILE_LEN);
    }
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

/// `(seed, mint_at, comment)` returned by `decrypt_inner`. Aliased to
/// quiet `clippy::type_complexity` now that the public `RecoveryArtifact`
/// API consumes this signature directly.
pub(crate) type DecryptedParts = ([u8; 32], Option<u64>, Option<String>);

/// Decrypt + parse a recovery-file byte slice into the seed and metadata.
/// Returns the raw seed (caller wraps it back into a `RecoveryArtifact`)
/// and the metadata fields.
pub(crate) fn decrypt_inner(
    bytes: &[u8],
    passphrase: &SecretString,
) -> Result<DecryptedParts, RecoveryError> {
    if bytes.len() < MIN_FILE_LEN {
        return Err(RecoveryError::TooSmall(bytes.len()));
    }
    if bytes.len() > MAX_FILE_LEN {
        return Err(RecoveryError::TooLarge(bytes.len()));
    }
    parse_header(&bytes[..HEADER_LEN])?;

    let salt: &[u8; SALT_LEN] = bytes[HEADER_LEN..HEADER_LEN + SALT_LEN]
        .try_into()
        .expect("range matches SALT_LEN");
    let nonce: &[u8; NONCE_LEN] = bytes[HEADER_LEN + SALT_LEN..HEADER_LEN + SALT_LEN + NONCE_LEN]
        .try_into()
        .expect("range matches NONCE_LEN");
    let ciphertext_and_tag = &bytes[HEADER_LEN + SALT_LEN + NONCE_LEN..];

    let header = serialize_header();
    let key = derive_key_argon2id(passphrase, salt);
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let plaintext_vec = cipher
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext_and_tag,
                aad: &header,
            },
        )
        .map_err(|_| RecoveryError::WrongPassphraseOrCorrupt)?;
    // Wrap plaintext in Zeroizing so it's wiped after we finish parsing.
    let plaintext: Zeroizing<Vec<u8>> = Zeroizing::new(plaintext_vec);

    let body: RecoveryFileBody = ciborium::de::from_reader(&plaintext[..]).map_err(|_| {
        RecoveryError::PayloadDecodeFailed("CBOR payload could not be parsed".to_string())
    })?;

    if body.format != FORMAT_STRING {
        return Err(RecoveryError::UnexpectedPayloadFormat {
            found: body.format,
            expected: FORMAT_STRING,
        });
    }
    if let Some(c) = body.comment.as_ref() {
        if c.len() > MAX_COMMENT_LEN {
            return Err(RecoveryError::CommentTooLong {
                actual: c.len(),
                max: MAX_COMMENT_LEN,
            });
        }
    }
    Ok((body.seed, body.mint_at, body.comment))
}

#[cfg(all(test, feature = "test-fixtures"))]
mod decrypt_tests {
    use super::*;

    #[test]
    fn round_trip_minimal() {
        let pass = SecretString::from("rt-test".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [9u8; 32],
            mint_at: None,
            comment: None,
        };
        let bytes = encrypt_with_params_for_test(&pass, &body, &[1u8; SALT_LEN], &[2u8; NONCE_LEN]);
        let (seed, mint_at, comment) = decrypt_inner(&bytes, &pass).unwrap();
        assert_eq!(seed, [9u8; 32]);
        assert_eq!(mint_at, None);
        assert_eq!(comment, None);
    }

    #[test]
    fn round_trip_with_full_metadata() {
        let pass = SecretString::from("rt-full".to_string());
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [11u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("primary owner".into()),
        };
        let bytes = encrypt_with_params_for_test(&pass, &body, &[3u8; SALT_LEN], &[4u8; NONCE_LEN]);
        let (seed, mint_at, comment) = decrypt_inner(&bytes, &pass).unwrap();
        assert_eq!(seed, [11u8; 32]);
        assert_eq!(mint_at, Some(1_700_000_000));
        assert_eq!(comment.as_deref(), Some("primary owner"));
    }

    fn fixture_bytes() -> Vec<u8> {
        let body = RecoveryFileBody {
            format: FORMAT_STRING.into(),
            seed: [13u8; 32],
            mint_at: Some(1_700_000_000),
            comment: Some("neg".into()),
        };
        encrypt_with_params_for_test(
            &SecretString::from("correct".to_string()),
            &body,
            &[7u8; SALT_LEN],
            &[8u8; NONCE_LEN],
        )
    }

    #[test]
    fn wrong_passphrase_fails_aead() {
        let bytes = fixture_bytes();
        let err = decrypt_inner(&bytes, &SecretString::from("wrong".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::WrongPassphraseOrCorrupt));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let mut bytes = fixture_bytes();
        let len = bytes.len();
        // Flip a byte in the middle of the ciphertext region.
        let ct_start = HEADER_LEN + SALT_LEN + NONCE_LEN;
        bytes[ct_start + (len - ct_start) / 2] ^= 0x01;
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::WrongPassphraseOrCorrupt));
    }

    /// Tampering with header bytes produces a strict-check rejection
    /// (`UnsupportedKdfParams`), NOT an AEAD-AAD failure
    /// (`WrongPassphraseOrCorrupt`). The strict-equality check at
    /// `wire::parse_header` is intentionally a stricter guard than the
    /// AEAD-AAD layer for the 13-byte header — every byte is checked
    /// against locked v1 values BEFORE Argon2id derivation, so the AAD
    /// binding is structurally redundant for header tampering in v1.
    /// (The AAD layer would matter for a future format version that
    /// allowed any header byte to vary; this test pins the v1 behavior.)
    #[test]
    fn tampered_header_caught_by_strict_check_pre_aad() {
        let mut bytes = fixture_bytes();
        // Mutate kdf_t low byte: 3 → 2.
        bytes[11] ^= 0x01;
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string()))
            .unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedKdfParams { id: 0x01 }));
    }

    #[test]
    fn tampered_kdf_params_rejected_before_argon2() {
        let mut bytes = fixture_bytes();
        // Set kdf_m_kib to a small value — the strict check must reject
        // this BEFORE we run Argon2id with attacker-controlled params.
        bytes[6..10].copy_from_slice(&1024u32.to_be_bytes());
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(
            err,
            RecoveryError::UnsupportedKdfParams { id: 0x01 }
        ));
    }

    #[test]
    fn wrong_magic_rejected() {
        let mut bytes = fixture_bytes();
        bytes[0] = b'X';
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::UnrecognizedFormat));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut bytes = fixture_bytes();
        bytes[4] = 0x02;
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::UnsupportedVersion(0x02)));
    }

    #[test]
    fn too_small_rejected() {
        let bytes = vec![0u8; 50];
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::TooSmall(50)));
    }

    #[test]
    fn too_large_rejected() {
        let bytes = vec![0u8; 2048];
        let err = decrypt_inner(&bytes, &SecretString::from("correct".to_string())).unwrap_err();
        assert!(matches!(err, RecoveryError::TooLarge(2048)));
    }
}

/// Production encrypt entry point: random salt + nonce per call. The body
/// is built by the caller from the seed + metadata.
pub(crate) fn encrypt_inner(
    passphrase: &SecretString,
    seed: &[u8; 32],
    mint_at: Option<u64>,
    comment: Option<String>,
) -> Result<Vec<u8>, RecoveryError> {
    if let Some(c) = comment.as_ref() {
        if c.len() > MAX_COMMENT_LEN {
            return Err(RecoveryError::CommentTooLong {
                actual: c.len(),
                max: MAX_COMMENT_LEN,
            });
        }
    }
    let body = RecoveryFileBody {
        format: FORMAT_STRING.into(),
        seed: *seed,
        mint_at,
        comment,
    };
    let plaintext: Zeroizing<Vec<u8>> =
        Zeroizing::new(crate::cbor::to_canonical(&body).expect("body always encodes"));
    let mut salt = [0u8; SALT_LEN];
    let mut nonce = [0u8; NONCE_LEN];
    OsRng.fill_bytes(&mut salt);
    OsRng.fill_bytes(&mut nonce);
    Ok(encrypt_core(passphrase, &plaintext, &salt, &nonce))
}

#[cfg(test)]
mod prod_encrypt_tests {
    use super::*;

    #[test]
    fn salt_rotates_per_encode() {
        let pass = SecretString::from("salt-rot".to_string());
        let a = encrypt_inner(&pass, &[5u8; 32], None, None).unwrap();
        let b = encrypt_inner(&pass, &[5u8; 32], None, None).unwrap();
        // Same payload, fresh salt + nonce → different ciphertexts.
        assert_ne!(a, b, "salt + nonce regen must produce different bytes");
        // Salts (offset HEADER_LEN..HEADER_LEN+SALT_LEN) must differ.
        assert_ne!(
            &a[HEADER_LEN..HEADER_LEN + SALT_LEN],
            &b[HEADER_LEN..HEADER_LEN + SALT_LEN]
        );
    }

    #[test]
    fn comment_at_max_length_succeeds() {
        let pass = SecretString::from("max-len".to_string());
        let max_comment = "a".repeat(MAX_COMMENT_LEN);
        let r = encrypt_inner(&pass, &[5u8; 32], None, Some(max_comment));
        assert!(r.is_ok());
    }

    #[test]
    fn comment_over_max_fails_at_encode() {
        let pass = SecretString::from("over-len".to_string());
        let too_long = "a".repeat(MAX_COMMENT_LEN + 1);
        let err = encrypt_inner(&pass, &[5u8; 32], None, Some(too_long)).unwrap_err();
        assert!(matches!(
            err,
            RecoveryError::CommentTooLong {
                actual: 257,
                max: 256
            }
        ));
    }
}

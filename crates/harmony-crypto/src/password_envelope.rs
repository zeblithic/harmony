//! Password-based at-rest encryption envelope: Argon2id key derivation followed
//! by XChaCha20-Poly1305 authenticated encryption.
//!
//! This is the shared kernel behind the app-side identity vault (`HRMI`), the
//! owner-state snapshot backup (`HRSS`), and the owner recovery file (`HRMR`).
//! Each caller frames its own header, chooses its own magic, and binds whatever
//! bytes it likes as `aad`; this module knows nothing about headers, versions,
//! or on-disk layout. That separation is what lets one primitive back three
//! byte-frozen formats — the format-distinguishing bytes live in the
//! caller-supplied `aad`, never in here.
//!
//! # Determinism / nonce reuse
//!
//! The caller supplies `salt` and `nonce`; this module generates no randomness,
//! so callers can reproduce byte-identical output for golden fixtures. **A
//! (key, nonce) pair must never be reused** — production callers draw a fresh
//! random salt and nonce per [`seal`].

use alloc::vec::Vec;

use argon2::{Algorithm, Argon2, Params, Version};
use chacha20poly1305::{
    aead::{Aead, KeyInit, Payload},
    XChaCha20Poly1305, XNonce,
};
use zeroize::Zeroizing;

use crate::CryptoError;

/// Derived-key / AEAD-key length in bytes (fixed at the XChaCha20 key size).
pub const KEY_LEN: usize = 32;

/// XChaCha20-Poly1305 nonce length in bytes.
pub const NONCE_LEN: usize = 24;

/// Poly1305 authentication tag length in bytes.
pub const TAG_LEN: usize = 16;

/// Argon2id cost parameters. Caller-supplied because each on-disk format encodes
/// these values in its own header and must round-trip byte-for-byte. The derived
/// output length is fixed at [`KEY_LEN`] (32 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Argon2idParams {
    m_kib: u32,
    t_cost: u32,
    p_cost: u32,
}

impl Argon2idParams {
    /// Validate and construct. Returns [`CryptoError::InvalidKdfParams`] if the
    /// triple is out of range for Argon2 (memory/iterations/lanes below the
    /// algorithm minimums), validated against the fixed 32-byte output length.
    pub fn new(m_kib: u32, t_cost: u32, p_cost: u32) -> Result<Self, CryptoError> {
        Params::new(m_kib, t_cost, p_cost, Some(KEY_LEN))
            .map_err(|_| CryptoError::InvalidKdfParams)?;
        Ok(Self {
            m_kib,
            t_cost,
            p_cost,
        })
    }
}

/// Argon2id-derive a 32-byte key. Fails (`InvalidKdfParams`) only on
/// out-of-range params or an Argon2-unacceptable salt length — never panics.
fn derive_key(
    password: &[u8],
    salt: &[u8],
    params: &Argon2idParams,
) -> Result<Zeroizing<[u8; KEY_LEN]>, CryptoError> {
    let argon_params = Params::new(params.m_kib, params.t_cost, params.p_cost, Some(KEY_LEN))
        .map_err(|_| CryptoError::InvalidKdfParams)?;
    let kdf = Argon2::new(Algorithm::Argon2id, Version::V0x13, argon_params);
    let mut out: Zeroizing<[u8; KEY_LEN]> = Zeroizing::new([0u8; KEY_LEN]);
    kdf.hash_password_into(password, salt, out.as_mut())
        .map_err(|_| CryptoError::InvalidKdfParams)?;
    Ok(out)
}

/// Argon2id-derive a key from `password`/`salt`/`params`, then
/// XChaCha20-Poly1305-seal `plaintext` under `nonce` (24 bytes), binding `aad`.
/// Returns `ciphertext ‖ tag` (16-byte Poly1305 tag). The derived key is
/// zeroized before return.
pub fn seal(
    password: &[u8],
    params: &Argon2idParams,
    salt: &[u8],
    nonce: &[u8],
    aad: &[u8],
    plaintext: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::InvalidNonceLength {
            expected: NONCE_LEN,
            got: nonce.len(),
        });
    }
    let key = derive_key(password, salt, params)?;
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    cipher
        .encrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: plaintext,
                aad,
            },
        )
        .map_err(|_| CryptoError::AeadEncryptFailed)
}

/// Inverse of [`seal`]. Returns the zeroizing plaintext, or a [`CryptoError`] on
/// any failure (wrong key / tampered ciphertext / tag mismatch). Callers that
/// must not expose a decryption oracle collapse every error into one
/// indistinguishable message.
pub fn open(
    password: &[u8],
    params: &Argon2idParams,
    salt: &[u8],
    nonce: &[u8],
    aad: &[u8],
    ciphertext: &[u8],
) -> Result<Zeroizing<Vec<u8>>, CryptoError> {
    if nonce.len() != NONCE_LEN {
        return Err(CryptoError::InvalidNonceLength {
            expected: NONCE_LEN,
            got: nonce.len(),
        });
    }
    let key = derive_key(password, salt, params)?;
    let cipher = XChaCha20Poly1305::new(key.as_ref().into());
    let pt = cipher
        .decrypt(
            XNonce::from_slice(nonce),
            Payload {
                msg: ciphertext,
                aad,
            },
        )
        .map_err(|_| CryptoError::AeadDecryptFailed)?;
    Ok(Zeroizing::new(pt))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Cheap params keep the unit tests fast — the primitive is param-generic,
    // so correctness does not depend on recovery's production 64 MiB cost.
    fn cheap() -> Argon2idParams {
        Argon2idParams::new(8, 1, 1).expect("8 KiB / t=1 / p=1 is valid")
    }

    #[test]
    fn round_trip() {
        let p = cheap();
        let salt = [0xABu8; 16];
        let nonce = [0xCDu8; NONCE_LEN];
        let ct = seal(b"pw", &p, &salt, &nonce, b"hdr", b"hello harmony").unwrap();
        assert_eq!(ct.len(), b"hello harmony".len() + TAG_LEN);
        let pt = open(b"pw", &p, &salt, &nonce, b"hdr", &ct).unwrap();
        assert_eq!(&pt[..], b"hello harmony");
    }

    #[test]
    fn deterministic_for_fixed_inputs() {
        let p = cheap();
        let salt = [1u8; 16];
        let nonce = [2u8; NONCE_LEN];
        let a = seal(b"pw", &p, &salt, &nonce, b"aad", b"msg").unwrap();
        let b = seal(b"pw", &p, &salt, &nonce, b"aad", b"msg").unwrap();
        assert_eq!(a, b, "identical inputs must produce identical bytes");
    }

    #[test]
    fn wrong_password_fails() {
        let p = cheap();
        let salt = [3u8; 16];
        let nonce = [4u8; NONCE_LEN];
        let ct = seal(b"right", &p, &salt, &nonce, b"", b"data").unwrap();
        let err = open(b"wrong", &p, &salt, &nonce, b"", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn wrong_aad_fails() {
        let p = cheap();
        let salt = [5u8; 16];
        let nonce = [6u8; NONCE_LEN];
        let ct = seal(b"pw", &p, &salt, &nonce, b"aad-1", b"data").unwrap();
        let err = open(b"pw", &p, &salt, &nonce, b"aad-2", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let p = cheap();
        let salt = [7u8; 16];
        let nonce = [8u8; NONCE_LEN];
        let mut ct = seal(b"pw", &p, &salt, &nonce, b"", b"data").unwrap();
        ct[0] ^= 0x01;
        let err = open(b"pw", &p, &salt, &nonce, b"", &ct).unwrap_err();
        assert!(matches!(err, CryptoError::AeadDecryptFailed));
    }

    #[test]
    fn wrong_nonce_length_rejected_not_panicked() {
        let p = cheap();
        let short = [0u8; 12];
        let err = seal(b"pw", &p, &[0u8; 16], &short, b"", b"data").unwrap_err();
        assert!(matches!(
            err,
            CryptoError::InvalidNonceLength {
                expected: 24,
                got: 12
            }
        ));
        let err = open(b"pw", &p, &[0u8; 16], &short, b"", b"xxxxxxxxxxxxxxxxx").unwrap_err();
        assert!(matches!(err, CryptoError::InvalidNonceLength { .. }));
    }

    #[test]
    fn invalid_params_rejected() {
        // t_cost = 0 is below Argon2's minimum (1).
        let err = Argon2idParams::new(65536, 0, 1).unwrap_err();
        assert!(matches!(err, CryptoError::InvalidKdfParams));
    }
}

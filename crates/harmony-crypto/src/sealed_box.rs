//! Anonymous-sender public-key encryption ("sealed box") over X25519.
//!
//! Encrypts a message to a recipient's X25519 public key such that only the
//! holder of the matching private key can open it, while the sender stays
//! anonymous (a fresh ephemeral keypair is used per message). This is **not**
//! NaCl/libsodium `crypto_box_seal` — it is a bespoke Harmony construction and
//! is byte-incompatible with libsodium. Domain separation is carried entirely
//! by the caller-supplied `info` label fed into the HKDF step (no AEAD AAD).
//!
//! ```text
//! seal(recipient_pub, plaintext, info):
//!   eph_sk        = random X25519 secret            (from the caller's RNG)
//!   shared        = X25519(eph_sk, recipient_pub)   (reject all-zero → InvalidPublicKey)
//!   key           = HKDF-SHA256(ikm=shared, salt=none, info=info, len=32)
//!   nonce         = random 12 bytes                 (from the caller's RNG)
//!   ct            = ChaCha20Poly1305(key, nonce, plaintext, aad="")
//!   envelope      = eph_pub(32) ‖ nonce(12) ‖ ct+tag        // ct+tag = plaintext_len + 16
//! ```
//!
//! `open` reverses it: split the envelope, X25519 with the recipient's private
//! key, re-derive the key, and AEAD-decrypt.
//!
//! # Warning
//!
//! The `info` label and this construction are a **frozen wire contract** —
//! existing ciphertext derives its AEAD key from a specific `info` string
//! (e.g. harmony-client's `harmony-zeb-249-epoch-key-seal`). Changing the
//! construction, the framing offsets, or an `info` value makes previously
//! sealed data unopenable. A fresh, unique nonce per message is mandatory:
//! callers MUST pass a cryptographically secure RNG.
//!
//! # Security: the ephemeral-key header is not authenticated
//!
//! The AEAD runs with empty AAD, so the 32-byte ephemeral-public-key header is
//! not bound to the ciphertext (this matches the frozen client wire format).
//! Because X25519 ignores the high bit of the u-coordinate, an attacker can
//! flip the top bit of the final header byte and the recipient still recovers
//! the **same** plaintext — the envelope is malleable in that bit. This is
//! benign for an anonymous sealed box: Poly1305 still guarantees plaintext
//! integrity, and the ephemeral key is unauthenticated by design. Callers must
//! nonetheless never treat a sealed envelope's exact bytes as a unique
//! identifier (e.g. for dedup or content-addressing), since a distinct-looking
//! envelope can decrypt to identical plaintext.

use alloc::vec::Vec;
use rand_core::CryptoRngCore;
use x25519_dalek::{EphemeralSecret, PublicKey, StaticSecret};

use crate::aead;
use crate::hkdf::DerivedKey;
use crate::CryptoError;

/// Length of the ephemeral-public-key prefix in a sealed envelope (32 bytes).
pub const EPHEMERAL_PUBKEY_LEN: usize = 32;

/// Fixed per-message overhead of a sealed envelope over the plaintext length:
/// ephemeral public key (32) + nonce (12) + Poly1305 tag (16) = 60 bytes.
pub const SEALED_BOX_OVERHEAD_LEN: usize =
    EPHEMERAL_PUBKEY_LEN + aead::NONCE_LENGTH + aead::TAG_LENGTH;

/// Seal `plaintext` to `recipient_x25519_pub` under domain label `info`.
///
/// A fresh ephemeral X25519 keypair and a fresh 12-byte nonce are drawn from
/// `rng` for every call — reusing an RNG that repeats output breaks security.
/// Returns the envelope `eph_pub ‖ nonce ‖ ciphertext+tag`
/// (`SEALED_BOX_OVERHEAD_LEN + plaintext.len()` bytes).
///
/// Rejects a recipient key that yields an all-zero (contributory-weak) shared
/// secret with [`CryptoError::InvalidPublicKey`].
pub fn seal(
    recipient_x25519_pub: &[u8; 32],
    plaintext: &[u8],
    info: &[u8],
    rng: &mut impl CryptoRngCore,
) -> Result<Vec<u8>, CryptoError> {
    let recipient_pub = PublicKey::from(*recipient_x25519_pub);
    let ephemeral = EphemeralSecret::random_from_rng(&mut *rng);
    let ephemeral_pub_bytes = *PublicKey::from(&ephemeral).as_bytes();

    let shared = ephemeral.diffie_hellman(&recipient_pub);
    if shared.as_bytes() == &[0u8; 32] {
        return Err(CryptoError::InvalidPublicKey);
    }
    let key = DerivedKey::new(shared.as_bytes(), None, info, aead::KEY_LENGTH)?;
    let key_arr: &[u8; aead::KEY_LENGTH] = key
        .as_bytes()
        .try_into()
        .expect("HKDF produced KEY_LENGTH bytes");

    let nonce = aead::generate_nonce(rng);
    let ciphertext = aead::encrypt(key_arr, &nonce, plaintext, b"")?;

    let mut out = Vec::with_capacity(EPHEMERAL_PUBKEY_LEN + aead::NONCE_LENGTH + ciphertext.len());
    out.extend_from_slice(&ephemeral_pub_bytes);
    out.extend_from_slice(&nonce);
    out.extend_from_slice(&ciphertext);
    Ok(out)
}

/// Open a sealed envelope with the recipient's X25519 private key under the
/// same domain label `info` used to seal it.
///
/// Returns [`CryptoError::CiphertextTooShort`] if the envelope is shorter than
/// [`SEALED_BOX_OVERHEAD_LEN`], [`CryptoError::InvalidPublicKey`] if the
/// embedded ephemeral key yields an all-zero shared secret, and
/// [`CryptoError::AeadDecryptFailed`] on wrong key / wrong `info` / tampering.
pub fn open(
    recipient_x25519_priv: &[u8; 32],
    sealed: &[u8],
    info: &[u8],
) -> Result<Vec<u8>, CryptoError> {
    if sealed.len() < SEALED_BOX_OVERHEAD_LEN {
        return Err(CryptoError::CiphertextTooShort);
    }
    let ephemeral_pub_bytes: [u8; 32] = sealed[0..EPHEMERAL_PUBKEY_LEN]
        .try_into()
        .expect("checked length");
    let nonce_bytes: [u8; aead::NONCE_LENGTH] = sealed
        [EPHEMERAL_PUBKEY_LEN..EPHEMERAL_PUBKEY_LEN + aead::NONCE_LENGTH]
        .try_into()
        .expect("checked length");
    let ciphertext = &sealed[EPHEMERAL_PUBKEY_LEN + aead::NONCE_LENGTH..];

    let recipient_secret = StaticSecret::from(*recipient_x25519_priv);
    let shared = recipient_secret.diffie_hellman(&PublicKey::from(ephemeral_pub_bytes));
    if shared.as_bytes() == &[0u8; 32] {
        return Err(CryptoError::InvalidPublicKey);
    }
    let key = DerivedKey::new(shared.as_bytes(), None, info, aead::KEY_LENGTH)?;
    let key_arr: &[u8; aead::KEY_LENGTH] = key
        .as_bytes()
        .try_into()
        .expect("HKDF produced KEY_LENGTH bytes");

    aead::decrypt(key_arr, &nonce_bytes, ciphertext, b"")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::x25519::{ed25519_priv_to_x25519, ed25519_pub_to_x25519};
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    // Fixed fixtures shared with the harmony-client cross-repo parity test.
    const KAT_ED_SEED: [u8; 32] = [0x24u8; 32];
    const KAT_INFO: &[u8] = b"harmony-zeb-249-epoch-key-seal";
    const KAT_PLAINTEXT: &[u8] = b"harmony sealed-box known-answer test vector";

    fn kat_recipient() -> ([u8; 32], [u8; 32]) {
        let sk = SigningKey::from_bytes(&KAT_ED_SEED);
        let recipient_priv = *ed25519_priv_to_x25519(&sk);
        let recipient_pub = ed25519_pub_to_x25519(&sk.verifying_key().to_bytes()).unwrap();
        (recipient_priv, recipient_pub)
    }

    #[test]
    fn round_trip() {
        let (recipient_priv, recipient_pub) = kat_recipient();
        let sealed = seal(&recipient_pub, KAT_PLAINTEXT, KAT_INFO, &mut OsRng).unwrap();
        assert_eq!(sealed.len(), SEALED_BOX_OVERHEAD_LEN + KAT_PLAINTEXT.len());
        let opened = open(&recipient_priv, &sealed, KAT_INFO).unwrap();
        assert_eq!(opened, KAT_PLAINTEXT);
    }

    #[test]
    fn round_trip_empty_plaintext() {
        let (recipient_priv, recipient_pub) = kat_recipient();
        let sealed = seal(&recipient_pub, b"", KAT_INFO, &mut OsRng).unwrap();
        assert_eq!(sealed.len(), SEALED_BOX_OVERHEAD_LEN);
        assert_eq!(open(&recipient_priv, &sealed, KAT_INFO).unwrap(), b"");
    }

    #[test]
    fn wrong_info_fails() {
        let (recipient_priv, recipient_pub) = kat_recipient();
        let sealed = seal(&recipient_pub, KAT_PLAINTEXT, KAT_INFO, &mut OsRng).unwrap();
        assert!(matches!(
            open(&recipient_priv, &sealed, b"different-domain"),
            Err(CryptoError::AeadDecryptFailed)
        ));
    }

    #[test]
    fn wrong_key_fails() {
        let (_, recipient_pub) = kat_recipient();
        let sealed = seal(&recipient_pub, KAT_PLAINTEXT, KAT_INFO, &mut OsRng).unwrap();
        let other_priv = *ed25519_priv_to_x25519(&SigningKey::from_bytes(&[0x99u8; 32]));
        assert!(matches!(
            open(&other_priv, &sealed, KAT_INFO),
            Err(CryptoError::AeadDecryptFailed)
        ));
    }

    #[test]
    fn seal_rejects_all_zero_recipient() {
        // The all-zero X25519 public key is a low-order point: every DH with it
        // yields an all-zero shared secret, which must be rejected up front.
        assert!(matches!(
            seal(&[0u8; 32], KAT_PLAINTEXT, KAT_INFO, &mut OsRng),
            Err(CryptoError::InvalidPublicKey)
        ));
    }

    #[test]
    fn open_rejects_all_zero_ephemeral() {
        // A valid-length envelope whose ephemeral-pubkey header is all zero
        // forces an all-zero shared secret on open — rejected before the AEAD
        // layer with InvalidPublicKey rather than AeadDecryptFailed.
        let (recipient_priv, _) = kat_recipient();
        let envelope = [0u8; SEALED_BOX_OVERHEAD_LEN];
        assert!(matches!(
            open(&recipient_priv, &envelope, KAT_INFO),
            Err(CryptoError::InvalidPublicKey)
        ));
    }

    #[test]
    fn open_rejects_short_envelope() {
        let (recipient_priv, _) = kat_recipient();
        let too_short = [0u8; SEALED_BOX_OVERHEAD_LEN - 1];
        assert!(matches!(
            open(&recipient_priv, &too_short, KAT_INFO),
            Err(CryptoError::CiphertextTooShort)
        ));
    }

    #[test]
    fn open_rejects_tampered_ciphertext() {
        let (recipient_priv, recipient_pub) = kat_recipient();
        let mut sealed = seal(&recipient_pub, KAT_PLAINTEXT, KAT_INFO, &mut OsRng).unwrap();
        let last = sealed.len() - 1;
        sealed[last] ^= 0x01;
        assert!(matches!(
            open(&recipient_priv, &sealed, KAT_INFO),
            Err(CryptoError::AeadDecryptFailed)
        ));
    }

    /// Frozen cross-repo decrypt-path known-answer test. The envelope below was
    /// produced once by [`seal`] and is decrypted by harmony-client's
    /// `dm_signing` open parity test against the SAME bytes — pinning that the
    /// framing offsets, HKDF schedule (`info` = zeb-249 default), and
    /// ChaCha20-Poly1305 layer are byte-identical across the two crates. DO NOT
    /// regenerate: it anchors the wire format of every sealed blob.
    #[test]
    fn frozen_open_kat() {
        let (recipient_priv, _) = kat_recipient();
        // Envelope produced once by `seal` (2026-07-23) with recipient derived
        // from Ed25519 seed [0x24; 32] and info = zeb-249 default. The matching
        // recipient X25519 private key is
        // f85a1fbfc2b76a3b6bca5c2f1f2d0884cb8f89f02fafbe4ede98efbe75d7e455.
        const FROZEN_ENVELOPE: &str = "0021bf9fce0c9b89eb3cf5f4c77cefa61c97cde1a8000902a9f86f03dc53bc158188f93da1cff420a0dda47f0b533087cc2812a74aaefe84df65cfe51315577cf0cecb77a5bc86d85ee14bdabfd0278e014adc2126a821557947423eaae99e177c97cf069c0fc6";
        let sealed = hex::decode(FROZEN_ENVELOPE).unwrap();
        assert_eq!(sealed.len(), SEALED_BOX_OVERHEAD_LEN + KAT_PLAINTEXT.len());
        let opened = open(&recipient_priv, &sealed, KAT_INFO).unwrap();
        assert_eq!(opened, KAT_PLAINTEXT);
    }
}

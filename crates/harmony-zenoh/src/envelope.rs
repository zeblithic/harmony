//! Binary E2EE message envelope for Harmony.
//!
//! Wire format (33-byte fixed header + variable ciphertext):
//!
//! ```text
//! [1B ver|type][16B sender_addr][12B nonce][4B sequence][N bytes ciphertext+tag]
//! ```
//!
//! The 33-byte header is passed as AAD to ChaCha20-Poly1305, cryptographically
//! binding the routing metadata to the encrypted payload.

use alloc::{
    string::{String, ToString},
    vec::Vec,
};
use harmony_crypto::{aead, hkdf};
use harmony_identity::{Identity, PrivateIdentity};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::ZenohError;

/// Current envelope format version.
pub const VERSION: u8 = 1;

/// Fixed header size in bytes.
pub const HEADER_SIZE: usize = 1 + 16 + 12 + 4; // 33

/// Minimum envelope size: header + Poly1305 tag (16 bytes for empty plaintext).
pub const MIN_ENVELOPE_SIZE: usize = HEADER_SIZE + aead::TAG_LENGTH;

/// Message type carried in the envelope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageType {
    /// Data message (publish).
    Put = 0,
    /// Tombstone (delete).
    Del = 1,
}

impl MessageType {
    /// Decode from the low 4 bits of the version|type byte.
    pub fn from_u8(val: u8) -> Result<Self, ZenohError> {
        match val {
            0 => Ok(Self::Put),
            1 => Ok(Self::Del),
            other => Err(ZenohError::InvalidMessageType(other)),
        }
    }
}

/// HKDF info string for envelope key derivation.
const HKDF_INFO: &[u8] = b"harmony-envelope-v1";

/// Derive shared symmetric key from ECDH + HKDF.
/// Salt = sender_addr || recipient_addr for directionality.
fn derive_shared_key(
    local: &PrivateIdentity,
    remote: &Identity,
    sender_address: &[u8; 16],
    recipient_address: &[u8; 16],
) -> Result<[u8; aead::KEY_LENGTH], String> {
    let shared_secret = local.ecdh(&remote.encryption_key);

    let mut salt = [0u8; 32];
    salt[..16].copy_from_slice(sender_address);
    salt[16..].copy_from_slice(recipient_address);

    let mut key_bytes = hkdf::derive_key(
        shared_secret.as_bytes(),
        Some(&salt),
        HKDF_INFO,
        aead::KEY_LENGTH,
    )
    .map_err(|e| e.to_string())?;

    let mut key = [0u8; aead::KEY_LENGTH];
    key.copy_from_slice(&key_bytes);
    key_bytes.zeroize();
    Ok(key)
}

/// A decoded Harmony message envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use]
pub struct HarmonyEnvelope {
    pub version: u8,
    pub msg_type: MessageType,
    pub sender_address: [u8; 16],
    pub sequence: u32,
    pub plaintext: Vec<u8>,
}

impl HarmonyEnvelope {
    /// Seal a plaintext message into an encrypted envelope.
    ///
    /// Performs ECDH key agreement between `sender` and `recipient`, derives a
    /// directional symmetric key via HKDF, encrypts with ChaCha20-Poly1305
    /// using the 33-byte header as AAD.
    pub fn seal(
        rng: &mut impl CryptoRngCore,
        msg_type: MessageType,
        sender: &PrivateIdentity,
        recipient: &Identity,
        sequence: u32,
        plaintext: &[u8],
    ) -> Result<Vec<u8>, ZenohError> {
        let nonce = aead::generate_nonce(rng);
        let sender_addr = sender.public_identity().address_hash;
        let mut key = derive_shared_key(sender, recipient, &sender_addr, &recipient.address_hash)
            .map_err(ZenohError::SealFailed)?;

        // Build header
        let mut header = [0u8; HEADER_SIZE];
        header[0] = (VERSION << 4) | (msg_type as u8);
        header[1..17].copy_from_slice(&sender_addr);
        header[17..29].copy_from_slice(&nonce);
        header[29..33].copy_from_slice(&sequence.to_be_bytes());

        // Encrypt with header as AAD
        let result = aead::encrypt(&key, &nonce, plaintext, &header)
            .map_err(|e| ZenohError::SealFailed(e.to_string()));
        key.zeroize();
        let ciphertext = result?;

        let mut out = Vec::with_capacity(HEADER_SIZE + ciphertext.len());
        out.extend_from_slice(&header);
        out.extend_from_slice(&ciphertext);
        Ok(out)
    }

    /// Open an encrypted envelope and recover the plaintext.
    ///
    /// Validates the header, derives the shared key from the recipient's
    /// private identity and the sender's public identity, then decrypts
    /// and authenticates the ciphertext (including the header as AAD).
    pub fn open(
        recipient: &PrivateIdentity,
        sender: &Identity,
        data: &[u8],
    ) -> Result<HarmonyEnvelope, ZenohError> {
        if data.len() < MIN_ENVELOPE_SIZE {
            return Err(ZenohError::EnvelopeTooShort(data.len(), MIN_ENVELOPE_SIZE));
        }

        let header = &data[..HEADER_SIZE];
        let ciphertext = &data[HEADER_SIZE..];

        let version = header[0] >> 4;
        if version != VERSION {
            return Err(ZenohError::UnsupportedVersion(version));
        }
        let msg_type = MessageType::from_u8(header[0] & 0x0F)?;

        let mut sender_address = [0u8; 16];
        sender_address.copy_from_slice(&header[1..17]);

        let mut nonce = [0u8; aead::NONCE_LENGTH];
        nonce.copy_from_slice(&header[17..29]);

        let sequence = u32::from_be_bytes(header[29..33].try_into().unwrap());

        // Validate sender_address matches the provided sender identity
        if sender_address != sender.address_hash {
            return Err(ZenohError::OpenFailed("sender address mismatch".into()));
        }

        // Derive shared key (salt: sender || recipient, same order as seal)
        let mut key = derive_shared_key(
            recipient,
            sender,
            &sender_address,
            &recipient.public_identity().address_hash,
        )
        .map_err(ZenohError::OpenFailed)?;

        let result = aead::decrypt(&key, &nonce, ciphertext, header)
            .map_err(|e| ZenohError::OpenFailed(e.to_string()));
        key.zeroize();
        let plaintext = result?;

        Ok(HarmonyEnvelope {
            version,
            msg_type,
            sender_address,
            sequence,
            plaintext,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn seal_open_roundtrip() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);
        let plaintext = b"hello harmony";

        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            42,
            plaintext,
        )
        .unwrap();

        let envelope =
            HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert_eq!(envelope.version, VERSION);
        assert_eq!(envelope.msg_type, MessageType::Put);
        assert_eq!(
            envelope.sender_address,
            sender.public_identity().address_hash
        );
        assert_eq!(envelope.sequence, 42);
        assert_eq!(envelope.plaintext, plaintext);
    }

    #[test]
    fn del_type_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Del,
            &sender,
            recipient.public_identity(),
            0,
            b"",
        )
        .unwrap();

        let envelope =
            HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert_eq!(envelope.msg_type, MessageType::Del);
        assert!(envelope.plaintext.is_empty());
    }

    #[test]
    fn empty_payload_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            0,
            b"",
        )
        .unwrap();

        assert_eq!(sealed.len(), MIN_ENVELOPE_SIZE);
        let envelope =
            HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert!(envelope.plaintext.is_empty());
    }

    #[test]
    fn large_payload_roundtrips() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);
        let plaintext = vec![0xAB; 8192];

        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            1,
            &plaintext,
        )
        .unwrap();

        let envelope =
            HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
        assert_eq!(envelope.plaintext, plaintext);
    }

    #[test]
    fn tampered_header_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            1,
            b"secret",
        )
        .unwrap();

        sealed[30] ^= 0xFF; // Flip byte in sequence field
        assert!(HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).is_err());
    }

    #[test]
    fn tampered_ciphertext_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            1,
            b"secret",
        )
        .unwrap();

        let last = sealed.len() - 1;
        sealed[last] ^= 0xFF;
        assert!(HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).is_err());
    }

    #[test]
    fn wrong_recipient_fails() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);
        let wrong = PrivateIdentity::generate(&mut rng);

        let sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            1,
            b"secret",
        )
        .unwrap();

        assert!(HarmonyEnvelope::open(&wrong, sender.public_identity(), &sealed).is_err());
    }

    #[test]
    fn sequence_preserved() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        for seq in [0, 1, 255, 65535, u32::MAX] {
            let sealed = HarmonyEnvelope::seal(
                &mut rng,
                MessageType::Put,
                &sender,
                recipient.public_identity(),
                seq,
                b"x",
            )
            .unwrap();
            let envelope =
                HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed).unwrap();
            assert_eq!(envelope.sequence, seq);
        }
    }

    #[test]
    fn version_type_byte_encoding() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let sealed_put = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            0,
            b"",
        )
        .unwrap();
        assert_eq!(sealed_put[0], 0x10); // version 1, type Put (0)

        let sealed_del = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Del,
            &sender,
            recipient.public_identity(),
            0,
            b"",
        )
        .unwrap();
        assert_eq!(sealed_del[0], 0x11); // version 1, type Del (1)
    }

    #[test]
    fn envelope_too_short_rejected() {
        let mut rng = OsRng;
        let recipient = PrivateIdentity::generate(&mut rng);
        let sender = PrivateIdentity::generate(&mut rng);

        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &[0u8; 32]);
        assert!(matches!(
            result,
            Err(ZenohError::EnvelopeTooShort(32, MIN_ENVELOPE_SIZE))
        ));
    }

    #[test]
    fn unsupported_version_rejected() {
        let mut rng = OsRng;
        let sender = PrivateIdentity::generate(&mut rng);
        let recipient = PrivateIdentity::generate(&mut rng);

        let mut sealed = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &sender,
            recipient.public_identity(),
            0,
            b"x",
        )
        .unwrap();

        sealed[0] = (2 << 4) | (sealed[0] & 0x0F); // version 2
        let result = HarmonyEnvelope::open(&recipient, sender.public_identity(), &sealed);
        assert!(matches!(result, Err(ZenohError::UnsupportedVersion(2))));
    }

    #[test]
    fn directionality_produces_different_ciphertexts() {
        let mut rng = OsRng;
        let alice = PrivateIdentity::generate(&mut rng);
        let bob = PrivateIdentity::generate(&mut rng);

        let a_to_b = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &alice,
            bob.public_identity(),
            1,
            b"same",
        )
        .unwrap();

        let b_to_a = HarmonyEnvelope::seal(
            &mut rng,
            MessageType::Put,
            &bob,
            alice.public_identity(),
            1,
            b"same",
        )
        .unwrap();

        // Different ciphertexts (different keys + different nonces)
        assert_ne!(a_to_b[HEADER_SIZE..], b_to_a[HEADER_SIZE..]);

        // Both decrypt correctly
        let env1 = HarmonyEnvelope::open(&bob, alice.public_identity(), &a_to_b).unwrap();
        let env2 = HarmonyEnvelope::open(&alice, bob.public_identity(), &b_to_a).unwrap();
        assert_eq!(env1.plaintext, b"same");
        assert_eq!(env2.plaintext, b"same");
    }
}

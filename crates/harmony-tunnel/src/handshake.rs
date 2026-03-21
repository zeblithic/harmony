use alloc::vec::Vec;
use harmony_crypto::aead::KEY_LENGTH;
use harmony_crypto::hash::blake3_hash;
use harmony_crypto::ml_dsa::{MlDsaPublicKey, MlDsaSecretKey, MlDsaSignature};
use harmony_crypto::ml_kem::{MlKemCiphertext, MlKemPublicKey, MlKemSharedSecret};
use rand_core::CryptoRngCore;
use zeroize::Zeroize;

use crate::error::TunnelError;

/// ML-KEM ciphertext length.
const CT_LEN: usize = 1088;
/// ML-DSA-65 public key length.
const DSA_PK_LEN: usize = 1952;
/// ML-DSA-65 signature length.
const SIG_LEN: usize = 3309;
/// Nonce length.
const NONCE_LEN: usize = 32;

/// First handshake message: initiator -> responder.
///
/// Wire format: [CT_LEN ciphertext][DSA_PK_LEN pubkey][NONCE_LEN nonce][SIG_LEN signature]
pub struct TunnelInit {
    pub ciphertext: MlKemCiphertext,
    pub initiator_pubkey: MlDsaPublicKey,
    pub nonce: [u8; NONCE_LEN],
    pub signature: MlDsaSignature,
}

/// The total byte length of a serialized TunnelInit.
pub const TUNNEL_INIT_LEN: usize = CT_LEN + DSA_PK_LEN + NONCE_LEN + SIG_LEN;

impl TunnelInit {
    /// Create a TunnelInit message.
    ///
    /// `responder_kem_pk` is the responder's ML-KEM-768 public key (from AnnounceRecord or contacts).
    /// `initiator_dsa_pk` / `initiator_dsa_sk` are the initiator's ML-DSA-65 keypair.
    pub fn create(
        rng: &mut impl CryptoRngCore,
        responder_kem_pk: &MlKemPublicKey,
        initiator_dsa_pk: &MlDsaPublicKey,
        initiator_dsa_sk: &MlDsaSecretKey,
    ) -> Result<(Self, MlKemSharedSecret), TunnelError> {
        // 1. ML-KEM encapsulate to responder's public key
        let (ciphertext, shared_secret) =
            harmony_crypto::ml_kem::encapsulate(rng, responder_kem_pk)?;

        // 2. Generate random nonce
        let mut nonce = [0u8; NONCE_LEN];
        rng.fill_bytes(&mut nonce);

        // 3. Build signed payload: ciphertext || pubkey || nonce
        let mut signed_payload = Vec::with_capacity(CT_LEN + DSA_PK_LEN + NONCE_LEN);
        signed_payload.extend_from_slice(ciphertext.as_bytes());
        signed_payload.extend_from_slice(&initiator_dsa_pk.as_bytes());
        signed_payload.extend_from_slice(&nonce);

        // 4. Sign
        let signature = harmony_crypto::ml_dsa::sign(initiator_dsa_sk, &signed_payload)?;

        Ok((
            Self {
                ciphertext,
                initiator_pubkey: MlDsaPublicKey::from_bytes(&initiator_dsa_pk.as_bytes())
                    .map_err(|_| TunnelError::MalformedHandshake {
                        reason: "pubkey clone failed",
                    })?,
                nonce,
                signature,
            },
            shared_secret,
        ))
    }

    /// Serialize to wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(TUNNEL_INIT_LEN);
        buf.extend_from_slice(self.ciphertext.as_bytes());
        buf.extend_from_slice(&self.initiator_pubkey.as_bytes());
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(self.signature.as_bytes());
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < TUNNEL_INIT_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: TUNNEL_INIT_LEN,
                got: data.len(),
            });
        }

        let mut offset = 0;
        let ciphertext = MlKemCiphertext::from_bytes(&data[offset..offset + CT_LEN])?;
        offset += CT_LEN;

        let initiator_pubkey = MlDsaPublicKey::from_bytes(&data[offset..offset + DSA_PK_LEN])
            .map_err(|_| TunnelError::MalformedHandshake {
                reason: "invalid ML-DSA public key",
            })?;
        offset += DSA_PK_LEN;

        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&data[offset..offset + NONCE_LEN]);
        offset += NONCE_LEN;

        let signature =
            MlDsaSignature::from_bytes(&data[offset..offset + SIG_LEN]).map_err(|_| {
                TunnelError::MalformedHandshake {
                    reason: "invalid ML-DSA signature",
                }
            })?;

        Ok(Self {
            ciphertext,
            initiator_pubkey,
            nonce,
            signature,
        })
    }

    /// Verify the initiator's signature over the signed payload.
    pub fn verify(&self) -> Result<(), TunnelError> {
        let mut signed_payload = Vec::with_capacity(CT_LEN + DSA_PK_LEN + NONCE_LEN);
        signed_payload.extend_from_slice(self.ciphertext.as_bytes());
        signed_payload.extend_from_slice(&self.initiator_pubkey.as_bytes());
        signed_payload.extend_from_slice(&self.nonce);

        harmony_crypto::ml_dsa::verify(&self.initiator_pubkey, &signed_payload, &self.signature)
            .map_err(|_| TunnelError::SignatureVerificationFailed)
    }
}

/// Second handshake message: responder -> initiator.
///
/// Wire format: [DSA_PK_LEN pubkey][NONCE_LEN nonce][SIG_LEN signature]
pub struct TunnelAccept {
    pub responder_pubkey: MlDsaPublicKey,
    pub nonce: [u8; NONCE_LEN],
    pub signature: MlDsaSignature,
}

/// The total byte length of a serialized TunnelAccept.
pub const TUNNEL_ACCEPT_LEN: usize = DSA_PK_LEN + NONCE_LEN + SIG_LEN;

/// Compute the BLAKE3 transcript hash over both handshake messages.
///
/// Includes all fields of TunnelInit and the non-signature fields of TunnelAccept,
/// binding both peers to the full handshake transcript.
pub fn compute_transcript_hash(init: &TunnelInit, accept: &TunnelAccept) -> [u8; 32] {
    let mut transcript = Vec::with_capacity(TUNNEL_INIT_LEN + DSA_PK_LEN + NONCE_LEN);
    // Full TunnelInit (including signature — binds initiator's identity)
    transcript.extend_from_slice(init.ciphertext.as_bytes());
    transcript.extend_from_slice(&init.initiator_pubkey.as_bytes());
    transcript.extend_from_slice(&init.nonce);
    transcript.extend_from_slice(init.signature.as_bytes());
    // TunnelAccept fields before signature
    transcript.extend_from_slice(&accept.responder_pubkey.as_bytes());
    transcript.extend_from_slice(&accept.nonce);

    blake3_hash(&transcript)
}

/// Compute the BLAKE3 transcript hash from raw fields, without requiring a
/// constructed `TunnelAccept`. Used during `TunnelAccept::create` before
/// the signature exists.
fn compute_transcript_hash_raw(
    init: &TunnelInit,
    responder_pubkey: &MlDsaPublicKey,
    responder_nonce: &[u8; NONCE_LEN],
) -> [u8; 32] {
    let mut transcript = Vec::with_capacity(TUNNEL_INIT_LEN + DSA_PK_LEN + NONCE_LEN);
    // Full TunnelInit (including signature — binds initiator's identity)
    transcript.extend_from_slice(init.ciphertext.as_bytes());
    transcript.extend_from_slice(&init.initiator_pubkey.as_bytes());
    transcript.extend_from_slice(&init.nonce);
    transcript.extend_from_slice(init.signature.as_bytes());
    // TunnelAccept non-signature fields
    transcript.extend_from_slice(&responder_pubkey.as_bytes());
    transcript.extend_from_slice(responder_nonce);

    blake3_hash(&transcript)
}

impl TunnelAccept {
    /// Create a TunnelAccept message in response to a TunnelInit.
    ///
    /// Decapsulates the shared secret, signs the transcript hash.
    pub fn create(
        rng: &mut impl CryptoRngCore,
        responder_kem_sk: &harmony_crypto::ml_kem::MlKemSecretKey,
        responder_dsa_pk: &MlDsaPublicKey,
        responder_dsa_sk: &MlDsaSecretKey,
        init: &TunnelInit,
    ) -> Result<(Self, MlKemSharedSecret), TunnelError> {
        // 1. Verify initiator's signature
        init.verify()?;

        // 2. Decapsulate shared secret
        let shared_secret =
            harmony_crypto::ml_kem::decapsulate(responder_kem_sk, &init.ciphertext)?;

        // 3. Generate responder nonce
        let mut nonce = [0u8; NONCE_LEN];
        rng.fill_bytes(&mut nonce);

        // 4. Clone the responder pubkey for the accept message
        let pubkey_clone =
            MlDsaPublicKey::from_bytes(&responder_dsa_pk.as_bytes()).map_err(|_| {
                TunnelError::MalformedHandshake {
                    reason: "pubkey clone failed",
                }
            })?;

        // 5. Compute transcript hash from raw fields (avoids needing a dummy signature)
        let transcript = compute_transcript_hash_raw(init, &pubkey_clone, &nonce);

        // 6. Sign the transcript
        let signature = harmony_crypto::ml_dsa::sign(responder_dsa_sk, &transcript)?;

        Ok((
            Self {
                responder_pubkey: pubkey_clone,
                nonce,
                signature,
            },
            shared_secret,
        ))
    }

    /// Serialize to wire format.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(TUNNEL_ACCEPT_LEN);
        buf.extend_from_slice(&self.responder_pubkey.as_bytes());
        buf.extend_from_slice(&self.nonce);
        buf.extend_from_slice(self.signature.as_bytes());
        buf
    }

    /// Deserialize from wire format.
    pub fn from_bytes(data: &[u8]) -> Result<Self, TunnelError> {
        if data.len() < TUNNEL_ACCEPT_LEN {
            return Err(TunnelError::FrameTooShort {
                expected: TUNNEL_ACCEPT_LEN,
                got: data.len(),
            });
        }

        let mut offset = 0;
        let responder_pubkey = MlDsaPublicKey::from_bytes(&data[offset..offset + DSA_PK_LEN])
            .map_err(|_| TunnelError::MalformedHandshake {
                reason: "invalid ML-DSA public key",
            })?;
        offset += DSA_PK_LEN;

        let mut nonce = [0u8; NONCE_LEN];
        nonce.copy_from_slice(&data[offset..offset + NONCE_LEN]);
        offset += NONCE_LEN;

        let signature =
            MlDsaSignature::from_bytes(&data[offset..offset + SIG_LEN]).map_err(|_| {
                TunnelError::MalformedHandshake {
                    reason: "invalid ML-DSA signature",
                }
            })?;

        Ok(Self {
            responder_pubkey,
            nonce,
            signature,
        })
    }

    /// Verify the responder's transcript signature.
    ///
    /// Caller must provide the original TunnelInit to reconstruct the transcript.
    pub fn verify(&self, init: &TunnelInit) -> Result<(), TunnelError> {
        let transcript = compute_transcript_hash(init, self);
        harmony_crypto::ml_dsa::verify(&self.responder_pubkey, &transcript, &self.signature)
            .map_err(|_| TunnelError::SignatureVerificationFailed)
    }
}

/// Directional session keys derived from the handshake.
pub struct SessionKeys {
    /// Initiator-to-responder encryption key (32 bytes).
    pub i2r_key: [u8; KEY_LENGTH],
    /// Responder-to-initiator encryption key (32 bytes).
    pub r2i_key: [u8; KEY_LENGTH],
}

impl Drop for SessionKeys {
    fn drop(&mut self) {
        self.i2r_key.zeroize();
        self.r2i_key.zeroize();
    }
}

/// Derive directional session keys from the ML-KEM shared secret and both nonces.
///
/// Uses HKDF-SHA256 with:
/// - IKM: the ML-KEM shared secret (32 bytes)
/// - Salt: nonce_i || nonce_r (64 bytes)
/// - Info: "i2r" or "r2i" (directional labels)
pub fn derive_session_keys(
    shared_secret: &[u8],
    nonce_i: &[u8; NONCE_LEN],
    nonce_r: &[u8; NONCE_LEN],
) -> SessionKeys {
    let mut salt = [0u8; NONCE_LEN * 2];
    salt[..NONCE_LEN].copy_from_slice(nonce_i);
    salt[NONCE_LEN..].copy_from_slice(nonce_r);

    let i2r_bytes =
        harmony_crypto::hkdf::derive_key(shared_secret, Some(&salt), b"i2r", KEY_LENGTH)
            .expect("HKDF-SHA256 with 32-byte output cannot fail");

    let r2i_bytes =
        harmony_crypto::hkdf::derive_key(shared_secret, Some(&salt), b"r2i", KEY_LENGTH)
            .expect("HKDF-SHA256 with 32-byte output cannot fail");

    let mut i2r_key = [0u8; KEY_LENGTH];
    let mut r2i_key = [0u8; KEY_LENGTH];
    i2r_key.copy_from_slice(&i2r_bytes);
    r2i_key.copy_from_slice(&r2i_bytes);

    SessionKeys { i2r_key, r2i_key }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn tunnel_init_roundtrip() {
        let (ml_kem_pk, _ml_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (ml_dsa_pk, ml_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init, _shared_secret) = TunnelInit::create(
            &mut OsRng, &ml_kem_pk, // responder's ML-KEM public key
            &ml_dsa_pk, // initiator's ML-DSA public key
            &ml_dsa_sk, // initiator's ML-DSA secret key
        )
        .unwrap();

        let bytes = init.to_bytes();
        let parsed = TunnelInit::from_bytes(&bytes).unwrap();

        assert_eq!(init.ciphertext.as_bytes(), parsed.ciphertext.as_bytes());
        assert_eq!(
            init.initiator_pubkey.as_bytes(),
            parsed.initiator_pubkey.as_bytes()
        );
        assert_eq!(init.nonce, parsed.nonce);
    }

    #[test]
    fn tunnel_accept_roundtrip() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        // Initiator creates TunnelInit
        let (init_msg, init_shared_secret) =
            TunnelInit::create(&mut OsRng, &resp_kem_pk, &init_dsa_pk, &init_dsa_sk).unwrap();

        // Responder creates TunnelAccept
        let (accept_msg, resp_shared_secret) = TunnelAccept::create(
            &mut OsRng,
            &resp_kem_sk,
            &resp_dsa_pk,
            &resp_dsa_sk,
            &init_msg,
        )
        .unwrap();

        // Both derived the same shared secret
        assert_eq!(init_shared_secret.as_bytes(), resp_shared_secret.as_bytes());

        // Roundtrip serialization
        let bytes = accept_msg.to_bytes();
        let parsed = TunnelAccept::from_bytes(&bytes).unwrap();
        assert_eq!(accept_msg.nonce, parsed.nonce);
    }

    #[test]
    fn transcript_signature_verifies() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init_msg, _) =
            TunnelInit::create(&mut OsRng, &resp_kem_pk, &init_dsa_pk, &init_dsa_sk).unwrap();

        let (accept_msg, _) = TunnelAccept::create(
            &mut OsRng,
            &resp_kem_sk,
            &resp_dsa_pk,
            &resp_dsa_sk,
            &init_msg,
        )
        .unwrap();

        // Verify the transcript signature
        let transcript = compute_transcript_hash(&init_msg, &accept_msg);
        harmony_crypto::ml_dsa::verify(
            &accept_msg.responder_pubkey,
            &transcript,
            &accept_msg.signature,
        )
        .unwrap();
    }

    #[test]
    fn session_keys_derived_correctly() {
        let (resp_kem_pk, resp_kem_sk) = harmony_crypto::ml_kem::generate(&mut OsRng);
        let (init_dsa_pk, init_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);
        let (resp_dsa_pk, resp_dsa_sk) = harmony_crypto::ml_dsa::generate(&mut OsRng);

        let (init_msg, init_ss) =
            TunnelInit::create(&mut OsRng, &resp_kem_pk, &init_dsa_pk, &init_dsa_sk).unwrap();

        let (accept_msg, resp_ss) = TunnelAccept::create(
            &mut OsRng,
            &resp_kem_sk,
            &resp_dsa_pk,
            &resp_dsa_sk,
            &init_msg,
        )
        .unwrap();

        // Both sides derive the same key pair
        let init_keys = derive_session_keys(init_ss.as_bytes(), &init_msg.nonce, &accept_msg.nonce);
        let resp_keys = derive_session_keys(resp_ss.as_bytes(), &init_msg.nonce, &accept_msg.nonce);

        // Initiator's send key == Responder's receive key
        assert_eq!(init_keys.i2r_key, resp_keys.i2r_key);
        // Responder's send key == Initiator's receive key
        assert_eq!(init_keys.r2i_key, resp_keys.r2i_key);
        // Keys are different from each other
        assert_ne!(init_keys.i2r_key, init_keys.r2i_key);
    }
}

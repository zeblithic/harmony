use crate::OwnerError;
use ed25519_dalek::{Signer, SigningKey, VerifyingKey, Signature};

/// Domain tags per cert type. Every signature is computed over
/// `tag || canonical_cbor_bytes(payload)`, which prevents a signature
/// valid in one cert context from being accepted in another.
pub mod tags {
    pub const ENROLLMENT: &[u8] = b"harmony-owner/v1/Enrollment";
    pub const VOUCHING: &[u8] = b"harmony-owner/v1/Vouching";
    pub const LIVENESS: &[u8] = b"harmony-owner/v1/Liveness";
    pub const REVOCATION: &[u8] = b"harmony-owner/v1/Revocation";
    pub const RECLAMATION: &[u8] = b"harmony-owner/v1/Reclamation";
}

pub fn sign_with_tag(
    sk: &SigningKey,
    tag: &[u8],
    payload_bytes: &[u8],
) -> Vec<u8> {
    let mut buf = Vec::with_capacity(tag.len() + payload_bytes.len());
    buf.extend_from_slice(tag);
    buf.extend_from_slice(payload_bytes);
    sk.sign(&buf).to_bytes().to_vec()
}

pub fn verify_with_tag(
    vk: &VerifyingKey,
    tag: &[u8],
    payload_bytes: &[u8],
    signature: &[u8],
    cert_type: &'static str,
) -> Result<(), OwnerError> {
    let sig_bytes: [u8; 64] = signature
        .try_into()
        .map_err(|_| OwnerError::InvalidSignature { cert_type })?;
    let sig = Signature::from_bytes(&sig_bytes);
    let mut buf = Vec::with_capacity(tag.len() + payload_bytes.len());
    buf.extend_from_slice(tag);
    buf.extend_from_slice(payload_bytes);
    vk.verify_strict(&buf, &sig).map_err(|_| OwnerError::InvalidSignature { cert_type })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn fresh_keypair() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    #[test]
    fn sign_then_verify_roundtrip() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let payload = b"hello, world";
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, payload);
        verify_with_tag(&vk, tags::ENROLLMENT, payload, &sig, "Enrollment").unwrap();
    }

    #[test]
    fn signature_with_wrong_tag_rejected() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let payload = b"hello, world";
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, payload);
        let result = verify_with_tag(&vk, tags::VOUCHING, payload, &sig, "Vouching");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn signature_with_wrong_payload_rejected() {
        let sk = fresh_keypair();
        let vk = sk.verifying_key();
        let sig = sign_with_tag(&sk, tags::ENROLLMENT, b"original");
        let result = verify_with_tag(&vk, tags::ENROLLMENT, b"tampered", &sig, "Enrollment");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn signature_with_wrong_key_rejected() {
        let sk_a = fresh_keypair();
        let sk_b = fresh_keypair();
        let vk_b = sk_b.verifying_key();
        let payload = b"hello";
        let sig = sign_with_tag(&sk_a, tags::ENROLLMENT, payload);
        let result = verify_with_tag(&vk_b, tags::ENROLLMENT, payload, &sig, "Enrollment");
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }
}

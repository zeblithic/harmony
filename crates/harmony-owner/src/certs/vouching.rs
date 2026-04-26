use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const VOUCHING_VERSION: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Stance {
    Vouch,
    Challenge,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VouchingCert {
    pub version: u8,
    #[serde(with = "crate::cbor::arr16")]
    pub owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    pub signer: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    pub target: [u8; 16],
    pub stance: Stance,
    pub issued_at: u64,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct VouchingSigningPayload {
    version: u8,
    #[serde(with = "crate::cbor::arr16")]
    owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    signer: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    target: [u8; 16],
    stance: Stance,
    issued_at: u64,
}

impl VouchingCert {
    /// Sign a VouchingCert. The `signer` identity hash is derived from
    /// `signer_sk` automatically (classical-only / ed25519). When PQ keys
    /// are introduced, a PQ-aware sign API will be added; the foot-gun of
    /// passing a wrong signer-id explicitly is eliminated for the v1
    /// classical path.
    pub fn sign(
        signer_sk: &SigningKey,
        owner_id: [u8; 16],
        target: [u8; 16],
        stance: Stance,
        issued_at: u64,
    ) -> Result<Self, OwnerError> {
        let signer = PubKeyBundle::classical_only(signer_sk.verifying_key().to_bytes()).identity_hash();
        let payload_bytes = cbor::to_canonical(&VouchingSigningPayload {
            version: VOUCHING_VERSION,
            owner_id,
            signer,
            target,
            stance,
            issued_at,
        })?;
        let signature = sign_with_tag(signer_sk, tags::VOUCHING, &payload_bytes);
        Ok(VouchingCert {
            version: VOUCHING_VERSION,
            owner_id,
            signer,
            target,
            stance,
            issued_at,
            signature,
        })
    }

    pub fn verify(&self, signer_pubkey: &VerifyingKey) -> Result<(), OwnerError> {
        if self.version != VOUCHING_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        // Bind the signer field to the verifying key the caller provided:
        // this prevents a cert that names device-A as signer from being
        // accepted as if signed by device-B (callers that only check the
        // ed25519 signature would otherwise be fooled).
        let expected_signer = PubKeyBundle::classical_only(signer_pubkey.to_bytes()).identity_hash();
        if self.signer != expected_signer {
            return Err(OwnerError::IdentityHashMismatch);
        }
        let payload_bytes = cbor::to_canonical(&VouchingSigningPayload {
            version: self.version,
            owner_id: self.owner_id,
            signer: self.signer,
            target: self.target,
            stance: self.stance,
            issued_at: self.issued_at,
        })?;
        verify_with_tag(signer_pubkey, tags::VOUCHING, &payload_bytes, &self.signature, "Vouching")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn vouch_signs_and_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(
            &sk,
            [1u8; 16],
            [3u8; 16],
            Stance::Vouch,
            1_700_000_000,
        ).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn challenge_signs_and_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(
            &sk,
            [1u8; 16],
            [3u8; 16],
            Stance::Challenge,
            1_700_000_000,
        ).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn signature_with_different_signer_key_rejected() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        let cert = VouchingCert::sign(&sk_a, [1u8; 16], [3u8; 16], Stance::Vouch, 1).unwrap();
        let result = cert.verify(&sk_b.verifying_key());
        // Either IdentityHashMismatch (signer-binding catches it first) or
        // InvalidSignature is acceptable — both prove the cert is rejected.
        assert!(matches!(
            result,
            Err(OwnerError::IdentityHashMismatch) | Err(OwnerError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn cert_with_lying_signer_field_rejected_by_verify() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        // Sign with A but lie that signer is B
        let lying_signer =
            PubKeyBundle::classical_only(sk_b.verifying_key().to_bytes()).identity_hash();
        let mut cert = VouchingCert::sign(&sk_a, [1u8; 16], [9u8; 16], Stance::Vouch, 1).unwrap();
        cert.signer = lying_signer;
        // Verifying with A's pubkey (the actual signer) should now fail because
        // cert.signer doesn't match A's identity hash.
        let result = cert.verify(&sk_a.verifying_key());
        assert!(matches!(result, Err(OwnerError::IdentityHashMismatch)));
    }
}

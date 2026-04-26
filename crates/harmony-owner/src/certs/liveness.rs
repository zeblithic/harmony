use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const LIVENESS_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LivenessCert {
    pub version: u8,
    #[serde(with = "crate::cbor::arr16")]
    pub owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    pub signer: [u8; 16],
    pub timestamp: u64,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct LivenessSigningPayload {
    version: u8,
    #[serde(with = "crate::cbor::arr16")]
    owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    signer: [u8; 16],
    timestamp: u64,
}

impl LivenessCert {
    /// Sign a LivenessCert. The `signer` identity hash is derived from
    /// `signer_sk` automatically (classical-only / ed25519). When PQ keys
    /// are introduced, a PQ-aware sign API will be added.
    pub fn sign(
        signer_sk: &SigningKey,
        owner_id: [u8; 16],
        timestamp: u64,
    ) -> Result<Self, OwnerError> {
        let signer = PubKeyBundle::classical_only(signer_sk.verifying_key().to_bytes()).identity_hash();
        let payload_bytes = cbor::to_canonical(&LivenessSigningPayload {
            version: LIVENESS_VERSION,
            owner_id,
            signer,
            timestamp,
        })?;
        let signature = sign_with_tag(signer_sk, tags::LIVENESS, &payload_bytes);
        Ok(LivenessCert {
            version: LIVENESS_VERSION,
            owner_id,
            signer,
            timestamp,
            signature,
        })
    }

    pub fn verify(&self, signer_pubkey: &VerifyingKey) -> Result<(), OwnerError> {
        if self.version != LIVENESS_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        // Bind the signer field to the provided verifying key so that a cert
        // claiming to be from device-A can never verify under device-B's key.
        let expected_signer = PubKeyBundle::classical_only(signer_pubkey.to_bytes()).identity_hash();
        if self.signer != expected_signer {
            return Err(OwnerError::IdentityHashMismatch);
        }
        let payload_bytes = cbor::to_canonical(&LivenessSigningPayload {
            version: self.version,
            owner_id: self.owner_id,
            signer: self.signer,
            timestamp: self.timestamp,
        })?;
        verify_with_tag(signer_pubkey, tags::LIVENESS, &payload_bytes, &self.signature, "Liveness")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn liveness_roundtrip() {
        let sk = SigningKey::generate(&mut OsRng);
        let cert = LivenessCert::sign(&sk, [1u8; 16], 1_700_000_000).unwrap();
        cert.verify(&sk.verifying_key()).unwrap();
    }

    #[test]
    fn liveness_rejected_with_wrong_key() {
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);
        let cert = LivenessCert::sign(&sk_a, [1u8; 16], 1).unwrap();
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
        let lying_signer =
            PubKeyBundle::classical_only(sk_b.verifying_key().to_bytes()).identity_hash();
        let mut cert = LivenessCert::sign(&sk_a, [1u8; 16], 1).unwrap();
        cert.signer = lying_signer;
        // Verifying with A's pubkey (the actual signer) should now fail because
        // cert.signer doesn't match A's identity hash.
        let result = cert.verify(&sk_a.verifying_key());
        assert!(matches!(result, Err(OwnerError::IdentityHashMismatch)));
    }
}

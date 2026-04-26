use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const REVOCATION_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationReason {
    Decommissioned,
    Lost,
    Compromised,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RevocationIssuer {
    SelfDevice,
    Master { master_pubkey: PubKeyBundle },
    Quorum {
        #[serde(with = "crate::cbor::arr16_vec")]
        signers: Vec<[u8; 16]>,
        #[serde(with = "crate::cbor::bytes_vec")]
        signatures: Vec<Vec<u8>>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RevocationCert {
    pub version: u8,
    #[serde(with = "crate::cbor::arr16")]
    pub owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    pub target: [u8; 16],
    pub issued_at: u64,
    pub issuer: RevocationIssuer,
    pub reason: RevocationReason,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, Serialize)]
struct RevocationSigningPayload<'a> {
    version: u8,
    #[serde(with = "crate::cbor::arr16")]
    owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    target: [u8; 16],
    issued_at: u64,
    issuer_kind: u8,
    #[serde(with = "serde_bytes")]
    issuer_data: Vec<u8>,
    reason: &'a RevocationReason,
}

impl RevocationCert {
    pub fn sign_self(
        device_sk: &SigningKey,
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
    ) -> Result<Self, OwnerError> {
        let issuer = RevocationIssuer::SelfDevice;
        let payload_bytes = cbor::to_canonical(&signing_payload(
            REVOCATION_VERSION, owner_id, target, issued_at, &issuer, &reason)?)?;
        let signature = sign_with_tag(device_sk, tags::REVOCATION, &payload_bytes);
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer,
            reason,
            signature,
        })
    }

    pub fn sign_master(
        master_sk: &SigningKey,
        master_pubkey: PubKeyBundle,
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
    ) -> Result<Self, OwnerError> {
        let owner_id = master_pubkey.identity_hash();
        let issuer = RevocationIssuer::Master { master_pubkey: master_pubkey.clone() };
        let payload_bytes = cbor::to_canonical(&signing_payload(
            REVOCATION_VERSION, owner_id, target, issued_at, &issuer, &reason)?)?;
        let signature = sign_with_tag(master_sk, tags::REVOCATION, &payload_bytes);
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer,
            reason,
            signature,
        })
    }

    /// Verify against a provided pubkey for the issuer (self-device's pubkey
    /// or master's pubkey for SelfDevice/Master variants respectively). For
    /// Quorum, full verification is delegated to OwnerState.
    pub fn verify(&self, issuer_pubkey: Option<&VerifyingKey>) -> Result<(), OwnerError> {
        if self.version != REVOCATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        match (&self.issuer, issuer_pubkey) {
            (RevocationIssuer::SelfDevice, Some(vk)) => {
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version, self.owner_id, self.target, self.issued_at, &self.issuer, &self.reason)?)?;
                verify_with_tag(vk, tags::REVOCATION, &payload_bytes, &self.signature, "Revocation")
            }
            (RevocationIssuer::Master { master_pubkey }, _) => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Revocation" })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version, self.owner_id, self.target, self.issued_at, &self.issuer, &self.reason)?)?;
                verify_with_tag(&vk, tags::REVOCATION, &payload_bytes, &self.signature, "Revocation")
            }
            (RevocationIssuer::Quorum { .. }, _) => {
                Err(OwnerError::QuorumRevocationNotImplemented)
            }
            (RevocationIssuer::SelfDevice, None) => {
                Err(OwnerError::InvalidSignature { cert_type: "Revocation" })
            }
        }
    }
}

fn signing_payload<'a>(
    version: u8,
    owner_id: [u8; 16],
    target: [u8; 16],
    issued_at: u64,
    issuer: &RevocationIssuer,
    reason: &'a RevocationReason,
) -> Result<RevocationSigningPayload<'a>, OwnerError> {
    let (issuer_kind, issuer_data) = match issuer {
        RevocationIssuer::SelfDevice => (0u8, Vec::new()),
        RevocationIssuer::Master { master_pubkey } => (1u8, cbor::to_canonical(master_pubkey)?),
        RevocationIssuer::Quorum { signers, .. } => (2u8, cbor::to_canonical(signers)?),
    };
    Ok(RevocationSigningPayload { version, owner_id, target, issued_at, issuer_kind, issuer_data, reason })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn self_revocation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let target = [9u8; 16];
        let cert = RevocationCert::sign_self(&sk, [1u8; 16], target, 1, RevocationReason::Decommissioned).unwrap();
        cert.verify(Some(&sk.verifying_key())).unwrap();
    }

    #[test]
    fn master_revocation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let master_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let cert = RevocationCert::sign_master(&sk, master_bundle, [9u8; 16], 1, RevocationReason::Compromised).unwrap();
        cert.verify(None).unwrap();
    }
}

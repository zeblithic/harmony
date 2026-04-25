use crate::cbor;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags, verify_with_tag};
use crate::OwnerError;
use ed25519_dalek::{SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const ENROLLMENT_VERSION: u8 = 1;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EnrollmentCert {
    pub version: u8,
    pub owner_id: [u8; 16],
    pub device_id: [u8; 16],
    pub device_pubkeys: PubKeyBundle,
    pub issued_at: u64,
    pub expires_at: Option<u64>,
    pub issuer: EnrollmentIssuer,
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EnrollmentIssuer {
    /// Master-signed. `master_pubkey` is embedded so verifier is self-contained.
    Master { master_pubkey: PubKeyBundle },

    /// K-quorum: each signer has its own EnrollmentCert under the same owner.
    /// Verifier walks back to those certs to fetch signers' pubkeys.
    Quorum {
        signers: Vec<[u8; 16]>,
        signatures: Vec<Vec<u8>>,
    },
}

#[derive(Debug, Clone, Serialize)]
struct EnrollmentSigningPayload<'a> {
    version: u8,
    owner_id: [u8; 16],
    device_id: [u8; 16],
    device_pubkeys: &'a PubKeyBundle,
    issued_at: u64,
    expires_at: Option<u64>,
    issuer_kind: u8, // 0 = Master, 1 = Quorum
    issuer_data: Vec<u8>, // CBOR-encoded inner data of the EnrollmentIssuer
}

impl EnrollmentCert {
    /// Issue a Master-signed Enrollment Cert.
    pub fn sign_master(
        owner_sk: &SigningKey,
        master_pubkey: PubKeyBundle,
        device_id: [u8; 16],
        device_pubkeys: PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
    ) -> Result<Self, OwnerError> {
        let owner_id = master_pubkey.identity_hash();
        let issuer = EnrollmentIssuer::Master { master_pubkey: master_pubkey.clone() };
        let payload_bytes = cbor::to_canonical(&signing_payload(
            ENROLLMENT_VERSION,
            owner_id,
            device_id,
            &device_pubkeys,
            issued_at,
            expires_at,
            &issuer,
        )?)?;
        let signature = sign_with_tag(owner_sk, tags::ENROLLMENT, &payload_bytes);
        Ok(EnrollmentCert {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer,
            signature,
        })
    }

    pub fn verify(&self) -> Result<(), OwnerError> {
        if self.version != ENROLLMENT_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        match &self.issuer {
            EnrollmentIssuer::Master { master_pubkey } => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Enrollment" })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version,
                    self.owner_id,
                    self.device_id,
                    &self.device_pubkeys,
                    self.issued_at,
                    self.expires_at,
                    &self.issuer,
                )?)?;
                verify_with_tag(&vk, tags::ENROLLMENT, &payload_bytes, &self.signature, "Enrollment")?;
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
            EnrollmentIssuer::Quorum { signers, signatures } => {
                // Quorum verification is delegated to OwnerState (which has access to
                // the full set of signer Enrollment Certs). This standalone verify()
                // only checks the signature count matches signers count.
                if signers.len() != signatures.len() {
                    return Err(OwnerError::InvalidSignature { cert_type: "Enrollment" });
                }
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
        }
    }
}

fn signing_payload<'a>(
    version: u8,
    owner_id: [u8; 16],
    device_id: [u8; 16],
    device_pubkeys: &'a PubKeyBundle,
    issued_at: u64,
    expires_at: Option<u64>,
    issuer: &'a EnrollmentIssuer,
) -> Result<EnrollmentSigningPayload<'a>, OwnerError> {
    let (issuer_kind, issuer_data) = match issuer {
        EnrollmentIssuer::Master { master_pubkey } => {
            (0u8, cbor::to_canonical(master_pubkey)?)
        }
        EnrollmentIssuer::Quorum { signers, .. } => {
            // Signatures NOT included in signing payload (chicken-and-egg);
            // each signer's signature covers the rest of the payload + the
            // signers list.
            (1u8, cbor::to_canonical(signers)?)
        }
    };
    Ok(EnrollmentSigningPayload {
        version,
        owner_id,
        device_id,
        device_pubkeys,
        issued_at,
        expires_at,
        issuer_kind,
        issuer_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pubkey_bundle::ClassicalKeys;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fresh_pubkey_bundle(ed_seed: u8, x_seed: u8) -> (SigningKey, PubKeyBundle) {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: sk.verifying_key().to_bytes(),
                x25519_pub: [x_seed; 32], // mock for test; real X25519 from harmony-identity
            },
            post_quantum: None,
        };
        let _ = ed_seed;
        (sk, bundle)
    }

    #[test]
    fn master_signed_enrollment_verifies() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        cert.verify().unwrap();
    }

    #[test]
    fn tampered_enrollment_signature_rejected() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let mut cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        // Tamper with timestamp
        cert.issued_at = 1_800_000_000;
        let result = cert.verify();
        assert!(matches!(result, Err(OwnerError::InvalidSignature { .. })));
    }

    #[test]
    fn enrollment_cbor_roundtrip() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_device_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();

        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_700_000_000,
            None,
        ).unwrap();

        let bytes = cbor::to_canonical(&cert).unwrap();
        let decoded: EnrollmentCert = cbor::from_bytes(&bytes).unwrap();
        assert_eq!(cert, decoded);
        decoded.verify().unwrap();
    }
}

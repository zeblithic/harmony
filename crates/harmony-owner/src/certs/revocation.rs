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
    Master {
        master_pubkey: PubKeyBundle,
    },
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
            REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            &issuer,
            &reason,
        )?)?;
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
        let issuer = RevocationIssuer::Master {
            master_pubkey: master_pubkey.clone(),
        };
        let payload_bytes = cbor::to_canonical(&signing_payload(
            REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            &issuer,
            &reason,
        )?)?;
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
                    self.version,
                    self.owner_id,
                    self.target,
                    self.issued_at,
                    &self.issuer,
                    &self.reason,
                )?)?;
                verify_with_tag(
                    vk,
                    tags::REVOCATION,
                    &payload_bytes,
                    &self.signature,
                    "Revocation",
                )
            }
            (RevocationIssuer::Master { master_pubkey }, _) => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature {
                        cert_type: "Revocation",
                    })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version,
                    self.owner_id,
                    self.target,
                    self.issued_at,
                    &self.issuer,
                    &self.reason,
                )?)?;
                verify_with_tag(
                    &vk,
                    tags::REVOCATION,
                    &payload_bytes,
                    &self.signature,
                    "Revocation",
                )
            }
            (RevocationIssuer::Quorum { .. }, _) => Err(OwnerError::QuorumRequiresSignerCerts),
            (RevocationIssuer::SelfDevice, None) => Err(OwnerError::InvalidSignature {
                cert_type: "Revocation",
            }),
        }
    }

    /// Canonical detached-signature payload for a Quorum-issued revocation
    /// (ZEB-677 §2). The signer set is part of the payload — fix it before
    /// signing. issuer_kind = 2 (Quorum) plus tags::REVOCATION give domain
    /// separation from enrollment quorum parts.
    pub fn quorum_signing_payload_bytes(
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: &RevocationReason,
        signers: &[[u8; 16]],
    ) -> Result<Vec<u8>, OwnerError> {
        let issuer_data = cbor::to_canonical(&signers.to_vec())?;
        cbor::to_canonical(&RevocationSigningPayload {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer_kind: 2, // Quorum
            issuer_data,
            reason,
        })
    }

    /// Sign one quorum part with the revocation domain tag.
    pub fn sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8> {
        sign_with_tag(sk, tags::REVOCATION, payload_bytes)
    }

    /// Assemble a Quorum-issued revocation from independently collected
    /// `(signer_device_id, detached_signature)` parts. Structural checks
    /// only (≥2, distinct); full verification needs the signer certs
    /// (`verify_quorum_with_signers`) or `OwnerState::add_revocation`.
    pub fn assemble_quorum(
        owner_id: [u8; 16],
        target: [u8; 16],
        issued_at: u64,
        reason: RevocationReason,
        parts: Vec<([u8; 16], Vec<u8>)>,
    ) -> Result<Self, OwnerError> {
        if parts.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: parts.len(),
            });
        }
        let unique: std::collections::HashSet<[u8; 16]> = parts.iter().map(|(id, _)| *id).collect();
        if unique.len() != parts.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Duplicate-Signer",
            });
        }
        let (signers, signatures): (Vec<[u8; 16]>, Vec<Vec<u8>>) = parts.into_iter().unzip();
        Ok(RevocationCert {
            version: REVOCATION_VERSION,
            owner_id,
            target,
            issued_at,
            issuer: RevocationIssuer::Quorum {
                signers,
                signatures,
            },
            reason,
            signature: Vec::new(),
        })
    }

    /// Peer-side full verification of a Quorum-issued revocation against
    /// presented signer enrollment certs. Same depth-1 policy as
    /// `EnrollmentCert::verify_quorum_with_signers`: each signer cert must
    /// be present, same-owner, Master-issued, valid at `now_secs`, and not
    /// issued after this revocation. Liveness/active-window policy is
    /// fleet-internal (`OwnerState::add_revocation`) and not checked here.
    pub fn verify_quorum_with_signers(
        &self,
        signer_certs: &[crate::certs::EnrollmentCert],
        now_secs: u64,
    ) -> Result<(), OwnerError> {
        if self.version != REVOCATION_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        let (signers, signatures) = match &self.issuer {
            RevocationIssuer::Quorum {
                signers,
                signatures,
            } => (signers, signatures),
            _ => {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Expected",
                })
            }
        };
        if signers.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: signers.len(),
            });
        }
        if signers.len() != signatures.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Length-Mismatch",
            });
        }
        let unique: std::collections::HashSet<[u8; 16]> = signers.iter().copied().collect();
        if unique.len() != signers.len() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Duplicate-Signer",
            });
        }
        // Quorum certs carry their signatures in the issuer; the top-level
        // field must be empty or it is unauthenticated malleable bytes — it
        // participates in struct equality and RevocationSet LWW tie-breaks
        // without being covered by any signature (ZEB-677).
        if !self.signature.is_empty() {
            return Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Nonempty-Signature",
            });
        }
        let payload_bytes = Self::quorum_signing_payload_bytes(
            self.owner_id,
            self.target,
            self.issued_at,
            &self.reason,
            signers,
        )?;
        for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
            let signer_cert = signer_certs
                .iter()
                .find(|c| c.device_id == *signer_id)
                .ok_or(OwnerError::NotEnrolled {
                    owner: self.owner_id,
                    device: *signer_id,
                })?;
            if signer_cert.owner_id != self.owner_id {
                return Err(OwnerError::WrongOwner {
                    expected: self.owner_id,
                    got: signer_cert.owner_id,
                });
            }
            if !matches!(
                signer_cert.issuer,
                crate::certs::EnrollmentIssuer::Master { .. }
            ) {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Signer-Not-Master",
                });
            }
            signer_cert.verify(now_secs)?;
            if signer_cert.issued_at > self.issued_at {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Backdated-Signer",
                });
            }
            let vk = VerifyingKey::from_bytes(&signer_cert.device_pubkeys.classical.ed25519_verify)
                .map_err(|_| OwnerError::InvalidSignature {
                    cert_type: "Revocation-Quorum-Member",
                })?;
            verify_with_tag(
                &vk,
                tags::REVOCATION,
                &payload_bytes,
                sig,
                "Revocation-Quorum-Member",
            )?;
        }
        Ok(())
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
    Ok(RevocationSigningPayload {
        version,
        owner_id,
        target,
        issued_at,
        issuer_kind,
        issuer_data,
        reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::EnrollmentCert;
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    fn signer_world() -> (
        [u8; 16],
        Vec<EnrollmentCert>, // Master-issued certs for A, B
        SigningKey,          // A sk
        SigningKey,          // B sk
    ) {
        let master_sk = SigningKey::generate(&mut OsRng);
        let master_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: master_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let owner_id = master_bundle.identity_hash();
        let mk = |seed: u8| -> (SigningKey, PubKeyBundle) {
            let sk = SigningKey::generate(&mut OsRng);
            let b = PubKeyBundle {
                classical: ClassicalKeys {
                    ed25519_verify: sk.verifying_key().to_bytes(),
                    x25519_pub: [seed; 32],
                },
                post_quantum: None,
            };
            (sk, b)
        };
        let (a_sk, a_bundle) = mk(1);
        let (b_sk, b_bundle) = mk(2);
        let a_cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle.clone(),
            a_bundle.identity_hash(),
            a_bundle,
            100,
            None,
        )
        .unwrap();
        let b_cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            b_bundle.identity_hash(),
            b_bundle,
            100,
            None,
        )
        .unwrap();
        (owner_id, vec![a_cert, b_cert], a_sk, b_sk)
    }

    fn assemble_quorum_revocation(
        owner_id: [u8; 16],
        signer_certs: &[EnrollmentCert],
        a_sk: &SigningKey,
        b_sk: &SigningKey,
        target: [u8; 16],
    ) -> RevocationCert {
        let signers = [signer_certs[0].device_id, signer_certs[1].device_id];
        let payload = RevocationCert::quorum_signing_payload_bytes(
            owner_id,
            target,
            1_000,
            &RevocationReason::Lost,
            &signers,
        )
        .unwrap();
        RevocationCert::assemble_quorum(
            owner_id,
            target,
            1_000,
            RevocationReason::Lost,
            vec![
                (signers[0], RevocationCert::sign_quorum_part(a_sk, &payload)),
                (signers[1], RevocationCert::sign_quorum_part(b_sk, &payload)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn quorum_revocation_assembles_and_chain_verifies() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        cert.verify_quorum_with_signers(&signer_certs, 2_000)
            .unwrap();
        // Standalone verify still fails closed.
        assert!(matches!(
            cert.verify(None),
            Err(OwnerError::QuorumRequiresSignerCerts)
        ));
    }

    #[test]
    fn quorum_revocation_rejects_tamper_and_reason_binding() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let mut cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        cert.reason = RevocationReason::Compromised; // reason is in the payload
        assert!(cert
            .verify_quorum_with_signers(&signer_certs, 2_000)
            .is_err());
    }

    #[test]
    fn quorum_revocation_rejects_cross_domain_signature() {
        // A signature made with the ENROLLMENT domain tag must not verify as
        // a REVOCATION quorum part (domain separation via signing tags).
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let signers = [signer_certs[0].device_id, signer_certs[1].device_id];
        let target = [9u8; 16];
        let rev_payload = RevocationCert::quorum_signing_payload_bytes(
            owner_id,
            target,
            1_000,
            &RevocationReason::Lost,
            &signers,
        )
        .unwrap();
        // Enrollment-domain signatures over the same bytes:
        let wrong = vec![
            (
                signers[0],
                EnrollmentCert::sign_quorum_part(&a_sk, &rev_payload),
            ),
            (
                signers[1],
                EnrollmentCert::sign_quorum_part(&b_sk, &rev_payload),
            ),
        ];
        let cert =
            RevocationCert::assemble_quorum(owner_id, target, 1_000, RevocationReason::Lost, wrong)
                .unwrap();
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn quorum_revocation_rejects_nonempty_top_level_signature() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let mut cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        // Unused for Quorum revocations — nonempty bytes are unauthenticated
        // malleability (struct equality + RevocationSet LWW tie-breaks).
        cert.signature = vec![0xAA];
        let err = cert.verify_quorum_with_signers(&signer_certs, 2_000);
        assert!(
            matches!(err, Err(OwnerError::InvalidSignature { cert_type }) if cert_type == "Revocation-Quorum-Nonempty-Signature")
        );
    }

    #[test]
    fn quorum_revocation_rejects_missing_signer_cert() {
        let (owner_id, signer_certs, a_sk, b_sk) = signer_world();
        let cert = assemble_quorum_revocation(owner_id, &signer_certs, &a_sk, &b_sk, [9u8; 16]);
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs[..1], 2_000),
            Err(OwnerError::NotEnrolled { .. })
        ));
    }

    #[test]
    fn quorum_revocation_assemble_rejects_single_and_duplicate_parts() {
        let (owner_id, signer_certs, a_sk, _b_sk) = signer_world();
        let a_id = signer_certs[0].device_id;
        let target = [9u8; 16];
        let payload = RevocationCert::quorum_signing_payload_bytes(
            owner_id,
            target,
            1_000,
            &RevocationReason::Lost,
            &[a_id],
        )
        .unwrap();
        let sig = RevocationCert::sign_quorum_part(&a_sk, &payload);
        assert!(matches!(
            RevocationCert::assemble_quorum(
                owner_id,
                target,
                1_000,
                RevocationReason::Lost,
                vec![(a_id, sig.clone())],
            ),
            Err(OwnerError::InsufficientQuorum { min: 2, got: 1 })
        ));
        assert!(matches!(
            RevocationCert::assemble_quorum(
                owner_id,
                target,
                1_000,
                RevocationReason::Lost,
                vec![(a_id, sig.clone()), (a_id, sig)],
            ),
            Err(OwnerError::InvalidSignature {
                cert_type: "Revocation-Quorum-Duplicate-Signer"
            })
        ));
    }

    #[test]
    fn self_revocation_verifies() {
        let sk = SigningKey::generate(&mut OsRng);
        let target = [9u8; 16];
        let cert =
            RevocationCert::sign_self(&sk, [1u8; 16], target, 1, RevocationReason::Decommissioned)
                .unwrap();
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
        let cert = RevocationCert::sign_master(
            &sk,
            master_bundle,
            [9u8; 16],
            1,
            RevocationReason::Compromised,
        )
        .unwrap();
        cert.verify(None).unwrap();
    }
}

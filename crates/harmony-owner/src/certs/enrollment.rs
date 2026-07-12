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
    #[serde(with = "crate::cbor::arr16")]
    pub owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
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
        #[serde(with = "crate::cbor::arr16_vec")]
        signers: Vec<[u8; 16]>,
        #[serde(with = "crate::cbor::bytes_vec")]
        signatures: Vec<Vec<u8>>,
    },
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct EnrollmentSigningPayload<'a> {
    pub(crate) version: u8,
    #[serde(with = "crate::cbor::arr16")]
    pub(crate) owner_id: [u8; 16],
    #[serde(with = "crate::cbor::arr16")]
    pub(crate) device_id: [u8; 16],
    pub(crate) device_pubkeys: &'a PubKeyBundle,
    pub(crate) issued_at: u64,
    pub(crate) expires_at: Option<u64>,
    pub(crate) issuer_kind: u8, // 0 = Master, 1 = Quorum
    #[serde(with = "serde_bytes")]
    pub(crate) issuer_data: Vec<u8>, // CBOR-encoded inner data of the EnrollmentIssuer
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
        let issuer = EnrollmentIssuer::Master {
            master_pubkey: master_pubkey.clone(),
        };
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

    /// `now_secs` is the current time in **Unix SECONDS** — the same unit as the
    /// cert's `issued_at`/`expires_at` and `OwnerState`'s `now`/`active_window_secs`
    /// (see `add_enrollment`/`active_devices`). Callers in millisecond domains (HLC
    /// `wall_ms`, `SystemTime::as_millis`) MUST convert (`/ 1000`) before calling.
    pub fn verify(&self, now_secs: u64) -> Result<(), OwnerError> {
        if self.version != ENROLLMENT_VERSION {
            return Err(OwnerError::UnknownVersion(self.version));
        }
        if let Some(exp) = self.expires_at {
            if now_secs > exp {
                return Err(OwnerError::EnrollmentCertExpired {
                    expires_at: exp,
                    now_secs,
                });
            }
        }
        match &self.issuer {
            EnrollmentIssuer::Master { master_pubkey } => {
                if master_pubkey.identity_hash() != self.owner_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                let vk = VerifyingKey::from_bytes(&master_pubkey.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature {
                        cert_type: "Enrollment",
                    })?;
                let payload_bytes = cbor::to_canonical(&signing_payload(
                    self.version,
                    self.owner_id,
                    self.device_id,
                    &self.device_pubkeys,
                    self.issued_at,
                    self.expires_at,
                    &self.issuer,
                )?)?;
                verify_with_tag(
                    &vk,
                    tags::ENROLLMENT,
                    &payload_bytes,
                    &self.signature,
                    "Enrollment",
                )?;
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
            EnrollmentIssuer::Quorum {
                signers,
                signatures,
            } => {
                // Quorum verification is delegated to OwnerState (which has access to
                // the full set of signer Enrollment Certs). This standalone verify()
                // performs the structural checks that don't require state lookup:
                // minimum quorum size, signers/signatures parity, distinct signers,
                // and device-id consistency.
                if signers.len() < 2 {
                    return Err(OwnerError::InsufficientQuorum {
                        min: 2,
                        got: signers.len(),
                    });
                }
                if signers.len() != signatures.len() {
                    return Err(OwnerError::InvalidSignature {
                        cert_type: "Enrollment",
                    });
                }
                let unique: std::collections::HashSet<[u8; 16]> = signers.iter().copied().collect();
                if unique.len() != signers.len() {
                    return Err(OwnerError::InvalidSignature {
                        cert_type: "Enrollment",
                    });
                }
                if self.device_pubkeys.identity_hash() != self.device_id {
                    return Err(OwnerError::IdentityHashMismatch);
                }
                Ok(())
            }
        }
    }

    /// Canonical detached-signature payload for a Quorum-issued enrollment
    /// cert. The signer set is part of the payload, so it must be fixed
    /// before any part is signed. Public so the ceremony can collect
    /// signatures across devices (ZEB-677); `OwnerState` and
    /// `enroll_via_quorum` use the same function, so signing and
    /// verification cannot drift.
    pub fn quorum_signing_payload_bytes(
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: &PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        signers: &[[u8; 16]],
    ) -> Result<Vec<u8>, OwnerError> {
        let issuer_data = cbor::to_canonical(&signers.to_vec())?;
        cbor::to_canonical(&EnrollmentSigningPayload {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer_kind: 1, // Quorum
            issuer_data,
        })
    }

    /// Sign one quorum part over `quorum_signing_payload_bytes` output with
    /// the correct domain tag. Thin wrapper so callers never touch raw tags.
    pub fn sign_quorum_part(sk: &SigningKey, payload_bytes: &[u8]) -> Vec<u8> {
        sign_with_tag(sk, tags::ENROLLMENT, payload_bytes)
    }

    /// Peer-side full verification of a Quorum-issued cert using presented
    /// signer enrollment certs (depth-1 chain carriage, ZEB-677 §2). Checks,
    /// in order: structural validity + own expiry (`verify`); Quorum issuer
    /// required; then per signer — a matching presented cert exists, is
    /// bound to the same owner, is Master-issued (depth-1), passes its own
    /// full `verify(now_secs)` (expiry checked before signatures so the
    /// error is deterministic), was not issued after this cert (backdating),
    /// and its enrolled ed25519 key validates the quorum part signature.
    /// Extra presented certs are ignored. Liveness/active-window policy is
    /// NOT checked here — that is fleet-internal (`OwnerState`) policy a
    /// peer has no data for.
    pub fn verify_quorum_with_signers(
        &self,
        signer_certs: &[EnrollmentCert],
        now_secs: u64,
    ) -> Result<(), OwnerError> {
        self.verify(now_secs)?;
        let (signers, signatures) = match &self.issuer {
            EnrollmentIssuer::Quorum {
                signers,
                signatures,
            } => (signers, signatures),
            EnrollmentIssuer::Master { .. } => {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Expected",
                })
            }
        };
        let payload_bytes = Self::quorum_signing_payload_bytes(
            self.owner_id,
            self.device_id,
            &self.device_pubkeys,
            self.issued_at,
            self.expires_at,
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
            if !matches!(signer_cert.issuer, EnrollmentIssuer::Master { .. }) {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Signer-Not-Master",
                });
            }
            signer_cert.verify(now_secs)?;
            if signer_cert.issued_at > self.issued_at {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Backdated-Signer",
                });
            }
            let vk = VerifyingKey::from_bytes(&signer_cert.device_pubkeys.classical.ed25519_verify)
                .map_err(|_| OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Quorum-Member",
                })?;
            verify_with_tag(
                &vk,
                tags::ENROLLMENT,
                &payload_bytes,
                sig,
                "Enrollment-Quorum-Member",
            )?;
        }
        Ok(())
    }

    /// Assemble a Quorum-issued cert from independently collected
    /// `(signer_device_id, detached_signature)` parts. Performs the
    /// structural quorum checks; full signature verification requires the
    /// signer certs (`verify_quorum_with_signers`) or `OwnerState`.
    pub fn assemble_quorum(
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        parts: Vec<([u8; 16], Vec<u8>)>,
    ) -> Result<Self, OwnerError> {
        if parts.len() < 2 {
            return Err(OwnerError::InsufficientQuorum {
                min: 2,
                got: parts.len(),
            });
        }
        let (signers, signatures): (Vec<[u8; 16]>, Vec<Vec<u8>>) = parts.into_iter().unzip();
        let cert = EnrollmentCert {
            version: ENROLLMENT_VERSION,
            owner_id,
            device_id,
            device_pubkeys,
            issued_at,
            expires_at,
            issuer: EnrollmentIssuer::Quorum {
                signers,
                signatures,
            },
            signature: Vec::new(),
        };
        // Structural checks (distinctness, parity, device-id binding) live in
        // verify(); expiry is irrelevant at issuance time, so verify at
        // issued_at.
        cert.verify(issued_at)?;
        Ok(cert)
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
        EnrollmentIssuer::Master { master_pubkey } => (0u8, cbor::to_canonical(master_pubkey)?),
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
        )
        .unwrap();

        cert.verify(0).unwrap();
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
        )
        .unwrap();

        // Tamper with timestamp
        cert.issued_at = 1_800_000_000;
        let result = cert.verify(0);
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
        )
        .unwrap();

        let bytes = cbor::to_canonical(&cert).unwrap();
        let decoded: EnrollmentCert = cbor::from_bytes(&bytes).unwrap();
        assert_eq!(cert, decoded);
        decoded.verify(0).unwrap();
    }

    #[test]
    fn verify_rejects_past_expiry() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let (_d_sk, device_bundle) = fresh_pubkey_bundle(3, 4);
        let device_id = device_bundle.identity_hash();
        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_000,
            Some(2_000),
        )
        .unwrap();
        assert!(matches!(
            cert.verify(2_001),
            Err(OwnerError::EnrollmentCertExpired {
                expires_at: 2_000,
                now_secs: 2_001
            })
        ));
        assert!(cert.verify(2_000).is_ok()); // exactly-at-expiry still valid (not strictly greater)
        assert!(cert.verify(1_500).is_ok());
    }

    #[test]
    fn assembled_quorum_cert_matches_shape_and_passes_structural_verify() {
        let (a_sk, a_bundle) = fresh_pubkey_bundle(1, 2);
        let (b_sk, b_bundle) = fresh_pubkey_bundle(3, 4);
        let (_new_sk, new_bundle) = fresh_pubkey_bundle(5, 6);
        let owner_id = [7u8; 16];
        let new_id = new_bundle.identity_hash();
        let signers = [a_bundle.identity_hash(), b_bundle.identity_hash()];

        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id,
            new_id,
            &new_bundle,
            1_000,
            None,
            &signers,
        )
        .unwrap();
        let parts = vec![
            (
                signers[0],
                EnrollmentCert::sign_quorum_part(&a_sk, &payload),
            ),
            (
                signers[1],
                EnrollmentCert::sign_quorum_part(&b_sk, &payload),
            ),
        ];
        let cert =
            EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle, 1_000, None, parts)
                .unwrap();
        assert!(
            matches!(&cert.issuer, EnrollmentIssuer::Quorum { signers: s, signatures } if s.len() == 2 && signatures.len() == 2)
        );
        assert!(cert.signature.is_empty()); // quorum sigs live in the issuer
        cert.verify(2_000).unwrap(); // structural verify passes
    }

    /// Master-enrolls devices A and B under one owner, quorum-enrolls a new
    /// device C signed by A+B, and returns everything a chain verifier needs.
    fn quorum_world() -> (
        [u8; 16],            // owner_id
        EnrollmentCert,      // quorum cert for C
        Vec<EnrollmentCert>, // signer certs [A, B] (Master-issued)
        SigningKey,          // A's device sk (for tamper/depth tests)
    ) {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(1, 2);
        let owner_id = master_bundle.identity_hash();
        let (a_sk, a_bundle) = fresh_pubkey_bundle(3, 4);
        let (b_sk, b_bundle) = fresh_pubkey_bundle(5, 6);
        let (_c_sk, c_bundle) = fresh_pubkey_bundle(7, 8);
        let a_cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle.clone(),
            a_bundle.identity_hash(),
            a_bundle.clone(),
            100,
            None,
        )
        .unwrap();
        let b_cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            b_bundle.identity_hash(),
            b_bundle.clone(),
            100,
            None,
        )
        .unwrap();
        let signers = [a_bundle.identity_hash(), b_bundle.identity_hash()];
        let c_id = c_bundle.identity_hash();
        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id, c_id, &c_bundle, 1_000, None, &signers,
        )
        .unwrap();
        let parts = vec![
            (
                signers[0],
                EnrollmentCert::sign_quorum_part(&a_sk, &payload),
            ),
            (
                signers[1],
                EnrollmentCert::sign_quorum_part(&b_sk, &payload),
            ),
        ];
        let c_cert =
            EnrollmentCert::assemble_quorum(owner_id, c_id, c_bundle, 1_000, None, parts).unwrap();
        (owner_id, c_cert, vec![a_cert, b_cert], a_sk)
    }

    #[test]
    fn quorum_chain_verifies_with_signer_certs() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        cert.verify_quorum_with_signers(&signer_certs, 2_000)
            .unwrap();
    }

    #[test]
    fn quorum_chain_rejects_missing_signer_cert() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        let only_a = &signer_certs[..1];
        assert!(matches!(
            cert.verify_quorum_with_signers(only_a, 2_000),
            Err(OwnerError::NotEnrolled { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_tampered_signature() {
        let (_owner, mut cert, signer_certs, _a_sk) = quorum_world();
        if let EnrollmentIssuer::Quorum { signatures, .. } = &mut cert.issuer {
            signatures[0][0] ^= 0xFF;
        }
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::InvalidSignature { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_wrong_owner_signer_cert() {
        let (_owner, cert, mut signer_certs, _a_sk) = quorum_world();
        // Re-issue A's cert under a DIFFERENT master: the signer cert is
        // internally valid but belongs to a foreign owner.
        let (other_master_sk, other_master_bundle) = fresh_pubkey_bundle(9, 10);
        let a_pub = signer_certs[0].device_pubkeys.clone();
        let a_id = signer_certs[0].device_id;
        signer_certs[0] = EnrollmentCert::sign_master(
            &other_master_sk,
            other_master_bundle,
            a_id,
            a_pub,
            100,
            None,
        )
        .unwrap();
        assert!(matches!(
            cert.verify_quorum_with_signers(&signer_certs, 2_000),
            Err(OwnerError::WrongOwner { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_quorum_issued_signer_cert_depth1() {
        let (owner_id, cert, mut signer_certs, a_sk) = quorum_world();
        // Replace A's Master cert with a structurally-valid Quorum-issued one
        // for the same device: depth-1 violation. (Signatures are junk-by-
        // construction — both parts signed by A — but the depth check must
        // fire before any signature validation.)
        let a = signer_certs[0].clone();
        let b_id = signer_certs[1].device_id;
        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id,
            a.device_id,
            &a.device_pubkeys,
            100,
            None,
            &[a.device_id, b_id],
        )
        .unwrap();
        let parts = vec![
            (
                a.device_id,
                EnrollmentCert::sign_quorum_part(&a_sk, &payload),
            ),
            (b_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload)),
        ];
        signer_certs[0] = EnrollmentCert::assemble_quorum(
            owner_id,
            a.device_id,
            a.device_pubkeys.clone(),
            100,
            None,
            parts,
        )
        .unwrap();
        let err = cert.verify_quorum_with_signers(&signer_certs, 2_000);
        assert!(
            matches!(err, Err(OwnerError::InvalidSignature { cert_type }) if cert_type == "Enrollment-Quorum-Signer-Not-Master")
        );
    }

    #[test]
    fn quorum_chain_rejects_expired_signer_cert() {
        let (_owner, cert, signer_certs, _a_sk) = quorum_world();
        // Mutating expires_at also breaks the signer cert's signature, but
        // the expiry check inside signer_cert.verify() fires FIRST, so the
        // error is deterministically EnrollmentCertExpired.
        let mut certs = signer_certs.clone();
        certs[0].expires_at = Some(150);
        assert!(matches!(
            cert.verify_quorum_with_signers(&certs, 2_000),
            Err(OwnerError::EnrollmentCertExpired { .. })
        ));
    }

    #[test]
    fn quorum_chain_rejects_master_cert_input() {
        let (_owner, _cert, signer_certs, _a_sk) = quorum_world();
        // A Master-issued cert may not be passed to the quorum verifier.
        let err = signer_certs[0].verify_quorum_with_signers(&signer_certs, 2_000);
        assert!(
            matches!(err, Err(OwnerError::InvalidSignature { cert_type }) if cert_type == "Enrollment-Quorum-Expected")
        );
    }

    #[test]
    fn quorum_chain_rejects_backdated_signer_cert() {
        let (_owner, cert, mut signer_certs, _a_sk) = quorum_world();
        // Signer enrolled AFTER the quorum cert was issued (cert at 1_000).
        // The mutation breaks A's cert signature too, but the backdating
        // guard runs only after signer_cert.verify() — so assert the
        // signature error surfaces (the chain is rejected either way; the
        // dedicated ordering-independent backdating coverage lives in the
        // OwnerState tests where a genuinely later-issued cert is signable).
        signer_certs[0].issued_at = 5_000;
        assert!(cert
            .verify_quorum_with_signers(&signer_certs, 6_000)
            .is_err());
    }

    #[test]
    fn assemble_quorum_rejects_single_part_and_duplicate_signers() {
        let (a_sk, a_bundle) = fresh_pubkey_bundle(1, 2);
        let (_new_sk, new_bundle) = fresh_pubkey_bundle(5, 6);
        let owner_id = [7u8; 16];
        let new_id = new_bundle.identity_hash();
        let a_id = a_bundle.identity_hash();

        let payload = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id,
            new_id,
            &new_bundle,
            1_000,
            None,
            &[a_id],
        )
        .unwrap();
        let one = vec![(a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload))];
        assert!(matches!(
            EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle.clone(), 1_000, None, one),
            Err(OwnerError::InsufficientQuorum { min: 2, got: 1 })
        ));

        let payload2 = EnrollmentCert::quorum_signing_payload_bytes(
            owner_id,
            new_id,
            &new_bundle,
            1_000,
            None,
            &[a_id, a_id],
        )
        .unwrap();
        let dup = vec![
            (a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload2)),
            (a_id, EnrollmentCert::sign_quorum_part(&a_sk, &payload2)),
        ];
        assert!(
            EnrollmentCert::assemble_quorum(owner_id, new_id, new_bundle, 1_000, None, dup)
                .is_err()
        );
    }

    #[test]
    fn verify_accepts_none_expiry_at_any_clock() {
        let (master_sk, master_bundle) = fresh_pubkey_bundle(5, 6);
        let (_d_sk, device_bundle) = fresh_pubkey_bundle(7, 8);
        let device_id = device_bundle.identity_hash();
        let cert = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_000,
            None,
        )
        .unwrap();
        assert!(cert.verify(0).is_ok());
        assert!(cert.verify(u64::MAX).is_ok());
    }
}

use crate::certs::{EnrollmentCert, EnrollmentIssuer, LivenessCert};
use crate::crdt::{RevocationSet, VouchingSet};
use crate::OwnerError;
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;

/// Aggregate state for one owner identity: enrollment certs per device,
/// vouching CRDT, revocation set, latest liveness per device.
///
/// Reclamation continuity is evaluated functionally via
/// `lifecycle::reclamation::evaluate_reclamation` against any candidate
/// `ReclamationCert`; it is not held as state here (no current consumer).
#[derive(Debug, Clone, Default)]
pub struct OwnerState {
    pub owner_id: [u8; 16],
    pub enrollments: HashMap<[u8; 16], EnrollmentCert>,
    pub vouching: VouchingSet,
    pub revocations: RevocationSet,
    pub liveness: HashMap<[u8; 16], LivenessCert>,
}

impl OwnerState {
    pub fn new(owner_id: [u8; 16]) -> Self {
        Self { owner_id, ..Default::default() }
    }

    pub fn add_enrollment(
        &mut self,
        cert: EnrollmentCert,
        now: u64,
        active_window_secs: u64,
    ) -> Result<(), OwnerError> {
        if cert.owner_id != self.owner_id {
            return Err(OwnerError::WrongOwner { expected: self.owner_id, got: cert.owner_id });
        }
        cert.verify()?;
        // For Quorum certs, also walk back to verify each signer's enrollment
        // is present and was issued before this cert's `issued_at`.
        if let EnrollmentIssuer::Quorum { signers, signatures } = &cert.issuer {
            if signers.len() < 2 {
                return Err(OwnerError::InsufficientQuorum { min: 2, got: signers.len() });
            }
            if signers.len() != signatures.len() {
                return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Length-Mismatch" });
            }
            let unique: std::collections::HashSet<[u8; 16]> = signers.iter().copied().collect();
            if unique.len() != signers.len() {
                return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Duplicate-Signer" });
            }
            for signer_id in signers {
                if self.is_revoked(*signer_id) {
                    return Err(OwnerError::Revoked { device: *signer_id });
                }
            }
            // Active-signer check: every quorum signer must have produced a
            // Liveness cert within the active window. Master enrollments
            // bypass this (they are by definition the cold authority).
            let active = self.active_devices(now, active_window_secs);
            let active_set: std::collections::HashSet<[u8; 16]> = active.into_iter().collect();
            for signer_id in signers {
                if !active_set.contains(signer_id) {
                    return Err(OwnerError::InvalidSignature {
                        cert_type: "Enrollment-Quorum-Inactive-Signer",
                    });
                }
            }
            // Verify quorum signatures: each signer in `signers` must have an
            // existing enrollment, and each signature in `signatures` must be
            // valid for that signer's pubkey. The signing payload is invariant
            // across iterations — compute it once.
            let payload_bytes = quorum_signing_payload(&cert)?;
            for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
                let signer_enrollment = self.enrollments.get(signer_id)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: *signer_id })?;
                if signer_enrollment.issued_at > cert.issued_at {
                    return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Backdated-Signer" });
                }
                let vk = VerifyingKey::from_bytes(&signer_enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" })?;
                crate::signing::verify_with_tag(
                    &vk,
                    crate::signing::tags::ENROLLMENT,
                    &payload_bytes,
                    sig,
                    "Enrollment-Quorum-Member",
                )?;
            }
        }
        // Reject older enrollments for an existing device_id (prevents
        // rollback). If the existing cert has the same issued_at, accept
        // idempotently if the content matches; reject as a conflict if not
        // (two different certs with the same timestamp is a forge attempt or
        // replica drift that should not silently overwrite).
        if let Some(existing) = self.enrollments.get(&cert.device_id) {
            if existing.issued_at > cert.issued_at {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Rollback-Rejected",
                });
            }
            if existing.issued_at == cert.issued_at && existing != &cert {
                return Err(OwnerError::InvalidSignature {
                    cert_type: "Enrollment-Conflict",
                });
            }
        }
        self.enrollments.insert(cert.device_id, cert);
        Ok(())
    }

    pub fn add_liveness(&mut self, cert: LivenessCert) -> Result<(), OwnerError> {
        if cert.owner_id != self.owner_id {
            return Err(OwnerError::WrongOwner { expected: self.owner_id, got: cert.owner_id });
        }
        if self.is_revoked(cert.signer) {
            return Err(OwnerError::Revoked { device: cert.signer });
        }
        let enrollment = self.enrollments.get(&cert.signer)
            .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.signer })?;
        let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Liveness" })?;
        cert.verify(&vk)?;
        match self.liveness.get(&cert.signer) {
            Some(existing) if existing.timestamp >= cert.timestamp => { /* keep newer */ }
            _ => { self.liveness.insert(cert.signer, cert); }
        }
        Ok(())
    }

    pub fn add_vouching(&mut self, cert: crate::certs::VouchingCert) -> Result<(), OwnerError> {
        if cert.owner_id != self.owner_id {
            return Err(OwnerError::WrongOwner { expected: self.owner_id, got: cert.owner_id });
        }
        if self.is_revoked(cert.signer) {
            return Err(OwnerError::Revoked { device: cert.signer });
        }
        let enrollment = self.enrollments.get(&cert.signer)
            .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.signer })?;
        let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Vouching" })?;
        cert.verify(&vk)?;
        self.vouching.insert(cert);
        Ok(())
    }

    pub fn add_revocation(&mut self, cert: crate::certs::RevocationCert) -> Result<(), OwnerError> {
        if cert.owner_id != self.owner_id {
            return Err(OwnerError::WrongOwner { expected: self.owner_id, got: cert.owner_id });
        }
        // SelfDevice verification needs the device's pubkey from its enrollment.
        match &cert.issuer {
            crate::certs::RevocationIssuer::SelfDevice => {
                let enrollment = self.enrollments.get(&cert.target)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.target })?;
                let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Revocation" })?;
                cert.verify(Some(&vk))?;
            }
            crate::certs::RevocationIssuer::Quorum { .. } => {
                return Err(OwnerError::QuorumRevocationNotImplemented);
            }
            crate::certs::RevocationIssuer::Master { .. } => {
                cert.verify(None)?;
            }
        }
        self.revocations.insert(cert);
        Ok(())
    }

    pub fn active_devices(&self, now: u64, active_window_secs: u64) -> Vec<[u8; 16]> {
        let cutoff = now.saturating_sub(active_window_secs);
        self.enrollments.keys()
            .filter(|id| !self.revocations.is_revoked(**id))
            .filter(|id| match self.liveness.get(*id) {
                Some(l) => l.timestamp >= cutoff,
                None => false,
            })
            .copied()
            .collect()
    }

    pub fn is_revoked(&self, device: [u8; 16]) -> bool {
        self.revocations.is_revoked(device)
    }
}

/// Compute the canonical payload bytes used for quorum signature verification.
/// Uses the shared `EnrollmentSigningPayload` from `certs/enrollment.rs` so
/// signing and verification cannot drift apart.
fn quorum_signing_payload(cert: &EnrollmentCert) -> Result<Vec<u8>, OwnerError> {
    use crate::cbor;
    use crate::certs::enrollment::EnrollmentSigningPayload;

    let signers = match &cert.issuer {
        EnrollmentIssuer::Quorum { signers, .. } => signers,
        _ => return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" }),
    };

    // issuer_data for Quorum = cbor(signers list), same as enrollment.rs
    let issuer_data = cbor::to_canonical(signers)?;

    cbor::to_canonical(&EnrollmentSigningPayload {
        version: cert.version,
        owner_id: cert.owner_id,
        device_id: cert.device_id,
        device_pubkeys: &cert.device_pubkeys,
        issued_at: cert.issued_at,
        expires_at: cert.expires_at,
        issuer_kind: 1, // Quorum = 1
        issuer_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::{EnrollmentCert, LivenessCert};
    use crate::pubkey_bundle::{ClassicalKeys, PubKeyBundle};
    use ed25519_dalek::{Signer, SigningKey};
    use rand::rngs::OsRng;

    fn keypair_and_bundle() -> (SigningKey, PubKeyBundle) {
        let sk = SigningKey::generate(&mut OsRng);
        let bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        (sk, bundle)
    }

    #[test]
    fn add_master_enrollment_and_liveness_makes_active() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();

        let enrollment = EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_id, device_bundle, 1_000_000, None
        ).unwrap();

        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(enrollment, 1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();

        let liveness = LivenessCert::sign(&device_sk, owner_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let active = state.active_devices(1_500_000, 24 * 60 * 60);
        assert_eq!(active, vec![device_id]);
    }

    #[test]
    fn revoked_device_is_not_active() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();

        let enrollment = EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_id, device_bundle, 1_000_000, None
        ).unwrap();

        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(enrollment, 1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();

        let liveness = LivenessCert::sign(&device_sk, owner_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let revocation = crate::certs::RevocationCert::sign_self(
            &device_sk, owner_id, device_id, 1_400_000,
            crate::certs::RevocationReason::Decommissioned,
        ).unwrap();
        state.add_revocation(revocation).unwrap();

        let active = state.active_devices(1_500_000, 24 * 60 * 60);
        assert!(active.is_empty());
        assert!(state.is_revoked(device_id));
    }

    #[test]
    fn cert_with_wrong_owner_id_rejected() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (_, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();
        let mut state = OwnerState::new(owner_id);

        // Cert under a different owner_id (just zero-bytes)
        let wrong_owner_cert = EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_id, device_bundle, 1_000_000, None
        ).unwrap();
        let mut tampered = wrong_owner_cert;
        tampered.owner_id = [99u8; 16]; // simulate cross-owner cert
        // (cert.verify() will fail first because owner_id mismatch breaks the signature,
        //  but the OwnerState invariant is the first guard now)
        let result = state.add_enrollment(tampered, 1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        assert!(matches!(result, Err(OwnerError::WrongOwner { .. })));
    }

    #[test]
    fn quorum_with_duplicate_signers_rejected() {
        use crate::certs::{EnrollmentIssuer, EnrollmentCert};
        use crate::cbor;

        // Build a state with one device A
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let device_a_id = bundle_a.identity_hash();
        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(EnrollmentCert::sign_master(
            &master_sk, master_bundle.clone(), device_a_id, bundle_a, 1_000_000, None
        ).unwrap(), 1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();

        // Forge a quorum cert with [A, A] — same signer twice
        let (_, bundle_c) = keypair_and_bundle();
        let device_c_id = bundle_c.identity_hash();
        let signers = vec![device_a_id, device_a_id];
        let issuer_data = cbor::to_canonical(&signers).unwrap();
        use serde::Serialize;
        #[derive(Serialize)]
        struct Payload<'a> {
            version: u8,
            owner_id: [u8; 16],
            device_id: [u8; 16],
            device_pubkeys: &'a crate::pubkey_bundle::PubKeyBundle,
            issued_at: u64,
            expires_at: Option<u64>,
            issuer_kind: u8,
            issuer_data: Vec<u8>,
        }
        let payload_bytes = cbor::to_canonical(&Payload {
            version: crate::certs::enrollment::ENROLLMENT_VERSION,
            owner_id,
            device_id: device_c_id,
            device_pubkeys: &bundle_c,
            issued_at: 1_001_000,
            expires_at: None,
            issuer_kind: 1,
            issuer_data,
        }).unwrap();
        let sig_a = crate::signing::sign_with_tag(&sk_a, crate::signing::tags::ENROLLMENT, &payload_bytes);
        let cert = EnrollmentCert {
            version: crate::certs::enrollment::ENROLLMENT_VERSION,
            owner_id,
            device_id: device_c_id,
            device_pubkeys: bundle_c,
            issued_at: 1_001_000,
            expires_at: None,
            issuer: EnrollmentIssuer::Quorum {
                signers: vec![device_a_id, device_a_id],
                signatures: vec![sig_a.clone(), sig_a],
            },
            signature: Vec::new(),
        };
        let result = state.add_enrollment(cert, 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        // Defense-in-depth: EnrollmentCert::verify now also catches duplicate
        // signers (returning cert_type "Enrollment"), and that fires before
        // OwnerState's own "Enrollment-Quorum-Duplicate-Signer" check. Either
        // rejection path is correct — both surface the same invariant.
        assert!(matches!(
            result,
            Err(OwnerError::InvalidSignature {
                cert_type: "Enrollment" | "Enrollment-Quorum-Duplicate-Signer"
            })
        ));
    }

    #[test]
    fn quorum_revocation_rejected_until_implemented() {
        use crate::certs::{RevocationCert, RevocationIssuer, RevocationReason};
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let device_a_id = bundle_a.identity_hash();
        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(EnrollmentCert::sign_master(
            &master_sk, master_bundle, device_a_id, bundle_a, 1_000_000, None
        ).unwrap(), 1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();

        // Construct a Quorum revocation manually
        let cert = RevocationCert {
            version: 1,
            owner_id,
            target: device_a_id,
            issued_at: 1_001_000,
            issuer: RevocationIssuer::Quorum {
                signers: vec![device_a_id],
                signatures: vec![sk_a.sign(b"dummy").to_bytes().to_vec()],
            },
            reason: RevocationReason::Compromised,
            signature: Vec::new(),
        };
        let result = state.add_revocation(cert);
        assert!(matches!(result, Err(OwnerError::QuorumRevocationNotImplemented)));
    }

    #[test]
    fn quorum_with_inactive_signer_rejected() {
        // Setup: enroll A and C via master, only A publishes liveness.
        // Attempt to enroll D via quorum [A, C] — C is inactive, should reject.
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let id_a = bundle_a.identity_hash();
        let (sk_c, bundle_c) = keypair_and_bundle();
        let id_c = bundle_c.identity_hash();

        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(
            EnrollmentCert::sign_master(&master_sk, master_bundle.clone(), id_a, bundle_a, 1_000_000, None).unwrap(),
            1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();
        state.add_enrollment(
            EnrollmentCert::sign_master(&master_sk, master_bundle, id_c, bundle_c, 1_000_001, None).unwrap(),
            1_000_001, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();

        // A publishes liveness, C does not.
        state.add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_000_500).unwrap()).unwrap();

        // Attempt quorum enrollment of new device D using [A, C] — C is inactive.
        use crate::cbor;
        let (_, bundle_d) = keypair_and_bundle();
        let id_d = bundle_d.identity_hash();
        let signers = vec![id_a, id_c];
        let issuer_data = cbor::to_canonical(&signers).unwrap();
        use crate::certs::enrollment::EnrollmentSigningPayload;
        let payload_bytes = cbor::to_canonical(&EnrollmentSigningPayload {
            version: crate::certs::enrollment::ENROLLMENT_VERSION,
            owner_id,
            device_id: id_d,
            device_pubkeys: &bundle_d,
            issued_at: 1_001_000,
            expires_at: None,
            issuer_kind: 1,
            issuer_data,
        }).unwrap();
        let sig_a = crate::signing::sign_with_tag(&sk_a, crate::signing::tags::ENROLLMENT, &payload_bytes);
        let sig_c = crate::signing::sign_with_tag(&sk_c, crate::signing::tags::ENROLLMENT, &payload_bytes);
        let cert = EnrollmentCert {
            version: crate::certs::enrollment::ENROLLMENT_VERSION,
            owner_id,
            device_id: id_d,
            device_pubkeys: bundle_d,
            issued_at: 1_001_000,
            expires_at: None,
            issuer: EnrollmentIssuer::Quorum {
                signers: vec![id_a, id_c],
                signatures: vec![sig_a, sig_c],
            },
            signature: Vec::new(),
        };
        let result = state.add_enrollment(cert, 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        assert!(matches!(
            result,
            Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Inactive-Signer" })
        ));
    }

    #[test]
    fn liveness_from_revoked_device_rejected() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let id_a = bundle_a.identity_hash();
        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(
            EnrollmentCert::sign_master(&master_sk, master_bundle, id_a, bundle_a, 1_000_000, None).unwrap(),
            1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();
        let rev = crate::certs::RevocationCert::sign_self(
            &sk_a, owner_id, id_a, 1_000_500, crate::certs::RevocationReason::Compromised
        ).unwrap();
        state.add_revocation(rev).unwrap();

        let liveness = LivenessCert::sign(&sk_a, owner_id, 1_001_000).unwrap();
        let result = state.add_liveness(liveness);
        assert!(matches!(result, Err(OwnerError::Revoked { device }) if device == id_a));
    }

    #[test]
    fn rollback_enrollment_rejected() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (_, device_bundle) = keypair_and_bundle();
        let device_id = device_bundle.identity_hash();
        let mut state = OwnerState::new(owner_id);

        // Newer enrollment (issued at later timestamp) is added first
        let newer = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle.clone(),
            device_id,
            device_bundle.clone(),
            2_000_000,
            None,
        )
        .unwrap();
        state
            .add_enrollment(newer, 2_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS)
            .unwrap();

        // Older enrollment for same device — should be rejected as rollback
        let older = EnrollmentCert::sign_master(
            &master_sk,
            master_bundle,
            device_id,
            device_bundle,
            1_000_000,
            None,
        )
        .unwrap();
        let result =
            state.add_enrollment(older, 2_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        assert!(matches!(
            result,
            Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Rollback-Rejected" })
        ));
    }

    #[test]
    fn vouching_from_revoked_device_rejected() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let id_a = bundle_a.identity_hash();
        let mut state = OwnerState::new(owner_id);
        state.add_enrollment(
            EnrollmentCert::sign_master(&master_sk, master_bundle, id_a, bundle_a, 1_000_000, None).unwrap(),
            1_000_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();
        let rev = crate::certs::RevocationCert::sign_self(
            &sk_a, owner_id, id_a, 1_000_500, crate::certs::RevocationReason::Compromised
        ).unwrap();
        state.add_revocation(rev).unwrap();

        let vouch = crate::certs::VouchingCert::sign(
            &sk_a, owner_id, [9u8; 16], crate::certs::Stance::Vouch, 1_001_000
        ).unwrap();
        let result = state.add_vouching(vouch);
        assert!(matches!(result, Err(OwnerError::Revoked { device }) if device == id_a));
    }
}

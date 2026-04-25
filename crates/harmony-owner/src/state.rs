use crate::certs::{EnrollmentCert, EnrollmentIssuer, LivenessCert, ReclamationCert};
use crate::crdt::{RevocationSet, VouchingSet};
use crate::OwnerError;
use ed25519_dalek::VerifyingKey;
use std::collections::HashMap;

/// Aggregate state for one owner identity: enrollment certs per device,
/// vouching CRDT, revocation set, latest liveness per device, optional
/// reclamation cert if this identity claims continuity from a predecessor.
#[derive(Debug, Clone, Default)]
pub struct OwnerState {
    pub owner_id: [u8; 16],
    pub enrollments: HashMap<[u8; 16], EnrollmentCert>,
    pub vouching: VouchingSet,
    pub revocations: RevocationSet,
    pub liveness: HashMap<[u8; 16], LivenessCert>,
    pub reclamation: Option<ReclamationCert>,
}

impl OwnerState {
    pub fn new(owner_id: [u8; 16]) -> Self {
        Self { owner_id, ..Default::default() }
    }

    pub fn add_enrollment(&mut self, cert: EnrollmentCert) -> Result<(), OwnerError> {
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
            // Verify quorum signatures: each signer in `signers` must have an
            // existing enrollment, and each signature in `signatures` must be
            // valid for that signer's pubkey.
            for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
                let signer_enrollment = self.enrollments.get(signer_id)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: *signer_id })?;
                if signer_enrollment.issued_at > cert.issued_at {
                    return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Backdated-Signer" });
                }
                let vk = VerifyingKey::from_bytes(&signer_enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" })?;
                let payload_bytes = quorum_signing_payload(&cert)?;
                crate::signing::verify_with_tag(
                    &vk,
                    crate::signing::tags::ENROLLMENT,
                    &payload_bytes,
                    sig,
                    "Enrollment-Quorum-Member",
                )?;
            }
        }
        self.enrollments.insert(cert.device_id, cert);
        Ok(())
    }

    pub fn add_liveness(&mut self, cert: LivenessCert) -> Result<(), OwnerError> {
        if cert.owner_id != self.owner_id {
            return Err(OwnerError::WrongOwner { expected: self.owner_id, got: cert.owner_id });
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
            _ => {
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
        state.add_enrollment(enrollment).unwrap();

        let liveness = LivenessCert::sign(&device_sk, owner_id, device_id, 1_500_000).unwrap();
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
        state.add_enrollment(enrollment).unwrap();

        let liveness = LivenessCert::sign(&device_sk, owner_id, device_id, 1_500_000).unwrap();
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
        let result = state.add_enrollment(tampered);
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
        ).unwrap()).unwrap();

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
        let result = state.add_enrollment(cert);
        assert!(matches!(result, Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Duplicate-Signer" })));
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
        ).unwrap()).unwrap();

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
}

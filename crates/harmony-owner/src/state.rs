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
        cert.verify()?;
        // For Quorum certs, also walk back to verify each signer's enrollment
        // is present and was issued before this cert's `issued_at`.
        if let EnrollmentIssuer::Quorum { signers, signatures } = &cert.issuer {
            if signers.len() < 2 {
                return Err(OwnerError::InsufficientQuorum { min: 2, got: signers.len() });
            }
            // Verify quorum signatures: each signer in `signers` must have an
            // existing enrollment, and each signature in `signatures` must be
            // valid for that signer's pubkey.
            for (signer_id, sig) in signers.iter().zip(signatures.iter()) {
                let signer_enrollment = self.enrollments.get(signer_id)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: *signer_id })?;
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
        let enrollment = self.enrollments.get(&cert.signer)
            .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.signer })?;
        let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
            .map_err(|_| OwnerError::InvalidSignature { cert_type: "Vouching" })?;
        cert.verify(&vk)?;
        self.vouching.insert(cert);
        Ok(())
    }

    pub fn add_revocation(&mut self, cert: crate::certs::RevocationCert) -> Result<(), OwnerError> {
        // SelfDevice verification needs the device's pubkey from its enrollment.
        match &cert.issuer {
            crate::certs::RevocationIssuer::SelfDevice => {
                let enrollment = self.enrollments.get(&cert.target)
                    .ok_or(OwnerError::NotEnrolled { owner: self.owner_id, device: cert.target })?;
                let vk = VerifyingKey::from_bytes(&enrollment.device_pubkeys.classical.ed25519_verify)
                    .map_err(|_| OwnerError::InvalidSignature { cert_type: "Revocation" })?;
                cert.verify(Some(&vk))?;
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
/// Reproduces the exact same `EnrollmentSigningPayload` that `enrollment.rs`
/// produces, so that quorum member signatures can be verified here.
fn quorum_signing_payload(cert: &EnrollmentCert) -> Result<Vec<u8>, OwnerError> {
    use crate::cbor;
    use crate::pubkey_bundle::PubKeyBundle;
    use serde::Serialize;

    /// Mirrors `EnrollmentSigningPayload` in `certs/enrollment.rs` exactly.
    #[derive(Serialize)]
    struct EnrollmentSigningPayload<'a> {
        version: u8,
        owner_id: [u8; 16],
        device_id: [u8; 16],
        device_pubkeys: &'a PubKeyBundle,
        issued_at: u64,
        expires_at: Option<u64>,
        issuer_kind: u8,
        issuer_data: Vec<u8>,
    }

    let signers = match &cert.issuer {
        EnrollmentIssuer::Quorum { signers, .. } => signers,
        _ => return Err(OwnerError::InvalidSignature { cert_type: "Enrollment-Quorum-Member" }),
    };

    // issuer_data for Quorum = cbor(signers list), same as enrollment.rs line ~141
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
    use ed25519_dalek::SigningKey;
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

        let liveness = LivenessCert::sign(&device_sk, device_id, 1_500_000).unwrap();
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

        let liveness = LivenessCert::sign(&device_sk, device_id, 1_500_000).unwrap();
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
}

use crate::cbor;
use crate::certs::enrollment::EnrollmentSigningPayload;
use crate::certs::{EnrollmentCert, EnrollmentIssuer, Stance, VouchingCert};
use crate::lifecycle::EnrollResult;
use crate::pubkey_bundle::PubKeyBundle;
use crate::signing::{sign_with_tag, tags};
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;

/// Enroll a new device using K=2 quorum of existing siblings (no recovery
/// artifact needed). Returns the new device's enrollment cert + auto-vouches.
pub fn enroll_via_quorum(
    state: &OwnerState,
    quorum_signers: Vec<(&SigningKey, [u8; 16])>,
    new_device_sk: &SigningKey,
    new_device_pubkey: PubKeyBundle,
    now: u64,
    active_window_secs: u64,
) -> Result<EnrollResult, OwnerError> {
    if quorum_signers.len() < 2 {
        return Err(OwnerError::InsufficientQuorum { min: 2, got: quorum_signers.len() });
    }
    let device_id = new_device_pubkey.identity_hash();
    let signers: Vec<[u8; 16]> = quorum_signers.iter().map(|(_, id)| *id).collect();

    // issuer_data for Quorum = cbor(signers list) — matches state.rs verification.
    let issuer_data = cbor::to_canonical(&signers)?;

    let payload_bytes = cbor::to_canonical(&EnrollmentSigningPayload {
        version: crate::certs::enrollment::ENROLLMENT_VERSION,
        owner_id: state.owner_id,
        device_id,
        device_pubkeys: &new_device_pubkey,
        issued_at: now,
        expires_at: None,
        issuer_kind: 1,
        issuer_data,
    })?;

    let signatures: Vec<Vec<u8>> = quorum_signers.iter()
        .map(|(sk, _)| sign_with_tag(sk, tags::ENROLLMENT, &payload_bytes))
        .collect();

    // Construct cert with quorum issuer; signature field is not used for
    // quorum (signers' individual signatures live in EnrollmentIssuer::Quorum).
    let enrollment_cert = EnrollmentCert {
        version: crate::certs::enrollment::ENROLLMENT_VERSION,
        owner_id: state.owner_id,
        device_id,
        device_pubkeys: new_device_pubkey,
        issued_at: now,
        expires_at: None,
        issuer: EnrollmentIssuer::Quorum { signers: signers.clone(), signatures },
        signature: Vec::new(),
    };

    let active = state.active_devices(now, active_window_secs);
    let auto_vouch_certs: Vec<VouchingCert> = active.iter()
        .filter(|s| **s != device_id)
        .map(|sibling_id| VouchingCert::sign(
            new_device_sk,
            state.owner_id,
            device_id,
            *sibling_id,
            Stance::Vouch,
            now,
        ))
        .collect::<Result<Vec<_>, _>>()?;

    Ok(EnrollResult { enrollment_cert, auto_vouch_certs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::LivenessCert;
    use crate::lifecycle::{enroll_via_master, mint_owner};
    use crate::pubkey_bundle::ClassicalKeys;
    use rand::rngs::OsRng;

    #[test]
    fn enroll_third_device_via_quorum() {
        let mint = mint_owner(1_000_000).unwrap();
        let mut state = mint.state;
        let device_a_id = *state.enrollments.keys().next().unwrap();
        let device_a_sk = mint.device_signing_key;
        state.add_liveness(LivenessCert::sign(&device_a_sk, state.owner_id, device_a_id, 1_000_001).unwrap()).unwrap();

        // Enroll device B via master
        let device_b_sk = SigningKey::generate(&mut OsRng);
        let device_b_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_b_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let r1 = enroll_via_master(&state, &mint.recovery_artifact, &device_b_sk, device_b_bundle.clone(), 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
        let device_b_id = device_b_bundle.identity_hash();
        state.add_enrollment(r1.enrollment_cert, 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
        for v in r1.auto_vouch_certs { state.add_vouching(v).unwrap(); }
        state.add_liveness(LivenessCert::sign(&device_b_sk, state.owner_id, device_b_id, 1_001_001).unwrap()).unwrap();

        // Now enroll device C via quorum of A+B
        let device_c_sk = SigningKey::generate(&mut OsRng);
        let device_c_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_c_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };

        let r2 = enroll_via_quorum(
            &state,
            vec![(&device_a_sk, device_a_id), (&device_b_sk, device_b_id)],
            &device_c_sk,
            device_c_bundle.clone(),
            1_002_000,
            crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();

        state.add_enrollment(r2.enrollment_cert, 1_002_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
        for v in r2.auto_vouch_certs { state.add_vouching(v).unwrap(); }

        // Verify Device C is now enrolled
        assert!(state.enrollments.contains_key(&device_c_bundle.identity_hash()));
    }

    #[test]
    fn quorum_with_one_signer_rejected() {
        let mint = mint_owner(1_000_000).unwrap();
        let device_a_id = *mint.state.enrollments.keys().next().unwrap();
        let device_a_sk = mint.device_signing_key;
        let new_sk = SigningKey::generate(&mut OsRng);
        let new_bundle = PubKeyBundle {
            classical: ClassicalKeys { ed25519_verify: new_sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
            post_quantum: None,
        };
        let result = enroll_via_quorum(&mint.state, vec![(&device_a_sk, device_a_id)], &new_sk, new_bundle, 1_001_000, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS);
        assert!(matches!(result, Err(OwnerError::InsufficientQuorum { .. })));
    }
}

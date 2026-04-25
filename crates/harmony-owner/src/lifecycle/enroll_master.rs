use crate::certs::{EnrollmentCert, Stance, VouchingCert};
use crate::lifecycle::RecoveryArtifact;
use crate::pubkey_bundle::PubKeyBundle;
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;

pub struct EnrollResult {
    pub enrollment_cert: EnrollmentCert,
    pub auto_vouch_certs: Vec<VouchingCert>,
}

/// Enroll a new device under the existing owner via the recovery artifact.
///
/// This brings the master signing key into RAM transiently, signs the new
/// device's enrollment cert, and immediately drops the master key. The new
/// device also auto-vouches for every active sibling.
///
/// Returns the new device's enrollment cert plus auto-vouches for siblings.
pub fn enroll_via_master(
    state: &OwnerState,
    artifact: &RecoveryArtifact,
    new_device_sk: &SigningKey,
    new_device_pubkey: PubKeyBundle,
    now: u64,
    active_window_secs: u64,
) -> Result<EnrollResult, OwnerError> {
    // Reconstruct master from artifact (transient).
    let master_sk = artifact.master_signing_key();
    let master_pubkey = master_pubkey_from_sk(&master_sk);

    if master_pubkey.identity_hash() != state.owner_id {
        return Err(OwnerError::WrongOwner {
            expected: state.owner_id,
            got: master_pubkey.identity_hash(),
        });
    }

    let device_id = new_device_pubkey.identity_hash();
    let enrollment_cert = EnrollmentCert::sign_master(
        &master_sk,
        master_pubkey,
        device_id,
        new_device_pubkey,
        now,
        None,
    )?;
    drop(master_sk); // wipe master from RAM

    // New device auto-vouches for every active sibling.
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

fn master_pubkey_from_sk(sk: &SigningKey) -> PubKeyBundle {
    use crate::pubkey_bundle::ClassicalKeys;
    PubKeyBundle {
        classical: ClassicalKeys { ed25519_verify: sk.verifying_key().to_bytes(), x25519_pub: [0u8; 32] },
        post_quantum: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::LivenessCert;
    use crate::lifecycle::mint_owner;
    use crate::pubkey_bundle::ClassicalKeys;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn enroll_second_device_via_master() {
        let mint = mint_owner(1_000_000).unwrap();
        // Device #1 is alive
        let device_a_id = *mint.state.enrollments.keys().next().unwrap();
        let mut state = mint.state;
        state.add_liveness(LivenessCert::sign(&mint.device_signing_key, device_a_id, 1_000_001).unwrap()).unwrap();

        // Generate device #2
        let device_b_sk = SigningKey::generate(&mut OsRng);
        let device_b_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: device_b_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };

        let result = enroll_via_master(
            &state,
            &mint.recovery_artifact,
            &device_b_sk,
            device_b_bundle,
            1_001_000,
            crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        ).unwrap();

        // Apply to state
        state.add_enrollment(result.enrollment_cert.clone()).unwrap();
        for v in &result.auto_vouch_certs {
            state.add_vouching(v.clone()).unwrap();
        }

        // Auto-vouch should reach exactly device A (and not B itself)
        assert_eq!(result.auto_vouch_certs.len(), 1);
        assert_eq!(result.auto_vouch_certs[0].target, device_a_id);
    }

    #[test]
    fn enroll_via_master_with_wrong_artifact_rejected() {
        // Mint two separate identities
        let mint_a = mint_owner(1_000_000).unwrap();
        let mint_b = mint_owner(1_000_001).unwrap();

        // Try to enroll a new device into mint_a's state using mint_b's artifact
        let new_sk = SigningKey::generate(&mut OsRng);
        let new_bundle = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: new_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        };
        let result = enroll_via_master(
            &mint_a.state,
            &mint_b.recovery_artifact, // wrong artifact
            &new_sk,
            new_bundle,
            1_002_000,
            crate::trust::DEFAULT_ACTIVE_WINDOW_SECS,
        );
        assert!(matches!(result, Err(crate::OwnerError::WrongOwner { .. })));
    }
}

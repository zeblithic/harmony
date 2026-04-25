use crate::certs::EnrollmentCert;
use crate::pubkey_bundle::{ClassicalKeys, PubKeyBundle};
use crate::state::OwnerState;
use crate::OwnerError;
use ed25519_dalek::SigningKey;
use rand_core::{OsRng, RngCore};
use zeroize::Zeroize;

/// 32-byte master seed. Format BIP39-wraps to 24 mnemonic words. Drop wipes.
pub struct RecoveryArtifact {
    seed: [u8; 32],
}

impl RecoveryArtifact {
    pub fn from_seed(seed: [u8; 32]) -> Self {
        Self { seed }
    }
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.seed
    }
    /// Reconstruct master signing key from the seed.
    pub fn master_signing_key(&self) -> SigningKey {
        SigningKey::from_bytes(&self.seed)
    }
    /// Reconstruct the master `PubKeyBundle` from the seed. This is the
    /// canonical source of truth for the master pubkey shape — every call
    /// site that needs the master bundle must use this method (or the
    /// shared `master_pubkey_bundle_from_sk` helper at mint time), not
    /// build the bundle inline, so any future change (e.g., HKDF-derived
    /// X25519) propagates uniformly.
    pub fn master_pubkey_bundle(&self) -> PubKeyBundle {
        master_pubkey_bundle_from_sk(&self.master_signing_key())
    }
}

/// Construct a master `PubKeyBundle` from a master signing key. Shared by
/// `mint_owner` (where we don't yet have a `RecoveryArtifact`) and by
/// `RecoveryArtifact::master_pubkey_bundle`. Single source of truth for
/// the master pubkey shape.
pub(crate) fn master_pubkey_bundle_from_sk(sk: &SigningKey) -> PubKeyBundle {
    PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32], // TODO v1.1: derive via HKDF from same seed
        },
        post_quantum: None,
    }
}

impl Drop for RecoveryArtifact {
    fn drop(&mut self) {
        self.seed.zeroize();
    }
}

pub struct MintResult {
    pub state: OwnerState,
    pub recovery_artifact: RecoveryArtifact,
    pub device_signing_key: SigningKey,
}

/// Mint a fresh owner identity with device #1.
///
/// Returns the OwnerState (with device #1 enrolled), the recovery artifact
/// (which the user must back up — it encodes the master key), and device
/// #1's signing key (which the device retains for ongoing operation).
///
/// IMPORTANT: After this returns, the master key is reconstructible only
/// from the recovery artifact. Callers must never persist the master key
/// outside that artifact.
pub fn mint_owner(now: u64) -> Result<MintResult, OwnerError> {
    let mut seed = [0u8; 32];
    OsRng.fill_bytes(&mut seed);
    let master_sk = SigningKey::from_bytes(&seed);
    let master_bundle = master_pubkey_bundle_from_sk(&master_sk);
    let owner_id = master_bundle.identity_hash();

    let device_sk = SigningKey::generate(&mut OsRng);
    let device_bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: device_sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    let device_id = device_bundle.identity_hash();

    let cert = EnrollmentCert::sign_master(
        &master_sk,
        master_bundle,
        device_id,
        device_bundle,
        now,
        None,
    )?;

    let mut state = OwnerState::new(owner_id);
    // Master enrollment bypasses the active-signer check, so the active
    // window value here is irrelevant; pass the trust default for consistency.
    state.add_enrollment(cert, now, crate::trust::DEFAULT_ACTIVE_WINDOW_SECS)?;

    let recovery_artifact = RecoveryArtifact::from_seed(seed);
    seed.zeroize();
    // master_sk is dropped here, signaling intent to wipe master from RAM
    // (callers should ensure they don't retain master_sk anywhere).
    drop(master_sk);

    Ok(MintResult {
        state,
        recovery_artifact,
        device_signing_key: device_sk,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mint_produces_active_device_one() {
        let result = mint_owner(1_700_000_000).unwrap();
        assert_eq!(result.state.enrollments.len(), 1);
        assert_eq!(result.recovery_artifact.as_bytes().len(), 32);
    }

    #[test]
    fn recovery_artifact_round_trip_yields_same_master_key() {
        let result = mint_owner(1_700_000_000).unwrap();
        let restored_sk = result.recovery_artifact.master_signing_key();
        let owner_id_via_restored = PubKeyBundle {
            classical: ClassicalKeys {
                ed25519_verify: restored_sk.verifying_key().to_bytes(),
                x25519_pub: [0u8; 32],
            },
            post_quantum: None,
        }.identity_hash();
        assert_eq!(owner_id_via_restored, result.state.owner_id);
    }
}

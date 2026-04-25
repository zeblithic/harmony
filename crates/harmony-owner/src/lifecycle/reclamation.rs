use crate::certs::{LivenessCert, ReclamationCert};
use crate::lifecycle::{mint_owner, MintResult};
use crate::OwnerError;

pub struct ReclamationMintResult {
    pub mint: MintResult,
    pub reclamation_cert: ReclamationCert,
}

/// Mint a fresh identity AND publish a Reclamation Cert claiming continuity
/// from a predecessor.
pub fn mint_reclaimed(
    claimed_predecessor: [u8; 16],
    challenge_window_secs: u64,
    note: String,
    now: u64,
) -> Result<ReclamationMintResult, OwnerError> {
    let mint = mint_owner(now)?;
    // Reconstruct master to sign the reclamation cert
    let master_sk = mint.recovery_artifact.master_signing_key();
    let master_pubkey = mint.recovery_artifact.master_pubkey_bundle();
    let reclamation_cert = ReclamationCert::sign(
        &master_sk,
        master_pubkey,
        claimed_predecessor,
        now,
        challenge_window_secs,
        note,
    )?;
    drop(master_sk);
    Ok(ReclamationMintResult { mint, reclamation_cert })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReclamationStatus {
    /// Window still open; no refutation observed yet.
    Pending,
    /// Window expired without refutation; honored at reduced trust.
    Honored,
    /// Refuted by a predecessor liveness cert with timestamp > issued_at.
    Refuted,
}

/// Evaluate a reclamation cert's status given the current time and any
/// observed liveness from devices under the predecessor identity.
pub fn evaluate_reclamation(
    cert: &ReclamationCert,
    predecessor_liveness_certs: &[LivenessCert],
    now: u64,
) -> ReclamationStatus {
    let refuted = predecessor_liveness_certs.iter()
        .any(|l| cert.is_refuted_by_timestamp(l.timestamp));
    if refuted {
        return ReclamationStatus::Refuted;
    }
    if cert.is_window_expired(now) {
        ReclamationStatus::Honored
    } else {
        ReclamationStatus::Pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::reclamation::DEFAULT_CHALLENGE_WINDOW_SECS;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn pending_during_window() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[], 1_000_500);
        assert_eq!(status, ReclamationStatus::Pending);
    }

    #[test]
    fn honored_after_window() {
        let result = mint_reclaimed([9u8; 16], 1000, "lost".into(), 1_000_000).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[], 1_002_000);
        assert_eq!(status, ReclamationStatus::Honored);
    }

    #[test]
    fn refuted_by_predecessor_liveness() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let predecessor_sk = SigningKey::generate(&mut OsRng);
        let liveness = LivenessCert::sign(&predecessor_sk, [9u8; 16], [9u8; 16], 1_000_500).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[liveness], 1_000_500);
        assert_eq!(status, ReclamationStatus::Refuted);
    }

    #[test]
    fn liveness_before_reclamation_does_not_refute() {
        let result = mint_reclaimed([9u8; 16], DEFAULT_CHALLENGE_WINDOW_SECS, "lost".into(), 1_000_000).unwrap();
        let predecessor_sk = SigningKey::generate(&mut OsRng);
        let liveness = LivenessCert::sign(&predecessor_sk, [9u8; 16], [9u8; 16], 999_999).unwrap();
        let status = evaluate_reclamation(&result.reclamation_cert, &[liveness], 1_000_500);
        assert_eq!(status, ReclamationStatus::Pending);
    }
}

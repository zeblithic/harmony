//! End-to-end threat scenarios from the design's threat-coverage matrix:
//! stolen master, contested reclamation, partition + replay.

use harmony_owner::{
    certs::{LivenessCert, RevocationCert, RevocationReason, Stance, VouchingCert},
    lifecycle::{enroll_via_master, evaluate_reclamation, mint_owner, mint_reclaimed, ReclamationStatus},
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
    trust::{evaluate_trust, RefusalReason, TrustDecision, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS},
    certs::reclamation::DEFAULT_CHALLENGE_WINDOW_SECS,
};
use ed25519_dalek::SigningKey;
use rand::rngs::OsRng;

fn fresh_device() -> (SigningKey, PubKeyBundle) {
    let sk = SigningKey::generate(&mut OsRng);
    let bundle = PubKeyBundle {
        classical: ClassicalKeys {
            ed25519_verify: sk.verifying_key().to_bytes(),
            x25519_pub: [0u8; 32],
        },
        post_quantum: None,
    };
    (sk, bundle)
}

#[test]
fn stolen_master_attacker_device_remains_provisional_when_real_devices_dont_vouch() {
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;
    state.add_liveness(LivenessCert::sign(&device_a_sk, state.owner_id, device_a_id, 1_000_001).unwrap()).unwrap();

    // Attacker has the recovery artifact and enrolls a malicious device.
    let (attacker_sk, attacker_bundle) = fresh_device();
    let attacker_id = attacker_bundle.identity_hash();
    let r = enroll_via_master(&state, &mint.recovery_artifact, &attacker_sk, attacker_bundle, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    state.add_enrollment(r.enrollment_cert, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    for v in r.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&attacker_sk, state.owner_id, attacker_id, 1_500_001).unwrap()).unwrap();

    // Real device A does NOT vouch for the attacker. Attacker should stay provisional.
    assert_eq!(
        evaluate_trust(&state, attacker_id, 1_500_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Provisional
    );

    // Real device A challenges. Attacker is refused.
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_a_id, attacker_id, Stance::Challenge, 1_500_002).unwrap()).unwrap();
    assert_eq!(
        evaluate_trust(&state, attacker_id, 1_500_002, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Refused(RefusalReason::ChallengedBySibling)
    );
}

#[test]
fn revoked_device_cannot_be_un_revoked() {
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;

    // Self-revoke
    let rev = RevocationCert::sign_self(&device_a_sk, state.owner_id, device_a_id, 1_000_500, RevocationReason::Decommissioned).unwrap();
    state.add_revocation(rev).unwrap();
    assert!(state.is_revoked(device_a_id));

    // Even an "older" revocation insertion does not un-revoke (it stays revoked, with the earliest cert kept).
    let rev2 = RevocationCert::sign_self(&device_a_sk, state.owner_id, device_a_id, 999_000, RevocationReason::Other("test".into())).unwrap();
    state.add_revocation(rev2).unwrap();
    assert!(state.is_revoked(device_a_id));
}

#[test]
fn reclamation_refuted_by_predecessor_liveness() {
    // Predecessor identity is alive
    let predecessor_mint = mint_owner(1_000_000).unwrap();
    let predecessor_owner_id = predecessor_mint.state.owner_id;
    let predecessor_device_id = *predecessor_mint.state.enrollments.keys().next().unwrap();
    let predecessor_sk = predecessor_mint.device_signing_key;

    // M2 publishes reclamation
    let reclaim = mint_reclaimed(predecessor_owner_id, DEFAULT_CHALLENGE_WINDOW_SECS, "thought all devices were lost".into(), 2_000_000).unwrap();

    // Predecessor publishes liveness within window
    let predecessor_liveness = LivenessCert::sign(&predecessor_sk, predecessor_owner_id, predecessor_device_id, 2_000_500).unwrap();

    let status = evaluate_reclamation(&reclaim.reclamation_cert, &[predecessor_liveness], 2_001_000);
    assert_eq!(status, ReclamationStatus::Refuted);
}

#[test]
fn reclamation_honored_after_silent_window() {
    let reclaim = mint_reclaimed([7u8; 16], 1000, "fire took everything".into(), 1_000_000).unwrap();
    let status = evaluate_reclamation(&reclaim.reclamation_cert, &[], 1_002_000);
    assert_eq!(status, ReclamationStatus::Honored);
}

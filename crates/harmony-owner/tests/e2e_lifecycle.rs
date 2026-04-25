//! End-to-end happy-path: mint, enroll three devices via mixed paths,
//! exchange vouches, verify trust evaluation, archive a stale device.

use harmony_owner::{
    certs::{LivenessCert, RevocationCert, RevocationReason, Stance, VouchingCert},
    lifecycle::{enroll_via_master, enroll_via_quorum, mint_owner},
    pubkey_bundle::{ClassicalKeys, PubKeyBundle},
    trust::{evaluate_trust, RefusalReason, TrustDecision, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS},
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
fn full_three_device_lifecycle() {
    // T0: Mint
    let mint = mint_owner(1_000_000).unwrap();
    let device_a_id = *mint.state.enrollments.keys().next().unwrap();
    let device_a_sk = mint.device_signing_key;
    let mut state = mint.state;
    state.add_liveness(LivenessCert::sign(&device_a_sk, state.owner_id, 1_000_001).unwrap()).unwrap();

    // Single-device → Full trust
    assert_eq!(
        evaluate_trust(&state, device_a_id, 1_000_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T1: Enroll Device B via master
    let (device_b_sk, device_b_bundle) = fresh_device();
    let device_b_id = device_b_bundle.identity_hash();
    let r1 = enroll_via_master(&state, &mint.recovery_artifact, &device_b_sk, device_b_bundle, 1_001_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    state.add_enrollment(r1.enrollment_cert, 1_001_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    for v in r1.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&device_b_sk, state.owner_id, 1_001_001).unwrap()).unwrap();

    // Device B is provisional (auto-vouches don't count for B because B vouched for A, not the other way)
    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_001_001, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Provisional
    );

    // Device A ratifies B
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_b_id, Stance::Vouch, 1_001_500).unwrap()).unwrap();

    // Device B is now Full
    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_001_500, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T2: Enroll Device C via quorum of A+B
    let (device_c_sk, device_c_bundle) = fresh_device();
    let device_c_id = device_c_bundle.identity_hash();
    let r2 = enroll_via_quorum(
        &state,
        vec![(&device_a_sk, device_a_id), (&device_b_sk, device_b_id)],
        &device_c_sk,
        device_c_bundle,
        1_002_000,
        DEFAULT_ACTIVE_WINDOW_SECS,
    ).unwrap();
    state.add_enrollment(r2.enrollment_cert, 1_002_000, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
    for v in r2.auto_vouch_certs { state.add_vouching(v).unwrap(); }
    state.add_liveness(LivenessCert::sign(&device_c_sk, state.owner_id, 1_002_001).unwrap()).unwrap();

    // Device A ratifies C
    state.add_vouching(VouchingCert::sign(&device_a_sk, state.owner_id, device_c_id, Stance::Vouch, 1_002_500).unwrap()).unwrap();

    assert_eq!(
        evaluate_trust(&state, device_c_id, 1_002_500, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Full
    );

    // T3: Revoke Device B (decommissioned)
    let revocation = RevocationCert::sign_self(&device_b_sk, state.owner_id, device_b_id, 1_003_000, RevocationReason::Decommissioned).unwrap();
    state.add_revocation(revocation).unwrap();

    assert_eq!(
        evaluate_trust(&state, device_b_id, 1_003_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS),
        TrustDecision::Refused(RefusalReason::Revoked)
    );

    // T4: Device C goes silent for 91 days. Active set should drop C.
    let way_later = 1_003_000 + 91 * 24 * 60 * 60;
    state.add_liveness(LivenessCert::sign(&device_a_sk, state.owner_id, way_later).unwrap()).unwrap();
    let active_now = state.active_devices(way_later, DEFAULT_ACTIVE_WINDOW_SECS);
    assert_eq!(active_now, vec![device_a_id]);
}

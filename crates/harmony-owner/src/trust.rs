use crate::state::OwnerState;
use std::collections::HashSet;

pub const DEFAULT_ACTIVE_WINDOW_SECS: u64 = 90 * 24 * 60 * 60;
pub const DEFAULT_FRESHNESS_WINDOW_SECS: u64 = 30 * 24 * 60 * 60;
pub const N_VOUCH_THRESHOLD_V1: usize = 1;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrustDecision {
    Full,
    Provisional,
    Refused(RefusalReason),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalReason {
    NotEnrolled,
    Revoked,
    StaleTrustState,
    ChallengedBySibling,
}

pub fn evaluate_trust(
    state: &OwnerState,
    target: [u8; 16],
    now: u64,
    active_window_secs: u64,
    freshness_window_secs: u64,
) -> TrustDecision {
    if state.is_revoked(target) {
        return TrustDecision::Refused(RefusalReason::Revoked);
    }
    if !state.enrollments.contains_key(&target) {
        return TrustDecision::Refused(RefusalReason::NotEnrolled);
    }
    let active = state.active_devices(now, active_window_secs);
    let active_set: HashSet<_> = active.iter().copied().collect();

    // Freshness: scoped to ACTIVE siblings (which already excludes revoked
    // devices via active_devices). Unrelated traffic from non-active devices
    // does NOT satisfy freshness, narrowing the suppression-attack surface:
    // a stale state cannot be revived by an attacker replaying old certs from
    // long-departed devices.
    let cutoff = now.saturating_sub(freshness_window_secs);
    let any_fresh = state
        .liveness
        .values()
        .filter(|l| active_set.contains(&l.signer))
        .any(|l| l.timestamp >= cutoff)
        || state
            .vouching
            .iter()
            .filter(|v| active_set.contains(&v.signer))
            .any(|v| v.issued_at >= cutoff);
    if !any_fresh {
        return TrustDecision::Refused(RefusalReason::StaleTrustState);
    }

    // Single-device case
    if active_set.len() == 1 && active_set.contains(&target) {
        return TrustDecision::Full;
    }

    // Challenges from active siblings
    let challenged = state.vouching.challenges_against(target)
        .any(|c| active_set.contains(&c.signer) && c.signer != target);
    if challenged {
        return TrustDecision::Refused(RefusalReason::ChallengedBySibling);
    }

    // Vouches from active siblings (excluding target itself)
    let vouches = state.vouching.vouches_for(target)
        .filter(|v| active_set.contains(&v.signer) && v.signer != target)
        .count();

    if vouches >= N_VOUCH_THRESHOLD_V1 {
        TrustDecision::Full
    } else {
        TrustDecision::Provisional
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::{EnrollmentCert, LivenessCert, Stance, VouchingCert};
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

    fn enroll_via_master(state: &mut OwnerState, master_sk: &SigningKey, master_bundle: PubKeyBundle, device_bundle: PubKeyBundle, ts: u64) -> [u8; 16] {
        let device_id = device_bundle.identity_hash();
        let cert = EnrollmentCert::sign_master(master_sk, master_bundle, device_id, device_bundle, ts, None).unwrap();
        state.add_enrollment(cert, ts, DEFAULT_ACTIVE_WINDOW_SECS).unwrap();
        device_id
    }

    #[test]
    fn single_device_yields_full_trust() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (device_sk, device_bundle) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let device_id = enroll_via_master(&mut state, &master_sk, master_bundle, device_bundle, 1_000_000);

        let liveness = LivenessCert::sign(&device_sk, owner_id, 1_500_000).unwrap();
        state.add_liveness(liveness).unwrap();

        let decision = evaluate_trust(&state, device_id, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Full);
    }

    #[test]
    fn second_device_no_vouch_is_provisional() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let _id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, owner_id, 1_500_000).unwrap()).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Provisional);
    }

    #[test]
    fn one_vouch_yields_full_trust() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let _id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, owner_id, 1_500_000).unwrap()).unwrap();

        let vouch = VouchingCert::sign(&sk_a, owner_id, id_b, Stance::Vouch, 1_400_000).unwrap();
        state.add_vouching(vouch).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Full);
    }

    #[test]
    fn challenge_overrides_vouch() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let _id_a = enroll_via_master(&mut state, &master_sk, master_bundle.clone(), bundle_a.clone(), 1_000_000);
        let id_b = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_b.clone(), 1_000_001);

        state.add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_500_000).unwrap()).unwrap();
        state.add_liveness(LivenessCert::sign(&sk_b, owner_id, 1_500_000).unwrap()).unwrap();

        // Vouch first, then challenge from same signer
        state.add_vouching(VouchingCert::sign(&sk_a, owner_id, id_b, Stance::Vouch, 1_400_000).unwrap()).unwrap();
        state.add_vouching(VouchingCert::sign(&sk_a, owner_id, id_b, Stance::Challenge, 1_450_000).unwrap()).unwrap();

        let decision = evaluate_trust(&state, id_b, 1_500_000, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Refused(RefusalReason::ChallengedBySibling));
    }

    #[test]
    fn revoked_device_liveness_does_not_satisfy_freshness() {
        // A and B are both alive at 1_500_000; B is then revoked at 1_500_500.
        // Jump well past the freshness window relative to when A was last alive.
        // Since B's liveness predates the freshness cutoff AND B is no longer
        // active, B's liveness must NOT satisfy freshness for the family.
        // A's liveness is also stale → refusal expected.
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();
        let (sk_b, bundle_b) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(
            &mut state,
            &master_sk,
            master_bundle.clone(),
            bundle_a.clone(),
            1_000_000,
        );
        let id_b = enroll_via_master(
            &mut state,
            &master_sk,
            master_bundle,
            bundle_b.clone(),
            1_000_001,
        );

        state
            .add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_500_000).unwrap())
            .unwrap();
        state
            .add_liveness(LivenessCert::sign(&sk_b, owner_id, 1_500_000).unwrap())
            .unwrap();
        let rev = crate::certs::RevocationCert::sign_self(
            &sk_b,
            owner_id,
            id_b,
            1_500_500,
            crate::certs::RevocationReason::Compromised,
        )
        .unwrap();
        state.add_revocation(rev).unwrap();

        // Now jump WAY past freshness. A's liveness is stale; B's is also stale
        // (and B is revoked anyway). Freshness scoped to active siblings → no
        // active sibling has fresh signal → refuse with StaleTrustState.
        let now = 1_500_000 + DEFAULT_FRESHNESS_WINDOW_SECS + 1;
        let decision = evaluate_trust(
            &state,
            id_a,
            now,
            DEFAULT_ACTIVE_WINDOW_SECS,
            DEFAULT_FRESHNESS_WINDOW_SECS,
        );
        assert_eq!(decision, TrustDecision::Refused(RefusalReason::StaleTrustState));
    }

    #[test]
    fn stale_trust_state_refuses() {
        let (master_sk, master_bundle) = keypair_and_bundle();
        let owner_id = master_bundle.identity_hash();
        let (sk_a, bundle_a) = keypair_and_bundle();

        let mut state = OwnerState::new(owner_id);
        let id_a = enroll_via_master(&mut state, &master_sk, master_bundle, bundle_a.clone(), 1_000_000);
        state.add_liveness(LivenessCert::sign(&sk_a, owner_id, 1_000_001).unwrap()).unwrap();

        // `now` is far past the freshness window
        let now = 1_000_001 + DEFAULT_FRESHNESS_WINDOW_SECS + 1;
        let decision = evaluate_trust(&state, id_a, now, DEFAULT_ACTIVE_WINDOW_SECS, DEFAULT_FRESHNESS_WINDOW_SECS);
        assert_eq!(decision, TrustDecision::Refused(RefusalReason::StaleTrustState));
    }
}

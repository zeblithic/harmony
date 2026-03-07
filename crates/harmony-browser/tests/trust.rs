use harmony_browser::{TrustDecision, TrustPolicy};

#[test]
fn unknown_author_gets_unknown_decision() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(None), TrustDecision::Unknown);
}

#[test]
fn identity_0_is_untrusted() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0b00000000)), TrustDecision::Untrusted);
}

#[test]
fn identity_1_is_untrusted() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0b00000001)), TrustDecision::Untrusted);
}

#[test]
fn identity_2_is_preview() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0b00000010)), TrustDecision::Preview);
}

#[test]
fn identity_3_is_full_trust() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0b00000011)), TrustDecision::FullTrust);
}

#[test]
fn full_score_0xff_is_full_trust() {
    let policy = TrustPolicy::new();
    assert_eq!(policy.decide(Some(0xFF)), TrustDecision::FullTrust);
}

#[test]
fn custom_threshold_overrides_default() {
    let mut policy = TrustPolicy::new();
    policy.set_preview_threshold(3);
    assert_eq!(policy.decide(Some(0b00000010)), TrustDecision::Untrusted);
    assert_eq!(policy.decide(Some(0b00000011)), TrustDecision::Preview);
}

#[test]
fn set_full_trust_threshold_clamps_preview() {
    let mut policy = TrustPolicy::new();
    // Lowering full_trust below preview must clamp preview down.
    // full_trust=1, preview clamped to 0 → identity 0 gets Preview, 1+ gets FullTrust
    policy.set_full_trust_threshold(1);
    assert_eq!(policy.decide(Some(0b00000000)), TrustDecision::Preview);
    assert_eq!(policy.decide(Some(0b00000001)), TrustDecision::FullTrust);
}

//! Spam scoring engine.
//!
//! A pure scoring function with no I/O. Takes authentication results and
//! heuristic signals, returns a numeric score and verdict.

/// SPF (Sender Policy Framework) authentication result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpfResult {
    Pass,
    Fail,
    SoftFail,
    None,
}

/// DKIM (DomainKeys Identified Mail) authentication result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DkimResult {
    Pass,
    Fail,
    Missing,
}

/// DMARC (Domain-based Message Authentication, Reporting, and Conformance) result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DmarcResult {
    Pass,
    Fail,
    None,
}

/// All signals that contribute to spam scoring.
#[derive(Debug, Clone)]
pub struct SpamSignals {
    // Layer 1: Connection-level
    pub dnsbl_listed: bool,
    pub fcrdns_pass: bool,

    // Layer 2: Envelope-level
    pub spf_result: SpfResult,

    // Layer 3: Content-level
    pub dkim_result: DkimResult,
    pub dmarc_result: DmarcResult,
    pub has_executable_attachment: bool,
    pub url_count: usize,
    pub empty_subject: bool,

    // Layer 4: Harmony-native trust
    pub known_harmony_sender: bool,
    /// Gateway trust score from 0.0 to 1.0, or `None` for unknown gateways.
    pub gateway_trust: Option<f32>,
    pub first_contact: bool,
}

/// Action to take based on the spam score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpamAction {
    /// Score <= 0: trusted, deliver immediately.
    Deliver,
    /// Score 1-4: borderline, deliver with X-Spam-Score header.
    DeliverWithScore,
    /// Score >= 5: reject at SMTP level.
    Reject,
}

/// The result of scoring a message.
#[derive(Debug, Clone)]
pub struct SpamVerdict {
    pub score: i32,
    pub action: SpamAction,
}

/// Score a message based on all available signals.
///
/// This is a pure function with no I/O. The scoring table is:
///
/// **Layer 1 -- Connection:**
/// - DNSBL listed: +10
/// - FCrDNS fail: +3
///
/// **Layer 2 -- Envelope:**
/// - SPF Fail: +3
/// - SPF SoftFail: +1
/// - SPF None: +1
///
/// **Layer 3 -- Content:**
/// - DKIM Fail: +3
/// - DKIM Missing: +1
/// - DMARC Fail: +3
/// - Executable attachment: +5
/// - URL count > 10: +1
/// - Empty subject: +1
///
/// **Layer 4 -- Harmony trust:**
/// - Known Harmony sender: -3
/// - Gateway trust >= 0.8: -2
/// - Gateway trust < 0.3: +2
/// - First contact: +1
///
/// **Thresholds** (default `reject_threshold` = 5):
/// - score <= 0: `Deliver`
/// - score 1..(reject_threshold-1): `DeliverWithScore`
/// - score >= reject_threshold: `Reject`
pub fn score(signals: &SpamSignals, reject_threshold: i32) -> SpamVerdict {
    let mut s: i32 = 0;

    // Layer 1: Connection-level
    if signals.dnsbl_listed {
        s += 10;
    }
    if !signals.fcrdns_pass {
        s += 3;
    }

    // Layer 2: Envelope-level
    match signals.spf_result {
        SpfResult::Pass => {}
        SpfResult::Fail => s += 3,
        SpfResult::SoftFail => s += 1,
        SpfResult::None => s += 1,
    }

    // Layer 3: Content-level
    match signals.dkim_result {
        DkimResult::Pass => {}
        DkimResult::Fail => s += 3,
        DkimResult::Missing => s += 1,
    }
    match signals.dmarc_result {
        DmarcResult::Pass => {}
        DmarcResult::Fail => s += 3,
        DmarcResult::None => {}
    }
    if signals.has_executable_attachment {
        s += 5;
    }
    if signals.url_count > 10 {
        s += 1;
    }
    if signals.empty_subject {
        s += 1;
    }

    // Layer 4: Harmony-native trust
    if signals.known_harmony_sender {
        s -= 3;
    }
    if let Some(trust) = signals.gateway_trust {
        if trust >= 0.8 {
            s -= 2;
        } else if trust < 0.3 {
            s += 2;
        }
    }
    if signals.first_contact {
        s += 1;
    }

    let action = if s <= 0 {
        SpamAction::Deliver
    } else if s < reject_threshold {
        SpamAction::DeliverWithScore
    } else {
        SpamAction::Reject
    };

    SpamVerdict { score: s, action }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a "clean" set of signals where everything looks good.
    fn clean_signals() -> SpamSignals {
        SpamSignals {
            dnsbl_listed: false,
            fcrdns_pass: true,
            spf_result: SpfResult::Pass,
            dkim_result: DkimResult::Pass,
            dmarc_result: DmarcResult::Pass,
            has_executable_attachment: false,
            url_count: 0,
            empty_subject: false,
            known_harmony_sender: false,
            gateway_trust: None,
            first_contact: false,
        }
    }

    #[test]
    fn clean_message_passes() {
        let verdict = score(&clean_signals(), 5);
        assert!(verdict.score <= 0, "expected score <= 0, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Deliver);
    }

    #[test]
    fn dnsbl_listed_rejects() {
        let signals = SpamSignals {
            dnsbl_listed: true,
            ..clean_signals()
        };
        let verdict = score(&signals, 5);
        assert!(verdict.score >= 5, "expected score >= 5, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Reject);
    }

    #[test]
    fn harmony_sender_gets_trust_bonus() {
        // Known Harmony sender (-3) with high gateway trust (-2) should
        // offset moderate failures like SPF SoftFail (+1) and DKIM Missing (+1).
        let signals = SpamSignals {
            known_harmony_sender: true,
            gateway_trust: Some(0.9),
            spf_result: SpfResult::SoftFail,
            dkim_result: DkimResult::Missing,
            ..clean_signals()
        };
        let verdict = score(&signals, 5);
        // Expected: 0 (base) + 1 (SPF SoftFail) + 1 (DKIM Missing) - 3 (known) - 2 (trust) = -3
        assert!(verdict.score <= 0, "expected score <= 0, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Deliver);
    }

    #[test]
    fn executable_attachment_rejects() {
        let signals = SpamSignals {
            has_executable_attachment: true,
            ..clean_signals()
        };
        let verdict = score(&signals, 5);
        // Expected: 0 (base) + 5 (executable) = 5
        assert!(verdict.score >= 5, "expected score >= 5, got {}", verdict.score);
        assert_eq!(verdict.action, SpamAction::Reject);
    }

    #[test]
    fn custom_reject_threshold() {
        // Executable attachment (+5) is rejected at default threshold 5,
        // but should be borderline at threshold 10.
        let signals = SpamSignals {
            has_executable_attachment: true,
            ..clean_signals()
        };
        let verdict = score(&signals, 10);
        assert_eq!(verdict.score, 5);
        assert_eq!(verdict.action, SpamAction::DeliverWithScore);
    }

    #[test]
    fn borderline_message_delivers_with_header() {
        // Mixed signals that land in the 1-4 range.
        // SPF SoftFail (+1) + DKIM Missing (+1) + first contact (+1) = 3
        let signals = SpamSignals {
            spf_result: SpfResult::SoftFail,
            dkim_result: DkimResult::Missing,
            first_contact: true,
            ..clean_signals()
        };
        let verdict = score(&signals, 5);
        assert!(
            (1..=4).contains(&verdict.score),
            "expected score in 1..=4, got {}",
            verdict.score
        );
        assert_eq!(verdict.action, SpamAction::DeliverWithScore);
    }
}

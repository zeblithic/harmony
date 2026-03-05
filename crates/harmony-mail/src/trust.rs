//! Cross-gateway trust coordination over the Harmony network.
//!
//! Gateways announce themselves to the network, report trust metrics about
//! other gateways they interact with, and aggregate those reports into
//! composite trust scores used for spam filtering and relay decisions.

use std::collections::HashMap;

use harmony_identity::identity::{ADDRESS_HASH_LENGTH, SIGNATURE_LENGTH};
use harmony_identity::{Identity, PrivateIdentity};

// ── Capability bitfield values ──────────────────────────────────────────

/// Gateway can receive inbound SMTP mail.
pub const CAP_INBOUND: u16 = 0x0001;

/// Gateway can send outbound SMTP mail.
pub const CAP_OUTBOUND: u16 = 0x0002;

/// Gateway can relay mail between other gateways.
pub const CAP_RELAY: u16 = 0x0004;

// ── GatewayAnnounce ─────────────────────────────────────────────────────

/// A gateway's announce to the network.
#[derive(Debug, Clone)]
pub struct GatewayAnnounce {
    /// The DNS domain this gateway serves (e.g. "mail.example.com").
    pub domain: String,
    /// The gateway's 128-bit Harmony address hash.
    pub gateway_address: [u8; ADDRESS_HASH_LENGTH],
    /// SMTP host for inbound delivery (e.g. "smtp.example.com").
    pub smtp_host: String,
    /// Capability bitfield (see `CAP_*` constants).
    pub capabilities: u16,
    /// Number of users this gateway serves.
    pub user_count: u32,
    /// Unix timestamp when the gateway started.
    pub uptime_since: u64,
    /// Ed25519 signature over the canonical announce bytes.
    pub signature: [u8; SIGNATURE_LENGTH],
}

impl GatewayAnnounce {
    /// Create a new signed gateway announce.
    pub fn new(
        domain: String,
        gateway: &PrivateIdentity,
        smtp_host: String,
        capabilities: u16,
        user_count: u32,
        uptime_since: u64,
    ) -> Self {
        let gateway_address = gateway.public_identity().address_hash;
        let mut announce = Self {
            domain,
            gateway_address,
            smtp_host,
            capabilities,
            user_count,
            uptime_since,
            signature: [0u8; SIGNATURE_LENGTH],
        };
        let canonical = announce.canonical_bytes();
        announce.signature = gateway.sign(&canonical);
        announce
    }

    /// Verify the announce signature against the given gateway identity.
    pub fn verify(&self, gateway_identity: &Identity) -> bool {
        let canonical = self.canonical_bytes();
        gateway_identity.verify(&canonical, &self.signature).is_ok()
    }

    /// Build the canonical byte sequence of all fields (excluding signature).
    ///
    /// Strings are length-prefixed with a u16 big-endian length.
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // domain: length-prefixed string
        buf.extend_from_slice(&(self.domain.len() as u16).to_be_bytes());
        buf.extend_from_slice(self.domain.as_bytes());
        // gateway_address: 16 bytes
        buf.extend_from_slice(&self.gateway_address);
        // smtp_host: length-prefixed string
        buf.extend_from_slice(&(self.smtp_host.len() as u16).to_be_bytes());
        buf.extend_from_slice(self.smtp_host.as_bytes());
        // capabilities: 2 bytes big-endian
        buf.extend_from_slice(&self.capabilities.to_be_bytes());
        // user_count: 4 bytes big-endian
        buf.extend_from_slice(&self.user_count.to_be_bytes());
        // uptime_since: 8 bytes big-endian
        buf.extend_from_slice(&self.uptime_since.to_be_bytes());
        buf
    }
}

// ── TrustMetrics ────────────────────────────────────────────────────────

/// Metrics reported about a gateway's behaviour over a time period.
#[derive(Debug, Clone, Copy)]
pub struct TrustMetrics {
    /// Total messages received from the subject gateway.
    pub messages_received: u32,
    /// Fraction of messages classified as spam (0.0..=1.0).
    pub spam_ratio: f32,
    /// Fraction of messages that bounced (0.0..=1.0).
    pub bounce_ratio: f32,
    /// Fraction of messages passing DKIM verification (0.0..=1.0).
    pub dkim_pass_ratio: f32,
    /// Fraction of messages passing SPF verification (0.0..=1.0).
    pub spf_pass_ratio: f32,
    /// Availability ratio over the period (0.0..=1.0).
    pub availability: f32,
}

// ── TrustReport ─────────────────────────────────────────────────────────

/// A trust report from one gateway about another.
#[derive(Debug, Clone)]
pub struct TrustReport {
    /// Address hash of the gateway filing this report.
    pub reporter_address: [u8; ADDRESS_HASH_LENGTH],
    /// Address hash of the gateway being reported on.
    pub subject_address: [u8; ADDRESS_HASH_LENGTH],
    /// Domain of the subject gateway.
    pub subject_domain: String,
    /// Start of the reporting period (unix timestamp).
    pub period_start: u64,
    /// End of the reporting period (unix timestamp).
    pub period_end: u64,
    /// Observed metrics for the subject gateway.
    pub metrics: TrustMetrics,
    /// Ed25519 signature over the canonical report bytes.
    pub signature: [u8; SIGNATURE_LENGTH],
}

impl TrustReport {
    /// Create a new signed trust report.
    pub fn new(
        reporter: &PrivateIdentity,
        subject_address: [u8; ADDRESS_HASH_LENGTH],
        subject_domain: String,
        period: (u64, u64),
        metrics: TrustMetrics,
    ) -> Self {
        let reporter_address = reporter.public_identity().address_hash;
        let mut report = Self {
            reporter_address,
            subject_address,
            subject_domain,
            period_start: period.0,
            period_end: period.1,
            metrics,
            signature: [0u8; SIGNATURE_LENGTH],
        };
        let canonical = report.canonical_bytes();
        report.signature = reporter.sign(&canonical);
        report
    }

    /// Verify the report signature against the given reporter identity.
    pub fn verify(&self, reporter_identity: &Identity) -> bool {
        let canonical = self.canonical_bytes();
        reporter_identity
            .verify(&canonical, &self.signature)
            .is_ok()
    }

    /// Build the canonical byte sequence of all fields (excluding signature).
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // reporter_address: 16 bytes
        buf.extend_from_slice(&self.reporter_address);
        // subject_address: 16 bytes
        buf.extend_from_slice(&self.subject_address);
        // subject_domain: length-prefixed string
        buf.extend_from_slice(&(self.subject_domain.len() as u16).to_be_bytes());
        buf.extend_from_slice(self.subject_domain.as_bytes());
        // period_start: 8 bytes big-endian
        buf.extend_from_slice(&self.period_start.to_be_bytes());
        // period_end: 8 bytes big-endian
        buf.extend_from_slice(&self.period_end.to_be_bytes());
        // metrics (all fixed-size)
        buf.extend_from_slice(&self.metrics.messages_received.to_be_bytes());
        buf.extend_from_slice(&self.metrics.spam_ratio.to_be_bytes());
        buf.extend_from_slice(&self.metrics.bounce_ratio.to_be_bytes());
        buf.extend_from_slice(&self.metrics.dkim_pass_ratio.to_be_bytes());
        buf.extend_from_slice(&self.metrics.spf_pass_ratio.to_be_bytes());
        buf.extend_from_slice(&self.metrics.availability.to_be_bytes());
        buf
    }
}

// ── AggregatedTrust ─────────────────────────────────────────────────────

/// Aggregated trust score for a gateway, computed from multiple reports.
#[derive(Debug, Clone)]
pub struct AggregatedTrust {
    /// Address hash of the gateway being scored.
    pub gateway_address: [u8; ADDRESS_HASH_LENGTH],
    /// Weighted trust score (0.0..=1.0).
    pub trust_score: f32,
    /// Number of reports that contributed to the score.
    pub report_count: usize,
}

// ── Aggregation ─────────────────────────────────────────────────────────

/// Compute a trust score from raw metrics.
///
/// Formula: `1.0 - (spam * 0.35 + bounce * 0.25 + (1-dkim) * 0.15 + (1-spf) * 0.15 + (1-availability) * 0.10)`
/// Result is clamped to 0.0..=1.0.
fn metrics_to_score(m: &TrustMetrics) -> f32 {
    let raw = 1.0
        - (m.spam_ratio * 0.35
            + m.bounce_ratio * 0.25
            + (1.0 - m.dkim_pass_ratio) * 0.15
            + (1.0 - m.spf_pass_ratio) * 0.15
            + (1.0 - m.availability) * 0.10);
    raw.clamp(0.0, 1.0)
}

/// Aggregate trust reports into a single trust score using weighted median.
///
/// Each report's metrics are converted to a score using [`metrics_to_score`],
/// then the weighted median is computed using `reporter_weights`. Reports
/// from reporters not present in the weights map are ignored.
///
/// Weighted median is used instead of weighted mean for robustness: a single
/// malicious reporter cannot skew the aggregate even with extreme values.
///
/// Returns `None` if no reports match any reporter in the weights map.
pub fn aggregate_trust(
    reports: &[TrustReport],
    reporter_weights: &HashMap<[u8; ADDRESS_HASH_LENGTH], f32>,
) -> Option<AggregatedTrust> {
    let mut scored: Vec<(f32, f32)> = Vec::new(); // (score, weight)
    let mut gateway_address = None;

    for report in reports {
        if let Some(&weight) = reporter_weights.get(&report.reporter_address) {
            if gateway_address.is_none() {
                gateway_address = Some(report.subject_address);
            }
            scored.push((metrics_to_score(&report.metrics), weight));
        }
    }

    if scored.is_empty() {
        return None;
    }

    let report_count = scored.len();

    // Sort by score for weighted median computation.
    scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let total_weight: f32 = scored.iter().map(|(_, w)| w).sum();
    if total_weight == 0.0 {
        return None;
    }

    // Weighted median: find the score where cumulative weight reaches half.
    let half = total_weight / 2.0;
    let mut cumulative = 0.0_f32;
    let mut trust_score = scored[0].0;
    for &(score, weight) in &scored {
        cumulative += weight;
        if cumulative >= half {
            trust_score = score;
            break;
        }
    }

    Some(AggregatedTrust {
        gateway_address: gateway_address.unwrap(),
        trust_score: trust_score.clamp(0.0, 1.0),
        report_count,
    })
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn gateway_announce_sign_verify() {
        let gateway = PrivateIdentity::generate(&mut OsRng);

        let announce = GatewayAnnounce::new(
            "mail.example.com".to_string(),
            &gateway,
            "smtp.example.com".to_string(),
            CAP_INBOUND | CAP_OUTBOUND,
            500,
            1_700_000_000,
        );

        // Signature should verify against the gateway's public identity.
        assert!(announce.verify(gateway.public_identity()));

        // Signature should NOT verify against a different identity.
        let other = PrivateIdentity::generate(&mut OsRng);
        assert!(!announce.verify(other.public_identity()));

        // Gateway address should match.
        assert_eq!(announce.gateway_address, gateway.public_identity().address_hash);
    }

    #[test]
    fn trust_report_sign_verify() {
        let reporter = PrivateIdentity::generate(&mut OsRng);
        let subject = PrivateIdentity::generate(&mut OsRng);

        let metrics = TrustMetrics {
            messages_received: 1000,
            spam_ratio: 0.02,
            bounce_ratio: 0.01,
            dkim_pass_ratio: 0.98,
            spf_pass_ratio: 0.99,
            availability: 0.999,
        };

        let report = TrustReport::new(
            &reporter,
            subject.public_identity().address_hash,
            "peer.example.com".to_string(),
            (1_700_000_000, 1_700_086_400),
            metrics,
        );

        // Signature should verify against the reporter's public identity.
        assert!(report.verify(reporter.public_identity()));

        // Signature should NOT verify against the subject's identity.
        assert!(!report.verify(subject.public_identity()));

        // Reporter address should match.
        assert_eq!(report.reporter_address, reporter.public_identity().address_hash);
    }

    #[test]
    fn aggregate_trust_single_report() {
        let reporter = PrivateIdentity::generate(&mut OsRng);
        let subject = PrivateIdentity::generate(&mut OsRng);

        let metrics = TrustMetrics {
            messages_received: 100,
            spam_ratio: 0.1,       // contributes 0.1 * 0.35 = 0.035
            bounce_ratio: 0.05,    // contributes 0.05 * 0.25 = 0.0125
            dkim_pass_ratio: 0.95, // contributes (1-0.95) * 0.15 = 0.0075
            spf_pass_ratio: 0.90,  // contributes (1-0.90) * 0.15 = 0.015
            availability: 0.99,    // contributes (1-0.99) * 0.10 = 0.001
        };
        // Expected score = 1.0 - (0.035 + 0.0125 + 0.0075 + 0.015 + 0.001) = 1.0 - 0.071 = 0.929

        let report = TrustReport::new(
            &reporter,
            subject.public_identity().address_hash,
            "peer.example.com".to_string(),
            (1_700_000_000, 1_700_086_400),
            metrics,
        );

        let mut weights = HashMap::new();
        weights.insert(reporter.public_identity().address_hash, 1.0);

        let result = aggregate_trust(&[report], &weights).unwrap();
        assert_eq!(result.report_count, 1);
        assert_eq!(result.gateway_address, subject.public_identity().address_hash);

        let expected = 0.929_f32;
        assert!(
            (result.trust_score - expected).abs() < 1e-5,
            "expected ~{}, got {}",
            expected,
            result.trust_score
        );
    }

    #[test]
    fn aggregate_trust_weighted() {
        let reporter_a = PrivateIdentity::generate(&mut OsRng);
        let reporter_b = PrivateIdentity::generate(&mut OsRng);
        let subject = PrivateIdentity::generate(&mut OsRng);

        // Reporter A: good metrics -> high score
        let metrics_a = TrustMetrics {
            messages_received: 500,
            spam_ratio: 0.0,
            bounce_ratio: 0.0,
            dkim_pass_ratio: 1.0,
            spf_pass_ratio: 1.0,
            availability: 1.0,
        };
        // Score A = 1.0

        // Reporter B: bad metrics -> low score
        let metrics_b = TrustMetrics {
            messages_received: 200,
            spam_ratio: 0.5,
            bounce_ratio: 0.5,
            dkim_pass_ratio: 0.0,
            spf_pass_ratio: 0.0,
            availability: 0.5,
        };
        // Score B = 1.0 - (0.5*0.35 + 0.5*0.25 + 1.0*0.15 + 1.0*0.15 + 0.5*0.10)
        //         = 1.0 - 0.65 = 0.35

        let report_a = TrustReport::new(
            &reporter_a,
            subject.public_identity().address_hash,
            "peer.example.com".to_string(),
            (1_700_000_000, 1_700_086_400),
            metrics_a,
        );
        let report_b = TrustReport::new(
            &reporter_b,
            subject.public_identity().address_hash,
            "peer.example.com".to_string(),
            (1_700_000_000, 1_700_086_400),
            metrics_b,
        );

        let mut weights = HashMap::new();
        // Reporter A has weight 0.8, Reporter B has weight 0.2
        weights.insert(reporter_a.public_identity().address_hash, 0.8);
        weights.insert(reporter_b.public_identity().address_hash, 0.2);

        let result = aggregate_trust(&[report_a, report_b], &weights).unwrap();
        assert_eq!(result.report_count, 2);

        // Weighted median: sorted by score [(0.35, w=0.2), (1.0, w=0.8)].
        // Cumulative at 0.35: 0.2 < 0.5, at 1.0: 1.0 >= 0.5.
        // Median = 1.0 (the high-weight good reporter dominates).
        let expected = 1.0_f32;
        assert!(
            (result.trust_score - expected).abs() < 1e-5,
            "expected ~{}, got {}",
            expected,
            result.trust_score
        );
    }
}

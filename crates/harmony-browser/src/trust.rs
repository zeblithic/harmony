use crate::TrustDecision;

/// Maps trust scores to rendering decisions.
///
/// Uses the identity dimension (bits 0-1) of the 8-bit trust score
/// to determine how content should be rendered. This matches the
/// `resolveMediaTrust` logic in harmony-client's TrustGraphService.
pub struct TrustPolicy {
    /// Minimum identity value for Preview (default: 2).
    preview_threshold: u8,
    /// Minimum identity value for FullTrust (default: 3).
    full_trust_threshold: u8,
}

impl TrustPolicy {
    pub fn new() -> Self {
        Self {
            preview_threshold: 2,
            full_trust_threshold: 3,
        }
    }

    /// Decide how to render content based on the author's trust score.
    ///
    /// - `None` → `Unknown` (author not in trust network)
    /// - Identity 0-1 → `Untrusted`
    /// - Identity 2 → `Preview`
    /// - Identity 3 → `FullTrust`
    pub fn decide(&self, score: Option<u8>) -> TrustDecision {
        let score = match score {
            Some(s) => s,
            None => return TrustDecision::Unknown,
        };
        let identity = score & 0x3;
        if identity >= self.full_trust_threshold {
            TrustDecision::FullTrust
        } else if identity >= self.preview_threshold {
            TrustDecision::Preview
        } else {
            TrustDecision::Untrusted
        }
    }

    pub fn set_preview_threshold(&mut self, threshold: u8) {
        self.preview_threshold = threshold;
        // Ensure full trust always requires strictly more than preview.
        if self.full_trust_threshold <= threshold {
            self.full_trust_threshold = threshold.saturating_add(1);
        }
    }

    pub fn set_full_trust_threshold(&mut self, threshold: u8) {
        self.full_trust_threshold = threshold;
    }
}

impl Default for TrustPolicy {
    fn default() -> Self {
        Self::new()
    }
}

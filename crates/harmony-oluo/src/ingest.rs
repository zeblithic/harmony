// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Ingest gate — Jain's privacy decision interface for incoming content.

use harmony_semantic::metadata::SidecarMetadata;

/// Jain's decision on whether and how to index incoming content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IngestDecision {
    /// Index with full persistence and all tiers.
    IndexFull,
    /// Index with limited persistence (lightweight entry, TTL-bounded).
    IndexLightweight { ttl_secs: u64 },
    /// Do not index this content.
    Reject,
}

/// The ingest gate trait — implemented by Jain, consumed by Oluo.
///
/// Before indexing content, Oluo presents the sidecar metadata to
/// the gate for a privacy decision. The gate may inspect privacy tier,
/// content type, source device, etc. to determine the index policy.
pub trait IngestGate {
    /// Decide whether and how to index content with the given metadata.
    fn decide(&self, metadata: &SidecarMetadata) -> IngestDecision;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ingest_decision_variants_exist() {
        let full = IngestDecision::IndexFull;
        assert_eq!(full, IngestDecision::IndexFull);

        let lightweight = IngestDecision::IndexLightweight { ttl_secs: 3600 };
        assert_eq!(
            lightweight,
            IngestDecision::IndexLightweight { ttl_secs: 3600 }
        );
        assert_ne!(
            lightweight,
            IngestDecision::IndexLightweight { ttl_secs: 7200 }
        );

        let reject = IngestDecision::Reject;
        assert_eq!(reject, IngestDecision::Reject);

        // All three variants are distinct.
        assert_ne!(full, reject);
        assert_ne!(full, IngestDecision::IndexLightweight { ttl_secs: 0 });
        assert_ne!(reject, IngestDecision::IndexLightweight { ttl_secs: 0 });
    }
}

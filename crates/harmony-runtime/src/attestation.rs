// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Trust tier model for multi-platform attestation.

/// Report of a node's platform attestation status.
#[derive(Debug, Clone)]
pub struct AttestationReport {
    /// Trust tier based on key binding and platform sovereignty.
    pub tier: AttestationTier,
    /// Optional platform-specific attestation evidence (TPM quote,
    /// Secure Enclave attestation, etc.). Opaque bytes.
    pub evidence: Option<Vec<u8>>,
}

/// Trust tier reflecting how strongly a node's keys are bound to hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationTier {
    /// Hardware-rooted keys on a sovereign OS (e.g., harmony-os + TPM).
    Sovereign,
    /// Hardware-rooted keys but vendor-controlled OS (e.g., Windows TPM).
    HardwareBound,
    /// No hardware attestation. Trust via behavioral scoring only.
    Unattested,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn attestation_tier_variants_exist() {
        let _ = AttestationTier::Sovereign;
        let _ = AttestationTier::HardwareBound;
        let _ = AttestationTier::Unattested;
    }

    #[test]
    fn attestation_report_without_evidence() {
        let report = AttestationReport {
            tier: AttestationTier::Unattested,
            evidence: None,
        };
        assert!(report.evidence.is_none());
    }

    #[test]
    fn attestation_report_with_evidence() {
        let evidence = vec![0x01, 0x02, 0x03];
        let report = AttestationReport {
            tier: AttestationTier::Sovereign,
            evidence: Some(evidence.clone()),
        };
        assert_eq!(report.evidence.unwrap(), evidence);
    }

    #[test]
    fn attestation_tier_debug_impl() {
        let tier = AttestationTier::HardwareBound;
        let _ = format!("{:?}", tier);
    }
}

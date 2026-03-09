// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Program descriptor — metadata about a deployed Kitri program.

use alloc::string::String;

use crate::retry::RetryPolicy;
use crate::trust::{CapabilitySet, TrustTier};

/// Metadata describing a deployed Kitri program.
#[derive(Debug, Clone)]
pub struct KitriProgram {
    pub cid: [u8; 32],
    pub name: String,
    pub version: String,
    pub trust_tier: TrustTier,
    pub capabilities: CapabilitySet,
    pub retry_policy: RetryPolicy,
    pub prefer_native: bool,
}

/// Compute a deterministic workflow ID: `BLAKE3(program_cid || input)`.
///
/// Same program + same input = same WorkflowId. This enables exactly-once
/// semantics: duplicate submissions return the cached result.
pub fn kitri_workflow_id(program_cid: &[u8; 32], input: &[u8]) -> [u8; 32] {
    use harmony_crypto::hash::blake3_hash;

    let mut preimage = alloc::vec::Vec::with_capacity(32 + input.len());
    preimage.extend_from_slice(program_cid);
    preimage.extend_from_slice(input);
    blake3_hash(&preimage)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retry::RetryPolicy;
    use crate::trust::{CapabilitySet, TrustTier};

    #[test]
    fn program_descriptor_creation() {
        let desc = KitriProgram {
            cid: [0xAA; 32],
            name: "shipment-verifier".into(),
            version: "0.1.0".into(),
            trust_tier: TrustTier::Owner,
            capabilities: CapabilitySet::new(),
            retry_policy: RetryPolicy::default(),
            prefer_native: true,
        };
        assert_eq!(desc.name, "shipment-verifier");
        assert!(desc.prefer_native);
    }

    #[test]
    fn workflow_id_deterministic() {
        let id1 = kitri_workflow_id(&[0xAA; 32], &[1, 2, 3]);
        let id2 = kitri_workflow_id(&[0xAA; 32], &[1, 2, 3]);
        assert_eq!(id1, id2);

        // Different input -> different id.
        let id3 = kitri_workflow_id(&[0xAA; 32], &[4, 5, 6]);
        assert_ne!(id1, id3);

        // Different program -> different id.
        let id4 = kitri_workflow_id(&[0xBB; 32], &[1, 2, 3]);
        assert_ne!(id1, id4);
    }
}

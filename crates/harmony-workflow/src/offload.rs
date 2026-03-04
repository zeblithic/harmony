/// Hints the caller can provide about the nature of a computation,
/// enabling the offload evaluator to make placement decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComputeHint {
    /// Prefer running on the local node even if peers are available.
    PreferLocal,
    /// Prefer offloading to a more powerful peer if one is reachable.
    PreferPowerful,
    /// Latency matters more than throughput; pick the fastest path.
    LatencySensitive,
    /// The result must be durably stored; pick a node with persistence.
    DurabilityRequired,
}

/// The decision made by the offload evaluator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffloadDecision {
    /// Execute the computation on this node.
    ExecuteLocally,
    /// Offload the computation to a specific peer identified by address.
    OffloadTo { peer: [u8; 16] },
}

/// Stub: always returns `ExecuteLocally`.
///
/// A real implementation would inspect peer capabilities, latency estimates,
/// and the hint to make an informed placement decision.
pub fn evaluate_offload(_hint: ComputeHint) -> OffloadDecision {
    OffloadDecision::ExecuteLocally
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_hint_variants_exist() {
        // Verify all variants can be constructed and compared.
        let hints = [
            ComputeHint::PreferLocal,
            ComputeHint::PreferPowerful,
            ComputeHint::LatencySensitive,
            ComputeHint::DurabilityRequired,
        ];
        for (i, a) in hints.iter().enumerate() {
            for (j, b) in hints.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn offload_decision_always_local() {
        // The stub always returns ExecuteLocally regardless of hint.
        assert_eq!(
            evaluate_offload(ComputeHint::PreferLocal),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::PreferPowerful),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::LatencySensitive),
            OffloadDecision::ExecuteLocally
        );
        assert_eq!(
            evaluate_offload(ComputeHint::DurabilityRequired),
            OffloadDecision::ExecuteLocally
        );
    }
}

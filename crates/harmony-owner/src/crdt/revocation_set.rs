use crate::certs::RevocationCert;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Strict Remove-Wins / monotonic add-only revocation set. Once a target
/// is in the set, no subsequent cert can remove it. The earliest-timestamp
/// revocation cert wins per target (so replays of older state don't lose
/// information about already-known revocations).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(from = "Vec<RevocationCert>", into = "Vec<RevocationCert>")]
pub struct RevocationSet {
    /// target -> earliest revocation cert seen
    cells: HashMap<[u8; 16], RevocationCert>,
}

impl From<Vec<RevocationCert>> for RevocationSet {
    fn from(certs: Vec<RevocationCert>) -> Self {
        let mut set = RevocationSet::default();
        for cert in certs {
            set.insert(cert);
        }
        set
    }
}

impl From<RevocationSet> for Vec<RevocationCert> {
    fn from(set: RevocationSet) -> Self {
        // Deterministic order: canonical CBOR sorts map entries but preserves
        // array order, so we must sort here. Sort by `target` (the cell key —
        // each target has at most one cert) for a stable wire format.
        let mut certs: Vec<_> = set.cells.into_values().collect();
        certs.sort_unstable_by_key(|cert| cert.target);
        certs
    }
}

impl RevocationSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, cert: RevocationCert) {
        let should_replace = match self.cells.get(&cert.target) {
            None => true,
            Some(existing) if existing.issued_at > cert.issued_at => true,
            Some(existing) if existing.issued_at == cert.issued_at => {
                // Deterministic tie-break: keep the lex-greater signature.
                cert.signature > existing.signature
            }
            _ => false,
        };
        if should_replace {
            self.cells.insert(cert.target, cert);
        }
    }

    pub fn merge(&mut self, other: RevocationSet) {
        for cert in other.cells.into_values() {
            self.insert(cert);
        }
    }

    pub fn is_revoked(&self, target: [u8; 16]) -> bool {
        self.cells.contains_key(&target)
    }

    pub fn cert_for(&self, target: [u8; 16]) -> Option<&RevocationCert> {
        self.cells.get(&target)
    }

    pub fn iter(&self) -> impl Iterator<Item = &RevocationCert> {
        self.cells.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::certs::RevocationReason;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn make_self_revocation(target: [u8; 16], ts: u64) -> RevocationCert {
        let sk = SigningKey::generate(&mut OsRng);
        RevocationCert::sign_self(&sk, [0u8; 16], target, ts, RevocationReason::Decommissioned).unwrap()
    }

    #[test]
    fn revocation_is_monotonic() {
        let mut set = RevocationSet::new();
        let target = [9u8; 16];

        set.insert(make_self_revocation(target, 100));
        assert!(set.is_revoked(target));

        // Idempotent
        set.insert(make_self_revocation(target, 200));
        assert!(set.is_revoked(target));
    }

    #[test]
    fn earliest_revocation_wins() {
        let mut set = RevocationSet::new();
        let target = [9u8; 16];

        let earlier = make_self_revocation(target, 100);
        let later = make_self_revocation(target, 200);

        // Insert later first, then earlier
        set.insert(later);
        set.insert(earlier);

        assert_eq!(set.cert_for(target).unwrap().issued_at, 100);
    }

    #[test]
    fn merge_preserves_revocations_from_both() {
        let target_a = [1u8; 16];
        let target_b = [2u8; 16];

        let mut s1 = RevocationSet::new();
        s1.insert(make_self_revocation(target_a, 100));

        let mut s2 = RevocationSet::new();
        s2.insert(make_self_revocation(target_b, 200));

        s1.merge(s2);
        assert!(s1.is_revoked(target_a));
        assert!(s1.is_revoked(target_b));
    }

    #[test]
    fn serialization_is_deterministic_across_insertion_orders() {
        // Bots flagged: HashMap iteration order randomizes the wire format.
        // The From<RevocationSet> for Vec<RevocationCert> impl now sorts by
        // target so equivalent sets produce byte-identical serialization.
        use crate::cbor;

        let mut targets = [[0u8; 16]; 5];
        for (i, t) in targets.iter_mut().enumerate() {
            *t = [(i as u8) * 17 + 1; 16];
        }
        let certs: Vec<_> = targets
            .iter()
            .map(|&t| make_self_revocation(t, 100))
            .collect();

        // Set A: insert forwards
        let mut set_a = RevocationSet::new();
        for c in certs.iter().cloned() {
            set_a.insert(c);
        }
        // Set B: insert backwards
        let mut set_b = RevocationSet::new();
        for c in certs.iter().rev().cloned() {
            set_b.insert(c);
        }

        let bytes_a = cbor::to_canonical(&set_a).unwrap();
        let bytes_b = cbor::to_canonical(&set_b).unwrap();
        assert_eq!(
            bytes_a, bytes_b,
            "equivalent RevocationSets with different insertion orders must \
             serialize to identical canonical CBOR"
        );
    }

    #[test]
    fn equal_timestamp_revocation_tie_breaks_deterministically() {
        // Same signing key + target + issued_at, but DIFFERENT reasons →
        // different signed payloads → different signatures even under
        // deterministic Ed25519. Both certs land in the same cell (keyed by
        // target only) and the tie-break decides which wins.
        let sk = SigningKey::generate(&mut OsRng);
        let target = [9u8; 16];

        let r_decom =
            RevocationCert::sign_self(&sk, [0u8; 16], target, 100, RevocationReason::Decommissioned)
                .unwrap();
        let r_lost = RevocationCert::sign_self(&sk, [0u8; 16], target, 100, RevocationReason::Lost)
            .unwrap();
        assert_ne!(
            r_decom.signature, r_lost.signature,
            "different reasons must produce different signatures"
        );

        let (lower, higher) = if r_decom.signature < r_lost.signature {
            (r_decom.clone(), r_lost.clone())
        } else {
            (r_lost.clone(), r_decom.clone())
        };

        // Order A: lower then higher → tie-break should replace lower with higher
        let mut s_a = RevocationSet::new();
        s_a.insert(lower.clone());
        s_a.insert(higher.clone());

        // Order B: higher then lower → tie-break should keep higher
        let mut s_b = RevocationSet::new();
        s_b.insert(higher.clone());
        s_b.insert(lower.clone());

        // Both replicas converge on `higher`.
        assert_eq!(
            s_a.cert_for(target).unwrap().signature,
            higher.signature
        );
        assert_eq!(
            s_b.cert_for(target).unwrap().signature,
            higher.signature
        );
    }
}

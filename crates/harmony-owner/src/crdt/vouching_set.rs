use crate::certs::{Stance, VouchingCert};
use std::collections::HashMap;

/// LWW-per-cell CRDT keyed by `(signer, target)`. Newer entries from the
/// same signer supersede older ones; signers cannot override each other.
#[derive(Debug, Clone, Default)]
pub struct VouchingSet {
    cells: HashMap<([u8; 16], [u8; 16]), VouchingCert>,
}

impl VouchingSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a cert. If a cert from the same signer about the same target
    /// already exists with a newer-or-equal timestamp, this is a no-op.
    pub fn insert(&mut self, cert: VouchingCert) {
        let key = (cert.signer, cert.target);
        let should_replace = match self.cells.get(&key) {
            None => true,
            Some(existing) if existing.issued_at < cert.issued_at => true,
            Some(existing) if existing.issued_at == cert.issued_at => {
                // Deterministic tie-break: keep the lex-greater signature.
                cert.signature > existing.signature
            }
            _ => false,
        };
        if should_replace {
            self.cells.insert(key, cert);
        }
    }

    /// Merge another VouchingSet into this one, applying LWW per cell.
    pub fn merge(&mut self, other: VouchingSet) {
        for cert in other.cells.into_values() {
            self.insert(cert);
        }
    }

    /// Vouches for target from active signers (filter applied externally).
    pub fn vouches_for(&self, target: [u8; 16]) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values().filter(move |c| c.target == target && c.stance == Stance::Vouch)
    }

    pub fn challenges_against(&self, target: [u8; 16]) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values().filter(move |c| c.target == target && c.stance == Stance::Challenge)
    }

    pub fn iter(&self) -> impl Iterator<Item = &VouchingCert> {
        self.cells.values()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn make_cert(signer: [u8; 16], target: [u8; 16], stance: Stance, ts: u64) -> VouchingCert {
        let sk = SigningKey::generate(&mut OsRng);
        VouchingCert::sign(&sk, [0u8; 16], signer, target, stance, ts).unwrap()
    }

    #[test]
    fn lww_per_cell() {
        let mut set = VouchingSet::new();
        let signer = [1u8; 16];
        let target = [2u8; 16];

        // Older Vouch
        set.insert(make_cert(signer, target, Stance::Vouch, 100));
        assert_eq!(set.vouches_for(target).count(), 1);

        // Newer Challenge from same signer — supersedes
        set.insert(make_cert(signer, target, Stance::Challenge, 200));
        assert_eq!(set.vouches_for(target).count(), 0);
        assert_eq!(set.challenges_against(target).count(), 1);

        // Older Vouch from same signer — no-op
        set.insert(make_cert(signer, target, Stance::Vouch, 150));
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn signers_cannot_override_each_other() {
        let mut set = VouchingSet::new();
        let target = [9u8; 16];
        let signer_a = [1u8; 16];
        let signer_b = [2u8; 16];

        set.insert(make_cert(signer_a, target, Stance::Vouch, 100));
        set.insert(make_cert(signer_b, target, Stance::Challenge, 200));

        assert_eq!(set.vouches_for(target).count(), 1);
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn merge_converges() {
        let target = [9u8; 16];
        let signer_a = [1u8; 16];
        let signer_b = [2u8; 16];

        let mut set1 = VouchingSet::new();
        set1.insert(make_cert(signer_a, target, Stance::Vouch, 100));

        let mut set2 = VouchingSet::new();
        set2.insert(make_cert(signer_b, target, Stance::Vouch, 200));

        set1.merge(set2);
        assert_eq!(set1.vouches_for(target).count(), 2);
    }

    #[test]
    fn equal_timestamp_tie_breaks_deterministically() {
        let signer = [1u8; 16];
        let target = [2u8; 16];

        // Two certs with same timestamp but different signatures (different signing keys produce different sigs)
        let c1 = make_cert(signer, target, Stance::Vouch, 100);
        let c2 = make_cert(signer, target, Stance::Vouch, 100);
        assert_ne!(c1.signature, c2.signature, "different signing keys should produce different sigs");

        let (lower, higher) = if c1.signature < c2.signature { (c1, c2) } else { (c2, c1) };

        // Order A: insert lower first, then higher
        let mut set_a = VouchingSet::new();
        set_a.insert(lower.clone());
        set_a.insert(higher.clone());

        // Order B: insert higher first, then lower
        let mut set_b = VouchingSet::new();
        set_b.insert(higher.clone());
        set_b.insert(lower.clone());

        // Both should converge on the higher-signature cert
        let a_winner = set_a.iter().next().unwrap();
        let b_winner = set_b.iter().next().unwrap();
        assert_eq!(a_winner.signature, b_winner.signature);
        assert_eq!(a_winner.signature, higher.signature);
    }
}

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
    use crate::pubkey_bundle::PubKeyBundle;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    /// Helper that signs a cert with the supplied key and returns the
    /// derived signer-id alongside the cert (so callers can predict the
    /// CRDT key without re-deriving).
    fn make_cert_with_key(sk: &SigningKey, target: [u8; 16], stance: Stance, ts: u64) -> ([u8; 16], VouchingCert) {
        let signer = PubKeyBundle::classical_only(sk.verifying_key().to_bytes()).identity_hash();
        let cert = VouchingCert::sign(sk, [0u8; 16], target, stance, ts).unwrap();
        (signer, cert)
    }

    #[test]
    fn lww_per_cell() {
        let mut set = VouchingSet::new();
        let sk = SigningKey::generate(&mut OsRng);
        let target = [2u8; 16];

        // Older Vouch
        let (_, c1) = make_cert_with_key(&sk, target, Stance::Vouch, 100);
        set.insert(c1);
        assert_eq!(set.vouches_for(target).count(), 1);

        // Newer Challenge from same signer — supersedes
        let (_, c2) = make_cert_with_key(&sk, target, Stance::Challenge, 200);
        set.insert(c2);
        assert_eq!(set.vouches_for(target).count(), 0);
        assert_eq!(set.challenges_against(target).count(), 1);

        // Older Vouch from same signer — no-op
        let (_, c3) = make_cert_with_key(&sk, target, Stance::Vouch, 150);
        set.insert(c3);
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn signers_cannot_override_each_other() {
        let mut set = VouchingSet::new();
        let target = [9u8; 16];
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);

        let (_, ca) = make_cert_with_key(&sk_a, target, Stance::Vouch, 100);
        let (_, cb) = make_cert_with_key(&sk_b, target, Stance::Challenge, 200);
        set.insert(ca);
        set.insert(cb);

        assert_eq!(set.vouches_for(target).count(), 1);
        assert_eq!(set.challenges_against(target).count(), 1);
    }

    #[test]
    fn merge_converges() {
        let target = [9u8; 16];
        let sk_a = SigningKey::generate(&mut OsRng);
        let sk_b = SigningKey::generate(&mut OsRng);

        let mut set1 = VouchingSet::new();
        let (_, ca) = make_cert_with_key(&sk_a, target, Stance::Vouch, 100);
        set1.insert(ca);

        let mut set2 = VouchingSet::new();
        let (_, cb) = make_cert_with_key(&sk_b, target, Stance::Vouch, 200);
        set2.insert(cb);

        set1.merge(set2);
        assert_eq!(set1.vouches_for(target).count(), 2);
    }

    #[test]
    fn equal_timestamp_tie_breaks_deterministically() {
        // Same signer + target + issued_at → same cell key. Different stance
        // produces a different signed payload (stance is part of the signing
        // payload), which produces a different signature even under
        // deterministic Ed25519. Both certs land in the same cell — tie-break
        // decides which wins.
        let signer_sk = SigningKey::generate(&mut OsRng);
        let target = [9u8; 16];

        let c_vouch = VouchingCert::sign(&signer_sk, [1u8; 16], target, Stance::Vouch, 100).unwrap();
        let c_challenge =
            VouchingCert::sign(&signer_sk, [1u8; 16], target, Stance::Challenge, 100).unwrap();
        assert_ne!(
            c_vouch.signature, c_challenge.signature,
            "different stances must produce different signatures"
        );
        // Sanity: same cell key (signer, target).
        assert_eq!(c_vouch.signer, c_challenge.signer);
        assert_eq!(c_vouch.target, c_challenge.target);
        assert_eq!(c_vouch.issued_at, c_challenge.issued_at);

        let (lower, higher) = if c_vouch.signature < c_challenge.signature {
            (c_vouch.clone(), c_challenge.clone())
        } else {
            (c_challenge.clone(), c_vouch.clone())
        };

        // Order A: lower then higher → tie-break should replace lower with higher
        let mut s_a = VouchingSet::new();
        s_a.insert(lower.clone());
        s_a.insert(higher.clone());

        // Order B: higher then lower → tie-break should keep higher (lower not
        // inserted because cert.signature > existing.signature is false).
        let mut s_b = VouchingSet::new();
        s_b.insert(higher.clone());
        s_b.insert(lower.clone());

        // Both replicas converge on `higher`.
        let a_winner = s_a.iter().next().unwrap();
        let b_winner = s_b.iter().next().unwrap();
        assert_eq!(a_winner.signature, higher.signature);
        assert_eq!(b_winner.signature, higher.signature);
    }
}

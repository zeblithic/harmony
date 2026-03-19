use harmony_identity::IdentityHash;

use crate::score::TrustScore;
use crate::store::TrustStore;

pub trait TrustLookup {
    fn score_for(&self, trustee: &IdentityHash) -> TrustScore;
}

impl TrustLookup for TrustStore {
    fn score_for(&self, trustee: &IdentityHash) -> TrustScore {
        self.effective_score(trustee)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;
    use crate::store::TrustStore;

    const LOCAL: [u8; 16] = [0x01; 16];
    const ALICE: [u8; 16] = [0xAA; 16];

    #[test]
    fn trust_store_implements_lookup() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, TrustScore::from_dimensions(3, 3, 2, 1), 1000);
        let lookup: &dyn TrustLookup = &store;
        let score = lookup.score_for(&ALICE);
        assert_eq!(score.identity(), 3);
        assert_eq!(score.endorsement(), 1);
    }

    #[test]
    fn lookup_unknown_returns_zero() {
        let store = TrustStore::new(LOCAL);
        let lookup: &dyn TrustLookup = &store;
        assert_eq!(lookup.score_for(&ALICE), TrustScore::UNKNOWN);
    }
}

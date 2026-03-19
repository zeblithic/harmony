use crate::score::TrustScore;
use harmony_identity::IdentityHash;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustEdge {
    pub truster: IdentityHash,
    pub trustee: IdentityHash,
    pub score: TrustScore,
    pub updated_at: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;

    #[test]
    fn edge_creation() {
        let edge = TrustEdge {
            truster: [0xAA; 16],
            trustee: [0xBB; 16],
            score: TrustScore::from_dimensions(3, 3, 2, 1),
            updated_at: 1000,
        };
        assert_eq!(edge.truster, [0xAA; 16]);
        assert_eq!(edge.score.identity(), 3);
    }

    #[test]
    fn edge_equality() {
        let a = TrustEdge {
            truster: [0x01; 16],
            trustee: [0x02; 16],
            score: TrustScore::new(0xFF),
            updated_at: 500,
        };
        assert_eq!(a, a.clone());
    }

    #[test]
    fn edge_serde_round_trip() {
        let edge = TrustEdge {
            truster: [0xCC; 16],
            trustee: [0xDD; 16],
            score: TrustScore::from_dimensions(1, 2, 3, 0),
            updated_at: 99999,
        };
        let bytes = postcard::to_allocvec(&edge).unwrap();
        let decoded: TrustEdge = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(decoded, edge);
    }
}

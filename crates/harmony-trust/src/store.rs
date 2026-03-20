use alloc::vec::Vec;
use harmony_identity::IdentityHash;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};

use crate::edge::TrustEdge;
use crate::error::TrustError;
use crate::score::TrustScore;

const FORMAT_VERSION: u8 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustStore {
    local_identity: IdentityHash,
    local_edges: HashMap<IdentityHash, LocalEdge>,
    received_edges: HashMap<IdentityHash, HashMap<IdentityHash, ReceivedEdge>>,
    /// Tombstone timestamps for removed local edges, preserving the
    /// staleness guard so a remove + stale re-insert cannot downgrade.
    #[serde(default)]
    removed_at: HashMap<IdentityHash, u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LocalEdge {
    score: TrustScore,
    updated_at: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReceivedEdge {
    score: TrustScore,
    updated_at: u64,
}

impl TrustStore {
    pub fn new(local_identity: IdentityHash) -> Self {
        Self {
            local_identity,
            local_edges: HashMap::new(),
            received_edges: HashMap::new(),
            removed_at: HashMap::new(),
        }
    }

    /// Set or update your trust score for an identity.
    /// Rejects updates with a strictly older timestamp (protects against
    /// out-of-order sync replay). Same-timestamp updates are accepted
    /// since local scores are direct user actions.
    pub fn set_score(&mut self, trustee: &IdentityHash, score: TrustScore, now: u64) {
        // Check both active edge and tombstone for staleness.
        let dominated_by_active = self
            .local_edges
            .get(trustee)
            .is_some_and(|e| e.updated_at > now);
        let dominated_by_tombstone = self.removed_at.get(trustee).is_some_and(|&t| t > now);
        if dominated_by_active || dominated_by_tombstone {
            return;
        }
        // Clear tombstone — this insert supersedes the removal.
        self.removed_at.remove(trustee);
        self.local_edges.insert(
            *trustee,
            LocalEdge {
                score,
                updated_at: now,
            },
        );
    }

    /// Remove your trust score for an identity. Returns the old score.
    /// Records a tombstone timestamp so stale re-insertions are rejected.
    pub fn remove_score(&mut self, trustee: &IdentityHash) -> Option<TrustScore> {
        self.local_edges.remove(trustee).map(|e| {
            self.removed_at.insert(*trustee, e.updated_at);
            e.score
        })
    }

    pub fn local_score(&self, trustee: &IdentityHash) -> Option<TrustScore> {
        self.local_edges.get(trustee).map(|e| e.score)
    }

    pub fn local_edges(&self) -> impl Iterator<Item = TrustEdge> + '_ {
        self.local_edges
            .iter()
            .map(move |(trustee, edge)| TrustEdge {
                truster: self.local_identity,
                trustee: *trustee,
                score: edge.score,
                updated_at: edge.updated_at,
            })
    }

    pub fn receive_edge(&mut self, edge: TrustEdge) {
        if edge.truster == self.local_identity {
            return;
        }
        let inner = self.received_edges.entry(edge.truster).or_default();
        match inner.get(&edge.trustee) {
            Some(existing) if existing.updated_at >= edge.updated_at => {}
            _ => {
                inner.insert(
                    edge.trustee,
                    ReceivedEdge {
                        score: edge.score,
                        updated_at: edge.updated_at,
                    },
                );
            }
        }
    }

    pub fn received_score(
        &self,
        truster: &IdentityHash,
        trustee: &IdentityHash,
    ) -> Option<TrustScore> {
        self.received_edges
            .get(truster)
            .and_then(|inner| inner.get(trustee))
            .map(|e| e.score)
    }

    pub fn received_edges_for(&self, trustee: &IdentityHash) -> Vec<TrustEdge> {
        let mut result = Vec::new();
        for (truster, inner) in &self.received_edges {
            if let Some(edge) = inner.get(trustee) {
                result.push(TrustEdge {
                    truster: *truster,
                    trustee: *trustee,
                    score: edge.score,
                    updated_at: edge.updated_at,
                });
            }
        }
        result
    }

    /// The effective trust score for an identity.
    ///
    /// Resolution chain:
    ///   1. Local edge (if exists) — your own subjective assessment wins.
    ///   2. `TrustScore::UNKNOWN` (0x00).
    ///
    /// Received edges are intentionally NOT consulted here. They are stored
    /// for future EigenTrust transitive computation (separate bead) and can
    /// be queried directly via `received_score()` / `received_edges_for()`.
    /// Until the EigenTrust algorithm is implemented, this method reflects
    /// only your own assessments — consumers that need network-derived trust
    /// should aggregate received edges themselves.
    pub fn effective_score(&self, trustee: &IdentityHash) -> TrustScore {
        match self.local_score(trustee) {
            Some(score) => score,
            None => TrustScore::UNKNOWN,
        }
    }

    /// Number of local trust edges.
    pub fn len(&self) -> usize {
        self.local_edges.len()
    }

    /// Whether the store has no local trust edges.
    pub fn is_empty(&self) -> bool {
        self.local_edges.is_empty()
    }

    pub fn serialize(&self) -> Result<Vec<u8>, TrustError> {
        let mut buf = Vec::new();
        buf.push(FORMAT_VERSION);
        let inner = postcard::to_allocvec(self)
            .map_err(|_| TrustError::SerializeError("postcard encode failed"))?;
        buf.extend_from_slice(&inner);
        Ok(buf)
    }

    pub fn deserialize(data: &[u8]) -> Result<Self, TrustError> {
        if data.is_empty() {
            return Err(TrustError::DeserializeError("empty data"));
        }
        if data[0] != FORMAT_VERSION {
            return Err(TrustError::DeserializeError("unsupported format version"));
        }
        postcard::from_bytes(&data[1..])
            .map_err(|_| TrustError::DeserializeError("postcard decode failed"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::TrustScore;

    const LOCAL: [u8; 16] = [0x01; 16];
    const ALICE: [u8; 16] = [0xAA; 16];
    const BOB: [u8; 16] = [0xBB; 16];
    const CAROL: [u8; 16] = [0xCC; 16];

    fn score(i: u8, c: u8, a: u8, e: u8) -> TrustScore {
        TrustScore::from_dimensions(i, c, a, e)
    }

    #[test]
    fn set_and_get_local_score() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 2, 1), 1000);
        let s = store.local_score(&ALICE).unwrap();
        assert_eq!(s.identity(), 3);
        assert_eq!(s.endorsement(), 1);
    }

    #[test]
    fn overwrite_local_score_newer_wins() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(1, 1, 1, 1), 1000);
        store.set_score(&ALICE, score(3, 3, 3, 3), 2000);
        assert_eq!(store.local_score(&ALICE).unwrap(), score(3, 3, 3, 3));
    }

    #[test]
    fn set_score_stale_rejected() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 3, 3), 2000);
        // Older timestamp should be rejected
        store.set_score(&ALICE, score(0, 0, 0, 0), 1000);
        assert_eq!(store.local_score(&ALICE).unwrap(), score(3, 3, 3, 3));
    }

    #[test]
    fn remove_then_stale_reinsert_rejected() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 3, 3), 2000);
        store.remove_score(&ALICE);
        // Stale re-insert (t=500 < removed t=2000) should be rejected
        store.set_score(&ALICE, score(0, 0, 0, 0), 500);
        assert!(store.local_score(&ALICE).is_none());
    }

    #[test]
    fn remove_then_newer_reinsert_accepted() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(1, 1, 1, 1), 1000);
        store.remove_score(&ALICE);
        // Newer re-insert (t=2000 > removed t=1000) should succeed
        store.set_score(&ALICE, score(3, 3, 3, 3), 2000);
        assert_eq!(store.local_score(&ALICE), Some(score(3, 3, 3, 3)));
    }

    #[test]
    fn set_score_same_timestamp_accepted() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(1, 1, 1, 1), 1000);
        // Same timestamp — direct user correction, should be accepted
        store.set_score(&ALICE, score(3, 3, 3, 3), 1000);
        assert_eq!(store.local_score(&ALICE).unwrap(), score(3, 3, 3, 3));
    }

    #[test]
    fn remove_local_score() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(2, 2, 2, 2), 1000);
        let removed = store.remove_score(&ALICE);
        assert_eq!(removed, Some(score(2, 2, 2, 2)));
        assert!(store.local_score(&ALICE).is_none());
    }

    #[test]
    fn remove_nonexistent_returns_none() {
        let mut store = TrustStore::new(LOCAL);
        assert!(store.remove_score(&ALICE).is_none());
    }

    #[test]
    fn local_score_unknown_returns_none() {
        let store = TrustStore::new(LOCAL);
        assert!(store.local_score(&ALICE).is_none());
    }

    #[test]
    fn iterate_local_edges() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 0, 0), 1000);
        store.set_score(&BOB, score(1, 1, 0, 0), 2000);
        let edges: Vec<_> = store.local_edges().collect();
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.truster == LOCAL));
        assert!(edges.iter().any(|e| e.trustee == ALICE));
        assert!(edges.iter().any(|e| e.trustee == BOB));
    }

    #[test]
    fn receive_and_query_edge() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(2, 2, 1, 0),
            updated_at: 1000,
        });
        assert_eq!(store.received_score(&ALICE, &BOB), Some(score(2, 2, 1, 0)));
    }

    #[test]
    fn newer_received_edge_wins() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(1, 1, 0, 0),
            updated_at: 1000,
        });
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(3, 3, 3, 3),
            updated_at: 2000,
        });
        assert_eq!(store.received_score(&ALICE, &BOB), Some(score(3, 3, 3, 3)));
    }

    #[test]
    fn stale_received_edge_rejected() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(3, 3, 3, 3),
            updated_at: 2000,
        });
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(0, 0, 0, 0),
            updated_at: 1000,
        });
        assert_eq!(store.received_score(&ALICE, &BOB), Some(score(3, 3, 3, 3)));
    }

    #[test]
    fn self_edge_ignored() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: LOCAL,
            trustee: ALICE,
            score: score(3, 3, 3, 3),
            updated_at: 1000,
        });
        assert!(store.received_score(&LOCAL, &ALICE).is_none());
    }

    #[test]
    fn received_edges_for_trustee() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: CAROL,
            score: score(3, 3, 0, 0),
            updated_at: 1000,
        });
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: CAROL,
            score: score(1, 1, 0, 0),
            updated_at: 2000,
        });
        store.receive_edge(TrustEdge {
            truster: ALICE,
            trustee: BOB,
            score: score(2, 2, 0, 0),
            updated_at: 1500,
        });
        let edges = store.received_edges_for(&CAROL);
        assert_eq!(edges.len(), 2);
        assert!(edges.iter().all(|e| e.trustee == CAROL));
    }

    #[test]
    fn received_edges_for_unknown_trustee_empty() {
        let store = TrustStore::new(LOCAL);
        assert!(store.received_edges_for(&ALICE).is_empty());
    }

    #[test]
    fn effective_score_local_wins() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 3, 3), 1000);
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: ALICE,
            score: score(0, 0, 0, 0),
            updated_at: 2000,
        });
        assert_eq!(store.effective_score(&ALICE), score(3, 3, 3, 3));
    }

    #[test]
    fn effective_score_unknown_when_no_local() {
        let mut store = TrustStore::new(LOCAL);
        store.receive_edge(TrustEdge {
            truster: BOB,
            trustee: ALICE,
            score: score(3, 3, 3, 3),
            updated_at: 1000,
        });
        assert_eq!(store.effective_score(&ALICE), TrustScore::UNKNOWN);
    }

    #[test]
    fn effective_score_completely_unknown() {
        let store = TrustStore::new(LOCAL);
        assert_eq!(store.effective_score(&ALICE), TrustScore::UNKNOWN);
    }

    #[test]
    fn effective_score_self_trust() {
        let store = TrustStore::new(LOCAL);
        assert_eq!(store.effective_score(&LOCAL), TrustScore::UNKNOWN);
        let mut store2 = TrustStore::new(LOCAL);
        store2.set_score(&LOCAL, score(3, 3, 3, 3), 1000);
        assert_eq!(store2.effective_score(&LOCAL), score(3, 3, 3, 3));
    }

    #[test]
    fn serialize_round_trip() {
        let mut store = TrustStore::new(LOCAL);
        store.set_score(&ALICE, score(3, 3, 2, 1), 1000);
        store.set_score(&BOB, score(1, 1, 0, 0), 2000);
        store.receive_edge(TrustEdge {
            truster: CAROL,
            trustee: ALICE,
            score: score(2, 2, 1, 0),
            updated_at: 3000,
        });
        let bytes = store.serialize().unwrap();
        let restored = TrustStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.local_score(&ALICE), Some(score(3, 3, 2, 1)));
        assert_eq!(restored.local_score(&BOB), Some(score(1, 1, 0, 0)));
        assert_eq!(
            restored.received_score(&CAROL, &ALICE),
            Some(score(2, 2, 1, 0))
        );
    }

    #[test]
    fn deserialize_empty_data() {
        assert!(matches!(
            TrustStore::deserialize(&[]),
            Err(TrustError::DeserializeError(_))
        ));
    }

    #[test]
    fn deserialize_bad_version() {
        assert!(matches!(
            TrustStore::deserialize(&[0xFF, 0x00]),
            Err(TrustError::DeserializeError(_))
        ));
    }

    #[test]
    fn len_and_is_empty() {
        let mut store = TrustStore::new(LOCAL);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
        store.set_score(&ALICE, score(1, 1, 1, 1), 1000);
        assert!(!store.is_empty());
        assert_eq!(store.len(), 1);
        store.set_score(&BOB, score(2, 2, 2, 2), 2000);
        assert_eq!(store.len(), 2);
    }

    #[test]
    fn empty_store_serialize_round_trip() {
        let store = TrustStore::new(LOCAL);
        let bytes = store.serialize().unwrap();
        let restored = TrustStore::deserialize(&bytes).unwrap();
        assert_eq!(restored.effective_score(&ALICE), TrustScore::UNKNOWN);
    }
}

// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OluoEngine — sans-I/O state machine for semantic search.

use alloc::vec::Vec;
use harmony_semantic::distance::hamming_distance;
use harmony_semantic::metadata::{PrivacyTier, SidecarMetadata};
use harmony_semantic::sidecar::SidecarHeader;
use harmony_semantic::tier::EmbeddingTier;

use crate::filter::RawSearchResult;
use crate::ingest::IngestDecision;
use crate::scope::SearchQuery;

/// An event submitted to the Oluo engine.
#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)] // Ingest carries SidecarHeader (280 bytes); acceptable for event type
pub enum OluoEvent {
    /// Submit a sidecar for indexing (Jain has approved).
    Ingest {
        header: SidecarHeader,
        metadata: SidecarMetadata,
        decision: IngestDecision,
        /// Current wall-clock time in milliseconds (caller-provided, sans-I/O).
        now_ms: u64,
    },
    /// Execute a search query.
    ///
    /// **Caller contract:** Search does not filter expired lightweight entries.
    /// The caller must send `EvictExpired` at an appropriate cadence to remove
    /// stale entries before querying. This separation keeps clock concerns out
    /// of the query path (consistent with the sans-I/O pattern).
    Search { query_id: u64, query: SearchQuery },
    /// A community published an updated trie root.
    SyncReceived {
        community_id: [u8; 32],
        trie_root: [u8; 32],
    },
    /// Timer tick: evict expired lightweight entries.
    EvictExpired { now_ms: u64 },
}

/// An action emitted by the Oluo engine for the caller to execute.
#[derive(Debug, Clone)]
pub enum OluoAction {
    /// The local index trie root changed.
    IndexUpdated { trie_root: [u8; 32] },
    /// Search results ready for Jain filtering.
    SearchResults {
        query_id: u64,
        results: Vec<RawSearchResult>,
    },
    /// Request a trie node blob from content store.
    FetchTrieNode { cid: [u8; 32] },
    /// Request a sidecar blob from content store.
    FetchSidecar { cid: [u8; 32] },
    /// Publish our trie root to a community.
    PublishTrieRoot {
        community_id: [u8; 32],
        root: [u8; 32],
    },
    /// Persist a blob to content-addressed storage.
    PersistBlob { cid: [u8; 32], data: Vec<u8> },
}

/// An entry in the in-memory flat index.
#[derive(Debug, Clone)]
struct IndexEntry {
    /// Tier 3 binary vector (256-bit) for distance comparison.
    tier3: [u8; 32],
    /// Merged metadata.
    metadata: SidecarMetadata,
    /// If Some, this is a lightweight entry with TTL (expires at this timestamp ms).
    expires_at: Option<u64>,
    /// When this entry was ingested (milliseconds since epoch).
    #[allow(dead_code)] // stored for future diagnostics and re-indexing
    ingested_at_ms: u64,
}

/// The Oluo search engine state machine.
pub struct OluoEngine {
    /// In-memory index entries (target_cid -> entry).
    entries: hashbrown::HashMap<[u8; 32], IndexEntry>,
    /// The collection threshold — when flat entries exceed this, a trie should be built.
    /// Reserved for future trie-building logic.
    #[allow(dead_code)]
    collection_threshold: u32,
}

impl OluoEngine {
    /// Create a new engine with an empty index and default threshold (256).
    pub fn new() -> Self {
        Self {
            entries: hashbrown::HashMap::new(),
            collection_threshold: 256,
        }
    }

    /// Create a new engine with a custom collection threshold.
    pub fn with_threshold(threshold: u32) -> Self {
        Self {
            entries: hashbrown::HashMap::new(),
            collection_threshold: threshold,
        }
    }

    /// Process an event and return resulting actions.
    pub fn handle(&mut self, event: OluoEvent) -> Vec<OluoAction> {
        match event {
            OluoEvent::Ingest {
                header,
                metadata,
                decision,
                now_ms,
            } => self.handle_ingest(header, metadata, decision, now_ms),
            OluoEvent::Search { query_id, query } => self.handle_search(query_id, query),
            OluoEvent::SyncReceived {
                community_id: _,
                trie_root,
            } => {
                // For now, request the trie root node from the content store.
                alloc::vec![OluoAction::FetchTrieNode { cid: trie_root }]
            }
            OluoEvent::EvictExpired { now_ms } => {
                // Evict entries at or past their expiry (strictly-greater keeps live entries).
                self.entries.retain(|_, entry| match entry.expires_at {
                    Some(expires) => expires > now_ms,
                    None => true,
                });
                Vec::new()
            }
        }
    }

    /// Number of indexed entries.
    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    fn handle_ingest(
        &mut self,
        header: SidecarHeader,
        metadata: SidecarMetadata,
        decision: IngestDecision,
        now_ms: u64,
    ) -> Vec<OluoAction> {
        // Reject decision — do not index.
        if decision == IngestDecision::Reject {
            return Vec::new();
        }

        // Privacy tier 3 (EncryptedEphemeral) — do not index.
        if metadata.privacy_tier == Some(PrivacyTier::EncryptedEphemeral) {
            return Vec::new();
        }

        let expires_at = match &decision {
            IngestDecision::IndexFull => None,
            IngestDecision::IndexLightweight { ttl_secs } => {
                Some(now_ms.saturating_add(ttl_secs.saturating_mul(1000)))
            }
            IngestDecision::Reject => unreachable!(),
        };

        let entry = IndexEntry {
            tier3: header.tier3,
            metadata,
            expires_at,
            ingested_at_ms: now_ms,
        };

        self.entries.insert(header.target_cid, entry);

        // Trie building deferred — return empty actions for now.
        Vec::new()
    }

    fn handle_search(&self, query_id: u64, query: SearchQuery) -> Vec<OluoAction> {
        // Only Tier 3 (256-bit) search is supported — the flat index stores tier3 vectors.
        // Non-T3 queries return empty results rather than producing incorrect scores.
        if query.tier != EmbeddingTier::T3 {
            return alloc::vec![OluoAction::SearchResults {
                query_id,
                results: Vec::new(),
            }];
        }

        let max_bits = EmbeddingTier::T3.bit_count() as f32;
        let mut scored: Vec<([u8; 32], f32, &IndexEntry)> = self
            .entries
            .iter()
            .map(|(cid, entry)| {
                let dist = hamming_distance(&query.embedding, &entry.tier3);
                let score = dist as f32 / max_bits;
                (*cid, score, entry)
            })
            .collect();

        // Sort by distance ascending (closest first).
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(core::cmp::Ordering::Equal));

        // Take top max_results.
        scored.truncate(query.max_results as usize);

        let results: Vec<RawSearchResult> = scored
            .into_iter()
            .map(|(cid, score, entry)| RawSearchResult {
                target_cid: cid,
                score,
                metadata: entry.metadata.clone(),
                overlays: Vec::new(),
            })
            .collect();

        alloc::vec![OluoAction::SearchResults { query_id, results }]
    }
}

impl Default for OluoEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_semantic::fingerprint::model_fingerprint;
    use harmony_semantic::tier::EmbeddingTier;

    use crate::scope::SearchScope;

    /// Build a test header with a specific tier3 value and target_cid.
    fn test_header(target_cid: [u8; 32], tier3: [u8; 32]) -> SidecarHeader {
        SidecarHeader {
            fingerprint: model_fingerprint("test-model"),
            target_cid,
            tier1: [0u8; 8],
            tier2: [0u8; 16],
            tier3,
            tier4: [0u8; 64],
            tier5: [0u8; 128],
        }
    }

    #[test]
    fn engine_ingest_stores_entry() {
        let mut engine = OluoEngine::new();
        assert_eq!(engine.entry_count(), 0);

        let header = test_header([0x01; 32], [0xAA; 32]);
        let metadata = SidecarMetadata::default();
        let decision = IngestDecision::IndexFull;

        let actions = engine.handle(OluoEvent::Ingest {
            header,
            metadata,
            decision,
            now_ms: 1_700_000_000_000,
        });

        assert!(actions.is_empty());
        assert_eq!(engine.entry_count(), 1);
    }

    #[test]
    fn engine_ingest_reject_blocks() {
        let mut engine = OluoEngine::new();

        let header = test_header([0x02; 32], [0xBB; 32]);
        let metadata = SidecarMetadata::default();
        let decision = IngestDecision::Reject;

        let actions = engine.handle(OluoEvent::Ingest {
            header,
            metadata,
            decision,
            now_ms: 1_700_000_000_000,
        });

        assert!(actions.is_empty());
        assert_eq!(engine.entry_count(), 0);
    }

    #[test]
    fn engine_ingest_privacy_tier_3_blocked() {
        let mut engine = OluoEngine::new();

        let header = test_header([0x03; 32], [0xCC; 32]);
        let metadata = SidecarMetadata {
            privacy_tier: Some(PrivacyTier::EncryptedEphemeral),
            ..SidecarMetadata::default()
        };
        let decision = IngestDecision::IndexFull;

        let actions = engine.handle(OluoEvent::Ingest {
            header,
            metadata,
            decision,
            now_ms: 1_700_000_000_000,
        });

        assert!(actions.is_empty());
        assert_eq!(engine.entry_count(), 0);
    }

    #[test]
    fn engine_search_non_t3_returns_empty() {
        let mut engine = OluoEngine::new();

        // Insert an entry so the index isn't empty.
        engine.handle(OluoEvent::Ingest {
            header: test_header([0x01; 32], [0xAA; 32]),
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_000,
        });

        let query = SearchQuery {
            embedding: [0xAA; 32],
            tier: EmbeddingTier::T1, // non-T3 — should be rejected
            scope: SearchScope::Personal,
            max_results: 10,
        };

        let actions = engine.handle(OluoEvent::Search { query_id: 99, query });
        match &actions[0] {
            OluoAction::SearchResults { query_id, results } => {
                assert_eq!(*query_id, 99);
                assert!(results.is_empty());
            }
            _ => panic!("expected SearchResults action"),
        }
    }

    #[test]
    fn engine_search_empty_returns_empty() {
        let mut engine = OluoEngine::new();

        let query = SearchQuery {
            embedding: [0xFF; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 10,
        };

        let actions = engine.handle(OluoEvent::Search { query_id: 1, query });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            OluoAction::SearchResults { query_id, results } => {
                assert_eq!(*query_id, 1);
                assert!(results.is_empty());
            }
            _ => panic!("expected SearchResults action"),
        }
    }

    #[test]
    fn engine_search_returns_nearest() {
        let mut engine = OluoEngine::new();

        // Insert 3 entries with known tier3 vectors.
        // Query will be [0x00; 32] (all zeros).
        // Entry A: [0x00; 32] — distance 0 (identical)
        // Entry B: [0x0F; 32] — distance 4 bits * 32 bytes = 128
        // Entry C: [0xFF; 32] — distance 8 bits * 32 bytes = 256
        let header_a = test_header([0x01; 32], [0x00; 32]);
        let header_b = test_header([0x02; 32], [0x0F; 32]);
        let header_c = test_header([0x03; 32], [0xFF; 32]);

        for header in [header_a, header_b, header_c] {
            engine.handle(OluoEvent::Ingest {
                header,
                metadata: SidecarMetadata::default(),
                decision: IngestDecision::IndexFull,
                now_ms: 1_700_000_000_000,
            });
        }

        assert_eq!(engine.entry_count(), 3);

        let query = SearchQuery {
            embedding: [0x00; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 3,
        };

        let actions = engine.handle(OluoEvent::Search {
            query_id: 42,
            query,
        });

        assert_eq!(actions.len(), 1);
        match &actions[0] {
            OluoAction::SearchResults { query_id, results } => {
                assert_eq!(*query_id, 42);
                assert_eq!(results.len(), 3);

                // Verify ranking: A (dist 0), B (dist 128), C (dist 256).
                assert_eq!(results[0].target_cid, [0x01; 32]);
                assert!((results[0].score - 0.0).abs() < f32::EPSILON);

                assert_eq!(results[1].target_cid, [0x02; 32]);
                assert!((results[1].score - 0.5).abs() < f32::EPSILON);

                assert_eq!(results[2].target_cid, [0x03; 32]);
                assert!((results[2].score - 1.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected SearchResults action"),
        }
    }

    #[test]
    fn engine_search_includes_logically_expired_without_eviction() {
        // Documents the caller contract: expired entries appear in search
        // results until EvictExpired is explicitly sent.
        let mut engine = OluoEngine::new();
        let ingest_time: u64 = 1_700_000_000_000;

        let header = test_header([0x05; 32], [0x00; 32]);
        engine.handle(OluoEvent::Ingest {
            header,
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexLightweight { ttl_secs: 10 },
            now_ms: ingest_time,
        });

        // Time has passed well beyond TTL, but no EvictExpired sent.
        let query = SearchQuery {
            embedding: [0x00; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 10,
        };
        let actions = engine.handle(OluoEvent::Search { query_id: 7, query });
        match &actions[0] {
            OluoAction::SearchResults { results, .. } => {
                // Expired entry still present — caller must evict first.
                assert_eq!(results.len(), 1);
            }
            _ => panic!("expected SearchResults"),
        }

        // After eviction, the entry is gone.
        engine.handle(OluoEvent::EvictExpired {
            now_ms: ingest_time + 20_000,
        });
        assert_eq!(engine.entry_count(), 0);
    }

    #[test]
    fn engine_evict_removes_expired() {
        let mut engine = OluoEngine::new();
        let ingest_time: u64 = 1_700_000_000_000; // realistic timestamp (ms)

        // Insert a lightweight entry with TTL of 60 seconds.
        let header = test_header([0x04; 32], [0xDD; 32]);
        let metadata = SidecarMetadata::default();
        let decision = IngestDecision::IndexLightweight { ttl_secs: 60 };

        engine.handle(OluoEvent::Ingest {
            header,
            metadata,
            decision,
            now_ms: ingest_time,
        });
        assert_eq!(engine.entry_count(), 1);

        // expires_at = ingest_time + 60_000 ms
        // Evict 1ms before expiry — should still be present.
        engine.handle(OluoEvent::EvictExpired {
            now_ms: ingest_time + 59_999,
        });
        assert_eq!(engine.entry_count(), 1);

        // Evict at expiry time — should be removed (strictly-greater keeps live entries).
        engine.handle(OluoEvent::EvictExpired {
            now_ms: ingest_time + 60_000,
        });
        assert_eq!(engine.entry_count(), 0);
    }
}

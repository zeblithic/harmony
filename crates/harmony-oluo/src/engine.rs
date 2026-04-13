// SPDX-License-Identifier: Apache-2.0 OR MIT
//! OluoEngine — sans-I/O state machine for semantic search.
//!
//! Uses `CompoundIndex` from `harmony-search` (USearch HNSW) for approximate
//! nearest-neighbor search on binary embeddings.

use harmony_search::{
    CompoundIndex, Match as SearchMatch, Metric, Quantization, SearchError, VectorIndexConfig,
};
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
    /// Timer tick: evict expired lightweight entries.
    EvictExpired { now_ms: u64 },
    /// A compacted base index was persisted to this path.
    CompactComplete {
        path: String,
        /// The generation this compaction was initiated for.
        generation: u64,
    },
}

/// An action emitted by the Oluo engine for the caller to execute.
#[derive(Debug, Clone)]
pub enum OluoAction {
    /// The local index was updated (vector added).
    IndexUpdated,
    /// Search results ready for Jain filtering.
    SearchResults {
        query_id: u64,
        results: Vec<RawSearchResult>,
    },
    /// Delta reached compaction threshold; caller should persist these bytes.
    CompactRequest {
        bytes: Vec<u8>,
        /// The compaction generation, must be passed back in `CompactComplete`.
        generation: u64,
    },
    /// An internal error occurred (index add/search/load_base failure).
    Error { message: String },
}

/// Metadata stored alongside each indexed vector.
struct EntryMetadata {
    /// CID of the content this entry references.
    target_cid: [u8; 32],
    /// Merged metadata.
    metadata: SidecarMetadata,
    /// If Some, this is a lightweight entry with TTL (expires at this timestamp ms).
    expires_at: Option<u64>,
    /// When this entry was ingested (milliseconds since epoch).
    #[allow(dead_code)] // stored for future diagnostics and re-indexing
    ingested_at_ms: u64,
}

/// The Oluo search engine state machine.
///
/// Wraps a `CompoundIndex` (USearch HNSW) with a metadata side-table.
/// Keys in the index are monotonically assigned u64 counters; the metadata
/// map holds the CID, privacy/TTL metadata, and ingestion timestamp.
pub struct OluoEngine {
    /// HNSW vector index (base + delta).
    index: CompoundIndex,
    /// Metadata keyed by the same u64 key used in the vector index.
    metadata: hashbrown::HashMap<u64, EntryMetadata>,
    /// Monotonic key counter for assigning index keys.
    key_counter: u64,
    /// Monotonic generation counter for compaction. Incremented each time
    /// a `CompactRequest` is emitted; used to reject stale `CompactComplete`.
    compact_generation: u64,
    /// Reverse lookup: CID -> index key. Ensures re-ingesting the same CID
    /// replaces the previous entry (deduplication).
    cid_to_key: hashbrown::HashMap<[u8; 32], u64>,
}

/// Unpack a 256-bit tier3 binary vector to 256 f32 values (0.0 / 1.0).
///
/// MSB-first ordering, matching `harmony_semantic::quantize_to_binary()`.
fn unpack_tier3(tier3: &[u8; 32]) -> Vec<f32> {
    let mut bits = Vec::with_capacity(256);
    for byte in tier3.iter() {
        for bit in (0..8).rev() {
            // MSB-first
            bits.push(if byte & (1 << bit) != 0 { 1.0 } else { 0.0 });
        }
    }
    bits
}

/// Default `VectorIndexConfig` for the 256-bit Hamming HNSW index.
fn default_index_config() -> VectorIndexConfig {
    VectorIndexConfig {
        dimensions: 256,
        metric: Metric::Hamming,
        quantization: Quantization::F32,
        capacity: 10_000,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
    }
}

impl OluoEngine {
    /// Create a new engine with an empty index and default compact threshold (1000).
    pub fn new() -> Self {
        let index = CompoundIndex::new(default_index_config(), 1000)
            .expect("default VectorIndexConfig must be valid");
        Self {
            index,
            metadata: hashbrown::HashMap::new(),
            key_counter: 0,
            compact_generation: 0,
            cid_to_key: hashbrown::HashMap::new(),
        }
    }

    /// Create a new engine with a custom compact threshold (used in tests).
    pub fn with_compact_threshold(threshold: usize) -> Self {
        let index = CompoundIndex::new(default_index_config(), threshold)
            .expect("default VectorIndexConfig must be valid");
        Self {
            index,
            metadata: hashbrown::HashMap::new(),
            key_counter: 0,
            compact_generation: 0,
            cid_to_key: hashbrown::HashMap::new(),
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
            OluoEvent::EvictExpired { now_ms } => self.handle_evict(now_ms),
            OluoEvent::CompactComplete { path, generation } => {
                self.handle_compact_complete(&path, generation)
            }
        }
    }

    /// Number of indexed entries (authoritative count including expired-but-not-evicted).
    pub fn entry_count(&self) -> usize {
        self.metadata.len()
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

        // CID deduplication: if this CID was previously ingested, reuse the key
        // and replace the old entry (mirrors HashMap::insert semantics).
        let (key, old_metadata) = if let Some(&existing_key) = self.cid_to_key.get(&header.target_cid) {
            let old = self.metadata.remove(&existing_key);
            (existing_key, old)
        } else {
            let k = self.key_counter;
            self.key_counter += 1;
            (k, None)
        };

        let f32_vector = unpack_tier3(&header.tier3);

        // Add to HNSW index — on failure, roll back and emit error.
        if let Err(e) = self.index.add(key, &f32_vector) {
            match &e {
                SearchError::RollbackFailed { .. } => {
                    // Index state is poisoned — the old vector is gone and could
                    // not be restored. Don't restore metadata; orphan the entry.
                    self.cid_to_key.remove(&header.target_cid);
                }
                _ => {
                    // Normal failure — old vector preserved, restore metadata.
                    if let Some(old) = old_metadata {
                        self.metadata.insert(key, old);
                    } else {
                        self.key_counter -= 1;
                    }
                }
            }
            return vec![OluoAction::Error {
                message: format!("index add failed: {e}"),
            }];
        }

        self.cid_to_key.insert(header.target_cid, key);
        self.metadata.insert(
            key,
            EntryMetadata {
                target_cid: header.target_cid,
                metadata,
                expires_at,
                ingested_at_ms: now_ms,
            },
        );

        let mut actions = Vec::new();

        // Check if compaction is needed.
        if self.index.should_compact() {
            match self.index.compact() {
                Ok(bytes) => {
                    self.compact_generation += 1;
                    actions.push(OluoAction::CompactRequest {
                        bytes,
                        generation: self.compact_generation,
                    });
                }
                Err(e) => {
                    actions.push(OluoAction::Error {
                        message: format!("compaction failed: {e}"),
                    });
                }
            }
        }

        actions.push(OluoAction::IndexUpdated);
        actions
    }

    fn handle_search(&self, query_id: u64, query: SearchQuery) -> Vec<OluoAction> {
        // max_results == 0 guard: return empty results immediately.
        if query.max_results == 0 {
            return vec![OluoAction::SearchResults {
                query_id,
                results: Vec::new(),
            }];
        }

        // Only Tier 3 (256-bit) search is supported.
        if query.tier != EmbeddingTier::T3 {
            return vec![OluoAction::SearchResults {
                query_id,
                results: Vec::new(),
            }];
        }

        let f32_vector = unpack_tier3(&query.embedding);

        // Over-fetch to compensate for entries that may have been evicted
        // from metadata but still live in the HNSW index.
        let headroom = self.evicted_headroom();
        let capped_headroom = headroom
            .min(query.max_results as usize * 5)
            .min(1000);
        let fetch_k = (query.max_results as usize).saturating_add(capped_headroom);

        let matches: Vec<SearchMatch> = match self.index.search(&f32_vector, fetch_k) {
            Ok(m) => m,
            Err(e) => {
                return vec![OluoAction::Error {
                    message: format!("search failed: {e}"),
                }];
            }
        };

        let mut results: Vec<RawSearchResult> = Vec::new();
        for m in matches {
            if let Some(entry) = self.metadata.get(&m.key) {
                // USearch Hamming on f32 XORs raw bytes and counts bits.
                // For 0.0/1.0 encoding: each differing dimension contributes
                // popcount(0x3F800000 XOR 0x00000000) = 8 bits.
                // Max distance = 256 dimensions * 8 bits = 2048.
                let score = m.distance / 2048.0;
                results.push(RawSearchResult {
                    target_cid: entry.target_cid,
                    score,
                    metadata: entry.metadata.clone(),
                    overlays: Vec::new(),
                });
                if results.len() >= query.max_results as usize {
                    break;
                }
            }
            // No metadata → entry was evicted; skip it.
        }

        vec![OluoAction::SearchResults { query_id, results }]
    }

    fn handle_evict(&mut self, now_ms: u64) -> Vec<OluoAction> {
        // Collect expired keys. We cannot remove vectors from CompoundIndex
        // easily (viewed base doesn't support remove), so we only remove
        // from metadata. Search results will filter out keyless entries.
        let expired: Vec<(u64, [u8; 32])> = self
            .metadata
            .iter()
            .filter_map(|(&key, entry)| match entry.expires_at {
                Some(expires) if expires <= now_ms => Some((key, entry.target_cid)),
                _ => None,
            })
            .collect();

        for (key, cid) in expired {
            self.metadata.remove(&key);
            self.cid_to_key.remove(&cid);
        }

        Vec::new()
    }

    fn handle_compact_complete(&mut self, path: &str, generation: u64) -> Vec<OluoAction> {
        // Reject stale CompactComplete: only accept if the generation matches
        // the current compact_generation. A newer compaction has been issued
        // since this one was started; loading the old base would overwrite
        // a newer in-memory base.
        if generation != self.compact_generation {
            return Vec::new();
        }
        if let Err(e) = self.index.load_base(path) {
            return vec![OluoAction::Error {
                message: format!("load_base failed: {e}"),
            }];
        }
        Vec::new()
    }

    /// Estimate of how many extra results to fetch to compensate for
    /// evicted-but-still-indexed vectors.
    fn evicted_headroom(&self) -> usize {
        // The index may contain vectors whose metadata has been removed.
        // Difference between unique index entries and metadata entries
        // gives a rough bound.
        self.index.unique_len().saturating_sub(self.metadata.len())
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

        assert_eq!(engine.entry_count(), 1);
        assert!(
            actions.iter().any(|a| matches!(a, OluoAction::IndexUpdated)),
            "expected IndexUpdated action"
        );
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

        let actions = engine.handle(OluoEvent::Search {
            query_id: 99,
            query,
        });
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
        let engine = OluoEngine::new();

        let query = SearchQuery {
            embedding: [0xFF; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 10,
        };

        // Use handle on a mutable ref even though we know it's a search
        let mut engine = engine;
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

                // Verify ranking order: A (dist 0), B (dist 128), C (dist 256).
                // Don't assert exact scores — USearch Hamming on f32 may produce
                // slightly different distances than pure hamming_distance.
                assert_eq!(results[0].target_cid, [0x01; 32]);
                assert_eq!(results[1].target_cid, [0x02; 32]);
                assert_eq!(results[2].target_cid, [0x03; 32]);

                // Verify ordering: A closer than B closer than C.
                assert!(results[0].score <= results[1].score);
                assert!(results[1].score <= results[2].score);
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

        // After eviction, metadata is removed. The vector stays in CompoundIndex
        // but search won't return it because there's no metadata to build a result.
        engine.handle(OluoEvent::EvictExpired {
            now_ms: ingest_time + 20_000,
        });
        assert_eq!(engine.entry_count(), 0);

        // Verify search returns empty after eviction.
        let query = SearchQuery {
            embedding: [0x00; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 10,
        };
        let actions = engine.handle(OluoEvent::Search { query_id: 8, query });
        match &actions[0] {
            OluoAction::SearchResults { results, .. } => {
                assert!(results.is_empty(), "evicted entry should not appear in search");
            }
            _ => panic!("expected SearchResults"),
        }
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

        // Evict at expiry time — should be removed (expires <= now_ms).
        engine.handle(OluoEvent::EvictExpired {
            now_ms: ingest_time + 60_000,
        });
        assert_eq!(engine.entry_count(), 0);
    }

    #[test]
    fn engine_compact_request_at_threshold() {
        // Set compact_threshold=2 so compaction triggers after 2 vectors.
        let mut engine = OluoEngine::with_compact_threshold(2);

        // First vector — no compaction yet.
        let actions = engine.handle(OluoEvent::Ingest {
            header: test_header([0x10; 32], [0xAA; 32]),
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_000,
        });
        assert!(
            !actions
                .iter()
                .any(|a| matches!(a, OluoAction::CompactRequest { .. })),
            "no CompactRequest after 1 vector"
        );

        // Second vector — should trigger compaction.
        let actions = engine.handle(OluoEvent::Ingest {
            header: test_header([0x20; 32], [0xBB; 32]),
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_001,
        });
        assert!(
            actions
                .iter()
                .any(|a| matches!(a, OluoAction::CompactRequest { .. })),
            "expected CompactRequest after reaching compact threshold"
        );
        // IndexUpdated should also be present.
        assert!(
            actions.iter().any(|a| matches!(a, OluoAction::IndexUpdated)),
            "expected IndexUpdated alongside CompactRequest"
        );
    }

    #[test]
    fn engine_compact_generation_tracking() {
        // Set compact_threshold=2 so compaction triggers after 2 vectors.
        let mut engine = OluoEngine::with_compact_threshold(2);

        // Insert 2 vectors to trigger compaction.
        engine.handle(OluoEvent::Ingest {
            header: test_header([0x10; 32], [0xAA; 32]),
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_000,
        });
        let actions = engine.handle(OluoEvent::Ingest {
            header: test_header([0x20; 32], [0xBB; 32]),
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_001,
        });

        // Extract the generation from the CompactRequest.
        let gen = actions.iter().find_map(|a| match a {
            OluoAction::CompactRequest { generation, .. } => Some(*generation),
            _ => None,
        });
        assert!(gen.is_some(), "expected CompactRequest with generation");
        let gen = gen.unwrap();
        assert_eq!(gen, 1, "first compaction should be generation 1");

        // A stale CompactComplete with wrong generation should be ignored.
        let stale_actions = engine.handle(OluoEvent::CompactComplete {
            path: "/tmp/fake_base".into(),
            generation: 999, // wrong generation
        });
        assert!(
            stale_actions.is_empty(),
            "stale CompactComplete should be silently ignored"
        );
    }

    #[test]
    fn engine_cid_deduplication() {
        let mut engine = OluoEngine::new();

        let cid = [0x42; 32];
        let header1 = test_header(cid, [0xAA; 32]);
        let header2 = test_header(cid, [0xBB; 32]); // same CID, different vector

        // Ingest twice with the same CID.
        engine.handle(OluoEvent::Ingest {
            header: header1,
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_000,
        });
        assert_eq!(engine.entry_count(), 1);

        engine.handle(OluoEvent::Ingest {
            header: header2,
            metadata: SidecarMetadata::default(),
            decision: IngestDecision::IndexFull,
            now_ms: 1_700_000_000_001,
        });
        // Should still be 1 entry — the old one was replaced.
        assert_eq!(engine.entry_count(), 1);

        // Search for the new vector [0xBB; 32] — should find it.
        let query = SearchQuery {
            embedding: [0xBB; 32],
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 10,
        };
        let actions = engine.handle(OluoEvent::Search { query_id: 1, query });
        match &actions[0] {
            OluoAction::SearchResults { results, .. } => {
                assert_eq!(results.len(), 1);
                assert_eq!(results[0].target_cid, cid);
            }
            _ => panic!("expected SearchResults"),
        }
    }

    #[test]
    fn engine_search_max_results_zero() {
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
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 0,
        };

        let actions = engine.handle(OluoEvent::Search {
            query_id: 50,
            query,
        });
        match &actions[0] {
            OluoAction::SearchResults { query_id, results } => {
                assert_eq!(*query_id, 50);
                assert!(results.is_empty(), "max_results=0 should return empty results");
            }
            _ => panic!("expected SearchResults action"),
        }
    }
}

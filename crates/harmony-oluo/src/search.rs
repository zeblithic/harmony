// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Search — collection scanning and trie traversal primitives.

use alloc::vec::Vec;
use harmony_semantic::collection::CollectionBlob;
use harmony_semantic::distance::hamming_distance;

/// A search hit with raw Hamming distance.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SearchHit {
    /// CID of the content.
    pub target_cid: [u8; 32],
    /// Hamming distance from the query vector.
    pub distance: u32,
}

/// Search a single collection blob for nearest neighbors.
///
/// Computes the Hamming distance between the query vector and each entry's
/// Tier 3 vector, returning all hits sorted by distance ascending.
/// Caller should truncate to `max_results` after merging across collections.
pub fn scan_collection(
    collection: &CollectionBlob,
    query_tier3: &[u8; 32],
    max_results: usize,
) -> Vec<SearchHit> {
    let mut hits: Vec<SearchHit> = collection
        .entries
        .iter()
        .map(|entry| {
            let distance = hamming_distance(&entry.tier3, query_tier3);
            SearchHit {
                target_cid: entry.target_cid,
                distance,
            }
        })
        .collect();

    hits.sort_by_key(|h| h.distance);
    hits.truncate(max_results);
    hits
}

#[cfg(test)]
mod tests {
    use alloc::vec;
    use alloc::vec::Vec;

    use harmony_semantic::collection::{CollectionBlob, CollectionEntry};

    use super::*;

    /// Helper: build a collection with entries having known tier3 vectors.
    fn make_collection(tier3_vecs: &[[u8; 32]]) -> CollectionBlob {
        let entries: Vec<CollectionEntry> = tier3_vecs
            .iter()
            .enumerate()
            .map(|(i, tier3)| {
                let mut target_cid = [0u8; 32];
                target_cid[0] = i as u8;
                CollectionEntry {
                    target_cid,
                    tier3: *tier3,
                }
            })
            .collect();
        let centroid = CollectionBlob::compute_centroid(&entries);
        CollectionBlob {
            fingerprint: [0x01, 0x02, 0x03, 0x04],
            centroid,
            entries,
        }
    }

    #[test]
    fn scan_collection_finds_nearest() {
        // 5 entries with distinct tier3 vectors.
        // Entry 3's tier3 is closest to the query (all 0xAA).
        let mut vecs = [[0u8; 32]; 5];
        vecs[0].fill(0x00); // distance to 0xAA: 32 * 4 = 128 (4 bits differ per byte)
        vecs[1].fill(0xFF); // distance to 0xAA: 32 * 4 = 128
        vecs[2].fill(0x55); // distance to 0xAA: 32 * 8 = 256 (all bits differ)
        vecs[3].fill(0xAA); // distance to 0xAA: 0 (exact match)
        vecs[4].fill(0xAB); // distance to 0xAA: 32 * 1 = 32 (1 bit per byte)

        let collection = make_collection(&vecs);
        let query: [u8; 32] = [0xAA; 32];

        let hits = scan_collection(&collection, &query, 10);

        assert_eq!(hits.len(), 5);
        // Entry 3 (exact match) should be first.
        assert_eq!(hits[0].target_cid[0], 3);
        assert_eq!(hits[0].distance, 0);
        // Entry 4 should be second (distance 32).
        assert_eq!(hits[1].target_cid[0], 4);
        assert_eq!(hits[1].distance, 32);
    }

    #[test]
    fn scan_collection_respects_max_results() {
        let mut vecs = [[0u8; 32]; 5];
        for (i, v) in vecs.iter_mut().enumerate() {
            v.fill(i as u8);
        }
        let collection = make_collection(&vecs);
        let query = [0u8; 32];

        let hits = scan_collection(&collection, &query, 2);

        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn scan_collection_empty() {
        let collection = CollectionBlob {
            fingerprint: [0; 4],
            centroid: [0; 32],
            entries: vec![],
        };
        let query = [0u8; 32];

        let hits = scan_collection(&collection, &query, 10);

        assert!(hits.is_empty());
    }
}

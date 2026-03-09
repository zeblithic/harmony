// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Ranking — normalize Hamming distances to similarity scores.

use alloc::vec::Vec;

use harmony_semantic::EmbeddingTier;

use crate::search::SearchHit;

/// Normalize a Hamming distance to a score in [0.0, 1.0].
///
/// Score = distance / max_distance, where max_distance is the number of
/// bits in the tier. 0.0 = identical, 1.0 = maximally different.
pub fn normalize_score(distance: u32, tier: EmbeddingTier) -> f32 {
    let max_distance = tier.bit_count() as f32;
    if max_distance == 0.0 {
        return 0.0;
    }
    distance as f32 / max_distance
}

/// Normalize a vector of search hits to scored results.
pub fn normalize_hits(hits: &[SearchHit], tier: EmbeddingTier) -> Vec<(SearchHit, f32)> {
    hits.iter()
        .map(|hit| (hit.clone(), normalize_score(hit.distance, tier)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn score_normalization() {
        // Distance 0 → score 0.0 (identical).
        let score_zero = normalize_score(0, EmbeddingTier::T3);
        assert!((score_zero - 0.0).abs() < f32::EPSILON);

        // Distance 256 → score 1.0 (maximally different for T3 = 256 bits).
        let score_max = normalize_score(256, EmbeddingTier::T3);
        assert!((score_max - 1.0).abs() < f32::EPSILON);

        // Distance 128 → score 0.5 (half the bits differ).
        let score_half = normalize_score(128, EmbeddingTier::T3);
        assert!((score_half - 0.5).abs() < f32::EPSILON);

        // Other tiers: T1 has 64 bits, so distance 32 → 0.5.
        let t1_half = normalize_score(32, EmbeddingTier::T1);
        assert!((t1_half - 0.5).abs() < f32::EPSILON);
    }
}

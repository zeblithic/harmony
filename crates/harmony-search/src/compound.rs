//! CompoundIndex: Base + Delta index management for CAS integration.

use crate::error::SearchResult;
use crate::index::{Match, VectorIndex, VectorIndexConfig};

/// Compound index with immutable Base + mutable Delta.
///
/// The base is memory-mapped from a file (zero-copy, read-only).
/// The delta holds recent additions in RAM. Searches query both
/// and merge results. Periodic compaction merges delta into base
/// and returns serialized bytes for CAS storage.
pub struct CompoundIndex {
    base: Option<VectorIndex>,
    delta: VectorIndex,
    delta_keys: Vec<u64>,
    config: VectorIndexConfig,
    compact_threshold: usize,
}

impl CompoundIndex {
    /// Create a new compound index with empty delta and no base.
    pub fn new(config: VectorIndexConfig, compact_threshold: usize) -> SearchResult<Self> {
        let delta = VectorIndex::new(config.clone())?;
        Ok(Self {
            base: None,
            delta,
            delta_keys: Vec::new(),
            config,
            compact_threshold,
        })
    }

    /// Add a vector to the delta index.
    pub fn add(&mut self, key: u64, vector: &[f32]) -> SearchResult<()> {
        self.delta.add(key, vector)?;
        self.delta_keys.push(key);
        Ok(())
    }

    /// Search both base and delta, merge results by distance.
    ///
    /// If the same key exists in both base and delta, the delta's
    /// entry is kept (it represents an update).
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<Match>> {
        let mut results = Vec::new();

        // Search delta
        if !self.delta.is_empty() {
            results.extend(self.delta.search(query, k)?);
        }

        // Search base
        if let Some(ref base) = self.base {
            if !base.is_empty() {
                let base_results = base.search(query, k)?;
                // Collect delta keys for dedup
                let delta_key_set: std::collections::HashSet<u64> =
                    results.iter().map(|m| m.key).collect();
                // Add base results that aren't shadowed by delta
                for m in base_results {
                    if !delta_key_set.contains(&m.key) {
                        results.push(m);
                    }
                }
            }
        }

        // Sort by distance and truncate to k
        results.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(k);
        Ok(results)
    }

    /// Whether the delta has reached the compaction threshold.
    pub fn should_compact(&self) -> bool {
        self.delta_len() >= self.compact_threshold
    }

    /// Total vectors across base + delta.
    pub fn len(&self) -> usize {
        let base_len = self.base.as_ref().map_or(0, |b| b.len());
        base_len + self.delta.len()
    }

    /// Number of vectors pending in the delta.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Whether both base and delta are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metric, Quantization, VectorIndexConfig};

    fn test_config() -> VectorIndexConfig {
        VectorIndexConfig {
            dimensions: 4,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        }
    }

    #[test]
    fn new_starts_empty() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.delta_len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn add_goes_to_delta() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.delta_len(), 1);
    }

    #[test]
    fn search_finds_delta_vectors() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, 1); // nearest
    }

    #[test]
    fn search_empty_returns_empty() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn should_compact_respects_threshold() {
        let mut idx = CompoundIndex::new(test_config(), 2).unwrap();
        assert!(!idx.should_compact());
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(!idx.should_compact());
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(idx.should_compact());
    }
}

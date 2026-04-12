//! CompoundIndex: Base + Delta index management for CAS integration.

use std::collections::HashSet;

use crate::error::{SearchError, SearchResult};
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
    delta_keys: HashSet<u64>,
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
            delta_keys: HashSet::new(),
            config,
            compact_threshold,
        })
    }

    /// Add a vector to the delta index.
    pub fn add(&mut self, key: u64, vector: &[f32]) -> SearchResult<()> {
        self.delta.add(key, vector)?;
        self.delta_keys.insert(key);
        Ok(())
    }

    /// Search both base and delta, merge results by distance.
    ///
    /// If the same key exists in both base and delta, the delta's
    /// entry is kept (it represents an update). Dedup is based on
    /// delta *membership* (all keys in delta), not just the top-k
    /// delta search results.
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<Match>> {
        if query.len() != self.config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "query length {} does not match index dimensions {}",
                query.len(),
                self.config.dimensions
            )));
        }

        let mut results = Vec::new();

        // Search delta
        if !self.delta.is_empty() {
            results.extend(self.delta.search(query, k)?);
        }

        // Search base
        if let Some(ref base) = self.base {
            if !base.is_empty() {
                // Over-fetch to compensate for shadowed keys that get filtered out
                let base_limit = base.len().min(k.saturating_add(self.delta_keys.len()));
                let base_results = base.search(query, base_limit)?;
                // Filter out base results for keys that exist anywhere in delta
                for m in base_results {
                    if !self.delta_keys.contains(&m.key) {
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

    /// Total vectors across base + delta (approximate — may double-count
    /// keys that exist in both due to updates).
    pub fn len(&self) -> usize {
        let base_len = self.base.as_ref().map_or(0, |b| b.len());
        base_len + self.delta.len()
    }

    /// Number of unique keys (exact count, accounts for shadowing).
    pub fn unique_len(&self) -> usize {
        let base_len = self.base.as_ref().map_or(0, |b| b.len());
        let shadowed = if self.base.is_some() {
            self.delta_keys
                .iter()
                .filter(|k| self.base.as_ref().map_or(false, |b| b.contains(**k)))
                .count()
        } else {
            0
        };
        base_len + self.delta_keys.len() - shadowed
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

    /// Merge delta into base and return serialized bytes.
    ///
    /// Uses serialize/deserialize to copy the base (not ANN search),
    /// ensuring no vectors are dropped during compaction. For shadowed
    /// keys (present in both base and delta), the base entry is removed
    /// before the delta entry is inserted.
    ///
    /// After compaction, the merged index is installed as the in-memory
    /// base and delta is reset. The returned bytes can be CAS-stored by
    /// the caller; once persisted, call `load_base()` to swap to the
    /// memory-mapped version.
    pub fn compact(&mut self) -> SearchResult<Vec<u8>> {
        let dims = self.config.dimensions;
        let mut buf = vec![0.0f32; dims];

        let merged = if let Some(ref base) = self.base {
            // Serialize base, load as writable copy — no vector loss
            let base_bytes = base.save_to_bytes()?;
            let loaded = VectorIndex::load_from_bytes(&base_bytes, self.config.clone())?;
            // Reserve extra capacity for delta insertions
            loaded.reserve(base.len() + self.delta_keys.len() + 100)?;
            // Remove base entries that delta will shadow (USearch rejects duplicate adds)
            for &key in &self.delta_keys {
                if loaded.contains(key) {
                    loaded.remove(key)?;
                }
            }
            loaded
        } else {
            let mut fresh_config = self.config.clone();
            fresh_config.capacity = self.delta_keys.len() + 100;
            VectorIndex::new(fresh_config)?
        };

        // Insert delta vectors
        for &key in &self.delta_keys {
            if self.delta.get(key, &mut buf)? {
                merged.add(key, &buf)?;
            }
        }

        // Serialize the merged result
        let bytes = merged.save_to_bytes()?;

        // Install merged as in-memory base before clearing delta
        // (no query-visible data loss between compact and load_base)
        self.base = Some(VectorIndex::load_from_bytes(&bytes, self.config.clone())?);

        // Reset delta
        self.delta = VectorIndex::new(self.config.clone())?;
        self.delta_keys.clear();

        Ok(bytes)
    }

    /// Load the base index from a file (memory-mapped, zero-copy).
    pub fn load_base(&mut self, path: &str) -> SearchResult<()> {
        self.base = Some(VectorIndex::view(path, self.config.clone())?);
        Ok(())
    }

    /// Load the base index from raw bytes.
    ///
    /// Loads into RAM (not memory-mapped). Suitable for tests
    /// and for the initial bootstrap case.
    pub fn load_base_from_bytes(&mut self, bytes: &[u8]) -> SearchResult<()> {
        self.base = Some(VectorIndex::load_from_bytes(bytes, self.config.clone())?);
        Ok(())
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
        assert_eq!(results[0].key, 1);
    }

    #[test]
    fn search_empty_returns_empty() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn search_wrong_dimensions_rejected() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        assert!(idx.search(&[1.0, 0.0], 1).is_err());
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

    #[test]
    fn compact_returns_bytes_and_resets_delta() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.delta_len(), 2);

        let _bytes = idx.compact().unwrap();
        assert_eq!(idx.delta_len(), 0);
        // Base installed automatically
        assert_eq!(idx.len(), 2);
    }

    #[test]
    fn compact_then_load_base_preserves_search() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        assert_eq!(idx.delta_len(), 0);
        assert_eq!(idx.len(), 2);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
    }

    #[test]
    fn search_merges_base_and_delta() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        assert_eq!(idx.unique_len(), 2);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn delta_shadows_base_on_same_key() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        // Add key 1 to base
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // "Update" key 1 in delta
        idx.add(1, &[0.0, 0.0, 0.0, 1.0]).unwrap();

        let results = idx.search(&[0.0, 0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
        assert!(results[0].distance < 0.01);

        // Compact again — the update persists through compaction
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        let results = idx.search(&[0.0, 0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn multiple_compact_cycles() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.delta_len(), 0);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn delta_keys_are_deduplicated() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        idx.add(3, &[0.0, 0.0, 1.0, 0.0]).unwrap();
        assert_eq!(idx.delta_keys.len(), 3);
    }

    #[test]
    fn unique_len_accounts_for_shadows() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // Shadow key 1 in delta
        idx.add(1, &[0.0, 0.0, 0.0, 1.0]).unwrap();

        assert_eq!(idx.len(), 3); // 2 base + 1 delta (double-counts)
        assert_eq!(idx.unique_len(), 2); // 2 unique keys
    }
}

//! Vector index wrapping USearch HNSW with Harmony-specific defaults.

use crate::error::{SearchError, SearchResult};
use usearch::ffi::{IndexOptions, MetricKind, ScalarKind};

/// Distance metric for vector comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Metric {
    /// Hamming distance on binary vectors. Best for Tier 3 embeddings (256-bit).
    Hamming,
    /// Cosine similarity. Best for dense float embeddings.
    Cosine,
    /// Squared L2 (Euclidean) distance.
    L2,
    /// Inner product distance.
    InnerProduct,
}

impl Metric {
    fn to_usearch(self) -> MetricKind {
        match self {
            Metric::Hamming => MetricKind::Hamming,
            Metric::Cosine => MetricKind::Cos,
            Metric::L2 => MetricKind::L2sq,
            Metric::InnerProduct => MetricKind::IP,
        }
    }
}

/// Storage quantization level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Quantization {
    /// Single-bit binary (for Hamming distance).
    Binary,
    /// 8-bit integer quantization.
    I8,
    /// 16-bit brain float.
    BF16,
    /// 16-bit half float.
    F16,
    /// 32-bit float (full precision).
    F32,
}

impl Quantization {
    fn to_usearch(self) -> ScalarKind {
        match self {
            Quantization::Binary => ScalarKind::B1,
            Quantization::I8 => ScalarKind::I8,
            Quantization::BF16 => ScalarKind::BF16,
            Quantization::F16 => ScalarKind::F16,
            Quantization::F32 => ScalarKind::F32,
        }
    }
}

/// Configuration for creating a vector index.
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// Number of dimensions per vector.
    pub dimensions: usize,
    /// Distance metric.
    pub metric: Metric,
    /// Storage quantization level.
    pub quantization: Quantization,
    /// Initial capacity (number of vectors to pre-allocate for).
    pub capacity: usize,
    /// HNSW graph connectivity (M parameter). Higher = better recall, more memory.
    pub connectivity: usize,
    /// Expansion factor during index construction (ef_construction).
    pub expansion_add: usize,
    /// Expansion factor during search (ef).
    pub expansion_search: usize,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            dimensions: 256,
            metric: Metric::Hamming,
            quantization: Quantization::Binary,
            capacity: 10_000,
            connectivity: 16,
            expansion_add: 128,
            expansion_search: 64,
        }
    }
}

/// A single search result: key + distance.
#[derive(Debug, Clone, Copy)]
pub struct Match {
    /// The key (ID) of the matched vector.
    pub key: u64,
    /// Distance to the query vector (lower = more similar).
    pub distance: f32,
}

/// HNSW vector index backed by USearch.
///
/// Provides approximate nearest-neighbor search with configurable metrics
/// and quantization levels. Designed for edge deployment with minimal
/// memory overhead.
pub struct VectorIndex {
    inner: usearch::Index,
    config: VectorIndexConfig,
}

impl VectorIndex {
    /// Create a new empty index with the given configuration.
    pub fn new(config: VectorIndexConfig) -> SearchResult<Self> {
        if config.dimensions == 0 {
            return Err(SearchError::InvalidConfig(
                "dimensions must be > 0".into(),
            ));
        }

        let opts = IndexOptions {
            dimensions: config.dimensions,
            metric: config.metric.to_usearch(),
            quantization: config.quantization.to_usearch(),
            connectivity: config.connectivity,
            expansion_add: config.expansion_add,
            expansion_search: config.expansion_search,
            multi: false,
        };

        let index = usearch::new_index(&opts)
            .map_err(|e| SearchError::Index(e.to_string()))?;

        index
            .reserve(config.capacity)
            .map_err(|e| SearchError::Index(e.to_string()))?;

        Ok(Self { inner: index, config })
    }

    /// Add a vector with the given key.
    pub fn add(&self, key: u64, vector: &[f32]) -> SearchResult<()> {
        self.inner
            .add(key, vector)
            .map_err(|e| SearchError::Index(e.to_string()))
    }

    /// Add a binary vector (packed bytes) with the given key.
    /// For Hamming distance on binary embeddings.
    pub fn add_bytes(&self, key: u64, vector: &[u8]) -> SearchResult<()> {
        // USearch expects f32 slice even for binary — convert packed bytes
        // to f32 where each element is 0.0 or 1.0 for individual bits.
        let bits: Vec<f32> = vector
            .iter()
            .flat_map(|byte| (0..8).map(move |bit| if byte & (1 << bit) != 0 { 1.0 } else { 0.0 }))
            .take(self.config.dimensions)
            .collect();
        self.add(key, &bits)
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<Match>> {
        let results = self
            .inner
            .search(query, k)
            .map_err(|e| SearchError::Index(e.to_string()))?;

        Ok(results
            .keys
            .into_iter()
            .zip(results.distances)
            .map(|(key, distance)| Match { key, distance })
            .collect())
    }

    /// Search with a binary query vector (packed bytes).
    pub fn search_bytes(&self, query: &[u8], k: usize) -> SearchResult<Vec<Match>> {
        let bits: Vec<f32> = query
            .iter()
            .flat_map(|byte| (0..8).map(move |bit| if byte & (1 << bit) != 0 { 1.0 } else { 0.0 }))
            .take(self.config.dimensions)
            .collect();
        self.search(&bits, k)
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.inner.size()
    }

    /// Whether the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Save the index to a file path.
    pub fn save(&self, path: &str) -> SearchResult<()> {
        self.inner
            .save(path)
            .map_err(|e| SearchError::Serialization(e.to_string()))
    }

    /// Load an index from a file path.
    pub fn load(path: &str, config: VectorIndexConfig) -> SearchResult<Self> {
        let index = Self::new(config)?;
        index
            .inner
            .load(path)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        Ok(index)
    }

    /// View an index from a memory-mapped file (no RAM copy).
    pub fn view(path: &str, config: VectorIndexConfig) -> SearchResult<Self> {
        let index = Self::new(config)?;
        index
            .inner
            .view(path)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        Ok(index)
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_index() {
        let config = VectorIndexConfig::default();
        let index = VectorIndex::new(config).unwrap();
        assert!(index.is_empty());
    }

    #[test]
    fn add_and_search_f32() {
        let config = VectorIndexConfig {
            dimensions: 3,
            metric: Metric::Cosine,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0]).unwrap();
        index.add(3, &[0.99, 0.01, 0.0]).unwrap();

        let results = index.search(&[1.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        // Nearest to [1,0,0] should be key 1 (exact match)
        assert_eq!(results[0].key, 1);
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn search_empty_returns_empty() {
        let config = VectorIndexConfig {
            dimensions: 3,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        let results = index.search(&[1.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn zero_dimensions_rejected() {
        let config = VectorIndexConfig {
            dimensions: 0,
            ..Default::default()
        };
        assert!(VectorIndex::new(config).is_err());
    }

    #[test]
    fn len_tracks_additions() {
        let config = VectorIndexConfig {
            dimensions: 4,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        assert_eq!(index.len(), 0);
        index.add(1, &[0.0; 4]).unwrap();
        index.add(2, &[1.0; 4]).unwrap();
        assert_eq!(index.len(), 2);
    }
}

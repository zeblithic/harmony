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

/// Create USearch IndexOptions from config without reserving capacity.
fn make_opts(config: &VectorIndexConfig) -> IndexOptions {
    IndexOptions {
        dimensions: config.dimensions,
        metric: config.metric.to_usearch(),
        quantization: config.quantization.to_usearch(),
        connectivity: config.connectivity,
        expansion_add: config.expansion_add,
        expansion_search: config.expansion_search,
        multi: false,
    }
}

impl VectorIndex {
    /// Create a new empty index with the given configuration.
    pub fn new(config: VectorIndexConfig) -> SearchResult<Self> {
        if config.dimensions == 0 {
            return Err(SearchError::InvalidConfig(
                "dimensions must be > 0".into(),
            ));
        }
        if config.connectivity == 0 {
            return Err(SearchError::InvalidConfig(
                "connectivity must be > 0".into(),
            ));
        }

        let index = usearch::new_index(&make_opts(&config))
            .map_err(|e| SearchError::Index(e.to_string()))?;

        index
            .reserve(config.capacity)
            .map_err(|e| SearchError::Index(e.to_string()))?;

        Ok(Self { inner: index, config })
    }

    /// Add a vector with the given key.
    pub fn add(&self, key: u64, vector: &[f32]) -> SearchResult<()> {
        if vector.len() != self.config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "vector length {} does not match index dimensions {}",
                vector.len(),
                self.config.dimensions
            )));
        }
        self.inner
            .add(key, vector)
            .map_err(|e| SearchError::Index(e.to_string()))
    }

    /// Add a binary vector (packed bytes) with the given key.
    /// Requires the index to be configured with Hamming metric.
    pub fn add_bytes(&self, key: u64, vector: &[u8]) -> SearchResult<()> {
        self.ensure_binary_mode()?;
        let bits = self.unpack_bits(vector)?;
        self.add(key, &bits)
    }

    /// Search for the k nearest neighbors of the query vector.
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<Match>> {
        if query.len() != self.config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "query length {} does not match index dimensions {}",
                query.len(),
                self.config.dimensions
            )));
        }
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
    /// Requires the index to be configured with Hamming metric.
    pub fn search_bytes(&self, query: &[u8], k: usize) -> SearchResult<Vec<Match>> {
        self.ensure_binary_mode()?;
        let bits = self.unpack_bits(query)?;
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
    ///
    /// The provided `config` must match the configuration used when the index
    /// was originally created and saved. Mismatched dimensions will be
    /// detected and rejected.
    pub fn load(path: &str, config: VectorIndexConfig) -> SearchResult<Self> {
        if config.dimensions == 0 {
            return Err(SearchError::InvalidConfig(
                "dimensions must be > 0".into(),
            ));
        }
        let inner = usearch::new_index(&make_opts(&config))
            .map_err(|e| SearchError::Index(e.to_string()))?;
        inner
            .load(path)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        let actual = inner.dimensions();
        if actual != config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "loaded index has {actual} dimensions, config says {}",
                config.dimensions
            )));
        }
        Ok(Self { inner, config })
    }

    /// View an index from a memory-mapped file (no RAM copy).
    ///
    /// The provided `config` must match the configuration used when the index
    /// was originally created and saved. Mismatched dimensions will be
    /// detected and rejected.
    pub fn view(path: &str, config: VectorIndexConfig) -> SearchResult<Self> {
        if config.dimensions == 0 {
            return Err(SearchError::InvalidConfig(
                "dimensions must be > 0".into(),
            ));
        }
        let inner = usearch::new_index(&make_opts(&config))
            .map_err(|e| SearchError::Index(e.to_string()))?;
        inner
            .view(path)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        let actual = inner.dimensions();
        if actual != config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "viewed index has {actual} dimensions, config says {}",
                config.dimensions
            )));
        }
        Ok(Self { inner, config })
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }

    /// Check that the index is configured for Hamming distance.
    fn ensure_binary_mode(&self) -> SearchResult<()> {
        if self.config.metric != Metric::Hamming {
            return Err(SearchError::InvalidConfig(
                "add_bytes/search_bytes require metric=Hamming".into(),
            ));
        }
        Ok(())
    }

    /// Unpack binary bytes to f32 bits, validating length.
    /// Uses MSB-first bit ordering to match harmony-semantic's
    /// `quantize_to_binary()` and harmony-oluo's `get_bit()`.
    fn unpack_bits(&self, bytes: &[u8]) -> SearchResult<Vec<f32>> {
        let expected_bytes = self.config.dimensions.div_ceil(8);
        if bytes.len() != expected_bytes {
            return Err(SearchError::InvalidConfig(format!(
                "binary vector must be exactly {expected_bytes} bytes for {} dimensions, got {}",
                self.config.dimensions,
                bytes.len()
            )));
        }
        let mut bits = Vec::with_capacity(self.config.dimensions);
        for i in 0..self.config.dimensions {
            let byte = bytes[i / 8];
            // MSB-first: bit 0 = most significant bit of byte 0
            let bit = (byte >> (7 - (i % 8))) & 1;
            bits.push(bit as f32);
        }
        Ok(bits)
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

    #[test]
    fn add_wrong_dimensions_rejected() {
        let config = VectorIndexConfig {
            dimensions: 3,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        // Too short
        assert!(index.add(1, &[1.0, 0.0]).is_err());
        // Too long
        assert!(index.add(2, &[1.0, 0.0, 0.0, 0.0]).is_err());
    }

    #[test]
    fn search_wrong_dimensions_rejected() {
        let config = VectorIndexConfig {
            dimensions: 3,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        index.add(1, &[1.0, 0.0, 0.0]).unwrap();
        assert!(index.search(&[1.0, 0.0], 1).is_err());
    }

    #[test]
    fn add_bytes_wrong_length_rejected() {
        // Note: bytes API unpacks to f32, so F32 quantization is correct here.
        // USearch B1 has different internal packing expectations.
        let config = VectorIndexConfig {
            dimensions: 16,
            metric: Metric::Hamming,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        // 16 dimensions needs exactly 2 bytes
        assert!(index.add_bytes(1, &[0xFF]).is_err()); // too short
        assert!(index.add_bytes(2, &[0xFF, 0xFF, 0xFF]).is_err()); // too long
        assert!(index.add_bytes(3, &[0xFF, 0x00]).is_ok()); // correct
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn bytes_api_rejects_non_hamming() {
        let config = VectorIndexConfig {
            dimensions: 8,
            metric: Metric::Cosine,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        assert!(index.add_bytes(1, &[0xFF]).is_err());
        assert!(index.search_bytes(&[0xFF], 1).is_err());
    }

    #[test]
    fn unpack_bits_msb_first() {
        let config = VectorIndexConfig {
            dimensions: 8,
            metric: Metric::Hamming,
            quantization: Quantization::F32,
            capacity: 10,
            ..Default::default()
        };
        let index = VectorIndex::new(config).unwrap();
        // 0x81 = 0b10000001: MSB-first → bits[0]=1, bits[7]=1, rest=0
        index.add_bytes(1, &[0x81]).unwrap();
        let results = index.search_bytes(&[0x81], 1).unwrap();
        assert_eq!(results[0].key, 1);
        assert!(results[0].distance < 0.01);
    }

    #[test]
    fn save_load_roundtrip() {
        let config = VectorIndexConfig {
            dimensions: 4,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        };
        let index = VectorIndex::new(config.clone()).unwrap();
        index.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let unique = format!(
            "harmony_search_test_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let dir = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_index.usearch");
        let path_str = path.to_str().unwrap();

        index.save(path_str).unwrap();

        let loaded = VectorIndex::load(path_str, config).unwrap();
        assert_eq!(loaded.len(), 2);

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].key, 1);

        std::fs::remove_dir_all(&dir).ok();
    }
}

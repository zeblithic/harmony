//! Vector similarity search for the Harmony mesh network.
//!
//! Wraps [USearch](https://github.com/unum-cloud/usearch) HNSW with
//! Harmony-specific abstractions: content-addressed serialization,
//! multi-metric support (Hamming, Cosine, L2), and a common API surface
//! for consumers across the ecosystem (oluo, engram, memo).
//!
//! # Usage
//!
//! ```rust
//! use harmony_search::{VectorIndex, VectorIndexConfig, Metric, Quantization};
//!
//! let config = VectorIndexConfig {
//!     dimensions: 3,
//!     metric: Metric::Cosine,
//!     quantization: Quantization::F32,
//!     capacity: 100,
//!     ..Default::default()
//! };
//! let index = VectorIndex::new(config).unwrap();
//!
//! // Add and search vectors
//! index.add(42, &[0.2, 0.6, 0.4]).unwrap();
//! let results = index.search(&[0.2, 0.6, 0.4], 1).unwrap();
//! assert_eq!(results[0].key, 42);
//! ```

mod error;
mod index;

pub use error::{SearchError, SearchResult};
pub use index::{Match, Metric, Quantization, VectorIndex, VectorIndexConfig};

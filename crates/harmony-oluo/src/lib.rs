// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Oluo — Harmony's semantic search engine.
//!
//! Uses USearch HNSW (via harmony-search) for approximate nearest-neighbor
//! search on binary embeddings with a sans-I/O state machine design.

pub mod engine;
pub mod error;
pub mod filter;
pub mod ingest;
pub mod scope;

pub use engine::{EntryMetadata, OluoAction, OluoEngine, OluoEvent};
pub use error::{OluoError, OluoResult};
pub use filter::{FilteredSearchResult, RawSearchResult, RetrievalContext, RetrievalFilter};
pub use ingest::{IngestDecision, IngestGate};
pub use scope::{SearchQuery, SearchScope};

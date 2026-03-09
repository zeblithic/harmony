// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Oluo — Harmony's semantic search engine.
//!
//! An adaptive-depth embedding trie with binary vector indexing,
//! content-addressed sidecars, and sans-I/O state machine design.

#![no_std]
extern crate alloc;

pub mod engine;
pub mod error;
pub mod filter;
pub mod ingest;
pub mod ranking;
pub mod scope;
pub mod search;
pub mod trie;
pub mod zenoh_keys;

pub use engine::{OluoAction, OluoEngine, OluoEvent};
pub use error::{OluoError, OluoResult};
pub use filter::{FilteredSearchResult, RawSearchResult, RetrievalContext, RetrievalFilter};
pub use ingest::{IngestDecision, IngestGate};
pub use ranking::normalize_score;
pub use scope::{SearchQuery, SearchScope};
pub use search::{scan_collection, SearchHit};
pub use trie::{get_bit, TrieNode};

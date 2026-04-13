// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Caller-side CAS persistence for OluoEngine snapshots.
//!
//! Two entry points:
//! - [`persist_snapshot`] — handles `PersistSnapshot` actions from the engine
//! - [`load_snapshot`] — restores an engine from CAS on startup

use std::path::{Path, PathBuf};

use harmony_content::book::BookStore;
use harmony_content::chunker::ChunkerConfig;
use harmony_content::cid::ContentId;
use harmony_content::dag;
use harmony_content::error::ContentError;
use harmony_search::SearchError;

use crate::engine::{EntryMetadata, OluoEngine};

const SNAPSHOT_VERSION: u32 = 1;
const HEAD_FILE: &str = "oluo_head.json";
const BASE_FILE: &str = "oluo_base.bin";

/// Manifest stored as a single CAS book, linking to the DAG roots
/// for the HNSW index and metadata sidecar.
#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct SnapshotManifest {
    pub version: u32,
    pub index_cid: [u8; 32],
    pub metadata_cid: [u8; 32],
    pub key_counter: u64,
    pub compact_generation: u64,
}

/// Errors from snapshot persistence operations.
#[derive(Debug, thiserror::Error)]
pub enum OluoPersistError {
    #[error("CAS operation failed: {0}")]
    Content(#[from] ContentError),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("manifest deserialization failed: {0}")]
    ManifestDeserialize(String),

    #[error("metadata deserialization failed: {0}")]
    MetadataDeserialize(String),

    #[error("engine restore failed: {0}")]
    Engine(#[from] SearchError),

    #[error("unsupported snapshot version: {0}")]
    UnsupportedVersion(u32),

    #[error("CID not found in store: {0}")]
    NotFound(String),
}

/// JSON structure for the local head file (`oluo_head.json`).
#[derive(serde::Serialize, serde::Deserialize)]
struct HeadFile {
    version: u32,
    head: String,
}

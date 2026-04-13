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

/// Persist a snapshot emitted by OluoEngine after compaction.
///
/// DAG-ingests index and metadata bytes into CAS, builds a
/// SnapshotManifest, writes the local index file for mmap,
/// and updates oluo_head.json.
///
/// Returns (local_index_path, generation) so the caller can
/// send `CompactComplete` back to the engine.
pub fn persist_snapshot(
    data_dir: &Path,
    store: &mut dyn BookStore,
    index_bytes: &[u8],
    metadata_bytes: &[u8],
    key_counter: u64,
    generation: u64,
) -> Result<(PathBuf, u64), OluoPersistError> {
    let config = ChunkerConfig::DEFAULT;

    // 1-2. DAG-ingest index and metadata into CAS.
    let index_cid = dag::ingest(index_bytes, &config, store)?;
    let metadata_cid = dag::ingest(metadata_bytes, &config, store)?;

    // 3-4. Build manifest and serialize with postcard.
    let manifest = SnapshotManifest {
        version: SNAPSHOT_VERSION,
        index_cid: index_cid.to_bytes(),
        metadata_cid: metadata_cid.to_bytes(),
        key_counter,
        compact_generation: generation,
    };
    let manifest_bytes = postcard::to_allocvec(&manifest)
        .map_err(|e| OluoPersistError::ManifestDeserialize(e.to_string()))?;

    // 5. Store manifest as a single CAS book.
    let manifest_cid = store.insert(&manifest_bytes)?;

    // 6. Write index bytes to local file for memory-mapping.
    let base_path = data_dir.join(BASE_FILE);
    atomic_write(&base_path, index_bytes)?;

    // 7. Update head file (last — crash safety).
    let head = HeadFile {
        version: SNAPSHOT_VERSION,
        head: hex::encode(manifest_cid.to_bytes()),
    };
    let head_json = serde_json::to_vec(&head)
        .map_err(|e| OluoPersistError::ManifestDeserialize(e.to_string()))?;
    atomic_write(&data_dir.join(HEAD_FILE), &head_json)?;

    Ok((base_path, generation))
}

/// Atomically write data to a file (write to .tmp, then rename).
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), OluoPersistError> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, data)?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::book::MemoryBookStore;

    fn make_test_snapshot() -> (Vec<u8>, Vec<u8>, u64, u64) {
        // Fake index bytes (just needs to be non-empty for DAG ingest)
        let index_bytes = vec![0xAA; 1024];
        // Fake metadata bytes (valid postcard for an empty BTreeMap)
        let metadata_bytes = postcard::to_allocvec(
            &std::collections::BTreeMap::<u64, EntryMetadata>::new(),
        )
        .unwrap();
        let key_counter = 42;
        let generation = 3;
        (index_bytes, metadata_bytes, key_counter, generation)
    }

    #[test]
    fn persist_snapshot_creates_head_file() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = MemoryBookStore::new();
        let (index_bytes, metadata_bytes, key_counter, generation) = make_test_snapshot();

        let (path, gen) = persist_snapshot(
            dir.path(),
            &mut store,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();

        assert_eq!(gen, generation);
        assert!(path.exists());
        assert!(dir.path().join(HEAD_FILE).exists());
    }
}

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

/// Load a snapshot from CAS and construct a ready-to-use OluoEngine.
///
/// Reads `oluo_head.json`, fetches the manifest from CAS,
/// reassembles index + metadata via DAG, restores the engine
/// via `from_snapshot`, and writes the local index file for mmap.
///
/// Returns `None` if no head file exists (fresh start).
pub fn load_snapshot(
    data_dir: &Path,
    store: &dyn BookStore,
    compact_threshold: usize,
) -> Result<Option<(OluoEngine, PathBuf, u64)>, OluoPersistError> {
    // 1. Read head file.
    let head_path = data_dir.join(HEAD_FILE);
    let head_bytes = match std::fs::read(&head_path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(OluoPersistError::Io(e)),
    };

    let head: HeadFile = serde_json::from_slice(&head_bytes)
        .map_err(|e| OluoPersistError::ManifestDeserialize(e.to_string()))?;

    if head.version != SNAPSHOT_VERSION {
        return Err(OluoPersistError::UnsupportedVersion(head.version));
    }

    // 2. Fetch manifest from CAS.
    let manifest_cid_bytes: [u8; 32] = hex::decode(&head.head)
        .map_err(|e| OluoPersistError::ManifestDeserialize(e.to_string()))?
        .try_into()
        .map_err(|_| OluoPersistError::ManifestDeserialize("invalid CID length".into()))?;
    let manifest_cid = ContentId::from_bytes(manifest_cid_bytes);

    let manifest_bytes = store
        .get(&manifest_cid)
        .ok_or_else(|| OluoPersistError::NotFound(head.head.clone()))?;

    // 3. Deserialize manifest.
    let manifest: SnapshotManifest = postcard::from_bytes(manifest_bytes)
        .map_err(|e| OluoPersistError::ManifestDeserialize(e.to_string()))?;

    if manifest.version != SNAPSHOT_VERSION {
        return Err(OluoPersistError::UnsupportedVersion(manifest.version));
    }

    // 4-5. Reassemble index and metadata from CAS.
    let index_cid = ContentId::from_bytes(manifest.index_cid);
    let metadata_cid = ContentId::from_bytes(manifest.metadata_cid);

    let index_bytes = dag::reassemble(&index_cid, store)?;
    let metadata_bytes = dag::reassemble(&metadata_cid, store)?;

    // 6. Deserialize metadata.
    let metadata_map: std::collections::BTreeMap<u64, EntryMetadata> =
        postcard::from_bytes(&metadata_bytes)
            .map_err(|e| OluoPersistError::MetadataDeserialize(e.to_string()))?;

    // 7. Restore engine.
    let engine = OluoEngine::from_snapshot(
        &index_bytes,
        metadata_map,
        manifest.key_counter,
        manifest.compact_generation,
        compact_threshold,
    )?;

    // 8. Write local index file for mmap.
    let base_path = data_dir.join(BASE_FILE);
    atomic_write(&base_path, &index_bytes)?;

    Ok(Some((engine, base_path, manifest.compact_generation)))
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
    fn load_snapshot_returns_none_when_no_head() {
        let dir = tempfile::tempdir().unwrap();
        let store = MemoryBookStore::new();

        let result = load_snapshot(dir.path(), &store, 1000).unwrap();
        assert!(result.is_none());
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

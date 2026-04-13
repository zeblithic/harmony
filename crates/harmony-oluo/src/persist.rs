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

    #[error("manifest serialization/deserialization failed: {0}")]
    ManifestSerde(String),

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
///
/// # Preconditions
///
/// `data_dir` must exist. This function does not create it.
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
        .map_err(|e| OluoPersistError::ManifestSerde(e.to_string()))?;

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
        .map_err(|e| OluoPersistError::ManifestSerde(e.to_string()))?;
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
///
/// # Preconditions
///
/// `data_dir` must exist. This function does not create it.
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
        .map_err(|e| OluoPersistError::ManifestSerde(e.to_string()))?;

    if head.version != SNAPSHOT_VERSION {
        return Err(OluoPersistError::UnsupportedVersion(head.version));
    }

    // 2. Fetch manifest from CAS.
    let manifest_cid_bytes: [u8; 32] = hex::decode(&head.head)
        .map_err(|e| OluoPersistError::ManifestSerde(e.to_string()))?
        .try_into()
        .map_err(|_| OluoPersistError::ManifestSerde("invalid CID length".into()))?;
    let manifest_cid = ContentId::from_bytes(manifest_cid_bytes);

    let manifest_bytes = store
        .get(&manifest_cid)
        .ok_or_else(|| OluoPersistError::NotFound(head.head.clone()))?;

    // 3. Deserialize manifest.
    let manifest: SnapshotManifest = postcard::from_bytes(manifest_bytes)
        .map_err(|e| OluoPersistError::ManifestSerde(e.to_string()))?;

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

/// Atomically write data to a file (write to .tmp, sync, then rename).
fn atomic_write(path: &Path, data: &[u8]) -> Result<(), OluoPersistError> {
    use std::io::Write;
    let tmp = path.with_extension("tmp");
    let mut file = std::fs::File::create(&tmp)?;
    file.write_all(data)?;
    file.sync_all()?;
    drop(file);
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::book::MemoryBookStore;
    use crate::engine::{OluoAction, OluoEngine, OluoEvent};
    use crate::scope::SearchScope;
    use harmony_semantic::metadata::SidecarMetadata;
    use harmony_semantic::sidecar::SidecarHeader;
    use harmony_semantic::tier::EmbeddingTier;
    use crate::ingest::IngestDecision;
    use crate::scope::SearchQuery;

    /// Helper: create a SidecarHeader with a specific tier3 embedding.
    fn test_header(tier3: [u8; 32], cid: [u8; 32]) -> SidecarHeader {
        SidecarHeader {
            fingerprint: [0u8; 4],
            target_cid: cid,
            tier1: [0u8; 8],
            tier2: [0u8; 16],
            tier3,
            tier4: [0u8; 64],
            tier5: [0u8; 128],
        }
    }

    /// Helper: ingest N entries and trigger compaction, returning the PersistSnapshot payload.
    fn ingest_and_compact(engine: &mut OluoEngine, count: usize) -> OluoAction {
        for i in 0..count {
            let mut tier3 = [0u8; 32];
            tier3[0] = i as u8;
            let mut cid = [0u8; 32];
            cid[0] = i as u8;
            cid[1] = 0xFF;

            let actions = engine.handle(OluoEvent::Ingest {
                header: test_header(tier3, cid),
                metadata: SidecarMetadata::default(),
                decision: IngestDecision::IndexFull,
                now_ms: 1000 + i as u64,
                scope: SearchScope::Personal,
                overlay_cids: vec![],
            });
            // Check if PersistSnapshot was emitted (compaction triggered)
            for action in &actions {
                if matches!(action, OluoAction::PersistSnapshot { .. }) {
                    return action.clone();
                }
            }
        }
        panic!("compaction was not triggered after {count} ingests");
    }

    #[test]
    fn persist_and_load_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = MemoryBookStore::new();

        // Create engine with low threshold so compaction triggers quickly.
        let mut engine = OluoEngine::with_compact_threshold(5);

        // Ingest entries until compaction triggers.
        let persist_action = ingest_and_compact(&mut engine, 10);

        let (index_bytes, metadata_bytes, key_counter, generation) = match persist_action {
            OluoAction::PersistSnapshot {
                index_bytes,
                metadata_bytes,
                key_counter,
                generation,
            } => (index_bytes, metadata_bytes, key_counter, generation),
            _ => unreachable!(),
        };

        // Search the original engine for a reference result.
        let query = SearchQuery {
            embedding: [0u8; 32], // matches first entry
            tier: EmbeddingTier::T3,
            scope: SearchScope::Personal,
            max_results: 5,
        };
        let original_actions = engine.handle(OluoEvent::Search {
            query_id: 1,
            query: query.clone(),
        });
        let original_results = match &original_actions[0] {
            OluoAction::SearchResults { results, .. } => results.clone(),
            _ => panic!("expected SearchResults"),
        };

        // Persist.
        let (base_path, gen) = persist_snapshot(
            dir.path(),
            &mut store,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();
        assert_eq!(gen, generation);
        assert!(base_path.exists());

        // Load.
        let (restored_engine, restored_path, restored_gen) =
            load_snapshot(dir.path(), &store, 5).unwrap().unwrap();
        assert_eq!(restored_gen, generation);
        assert!(restored_path.exists());

        // Search the restored engine and compare.
        let mut restored_engine = restored_engine;
        let restored_actions = restored_engine.handle(OluoEvent::Search {
            query_id: 2,
            query,
        });
        let restored_results = match &restored_actions[0] {
            OluoAction::SearchResults { results, .. } => results.clone(),
            _ => panic!("expected SearchResults"),
        };

        // Same number of results, same CIDs (order may vary by distance).
        assert_eq!(original_results.len(), restored_results.len());
        let mut orig_cids: Vec<[u8; 32]> =
            original_results.iter().map(|r| r.target_cid).collect();
        let mut rest_cids: Vec<[u8; 32]> =
            restored_results.iter().map(|r| r.target_cid).collect();
        orig_cids.sort();
        rest_cids.sort();
        assert_eq!(orig_cids, rest_cids);
    }

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

    #[test]
    fn manifest_fields_match_persist_payload() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = MemoryBookStore::new();
        let (index_bytes, metadata_bytes, key_counter, generation) = make_test_snapshot();

        persist_snapshot(
            dir.path(),
            &mut store,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();

        // Read back the head file to get the manifest CID.
        let head_bytes = std::fs::read(dir.path().join(HEAD_FILE)).unwrap();
        let head: HeadFile = serde_json::from_slice(&head_bytes).unwrap();
        let manifest_cid_bytes: [u8; 32] = hex::decode(&head.head).unwrap().try_into().unwrap();
        let manifest_cid = ContentId::from_bytes(manifest_cid_bytes);

        // Fetch and deserialize the manifest.
        let manifest_bytes = store.get(&manifest_cid).unwrap();
        let manifest: SnapshotManifest = postcard::from_bytes(manifest_bytes).unwrap();

        assert_eq!(manifest.version, SNAPSHOT_VERSION);
        assert_eq!(manifest.key_counter, key_counter);
        assert_eq!(manifest.compact_generation, generation);

        // Verify index and metadata CIDs resolve to the original bytes.
        let index_cid = ContentId::from_bytes(manifest.index_cid);
        let metadata_cid = ContentId::from_bytes(manifest.metadata_cid);
        let recovered_index = dag::reassemble(&index_cid, &store).unwrap();
        let recovered_metadata = dag::reassemble(&metadata_cid, &store).unwrap();
        assert_eq!(recovered_index, index_bytes);
        assert_eq!(recovered_metadata, metadata_bytes);
    }

    #[test]
    fn load_after_head_deleted_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let mut store = MemoryBookStore::new();
        let (index_bytes, metadata_bytes, key_counter, generation) = make_test_snapshot();

        persist_snapshot(
            dir.path(),
            &mut store,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();

        // Simulate crash: delete head file.
        std::fs::remove_file(dir.path().join(HEAD_FILE)).unwrap();

        let result = load_snapshot(dir.path(), &store, 1000).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn persist_same_data_twice_produces_same_head() {
        let dir1 = tempfile::tempdir().unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        let mut store1 = MemoryBookStore::new();
        let mut store2 = MemoryBookStore::new();
        let (index_bytes, metadata_bytes, key_counter, generation) = make_test_snapshot();

        persist_snapshot(
            dir1.path(),
            &mut store1,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();

        persist_snapshot(
            dir2.path(),
            &mut store2,
            &index_bytes,
            &metadata_bytes,
            key_counter,
            generation,
        )
        .unwrap();

        let head1 = std::fs::read_to_string(dir1.path().join(HEAD_FILE)).unwrap();
        let head2 = std::fs::read_to_string(dir2.path().join(HEAD_FILE)).unwrap();
        assert_eq!(head1, head2);
    }
}

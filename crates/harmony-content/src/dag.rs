use crate::blob::BlobStore;
use crate::bundle::{BundleBuilder, MAX_BUNDLE_ENTRIES};
use crate::chunker::{chunk_all, ChunkerConfig};
use crate::cid::ContentId;
use crate::error::ContentError;

/// Ingest raw data into the content store, returning the root CID.
///
/// Chunks the data with FastCDC, inserts each chunk as a blob, and builds
/// a Merkle DAG of bundles. For data that fits in a single chunk, returns
/// the bare blob CID (no bundle wrapper).
///
/// The root bundle (if any) includes inline metadata with the total file
/// size and chunk count. Timestamp and MIME are set to zero (placeholders).
pub fn ingest(
    data: &[u8],
    config: &ChunkerConfig,
    store: &mut dyn BlobStore,
) -> Result<ContentId, ContentError> {
    if data.is_empty() {
        return Err(ContentError::EmptyData);
    }

    let ranges = chunk_all(data, config)?;

    // Single chunk — store as a bare blob, no bundle wrapper.
    if ranges.len() == 1 {
        return store.insert(data);
    }

    // Multiple chunks — insert each as a blob.
    let chunk_count = ranges.len();
    let mut blob_cids: Vec<ContentId> = Vec::with_capacity(chunk_count);
    for range in &ranges {
        let cid = store.insert(&data[range.clone()])?;
        blob_cids.push(cid);
    }

    // Build the bundle tree bottom-up.
    let mut current_level = blob_cids;
    let mut is_root = false;

    loop {
        // If we're down to few enough CIDs for a single bundle, this is the root.
        if current_level.len() <= MAX_BUNDLE_ENTRIES {
            is_root = true;
        }

        let mut next_level: Vec<ContentId> = Vec::new();

        for group in current_level.chunks(MAX_BUNDLE_ENTRIES) {
            let mut builder = BundleBuilder::new();
            for cid in group {
                builder.add(*cid);
            }

            // Attach inline metadata to the root bundle only.
            if is_root && next_level.is_empty() {
                builder.with_metadata(
                    data.len() as u64,
                    chunk_count as u32,
                    0,          // timestamp placeholder
                    [0u8; 8],   // MIME placeholder
                );
            }

            let (bundle_bytes, bundle_cid) = builder.build()?;
            store.store(bundle_cid, bundle_bytes);
            next_level.push(bundle_cid);
        }

        if next_level.len() == 1 {
            return Ok(next_level[0]);
        }

        current_level = next_level;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;
    use crate::bundle;
    use crate::chunker::ChunkerConfig;
    use crate::cid::CidType;

    /// Small chunker config for fast tests (min=64, avg=128, max=256 bytes).
    fn test_config() -> ChunkerConfig {
        ChunkerConfig {
            min_chunk: 64,
            avg_chunk: 128,
            max_chunk: 256,
        }
    }

    #[test]
    fn module_compiles() {
        let _store = MemoryBlobStore::new();
        let _config = test_config();
    }

    #[test]
    fn ingest_empty_returns_error() {
        let mut store = MemoryBlobStore::new();
        let result = ingest(&[], &test_config(), &mut store);
        assert!(matches!(result, Err(ContentError::EmptyData)));
    }

    #[test]
    fn ingest_small_returns_blob_cid() {
        let mut store = MemoryBlobStore::new();
        let data = b"hello small file";
        let cid = ingest(data, &test_config(), &mut store).unwrap();
        assert_eq!(cid.cid_type(), CidType::Blob);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn ingest_multi_chunk_returns_bundle_cid() {
        let mut store = MemoryBlobStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let cid = ingest(&data, &test_config(), &mut store).unwrap();
        assert!(matches!(cid.cid_type(), CidType::Bundle(_)));
    }

    #[test]
    fn ingest_exact_max_chunk_returns_blob() {
        let mut store = MemoryBlobStore::new();
        let config = test_config();
        // Data exactly at max_chunk — single chunk, no bundle.
        let data = vec![0xAB; config.max_chunk];
        let cid = ingest(&data, &config, &mut store).unwrap();
        assert_eq!(cid.cid_type(), CidType::Blob);
    }

    #[test]
    fn ingest_multi_chunk_has_inline_metadata() {
        let mut store = MemoryBlobStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root_cid = ingest(&data, &test_config(), &mut store).unwrap();

        // Parse the root bundle and check first entry is InlineMetadata.
        let root_bytes = store.get(&root_cid).unwrap();
        let entries = bundle::parse_bundle(root_bytes).unwrap();
        assert_eq!(entries[0].cid_type(), CidType::InlineMetadata);

        let (total_size, chunk_count, _ts, _mime) =
            entries[0].parse_inline_metadata().unwrap();
        assert_eq!(total_size, data.len() as u64);
        assert!(chunk_count > 1);
    }
}

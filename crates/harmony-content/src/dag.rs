use crate::book::BookStore;
use crate::bundle::{self, BundleBuilder, MAX_BUNDLE_ENTRIES};
use crate::chunker::{chunk_all, ChunkerConfig};
use crate::cid::{CidType, ContentId};
use crate::error::ContentError;
use alloc::vec::Vec;

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
    store: &mut dyn BookStore,
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
                    0,        // timestamp placeholder
                    [0u8; 8], // MIME placeholder
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

/// Walk a Merkle DAG from the root, returning leaf CIDs in order.
///
/// Performs a depth-first left-to-right traversal. For a bare blob root,
/// returns a single-element vec. For a bundle root, recursively descends
/// through child bundles, collecting leaf CIDs. Sentinel `InlineData` entries
/// (metadata CIDs) are skipped; non-sentinel `InlineData` CIDs carry real
/// inline data and are included. `Stream` CIDs are not walked recursively.
///
/// Returns `MissingContent` if any referenced CID is not in the store.
pub fn walk(root_cid: &ContentId, store: &dyn BookStore) -> Result<Vec<ContentId>, ContentError> {
    let mut result = Vec::new();
    walk_recursive(root_cid, store, &mut result)?;
    Ok(result)
}

fn walk_recursive(
    cid: &ContentId,
    store: &dyn BookStore,
    result: &mut Vec<ContentId>,
) -> Result<(), ContentError> {
    match cid.cid_type() {
        CidType::Book => {
            result.push(*cid);
        }
        CidType::Bundle(_) => {
            let data = store
                .get(cid)
                .ok_or(ContentError::MissingContent { cid: *cid })?;
            let children = bundle::parse_bundle(data)?;
            for child in children {
                walk_recursive(child, store, result)?;
            }
        }
        CidType::InlineData => {
            // Sentinel inline CIDs are metadata — skip them.
            // Non-sentinel inline CIDs carry real data — include them.
            if !cid.is_sentinel() {
                result.push(*cid);
            }
        }
        CidType::Stream => {
            // Streams are not walked recursively — they're processed
            // via streaming iteration, not DAG traversal.
        }
    }
    Ok(())
}

/// Reassemble original data from a Merkle DAG root CID.
///
/// Walks the DAG to collect blob CIDs in order, fetches each blob from the
/// store, and concatenates them. If the root is a bundle with inline metadata,
/// pre-allocates the output buffer using the total file size.
///
/// Guarantees: `reassemble(ingest(data)) == data` for all non-empty inputs.
pub fn reassemble(root_cid: &ContentId, store: &dyn BookStore) -> Result<Vec<u8>, ContentError> {
    // Try to pre-allocate using inline metadata if available.
    let capacity = estimate_size(root_cid, store);
    let mut output = Vec::with_capacity(capacity);

    let blob_cids = walk(root_cid, store)?;
    for cid in &blob_cids {
        if cid.cid_type() == CidType::InlineData {
            // Non-sentinel inline CIDs carry data in the CID itself.
            let inline = cid.extract_inline_data()?;
            output.extend_from_slice(&inline);
        } else {
            let data = store
                .get(cid)
                .ok_or(ContentError::MissingContent { cid: *cid })?;
            output.extend_from_slice(data);
        }
    }

    Ok(output)
}

/// Estimate total data size from inline metadata (if present).
/// Returns 0 if unavailable — Vec will grow dynamically.
fn estimate_size(root_cid: &ContentId, store: &dyn BookStore) -> usize {
    if let CidType::Bundle(_) = root_cid.cid_type() {
        if let Some(data) = store.get(root_cid) {
            if let Ok(entries) = bundle::parse_bundle(data) {
                if let Some(first) = entries.first() {
                    if first.cid_type() == CidType::InlineData {
                        if let Ok((total_size, _, _, _)) = first.parse_inline_metadata() {
                            return total_size as usize;
                        }
                    }
                }
            }
        }
    } else if let CidType::Book = root_cid.cid_type() {
        return root_cid.payload_size() as usize;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::book::MemoryBookStore;
    use crate::chunker::ChunkerConfig;

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
        let _store = MemoryBookStore::new();
        let _config = test_config();
    }

    #[test]
    fn ingest_empty_returns_error() {
        let mut store = MemoryBookStore::new();
        let result = ingest(&[], &test_config(), &mut store);
        assert!(matches!(result, Err(ContentError::EmptyData)));
    }

    #[test]
    fn ingest_small_returns_book_cid() {
        let mut store = MemoryBookStore::new();
        let data = b"hello small file";
        let cid = ingest(data, &test_config(), &mut store).unwrap();
        assert_eq!(cid.cid_type(), CidType::Book);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn ingest_multi_chunk_returns_bundle_cid() {
        let mut store = MemoryBookStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let cid = ingest(&data, &test_config(), &mut store).unwrap();
        assert!(matches!(cid.cid_type(), CidType::Bundle(_)));
    }

    #[test]
    fn ingest_exact_max_chunk_returns_book() {
        let mut store = MemoryBookStore::new();
        let config = test_config();
        // Data exactly at max_chunk — single chunk, no bundle.
        let data = vec![0xAB; config.max_chunk];
        let cid = ingest(&data, &config, &mut store).unwrap();
        assert_eq!(cid.cid_type(), CidType::Book);
    }

    #[test]
    fn ingest_multi_chunk_has_inline_metadata() {
        let mut store = MemoryBookStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root_cid = ingest(&data, &test_config(), &mut store).unwrap();

        // Parse the root bundle and check first entry is InlineData.
        let root_bytes = store.get(&root_cid).unwrap();
        let entries = bundle::parse_bundle(root_bytes).unwrap();
        assert_eq!(entries[0].cid_type(), CidType::InlineData);

        let (total_size, chunk_count, _ts, _mime) = entries[0].parse_inline_metadata().unwrap();
        assert_eq!(total_size, data.len() as u64);
        assert!(chunk_count > 1);
    }

    #[test]
    fn walk_book_returns_single_cid() {
        let mut store = MemoryBookStore::new();
        let data = b"small book";
        let cid = store.insert(data).unwrap();
        let books = walk(&cid, &store).unwrap();
        assert_eq!(books, vec![cid]);
    }

    #[test]
    fn walk_bundle_returns_books_in_order() {
        let mut store = MemoryBookStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let books = walk(&root, &store).unwrap();

        // Should have multiple books, all of type Book.
        assert!(books.len() > 1);
        for cid in &books {
            assert_eq!(cid.cid_type(), CidType::Book);
        }
    }

    #[test]
    fn walk_missing_cid_returns_error() {
        let store = MemoryBookStore::new();
        // Ingest into a different store, then walk from empty store.
        let mut other_store = MemoryBookStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut other_store).unwrap();

        let result = walk(&root, &store);
        assert!(matches!(result, Err(ContentError::MissingContent { .. })));
    }

    #[test]
    fn round_trip_small_data() {
        let mut store = MemoryBookStore::new();
        let data = b"hello round trip";
        let root = ingest(data.as_slice(), &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_medium_data() {
        let mut store = MemoryBookStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_large_data() {
        let mut store = MemoryBookStore::new();
        // 8KB+ should produce two+ bundle levels with small config.
        let data: Vec<u8> = (0..8192).map(|i| (i * 41 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_exact_max_chunk() {
        let mut store = MemoryBookStore::new();
        let config = test_config();
        let data = vec![0xCD; config.max_chunk];
        let root = ingest(&data, &config, &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn structural_sharing_across_versions() {
        let mut store = MemoryBookStore::new();
        let config = test_config();

        // Version 1
        let v1: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let _root_v1 = ingest(&v1, &config, &mut store).unwrap();
        let count_after_v1 = store.len();

        // Version 2: small edit in the middle
        let mut v2 = v1.clone();
        v2[1024..1034].copy_from_slice(&[0xFF; 10]);
        let root_v2 = ingest(&v2, &config, &mut store).unwrap();
        let count_after_v2 = store.len();

        // V2 should add fewer new entries than V1 did (shared chunks).
        let new_entries = count_after_v2 - count_after_v1;
        assert!(
            new_entries < count_after_v1,
            "v2 added {} entries, v1 had {} — expected fewer due to dedup",
            new_entries,
            count_after_v1
        );

        // V2 should still round-trip correctly.
        let recovered_v2 = reassemble(&root_v2, &store).unwrap();
        assert_eq!(recovered_v2, v2);
    }
}

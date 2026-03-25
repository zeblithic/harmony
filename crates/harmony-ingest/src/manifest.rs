//! Manifest building: ManifestHeader + shard CID list → Merkle DAG.

use harmony_content::book::MemoryBookStore;
use harmony_content::chunker::ChunkerConfig;
use harmony_content::cid::ContentId;
use harmony_content::dag;
use harmony_engram::ManifestHeader;

/// Build the manifest DAG from header + shard CIDs.
///
/// Returns (root_cid, store_containing_all_dag_books).
pub fn build_manifest(
    header: &ManifestHeader,
    shard_cids: &[[u8; 32]],
) -> Result<(ContentId, MemoryBookStore), String> {
    // Serialize header.
    let header_bytes = header
        .to_bytes()
        .map_err(|e| format!("manifest header serialize: {e}"))?;

    // Concatenate: [header_bytes | shard_cid_bytes]
    let cid_bytes_len = shard_cids.len() * 32;
    let mut combined = Vec::with_capacity(header_bytes.len() + cid_bytes_len);
    combined.extend_from_slice(&header_bytes);
    for cid in shard_cids {
        combined.extend_from_slice(cid);
    }

    // Ingest into a fresh MemoryBookStore.
    let mut store = MemoryBookStore::new();
    let root_cid = dag::ingest(&combined, &ChunkerConfig::DEFAULT, &mut store)
        .map_err(|e| format!("dag ingest: {e}"))?;

    Ok((root_cid, store))
}

/// Construct a ManifestHeader from config + tensor metadata.
pub fn make_header(
    version: String,
    embedding_dim: u32,
    num_heads: u32,
    hash_seeds: Vec<u64>,
    total_entries: u64,
    shard_size: u32,
    num_shards: u64,
) -> ManifestHeader {
    ManifestHeader {
        version,
        embedding_dim,
        dtype_bytes: 2, // f16
        num_heads,
        hash_seeds,
        total_entries,
        shard_size,
        num_shards,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use harmony_content::dag;

    #[test]
    fn build_and_recover_manifest() {
        let header = make_header(
            "v1".into(),
            4,    // embedding_dim
            2,    // num_heads
            vec![42, 99],
            6,    // total_entries
            3,    // shard_size
            2,    // num_shards
        );

        // 2 dummy shard CIDs.
        let shard_cids: Vec<[u8; 32]> = vec![[0xAA; 32], [0xBB; 32]];

        let (root_cid, store) = build_manifest(&header, &shard_cids).unwrap();

        // Reassemble the manifest from the DAG.
        let reassembled = dag::reassemble(&root_cid, &store).unwrap();

        // First part is the postcard-encoded header.
        let header_bytes = header.to_bytes().unwrap();
        let recovered_header =
            ManifestHeader::from_bytes(&reassembled[..header_bytes.len()]).unwrap();
        assert_eq!(recovered_header, header);

        // Remaining bytes are shard CIDs.
        let cid_bytes = &reassembled[header_bytes.len()..];
        assert_eq!(cid_bytes.len(), 64); // 2 × 32
        assert_eq!(&cid_bytes[0..32], &[0xAA; 32]);
        assert_eq!(&cid_bytes[32..64], &[0xBB; 32]);
    }

    #[test]
    fn into_books_returns_all_dag_nodes() {
        let header = make_header("v1".into(), 4, 1, vec![0], 3, 3, 1);
        let shard_cids = vec![[0x11; 32]];
        let (_root_cid, store) = build_manifest(&header, &shard_cids).unwrap();

        // Store should contain at least 1 book (the manifest data).
        let books: Vec<_> = store.into_books().collect();
        assert!(!books.is_empty());
        for (cid, data) in &books {
            assert!(!data.is_empty());
            assert_eq!(cid.to_bytes().len(), 32);
        }
    }
}

# Merkle DAG Builder and Walker Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the end-to-end content pipeline — `ingest()` splits data into chunks and builds a Merkle DAG, `walk()` traverses a DAG to yield ordered blob CIDs, and `reassemble()` reconstructs the original data — with the round-trip guarantee `reassemble(ingest(data)) == data`.

**Architecture:** Three free functions in a new `dag.rs` module composing existing building blocks (`chunk_all`, `BlobStore`, `BundleBuilder`, `parse_bundle`). No struct, no state — pure functions operating on `&dyn BlobStore` / `&mut dyn BlobStore`. Sans-I/O.

**Tech Stack:** Rust, existing `harmony-content` crate primitives (blob, bundle, chunker, cid)

---

### Task 1: Error Variants + Module Scaffolding

**Files:**
- Create: `crates/harmony-content/src/dag.rs`
- Modify: `crates/harmony-content/src/error.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Context:** We need two new error variants and the dag module registered. `ContentError` is in `error.rs`. `ContentId` already implements `Display` (shows hex prefix + type + size), which is used in the `MissingContent` error message.

**Step 1: Add error variants to `crates/harmony-content/src/error.rs`**

Add after the `InvalidChunkerConfig` variant (line 23):

```rust
    #[error("cannot ingest empty data")]
    EmptyData,

    #[error("content not found in store: {cid}")]
    MissingContent { cid: ContentId },
```

This requires adding `use crate::cid::ContentId;` at the top of error.rs.

**Step 2: Create `crates/harmony-content/src/dag.rs` with stub + test**

```rust
use crate::blob::BlobStore;
use crate::bundle::{self, BundleBuilder, MAX_BUNDLE_ENTRIES};
use crate::chunker::{chunk_all, ChunkerConfig};
use crate::cid::{CidType, ContentId};
use crate::error::ContentError;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;
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
        let _store = MemoryBlobStore::new();
        let _config = test_config();
    }
}
```

**Step 3: Add `pub mod dag;` to `crates/harmony-content/src/lib.rs`**

```rust
pub mod blob;
pub mod bundle;
pub mod chunker;
pub mod cid;
pub mod dag;
pub mod error;
```

**Step 4: Run tests to verify compilation**

Run: `cargo test -p harmony-content dag::tests::module_compiles`
Expected: PASS

Run: `cargo clippy -p harmony-content`
Expected: Clean (may have unused import warnings for dag.rs — fine, they'll resolve in Task 2)

**Step 5: Commit**

```bash
git add crates/harmony-content/src/dag.rs crates/harmony-content/src/error.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add dag module scaffolding and error variants"
```

---

### Task 2: `ingest()` — Data to Root CID

**Files:**
- Modify: `crates/harmony-content/src/dag.rs`

**Context:** `ingest()` is the entry point for the content pipeline. It takes raw data, chunks it with FastCDC, inserts blobs into the store, and builds a bundle tree. Key APIs to compose:

- `chunk_all(data, config) -> Result<Vec<Range<usize>>, ContentError>` — splits data into byte ranges
- `store.insert(chunk) -> Result<ContentId, ContentError>` — inserts a blob, returns its CID
- `BundleBuilder::new().add(cid).with_metadata(...).build() -> Result<(Vec<u8>, ContentId), ContentError>` — builds a bundle
- `store.store(cid, bytes)` — stores bundle bytes under a pre-computed CID
- `MAX_BUNDLE_ENTRIES` = 32,767 — max CIDs per bundle

For single-chunk data, return the blob CID directly (no bundle wrapper). For multi-chunk data, build bundle tree with inline metadata on the root.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/dag.rs` (before `#[cfg(test)]`):

```rust
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
```

Add tests:

```rust
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
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content ingest`
Expected: 5 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/dag.rs
git commit -m "feat(content): add ingest() — data to Merkle DAG root CID"
```

---

### Task 3: `walk()` — Root CID to Ordered Blob CIDs

**Files:**
- Modify: `crates/harmony-content/src/dag.rs`

**Context:** `walk()` takes a root CID and a store, and returns all leaf blob CIDs in depth-first left-to-right order. For a blob root, returns `vec![blob_cid]`. For a bundle root, recursively parses children, skipping InlineMetadata entries. Max recursion is 7 (capped by CID type system).

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/dag.rs` (after `ingest()`, before `#[cfg(test)]`):

```rust
/// Walk a Merkle DAG from the root, returning leaf blob CIDs in order.
///
/// Performs a depth-first left-to-right traversal. For a bare blob root,
/// returns a single-element vec. For a bundle root, recursively descends
/// through child bundles, collecting blob CIDs. InlineMetadata entries
/// are skipped (they carry metadata, not data).
///
/// Returns `MissingContent` if any referenced CID is not in the store.
pub fn walk(root_cid: &ContentId, store: &dyn BlobStore) -> Result<Vec<ContentId>, ContentError> {
    let mut result = Vec::new();
    walk_recursive(root_cid, store, &mut result)?;
    Ok(result)
}

fn walk_recursive(
    cid: &ContentId,
    store: &dyn BlobStore,
    result: &mut Vec<ContentId>,
) -> Result<(), ContentError> {
    match cid.cid_type() {
        CidType::Blob => {
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
        CidType::InlineMetadata => {
            // Skip — metadata entries don't carry data.
        }
        _ => {
            // Reserved types — should not appear in a well-formed DAG.
            return Err(ContentError::MissingContent { cid: *cid });
        }
    }
    Ok(())
}
```

Add tests:

```rust
    #[test]
    fn walk_blob_returns_single_cid() {
        let mut store = MemoryBlobStore::new();
        let data = b"small blob";
        let cid = store.insert(data).unwrap();
        let blobs = walk(&cid, &store).unwrap();
        assert_eq!(blobs, vec![cid]);
    }

    #[test]
    fn walk_bundle_returns_blobs_in_order() {
        let mut store = MemoryBlobStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let blobs = walk(&root, &store).unwrap();

        // Should have multiple blobs, all of type Blob.
        assert!(blobs.len() > 1);
        for cid in &blobs {
            assert_eq!(cid.cid_type(), CidType::Blob);
        }
    }

    #[test]
    fn walk_missing_cid_returns_error() {
        let store = MemoryBlobStore::new();
        // Ingest into a different store, then walk from empty store.
        let mut other_store = MemoryBlobStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut other_store).unwrap();

        let result = walk(&root, &store);
        assert!(matches!(result, Err(ContentError::MissingContent { .. })));
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content walk`
Expected: 3 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/dag.rs
git commit -m "feat(content): add walk() — depth-first Merkle DAG traversal"
```

---

### Task 4: `reassemble()` — Root CID to Original Data

**Files:**
- Modify: `crates/harmony-content/src/dag.rs`

**Context:** `reassemble()` calls `walk()` to get ordered blob CIDs, fetches each from the store, and concatenates into `Vec<u8>`. For root bundles with inline metadata, pre-allocates using `total_size`. The critical property: `reassemble(ingest(data)) == data`.

**Step 1: Write the implementation and tests**

Add to `crates/harmony-content/src/dag.rs` (after `walk_recursive()`, before `#[cfg(test)]`):

```rust
/// Reassemble original data from a Merkle DAG root CID.
///
/// Walks the DAG to collect blob CIDs in order, fetches each blob from the
/// store, and concatenates them. If the root is a bundle with inline metadata,
/// pre-allocates the output buffer using the total file size.
///
/// Guarantees: `reassemble(ingest(data)) == data` for all non-empty inputs.
pub fn reassemble(
    root_cid: &ContentId,
    store: &dyn BlobStore,
) -> Result<Vec<u8>, ContentError> {
    // Try to pre-allocate using inline metadata if available.
    let capacity = estimate_size(root_cid, store);
    let mut output = Vec::with_capacity(capacity);

    let blob_cids = walk(root_cid, store)?;
    for cid in &blob_cids {
        let data = store
            .get(cid)
            .ok_or(ContentError::MissingContent { cid: *cid })?;
        output.extend_from_slice(data);
    }

    Ok(output)
}

/// Estimate total data size from inline metadata (if present).
/// Returns 0 if unavailable — Vec will grow dynamically.
fn estimate_size(root_cid: &ContentId, store: &dyn BlobStore) -> usize {
    if let CidType::Bundle(_) = root_cid.cid_type() {
        if let Some(data) = store.get(root_cid) {
            if let Ok(entries) = bundle::parse_bundle(data) {
                if let Some(first) = entries.first() {
                    if first.cid_type() == CidType::InlineMetadata {
                        if let Ok((total_size, _, _, _)) = first.parse_inline_metadata() {
                            return total_size as usize;
                        }
                    }
                }
            }
        }
    } else if let CidType::Blob = root_cid.cid_type() {
        return root_cid.payload_size() as usize;
    }
    0
}
```

Add tests:

```rust
    #[test]
    fn round_trip_small_data() {
        let mut store = MemoryBlobStore::new();
        let data = b"hello round trip";
        let root = ingest(data.as_slice(), &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_medium_data() {
        let mut store = MemoryBlobStore::new();
        let data: Vec<u8> = (0..2048).map(|i| (i * 37 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_large_data() {
        let mut store = MemoryBlobStore::new();
        // 8KB+ should produce two+ bundle levels with small config.
        let data: Vec<u8> = (0..8192).map(|i| (i * 41 % 256) as u8).collect();
        let root = ingest(&data, &test_config(), &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }

    #[test]
    fn round_trip_exact_max_chunk() {
        let mut store = MemoryBlobStore::new();
        let config = test_config();
        let data = vec![0xCD; config.max_chunk];
        let root = ingest(&data, &config, &mut store).unwrap();
        let recovered = reassemble(&root, &store).unwrap();
        assert_eq!(recovered, data);
    }
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content round_trip`
Expected: 4 tests PASS

**Step 3: Commit**

```bash
git add crates/harmony-content/src/dag.rs
git commit -m "feat(content): add reassemble() with round-trip guarantee"
```

---

### Task 5: Structural Sharing + Full Suite Verification

**Files:**
- Modify: `crates/harmony-content/src/dag.rs`

**Context:** The final test verifies deduplication across versions — the core value proposition of content-addressed storage. Two versions of data with a small edit should share most blobs in the store.

**Step 1: Write the test**

```rust
    #[test]
    fn structural_sharing_across_versions() {
        let mut store = MemoryBlobStore::new();
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
```

**Step 2: Run the full test suite**

Run: `cargo test -p harmony-content`
Expected: All tests PASS (77 existing + ~14 new dag tests)

Run: `cargo test --workspace`
Expected: All workspace tests PASS

Run: `cargo clippy -p harmony-content`
Expected: No warnings

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 3: Commit**

```bash
git add crates/harmony-content/src/dag.rs
git commit -m "test(content): add structural sharing and full DAG pipeline tests"
```

# Merkle DAG Builder and Walker Design

**Goal:** Implement the end-to-end content pipeline: `ingest(data) → root CID`, `walk(root_cid) → ordered blob CIDs`, `reassemble(root_cid) → original data`, with the round-trip guarantee `reassemble(ingest(data)) == data`.

**Architecture:** Three free functions in a new `dag.rs` module operating on `&dyn BlobStore`. Composes existing building blocks: `chunk_all()` for splitting, `BlobStore` for storage, `BundleBuilder` for tree construction, `parse_bundle()` for traversal. No struct, no state — pure functions. Sans-I/O.

---

## `ingest()` — Data to Root CID

```rust
pub fn ingest(
    data: &[u8],
    config: &ChunkerConfig,
    store: &mut dyn BlobStore,
) -> Result<ContentId, ContentError>
```

1. Empty data → `ContentError::EmptyData`.
2. Chunk with `chunk_all(data, config)`.
3. Single chunk → `store.insert(data)`, return bare blob CID (no bundle wrapper).
4. Multiple chunks:
   - Insert each chunk → collect blob CIDs.
   - Group into bundles of up to `MAX_BUNDLE_ENTRIES` (32,767) via `BundleBuilder`.
   - Root-level bundle gets inline metadata: `total_size=data.len()`, `chunk_count`, `timestamp=0`, `mime=[0;8]`.
   - If multiple L1 bundles, group into L2, repeat until single root remains.
   - Return root CID.

Timestamp and MIME are placeholders (zeros). Higher-level API can set them later.

## `walk()` — Root CID to Ordered Blob CIDs

```rust
pub fn walk(
    root_cid: &ContentId,
    store: &dyn BlobStore,
) -> Result<Vec<ContentId>, ContentError>
```

1. Blob root → `vec![*root_cid]`.
2. Bundle root → fetch, parse, depth-first left-to-right:
   - Blob child → append to result.
   - Bundle child → recurse.
   - InlineMetadata child → skip.
3. Missing CID → `ContentError::MissingContent { cid }`.
4. Unexpected CID type → error.

Max recursion depth is 7 (enforced by CID type system). No recursion limit needed.

## `reassemble()` — Root CID to Original Data

```rust
pub fn reassemble(
    root_cid: &ContentId,
    store: &dyn BlobStore,
) -> Result<Vec<u8>, ContentError>
```

1. `walk(root_cid, store)` → ordered blob CIDs.
2. Fetch each blob, concatenate into `Vec<u8>`.
3. Pre-allocate using inline metadata `total_size` if available.

Core property: `reassemble(ingest(data)) == data` for all non-empty inputs.

## Error Handling

Two new `ContentError` variants:

- `EmptyData` — ingest given zero-length data.
- `MissingContent { cid: ContentId }` — walk encounters CID not in store.

All other errors propagate from existing operations (chunk_all, insert, BundleBuilder::build, parse_bundle).

## Testing (~15 tests)

Using small chunker config (min=64, avg=128, max=256) and MemoryBlobStore.

**Round-trip:** small (50B, single blob), medium (2KB, one bundle level), large (8KB+, two+ levels), exact boundary (256B).

**Structural:** single blob returns blob CID not bundle, multi-chunk returns bundle CID, walk on blob returns vec of one, walk preserves order.

**Errors:** ingest empty → EmptyData, walk missing CID → MissingContent.

**Deduplication:** two versions with small edit, second ingest adds fewer blobs.

**Metadata:** multi-chunk root has InlineMetadata with correct total_size and chunk_count.

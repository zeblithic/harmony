# Blob Store & Bundle Construction Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the blob store and bundle construction layer — the data primitives that compose content-addressed blobs into recursive Merkle DAGs.

**Architecture:** Two new modules in the existing `harmony-content` crate: `blob.rs` (content-addressed key-value store) and `bundle.rs` (zero-copy bundle parsing + BundleBuilder). The `BlobStore` trait abstracts storage; `MemoryBlobStore` is the in-memory reference implementation. Bundles are flat arrays of 32-byte `ContentId` structs — zero-copy parseable via `#[repr(C)]` reinterpret. `BundleBuilder` collects child CIDs, optionally prepends inline metadata, validates depth, and produces the bundle bytes + CID.

**Tech Stack:** Rust, `harmony-crypto` (SHA-256), `harmony-content::cid` (ContentId, CidType), `std::collections::HashMap`

---

### Task 1: `BlobStore` trait and `MemoryBlobStore`

**Files:**
- Create: `crates/harmony-content/src/blob.rs`
- Modify: `crates/harmony-content/src/lib.rs` (add `pub mod blob;`)

**Step 1: Write the failing tests**

Create `crates/harmony-content/src/blob.rs` with tests first, then the implementation above them:

```rust
use std::collections::HashMap;

use crate::cid::ContentId;
use crate::error::ContentError;

/// A content-addressed store for blob data.
pub trait BlobStore {
    /// Insert raw blob data, returning the blob's ContentId.
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError>;

    /// Store data under a pre-computed CID (used for bundles).
    fn store(&mut self, cid: ContentId, data: Vec<u8>);

    /// Retrieve data by CID.
    fn get(&self, cid: &ContentId) -> Option<&[u8]>;

    /// Check if a CID exists in the store.
    fn contains(&self, cid: &ContentId) -> bool;
}

/// In-memory content-addressed store backed by a HashMap.
pub struct MemoryBlobStore {
    data: HashMap<ContentId, Vec<u8>>,
}

impl MemoryBlobStore {
    pub fn new() -> Self {
        MemoryBlobStore {
            data: HashMap::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl Default for MemoryBlobStore {
    fn default() -> Self {
        Self::new()
    }
}

impl BlobStore for MemoryBlobStore {
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        let cid = ContentId::for_blob(data)?;
        self.data.entry(cid).or_insert_with(|| data.to_vec());
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.data.entry(cid).or_insert(data);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.data.get(cid).map(|v| v.as_slice())
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.data.contains_key(cid)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::CidType;

    #[test]
    fn insert_and_get_round_trip() {
        let mut store = MemoryBlobStore::new();
        let data = b"hello harmony blob store";
        let cid = store.insert(data).unwrap();
        assert_eq!(cid.cid_type(), CidType::Blob);
        assert_eq!(cid.payload_size(), data.len() as u32);
        assert_eq!(store.get(&cid).unwrap(), data);
    }

    #[test]
    fn duplicate_insert_returns_same_cid() {
        let mut store = MemoryBlobStore::new();
        let data = b"duplicate data";
        let cid1 = store.insert(data).unwrap();
        let cid2 = store.insert(data).unwrap();
        assert_eq!(cid1, cid2);
        assert_eq!(store.len(), 1);
    }

    #[test]
    fn get_unknown_returns_none() {
        let store = MemoryBlobStore::new();
        let cid = ContentId::for_blob(b"not stored").unwrap();
        assert!(store.get(&cid).is_none());
        assert!(!store.contains(&cid));
    }

    #[test]
    fn contains_reflects_state() {
        let mut store = MemoryBlobStore::new();
        let cid = store.insert(b"exists").unwrap();
        assert!(store.contains(&cid));
    }

    #[test]
    fn store_raw_for_bundle_data() {
        let mut store = MemoryBlobStore::new();
        let blob_a = ContentId::for_blob(b"aaa").unwrap();
        let blob_b = ContentId::for_blob(b"bbb").unwrap();

        // Build bundle bytes manually
        let mut bundle_bytes = Vec::new();
        bundle_bytes.extend_from_slice(&blob_a.to_bytes());
        bundle_bytes.extend_from_slice(&blob_b.to_bytes());
        let bundle_cid =
            ContentId::for_bundle(&bundle_bytes, &[blob_a, blob_b]).unwrap();

        store.store(bundle_cid, bundle_bytes.clone());
        assert_eq!(store.get(&bundle_cid).unwrap(), bundle_bytes.as_slice());
    }
}
```

**Step 2: Add module to `lib.rs`**

Modify `crates/harmony-content/src/lib.rs`:

```rust
pub mod blob;
pub mod cid;
pub mod error;
```

**Step 3: Run tests to verify they pass**

Run: `cargo test -p harmony-content`
Expected: All tests PASS (both old cid tests + new blob tests).

**Step 4: Run clippy**

Run: `cargo clippy -p harmony-content`
Expected: No warnings.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/blob.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add BlobStore trait and MemoryBlobStore"
```

---

### Task 2: Zero-copy bundle parsing and depth validation

**Files:**
- Create: `crates/harmony-content/src/bundle.rs`
- Modify: `crates/harmony-content/src/lib.rs` (add `pub mod bundle;`)
- Modify: `crates/harmony-content/src/error.rs` (add `InvalidBundleLength` variant)

**Step 1: Add the error variant**

Add to `ContentError` in `crates/harmony-content/src/error.rs`:

```rust
    #[error("invalid bundle length: {len} is not a multiple of 32")]
    InvalidBundleLength { len: usize },
```

**Step 2: Write `bundle.rs` with parsing and validation**

Create `crates/harmony-content/src/bundle.rs`:

```rust
use crate::cid::{ContentId, CONTENT_HASH_LEN};
use crate::error::ContentError;

/// Size of a single ContentId in bytes.
pub const CID_SIZE: usize = 32;

/// Maximum number of CIDs that fit in a single bundle (1MB payload / 32 bytes).
pub const MAX_BUNDLE_ENTRIES: usize = crate::cid::MAX_PAYLOAD_SIZE / CID_SIZE;

/// Parse a bundle's raw bytes into a slice of ContentIds (zero-copy).
///
/// # Safety invariants (all verified at compile time or by assertions):
/// - `ContentId` is `#[repr(C)]` with fields `[u8; 28]` + `[u8; 4]` = 32 bytes
/// - `ContentId` has alignment 1 (only byte-array fields)
/// - Input length is checked to be a multiple of 32
pub fn parse_bundle(data: &[u8]) -> Result<&[ContentId], ContentError> {
    if data.len() % CID_SIZE != 0 {
        return Err(ContentError::InvalidBundleLength { len: data.len() });
    }

    // Compile-time guarantee that ContentId is exactly 32 bytes.
    const _: () = assert!(std::mem::size_of::<ContentId>() == CID_SIZE);
    const _: () = assert!(std::mem::align_of::<ContentId>() == 1);

    let count = data.len() / CID_SIZE;

    // SAFETY: ContentId is #[repr(C)], exactly 32 bytes, alignment 1 (byte arrays only).
    // Input is checked to be a multiple of 32 bytes. &[u8] has alignment 1, matching
    // ContentId's alignment requirement.
    let cids = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const ContentId, count) };
    Ok(cids)
}

/// Validate that all children in a bundle have depth strictly less than the parent.
///
/// Returns `Ok(())` if valid, `Err(DepthViolation)` if any child has depth >= parent_depth.
pub fn validate_bundle_depth(
    parent_depth: u8,
    children: &[ContentId],
) -> Result<(), ContentError> {
    for child in children {
        let child_depth = child.cid_type().depth();
        if child_depth >= parent_depth {
            return Err(ContentError::DepthViolation {
                child: child_depth,
                parent: parent_depth,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::CidType;

    #[test]
    fn parse_valid_bundle() {
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&blob_a.to_bytes());
        bytes.extend_from_slice(&blob_b.to_bytes());

        let cids = parse_bundle(&bytes).unwrap();
        assert_eq!(cids.len(), 2);
        assert_eq!(cids[0], blob_a);
        assert_eq!(cids[1], blob_b);
    }

    #[test]
    fn parse_empty_bundle() {
        let cids = parse_bundle(&[]).unwrap();
        assert!(cids.is_empty());
    }

    #[test]
    fn parse_rejects_non_aligned_length() {
        let result = parse_bundle(&[0u8; 33]);
        assert!(matches!(
            result,
            Err(ContentError::InvalidBundleLength { len: 33 })
        ));
    }

    #[test]
    fn validate_depth_accepts_valid_sparse_tree() {
        // L4 bundle containing only L2 children — valid (sparse)
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));

        // L4 parent should accept L2 children
        assert!(validate_bundle_depth(4, &[l2, l2]).is_ok());
    }

    #[test]
    fn validate_depth_rejects_equal_depth() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        assert_eq!(l1.cid_type(), CidType::Bundle(1));

        // L1 parent cannot contain L1 children (depth not strictly less)
        assert!(validate_bundle_depth(1, &[l1]).is_err());
    }

    #[test]
    fn validate_depth_rejects_deeper_child() {
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();
        assert_eq!(l2.cid_type(), CidType::Bundle(2));

        // L1 parent cannot contain L2 child
        assert!(validate_bundle_depth(1, &[l2]).is_err());
    }

    #[test]
    fn validate_depth_accepts_mixed_children() {
        // L3 bundle containing [blob, L1, L2] — valid (mixed depths, all < 3)
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let l1_bytes = blob.to_bytes().to_vec();
        let l1 = ContentId::for_bundle(&l1_bytes, &[blob]).unwrap();
        let l2_bytes = l1.to_bytes().to_vec();
        let l2 = ContentId::for_bundle(&l2_bytes, &[l1]).unwrap();

        assert!(validate_bundle_depth(3, &[blob, l1, l2]).is_ok());
    }
}
```

**Step 3: Add module to `lib.rs`**

Modify `crates/harmony-content/src/lib.rs`:

```rust
pub mod blob;
pub mod bundle;
pub mod cid;
pub mod error;
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-content`
Expected: No warnings.

**Step 6: Commit**

```bash
git add crates/harmony-content/src/bundle.rs crates/harmony-content/src/lib.rs crates/harmony-content/src/error.rs
git commit -m "feat(content): add zero-copy bundle parsing and depth validation"
```

---

### Task 3: `BundleBuilder`

**Files:**
- Modify: `crates/harmony-content/src/bundle.rs`
- Modify: `crates/harmony-content/src/error.rs` (add `EmptyBundle` variant)

**Step 1: Add the error variant**

Add to `ContentError` in `crates/harmony-content/src/error.rs`:

```rust
    #[error("cannot build an empty bundle")]
    EmptyBundle,
```

**Step 2: Write the `BundleBuilder` struct**

Add to `crates/harmony-content/src/bundle.rs` (above the `#[cfg(test)]` block):

```rust
/// Builder for constructing bundles from child CIDs.
///
/// Collects child CIDs, optionally prepends an inline metadata CID,
/// validates depth constraints, and produces the bundle bytes + CID.
pub struct BundleBuilder {
    children: Vec<ContentId>,
    metadata: Option<ContentId>,
}

impl BundleBuilder {
    /// Create a new empty builder.
    pub fn new() -> Self {
        BundleBuilder {
            children: Vec::new(),
            metadata: None,
        }
    }

    /// Add a child CID to the bundle.
    pub fn add(&mut self, cid: ContentId) -> &mut Self {
        self.children.push(cid);
        self
    }

    /// Set inline metadata for a root bundle.
    ///
    /// The metadata CID will be prepended as the first entry.
    pub fn with_metadata(
        &mut self,
        total_size: u64,
        chunk_count: u32,
        timestamp: u64,
        mime: [u8; 8],
    ) -> &mut Self {
        self.metadata = Some(ContentId::inline_metadata(
            total_size,
            chunk_count,
            timestamp,
            mime,
        ));
        self
    }

    /// Build the bundle, returning the raw bytes and the bundle's CID.
    ///
    /// The bundle's depth is `max(child_depths) + 1`. If inline metadata
    /// was set, it is prepended as the first CID in the bundle bytes
    /// (metadata CIDs have depth 0, so they don't affect bundle depth).
    pub fn build(&self) -> Result<(Vec<u8>, ContentId), ContentError> {
        if self.children.is_empty() {
            return Err(ContentError::EmptyBundle);
        }

        // Collect all entries: optional metadata + children
        let mut entries: Vec<ContentId> = Vec::new();
        if let Some(meta) = &self.metadata {
            entries.push(*meta);
        }
        entries.extend_from_slice(&self.children);

        // Serialize to bytes
        let mut bundle_bytes =
            Vec::with_capacity(entries.len() * CID_SIZE);
        for cid in &entries {
            bundle_bytes.extend_from_slice(&cid.to_bytes());
        }

        // Compute bundle CID (depth based on all entries including metadata)
        let bundle_cid =
            ContentId::for_bundle(&bundle_bytes, &entries)?;

        Ok((bundle_bytes, bundle_cid))
    }

    /// Return the number of child CIDs (not counting metadata).
    pub fn len(&self) -> usize {
        self.children.len()
    }

    /// Check if the builder has no children.
    pub fn is_empty(&self) -> bool {
        self.children.is_empty()
    }
}

impl Default for BundleBuilder {
    fn default() -> Self {
        Self::new()
    }
}
```

**Step 3: Add tests for BundleBuilder**

Add to the `tests` module in `bundle.rs`:

```rust
    #[test]
    fn builder_basic_two_blobs() {
        let blob_a = ContentId::for_blob(b"chunk a").unwrap();
        let blob_b = ContentId::for_blob(b"chunk b").unwrap();

        let mut builder = BundleBuilder::new();
        builder.add(blob_a).add(blob_b);

        let (bytes, cid) = builder.build().unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1));
        assert_eq!(cid.payload_size(), 64); // 2 * 32
        assert_eq!(bytes.len(), 64);

        // Parse back and verify
        let parsed = parse_bundle(&bytes).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0], blob_a);
        assert_eq!(parsed[1], blob_b);
    }

    #[test]
    fn builder_with_metadata() {
        let blob = ContentId::for_blob(b"data").unwrap();
        let mut builder = BundleBuilder::new();
        builder
            .add(blob)
            .with_metadata(1000, 1, 0, *b"text/pln");

        let (bytes, cid) = builder.build().unwrap();
        assert_eq!(cid.cid_type(), CidType::Bundle(1));
        // 2 entries: metadata + blob
        assert_eq!(bytes.len(), 64);

        let parsed = parse_bundle(&bytes).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].cid_type(), CidType::InlineMetadata);
        assert_eq!(parsed[1], blob);
    }

    #[test]
    fn builder_depth_cascade() {
        let blob = ContentId::for_blob(b"leaf").unwrap();

        // L1 bundle
        let mut b1 = BundleBuilder::new();
        b1.add(blob);
        let (_, l1_cid) = b1.build().unwrap();
        assert_eq!(l1_cid.cid_type(), CidType::Bundle(1));

        // L2 bundle wrapping the L1
        let mut b2 = BundleBuilder::new();
        b2.add(l1_cid);
        let (_, l2_cid) = b2.build().unwrap();
        assert_eq!(l2_cid.cid_type(), CidType::Bundle(2));
    }

    #[test]
    fn builder_rejects_empty() {
        let builder = BundleBuilder::new();
        assert!(matches!(
            builder.build(),
            Err(ContentError::EmptyBundle)
        ));
    }

    #[test]
    fn builder_rejects_depth_overflow() {
        // Build up to depth 7 then try to wrap
        let blob = ContentId::for_blob(b"leaf").unwrap();
        let mut current = blob;
        for _ in 0..7 {
            let mut b = BundleBuilder::new();
            b.add(current);
            let (_, cid) = b.build().unwrap();
            current = cid;
        }
        assert_eq!(current.cid_type(), CidType::Bundle(7));

        let mut b = BundleBuilder::new();
        b.add(current);
        assert!(b.build().is_err());
    }
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 5: Run clippy**

Run: `cargo clippy -p harmony-content`
Expected: No warnings.

**Step 6: Commit**

```bash
git add crates/harmony-content/src/bundle.rs crates/harmony-content/src/error.rs
git commit -m "feat(content): add BundleBuilder with metadata and depth validation"
```

---

### Task 4: Integration tests — full blob + bundle pipeline

**Files:**
- Modify: `crates/harmony-content/src/bundle.rs` (add integration tests to test module)

These tests verify the full pipeline: insert blobs into a store, build a bundle, store the bundle, retrieve it, parse it, and verify the children match.

**Step 1: Add integration tests**

Add to the `tests` module in `bundle.rs`:

```rust
    use crate::blob::{BlobStore, MemoryBlobStore};

    #[test]
    fn full_pipeline_store_blobs_build_bundle_retrieve() {
        let mut store = MemoryBlobStore::new();

        // Insert blobs
        let cid_a = store.insert(b"chunk alpha").unwrap();
        let cid_b = store.insert(b"chunk beta").unwrap();
        let cid_c = store.insert(b"chunk gamma").unwrap();

        // Build bundle
        let mut builder = BundleBuilder::new();
        builder.add(cid_a).add(cid_b).add(cid_c);
        let (bundle_bytes, bundle_cid) = builder.build().unwrap();

        // Store bundle
        store.store(bundle_cid, bundle_bytes);

        // Retrieve and parse
        let retrieved = store.get(&bundle_cid).unwrap();
        let children = parse_bundle(retrieved).unwrap();
        assert_eq!(children.len(), 3);
        assert_eq!(children[0], cid_a);
        assert_eq!(children[1], cid_b);
        assert_eq!(children[2], cid_c);

        // Verify each child blob is still retrievable
        assert_eq!(store.get(&cid_a).unwrap(), b"chunk alpha");
        assert_eq!(store.get(&cid_b).unwrap(), b"chunk beta");
        assert_eq!(store.get(&cid_c).unwrap(), b"chunk gamma");
    }

    #[test]
    fn full_pipeline_with_metadata_root_bundle() {
        let mut store = MemoryBlobStore::new();

        let cid_a = store.insert(b"part 1").unwrap();
        let cid_b = store.insert(b"part 2").unwrap();

        let mut builder = BundleBuilder::new();
        builder
            .add(cid_a)
            .add(cid_b)
            .with_metadata(12, 2, 1709337600000, *b"text/pln");

        let (bundle_bytes, bundle_cid) = builder.build().unwrap();
        store.store(bundle_cid, bundle_bytes);

        let retrieved = store.get(&bundle_cid).unwrap();
        let entries = parse_bundle(retrieved).unwrap();

        // First entry is inline metadata
        assert_eq!(entries[0].cid_type(), CidType::InlineMetadata);
        let (total_size, chunk_count, ts, mime) =
            entries[0].parse_inline_metadata().unwrap();
        assert_eq!(total_size, 12);
        assert_eq!(chunk_count, 2);
        assert_eq!(ts, 1709337600000);
        assert_eq!(&mime, b"text/pln");

        // Remaining entries are the blobs
        assert_eq!(entries[1], cid_a);
        assert_eq!(entries[2], cid_b);
    }

    #[test]
    fn two_level_bundle_tree() {
        let mut store = MemoryBlobStore::new();

        // Create 4 blobs
        let cids: Vec<ContentId> = (0..4)
            .map(|i| store.insert(format!("chunk {i}").as_bytes()).unwrap())
            .collect();

        // L1 bundle: first 2 blobs
        let mut b1 = BundleBuilder::new();
        b1.add(cids[0]).add(cids[1]);
        let (b1_bytes, b1_cid) = b1.build().unwrap();
        assert_eq!(b1_cid.cid_type(), CidType::Bundle(1));
        store.store(b1_cid, b1_bytes);

        // L1 bundle: last 2 blobs
        let mut b2 = BundleBuilder::new();
        b2.add(cids[2]).add(cids[3]);
        let (b2_bytes, b2_cid) = b2.build().unwrap();
        store.store(b2_cid, b2_bytes);

        // L2 root bundle: two L1 bundles
        let mut root_builder = BundleBuilder::new();
        root_builder.add(b1_cid).add(b2_cid);
        let (root_bytes, root_cid) = root_builder.build().unwrap();
        assert_eq!(root_cid.cid_type(), CidType::Bundle(2));
        store.store(root_cid, root_bytes);

        // Walk the tree: root → L1s → blobs
        let root_children = parse_bundle(store.get(&root_cid).unwrap()).unwrap();
        assert_eq!(root_children.len(), 2);

        let l1a_children =
            parse_bundle(store.get(&root_children[0]).unwrap()).unwrap();
        assert_eq!(l1a_children.len(), 2);
        assert_eq!(l1a_children[0], cids[0]);
        assert_eq!(l1a_children[1], cids[1]);

        let l1b_children =
            parse_bundle(store.get(&root_children[1]).unwrap()).unwrap();
        assert_eq!(l1b_children[0], cids[2]);
        assert_eq!(l1b_children[1], cids[3]);

        // All leaf data is correct
        assert_eq!(store.get(&cids[0]).unwrap(), b"chunk 0");
        assert_eq!(store.get(&cids[3]).unwrap(), b"chunk 3");
    }
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content`
Expected: All tests PASS.

**Step 3: Run full quality gates**

Run: `cargo test -p harmony-content && cargo clippy -p harmony-content && cargo fmt --all -- --check`
Expected: All pass.

**Step 4: Run workspace tests**

Run: `cargo test --workspace`
Expected: All pass, no regressions.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/bundle.rs
git commit -m "test(content): add integration tests for blob store + bundle pipeline"
```

---

### Summary

| Task | What it builds | Tests |
|---|---|---|
| 1 | `BlobStore` trait, `MemoryBlobStore` | 5 |
| 2 | Zero-copy `parse_bundle`, `validate_bundle_depth` | 7 |
| 3 | `BundleBuilder` with metadata + depth validation | 5 |
| 4 | Full pipeline integration tests (store + bundle tree walk) | 3 |

Total: ~20 tests across 4 commits.

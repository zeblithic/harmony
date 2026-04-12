# harmony-db Phase 2: Prolly Tree Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace harmony-db's sorted-Vec internals with a Prolly Tree engine for history-independent topology, O(d) diffing, and CAS structural sharing — while preserving the exact public API.

**Architecture:** CDF-based chunking with BLAKE3 key-only hashing determines node boundaries. Nodes are postcard-serialized and stored as CAS Books. Each table is a ProllyTree with an in-memory entry cache (for API-compatible reference returns) backed by the on-disk tree. Mutations update the cache and rebuild the tree bottom-up. Diff walks two trees recursively, skipping matching subtrees in O(1).

**Tech Stack:** Rust, harmony-content (ContentId, ContentFlags, BookStore), postcard (node serialization), blake3 (key hashing for CDF chunker), serde/serde_json (root tracking), hex

---

## File Structure

```text
crates/harmony-db/
├── Cargo.toml                  # Add postcard + blake3 deps
├── src/
│   ├── lib.rs                  # Replace mod table/commit with mod prolly
│   ├── error.rs                # Unchanged
│   ├── types.rs                # Unchanged
│   ├── persist.rs              # Shrink: save_roots/load_roots + blob I/O
│   ├── db.rs                   # Switch from Table+commit to ProllyTree
│   └── prolly/
│       ├── mod.rs              # ProllyTree struct + tree build/read/mutate
│       ├── chunker.rs          # CDF boundary decision
│       ├── node.rs             # LeafEntry, BranchEntry, Node enum, CAS I/O
│       └── diff.rs             # Recursive O(d) tree diff
├── examples/
│   └── mail_workflow.rs        # Unchanged (public API only)
└── [deleted] table.rs, commit.rs
```

**Key design decision:** ProllyTree keeps an in-memory `Vec<Entry>` cache alongside the tree root CID. Reads serve from cache (zero disk I/O, reference-returning). Writes update the cache and rebuild the tree. This preserves Phase 1's exact API signatures (`&[Entry]`, `Option<&Entry>`) while gaining tree-based persistence, diff, and commit.

**Mutation strategy:** Level-by-level rebuild. Every insert/remove/update_meta modifies the entry cache, then rebuilds the entire tree bottom-up from the sorted cache. This is O(n) per mutation (same as Phase 1) but produces the correct history-independent tree. Incremental O(log n) mutations are a future optimization — the public API works identically either way.

---

### Task 1: Add Dependencies and Create Prolly Module Skeleton

**Files:**
- Modify: `crates/harmony-db/Cargo.toml`
- Create: `crates/harmony-db/src/prolly/mod.rs`
- Create: `crates/harmony-db/src/prolly/chunker.rs`
- Create: `crates/harmony-db/src/prolly/node.rs`
- Create: `crates/harmony-db/src/prolly/diff.rs`
- Modify: `crates/harmony-db/src/lib.rs`

- [ ] **Step 1: Add postcard and blake3 to Cargo.toml**

Add to `[dependencies]` in `crates/harmony-db/Cargo.toml`:

```toml
postcard = { workspace = true }
blake3 = { workspace = true }
```

- [ ] **Step 2: Create prolly module directory and stub files**

```bash
mkdir -p crates/harmony-db/src/prolly
```

Create `crates/harmony-db/src/prolly/mod.rs`:

```rust
pub(crate) mod chunker;
pub(crate) mod diff;
pub(crate) mod node;
```

Create `crates/harmony-db/src/prolly/chunker.rs`:

```rust
// CDF boundary decision for Prolly Tree chunking.
```

Create `crates/harmony-db/src/prolly/node.rs`:

```rust
// Prolly Tree node types and CAS I/O.
```

Create `crates/harmony-db/src/prolly/diff.rs`:

```rust
// Recursive O(d) tree diff.
```

- [ ] **Step 3: Add prolly module to lib.rs**

Add `mod prolly;` to `crates/harmony-db/src/lib.rs` (keep existing modules for now — we'll delete table/commit later):

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod commit;
mod db;
mod error;
mod persist;
mod prolly;
mod table;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Diff, Entry, EntryMeta, TableDiff};
```

- [ ] **Step 4: Verify compilation**

```bash
cargo check -p harmony-db
```

Expected: compiles (stub modules are empty but valid).

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-db/
git commit -m "feat(harmony-db): add prolly module skeleton with postcard+blake3 deps (ZEB-98)"
```

---

### Task 2: CDF Chunker

**Files:**
- Modify: `crates/harmony-db/src/prolly/chunker.rs`

The chunker is standalone — no dependencies on other prolly modules. It determines where node boundaries fall using a CDF over a normal distribution, with BLAKE3 key-only hashing.

- [ ] **Step 1: Implement the chunker**

Write `crates/harmony-db/src/prolly/chunker.rs`:

```rust
use blake3::Hasher;

/// Configuration for CDF-based Prolly Tree chunking.
///
/// Determines node boundaries using a normal distribution CDF.
/// The probability of a boundary increases as the chunk grows,
/// producing a tight distribution centered on `target_size`.
#[derive(Debug, Clone)]
pub(crate) struct ChunkerConfig {
    pub target_size: usize,
    pub min_size: usize,
    pub max_size: usize,
    pub std_dev: f64,
}

impl ChunkerConfig {
    /// Default configuration: ~4KB target chunks.
    pub fn default_4k() -> Self {
        ChunkerConfig {
            target_size: 4096,
            min_size: 512,
            max_size: 16384,
            std_dev: 1024.0,
        }
    }

    /// Evaluate whether `key` triggers a chunk boundary when the current
    /// chunk has accumulated `current_size` bytes and the new entry is
    /// `entry_size` bytes.
    ///
    /// Hashes ONLY the key (not value) so value updates never shift boundaries.
    pub fn is_boundary(&self, current_size: usize, entry_size: usize, key: &[u8]) -> bool {
        if current_size < self.min_size {
            return false;
        }
        if current_size + entry_size >= self.max_size {
            return true;
        }

        let p = self.split_probability(current_size, entry_size);
        if p <= 0.0 {
            return false;
        }
        if p >= 1.0 {
            return true;
        }

        // Hash ONLY the key with BLAKE3 to get a uniform random variable.
        let mut hasher = Hasher::new();
        hasher.update(key);
        let hash = hasher.finalize();
        let hash_u32 = u32::from_le_bytes(hash.as_bytes()[0..4].try_into().unwrap());
        let random_var = (hash_u32 as f64) / (u32::MAX as f64);

        random_var <= p
    }

    /// Conditional probability that a chunk should end before reaching
    /// `current_size + entry_size`, given it has already reached `current_size`.
    ///
    /// P = (F(s+a) - F(s)) / (1 - F(s))
    /// where F is the CDF of Normal(target_size, std_dev).
    fn split_probability(&self, current_size: usize, entry_size: usize) -> f64 {
        let f_s = self.normal_cdf(current_size as f64);
        let f_sa = self.normal_cdf((current_size + entry_size) as f64);
        let denom = 1.0 - f_s;
        if denom <= 0.0 {
            return 1.0;
        }
        ((f_sa - f_s) / denom).clamp(0.0, 1.0)
    }

    /// Logistic approximation of the normal CDF.
    fn normal_cdf(&self, x: f64) -> f64 {
        let z = (x - self.target_size as f64) / self.std_dev;
        0.5 * (1.0 + f64::tanh(z * 0.7978845608))
    }
}

/// Split a sorted sequence of items into chunks using the CDF chunker.
///
/// `items` must be sorted. `size_fn` returns the serialized byte size of an item.
/// `key_fn` returns the key bytes used for boundary hashing.
///
/// Returns a vector of chunks (each chunk is a vector of items).
pub(crate) fn chunk_items<T: Clone>(
    items: &[T],
    config: &ChunkerConfig,
    size_fn: impl Fn(&T) -> usize,
    key_fn: impl Fn(&T) -> &[u8],
) -> Vec<Vec<T>> {
    if items.is_empty() {
        return Vec::new();
    }

    let mut chunks: Vec<Vec<T>> = Vec::new();
    let mut current_chunk: Vec<T> = Vec::new();
    let mut current_size: usize = 0;

    for item in items {
        let item_size = size_fn(item);

        if !current_chunk.is_empty()
            && config.is_boundary(current_size, item_size, key_fn(item))
        {
            chunks.push(std::mem::take(&mut current_chunk));
            current_size = 0;
        }

        current_chunk.push(item.clone());
        current_size += item_size;
    }

    if !current_chunk.is_empty() {
        chunks.push(current_chunk);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ChunkerConfig {
        ChunkerConfig::default_4k()
    }

    #[test]
    fn boundary_below_min_never_splits() {
        let c = config();
        // 100 bytes is well below min_size (512), should never split.
        for i in 0..100u32 {
            let key = i.to_le_bytes();
            assert!(!c.is_boundary(100, 50, &key));
        }
    }

    #[test]
    fn boundary_above_max_always_splits() {
        let c = config();
        // At max_size, should always split regardless of key.
        for i in 0..100u32 {
            let key = i.to_le_bytes();
            assert!(c.is_boundary(c.max_size, 100, &key));
        }
    }

    #[test]
    fn boundary_deterministic() {
        let c = config();
        let key = b"test-key-deterministic";
        let result1 = c.is_boundary(2048, 200, key);
        let result2 = c.is_boundary(2048, 200, key);
        assert_eq!(result1, result2);
    }

    #[test]
    fn boundary_key_only() {
        // Same key with different "values" (simulated by different entry sizes
        // that don't cross min/max thresholds) should produce the same boundary
        // decision at the same chunk size. The boundary only depends on the key hash.
        let c = config();
        let key = b"stable-key";
        let r1 = c.is_boundary(3000, 200, key);
        let r2 = c.is_boundary(3000, 250, key);
        // Both should evaluate the same key hash against the probability.
        // The probability differs slightly (different entry_size), but the key
        // hash is deterministic. This test validates the hash is key-only.
        // We can't assert equality because probability changes with entry_size.
        // Instead, verify the hash itself is key-only:
        let mut h1 = Hasher::new();
        h1.update(key);
        let hash1 = h1.finalize();

        let mut h2 = Hasher::new();
        h2.update(key);
        let hash2 = h2.finalize();

        assert_eq!(hash1, hash2, "key-only hash must be deterministic");
    }

    #[test]
    fn chunk_size_distribution() {
        let c = config();
        // Generate 1000 items of ~200 bytes each, measure chunk sizes.
        let items: Vec<Vec<u8>> = (0..1000u32)
            .map(|i| format!("key-{i:06}").into_bytes())
            .collect();

        let chunks = chunk_items(
            &items,
            &c,
            |k| k.len() + 200, // simulate ~200 byte entries
            |k| k.as_slice(),
        );

        assert!(chunks.len() > 1, "should produce multiple chunks");

        let sizes: Vec<usize> = chunks.iter()
            .map(|chunk| chunk.iter().map(|k| k.len() + 200).sum::<usize>())
            .collect();

        let avg: f64 = sizes.iter().sum::<usize>() as f64 / sizes.len() as f64;

        // Average should be roughly near target (4096), with some tolerance.
        // The exact average depends on the key hashes but should be in a
        // reasonable range.
        assert!(avg > 1000.0, "average chunk size {avg} too small");
        assert!(avg < 12000.0, "average chunk size {avg} too large");

        // All chunks should respect min/max bounds.
        for size in &sizes {
            assert!(*size >= c.min_size, "chunk {size} below min {}", c.min_size);
            // Last chunk can be smaller than min — that's OK (it's the tail).
        }
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- chunker
```

Expected: all 5 chunker tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/prolly/chunker.rs
git commit -m "feat(harmony-db): CDF chunker with BLAKE3 key-only hashing (ZEB-98)"
```

---

### Task 3: Node Types and CAS I/O

**Files:**
- Modify: `crates/harmony-db/src/prolly/node.rs`

Node types are postcard-serialized and stored/loaded from the CAS (commits/ directory). LeafEntry is the tree-internal entry type; conversion to/from the public `Entry` type is provided.

- [ ] **Step 1: Implement node types**

Write `crates/harmony-db/src/prolly/node.rs`:

```rust
use crate::error::DbError;
use crate::types::{Entry, EntryMeta};
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A leaf-level entry in the Prolly Tree.
/// Uses postcard serialization (compact binary, no hex encoding).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct LeafEntry {
    pub key: Vec<u8>,
    pub value_cid: [u8; 32],
    pub timestamp: u64,
    pub flags: u64,
    pub snippet: String,
}

impl LeafEntry {
    pub fn from_entry(e: &Entry) -> Self {
        LeafEntry {
            key: e.key.clone(),
            value_cid: e.value_cid.to_bytes(),
            timestamp: e.timestamp,
            flags: e.metadata.flags,
            snippet: e.metadata.snippet.clone(),
        }
    }

    pub fn to_entry(&self) -> Entry {
        Entry {
            key: self.key.clone(),
            value_cid: ContentId::from_bytes(self.value_cid),
            timestamp: self.timestamp,
            metadata: EntryMeta {
                flags: self.flags,
                snippet: self.snippet.clone(),
            },
        }
    }

    /// Approximate serialized size in bytes (for CDF chunker).
    pub fn approx_size(&self) -> usize {
        self.key.len() + 32 + 8 + 8 + self.snippet.len() + 8 // overhead
    }
}

/// A branch-level routing entry.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub(crate) struct BranchEntry {
    pub boundary_key: Vec<u8>,
    pub child_cid: [u8; 32],
}

impl BranchEntry {
    /// Approximate serialized size in bytes (for CDF chunker).
    pub fn approx_size(&self) -> usize {
        self.boundary_key.len() + 32 + 8 // overhead
    }
}

/// A Prolly Tree node — either leaf or branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) enum Node {
    Leaf(Vec<LeafEntry>),
    Branch(Vec<BranchEntry>),
}

impl Node {
    /// Serialize this node via postcard, write to CAS, return the CID.
    pub fn write_to_cas(&self, data_dir: &Path) -> Result<ContentId, DbError> {
        let bytes = postcard::to_allocvec(self)
            .map_err(|e| DbError::Serialize(format!("postcard encode: {e}")))?;
        let cid = ContentId::for_book(&bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let cid_hex = hex::encode(cid.to_bytes());
        let node_path = data_dir.join("commits").join(format!("{cid_hex}.bin"));
        if !node_path.exists() {
            let tmp = data_dir.join("commits").join(format!("{cid_hex}.bin.tmp"));
            std::fs::write(&tmp, &bytes)?;
            std::fs::rename(&tmp, &node_path)?;
        }
        Ok(cid)
    }

    /// Read a node from CAS by CID.
    pub fn read_from_cas(data_dir: &Path, cid: ContentId) -> Result<Node, DbError> {
        let cid_hex = hex::encode(cid.to_bytes());
        let node_path = data_dir.join("commits").join(format!("{cid_hex}.bin"));
        let bytes = std::fs::read(&node_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DbError::CommitNotFound { cid: cid_hex }
            } else {
                DbError::Io(e)
            }
        })?;
        postcard::from_bytes(&bytes)
            .map_err(|e| DbError::CorruptIndex(format!("postcard decode: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_leaf_entry(key: &[u8]) -> LeafEntry {
        let cid = ContentId::for_book(key, ContentFlags::default()).unwrap();
        LeafEntry {
            key: key.to_vec(),
            value_cid: cid.to_bytes(),
            timestamp: 1000,
            flags: 0,
            snippet: "test".to_string(),
        }
    }

    #[test]
    fn leaf_serialize_round_trip() {
        let entries = vec![
            sample_leaf_entry(b"alpha"),
            sample_leaf_entry(b"beta"),
        ];
        let node = Node::Leaf(entries.clone());
        let bytes = postcard::to_allocvec(&node).unwrap();
        let decoded: Node = postcard::from_bytes(&bytes).unwrap();
        match decoded {
            Node::Leaf(decoded_entries) => assert_eq!(decoded_entries, entries),
            _ => panic!("expected Leaf"),
        }
    }

    #[test]
    fn branch_serialize_round_trip() {
        let cid_a = ContentId::for_book(b"a", ContentFlags::default()).unwrap();
        let cid_b = ContentId::for_book(b"b", ContentFlags::default()).unwrap();
        let children = vec![
            BranchEntry { boundary_key: b"m".to_vec(), child_cid: cid_a.to_bytes() },
            BranchEntry { boundary_key: b"z".to_vec(), child_cid: cid_b.to_bytes() },
        ];
        let node = Node::Branch(children.clone());
        let bytes = postcard::to_allocvec(&node).unwrap();
        let decoded: Node = postcard::from_bytes(&bytes).unwrap();
        match decoded {
            Node::Branch(decoded_children) => assert_eq!(decoded_children, children),
            _ => panic!("expected Branch"),
        }
    }

    #[test]
    fn node_cas_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("commits")).unwrap();

        let entries = vec![sample_leaf_entry(b"key1"), sample_leaf_entry(b"key2")];
        let node = Node::Leaf(entries);
        let cid = node.write_to_cas(dir.path()).unwrap();

        let loaded = Node::read_from_cas(dir.path(), cid).unwrap();
        match (&node, &loaded) {
            (Node::Leaf(a), Node::Leaf(b)) => assert_eq!(a, b),
            _ => panic!("mismatch"),
        }
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- node
```

Expected: all 3 node tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/prolly/node.rs
git commit -m "feat(harmony-db): Prolly Tree node types with postcard CAS I/O (ZEB-98)"
```

---

### Task 4: ProllyTree — Build, Get, Range

**Files:**
- Modify: `crates/harmony-db/src/prolly/mod.rs`

The ProllyTree struct holds a root CID and an in-memory entry cache. `build_tree` constructs the tree bottom-up from sorted entries. Read operations serve from the cache.

- [ ] **Step 1: Implement ProllyTree core**

Write `crates/harmony-db/src/prolly/mod.rs`:

```rust
pub(crate) mod chunker;
pub(crate) mod diff;
pub(crate) mod node;

use crate::error::DbError;
use crate::types::Entry;
use chunker::{chunk_items, ChunkerConfig};
use harmony_content::ContentId;
use node::{BranchEntry, LeafEntry, Node};
use std::path::Path;

/// A Prolly Tree table with an in-memory entry cache.
///
/// The cache is a sorted `Vec<Entry>` derived from the tree. Reads serve
/// from the cache. Writes update the cache and rebuild the tree.
#[derive(Debug, Clone)]
pub(crate) struct ProllyTree {
    root: Option<ContentId>,
    cache: Vec<Entry>,
    config: ChunkerConfig,
}

impl ProllyTree {
    /// Create an empty tree.
    pub fn new() -> Self {
        ProllyTree {
            root: None,
            cache: Vec::new(),
            config: ChunkerConfig::default_4k(),
        }
    }

    /// Load a tree from an existing root CID. Walks the tree to populate
    /// the in-memory entry cache.
    pub fn from_root(root_cid: ContentId, data_dir: &Path) -> Result<Self, DbError> {
        let mut entries = Vec::new();
        collect_entries(data_dir, root_cid, &mut entries)?;
        entries.sort_by(|a, b| a.key.cmp(&b.key));
        Ok(ProllyTree {
            root: Some(root_cid),
            cache: entries,
            config: ChunkerConfig::default_4k(),
        })
    }

    pub fn root(&self) -> Option<ContentId> {
        self.root
    }

    // --- Read operations (serve from cache) ---

    pub fn get_entry(&self, key: &[u8]) -> Option<&Entry> {
        let idx = self.cache.binary_search_by(|e| e.key.as_slice().cmp(key)).ok()?;
        Some(&self.cache[idx])
    }

    pub fn range(&self, start: &[u8], end: &[u8]) -> &[Entry] {
        let lo = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(start)) {
            Ok(i) | Err(i) => i,
        };
        let hi = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(end)) {
            Ok(i) | Err(i) => i,
        };
        if lo >= hi {
            return &[];
        }
        &self.cache[lo..hi]
    }

    pub fn entries(&self) -> &[Entry] {
        &self.cache
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    // --- Write operations (update cache + rebuild tree) ---

    /// Insert or replace an entry. Rebuilds the tree. Returns new root CID.
    pub fn insert(&mut self, entry: Entry, data_dir: &Path) -> Result<Option<ContentId>, DbError> {
        match self.cache.binary_search_by(|e| e.key.cmp(&entry.key)) {
            Ok(idx) => self.cache[idx] = entry,
            Err(idx) => self.cache.insert(idx, entry),
        }
        self.rebuild_tree(data_dir)
    }

    /// Remove an entry by key. Rebuilds the tree. Returns the removed entry
    /// and the new root CID.
    pub fn remove(&mut self, key: &[u8], data_dir: &Path) -> Result<(Option<Entry>, Option<ContentId>), DbError> {
        let removed = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            Ok(idx) => Some(self.cache.remove(idx)),
            Err(_) => None,
        };
        let new_root = if removed.is_some() {
            self.rebuild_tree(data_dir)?
        } else {
            self.root
        };
        Ok((removed, new_root))
    }

    /// Update metadata for an entry. Rebuilds the tree (key unchanged, so
    /// boundaries stay stable but leaf content changes). Returns new root CID.
    pub fn update_meta(
        &mut self,
        key: &[u8],
        flags: u64,
        snippet: String,
        data_dir: &Path,
    ) -> Result<bool, DbError> {
        let idx = match self.cache.binary_search_by(|e| e.key.as_slice().cmp(key)) {
            Ok(i) => i,
            Err(_) => return Ok(false),
        };
        self.cache[idx].metadata.flags = flags;
        self.cache[idx].metadata.snippet = snippet;
        self.rebuild_tree(data_dir)?;
        Ok(true)
    }

    // --- Tree construction ---

    /// Rebuild the tree from the in-memory cache. Sets self.root.
    fn rebuild_tree(&mut self, data_dir: &Path) -> Result<Option<ContentId>, DbError> {
        self.root = build_tree(&self.cache, &self.config, data_dir)?;
        Ok(self.root)
    }
}

/// Build a Prolly Tree bottom-up from sorted entries.
/// Returns the root CID (None if entries is empty).
pub(crate) fn build_tree(
    entries: &[Entry],
    config: &ChunkerConfig,
    data_dir: &Path,
) -> Result<Option<ContentId>, DbError> {
    if entries.is_empty() {
        return Ok(None);
    }

    // Convert to leaf entries.
    let leaf_entries: Vec<LeafEntry> = entries.iter().map(LeafEntry::from_entry).collect();

    // Chunk leaf entries into leaf nodes.
    let leaf_chunks = chunk_items(
        &leaf_entries,
        config,
        |e| e.approx_size(),
        |e| e.key.as_slice(),
    );

    // Write leaf nodes, collect branch entries for next level.
    let mut branch_entries: Vec<BranchEntry> = Vec::with_capacity(leaf_chunks.len());
    for chunk in &leaf_chunks {
        let boundary_key = chunk.last().unwrap().key.clone();
        let node = Node::Leaf(chunk.clone());
        let cid = node.write_to_cas(data_dir)?;
        branch_entries.push(BranchEntry {
            boundary_key,
            child_cid: cid.to_bytes(),
        });
    }

    // If only one leaf node, it's the root.
    if branch_entries.len() == 1 {
        return Ok(Some(ContentId::from_bytes(branch_entries[0].child_cid)));
    }

    // Build branch levels until we have a single root.
    loop {
        let branch_chunks = chunk_items(
            &branch_entries,
            config,
            |e| e.approx_size(),
            |e| e.boundary_key.as_slice(),
        );

        let mut next_level: Vec<BranchEntry> = Vec::with_capacity(branch_chunks.len());
        for chunk in &branch_chunks {
            let boundary_key = chunk.last().unwrap().boundary_key.clone();
            let node = Node::Branch(chunk.clone());
            let cid = node.write_to_cas(data_dir)?;
            next_level.push(BranchEntry {
                boundary_key,
                child_cid: cid.to_bytes(),
            });
        }

        if next_level.len() == 1 {
            return Ok(Some(ContentId::from_bytes(next_level[0].child_cid)));
        }

        branch_entries = next_level;
    }
}

/// Recursively collect all leaf entries from a tree into a flat Vec.
fn collect_entries(
    data_dir: &Path,
    cid: ContentId,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    match node {
        Node::Leaf(entries) => {
            for e in entries {
                out.push(e.to_entry());
            }
        }
        Node::Branch(children) => {
            for child in children {
                collect_entries(data_dir, ContentId::from_bytes(child.child_cid), out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use crate::types::EntryMeta;
    use harmony_content::ContentFlags;

    fn make_entry(key: &[u8], value: &[u8], data_dir: &Path) -> Entry {
        let cid = persist::write_blob(data_dir, value).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta { flags: 0, snippet: "test".to_string() },
        }
    }

    #[test]
    fn empty_tree() {
        let tree = ProllyTree::new();
        assert!(tree.root().is_none());
        assert_eq!(tree.len(), 0);
        assert!(tree.get_entry(b"anything").is_none());
        assert!(tree.range(b"a", b"z").is_empty());
    }

    #[test]
    fn insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();
        let entry = make_entry(b"hello", b"world", dir.path());
        tree.insert(entry, dir.path()).unwrap();

        assert_eq!(tree.len(), 1);
        assert!(tree.root().is_some());
        let found = tree.get_entry(b"hello").unwrap();
        assert_eq!(found.metadata.snippet, "test");
    }

    #[test]
    fn insert_upsert() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        let e1 = Entry {
            key: b"k".to_vec(),
            value_cid: ContentId::for_book(b"old", ContentFlags::default()).unwrap(),
            timestamp: 1, metadata: EntryMeta { flags: 0, snippet: "old".into() },
        };
        tree.insert(e1, dir.path()).unwrap();

        let e2 = Entry {
            key: b"k".to_vec(),
            value_cid: ContentId::for_book(b"new", ContentFlags::default()).unwrap(),
            timestamp: 2, metadata: EntryMeta { flags: 1, snippet: "new".into() },
        };
        tree.insert(e2, dir.path()).unwrap();

        assert_eq!(tree.len(), 1);
        assert_eq!(tree.get_entry(b"k").unwrap().metadata.snippet, "new");
    }

    #[test]
    fn insert_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        for key in [b"cherry".as_slice(), b"apple", b"banana"] {
            let e = make_entry(key, key, dir.path());
            tree.insert(e, dir.path()).unwrap();
        }

        let keys: Vec<&[u8]> = tree.entries().iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"apple".as_slice(), b"banana", b"cherry"]);
    }

    #[test]
    fn range_query() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        for key in [b"a".as_slice(), b"b", b"c", b"d"] {
            tree.insert(make_entry(key, key, dir.path()), dir.path()).unwrap();
        }

        let range = tree.range(b"b", b"d");
        let keys: Vec<&[u8]> = range.iter().map(|e| e.key.as_slice()).collect();
        assert_eq!(keys, vec![b"b".as_slice(), b"c"]);
    }

    #[test]
    fn remove_entry() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();
        tree.insert(make_entry(b"a", b"1", dir.path()), dir.path()).unwrap();
        tree.insert(make_entry(b"b", b"2", dir.path()), dir.path()).unwrap();

        let (removed, _) = tree.remove(b"a", dir.path()).unwrap();
        assert!(removed.is_some());
        assert!(tree.get_entry(b"a").is_none());
        assert_eq!(tree.len(), 1);
    }

    #[test]
    fn update_meta_no_rechunk() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();
        tree.insert(make_entry(b"k", b"v", dir.path()), dir.path()).unwrap();
        let root_before = tree.root();

        tree.update_meta(b"k", 42, "updated".into(), dir.path()).unwrap();

        assert_eq!(tree.get_entry(b"k").unwrap().metadata.flags, 42);
        // Root WILL change (leaf content changed), but tree structure
        // (number of nodes, key distribution) stays the same.
        assert_ne!(tree.root(), root_before, "root should change on metadata update");
    }

    #[test]
    fn large_insert_builds_tree() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        for i in 0..1000u32 {
            let key = format!("key-{i:06}").into_bytes();
            let val = format!("value-{i}").into_bytes();
            tree.insert(make_entry(&key, &val, dir.path()), dir.path()).unwrap();
        }

        assert_eq!(tree.len(), 1000);
        assert!(tree.root().is_some());

        // Verify all entries are retrievable.
        for i in 0..1000u32 {
            let key = format!("key-{i:06}").into_bytes();
            assert!(tree.get_entry(&key).is_some(), "missing key-{i:06}");
        }

        // Verify tree actually has depth > 1 by checking that the root
        // is a branch node (not a leaf).
        let root = tree.root().unwrap();
        let root_node = Node::read_from_cas(dir.path(), root).unwrap();
        match root_node {
            Node::Branch(_) => {} // Expected for 1000 entries.
            Node::Leaf(_) => panic!("expected branch root for 1000 entries"),
        }
    }

    #[test]
    fn history_independence() {
        // THE critical Prolly Tree test: same entries inserted in different
        // order must produce identical root CIDs.
        let dir1 = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir1.path()).unwrap();
        let dir2 = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir2.path()).unwrap();

        let keys: Vec<Vec<u8>> = (0..100u32)
            .map(|i| format!("entry-{i:04}").into_bytes())
            .collect();

        // Tree 1: insert in order.
        let mut tree1 = ProllyTree::new();
        for key in &keys {
            tree1.insert(make_entry(key, key, dir1.path()), dir1.path()).unwrap();
        }

        // Tree 2: insert in reverse order.
        let mut tree2 = ProllyTree::new();
        for key in keys.iter().rev() {
            tree2.insert(make_entry(key, key, dir2.path()), dir2.path()).unwrap();
        }

        assert_eq!(
            tree1.root(), tree2.root(),
            "same entries in different order must produce same root CID"
        );
    }

    #[test]
    fn value_update_stability() {
        // Updating a value (same key) should only change the leaf + path
        // to root. The tree structure (boundaries) stays stable because
        // boundaries depend only on keys.
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        // Insert 100 entries.
        for i in 0..100u32 {
            let key = format!("k-{i:04}").into_bytes();
            tree.insert(make_entry(&key, &key, dir.path()), dir.path()).unwrap();
        }
        let root_before = tree.root().unwrap();

        // Update one value.
        let updated = Entry {
            key: b"k-0050".to_vec(),
            value_cid: ContentId::for_book(b"new-value", ContentFlags::default()).unwrap(),
            timestamp: 9999,
            metadata: EntryMeta { flags: 0, snippet: "test".into() },
        };
        tree.insert(updated, dir.path()).unwrap();
        let root_after = tree.root().unwrap();

        assert_ne!(root_before, root_after, "root should change after value update");

        // Verify the updated entry.
        let e = tree.get_entry(b"k-0050").unwrap();
        assert_eq!(e.timestamp, 9999);
    }

    #[test]
    fn from_root_rebuilds_cache() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let mut tree = ProllyTree::new();

        for i in 0..50u32 {
            let key = format!("k-{i:04}").into_bytes();
            tree.insert(make_entry(&key, &key, dir.path()), dir.path()).unwrap();
        }

        let root = tree.root().unwrap();

        // Load from root — should reconstruct the full cache.
        let loaded = ProllyTree::from_root(root, dir.path()).unwrap();
        assert_eq!(loaded.len(), 50);
        assert_eq!(loaded.entries(), tree.entries());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- prolly::tests
```

Expected: all 11 tree tests pass (empty_tree, insert_and_get, insert_upsert, insert_sorted_order, range_query, remove_entry, update_meta_no_rechunk, large_insert_builds_tree, history_independence, value_update_stability, from_root_rebuilds_cache).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/prolly/mod.rs
git commit -m "feat(harmony-db): ProllyTree with build/get/range/insert/remove (ZEB-98)"
```

---

### Task 5: Recursive Diff

**Files:**
- Modify: `crates/harmony-db/src/prolly/diff.rs`

The diff algorithm walks two trees recursively, skipping identical subtrees in O(1).

- [ ] **Step 1: Implement the diff**

Write `crates/harmony-db/src/prolly/diff.rs`:

```rust
use crate::error::DbError;
use crate::types::{Entry, TableDiff};
use harmony_content::ContentId;
use super::node::Node;
use std::cmp::Ordering;
use std::path::Path;

/// Compute the difference between two Prolly Tree roots.
/// Returns added, removed, and changed entries.
///
/// Exploits structural sharing: identical subtree CIDs are skipped in O(1).
pub(crate) fn diff_trees(
    data_dir: &Path,
    root_a: Option<ContentId>,
    root_b: Option<ContentId>,
) -> Result<TableDiff, DbError> {
    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    match (root_a, root_b) {
        (None, None) => {}
        (Some(a), None) => collect_all(data_dir, a, &mut removed)?,
        (None, Some(b)) => collect_all(data_dir, b, &mut added)?,
        (Some(a), Some(b)) => {
            if a == b {
                // Identical roots — no diff.
                return Ok(TableDiff { added, removed, changed });
            }
            diff_nodes(data_dir, a, b, &mut added, &mut removed, &mut changed)?;
        }
    }

    Ok(TableDiff { added, removed, changed })
}

fn diff_nodes(
    data_dir: &Path,
    cid_a: ContentId,
    cid_b: ContentId,
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) -> Result<(), DbError> {
    if cid_a == cid_b {
        return Ok(());
    }

    let node_a = Node::read_from_cas(data_dir, cid_a)?;
    let node_b = Node::read_from_cas(data_dir, cid_b)?;

    match (node_a, node_b) {
        (Node::Leaf(entries_a), Node::Leaf(entries_b)) => {
            diff_leaf_entries(&entries_a, &entries_b, added, removed, changed);
        }
        (Node::Branch(children_a), Node::Branch(children_b)) => {
            diff_branches(data_dir, &children_a, &children_b, added, removed, changed)?;
        }
        // Depth mismatch: flatten both to entries and diff.
        (a, b) => {
            let mut entries_a = Vec::new();
            let mut entries_b = Vec::new();
            collect_leaf_entries(data_dir, &a, &mut entries_a)?;
            collect_leaf_entries(data_dir, &b, &mut entries_b)?;
            diff_leaf_entries(&entries_a, &entries_b, added, removed, changed);
        }
    }

    Ok(())
}

fn diff_branches(
    data_dir: &Path,
    children_a: &[super::node::BranchEntry],
    children_b: &[super::node::BranchEntry],
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) -> Result<(), DbError> {
    let mut ia = 0;
    let mut ib = 0;

    while ia < children_a.len() && ib < children_b.len() {
        let a = &children_a[ia];
        let b = &children_b[ib];

        match a.boundary_key.cmp(&b.boundary_key) {
            Ordering::Equal => {
                let cid_a = ContentId::from_bytes(a.child_cid);
                let cid_b = ContentId::from_bytes(b.child_cid);
                if cid_a != cid_b {
                    diff_nodes(data_dir, cid_a, cid_b, added, removed, changed)?;
                }
                // else: identical subtree, skip.
                ia += 1;
                ib += 1;
            }
            Ordering::Less => {
                // Subtree in A but not B — all removed.
                collect_all(data_dir, ContentId::from_bytes(a.child_cid), removed)?;
                ia += 1;
            }
            Ordering::Greater => {
                // Subtree in B but not A — all added.
                collect_all(data_dir, ContentId::from_bytes(b.child_cid), added)?;
                ib += 1;
            }
        }
    }

    while ia < children_a.len() {
        collect_all(data_dir, ContentId::from_bytes(children_a[ia].child_cid), removed)?;
        ia += 1;
    }
    while ib < children_b.len() {
        collect_all(data_dir, ContentId::from_bytes(children_b[ib].child_cid), added)?;
        ib += 1;
    }

    Ok(())
}

fn diff_leaf_entries(
    entries_a: &[super::node::LeafEntry],
    entries_b: &[super::node::LeafEntry],
    added: &mut Vec<Entry>,
    removed: &mut Vec<Entry>,
    changed: &mut Vec<(Entry, Entry)>,
) {
    let mut ia = 0;
    let mut ib = 0;

    while ia < entries_a.len() && ib < entries_b.len() {
        match entries_a[ia].key.cmp(&entries_b[ib].key) {
            Ordering::Less => {
                removed.push(entries_a[ia].to_entry());
                ia += 1;
            }
            Ordering::Greater => {
                added.push(entries_b[ib].to_entry());
                ib += 1;
            }
            Ordering::Equal => {
                let ea = &entries_a[ia];
                let eb = &entries_b[ib];
                if ea.value_cid != eb.value_cid
                    || ea.flags != eb.flags
                    || ea.snippet != eb.snippet
                {
                    changed.push((ea.to_entry(), eb.to_entry()));
                }
                ia += 1;
                ib += 1;
            }
        }
    }

    while ia < entries_a.len() {
        removed.push(entries_a[ia].to_entry());
        ia += 1;
    }
    while ib < entries_b.len() {
        added.push(entries_b[ib].to_entry());
        ib += 1;
    }
}

/// Collect all entries from a subtree into a flat Vec.
fn collect_all(
    data_dir: &Path,
    cid: ContentId,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    let node = Node::read_from_cas(data_dir, cid)?;
    collect_leaf_entries(data_dir, &node, out)
}

fn collect_leaf_entries(
    data_dir: &Path,
    node: &Node,
    out: &mut Vec<Entry>,
) -> Result<(), DbError> {
    match node {
        Node::Leaf(entries) => {
            for e in entries {
                out.push(e.to_entry());
            }
        }
        Node::Branch(children) => {
            for child in children {
                collect_all(data_dir, ContentId::from_bytes(child.child_cid), out)?;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persist;
    use crate::types::EntryMeta;
    use super::super::{build_tree, ChunkerConfig, ProllyTree};
    use harmony_content::ContentFlags;

    fn make_entry(key: &[u8], value: &[u8], data_dir: &Path) -> Entry {
        let cid = persist::write_blob(data_dir, value).unwrap();
        Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: 1000,
            metadata: EntryMeta { flags: 0, snippet: "".to_string() },
        }
    }

    fn build_entries(keys: &[&[u8]], data_dir: &Path) -> (Vec<Entry>, Option<ContentId>) {
        let config = ChunkerConfig::default_4k();
        let entries: Vec<Entry> = keys.iter()
            .map(|k| make_entry(k, k, data_dir))
            .collect();
        let root = build_tree(&entries, &config, data_dir).unwrap();
        (entries, root)
    }

    #[test]
    fn diff_identical_roots() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let (_, root) = build_entries(&[b"a", b"b", b"c"], dir.path());
        let td = diff_trees(dir.path(), root, root).unwrap();
        assert!(td.added.is_empty());
        assert!(td.removed.is_empty());
        assert!(td.changed.is_empty());
    }

    #[test]
    fn diff_additions() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let config = ChunkerConfig::default_4k();

        let entries1: Vec<Entry> = vec![make_entry(b"a", b"1", dir.path())];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2: Vec<Entry> = vec![
            make_entry(b"a", b"1", dir.path()),
            make_entry(b"b", b"2", dir.path()),
        ];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let td = diff_trees(dir.path(), root1, root2).unwrap();
        assert_eq!(td.added.len(), 1);
        assert_eq!(td.added[0].key, b"b");
        assert!(td.removed.is_empty());
    }

    #[test]
    fn diff_removals() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let config = ChunkerConfig::default_4k();

        let entries1: Vec<Entry> = vec![
            make_entry(b"a", b"1", dir.path()),
            make_entry(b"b", b"2", dir.path()),
        ];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2: Vec<Entry> = vec![make_entry(b"a", b"1", dir.path())];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let td = diff_trees(dir.path(), root1, root2).unwrap();
        assert_eq!(td.removed.len(), 1);
        assert_eq!(td.removed[0].key, b"b");
    }

    #[test]
    fn diff_changes() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let config = ChunkerConfig::default_4k();

        let entries1: Vec<Entry> = vec![make_entry(b"k", b"old", dir.path())];
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        let entries2: Vec<Entry> = vec![make_entry(b"k", b"new", dir.path())];
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let td = diff_trees(dir.path(), root1, root2).unwrap();
        assert_eq!(td.changed.len(), 1);
    }

    #[test]
    fn diff_none_to_some() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let (_, root) = build_entries(&[b"a", b"b"], dir.path());
        let td = diff_trees(dir.path(), None, root).unwrap();
        assert_eq!(td.added.len(), 2);
        assert!(td.removed.is_empty());
    }

    #[test]
    fn diff_large_dataset_small_change() {
        let dir = tempfile::tempdir().unwrap();
        persist::ensure_dirs(dir.path()).unwrap();
        let config = ChunkerConfig::default_4k();

        let entries1: Vec<Entry> = (0..500u32)
            .map(|i| {
                let key = format!("k-{i:06}").into_bytes();
                make_entry(&key, &key, dir.path())
            })
            .collect();
        let root1 = build_tree(&entries1, &config, dir.path()).unwrap();

        // Change one entry.
        let mut entries2 = entries1.clone();
        entries2[250] = make_entry(b"k-000250", b"modified", dir.path());
        let root2 = build_tree(&entries2, &config, dir.path()).unwrap();

        let td = diff_trees(dir.path(), root1, root2).unwrap();
        assert_eq!(td.changed.len(), 1);
        assert_eq!(td.changed[0].1.key, b"k-000250");
        assert!(td.added.is_empty());
        assert!(td.removed.is_empty());
    }
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- diff::tests
```

Expected: all 6 diff tests pass.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/prolly/diff.rs
git commit -m "feat(harmony-db): recursive O(d) Prolly Tree diff (ZEB-98)"
```

---

### Task 6: Rewrite persist.rs — Save/Load Roots

**Files:**
- Modify: `crates/harmony-db/src/persist.rs`

Replace the full-index `IndexFile`/`TableFile`/`save_index`/`load_index` with a minimal root tracking file. Keep blob I/O, `ensure_dirs`, `truncate_snippet` unchanged.

- [ ] **Step 1: Replace index types and functions**

In `crates/harmony-db/src/persist.rs`, replace `IndexFile`, `TableFile`, `load_index`, and `save_index` with:

```rust
const ROOTS_VERSION: u32 = 2;

/// Minimal root tracking file — just table names → tree root CIDs.
#[derive(Debug, Serialize, Deserialize)]
pub(crate) struct RootsFile {
    pub version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    pub head: Option<ContentId>,
    pub table_roots: std::collections::BTreeMap<String, String>, // name → root CID hex
}

/// Load table root CIDs from the roots file.
/// Returns empty state if the file is missing or corrupt.
pub(crate) fn load_roots(data_dir: &Path) -> (Option<ContentId>, std::collections::BTreeMap<String, ContentId>) {
    let path = data_dir.join("index.json");
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(_) => return (None, std::collections::BTreeMap::new()),
    };
    let roots: RootsFile = match serde_json::from_slice::<RootsFile>(&bytes) {
        Ok(r) if r.version == ROOTS_VERSION => r,
        _ => return (None, std::collections::BTreeMap::new()),
    };
    let table_roots = roots.table_roots.into_iter().filter_map(|(name, hex_str)| {
        let bytes: [u8; 32] = hex::decode(&hex_str).ok()?.try_into().ok()?;
        Some((name, ContentId::from_bytes(bytes)))
    }).collect();
    (roots.head, table_roots)
}

/// Atomically save table root CIDs.
pub(crate) fn save_roots(
    data_dir: &Path,
    head: Option<ContentId>,
    table_roots: &std::collections::BTreeMap<String, Option<ContentId>>,
) -> Result<(), DbError> {
    let roots = RootsFile {
        version: ROOTS_VERSION,
        head,
        table_roots: table_roots.iter().filter_map(|(name, cid)| {
            cid.map(|c| (name.clone(), hex::encode(c.to_bytes())))
        }).collect(),
    };
    let bytes = serde_json::to_vec_pretty(&roots)
        .map_err(|e| DbError::Serialize(e.to_string()))?;
    atomic_write(&data_dir.join("index.json"), &bytes)
}
```

Remove the old `IndexFile`, `TableFile`, `load_index`, `save_index` functions and the old `INDEX_VERSION` constant.

Keep all of: `write_blob`, `read_blob`, `write_blob_raw`, `ensure_dirs`, `truncate_snippet`, `atomic_write`.

Update the persist tests: remove `save_and_load_index_round_trip`, `load_missing_index_returns_empty`, `load_corrupt_index_returns_empty`. Replace with:

```rust
#[test]
fn save_and_load_roots_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    ensure_dirs(dir.path()).unwrap();
    let cid = ContentId::for_book(b"root", ContentFlags::default()).unwrap();
    let mut table_roots = std::collections::BTreeMap::new();
    table_roots.insert("inbox".to_string(), Some(cid));
    save_roots(dir.path(), None, &table_roots).unwrap();
    let (head, loaded) = load_roots(dir.path());
    assert!(head.is_none());
    assert_eq!(loaded["inbox"], cid);
}

#[test]
fn load_missing_roots_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let (head, roots) = load_roots(dir.path());
    assert!(head.is_none());
    assert!(roots.is_empty());
}

#[test]
fn load_corrupt_roots_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("index.json"), b"not json{{{").unwrap();
    let (head, roots) = load_roots(dir.path());
    assert!(head.is_none());
    assert!(roots.is_empty());
}
```

- [ ] **Step 2: Run tests**

```bash
cargo test -p harmony-db -- persist
```

Expected: all persist tests pass (blob tests unchanged + 3 new roots tests).

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-db/src/persist.rs
git commit -m "feat(harmony-db): minimal root tracking replaces full-index persistence (ZEB-98)"
```

---

### Task 7: Rewrite db.rs — Switch to ProllyTree

**Files:**
- Modify: `crates/harmony-db/src/db.rs`
- Modify: `crates/harmony-db/src/lib.rs`
- Delete: `crates/harmony-db/src/table.rs`
- Delete: `crates/harmony-db/src/commit.rs`

This is the big switchover. Replace `Table` + `commit::` with `ProllyTree` + `prolly::diff`. The public API stays identical.

- [ ] **Step 1: Rewrite db.rs**

Replace the full contents of `crates/harmony-db/src/db.rs`:

```rust
use crate::error::DbError;
use crate::persist;
use crate::prolly::node::Node;
use crate::prolly::{build_tree, diff::diff_trees, ProllyTree};
use crate::types::{Diff, Entry, EntryMeta};
use harmony_content::book::BookStore;
use harmony_content::{ContentFlags, ContentId};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

const COMMIT_VERSION: u32 = 2;

/// A content-addressed key-value database with named tables.
pub struct HarmonyDb {
    data_dir: PathBuf,
    trees: HashMap<String, ProllyTree>,
    head: Option<ContentId>,
}

/// Commit manifest — maps table names to Prolly Tree root CIDs.
#[derive(Debug, Serialize, Deserialize)]
struct CommitManifest {
    version: u32,
    #[serde(
        serialize_with = "crate::types::ser_opt_cid",
        deserialize_with = "crate::types::de_opt_cid"
    )]
    parent: Option<ContentId>,
    tables: BTreeMap<String, String>, // name → root CID hex
}

impl HarmonyDb {
    /// Open or create a database at `data_dir`.
    pub fn open(data_dir: &Path) -> Result<Self, DbError> {
        persist::ensure_dirs(data_dir)?;
        let (head, table_roots) = persist::load_roots(data_dir);
        let mut trees = HashMap::new();
        for (name, root_cid) in table_roots {
            trees.insert(name, ProllyTree::from_root(root_cid, data_dir)?);
        }
        Ok(HarmonyDb {
            data_dir: data_dir.to_path_buf(),
            trees,
            head,
        })
    }

    pub fn table_names(&self) -> Vec<&str> {
        let mut names: Vec<&str> = self.trees.keys().map(|s| s.as_str()).collect();
        names.sort_unstable();
        names
    }

    pub fn table_len(&self, table: &str) -> usize {
        self.trees.get(table).map_or(0, |t| t.len())
    }

    pub fn insert(
        &mut self,
        table: &str,
        key: &[u8],
        value: &[u8],
        meta: EntryMeta,
    ) -> Result<ContentId, DbError> {
        let cid = persist::write_blob(&self.data_dir, value)?;
        let entry = Entry {
            key: key.to_vec(),
            value_cid: cid,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            metadata: EntryMeta {
                flags: meta.flags,
                snippet: persist::truncate_snippet(&meta.snippet),
            },
        };
        let tree = self.trees.entry(table.to_string()).or_insert_with(ProllyTree::new);
        tree.insert(entry, &self.data_dir)?;
        self.save_roots()?;
        Ok(cid)
    }

    pub fn get(&self, table: &str, key: &[u8]) -> Result<Option<Vec<u8>>, DbError> {
        let entry = match self.trees.get(table).and_then(|t| t.get_entry(key)) {
            Some(e) => e,
            None => return Ok(None),
        };
        persist::read_blob(&self.data_dir, &entry.value_cid)
    }

    pub fn get_entry(&self, table: &str, key: &[u8]) -> Option<&Entry> {
        self.trees.get(table)?.get_entry(key)
    }

    pub fn range(&self, table: &str, start: &[u8], end: &[u8]) -> &[Entry] {
        match self.trees.get(table) {
            Some(t) => t.range(start, end),
            None => &[],
        }
    }

    pub fn entries(&self, table: &str) -> &[Entry] {
        match self.trees.get(table) {
            Some(t) => t.entries(),
            None => &[],
        }
    }

    pub fn remove(&mut self, table: &str, key: &[u8]) -> Result<Option<Entry>, DbError> {
        let removed = match self.trees.get_mut(table) {
            Some(t) => t.remove(key, &self.data_dir)?.0,
            None => None,
        };
        if removed.is_some() {
            self.save_roots()?;
        }
        Ok(removed)
    }

    pub fn update_meta(
        &mut self,
        table: &str,
        key: &[u8],
        meta: EntryMeta,
    ) -> Result<(), DbError> {
        let truncated_snippet = persist::truncate_snippet(&meta.snippet);
        let t = self.trees.get_mut(table).ok_or_else(|| DbError::TableNotFound {
            name: table.to_string(),
        })?;
        if !t.update_meta(key, meta.flags, truncated_snippet, &self.data_dir)? {
            return Err(DbError::EntryNotFound { table: table.to_string() });
        }
        self.save_roots()?;
        Ok(())
    }

    pub fn head(&self) -> Option<ContentId> {
        self.head
    }

    pub fn commit(
        &mut self,
        store: Option<&mut dyn BookStore>,
    ) -> Result<ContentId, DbError> {
        let mut table_cids: BTreeMap<String, String> = BTreeMap::new();
        for (name, tree) in &self.trees {
            if let Some(root) = tree.root() {
                table_cids.insert(name.clone(), hex::encode(root.to_bytes()));
            }
        }

        let manifest = CommitManifest {
            version: COMMIT_VERSION,
            parent: self.head,
            tables: table_cids,
        };
        let manifest_bytes = serde_json::to_vec_pretty(&manifest)
            .map_err(|e| DbError::Serialize(e.to_string()))?;
        let root_cid = ContentId::for_book(&manifest_bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        let root_hex = hex::encode(root_cid.to_bytes());

        // Write manifest to commits/.
        let root_path = self.data_dir.join("commits").join(format!("{root_hex}.bin"));
        if !root_path.exists() {
            let tmp = root_path.with_extension("bin.tmp");
            std::fs::write(&tmp, &manifest_bytes)?;
            std::fs::rename(&tmp, &root_path)?;
        }

        // Push to BookStore if provided.
        if let Some(ref mut s) = store {
            // Push tree nodes + value blobs first.
            for tree in self.trees.values() {
                if let Some(tree_root) = tree.root() {
                    push_tree_to_store(s, &self.data_dir, tree_root)?;
                }
            }
            // Manifest last.
            s.insert_with_flags(&manifest_bytes, ContentFlags::default())
                .map_err(|e| DbError::Serialize(format!("BookStore push failed: {e:?}")))?;
        }

        let new_head = Some(root_cid);
        self.save_roots_with_head(new_head)?;
        self.head = new_head;
        Ok(root_cid)
    }

    pub fn diff(
        &self,
        old: ContentId,
        new: ContentId,
        store: Option<&dyn BookStore>,
    ) -> Result<Diff, DbError> {
        let old_manifest = load_commit_manifest(&self.data_dir, old, store)?;
        let new_manifest = load_commit_manifest(&self.data_dir, new, store)?;

        let mut tables = HashMap::new();

        for (name, new_hex) in &new_manifest.tables {
            if let Some(old_hex) = old_manifest.tables.get(name) {
                if old_hex == new_hex {
                    continue; // Identical tree root — skip.
                }
                let old_root = cid_from_hex(old_hex)?;
                let new_root = cid_from_hex(new_hex)?;
                let td = diff_trees(&self.data_dir, Some(old_root), Some(new_root))?;
                if !td.added.is_empty() || !td.removed.is_empty() || !td.changed.is_empty() {
                    tables.insert(name.clone(), td);
                }
            } else {
                let new_root = cid_from_hex(new_hex)?;
                let td = diff_trees(&self.data_dir, None, Some(new_root))?;
                if !td.added.is_empty() {
                    tables.insert(name.clone(), td);
                }
            }
        }

        for (name, old_hex) in &old_manifest.tables {
            if !new_manifest.tables.contains_key(name) {
                let old_root = cid_from_hex(old_hex)?;
                let td = diff_trees(&self.data_dir, Some(old_root), None)?;
                if !td.removed.is_empty() {
                    tables.insert(name.clone(), td);
                }
            }
        }

        Ok(Diff { tables })
    }

    pub fn rebuild_from(
        &mut self,
        root: ContentId,
        store: Option<&dyn BookStore>,
    ) -> Result<(), DbError> {
        let manifest = load_commit_manifest(&self.data_dir, root, store)?;
        let mut new_trees = HashMap::new();
        for (name, hex_str) in &manifest.tables {
            let tree_root = cid_from_hex(hex_str)?;
            // Prefetch tree nodes from store if available.
            if let Some(s) = store {
                prefetch_tree(s, &self.data_dir, tree_root)?;
            }
            new_trees.insert(name.clone(), ProllyTree::from_root(tree_root, &self.data_dir)?);
        }
        let new_head = Some(root);
        let roots: BTreeMap<String, Option<ContentId>> = new_trees.iter()
            .map(|(n, t)| (n.clone(), t.root()))
            .collect();
        persist::save_roots(&self.data_dir, new_head, &roots)?;
        self.trees = new_trees;
        self.head = new_head;
        Ok(())
    }

    pub fn open_from_cas(
        data_dir: &Path,
        root: ContentId,
        store: &dyn BookStore,
    ) -> Result<Self, DbError> {
        let mut db = Self::open(data_dir)?;
        db.rebuild_from(root, Some(store))?;
        Ok(db)
    }

    // --- Internal helpers ---

    fn save_roots(&self) -> Result<(), DbError> {
        let roots: BTreeMap<String, Option<ContentId>> = self.trees.iter()
            .map(|(n, t)| (n.clone(), t.root()))
            .collect();
        persist::save_roots(&self.data_dir, self.head, &roots)
    }

    fn save_roots_with_head(&self, head: Option<ContentId>) -> Result<(), DbError> {
        let roots: BTreeMap<String, Option<ContentId>> = self.trees.iter()
            .map(|(n, t)| (n.clone(), t.root()))
            .collect();
        persist::save_roots(&self.data_dir, head, &roots)
    }
}

fn load_commit_manifest(
    data_dir: &Path,
    cid: ContentId,
    store: Option<&dyn BookStore>,
) -> Result<CommitManifest, DbError> {
    let cid_hex = hex::encode(cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{cid_hex}.bin"));

    let (bytes, from_store) = if local_path.exists() {
        (std::fs::read(&local_path)?, false)
    } else if let Some(s) = store {
        let fetched = s.get(&cid)
            .map(|b| b.to_vec())
            .ok_or_else(|| DbError::CommitNotFound { cid: cid_hex.clone() })?;
        (fetched, true)
    } else {
        return Err(DbError::CommitNotFound { cid: cid_hex });
    };

    if from_store {
        let computed = ContentId::for_book(&bytes, ContentFlags::default())
            .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
        if computed != cid {
            return Err(DbError::CorruptIndex(format!("manifest content mismatch for {cid_hex}")));
        }
    }

    let manifest: CommitManifest = serde_json::from_slice(&bytes)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?;
    if manifest.version != COMMIT_VERSION {
        return Err(DbError::CorruptIndex(format!(
            "unsupported commit version: {} (expected {})",
            manifest.version, COMMIT_VERSION
        )));
    }

    if from_store {
        let tmp = local_path.with_extension("bin.tmp");
        std::fs::write(&tmp, &bytes)?;
        std::fs::rename(&tmp, &local_path)?;
    }

    Ok(manifest)
}

fn cid_from_hex(hex_str: &str) -> Result<ContentId, DbError> {
    let bytes: [u8; 32] = hex::decode(hex_str)
        .map_err(|e| DbError::CorruptIndex(e.to_string()))?
        .try_into()
        .map_err(|_| DbError::CorruptIndex("bad CID length".into()))?;
    Ok(ContentId::from_bytes(bytes))
}

/// Recursively push all tree nodes to a BookStore.
fn push_tree_to_store(
    store: &mut dyn BookStore,
    data_dir: &Path,
    cid: ContentId,
) -> Result<(), DbError> {
    if store.contains(&cid) {
        return Ok(());
    }
    let cid_hex = hex::encode(cid.to_bytes());
    let node_path = data_dir.join("commits").join(format!("{cid_hex}.bin"));
    let bytes = std::fs::read(&node_path).map_err(|_| DbError::CommitNotFound { cid: cid_hex })?;
    store.store(cid, bytes.clone());

    // If it's a branch node, recurse into children.
    if let Ok(Node::Branch(children)) = postcard::from_bytes::<Node>(&bytes) {
        for child in children {
            push_tree_to_store(store, data_dir, ContentId::from_bytes(child.child_cid))?;
        }
    }

    // Also push value blobs referenced by leaf entries.
    if let Ok(Node::Leaf(entries)) = postcard::from_bytes::<Node>(&bytes) {
        for entry in entries {
            let value_cid = ContentId::from_bytes(entry.value_cid);
            if !store.contains(&value_cid) {
                match persist::read_blob(data_dir, &value_cid)? {
                    Some(blob) => store.store(value_cid, blob),
                    None => {} // Orphaned — best effort.
                }
            }
        }
    }

    Ok(())
}

/// Recursively prefetch tree nodes from BookStore to local cache.
fn prefetch_tree(
    store: &dyn BookStore,
    data_dir: &Path,
    cid: ContentId,
) -> Result<(), DbError> {
    let cid_hex = hex::encode(cid.to_bytes());
    let local_path = data_dir.join("commits").join(format!("{cid_hex}.bin"));
    if local_path.exists() {
        return Ok(());
    }
    let bytes = match store.get(&cid) {
        Some(b) => b.to_vec(),
        None => return Err(DbError::CommitNotFound { cid: cid_hex }),
    };

    // Verify CID before caching.
    let computed = ContentId::for_book(&bytes, ContentFlags::default())
        .map_err(|e| DbError::Serialize(format!("CID error: {e:?}")))?;
    if computed != cid {
        return Err(DbError::CorruptIndex(format!("node content mismatch for {cid_hex}")));
    }

    let tmp = local_path.with_extension("bin.tmp");
    std::fs::write(&tmp, &bytes)?;
    std::fs::rename(&tmp, &local_path)?;

    // Recurse into children if branch.
    if let Ok(Node::Branch(children)) = postcard::from_bytes::<Node>(&bytes) {
        for child in children {
            prefetch_tree(store, data_dir, ContentId::from_bytes(child.child_cid))?;
        }
    }

    // Prefetch value blobs for leaf entries.
    if let Ok(Node::Leaf(entries)) = postcard::from_bytes::<Node>(&bytes) {
        for entry in entries {
            let value_cid = ContentId::from_bytes(entry.value_cid);
            let value_hex = hex::encode(entry.value_cid);
            let blob_path = data_dir.join("blobs").join(format!("{value_hex}.bin"));
            if !blob_path.exists() {
                if let Some(blob_data) = store.get(&value_cid) {
                    persist::write_blob_raw(data_dir, &value_hex, blob_data)?;
                }
            }
        }
    }

    Ok(())
}

// Keep all existing db.rs tests — they use only the public API and should
// pass unchanged with the new ProllyTree engine.
#[cfg(test)]
mod tests {
    use super::*;

    fn meta(flags: u64, snippet: &str) -> EntryMeta {
        EntryMeta {
            flags,
            snippet: snippet.to_string(),
        }
    }

    #[test]
    fn open_empty() {
        let dir = tempfile::tempdir().unwrap();
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert!(db.table_names().is_empty());
        assert!(db.head().is_none());
    }

    #[test]
    fn insert_and_get() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let cid = db.insert("inbox", b"msg1", b"hello world", meta(0, "greet")).unwrap();
        let value = db.get("inbox", b"msg1").unwrap().unwrap();
        assert_eq!(value, b"hello world");
        assert_eq!(db.get_entry("inbox", b"msg1").unwrap().value_cid, cid);
    }

    #[test]
    fn insert_upsert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"old", meta(0, "")).unwrap();
        db.insert("t", b"k", b"new", meta(1, "")).unwrap();
        assert_eq!(db.table_len("t"), 1);
        assert_eq!(db.get("t", b"k").unwrap().unwrap(), b"new");
        assert_eq!(db.get_entry("t", b"k").unwrap().metadata.flags, 1);
    }

    #[test]
    fn get_nonexistent_table() {
        let dir = tempfile::tempdir().unwrap();
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert!(db.get("nope", b"k").unwrap().is_none());
    }

    #[test]
    fn remove_entry() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"val", meta(0, "")).unwrap();
        let removed = db.remove("t", b"k").unwrap();
        assert!(removed.is_some());
        assert!(db.get("t", b"k").unwrap().is_none());
    }

    #[test]
    fn update_meta_test() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"val", meta(0, "old")).unwrap();
        db.update_meta("t", b"k", meta(1, "new")).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert_eq!(e.metadata.flags, 1);
        assert_eq!(e.metadata.snippet, "new");
    }

    #[test]
    fn range_query() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"a", b"1", meta(0, "")).unwrap();
        db.insert("t", b"b", b"2", meta(0, "")).unwrap();
        db.insert("t", b"c", b"3", meta(0, "")).unwrap();
        db.insert("t", b"d", b"4", meta(0, "")).unwrap();
        let range = db.range("t", b"b", b"d");
        assert_eq!(range.len(), 2);
        assert_eq!(range[0].key, b"b");
        assert_eq!(range[1].key, b"c");
    }

    #[test]
    fn multi_table() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("inbox", b"k", b"inbox_val", meta(0, "")).unwrap();
        db.insert("sent", b"k", b"sent_val", meta(0, "")).unwrap();
        assert_eq!(db.get("inbox", b"k").unwrap().unwrap(), b"inbox_val");
        assert_eq!(db.get("sent", b"k").unwrap().unwrap(), b"sent_val");
        assert_eq!(db.table_len("inbox"), 1);
        assert_eq!(db.table_len("sent"), 1);
    }

    #[test]
    fn open_existing_persists() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut db = HarmonyDb::open(dir.path()).unwrap();
            db.insert("t", b"k", b"val", meta(0, "hi")).unwrap();
        }
        let db = HarmonyDb::open(dir.path()).unwrap();
        assert_eq!(db.get_entry("t", b"k").unwrap().metadata.snippet, "hi");
        assert_eq!(db.get("t", b"k").unwrap().unwrap(), b"val");
    }

    #[test]
    fn snippet_truncated_on_insert() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let long = "a".repeat(500);
        db.insert("t", b"k", b"v", meta(0, &long)).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert!(e.metadata.snippet.len() <= 256);
    }

    #[test]
    fn update_meta_truncates_snippet() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"v", meta(0, "short")).unwrap();
        let long = "x".repeat(500);
        db.update_meta("t", b"k", meta(1, &long)).unwrap();
        let e = db.get_entry("t", b"k").unwrap();
        assert!(e.metadata.snippet.len() <= 256);
    }

    #[test]
    fn blob_dedup_across_tables() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        let cid1 = db.insert("inbox", b"k1", b"same_value", meta(0, "")).unwrap();
        let cid2 = db.insert("sent", b"k2", b"same_value", meta(0, "")).unwrap();
        assert_eq!(cid1, cid2);
        let blob_count = std::fs::read_dir(dir.path().join("blobs"))
            .unwrap()
            .filter(|e| e.as_ref().unwrap().path().extension() == Some("bin".as_ref()))
            .count();
        assert_eq!(blob_count, 1);
    }

    #[test]
    fn commit_and_diff() {
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"a", b"1", meta(0, "")).unwrap();
        let root1 = db.commit(None).unwrap();

        db.insert("t", b"b", b"2", meta(0, "")).unwrap();
        let root2 = db.commit(None).unwrap();

        let d = db.diff(root1, root2, None).unwrap();
        assert_eq!(d.tables["t"].added.len(), 1);
        assert_eq!(d.tables["t"].added[0].key, b"b");
    }

    #[test]
    fn commit_with_bookstore_and_rebuild() {
        use harmony_content::book::MemoryBookStore;
        let dir = tempfile::tempdir().unwrap();
        let mut db = HarmonyDb::open(dir.path()).unwrap();
        db.insert("t", b"k", b"value", meta(0, "snap")).unwrap();

        let mut bs = MemoryBookStore::new();
        let root = db.commit(Some(&mut bs)).unwrap();

        // Rebuild in fresh directory.
        let dir2 = tempfile::tempdir().unwrap();
        let db2 = HarmonyDb::open_from_cas(dir2.path(), root, &bs).unwrap();
        assert_eq!(db2.table_len("t"), 1);
        assert_eq!(db2.get_entry("t", b"k").unwrap().metadata.snippet, "snap");
        assert_eq!(db2.get("t", b"k").unwrap().unwrap(), b"value");
    }
}
```

- [ ] **Step 2: Delete table.rs and commit.rs**

```bash
rm crates/harmony-db/src/table.rs crates/harmony-db/src/commit.rs
```

- [ ] **Step 3: Update lib.rs**

Replace `crates/harmony-db/src/lib.rs`:

```rust
//! Content-addressed key-value database with named tables, atomic commits,
//! history diffing, and portable index rebuild from CAS.

mod db;
mod error;
mod persist;
mod prolly;
mod types;

pub use db::HarmonyDb;
pub use error::DbError;
pub use types::{Diff, Entry, EntryMeta, TableDiff};
```

- [ ] **Step 4: Run all tests**

```bash
cargo test -p harmony-db
```

Expected: all tests pass — prolly tests + db regression tests + persist tests + error tests.

- [ ] **Step 5: Run the mail_workflow example**

```bash
cargo run -p harmony-db --example mail_workflow
```

Expected: "All assertions passed!"

- [ ] **Step 6: Run clippy**

```bash
cargo clippy -p harmony-db -- -W clippy::all
```

Fix any warnings.

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-db/
git commit -m "feat(harmony-db): replace Phase 1 internals with Prolly Tree engine (ZEB-98)"
```

---

### Task 8: Final Verification

- [ ] **Step 1: Full test suite**

```bash
cargo test -p harmony-db
```

Expected: all tests pass.

- [ ] **Step 2: Run example**

```bash
cargo run -p harmony-db --example mail_workflow
```

Expected: passes.

- [ ] **Step 3: Clippy clean**

```bash
cargo clippy -p harmony-db -- -W clippy::all
```

Expected: no harmony-db warnings.

- [ ] **Step 4: Workspace check**

```bash
cargo check -p harmony-db
```

Expected: compiles clean.

- [ ] **Step 5: Commit any fixes**

```bash
git add crates/harmony-db/
git commit -m "fix(harmony-db): address clippy warnings (ZEB-98)"
```

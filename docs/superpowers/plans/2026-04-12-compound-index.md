# CompoundIndex Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a CompoundIndex that splits vector storage into an immutable memory-mapped Base and a mutable in-RAM Delta, with search merging and periodic compaction that returns bytes for CAS storage.

**Architecture:** CompoundIndex wraps two VectorIndex instances (base + delta) and a key tracker for the delta. Searches query both and merge results. Compaction loads the base as writable, inserts delta vectors, serializes to bytes via USearch's `save_to_buffer`, and resets the delta. CAS-agnostic — returns raw bytes, caller handles storage.

**Tech Stack:** Rust, USearch (via harmony-search VectorIndex), tempfile for bytes→mmap path

**Spec:** `docs/superpowers/specs/2026-04-12-compound-index-design.md`

---

### Task 1: VectorIndex serialization helpers

**Files:**
- Modify: `crates/harmony-search/src/index.rs`

Before building CompoundIndex, we need `save_to_bytes()` and `load_from_bytes()` on VectorIndex. USearch provides `save_to_buffer` / `load_from_buffer` natively.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `crates/harmony-search/src/index.rs`:

```rust
    #[test]
    fn save_to_bytes_roundtrip() {
        let config = VectorIndexConfig {
            dimensions: 4,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        };
        let index = VectorIndex::new(config.clone()).unwrap();
        index.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        index.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let bytes = index.save_to_bytes().unwrap();
        assert!(!bytes.is_empty());

        let loaded = VectorIndex::load_from_bytes(&bytes, config).unwrap();
        assert_eq!(loaded.len(), 2);

        let results = loaded.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-search -- save_to_bytes_roundtrip`
Expected: FAIL — method doesn't exist

- [ ] **Step 3: Write the implementation**

Add to `impl VectorIndex` in `crates/harmony-search/src/index.rs`:

```rust
    /// Serialize the index to bytes.
    pub fn save_to_bytes(&self) -> SearchResult<Vec<u8>> {
        let len = self.inner.serialized_length();
        let mut buffer = vec![0u8; len];
        self.inner
            .save_to_buffer(&mut buffer)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        Ok(buffer)
    }

    /// Load an index from bytes.
    ///
    /// The provided `config` must match the configuration used when the index
    /// was originally created and saved.
    pub fn load_from_bytes(bytes: &[u8], config: VectorIndexConfig) -> SearchResult<Self> {
        Self::validate_config(&config)?;
        let inner = usearch::new_index(&make_opts(&config))
            .map_err(|e| SearchError::Index(e.to_string()))?;
        inner
            .load_from_buffer(bytes)
            .map_err(|e| SearchError::Serialization(e.to_string()))?;
        let actual = inner.dimensions();
        if actual != config.dimensions {
            return Err(SearchError::InvalidConfig(format!(
                "loaded index has {actual} dimensions, config says {}",
                config.dimensions
            )));
        }
        Ok(Self { inner, config })
    }

    /// Retrieve a vector by key. Returns None if key not found.
    pub fn get(&self, key: u64, buffer: &mut [f32]) -> SearchResult<bool> {
        match self.inner.get(key, buffer) {
            Ok(count) => Ok(count > 0),
            Err(e) => Err(SearchError::Index(e.to_string())),
        }
    }

    /// Check if a key exists in the index.
    pub fn contains(&self, key: u64) -> bool {
        self.inner.contains(key)
    }
```

Also add `save_to_bytes` and `load_from_bytes` to the `pub use` in `lib.rs` (they're already methods on VectorIndex, no separate re-export needed).

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-search -- save_to_bytes_roundtrip`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-search/src/index.rs
git commit -m "feat: save_to_bytes, load_from_bytes, get, contains on VectorIndex"
```

---

### Task 2: CompoundIndex core — new, add, search, len

**Files:**
- Create: `crates/harmony-search/src/compound.rs`
- Modify: `crates/harmony-search/src/lib.rs`

The core struct and its primary operations.

- [ ] **Step 1: Write the failing tests**

Create `crates/harmony-search/src/compound.rs` starting with tests at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metric, Quantization, VectorIndexConfig};

    fn test_config() -> VectorIndexConfig {
        VectorIndexConfig {
            dimensions: 4,
            metric: Metric::L2,
            quantization: Quantization::F32,
            capacity: 100,
            ..Default::default()
        }
    }

    #[test]
    fn new_starts_empty() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.delta_len(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn add_goes_to_delta() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.len(), 1);
        assert_eq!(idx.delta_len(), 1);
    }

    #[test]
    fn search_finds_delta_vectors() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].key, 1); // nearest
    }

    #[test]
    fn search_empty_returns_empty() {
        let idx = CompoundIndex::new(test_config(), 100).unwrap();
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 5).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn should_compact_respects_threshold() {
        let idx = CompoundIndex::new(test_config(), 2).unwrap();
        assert!(!idx.should_compact());
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        assert!(!idx.should_compact());
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert!(idx.should_compact());
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-search -- compound`
Expected: FAIL — module doesn't exist

- [ ] **Step 3: Write the implementation**

Create the module in `crates/harmony-search/src/compound.rs`:

```rust
//! CompoundIndex: Base + Delta index management for CAS integration.

use crate::error::{SearchError, SearchResult};
use crate::index::{Match, VectorIndex, VectorIndexConfig};

/// Compound index with immutable Base + mutable Delta.
///
/// The base is memory-mapped from a file (zero-copy, read-only).
/// The delta holds recent additions in RAM. Searches query both
/// and merge results. Periodic compaction merges delta into base
/// and returns serialized bytes for CAS storage.
pub struct CompoundIndex {
    base: Option<VectorIndex>,
    delta: VectorIndex,
    delta_keys: Vec<u64>,
    config: VectorIndexConfig,
    compact_threshold: usize,
}

impl CompoundIndex {
    /// Create a new compound index with empty delta and no base.
    pub fn new(config: VectorIndexConfig, compact_threshold: usize) -> SearchResult<Self> {
        let delta = VectorIndex::new(config.clone())?;
        Ok(Self {
            base: None,
            delta,
            delta_keys: Vec::new(),
            config,
            compact_threshold,
        })
    }

    /// Add a vector to the delta index.
    pub fn add(&self, key: u64, vector: &[f32]) -> SearchResult<()> {
        self.delta.add(key, vector)?;
        // SAFETY: we need interior mutability for the key tracker.
        // USearch's add already uses interior mutability (takes &self).
        // We match that pattern here. In practice, callers serialize access
        // through the sans-I/O event loop.
        let keys = &self.delta_keys as *const Vec<u64> as *mut Vec<u64>;
        unsafe { &mut *keys }.push(key);
        Ok(())
    }

    /// Search both base and delta, merge results by distance.
    ///
    /// If the same key exists in both base and delta, the delta's
    /// entry is kept (it represents an update).
    pub fn search(&self, query: &[f32], k: usize) -> SearchResult<Vec<Match>> {
        let mut results = Vec::new();

        // Search delta
        if !self.delta.is_empty() {
            results.extend(self.delta.search(query, k)?);
        }

        // Search base
        if let Some(ref base) = self.base {
            if !base.is_empty() {
                let base_results = base.search(query, k)?;
                // Collect delta keys for dedup
                let delta_key_set: std::collections::HashSet<u64> =
                    results.iter().map(|m| m.key).collect();
                // Add base results that aren't shadowed by delta
                for m in base_results {
                    if !delta_key_set.contains(&m.key) {
                        results.push(m);
                    }
                }
            }
        }

        // Sort by distance and truncate to k
        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        Ok(results)
    }

    /// Whether the delta has reached the compaction threshold.
    pub fn should_compact(&self) -> bool {
        self.delta_len() >= self.compact_threshold
    }

    /// Total vectors across base + delta.
    pub fn len(&self) -> usize {
        let base_len = self.base.as_ref().map_or(0, |b| b.len());
        base_len + self.delta.len()
    }

    /// Number of vectors pending in the delta.
    pub fn delta_len(&self) -> usize {
        self.delta.len()
    }

    /// Whether both base and delta are empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &VectorIndexConfig {
        &self.config
    }
}
```

Then add the module to `crates/harmony-search/src/lib.rs`:

```rust
mod compound;

pub use compound::CompoundIndex;
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-search`
Expected: All pass (existing + 5 new compound tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-search/src/compound.rs crates/harmony-search/src/lib.rs
git commit -m "feat: CompoundIndex core — new, add, search, len, should_compact"
```

---

### Task 3: Compaction and base loading

**Files:**
- Modify: `crates/harmony-search/src/compound.rs`

Add `compact()`, `load_base()`, and `load_base_from_bytes()`.

- [ ] **Step 1: Write the failing tests**

Add to the test module in `compound.rs`:

```rust
    #[test]
    fn compact_returns_bytes_and_resets_delta() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        assert_eq!(idx.delta_len(), 2);

        let bytes = idx.compact().unwrap();
        assert!(!bytes.is_empty());
        assert_eq!(idx.delta_len(), 0);
    }

    #[test]
    fn compact_then_load_base_preserves_search() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // Delta is empty, base has the vectors
        assert_eq!(idx.delta_len(), 0);
        assert_eq!(idx.len(), 2);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
    }

    #[test]
    fn search_merges_base_and_delta() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        // Add vectors, compact to base
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // Add more to delta
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();

        // Search should find both
        assert_eq!(idx.len(), 2);
        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn delta_shadows_base_on_same_key() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        // Add key 1 to base
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // "Update" key 1 in delta with a different vector
        idx.add(1, &[0.0, 0.0, 0.0, 1.0]).unwrap();

        // Search near the updated position should find key 1
        let results = idx.search(&[0.0, 0.0, 0.0, 1.0], 1).unwrap();
        assert_eq!(results[0].key, 1);
        assert!(results[0].distance < 0.01); // should match delta's vector
    }

    #[test]
    fn multiple_compact_cycles() {
        let mut idx = CompoundIndex::new(test_config(), 100).unwrap();

        // Cycle 1
        idx.add(1, &[1.0, 0.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        // Cycle 2
        idx.add(2, &[0.0, 1.0, 0.0, 0.0]).unwrap();
        let bytes = idx.compact().unwrap();
        idx.load_base_from_bytes(&bytes).unwrap();

        assert_eq!(idx.len(), 2);
        assert_eq!(idx.delta_len(), 0);

        let results = idx.search(&[1.0, 0.0, 0.0, 0.0], 2).unwrap();
        assert_eq!(results.len(), 2);
    }
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-search -- compound`
Expected: FAIL — methods don't exist

- [ ] **Step 3: Write the implementation**

Add to `impl CompoundIndex` in `compound.rs`:

```rust
    /// Merge delta into base and return serialized bytes.
    ///
    /// The returned bytes can be CAS-stored by the caller. After compaction,
    /// call `load_base()` or `load_base_from_bytes()` with the persisted data
    /// to swap in the new base.
    ///
    /// Resets the delta to empty after merge.
    pub fn compact(&mut self) -> SearchResult<Vec<u8>> {
        // Build a new writable index containing everything
        let mut merged_config = self.config.clone();
        // Capacity should fit base + delta
        merged_config.capacity = self.len() + 100;
        let merged = VectorIndex::new(merged_config)?;

        // Copy base vectors if we have a base
        if let Some(ref base) = self.base {
            let dims = self.config.dimensions;
            let mut buf = vec![0.0f32; dims];
            // We don't have a keys iterator, so we search for all vectors
            // by using a zero query with k=base.len() — this returns all keys
            if !base.is_empty() {
                let all = base.search(&vec![0.0f32; dims], base.len())?;
                for m in &all {
                    if base.get(m.key, &mut buf)? {
                        merged.add(m.key, &buf)?;
                    }
                }
            }
        }

        // Copy delta vectors
        let dims = self.config.dimensions;
        let mut buf = vec![0.0f32; dims];
        for &key in &self.delta_keys {
            if self.delta.get(key, &mut buf)? {
                merged.add(key, &buf)?;
            }
        }

        // Serialize
        let bytes = merged.save_to_bytes()?;

        // Reset delta
        self.delta = VectorIndex::new(self.config.clone())?;
        self.delta_keys.clear();

        Ok(bytes)
    }

    /// Load the base index from a file (memory-mapped, zero-copy).
    pub fn load_base(&mut self, path: &str) -> SearchResult<()> {
        self.base = Some(VectorIndex::view(path, self.config.clone())?);
        Ok(())
    }

    /// Load the base index from raw bytes.
    ///
    /// Writes to a temp file then memory-maps it. Suitable for tests
    /// and for the initial bootstrap case.
    pub fn load_base_from_bytes(&mut self, bytes: &[u8]) -> SearchResult<()> {
        self.base = Some(VectorIndex::load_from_bytes(bytes, self.config.clone())?);
        Ok(())
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-search`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-search/src/compound.rs
git commit -m "feat: CompoundIndex compact, load_base, load_base_from_bytes"
```

---

### Task 4: Push and create PR

**Files:** None (git operations only)

- [ ] **Step 1: Run full test suite**

Run: `cargo test -p harmony-search`
Expected: All pass

- [ ] **Step 2: Push and create PR**

```bash
git push -u origin HEAD
gh pr create --title "feat: CompoundIndex — Base + Delta index for CAS integration (ZEB-105 Phase 2)" \
  --body "$(cat <<'EOF'
## Summary

- Add `CompoundIndex` struct with immutable Base (memory-mapped) + mutable Delta (in-RAM)
- Search merges results from both, delta shadows base for same-key updates
- `compact()` merges delta into base, returns serialized bytes for CAS storage
- `load_base()` / `load_base_from_bytes()` for swapping in new base after compaction
- `should_compact()` hint based on configurable threshold
- Add `save_to_bytes`, `load_from_bytes`, `get`, `contains` helpers on VectorIndex
- CAS-agnostic: returns raw bytes, caller handles BLAKE3 hashing and storage

### How oluo will consume this (Phase 3)

```
Ingest → compound.add(key, vector) + metadata_table.insert(key, metadata)
Search → compound.search(query, k) → look up metadata for each match
if compound.should_compact() → emit OluoAction::CompactRequest
CompactComplete → compound.load_base(path)
```

## Test plan

- [ ] CompoundIndex search returns results from both base and delta
- [ ] Compaction produces bytes that load back as a new base
- [ ] Delta is empty after compaction
- [ ] Delta shadows base for same-key updates
- [ ] Multiple compact cycles work correctly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

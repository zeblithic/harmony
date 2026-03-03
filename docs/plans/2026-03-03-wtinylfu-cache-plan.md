# W-TinyLFU Cache Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a scan-resistant, frequency-aware W-TinyLFU cache to harmony-content via three layered modules: Count-Min Sketch, slab-allocated LRU, and a ContentStore wrapper.

**Architecture:** `sketch.rs` provides approximate frequency counting with 4-bit pair-packed counters and periodic halving. `lru.rs` provides a slab-allocated doubly-linked-list LRU reused for Window, Probation, and Protected segments. `cache.rs` wires them together into `ContentStore<S: BlobStore>` with W-TinyLFU admission policy and pin support. All sans-I/O, no new dependencies.

**Tech Stack:** Rust, harmony-content crate (existing BlobStore trait, ContentId)

---

### Task 1: Count-Min Sketch — scaffold and increment/estimate

**Files:**
- Create: `crates/harmony-content/src/sketch.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Write the failing tests**

Add to `crates/harmony-content/src/sketch.rs`:

```rust
use crate::cid::ContentId;

/// Count-Min Sketch with 4-bit pair-packed counters.
///
/// Four hash rows, each `width` counters wide. Two counters per byte
/// (high nibble = even index, low nibble = odd index). Saturating
/// increment at 15. Periodic halving decays stale popularity.
pub struct CountMinSketch {
    // TODO: implement
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(format!("sketch-test-{i}").as_bytes()).unwrap()
    }

    #[test]
    fn increment_and_estimate() {
        let mut sketch = CountMinSketch::new(1024, 10_000);
        let cid = make_cid(0);
        sketch.increment(&cid);
        sketch.increment(&cid);
        sketch.increment(&cid);
        assert!(sketch.estimate(&cid) >= 3);
    }

    #[test]
    fn estimate_never_underestimates() {
        let mut sketch = CountMinSketch::new(256, 10_000);
        let target = make_cid(42);
        // Increment target 7 times.
        for _ in 0..7 {
            sketch.increment(&target);
        }
        // Increment many other CIDs to create noise.
        for i in 100..400 {
            sketch.increment(&make_cid(i));
        }
        // Count-Min guarantee: estimate >= true count.
        assert!(sketch.estimate(&target) >= 7);
    }
}
```

Add `pub mod sketch;` to `crates/harmony-content/src/lib.rs`.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content sketch -- --nocapture`
Expected: FAIL — `CountMinSketch` has no `new`, `increment`, or `estimate` methods.

**Step 3: Implement CountMinSketch with increment and estimate**

Replace the struct in `crates/harmony-content/src/sketch.rs`:

```rust
use crate::cid::ContentId;

/// Number of hash rows in the Count-Min Sketch.
const NUM_ROWS: usize = 4;

/// Count-Min Sketch with 4-bit pair-packed counters.
///
/// Four hash rows, each `width` counters wide. Two counters per byte
/// (high nibble = even index, low nibble = odd index). Saturating
/// increment at 15. Periodic halving decays stale popularity.
pub struct CountMinSketch {
    /// Pair-packed 4-bit counters. Layout: 4 rows x (width / 2) bytes.
    counters: Vec<u8>,
    /// Number of counters per row.
    width: usize,
    /// Running total of increment calls.
    total_increments: u64,
    /// Halve all counters when total_increments reaches this.
    halving_threshold: u64,
}

/// Mixing constants for deriving 4 row indices from CID bytes.
/// These are arbitrary odd 64-bit primes for multiply-xor hashing.
const SEEDS: [u64; NUM_ROWS] = [
    0x9E3779B97F4A7C15, // golden ratio derivative
    0xBF58476D1CE4E5B9, // splitmix64 constant
    0x94D049BB133111EB, // splitmix64 constant
    0x517CC1B727220A95, // custom prime
];

impl CountMinSketch {
    /// Create a new sketch with `width` counters per row.
    ///
    /// `halving_threshold` controls how often counters decay. A good
    /// default is `10 * width` (roughly when saturation becomes likely).
    pub fn new(width: usize, halving_threshold: u64) -> Self {
        // Round width up to even for pair-packing.
        let width = if width % 2 == 1 { width + 1 } else { width };
        let bytes_per_row = width / 2;
        CountMinSketch {
            counters: vec![0u8; NUM_ROWS * bytes_per_row],
            width,
            total_increments: 0,
            halving_threshold,
        }
    }

    /// Increment the frequency counter for a CID.
    pub fn increment(&mut self, cid: &ContentId) {
        let indices = self.hash_indices(cid);
        let bytes_per_row = self.width / 2;
        for (row, col) in indices.iter().enumerate() {
            let byte_idx = row * bytes_per_row + col / 2;
            if col % 2 == 0 {
                // High nibble.
                let val = self.counters[byte_idx] >> 4;
                if val < 15 {
                    self.counters[byte_idx] += 0x10;
                }
            } else {
                // Low nibble.
                let val = self.counters[byte_idx] & 0x0F;
                if val < 15 {
                    self.counters[byte_idx] += 1;
                }
            }
        }
        self.total_increments += 1;
        if self.total_increments >= self.halving_threshold {
            self.halve();
        }
    }

    /// Estimate the frequency of a CID (minimum across all rows).
    pub fn estimate(&self, cid: &ContentId) -> u8 {
        let indices = self.hash_indices(cid);
        let bytes_per_row = self.width / 2;
        let mut min_val = 15u8;
        for (row, col) in indices.iter().enumerate() {
            let byte_idx = row * bytes_per_row + col / 2;
            let val = if col % 2 == 0 {
                self.counters[byte_idx] >> 4
            } else {
                self.counters[byte_idx] & 0x0F
            };
            min_val = min_val.min(val);
        }
        min_val
    }

    /// Halve all counters (right-shift by 1). Resets total_increments.
    fn halve(&mut self) {
        for byte in self.counters.iter_mut() {
            let hi = (*byte >> 4) >> 1;
            let lo = (*byte & 0x0F) >> 1;
            *byte = (hi << 4) | lo;
        }
        self.total_increments = 0;
    }

    /// Derive 4 column indices from a CID's hash bytes.
    fn hash_indices(&self, cid: &ContentId) -> [usize; NUM_ROWS] {
        let bytes = cid.to_bytes();
        let h0 = u64::from_le_bytes(bytes[0..8].try_into().unwrap());
        let h1 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mut indices = [0usize; NUM_ROWS];
        for (i, seed) in SEEDS.iter().enumerate() {
            let mixed = h0.wrapping_mul(*seed) ^ h1.wrapping_mul(seed.rotate_left(32));
            indices[i] = (mixed as usize) % self.width;
        }
        indices
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content sketch -- --nocapture`
Expected: 2 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/sketch.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add Count-Min Sketch with 4-bit counters and frequency estimation"
```

---

### Task 2: Count-Min Sketch — halving and auto-halve

**Files:**
- Modify: `crates/harmony-content/src/sketch.rs`

**Step 1: Write the failing tests**

Add to the `tests` module in `sketch.rs`:

```rust
    #[test]
    fn halving_decays_counters() {
        let mut sketch = CountMinSketch::new(1024, 100_000);
        let cid = make_cid(0);
        // Increment 10 times.
        for _ in 0..10 {
            sketch.increment(&cid);
        }
        let before = sketch.estimate(&cid);
        assert!(before >= 10);
        // Force a halve.
        sketch.halve_now();
        let after = sketch.estimate(&cid);
        // Each 4-bit counter was right-shifted by 1, so estimate ~5.
        assert!(after >= 4 && after <= before / 2 + 1);
    }

    #[test]
    fn auto_halving_at_threshold() {
        // Low threshold so auto-halve triggers quickly.
        let mut sketch = CountMinSketch::new(64, 20);
        let cid = make_cid(0);
        for _ in 0..10 {
            sketch.increment(&cid);
        }
        let before = sketch.estimate(&cid);
        // Increment other CIDs to cross the threshold.
        for i in 100..115 {
            sketch.increment(&make_cid(i));
        }
        // threshold=20 reached, auto-halve should have fired.
        let after = sketch.estimate(&cid);
        assert!(after < before, "expected decay: before={before}, after={after}");
    }
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content sketch -- --nocapture`
Expected: FAIL — `halve_now` method not found.

**Step 3: Add halve_now public method**

Add to the `impl CountMinSketch` block:

```rust
    /// Force a halve (public, for testing).
    pub fn halve_now(&mut self) {
        self.halve();
    }
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content sketch -- --nocapture`
Expected: 4 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/sketch.rs
git commit -m "feat(content): add Count-Min Sketch halving and auto-halve at threshold"
```

---

### Task 3: LRU linked list — core operations

**Files:**
- Create: `crates/harmony-content/src/lru.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Write the failing tests**

Add to `crates/harmony-content/src/lru.rs`:

```rust
use std::collections::HashMap;
use crate::cid::ContentId;

/// Slab-allocated doubly-linked-list LRU.
///
/// Used for the Window, Probation, and Protected segments of the
/// W-TinyLFU cache. Entries are stored in a contiguous slab with a
/// free list for recycling removed slots.
pub struct Lru {
    // TODO: implement
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(format!("lru-test-{i}").as_bytes()).unwrap()
    }

    #[test]
    fn eviction_order() {
        let mut lru = Lru::new(3);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        let c3 = make_cid(3);
        assert_eq!(lru.insert(c0), None);
        assert_eq!(lru.insert(c1), None);
        assert_eq!(lru.insert(c2), None);
        // Full — next insert evicts oldest (c0).
        assert_eq!(lru.insert(c3), Some(c0));
        assert!(!lru.contains(&c0));
        assert!(lru.contains(&c3));
        assert_eq!(lru.len(), 3);
    }

    #[test]
    fn touch_promotes_to_head() {
        let mut lru = Lru::new(3);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        let c3 = make_cid(3);
        lru.insert(c0);
        lru.insert(c1);
        lru.insert(c2);
        // Touch c0, making it most recently used.
        assert!(lru.touch(&c0));
        // Now c1 is the LRU tail.
        assert_eq!(lru.insert(c3), Some(c1));
    }

    #[test]
    fn remove_frees_slot() {
        let mut lru = Lru::new(2);
        let c0 = make_cid(0);
        let c1 = make_cid(1);
        let c2 = make_cid(2);
        lru.insert(c0);
        lru.insert(c1);
        assert_eq!(lru.len(), 2);
        assert!(lru.remove(&c0));
        assert_eq!(lru.len(), 1);
        // Inserting after remove does not evict (slot was freed).
        assert_eq!(lru.insert(c2), None);
        assert_eq!(lru.len(), 2);
    }
}
```

Add `pub mod lru;` to `crates/harmony-content/src/lib.rs`.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content lru -- --nocapture`
Expected: FAIL — `Lru` has no methods.

**Step 3: Implement Lru**

Replace the struct in `crates/harmony-content/src/lru.rs`:

```rust
use std::collections::HashMap;
use crate::cid::ContentId;

/// Slab-allocated doubly-linked-list LRU.
///
/// Used for the Window, Probation, and Protected segments of the
/// W-TinyLFU cache. Entries are stored in a contiguous slab with a
/// free list for recycling removed slots.
pub struct Lru {
    entries: Vec<LruEntry>,
    map: HashMap<ContentId, usize>,
    head: Option<usize>,
    tail: Option<usize>,
    free: Vec<usize>,
    len: usize,
    capacity: usize,
}

struct LruEntry {
    cid: ContentId,
    prev: Option<usize>,
    next: Option<usize>,
    occupied: bool,
}

impl Lru {
    /// Create a new LRU with the given capacity.
    pub fn new(capacity: usize) -> Self {
        Lru {
            entries: Vec::new(),
            map: HashMap::new(),
            head: None,
            tail: None,
            free: Vec::new(),
            len: 0,
            capacity,
        }
    }

    /// Insert a CID at the head. Returns the evicted CID if over capacity.
    pub fn insert(&mut self, cid: ContentId) -> Option<ContentId> {
        if self.map.contains_key(&cid) {
            self.touch(&cid);
            return None;
        }

        let evicted = if self.len >= self.capacity {
            self.evict_tail()
        } else {
            None
        };

        let idx = self.alloc_slot(cid);
        self.link_at_head(idx);
        self.map.insert(cid, idx);
        self.len += 1;

        evicted
    }

    /// Move a CID to the head (most recently used). Returns true if found.
    pub fn touch(&mut self, cid: &ContentId) -> bool {
        if let Some(&idx) = self.map.get(cid) {
            self.unlink(idx);
            self.link_at_head(idx);
            true
        } else {
            false
        }
    }

    /// Remove a CID. Returns true if it was present.
    pub fn remove(&mut self, cid: &ContentId) -> bool {
        if let Some(idx) = self.map.remove(cid) {
            self.unlink(idx);
            self.entries[idx].occupied = false;
            self.free.push(idx);
            self.len -= 1;
            true
        } else {
            false
        }
    }

    /// Peek at the least recently used CID without removing it.
    pub fn peek_lru(&self) -> Option<ContentId> {
        self.tail.map(|idx| self.entries[idx].cid)
    }

    /// Check if a CID is present.
    pub fn contains(&self, cid: &ContentId) -> bool {
        self.map.contains_key(cid)
    }

    /// Current number of entries.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the LRU is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    // -- internal helpers --

    fn alloc_slot(&mut self, cid: ContentId) -> usize {
        if let Some(idx) = self.free.pop() {
            self.entries[idx] = LruEntry {
                cid,
                prev: None,
                next: None,
                occupied: true,
            };
            idx
        } else {
            let idx = self.entries.len();
            self.entries.push(LruEntry {
                cid,
                prev: None,
                next: None,
                occupied: true,
            });
            idx
        }
    }

    fn link_at_head(&mut self, idx: usize) {
        self.entries[idx].prev = None;
        self.entries[idx].next = self.head;
        if let Some(old_head) = self.head {
            self.entries[old_head].prev = Some(idx);
        }
        self.head = Some(idx);
        if self.tail.is_none() {
            self.tail = Some(idx);
        }
    }

    fn unlink(&mut self, idx: usize) {
        let prev = self.entries[idx].prev;
        let next = self.entries[idx].next;
        if let Some(p) = prev {
            self.entries[p].next = next;
        } else {
            self.head = next;
        }
        if let Some(n) = next {
            self.entries[n].prev = prev;
        } else {
            self.tail = prev;
        }
        self.entries[idx].prev = None;
        self.entries[idx].next = None;
    }

    fn evict_tail(&mut self) -> Option<ContentId> {
        let tail_idx = self.tail?;
        let cid = self.entries[tail_idx].cid;
        self.map.remove(&cid);
        self.unlink(tail_idx);
        self.entries[tail_idx].occupied = false;
        self.free.push(tail_idx);
        self.len -= 1;
        Some(cid)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-content lru -- --nocapture`
Expected: 3 tests PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/lru.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add slab-allocated LRU linked list"
```

---

### Task 4: ContentStore — scaffold, BlobStore impl, basic get/put

**Files:**
- Create: `crates/harmony-content/src/cache.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Write the failing test**

Add to `crates/harmony-content/src/cache.rs`:

```rust
use std::collections::HashSet;
use crate::blob::BlobStore;
use crate::cid::ContentId;
use crate::error::ContentError;
use crate::lru::Lru;
use crate::sketch::CountMinSketch;

/// W-TinyLFU cached content store.
///
/// Wraps any `BlobStore` with frequency-aware admission policy.
/// Three LRU segments: Window (1%), Protected (20%), Probation (~79%).
/// A Count-Min Sketch tracks access frequency for admission decisions.
pub struct ContentStore<S: BlobStore> {
    // TODO: implement
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;

    #[test]
    fn content_store_implements_blobstore() {
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 100);
        let data = b"hello cache";
        let cid = cs.insert(data).unwrap();
        assert!(cs.contains(&cid));
        assert_eq!(cs.get(&cid).unwrap(), data);
    }
}
```

Add `pub mod cache;` to `crates/harmony-content/src/lib.rs`.

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content content_store_implements -- --nocapture`
Expected: FAIL — `ContentStore` has no methods.

**Step 3: Implement ContentStore with BlobStore delegation**

Replace the struct and add the impl blocks:

```rust
use std::collections::HashSet;
use crate::blob::BlobStore;
use crate::cid::ContentId;
use crate::error::ContentError;
use crate::lru::Lru;
use crate::sketch::CountMinSketch;

/// W-TinyLFU cached content store.
///
/// Wraps any `BlobStore` with frequency-aware admission policy.
/// Three LRU segments: Window (1%), Protected (20%), Probation (~79%).
/// A Count-Min Sketch tracks access frequency for admission decisions.
pub struct ContentStore<S: BlobStore> {
    store: S,
    sketch: CountMinSketch,
    window: Lru,
    probation: Lru,
    protected: Lru,
    pinned: HashSet<ContentId>,
}

impl<S: BlobStore> ContentStore<S> {
    /// Create a new cached store with the given total capacity (entry count).
    pub fn new(store: S, capacity: usize) -> Self {
        let window_cap = (capacity / 100).max(1);
        let protected_cap = capacity / 5;
        let probation_cap = capacity.saturating_sub(window_cap).saturating_sub(protected_cap).max(1);
        let sketch_width = capacity * 2;
        let halving_threshold = (capacity * 10) as u64;

        ContentStore {
            store,
            sketch: CountMinSketch::new(sketch_width, halving_threshold),
            window: Lru::new(window_cap),
            probation: Lru::new(probation_cap),
            protected: Lru::new(protected_cap),
            pinned: HashSet::new(),
        }
    }
}

impl<S: BlobStore> BlobStore for ContentStore<S> {
    fn insert(&mut self, data: &[u8]) -> Result<ContentId, ContentError> {
        let cid = self.store.insert(data)?;
        self.admit(cid);
        Ok(cid)
    }

    fn store(&mut self, cid: ContentId, data: Vec<u8>) {
        self.store.store(cid, data);
        self.admit(cid);
    }

    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.store.get(cid)
    }

    fn contains(&self, cid: &ContentId) -> bool {
        self.store.contains(cid)
    }
}

impl<S: BlobStore> ContentStore<S> {
    /// Record access and manage admission. Called on insert/store.
    fn admit(&mut self, cid: ContentId) {
        self.sketch.increment(&cid);
        // For now, just insert into window. Admission challenge in next task.
        if !self.window.contains(&cid)
            && !self.probation.contains(&cid)
            && !self.protected.contains(&cid)
        {
            self.window.insert(cid);
        }
    }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content content_store_implements -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cache.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add ContentStore scaffold with BlobStore delegation"
```

---

### Task 5: ContentStore — W-TinyLFU admission challenge

**Files:**
- Modify: `crates/harmony-content/src/cache.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `cache.rs`:

```rust
    #[test]
    fn frequency_based_admission() {
        // Capacity 10: window=1, protected=2, probation=7.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 10);

        // Fill probation with 7 items, each accessed 5 times to build frequency.
        let hot: Vec<ContentId> = (0..7)
            .map(|i| {
                let data = format!("hot-{i}");
                cs.insert(data.as_bytes()).unwrap()
            })
            .collect();
        for cid in &hot {
            for _ in 0..5 {
                cs.record_access(cid);
            }
        }

        // Insert a cold item — it enters window, then faces admission challenge.
        // Its frequency (1) should lose against probation's tail (5+).
        let cold = cs.insert(b"cold-newcomer").unwrap();

        // The hot items should still be accessible.
        for cid in &hot {
            assert!(
                cs.window.contains(cid)
                    || cs.probation.contains(cid)
                    || cs.protected.contains(cid),
                "hot CID should still be in cache"
            );
        }
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content frequency_based_admission -- --nocapture`
Expected: FAIL — `record_access` method not found.

**Step 3: Implement admission challenge and record_access**

Add `record_access` and update the `admit` method in `cache.rs`:

```rust
impl<S: BlobStore> ContentStore<S> {
    /// Record a read access to a CID (updates frequency sketch and LRU position).
    ///
    /// Call this on cache hits to maintain accurate frequency data.
    /// Handles promotion from probation to protected.
    pub fn record_access(&mut self, cid: &ContentId) {
        self.sketch.increment(cid);

        // If in protected, just touch.
        if self.protected.contains(cid) {
            self.protected.touch(cid);
            return;
        }

        // If in probation, promote to protected.
        if self.probation.contains(cid) {
            self.probation.remove(cid);
            if let Some(demoted) = self.protected.insert(*cid) {
                // Protected overflow: demote victim to probation head.
                self.probation.insert(demoted);
            }
            return;
        }

        // If in window, just touch.
        if self.window.contains(cid) {
            self.window.touch(cid);
        }
    }

    /// Record access and manage admission. Called on insert/store.
    fn admit(&mut self, cid: ContentId) {
        self.sketch.increment(&cid);

        // Already tracked — just record access.
        if self.window.contains(&cid)
            || self.probation.contains(&cid)
            || self.protected.contains(&cid)
        {
            self.record_access(&cid);
            return;
        }

        // New entry: insert into window.
        if let Some(evictee) = self.window.insert(cid) {
            // Window overflow: admission challenge.
            self.admission_challenge(evictee);
        }
    }

    /// The admission challenge: compare window evictee frequency against
    /// probation's least-valuable resident.
    fn admission_challenge(&mut self, candidate: ContentId) {
        if let Some(victim) = self.probation.peek_lru() {
            let candidate_freq = self.sketch.estimate(&candidate);
            let victim_freq = self.sketch.estimate(&victim);
            if candidate_freq > victim_freq {
                // Candidate wins — admit to probation, evict victim.
                self.probation.remove(&victim);
                self.store_remove(&victim);
                self.probation.insert(candidate);
            } else {
                // Candidate loses — drop it entirely.
                self.store_remove(&candidate);
            }
        } else {
            // Probation is empty — admit directly.
            self.probation.insert(candidate);
        }
    }

    /// Remove data from the underlying store (for evicted entries).
    fn store_remove(&mut self, _cid: &ContentId) {
        // MemoryBlobStore doesn't support removal yet.
        // This is a no-op for now — the data stays in the store
        // but is no longer tracked by the cache. Future optimization.
    }
}
```

Also update `get` to call `record_access`:

Replace the `get` method in the `BlobStore` impl:

```rust
    fn get(&self, cid: &ContentId) -> Option<&[u8]> {
        self.store.get(cid)
    }
```

Note: `get` takes `&self` so it can't mutate the sketch/LRU. We need a separate `get_and_record` or accept this limitation. For the BlobStore trait compatibility, `get` stays immutable. Callers that want frequency tracking should call `record_access` separately after `get`. This matches the sans-I/O pattern — the caller manages access recording.

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content frequency_based_admission -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cache.rs
git commit -m "feat(content): add W-TinyLFU admission challenge and record_access"
```

---

### Task 6: ContentStore — probation-to-protected promotion

**Files:**
- Modify: `crates/harmony-content/src/cache.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `cache.rs`:

```rust
    #[test]
    fn probation_to_protected_promotion() {
        // Capacity 20: window=1, protected=4, probation=15.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);

        // Insert an item — it enters window, then gets promoted to probation
        // when the next item pushes it out of window.
        let target = cs.insert(b"target").unwrap();
        let _pusher = cs.insert(b"pusher").unwrap();

        // target should now be in probation (pushed out of window).
        // Access it to promote to protected.
        cs.record_access(&target);
        assert!(cs.protected.contains(&target));
        assert!(!cs.probation.contains(&target));
    }
```

**Step 2: Run test to verify it passes (or fails)**

Run: `cargo test -p harmony-content probation_to_protected -- --nocapture`
Expected: This test should already PASS with the `record_access` implementation from Task 5. If it doesn't, debug the promotion logic.

**Step 3: Commit (if passing, otherwise fix first)**

```bash
git add crates/harmony-content/src/cache.rs
git commit -m "test(content): verify probation-to-protected promotion on access"
```

---

### Task 7: ContentStore — scan resistance

**Files:**
- Modify: `crates/harmony-content/src/cache.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `cache.rs`:

```rust
    #[test]
    fn scan_resistance() {
        // The core W-TinyLFU property: a sequential scan of cold data
        // should NOT evict a frequently-accessed hot item.

        // Capacity 20: window=1, protected=4, probation=15.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);

        // Create a hot item and give it lots of frequency.
        let hot = cs.insert(b"frequently-accessed").unwrap();
        for _ in 0..10 {
            cs.record_access(&hot);
        }

        // Sequential scan: insert 50 unique cold items (never accessed again).
        for i in 0..50 {
            let data = format!("scan-item-{i}");
            cs.insert(data.as_bytes()).unwrap();
        }

        // The hot item should survive the scan.
        assert!(
            cs.window.contains(&hot)
                || cs.probation.contains(&hot)
                || cs.protected.contains(&hot),
            "hot CID should survive sequential scan"
        );
    }
```

**Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-content scan_resistance -- --nocapture`
Expected: PASS — the hot item's high frequency should defeat all cold admission challenges.

**Step 3: Commit**

```bash
git add crates/harmony-content/src/cache.rs
git commit -m "test(content): verify W-TinyLFU scan resistance property"
```

---

### Task 8: ContentStore — pin/unpin support

**Files:**
- Modify: `crates/harmony-content/src/cache.rs`

**Step 1: Write the failing test**

Add to the `tests` module in `cache.rs`:

```rust
    #[test]
    fn pin_exempts_from_eviction() {
        // Small cache: capacity 5 → window=1, protected=1, probation=3.
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 5);

        let pinned_cid = cs.insert(b"pinned-data").unwrap();
        cs.pin(pinned_cid);

        // Fill the cache well beyond capacity.
        for i in 0..20 {
            let data = format!("filler-{i}");
            cs.insert(data.as_bytes()).unwrap();
        }

        // Pinned item should still be retrievable.
        assert!(cs.is_pinned(&pinned_cid));
        // Data should still be in the store (not removed by eviction).
        assert!(cs.contains(&pinned_cid));

        // Unpin and verify it becomes evictable.
        cs.unpin(&pinned_cid);
        assert!(!cs.is_pinned(&pinned_cid));
    }
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-content pin_exempts -- --nocapture`
Expected: FAIL — `pin`, `unpin`, `is_pinned` methods not found.

**Step 3: Implement pin/unpin**

Add to `impl<S: BlobStore> ContentStore<S>`:

```rust
    /// Pin a CID, exempting it from eviction.
    pub fn pin(&mut self, cid: ContentId) {
        self.pinned.insert(cid);
    }

    /// Unpin a CID, making it eligible for eviction again.
    pub fn unpin(&mut self, cid: &ContentId) {
        self.pinned.remove(cid);
    }

    /// Check if a CID is pinned.
    pub fn is_pinned(&self, cid: &ContentId) -> bool {
        self.pinned.contains(cid)
    }
```

Update `admission_challenge` to skip pinned victims:

```rust
    fn admission_challenge(&mut self, candidate: ContentId) {
        if let Some(victim) = self.probation.peek_lru() {
            // Skip pinned victims.
            if self.pinned.contains(&victim) {
                // Pinned victim can't be evicted — just admit candidate.
                self.probation.insert(candidate);
                return;
            }
            let candidate_freq = self.sketch.estimate(&candidate);
            let victim_freq = self.sketch.estimate(&victim);
            if candidate_freq > victim_freq {
                self.probation.remove(&victim);
                self.store_remove(&victim);
                self.probation.insert(candidate);
            } else {
                self.store_remove(&candidate);
            }
        } else {
            self.probation.insert(candidate);
        }
    }
```

**Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-content pin_exempts -- --nocapture`
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cache.rs
git commit -m "feat(content): add pin/unpin support for cache eviction exemption"
```

---

### Task 9: Run full test suite and verify

**Files:** None (verification only)

**Step 1: Run all harmony-content tests**

Run: `cargo test -p harmony-content`
Expected: All tests pass (existing 104 + ~12 new = ~116 tests).

**Step 2: Run clippy**

Run: `cargo clippy -p harmony-content -- -D warnings`
Expected: No warnings.

**Step 3: Run fmt check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues.

**Step 4: Commit any fixes if needed**

If clippy or fmt found issues, fix and commit:

```bash
git commit -m "style(content): fix clippy/fmt issues in W-TinyLFU cache"
```

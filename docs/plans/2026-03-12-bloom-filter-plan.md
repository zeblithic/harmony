# Bloom Filter Content Discovery Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Nodes periodically broadcast Bloom filters of their cached CID set so peers can skip content queries to nodes that definitely don't have the content.

**Architecture:** A `BloomFilter` data structure in `harmony-content` uses Kirsch-Mitzenmacher double-hashing over 28-byte CID hashes. `StorageTier` tracks cache mutations and emits `BroadcastFilter` actions on a hybrid timer/threshold trigger. `ContentStore` exposes `iter_admitted()` so the filter can be rebuilt from scratch each broadcast. `harmony-zenoh` gets a `filters` namespace module. `NodeRuntime` maintains a per-peer filter table and checks it before dispatching content queries.

**Tech Stack:** Rust, `no_std` + `alloc`, harmony-content, harmony-zenoh, harmony-node

---

### Task 1: BloomFilter Data Structure — Core

**Files:**
- Create: `crates/harmony-content/src/bloom.rs`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Create `bloom.rs` with the struct and `new()` constructor**

```rust
//! Bloom filter for approximate set membership testing.
//!
//! Used for content discovery: nodes broadcast filters of their cached CID set
//! so peers can skip queries to nodes that definitely don't have the content.
//! False positives (unnecessary queries) are tolerable. False negatives
//! (missed content) are impossible for items present at filter-build time.

use alloc::vec;
use alloc::vec::Vec;
use core::fmt;

use crate::cid::ContentId;

/// Bloom filter for approximate CID set membership.
///
/// Sized at construction for a target item count and false positive rate.
/// Uses Kirsch-Mitzenmacher double hashing: k hash indices derived from
/// two base hashes of the 28-byte CID hash field.
pub struct BloomFilter {
    /// Bit array packed into u64 words.
    bits: Vec<u64>,
    /// Number of hash functions (k).
    num_hashes: u32,
    /// Total number of bits in the filter (m).
    num_bits: u32,
}

impl fmt::Debug for BloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BloomFilter")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .field("size_bytes", &(self.bits.len() * 8))
            .finish_non_exhaustive()
    }
}

impl BloomFilter {
    /// Create a filter sized for `expected_items` at the given false positive rate.
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is 0 or `fp_rate` is not in `(0.0, 1.0)`.
    pub fn new(expected_items: u32, fp_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fp_rate > 0.0 && fp_rate < 1.0,
            "fp_rate must be in (0.0, 1.0)"
        );

        let n = expected_items as f64;
        let ln2 = core::f64::consts::LN_2;

        // m = -n * ln(p) / (ln2)^2
        let m = (-n * fp_rate.ln() / (ln2 * ln2)).ceil() as u32;
        // k = (m/n) * ln2
        let k = ((m as f64 / n) * ln2).round() as u32;
        let k = k.max(1); // at least 1 hash function

        // Round up to u64 boundary.
        let num_words = ((m as usize) + 63) / 64;

        BloomFilter {
            bits: vec![0u64; num_words],
            num_hashes: k,
            num_bits: m,
        }
    }

    /// Number of bits in this filter.
    pub fn num_bits(&self) -> u32 {
        self.num_bits
    }

    /// Number of hash functions used.
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }
}
```

**Step 2: Register the module in `lib.rs`**

Add `pub mod bloom;` after `pub mod blob;` in `crates/harmony-content/src/lib.rs`.

**Step 3: Write tests for `new()` sizing**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sizes_correctly_for_100k_items() {
        let bf = BloomFilter::new(100_000, 0.001);
        // m ≈ 1,437,759 bits, k ≈ 10
        assert!(bf.num_bits() >= 1_400_000 && bf.num_bits() <= 1_500_000);
        assert_eq!(bf.num_hashes(), 10);
    }

    #[test]
    fn new_sizes_correctly_for_1k_items() {
        let bf = BloomFilter::new(1_000, 0.001);
        assert!(bf.num_bits() >= 14_000 && bf.num_bits() <= 15_000);
        assert_eq!(bf.num_hashes(), 10);
    }

    #[test]
    #[should_panic(expected = "expected_items must be > 0")]
    fn new_rejects_zero_items() {
        BloomFilter::new(0, 0.01);
    }

    #[test]
    #[should_panic(expected = "fp_rate must be in (0.0, 1.0)")]
    fn new_rejects_zero_fp_rate() {
        BloomFilter::new(100, 0.0);
    }

    #[test]
    fn debug_impl_shows_summary() {
        let bf = BloomFilter::new(1000, 0.01);
        let dbg = format!("{bf:?}");
        assert!(dbg.contains("BloomFilter"));
        assert!(dbg.contains("num_bits"));
        assert!(dbg.contains("num_hashes"));
    }
}
```

**Step 4: Run tests**

```bash
cargo test -p harmony-content bloom
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/bloom.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): BloomFilter struct with sizing constructor"
```

---

### Task 2: BloomFilter — Insert, Query, Clear

**Files:**
- Modify: `crates/harmony-content/src/bloom.rs`

**Step 1: Add hashing and insert/may_contain/clear methods**

```rust
/// Mixing constants for Kirsch-Mitzenmacher double hashing.
/// Same SplitMix64 constants as sketch.rs for consistency.
const SEED_A: u64 = 0x9E3779B97F4A7C15;
const SEED_B: u64 = 0xBF58476D1CE4E5B9;

impl BloomFilter {
    // ... existing methods ...

    /// Insert a CID into the filter.
    pub fn insert(&mut self, cid: &ContentId) {
        let (h1, h2) = Self::hash_pair(cid);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// Check if a CID might be in the filter.
    ///
    /// Returns `false` only if the CID is **definitely** absent.
    /// Returns `true` if the CID is **possibly** present (may be a false positive).
    pub fn may_contain(&self, cid: &ContentId) -> bool {
        let (h1, h2) = Self::hash_pair(cid);
        for i in 0..self.num_hashes {
            let idx = self.bit_index(h1, h2, i);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// Reset all bits to zero, making the filter empty.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
    }

    /// Derive two base hashes from a CID's 28-byte hash field.
    ///
    /// Kirsch-Mitzenmacher: k indices = h1 + i*h2, so we only need 2 hashes.
    fn hash_pair(cid: &ContentId) -> (u64, u64) {
        let hash = &cid.hash;
        let a = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        let b = u64::from_le_bytes(hash[8..16].try_into().unwrap());

        let h1 = a.wrapping_mul(SEED_A) ^ b;
        let h2 = b.wrapping_mul(SEED_B) ^ a;
        (h1, h2)
    }

    /// Compute the i-th bit index using Kirsch-Mitzenmacher scheme.
    fn bit_index(&self, h1: u64, h2: u64, i: u32) -> usize {
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined % self.num_bits as u64) as usize
    }

    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 == 1
    }
}
```

**Step 2: Add tests**

```rust
    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(
            format!("bloom-test-{i}").as_bytes(),
            crate::cid::ContentFlags::default(),
        )
        .unwrap()
    }

    #[test]
    fn insert_and_may_contain_round_trip() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let cid = make_cid(42);
        assert!(!bf.may_contain(&cid));
        bf.insert(&cid);
        assert!(bf.may_contain(&cid));
    }

    #[test]
    fn empty_filter_returns_false() {
        let bf = BloomFilter::new(1000, 0.01);
        for i in 0..100 {
            assert!(!bf.may_contain(&make_cid(i)));
        }
    }

    #[test]
    fn clear_resets_all_bits() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..100 {
            bf.insert(&make_cid(i));
        }
        bf.clear();
        for i in 0..100 {
            assert!(!bf.may_contain(&make_cid(i)));
        }
    }

    #[test]
    fn no_false_negatives() {
        // Insert 1000 items, verify all are found.
        let mut bf = BloomFilter::new(1000, 0.01);
        let cids: Vec<ContentId> = (0..1000).map(make_cid).collect();
        for cid in &cids {
            bf.insert(cid);
        }
        for cid in &cids {
            assert!(bf.may_contain(cid), "false negative detected");
        }
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        // Insert 1000 items at 1% FP rate, check 10000 non-members.
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&make_cid(i));
        }
        let mut false_positives = 0;
        for i in 1000..11000 {
            if bf.may_contain(&make_cid(i)) {
                false_positives += 1;
            }
        }
        let fp_rate = false_positives as f64 / 10000.0;
        // Allow 3x theoretical rate to account for variance.
        assert!(
            fp_rate < 0.03,
            "FP rate {fp_rate:.4} exceeds 3% (expected ~1%)"
        );
    }
```

**Step 3: Run tests**

```bash
cargo test -p harmony-content bloom
```
Expected: PASS

**Step 4: Commit**

```bash
git add crates/harmony-content/src/bloom.rs
git commit -m "feat(content): BloomFilter insert, may_contain, clear with double hashing"
```

---

### Task 3: BloomFilter — Serialization and Estimated Count

**Files:**
- Modify: `crates/harmony-content/src/bloom.rs`

**Step 1: Add error type, serialization, and estimated_count**

```rust
/// Errors for Bloom filter deserialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// Payload too short to contain the header.
    HeaderTruncated,
    /// Payload length doesn't match the declared bit count.
    LengthMismatch { expected: usize, got: usize },
    /// num_bits is zero.
    ZeroBits,
    /// num_hashes is zero.
    ZeroHashes,
}

impl core::fmt::Display for FilterError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::HeaderTruncated => write!(f, "filter payload too short for header"),
            Self::LengthMismatch { expected, got } => {
                write!(f, "expected {expected} body bytes, got {got}")
            }
            Self::ZeroBits => write!(f, "num_bits must be > 0"),
            Self::ZeroHashes => write!(f, "num_hashes must be > 0"),
        }
    }
}

/// Header size: num_bits(4) + num_hashes(4) + item_count(4) = 12 bytes.
const HEADER_SIZE: usize = 12;

impl BloomFilter {
    // ... existing methods ...

    /// Serialize for broadcast.
    ///
    /// Wire format: `[num_bits: u32 BE][num_hashes: u32 BE][item_count: u32 BE][bit array]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let body_len = (self.bits.len()) * 8;
        let mut buf = Vec::with_capacity(HEADER_SIZE + body_len);
        buf.extend_from_slice(&self.num_bits.to_be_bytes());
        buf.extend_from_slice(&self.num_hashes.to_be_bytes());
        buf.extend_from_slice(&self.estimated_count().to_be_bytes());
        for &word in &self.bits {
            buf.extend_from_slice(&word.to_le_bytes());
        }
        buf
    }

    /// Deserialize from broadcast payload.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FilterError> {
        if bytes.len() < HEADER_SIZE {
            return Err(FilterError::HeaderTruncated);
        }
        let num_bits = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let num_hashes = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        // item_count at bytes[8..12] is informational — not stored in the struct.

        if num_bits == 0 {
            return Err(FilterError::ZeroBits);
        }
        if num_hashes == 0 {
            return Err(FilterError::ZeroHashes);
        }

        let num_words = ((num_bits as usize) + 63) / 64;
        let expected_body = num_words * 8;
        let body = &bytes[HEADER_SIZE..];
        if body.len() != expected_body {
            return Err(FilterError::LengthMismatch {
                expected: expected_body,
                got: body.len(),
            });
        }

        let bits: Vec<u64> = body
            .chunks_exact(8)
            .map(|chunk| u64::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(BloomFilter {
            bits,
            num_hashes,
            num_bits,
        })
    }

    /// Approximate number of items inserted, estimated from bit density.
    ///
    /// Uses the formula: `n* = -(m/k) * ln(1 - X/m)` where X = number of set bits.
    pub fn estimated_count(&self) -> u32 {
        let set_bits: u64 = self.bits.iter().map(|w| w.count_ones() as u64).sum();
        if set_bits == 0 {
            return 0;
        }
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;
        let x = set_bits as f64;
        // Clamp x/m to avoid ln(0).
        let ratio = (x / m).min(0.9999);
        let estimate = -(m / k) * (1.0 - ratio).ln();
        estimate.round() as u32
    }
}
```

**Step 2: Add tests**

```rust
    #[test]
    fn serialization_round_trip() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..500 {
            bf.insert(&make_cid(i));
        }
        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();
        assert_eq!(bf.num_bits(), bf2.num_bits());
        assert_eq!(bf.num_hashes(), bf2.num_hashes());
        // All previously inserted CIDs should still be found.
        for i in 0..500 {
            assert!(bf2.may_contain(&make_cid(i)));
        }
    }

    #[test]
    fn from_bytes_rejects_truncated() {
        assert_eq!(
            BloomFilter::from_bytes(&[0; 8]).unwrap_err(),
            FilterError::HeaderTruncated
        );
    }

    #[test]
    fn from_bytes_rejects_wrong_body_length() {
        let bf = BloomFilter::new(100, 0.01);
        let mut bytes = bf.to_bytes();
        bytes.truncate(HEADER_SIZE + 8); // leave only 1 word
        let err = BloomFilter::from_bytes(&bytes).unwrap_err();
        assert!(matches!(err, FilterError::LengthMismatch { .. }));
    }

    #[test]
    fn estimated_count_accuracy() {
        let mut bf = BloomFilter::new(10_000, 0.01);
        for i in 0..5_000 {
            bf.insert(&make_cid(i));
        }
        let estimate = bf.estimated_count();
        // Should be within 10% of 5000.
        assert!(
            estimate >= 4500 && estimate <= 5500,
            "estimate {estimate} not within 10% of 5000"
        );
    }

    #[test]
    fn estimated_count_zero_for_empty() {
        let bf = BloomFilter::new(1000, 0.01);
        assert_eq!(bf.estimated_count(), 0);
    }
```

**Step 3: Run tests**

```bash
cargo test -p harmony-content bloom
```
Expected: PASS

**Step 4: Commit**

```bash
git add crates/harmony-content/src/bloom.rs
git commit -m "feat(content): BloomFilter serialization and estimated_count"
```

---

### Task 4: ContentStore `iter_admitted()` Iterator

**Files:**
- Modify: `crates/harmony-content/src/lru.rs`
- Modify: `crates/harmony-content/src/cache.rs`

**Step 1: Add `iter()` method to `Lru`**

Add this to the `impl Lru` block in `crates/harmony-content/src/lru.rs`:

```rust
    /// Iterate over all CIDs in the LRU from head (most recent) to tail.
    pub fn iter(&self) -> LruIter<'_> {
        LruIter {
            entries: &self.entries,
            current: self.head,
        }
    }
```

Add the iterator struct after the `impl Lru` block:

```rust
/// Iterator over CIDs in an LRU segment, from head to tail.
pub struct LruIter<'a> {
    entries: &'a [LruEntry],
    current: Option<usize>,
}

impl<'a> Iterator for LruIter<'a> {
    type Item = ContentId;

    fn next(&mut self) -> Option<Self::Item> {
        let idx = self.current?;
        let entry = &self.entries[idx];
        self.current = entry.next;
        Some(entry.cid)
    }
}
```

**Step 2: Add `iter_admitted()` to `ContentStore`**

Add this method to the `impl<S: BlobStore> ContentStore<S>` block in `crates/harmony-content/src/cache.rs`:

```rust
    /// Iterate over all admitted CIDs (window + probation + protected).
    ///
    /// Used to rebuild Bloom filters from the current cache state.
    /// Order is not guaranteed across segments.
    pub fn iter_admitted(&self) -> impl Iterator<Item = ContentId> + '_ {
        self.window
            .iter()
            .chain(self.probation.iter())
            .chain(self.protected.iter())
    }
```

**Step 3: Add tests**

In `lru.rs` tests:
```rust
    #[test]
    fn iter_visits_all_entries_head_to_tail() {
        let mut lru = Lru::new(5);
        lru.insert(make_cid(1));
        lru.insert(make_cid(2));
        lru.insert(make_cid(3));
        let cids: Vec<ContentId> = lru.iter().collect();
        assert_eq!(cids.len(), 3);
        // Most recent first (head = last inserted).
        assert_eq!(cids[0], make_cid(3));
        assert_eq!(cids[2], make_cid(1));
    }

    #[test]
    fn iter_empty_lru() {
        let lru = Lru::new(5);
        assert_eq!(lru.iter().count(), 0);
    }
```

In `cache.rs` tests:
```rust
    #[test]
    fn iter_admitted_returns_all_cached_cids() {
        let store = MemoryBlobStore::new();
        let mut cs = ContentStore::new(store, 20);
        let mut expected = Vec::new();
        for i in 0..10 {
            let cid = cs.insert(format!("item-{i}").as_bytes()).unwrap();
            expected.push(cid);
        }
        let admitted: Vec<ContentId> = cs.iter_admitted().collect();
        assert_eq!(admitted.len(), 10);
        for cid in &expected {
            assert!(admitted.contains(cid), "missing CID from iter_admitted");
        }
    }
```

**Step 4: Run tests**

```bash
cargo test -p harmony-content lru::tests::iter && cargo test -p harmony-content cache::tests::iter_admitted
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-content/src/lru.rs crates/harmony-content/src/cache.rs
git commit -m "feat(content): iter_admitted() for Bloom filter rebuild from cache state"
```

---

### Task 5: Zenoh `filters` Namespace Module

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

**Step 1: Add the `filters` module**

Add after the `announce` module (after line 174):

```rust
/// Content filter broadcasts.
///
/// Nodes periodically broadcast Bloom filters of their cached CID set.
/// Receiving nodes check these filters before dispatching content queries,
/// skipping peers that definitely don't have the content.
pub mod filters {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/filters`
    pub const PREFIX: &str = "harmony/filters";

    /// Content filter prefix: `harmony/filters/content`
    pub const CONTENT_PREFIX: &str = "harmony/filters/content";

    /// Subscribe to all content filters: `harmony/filters/content/**`
    pub const CONTENT_SUB: &str = "harmony/filters/content/**";

    /// Content filter key: `harmony/filters/content/{node_addr}`
    pub fn content_key(node_addr: &str) -> String {
        format!("{CONTENT_PREFIX}/{node_addr}")
    }
}
```

**Step 2: Add `filters::PREFIX` to the cross-tier consistency test**

In the `all_prefixes_start_with_root` test, add `filters::PREFIX` to the prefixes array.

**Step 3: Add tests**

```rust
    // ── Tier 2: Filters ──────────────────────────────────────────

    #[test]
    fn filters_content_key() {
        assert_eq!(
            filters::content_key("abc123"),
            "harmony/filters/content/abc123"
        );
    }

    #[test]
    fn filters_subscription_pattern() {
        assert_eq!(filters::CONTENT_SUB, "harmony/filters/content/**");
    }
```

**Step 4: Run tests**

```bash
cargo test -p harmony-zenoh namespace
```
Expected: PASS

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): filters namespace for Bloom filter broadcasts"
```

---

### Task 6: StorageTier — Filter Broadcast Config, Event, and Action

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Add FilterBroadcastConfig, new event and action variants**

Add after `ContentPolicy`:

```rust
/// Configuration for periodic Bloom filter broadcasts.
#[derive(Debug, Clone)]
pub struct FilterBroadcastConfig {
    /// Broadcast after this many cache admissions.
    pub mutation_threshold: u32,
    /// Maximum seconds between broadcasts (even if no mutations).
    /// The runtime injects `FilterTimerTick` events at this interval.
    pub max_interval_ticks: u32,
    /// Expected item count for sizing the Bloom filter (should match cache_capacity).
    pub expected_items: u32,
    /// Target false positive rate.
    pub fp_rate: f64,
}

impl Default for FilterBroadcastConfig {
    fn default() -> Self {
        Self {
            mutation_threshold: 100,
            max_interval_ticks: 30,
            expected_items: 1024,
            fp_rate: 0.001,
        }
    }
}
```

Add `FilterTimerTick` variant to `StorageTierEvent`:

```rust
    /// Timer tick for periodic filter broadcasts. Injected by runtime at
    /// `FilterBroadcastConfig::max_interval_ticks` intervals.
    FilterTimerTick,
```

Add `BroadcastFilter` variant to `StorageTierAction`:

```rust
    /// Broadcast a Bloom filter snapshot of the cached CID set.
    BroadcastFilter { key_expr: String, payload: Vec<u8> },
```

**Step 2: Add filter state to `StorageTier` struct and `new()`**

Add fields to `StorageTier`:

```rust
    /// Filter broadcast configuration.
    filter_config: FilterBroadcastConfig,
    /// Cache mutations since last filter broadcast.
    mutations_since_broadcast: u32,
```

Update `StorageTier::new()` to accept `FilterBroadcastConfig` and initialize the fields. Add `harmony/filters/content/**` to subscriber declarations.

**Step 3: Add filter_config accessor**

```rust
    /// Read-only access to the filter broadcast configuration.
    pub fn filter_config(&self) -> &FilterBroadcastConfig {
        &self.filter_config
    }
```

**Step 4: Add tests**

```rust
    #[test]
    fn filter_broadcast_config_defaults() {
        let config = FilterBroadcastConfig::default();
        assert_eq!(config.mutation_threshold, 100);
        assert_eq!(config.max_interval_ticks, 30);
        assert_eq!(config.expected_items, 1024);
        assert!((config.fp_rate - 0.001).abs() < f64::EPSILON);
    }
```

**Step 5: Run tests**

```bash
cargo test -p harmony-content storage_tier
```
Expected: PASS (existing tests must still pass — the new `FilterBroadcastConfig` parameter is added to `StorageTier::new()` which means updating `make_tier_with_policy` and all direct `StorageTier::new()` calls in tests to pass the new config. Use `FilterBroadcastConfig::default()` everywhere.)

**Step 6: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): FilterBroadcastConfig, BroadcastFilter action, FilterTimerTick event"
```

---

### Task 7: StorageTier — Mutation Tracking and Filter Rebuild

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Step 1: Increment mutation counter on admission**

In `handle_transit()` after `self.cache.store_preadmitted(...)`:
```rust
self.mutations_since_broadcast += 1;
```

In `handle_publish()` after `self.cache.store(...)`:
```rust
self.mutations_since_broadcast += 1;
```

**Step 2: Add rebuild_filter method**

```rust
    /// Rebuild the Bloom filter from current cache contents and emit broadcast action.
    fn rebuild_filter(&mut self) -> StorageTierAction {
        use crate::bloom::BloomFilter;
        use harmony_zenoh::namespace::filters;

        let mut filter = BloomFilter::new(
            self.filter_config.expected_items,
            self.filter_config.fp_rate,
        );

        for cid in self.cache.iter_admitted() {
            // Only include CIDs that would be served (respect content policy).
            if self.class_admits(&cid) {
                filter.insert(&cid);
            }
        }

        self.mutations_since_broadcast = 0;

        // Use empty string as node_addr placeholder — runtime fills in actual address.
        let key_expr = filters::content_key("");
        StorageTierAction::BroadcastFilter {
            key_expr,
            payload: filter.to_bytes(),
        }
    }
```

**Step 3: Check mutation threshold after transit/publish, handle FilterTimerTick**

Add to the end of `handle_transit()` and `handle_publish()` (before returning `actions`):

```rust
        if self.mutations_since_broadcast >= self.filter_config.mutation_threshold {
            actions.push(self.rebuild_filter());
        }
```

Add `FilterTimerTick` handler in the `handle()` match:

```rust
            StorageTierEvent::FilterTimerTick => {
                let action = self.rebuild_filter();
                vec![action]
            }
```

**Step 4: Add tests**

```rust
    #[test]
    fn filter_broadcast_at_mutation_threshold() {
        let filter_config = FilterBroadcastConfig {
            mutation_threshold: 3,
            ..FilterBroadcastConfig::default()
        };
        let budget = StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        };
        let (mut tier, _) = StorageTier::new(
            MemoryBlobStore::new(),
            budget,
            ContentPolicy::default(),
            filter_config,
        );

        // First 2 transits: no BroadcastFilter.
        for i in 0..2 {
            let (cid, data) = cid_with_class(format!("item-{i}").as_bytes(), false, false);
            let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
            assert!(
                !actions.iter().any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
                "no filter broadcast before threshold"
            );
        }

        // 3rd transit hits threshold — should include BroadcastFilter.
        let (cid, data) = cid_with_class(b"item-2", false, false);
        let actions = tier.handle(StorageTierEvent::TransitContent { cid, data });
        assert!(
            actions.iter().any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
            "filter broadcast at mutation threshold"
        );
    }

    #[test]
    fn filter_timer_tick_triggers_broadcast() {
        let mut tier = make_tier_with_policy(ContentPolicy::default());
        let actions = tier.handle(StorageTierEvent::FilterTimerTick);
        assert_eq!(actions.len(), 1);
        assert!(matches!(&actions[0], StorageTierAction::BroadcastFilter { .. }));
    }

    #[test]
    fn filter_excludes_encrypted_ephemeral() {
        use crate::bloom::BloomFilter;

        let mut tier = make_tier_with_policy(ContentPolicy::default());

        // Store a PublicDurable item.
        let (pub_cid, pub_data) = cid_with_class(b"public doc", false, false);
        tier.handle(StorageTierEvent::PublishContent {
            cid: pub_cid,
            data: pub_data,
        });

        // Trigger filter rebuild via timer.
        let actions = tier.handle(StorageTierEvent::FilterTimerTick);
        let broadcast = actions.iter().find(|a| matches!(a, StorageTierAction::BroadcastFilter { .. }));
        let payload = match broadcast.unwrap() {
            StorageTierAction::BroadcastFilter { payload, .. } => payload,
            _ => unreachable!(),
        };

        let filter = BloomFilter::from_bytes(payload).unwrap();
        assert!(filter.may_contain(&pub_cid), "PublicDurable should be in filter");

        // EncryptedEphemeral should never appear even if somehow cached.
        let (ee_cid, _) = cid_with_class(b"secret", true, true);
        assert!(!filter.may_contain(&ee_cid), "EncryptedEphemeral should not be in filter");
    }
```

**Step 5: Run tests**

```bash
cargo test -p harmony-content storage_tier
```
Expected: PASS

**Step 6: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): mutation-tracked filter rebuild with BroadcastFilter emission"
```

---

### Task 8: NodeRuntime — PeerFilterTable and Filter Subscription

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Step 1: Add PeerFilterTable struct**

```rust
use harmony_content::bloom::BloomFilter;

/// Bloom filter received from a peer, with metadata.
struct PeerFilter {
    filter: BloomFilter,
    /// Tick count when this filter was received (for staleness).
    received_tick: u64,
    /// Approximate item count reported by the peer.
    item_count: u32,
}

/// Per-peer Bloom filter table for content query routing.
///
/// When a content query misses the local cache, the runtime checks
/// each peer's filter before dispatching the query. If the filter
/// says "definitely not here", the peer is skipped.
struct PeerFilterTable {
    filters: HashMap<String, PeerFilter>,
    /// Filters older than this many ticks are considered stale and dropped.
    staleness_ticks: u64,
}

impl PeerFilterTable {
    fn new(staleness_ticks: u64) -> Self {
        Self {
            filters: HashMap::new(),
            staleness_ticks,
        }
    }

    /// Update or insert a peer's filter.
    fn upsert(&mut self, peer_addr: String, filter: BloomFilter, item_count: u32, tick: u64) {
        self.filters.insert(
            peer_addr,
            PeerFilter {
                filter,
                received_tick: tick,
                item_count,
            },
        );
    }

    /// Check if a peer's filter says they might have a CID.
    /// Returns `true` (query the peer) if: no filter, stale filter, or filter says "maybe".
    /// Returns `false` (skip the peer) only if filter says "definitely not".
    fn should_query(&self, peer_addr: &str, cid: &ContentId, current_tick: u64) -> bool {
        match self.filters.get(peer_addr) {
            None => true, // No filter = assume they might have it.
            Some(pf) => {
                if current_tick.saturating_sub(pf.received_tick) > self.staleness_ticks {
                    true // Stale filter = assume they might have it.
                } else {
                    pf.filter.may_contain(cid)
                }
            }
        }
    }

    /// Remove stale entries.
    fn evict_stale(&mut self, current_tick: u64) {
        self.filters.retain(|_, pf| {
            current_tick.saturating_sub(pf.received_tick) <= self.staleness_ticks
        });
    }
}
```

**Step 2: Add `peer_filters` field and tick counter to `NodeRuntime`**

Add to struct:
```rust
    peer_filters: PeerFilterTable,
    tick_count: u64,
```

Initialize in `new()`:
```rust
    peer_filters: PeerFilterTable::new(90), // ~90 ticks staleness
    tick_count: 0,
```

Increment `tick_count` at the start of `tick()`.

**Step 3: Wire `BroadcastFilter` in `dispatch_storage_actions`**

```rust
                StorageTierAction::BroadcastFilter { key_expr, payload } => {
                    out.push(RuntimeAction::Publish { key_expr, payload });
                }
```

**Step 4: Add `FilterBroadcastConfig` to `NodeConfig` and wire it through**

Add to `NodeConfig`:
```rust
    /// Bloom filter broadcast configuration.
    pub filter_config: FilterBroadcastConfig,
```

Pass it to `StorageTier::new()` in `NodeRuntime::new()`.

Subscribe to `harmony/filters/content/**` in startup actions.

**Step 5: Add tests**

```rust
    #[test]
    fn peer_filter_table_skips_definite_miss() {
        let mut table = PeerFilterTable::new(100);
        let mut filter = BloomFilter::new(1000, 0.01);
        let cid_in = ContentId::for_blob(b"present", ContentFlags::default()).unwrap();
        let cid_out = ContentId::for_blob(b"absent", ContentFlags::default()).unwrap();
        filter.insert(&cid_in);
        table.upsert("peer-1".into(), filter, 1, 10);

        assert!(table.should_query("peer-1", &cid_in, 10));
        assert!(!table.should_query("peer-1", &cid_out, 10));
    }

    #[test]
    fn peer_filter_table_queries_unknown_peer() {
        let table = PeerFilterTable::new(100);
        let cid = ContentId::for_blob(b"test", ContentFlags::default()).unwrap();
        assert!(table.should_query("unknown", &cid, 10));
    }

    #[test]
    fn peer_filter_table_queries_stale_filter() {
        let mut table = PeerFilterTable::new(100);
        let filter = BloomFilter::new(1000, 0.01);
        // Empty filter — would return false for everything.
        table.upsert("peer-1".into(), filter, 0, 10);

        let cid = ContentId::for_blob(b"test", ContentFlags::default()).unwrap();
        // At tick 10: filter is fresh, should return false (empty filter).
        assert!(!table.should_query("peer-1", &cid, 10));
        // At tick 200: filter is stale, should return true (safe default).
        assert!(table.should_query("peer-1", &cid, 200));
    }
```

**Step 6: Run tests**

```bash
cargo test -p harmony-node
```
Expected: PASS

**Step 7: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): PeerFilterTable for Bloom filter query routing"
```

---

### Task 9: CLI Flags for Filter Configuration

**Files:**
- Modify: `crates/harmony-node/src/main.rs`

**Step 1: Add CLI flags to the `Run` command**

```rust
        /// Bloom filter broadcast interval in seconds
        #[arg(long, default_value_t = 30)]
        filter_broadcast_interval: u32,
        /// Bloom filter broadcast mutation threshold
        #[arg(long, default_value_t = 100)]
        filter_mutation_threshold: u32,
```

**Step 2: Wire into `FilterBroadcastConfig` in the `run()` function**

```rust
            let filter_config = FilterBroadcastConfig {
                mutation_threshold: filter_mutation_threshold,
                max_interval_ticks: filter_broadcast_interval,
                expected_items: cache_capacity as u32,
                fp_rate: 0.001,
            };
```

Pass `filter_config` to `NodeConfig`.

**Step 3: Add print lines for filter config**

```rust
            println!("  Filter interval: {filter_broadcast_interval}s / {filter_mutation_threshold} mutations");
```

**Step 4: Add CLI parse test**

```rust
    #[test]
    fn cli_parses_run_with_filter_config() {
        let cli = Cli::try_parse_from([
            "harmony",
            "run",
            "--filter-broadcast-ticks",
            "60",
            "--filter-mutation-threshold",
            "200",
        ])
        .unwrap();
        if let Commands::Run {
            filter_broadcast_interval,
            filter_mutation_threshold,
            ..
        } = cli.command
        {
            assert_eq!(filter_broadcast_interval, 60);
            assert_eq!(filter_mutation_threshold, 200);
        } else {
            panic!("expected Run command");
        }
    }
```

**Step 5: Run tests**

```bash
cargo test -p harmony-node
```
Expected: PASS

**Step 6: Commit**

```bash
git add crates/harmony-node/src/main.rs
git commit -m "feat(node): CLI flags for Bloom filter broadcast configuration"
```

---

### Task 10: Final — Clippy Clean and Full Test Suite

**Step 1: Run full workspace tests**

```bash
cargo test --workspace
```
Expected: All tests PASS

**Step 2: Run clippy**

```bash
cargo clippy --workspace
```
Expected: Zero warnings

**Step 3: Fix any issues found**

**Step 4: Commit if any fixes were needed**

```bash
git commit -m "fix: clippy and test cleanup for Bloom filter feature"
```

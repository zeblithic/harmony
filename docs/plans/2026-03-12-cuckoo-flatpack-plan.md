# Cuckoo Filter & Flatpack Reverse Index Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a cuckoo filter data structure and Flatpack reverse index (`child_cid → [bundle_cids...]`) to enable GC safety, dedup discovery, and impact analysis.

**Architecture:** CuckooFilter (`cuckoo.rs`) mirrors BloomFilter's API conventions. FlatpackIndex (`flatpack.rs`) owns the reverse/forward maps and a CuckooFilter instance. StorageTier delegates bundle admission/eviction to FlatpackIndex and surfaces `BroadcastCuckooFilter` actions alongside existing `BroadcastFilter` actions. NodeRuntime's PeerFilterTable gains a second filter slot for cuckoo filters. Zenoh namespace gains `harmony/filters/flatpack/{node_addr}`.

**Tech Stack:** Rust (edition 2021, `no_std` with `alloc`), same SplitMix64 hashing as `bloom.rs` and `sketch.rs`.

**Design doc:** `docs/plans/2026-03-12-cuckoo-flatpack-design.md`

**Reference files you'll need to read before implementing:**
- `crates/harmony-content/src/bloom.rs` — API conventions, FilterError pattern, hash_pair(), SplitMix64
- `crates/harmony-content/src/storage_tier.rs` — StorageTierEvent/Action enums, mutation counter, rebuild_filter(), FilterBroadcastConfig
- `crates/harmony-content/src/cache.rs` — ContentStore, iter_admitted()
- `crates/harmony-zenoh/src/namespace.rs` — filters module pattern
- `crates/harmony-node/src/runtime.rs` — PeerFilter, PeerFilterTable, route_subscription(), dispatch_storage_actions()

---

### Task 1: CuckooFilter core — struct, new(), insert(), may_contain()

**Files:**
- Create: `crates/harmony-content/src/cuckoo.rs`
- Modify: `crates/harmony-content/src/lib.rs` — add `pub mod cuckoo;`

**Context:** This mirrors `bloom.rs` in structure. The CuckooFilter uses 4-entry buckets with 12-bit fingerprints stored in `u16`. Hashing uses the same SplitMix64 mixing constants (`SEED_A`, `SEED_B`) from `bloom.rs` over `ContentId.hash`. The alternating-index trick: `i2 = i1 XOR hash(fingerprint)`.

**Step 1: Write the failing tests**

In `crates/harmony-content/src/cuckoo.rs`, add the module doc, imports, struct definition, and a `#[cfg(test)] mod tests` block with these initial tests:

```rust
//! Cuckoo filter for content discovery with deletion support.
//!
//! Nodes broadcast cuckoo filters of their Flatpack reverse index entries
//! so peers can skip reverse-lookup queries for child CIDs that are
//! definitely absent. Uses 4-entry buckets with 12-bit fingerprints and
//! the standard alternating-index trick for O(1) lookups.

use alloc::vec::Vec;
use core::fmt;

use crate::cid::ContentId;

/// SplitMix64 mixing constant A (shared with bloom.rs).
const SEED_A: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64 mixing constant B (shared with bloom.rs).
const SEED_B: u64 = 0xBF58476D1CE4E5B9;

/// Entries per bucket.
const BUCKET_SIZE: usize = 4;

/// Maximum relocation attempts before declaring the filter full.
const DEFAULT_MAX_KICKS: u32 = 500;

/// Maximum allowed `num_buckets` in deserialized filters.
/// 100K items at 95% load → ~26,316 buckets. Cap at 1M for defense-in-depth.
const MAX_BUCKETS: u32 = 1_000_000;

/// Empty fingerprint sentinel (a valid fingerprint is always non-zero).
const EMPTY: u16 = 0;

/// Errors from cuckoo filter operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuckooError {
    /// Filter is full — relocation limit exceeded.
    FilterFull,
    /// Deserialization error: input too short.
    HeaderTruncated,
    /// Deserialization error: body length mismatch.
    LengthMismatch { expected: usize, got: usize },
    /// Deserialization error: num_buckets is zero.
    ZeroBuckets,
    /// Deserialization error: num_buckets exceeds maximum.
    TooManyBuckets { got: u32, max: u32 },
}

impl fmt::Display for CuckooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CuckooError::FilterFull => write!(f, "cuckoo filter is full"),
            CuckooError::HeaderTruncated => write!(f, "input shorter than 8-byte header"),
            CuckooError::LengthMismatch { expected, got } => {
                write!(f, "body length mismatch: expected {expected}, got {got}")
            }
            CuckooError::ZeroBuckets => write!(f, "num_buckets must be non-zero"),
            CuckooError::TooManyBuckets { got, max } => {
                write!(f, "num_buckets {got} exceeds maximum {max}")
            }
        }
    }
}

/// A cuckoo filter for approximate set membership with deletion support.
///
/// Uses 4-entry buckets with 12-bit fingerprints. Supports `insert`,
/// `delete`, and `may_contain`. False positives are possible; false
/// negatives are not (assuming no hash collisions and correct usage:
/// only delete items that were previously inserted).
pub struct CuckooFilter {
    /// Bucket array. Each bucket holds 4 fingerprints (u16, only low 12 bits used).
    /// EMPTY (0) means the slot is vacant.
    buckets: Vec<[u16; BUCKET_SIZE]>,
    num_buckets: u32,
    count: u32,
    max_kicks: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(
            format!("cuckoo-test-{i}").as_bytes(),
            crate::cid::ContentFlags::default(),
        )
        .unwrap()
    }

    #[test]
    fn insert_and_may_contain_round_trip() {
        let mut cf = CuckooFilter::new(1000);
        let cid = make_cid(42);
        assert!(!cf.may_contain(&cid));
        cf.insert(&cid).unwrap();
        assert!(cf.may_contain(&cid));
        assert_eq!(cf.count(), 1);
    }

    #[test]
    fn empty_filter_returns_false() {
        let cf = CuckooFilter::new(1000);
        for i in 0..100 {
            assert!(!cf.may_contain(&make_cid(i)));
        }
        assert_eq!(cf.count(), 0);
    }

    #[test]
    fn insert_many_items() {
        let mut cf = CuckooFilter::new(1000);
        for i in 0..500 {
            cf.insert(&make_cid(i)).unwrap();
        }
        // All inserted items must be found.
        for i in 0..500 {
            assert!(cf.may_contain(&make_cid(i)), "false negative for item {i}");
        }
        assert_eq!(cf.count(), 500);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content cuckoo::tests --no-run 2>&1 | head -20`
Expected: Compilation error — `CuckooFilter::new`, `insert`, `may_contain`, `count` not implemented.

**Step 3: Implement CuckooFilter core**

Add to `crates/harmony-content/src/cuckoo.rs` (above the tests module):

```rust
impl fmt::Debug for CuckooFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CuckooFilter")
            .field("num_buckets", &self.num_buckets)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl CuckooFilter {
    /// Create a cuckoo filter sized for `capacity` items.
    ///
    /// The number of buckets is `ceil(capacity / (BUCKET_SIZE * load_factor))`
    /// where load_factor ≈ 0.95 for 4-entry buckets.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: u32) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        // Target 95% load factor for 4-entry buckets.
        let num_buckets = ((capacity as f64) / (BUCKET_SIZE as f64 * 0.95)).ceil() as u32;
        let num_buckets = num_buckets.max(1);
        Self {
            buckets: alloc::vec![[EMPTY; BUCKET_SIZE]; num_buckets as usize],
            num_buckets,
            count: 0,
            max_kicks: DEFAULT_MAX_KICKS,
        }
    }

    /// Current item count.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Number of buckets.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Insert a CID into the filter.
    ///
    /// Returns `Err(CuckooError::FilterFull)` if the relocation chain exceeds
    /// `max_kicks` without finding an empty slot.
    pub fn insert(&mut self, cid: &ContentId) -> Result<(), CuckooError> {
        let fp = fingerprint(cid);
        let (i1, i2) = bucket_indices(cid, fp, self.num_buckets);

        // Try bucket i1.
        if self.try_insert_bucket(i1, fp) {
            self.count = self.count.saturating_add(1);
            return Ok(());
        }
        // Try bucket i2.
        if self.try_insert_bucket(i2, fp) {
            self.count = self.count.saturating_add(1);
            return Ok(());
        }

        // Both full — relocate.
        self.relocate(i1, fp)
    }

    /// Check if a CID might be in the filter.
    ///
    /// Returns `false` only if the CID is definitely absent.
    pub fn may_contain(&self, cid: &ContentId) -> bool {
        let fp = fingerprint(cid);
        let (i1, i2) = bucket_indices(cid, fp, self.num_buckets);
        self.bucket_contains(i1, fp) || self.bucket_contains(i2, fp)
    }

    /// Reset all buckets to empty.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = [EMPTY; BUCKET_SIZE];
        }
        self.count = 0;
    }

    fn try_insert_bucket(&mut self, idx: u32, fp: u16) -> bool {
        let bucket = &mut self.buckets[idx as usize];
        for slot in bucket.iter_mut() {
            if *slot == EMPTY {
                *slot = fp;
                return true;
            }
        }
        false
    }

    fn bucket_contains(&self, idx: u32, fp: u16) -> bool {
        self.buckets[idx as usize].iter().any(|&s| s == fp)
    }

    fn relocate(&mut self, mut idx: u32, mut fp: u16) -> Result<(), CuckooError> {
        // Use a simple deterministic "random" slot selection based on fp.
        for _ in 0..self.max_kicks {
            // Pick a slot to evict (cycle through based on fingerprint).
            let slot = (fp as usize) % BUCKET_SIZE;
            let bucket = &mut self.buckets[idx as usize];
            core::mem::swap(&mut bucket[slot], &mut fp);

            // Compute alternate bucket for evicted fingerprint.
            idx = alt_index(idx, fp, self.num_buckets);

            if self.try_insert_bucket(idx, fp) {
                self.count = self.count.saturating_add(1);
                return Ok(());
            }
        }
        Err(CuckooError::FilterFull)
    }
}

/// Extract a 12-bit fingerprint from a CID.
///
/// Uses the first 2 bytes of the CID hash mixed with SplitMix64.
/// The fingerprint is guaranteed non-zero (EMPTY is reserved as sentinel).
fn fingerprint(cid: &ContentId) -> u16 {
    let a = u64::from_le_bytes(cid.hash[0..8].try_into().unwrap());
    let mixed = a.wrapping_mul(SEED_A);
    let fp = (mixed >> 32) as u16 & 0x0FFF; // 12 bits
    if fp == EMPTY { 1 } else { fp } // ensure non-zero
}

/// Compute the two candidate bucket indices for a CID.
fn bucket_indices(cid: &ContentId, fp: u16, num_buckets: u32) -> (u32, u32) {
    let b = u64::from_le_bytes(cid.hash[8..16].try_into().unwrap());
    let i1 = (b.wrapping_mul(SEED_B) % num_buckets as u64) as u32;
    let i2 = alt_index(i1, fp, num_buckets);
    (i1, i2)
}

/// Compute the alternate bucket index: `i XOR hash(fingerprint) % num_buckets`.
fn alt_index(i: u32, fp: u16, num_buckets: u32) -> u32 {
    let fp_hash = (fp as u64).wrapping_mul(SEED_A);
    ((i as u64) ^ fp_hash) as u32 % num_buckets
}
```

Also add to `crates/harmony-content/src/lib.rs`:

```rust
pub mod cuckoo;
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content cuckoo::tests`
Expected: 3 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cuckoo.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add CuckooFilter core — new, insert, may_contain"
```

---

### Task 2: CuckooFilter delete()

**Files:**
- Modify: `crates/harmony-content/src/cuckoo.rs`

**Context:** Delete searches both candidate buckets for a matching fingerprint and removes the first match. Returns `true` if found, `false` if not. Only delete items that were previously inserted — deleting a non-inserted item can cause false negatives.

**Step 1: Write the failing test**

```rust
#[test]
fn delete_removes_item() {
    let mut cf = CuckooFilter::new(1000);
    let cid = make_cid(42);
    cf.insert(&cid).unwrap();
    assert!(cf.may_contain(&cid));
    assert_eq!(cf.count(), 1);
    assert!(cf.delete(&cid));
    assert!(!cf.may_contain(&cid));
    assert_eq!(cf.count(), 0);
}

#[test]
fn delete_nonexistent_returns_false() {
    let mut cf = CuckooFilter::new(1000);
    let cid = make_cid(42);
    assert!(!cf.delete(&cid));
    assert_eq!(cf.count(), 0);
}

#[test]
fn delete_then_reinsert() {
    let mut cf = CuckooFilter::new(1000);
    let cid = make_cid(42);
    cf.insert(&cid).unwrap();
    cf.delete(&cid);
    assert!(!cf.may_contain(&cid));
    cf.insert(&cid).unwrap();
    assert!(cf.may_contain(&cid));
    assert_eq!(cf.count(), 1);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content cuckoo::tests::delete --no-run 2>&1 | head -10`
Expected: Compilation error — `delete` not found.

**Step 3: Implement delete**

Add to the `impl CuckooFilter` block:

```rust
/// Delete a CID from the filter.
///
/// Returns `true` if the fingerprint was found and removed.
/// Returns `false` if the CID was not in the filter.
///
/// **Important:** Only delete items that were previously inserted.
/// Deleting a non-inserted item that shares a fingerprint with an
/// inserted item will remove the wrong entry, causing false negatives.
pub fn delete(&mut self, cid: &ContentId) -> bool {
    let fp = fingerprint(cid);
    let (i1, i2) = bucket_indices(cid, fp, self.num_buckets);
    if self.try_delete_bucket(i1, fp) {
        self.count = self.count.saturating_sub(1);
        return true;
    }
    if self.try_delete_bucket(i2, fp) {
        self.count = self.count.saturating_sub(1);
        return true;
    }
    false
}

fn try_delete_bucket(&mut self, idx: u32, fp: u16) -> bool {
    let bucket = &mut self.buckets[idx as usize];
    for slot in bucket.iter_mut() {
        if *slot == fp {
            *slot = EMPTY;
            return true;
        }
    }
    false
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content cuckoo::tests`
Expected: 6 tests pass (3 old + 3 new).

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cuckoo.rs
git commit -m "feat(content): add CuckooFilter::delete()"
```

---

### Task 3: CuckooFilter serialization and false positive rate test

**Files:**
- Modify: `crates/harmony-content/src/cuckoo.rs`

**Context:** Wire format: `[num_buckets: u32 BE][item_count: u32 BE][bucket data: num_buckets * 4 * 2 bytes, each u16 BE]`. Header is 8 bytes. Mirrors bloom.rs defensive bounds checks in `from_bytes`. The FP rate test inserts n items and checks m random non-members — at 12-bit fingerprints with 4-entry buckets, theoretical FP rate is ~0.025% which is well under our 0.1% target.

**Step 1: Write the failing tests**

```rust
#[test]
fn serialization_round_trip() {
    let mut cf = CuckooFilter::new(1000);
    for i in 0..100 {
        cf.insert(&make_cid(i)).unwrap();
    }
    let bytes = cf.to_bytes();
    let cf2 = CuckooFilter::from_bytes(&bytes).unwrap();

    assert_eq!(cf.num_buckets(), cf2.num_buckets());
    assert_eq!(cf.count(), cf2.count());

    // Verify membership preserved.
    for i in 0..100 {
        assert!(cf2.may_contain(&make_cid(i)));
    }
}

#[test]
fn from_bytes_rejects_truncated() {
    let err = CuckooFilter::from_bytes(&[0u8; 4]).unwrap_err();
    assert_eq!(err, CuckooError::HeaderTruncated);
}

#[test]
fn from_bytes_rejects_zero_buckets() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&0u32.to_be_bytes()); // num_buckets = 0
    buf.extend_from_slice(&0u32.to_be_bytes()); // item_count = 0
    let err = CuckooFilter::from_bytes(&buf).unwrap_err();
    assert_eq!(err, CuckooError::ZeroBuckets);
}

#[test]
fn from_bytes_rejects_excessive_buckets() {
    let too_many = MAX_BUCKETS + 1;
    let mut buf = Vec::new();
    buf.extend_from_slice(&too_many.to_be_bytes());
    buf.extend_from_slice(&0u32.to_be_bytes());
    let err = CuckooFilter::from_bytes(&buf).unwrap_err();
    assert_eq!(err, CuckooError::TooManyBuckets { got: too_many, max: MAX_BUCKETS });
}

#[test]
fn from_bytes_rejects_wrong_body_length() {
    let mut cf = CuckooFilter::new(100);
    cf.insert(&make_cid(0)).unwrap();
    let mut bytes = cf.to_bytes();
    bytes.truncate(bytes.len() - 2); // remove one u16
    let err = CuckooFilter::from_bytes(&bytes).unwrap_err();
    match err {
        CuckooError::LengthMismatch { .. } => {}
        other => panic!("expected LengthMismatch, got {other:?}"),
    }
}

#[test]
fn false_positive_rate_within_bounds() {
    let mut cf = CuckooFilter::new(1000);
    for i in 0..1000 {
        cf.insert(&make_cid(i)).unwrap();
    }
    // Check 10000 non-members.
    let mut false_positives = 0;
    for i in 10_000..20_000 {
        if cf.may_contain(&make_cid(i)) {
            false_positives += 1;
        }
    }
    let fp_rate = false_positives as f64 / 10_000.0;
    // 12-bit fingerprints, 4-entry buckets → theoretical ~0.025%.
    // Allow up to 0.5% for safety margin.
    assert!(
        fp_rate <= 0.005,
        "false positive rate {fp_rate:.4} exceeds 0.5% bound"
    );
}

#[test]
fn insert_until_full_returns_error() {
    // Small filter: capacity 10 → ~3 buckets.
    let mut cf = CuckooFilter::new(10);
    let mut full_count = 0;
    for i in 0..100 {
        match cf.insert(&make_cid(i)) {
            Ok(()) => full_count += 1,
            Err(CuckooError::FilterFull) => break,
        }
    }
    // Should have inserted some items before getting full.
    assert!(full_count >= 10, "should insert at least capacity items");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content cuckoo::tests --no-run 2>&1 | head -10`
Expected: Compilation error — `to_bytes`, `from_bytes` not found.

**Step 3: Implement serialization**

Add to the `impl CuckooFilter` block:

```rust
/// Serialize the filter to bytes.
///
/// Wire format: `[num_buckets: u32 BE][item_count: u32 BE][bucket data]`
/// Bucket data: `num_buckets * BUCKET_SIZE * 2` bytes (each u16 big-endian).
pub fn to_bytes(&self) -> Vec<u8> {
    let header_size = 8;
    let body_size = self.num_buckets as usize * BUCKET_SIZE * 2;
    let mut buf = Vec::with_capacity(header_size + body_size);

    buf.extend_from_slice(&self.num_buckets.to_be_bytes());
    buf.extend_from_slice(&self.count.to_be_bytes());

    for bucket in &self.buckets {
        for &fp in bucket {
            buf.extend_from_slice(&fp.to_be_bytes());
        }
    }
    buf
}

/// Deserialize a filter from bytes.
pub fn from_bytes(bytes: &[u8]) -> Result<Self, CuckooError> {
    const HEADER_SIZE: usize = 8;
    if bytes.len() < HEADER_SIZE {
        return Err(CuckooError::HeaderTruncated);
    }

    let num_buckets = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
    let count = u32::from_be_bytes(bytes[4..8].try_into().unwrap());

    if num_buckets == 0 {
        return Err(CuckooError::ZeroBuckets);
    }
    if num_buckets > MAX_BUCKETS {
        return Err(CuckooError::TooManyBuckets {
            got: num_buckets,
            max: MAX_BUCKETS,
        });
    }

    let expected_body = num_buckets as usize * BUCKET_SIZE * 2;
    let got_body = bytes.len() - HEADER_SIZE;
    if got_body != expected_body {
        return Err(CuckooError::LengthMismatch {
            expected: expected_body,
            got: got_body,
        });
    }

    let mut buckets = alloc::vec![[EMPTY; BUCKET_SIZE]; num_buckets as usize];
    let mut offset = HEADER_SIZE;
    for bucket in &mut buckets {
        for slot in bucket.iter_mut() {
            *slot = u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap());
            offset += 2;
        }
    }

    Ok(CuckooFilter {
        buckets,
        num_buckets,
        count,
        max_kicks: DEFAULT_MAX_KICKS,
    })
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content cuckoo::tests`
Expected: All 13 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/cuckoo.rs
git commit -m "feat(content): add CuckooFilter serialization and FP rate test"
```

---

### Task 4: FlatpackIndex core — on_bundle_admitted, on_bundle_evicted, lookup, is_referenced

**Files:**
- Create: `crates/harmony-content/src/flatpack.rs`
- Modify: `crates/harmony-content/src/lib.rs` — add `pub mod flatpack;`

**Context:** FlatpackIndex owns a reverse map (`child_cid → Set<bundle_cid>`), forward map (`bundle_cid → Vec<child_cid>`), and a CuckooFilter. StorageTier calls `on_bundle_admitted(bundle_cid, child_cids)` when a bundle is stored. It calls `on_bundle_evicted(bundle_cid)` when a bundle is evicted — the index uses the forward map to find which child CIDs to clean up, and only deletes a child from the cuckoo filter if no other bundle references it.

The `capacity` parameter for `FlatpackIndex::new()` sizes the internal CuckooFilter.

**Step 1: Write the failing tests**

```rust
//! Flatpack reverse index: child_cid → set of bundle_cids.
//!
//! Enables GC safety (is this blob referenced?), dedup discovery (which
//! bundles share this blob?), and impact analysis (if this blob is corrupt,
//! what's affected?). Each node maintains a local index of bundles it has
//! cached, with a cuckoo filter for broadcast-based query pre-filtering.

use alloc::vec::Vec;
use core::fmt;
#[cfg(not(feature = "std"))]
use hashbrown::{HashMap, HashSet};
#[cfg(feature = "std")]
use std::collections::{HashMap, HashSet};

use crate::cid::ContentId;
use crate::cuckoo::CuckooFilter;

/// Reverse index mapping child CIDs to the bundles that reference them.
pub struct FlatpackIndex {
    /// child_cid → set of bundle_cids that reference it
    reverse: HashMap<ContentId, HashSet<ContentId>>,
    /// bundle_cid → list of child_cids (for cleanup on eviction)
    forward: HashMap<ContentId, Vec<ContentId>>,
    /// Cuckoo filter summarizing which child_cids have reverse entries
    filter: CuckooFilter,
    /// Number of mutations since last cuckoo filter broadcast
    mutations_since_broadcast: u32,
}

impl fmt::Debug for FlatpackIndex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FlatpackIndex")
            .field("bundles", &self.forward.len())
            .field("child_cids", &self.reverse.len())
            .field("filter", &self.filter)
            .field("mutations_since_broadcast", &self.mutations_since_broadcast)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cid::ContentFlags;

    fn make_cid(label: &str) -> ContentId {
        ContentId::for_blob(label.as_bytes(), ContentFlags::default()).unwrap()
    }

    #[test]
    fn admitted_bundle_creates_reverse_entries() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid("bundle-1");
        let child_a = make_cid("child-a");
        let child_b = make_cid("child-b");

        idx.on_bundle_admitted(bundle, vec![child_a, child_b]);

        assert!(idx.is_referenced(&child_a));
        assert!(idx.is_referenced(&child_b));
        let refs = idx.lookup(&child_a).unwrap();
        assert!(refs.contains(&bundle));
        assert_eq!(refs.len(), 1);
    }

    #[test]
    fn evicted_bundle_removes_reverse_entries() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid("bundle-1");
        let child = make_cid("child-a");

        idx.on_bundle_admitted(bundle, vec![child]);
        assert!(idx.is_referenced(&child));

        idx.on_bundle_evicted(&bundle);
        assert!(!idx.is_referenced(&child));
        assert!(idx.lookup(&child).is_none());
    }

    #[test]
    fn shared_child_survives_partial_eviction() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle_1 = make_cid("bundle-1");
        let bundle_2 = make_cid("bundle-2");
        let shared = make_cid("shared-child");

        idx.on_bundle_admitted(bundle_1, vec![shared]);
        idx.on_bundle_admitted(bundle_2, vec![shared]);

        // Both reference shared — evicting one should not remove it.
        idx.on_bundle_evicted(&bundle_1);
        assert!(idx.is_referenced(&shared));
        let refs = idx.lookup(&shared).unwrap();
        assert!(refs.contains(&bundle_2));
        assert!(!refs.contains(&bundle_1));

        // Evict the second — now shared is unreferenced.
        idx.on_bundle_evicted(&bundle_2);
        assert!(!idx.is_referenced(&shared));
    }

    #[test]
    fn cuckoo_filter_tracks_child_cids() {
        let mut idx = FlatpackIndex::new(1000);
        let bundle = make_cid("bundle-1");
        let child = make_cid("child-a");
        let non_child = make_cid("not-a-child");

        idx.on_bundle_admitted(bundle, vec![child]);

        // Cuckoo filter should say "maybe" for the child.
        assert!(idx.filter.may_contain(&child));
        // And "definitely not" for the non-child (modulo FP).
        // We can't assert !may_contain due to FP possibility,
        // but we can verify the child is removed after eviction.

        idx.on_bundle_evicted(&bundle);
        // After eviction, cuckoo filter should no longer contain the child.
        // (No FP concern here — we explicitly deleted it.)
        assert!(!idx.filter.may_contain(&child));
    }

    #[test]
    fn lookup_missing_returns_none() {
        let idx = FlatpackIndex::new(1000);
        assert!(idx.lookup(&make_cid("nonexistent")).is_none());
    }

    #[test]
    fn evict_unknown_bundle_is_noop() {
        let mut idx = FlatpackIndex::new(1000);
        idx.on_bundle_evicted(&make_cid("never-admitted"));
        // No panic, no state change.
        assert_eq!(idx.reverse.len(), 0);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content flatpack::tests --no-run 2>&1 | head -10`
Expected: Compilation error — `FlatpackIndex` methods not implemented.

**Step 3: Implement FlatpackIndex core**

Add to `crates/harmony-content/src/flatpack.rs` (above the tests module):

```rust
impl FlatpackIndex {
    /// Create a new empty Flatpack index.
    ///
    /// `capacity` sizes the internal cuckoo filter for the expected number
    /// of unique child CIDs across all tracked bundles.
    pub fn new(capacity: u32) -> Self {
        Self {
            reverse: HashMap::new(),
            forward: HashMap::new(),
            filter: CuckooFilter::new(capacity),
            mutations_since_broadcast: 0,
        }
    }

    /// Register a bundle and its child CIDs in the index.
    ///
    /// Called by StorageTier when a bundle is admitted to cache. StorageTier
    /// parses the bundle payload and passes the structured CID list — the
    /// index never touches raw bundle bytes.
    pub fn on_bundle_admitted(&mut self, bundle_cid: ContentId, child_cids: Vec<ContentId>) {
        for &child in &child_cids {
            let entry = self.reverse.entry(child).or_insert_with(HashSet::new);
            if entry.is_empty() {
                // New child CID — add to cuckoo filter.
                // Ignore FilterFull — the filter is best-effort.
                let _ = self.filter.insert(&child);
            }
            entry.insert(bundle_cid);
        }
        self.forward.insert(bundle_cid, child_cids);
        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
    }

    /// Remove a bundle and clean up its reverse entries.
    ///
    /// Called by StorageTier when a bundle is evicted from cache. Only
    /// removes a child CID from the cuckoo filter if no other bundle
    /// still references it.
    pub fn on_bundle_evicted(&mut self, bundle_cid: &ContentId) {
        let Some(child_cids) = self.forward.remove(bundle_cid) else {
            return;
        };
        for child in &child_cids {
            if let Some(bundles) = self.reverse.get_mut(child) {
                bundles.remove(bundle_cid);
                if bundles.is_empty() {
                    self.reverse.remove(child);
                    self.filter.delete(child);
                }
            }
        }
        self.mutations_since_broadcast = self.mutations_since_broadcast.saturating_add(1);
    }

    /// Look up which bundles reference a given child CID.
    pub fn lookup(&self, child_cid: &ContentId) -> Option<&HashSet<ContentId>> {
        self.reverse.get(child_cid).filter(|s| !s.is_empty())
    }

    /// Check if a child CID is referenced by any local bundle.
    ///
    /// Used for GC safety: if this returns `true`, the blob should not be
    /// evicted from cache.
    pub fn is_referenced(&self, child_cid: &ContentId) -> bool {
        self.reverse.get(child_cid).map_or(false, |s| !s.is_empty())
    }

    /// Number of mutations since the last cuckoo filter broadcast.
    pub fn mutations_since_broadcast(&self) -> u32 {
        self.mutations_since_broadcast
    }

    /// Reset the mutation counter (called after broadcast).
    pub fn reset_mutation_counter(&mut self) {
        self.mutations_since_broadcast = 0;
    }

    /// Access the cuckoo filter for serialization.
    pub fn filter(&self) -> &CuckooFilter {
        &self.filter
    }

    /// Rebuild the cuckoo filter from the current reverse map contents.
    ///
    /// Useful after many insert/delete cycles that may have degraded the
    /// filter's accuracy.
    pub fn rebuild_filter(&mut self) {
        self.filter.clear();
        for child_cid in self.reverse.keys() {
            let _ = self.filter.insert(child_cid);
        }
    }
}
```

Also add to `crates/harmony-content/src/lib.rs`:

```rust
pub mod flatpack;
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content flatpack::tests`
Expected: 6 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/flatpack.rs crates/harmony-content/src/lib.rs
git commit -m "feat(content): add FlatpackIndex — reverse map, forward map, cuckoo filter, GC safety"
```

---

### Task 5: Zenoh namespace — flatpack filter and query key expressions

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

**Context:** Add to the existing `filters` module: flatpack filter prefix, subscription pattern, and key builder. Also add a new `flatpack` module for query key expressions (matching the `content` module's shard pattern approach). The filters module already has `CONTENT_PREFIX`, `CONTENT_SUB`, and `content_key()` — add parallel constants for flatpack.

**Step 1: Write the failing tests**

Add to the `tests` module in `namespace.rs`:

```rust
// ── Tier 2: Flatpack Filters ─────────────────────────────────────

#[test]
fn filters_flatpack_key() {
    assert_eq!(
        filters::flatpack_key("abc123"),
        "harmony/filters/flatpack/abc123"
    );
}

#[test]
fn filters_flatpack_subscription_pattern() {
    assert_eq!(filters::FLATPACK_SUB, "harmony/filters/flatpack/**");
}

// ── Tier 2: Flatpack Queries ──────────────────────────────────────

#[test]
fn flatpack_query_key() {
    assert_eq!(
        flatpack::query_key("abcdef1234567890"),
        "harmony/flatpack/abcdef1234567890"
    );
}

#[test]
fn flatpack_subscription_pattern() {
    assert_eq!(flatpack::SUB, "harmony/flatpack/**");
}
```

Also add `flatpack::PREFIX` to the `all_prefixes_start_with_root` test array.

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh namespace::tests --no-run 2>&1 | head -10`
Expected: Compilation error — `filters::flatpack_key`, `filters::FLATPACK_SUB`, `flatpack` module not found.

**Step 3: Implement namespace additions**

Add to the `filters` module in `namespace.rs`:

```rust
/// Flatpack filter prefix: `harmony/filters/flatpack`
pub const FLATPACK_PREFIX: &str = "harmony/filters/flatpack";

/// Subscribe to all flatpack filters: `harmony/filters/flatpack/**`
pub const FLATPACK_SUB: &str = "harmony/filters/flatpack/**";

/// Flatpack filter key: `harmony/filters/flatpack/{node_addr}`
pub fn flatpack_key(node_addr: &str) -> String {
    format!("{FLATPACK_PREFIX}/{node_addr}")
}
```

Add a new `flatpack` module after the `filters` module:

```rust
/// Flatpack reverse-lookup query key expressions.
///
/// Nodes query `harmony/flatpack/{child_cid_hex}` to discover which
/// bundles reference a given child CID. Peers with matching index
/// entries respond with their bundle CID lists.
pub mod flatpack {
    use alloc::{format, string::String};
    /// Base prefix: `harmony/flatpack`
    pub const PREFIX: &str = "harmony/flatpack";

    /// Subscribe to all flatpack queries: `harmony/flatpack/**`
    pub const SUB: &str = "harmony/flatpack/**";

    /// Flatpack query key: `harmony/flatpack/{child_cid_hex}`
    pub fn query_key(child_cid_hex: &str) -> String {
        format!("{PREFIX}/{child_cid_hex}")
    }
}
```

Update the `all_prefixes_start_with_root` test to include `flatpack::PREFIX`.

**Step 4: Run tests**

Run: `cargo test -p harmony-zenoh namespace::tests`
Expected: All tests pass (existing + 4 new).

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add flatpack filter and query namespace key expressions"
```

---

### Task 6: StorageTier integration — BroadcastCuckooFilter action, FlatpackIndex wiring

**Files:**
- Modify: `crates/harmony-content/src/storage_tier.rs`

**Context:** StorageTier needs to: (1) own a FlatpackIndex, (2) call `on_bundle_admitted()` when a bundle is stored (detecting bundles by content class or a new type tag), (3) call `on_bundle_evicted()` — but eviction hooks don't exist yet in ContentStore, so this is scaffolded, (4) emit `BroadcastCuckooFilter` actions alongside `BroadcastFilter`. For now, bundle detection is deferred since the bundle content type isn't fully wired yet — we add the FlatpackIndex field, the new action variant, and the rebuild method, and hook the timer tick to also rebuild the cuckoo filter.

**Step 1: Write the failing tests**

Add tests in the `storage_tier.rs` `#[cfg(test)] mod tests` block:

```rust
#[test]
fn cuckoo_filter_broadcast_on_timer_tick() {
    let (mut tier, _) = make_tier();
    let actions = tier.handle(StorageTierEvent::FilterTimerTick);
    // Timer tick should produce both bloom and cuckoo broadcasts.
    assert!(
        actions.iter().any(|a| matches!(a, StorageTierAction::BroadcastFilter { .. })),
        "should emit BroadcastFilter"
    );
    assert!(
        actions.iter().any(|a| matches!(a, StorageTierAction::BroadcastCuckooFilter { .. })),
        "should emit BroadcastCuckooFilter"
    );
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-content storage_tier::tests::cuckoo --no-run 2>&1 | head -10`
Expected: Compilation error — `BroadcastCuckooFilter` variant not found.

**Step 3: Implement StorageTier changes**

Add `BroadcastCuckooFilter` to `StorageTierAction`:

```rust
/// Broadcast a cuckoo filter snapshot of the Flatpack reverse index.
///
/// The runtime constructs the full key expression (including node address)
/// since `StorageTier` is sans-I/O and has no identity context.
BroadcastCuckooFilter { payload: Vec<u8> },
```

Add `flatpack` field to `StorageTier`:

```rust
use crate::flatpack::FlatpackIndex;

// In the struct:
/// Flatpack reverse index (child_cid → bundle_cids).
flatpack: FlatpackIndex,
```

Initialize in `StorageTier::new()`:

```rust
flatpack: FlatpackIndex::new(filter_config.expected_items),
```

Add `rebuild_cuckoo_filter` method:

```rust
/// Rebuild the cuckoo filter from the Flatpack index and return
/// a `BroadcastCuckooFilter` action. Resets the Flatpack mutation counter.
fn rebuild_cuckoo_filter(&mut self) -> StorageTierAction {
    self.flatpack.rebuild_filter();
    self.flatpack.reset_mutation_counter();
    StorageTierAction::BroadcastCuckooFilter {
        payload: self.flatpack.filter().to_bytes(),
    }
}
```

Modify `rebuild_filter` (called on timer tick) to also return the cuckoo filter, or modify the `FilterTimerTick` handler. The cleanest approach: modify the `FilterTimerTick` handler to return both:

In the `handle` method, change the `FilterTimerTick` arm from:

```rust
StorageTierEvent::FilterTimerTick => {
    vec![self.rebuild_filter()]
}
```

To:

```rust
StorageTierEvent::FilterTimerTick => {
    vec![self.rebuild_filter(), self.rebuild_cuckoo_filter()]
}
```

Also modify the threshold-triggered broadcast spots (in `handle_transit`, `handle_publish`, `DiskReadComplete`) to also emit cuckoo filter broadcasts when the flatpack mutation counter reaches threshold. Add a helper:

```rust
/// Check if the Flatpack index needs a cuckoo filter broadcast.
fn maybe_broadcast_cuckoo(&mut self) -> Option<StorageTierAction> {
    if self.flatpack.mutations_since_broadcast() >= self.filter_config.mutation_threshold {
        Some(self.rebuild_cuckoo_filter())
    } else {
        None
    }
}
```

Call `maybe_broadcast_cuckoo()` after each `rebuild_filter()` call in the threshold checks, appending the action if `Some`.

Add read-only access to the flatpack index:

```rust
/// Read-only access to the Flatpack reverse index.
pub fn flatpack(&self) -> &FlatpackIndex {
    &self.flatpack
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-content storage_tier::tests`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-content/src/storage_tier.rs
git commit -m "feat(content): wire FlatpackIndex into StorageTier with BroadcastCuckooFilter action"
```

---

### Task 7: NodeRuntime — PeerFilterTable cuckoo slot, subscribe to flatpack broadcasts

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

**Context:** PeerFilter currently holds one BloomFilter. It needs a second field for an optional CuckooFilter. PeerFilterTable needs a `should_query_flatpack()` method that checks the cuckoo filter. The runtime needs to subscribe to `harmony/filters/flatpack/**` and route received cuckoo filter payloads into PeerFilterTable. The `dispatch_storage_actions` method needs to handle `BroadcastCuckooFilter` actions alongside `BroadcastFilter`.

**Step 1: Write the failing tests**

```rust
#[test]
fn cuckoo_filter_broadcast_dispatched() {
    use harmony_content::blob::MemoryBlobStore;
    use harmony_content::cid::ContentFlags;

    let config = NodeConfig {
        storage_budget: StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        },
        compute_budget: InstructionBudget { fuel: 1000 },
        schedule: Default::default(),
        content_policy: ContentPolicy::default(),
        filter_broadcast_config: FilterBroadcastConfig::default(),
        node_addr: "cuckoo-test".to_string(),
    };
    let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

    // Force a timer tick which should produce both filter broadcasts.
    for _ in 0..31 {
        rt.tick();
    }
    let actions = rt.tick();

    // Should have both a bloom and a cuckoo filter publish.
    let filter_publishes: Vec<_> = actions
        .iter()
        .filter(|a| matches!(a, RuntimeAction::Publish { key_expr, .. }
            if key_expr.starts_with("harmony/filters/")))
        .collect();
    assert!(
        filter_publishes.len() >= 2,
        "expected bloom + cuckoo filter broadcasts, got {}",
        filter_publishes.len()
    );
}

#[test]
fn route_subscription_parses_cuckoo_filter() {
    use harmony_content::blob::MemoryBlobStore;
    use harmony_content::cuckoo::CuckooFilter;

    let config = NodeConfig {
        storage_budget: StorageBudget {
            cache_capacity: 100,
            max_pinned_bytes: 1_000_000,
        },
        compute_budget: InstructionBudget { fuel: 1000 },
        schedule: Default::default(),
        content_policy: ContentPolicy::default(),
        filter_broadcast_config: FilterBroadcastConfig::default(),
        node_addr: "self-node".to_string(),
    };
    let (mut rt, _) = NodeRuntime::new(config, MemoryBlobStore::new());

    // Create and serialize a cuckoo filter.
    let cf = CuckooFilter::new(100);
    let payload = cf.to_bytes();

    // Send it as a flatpack filter from a peer.
    rt.push_event(RuntimeEvent::SubscriptionMessage {
        key_expr: "harmony/filters/flatpack/peer-xyz".to_string(),
        payload,
    });
    rt.tick();

    // No parse errors.
    assert_eq!(rt.peer_filter_parse_errors(), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-node runtime::tests --no-run 2>&1 | head -10`

**Step 3: Implement runtime changes**

1. Add `use harmony_content::cuckoo::CuckooFilter;` to imports.

2. Modify `PeerFilter` to hold both filters:

```rust
struct PeerFilter {
    content_filter: Option<BloomFilter>,
    flatpack_filter: Option<CuckooFilter>,
    received_tick: u64,
}
```

3. Update `PeerFilterTable::upsert` to take a filter type parameter (or add a second upsert method):

```rust
fn upsert_content(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
    let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
        content_filter: None,
        flatpack_filter: None,
        received_tick: tick,
    });
    entry.content_filter = Some(filter);
    entry.received_tick = tick;
}

fn upsert_flatpack(&mut self, peer_addr: String, filter: CuckooFilter, tick: u64) {
    let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
        content_filter: None,
        flatpack_filter: None,
        received_tick: tick,
    });
    entry.flatpack_filter = Some(filter);
    entry.received_tick = tick;
}
```

4. Update `should_query` to use `content_filter`:

```rust
fn should_query(&self, peer_addr: &str, cid: &ContentId, current_tick: u64) -> bool {
    match self.filters.get(peer_addr) {
        None => true,
        Some(pf) => {
            if current_tick.saturating_sub(pf.received_tick) > self.staleness_ticks {
                true
            } else {
                match &pf.content_filter {
                    Some(bf) => bf.may_contain(cid),
                    None => true,
                }
            }
        }
    }
}
```

5. Add `should_query_flatpack`:

```rust
/// Returns true if the peer should be queried for flatpack entries.
fn should_query_flatpack(&self, peer_addr: &str, child_cid: &ContentId, current_tick: u64) -> bool {
    match self.filters.get(peer_addr) {
        None => true,
        Some(pf) => {
            if current_tick.saturating_sub(pf.received_tick) > self.staleness_ticks {
                true
            } else {
                match &pf.flatpack_filter {
                    Some(cf) => cf.may_contain(child_cid),
                    None => true,
                }
            }
        }
    }
}
```

6. Update `route_subscription` to also check for flatpack filter broadcasts:

```rust
fn route_subscription(&mut self, key_expr: String, payload: Vec<u8>) {
    // Check if this is a content filter broadcast.
    if let Some(peer_addr) = key_expr
        .strip_prefix(harmony_zenoh::namespace::filters::CONTENT_PREFIX)
        .and_then(|s| s.strip_prefix('/'))
    {
        if peer_addr != self.node_addr {
            match BloomFilter::from_bytes(&payload) {
                Ok(filter) => {
                    self.peer_filters.upsert_content(
                        peer_addr.to_string(),
                        filter,
                        self.tick_count,
                    );
                }
                Err(_) => {
                    self.peer_filters.record_parse_error();
                }
            }
        }
        return;
    }

    // Check if this is a flatpack filter broadcast.
    if let Some(peer_addr) = key_expr
        .strip_prefix(harmony_zenoh::namespace::filters::FLATPACK_PREFIX)
        .and_then(|s| s.strip_prefix('/'))
    {
        if peer_addr != self.node_addr {
            match CuckooFilter::from_bytes(&payload) {
                Ok(filter) => {
                    self.peer_filters.upsert_flatpack(
                        peer_addr.to_string(),
                        filter,
                        self.tick_count,
                    );
                }
                Err(_) => {
                    self.peer_filters.record_parse_error();
                }
            }
        }
        return;
    }

    if let Some(event) = self.parse_subscription_event(&key_expr, payload) {
        self.storage_queue.push_back(event);
    }
}
```

7. Subscribe to flatpack filter broadcasts in `new()`:

```rust
actions.push(RuntimeAction::Subscribe {
    key_expr: harmony_zenoh::namespace::filters::FLATPACK_SUB.to_string(),
});
```

8. Handle `BroadcastCuckooFilter` in `dispatch_storage_actions`:

```rust
StorageTierAction::BroadcastCuckooFilter { payload } => {
    self.pending_cuckoo_broadcast = Some(payload);
    // No timer reset needed — cuckoo shares the same broadcast cycle.
}
```

9. Add `pending_cuckoo_broadcast: Option<Vec<u8>>` field to `NodeRuntime`, initialize to `None`.

10. Flush the cuckoo broadcast alongside the bloom broadcast in the tick method:

```rust
if let Some(payload) = self.pending_cuckoo_broadcast.take() {
    let key_expr =
        harmony_zenoh::namespace::filters::flatpack_key(&self.node_addr);
    actions.push(RuntimeAction::Publish { key_expr, payload });
}
```

11. Add public method:

```rust
/// Check if a peer should be queried for flatpack reverse-lookup entries.
pub fn should_query_peer_flatpack(&self, peer_addr: &str, child_cid: &ContentId) -> bool {
    self.peer_filters
        .should_query_flatpack(peer_addr, child_cid, self.tick_count)
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-node runtime::tests`
Expected: All tests pass (existing + new).

**Step 5: Run full workspace**

Run: `cargo test --workspace`
Expected: All tests pass.

Run: `cargo clippy --workspace`
Expected: Zero warnings.

**Step 6: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(node): wire cuckoo filter into PeerFilterTable and filter broadcast dispatch"
```

---

### Task 8: Integration test — full cycle

**Files:**
- Modify: `crates/harmony-content/src/flatpack.rs` — add integration-style test

**Context:** Test the full cycle: admit bundle → index built → cuckoo filter serialization round-trip → membership confirms child CIDs. Also test GC safety: referenced child blocks eviction consideration, unreferenced child allows it.

**Step 1: Write the integration tests**

Add to the test module in `flatpack.rs`:

```rust
#[test]
fn full_cycle_admit_broadcast_deserialize_lookup() {
    use crate::cuckoo::CuckooFilter;

    let mut idx = FlatpackIndex::new(1000);
    let bundle = make_cid("bundle-1");
    let child_a = make_cid("child-a");
    let child_b = make_cid("child-b");

    // Admit bundle.
    idx.on_bundle_admitted(bundle, vec![child_a, child_b]);

    // Serialize cuckoo filter (simulates broadcast).
    let bytes = idx.filter().to_bytes();

    // Deserialize (simulates receiving peer).
    let remote_filter = CuckooFilter::from_bytes(&bytes).unwrap();

    // Remote filter should say "maybe" for both children.
    assert!(remote_filter.may_contain(&child_a));
    assert!(remote_filter.may_contain(&child_b));

    // Remote filter should say "definitely not" for unrelated CIDs (modulo FP).
    let unrelated = make_cid("not-a-child");
    // Can't assert !may_contain (FP), but verify count matches.
    assert_eq!(remote_filter.count(), 2);
}

#[test]
fn gc_safety_blocks_referenced_eviction() {
    let mut idx = FlatpackIndex::new(1000);
    let bundle = make_cid("pinned-bundle");
    let blob = make_cid("important-blob");

    idx.on_bundle_admitted(bundle, vec![blob]);

    // Simulate GC check before evicting blob.
    assert!(idx.is_referenced(&blob), "referenced blob must not be evicted");

    // After bundle eviction, blob becomes unreferenced.
    idx.on_bundle_evicted(&bundle);
    assert!(!idx.is_referenced(&blob), "unreferenced blob can be evicted");
}

#[test]
fn rebuild_filter_matches_current_state() {
    let mut idx = FlatpackIndex::new(1000);
    let bundle = make_cid("bundle-1");
    let child_a = make_cid("child-a");
    let child_b = make_cid("child-b");
    let child_c = make_cid("child-c");

    // Admit, then evict one child's only bundle.
    let bundle_2 = make_cid("bundle-2");
    idx.on_bundle_admitted(bundle, vec![child_a, child_b]);
    idx.on_bundle_admitted(bundle_2, vec![child_c]);
    idx.on_bundle_evicted(&bundle_2);

    // Rebuild filter from scratch.
    idx.rebuild_filter();

    // Filter should match: child_a and child_b yes, child_c no.
    assert!(idx.filter().may_contain(&child_a));
    assert!(idx.filter().may_contain(&child_b));
    // child_c was evicted and should have been deleted from filter.
    assert!(!idx.filter().may_contain(&child_c));
}

#[test]
fn mutation_counter_increments() {
    let mut idx = FlatpackIndex::new(1000);
    assert_eq!(idx.mutations_since_broadcast(), 0);

    idx.on_bundle_admitted(make_cid("b1"), vec![make_cid("c1")]);
    assert_eq!(idx.mutations_since_broadcast(), 1);

    idx.on_bundle_evicted(&make_cid("b1"));
    assert_eq!(idx.mutations_since_broadcast(), 2);

    idx.reset_mutation_counter();
    assert_eq!(idx.mutations_since_broadcast(), 0);
}
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content flatpack::tests`
Expected: All 10 tests pass.

**Step 3: Final workspace check**

Run: `cargo test --workspace`
Run: `cargo clippy --workspace`
Run: `cargo fmt --all -- --check`
Expected: All pass.

**Step 4: Commit**

```bash
git add crates/harmony-content/src/flatpack.rs
git commit -m "test(content): add integration tests for Flatpack full cycle, GC safety, and rebuild"
```

---

## Summary

| Task | What | Key Files |
|------|------|-----------|
| 1 | CuckooFilter core (new, insert, may_contain) | `cuckoo.rs` (new), `lib.rs` |
| 2 | CuckooFilter delete | `cuckoo.rs` |
| 3 | CuckooFilter serialization + FP rate test | `cuckoo.rs` |
| 4 | FlatpackIndex core (admit, evict, lookup, is_referenced) | `flatpack.rs` (new), `lib.rs` |
| 5 | Zenoh namespace (flatpack filter + query keys) | `namespace.rs` |
| 6 | StorageTier integration (BroadcastCuckooFilter, FlatpackIndex field) | `storage_tier.rs` |
| 7 | NodeRuntime (PeerFilter cuckoo slot, subscribe, dispatch) | `runtime.rs` |
| 8 | Integration tests (full cycle, GC safety, rebuild) | `flatpack.rs` |

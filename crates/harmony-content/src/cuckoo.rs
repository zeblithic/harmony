//! Cuckoo filter for compact approximate set membership testing.
//!
//! A space-efficient probabilistic data structure that supports both
//! insertion and membership queries on [`ContentId`] values.  Uses
//! 4-entry buckets with 12-bit fingerprints, achieving higher space
//! efficiency than a Bloom filter at comparable false positive rates
//! while also supporting deletion.
//!
//! Hashing reuses the SplitMix64 constants from [`super::bloom`] so
//! both filters derive their entropy from the same CID hash bytes.

use alloc::{vec, vec::Vec};
use core::fmt;

use crate::cid::ContentId;

/// SplitMix64 mixing constant A.
const SEED_A: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64 mixing constant B.
const SEED_B: u64 = 0xBF58476D1CE4E5B9;

/// Number of entries per bucket.
const BUCKET_SIZE: usize = 4;

/// Bit width of each fingerprint.
const FINGERPRINT_BITS: u32 = 12;

/// Mask for extracting a 12-bit fingerprint.
const FINGERPRINT_MASK: u16 = 0xFFF;

/// Empty sentinel — a fingerprint of zero means the slot is vacant.
const EMPTY: u16 = 0;

/// Default maximum number of eviction kicks before declaring the filter full.
const DEFAULT_MAX_KICKS: u32 = 500;

/// Maximum number of buckets allowed (defense-in-depth for deserialization).
///
/// Used by [`CuckooFilter::from_bytes`] to reject untrusted headers.
const MAX_BUCKETS: u32 = 1_000_000;

/// Target load factor used to size the bucket array.
const LOAD_FACTOR: f64 = 0.95;

/// Errors that can occur when operating on a [`CuckooFilter`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CuckooError {
    /// The filter is full and cannot accept more items after maximum kicks.
    FilterFull,
    /// Input is shorter than the serialization header.
    HeaderTruncated,
    /// Body length does not match the expected bucket data size.
    LengthMismatch {
        /// Expected body length in bytes.
        expected: usize,
        /// Actual body length in bytes.
        got: usize,
    },
    /// `num_buckets` in the header is zero.
    ZeroBuckets,
    /// `num_buckets` exceeds the maximum allowed value.
    TooManyBuckets {
        /// The value found in the header.
        got: u32,
        /// The maximum allowed value.
        max: u32,
    },
    /// `num_buckets` is not a power of two (required for alt_index involution).
    NotPowerOfTwo {
        /// The value found in the header.
        got: u32,
    },
}

impl fmt::Display for CuckooError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CuckooError::FilterFull => {
                write!(f, "cuckoo filter is full after {DEFAULT_MAX_KICKS} kicks")
            }
            CuckooError::HeaderTruncated => write!(f, "input shorter than serialization header"),
            CuckooError::LengthMismatch { expected, got } => {
                write!(f, "body length mismatch: expected {expected}, got {got}")
            }
            CuckooError::ZeroBuckets => write!(f, "num_buckets must be non-zero"),
            CuckooError::TooManyBuckets { got, max } => {
                write!(f, "num_buckets {got} exceeds maximum {max}")
            }
            CuckooError::NotPowerOfTwo { got } => {
                write!(f, "num_buckets {got} is not a power of two")
            }
        }
    }
}

/// A cuckoo filter for approximate set membership testing of [`ContentId`] values.
///
/// Each bucket holds [`BUCKET_SIZE`] fingerprint slots stored as `u16` values.
/// Fingerprints are 12 bits wide; the value `0` is reserved as the empty
/// sentinel, so all stored fingerprints are guaranteed non-zero.
pub struct CuckooFilter {
    /// Bucket array: each bucket is `BUCKET_SIZE` fingerprint slots.
    buckets: Vec<[u16; BUCKET_SIZE]>,
    /// Number of buckets in the filter.
    num_buckets: u32,
    /// Maximum eviction kicks before declaring the filter full.
    max_kicks: u32,
    /// Number of items currently stored.
    count: u32,
}

impl fmt::Debug for CuckooFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CuckooFilter")
            .field("num_buckets", &self.num_buckets)
            .field("count", &self.count)
            .finish_non_exhaustive()
    }
}

impl CuckooFilter {
    /// Create a new cuckoo filter sized for the expected capacity.
    ///
    /// The number of buckets is computed as `ceil(capacity / (BUCKET_SIZE * LOAD_FACTOR))`
    /// then rounded up to the next power of 2. Power-of-2 sizing is required
    /// so that `alt_index(alt_index(i, fp), fp) == i` — the XOR-based
    /// alternate index trick only works when the modulus is a power of 2.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: u32) -> Self {
        assert!(capacity > 0, "capacity must be > 0");

        let slots_needed = (capacity as f64) / LOAD_FACTOR;
        let num_buckets = (slots_needed / BUCKET_SIZE as f64).ceil() as u32;
        // Round up to next power of 2 so that alt_index is a proper involution.
        let num_buckets = num_buckets.max(1).next_power_of_two();

        CuckooFilter {
            buckets: vec![[EMPTY; BUCKET_SIZE]; num_buckets as usize],
            num_buckets,
            max_kicks: DEFAULT_MAX_KICKS,
            count: 0,
        }
    }

    /// Insert a content ID into the filter.
    ///
    /// Returns `Ok(())` on success, or `Err(CuckooError::FilterFull)` if the
    /// item could not be placed after [`DEFAULT_MAX_KICKS`] eviction attempts.
    ///
    /// On failure the filter is unchanged — the eviction chain is rolled back
    /// so that no previously-inserted fingerprint is lost.
    pub fn insert(&mut self, cid: &ContentId) -> Result<(), CuckooError> {
        let (fp, i1) = fingerprint_and_index(cid, self.num_buckets);
        let i2 = alt_index(i1, fp, self.num_buckets);

        // Try the two candidate buckets first.
        if self.try_insert_at(i1 as usize, fp) {
            self.count = self.count.saturating_add(1);
            return Ok(());
        }
        if self.try_insert_at(i2 as usize, fp) {
            self.count = self.count.saturating_add(1);
            return Ok(());
        }

        // Both buckets are full — begin eviction.
        // Track each swap so we can roll back on failure.
        let mut idx = if fp as u32 % 2 == 0 { i1 } else { i2 };
        let mut evicted_fp = fp;
        let mut rollback: Vec<(u32, usize, u16)> = Vec::new();

        for _ in 0..self.max_kicks {
            let slot = evicted_fp as usize % BUCKET_SIZE;
            let original_fp = self.buckets[idx as usize][slot];
            self.buckets[idx as usize][slot] = evicted_fp;
            rollback.push((idx, slot, original_fp));
            evicted_fp = original_fp;

            idx = alt_index(idx, evicted_fp, self.num_buckets);
            if self.try_insert_at(idx as usize, evicted_fp) {
                self.count = self.count.saturating_add(1);
                return Ok(());
            }
        }

        // Eviction failed — roll back all swaps in reverse order.
        for (bucket_idx, slot, original_fp) in rollback.into_iter().rev() {
            self.buckets[bucket_idx as usize][slot] = original_fp;
        }
        Err(CuckooError::FilterFull)
    }

    /// Test whether a content ID *may* be in the filter.
    ///
    /// Returns `true` if the item might be present (with some false positive
    /// probability), or `false` if the item is definitely absent.
    pub fn may_contain(&self, cid: &ContentId) -> bool {
        let (fp, i1) = fingerprint_and_index(cid, self.num_buckets);
        let i2 = alt_index(i1, fp, self.num_buckets);

        self.bucket_contains(i1 as usize, fp) || self.bucket_contains(i2 as usize, fp)
    }

    /// Clear all buckets, resetting the filter to empty.
    pub fn clear(&mut self) {
        for bucket in &mut self.buckets {
            *bucket = [EMPTY; BUCKET_SIZE];
        }
        self.count = 0;
    }

    /// Returns the number of items currently stored in the filter.
    pub fn count(&self) -> u32 {
        self.count
    }

    /// Returns the number of buckets in the filter.
    pub fn num_buckets(&self) -> u32 {
        self.num_buckets
    }

    /// Try to insert a fingerprint into a bucket, returning true on success.
    ///
    /// Does NOT check for duplicate fingerprints — two distinct CIDs can
    /// legitimately share the same 12-bit fingerprint (that's how false
    /// positives arise). Callers that need set semantics (no duplicate CIDs)
    /// must guard against re-insertion externally.
    fn try_insert_at(&mut self, bucket_idx: usize, fp: u16) -> bool {
        for slot in &mut self.buckets[bucket_idx] {
            if *slot == EMPTY {
                *slot = fp;
                return true;
            }
        }
        false
    }

    /// Check whether a bucket contains a given fingerprint.
    fn bucket_contains(&self, bucket_idx: usize, fp: u16) -> bool {
        self.buckets[bucket_idx].contains(&fp)
    }

    /// Serialize the filter to a byte vector.
    ///
    /// Wire format: `[num_buckets: u32 BE][item_count: u32 BE][bucket data]`
    /// where bucket data is `num_buckets * BUCKET_SIZE` fingerprints, each as
    /// a `u16` in big-endian order.
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

    /// Deserialize a filter from bytes produced by [`to_bytes`](Self::to_bytes).
    ///
    /// Validates that the header is present, `num_buckets` is in range, and
    /// the body length matches the declared bucket count.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CuckooError> {
        const HEADER_SIZE: usize = 8;
        if bytes.len() < HEADER_SIZE {
            return Err(CuckooError::HeaderTruncated);
        }
        let num_buckets = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        // Wire count is ignored — recomputed from actual bucket contents below.
        let _wire_count = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        if num_buckets == 0 {
            return Err(CuckooError::ZeroBuckets);
        }
        if num_buckets > MAX_BUCKETS {
            return Err(CuckooError::TooManyBuckets {
                got: num_buckets,
                max: MAX_BUCKETS,
            });
        }
        if !num_buckets.is_power_of_two() {
            return Err(CuckooError::NotPowerOfTwo { got: num_buckets });
        }
        let expected_body = num_buckets as usize * BUCKET_SIZE * 2;
        let got_body = bytes.len() - HEADER_SIZE;
        if got_body != expected_body {
            return Err(CuckooError::LengthMismatch {
                expected: expected_body,
                got: got_body,
            });
        }
        let mut buckets = vec![[EMPTY; BUCKET_SIZE]; num_buckets as usize];
        let mut offset = HEADER_SIZE;
        for bucket in &mut buckets {
            for slot in bucket.iter_mut() {
                *slot = u16::from_be_bytes(bytes[offset..offset + 2].try_into().unwrap());
                offset += 2;
            }
        }
        // Recompute count from actual bucket contents rather than trusting
        // the wire value — prevents a malicious or corrupted payload from
        // providing a misleading count.
        let actual_count = buckets
            .iter()
            .flat_map(|b| b.iter())
            .filter(|&&slot| slot != EMPTY)
            .count() as u32;
        Ok(CuckooFilter {
            buckets,
            num_buckets,
            count: actual_count,
            max_kicks: DEFAULT_MAX_KICKS,
        })
    }

    /// Remove a content ID from the filter, returning `true` if it was found.
    ///
    /// Computes the fingerprint and both candidate bucket indices, then
    /// searches both buckets for a matching fingerprint. If found, the slot
    /// is cleared and the count decremented.
    pub fn delete(&mut self, cid: &ContentId) -> bool {
        let (fp, i1) = fingerprint_and_index(cid, self.num_buckets);
        let i2 = alt_index(i1, fp, self.num_buckets);
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

    /// Try to remove a fingerprint from the given bucket, returning `true` if found.
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
}

/// Derive a non-zero 12-bit fingerprint and primary bucket index from a CID.
///
/// Uses the same hash byte ranges as [`super::bloom::hash_pair`]:
/// - `hash[0..8]` mixed with `SEED_A` produces the fingerprint
/// - `hash[8..16]` mixed with `SEED_B` produces the bucket index
fn fingerprint_and_index(cid: &ContentId, num_buckets: u32) -> (u16, u32) {
    let a = u64::from_le_bytes(cid.hash[0..8].try_into().unwrap());
    let b = u64::from_le_bytes(cid.hash[8..16].try_into().unwrap());

    let fp_raw = a.wrapping_mul(SEED_A) ^ b;
    let fp = to_nonzero_fingerprint(fp_raw);

    let idx_hash = b.wrapping_mul(SEED_B) ^ a;
    let idx = (idx_hash % num_buckets as u64) as u32;

    (fp, idx)
}

/// Collapse a u64 into a non-zero 12-bit fingerprint.
///
/// Takes the bottom 12 bits; if zero, folds in the upper bits until non-zero,
/// falling back to 1 as a last resort.
fn to_nonzero_fingerprint(raw: u64) -> u16 {
    let mut fp = (raw as u16) & FINGERPRINT_MASK;
    if fp == 0 {
        fp = ((raw >> FINGERPRINT_BITS) as u16) & FINGERPRINT_MASK;
    }
    if fp == 0 {
        fp = ((raw >> (FINGERPRINT_BITS * 2)) as u16) & FINGERPRINT_MASK;
    }
    if fp == 0 {
        fp = 1;
    }
    fp
}

/// Compute the alternate bucket index for a given fingerprint.
///
/// `alt = (index XOR hash(fingerprint)) % num_buckets`
///
/// The fingerprint is hashed before XOR to break clustering that would occur
/// if nearby indices and similar fingerprints mapped to the same alternate.
fn alt_index(index: u32, fp: u16, num_buckets: u32) -> u32 {
    let fp_hash = (fp as u32).wrapping_mul(SEED_A as u32);
    (index ^ fp_hash) % num_buckets
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
        for i in 0..500 {
            assert!(cf.may_contain(&make_cid(i)), "false negative for item {i}");
        }
        assert_eq!(cf.count(), 500);
    }

    #[test]
    #[should_panic(expected = "capacity must be > 0")]
    fn new_rejects_zero_capacity() {
        CuckooFilter::new(0);
    }

    #[test]
    fn debug_impl_shows_summary() {
        let cf = CuckooFilter::new(1000);
        let dbg = format!("{cf:?}");
        assert!(
            dbg.contains("CuckooFilter"),
            "missing 'CuckooFilter': {dbg}"
        );
        assert!(dbg.contains("num_buckets"), "missing 'num_buckets': {dbg}");
        assert!(dbg.contains("count"), "missing 'count': {dbg}");
        assert!(dbg.contains(".."), "should use finish_non_exhaustive()");
    }

    #[test]
    fn clear_resets_all() {
        let mut cf = CuckooFilter::new(1000);
        for i in 0..50 {
            cf.insert(&make_cid(i)).unwrap();
        }
        assert_eq!(cf.count(), 50);
        cf.clear();
        assert_eq!(cf.count(), 0);
        for i in 0..50 {
            assert!(!cf.may_contain(&make_cid(i)), "item {i} survived clear");
        }
    }

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
        buf.extend_from_slice(&0u32.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        let err = CuckooFilter::from_bytes(&buf).unwrap_err();
        assert_eq!(err, CuckooError::ZeroBuckets);
    }

    #[test]
    fn from_bytes_rejects_non_power_of_two() {
        // 3 is valid (non-zero, within bounds) but not a power of 2.
        let num_buckets: u32 = 3;
        let mut buf = Vec::new();
        buf.extend_from_slice(&num_buckets.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        // Provide correct body length for 3 buckets.
        buf.extend_from_slice(&vec![0u8; 3 * 4 * 2]);
        let err = CuckooFilter::from_bytes(&buf).unwrap_err();
        assert_eq!(err, CuckooError::NotPowerOfTwo { got: 3 });
    }

    #[test]
    fn from_bytes_rejects_excessive_buckets() {
        let too_many = super::MAX_BUCKETS + 1;
        let mut buf = Vec::new();
        buf.extend_from_slice(&too_many.to_be_bytes());
        buf.extend_from_slice(&0u32.to_be_bytes());
        let err = CuckooFilter::from_bytes(&buf).unwrap_err();
        assert_eq!(
            err,
            CuckooError::TooManyBuckets {
                got: too_many,
                max: super::MAX_BUCKETS,
            }
        );
    }

    #[test]
    fn from_bytes_rejects_wrong_body_length() {
        let mut cf = CuckooFilter::new(100);
        cf.insert(&make_cid(0)).unwrap();
        let mut bytes = cf.to_bytes();
        bytes.truncate(bytes.len() - 2);
        let err = CuckooFilter::from_bytes(&bytes).unwrap_err();
        match err {
            CuckooError::LengthMismatch { .. } => {}
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        // Over-provision so all 1000 items fit without hitting FilterFull.
        let mut cf = CuckooFilter::new(2000);
        for i in 0..1000 {
            cf.insert(&make_cid(i)).unwrap();
        }
        let mut false_positives = 0;
        for i in 10_000..20_000 {
            if cf.may_contain(&make_cid(i)) {
                false_positives += 1;
            }
        }
        let fp_rate = false_positives as f64 / 10_000.0;
        // With 12-bit fingerprints the theoretical FP rate is ~0.2%.
        // Power-of-2 bucket rounding can push the observed rate slightly
        // higher at low load factors, so we allow up to 1%.
        assert!(
            fp_rate <= 0.01,
            "false positive rate {fp_rate:.4} exceeds 1% bound"
        );
    }

    #[test]
    fn insert_until_full_returns_error() {
        let mut cf = CuckooFilter::new(10);
        let mut full_count = 0;
        for i in 0..100 {
            match cf.insert(&make_cid(i)) {
                Ok(()) => full_count += 1,
                Err(CuckooError::FilterFull) => break,
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        assert!(
            full_count >= 10,
            "should insert at least capacity items, got {full_count}"
        );
    }

    #[test]
    fn failed_insert_does_not_corrupt_existing_items() {
        // Fill a small filter, then attempt an insert that fails.
        // All previously-inserted items must still be found.
        let mut cf = CuckooFilter::new(10);
        let mut inserted = Vec::new();
        for i in 0..200 {
            let cid = make_cid(i);
            match cf.insert(&cid) {
                Ok(()) => inserted.push(cid),
                Err(CuckooError::FilterFull) => {
                    // Verify rollback: every previously-inserted item is still present.
                    for prev in &inserted {
                        assert!(
                            cf.may_contain(prev),
                            "false negative after failed insert (rollback broken)"
                        );
                    }
                    return;
                }
                Err(other) => panic!("unexpected error: {other:?}"),
            }
        }
        // If we never hit FilterFull, that's fine — test still passes.
    }

    #[test]
    fn alt_index_is_involution() {
        // For correctness, alt_index(alt_index(i, fp), fp) must equal i.
        // This property requires power-of-2 bucket counts.
        let cf = CuckooFilter::new(1000);
        for i in 0..100 {
            let cid = make_cid(i);
            let (fp, i1) = fingerprint_and_index(&cid, cf.num_buckets);
            let i2 = alt_index(i1, fp, cf.num_buckets);
            let i1_back = alt_index(i2, fp, cf.num_buckets);
            assert_eq!(
                i1, i1_back,
                "alt_index is not an involution: i1={i1}, i2={i2}, alt(i2)={i1_back}"
            );
        }
    }

    #[test]
    fn num_buckets_is_power_of_two() {
        for cap in [1, 10, 100, 1000, 10_000, 100_000] {
            let cf = CuckooFilter::new(cap);
            assert!(
                cf.num_buckets.is_power_of_two(),
                "num_buckets {} is not power of 2 for capacity {cap}",
                cf.num_buckets
            );
        }
    }

    #[test]
    fn high_load_no_false_negatives() {
        // Exercise the eviction path at ~80% load to verify alt_index
        // correctness under pressure.
        let capacity = 1000;
        let target = (capacity as f64 * 0.8) as usize;
        let mut cf = CuckooFilter::new(capacity);
        let mut inserted = Vec::new();
        for i in 0..target * 2 {
            let cid = make_cid(i);
            if cf.insert(&cid).is_ok() {
                inserted.push(cid);
            }
            if inserted.len() >= target {
                break;
            }
        }
        // Every successfully inserted item must be found.
        for cid in &inserted {
            assert!(
                cf.may_contain(cid),
                "false negative at load {}/{}",
                inserted.len(),
                capacity
            );
        }
    }
}

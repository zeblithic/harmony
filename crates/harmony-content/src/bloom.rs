//! Bloom filter for content discovery.
//!
//! Nodes broadcast Bloom filters of their cached CID set so peers can
//! skip queries for content that is definitely absent. Uses
//! Kirsch-Mitzenmacher double hashing to derive *k* bit indices from
//! two base hashes extracted from the CID's truncated SHA-256.

use alloc::{vec, vec::Vec};
use core::fmt;

use crate::cid::ContentId;

/// SplitMix64 mixing constant A.
const SEED_A: u64 = 0x9E3779B97F4A7C15;

/// SplitMix64 mixing constant B.
const SEED_B: u64 = 0xBF58476D1CE4E5B9;

/// Size of the serialization header in bytes: num_bits (4) + num_hashes (4) + item_count (4).
const HEADER_SIZE: usize = 12;

/// Maximum allowed `num_hashes` (k) in a deserialized filter.
///
/// Optimal k for any practical false positive rate is under 25. Capping at 100
/// prevents CPU-intensive filters from untrusted peers (a malicious broadcast
/// with `num_hashes = u32::MAX` would cause ~4 billion iterations in `may_contain`).
const MAX_HASHES: u32 = 100;

/// Maximum allowed `num_bits` (m) in a deserialized filter.
///
/// Derived from the CLI's maximum cache capacity (200M items at fp_rate=0.001
/// yields ~2.88 billion bits). Capping at 3 billion provides slight headroom
/// and prevents untrusted peers from forcing large memory allocations (~375 MB
/// at the cap). The receiver must still transmit a matching body, but
/// defense-in-depth caps the allocation here.
const MAX_BITS: u32 = 3_000_000_000;

/// Errors that can occur when deserializing a [`BloomFilter`] from bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterError {
    /// Input is shorter than the 12-byte header.
    HeaderTruncated,
    /// Body length does not match the expected number of u64 words.
    LengthMismatch {
        /// Expected body length in bytes.
        expected: usize,
        /// Actual body length in bytes.
        got: usize,
    },
    /// `num_bits` in the header is zero.
    ZeroBits,
    /// `num_hashes` in the header is zero.
    ZeroHashes,
    /// `num_hashes` exceeds the maximum allowed value.
    TooManyHashes {
        /// The value found in the header.
        got: u32,
        /// The maximum allowed value.
        max: u32,
    },
    /// `num_bits` exceeds the maximum allowed value.
    TooManyBits {
        /// The value found in the header.
        got: u32,
        /// The maximum allowed value.
        max: u32,
    },
}

impl fmt::Display for FilterError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FilterError::HeaderTruncated => write!(f, "input shorter than 12-byte header"),
            FilterError::LengthMismatch { expected, got } => {
                write!(f, "body length mismatch: expected {expected}, got {got}")
            }
            FilterError::ZeroBits => write!(f, "num_bits must be non-zero"),
            FilterError::ZeroHashes => write!(f, "num_hashes must be non-zero"),
            FilterError::TooManyHashes { got, max } => {
                write!(f, "num_hashes {got} exceeds maximum {max}")
            }
            FilterError::TooManyBits { got, max } => {
                write!(f, "num_bits {got} exceeds maximum {max}")
            }
        }
    }
}

/// A Bloom filter for approximate set membership testing of [`ContentId`] values.
///
/// Provides probabilistic membership queries: [`may_contain`](BloomFilter::may_contain) may
/// return false positives but never false negatives (assuming no hash collisions).
/// The false positive rate is controlled by the `fp_rate` parameter at construction.
pub struct BloomFilter {
    /// Bit array stored as packed u64 words.
    bits: Vec<u64>,
    /// Number of independent hash functions (k).
    num_hashes: u32,
    /// Total number of bits in the filter (m).
    num_bits: u32,
    /// Number of items inserted (for serialization).
    item_count: u32,
}

impl fmt::Debug for BloomFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let set_bits = self.count_set_bits();
        f.debug_struct("BloomFilter")
            .field("num_bits", &self.num_bits)
            .field("num_hashes", &self.num_hashes)
            .field("item_count", &self.item_count)
            .field("set_bits", &set_bits)
            .finish_non_exhaustive()
    }
}

impl BloomFilter {
    /// Create a new Bloom filter sized for the expected number of items and
    /// desired false positive rate.
    ///
    /// Uses the classic sizing formulas:
    /// - `m = -n * ln(p) / (ln 2)^2`
    /// - `k = (m / n) * ln 2`
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is zero or `fp_rate` is not in `(0.0, 1.0)`.
    pub fn new(expected_items: u32, fp_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be > 0");
        assert!(
            fp_rate > 0.0 && fp_rate < 1.0,
            "fp_rate must be in (0.0, 1.0)"
        );

        let n = expected_items as f64;
        let ln2 = core::f64::consts::LN_2;
        let ln2_sq = ln2 * ln2;

        // m = -n * ln(p) / (ln2)^2
        let m = -n * libm::log(fp_rate) / ln2_sq;
        assert!(
            m <= u32::MAX as f64,
            "Bloom filter requires {} bits which exceeds u32::MAX; reduce expected_items or increase fp_rate",
            m as u64,
        );
        let num_bits = m.ceil() as u32;

        // k = (m / n) * ln2
        let k = (m / n) * ln2;
        let num_hashes = k.round().max(1.0) as u32;

        // Number of u64 words needed.
        let num_words = (num_bits as usize).div_ceil(64);

        BloomFilter {
            bits: vec![0u64; num_words],
            num_hashes,
            num_bits,
            item_count: 0,
        }
    }

    /// Returns the total number of bits in the filter (m).
    pub fn num_bits(&self) -> u32 {
        self.num_bits
    }

    /// Returns the number of hash functions (k).
    pub fn num_hashes(&self) -> u32 {
        self.num_hashes
    }

    /// Returns the exact number of items inserted (tracked on each `insert` call).
    pub fn item_count(&self) -> u32 {
        self.item_count
    }

    /// Insert a content ID into the filter.
    pub fn insert(&mut self, cid: &ContentId) {
        let (h1, h2) = hash_pair(cid);
        for i in 0..self.num_hashes {
            let idx = bit_index(h1, h2, i, self.num_bits);
            self.set_bit(idx);
        }
        self.item_count = self.item_count.saturating_add(1);
    }

    /// Test whether a content ID *may* be in the filter.
    ///
    /// Returns `true` if the item might be present (with some false positive
    /// probability), or `false` if the item is definitely absent.
    pub fn may_contain(&self, cid: &ContentId) -> bool {
        let (h1, h2) = hash_pair(cid);
        for i in 0..self.num_hashes {
            let idx = bit_index(h1, h2, i, self.num_bits);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }

    /// Clear all bits, resetting the filter to empty.
    pub fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.item_count = 0;
    }

    /// Estimate the number of items inserted based on the number of set bits.
    ///
    /// Uses the formula: `n* = -(m/k) * ln(1 - X/m)` where X is the number
    /// of set bits.
    pub fn estimated_count(&self) -> u32 {
        let x = self.count_set_bits() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        if x == 0.0 {
            return 0;
        }

        let ratio = x / m;
        // Clamp to avoid ln(0) when all bits are set.
        if ratio >= 1.0 {
            return u32::MAX;
        }

        let estimate = -(m / k) * libm::log(1.0 - ratio);
        if estimate > u32::MAX as f64 {
            return u32::MAX;
        }
        estimate.round() as u32
    }

    /// Serialize the filter to bytes.
    ///
    /// Wire format: `[num_bits: u32 BE][num_hashes: u32 BE][item_count: u32 BE][bit array as u64 LE words]`
    pub fn to_bytes(&self) -> Vec<u8> {
        let body_len = self.bits.len() * 8;
        let mut buf = Vec::with_capacity(HEADER_SIZE + body_len);

        buf.extend_from_slice(&self.num_bits.to_be_bytes());
        buf.extend_from_slice(&self.num_hashes.to_be_bytes());
        buf.extend_from_slice(&self.item_count.to_be_bytes());

        for &word in &self.bits {
            buf.extend_from_slice(&word.to_le_bytes());
        }

        buf
    }

    /// Deserialize a filter from bytes.
    ///
    /// Returns a [`FilterError`] if the input is malformed.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FilterError> {
        if bytes.len() < HEADER_SIZE {
            return Err(FilterError::HeaderTruncated);
        }

        let num_bits = u32::from_be_bytes(bytes[0..4].try_into().unwrap());
        let num_hashes = u32::from_be_bytes(bytes[4..8].try_into().unwrap());
        let item_count = u32::from_be_bytes(bytes[8..12].try_into().unwrap());

        if num_bits == 0 {
            return Err(FilterError::ZeroBits);
        }
        if num_bits > MAX_BITS {
            return Err(FilterError::TooManyBits {
                got: num_bits,
                max: MAX_BITS,
            });
        }
        if num_hashes == 0 {
            return Err(FilterError::ZeroHashes);
        }
        if num_hashes > MAX_HASHES {
            return Err(FilterError::TooManyHashes {
                got: num_hashes,
                max: MAX_HASHES,
            });
        }

        let num_words = (num_bits as usize).div_ceil(64);
        let expected_body = num_words * 8;
        let got_body = bytes.len() - HEADER_SIZE;

        if got_body != expected_body {
            return Err(FilterError::LengthMismatch {
                expected: expected_body,
                got: got_body,
            });
        }

        let mut bits = vec![0u64; num_words];
        for (i, word) in bits.iter_mut().enumerate() {
            let offset = HEADER_SIZE + i * 8;
            *word = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
        }

        Ok(BloomFilter {
            bits,
            num_hashes,
            num_bits,
            item_count,
        })
    }

    /// Set a bit at the given index.
    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    /// Get a bit at the given index.
    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Count the number of set bits across all words.
    fn count_set_bits(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }
}

/// Extract two base hashes from a CID for Kirsch-Mitzenmacher double hashing.
///
/// Reads `hash[0..8]` and `hash[8..16]` as little-endian u64 values and mixes
/// them with SplitMix64 constants.
fn hash_pair(cid: &ContentId) -> (u64, u64) {
    let a = u64::from_le_bytes(cid.hash[0..8].try_into().unwrap());
    let b = u64::from_le_bytes(cid.hash[8..16].try_into().unwrap());
    let h1 = a.wrapping_mul(SEED_A) ^ b;
    let h2 = b.wrapping_mul(SEED_B) ^ a;
    (h1, h2)
}

/// Compute the bit index for the i-th hash function using double hashing.
///
/// `index = (h1 + i * h2) % num_bits`
fn bit_index(h1: u64, h2: u64, i: u32, num_bits: u32) -> usize {
    let i = i as u64;
    let combined = h1.wrapping_add(i.wrapping_mul(h2));
    (combined % num_bits as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(
            format!("bloom-test-{i}").as_bytes(),
            crate::cid::ContentFlags::default(),
        )
        .unwrap()
    }

    #[test]
    fn new_sizes_correctly_for_100k_items() {
        let bf = BloomFilter::new(100_000, 0.001);
        let m = bf.num_bits();
        assert!(
            m >= 1_400_000 && m <= 1_500_000,
            "expected m in 1.4M-1.5M, got {m}"
        );
        assert_eq!(bf.num_hashes(), 10, "expected k=10 for 0.1% FP rate");
    }

    #[test]
    fn new_sizes_correctly_for_1k_items() {
        let bf = BloomFilter::new(1_000, 0.001);
        let m = bf.num_bits();
        assert!(
            m >= 14_000 && m <= 15_000,
            "expected m in 14K-15K, got {m}"
        );
        assert_eq!(bf.num_hashes(), 10, "expected k=10 for 0.1% FP rate");
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_items() {
        BloomFilter::new(0, 0.01);
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_fp_rate() {
        BloomFilter::new(1000, 0.0);
    }

    #[test]
    fn debug_impl_shows_summary() {
        let bf = BloomFilter::new(1000, 0.01);
        let dbg = format!("{bf:?}");
        assert!(dbg.contains("num_bits"));
        assert!(dbg.contains("num_hashes"));
        assert!(dbg.contains("item_count"));
        assert!(dbg.contains("set_bits"));
        assert!(dbg.contains(".."), "should use finish_non_exhaustive()");
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
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..1000 {
            bf.insert(&make_cid(i));
        }
        for i in 0..1000 {
            assert!(
                bf.may_contain(&make_cid(i)),
                "false negative for item {i}"
            );
        }
    }

    #[test]
    fn false_positive_rate_within_bounds() {
        let mut bf = BloomFilter::new(1000, 0.01);
        // Insert 1000 items.
        for i in 0..1000 {
            bf.insert(&make_cid(i));
        }
        // Check 10000 non-members.
        let mut false_positives = 0;
        for i in 10_000..20_000 {
            if bf.may_contain(&make_cid(i)) {
                false_positives += 1;
            }
        }
        let fp_rate = false_positives as f64 / 10_000.0;
        assert!(
            fp_rate <= 0.03,
            "false positive rate {fp_rate:.4} exceeds 3% bound"
        );
    }

    #[test]
    fn serialization_round_trip() {
        let mut bf = BloomFilter::new(1000, 0.01);
        for i in 0..100 {
            bf.insert(&make_cid(i));
        }
        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();

        assert_eq!(bf.num_bits(), bf2.num_bits());
        assert_eq!(bf.num_hashes(), bf2.num_hashes());
        assert_eq!(bf.item_count, bf2.item_count);
        assert_eq!(bf.bits, bf2.bits);

        // Verify membership is preserved.
        for i in 0..100 {
            assert!(bf2.may_contain(&make_cid(i)));
        }
    }

    #[test]
    fn from_bytes_rejects_truncated() {
        let err = BloomFilter::from_bytes(&[0u8; 8]).unwrap_err();
        assert_eq!(err, FilterError::HeaderTruncated);
    }

    #[test]
    fn from_bytes_rejects_wrong_body_length() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.insert(&make_cid(0));
        let mut bytes = bf.to_bytes();
        // Truncate body by 8 bytes.
        bytes.truncate(bytes.len() - 8);
        let err = BloomFilter::from_bytes(&bytes).unwrap_err();
        match err {
            FilterError::LengthMismatch { .. } => {}
            other => panic!("expected LengthMismatch, got {other:?}"),
        }
    }

    #[test]
    fn estimated_count_accuracy() {
        let mut bf = BloomFilter::new(2000, 0.01);
        let actual = 1000u32;
        for i in 0..actual {
            bf.insert(&make_cid(i as usize));
        }
        let est = bf.estimated_count();
        let diff = (est as i64 - actual as i64).unsigned_abs();
        assert!(
            diff <= (actual as u64) / 10,
            "estimated {est} too far from actual {actual} (diff {diff})"
        );
    }

    #[test]
    fn estimated_count_zero_for_empty() {
        let bf = BloomFilter::new(1000, 0.01);
        assert_eq!(bf.estimated_count(), 0);
    }

    #[test]
    fn from_bytes_rejects_excessive_num_hashes() {
        // Craft a header with num_hashes > MAX_HASHES.
        let mut header = Vec::new();
        header.extend_from_slice(&64u32.to_be_bytes()); // num_bits = 64
        header.extend_from_slice(&101u32.to_be_bytes()); // num_hashes = 101 > MAX_HASHES
        header.extend_from_slice(&0u32.to_be_bytes()); // item_count = 0
        header.extend_from_slice(&0u64.to_le_bytes()); // 1 word for 64 bits
        let err = BloomFilter::from_bytes(&header).unwrap_err();
        assert_eq!(
            err,
            FilterError::TooManyHashes {
                got: 101,
                max: super::MAX_HASHES,
            }
        );
    }

    #[test]
    fn from_bytes_rejects_excessive_num_bits() {
        // num_bits = MAX_BITS + 1 should be rejected at the header check.
        // No body is needed — from_bytes returns TooManyBits before examining it.
        let num_bits = super::MAX_BITS + 1;
        let mut buf = Vec::new();
        buf.extend_from_slice(&num_bits.to_be_bytes()); // num_bits
        buf.extend_from_slice(&10u32.to_be_bytes()); // num_hashes
        buf.extend_from_slice(&0u32.to_be_bytes()); // item_count
        let err = BloomFilter::from_bytes(&buf).unwrap_err();
        assert_eq!(
            err,
            FilterError::TooManyBits {
                got: num_bits,
                max: super::MAX_BITS,
            }
        );
    }

    #[test]
    #[should_panic(expected = "exceeds u32::MAX")]
    fn new_panics_on_bit_count_overflow() {
        // 500_000_000 items at 0.001 FP rate needs ~7.2 billion bits > u32::MAX.
        BloomFilter::new(500_000_000, 0.001);
    }
}

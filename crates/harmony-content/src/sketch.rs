//! Count-Min Sketch with 4-bit pair-packed counters.
//!
//! Used for approximate frequency counting in the W-TinyLFU admission policy.
//! Each counter occupies 4 bits (max value 15), and counters are packed two per
//! byte: the high nibble holds even-indexed counters and the low nibble holds
//! odd-indexed counters.

use alloc::{vec, vec::Vec};
use core::fmt;

use crate::cid::ContentId;

/// Number of independent hash rows in the sketch.
const NUM_ROWS: usize = 4;

/// Mixing seeds for deriving row indices from CID bytes.
const SEEDS: [u64; NUM_ROWS] = [
    0x9E3779B97F4A7C15,
    0xBF58476D1CE4E5B9,
    0x94D049BB133111EB,
    0x517CC1B727220A95,
];

/// A Count-Min Sketch with 4-bit pair-packed counters.
///
/// Provides approximate frequency estimation: `estimate` may overcount but
/// never undercounts (before halving). Counters saturate at 15 and are
/// periodically halved to implement aging.
pub struct CountMinSketch {
    /// 4-bit counters packed two per byte.
    /// High nibble = even-indexed counter, low nibble = odd-indexed counter.
    /// Layout: `NUM_ROWS` rows of `width / 2` bytes each.
    counters: Vec<u8>,
    /// Number of logical counters per row.
    width: usize,
    /// Total number of increment operations since last halving.
    total_increments: u64,
    /// When `total_increments` reaches this threshold, all counters are halved.
    halving_threshold: u64,
}

impl fmt::Debug for CountMinSketch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CountMinSketch")
            .field("width", &self.width)
            .field("total_increments", &self.total_increments)
            .field("halving_threshold", &self.halving_threshold)
            .finish_non_exhaustive()
    }
}

impl CountMinSketch {
    /// Create a new sketch with the given width (rounded up to even) and
    /// halving threshold.
    ///
    /// Allocates `NUM_ROWS * (width / 2)` bytes for counters.
    pub fn new(width: usize, halving_threshold: u64) -> Self {
        assert!(width >= 2, "CountMinSketch width must be at least 2");
        // Round width up to the nearest even number.
        let width = (width + 1) & !1;
        let bytes_per_row = width / 2;
        let total_bytes = NUM_ROWS * bytes_per_row;

        CountMinSketch {
            counters: vec![0u8; total_bytes],
            width,
            total_increments: 0,
            halving_threshold,
        }
    }

    /// Increment the counters for the given content ID.
    ///
    /// Each of the 4 row counters is incremented by 1, saturating at 15.
    /// If total increments reaches the halving threshold, all counters are
    /// halved.
    pub fn increment(&mut self, cid: &ContentId) {
        let indices = self.hash_indices(cid);
        let bytes_per_row = self.width / 2;

        for (row, &col) in indices.iter().enumerate() {
            let byte_offset = row * bytes_per_row + col / 2;
            let byte = &mut self.counters[byte_offset];

            if col % 2 == 0 {
                // Even index: high nibble
                let val = *byte >> 4;
                if val < 15 {
                    *byte = ((val + 1) << 4) | (*byte & 0x0F);
                }
            } else {
                // Odd index: low nibble
                let val = *byte & 0x0F;
                if val < 15 {
                    *byte = (*byte & 0xF0) | (val + 1);
                }
            }
        }

        self.total_increments += 1;
        if self.total_increments >= self.halving_threshold {
            self.halve();
        }
    }

    /// Estimate the frequency of the given content ID.
    ///
    /// Returns the minimum counter value across all 4 rows, which is the
    /// Count-Min Sketch's conservative estimate.
    pub fn estimate(&self, cid: &ContentId) -> u8 {
        let indices = self.hash_indices(cid);
        let bytes_per_row = self.width / 2;
        let mut min = u8::MAX;

        for (row, &col) in indices.iter().enumerate() {
            let byte_offset = row * bytes_per_row + col / 2;
            let byte = self.counters[byte_offset];

            let val = if col % 2 == 0 {
                byte >> 4 // Even index: high nibble
            } else {
                byte & 0x0F // Odd index: low nibble
            };

            if val < min {
                min = val;
            }
        }

        min
    }

    /// Force a halve (public, for testing).
    pub fn halve_now(&mut self) {
        self.halve();
    }

    /// Halve all counters by right-shifting each 4-bit value by 1.
    ///
    /// This implements frequency aging: recent items retain higher counts
    /// while old items decay. Resets `total_increments` to zero.
    fn halve(&mut self) {
        for byte in &mut self.counters {
            let high = (*byte >> 4) >> 1; // halve the high nibble
            let low = (*byte & 0x0F) >> 1; // halve the low nibble
            *byte = (high << 4) | low;
        }
        self.total_increments = 0;
    }

    /// Compute the column index for each row from the CID's hash bytes.
    ///
    /// Extracts two `u64` values from the first 16 bytes of the CID hash,
    /// then mixes each with the row seed via wrapping multiplication and XOR
    /// to produce a column index.
    fn hash_indices(&self, cid: &ContentId) -> [usize; NUM_ROWS] {
        let hash = &cid.hash;

        // Extract two u64 from bytes [0..8] and [8..16].
        let h0 = u64::from_le_bytes(hash[0..8].try_into().unwrap());
        let h1 = u64::from_le_bytes(hash[8..16].try_into().unwrap());

        let mut indices = [0usize; NUM_ROWS];
        for (i, seed) in SEEDS.iter().enumerate() {
            let mixed = h0.wrapping_mul(*seed) ^ h1.wrapping_mul(seed.rotate_left(32));
            indices[i] = (mixed as usize) % self.width;
        }

        indices
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cid(i: usize) -> ContentId {
        ContentId::for_blob(
            format!("sketch-test-{i}").as_bytes(),
            crate::cid::ContentFlags::default(),
        )
        .unwrap()
    }

    #[test]
    fn debug_impl_shows_summary() {
        let sketch = CountMinSketch::new(256, 10_000);
        let dbg = format!("{sketch:?}");
        assert!(dbg.contains("width: 256"));
        assert!(dbg.contains("total_increments: 0"));
        assert!(dbg.contains("halving_threshold: 10000"));
        assert!(dbg.contains(".."), "should use finish_non_exhaustive()");
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
        for _ in 0..7 {
            sketch.increment(&target);
        }
        for i in 100..400 {
            sketch.increment(&make_cid(i));
        }
        assert!(sketch.estimate(&target) >= 7);
    }

    #[test]
    fn halving_decays_counters() {
        let mut sketch = CountMinSketch::new(1024, 100_000);
        let cid = make_cid(0);
        for _ in 0..10 {
            sketch.increment(&cid);
        }
        let before = sketch.estimate(&cid);
        assert!(before >= 10);
        sketch.halve_now();
        let after = sketch.estimate(&cid);
        assert!(after >= 4 && after <= before / 2 + 1);
    }

    #[test]
    fn auto_halving_at_threshold() {
        let mut sketch = CountMinSketch::new(64, 20);
        let cid = make_cid(0);
        for _ in 0..10 {
            sketch.increment(&cid);
        }
        let before = sketch.estimate(&cid);
        for i in 100..115 {
            sketch.increment(&make_cid(i));
        }
        let after = sketch.estimate(&cid);
        assert!(
            after < before,
            "expected decay: before={before}, after={after}"
        );
    }

    #[test]
    fn hash_rows_are_independent() {
        // Verify that different CIDs produce distinct index sets across rows,
        // confirming that h1 is mixed per-row (not a constant offset).
        let sketch = CountMinSketch::new(1024, 10_000);
        let indices_a = sketch.hash_indices(&make_cid(0));
        let indices_b = sketch.hash_indices(&make_cid(1));

        // With 4 rows and width 1024, two different CIDs should differ in
        // at least one row index. (Probability of full collision ~= 1/1024^4.)
        assert_ne!(
            indices_a, indices_b,
            "two different CIDs should not produce identical index sets"
        );
    }
}

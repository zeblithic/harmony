//! Lloyd-Max optimal quantizer for Gaussian-distributed data (3-bit, 8-level).
//!
//! The codebook is pre-computed for the standard normal distribution N(0,1),
//! then scaled by the target standard deviation at construction time. After
//! random orthogonal rotation, KV cache vector coordinates are approximately
//! i.i.d. Gaussian, making this codebook near-optimal.

/// 8-level Lloyd-Max codebook for Gaussian data.
///
/// Boundaries partition the real line into 8 intervals.
/// Each interval maps to a centroid (reconstruction value).
/// The codebook is symmetric around zero.
pub struct LloydMaxCodebook {
    /// 7 decision boundaries (sorted ascending).
    pub boundaries: [f32; 7],
    /// 8 reconstruction centroids (sorted ascending).
    pub centroids: [f32; 8],
}

/// Standard N(0,1) Lloyd-Max 8-level boundaries.
const GAUSSIAN_BOUNDARIES: [f64; 7] = [
    -1.7479, -1.0500, -0.5006, 0.0, 0.5006, 1.0500, 1.7479,
];

/// Standard N(0,1) Lloyd-Max 8-level centroids (conditional means).
const GAUSSIAN_CENTROIDS: [f64; 8] = [
    -2.1520, -1.3440, -0.7560, -0.2451, 0.2451, 0.7560, 1.3440, 2.1520,
];

/// Normalized MSE of 3-bit Lloyd-Max on N(0,1): MSE / σ².
/// Used to estimate expected quantization error for QJL correction.
pub const NMSE_3BIT_GAUSSIAN: f32 = 0.03454;

impl LloydMaxCodebook {
    /// Create a codebook for Gaussian data with standard deviation `sigma`.
    ///
    /// Scales the standard N(0,1) codebook by `sigma`. For unit vector
    /// coordinates of dimension d, use `sigma = 1.0 / sqrt(d)`.
    pub fn gaussian(sigma: f32) -> Self {
        let s = sigma as f64;
        let mut boundaries = [0.0f32; 7];
        let mut centroids = [0.0f32; 8];
        for i in 0..7 {
            boundaries[i] = (GAUSSIAN_BOUNDARIES[i] * s) as f32;
        }
        for i in 0..8 {
            centroids[i] = (GAUSSIAN_CENTROIDS[i] * s) as f32;
        }
        Self {
            boundaries,
            centroids,
        }
    }

    /// Quantize a value to its 3-bit index (0–7).
    ///
    /// Performs linear scan over 7 boundaries. For 7 comparisons this is
    /// faster than binary search due to branch prediction.
    pub fn quantize(&self, value: f32) -> u8 {
        for (i, &b) in self.boundaries.iter().enumerate() {
            if value < b {
                return i as u8;
            }
        }
        7
    }

    /// Dequantize a 3-bit index back to its centroid value.
    pub fn dequantize(&self, index: u8) -> f32 {
        self.centroids[index.min(7) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_codebook_centroids_are_ordered() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        for i in 1..8 {
            assert!(
                cb.centroids[i] > cb.centroids[i - 1],
                "centroids not ascending: [{}]={} >= [{}]={}",
                i - 1, cb.centroids[i - 1], i, cb.centroids[i]
            );
        }
    }

    #[test]
    fn gaussian_codebook_boundaries_are_ordered() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        for i in 1..7 {
            assert!(
                cb.boundaries[i] > cb.boundaries[i - 1],
                "boundaries not ascending: [{}]={} >= [{}]={}",
                i - 1, cb.boundaries[i - 1], i, cb.boundaries[i]
            );
        }
    }

    #[test]
    fn gaussian_codebook_is_symmetric() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        for i in 0..4 {
            assert!(
                (cb.centroids[i] + cb.centroids[7 - i]).abs() < 1e-4,
                "centroids not symmetric: [{}]={} + [{}]={}",
                i, cb.centroids[i], 7 - i, cb.centroids[7 - i]
            );
        }
        for i in 0..3 {
            assert!(
                (cb.boundaries[i] + cb.boundaries[6 - i]).abs() < 1e-4,
                "boundaries not symmetric"
            );
        }
        assert!(cb.boundaries[3].abs() < 1e-6, "middle boundary should be 0");
    }

    #[test]
    fn quantize_returns_valid_indices() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        for x in [-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0] {
            let idx = cb.quantize(x);
            assert!(idx <= 7, "index {idx} out of range for x={x}");
        }
    }

    #[test]
    fn quantize_extreme_values() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        assert_eq!(cb.quantize(-100.0), 0);
        assert_eq!(cb.quantize(100.0), 7);
    }

    #[test]
    fn quantize_dequantize_bounded_error() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let idx = cb.quantize(x);
            let reconstructed = cb.dequantize(idx);
            let err = (x - reconstructed).abs();
            assert!(
                err < 1.5,
                "error {err} too large for x={x} -> idx={idx} -> {reconstructed}"
            );
        }
    }

    #[test]
    fn scaling_shifts_codebook() {
        let cb1 = LloydMaxCodebook::gaussian(1.0);
        let cb2 = LloydMaxCodebook::gaussian(0.5);
        for i in 0..8 {
            assert!(
                (cb2.centroids[i] - cb1.centroids[i] * 0.5).abs() < 1e-6,
                "scaling failed at centroid {i}"
            );
        }
    }

    #[test]
    fn dequantize_clamps_index() {
        let cb = LloydMaxCodebook::gaussian(1.0);
        assert_eq!(cb.dequantize(255), cb.dequantize(7));
    }
}

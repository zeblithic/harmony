//! Quantized Johnson-Lindenstrauss (QJL) error correction.
//!
//! Projects quantization residuals through a random ±1 (Rademacher) matrix
//! and stores only the sign bits. During decompression, the sign bits are
//! decoded into a correction vector that reduces reconstruction error.
//! The correction provides an unbiased inner product estimator.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use super::packing::{pack_1bit, unpack_1bit};

/// Random ±1 (Rademacher) matrix for JL projection, stored as packed bits.
///
/// Row-major layout: entry (row, col) at bit index `row * dim + col`.
/// Bit = 1 means +1, bit = 0 means -1.
pub struct JlMatrix {
    packed: Vec<u8>,
    dim: usize,
}

impl JlMatrix {
    /// Generate a deterministic Rademacher matrix from a seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let total_bits = dim * dim;
        let values: Vec<bool> = (0..total_bits).map(|_| rng.gen_bool(0.5)).collect();
        Self {
            packed: pack_1bit(&values),
            dim,
        }
    }

    /// Get entry (row, col) as +1.0 or -1.0.
    fn get(&self, row: usize, col: usize) -> f32 {
        let idx = row * self.dim + col;
        if (self.packed[idx / 8] >> (idx % 8)) & 1 == 1 {
            1.0
        } else {
            -1.0
        }
    }

    /// Encode: project residual through JL matrix and return sign bits.
    ///
    /// Computes `projected = M @ residual`, returns `pack_1bit(sign(projected))`.
    pub fn encode_signs(&self, residual: &[f32]) -> Vec<u8> {
        let dim = self.dim;
        let signs: Vec<bool> = (0..dim)
            .map(|row| {
                let mut dot = 0.0f32;
                for col in 0..dim {
                    dot += self.get(row, col) * residual[col];
                }
                dot > 0.0
            })
            .collect();
        pack_1bit(&signs)
    }

    /// Decode: reconstruct correction vector from sign bits.
    ///
    /// Computes `correction = (scale / (dim * sqrt(dim))) * M^T @ sign_vector`
    /// where sign bits are decoded as +1 (true) or -1 (false).
    ///
    /// The extra `1/sqrt(dim)` factor normalises the Rademacher rows (each row
    /// has L2 norm `sqrt(dim)`), making the estimator consistent with the
    /// standard 1-bit CS result for unit-row-norm sensing matrices.
    ///
    /// The caller should pass `scale = sqrt(π/2) * expected_residual_norm`.
    pub fn decode_correction(&self, packed_signs: &[u8], scale: f32) -> Vec<f32> {
        let dim = self.dim;
        let signs = unpack_1bit(packed_signs, dim);
        // Divide by dim * sqrt(dim) to account for unnormalised Rademacher rows.
        let scale_factor = scale / (dim as f32 * (dim as f32).sqrt());
        let mut correction = vec![0.0f32; dim];
        // M^T @ sign_vector: for each output col, dot column of M with sign vector
        for col in 0..dim {
            let mut dot = 0.0f32;
            for row in 0..dim {
                let jl_val = self.get(row, col);
                let sign_val = if signs[row] { 1.0 } else { -1.0 };
                dot += jl_val * sign_val;
            }
            correction[col] = scale_factor * dot;
        }
        correction
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jl_matrix_has_correct_size() {
        let jl = JlMatrix::new(80, 42);
        assert_eq!(jl.packed.len(), 800); // 80 * 80 / 8
        assert_eq!(jl.dim, 80);
    }

    #[test]
    fn jl_matrix_deterministic_from_seed() {
        let jl1 = JlMatrix::new(16, 42);
        let jl2 = JlMatrix::new(16, 42);
        assert_eq!(jl1.packed, jl2.packed);
    }

    #[test]
    fn jl_matrix_different_seeds_differ() {
        let jl1 = JlMatrix::new(16, 42);
        let jl2 = JlMatrix::new(16, 99);
        assert_ne!(jl1.packed, jl2.packed);
    }

    #[test]
    fn jl_matrix_entries_are_plus_minus_one() {
        let jl = JlMatrix::new(8, 42);
        for row in 0..8 {
            for col in 0..8 {
                let v = jl.get(row, col);
                assert!(
                    v == 1.0 || v == -1.0,
                    "entry [{row},{col}] = {v}, expected ±1"
                );
            }
        }
    }

    #[test]
    fn encode_signs_correct_length() {
        let jl = JlMatrix::new(80, 42);
        let residual = vec![0.1f32; 80];
        let signs = jl.encode_signs(&residual);
        assert_eq!(signs.len(), 10); // 80 / 8
    }

    #[test]
    fn encode_signs_of_zero_residual() {
        let jl = JlMatrix::new(8, 42);
        let residual = vec![0.0f32; 8];
        let signs = jl.encode_signs(&residual);
        let unpacked = unpack_1bit(&signs, 8);
        assert!(unpacked.iter().all(|&s| !s), "sign(0) should be false");
    }

    #[test]
    fn decode_correction_has_correct_length() {
        let jl = JlMatrix::new(80, 42);
        let signs = vec![0u8; 10];
        let correction = jl.decode_correction(&signs, 1.0);
        assert_eq!(correction.len(), 80);
    }

    #[test]
    fn decode_correction_with_zero_scale_is_zero() {
        let jl = JlMatrix::new(8, 42);
        let signs = vec![0xFFu8];
        let correction = jl.decode_correction(&signs, 0.0);
        assert!(
            correction.iter().all(|&v| v.abs() < 1e-10),
            "zero scale should produce zero correction"
        );
    }

    #[test]
    fn correction_reduces_error() {
        let dim = 32;
        let jl = JlMatrix::new(dim, 42);

        let residual: Vec<f32> = (0..dim).map(|i| (i as f32 - 16.0) * 0.01).collect();
        let residual_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        let signs = jl.encode_signs(&residual);

        let scale = core::f32::consts::FRAC_PI_2.sqrt() * residual_norm;
        let correction = jl.decode_correction(&signs, scale);

        let err_without: f32 = residual.iter().map(|x| x * x).sum();

        let err_with: f32 = residual
            .iter()
            .zip(correction.iter())
            .map(|(r, c)| (r - c) * (r - c))
            .sum();

        assert!(
            err_with < err_without,
            "correction should reduce error: {err_with} >= {err_without}"
        );
    }
}

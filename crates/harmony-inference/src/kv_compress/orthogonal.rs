//! Random orthogonal matrix generation via modified Gram-Schmidt.
//!
//! Generates a deterministic orthogonal matrix from a seed. Used for
//! preconditioning KV cache vectors before quantization — smooths
//! channel-wise outliers so all dimensions have similar distributions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate a random orthogonal matrix of size `dim × dim`.
///
/// Uses a seeded RNG for deterministic output, then orthogonalizes
/// via modified Gram-Schmidt. The matrix is generated in f64 for
/// numerical precision, returned as f32 for use.
///
/// Returns the matrix in row-major order: `result[row * dim + col]`.
pub fn generate_orthogonal_matrix(dim: usize, seed: u64) -> Vec<f32> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Generate random matrix in f64 for numerical precision
    let mut m: Vec<f64> = (0..dim * dim)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();

    // Modified Gram-Schmidt (column-wise)
    for i in 0..dim {
        // Orthogonalize column i against all previous columns
        for j in 0..i {
            let dot = col_dot(&m, dim, j, i);
            for row in 0..dim {
                m[row * dim + i] -= dot * m[row * dim + j];
            }
        }

        // Normalize column i
        let norm = col_norm(&m, dim, i);
        if norm < 1e-12 {
            // Degenerate — regenerate column and re-orthogonalize
            for row in 0..dim {
                m[row * dim + i] = rng.gen_range(-1.0..1.0);
            }
            for j in 0..i {
                let dot = col_dot(&m, dim, j, i);
                for row in 0..dim {
                    m[row * dim + i] -= dot * m[row * dim + j];
                }
            }
            let norm = col_norm(&m, dim, i);
            for row in 0..dim {
                m[row * dim + i] /= norm;
            }
        } else {
            for row in 0..dim {
                m[row * dim + i] /= norm;
            }
        }
    }

    m.iter().map(|&v| v as f32).collect()
}

/// Transpose a row-major square matrix.
pub fn transpose(matrix: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for j in 0..dim {
            result[j * dim + i] = matrix[i * dim + j];
        }
    }
    result
}

/// Multiply row-major square matrix by vector: result = M @ v.
pub fn matvec(matrix: &[f32], vector: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    for i in 0..dim {
        let mut sum = 0.0f32;
        let row_start = i * dim;
        for j in 0..dim {
            sum += matrix[row_start + j] * vector[j];
        }
        result[i] = sum;
    }
    result
}

/// Multiply two row-major square matrices: result = A @ B.
pub fn matmul(a: &[f32], b: &[f32], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim * dim];
    for i in 0..dim {
        for k in 0..dim {
            let a_ik = a[i * dim + k];
            for j in 0..dim {
                result[i * dim + j] += a_ik * b[k * dim + j];
            }
        }
    }
    result
}

fn col_norm(matrix: &[f64], dim: usize, col: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..dim {
        let v = matrix[row * dim + col];
        sum += v * v;
    }
    sum.sqrt()
}

fn col_dot(matrix: &[f64], dim: usize, col_a: usize, col_b: usize) -> f64 {
    let mut sum = 0.0;
    for row in 0..dim {
        sum += matrix[row * dim + col_a] * matrix[row * dim + col_b];
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_matrix_has_correct_size() {
        let q = generate_orthogonal_matrix(8, 42);
        assert_eq!(q.len(), 64);
    }

    #[test]
    fn orthogonal_matrix_is_orthogonal() {
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42);
        let qt = transpose(&q, dim);
        let product = matmul(&q, &qt, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = product[i * dim + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Q@Q^T[{i},{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn orthogonal_matrix_columns_are_unit_length() {
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42);
        for col in 0..dim {
            let mut norm_sq = 0.0f32;
            for row in 0..dim {
                let v = q[row * dim + col];
                norm_sq += v * v;
            }
            assert!(
                (norm_sq - 1.0).abs() < 1e-5,
                "column {col} norm² = {norm_sq}, expected 1.0"
            );
        }
    }

    #[test]
    fn orthogonal_matrix_deterministic_from_seed() {
        let q1 = generate_orthogonal_matrix(8, 42);
        let q2 = generate_orthogonal_matrix(8, 42);
        assert_eq!(q1, q2);
    }

    #[test]
    fn different_seeds_produce_different_matrices() {
        let q1 = generate_orthogonal_matrix(8, 42);
        let q2 = generate_orthogonal_matrix(8, 99);
        assert_ne!(q1, q2);
    }

    #[test]
    fn transpose_is_inverse() {
        let dim = 8;
        let q = generate_orthogonal_matrix(dim, 42);
        let qt = transpose(&q, dim);
        let product = matmul(&qt, &q, dim);
        for i in 0..dim {
            for j in 0..dim {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = product[i * dim + j];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "Q^T@Q[{i},{j}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn matvec_identity_is_passthrough() {
        let dim = 4;
        let mut identity = vec![0.0f32; dim * dim];
        for i in 0..dim {
            identity[i * dim + i] = 1.0;
        }
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let result = matvec(&identity, &v, dim);
        assert_eq!(result, v);
    }

    #[test]
    fn rotation_preserves_norm() {
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let rotated = matvec(&q, &v, dim);
        let rot_norm: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-4,
            "rotation changed norm: {orig_norm} -> {rot_norm}"
        );
    }

    #[test]
    fn rotate_then_inverse_recovers_original() {
        let dim = 16;
        let q = generate_orthogonal_matrix(dim, 42);
        let qt = transpose(&q, dim);
        let v: Vec<f32> = (0..dim).map(|i| (i as f32 - 8.0) * 0.3).collect();
        let rotated = matvec(&q, &v, dim);
        let recovered = matvec(&qt, &rotated, dim);
        for i in 0..dim {
            assert!(
                (v[i] - recovered[i]).abs() < 1e-4,
                "recovery failed at [{i}]: {} vs {}",
                v[i], recovered[i]
            );
        }
    }

    #[test]
    fn works_with_dim_80() {
        let dim = 80;
        let q = generate_orthogonal_matrix(dim, 42);
        assert_eq!(q.len(), 6400);
        let mut dot = 0.0f32;
        for row in 0..dim {
            dot += q[row * dim] * q[row * dim + 1];
        }
        assert!(dot.abs() < 1e-4, "columns 0,1 dot product = {dot}");
    }
}

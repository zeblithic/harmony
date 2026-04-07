# Phase 0b: TurboQuant Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 3-bit uniform quantization in `kv_compress.rs` with a three-stage PolarQuant + QJL pipeline (random orthogonal rotation → Cartesian Lloyd-Max → 1-bit sign correction). 44 bytes/vector at head_dim=80 vs 56 bytes current.

**Architecture:** Stateful `TurboQuantState` initialized once at model load holds a random orthogonal matrix, Lloyd-Max codebook, and JL projection matrix. The existing `kv_compress.rs` file becomes a `kv_compress/` module directory with focused submodules for each math component. The external `InferenceCache` API gains a `&TurboQuantState` parameter on `compress()`/`decompress()`.

**Tech Stack:** Rust, candle-core (tensor bridge), rand (seeded RNG), serde + postcard (serialization)

**Spec:** `docs/superpowers/specs/2026-04-06-phase0b-turboquant-design.md`

**Existing code:** `crates/harmony-inference/src/kv_compress.rs` (332 lines, to be replaced)

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/harmony-inference/src/kv_compress.rs` | Delete | Replaced by module directory |
| `crates/harmony-inference/src/kv_compress/mod.rs` | Create | Public API: `TurboQuantConfig`, `TurboQuantState`, `TurboQuantVec`, `CompressedKvLayer`, compress/decompress tensor bridge |
| `crates/harmony-inference/src/kv_compress/packing.rs` | Create | 3-bit and 1-bit pack/unpack (extracted from old code + extended) |
| `crates/harmony-inference/src/kv_compress/lloyd_max.rs` | Create | Gaussian Lloyd-Max codebook, quantize/dequantize |
| `crates/harmony-inference/src/kv_compress/orthogonal.rs` | Create | Random orthogonal matrix via modified Gram-Schmidt |
| `crates/harmony-inference/src/kv_compress/qjl.rs` | Create | Rademacher JL matrix, sign encode/decode, residual correction |
| `crates/harmony-inference/src/lib.rs` | Modify | Update `compress()`/`decompress()` signatures to take `&TurboQuantState` |

---

### Task 1: Module restructure and bit packing

**Files:**
- Delete: `crates/harmony-inference/src/kv_compress.rs`
- Create: `crates/harmony-inference/src/kv_compress/mod.rs`
- Create: `crates/harmony-inference/src/kv_compress/packing.rs`

Convert `kv_compress.rs` from a single file to a module directory. Extract bit packing
into its own submodule. Add 1-bit packing (new). The old quantization code stays
temporarily in `mod.rs` so existing lib.rs tests keep compiling — it gets replaced in
Task 5.

- [ ] **Step 1: Create the directory and move the file**

```bash
mkdir -p crates/harmony-inference/src/kv_compress
mv crates/harmony-inference/src/kv_compress.rs crates/harmony-inference/src/kv_compress/mod.rs
```

- [ ] **Step 2: Verify existing tests still pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress
```

Expected: all tests pass (the module path `kv_compress` resolves identically to a directory with `mod.rs`).

- [ ] **Step 3: Create `packing.rs` with tests**

Create `crates/harmony-inference/src/kv_compress/packing.rs`:

```rust
//! Bit packing utilities for 3-bit and 1-bit quantized values.

/// Pack 3-bit values (each 0..7) into bytes, 8 values per 3 bytes.
///
/// Bit layout within each 3-byte group (little-endian):
/// ```text
/// byte 0: [v0₂ v0₁ v0₀ | v1₂ v1₁ v1₀ | v2₁ v2₀]
/// byte 1: [v2₂ | v3₂ v3₁ v3₀ | v4₂ v4₁ v4₀ | v5₀]
/// byte 2: [v5₂ v5₁ | v6₂ v6₁ v6₀ | v7₂ v7₁ v7₀]
/// ```
pub fn pack_3bit(values: &[u8]) -> Vec<u8> {
    let num_bytes = (values.len() * 3).div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        packed[byte_idx] |= (v & 0x7) << bit_idx;
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            packed[byte_idx + 1] |= (v & 0x7) >> (8 - bit_idx);
        }
    }
    packed
}

/// Unpack 3-bit values from packed bytes.
pub fn unpack_3bit(packed: &[u8], count: usize) -> Vec<u8> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let bit_offset = i * 3;
        let byte_idx = bit_offset / 8;
        let bit_idx = bit_offset % 8;
        let mut v = (packed[byte_idx] >> bit_idx) & 0x7;
        if bit_idx > 5 && byte_idx + 1 < packed.len() {
            v |= (packed[byte_idx + 1] << (8 - bit_idx)) & 0x7;
        }
        values.push(v);
    }
    values
}

/// Pack boolean values into bytes (1 bit per value, LSB first).
pub fn pack_1bit(values: &[bool]) -> Vec<u8> {
    let num_bytes = values.len().div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &v) in values.iter().enumerate() {
        if v {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

/// Unpack boolean values from packed bytes (LSB first).
pub fn unpack_1bit(packed: &[u8], count: usize) -> Vec<bool> {
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        values.push((packed[i / 8] >> (i % 8)) & 1 == 1);
    }
    values
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 3-bit packing ──

    #[test]
    fn pack_3bit_unpack_roundtrip() {
        let values: Vec<u8> = (0..16).map(|i| i % 8).collect();
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_3bit_all_sevens() {
        let values = vec![7u8; 24];
        let packed = pack_3bit(&values);
        let unpacked = unpack_3bit(&packed, 24);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_3bit_all_zeros() {
        let values = vec![0u8; 128];
        let packed = pack_3bit(&values);
        assert_eq!(packed.len(), 48); // 128 * 3 / 8 = 48
        let unpacked = unpack_3bit(&packed, 128);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_3bit_byte_count() {
        // 80 values * 3 bits = 240 bits = 30 bytes
        let values = vec![3u8; 80];
        let packed = pack_3bit(&values);
        assert_eq!(packed.len(), 30);
    }

    // ── 1-bit packing ──

    #[test]
    fn pack_1bit_unpack_roundtrip() {
        let values: Vec<bool> = (0..80).map(|i| i % 3 == 0).collect();
        let packed = pack_1bit(&values);
        let unpacked = unpack_1bit(&packed, values.len());
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_1bit_all_true() {
        let values = vec![true; 80];
        let packed = pack_1bit(&values);
        assert_eq!(packed.len(), 10); // 80 / 8 = 10
        // All bytes should be 0xFF
        assert!(packed.iter().all(|&b| b == 0xFF));
        let unpacked = unpack_1bit(&packed, 80);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_1bit_all_false() {
        let values = vec![false; 80];
        let packed = pack_1bit(&values);
        assert_eq!(packed.len(), 10);
        assert!(packed.iter().all(|&b| b == 0));
        let unpacked = unpack_1bit(&packed, 80);
        assert_eq!(values, unpacked);
    }

    #[test]
    fn pack_1bit_byte_count() {
        // 80 bits = 10 bytes
        assert_eq!(pack_1bit(&vec![false; 80]).len(), 10);
        // 81 bits = 11 bytes (ceil)
        assert_eq!(pack_1bit(&vec![false; 81]).len(), 11);
    }

    #[test]
    fn pack_1bit_non_multiple_of_8() {
        let values = vec![true, false, true, false, true]; // 5 bits
        let packed = pack_1bit(&values);
        assert_eq!(packed.len(), 1); // ceil(5/8) = 1
        let unpacked = unpack_1bit(&packed, 5);
        assert_eq!(values, unpacked);
    }
}
```

- [ ] **Step 4: Update `mod.rs` to use the packing submodule**

In `crates/harmony-inference/src/kv_compress/mod.rs`:

Add at the top (after the module doc and imports):
```rust
pub(crate) mod packing;
```

Replace the `pack_3bit` and `unpack_3bit` function definitions (lines 64-95 of the old file) with imports:
```rust
use packing::{pack_3bit, unpack_3bit};
```

Remove the 3 packing tests from `mod.rs` (`pack_unpack_roundtrip`, `pack_unpack_all_sevens`, `pack_unpack_all_zeros`) — they now live in `packing.rs`.

- [ ] **Step 5: Verify all tests pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress
```

Expected: all existing tests still pass, plus new 1-bit packing tests.

- [ ] **Step 6: Commit**

```bash
git add -A crates/harmony-inference/src/kv_compress/
git rm crates/harmony-inference/src/kv_compress.rs 2>/dev/null || true
git add crates/harmony-inference/src/kv_compress/
git commit -m "refactor(inference): convert kv_compress to module directory, extract packing"
```

---

### Task 2: Lloyd-Max codebook

**Files:**
- Create: `crates/harmony-inference/src/kv_compress/lloyd_max.rs`
- Modify: `crates/harmony-inference/src/kv_compress/mod.rs` (add `pub(crate) mod lloyd_max;`)

A 3-bit (8-level) Lloyd-Max codebook optimized for Gaussian data. The codebook
boundaries and centroids are hardcoded constants derived from N(0,1), then scaled
by the target standard deviation at construction time.

- [ ] **Step 1: Write failing tests**

Create `crates/harmony-inference/src/kv_compress/lloyd_max.rs`:

```rust
//! Lloyd-Max optimal quantizer for Gaussian-distributed data (3-bit, 8-level).
//!
//! The codebook is pre-computed for the standard normal distribution N(0,1),
//! then scaled by the target standard deviation at construction time. After
//! random orthogonal rotation, KV cache vector coordinates are approximately
//! i.i.d. Gaussian, making this codebook near-optimal.

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
        // Middle boundary is at 0
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
        // For values within ±3σ, quantization error < max step size
        for i in -30..=30 {
            let x = i as f32 * 0.1;
            let idx = cb.quantize(x);
            let reconstructed = cb.dequantize(idx);
            let err = (x - reconstructed).abs();
            // Max step between centroids is ~0.5 for N(0,1) inner levels
            // Max error should be < half the largest step
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
        // Scaled codebook centroids should be half
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
        // Index > 7 should be clamped to 7
        assert_eq!(cb.dequantize(255), cb.dequantize(7));
    }
}
```

- [ ] **Step 2: Add module declaration and run tests to verify they fail**

In `crates/harmony-inference/src/kv_compress/mod.rs`, add after the packing module line:
```rust
pub(crate) mod lloyd_max;
```

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::lloyd_max 2>&1 | head -20
```

Expected: compilation error — `LloydMaxCodebook` not defined.

- [ ] **Step 3: Implement LloydMaxCodebook**

Add above the `#[cfg(test)]` block in `lloyd_max.rs`:

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::lloyd_max
```

Expected: 8 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/kv_compress/lloyd_max.rs
git add crates/harmony-inference/src/kv_compress/mod.rs
git commit -m "feat(inference): Lloyd-Max Gaussian codebook for TurboQuant"
```

---

### Task 3: Random orthogonal matrix

**Files:**
- Create: `crates/harmony-inference/src/kv_compress/orthogonal.rs`
- Modify: `crates/harmony-inference/src/kv_compress/mod.rs` (add module declaration)

Generates a deterministic random orthogonal matrix via seeded RNG + modified
Gram-Schmidt. Also provides `transpose()` and `matvec()` helpers used by the
compress/decompress pipeline.

- [ ] **Step 1: Write failing tests**

Create `crates/harmony-inference/src/kv_compress/orthogonal.rs`:

```rust
//! Random orthogonal matrix generation via modified Gram-Schmidt.
//!
//! Generates a deterministic orthogonal matrix from a seed. Used for
//! preconditioning KV cache vectors before quantization — smooths
//! channel-wise outliers so all dimensions have similar distributions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orthogonal_matrix_has_correct_size() {
        let q = generate_orthogonal_matrix(8, 42);
        assert_eq!(q.len(), 64); // 8 * 8
    }

    #[test]
    fn orthogonal_matrix_is_orthogonal() {
        // Q @ Q^T should be approximately identity
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
        assert_eq!(q1, q2, "same seed should produce identical matrix");
    }

    #[test]
    fn different_seeds_produce_different_matrices() {
        let q1 = generate_orthogonal_matrix(8, 42);
        let q2 = generate_orthogonal_matrix(8, 99);
        assert_ne!(q1, q2, "different seeds should produce different matrices");
    }

    #[test]
    fn transpose_is_inverse() {
        // For orthogonal Q: Q^T @ Q = I
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
        // Build identity matrix
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
                v[i],
                recovered[i]
            );
        }
    }

    #[test]
    fn works_with_dim_80() {
        let dim = 80;
        let q = generate_orthogonal_matrix(dim, 42);
        assert_eq!(q.len(), 6400);

        // Spot-check orthogonality: first two columns
        let mut dot = 0.0f32;
        for row in 0..dim {
            dot += q[row * dim] * q[row * dim + 1];
        }
        assert!(dot.abs() < 1e-4, "columns 0,1 dot product = {dot}");
    }
}
```

- [ ] **Step 2: Add module declaration and run tests to verify they fail**

In `crates/harmony-inference/src/kv_compress/mod.rs`, add:
```rust
pub(crate) mod orthogonal;
```

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::orthogonal 2>&1 | head -20
```

Expected: compilation error — functions not defined.

- [ ] **Step 3: Implement the orthogonal matrix module**

Add above the `#[cfg(test)]` block in `orthogonal.rs`:

```rust
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::orthogonal
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/kv_compress/orthogonal.rs
git add crates/harmony-inference/src/kv_compress/mod.rs
git commit -m "feat(inference): random orthogonal matrix via modified Gram-Schmidt"
```

---

### Task 4: QJL projection

**Files:**
- Create: `crates/harmony-inference/src/kv_compress/qjl.rs`
- Modify: `crates/harmony-inference/src/kv_compress/mod.rs` (add module declaration)

Rademacher ±1 JL matrix for 1-bit sign-based error correction. Encodes the
quantization residual as sign bits and decodes a correction vector during
decompression.

- [ ] **Step 1: Write failing tests**

Create `crates/harmony-inference/src/kv_compress/qjl.rs`:

```rust
//! Quantized Johnson-Lindenstrauss (QJL) error correction.
//!
//! Projects quantization residuals through a random ±1 (Rademacher) matrix
//! and stores only the sign bits. During decompression, the sign bits are
//! decoded into a correction vector that reduces reconstruction error.
//! The correction provides an unbiased inner product estimator.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use super::packing::{pack_1bit, unpack_1bit};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jl_matrix_has_correct_size() {
        let jl = JlMatrix::new(80, 42);
        // 80 * 80 = 6400 bits = 800 bytes
        assert_eq!(jl.packed.len(), 800);
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
        assert_eq!(signs.len(), 10); // 80 bits = 10 bytes
    }

    #[test]
    fn encode_signs_of_zero_residual() {
        let jl = JlMatrix::new(8, 42);
        let residual = vec![0.0f32; 8];
        let signs = jl.encode_signs(&residual);
        // All projections are 0 → sign(0) = false (not positive)
        let unpacked = unpack_1bit(&signs, 8);
        assert!(unpacked.iter().all(|&s| !s), "sign(0) should be false");
    }

    #[test]
    fn decode_correction_has_correct_length() {
        let jl = JlMatrix::new(80, 42);
        let signs = vec![0u8; 10]; // 80 bits, all false
        let correction = jl.decode_correction(&signs, 1.0);
        assert_eq!(correction.len(), 80);
    }

    #[test]
    fn decode_correction_with_zero_scale_is_zero() {
        let jl = JlMatrix::new(8, 42);
        let signs = vec![0xFFu8]; // 8 bits, all true
        let correction = jl.decode_correction(&signs, 0.0);
        assert!(
            correction.iter().all(|&v| v.abs() < 1e-10),
            "zero scale should produce zero correction"
        );
    }

    #[test]
    fn correction_reduces_error() {
        // Generate a known residual, encode signs, decode correction.
        // The correction should reduce the L2 error vs the original residual.
        let dim = 32;
        let jl = JlMatrix::new(dim, 42);

        // Residual: a structured vector
        let residual: Vec<f32> = (0..dim).map(|i| (i as f32 - 16.0) * 0.01).collect();
        let residual_norm: f32 = residual.iter().map(|x| x * x).sum::<f32>().sqrt();

        let signs = jl.encode_signs(&residual);

        // Scale = sqrt(π/2) * ||residual||
        let scale = core::f32::consts::FRAC_PI_2.sqrt() * residual_norm;
        let correction = jl.decode_correction(&signs, scale);

        // Error without correction = ||residual||²
        let err_without: f32 = residual.iter().map(|x| x * x).sum();

        // Error with correction = ||residual - correction||²
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
```

- [ ] **Step 2: Add module declaration and run tests to verify they fail**

In `crates/harmony-inference/src/kv_compress/mod.rs`, add:
```rust
pub(crate) mod qjl;
```

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::qjl 2>&1 | head -20
```

Expected: compilation error — `JlMatrix` not defined.

- [ ] **Step 3: Implement JlMatrix**

Add above the `#[cfg(test)]` block in `qjl.rs`:

```rust
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
    /// Computes `correction = (scale / dim) * M^T @ sign_vector`
    /// where sign bits are decoded as +1 (true) or -1 (false).
    ///
    /// The caller should pass `scale = sqrt(π/2) * expected_residual_norm`.
    pub fn decode_correction(&self, packed_signs: &[u8], scale: f32) -> Vec<f32> {
        let dim = self.dim;
        let signs = unpack_1bit(packed_signs, dim);
        let scale_over_dim = scale / dim as f32;
        let mut correction = vec![0.0f32; dim];
        // M^T @ sign_vector: for each output col, dot column of M with sign vector
        for col in 0..dim {
            let mut dot = 0.0f32;
            for row in 0..dim {
                let jl_val = self.get(row, col);
                let sign_val = if signs[row] { 1.0 } else { -1.0 };
                dot += jl_val * sign_val;
            }
            correction[col] = scale_over_dim * dot;
        }
        correction
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::qjl
```

Expected: 10 tests pass.

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-inference/src/kv_compress/qjl.rs
git add crates/harmony-inference/src/kv_compress/mod.rs
git commit -m "feat(inference): QJL sign-bit error correction for TurboQuant"
```

---

### Task 5: TurboQuant pipeline

**Files:**
- Rewrite: `crates/harmony-inference/src/kv_compress/mod.rs`

Replace the old uniform quantization code with the TurboQuant pipeline. This is
the largest task — it wires together all four submodules (packing, lloyd_max,
orthogonal, qjl) into `TurboQuantState` with `compress_tensor` / `decompress_tensor`.

**Important:** This task rewrites `mod.rs` completely. The old `QuantizedVec`,
`quantize_vec`, `dequantize_vec`, `compress_tensor`, `decompress_tensor` functions
are all replaced. The old tests in mod.rs are replaced with new TurboQuant tests.
The lib.rs integration tests will be updated in Task 6.

- [ ] **Step 1: Write the new `mod.rs` with tests**

Replace the entire contents of `crates/harmony-inference/src/kv_compress/mod.rs` with:

```rust
//! TurboQuant KV cache compression.
//!
//! Three-stage pipeline replacing uniform 3-bit quantization:
//! 1. Random orthogonal rotation (smooths channel-wise outliers)
//! 2. PolarQuant: radius + Cartesian Lloyd-Max on unit vector (3-bit, global codebook)
//! 3. QJL: 1-bit sign projection of quantization residual (error correction)
//!
//! Feature-gated behind `kv-compress`.

pub(crate) mod lloyd_max;
pub(crate) mod orthogonal;
pub(crate) mod packing;
pub(crate) mod qjl;

use crate::error::InferenceError;
use candle_core::{DType, Device, Tensor};
use serde::{Deserialize, Serialize};

use lloyd_max::{LloydMaxCodebook, NMSE_3BIT_GAUSSIAN};
use orthogonal::{generate_orthogonal_matrix, matvec, transpose};
use packing::{pack_3bit, unpack_3bit};
use qjl::JlMatrix;

/// Configuration for TurboQuant compression.
#[derive(Debug, Clone)]
pub struct TurboQuantConfig {
    /// Dimension of each KV head vector.
    pub head_dim: usize,
    /// Deterministic seed for random matrix generation.
    pub seed: u64,
}

/// Precomputed state for TurboQuant compression/decompression.
///
/// Initialized once at model load. Immutable after creation. Multiple
/// caches (different conversations) share the same state via `&TurboQuantState`.
pub struct TurboQuantState {
    config: TurboQuantConfig,
    /// Random orthogonal matrix Q, row-major [head_dim × head_dim].
    ortho_matrix: Vec<f32>,
    /// Q^T (inverse rotation), row-major [head_dim × head_dim].
    ortho_transpose: Vec<f32>,
    /// Lloyd-Max codebook scaled for unit vector coordinates.
    codebook: LloydMaxCodebook,
    /// Rademacher JL matrix for sign-bit error correction.
    jl_matrix: JlMatrix,
    /// Precomputed: sqrt(π/2) * sqrt(NMSE).
    correction_scale: f32,
}

impl TurboQuantState {
    /// Initialize TurboQuant state from config.
    ///
    /// Generates the random orthogonal matrix and JL matrix deterministically
    /// from the seed. Uses different seed offsets for the two matrices to
    /// avoid correlation.
    pub fn new(config: &TurboQuantConfig) -> Result<Self, InferenceError> {
        if config.head_dim == 0 {
            return Err(InferenceError::CompressionFailed(
                "head_dim must be > 0".into(),
            ));
        }

        let dim = config.head_dim;
        let ortho_matrix = generate_orthogonal_matrix(dim, config.seed);
        let ortho_transpose = transpose(&ortho_matrix, dim);

        // Unit vector coordinates ~ N(0, 1/dim), so sigma = 1/sqrt(dim)
        let sigma = 1.0 / (dim as f32).sqrt();
        let codebook = LloydMaxCodebook::gaussian(sigma);

        // Use offset seed for JL matrix to avoid correlation
        let jl_matrix = JlMatrix::new(dim, config.seed.wrapping_add(0x517cc1b727220a95));

        // Precompute correction scale: sqrt(π/2) * sqrt(NMSE)
        let expected_relative_error = NMSE_3BIT_GAUSSIAN.sqrt();
        let correction_scale =
            core::f32::consts::FRAC_PI_2.sqrt() * expected_relative_error;

        Ok(Self {
            config: config.clone(),
            ortho_matrix,
            ortho_transpose,
            codebook,
            jl_matrix,
            correction_scale,
        })
    }

    /// Compress a single f32 vector through the TurboQuant pipeline.
    fn compress_vec(&self, data: &[f32]) -> TurboQuantVec {
        let dim = self.config.head_dim;

        // Stage 1: Rotate
        let rotated = matvec(&self.ortho_matrix, data, dim);

        // Stage 2: PolarQuant — radius + Lloyd-Max on unit vector
        let radius: f32 = rotated.iter().map(|x| x * x).sum::<f32>().sqrt();

        if radius < 1e-30 {
            // Zero vector — no quantization needed
            return TurboQuantVec {
                radius: 0.0,
                angles: vec![0u8; (dim * 3).div_ceil(8)],
                signs: vec![0u8; dim.div_ceil(8)],
            };
        }

        let inv_radius = 1.0 / radius;
        let indices: Vec<u8> = rotated
            .iter()
            .map(|&v| self.codebook.quantize(v * inv_radius))
            .collect();
        let angles = pack_3bit(&indices);

        // Reconstruct for residual computation
        let reconstructed: Vec<f32> = indices
            .iter()
            .map(|&idx| self.codebook.dequantize(idx) * radius)
            .collect();

        // Stage 3: QJL — sign-bit encode of residual
        let residual: Vec<f32> = rotated
            .iter()
            .zip(reconstructed.iter())
            .map(|(r, q)| r - q)
            .collect();
        let signs = self.jl_matrix.encode_signs(&residual);

        TurboQuantVec {
            radius,
            angles,
            signs,
        }
    }

    /// Decompress a TurboQuantVec back to an f32 vector.
    fn decompress_vec(&self, qv: &TurboQuantVec) -> Vec<f32> {
        let dim = self.config.head_dim;

        if qv.radius.abs() < 1e-30 {
            return vec![0.0f32; dim];
        }

        // Stage 2 (reverse): Dequantize unit vector, scale by radius
        let indices = unpack_3bit(&qv.angles, dim);
        let mut reconstructed: Vec<f32> = indices
            .iter()
            .map(|&idx| self.codebook.dequantize(idx) * qv.radius)
            .collect();

        // Stage 3 (reverse): QJL correction
        let scale = self.correction_scale * qv.radius;
        let correction = self.jl_matrix.decode_correction(&qv.signs, scale);
        for i in 0..dim {
            reconstructed[i] += correction[i];
        }

        // Stage 1 (reverse): Inverse rotate
        matvec(&self.ortho_transpose, &reconstructed, dim)
    }

    /// Compress an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
    ///
    /// Returns (compressed_vecs, seq_len). One `TurboQuantVec` per (head, token)
    /// pair, stored row-major.
    pub fn compress_tensor(
        &self,
        tensor: &Tensor,
    ) -> Result<(Vec<TurboQuantVec>, usize), InferenceError> {
        let (batch, num_heads, seq_len, head_dim) = tensor
            .dims4()
            .map_err(|e| InferenceError::CompressionFailed(format!("unexpected shape: {e}")))?;
        if batch != 1 {
            return Err(InferenceError::CompressionFailed(format!(
                "expected batch=1, got {batch}"
            )));
        }
        if head_dim != self.config.head_dim {
            return Err(InferenceError::CompressionFailed(format!(
                "head_dim mismatch: state has {}, tensor has {head_dim}",
                self.config.head_dim
            )));
        }

        let total_vecs = num_heads * seq_len;
        let flat = tensor
            .to_dtype(DType::F32)
            .and_then(|t| t.reshape((total_vecs, head_dim)))
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;
        let rows = flat
            .to_vec2::<f32>()
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))?;

        let vecs: Vec<TurboQuantVec> = rows.iter().map(|row| self.compress_vec(row)).collect();
        Ok((vecs, seq_len))
    }

    /// Decompress to an f16 tensor `[1, num_kv_heads, seq_len, head_dim]`.
    pub fn decompress_tensor(
        &self,
        vecs: &[TurboQuantVec],
        num_kv_heads: usize,
        seq_len: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Tensor, InferenceError> {
        if head_dim != self.config.head_dim {
            return Err(InferenceError::CompressionFailed(format!(
                "head_dim mismatch: state has {}, requested {head_dim}",
                self.config.head_dim
            )));
        }
        let expected = num_kv_heads * seq_len;
        if vecs.len() != expected {
            return Err(InferenceError::CompressionFailed(format!(
                "expected {expected} vecs, got {}",
                vecs.len()
            )));
        }

        let mut flat = Vec::with_capacity(expected * head_dim);
        for qv in vecs {
            flat.extend_from_slice(&self.decompress_vec(qv));
        }

        Tensor::from_vec(flat, (1, num_kv_heads, seq_len, head_dim), device)
            .and_then(|t| t.to_dtype(DType::F16))
            .map_err(|e| InferenceError::CompressionFailed(e.to_string()))
    }
}

/// A single TurboQuant-compressed vector.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TurboQuantVec {
    /// Vector magnitude (L2 norm of rotated vector).
    pub radius: f32,
    /// 3-bit packed Lloyd-Max indices for unit vector coordinates.
    pub angles: Vec<u8>,
    /// 1-bit packed QJL sign projections.
    pub signs: Vec<u8>,
}

impl TurboQuantVec {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        4 + self.angles.len() + self.signs.len()
    }
}

/// Compressed representation of one layer's K and V tensors.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompressedKvLayer {
    /// Compressed key vectors, row-major: [head0_tok0, head0_tok1, ..., headH_tokT].
    pub k: Vec<TurboQuantVec>,
    /// Compressed value vectors, same layout as k.
    pub v: Vec<TurboQuantVec>,
    /// Sequence length at compression time.
    pub seq_len: usize,
}

impl CompressedKvLayer {
    /// Approximate memory usage in bytes.
    pub fn byte_size(&self) -> usize {
        self.k.iter().map(|q| q.byte_size()).sum::<usize>()
            + self.v.iter().map(|q| q.byte_size()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> TurboQuantConfig {
        TurboQuantConfig {
            head_dim: 80,
            seed: 42,
        }
    }

    fn test_state() -> TurboQuantState {
        TurboQuantState::new(&test_config()).unwrap()
    }

    // ── State construction ──

    #[test]
    fn new_state_succeeds() {
        let state = test_state();
        assert_eq!(state.config.head_dim, 80);
    }

    #[test]
    fn new_state_rejects_zero_dim() {
        let cfg = TurboQuantConfig {
            head_dim: 0,
            seed: 42,
        };
        assert!(TurboQuantState::new(&cfg).is_err());
    }

    // ── Per-vector compress/decompress ──

    #[test]
    fn compress_vec_produces_correct_sizes() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        assert_eq!(qv.angles.len(), 30); // ceil(80 * 3 / 8)
        assert_eq!(qv.signs.len(), 10);  // ceil(80 / 8)
        assert!(qv.radius > 0.0);
    }

    #[test]
    fn compress_decompress_vec_bounded_error() {
        let state = test_state();
        // Generate a realistic vector (random-ish values)
        let data: Vec<f32> = (0..80).map(|i| ((i * 7 + 3) % 13) as f32 * 0.1 - 0.6).collect();
        let data_norm: f32 = data.iter().map(|x| x * x).sum::<f32>().sqrt();

        let qv = state.compress_vec(&data);
        let recovered = state.decompress_vec(&qv);

        let err: f32 = data
            .iter()
            .zip(recovered.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            .sqrt();

        // Relative error should be < 30% (3-bit quantization + QJL correction)
        let relative_err = err / data_norm;
        assert!(
            relative_err < 0.3,
            "relative error {relative_err:.4} too large (norm={data_norm:.4}, err={err:.4})"
        );
    }

    #[test]
    fn zero_vector_roundtrips() {
        let state = test_state();
        let data = vec![0.0f32; 80];
        let qv = state.compress_vec(&data);
        assert!(qv.radius.abs() < 1e-20);
        let recovered = state.decompress_vec(&qv);
        assert!(recovered.iter().all(|&v| v.abs() < 1e-10));
    }

    #[test]
    fn byte_size_matches_budget() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        // 4 (f32 radius) + 30 (angles) + 10 (signs) = 44
        assert_eq!(qv.byte_size(), 44);
    }

    // ── Tensor bridge ──

    #[test]
    fn compress_tensor_roundtrip() {
        let state = test_state();
        let shape = (1, 8, 4, 80);
        let tensor = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();

        let (vecs, seq_len) = state.compress_tensor(&tensor).unwrap();
        assert_eq!(seq_len, 4);
        assert_eq!(vecs.len(), 32); // 8 heads * 4 tokens

        let restored = state
            .decompress_tensor(&vecs, 8, 4, 80, &Device::Cpu)
            .unwrap();
        assert_eq!(restored.dims4().unwrap(), (1, 8, 4, 80));
        assert_eq!(restored.dtype(), DType::F16);
    }

    #[test]
    fn compress_tensor_validates_head_dim() {
        let state = test_state(); // head_dim=80
        // Tensor with head_dim=128 should fail
        let tensor = Tensor::rand(0f32, 1f32, (1, 8, 4, 128), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert!(state.compress_tensor(&tensor).is_err());
    }

    #[test]
    fn compress_tensor_validates_batch_size() {
        let state = test_state();
        let tensor = Tensor::rand(0f32, 1f32, (2, 8, 4, 80), &Device::Cpu)
            .unwrap()
            .to_dtype(DType::F16)
            .unwrap();
        assert!(state.compress_tensor(&tensor).is_err());
    }

    // ── Serialization ──

    #[test]
    fn turboquant_vec_serde_roundtrip() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);

        let bytes = postcard::to_allocvec(&qv).unwrap();
        let restored: TurboQuantVec = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(qv.radius, restored.radius);
        assert_eq!(qv.angles, restored.angles);
        assert_eq!(qv.signs, restored.signs);
    }

    #[test]
    fn compressed_kv_layer_serde_roundtrip() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 4],
            v: vec![qv; 4],
            seq_len: 2,
        };

        let bytes = postcard::to_allocvec(&layer).unwrap();
        let restored: CompressedKvLayer = postcard::from_bytes(&bytes).unwrap();

        assert_eq!(layer.seq_len, restored.seq_len);
        assert_eq!(layer.k.len(), restored.k.len());
        assert_eq!(layer.v.len(), restored.v.len());
    }

    #[test]
    fn compressed_kv_layer_byte_size() {
        let state = test_state();
        let data = vec![1.0f32; 80];
        let qv = state.compress_vec(&data);
        let layer = CompressedKvLayer {
            k: vec![qv.clone(); 8],
            v: vec![qv; 8],
            seq_len: 1,
        };
        // 2 tensors * 8 vecs * 44 bytes = 704
        assert_eq!(layer.byte_size(), 704);
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress::tests
```

Expected: 12 mod-level tests pass. The submodule tests should also still pass:

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress kv_compress
```

Expected: all kv_compress tests pass (packing + lloyd_max + orthogonal + qjl + mod tests).

**Note:** The lib.rs `kv_compress_cache_tests` will fail at this point because the
API changed (compress/decompress are now methods on TurboQuantState, not free functions).
This is expected — Task 6 fixes the integration layer.

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-inference/src/kv_compress/mod.rs
git commit -m "feat(inference): TurboQuant pipeline — three-stage compress/decompress"
```

---

### Task 6: InferenceCache integration

**Files:**
- Modify: `crates/harmony-inference/src/lib.rs`

Update the `InferenceCache` compress/decompress methods to accept `&TurboQuantState`.
Update all existing `kv_compress_cache_tests` to work with the new API. The error
tolerances for the roundtrip test are adjusted for TurboQuant's different
reconstruction error characteristics.

- [ ] **Step 1: Update `compress()` and `decompress()` signatures**

In `crates/harmony-inference/src/lib.rs`, change:

```rust
pub fn compress(&mut self) -> Result<(), InferenceError> {
```
to:
```rust
pub fn compress(&mut self, tq: &kv_compress::TurboQuantState) -> Result<(), InferenceError> {
```

And update the body — replace:
```rust
                    let (k_vecs, seq_len) = kv_compress::compress_tensor(k)?;
                    let (v_vecs, _) = kv_compress::compress_tensor(v)?;
```
with:
```rust
                    let (k_vecs, seq_len) = tq.compress_tensor(k)?;
                    let (v_vecs, _) = tq.compress_tensor(v)?;
```

Similarly, change:
```rust
pub fn decompress(&mut self) -> Result<(), InferenceError> {
```
to:
```rust
pub fn decompress(&mut self, tq: &kv_compress::TurboQuantState) -> Result<(), InferenceError> {
```

And update the body — replace:
```rust
                    let k = kv_compress::decompress_tensor(
                        &c.k,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
                    let v = kv_compress::decompress_tensor(
                        &c.v,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
```
with:
```rust
                    let k = tq.decompress_tensor(
                        &c.k,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
                    let v = tq.decompress_tensor(
                        &c.v,
                        self.num_kv_heads,
                        c.seq_len,
                        self.head_dim,
                        &device,
                    )?;
```

- [ ] **Step 2: Update `kv_compress_cache_tests`**

The test module `kv_compress_cache_tests` in `lib.rs` needs these changes:

**Add test helper at the top of the test module:**
```rust
    fn test_tq_state(head_dim: usize) -> kv_compress::TurboQuantState {
        kv_compress::TurboQuantState::new(&kv_compress::TurboQuantConfig {
            head_dim,
            seed: 42,
        })
        .unwrap()
    }
```

**Update `cache_with_data` to use head_dim=80** (TurboQuant state is configured for head_dim):

Change all calls to `cache_with_data(2, 8, 128, ...)` to `cache_with_data(2, 8, 80, ...)`.
Change all references to `head_dim` of 128 to 80.

**Update every test that calls `compress()` or `decompress()`** to pass `&tq`:

For example, change:
```rust
cache.compress().unwrap();
```
to:
```rust
let tq = test_tq_state(80);
cache.compress(&tq).unwrap();
```

And:
```rust
cache.decompress().unwrap();
```
to:
```rust
cache.decompress(&tq).unwrap();
```

**Update the roundtrip error tolerance:**

In `compress_decompress_roundtrip`, the max reconstruction error check:
```rust
assert!(max_err < 0.15, ...);
```
Change to:
```rust
// TurboQuant: 3-bit Lloyd-Max + QJL correction, higher per-element error
// but unbiased. Relative error should be < 50% per element.
assert!(max_err < 0.5, "max reconstruction error {max_err} too large");
```

**Update shape assertions** in tests that check tensor dimensions — change `128` to `80`:
```rust
assert_eq!(k.dims4().unwrap(), (1, 8, 16, 80));
assert_eq!(v.dims4().unwrap(), (1, 8, 16, 80));
```

**Update the `compress_atomic_on_error` test:** The current test creates a 3D tensor
to trigger a `dims4()` error. This should still work since `compress_tensor` still
calls `dims4()`.

- [ ] **Step 3: Verify full test suite passes**

```bash
cd crates/harmony-inference && cargo test --lib --features kv-compress
```

Expected: all tests pass — kv_compress submodule tests + updated cache tests.

```bash
cd crates/harmony-inference && cargo test --lib
```

Expected: all non-kv-compress tests also pass (block_attnres, uq_head, etc.).

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-inference/src/lib.rs
git commit -m "feat(inference): update InferenceCache to use TurboQuantState"
```

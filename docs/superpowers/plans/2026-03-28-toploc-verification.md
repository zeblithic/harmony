# TOPLOC Verification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 258-byte TOPLOC polynomial proofs to prefill KV cache blobs, enabling trustless verification of cache integrity between mesh nodes.

**Architecture:** A new `toploc.rs` module in harmony-speculative (behind `prefill` feature) implements pure integer math: top-k mantissa extraction, injective modulus search, Newton Divided Differences → monomial form, Horner evaluation. `prefill.rs` gains a `proofs` field on `PrefillCacheHeader`, a `store_prefill_cache_with_proofs()` function, and magic v2 support. No new dependencies.

**Tech Stack:** Rust, pure integer arithmetic (u16/u32), IEEE 754 bit manipulation, harmony-crypto SHA-256 for deterministic sampling

**Spec:** `docs/superpowers/specs/2026-03-28-toploc-verification-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `crates/harmony-speculative/src/toploc.rs` | Create | Math core: modulus search, NDD, Horner, mantissa extraction, top-k |
| `crates/harmony-speculative/src/toploc_proof.rs` | Create | TocProof type, generate_proofs, verify_proofs, sampling, VerifyResult |
| `crates/harmony-speculative/src/prefill.rs` | Modify | Add proofs to header, magic v2, store_with_proofs, load handles v1+v2 |
| `crates/harmony-speculative/src/lib.rs` | Modify | Module declarations for toploc + toploc_proof |

**Why two files:** `toploc.rs` is pure math (no candle, no CAS, no types — just functions on `[u16; 128]`). `toploc_proof.rs` is the integration layer (TocProof struct, candle tensor extraction, generate/verify). This separation keeps the math independently testable and the tensor integration focused.

---

### Task 1: Integer Math Core (toploc.rs)

**Files:**
- Create: `crates/harmony-speculative/src/toploc.rs`
- Modify: `crates/harmony-speculative/src/lib.rs` (module declaration)

This is the largest task — all the pure math functions. No tensor dependencies, no external types.

- [ ] **Step 1: Create toploc.rs with extended_gcd and modular inverse**

Create `crates/harmony-speculative/src/toploc.rs`:

```rust
//! TOPLOC integer math core.
//!
//! Pure integer arithmetic over finite fields Z_m for polynomial proof
//! generation and verification. No heap allocation, no floating point.
//!
//! Feature-gated behind `prefill`.

/// Number of top-k activations encoded per proof.
pub(crate) const TOP_K: usize = 128;

/// Tokens per proof chunk.
pub(crate) const CHUNK_SIZE: usize = 32;

/// Starting modulus for the injective search — largest prime that fits in u16.
/// Must be prime so that all nonzero differences have modular inverses in NDD.
const START_MODULUS: u16 = 65521;

/// Extended Euclidean Algorithm.
/// Returns (gcd, x) such that a*x + b*y = gcd for some y.
fn extended_gcd(a: i64, b: i64) -> (i64, i64) {
    if a == 0 {
        return (b, 0);
    }
    let (g, x1) = extended_gcd(b % a, a);
    (g, x1 - (b / a) * x1.wrapping_mul(0) + x1) // placeholder — see step 3
}
```

Actually, let me write the complete, correct implementations. The extended_gcd is small:

```rust
//! TOPLOC integer math core.
//!
//! Pure integer arithmetic over finite fields Z_m for polynomial proof
//! generation and verification. No heap allocation in the hot path.
//!
//! Feature-gated behind `prefill`.

/// Number of top-k activations encoded per proof.
pub(crate) const TOP_K: usize = 128;

/// Tokens per proof chunk.
pub(crate) const CHUNK_SIZE: usize = 32;

/// Extended Euclidean Algorithm (iterative).
/// Returns (gcd, x) such that a*x ≡ gcd (mod b).
pub(crate) fn extended_gcd(a: u32, b: u32) -> (u32, i64) {
    let (mut old_r, mut r) = (a as i64, b as i64);
    let (mut old_s, mut s) = (1_i64, 0_i64);
    while r != 0 {
        let q = old_r / r;
        let tmp_r = r;
        r = old_r - q * r;
        old_r = tmp_r;
        let tmp_s = s;
        s = old_s - q * s;
        old_s = tmp_s;
    }
    (old_r as u32, old_s)
}

/// Modular multiplicative inverse of a in Z_m.
/// Panics if gcd(a, m) != 1 (no inverse exists).
pub(crate) fn mod_inverse(a: u16, m: u16) -> u16 {
    let (g, x) = extended_gcd(a as u32, m as u32);
    assert_eq!(g, 1, "no modular inverse: gcd({a}, {m}) = {g}");
    x.rem_euclid(m as i64) as u16
}

/// Find a prime m such that all indices are unique mod m.
/// Starts at 65521 (largest prime ≤ u16::MAX) and searches downward
/// through primes only. Primality ensures all nonzero elements have
/// modular inverses, which NDD interpolation requires.
pub(crate) fn find_injective_modulus(indices: &[u16; TOP_K]) -> u16 {
    let mut seen = [false; 65536];
    let mut m = START_MODULUS;
    loop {
        // Clear seen array for this attempt
        for s in seen[..m as usize + 1].iter_mut() {
            *s = false;
        }
        let mut collision = false;
        for &idx in indices {
            let mapped = (idx as u32 % m as u32) as usize;
            if seen[mapped] {
                collision = true;
                break;
            }
            seen[mapped] = true;
        }
        if !collision {
            return m;
        }
        // Find next smaller prime
        m -= if m % 2 == 0 { 1 } else { 2 };
        while !is_prime(m) {
            m -= if m % 2 == 0 { 1 } else { 2 };
        }
        assert!(m >= TOP_K as u16, "no injective prime modulus found");
    }
}

/// Simple trial-division primality test. Only used during modulus search
/// (at most a few iterations, small numbers ≤ 65521).
fn is_prime(n: u16) -> bool {
    if n < 2 { return false; }
    if n < 4 { return true; }
    if n % 2 == 0 || n % 3 == 0 { return false; }
    let mut i = 5u32;
    while i * i <= n as u32 {
        if n as u32 % i == 0 || n as u32 % (i + 2) == 0 { return false; }
        i += 6;
    }
    true
}

/// Newton Divided Differences interpolation over Z_m, converted to monomial form.
///
/// Given k (x, y) pairs, computes the unique degree-(k-1) interpolating polynomial
/// and returns its coefficients in monomial (power) basis: c_0 + c_1*x + c_2*x^2 + ...
///
/// Two phases:
/// 1. NDD: build divided difference table to get Newton basis coefficients
/// 2. Convert Newton basis to monomial form via synthetic expansion
pub(crate) fn ndd_interpolate(
    xs: &[u16; TOP_K],
    ys: &[u16; TOP_K],
    m: u16,
) -> [u16; TOP_K] {
    let m32 = m as u32;
    let k = TOP_K;

    // Phase 1: Newton divided differences
    // ndd[i] will hold the i-th divided difference coefficient
    let mut ndd = [0u32; TOP_K];
    for i in 0..k {
        ndd[i] = ys[i] as u32 % m32;
    }
    for j in 1..k {
        for i in (j..k).rev() {
            let num = (ndd[i] + m32 - ndd[i - 1]) % m32;
            let den = ((xs[i] as u32 + m32 - xs[i - j] as u32) % m32) as u16;
            let inv = mod_inverse(den, m);
            ndd[i] = (num * inv as u32) % m32;
        }
    }

    // Phase 2: Convert Newton basis to monomial form
    // Newton: p(x) = ndd[0] + ndd[1](x - xs[0]) + ndd[2](x - xs[0])(x - xs[1]) + ...
    // Expand iteratively into monomial coefficients
    let mut mono = [0u32; TOP_K];
    mono[0] = ndd[k - 1];

    for i in (0..k - 1).rev() {
        // Multiply current polynomial by (x - xs[i]):
        // new[j] = old[j-1] - xs[i] * old[j]
        for j in (1..k).rev() {
            mono[j] = (mono[j - 1] + m32 - (xs[i] as u32 * mono[j]) % m32) % m32;
        }
        mono[0] = (m32 - (xs[i] as u32 * mono[0]) % m32 + ndd[i]) % m32;
    }

    let mut result = [0u16; TOP_K];
    for i in 0..k {
        result[i] = mono[i] as u16;
    }
    result
}

/// Evaluate polynomial in monomial form at x using Horner's method over Z_m.
/// p(x) = c[0] + c[1]*x + c[2]*x^2 + ... = c[0] + x*(c[1] + x*(c[2] + ...))
/// O(k). All arithmetic in u32 to prevent overflow on 32-bit targets.
pub(crate) fn horner_evaluate(coeffs: &[u16; TOP_K], x: u16, m: u16) -> u16 {
    let m32 = m as u32;
    let x32 = x as u32;
    let mut result = coeffs[TOP_K - 1] as u32;
    for i in (0..TOP_K - 1).rev() {
        result = (result * x32 + coeffs[i] as u32) % m32;
    }
    result as u16
}

/// Extract IEEE 754 mantissa (lower 16 bits of 23-bit fraction) from an f32.
#[inline]
pub(crate) fn f32_mantissa_u16(value: f32) -> u16 {
    (value.to_bits() & 0x007F_FFFF) as u16
}

/// Extract IEEE 754 exponent (8 bits) from an f32.
#[inline]
pub(crate) fn f32_exponent(value: f32) -> u8 {
    ((value.to_bits() >> 23) & 0xFF) as u8
}

/// Find the top-k elements by absolute magnitude in an f32 slice.
/// Returns (indices, mantissa_values) as fixed-size arrays.
/// Panics if data.len() < k.
pub(crate) fn extract_top_k(data: &[f32], k: usize) -> (Vec<u16>, Vec<u16>) {
    assert!(data.len() >= k, "data length {} < k {}", data.len(), k);

    // Collect (abs_value, original_index) pairs
    let mut indexed: Vec<(f32, usize)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();

    // Partial sort: get top k by descending absolute value
    indexed.select_nth_unstable_by(k - 1, |a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

    let top_k = &indexed[..k];

    let indices: Vec<u16> = top_k.iter().map(|(_, i)| *i as u16).collect();
    let mantissas: Vec<u16> = top_k.iter().map(|(_, i)| f32_mantissa_u16(data[*i])).collect();

    (indices, mantissas)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_gcd_known_vectors() {
        // gcd(3, 7) = 1, 3*5 ≡ 1 (mod 7)
        let inv = mod_inverse(3, 7);
        assert_eq!((3u32 * inv as u32) % 7, 1);

        // gcd(5, 13) = 1, 5*8 ≡ 1 (mod 13)
        let inv = mod_inverse(5, 13);
        assert_eq!((5u32 * inv as u32) % 13, 1);
    }

    #[test]
    fn mod_inverse_all_coprime_to_prime() {
        let m = 65521u16; // largest prime ≤ 65535
        for a in [1u16, 2, 127, 1000, 65520] {
            let inv = mod_inverse(a, m);
            assert_eq!((a as u32 * inv as u32) % m as u32, 1, "inverse of {a} mod {m}");
        }
    }

    #[test]
    fn injective_modulus_finds_valid_prime() {
        let mut indices = [0u16; TOP_K];
        for i in 0..TOP_K {
            indices[i] = (i * 31 + 7) as u16; // spread out indices
        }
        let m = find_injective_modulus(&indices);
        // Verify modulus is prime
        assert!(is_prime(m), "modulus {m} should be prime");
        // Verify all indices are unique mod m
        let mut seen = std::collections::HashSet::new();
        for &idx in &indices {
            assert!(seen.insert(idx as u32 % m as u32), "collision at {idx} mod {m}");
        }
    }

    #[test]
    fn injective_modulus_with_scattered_indices() {
        // Simulate real top-k extraction: indices scattered across 0..4095
        let mut indices = [0u16; TOP_K];
        for i in 0..TOP_K {
            indices[i] = ((i * 251 + 13) % 4096) as u16;
        }
        let m = find_injective_modulus(&indices);
        assert!(is_prime(m));
        let mut seen = std::collections::HashSet::new();
        for &idx in &indices {
            assert!(seen.insert(idx as u32 % m as u32));
        }
    }

    #[test]
    fn injective_modulus_deterministic() {
        let mut indices = [0u16; TOP_K];
        for i in 0..TOP_K {
            indices[i] = (i * 17) as u16;
        }
        let m1 = find_injective_modulus(&indices);
        let m2 = find_injective_modulus(&indices);
        assert_eq!(m1, m2);
    }

    #[test]
    fn ndd_interpolate_roundtrip() {
        // Small test: use 128 points with known values
        let mut xs = [0u16; TOP_K];
        let mut ys = [0u16; TOP_K];
        let m: u16 = 65521; // prime modulus
        for i in 0..TOP_K {
            xs[i] = i as u16;
            ys[i] = ((i * i + 3 * i + 7) % m as usize) as u16;
        }
        let coeffs = ndd_interpolate(&xs, &ys, m);
        // Verify: evaluating at each x_i should return y_i
        for i in 0..TOP_K {
            let result = horner_evaluate(&coeffs, xs[i], m);
            assert_eq!(result, ys[i], "mismatch at x={}", xs[i]);
        }
    }

    #[test]
    fn horner_matches_direct() {
        // Simple polynomial: p(x) = 3 + 2x + x^2 mod 65521
        let m = 65521u16;
        let mut coeffs = [0u16; TOP_K];
        coeffs[0] = 3;
        coeffs[1] = 2;
        coeffs[2] = 1;
        // p(5) = 3 + 10 + 25 = 38
        assert_eq!(horner_evaluate(&coeffs, 5, m), 38);
        // p(0) = 3
        assert_eq!(horner_evaluate(&coeffs, 0, m), 3);
    }

    #[test]
    fn mantissa_extraction_correct() {
        // 1.0f32 = 0x3F800000 → exponent=127, mantissa=0
        assert_eq!(f32_mantissa_u16(1.0), 0);
        assert_eq!(f32_exponent(1.0), 127);
        // 1.5f32 = 0x3FC00000 → mantissa = 0x400000 → lower 16 = 0
        assert_eq!(f32_exponent(1.5), 127);
        // -2.0f32 → same exponent as 2.0 (128), mantissa = 0
        assert_eq!(f32_exponent(-2.0), 128);
        assert_eq!(f32_mantissa_u16(-2.0), 0);
    }

    #[test]
    fn extract_top_k_selects_highest() {
        let mut data = vec![0.0f32; 256];
        // Place known large values
        data[10] = 100.0;
        data[20] = -99.0;
        data[30] = 98.0;
        let (indices, _mantissas) = extract_top_k(&data, 3);
        let mut sorted_indices: Vec<u16> = indices;
        sorted_indices.sort();
        assert_eq!(sorted_indices, vec![10, 20, 30]);
    }
}
```

- [ ] **Step 2: Add module declaration**

In `crates/harmony-speculative/src/lib.rs`, add after the `pub mod prefill;` line:

```rust
#[cfg(feature = "prefill")]
pub(crate) mod toploc;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-speculative --features prefill -- toploc::tests`
Expected: PASS (7 tests)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-speculative/src/toploc.rs crates/harmony-speculative/src/lib.rs
git commit -m "feat(speculative): add TOPLOC integer math core

Extended GCD, modular inverse, injective modulus search, NDD interpolation
with Newton-to-monomial conversion, Horner evaluation, IEEE 754 mantissa
extraction, top-k selection. All pure integer arithmetic over Z_m."
```

---

### Task 2: TocProof Type, Sampling, Generate, Verify (toploc_proof.rs)

**Files:**
- Create: `crates/harmony-speculative/src/toploc_proof.rs`
- Modify: `crates/harmony-speculative/src/lib.rs` (module declaration)

- [ ] **Step 1: Create toploc_proof.rs with types + sampling + generate + verify**

Create `crates/harmony-speculative/src/toploc_proof.rs`:

```rust
//! TOPLOC proof generation and verification.
//!
//! Wraps the integer math core (toploc.rs) with candle tensor extraction
//! and deterministic sampling. Feature-gated behind `prefill`.

use candle_core::{DType, Tensor};
use harmony_crypto::hash::full_hash;
use serde::{Deserialize, Serialize};

use crate::prefill::{PrefillCacheHeader, PrefillError};
use crate::toploc::{self, TOP_K, CHUNK_SIZE};
use harmony_inference::InferenceCache;

/// A single TOPLOC proof: polynomial encoding of top-128 mantissas
/// from a KV tensor slice.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TocProof {
    /// Injective modulus for the finite field Z_m.
    pub modulus: u16,
    /// 128 polynomial coefficients in monomial form over Z_m.
    pub coefficients: [u16; TOP_K],
    /// Which layer was sampled.
    pub layer: u16,
    /// Which KV head was sampled.
    pub head: u16,
    /// Starting token index for this chunk.
    pub token_offset: u32,
}

/// Result of TOPLOC verification.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    pub proofs_checked: usize,
    pub proofs_passed: usize,
    pub details: Vec<ProofCheckDetail>,
}

impl VerifyResult {
    pub fn is_valid(&self) -> bool {
        self.proofs_checked > 0 && self.proofs_passed == self.proofs_checked
    }
}

#[derive(Debug, Clone)]
pub struct ProofCheckDetail {
    pub layer: u16,
    pub head: u16,
    pub token_offset: u32,
    pub agreement_rate: f32,
    pub mean_mantissa_diff: f32,
    pub median_mantissa_diff: f32,
    pub passed: bool,
}

// ── Thresholds ───────────────────────────────────────────────────────

const AGREEMENT_THRESHOLD: f32 = 0.5;
const MEAN_DIFF_THRESHOLD: f32 = 256.0;
const MEDIAN_DIFF_THRESHOLD: f32 = 128.0;

// ── Deterministic sampling ───────────────────────────────────────────

/// Derive (layer, head) for a given chunk index from the header hash.
fn sample_coordinates(
    header: &PrefillCacheHeader,
    chunk_idx: usize,
    num_layers: usize,
    num_kv_heads: usize,
) -> (usize, usize) {
    let mut seed_input = Vec::with_capacity(68);
    seed_input.extend_from_slice(&header.model_cid);
    seed_input.extend_from_slice(&header.token_hash);
    seed_input.extend_from_slice(&(chunk_idx as u32).to_le_bytes());
    let hash = full_hash(&seed_input);
    let layer = u16::from_le_bytes([hash[0], hash[1]]) as usize % num_layers;
    let head = u16::from_le_bytes([hash[2], hash[3]]) as usize % num_kv_heads;
    (layer, head)
}

// ── Tensor extraction helper ─────────────────────────────────────────

/// Extract a flattened f32 slice from a KV tensor at a specific head and token range.
/// Input shape: [1, num_kv_heads, seq_len, head_dim]
/// Output: Vec<f32> of length min(chunk_size, available) * head_dim
fn extract_kv_slice(
    tensor: &Tensor,
    head: usize,
    token_offset: usize,
    chunk_size: usize,
) -> Result<Vec<f32>, PrefillError> {
    let (_b, _h, seq_len, head_dim) = tensor.dims4()
        .map_err(|e| PrefillError::SerializationFailed(format!("tensor shape: {e}")))?;
    let end = (token_offset + chunk_size).min(seq_len);
    let len = end - token_offset;
    if len == 0 {
        return Err(PrefillError::SerializationFailed("empty token range".into()));
    }
    // Narrow: [1, head:head+1, token_offset:end, :]
    let slice = tensor
        .i((0, head, token_offset..end, ..))
        .and_then(|t| t.to_dtype(DType::F32))
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
    Ok(slice)
}

// ── Proof generation ─────────────────────────────────────────────────

/// Generate TOPLOC proofs from a full-precision (uncompressed) KV cache.
/// Must be called BEFORE compress().
pub fn generate_proofs(
    cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<Vec<TocProof>, PrefillError> {
    if cache.is_compressed() {
        return Err(PrefillError::SerializationFailed(
            "cache must not be compressed for proof generation".into(),
        ));
    }

    let num_layers = header.num_layers as usize;
    let num_kv_heads = header.num_kv_heads as usize;
    let seq_len = cache.position;
    let num_chunks = (seq_len + CHUNK_SIZE - 1) / CHUNK_SIZE;

    let mut proofs = Vec::with_capacity(num_chunks);

    for chunk_idx in 0..num_chunks {
        let (layer, head) = sample_coordinates(header, chunk_idx, num_layers, num_kv_heads);
        let token_offset = chunk_idx * CHUNK_SIZE;

        // Skip if layer not populated
        let (k_tensor, _v_tensor) = match &cache.layers[layer] {
            Some((k, v)) => (k, v),
            None => continue,
        };

        let flat = extract_kv_slice(k_tensor, head, token_offset, CHUNK_SIZE)?;

        if flat.len() < TOP_K {
            continue; // too few elements for a proof
        }

        let (raw_indices, mantissas) = toploc::extract_top_k(&flat, TOP_K);

        // Convert to fixed-size arrays
        let mut idx_arr = [0u16; TOP_K];
        let mut man_arr = [0u16; TOP_K];
        for i in 0..TOP_K {
            idx_arr[i] = raw_indices[i];
            man_arr[i] = mantissas[i];
        }

        let modulus = toploc::find_injective_modulus(&idx_arr);

        // Reduce indices mod m
        for idx in &mut idx_arr {
            *idx = (*idx as u32 % modulus as u32) as u16;
        }

        let coefficients = toploc::ndd_interpolate(&idx_arr, &man_arr, modulus);

        proofs.push(TocProof {
            modulus,
            coefficients,
            layer: layer as u16,
            head: head as u16,
            token_offset: token_offset as u32,
        });
    }

    Ok(proofs)
}

// ── Proof verification ───────────────────────────────────────────────

/// Verify TOPLOC proofs against a locally-computed KV cache.
pub fn verify_proofs(
    local_cache: &InferenceCache,
    header: &PrefillCacheHeader,
) -> Result<VerifyResult, PrefillError> {
    let proofs = &header.proofs;

    if proofs.is_empty() {
        return Ok(VerifyResult {
            proofs_checked: 0,
            proofs_passed: 0,
            details: vec![],
        });
    }

    let mut details = Vec::with_capacity(proofs.len());

    for proof in proofs {
        let layer = proof.layer as usize;
        let head = proof.head as usize;
        let token_offset = proof.token_offset as usize;

        // Extract local KV slice
        let (k_tensor, _) = match &local_cache.layers[layer] {
            Some((k, v)) => (k, v),
            None => {
                details.push(ProofCheckDetail {
                    layer: proof.layer,
                    head: proof.head,
                    token_offset: proof.token_offset,
                    agreement_rate: 0.0,
                    mean_mantissa_diff: f32::INFINITY,
                    median_mantissa_diff: f32::INFINITY,
                    passed: false,
                });
                continue;
            }
        };

        let flat = extract_kv_slice(k_tensor, head, token_offset, CHUNK_SIZE)?;

        if flat.len() < TOP_K {
            details.push(ProofCheckDetail {
                layer: proof.layer,
                head: proof.head,
                token_offset: proof.token_offset,
                agreement_rate: 0.0,
                mean_mantissa_diff: f32::INFINITY,
                median_mantissa_diff: f32::INFINITY,
                passed: false,
            });
            continue;
        }

        let (local_indices, local_mantissas) = toploc::extract_top_k(&flat, TOP_K);

        // Evaluate proof polynomial at each local index
        let mut diffs: Vec<f32> = Vec::new();
        for i in 0..TOP_K {
            let x = (local_indices[i] as u32 % proof.modulus as u32) as u16;
            let claimed = toploc::horner_evaluate(&proof.coefficients, x, proof.modulus);
            let local = local_mantissas[i];
            let diff = (claimed as i32 - local as i32).unsigned_abs() as f32;
            diffs.push(diff);
        }

        // Agreement: indices where diff is within mantissa threshold
        let agreeing = diffs.iter().filter(|&&d| d <= MEAN_DIFF_THRESHOLD).count();
        let agreement_rate = agreeing as f32 / TOP_K as f32;

        // Stats on agreeing diffs only
        let mut agreeing_diffs: Vec<f32> = diffs.iter()
            .copied()
            .filter(|&d| d <= MEAN_DIFF_THRESHOLD)
            .collect();

        let (mean_diff, median_diff) = if agreeing_diffs.is_empty() {
            (f32::INFINITY, f32::INFINITY)
        } else {
            let sum: f32 = agreeing_diffs.iter().sum();
            let mean = sum / agreeing_diffs.len() as f32;
            agreeing_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median = agreeing_diffs[agreeing_diffs.len() / 2];
            (mean, median)
        };

        let passed = agreement_rate >= AGREEMENT_THRESHOLD
            && mean_diff <= MEAN_DIFF_THRESHOLD
            && median_diff <= MEDIAN_DIFF_THRESHOLD;

        details.push(ProofCheckDetail {
            layer: proof.layer,
            head: proof.head,
            token_offset: proof.token_offset,
            agreement_rate,
            mean_mantissa_diff: mean_diff,
            median_mantissa_diff: median_diff,
            passed,
        });
    }

    let proofs_passed = details.iter().filter(|d| d.passed).count();

    Ok(VerifyResult {
        proofs_checked: details.len(),
        proofs_passed,
        details,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sampling_deterministic() {
        let header = PrefillCacheHeader {
            magic: *b"HKV\x02",
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 64,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
            proofs: vec![],
        };
        let (l1, h1) = sample_coordinates(&header, 0, 28, 8);
        let (l2, h2) = sample_coordinates(&header, 0, 28, 8);
        assert_eq!((l1, h1), (l2, h2));
    }

    #[test]
    fn sampling_varies_across_chunks() {
        let header = PrefillCacheHeader {
            magic: *b"HKV\x02",
            model_cid: [0xAB; 32],
            token_hash: [0xCD; 32],
            token_count: 128,
            num_layers: 28,
            num_kv_heads: 8,
            head_dim: 128,
            quant_bits: 3,
            proofs: vec![],
        };
        let coords: Vec<_> = (0..4)
            .map(|i| sample_coordinates(&header, i, 28, 8))
            .collect();
        // At least 2 different (layer, head) pairs across 4 chunks
        let unique: std::collections::HashSet<_> = coords.into_iter().collect();
        assert!(unique.len() >= 2, "sampling should vary across chunks");
    }

    #[test]
    fn proof_serde_roundtrip() {
        let proof = TocProof {
            modulus: 65521,
            coefficients: [42u16; TOP_K],
            layer: 14,
            head: 3,
            token_offset: 64,
        };
        let bytes = postcard::to_allocvec(&proof).unwrap();
        let restored: TocProof = postcard::from_bytes(&bytes).unwrap();
        assert_eq!(proof, restored);
    }
}
```

- [ ] **Step 2: Add module declaration**

In `crates/harmony-speculative/src/lib.rs`, add:

```rust
#[cfg(feature = "prefill")]
pub mod toploc_proof;
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-speculative --features prefill -- toploc_proof::tests`
Expected: PASS (3 tests)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-speculative/src/toploc_proof.rs crates/harmony-speculative/src/lib.rs
git commit -m "feat(speculative): add TocProof type with generate/verify functions

TocProof struct, deterministic sampling via SHA-256, proof generation
from uncompressed KV cache, verification via Horner polynomial evaluation
with agreement rate + mantissa diff thresholds."
```

---

### Task 3: PrefillCacheHeader Changes (magic v2, proofs field)

**Files:**
- Modify: `crates/harmony-speculative/src/prefill.rs`

- [ ] **Step 1: Add proofs field, magic v2, and store_with_proofs**

In `crates/harmony-speculative/src/prefill.rs`:

Add magic v2 constant:
```rust
/// Magic for proof-bearing blobs (version 2).
pub const PREFILL_MAGIC_V2: [u8; 4] = *b"HKV\x02";
```

Add `proofs` field to `PrefillCacheHeader`:
```rust
pub struct PrefillCacheHeader {
    // ... all existing fields ...

    /// TOPLOC proofs for KV cache integrity verification.
    /// Empty for HKV\x01 blobs (no proofs).
    pub proofs: Vec<crate::toploc_proof::TocProof>,
}
```

Add `store_prefill_cache_with_proofs`:
```rust
/// Store a compressed KV cache with TOPLOC proofs in CAS.
/// Emits HKV\x02 format with proofs embedded in the header.
pub fn store_prefill_cache_with_proofs(
    cache: &InferenceCache,
    model_cid: &ContentId,
    token_ids: &[u32],
    proofs: Vec<crate::toploc_proof::TocProof>,
    store: &mut dyn BookStore,
) -> Result<ContentId, PrefillError> {
    // Same validation as store_prefill_cache
    if !cache.is_compressed() {
        return Err(PrefillError::CacheNotCompressed);
    }
    if token_ids.len() != cache.position {
        return Err(PrefillError::SerializationFailed(format!(
            "token_ids length {} does not match cache position {}",
            token_ids.len(), cache.position
        )));
    }

    let header = PrefillCacheHeader {
        magic: PREFILL_MAGIC_V2,
        model_cid: model_cid.to_bytes(),
        token_hash: token_hash(token_ids),
        token_count: u32::try_from(cache.position)
            .map_err(|_| PrefillError::SerializationFailed(format!("position {} exceeds u32", cache.position)))?,
        num_layers: u16::try_from(cache.num_layers)
            .map_err(|_| PrefillError::SerializationFailed(format!("num_layers {} exceeds u16", cache.num_layers)))?,
        num_kv_heads: u16::try_from(cache.num_kv_heads)
            .map_err(|_| PrefillError::SerializationFailed(format!("num_kv_heads {} exceeds u16", cache.num_kv_heads)))?,
        head_dim: u16::try_from(cache.head_dim)
            .map_err(|_| PrefillError::SerializationFailed(format!("head_dim {} exceeds u16", cache.head_dim)))?,
        quant_bits: SUPPORTED_QUANT_BITS,
        proofs,
    };

    let header_bytes = postcard::to_allocvec(&header)
        .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
    let payload_bytes = cache.serialize_compressed()?;

    let mut blob = Vec::with_capacity(4 + header_bytes.len() + payload_bytes.len());
    blob.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    blob.extend_from_slice(&header_bytes);
    blob.extend_from_slice(&payload_bytes);

    let root = dag::ingest(&blob, &ChunkerConfig::DEFAULT, store)?;
    Ok(root)
}
```

Update existing `store_prefill_cache` to populate `proofs: vec![]` on the header.

Update `load_prefill_cache` to handle v1/v2 deserialization. The key issue: v1 blobs were serialized WITHOUT the `proofs` field, so `postcard::from_bytes` into the new struct would fail. Solution: peek at the magic from the raw header bytes, then deserialize to the appropriate struct.

Define a v1 struct (private, for deserialization only):
```rust
/// V1 header (no proofs). Used only for deserializing HKV\x01 blobs.
#[derive(Deserialize)]
struct PrefillCacheHeaderV1 {
    magic: [u8; 4],
    model_cid: [u8; 32],
    token_hash: [u8; 32],
    token_count: u32,
    num_layers: u16,
    num_kv_heads: u16,
    head_dim: u16,
    quant_bits: u8,
}
```

Then in `load_prefill_cache`, after reading `header_len` and `header_end`:
```rust
    // Peek at magic to determine version (first 4 bytes of header)
    let header_bytes = &blob[4..header_end];
    if header_bytes.len() < 4 {
        return Err(PrefillError::SerializationFailed("header too short for magic".into()));
    }
    let magic: [u8; 4] = header_bytes[0..4].try_into().unwrap();

    let header = if magic == PREFILL_MAGIC {
        // V1: deserialize without proofs, then convert
        let v1: PrefillCacheHeaderV1 = postcard::from_bytes(header_bytes)
            .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?;
        PrefillCacheHeader {
            magic: v1.magic,
            model_cid: v1.model_cid,
            token_hash: v1.token_hash,
            token_count: v1.token_count,
            num_layers: v1.num_layers,
            num_kv_heads: v1.num_kv_heads,
            head_dim: v1.head_dim,
            quant_bits: v1.quant_bits,
            proofs: vec![],
        }
    } else if magic == PREFILL_MAGIC_V2 {
        // V2: deserialize with proofs
        postcard::from_bytes(header_bytes)
            .map_err(|e| PrefillError::SerializationFailed(e.to_string()))?
    } else {
        return Err(PrefillError::InvalidMagic);
    };
```

Remove the old `postcard::from_bytes` + magic check that followed.

- [ ] **Step 2: Fix existing tests**

The existing tests construct `PrefillCacheHeader` without the `proofs` field. Add `proofs: vec![]` to all existing test header constructions. Also update the `header_serde_roundtrip` and `header_magic_validation` tests in both `prefill::tests` and `toploc_proof::tests`.

- [ ] **Step 3: Add new tests**

Add to `prefill::tests`:

```rust
    #[test]
    fn load_v1_blob_has_empty_proofs() {
        // store_prefill_cache emits HKV\x01 with empty proofs
        let cache = compressed_cache(2, 8, 128, 16);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..16).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache(&cache, &model_cid, &token_ids, &mut store).unwrap();
        let (_, header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert!(header.proofs.is_empty());
        assert_eq!(header.magic, PREFILL_MAGIC); // v1
    }

    #[test]
    fn store_with_proofs_uses_v2_magic() {
        let cache = compressed_cache(2, 8, 128, 16);
        let model_cid = fake_model_cid();
        let token_ids: Vec<u32> = (0..16).collect();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache_with_proofs(
            &cache, &model_cid, &token_ids, vec![], &mut store
        ).unwrap();
        let (_, header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert_eq!(header.magic, PREFILL_MAGIC_V2);
    }
```

- [ ] **Step 4: Run all tests**

Run: `cargo test -p harmony-speculative --features prefill`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-speculative/src/prefill.rs
git commit -m "feat(speculative): add proofs to PrefillCacheHeader with magic v2

PrefillCacheHeader gains proofs: Vec<TocProof> field. store_prefill_cache
emits HKV\x01 (no proofs), store_prefill_cache_with_proofs emits HKV\x02.
load_prefill_cache accepts both versions."
```

---

### Task 4: Integration Tests (generate → store → load → verify)

**Files:**
- Modify: `crates/harmony-speculative/src/toploc_proof.rs` (add integration tests)

- [ ] **Step 1: Add integration tests**

Add to `toploc_proof::tests`:

```rust
    use candle_core::{DType, Device, Tensor};
    use harmony_content::book::MemoryBookStore;
    use harmony_content::cid::ContentId;
    use crate::prefill::{
        PREFILL_MAGIC_V2, store_prefill_cache_with_proofs, load_prefill_cache, token_hash,
    };

    fn make_cache_with_data(num_layers: usize, num_kv_heads: usize, head_dim: usize, n_tokens: usize) -> InferenceCache {
        let mut cache = InferenceCache::new(num_layers, head_dim, num_kv_heads);
        if n_tokens > 0 {
            let shape = (1, num_kv_heads, n_tokens, head_dim);
            let k = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            let v = Tensor::rand(0f32, 1f32, shape, &Device::Cpu)
                .unwrap().to_dtype(DType::F16).unwrap();
            cache.layers[0] = Some((k, v));
            cache.position = n_tokens;
        }
        cache
    }

    fn make_header(cache: &InferenceCache, token_ids: &[u32]) -> PrefillCacheHeader {
        PrefillCacheHeader {
            magic: PREFILL_MAGIC_V2,
            model_cid: [0xAB; 32],
            token_hash: token_hash(token_ids),
            token_count: cache.position as u32,
            num_layers: cache.num_layers as u16,
            num_kv_heads: cache.num_kv_heads as u16,
            head_dim: cache.head_dim as u16,
            quant_bits: 3,
            proofs: vec![],
        }
    }

    #[test]
    fn generate_verify_roundtrip() {
        // Need enough tokens for at least one 32-token chunk with 4096 elements
        // 32 tokens * 128 head_dim = 4096 >= TOP_K (128)
        let cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        assert!(!proofs.is_empty(), "should generate at least one proof");

        // Verify against the same cache
        let mut header_with_proofs = header;
        header_with_proofs.proofs = proofs;

        let result = verify_proofs(&cache, &header_with_proofs).unwrap();
        assert!(result.is_valid(), "verification should pass for identical cache: {:?}", result.details);
    }

    #[test]
    fn verify_rejects_tampered_cache() {
        let cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        assert!(!proofs.is_empty());

        // Create a different cache (different random data)
        let tampered = make_cache_with_data(2, 8, 128, 32);

        let mut header_with_proofs = header;
        header_with_proofs.proofs = proofs;

        let result = verify_proofs(&tampered, &header_with_proofs).unwrap();
        assert!(!result.is_valid(), "verification should fail for tampered cache");
    }

    #[test]
    fn proofs_in_header_roundtrip() {
        let mut cache = make_cache_with_data(2, 8, 128, 32);
        let token_ids: Vec<u32> = (0..32).collect();
        let header = make_header(&cache, &token_ids);

        let proofs = generate_proofs(&cache, &header).unwrap();
        let proof_count = proofs.len();

        // Compress and store with proofs
        cache.compress().unwrap();
        let model_cid = ContentId::for_book(b"test-model", Default::default()).unwrap();
        let mut store = MemoryBookStore::new();

        let root = store_prefill_cache_with_proofs(
            &cache, &model_cid, &token_ids, proofs, &mut store
        ).unwrap();

        // Load and check proofs survived
        let (_, loaded_header) = load_prefill_cache(&root, &model_cid, &store).unwrap();
        assert_eq!(loaded_header.magic, PREFILL_MAGIC_V2);
        assert_eq!(loaded_header.proofs.len(), proof_count);
    }
```

- [ ] **Step 2: Run all tests**

Run: `cargo test -p harmony-speculative --features prefill`
Expected: ALL PASS

- [ ] **Step 3: Run clippy**

Run: `cargo clippy -p harmony-speculative --features prefill -- -D warnings`
Expected: Clean (ignoring pre-existing harmony-content identity_op)

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-speculative/src/toploc_proof.rs
git commit -m "test(speculative): add TOPLOC integration tests

generate_verify_roundtrip, verify_rejects_tampered_cache,
proofs_in_header_roundtrip (full CAS store→load→verify pipeline)."
```

---

## Test Matrix Summary

| Test | Module | What | Task |
|------|--------|------|------|
| `extended_gcd_known_vectors` | toploc | Modular inverse correctness | 1 |
| `mod_inverse_all_coprime_to_prime` | toploc | Inverse for multiple values mod prime | 1 |
| `injective_modulus_finds_valid` | toploc | All indices unique mod m | 1 |
| `injective_modulus_deterministic` | toploc | Same input → same m | 1 |
| `ndd_interpolate_roundtrip` | toploc | Interpolate then evaluate → original values | 1 |
| `horner_matches_direct` | toploc | Horner matches naive calculation | 1 |
| `mantissa_extraction_correct` | toploc | IEEE 754 bit extraction | 1 |
| `extract_top_k_selects_highest` | toploc | Correct top-k by magnitude | 1 |
| `sampling_deterministic` | toploc_proof | Same header → same coordinates | 2 |
| `sampling_varies_across_chunks` | toploc_proof | Different chunks → different coordinates | 2 |
| `proof_serde_roundtrip` | toploc_proof | TocProof postcard roundtrip | 2 |
| `load_v1_blob_has_empty_proofs` | prefill | HKV\x01 → empty proofs | 3 |
| `store_with_proofs_uses_v2_magic` | prefill | store_with_proofs → HKV\x02 | 3 |
| `generate_verify_roundtrip` | toploc_proof | Same cache → verification passes | 4 |
| `verify_rejects_tampered_cache` | toploc_proof | Different cache → verification fails | 4 |
| `proofs_in_header_roundtrip` | toploc_proof | Proofs survive CAS store→load | 4 |

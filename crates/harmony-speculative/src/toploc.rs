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

/// Starting modulus for the injective search — largest prime that fits in u16.
/// Must be prime so that all nonzero differences have modular inverses in NDD.
const START_MODULUS: u16 = 65521;

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

/// Simple trial-division primality test.
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

/// Find a prime m such that all indices are unique mod m.
/// Starts at 65521 (largest prime ≤ u16::MAX) and searches downward
/// through primes only. Primality ensures all nonzero elements have
/// modular inverses, which NDD interpolation requires.
pub(crate) fn find_injective_modulus(indices: &[u16; TOP_K]) -> u16 {
    let mut seen = vec![false; 65536];
    let mut m = START_MODULUS;
    loop {
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
        m -= if m % 2 == 0 { 1 } else { 2 };
        while !is_prime(m) {
            m -= if m % 2 == 0 { 1 } else { 2 };
        }
        assert!(m >= TOP_K as u16, "no injective prime modulus found");
    }
}

/// Newton Divided Differences interpolation over Z_m, converted to monomial form.
///
/// Two phases:
/// 1. NDD: build divided difference table to get Newton basis coefficients
/// 2. Convert Newton basis to monomial (power) basis via synthetic expansion
pub(crate) fn ndd_interpolate(
    xs: &[u16; TOP_K],
    ys: &[u16; TOP_K],
    m: u16,
) -> [u16; TOP_K] {
    let m32 = m as u32;
    let k = TOP_K;

    // Phase 1: Newton divided differences
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
    // Newton: p(x) = ndd[0] + ndd[1](x-x0) + ndd[2](x-x0)(x-x1) + ...
    // Build up from highest degree: start with ndd[k-1], multiply by (x - xs[i]), add ndd[i]
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
/// Reserved for future exponent intersection verification.
#[inline]
#[allow(dead_code)]
pub(crate) fn f32_exponent(value: f32) -> u8 {
    ((value.to_bits() >> 23) & 0xFF) as u8
}

/// Find the top-k elements by absolute magnitude in an f32 slice.
/// Returns (indices, mantissa_values).
/// Panics if data.len() < k.
pub(crate) fn extract_top_k(data: &[f32], k: usize) -> (Vec<u16>, Vec<u16>) {
    assert!(data.len() >= k, "data length {} < k {}", data.len(), k);

    let mut indexed: Vec<(f32, usize)> = data.iter()
        .enumerate()
        .map(|(i, &v)| (v.abs(), i))
        .collect();

    // Stable sort with index tie-breaking for cross-platform reproducibility.
    // NaN values sort to the end (treated as smallest).
    indexed.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });

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
        let inv = mod_inverse(3, 7);
        assert_eq!((3u32 * inv as u32) % 7, 1);
        let inv = mod_inverse(5, 13);
        assert_eq!((5u32 * inv as u32) % 13, 1);
    }

    #[test]
    fn mod_inverse_all_coprime_to_prime() {
        let m = 65521u16;
        for a in [1u16, 2, 127, 1000, 65520] {
            let inv = mod_inverse(a, m);
            assert_eq!((a as u32 * inv as u32) % m as u32, 1, "inverse of {a} mod {m}");
        }
    }

    #[test]
    fn injective_modulus_finds_valid_prime() {
        let mut indices = [0u16; TOP_K];
        for i in 0..TOP_K {
            indices[i] = (i * 31 + 7) as u16;
        }
        let m = find_injective_modulus(&indices);
        assert!(is_prime(m), "modulus {m} should be prime");
        let mut seen = std::collections::HashSet::new();
        for &idx in &indices {
            assert!(seen.insert(idx as u32 % m as u32), "collision at {idx} mod {m}");
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
    fn injective_modulus_with_scattered_indices() {
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
    fn ndd_interpolate_roundtrip() {
        let mut xs = [0u16; TOP_K];
        let mut ys = [0u16; TOP_K];
        let m: u16 = 65521;
        for i in 0..TOP_K {
            xs[i] = i as u16;
            ys[i] = ((i * i + 3 * i + 7) % m as usize) as u16;
        }
        let coeffs = ndd_interpolate(&xs, &ys, m);
        for i in 0..TOP_K {
            let result = horner_evaluate(&coeffs, xs[i], m);
            assert_eq!(result, ys[i], "mismatch at x={}", xs[i]);
        }
    }

    #[test]
    fn horner_matches_direct() {
        let m = 65521u16;
        let mut coeffs = [0u16; TOP_K];
        coeffs[0] = 3;
        coeffs[1] = 2;
        coeffs[2] = 1;
        assert_eq!(horner_evaluate(&coeffs, 5, m), 38);
        assert_eq!(horner_evaluate(&coeffs, 0, m), 3);
    }

    #[test]
    fn mantissa_extraction_correct() {
        assert_eq!(f32_mantissa_u16(1.0), 0);
        assert_eq!(f32_exponent(1.0), 127);
        assert_eq!(f32_exponent(-2.0), 128);
        assert_eq!(f32_mantissa_u16(-2.0), 0);
    }

    #[test]
    fn extract_top_k_selects_highest() {
        let mut data = vec![0.0f32; 256];
        data[10] = 100.0;
        data[20] = -99.0;
        data[30] = 98.0;
        let (indices, _mantissas) = extract_top_k(&data, 3);
        let mut sorted_indices: Vec<u16> = indices;
        sorted_indices.sort();
        assert_eq!(sorted_indices, vec![10, 20, 30]);
    }
}

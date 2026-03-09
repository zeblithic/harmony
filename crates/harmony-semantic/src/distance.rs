// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Hamming distance — XOR + POPCNT over binary vectors.

/// Compute the Hamming distance between two binary vectors of equal length.
///
/// This is the number of bit positions where the two vectors differ.
/// Implemented as XOR each byte pair, then count set bits (POPCNT).
/// The compiler auto-vectorizes this for SSE/AVX when available.
pub fn hamming_distance(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must be equal length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum()
}

/// Compute the Hamming distance using u64 chunks for better throughput.
///
/// Processes 8 bytes at a time. Falls back to byte-level for the remainder.
pub fn hamming_distance_fast(a: &[u8], b: &[u8]) -> u32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must be equal length");
    let chunks = a.len() / 8;
    let mut dist = 0u32;

    for i in 0..chunks {
        let offset = i * 8;
        let va = u64::from_le_bytes(a[offset..offset + 8].try_into().unwrap());
        let vb = u64::from_le_bytes(b[offset..offset + 8].try_into().unwrap());
        dist += (va ^ vb).count_ones();
    }

    // Handle remaining bytes.
    let remainder = chunks * 8;
    for i in remainder..a.len() {
        dist += (a[i] ^ b[i]).count_ones();
    }

    dist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hamming_distance_identical_is_zero() {
        let a = [0xAB, 0xCD, 0xEF, 0x01];
        assert_eq!(hamming_distance(&a, &a), 0);
    }

    #[test]
    fn hamming_distance_all_different() {
        // 0x00 vs 0xFF: every bit differs → 8 bits per byte.
        let a = [0x00u8; 4];
        let b = [0xFFu8; 4];
        assert_eq!(hamming_distance(&a, &b), 32);
    }

    #[test]
    fn hamming_distance_known_vector() {
        // 0b00001111 vs 0b11110000 = 8 differing bits.
        // 0b10101010 vs 0b01010101 = 8 differing bits.
        // Total = 16.
        let a = [0x0F, 0xAA];
        let b = [0xF0, 0x55];
        assert_eq!(hamming_distance(&a, &b), 16);
    }

    #[test]
    fn hamming_distance_fast_matches_simple() {
        // Test with various sizes — including exact multiples of 8 and remainders.
        let cases: &[usize] = &[0, 1, 7, 8, 9, 15, 16, 31, 32, 33];
        for &len in cases {
            let a: alloc::vec::Vec<u8> = (0..len).map(|i| (i * 37) as u8).collect();
            let b: alloc::vec::Vec<u8> = (0..len).map(|i| (i * 53 + 7) as u8).collect();
            assert_eq!(
                hamming_distance(&a, &b),
                hamming_distance_fast(&a, &b),
                "mismatch for len={len}"
            );
        }
    }

    #[test]
    fn hamming_distance_tier3_scale() {
        // 32-byte vectors (256-bit, realistic tier 3 size).
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        // Set up known pattern: only first byte differs.
        a[0] = 0xFF;
        b[0] = 0x00;
        // Rest identical.
        for i in 1..32 {
            a[i] = 0xAA;
            b[i] = 0xAA;
        }
        assert_eq!(hamming_distance(&a, &b), 8);
        assert_eq!(hamming_distance_fast(&a, &b), 8);
    }
}

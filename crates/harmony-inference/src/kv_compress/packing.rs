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
        assert_eq!(pack_1bit(&vec![false; 80]).len(), 10);
        assert_eq!(pack_1bit(&vec![false; 81]).len(), 11);
    }

    #[test]
    fn pack_1bit_non_multiple_of_8() {
        let values = vec![true, false, true, false, true]; // 5 bits
        let packed = pack_1bit(&values);
        assert_eq!(packed.len(), 1);
        let unpacked = unpack_1bit(&packed, 5);
        assert_eq!(values, unpacked);
    }
}

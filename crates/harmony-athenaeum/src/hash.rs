// SPDX-License-Identifier: Apache-2.0 OR MIT
//! Hash functions for address derivation.

use sha2::{Digest, Sha224, Sha256};

use crate::addr::Algorithm;

/// Compute full SHA-256 hash (32 bytes).
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Compute full SHA-224 hash (28 bytes).
pub fn sha224_hash(data: &[u8]) -> [u8; 28] {
    let mut hasher = Sha224::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Derive 28 hash address bits from page data using the specified algorithm.
///
/// Hashes `data` with the algorithm's hash function, takes a 4-byte window
/// from the MSB or LSB end of the digest, interprets it as a big-endian u32,
/// and masks to the lower 28 bits.
pub(crate) fn derive_hash_bits(data: &[u8], algorithm: Algorithm) -> u32 {
    match algorithm {
        Algorithm::Sha256Msb => {
            let h = sha256_hash(data);
            u32::from_be_bytes([h[0], h[1], h[2], h[3]]) & 0x0FFF_FFFF
        }
        Algorithm::Sha256Lsb => {
            let h = sha256_hash(data);
            u32::from_be_bytes([h[28], h[29], h[30], h[31]]) & 0x0FFF_FFFF
        }
        Algorithm::Sha224Msb => {
            let h = sha224_hash(data);
            u32::from_be_bytes([h[0], h[1], h[2], h[3]]) & 0x0FFF_FFFF
        }
        Algorithm::Sha224Lsb => {
            let h = sha224_hash(data);
            u32::from_be_bytes([h[24], h[25], h[26], h[27]]) & 0x0FFF_FFFF
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::Algorithm;

    #[test]
    fn sha256_msb_deterministic() {
        let bits1 = derive_hash_bits(b"hello", Algorithm::Sha256Msb);
        let bits2 = derive_hash_bits(b"hello", Algorithm::Sha256Msb);
        assert_eq!(bits1, bits2);
    }

    #[test]
    fn sha256_msb_vs_lsb_differ() {
        let msb = derive_hash_bits(b"test data", Algorithm::Sha256Msb);
        let lsb = derive_hash_bits(b"test data", Algorithm::Sha256Lsb);
        assert_ne!(msb, lsb);
    }

    #[test]
    fn sha224_msb_vs_sha256_msb_differ() {
        let sha256 = derive_hash_bits(b"test data", Algorithm::Sha256Msb);
        let sha224 = derive_hash_bits(b"test data", Algorithm::Sha224Msb);
        assert_ne!(sha256, sha224);
    }

    #[test]
    fn hash_bits_fit_in_28_bits() {
        for algo in Algorithm::ALL {
            let bits = derive_hash_bits(b"anything", algo);
            assert!(
                bits <= 0x0FFF_FFFF,
                "hash bits {:#x} exceed 28-bit max for {:?}",
                bits,
                algo,
            );
        }
    }

    #[test]
    fn different_data_different_bits() {
        let a = derive_hash_bits(b"alice", Algorithm::Sha256Msb);
        let b = derive_hash_bits(b"bob", Algorithm::Sha256Msb);
        assert_ne!(a, b);
    }

    #[test]
    fn full_hash_sha256_known_vector() {
        // SHA-256("") = e3b0c442...
        let hash = sha256_hash(b"");
        assert_eq!(hash[0], 0xe3);
        assert_eq!(hash[1], 0xb0);
        assert_eq!(hash[2], 0xc4);
        assert_eq!(hash[3], 0x42);
    }

    #[test]
    fn full_hash_sha224_known_vector() {
        // SHA-224("") = d14a028c...
        let hash = sha224_hash(b"");
        assert_eq!(hash[0], 0xd1);
        assert_eq!(hash[1], 0x4a);
        assert_eq!(hash[2], 0x02);
        assert_eq!(hash[3], 0x8c);
    }

    #[test]
    fn all_four_algorithms_produce_values() {
        let data = b"test page data for athenaeum";
        for algo in Algorithm::ALL {
            let bits = derive_hash_bits(data, algo);
            assert!(bits > 0, "{:?} produced zero for non-trivial data", algo);
        }
    }
}

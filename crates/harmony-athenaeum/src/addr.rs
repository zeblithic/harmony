// SPDX-License-Identifier: Apache-2.0 OR MIT
//! PageAddr — 32-bit content-addressed page identifier.
//!
//! ## 32-bit Layout
//!
//! ```text
//! ┌──────────┬────────────────────────────────┬──────────┐
//! │ algo (2) │         hash_bits (28)         │ cksum(2) │
//! │ bits 31-30│        bits 29-2              │ bits 1-0 │
//! └──────────┴────────────────────────────────┴──────────┘
//! ```
//!
//! Pages are always 4KB. Depth and size_exponent are gone.

/// Fixed page size in bytes.
pub const PAGE_SIZE: usize = 4096;

/// Number of pages per book.
pub const PAGES_PER_BOOK: usize = 256;

/// Maximum book size in bytes (1 MB).
pub const BOOK_MAX_SIZE: usize = PAGE_SIZE * PAGES_PER_BOOK;

/// Number of hash algorithm variants.
pub const ALGO_COUNT: usize = 4;

/// Sentinel value for an empty/null page slot.
///
/// Bits 31-2 are all zero, so XOR-fold produces `00`, but the checksum
/// field is `11`. This deliberately fails checksum validation.
pub const NULL_PAGE: u32 = 0x00000003;

/// Hash algorithm selector for address derivation.
///
/// The 2-bit algorithm field selects both the hash function and which
/// end of the digest to use as the address window.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Algorithm {
    /// SHA-256, first 4 bytes of digest.
    Sha256Msb = 0,
    /// SHA-256, last 4 bytes of digest.
    Sha256Lsb = 1,
    /// SHA-224, first 4 bytes of digest.
    Sha224Msb = 2,
    /// SHA-224, last 4 bytes of digest.
    Sha224Lsb = 3,
}

impl Algorithm {
    /// All algorithm variants in selector order.
    pub const ALL: [Algorithm; ALGO_COUNT] = [
        Algorithm::Sha256Msb,
        Algorithm::Sha256Lsb,
        Algorithm::Sha224Msb,
        Algorithm::Sha224Lsb,
    ];
}

/// A 32-bit content-addressed page identifier.
///
/// Layout: `[algo:2][hash_bits:28][checksum:2]`
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageAddr(pub(crate) u32);

impl PageAddr {
    /// Construct a `PageAddr` from hash bits and an algorithm selector.
    ///
    /// The 2-bit XOR-fold checksum is computed automatically.
    /// `hash_bits` must fit in 28 bits (i.e., `hash_bits <= 0x0FFF_FFFF`).
    pub fn new(hash_bits: u32, algorithm: Algorithm) -> Self {
        debug_assert!(hash_bits <= 0x0FFF_FFFF, "hash_bits must fit in 28 bits");
        let algo = (algorithm as u32) & 0x3;
        // Pack: algo in bits 31-30, hash_bits in bits 29-2
        let upper = (algo << 30) | ((hash_bits & 0x0FFF_FFFF) << 2);
        let checksum = Self::compute_checksum(upper);
        PageAddr(upper | checksum as u32)
    }

    /// Derive a `PageAddr` from page data using the specified algorithm.
    pub fn from_data(data: &[u8], algorithm: Algorithm) -> Self {
        let hash_bits = crate::hash::derive_hash_bits(data, algorithm);
        Self::new(hash_bits, algorithm)
    }

    /// Extract the 2-bit algorithm selector.
    pub fn algorithm(&self) -> Algorithm {
        match (self.0 >> 30) & 0x3 {
            0 => Algorithm::Sha256Msb,
            1 => Algorithm::Sha256Lsb,
            2 => Algorithm::Sha224Msb,
            3 => Algorithm::Sha224Lsb,
            _ => unreachable!(),
        }
    }

    /// Extract the 28-bit hash address (bits 29-2).
    pub fn hash_bits(&self) -> u32 {
        (self.0 >> 2) & 0x0FFF_FFFF
    }

    /// Extract the 2-bit checksum (bits 1-0).
    pub fn checksum(&self) -> u8 {
        (self.0 & 0x3) as u8
    }

    /// Verify that the checksum matches the upper 30 bits.
    pub fn verify_checksum(&self) -> bool {
        let upper = self.0 & !0x3; // bits 31-2, with bits 1-0 zeroed
        let expected = Self::compute_checksum(upper);
        self.checksum() == expected
    }

    /// Verify that `data` matches this address by re-deriving the hash bits.
    pub fn verify_data(&self, data: &[u8]) -> bool {
        let expected = crate::hash::derive_hash_bits(data, self.algorithm());
        expected == self.hash_bits()
    }

    /// Compute 2-bit XOR-fold checksum of the upper 30 bits (bits 31-2).
    ///
    /// Takes the 30 bits, splits into 15 pairs, XORs all pairs together
    /// to produce a 2-bit result.
    fn compute_checksum(upper: u32) -> u8 {
        // `upper` has the meaningful bits in positions 31-2.
        // Shift right by 2 to get 30 contiguous bits in positions 29-0.
        let bits30 = upper >> 2;
        let mut result = 0u8;
        for i in 0..15 {
            result ^= ((bits30 >> (i * 2)) & 0x3) as u8;
        }
        result & 0x3
    }
}

impl core::fmt::Debug for PageAddr {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "PageAddr({:#010x}, {:?})", self.0, self.algorithm(),)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(PAGES_PER_BOOK, 256);
        assert_eq!(BOOK_MAX_SIZE, 4096 * 256);
        assert_eq!(BOOK_MAX_SIZE, 1024 * 1024); // 1 MB
        assert_eq!(ALGO_COUNT, 4);
        assert_eq!(NULL_PAGE, 0x00000003);
    }

    #[test]
    fn null_page_fails_checksum() {
        let null = PageAddr(NULL_PAGE);
        assert!(!null.verify_checksum(), "NULL_PAGE must fail checksum");
        // Bits 31-2 are all zero, so XOR-fold = 00, but checksum field = 11
        assert_eq!(null.checksum(), 0b11);
    }

    #[test]
    fn round_trip_all_fields() {
        let addr = PageAddr::new(0x0ABC_DEF0, Algorithm::Sha256Msb);
        assert_eq!(addr.hash_bits(), 0x0ABC_DEF0);
        assert_eq!(addr.algorithm(), Algorithm::Sha256Msb);
        assert!(addr.verify_checksum());
    }

    #[test]
    fn algorithm_variants() {
        for algo in Algorithm::ALL {
            let addr = PageAddr::new(42, algo);
            assert_eq!(addr.algorithm(), algo);
            assert!(addr.verify_checksum());
        }
    }

    #[test]
    fn checksum_detects_single_bit_flip() {
        // XOR-fold checksum guarantees 100% detection of single-bit flips
        // in bits 2-29: each flip toggles exactly one pair in the fold,
        // changing the computed checksum while stored bits remain unchanged.
        let addr = PageAddr::new(0x0123_4567, Algorithm::Sha224Lsb);
        assert!(addr.verify_checksum());

        let mut detected = 0u32;
        let total_bits = 28; // bits 2 through 29
        for bit in 2..30 {
            let corrupted = PageAddr(addr.0 ^ (1 << bit));
            if !corrupted.verify_checksum() {
                detected += 1;
            }
        }
        assert_eq!(
            detected, total_bits,
            "2-bit XOR-fold checksum must detect every single-bit flip in hash bits"
        );
    }

    #[test]
    fn checksum_xor_fold_computation() {
        // All-zero upper bits → checksum 00
        let addr = PageAddr::new(0, Algorithm::Sha256Msb);
        // algo=00, hash=0 → upper 30 bits all zero → XOR fold = 00
        assert_eq!(addr.checksum(), 0b00);
        assert!(addr.verify_checksum());
    }

    #[test]
    fn max_hash_bits_value() {
        let max_hash = 0x0FFF_FFFF;
        let addr = PageAddr::new(max_hash, Algorithm::Sha224Lsb);
        assert_eq!(addr.hash_bits(), max_hash);
        assert!(addr.verify_checksum());
        // Verify the raw u32 has algo in bits 31-30 and hash in bits 29-2
        // algo=11 → bits 31-30 = 0xC000_0000
        // hash=0x0FFF_FFFF << 2 = 0x3FFF_FFFC
        // combined upper = 0xFFFF_FFFC
        let expected_upper = 0xFFFF_FFFC;
        assert_eq!(addr.0 & !0x3, expected_upper);
    }

    #[test]
    fn from_data_produces_valid_addr() {
        let data = b"hello athenaeum";
        let addr = PageAddr::from_data(data, Algorithm::Sha256Msb);
        assert!(addr.verify_checksum());
        assert_eq!(addr.algorithm(), Algorithm::Sha256Msb);
    }

    #[test]
    fn verify_data_matches_address() {
        let data = b"verify me";
        let addr = PageAddr::from_data(data, Algorithm::Sha256Msb);
        assert!(addr.verify_data(data));
        assert!(!addr.verify_data(b"wrong data"));
    }

    #[test]
    fn from_data_deterministic() {
        let data = b"deterministic";
        let a1 = PageAddr::from_data(data, Algorithm::Sha256Msb);
        let a2 = PageAddr::from_data(data, Algorithm::Sha256Msb);
        assert_eq!(a1, a2);
    }

    #[test]
    fn different_algorithms_produce_different_addresses() {
        let data = b"same data different algos";
        let a1 = PageAddr::from_data(data, Algorithm::Sha256Msb);
        let a2 = PageAddr::from_data(data, Algorithm::Sha224Lsb);
        // Different algorithms → different raw values (algo bits differ at minimum)
        assert_ne!(a1.0, a2.0);
    }

    #[test]
    fn debug_format() {
        let addr = PageAddr::new(42, Algorithm::Sha256Msb);
        let debug = alloc::format!("{:?}", addr);
        assert!(debug.contains("PageAddr("));
        assert!(debug.contains("Sha256Msb"));
    }

    #[test]
    fn from_raw_u32_round_trip() {
        let addr = PageAddr::new(100, Algorithm::Sha256Lsb);
        let raw = addr.0;
        let restored = PageAddr(raw);
        assert_eq!(restored.hash_bits(), 100);
        assert_eq!(restored.algorithm(), Algorithm::Sha256Lsb);
        assert!(restored.verify_checksum());
    }
}

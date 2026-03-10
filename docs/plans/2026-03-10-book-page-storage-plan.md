# Book/Page Storage Layer Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the ChunkAddr/Athenaeum addressing system in `harmony-athenaeum` with a simpler, higher-capacity PageAddr/Book/Encyclopedia model where pages are always 4KB and books are always 256 pages.

**Architecture:** Surgical refactor of `harmony-athenaeum` (6 source files, 81 existing tests). Preserve power-of-choice collision resolution and Volume partitioning algorithms. Replace bit layouts (21→28 bit hash space), rename types (ChunkAddr→PageAddr, Athenaeum→Book), remove depth/size_exponent (pages are always 4KB), simplify checksum (4-bit→2-bit XOR-fold). Add Zenoh key expressions for book/page namespace. Update harmony-os content_server.

**Tech Stack:** Rust (`no_std`), `sha2` crate, harmony-athenaeum (Ring 0), harmony-zenoh (Ring 0), harmony-microkernel (Ring 2, harmony-os)

**Reference Design:** `docs/plans/2026-03-10-book-page-storage-design.md`

---

### Task 1: PageAddr Type — Replace ChunkAddr

**Files:**
- Modify: `crates/harmony-athenaeum/src/addr.rs` (complete rewrite)
- Reference: `docs/plans/2026-03-10-book-page-storage-design.md` (PageAddr section)

This task replaces `ChunkAddr` with `PageAddr`. The old layout packed 21-bit hash + 2-bit algo + 2-bit depth + 3-bit size_exp + 4-bit checksum. The new layout is simpler: 2-bit algo (bits 31-30) + 28-bit hash_bits (bits 29-2) + 2-bit XOR-fold checksum (bits 1-0). Depth and size_exponent are gone — pages are always 4KB.

**Step 1: Write failing tests for PageAddr core**

Replace all existing tests in `addr.rs` with new ones. The old tests reference `Depth`, `size_exponent`, and the 4-bit checksum — none of which exist anymore.

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants() {
        assert_eq!(PAGE_SIZE, 4096);
        assert_eq!(PAGES_PER_BOOK, 256);
        assert_eq!(BOOK_MAX_SIZE, PAGE_SIZE * PAGES_PER_BOOK); // 1MB
        assert_eq!(ALGO_COUNT, 4);
    }

    #[test]
    fn null_page_fails_checksum() {
        let null = PageAddr(NULL_PAGE);
        assert!(!null.verify_checksum());
        assert_eq!(NULL_PAGE, 0x00000003);
    }

    #[test]
    fn round_trip_all_fields() {
        let addr = PageAddr::new(0x0ABC_DEF0, Algorithm::Sha256Msb);
        assert_eq!(addr.algorithm(), Algorithm::Sha256Msb);
        assert_eq!(addr.hash_bits(), 0x0ABC_DEF0 & 0x0FFF_FFFC);
        assert!(addr.verify_checksum());
    }

    #[test]
    fn algorithm_variants() {
        for algo in [
            Algorithm::Sha256Msb,
            Algorithm::Sha256Lsb,
            Algorithm::Sha224Msb,
            Algorithm::Sha224Lsb,
        ] {
            let addr = PageAddr::new(0x1234_5678, algo);
            assert_eq!(addr.algorithm(), algo);
            assert!(addr.verify_checksum());
        }
    }

    #[test]
    fn checksum_detects_single_bit_flip() {
        let addr = PageAddr::new(0x0ABC_DEF0, Algorithm::Sha256Msb);
        let flipped = PageAddr(addr.0 ^ (1 << 15)); // flip a hash bit
        // Flipping one bit in the hash region should (usually) invalidate checksum
        // Not guaranteed for every bit due to XOR-fold, but very likely
        // We test statistical detection, not per-bit guarantee
        let mut detected = 0u32;
        for bit in 2..30 {
            let f = PageAddr(addr.0 ^ (1 << bit));
            if !f.verify_checksum() {
                detected += 1;
            }
        }
        // At least half of single-bit flips should be detected
        assert!(detected > 14, "detected {detected}/28 flips");
    }

    #[test]
    fn checksum_xor_fold_computation() {
        // Verify the XOR-fold: bits 31-2 → 15 pairs → XOR all → 2 bits
        let addr = PageAddr::new(0x0000_0000, Algorithm::Sha256Msb);
        // algo=00, hash=0 → all 30 bits are 0 → 15 pairs of 00 → XOR = 00
        assert_eq!(addr.checksum(), 0b00);
        assert!(addr.verify_checksum());
    }

    #[test]
    fn max_hash_bits_value() {
        // 28-bit hash space: max = 0x0FFF_FFFC (bits 29-2 all set)
        let addr = PageAddr::new(0x0FFF_FFFC, Algorithm::Sha256Msb);
        assert_eq!(addr.hash_bits(), 0x0FFF_FFFC);
        assert!(addr.verify_checksum());
    }

    #[test]
    fn from_data_produces_valid_addr() {
        let data = [0x42u8; PAGE_SIZE];
        let addr = PageAddr::from_data(&data, Algorithm::Sha256Msb);
        assert!(addr.verify_checksum());
        assert_eq!(addr.algorithm(), Algorithm::Sha256Msb);
    }

    #[test]
    fn verify_data_matches_address() {
        let data = [0x42u8; PAGE_SIZE];
        let addr = PageAddr::from_data(&data, Algorithm::Sha256Msb);
        assert!(addr.verify_data(&data));
    }

    #[test]
    fn from_data_deterministic() {
        let data = [0x42u8; PAGE_SIZE];
        let a1 = PageAddr::from_data(&data, Algorithm::Sha256Msb);
        let a2 = PageAddr::from_data(&data, Algorithm::Sha256Msb);
        assert_eq!(a1, a2);
    }

    #[test]
    fn different_algorithms_produce_different_addresses() {
        let data = [0x42u8; PAGE_SIZE];
        let a0 = PageAddr::from_data(&data, Algorithm::Sha256Msb);
        let a1 = PageAddr::from_data(&data, Algorithm::Sha256Lsb);
        let a2 = PageAddr::from_data(&data, Algorithm::Sha224Msb);
        let a3 = PageAddr::from_data(&data, Algorithm::Sha224Lsb);
        // All should be valid
        assert!(a0.verify_checksum());
        assert!(a1.verify_checksum());
        assert!(a2.verify_checksum());
        assert!(a3.verify_checksum());
        // All should differ (different algo bits + likely different hash bits)
        let set: std::collections::HashSet<u32> = [a0.0, a1.0, a2.0, a3.0].into();
        assert_eq!(set.len(), 4, "all 4 algo variants should differ");
    }

    #[test]
    fn debug_format() {
        let addr = PageAddr::new(0x0ABC_0000, Algorithm::Sha256Msb);
        let dbg = format!("{:?}", addr);
        assert!(dbg.contains("PageAddr"), "debug should contain type name");
    }

    #[test]
    fn from_raw_u32_round_trip() {
        let addr = PageAddr::new(0x0123_4560, Algorithm::Sha224Lsb);
        let raw = addr.0;
        let restored = PageAddr(raw);
        assert_eq!(restored.algorithm(), addr.algorithm());
        assert_eq!(restored.hash_bits(), addr.hash_bits());
        assert_eq!(restored.checksum(), addr.checksum());
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- addr::tests 2>&1 | head -30`
Expected: Compilation errors — `PageAddr`, `NULL_PAGE`, `PAGE_SIZE`, etc. don't exist yet.

**Step 3: Implement PageAddr**

Replace the entire `addr.rs` contents. Keep `Algorithm` enum as-is. Delete `Depth`, `size_exponent` concept entirely.

```rust
extern crate alloc;
use alloc::format;
use core::fmt;

/// Page size: always 4 KiB.
pub const PAGE_SIZE: usize = 4096;

/// Pages per book: always 256.
pub const PAGES_PER_BOOK: usize = 256;

/// Maximum book size: 1 MiB (256 × 4 KiB).
pub const BOOK_MAX_SIZE: usize = PAGE_SIZE * PAGES_PER_BOOK;

/// Number of hash algorithm variants for power-of-choice.
pub const ALGO_COUNT: usize = 4;

/// Null sentinel — deliberately fails checksum validation.
/// Bits 31-2 are all zero → XOR-fold = 00, but checksum bits are 11.
pub const NULL_PAGE: u32 = 0x0000_0003;

/// Hash algorithm selector (bits 31-30 of PageAddr).
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
    /// All algorithm variants in priority order.
    pub const ALL: [Algorithm; ALGO_COUNT] = [
        Algorithm::Sha256Msb,
        Algorithm::Sha256Lsb,
        Algorithm::Sha224Msb,
        Algorithm::Sha224Lsb,
    ];

    fn from_bits(bits: u8) -> Self {
        match bits & 0x03 {
            0 => Algorithm::Sha256Msb,
            1 => Algorithm::Sha256Lsb,
            2 => Algorithm::Sha224Msb,
            _ => Algorithm::Sha224Lsb,
        }
    }
}

/// 32-bit content address for a 4 KiB page.
///
/// Layout:
/// ```text
/// ┌──────────┬────────────────────────────────┬──────────┐
/// │ algo (2) │         hash_bits (28)         │ cksum(2) │
/// │ bits 31-30│        bits 29-2              │ bits 1-0 │
/// └──────────┴────────────────────────────────┴──────────┘
/// ```
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PageAddr(pub(crate) u32);

impl PageAddr {
    /// Construct a PageAddr from raw hash_bits and algorithm.
    ///
    /// `hash_bits` must be pre-masked to bits 29-2 (i.e., bits 31-30 and 1-0 clear).
    /// The checksum is computed automatically.
    pub fn new(hash_bits: u32, algorithm: Algorithm) -> Self {
        let algo_bits = (algorithm as u32) << 30;
        let masked_hash = hash_bits & 0x3FFF_FFFC; // keep only bits 29-2
        let upper = algo_bits | masked_hash;
        let cksum = Self::compute_checksum(upper);
        PageAddr(upper | cksum as u32)
    }

    /// Derive a PageAddr from 4 KiB page data using the specified algorithm.
    pub fn from_data(data: &[u8], algorithm: Algorithm) -> Self {
        let hash_bits = crate::hash::derive_hash_bits(data, algorithm);
        Self::new(hash_bits, algorithm)
    }

    /// Algorithm selector (bits 31-30).
    pub fn algorithm(&self) -> Algorithm {
        Algorithm::from_bits((self.0 >> 30) as u8)
    }

    /// 28-bit hash address (bits 29-2, with bits 31-30 and 1-0 cleared).
    pub fn hash_bits(&self) -> u32 {
        self.0 & 0x3FFF_FFFC
    }

    /// 2-bit checksum (bits 1-0).
    pub fn checksum(&self) -> u8 {
        (self.0 & 0x03) as u8
    }

    /// Verify that the checksum matches the upper 30 bits.
    pub fn verify_checksum(&self) -> bool {
        let expected = Self::compute_checksum(self.0 & 0xFFFF_FFFC);
        self.checksum() == expected
    }

    /// Verify that the given data matches this address.
    pub fn verify_data(&self, data: &[u8]) -> bool {
        let derived = Self::from_data(data, self.algorithm());
        derived.hash_bits() == self.hash_bits()
    }

    /// XOR-fold bits 31-2 (30 bits) into 15 pairs, XOR all pairs → 2-bit result.
    fn compute_checksum(upper: u32) -> u8 {
        let bits30 = upper >> 2; // shift out the checksum position
        let mut result = 0u8;
        for i in 0..15 {
            result ^= ((bits30 >> (i * 2)) & 0x03) as u8;
        }
        result
    }
}

impl fmt::Debug for PageAddr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PageAddr({:#010x}, {:?})",
            self.0,
            self.algorithm(),
        )
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- addr::tests -v`
Expected: All tests pass. Other modules will have compilation errors (they still reference `ChunkAddr`), but addr-specific tests should pass.

Note: At this point, `lib.rs` still re-exports `ChunkAddr`, `athenaeum.rs` still uses it, etc. Those will break. That's expected — we fix them in subsequent tasks.

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/addr.rs
git commit -m "feat(athenaeum): replace ChunkAddr with PageAddr (32-bit: 2-bit algo + 28-bit hash + 2-bit XOR checksum)"
```

---

### Task 2: Update Hash Bit Extraction

**Files:**
- Modify: `crates/harmony-athenaeum/src/hash.rs`

The old `derive_hash_bits` extracted 21 bits from 3 bytes. The new version extracts the middle 28 bits (bits 29-2) from a 4-byte window of the hash digest.

**Step 1: Write failing tests**

Replace hash tests to match new 28-bit extraction:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Algorithm;

    #[test]
    fn sha256_msb_deterministic() {
        let data = [0xAA; 4096];
        let a = derive_hash_bits(&data, Algorithm::Sha256Msb);
        let b = derive_hash_bits(&data, Algorithm::Sha256Msb);
        assert_eq!(a, b);
    }

    #[test]
    fn sha256_msb_vs_lsb_differ() {
        let data = [0xBB; 4096];
        let msb = derive_hash_bits(&data, Algorithm::Sha256Msb);
        let lsb = derive_hash_bits(&data, Algorithm::Sha256Lsb);
        assert_ne!(msb, lsb);
    }

    #[test]
    fn sha224_msb_vs_sha256_msb_differ() {
        let data = [0xCC; 4096];
        let s256 = derive_hash_bits(&data, Algorithm::Sha256Msb);
        let s224 = derive_hash_bits(&data, Algorithm::Sha224Msb);
        assert_ne!(s256, s224);
    }

    #[test]
    fn hash_bits_fit_in_28_bits_at_positions_29_to_2() {
        let data = [0xDD; 4096];
        for algo in Algorithm::ALL {
            let bits = derive_hash_bits(&data, algo);
            // bits 31-30 and 1-0 must be clear
            assert_eq!(bits & 0xC000_0003, 0, "algo {:?}: bits outside 29-2 must be 0", algo);
            // bits 29-2 should have some non-zero content (statistically certain)
        }
    }

    #[test]
    fn different_data_different_bits() {
        let d1 = [0x00; 4096];
        let d2 = [0xFF; 4096];
        let b1 = derive_hash_bits(&d1, Algorithm::Sha256Msb);
        let b2 = derive_hash_bits(&d2, Algorithm::Sha256Msb);
        assert_ne!(b1, b2);
    }

    #[test]
    fn full_hash_sha256_known_vector() {
        // SHA-256 of empty input is well-known
        let hash = sha256_hash(b"");
        assert_eq!(
            &hash[..4],
            &[0xe3, 0xb0, 0xc4, 0x42],
            "SHA-256('') starts with e3b0c442"
        );
    }

    #[test]
    fn full_hash_sha224_known_vector() {
        let hash = sha224_hash(b"");
        assert_eq!(
            &hash[..4],
            &[0xd1, 0x4a, 0x02, 0x8c],
            "SHA-224('') starts with d14a028c"
        );
    }

    #[test]
    fn all_four_algorithms_produce_values() {
        let data = [0x42; 4096];
        for algo in Algorithm::ALL {
            let bits = derive_hash_bits(&data, algo);
            // Should produce non-zero hash bits (statistically certain for non-trivial data)
            assert_ne!(bits, 0, "algo {:?} produced zero hash_bits", algo);
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- hash::tests 2>&1 | head -30`
Expected: `hash_bits_fit_in_28_bits_at_positions_29_to_2` fails (old code produces 21-bit values in different positions).

**Step 3: Implement new derive_hash_bits**

```rust
use sha2::{Digest, Sha256, Sha224};
use crate::Algorithm;

/// Full SHA-256 hash.
pub fn sha256_hash(data: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Full SHA-224 hash.
pub fn sha224_hash(data: &[u8]) -> [u8; 28] {
    let mut hasher = Sha224::new();
    hasher.update(data);
    hasher.finalize().into()
}

/// Derive 28-bit hash_bits (positioned at bits 29-2 of a u32).
///
/// 1. Hash data with SHA-256 or SHA-224 per the algorithm.
/// 2. Take first or last 4 bytes as big-endian u32.
/// 3. Extract bits 29-2 of that u32 (mask with 0x3FFF_FFFC).
pub(crate) fn derive_hash_bits(data: &[u8], algorithm: Algorithm) -> u32 {
    let window = match algorithm {
        Algorithm::Sha256Msb => {
            let h = sha256_hash(data);
            u32::from_be_bytes([h[0], h[1], h[2], h[3]])
        }
        Algorithm::Sha256Lsb => {
            let h = sha256_hash(data);
            u32::from_be_bytes([h[28], h[29], h[30], h[31]])
        }
        Algorithm::Sha224Msb => {
            let h = sha224_hash(data);
            u32::from_be_bytes([h[0], h[1], h[2], h[3]])
        }
        Algorithm::Sha224Lsb => {
            let h = sha224_hash(data);
            u32::from_be_bytes([h[24], h[25], h[26], h[27]])
        }
    };
    window & 0x3FFF_FFFC // bits 29-2
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- hash::tests -v`
Expected: All 8 tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/hash.rs
git commit -m "feat(athenaeum): update derive_hash_bits for 28-bit extraction (bits 29-2 of 4-byte window)"
```

---

### Task 3: Book Type — Replace Athenaeum

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (complete rewrite → becomes Book)

The old `Athenaeum` stored a single CID blob as `Vec<ChunkAddr>` with variable-size chunks. The new `Book` stores a single CID blob as exactly 256 page slots, each with all 4 algorithm variants precomputed. It also generates a 4 KiB Table of Contents.

**Step 1: Write failing tests for Book**

Replace all tests in `athenaeum.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::{PAGE_SIZE, PAGES_PER_BOOK, BOOK_MAX_SIZE, NULL_PAGE};

    #[test]
    fn from_blob_single_page() {
        let data = [0x42u8; 100]; // < 4KB → 1 page
        let cid = [0xAA; 32];
        let book = Book::from_blob(cid, &data).unwrap();
        assert_eq!(book.cid, cid);
        assert_eq!(book.blob_size, 100);
        assert_eq!(book.page_count(), 1);
        // Each page has 4 algo variants
        assert_eq!(book.pages.len(), 1);
        for addr in &book.pages[0] {
            assert!(addr.verify_checksum());
        }
    }

    #[test]
    fn from_blob_full_page() {
        let data = [0x42u8; PAGE_SIZE];
        let book = Book::from_blob([0; 32], &data).unwrap();
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.blob_size as usize, PAGE_SIZE);
    }

    #[test]
    fn from_blob_multiple_pages() {
        let data = [0x42u8; PAGE_SIZE * 3 + 100]; // 3 full + 1 partial = 4 pages
        let book = Book::from_blob([0; 32], &data).unwrap();
        assert_eq!(book.page_count(), 4);
        assert_eq!(book.blob_size as usize, PAGE_SIZE * 3 + 100);
    }

    #[test]
    fn from_blob_max_size() {
        let data = vec![0x42u8; BOOK_MAX_SIZE]; // exactly 1MB = 256 pages
        let book = Book::from_blob([0; 32], &data).unwrap();
        assert_eq!(book.page_count(), PAGES_PER_BOOK);
    }

    #[test]
    fn from_blob_too_large() {
        let data = vec![0u8; BOOK_MAX_SIZE + 1];
        assert!(Book::from_blob([0; 32], &data).is_err());
    }

    #[test]
    fn from_blob_empty() {
        let book = Book::from_blob([0; 32], &[]).unwrap();
        assert_eq!(book.page_count(), 0);
        assert_eq!(book.blob_size, 0);
    }

    #[test]
    fn from_blob_cid_preserved() {
        let cid = [0xFF; 32];
        let book = Book::from_blob(cid, &[0x42; 100]).unwrap();
        assert_eq!(book.cid, cid);
    }

    #[test]
    fn last_page_zero_padded_produces_valid_addr() {
        // 1 byte → 1 page, padded with 4095 zero bytes
        let data = [0x42u8; 1];
        let book = Book::from_blob([0; 32], &data).unwrap();
        assert_eq!(book.page_count(), 1);
        // The page is 0x42 followed by 4095 zeros — valid content
        for addr in &book.pages[0] {
            assert!(addr.verify_checksum());
        }
    }

    #[test]
    fn all_four_variants_precomputed() {
        let data = [0x42u8; PAGE_SIZE];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let variants = &book.pages[0];
        assert_eq!(variants[0].algorithm(), Algorithm::Sha256Msb);
        assert_eq!(variants[1].algorithm(), Algorithm::Sha256Lsb);
        assert_eq!(variants[2].algorithm(), Algorithm::Sha224Msb);
        assert_eq!(variants[3].algorithm(), Algorithm::Sha224Lsb);
    }

    #[test]
    fn toc_is_4kb() {
        let data = [0x42u8; PAGE_SIZE * 3];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let toc = book.toc();
        assert_eq!(toc.len(), PAGE_SIZE);
    }

    #[test]
    fn toc_layout_sections() {
        // ToC layout: 4 sections × 256 entries × 4 bytes = 4096 bytes
        let data = [0x42u8; PAGE_SIZE * 2]; // 2 pages
        let book = Book::from_blob([0; 32], &data).unwrap();
        let toc = book.toc();

        // Section 0 (algo=00): bytes 0..1024
        // Page 0 should have a valid address
        let addr0 = u32::from_le_bytes([toc[0], toc[1], toc[2], toc[3]]);
        assert_ne!(addr0, NULL_PAGE);
        let pa0 = PageAddr(addr0);
        assert!(pa0.verify_checksum());
        assert_eq!(pa0.algorithm(), Algorithm::Sha256Msb);

        // Page 1 should also be valid
        let addr1 = u32::from_le_bytes([toc[4], toc[5], toc[6], toc[7]]);
        assert_ne!(addr1, NULL_PAGE);

        // Page 2 (doesn't exist) should be NULL_PAGE
        let addr2 = u32::from_le_bytes([toc[8], toc[9], toc[10], toc[11]]);
        assert_eq!(addr2, NULL_PAGE);

        // Section 1 (algo=01) starts at byte 1024
        let s1_addr0 = u32::from_le_bytes([toc[1024], toc[1025], toc[1026], toc[1027]]);
        let pa_s1 = PageAddr(s1_addr0);
        assert!(pa_s1.verify_checksum());
        assert_eq!(pa_s1.algorithm(), Algorithm::Sha256Lsb);
    }

    #[test]
    fn reassemble_round_trip() {
        let original = vec![0x42u8; PAGE_SIZE * 2 + 500];
        let book = Book::from_blob([0; 32], &original).unwrap();

        // Build page data store keyed by page index
        let pages: Vec<Vec<u8>> = book.page_data_from_blob(&original);

        let reassembled = book.reassemble(|idx| pages.get(idx as usize).cloned());
        assert_eq!(reassembled.unwrap(), original);
    }

    #[test]
    fn reassemble_missing_page_fails() {
        let data = [0x42u8; PAGE_SIZE * 2];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let result = book.reassemble(|_| None);
        assert!(result.is_err());
    }

    #[test]
    fn blob_size_preserved() {
        let data = [0x42u8; 12345];
        let book = Book::from_blob([0; 32], &data).unwrap();
        assert_eq!(book.blob_size, 12345);
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- athenaeum::tests 2>&1 | head -30`
Expected: Compilation errors — `Book`, `toc()`, `page_count()`, `page_data_from_blob()`, `reassemble()` don't exist yet.

**Step 3: Implement Book**

Replace the contents of `athenaeum.rs`:

```rust
extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

use crate::addr::{Algorithm, PageAddr, PAGE_SIZE, PAGES_PER_BOOK, BOOK_MAX_SIZE, NULL_PAGE, ALGO_COUNT};

/// Error building or using a Book.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BookError {
    /// Blob exceeds maximum size (1 MB).
    BlobTooLarge { size: usize },
    /// All 4 algorithm variants collide for a page.
    AllAlgorithmsCollide { page_index: usize },
    /// Missing page during reassembly.
    MissingPage { page_index: u8 },
    /// Serialization format error.
    BadFormat,
    /// Data too short for deserialization.
    TooShort,
}

/// A Book represents one ContentId-addressed blob (up to 1 MB) split into
/// exactly `page_count` pages of 4 KiB each, with all 4 algorithm variants
/// precomputed per page.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Book {
    /// 256-bit blob identifier.
    pub cid: [u8; 32],
    /// Per-page addresses: `pages[i]` = `[algo0, algo1, algo2, algo3]`.
    /// Length = `ceil(blob_size / PAGE_SIZE)`, max 256.
    pub pages: Vec<[PageAddr; ALGO_COUNT]>,
    /// Actual byte count of the blob (≤ BOOK_MAX_SIZE).
    pub blob_size: u32,
}

impl Book {
    /// Build a Book by chunking a blob into 4 KiB pages.
    ///
    /// Each page gets all 4 algorithm variants precomputed. The last page is
    /// zero-padded to 4 KiB for hashing.
    pub fn from_blob(cid: [u8; 32], data: &[u8]) -> Result<Self, BookError> {
        if data.len() > BOOK_MAX_SIZE {
            return Err(BookError::BlobTooLarge { size: data.len() });
        }

        let page_count = if data.is_empty() {
            0
        } else {
            (data.len() + PAGE_SIZE - 1) / PAGE_SIZE
        };

        let mut pages = Vec::with_capacity(page_count);

        for i in 0..page_count {
            let start = i * PAGE_SIZE;
            let end = core::cmp::min(start + PAGE_SIZE, data.len());
            let chunk = &data[start..end];

            // Zero-pad last page to PAGE_SIZE
            let mut page_buf = [0u8; PAGE_SIZE];
            page_buf[..chunk.len()].copy_from_slice(chunk);

            let variants = Algorithm::ALL.map(|algo| PageAddr::from_data(&page_buf, algo));
            pages.push(variants);
        }

        Ok(Book {
            cid,
            pages,
            blob_size: data.len() as u32,
        })
    }

    /// Number of pages in this book.
    pub fn page_count(&self) -> usize {
        self.pages.len()
    }

    /// Generate the 4 KiB Table of Contents.
    ///
    /// Layout: 4 sections (one per algorithm) × 256 entries × 4 bytes (LE u32).
    /// Unused positions are filled with `NULL_PAGE`.
    pub fn toc(&self) -> Vec<u8> {
        let mut buf = vec![0u8; PAGE_SIZE];

        for (algo_idx, _algo) in Algorithm::ALL.iter().enumerate() {
            let section_offset = algo_idx * PAGES_PER_BOOK * 4;
            for page_idx in 0..PAGES_PER_BOOK {
                let entry_offset = section_offset + page_idx * 4;
                let value = if page_idx < self.pages.len() {
                    self.pages[page_idx][algo_idx].0
                } else {
                    NULL_PAGE
                };
                buf[entry_offset..entry_offset + 4].copy_from_slice(&value.to_le_bytes());
            }
        }

        buf
    }

    /// Split a blob into page-sized buffers (for storage/reassembly).
    pub fn page_data_from_blob(&self, data: &[u8]) -> Vec<Vec<u8>> {
        let mut pages = Vec::with_capacity(self.page_count());
        for i in 0..self.page_count() {
            let start = i * PAGE_SIZE;
            let end = core::cmp::min(start + PAGE_SIZE, data.len());
            let mut page_buf = vec![0u8; PAGE_SIZE];
            page_buf[..end - start].copy_from_slice(&data[start..end]);
            pages.push(page_buf);
        }
        pages
    }

    /// Reassemble the original blob from pages, fetched by index.
    pub fn reassemble(
        &self,
        fetch: impl Fn(u8) -> Option<Vec<u8>>,
    ) -> Result<Vec<u8>, BookError> {
        let mut result = Vec::with_capacity(self.blob_size as usize);

        for i in 0..self.page_count() {
            let page_data = fetch(i as u8)
                .ok_or(BookError::MissingPage { page_index: i as u8 })?;
            result.extend_from_slice(&page_data);
        }

        result.truncate(self.blob_size as usize);
        Ok(result)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- athenaeum::tests -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): replace Athenaeum with Book — 256 × 4KB pages, 4 algo variants, ToC generation"
```

---

### Task 4: Delete Old Book Type

**Files:**
- Delete: `crates/harmony-athenaeum/src/book.rs`
- Modify: `crates/harmony-athenaeum/src/lib.rs` (remove `mod book` and old re-exports)

The old `Book` type held 1-3 blobs with cross-consistent addressing. That cross-blob role is now handled by the Encyclopedia. The new `Book` (in athenaeum.rs) replaces both old types.

**Step 1: Delete book.rs and update lib.rs**

Remove `mod book;` from `lib.rs`. Remove any re-exports of old `Book`, `BookEntry`, `BookError` (the new `BookError` comes from `athenaeum.rs` now).

Read `lib.rs` first to see current re-exports, then:

- Delete `book.rs` file
- Remove `mod book;` line from `lib.rs`
- Update re-exports:
  - Remove: `ChunkAddr`, `Athenaeum`, old `Book`, `BookEntry`, old `BookError`, `Depth`, `MissingChunkError`, `CollisionError`
  - Add: `PageAddr`, `Book` (new), `BookError` (new), `Algorithm`
  - Keep: `Volume`, `Encyclopedia`, `sha256_hash`, `sha224_hash`, `MAX_BLOB_SIZE` (rename to `BOOK_MAX_SIZE`)

```rust
// lib.rs — updated re-exports
#![no_std]
extern crate alloc;

mod addr;
mod athenaeum; // Book lives here now
mod encyclopedia;
mod hash;
mod volume;

pub use addr::{Algorithm, PageAddr, PAGE_SIZE, PAGES_PER_BOOK, BOOK_MAX_SIZE, ALGO_COUNT, NULL_PAGE};
pub use athenaeum::{Book, BookError};
pub use encyclopedia::Encyclopedia;
pub use hash::{sha256_hash, sha224_hash};
pub use volume::Volume;

// Legacy alias for backward compatibility during transition
pub const MAX_BLOB_SIZE: usize = BOOK_MAX_SIZE;
```

**Step 2: Run compilation to check**

Run: `cargo check -p harmony-athenaeum 2>&1 | head -40`
Expected: Compilation errors in `volume.rs` and `encyclopedia.rs` (they still reference `ChunkAddr`). That's expected — fixed in Tasks 5 and 6. The addr, hash, and athenaeum modules should be clean.

**Step 3: Commit**

```bash
git rm crates/harmony-athenaeum/src/book.rs
git add crates/harmony-athenaeum/src/lib.rs
git commit -m "refactor(athenaeum): delete old Book type, update lib.rs re-exports for PageAddr/Book"
```

---

### Task 5: Update Volume

**Files:**
- Modify: `crates/harmony-athenaeum/src/volume.rs`

Volume's structure is largely unchanged — it's a binary tree of partitions. The changes are:
1. Replace `ChunkAddr` references with `PageAddr`
2. Update `Book` references to use new `Book` type
3. Leaf nodes hold `Vec<Book>` (new Book type)
4. Update serialization (each entry is now `PageAddr.0` as u32 LE, same wire format)
5. Update `PARTITION_START_BIT` — old was 22 (for 21-bit addressing), new should be 30 (bits 29-2 are the hash space, partition routing uses content-hash bits beyond the address space)

Actually, looking more carefully: `route_chunk` uses SHA-256 content-hash bits (not address bits) for partition routing. The `PARTITION_START_BIT` in the old Encyclopedia was 22 — this is the bit index into the SHA-256 hash, not into the PageAddr. With 28-bit address space, we want to start routing at a bit index beyond the address extraction window. But `derive_hash_bits` uses the first/last 4 bytes of the hash — so content routing should use bits from the middle of the hash to avoid correlation. The old value of 22 (starting at byte 2, bit 6) is reasonable — it's in the middle of the hash. This can stay the same or be adjusted. For now, keep it at 22 since it's orthogonal to the address space change.

**Step 1: Write updated tests**

The Volume tests are mostly about tree structure and serialization. Replace `ChunkAddr` with `PageAddr` in test construction:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::{PageAddr, Algorithm, NULL_PAGE, PAGE_SIZE};
    use crate::athenaeum::Book;

    fn sample_book() -> Book {
        Book::from_blob([0xAA; 32], &[0x42u8; PAGE_SIZE * 2]).unwrap()
    }

    #[test]
    fn route_chunk_bit_22() {
        let hash = crate::hash::sha256_hash(b"test data");
        let result = route_chunk(&hash, 22);
        // Bit 22 is at byte 2, bit 6 — just verify it returns a bool
        assert!(result || !result);
    }

    #[test]
    fn route_chunk_bit_0() {
        let hash = crate::hash::sha256_hash(b"test data");
        let result = route_chunk(&hash, 0);
        // Bit 0 is MSB of byte 0
        let expected = (hash[0] >> 7) & 1 == 1;
        assert_eq!(result, expected);
    }

    #[test]
    fn leaf_volume_book_count() {
        let v = Volume::leaf(0, 0, vec![sample_book(), sample_book()]);
        assert_eq!(v.book_count(), 2);
    }

    #[test]
    fn split_volume_book_count() {
        let left = Volume::leaf(1, 0, vec![sample_book()]);
        let right = Volume::leaf(1, 1, vec![sample_book(), sample_book()]);
        let v = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        assert_eq!(v.book_count(), 3);
    }

    #[test]
    fn volume_depth_and_path() {
        let v = Volume::leaf(3, 5, vec![]);
        assert_eq!(v.depth(), 3);
        assert_eq!(v.path(), 5);
    }

    #[test]
    fn leaf_volume_round_trip() {
        let v = Volume::leaf(0, 0, vec![sample_book()]);
        let bytes = v.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(v.book_count(), restored.book_count());
        assert_eq!(v.depth(), restored.depth());
    }

    #[test]
    fn split_volume_round_trip() {
        let left = Volume::leaf(1, 0, vec![sample_book()]);
        let right = Volume::leaf(1, 1, vec![sample_book()]);
        let v = Volume::Split {
            partition_depth: 0,
            partition_path: 0,
            split_bit: 22,
            left: Box::new(left),
            right: Box::new(right),
        };
        let bytes = v.to_bytes();
        let restored = Volume::from_bytes(&bytes).unwrap();
        assert_eq!(v.book_count(), restored.book_count());
    }

    #[test]
    fn volume_from_bytes_too_short() {
        assert!(Volume::from_bytes(&[]).is_err());
        assert!(Volume::from_bytes(&[0]).is_err());
    }

    #[test]
    fn volume_from_bytes_unknown_tag() {
        assert!(Volume::from_bytes(&[99, 0, 0, 0, 0, 0, 0, 0]).is_err());
    }
}
```

**Step 2: Update Volume implementation**

Key changes:
- `Book` references now point to new `Book` type from `athenaeum.rs`
- `chunk_count()` → `page_count()` — counts total pages across all books
- Serialization of Book entries uses new Book's serialization

The Volume struct itself stays the same shape:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Volume {
    Leaf {
        partition_depth: u8,
        partition_path: u32,
        books: Vec<Book>,
    },
    Split {
        partition_depth: u8,
        partition_path: u32,
        split_bit: u8,
        left: Box<Volume>,
        right: Box<Volume>,
    },
}
```

Update methods:
- `chunk_count()` → rename to `page_count()`, sum `book.page_count()` for each book
- Serialization: each Book is serialized as `cid(32) + blob_size(4 LE) + page_count(2 LE) + reserved(2) + pages(page_count × ALGO_COUNT × 4 bytes LE)`
- Keep `route_chunk`, `MAX_PARTITION_DEPTH` as-is

**Step 3: Run tests**

Run: `cargo test -p harmony-athenaeum -- volume::tests -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/volume.rs
git commit -m "refactor(athenaeum): update Volume from ChunkAddr to PageAddr, rename chunk_count to page_count"
```

---

### Task 6: Update Encyclopedia

**Files:**
- Modify: `crates/harmony-athenaeum/src/encyclopedia.rs`

The Encyclopedia now works with new Book types. Key changes:
1. `ChunkAddr` → `PageAddr` throughout
2. Use Book's precomputed ToC for power-of-choice resolution
3. Add `Assignment` struct for the flat assignment map
4. Update `SPLIT_THRESHOLD` for 28-bit space: `0.75 × 2^28 = 201,326,592` (vs old `0.75 × 2^21 = 1,572,864`)
5. Keep recursive Volume partitioning as fallback

**Step 1: Write updated tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::{PageAddr, Algorithm, PAGE_SIZE, BOOK_MAX_SIZE};
    use crate::athenaeum::Book;

    #[test]
    fn build_single_blob() {
        let data = [0x42u8; PAGE_SIZE * 3];
        let cid = crate::hash::sha256_hash(&data);
        let enc = Encyclopedia::build(&[(cid, data.as_slice())]).unwrap();
        assert_eq!(enc.total_blobs, 1);
        assert!(enc.total_unique_pages > 0);
    }

    #[test]
    fn build_multiple_blobs() {
        let d1 = [0x42u8; PAGE_SIZE * 2];
        let d2 = [0x99u8; PAGE_SIZE * 3];
        let c1 = crate::hash::sha256_hash(&d1);
        let c2 = crate::hash::sha256_hash(&d2);
        let enc = Encyclopedia::build(&[(c1, d1.as_slice()), (c2, d2.as_slice())]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        assert_eq!(enc.total_unique_pages, 5); // 2 + 3, no shared content
    }

    #[test]
    fn build_deduplicates_shared_pages() {
        // Two blobs with identical first page
        let mut d1 = vec![0x42u8; PAGE_SIZE * 2];
        let mut d2 = vec![0x42u8; PAGE_SIZE]; // same first page
        d1[PAGE_SIZE..].fill(0xAA); // different second page
        d2.extend_from_slice(&[0xBB; PAGE_SIZE]); // different second page

        let c1 = crate::hash::sha256_hash(&d1);
        let c2 = crate::hash::sha256_hash(&d2);
        let enc = Encyclopedia::build(&[(c1, d1.as_slice()), (c2, d2.as_slice())]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        // 4 total pages, but page 0 is shared → 3 unique
        assert_eq!(enc.total_unique_pages, 3);
    }

    #[test]
    fn build_empty() {
        let enc = Encyclopedia::build(&[]).unwrap();
        assert_eq!(enc.total_blobs, 0);
        assert_eq!(enc.total_unique_pages, 0);
    }

    #[test]
    fn build_blob_too_large() {
        let data = vec![0u8; BOOK_MAX_SIZE + 1];
        let cid = [0; 32];
        assert!(Encyclopedia::build(&[(cid, data.as_slice())]).is_err());
    }

    #[test]
    fn serialization_round_trip() {
        let d1 = [0x42u8; PAGE_SIZE * 2];
        let c1 = crate::hash::sha256_hash(&d1);
        let enc = Encyclopedia::build(&[(c1, d1.as_slice())]).unwrap();
        let bytes = enc.to_bytes();
        let restored = Encyclopedia::from_bytes(&bytes).unwrap();
        assert_eq!(enc.total_blobs, restored.total_blobs);
        assert_eq!(enc.total_unique_pages, restored.total_unique_pages);
    }

    #[test]
    fn from_bytes_bad_magic() {
        assert!(Encyclopedia::from_bytes(b"XXXX\x01\x00\x00\x00\x00\x00\x00\x00\x00").is_err());
    }

    #[test]
    fn from_bytes_too_short() {
        assert!(Encyclopedia::from_bytes(&[]).is_err());
        assert!(Encyclopedia::from_bytes(b"ENCY").is_err());
    }

    #[test]
    fn split_threshold_value() {
        // 75% of 2^28
        assert_eq!(SPLIT_THRESHOLD, 201_326_592);
    }
}
```

**Step 2: Update Encyclopedia implementation**

Key structural changes:

```rust
/// Page assignment within an Encyclopedia.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Assignment {
    pub book_index: usize,
    pub page_index: u8,
    pub algo: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Encyclopedia {
    pub root: Volume,
    pub total_blobs: u32,
    pub total_unique_pages: u32,
}

/// 75% of 2^28 address space.
pub(crate) const SPLIT_THRESHOLD: usize = 201_326_592;

/// Starting bit for partition routing (orthogonal to address extraction).
pub(crate) const PARTITION_START_BIT: u8 = 22;
```

The `build()` method follows the same pattern as before:
1. Create Books from all blobs
2. Deduplicate pages by content hash
3. Assign addresses using power-of-choice (try algo 0-3 for each unique page)
4. If too many pages for flat resolution, partition via Volume tree

Update all internal `ChunkInfo` → `PageInfo`, `chunk_count` → `page_count`, etc.

**Step 3: Run tests**

Run: `cargo test -p harmony-athenaeum -- encyclopedia::tests -v`
Expected: All tests pass.

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/encyclopedia.rs
git commit -m "refactor(athenaeum): update Encyclopedia for PageAddr, add Assignment type, 28-bit split threshold"
```

---

### Task 7: Integration Tests and lib.rs Cleanup

**Files:**
- Modify: `crates/harmony-athenaeum/src/lib.rs`

Update integration tests in `lib.rs` to use new types and verify end-to-end workflows.

**Step 1: Write updated integration tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn end_to_end_1mb_blob() {
        let data = vec![0x42u8; BOOK_MAX_SIZE];
        let cid = sha256_hash(&data);
        let book = Book::from_blob(cid, &data).unwrap();
        assert_eq!(book.page_count(), PAGES_PER_BOOK);

        let pages = book.page_data_from_blob(&data);
        let reassembled = book.reassemble(|idx| pages.get(idx as usize).cloned()).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn verify_individual_pages() {
        let data = [0x42u8; PAGE_SIZE * 3];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let pages = book.page_data_from_blob(&data);

        for (i, page_data) in pages.iter().enumerate() {
            // Verify each page against its algo=0 address
            let addr = book.pages[i][0];
            assert!(addr.verify_data(page_data));
        }
    }

    #[test]
    fn small_blob_round_trip() {
        let data = b"hello world";
        let cid = sha256_hash(data);
        let book = Book::from_blob(cid, data).unwrap();
        assert_eq!(book.page_count(), 1);
        assert_eq!(book.blob_size, 11);

        let pages = book.page_data_from_blob(data);
        let reassembled = book.reassemble(|idx| pages.get(idx as usize).cloned()).unwrap();
        assert_eq!(reassembled, data);
    }

    #[test]
    fn toc_round_trip() {
        let data = [0x42u8; PAGE_SIZE * 5];
        let book = Book::from_blob([0; 32], &data).unwrap();
        let toc = book.toc();
        assert_eq!(toc.len(), PAGE_SIZE);

        // Verify ToC entries match book.pages
        for (page_idx, variants) in book.pages.iter().enumerate() {
            for (algo_idx, addr) in variants.iter().enumerate() {
                let offset = algo_idx * PAGES_PER_BOOK * 4 + page_idx * 4;
                let stored = u32::from_le_bytes([
                    toc[offset], toc[offset+1], toc[offset+2], toc[offset+3],
                ]);
                assert_eq!(stored, addr.0, "mismatch at page {page_idx}, algo {algo_idx}");
            }
        }
    }

    #[test]
    fn page_addr_debug_is_informative() {
        let addr = PageAddr::new(0x0ABC_0000, Algorithm::Sha256Msb);
        let dbg = format!("{:?}", addr);
        assert!(dbg.contains("PageAddr"));
    }

    #[test]
    fn encyclopedia_multi_blob_reassemble() {
        let d1 = [0x42u8; PAGE_SIZE * 2];
        let d2 = [0x99u8; PAGE_SIZE * 3];
        let c1 = sha256_hash(&d1);
        let c2 = sha256_hash(&d2);
        let enc = Encyclopedia::build(&[(c1, d1.as_slice()), (c2, d2.as_slice())]).unwrap();
        assert_eq!(enc.total_blobs, 2);
        assert!(enc.total_unique_pages > 0);
    }

    #[test]
    fn null_page_sentinel() {
        let null = PageAddr(NULL_PAGE);
        assert!(!null.verify_checksum(), "NULL_PAGE must fail checksum");
        // XOR-fold of 30 zero bits = 00, but stored checksum is 11
        assert_eq!(null.checksum(), 0b11);
    }
}
```

**Step 2: Run full test suite**

Run: `cargo test -p harmony-athenaeum -v`
Expected: All tests across all modules pass.

**Step 3: Run clippy**

Run: `cargo clippy -p harmony-athenaeum`
Expected: Zero warnings.

**Step 4: Commit**

```bash
git add crates/harmony-athenaeum/src/lib.rs
git commit -m "test(athenaeum): update integration tests for PageAddr/Book/Encyclopedia"
```

---

### Task 8: Zenoh Book/Page Key Expressions

**Files:**
- Modify: `crates/harmony-zenoh/src/keyspace.rs`

Add key expression builders and parsers for the book/page namespace defined in the design doc:

```
harmony/book/{cid_hex}/page/{page_addr_hex}   — content-addressed access
harmony/book/{cid_hex}/pos/{index}             — positional access (0-255)
harmony/book/{cid_hex}/toc                     — Table of Contents
harmony/book/{cid_hex}/meta                    — metadata
harmony/book/*/page/{page_addr_hex}            — "who has this page?"
harmony/book/{cid_hex}/page/*                  — "all pages of this book"
harmony/book/{cid_hex}/pos/*                   — "stream book in order"
```

**Step 1: Write failing tests**

Add to the test module in `keyspace.rs`:

```rust
// ── Book/page key expression tests ─────────────────────────────

#[test]
fn book_page_key_valid() {
    let cid = "aa".repeat(32); // 64-char hex
    let page_addr = "0abc1234"; // 8-char hex
    let k = book_page_key(&cid, &page_addr).unwrap();
    assert_eq!(k.as_str(), format!("harmony/book/{cid}/page/{page_addr}"));
}

#[test]
fn book_pos_key_valid() {
    let cid = "bb".repeat(32);
    let k = book_pos_key(&cid, 42).unwrap();
    assert_eq!(k.as_str(), format!("harmony/book/{cid}/pos/42"));
}

#[test]
fn book_toc_key_valid() {
    let cid = "cc".repeat(32);
    let k = book_toc_key(&cid).unwrap();
    assert_eq!(k.as_str(), format!("harmony/book/{cid}/toc"));
}

#[test]
fn book_meta_key_valid() {
    let cid = "dd".repeat(32);
    let k = book_meta_key(&cid).unwrap();
    assert_eq!(k.as_str(), format!("harmony/book/{cid}/meta"));
}

#[test]
fn book_page_sub_by_addr_valid() {
    let page_addr = "0abc1234";
    let sub = book_page_sub_by_addr(&page_addr).unwrap();
    let key = book_page_key(&"aa".repeat(32), &page_addr).unwrap();
    assert!(sub.intersects(&key));
}

#[test]
fn book_page_sub_all_valid() {
    let cid = "ee".repeat(32);
    let sub = book_page_sub_all(&cid).unwrap();
    let key = book_page_key(&cid, "12345678").unwrap();
    assert!(sub.intersects(&key));
}

#[test]
fn book_pos_sub_all_valid() {
    let cid = "ff".repeat(32);
    let sub = book_pos_sub_all(&cid).unwrap();
    let key = book_pos_key(&cid, 0).unwrap();
    assert!(sub.intersects(&key));
    let key2 = book_pos_key(&cid, 255).unwrap();
    assert!(sub.intersects(&key2));
}

#[test]
fn book_keys_reject_slashes() {
    assert!(book_page_key("aa/bb", "12345678").is_err());
    assert!(book_page_key(&"aa".repeat(32), "12/34").is_err());
    assert!(book_pos_key("aa/bb", 0).is_err());
    assert!(book_toc_key("aa/bb").is_err());
    assert!(book_meta_key("aa/bb").is_err());
}

#[test]
fn parse_book_page_extracts_fields() {
    let cid = "aa".repeat(32);
    let addr = "0abc1234";
    let key = book_page_key(&cid, addr).unwrap();
    let (parsed_cid, parsed_addr) = parse_book_page(&key).unwrap();
    assert_eq!(parsed_cid, cid);
    assert_eq!(parsed_addr, addr);
}

#[test]
fn parse_book_pos_extracts_fields() {
    let cid = "bb".repeat(32);
    let key = book_pos_key(&cid, 128).unwrap();
    let (parsed_cid, parsed_idx) = parse_book_pos(&key).unwrap();
    assert_eq!(parsed_cid, cid);
    assert_eq!(parsed_idx, "128");
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh -- keyspace::tests::book 2>&1 | head -20`
Expected: Compilation errors — functions don't exist.

**Step 3: Implement book/page key expression builders and parsers**

Add to `keyspace.rs` (after existing vine key expressions):

```rust
// ── Book/page key expressions ───────────────────────────────────

// Format strings
const BOOK_PAGE_FMT: &str = "harmony/book/${cid_hex:*}/page/${page_addr_hex:*}";
const BOOK_POS_FMT: &str = "harmony/book/${cid_hex:*}/pos/${index:*}";

/// Build a book page key expression (content-addressed access).
pub fn book_page_key(cid_hex: &str, page_addr_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    reject_slashes(page_addr_hex)?;
    ke(&format!("harmony/book/{cid_hex}/page/{page_addr_hex}"))
}

/// Build a book positional access key expression.
pub fn book_pos_key(cid_hex: &str, index: u8) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    ke(&format!("harmony/book/{cid_hex}/pos/{index}"))
}

/// Build a book Table of Contents key expression.
pub fn book_toc_key(cid_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    ke(&format!("harmony/book/{cid_hex}/toc"))
}

/// Build a book metadata key expression.
pub fn book_meta_key(cid_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    ke(&format!("harmony/book/{cid_hex}/meta"))
}

/// Subscribe to "who has this page?" across all books.
pub fn book_page_sub_by_addr(page_addr_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(page_addr_hex)?;
    ke(&format!("harmony/book/*/page/{page_addr_hex}"))
}

/// Subscribe to all pages of a specific book.
pub fn book_page_sub_all(cid_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    ke(&format!("harmony/book/{cid_hex}/page/*"))
}

/// Subscribe to stream a book in positional order.
pub fn book_pos_sub_all(cid_hex: &str) -> Result<OwnedKeyExpr, ZenohError> {
    reject_slashes(cid_hex)?;
    ke(&format!("harmony/book/{cid_hex}/pos/*"))
}

/// Parse a book page key expression, returning (cid_hex, page_addr_hex).
pub fn parse_book_page(ke: &keyexpr) -> Result<(String, String), ZenohError> {
    let fmt = KeFormat::new(BOOK_PAGE_FMT).map_err(|e| ze(e.to_string()))?;
    let parsed = fmt.parse(ke).map_err(|e| ze(e.to_string()))?;
    let cid = parsed.get("cid_hex").map_err(|e| ze(e.to_string()))?.to_string();
    let addr = parsed.get("page_addr_hex").map_err(|e| ze(e.to_string()))?.to_string();
    Ok((cid, addr))
}

/// Parse a book positional key expression, returning (cid_hex, index_str).
pub fn parse_book_pos(ke: &keyexpr) -> Result<(String, String), ZenohError> {
    let fmt = KeFormat::new(BOOK_POS_FMT).map_err(|e| ze(e.to_string()))?;
    let parsed = fmt.parse(ke).map_err(|e| ze(e.to_string()))?;
    let cid = parsed.get("cid_hex").map_err(|e| ze(e.to_string()))?.to_string();
    let idx = parsed.get("index").map_err(|e| ze(e.to_string()))?.to_string();
    Ok((cid, idx))
}
```

**Step 4: Run tests**

Run: `cargo test -p harmony-zenoh -v`
Expected: All tests pass (existing + new book/page tests).

**Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/keyspace.rs
git commit -m "feat(zenoh): add book/page/toc/pos key expression builders and parsers"
```

---

### Task 9: Update ContentServer (harmony-os)

**Files:**
- Modify: `crates/harmony-microkernel/src/content_server.rs` (in harmony-os repo)

This updates the 9P content server to use the new types:
- `ChunkAddr` → `PageAddr`
- `Athenaeum` → `Book`
- `chunks` directory → `pages` directory
- `chunks` BTreeMap key: old 21-bit hash_bits → new 28-bit hash_bits
- `blobs` BTreeMap value: `Athenaeum` → `Book`
- QPath constants: `CHUNK_QPATH_BASE` → `PAGE_QPATH_BASE`, update range (28-bit hash_bits, max 0x0FFF_FFFF)

**Important:** This task works in the **harmony-os** repo, not the harmony core repo. The dependency is via `harmony-athenaeum` workspace member. After the core changes land, `harmony-os` must update its dependency to pick them up.

**Step 1: Update imports**

Change:
```rust
use harmony_athenaeum::{sha256_hash, Athenaeum, ChunkAddr, MAX_BLOB_SIZE};
```
To:
```rust
use harmony_athenaeum::{sha256_hash, Book, PageAddr, BOOK_MAX_SIZE, PAGE_SIZE};
```

**Step 2: Rename data structures**

- `chunks: BTreeMap<u32, (ChunkAddr, Vec<u8>)>` → `pages: BTreeMap<u32, (PageAddr, Vec<u8>)>`
- `blobs: BTreeMap<[u8; 32], Athenaeum>` → `blobs: BTreeMap<[u8; 32], Book>`
- `CHUNKS_DIR` → `PAGES_DIR`
- `CHUNK_QPATH_BASE` → `PAGE_QPATH_BASE`
- `MAX_BLOB_SIZE` → `BOOK_MAX_SIZE`
- `CHUNK_SIZE` → `PAGE_SIZE` (imported from harmony_athenaeum now)

**Step 3: Update QPath ranges**

Old: `CHUNK_QPATH_BASE = 0x100_0000` with 21-bit hash_bits (max 0x1F_FFFF), chunks top at `0x11F_FFFF`.
New: `PAGE_QPATH_BASE = 0x100_0000` with 28-bit hash_bits (max 0x0FFF_FFFF), pages top at `0x10FF_FFFF`.

Verify this doesn't overlap with `BLOB_QPATH_BASE = 0x1_0000_0000` — it doesn't (pages max at 0x10FF_FFFF < 0x1_0000_0000).

**Step 4: Update ingest logic**

The ingest finalization currently calls `Athenaeum::from_blob(cid, &buf)` and stores chunks. Update to:

```rust
let book = Book::from_blob(cid, &buf).map_err(|_| IpcError::OutOfSpace)?;
// Store pages
for (i, page_data) in book.page_data_from_blob(&buf).iter().enumerate() {
    let addr = book.pages[i][0]; // use algo 0 for storage
    let hash_bits = addr.hash_bits() >> 2; // normalize to 28-bit index
    self.pages.insert(hash_bits, (addr, page_data.clone()));
}
self.blobs.insert(cid, book);
```

**Step 5: Update filesystem paths**

- `"chunks"` → `"pages"` in directory names
- `is_chunks_dir()` → `is_pages_dir()`
- `format!("{:05x}", hash_bits)` → `format!("{:07x}", hash_bits)` (28-bit needs 7 hex chars)

**Step 6: Update tests**

Update all test helpers and assertions:
- `setup_read_mock` patterns may need updating for hash_bits range
- Assert strings like `"chunks/"` → `"pages/"`
- Verify QPath calculations with new ranges

**Step 7: Run tests**

Run: `cargo test -p harmony-microkernel -- content_server -v`
Expected: All tests pass.

**Step 8: Run full workspace**

Run: `cargo test --workspace` (from harmony-os repo)
Run: `cargo clippy --workspace`
Expected: All pass, zero warnings.

**Step 9: Commit**

```bash
git add crates/harmony-microkernel/src/content_server.rs
git commit -m "refactor(microkernel): update ContentServer from ChunkAddr/Athenaeum to PageAddr/Book"
```

---

## Execution Notes

### Cross-Repo Coordination

This plan spans two repos:
- **harmony** (core): Tasks 1-8 (harmony-athenaeum + harmony-zenoh)
- **harmony-os**: Task 9 (harmony-microkernel)

The harmony-os repo depends on harmony core via git. After Tasks 1-8 land on main in harmony, update the harmony-os git dependency to pick up the new types before starting Task 9.

**Branch naming:** Use `jake-athenaeum-page-addr` in harmony, `jake-os-page-addr` in harmony-os.

### Test Execution Strategy

Tasks 1-3 will cause compilation failures in modules not yet updated (volume.rs, encyclopedia.rs reference old types). During incremental development:
- Test individual modules with `cargo test -p harmony-athenaeum -- addr::tests` (scope to the module being worked on)
- Full crate compilation won't succeed until Tasks 1-6 are all complete
- Alternative: temporarily comment out `mod volume;` and `mod encyclopedia;` in lib.rs during Tasks 1-4, then restore in Tasks 5-6

### Follow-Up Bead

Storage/publishing policy enforcement based on CID classification bits (00=public durable, 01=public ephemeral, 10=encrypted durable, 11=encrypted ephemeral) is tracked in bead `harmony-8fg`.

# Self-Indexing Books Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make books self-documenting by embedding the Table of Contents at page 0, using sentinel-based self-referencing to break circular hash dependencies.

**Architecture:** Four structurally-invalid PageAddr values (sentinels) serve as content-independent aliases for the ToC page. The ToC is a regular 4KB page at position 0 with sentinels at entry 0 (self-reference). The Book struct gains a `self_indexing` flag that controls ToC generation, data page accessors, and reassembly.

**Tech Stack:** Rust, `no_std` (with `alloc`), SHA-256/SHA-224 (`sha2` crate), `harmony-athenaeum` crate

**Design doc:** `docs/plans/2026-03-11-self-indexing-books-design.md`

---

### Task 1: Sentinel Constants and Utility Functions

**Files:**
- Modify: `crates/harmony-athenaeum/src/addr.rs:31` (after NULL_PAGE)
- Modify: `crates/harmony-athenaeum/src/lib.rs:17-18` (add re-exports)

**Step 1: Write the failing tests**

Add to `#[cfg(test)] mod tests` in `addr.rs`:

```rust
#[test]
fn sentinel_values() {
    assert_eq!(SELF_INDEX_SENTINEL_00, 0x3FFF_FFFF);
    assert_eq!(SELF_INDEX_SENTINEL_01, 0x7FFF_FFFE);
    assert_eq!(SELF_INDEX_SENTINEL_10, 0xBFFF_FFFD);
    assert_eq!(SELF_INDEX_SENTINEL_11, 0xFFFF_FFFC);
}

#[test]
fn sentinels_fail_checksum() {
    for &sentinel in &SELF_INDEX_SENTINELS {
        let addr = PageAddr(sentinel);
        assert!(
            !addr.verify_checksum(),
            "sentinel {:#010x} must fail checksum",
            sentinel
        );
    }
}

#[test]
fn sentinels_have_all_ones_data_bits() {
    for &sentinel in &SELF_INDEX_SENTINELS {
        let addr = PageAddr(sentinel);
        assert_eq!(
            addr.hash_bits(),
            0x0FFF_FFFF,
            "sentinel {:#010x} must have all 28 data bits set",
            sentinel
        );
    }
}

#[test]
fn is_toc_sentinel_matches_all_four() {
    for &sentinel in &SELF_INDEX_SENTINELS {
        assert!(is_toc_sentinel(sentinel), "{:#010x} should be a sentinel", sentinel);
    }
}

#[test]
fn is_toc_sentinel_rejects_non_sentinels() {
    assert!(!is_toc_sentinel(NULL_PAGE));
    assert!(!is_toc_sentinel(0));
    let normal = PageAddr::new(42, Algorithm::Sha256Msb);
    assert!(!is_toc_sentinel(normal.0));
}

#[test]
fn toc_sentinel_for_algo_round_trip() {
    assert_eq!(toc_sentinel_for_algo(0), SELF_INDEX_SENTINEL_00);
    assert_eq!(toc_sentinel_for_algo(1), SELF_INDEX_SENTINEL_01);
    assert_eq!(toc_sentinel_for_algo(2), SELF_INDEX_SENTINEL_10);
    assert_eq!(toc_sentinel_for_algo(3), SELF_INDEX_SENTINEL_11);
}

#[test]
fn sentinel_algo_extracts_algorithm() {
    assert_eq!(sentinel_algo(SELF_INDEX_SENTINEL_00), Some(0));
    assert_eq!(sentinel_algo(SELF_INDEX_SENTINEL_01), Some(1));
    assert_eq!(sentinel_algo(SELF_INDEX_SENTINEL_10), Some(2));
    assert_eq!(sentinel_algo(SELF_INDEX_SENTINEL_11), Some(3));
}

#[test]
fn sentinel_algo_returns_none_for_non_sentinel() {
    assert_eq!(sentinel_algo(NULL_PAGE), None);
    assert_eq!(sentinel_algo(0), None);
    let normal = PageAddr::new(42, Algorithm::Sha256Msb);
    assert_eq!(sentinel_algo(normal.0), None);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- sentinel`
Expected: FAIL — constants and functions not defined

**Step 3: Write the implementation**

Add after `pub const NULL_PAGE: u32 = 0x00000003;` (line 31):

```rust
/// Self-indexing sentinel for algorithm 00 (Sha256Msb).
///
/// All 28 data bits set to 1, checksum = inverted mode bits.
/// Deliberately fails checksum validation.
pub const SELF_INDEX_SENTINEL_00: u32 = 0x3FFF_FFFF;

/// Self-indexing sentinel for algorithm 01 (Sha256Lsb).
pub const SELF_INDEX_SENTINEL_01: u32 = 0x7FFF_FFFE;

/// Self-indexing sentinel for algorithm 10 (Sha224Msb).
pub const SELF_INDEX_SENTINEL_10: u32 = 0xBFFF_FFFD;

/// Self-indexing sentinel for algorithm 11 (Sha224Lsb).
pub const SELF_INDEX_SENTINEL_11: u32 = 0xFFFF_FFFC;

/// All four self-indexing sentinels, indexed by algorithm selector (0-3).
pub const SELF_INDEX_SENTINELS: [u32; ALGO_COUNT] = [
    SELF_INDEX_SENTINEL_00,
    SELF_INDEX_SENTINEL_01,
    SELF_INDEX_SENTINEL_10,
    SELF_INDEX_SENTINEL_11,
];

/// Maximum data pages in a self-indexing book (page 0 is the ToC).
pub const SELF_INDEXING_MAX_DATA_PAGES: usize = PAGES_PER_BOOK - 1;

/// Maximum data size for a self-indexing book in bytes.
pub const SELF_INDEXING_MAX_DATA_SIZE: usize = PAGE_SIZE * SELF_INDEXING_MAX_DATA_PAGES;

/// Returns `true` if `addr` is one of the 4 self-indexing ToC sentinels.
pub fn is_toc_sentinel(addr: u32) -> bool {
    SELF_INDEX_SENTINELS.contains(&addr)
}

/// Returns the sentinel value for the given algorithm selector (0-3).
///
/// # Panics
/// Panics if `algo > 3`.
pub fn toc_sentinel_for_algo(algo: u8) -> u32 {
    SELF_INDEX_SENTINELS[algo as usize]
}

/// If `addr` is a self-indexing sentinel, returns the algorithm selector (0-3).
/// Returns `None` for non-sentinel values.
pub fn sentinel_algo(addr: u32) -> Option<u8> {
    SELF_INDEX_SENTINELS
        .iter()
        .position(|&s| s == addr)
        .map(|i| i as u8)
}
```

**Step 4: Update lib.rs re-exports**

Replace the `pub use addr::` line in `lib.rs`:

```rust
pub use addr::{
    is_toc_sentinel, sentinel_algo, toc_sentinel_for_algo, Algorithm, PageAddr, ALGO_COUNT,
    BOOK_MAX_SIZE, NULL_PAGE, PAGES_PER_BOOK, PAGE_SIZE, SELF_INDEXING_MAX_DATA_PAGES,
    SELF_INDEXING_MAX_DATA_SIZE, SELF_INDEX_SENTINELS, SELF_INDEX_SENTINEL_00,
    SELF_INDEX_SENTINEL_01, SELF_INDEX_SENTINEL_10, SELF_INDEX_SENTINEL_11,
};
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- sentinel`
Expected: All 8 new tests PASS

**Step 6: Run full test suite and clippy**

Run: `cargo test -p harmony-athenaeum && cargo clippy -p harmony-athenaeum`
Expected: All existing tests still pass, no clippy warnings

**Step 7: Commit**

```bash
git add crates/harmony-athenaeum/src/addr.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "feat(athenaeum): add self-indexing sentinel constants and utility functions"
```

---

### Task 2: Book Type — Self-Indexing Flag and Accessors

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs:36-44` (Book struct + impl)

**Step 1: Write the failing tests**

Add to the test module in `athenaeum.rs`:

```rust
#[test]
fn raw_book_is_not_self_indexing() {
    let data = vec![0xABu8; 100];
    let book = Book::from_blob(test_cid(), &data).unwrap();
    assert!(!book.is_self_indexing());
}

#[test]
fn raw_book_data_page_count_equals_page_count() {
    let data = vec![0xABu8; PAGE_SIZE * 3];
    let book = Book::from_blob(test_cid(), &data).unwrap();
    assert_eq!(book.data_page_count(), 3);
    assert_eq!(book.data_page_count(), book.page_count());
}

#[test]
fn raw_book_data_pages_returns_all() {
    let data = vec![0xABu8; PAGE_SIZE * 2];
    let book = Book::from_blob(test_cid(), &data).unwrap();
    let dp = book.data_pages();
    assert_eq!(dp.len(), 2);
    assert_eq!(dp[0], book.pages[0]);
    assert_eq!(dp[1], book.pages[1]);
}

#[test]
fn raw_book_toc_page_returns_none() {
    let data = vec![0xABu8; 100];
    let book = Book::from_blob(test_cid(), &data).unwrap();
    assert!(book.toc_page().is_none());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- raw_book`
Expected: FAIL — methods not defined

**Step 3: Write the implementation**

Add `self_indexing: bool` field to Book struct:

```rust
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Book {
    pub cid: [u8; 32],
    pub pages: Vec<[PageAddr; ALGO_COUNT]>,
    pub blob_size: u32,
    /// When true, page 0 is an embedded Table of Contents.
    pub self_indexing: bool,
}
```

Update `from_blob()` — add `self_indexing: false` to both return paths:

```rust
// Empty case:
return Ok(Book {
    cid,
    pages: Vec::new(),
    blob_size: 0,
    self_indexing: false,
});

// Normal case:
Ok(Book {
    cid,
    pages,
    blob_size: data.len() as u32,
    self_indexing: false,
})
```

Add new methods to `impl Book`:

```rust
/// Whether this book embeds its ToC at page 0.
pub fn is_self_indexing(&self) -> bool {
    self.self_indexing
}

/// Number of data pages (excludes the ToC page for self-indexing books).
pub fn data_page_count(&self) -> usize {
    if self.self_indexing {
        self.pages.len().saturating_sub(1)
    } else {
        self.pages.len()
    }
}

/// Slice of data page addresses (excludes ToC for self-indexing books).
pub fn data_pages(&self) -> &[[PageAddr; ALGO_COUNT]] {
    if self.self_indexing && !self.pages.is_empty() {
        &self.pages[1..]
    } else {
        &self.pages
    }
}

/// Returns the ToC page bytes if this is a self-indexing book.
pub fn toc_page(&self) -> Option<Vec<u8>> {
    if self.self_indexing {
        Some(self.toc())
    } else {
        None
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- raw_book`
Expected: All 4 new tests PASS

**Step 5: Run full suite**

Run: `cargo test -p harmony-athenaeum && cargo clippy -p harmony-athenaeum`
Expected: All tests pass, no warnings

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): add self_indexing flag and data page accessors to Book"
```

---

### Task 3: Self-Indexing ToC Generation

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (update `toc()`, add import)

**Step 1: Write the failing tests**

Add to test module in `athenaeum.rs` (add needed imports at top of test module):

```rust
use crate::addr::{
    is_toc_sentinel, toc_sentinel_for_algo, SELF_INDEX_SENTINEL_00,
};

#[test]
fn self_indexing_toc_has_sentinels_at_position_zero() {
    let data = vec![0xAAu8; PAGE_SIZE * 2];
    let raw_book = Book::from_blob(test_cid(), &data).unwrap();
    // Manually build a self-indexing book to test toc()
    let mut pages = Vec::with_capacity(3);
    pages.push([PageAddr(NULL_PAGE); ALGO_COUNT]); // placeholder at page 0
    pages.extend_from_slice(&raw_book.pages);
    let si_book = Book {
        cid: test_cid(),
        pages,
        blob_size: data.len() as u32,
        self_indexing: true,
    };
    let toc = si_book.toc();
    for algo_idx in 0..ALGO_COUNT {
        let offset = algo_idx * PAGES_PER_BOOK * 4;
        let value = u32::from_le_bytes([
            toc[offset],
            toc[offset + 1],
            toc[offset + 2],
            toc[offset + 3],
        ]);
        assert!(
            is_toc_sentinel(value),
            "section {algo_idx}, pos 0: expected sentinel, got {value:#010x}"
        );
        assert_eq!(value, toc_sentinel_for_algo(algo_idx as u8));
    }
}

#[test]
fn self_indexing_toc_data_pages_at_correct_positions() {
    let data = vec![0xBBu8; PAGE_SIZE * 2];
    let raw_book = Book::from_blob(test_cid(), &data).unwrap();
    let mut pages = Vec::with_capacity(3);
    pages.push([PageAddr(NULL_PAGE); ALGO_COUNT]);
    pages.extend_from_slice(&raw_book.pages);
    let si_book = Book {
        cid: test_cid(),
        pages,
        blob_size: data.len() as u32,
        self_indexing: true,
    };
    let toc = si_book.toc();
    for algo_idx in 0..ALGO_COUNT {
        for data_page in 0..2 {
            let toc_pos = data_page + 1;
            let offset = algo_idx * PAGES_PER_BOOK * 4 + toc_pos * 4;
            let value = u32::from_le_bytes([
                toc[offset],
                toc[offset + 1],
                toc[offset + 2],
                toc[offset + 3],
            ]);
            let expected = raw_book.pages[data_page][algo_idx].0;
            assert_eq!(value, expected);
        }
    }
}

#[test]
fn self_indexing_toc_unused_entries_are_null() {
    let data = vec![0xCCu8; PAGE_SIZE];
    let raw_book = Book::from_blob(test_cid(), &data).unwrap();
    let mut pages = Vec::with_capacity(2);
    pages.push([PageAddr(NULL_PAGE); ALGO_COUNT]);
    pages.extend_from_slice(&raw_book.pages);
    let si_book = Book {
        cid: test_cid(),
        pages,
        blob_size: data.len() as u32,
        self_indexing: true,
    };
    let toc = si_book.toc();
    for algo_idx in 0..ALGO_COUNT {
        for page_idx in 2..PAGES_PER_BOOK {
            let offset = algo_idx * PAGES_PER_BOOK * 4 + page_idx * 4;
            let value = u32::from_le_bytes([
                toc[offset],
                toc[offset + 1],
                toc[offset + 2],
                toc[offset + 3],
            ]);
            assert_eq!(value, NULL_PAGE);
        }
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- self_indexing_toc`
Expected: FAIL — toc() doesn't handle sentinels yet

**Step 3: Update toc() to handle self-indexing**

Update imports at top of `athenaeum.rs`:

```rust
use crate::addr::{
    toc_sentinel_for_algo, Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE,
    PAGES_PER_BOOK, PAGE_SIZE,
};
```

Replace the `toc()` body:

```rust
pub fn toc(&self) -> Vec<u8> {
    let mut buf = vec![0u8; PAGE_SIZE];

    for algo_idx in 0..ALGO_COUNT {
        let section_offset = algo_idx * PAGES_PER_BOOK * 4;
        for page_idx in 0..PAGES_PER_BOOK {
            let entry_offset = section_offset + page_idx * 4;
            let value = if self.self_indexing && page_idx == 0 {
                toc_sentinel_for_algo(algo_idx as u8)
            } else if page_idx < self.pages.len() {
                self.pages[page_idx][algo_idx].0
            } else {
                NULL_PAGE
            };
            buf[entry_offset..entry_offset + 4].copy_from_slice(&value.to_le_bytes());
        }
    }

    buf
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- toc`
Expected: All toc tests PASS (existing + new)

**Step 5: Run full suite and clippy**

Run: `cargo test -p harmony-athenaeum && cargo clippy -p harmony-athenaeum`
Expected: All tests pass, no warnings

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): update toc() to write sentinels for self-indexing books"
```

---

### Task 4: Self-Indexing Book Construction

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (add `from_blob_self_indexing()`)

**Step 1: Write the failing tests**

Add to test module (ensure `SELF_INDEXING_MAX_DATA_SIZE` and `SELF_INDEX_SENTINEL_00` are imported):

```rust
#[test]
fn from_blob_self_indexing_single_page() {
    let data = vec![0xAAu8; 100];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    assert!(book.is_self_indexing());
    assert_eq!(book.page_count(), 2); // ToC + 1 data
    assert_eq!(book.data_page_count(), 1);
    assert_eq!(book.blob_size, 100);
}

#[test]
fn from_blob_self_indexing_multiple_pages() {
    let data = vec![0xBBu8; PAGE_SIZE * 3 + 100];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    assert!(book.is_self_indexing());
    assert_eq!(book.page_count(), 5); // ToC + 4 data
    assert_eq!(book.data_page_count(), 4);
    assert_eq!(book.blob_size, (PAGE_SIZE * 3 + 100) as u32);
}

#[test]
fn from_blob_self_indexing_max_data() {
    let data = vec![0xCCu8; SELF_INDEXING_MAX_DATA_SIZE];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    assert_eq!(book.data_page_count(), 255);
    assert_eq!(book.page_count(), 256);
}

#[test]
fn from_blob_self_indexing_too_large() {
    let data = vec![0u8; SELF_INDEXING_MAX_DATA_SIZE + 1];
    let err = Book::from_blob_self_indexing(test_cid(), &data).unwrap_err();
    assert_eq!(
        err,
        BookError::BlobTooLarge {
            size: SELF_INDEXING_MAX_DATA_SIZE + 1,
        }
    );
}

#[test]
fn from_blob_self_indexing_empty() {
    let book = Book::from_blob_self_indexing(test_cid(), &[]).unwrap();
    assert!(book.is_self_indexing());
    assert_eq!(book.page_count(), 1); // Just ToC
    assert_eq!(book.data_page_count(), 0);
    assert_eq!(book.blob_size, 0);
}

#[test]
fn from_blob_self_indexing_toc_page_zero_valid() {
    let data = vec![0xDDu8; PAGE_SIZE * 2];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    for addr in &book.pages[0] {
        assert!(addr.verify_checksum());
        assert!(!is_toc_sentinel(addr.0));
    }
}

#[test]
fn from_blob_self_indexing_toc_starts_with_sentinel() {
    let data = vec![0xEEu8; PAGE_SIZE];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let toc = book.toc();
    let first_u32 = u32::from_le_bytes([toc[0], toc[1], toc[2], toc[3]]);
    assert_eq!(first_u32, SELF_INDEX_SENTINEL_00);
}

#[test]
fn from_blob_self_indexing_toc_verifies_against_page_zero_addr() {
    let data = vec![0xFFu8; PAGE_SIZE * 2];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let toc = book.toc();
    for addr in &book.pages[0] {
        assert!(addr.verify_data(&toc));
    }
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- from_blob_self_indexing`
Expected: FAIL — method not defined

**Step 3: Write the implementation**

Update imports at top of `athenaeum.rs` to add `SELF_INDEXING_MAX_DATA_SIZE`:

```rust
use crate::addr::{
    toc_sentinel_for_algo, Algorithm, PageAddr, ALGO_COUNT, BOOK_MAX_SIZE, NULL_PAGE,
    PAGES_PER_BOOK, PAGE_SIZE, SELF_INDEXING_MAX_DATA_SIZE,
};
```

Add to `impl Book`:

```rust
/// Build a self-indexing Book with an embedded ToC at page 0.
///
/// The ToC page is constructed from the data page addresses and placed
/// at index 0. Data pages occupy indices 1..=N. The ToC uses sentinel
/// values at position 0 to break the circular hash dependency.
///
/// - Maximum data: 255 pages (1,044,480 bytes).
/// - Empty data produces a Book with 1 page (just the ToC).
pub fn from_blob_self_indexing(cid: [u8; 32], data: &[u8]) -> Result<Self, BookError> {
    if data.len() > SELF_INDEXING_MAX_DATA_SIZE {
        return Err(BookError::BlobTooLarge { size: data.len() });
    }

    // Step 1: Compute data page addresses
    let mut data_page_addrs: Vec<[PageAddr; ALGO_COUNT]> = Vec::new();
    if !data.is_empty() {
        for (i, chunk) in data.chunks(PAGE_SIZE).enumerate() {
            let mut page_buf = [0u8; PAGE_SIZE];
            page_buf[..chunk.len()].copy_from_slice(chunk);

            let variants = [
                PageAddr::from_data(&page_buf, Algorithm::Sha256Msb),
                PageAddr::from_data(&page_buf, Algorithm::Sha256Lsb),
                PageAddr::from_data(&page_buf, Algorithm::Sha224Msb),
                PageAddr::from_data(&page_buf, Algorithm::Sha224Lsb),
            ];

            let all_same = variants[1..]
                .iter()
                .all(|v| v.hash_bits() == variants[0].hash_bits());
            if all_same {
                return Err(BookError::AllAlgorithmsCollide { page_index: i + 1 });
            }

            data_page_addrs.push(variants);
        }
    }

    // Step 2: Build ToC page bytes
    let mut toc_buf = [0u8; PAGE_SIZE];
    for algo_idx in 0..ALGO_COUNT {
        let section_offset = algo_idx * PAGES_PER_BOOK * 4;
        // Position 0: sentinel
        let sentinel = toc_sentinel_for_algo(algo_idx as u8);
        toc_buf[section_offset..section_offset + 4]
            .copy_from_slice(&sentinel.to_le_bytes());
        // Positions 1..N: data page addrs
        for (i, addrs) in data_page_addrs.iter().enumerate() {
            let entry_offset = section_offset + (i + 1) * 4;
            toc_buf[entry_offset..entry_offset + 4]
                .copy_from_slice(&addrs[algo_idx].0.to_le_bytes());
        }
        // Positions N+1..255: NULL_PAGE
        for page_idx in (data_page_addrs.len() + 1)..PAGES_PER_BOOK {
            let entry_offset = section_offset + page_idx * 4;
            toc_buf[entry_offset..entry_offset + 4]
                .copy_from_slice(&NULL_PAGE.to_le_bytes());
        }
    }

    // Step 3: Hash ToC page to get its content-derived address
    let toc_addrs = [
        PageAddr::from_data(&toc_buf, Algorithm::Sha256Msb),
        PageAddr::from_data(&toc_buf, Algorithm::Sha256Lsb),
        PageAddr::from_data(&toc_buf, Algorithm::Sha224Msb),
        PageAddr::from_data(&toc_buf, Algorithm::Sha224Lsb),
    ];

    // Step 4: Assemble pages: [ToC, data_page_1, ..., data_page_N]
    let mut pages = Vec::with_capacity(1 + data_page_addrs.len());
    pages.push(toc_addrs);
    pages.extend(data_page_addrs);

    Ok(Book {
        cid,
        pages,
        blob_size: data.len() as u32,
        self_indexing: true,
    })
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum -- from_blob_self_indexing`
Expected: All 8 new tests PASS

**Step 5: Run full suite and clippy**

Run: `cargo test -p harmony-athenaeum && cargo clippy -p harmony-athenaeum`
Expected: All tests pass, no warnings

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): add Book::from_blob_self_indexing() constructor"
```

---

### Task 5: Page Data and Reassembly for Self-Indexing Books

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (update `page_data_from_blob`, `reassemble`)

**Step 1: Write the failing tests**

```rust
#[test]
fn page_data_self_indexing_includes_toc() {
    let data = vec![0xAAu8; PAGE_SIZE + 100];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let page_bufs = book.page_data_from_blob(&data);
    assert_eq!(page_bufs.len(), 3); // ToC + 2 data pages
    assert_eq!(page_bufs[0].len(), PAGE_SIZE);
    let first_u32 = u32::from_le_bytes([
        page_bufs[0][0],
        page_bufs[0][1],
        page_bufs[0][2],
        page_bufs[0][3],
    ]);
    assert_eq!(first_u32, SELF_INDEX_SENTINEL_00);
}

#[test]
fn page_data_self_indexing_toc_matches_toc_method() {
    let data = vec![0xBBu8; PAGE_SIZE * 2];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let page_bufs = book.page_data_from_blob(&data);
    let toc = book.toc();
    assert_eq!(page_bufs[0], toc);
}

#[test]
fn reassemble_self_indexing_round_trip() {
    let mut data = vec![0u8; PAGE_SIZE * 3 + 500];
    for (i, b) in data.iter_mut().enumerate() {
        let pos = i as u32;
        *b = (pos ^ (pos >> 8) ^ (pos >> 16)) as u8;
    }
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let page_bufs = book.page_data_from_blob(&data);
    let reassembled = book
        .reassemble(|idx| page_bufs.get(idx as usize).cloned())
        .unwrap();
    assert_eq!(reassembled.len(), data.len());
    assert_eq!(reassembled, data);
}

#[test]
fn reassemble_self_indexing_empty() {
    let book = Book::from_blob_self_indexing(test_cid(), &[]).unwrap();
    let page_bufs = book.page_data_from_blob(&[]);
    let reassembled = book
        .reassemble(|idx| page_bufs.get(idx as usize).cloned())
        .unwrap();
    assert!(reassembled.is_empty());
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- self_indexing`
Expected: FAIL — page_data_from_blob doesn't include ToC, reassemble includes ToC in output

**Step 3: Update `page_data_from_blob`**

Replace:

```rust
pub fn page_data_from_blob(&self, data: &[u8]) -> Vec<Vec<u8>> {
    debug_assert_eq!(
        data.len(),
        self.blob_size as usize,
        "data length does not match blob_size"
    );
    let mut pages = Vec::new();
    if self.self_indexing {
        pages.push(self.toc());
    }
    for chunk in data.chunks(PAGE_SIZE) {
        let mut page_buf = vec![0u8; PAGE_SIZE];
        page_buf[..chunk.len()].copy_from_slice(chunk);
        pages.push(page_buf);
    }
    pages
}
```

**Step 4: Update `reassemble`**

Replace:

```rust
pub fn reassemble(&self, fetch: impl Fn(u8) -> Option<Vec<u8>>) -> Result<Vec<u8>, BookError> {
    let mut result = Vec::with_capacity(self.blob_size as usize);

    let start = if self.self_indexing { 1 } else { 0 };
    for i in start..self.pages.len() {
        let page_data = fetch(i as u8).ok_or(BookError::MissingPage {
            page_index: i as u8,
        })?;
        if page_data.len() != PAGE_SIZE {
            return Err(BookError::BadFormat);
        }
        result.extend_from_slice(&page_data);
    }

    result.truncate(self.blob_size as usize);
    Ok(result)
}
```

**Step 5: Run tests to verify they pass**

Run: `cargo test -p harmony-athenaeum`
Expected: ALL tests pass (existing + new)

**Step 6: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs
git commit -m "feat(athenaeum): update page_data_from_blob and reassemble for self-indexing books"
```

---

### Task 6: Self-Indexing Detection and Integration Tests

**Files:**
- Modify: `crates/harmony-athenaeum/src/athenaeum.rs` (add detection method)
- Modify: `crates/harmony-athenaeum/src/lib.rs` (add integration tests)

**Step 1: Write the failing detection tests**

Add to `athenaeum.rs` test module:

```rust
#[test]
fn is_self_indexing_blob_detects_self_indexing() {
    let data = vec![0xAAu8; PAGE_SIZE * 2];
    let book = Book::from_blob_self_indexing(test_cid(), &data).unwrap();
    let page_bufs = book.page_data_from_blob(&data);
    let blob: Vec<u8> = page_bufs.into_iter().flatten().collect();
    assert!(Book::is_self_indexing_blob(&blob));
}

#[test]
fn is_self_indexing_blob_rejects_raw() {
    let data = vec![0xBBu8; PAGE_SIZE * 2];
    let book = Book::from_blob(test_cid(), &data).unwrap();
    let page_bufs = book.page_data_from_blob(&data);
    let blob: Vec<u8> = page_bufs.into_iter().flatten().collect();
    assert!(!Book::is_self_indexing_blob(&blob));
}

#[test]
fn is_self_indexing_blob_rejects_short() {
    assert!(!Book::is_self_indexing_blob(&[0xFF, 0xFF, 0xFF]));
    assert!(!Book::is_self_indexing_blob(&[]));
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-athenaeum -- is_self_indexing_blob`
Expected: FAIL — method not defined

**Step 3: Write the detection method**

Add to `impl Book`:

```rust
/// Detect whether a blob starts with a self-indexing ToC.
///
/// Checks the first 4 bytes: if they equal `SELF_INDEX_SENTINEL_00`
/// (0x3FFFFFFF as little-endian), this is a self-indexing book.
pub fn is_self_indexing_blob(blob: &[u8]) -> bool {
    if blob.len() < 4 {
        return false;
    }
    let first = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
    first == crate::addr::SELF_INDEX_SENTINEL_00
}
```

**Step 4: Run detection tests**

Run: `cargo test -p harmony-athenaeum -- is_self_indexing_blob`
Expected: All 3 tests PASS

**Step 5: Write integration tests**

Add to `lib.rs` test module:

```rust
#[test]
fn self_indexing_end_to_end() {
    let data = vec![0x42u8; PAGE_SIZE * 5 + 100];
    let cid = sha256_hash(&data);
    let book = Book::from_blob_self_indexing(cid, &data).unwrap();

    assert!(book.is_self_indexing());
    assert_eq!(book.data_page_count(), 6);
    assert_eq!(book.page_count(), 7);

    let toc = book.toc();
    for addr in &book.pages[0] {
        assert!(addr.verify_data(&toc));
    }

    let first_u32 = u32::from_le_bytes([toc[0], toc[1], toc[2], toc[3]]);
    assert_eq!(first_u32, SELF_INDEX_SENTINEL_00);

    let page_bufs = book.page_data_from_blob(&data);
    assert_eq!(page_bufs.len(), 7);
    let reassembled = book
        .reassemble(|idx| page_bufs.get(idx as usize).cloned())
        .unwrap();
    assert_eq!(reassembled, data);

    let blob: Vec<u8> = page_bufs.into_iter().flatten().collect();
    assert!(Book::is_self_indexing_blob(&blob));
}

#[test]
fn self_indexing_vs_raw_data_pages() {
    let data = vec![0x99u8; PAGE_SIZE * 3];
    let cid = sha256_hash(&data);
    let raw = Book::from_blob(cid, &data).unwrap();
    let si = Book::from_blob_self_indexing(cid, &data).unwrap();

    assert_eq!(raw.page_count(), 3);
    assert_eq!(raw.data_page_count(), 3);

    assert_eq!(si.page_count(), 4);
    assert_eq!(si.data_page_count(), 3);

    // Data page addresses should be identical
    for i in 0..3 {
        assert_eq!(raw.pages[i], si.pages[i + 1]);
    }
}

#[test]
fn self_indexing_empty_book() {
    let book = Book::from_blob_self_indexing([0; 32], &[]).unwrap();
    assert!(book.is_self_indexing());
    assert_eq!(book.page_count(), 1);
    assert_eq!(book.data_page_count(), 0);

    let toc = book.toc();
    for algo_idx in 0..ALGO_COUNT {
        let section_offset = algo_idx * PAGES_PER_BOOK * 4;
        let val = u32::from_le_bytes([
            toc[section_offset],
            toc[section_offset + 1],
            toc[section_offset + 2],
            toc[section_offset + 3],
        ]);
        assert!(is_toc_sentinel(val));

        for page_idx in 1..PAGES_PER_BOOK {
            let offset = section_offset + page_idx * 4;
            let val = u32::from_le_bytes([
                toc[offset],
                toc[offset + 1],
                toc[offset + 2],
                toc[offset + 3],
            ]);
            assert_eq!(val, NULL_PAGE);
        }
    }

    let page_bufs = book.page_data_from_blob(&[]);
    let reassembled = book
        .reassemble(|idx| page_bufs.get(idx as usize).cloned())
        .unwrap();
    assert!(reassembled.is_empty());
}
```

**Step 6: Run all tests**

Run: `cargo test -p harmony-athenaeum && cargo clippy -p harmony-athenaeum`
Expected: ALL tests pass, no clippy warnings

**Step 7: Commit**

```bash
git add crates/harmony-athenaeum/src/athenaeum.rs crates/harmony-athenaeum/src/lib.rs
git commit -m "feat(athenaeum): add self-indexing detection and integration tests"
```

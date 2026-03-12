# Self-Indexing Books Design

## Problem

A Book's Table of Contents (ToC) is currently a sidecar — a separate 4KB page
listing the PageAddrs of every page in the book. This makes books dependent on
external metadata to be useful. Embedding the ToC inside the book (at page 0)
makes every book self-documenting, but creates a circular hash dependency: the
ToC contains PageAddrs derived from page content hashes, and if the ToC is
itself a page, its own PageAddr depends on its content, which contains its own
PageAddr.

## Solution: Sentinel-Based Self-Referencing

Four structurally-invalid PageAddr values serve as content-independent aliases
for the ToC page. Because sentinels are fixed constants — not derived from the
ToC's content — the circular dependency is broken.

### Sentinel Construction

The PageAddr checksum is the XOR-fold of all 15 two-bit pairs of bits 31-2.
For a value with mode bits `MM` and 28 data bits all `1`, the correct checksum
is `MM` itself (14 pairs of `11` XOR to `00`, then XOR with `MM` = `MM`).

By storing the **inverted** mode bits as the checksum, the sentinel always
fails validation:

| Algo | Sentinel (hex) | Binary layout | Correct checksum | Stored checksum |
|------|---------------|---------------|-----------------|-----------------|
| `00` | `0x3FFFFFFF` | `00`-28×`1`-`11` | `00` | `11` |
| `01` | `0x7FFFFFFE` | `01`-28×`1`-`10` | `01` | `10` |
| `10` | `0xBFFFFFFD` | `10`-28×`1`-`01` | `10` | `01` |
| `11` | `0xFFFFFFFC` | `11`-28×`1`-`00` | `11` | `00` |

Invariant: XOR result `00` = valid page, XOR result = inverted mode = ToC
sentinel, and `01`/`10` remain genuinely invalid. Both valid pages and sentinels
benefit from one bit of "checksum" — they're a single bit-flip apart, with the
other two states being corruption.

These sit alongside the existing `NULL_PAGE = 0x00000003` (all data bits zero,
checksum fails).

### Utility Functions

- `is_toc_sentinel(addr: u32) -> bool` — matches any of the 4 sentinels
- `toc_sentinel_for_algo(algo: u8) -> u32` — returns sentinel for algorithm 0-3
- `sentinel_algo(addr: u32) -> Option<u8>` — if sentinel, which algorithm?

## ToC Page Layout (4096 bytes)

The ToC is a regular 4KB page at position 0. Four 1KB sections, one per
algorithm encoding:

```
Offset 0x000..0x400:  Section 00 — 256 × 4-byte PageAddrs
Offset 0x400..0x800:  Section 01 — 256 × 4-byte PageAddrs
Offset 0x800..0xC00:  Section 10 — 256 × 4-byte PageAddrs
Offset 0xC00..0x1000: Section 11 — 256 × 4-byte PageAddrs
```

Within each section, entry `[n]` is the PageAddr for page `n` in that encoding:

| Entry | Value | Meaning |
|-------|-------|---------|
| `[0]` | That section's sentinel | Self-reference ("I am the ToC") |
| `[n]` where `n < page_count` | Content-derived PageAddr | Normal page |
| `[n]` where `n >= page_count` | `NULL_PAGE (0x00000003)` | Page doesn't exist |

**Instant detection:** The first 4 bytes of any self-indexing book are
`0x3FFFFFFF`. A single `u32` read at offset 0 distinguishes self-indexing
books from raw books.

## Book Type Changes

The `Book` struct gains a `self_indexing` flag. When enabled, 260 PageAddr
entries exist in memory: 256 content-derived (including the ToC page's real
address at index 0) + 4 sentinel aliases that resolve to page 0.

### Construction Flow

1. Accept up to 255 data pages (pages 1-255)
2. Compute content-derived PageAddrs for each data page (all 4 algos)
3. Build the ToC page: sentinels at position 0, real PageAddrs at 1-255,
   `NULL_PAGE` for unused positions
4. Hash the ToC page to get its content-derived PageAddr (all 4 algos)
5. Store: `pages[0]` = ToC's content-derived PageAddr,
   `pages[1..=255]` = data PageAddrs
6. Register the 4 sentinel aliases pointing to page 0
7. Blob = `[toc_page | data_page_1 | ... | data_page_N | zero_padding...]` (1MB)

### Reading Flow

1. Read first 4 bytes. If `== 0x3FFFFFFF`, self-indexing book.
2. Parse page 0 as ToC: extract all PageAddrs, register 4 sentinels.
3. Pages 1-255 are data.

### API Additions

- `Book::data_pages()` — pages 1-255 only (skips ToC)
- `Book::data_page_count()` — 255 max for self-indexing, 256 for raw
- `Book::toc_page()` — returns ToC page bytes if self-indexing
- `Book::is_self_indexing() -> bool`

### PageAddr Lookup

When resolving a PageAddr against a book: check `toc_sentinels` first (O(4)).
If match, return page 0. Otherwise search `pages[0..256]` normally.

## Encyclopedia Integration

**Assembly:** Each self-indexing book holds 255 data pages. A 4GB file needs
~4,017 books instead of 4,096 (~0.4% overhead for self-documentation).

**Collision resolution:** ToC pages participate in the global PageAddr space
as normal content-derived addresses. Sentinel values are per-book aliases and
are NOT registered globally.

**Reassembly:** `Book::data_pages()` handles stripping automatically. Raw books
emit all 256 pages; self-indexing books emit pages 1-255.

**No changes to Volume or Encyclopedia structs.** Self-indexing logic is fully
encapsulated in `Book`.

## Edge Cases

| Scenario | Behavior |
|----------|----------|
| Empty book (0 data pages) | Valid: ToC page only, all entries are sentinel/NULL_PAGE |
| Raw book starting with `0x3FFFFFFF` | Validate all 4 sentinels at offsets 0, 1024, 2048, 3072. A raw book matching all 4 is 128 bits of coincidence. |
| Corrupted ToC | Content-derived PageAddr won't match CID. Standard integrity check. |
| Sentinel in data PageAddr | Impossible: sentinels deliberately fail checksum. |

**Graceful degradation:** If sentinel validation fails, treat as raw book.

## Scope

Changes to `harmony-athenaeum` only:
- `addr.rs` — sentinel constants + utility functions
- `athenaeum.rs` — Book type, construction, reading, API additions
- Tests for all of the above

No changes to: Encyclopedia, Volume, CID, chunker, Zenoh keyspace.

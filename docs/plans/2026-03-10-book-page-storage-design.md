# Book/Page Storage Layer Redesign

## Goal

Replace the ChunkAddr/Athenaeum/Book addressing system with a simpler, higher-capacity PageAddr/Book/Encyclopedia model. Pages are always 4KB, no subdivision. Books are always 256 pages (1MB). Encyclopedias assign unambiguous addresses across collections of books using power-of-choice with 4 hash algorithms.

## Architecture

```
Library (collection name, top-level)
└── Encyclopedia (authoritative address blueprint, computed once)
    ├── Volume (recursive partition, rarely needed with 2^28 space)
    │   └── Book (single CID blob ≤1MB = 256 × 4KB pages)
    │       ├── Page 0..255 (4KB each, content-addressed)
    │       └── Table of Contents (4KB, all 4 address variants precomputed)
    └── Book (leaf — no Volume needed for small collections)
```

## PageAddr — 32-bit content address

```
┌──────────┬────────────────────────────────┬──────────┐
│ algo (2) │         hash_bits (28)         │ cksum(2) │
│ bits 31-30│        bits 29-2              │ bits 1-0 │
└──────────┴────────────────────────────────┴──────────┘
```

**Algorithm selector (bits 31-30):**

| Value | Algorithm | Window |
|-------|-----------|--------|
| `00` | SHA-256 | first 4 bytes |
| `01` | SHA-256 | last 4 bytes |
| `10` | SHA-224 | first 4 bytes |
| `11` | SHA-224 | last 4 bytes |

**Derivation:**

1. Hash the 4KB page content with the selected algorithm (SHA-256 or SHA-224).
2. Take the first or last 4 bytes of the hash digest (big-endian u32).
3. Extract bits 29-2 of that u32 — these are the 28-bit `hash_bits`.
4. Set bits 31-30 to the algorithm selector.
5. Compute checksum: XOR-fold bits 31-2 (30 bits) into 15 pairs, XOR all pairs → 2-bit result.
6. Set bits 1-0 to the checksum.

**Checksum verification:** Given any u32, split bits 31-2 into 15 pairs, XOR-fold, compare to bits 1-0. ~25% of random values pass. All valid PageAddrs are self-consistent.

**Content verification:** Re-derive hash_bits using the algorithm indicated by bits 31-30. Match confirms authenticity.

**Null sentinel:** `0x00000003` — bits 31-2 are all zero, XOR-fold = `00`, but checksum bits are `11`. Deliberately fails checksum validation. Means "this page position does not exist" (null pointer). Can never collide with a naturally-derived PageAddr.

**Constants:**

```rust
const PAGE_SIZE: usize = 4096;
const PAGES_PER_BOOK: usize = 256;
const BOOK_MAX_SIZE: usize = PAGE_SIZE * PAGES_PER_BOOK; // 1MB
const ALGO_COUNT: usize = 4;
const NULL_PAGE: u32 = 0x00000003;
```

**Birthday paradox threshold:** With 28-bit hash space (2^28 = 268M), collision probability reaches 50% at ~16,000 pages. A single book has at most 256 pages. Even large Encyclopedia collections with 4 algorithm choices have comfortable margin.

## Book — single CID blob chunked into pages

A Book represents one ContentId-addressed blob (up to 1MB) split into exactly 256 pages of 4KB each.

```rust
struct Book {
    cid: ContentId,       // 256-bit blob identifier
    pages: Vec<[PageAddr; 4]>,  // up to 256 entries, all 4 algo variants each
    blob_size: u32,       // actual byte count (≤ 1MB)
}
```

- Page count = `ceil(blob_size / 4096)`.
- Last page is zero-padded to 4KB for hashing, but `blob_size` tells how many bytes are real.
- Pages past the written content contain 4KB of zeroes and get the naturally-derived address of the zero page (SHA-256 of 4096 zero bytes = `5656...1b01`). These deduplicate across all books.
- The 20-bit length field in the CID encodes the actual content size.

**Table of Contents (ToC):** A 4KB page generated when a book is created.

```
ToC layout (4096 bytes):

Bytes 0..1024:     Section 0 — algo=00 addresses, 256 × 4-byte PageAddr
Bytes 1024..2048:  Section 1 — algo=01 addresses, 256 × 4-byte PageAddr
Bytes 2048..3072:  Section 2 — algo=10 addresses, 256 × 4-byte PageAddr
Bytes 3072..4096:  Section 3 — algo=11 addresses, 256 × 4-byte PageAddr
```

All 4 algorithm variants are precomputed for every page position, regardless of which address the Encyclopedia ultimately assigns. Unused positions (past the page count) are filled with `NULL_PAGE` (`0x00000003`).

The ToC is itself a content-addressable 4KB page associated with the book's CID.

**Replaces:** `Athenaeum` (retired name).

## Encyclopedia — authoritative address blueprint

The Encyclopedia assigns a single unambiguous PageAddr to each page across a collection of books, using power-of-choice collision resolution.

```rust
struct Encyclopedia {
    books: Vec<Book>,
    assignments: BTreeMap<u32, Assignment>,  // hash_bits → assignment
}

struct Assignment {
    book_index: usize,  // which book in the collection
    page_index: u8,     // position within that book (0-255)
    algo: u8,           // which of the 4 algorithms was chosen (0-3)
}
```

**Build process:**

1. Ingest books — each brings its precomputed ToC.
2. For each page across all books, try algo 0. If the 28-bit hash_bits are unoccupied, assign. If occupied by same content, dedup. If occupied by different content, try algo 1, 2, 3.
3. If all 4 collide with different content: `AllAlgorithmsCollide` error (astronomically rare).
4. Result: a complete, collision-free mapping computed once and valid forever.

**Page lookup outcomes:**

- **Hit** — exactly one assignment → return content.
- **Miss** — no assignment → page fault (not found).
- **Collision** — only possible from PageAddrs outside this Encyclopedia's scope. Internal assignments are collision-free by construction.

**Volume subdivision:** The existing binary-tree partitioning by SHA-256 content-hash bits remains available for extremely large collections, but triggers much less often with 2^28 address space (128× larger than old 2^21).

## Zenoh key expressions

New key expression builders added to `harmony-zenoh` keyspace:

```
Content-addressed access:
  harmony/book/{cid_hex}/page/{page_addr_hex}

Positional access:
  harmony/book/{cid_hex}/pos/{index}        ← index 0-255 (decimal)

Metadata:
  harmony/book/{cid_hex}/toc               ← 4KB Table of Contents
  harmony/book/{cid_hex}/meta              ← blob_size, page count, etc.

Wildcard patterns:
  harmony/book/*/page/{page_addr_hex}       ← "who has this page?"
  harmony/book/{cid_hex}/page/*             ← "all pages of this book"
  harmony/book/{cid_hex}/pos/*              ← "stream book in order"
```

`cid_hex` = 64-char hex of 256-bit ContentId. `page_addr_hex` = 8-char hex of 32-bit PageAddr.

Publishing policy enforcement (which CID classes get published on Zenoh) is out of scope — see bead `harmony-8fg`.

## Code changes

### harmony-content (Ring 0)

| Current | Action | New |
|---------|--------|-----|
| `ChunkAddr` | Delete & replace | `PageAddr` |
| `Athenaeum` | Rename & refactor | `Book` |
| `Book` (old, 1-3 blobs) | Delete | Role absorbed by `Encyclopedia` |
| `Volume` | Keep, update to `PageAddr` | `Volume` |
| `Encyclopedia` | Expand as single entry point | `Encyclopedia` |
| `bundle.rs` | Keep as-is | DAG/bundle serialization unchanged |
| `dag.rs` | Keep as-is | Merkle DAG unchanged |
| `storage_tier.rs` | Update ChunkAddr → PageAddr refs | |

### harmony-zenoh (Ring 0)

| File | Action |
|------|--------|
| `keyspace.rs` | Add book/page/toc/pos key expression builders and parsers |

### harmony-microkernel (harmony-os, Ring 2)

| File | Action |
|------|--------|
| `content_server.rs` | Update `/store/chunks/` → `/store/pages/`, ChunkAddr → PageAddr, Athenaeum → Book |

### Files deleted

- `athenaeum.rs` → logic moves into new `book.rs`
- Old `book.rs` → role absorbed by `encyclopedia.rs`

## Out of scope

- Storage/publishing policy enforcement based on CID classification bits → bead `harmony-8fg`
- `zenoh_bridge.rs` / `reticulum_bridge.rs` implementation
- Incremental Encyclopedia updates (add/remove books)
- Distributed Encyclopedia build coordination

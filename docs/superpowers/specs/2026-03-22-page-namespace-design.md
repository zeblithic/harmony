# Page Addressing Namespace for Zenoh

**Bead:** harmony-psp
**Date:** 2026-03-22
**Status:** Draft

## Problem

Pages (4KB chunks within 1MB books) are currently addressable only via their
containing book: `harmony/book/{cid}/page/{page_addr}`. You must know the
book CID to find a page. There's no way to ask "who has a page with this
32-bit address?" across all books — you'd have to query every book individually.

## Solution

Add a page-first Zenoh namespace: `harmony/page/<addr_00>/<addr_01>/<addr_10>/<addr_11>/<book_cid>/<page_num>`. All 4 algorithm variants of the PageAddr are path segments, enabling lookup
with as little or as much information as available. The existing
`harmony/book/{cid}/page/` namespace coexists for book-first access.

## Design Decisions

### Coexist with existing book namespace (not replace)

Two different queries, two different namespaces:
- `harmony/page/...` — "who has this page address?" (page-first discovery)
- `harmony/book/{cid}/page/...` — "give me page N of this book" (book-first access)

### All 4 address variants always populated on publish

Each page has 4 PageAddr values (MSB SHA-256, LSB SHA-256, SHA-224 MSB,
SHA-224 LSB). All 4 are path segments in every published entry. Consumers
query with wildcards for unknown slots. One publish per page, maximum
flexibility for consumers.

### Adaptive response (metadata by default, data on request)

Published entries contain lightweight metadata (book CID + page_num). When
a consumer queries and gets a single unambiguous match, the responder can
inline the 4KB page data in the reply. Multiple matches return metadata
only for disambiguation. (Implemented in follow-up bead `harmony-594`.)

### MSB SHA-256 book CID by default

Only the MSB SHA-256 variant of the 256-bit book ContentId is used in the
path. All three valid hash variants produce the same book content, so one
is sufficient for uniqueness. The other two can be published on a case-by-case
basis for collision resistance, at 3x the memory cost.

### Existing PageAddr and Book types unchanged

The `harmony-athenaeum` crate already has `PageAddr` (32-bit, 4 algorithm
variants), `Book` (up to 256 pages with `[PageAddr; 4]` per page), and
`BookType::SelfIndexing` (page 0 is a ToC). This spec adds only Zenoh
namespace helpers — no changes to athenaeum internals.

## Namespace Structure

**Path:** `harmony/page/<addr_00>/<addr_01>/<addr_10>/<addr_11>/<book_cid>/<page_num>`

| Segment | Format | Description |
|---------|--------|-------------|
| `<addr_00>` | 8 hex chars | MSB SHA-256 PageAddr (mode 00) |
| `<addr_01>` | 8 hex chars | LSB SHA-256 PageAddr (mode 01) |
| `<addr_10>` | 8 hex chars | SHA-224 MSB PageAddr (mode 10) |
| `<addr_11>` | 8 hex chars | SHA-224 LSB PageAddr (mode 11) |
| `<book_cid>` | 64 hex chars | 256-bit ContentId of the containing book |
| `<page_num>` | decimal 0-255 | Page position within the book |

### 32-bit PageAddr layout (per `harmony-athenaeum/src/addr.rs`)

```
[algo:2 bits][hash_bits:28 bits][checksum:2 bits]
 bits 31-30    bits 29-2          bits 1-0
```

- Algorithm: 00=MSB SHA-256, 01=LSB SHA-256, 10=SHA-224 MSB, 11=SHA-224 LSB
- Hash bits: middle 28 bits from the algorithm-specific hash window (mask, no shift)
- Checksum: 2-bit XOR fold of the 30-bit prefix

### Query patterns

| What you know | Query | Wildcards |
|---|---|---|
| Mode-00 only | `harmony/page/{00}/*/*/*/*/*` | 5 |
| Mode-00 + 01 | `harmony/page/{00}/{01}/*/*/*/*` | 4 |
| All 4 addrs | `harmony/page/{00}/{01}/{10}/{11}/*/*` | 2 |
| All 4 + book | `harmony/page/{00}/{01}/{10}/{11}/{cid}/*` | 1 |
| Exact page | `harmony/page/{00}/{01}/{10}/{11}/{cid}/{num}` | 0 |
| All pages of book | `harmony/page/*/*/*/*/{cid}/*` | 4+1 |
| Specific book+pos | `harmony/page/*/*/*/*/{cid}/{num}` | 4 |

**Disambiguation:** When a short query (e.g., mode-00 only) returns multiple
results, the consumer sees distinct path entries and inspects the remaining
segments to choose. If a single mode-00 address uniquely resolves to one
page, that's all the consumer needs.

**Non-terminal wildcards:** Patterns like `harmony/page/*/*/*/*/{cid}/*` use
`*` in non-terminal positions. These work for Zenoh `session.get()` queries
but may not be registerable as subscribers in Zenoh 1.x.

## Key Expression Helpers

```rust
pub mod page {
    pub const PREFIX: &str = "harmony/page";
    pub const SUB: &str = "harmony/page/**";

    /// Full page key with all 4 address variants + book CID + page number.
    pub fn page_key(
        addr_00: &str, addr_01: &str, addr_10: &str, addr_11: &str,
        book_cid: &str, page_num: u8,
    ) -> String;

    /// Query by mode-00 address only (MSB SHA-256, most common).
    pub fn query_by_addr00(addr_00: &str) -> String;

    /// Query by mode-00 + mode-01 addresses.
    pub fn query_by_addr00_01(addr_00: &str, addr_01: &str) -> String;

    /// Query all pages of a specific book (non-terminal wildcard caveat).
    pub fn query_by_book(book_cid: &str) -> String;

    /// Query a specific page by book + position (non-terminal wildcard caveat).
    pub fn query_by_book_and_pos(book_cid: &str, page_num: u8) -> String;
}
```

All addresses are canonical lowercase hex via `hex::encode()`.

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod page` with PREFIX, SUB, key builders, query helpers |

## What is NOT in Scope

- No runtime integration (bead `harmony-594`)
- No Bloom filter for page addresses (bead `harmony-7np`)
- No changes to harmony-athenaeum (PageAddr types already exist)
- No adaptive response logic (runtime concern)
- No Q8 encoding changes (harmony-stq8 concern)
- No changes to existing `harmony/book/{cid}/page/` namespace (coexists)

## Testing

- `page_key_format` — full path with all 6 segments
- `query_by_addr00` — 5 trailing wildcards
- `query_by_addr00_01` — 4 trailing wildcards
- `query_by_book` — 4 leading wildcards + 1 trailing
- `query_by_book_and_pos` — 4 leading wildcards
- `page_subscription_pattern` — SUB constant
- `all_prefixes_start_with_root` — include page::PREFIX

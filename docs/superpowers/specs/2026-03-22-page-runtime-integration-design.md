# Page Namespace Runtime Integration

**Bead:** harmony-594
**Date:** 2026-03-22
**Status:** Draft

## Problem

The page namespace (`harmony/page/...`) defines the key expressions but
nothing populates them. When a book is stored, its pages are not published
to the namespace and there's no queryable to serve page data. The namespace
is defined but inert.

## Solution

Wire page publishing and querying into the NodeRuntime. When a book is stored,
compute its PageAddr index and publish lightweight metadata entries. Declare a
queryable on `harmony/page/**` that serves adaptive responses: inline 4KB data
for unique matches, metadata-only for ambiguous queries.

## Design Decisions

### Publish metadata + queryable for data (two-phase)

Published metadata entries persist in Zenoh even when the node is offline,
enabling passive discovery. The queryable serves actual page data on demand
with adaptive responses. Same pattern as content: Bloom filters for discovery,
queryables for data.

### Adaptive response driven by match count (not consumer flags)

The query pattern IS the intent signal. Exact query (0 wildcards) → inline 4KB.
Partial query (wildcards) with 1 match → inline 4KB. Multiple matches →
metadata only for each. No extra protocol, no payload flags, no round-trip
negotiation. The specificity of your query determines what you get back.

### Logic lives in NodeRuntime (not StorageTier)

The runtime already bridges StorageTier → Zenoh publishing. Page computation
uses `harmony-athenaeum::Book::from_book()` which is an athenaeum concern,
not a storage-tier concern. The runtime handles the orchestration.

### One queryable on `harmony/page/**`

Single declaration at startup. The handler parses the key expression to
extract concrete vs wildcard segments and matches against the local page
index. Same pattern as content queryables.

### PageIndex keyed by mode-00 (MSB SHA-256)

Mode-00 is the most common query (default hash variant). The index stores
all 4 variants per entry for filtering when the query includes additional
address modes.

## Architecture

### PageIndex (in-memory)

```rust
struct PageIndexEntry {
    book_cid: ContentId,
    page_num: u8,
    addrs: [PageAddr; 4],  // all 4 algorithm variants
}

struct PageIndex {
    by_addr00: HashMap<PageAddr, Vec<PageIndexEntry>>,
}
```

Methods:
- `insert_book(cid: ContentId, book: &Book)` — index all data pages
- `lookup(addr_00: PageAddr) -> &[PageIndexEntry]` — primary lookup
- `match_query(addr_00: Option<PageAddr>, addr_01: Option<PageAddr>, ..., book_cid: Option<ContentId>, page_num: Option<u8>) -> Vec<&PageIndexEntry>` — filter by whatever segments are concrete

### Publishing flow

1. Runtime receives `StorageTierAction::AnnounceContent` for a book (depth-0 CID)
2. Fetch book data from `BookStore::get(&cid)`
3. Compute `Book::from_book(cid.to_bytes(), data)`
4. Call `page_index.insert_book(cid, &book)`
5. For each data page, emit `RuntimeAction::Publish`:
   - Key: `harmony/page/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/{page_num}`
   - Payload: single byte `page_num` (key encodes everything else)

Only processes depth-0 CIDs (`cid.cid_type() == CidType::Book`).

### Queryable flow

1. Startup: declare queryable on `harmony/page/**`
2. Query arrives with key expression (may contain wildcards)
3. Parse key expression — extract concrete segments
4. Look up matches in `PageIndex`
5. Response based on match count:
   - **0 matches** — no reply
   - **1 match** — reply with `0x01` header + 4096 bytes of page data
     (sliced from BookStore: `book_data[page_num * 4096 .. (page_num + 1) * 4096]`)
   - **Multiple matches** — one reply per match with `0x00` header + the
     full page key expression (consumer inspects keys to disambiguate)

### Reply format

```
[0x00] = metadata reply: remaining bytes are the full key expression (UTF-8)
[0x01] = data reply: remaining 4096 bytes are the page content
```

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/Cargo.toml` | Add `harmony-athenaeum` dependency |
| `crates/harmony-node/src/runtime.rs` | PageIndex struct, populate on book store, declare page queryable at startup, page publish on AnnounceContent |
| `crates/harmony-node/src/event_loop.rs` | Route page queries to runtime, serve adaptive responses |

## What is NOT in Scope

- No Bloom filter for pages (bead `harmony-7np`)
- No persistence of PageIndex across restarts (rebuilt lazily from BookStore)
- No changes to harmony-athenaeum (uses existing Book::from_book, PageAddr)
- No changes to harmony-zenoh (namespace already defined)
- No query forwarding to other peers (local index only)

## Testing

- `page_index_insert_and_lookup` — store book, verify PageAddrs indexed
- `page_index_multiple_books` — two books sharing a mode-00 address, both returned
- `page_query_exact_returns_data` — exact query → 0x01 header + 4KB
- `page_query_ambiguous_returns_metadata` — partial query, multiple matches → 0x00 headers
- `page_publish_on_book_store` — verify Publish actions emitted after storing a book

# Page-Level Bloom Filter

**Bead:** harmony-7np
**Date:** 2026-03-22
**Status:** Draft

## Problem

Nodes index pages and serve them via the `harmony/page/**` queryable, but peers
have no way to know which nodes hold which pages before querying. Every page
query fans out to all peers. The content and memo namespaces already solve this
with Bloom filter broadcasts — page needs the same.

## Solution

Broadcast a Bloom filter of mode-00 PageAddr values so peers can pre-check
"does this node have a page with this address?" before issuing a queryable
request. Fourth instance of the established content/memo/flatpack filter pattern.

## Design Decisions

### Mode-00 addresses only (not all 4 variants)

Mode-00 (MSB SHA-256) is the default and most common lookup. Inserting all 4
variants would 4x the filter size for marginal benefit — queries starting from
other modes can still use the queryable directly. The Bloom filter is a
pre-check optimization, not a gatekeeper.

### Generic `insert_bytes`/`may_contain_bytes` on BloomFilter

The existing BloomFilter API is hardcoded to `ContentId` (32 bytes). PageAddr
is 4 bytes. Rather than wastefully zero-padding into a ContentId, add generic
byte-slice methods that hash the input via SHA-256 first, then feed the digest
into the existing SplitMix64 double-hashing. The ContentId-specific methods
stay unchanged.

### Timer-based only (no mutation threshold)

The page index only changes when new books are stored — much less frequent than
cache admissions. A mutation threshold adds complexity for negligible benefit.
Broadcast on the existing `FilterTimerTick` alongside content/memo filters.

### Skip broadcast when page index is empty

If no books have been indexed, don't broadcast an empty filter. Same pattern
as the memo filter.

## Architecture

### BloomFilter Extension (harmony-content)

```rust
// New methods on BloomFilter:
pub fn insert_bytes(&mut self, data: &[u8])
pub fn may_contain_bytes(&self, data: &[u8]) -> bool
```

Internal: SHA-256 hash the input bytes to produce a 32-byte digest, then read
`digest[0..8]` and `digest[8..16]` as the two base hashes (same SplitMix64
mixing as the ContentId path).

### Namespace (harmony-zenoh)

```
harmony/filters/page/{node_addr}
```

Constants:
- `PAGE_PREFIX = "harmony/filters/page"`
- `PAGE_SUB = "harmony/filters/page/**"`
- `page_key(node_addr: &str) -> String`

### PageIndex Accessor (harmony-node)

```rust
pub fn addr00_iter(&self) -> impl Iterator<Item = &PageAddr>
```

Exposes the mode-00 key set without leaking the internal HashMap.

### Build + Broadcast (harmony-node runtime)

On `FilterTimerTick`, if `!page_index.is_empty()`:

```rust
let mut page_filter = BloomFilter::new(expected_items, fp_rate);
for addr in self.page_index.addr00_iter() {
    page_filter.insert_bytes(&addr.to_bytes());
}
self.pending_page_broadcast = Some(page_filter.to_bytes());
```

Emitted as `RuntimeAction::Publish` with key `filters::page_key(&self.node_addr)`.

### Consume (harmony-node runtime)

In `route_subscription`, match `filters::PAGE_PREFIX`:

```rust
if let Some(peer_addr) = key_expr.strip_prefix(filters::PAGE_PREFIX)... {
    match BloomFilter::from_bytes(&payload) {
        Ok(filter) => self.peer_filters.upsert_page(peer_addr, filter, tick),
        Err(_) => self.peer_filters.record_parse_error(),
    }
}
```

### PeerFilterTable Extension

Add to `PeerFilter`:
- `page_filter: Option<BloomFilter>`
- `page_received_tick: u64`

Methods:
- `upsert_page(peer_addr: String, filter: BloomFilter, tick: u64)`
- `should_query_page(peer_addr: &str, addr: &PageAddr, tick: u64) -> bool`
  — calls `may_contain_bytes(&addr.to_bytes())`

Staleness model identical to content/memo — stale → treat as absent → query.

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-content/src/bloom.rs` | Add `insert_bytes`, `may_contain_bytes` |
| `crates/harmony-zenoh/src/namespace.rs` | Add `filters::PAGE_PREFIX`, `PAGE_SUB`, `page_key` |
| `crates/harmony-node/src/page_index.rs` | Add `addr00_iter()` |
| `crates/harmony-node/src/runtime.rs` | Build + broadcast page filter, consume in `route_subscription`, extend `PeerFilterTable` |

## What is NOT in Scope

- No mutation-threshold triggering (timer only)
- No changes to the page queryable (already works)
- No peer-selection logic that uses the filter for routing (future work)
- No persistence of the filter across restarts (rebuilt from PageIndex)

## Testing

- `bloom_insert_bytes_and_may_contain` — round-trip for arbitrary byte slices
- `bloom_bytes_definite_miss` — absent item returns false
- `page_filter_broadcast_on_timer` — store book, tick, verify Publish action
- `peer_page_filter_upsert_and_query` — PeerFilterTable round-trip
- `peer_page_filter_staleness` — stale filter → query anyway

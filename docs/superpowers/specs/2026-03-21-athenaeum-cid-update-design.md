# Athenaeum CID Format Update Design

**Date:** 2026-03-21
**Status:** Implemented (with deviations — see note below)
**Scope:** `harmony-athenaeum` crate — update for new 256-bit ContentId format
**Bead:** harmony-7rq

**Implementation note:** The original design proposed changing `route_page()` to `&[u8; 28]` and reducing `MAX_PARTITION_DEPTH` to 223. During implementation, we discovered that internal partition routing uses SHA-256 page content hashes (full 32 bytes), not CIDs — so `route_page()` and `MAX_PARTITION_DEPTH` are unaffected. Instead, a new `route_hash()` method was added that extracts bits directly from the 28-byte CID hash portion without zero-padding.

## Overview

The CID format was redesigned in harmony-0h7 (PR #90). The new 32-byte ContentId layout is:

```
[4 mode][6 depth][20 size][2 checksum][224 hash]
 ← 4 bytes (32 bits) header →          ← 28 bytes hash →
```

harmony-athenaeum uses CID bytes for two purposes:
1. **Storage** — `Book.cid: [u8; 32]` stores the full CID as opaque bytes
2. **Routing** — `route_page()` and `Encyclopedia` extract individual bits for partition tree traversal

The routing use is affected: bits 0-31 are now metadata (mode, depth, size, checksum), not hash bits. Routing on metadata creates pathological partition distribution because many CIDs share the same prefix.

## Design

### route_page() — hash-only routing

Change `route_page()` to accept the 28-byte hash portion instead of the full 32-byte CID:

**Before:**
```rust
pub fn route_page(content_hash: &[u8; 32], bit_index: u8) -> bool
```

**After:**
```rust
pub fn route_page(hash: &[u8; 28], bit_index: u8) -> bool
```

Callers extract the hash from the CID (`&cid[4..]`) before calling. This maintains the clean boundary — athenaeum never parses CID headers.

Max `bit_index` drops from 255 to 223. `MAX_PARTITION_DEPTH` (currently 228) must be reduced to 223 to stay within the hash bit space.

### Encyclopedia — same change

`Encyclopedia::build()` uses CID bytes for power-of-choice address assignment. Update internal hash extraction to use 28-byte hash slices, matching the route_page change.

### Book.cid — no change

`Book.cid` stays `[u8; 32]`. The Book stores the full CID as opaque bytes. `Book::from_book()` receives a CID as a parameter (doesn't compute one) — no change needed.

### No new dependencies

harmony-athenaeum stays independent of harmony-content. The caller (harmony-content or harmony-node) is responsible for constructing ContentId and extracting the hash portion before passing it to athenaeum routing functions.

### What doesn't change

- `PageAddr` — hashes page data, not CIDs
- `Book` struct layout and serialization
- `Volume` serialization format
- Encrypted book handling
- `Book::from_book()` / `from_book_self_indexing()` / `from_book_encrypted()` constructors

### Constants

| Constant | Old | New | Reason |
|----------|-----|-----|--------|
| `MAX_PARTITION_DEPTH` | 228 | 223 | 224 hash bits - 1 (0-indexed) |

### Test updates

- Tests that call `route_page()` pass `&hash[4..]` (28 bytes) instead of `&hash` (32 bytes)
- Tests that build Encyclopedias pass 28-byte hash slices
- Existing tests using `sha256_hash()` directly: extract bytes 4-31 for routing, keep full 32 bytes for `Book.cid`

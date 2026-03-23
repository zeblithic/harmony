# Page Namespace Runtime Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Auto-publish page metadata when books are stored and serve page data via a queryable with adaptive responses.

**Architecture:** Add `PageIndex` to NodeRuntime for tracking pages by mode-00 PageAddr. When StorageTier announces a book, compute its pages via `Book::from_book()` and publish entries to the `/page/` namespace. Register a queryable on `harmony/page/**` that serves 4KB data for unique matches and metadata for ambiguous ones.

**Tech Stack:** Rust, harmony-athenaeum (Book, PageAddr), harmony-zenoh (page namespace), harmony-content (ContentId, BookStore)

**Spec:** `docs/superpowers/specs/2026-03-22-page-runtime-integration-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/Cargo.toml` | Add `harmony-athenaeum` dependency |
| `crates/harmony-node/src/page_index.rs` | **New**: PageIndex struct, insert, lookup, match queries |
| `crates/harmony-node/src/runtime.rs` | Own PageIndex, publish pages on AnnounceContent, register page queryable, route page queries |
| `crates/harmony-node/src/main.rs` | Add `mod page_index;` |

---

### Task 1: PageIndex data structure (TDD)

**Files:**
- Create: `crates/harmony-node/src/page_index.rs`
- Modify: `crates/harmony-node/src/main.rs` (add `mod page_index;`)
- Modify: `crates/harmony-node/Cargo.toml` (add `harmony-athenaeum`)

Pure data structure — no runtime integration yet.

- [ ] **Step 1: Add harmony-athenaeum dependency**

In `crates/harmony-node/Cargo.toml`, add:
```toml
harmony-athenaeum = { workspace = true }
```

Check root `Cargo.toml` workspace deps — if `harmony-athenaeum` is not listed, add it:
```toml
harmony-athenaeum = { path = "crates/harmony-athenaeum" }
```

- [ ] **Step 2: Add `mod page_index;` to main.rs**

Add after existing mod declarations.

- [ ] **Step 3: Create page_index.rs with types and tests**

```rust
use std::collections::HashMap;
use harmony_athenaeum::{Book, PageAddr, ALGO_COUNT};
use harmony_content::ContentId;

/// Entry in the page index mapping a page to its containing book.
#[derive(Debug, Clone)]
pub struct PageIndexEntry {
    pub book_cid: ContentId,
    pub page_num: u8,
    pub addrs: [PageAddr; ALGO_COUNT],
}

/// In-memory index of pages keyed by mode-00 (MSB SHA-256) PageAddr.
pub struct PageIndex {
    by_addr00: HashMap<PageAddr, Vec<PageIndexEntry>>,
}
```

Methods:
- `new() -> Self`
- `insert_book(cid: ContentId, book: &Book)` — iterate `book.data_pages()`, insert each page's mode-00 addr as key with entry containing all 4 addrs
- `lookup(&self, addr_00: &PageAddr) -> &[PageIndexEntry]` — primary lookup
- `match_query(&self, addr_00: Option<PageAddr>, addr_01: Option<PageAddr>, addr_10: Option<PageAddr>, addr_11: Option<PageAddr>, book_cid: Option<&ContentId>, page_num: Option<u8>) -> Vec<&PageIndexEntry>` — filter by concrete segments
- `len() -> usize` — total entry count
- `is_empty() -> bool`

Tests:
- `insert_and_lookup` — create book from data, insert, lookup by mode-00 addr
- `multiple_books_same_addr` — two different books that happen to share a mode-00 addr (construct test data)
- `match_query_filters_by_book_cid` — verify filtering works
- `match_query_filters_by_page_num` — verify filtering works
- `empty_index` — lookup returns empty slice

For test data, use `Book::from_book([0x11; 32], &[0u8; 4096])` (a 1-page book with dummy data).

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-node -- page_index`
Expected: all tests pass

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/Cargo.toml crates/harmony-node/src/page_index.rs crates/harmony-node/src/main.rs Cargo.toml
git commit -m "feat(harmony-node): add PageIndex for page-first addressing

PageIndex keyed by mode-00 PageAddr with full 4-variant entries.
insert_book() from Book, lookup, match_query with filtering.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Page publishing on book store

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

When a book is stored and announced, also compute and publish its page entries.

- [ ] **Step 1: Add PageIndex to NodeRuntime**

Add `page_index: crate::page_index::PageIndex` to the runtime struct.
Initialize `page_index: PageIndex::new()` in `new()`.

- [ ] **Step 2: Publish page entries after AnnounceContent**

In `dispatch_storage_actions()`, after the `AnnounceContent` arm emits the Publish action (line ~1198), add page publishing logic:

```rust
StorageTierAction::AnnounceContent { key_expr, payload } => {
    out.push(RuntimeAction::Publish { key_expr, payload });

    // Page publishing: if this is a book (depth 0), compute page index
    // and publish metadata entries to harmony/page/ namespace.
    // Extract CID from the announce key expression.
    if let Some(cid_hex) = key_expr.strip_prefix("harmony/announce/") {
        if let Ok(cid_bytes) = hex::decode(cid_hex) {
            if cid_bytes.len() == 32 {
                let cid = ContentId::from_bytes(cid_bytes.try_into().unwrap());
                if matches!(cid.cid_type(), harmony_content::cid::CidType::Book) {
                    if let Some(data) = self.storage.get(&cid) {
                        self.publish_book_pages(&cid, data, out);
                    }
                }
            }
        }
    }
}
```

Implement `publish_book_pages()`:

```rust
fn publish_book_pages(&mut self, cid: &ContentId, data: &[u8], out: &mut Vec<RuntimeAction>) {
    let cid_bytes: [u8; 32] = cid.to_bytes();
    let Ok(book) = Book::from_book(cid_bytes, data) else { return };

    self.page_index.insert_book(*cid, &book);

    let book_cid_hex = hex::encode(cid_bytes);
    for (i, addrs) in book.data_pages().iter().enumerate() {
        let page_num = i as u8;
        let addr_00 = hex::encode(addrs[0].to_bytes());
        let addr_01 = hex::encode(addrs[1].to_bytes());
        let addr_10 = hex::encode(addrs[2].to_bytes());
        let addr_11 = hex::encode(addrs[3].to_bytes());

        let key_expr = harmony_zenoh::namespace::page::page_key(
            &addr_00, &addr_01, &addr_10, &addr_11,
            &book_cid_hex, page_num,
        );
        out.push(RuntimeAction::Publish {
            key_expr,
            payload: vec![page_num],
        });
    }
}
```

Note: Check the actual `Book::from_book` signature and `PageAddr` API. `PageAddr::to_bytes()` should return `[u8; 4]`. Adapt as needed. Also check if `self.storage.get()` is the right way to access stored content — it might be `self.storage.cache.get()` or similar.

- [ ] **Step 3: Add test**

```rust
#[test]
fn page_publish_on_book_store() {
    // Create a runtime, push a book via PublishContent event,
    // tick, and verify Publish actions include page namespace keys.
}
```

This may be complex to set up fully. At minimum, verify `publish_book_pages` produces the expected number of Publish actions for a known book size.

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-node -v`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): publish page metadata on book store

Compute Book from stored data, index pages, emit Publish actions
to harmony/page/ namespace for each data page. PageIndex populated
for queryable lookups.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Page queryable with adaptive response

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

Declare a page queryable and route page queries with adaptive responses.

- [ ] **Step 1: Register page queryable at startup**

In the `new()` constructor, alongside existing queryable declarations, add:

```rust
// Declare page queryable
let page_qbl_id = queryable_router.declare(
    harmony_zenoh::namespace::page::SUB,  // "harmony/page/**"
)?;
```

Store `page_qbl_id` in a new field: `page_queryable_id: QueryableId`.

- [ ] **Step 2: Route page queries in route_query()**

In `route_query()`, add a branch for the page queryable alongside the existing storage and compute branches:

```rust
} else if queryable_id == self.page_queryable_id {
    let actions = self.handle_page_query(query_id, &key_expr);
    self.pending_direct_actions.extend(actions);
}
```

- [ ] **Step 3: Implement handle_page_query()**

```rust
fn handle_page_query(&self, query_id: u64, key_expr: &str) -> Vec<RuntimeAction> {
    // Parse key_expr to extract concrete segments
    let parsed = self.parse_page_key_expr(key_expr);

    // Look up matches
    let matches = self.page_index.match_query(
        parsed.addr_00, parsed.addr_01, parsed.addr_10, parsed.addr_11,
        parsed.book_cid.as_ref(), parsed.page_num,
    );

    if matches.is_empty() {
        return vec![];
    }

    if matches.len() == 1 {
        // Unique match — serve 4KB inline
        let entry = matches[0];
        if let Some(data) = self.storage.get(&entry.book_cid) {
            let start = entry.page_num as usize * 4096;
            let end = (start + 4096).min(data.len());
            let mut payload = Vec::with_capacity(1 + 4096);
            payload.push(0x01); // data reply header
            payload.extend_from_slice(&data[start..end]);
            return vec![RuntimeAction::SendReply { query_id, payload }];
        }
    }

    // Multiple matches — metadata only
    let mut actions = Vec::new();
    for entry in matches {
        let key = harmony_zenoh::namespace::page::page_key(
            &hex::encode(entry.addrs[0].to_bytes()),
            &hex::encode(entry.addrs[1].to_bytes()),
            &hex::encode(entry.addrs[2].to_bytes()),
            &hex::encode(entry.addrs[3].to_bytes()),
            &hex::encode(entry.book_cid.to_bytes()),
            entry.page_num,
        );
        let mut payload = Vec::with_capacity(1 + key.len());
        payload.push(0x00); // metadata reply header
        payload.extend_from_slice(key.as_bytes());
        actions.push(RuntimeAction::SendReply { query_id, payload });
    }
    actions
}
```

Implement `parse_page_key_expr()` that splits on `/` and returns an options struct with concrete values where segments are not `*`.

- [ ] **Step 4: Add startup queryable action**

Ensure the `DeclareQueryable` action for `harmony/page/**` is emitted in the startup actions Vec.

- [ ] **Step 5: Add tests**

- `page_query_exact_returns_data` — store a book, query exact page, verify 0x01 header + 4KB
- `page_query_ambiguous_returns_metadata` — store two books with overlapping mode-00 addr (if possible with test data), query by mode-00, verify 0x00 headers

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-node -v`

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): page queryable with adaptive response

Declare queryable on harmony/page/**. Exact match → 0x01 header + 4KB
page data. Multiple matches → 0x00 header + key expression metadata
per match. Parses key expression for concrete vs wildcard segments.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Full verification

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-node`

- [ ] **Step 3: Run fmt**

Run: `cargo fmt --all -- --check`

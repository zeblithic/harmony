# Page Addressing Namespace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add page-first Zenoh namespace helpers for discovering pages by their 32-bit addresses across all books.

**Architecture:** Add `pub mod page` to `harmony-zenoh/src/namespace.rs` with key builders and query helpers. Follows the existing namespace module pattern (PREFIX, SUB, format-string builders). Single-file change.

**Tech Stack:** Rust, harmony-zenoh (namespace conventions)

**Spec:** `docs/superpowers/specs/2026-03-22-page-namespace-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-zenoh/src/namespace.rs` | Add `pub mod page` with PREFIX, SUB, page_key, query helpers |

---

### Task 1: Add page namespace module with key builders and tests

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

Single task — add the `pub mod page` module following the existing namespace pattern.

- [ ] **Step 1: Add `pub mod page` to namespace.rs**

Add between the `memo` module and `#[cfg(test)]`, following the exact pattern of existing modules:

```rust
/// Page-first content addressing namespace.
///
/// Path: `harmony/page/<addr_00>/<addr_01>/<addr_10>/<addr_11>/<book_cid>/<page_num>`
///
/// Each page has 4 algorithm-variant PageAddr values (32-bit each, 8 hex chars):
/// - `addr_00`: MSB SHA-256 (mode bits = 00)
/// - `addr_01`: LSB SHA-256 (mode bits = 01)
/// - `addr_10`: SHA-224 MSB (mode bits = 10)
/// - `addr_11`: SHA-224 LSB (mode bits = 11)
///
/// The `book_cid` is the 256-bit ContentId of the containing book (64 hex chars).
/// The `page_num` is the page's position within the book (0-255, decimal).
///
/// Coexists with `harmony/book/{cid}/page/{page_addr}` (book-first access).
/// This namespace enables page-first discovery: "who has this page address?"
///
/// All addresses are canonical lowercase hex.
pub mod page {
    use alloc::{format, string::String};

    /// Base prefix: `harmony/page`
    pub const PREFIX: &str = "harmony/page";

    /// Subscribe to all page events: `harmony/page/**`
    pub const SUB: &str = "harmony/page/**";

    /// Full page key with all 4 address variants, book CID, and page number.
    ///
    /// `harmony/page/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/{page_num}`
    pub fn page_key(
        addr_00: &str,
        addr_01: &str,
        addr_10: &str,
        addr_11: &str,
        book_cid: &str,
        page_num: u8,
    ) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/{addr_10}/{addr_11}/{book_cid}/{page_num}")
    }

    /// Query by mode-00 address only (MSB SHA-256, most common).
    ///
    /// `harmony/page/{addr_00}/*/*/*/*/*`
    pub fn query_by_addr00(addr_00: &str) -> String {
        format!("{PREFIX}/{addr_00}/*/*/*/*/*")
    }

    /// Query by mode-00 + mode-01 addresses.
    ///
    /// `harmony/page/{addr_00}/{addr_01}/*/*/*/*`
    pub fn query_by_addr00_01(addr_00: &str, addr_01: &str) -> String {
        format!("{PREFIX}/{addr_00}/{addr_01}/*/*/*/*")
    }

    /// Query all pages of a specific book.
    ///
    /// `harmony/page/*/*/*/*/{book_cid}/*`
    ///
    /// This pattern uses `*` in non-terminal positions, which works for
    /// Zenoh `session.get()` queries but may not be registerable as a
    /// subscriber in Zenoh 1.x. For subscription use cases, subscribe to
    /// `harmony/page/**` and filter at the application layer.
    pub fn query_by_book(book_cid: &str) -> String {
        format!("{PREFIX}/*/*/*/*/{book_cid}/*")
    }

    /// Query a specific page by book CID + position.
    ///
    /// `harmony/page/*/*/*/*/{book_cid}/{page_num}`
    ///
    /// This pattern uses `*` in non-terminal positions, which works for
    /// Zenoh `session.get()` queries but may not be registerable as a
    /// subscriber in Zenoh 1.x.
    pub fn query_by_book_and_pos(book_cid: &str, page_num: u8) -> String {
        format!("{PREFIX}/*/*/*/*/{book_cid}/{page_num}")
    }
}
```

- [ ] **Step 2: Add `page::PREFIX` to `all_prefixes_start_with_root` test**

Find the existing `all_prefixes_start_with_root` test in the `#[cfg(test)]` section and add `page::PREFIX` to the prefixes array.

- [ ] **Step 3: Add page namespace tests**

Add to the test section:

```rust
    // ── Page namespace ──────────────────────────────────────────────

    #[test]
    fn page_key_format() {
        let key = page::page_key("aabb0011", "ccdd2233", "eeff4455", "00116677", "ff".repeat(32).as_str(), 42);
        assert_eq!(
            key,
            format!(
                "harmony/page/aabb0011/ccdd2233/eeff4455/00116677/{}/42",
                "ff".repeat(32)
            )
        );
    }

    #[test]
    fn page_key_page_zero() {
        let key = page::page_key("00000000", "11111111", "22222222", "33333333", &"aa".repeat(32), 0);
        assert!(key.ends_with("/0"));
    }

    #[test]
    fn page_key_page_255() {
        let key = page::page_key("00000000", "11111111", "22222222", "33333333", &"aa".repeat(32), 255);
        assert!(key.ends_with("/255"));
    }

    #[test]
    fn page_query_by_addr00() {
        let q = page::query_by_addr00("aabb0011");
        assert_eq!(q, "harmony/page/aabb0011/*/*/*/*/*");
        // Verify exactly 5 wildcards after the address
        assert_eq!(q.matches('*').count(), 5);
    }

    #[test]
    fn page_query_by_addr00_01() {
        let q = page::query_by_addr00_01("aabb0011", "ccdd2233");
        assert_eq!(q, "harmony/page/aabb0011/ccdd2233/*/*/*/*");
        assert_eq!(q.matches('*').count(), 4);
    }

    #[test]
    fn page_query_by_book() {
        let cid = "bb".repeat(32);
        let q = page::query_by_book(&cid);
        assert_eq!(q, format!("harmony/page/*/*/*/*/{cid}/*"));
        // 4 leading wildcards + 1 trailing = 5 total
        assert_eq!(q.matches('*').count(), 5);
    }

    #[test]
    fn page_query_by_book_and_pos() {
        let cid = "cc".repeat(32);
        let q = page::query_by_book_and_pos(&cid, 100);
        assert_eq!(q, format!("harmony/page/*/*/*/*/{cid}/100"));
        assert_eq!(q.matches('*').count(), 4);
    }

    #[test]
    fn page_subscription_pattern() {
        assert_eq!(page::SUB, "harmony/page/**");
    }
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p harmony-zenoh -v`
Expected: all tests pass (existing + 8 new page tests)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add page-first addressing namespace

harmony/page/<addr_00>/<addr_01>/<addr_10>/<addr_11>/<book_cid>/<page_num>
for page-first discovery across all books. Query helpers for partial
address lookup with wildcards. 8 new tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Verification

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-zenoh`
Expected: no new warnings

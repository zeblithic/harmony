# Page-Level Bloom Filter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Broadcast a Bloom filter of page addresses so peers can pre-check before querying the page queryable.

**Architecture:** Add generic `insert_bytes`/`may_contain_bytes` to `BloomFilter`, define the `harmony/filters/page/` namespace, extend `PeerFilterTable` with a page filter slot, and build+broadcast+consume the page filter alongside existing content/memo filters on the timer tick.

**Tech Stack:** Rust, harmony-content (BloomFilter), harmony-zenoh (namespace), harmony-node (runtime, page_index)

**Spec:** `docs/superpowers/specs/2026-03-22-page-bloom-filter-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-content/src/bloom.rs` | Add `insert_bytes`/`may_contain_bytes` + `hash_pair_bytes` helper |
| `crates/harmony-zenoh/src/namespace.rs` | Add `PAGE_PREFIX`, `PAGE_SUB`, `page_key()` to `filters` module |
| `crates/harmony-node/src/page_index.rs` | Add `addr00_iter()` accessor |
| `crates/harmony-node/src/runtime.rs` | `PeerFilter` + `PeerFilterTable` page slot, build/broadcast/consume |

---

### Task 1: Generic byte-slice methods on BloomFilter

**Files:**
- Modify: `crates/harmony-content/src/bloom.rs`

- [ ] **Step 1: Add `hash_pair_bytes` helper function**

After the existing `hash_pair` function (~line 346), add:

```rust
/// Extract two base hashes from an arbitrary byte slice.
///
/// SHA-256 hashes the input to produce a 32-byte digest, then reads
/// `digest[0..8]` and `digest[8..16]` as little-endian u64 values
/// with the same SplitMix64 mixing as [`hash_pair`].
fn hash_pair_bytes(data: &[u8]) -> (u64, u64) {
    let digest = harmony_crypto::hash::full_hash(data);
    let a = u64::from_le_bytes(digest[0..8].try_into().unwrap());
    let b = u64::from_le_bytes(digest[8..16].try_into().unwrap());
    let h1 = a.wrapping_mul(SEED_A) ^ b;
    let h2 = b.wrapping_mul(SEED_B) ^ a;
    (h1, h2)
}
```

- [ ] **Step 2: Add `insert_bytes` method**

After the existing `insert` method (~line 191), add:

```rust
    /// Insert an arbitrary byte slice into the filter.
    ///
    /// The bytes are SHA-256 hashed internally to derive the Bloom filter
    /// bit indices. Use this for types other than [`ContentId`].
    pub fn insert_bytes(&mut self, data: &[u8]) {
        let (h1, h2) = hash_pair_bytes(data);
        for i in 0..self.num_hashes {
            let idx = bit_index(h1, h2, i, self.num_bits);
            self.set_bit(idx);
        }
        self.item_count = self.item_count.saturating_add(1);
    }
```

- [ ] **Step 3: Add `may_contain_bytes` method**

After the existing `may_contain` method (~line 206), add:

```rust
    /// Test whether an arbitrary byte slice *may* be in the filter.
    ///
    /// Returns `true` if the item might be present (with some false positive
    /// probability), or `false` if the item is definitely absent.
    pub fn may_contain_bytes(&self, data: &[u8]) -> bool {
        let (h1, h2) = hash_pair_bytes(data);
        for i in 0..self.num_hashes {
            let idx = bit_index(h1, h2, i, self.num_bits);
            if !self.get_bit(idx) {
                return false;
            }
        }
        true
    }
```

- [ ] **Step 4: Add tests**

In the `tests` module at the bottom of bloom.rs, add:

```rust
    #[test]
    fn insert_bytes_and_may_contain_round_trip() {
        let mut bf = BloomFilter::new(1000, 0.01);
        let data = b"hello page addr";
        bf.insert_bytes(data);
        assert!(bf.may_contain_bytes(data));
        assert_eq!(bf.item_count(), 1);
    }

    #[test]
    fn bytes_definite_miss() {
        let mut bf = BloomFilter::new(1000, 0.01);
        bf.insert_bytes(b"present");
        assert!(!bf.may_contain_bytes(b"absent"));
    }

    #[test]
    fn bytes_serialization_round_trip() {
        let mut bf = BloomFilter::new(100, 0.01);
        bf.insert_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]);
        let bytes = bf.to_bytes();
        let bf2 = BloomFilter::from_bytes(&bytes).unwrap();
        assert!(bf2.may_contain_bytes(&[0xDE, 0xAD, 0xBE, 0xEF]));
        assert!(!bf2.may_contain_bytes(&[0xCA, 0xFE]));
    }
```

- [ ] **Step 5: Run tests**

Run: `cargo test -p harmony-content -- bloom`
Expected: all tests pass (existing + 3 new)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-content/src/bloom.rs
git commit -m "feat(bloom): add insert_bytes/may_contain_bytes for non-ContentId types

SHA-256 hashes arbitrary byte slices to derive Bloom filter bit indices.
Same SplitMix64 double-hashing as the ContentId path. Enables page
address Bloom filters without wasteful zero-padding.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Page filter namespace constants

**Files:**
- Modify: `crates/harmony-zenoh/src/namespace.rs`

- [ ] **Step 1: Add page filter constants to `filters` module**

In `crates/harmony-zenoh/src/namespace.rs`, inside the `pub mod filters` block (~line 217, after `memo_key`), add:

```rust
    /// Page filter prefix: `harmony/filters/page`
    pub const PAGE_PREFIX: &str = "harmony/filters/page";

    /// Subscribe to all page filters: `harmony/filters/page/**`
    pub const PAGE_SUB: &str = "harmony/filters/page/**";

    /// Page filter key: `harmony/filters/page/{node_addr}`
    pub fn page_key(node_addr: &str) -> String {
        format!("{PAGE_PREFIX}/{node_addr}")
    }
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-zenoh`
Expected: all tests pass (no new tests needed — these are constants)

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-zenoh/src/namespace.rs
git commit -m "feat(zenoh): add harmony/filters/page namespace constants

PAGE_PREFIX, PAGE_SUB, page_key() following the content/memo/flatpack
filter namespace pattern.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: PageIndex accessor + PeerFilterTable extension

**Files:**
- Modify: `crates/harmony-node/src/page_index.rs`
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add `addr00_iter` to PageIndex**

In `crates/harmony-node/src/page_index.rs`, after the `is_empty` method (~line 100), add:

```rust
    /// Iterate over all indexed mode-00 (Sha256Msb) page addresses.
    ///
    /// Used by the page Bloom filter builder to enumerate local page addresses
    /// without exposing the internal HashMap.
    pub fn addr00_iter(&self) -> impl Iterator<Item = &PageAddr> {
        self.by_addr00.keys()
    }
```

- [ ] **Step 2: Add page filter fields to `PeerFilter` struct**

In `crates/harmony-node/src/runtime.rs`, in the `PeerFilter` struct (~line 256-263), add two new fields:

```rust
struct PeerFilter {
    content_filter: Option<BloomFilter>,
    flatpack_filter: Option<CuckooFilter>,
    memo_filter: Option<BloomFilter>,
    page_filter: Option<BloomFilter>,       // NEW
    content_received_tick: u64,
    flatpack_received_tick: u64,
    memo_received_tick: u64,
    page_received_tick: u64,                // NEW
}
```

- [ ] **Step 3: Update all `PeerFilter` constructors**

Every `or_insert(PeerFilter { ... })` call in `PeerFilterTable` must include the new fields. There are 3 existing constructors (in `upsert_content`, `upsert_flatpack`, `upsert_memo`). Add to each:

```rust
    page_filter: None,
    page_received_tick: 0,
```

- [ ] **Step 4: Add `upsert_page` method**

After `upsert_memo` (~line 327), add:

```rust
    fn upsert_page(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
        let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
            content_filter: None,
            flatpack_filter: None,
            memo_filter: None,
            page_filter: None,
            content_received_tick: 0,
            flatpack_received_tick: 0,
            memo_received_tick: 0,
            page_received_tick: tick,
        });
        entry.page_filter = Some(filter);
        entry.page_received_tick = tick;
    }
```

- [ ] **Step 5: Add `should_query_page` method**

After `should_query_memo` (~line 389), add:

```rust
    /// Returns true if the peer should be queried for page addresses.
    fn should_query_page(
        &self,
        peer_addr: &str,
        page_addr: &harmony_athenaeum::PageAddr,
        current_tick: u64,
    ) -> bool {
        match self.filters.get(peer_addr) {
            None => true,
            Some(pf) => {
                if current_tick.saturating_sub(pf.page_received_tick) > self.staleness_ticks {
                    true
                } else {
                    match &pf.page_filter {
                        Some(bf) => bf.may_contain_bytes(&page_addr.to_bytes()),
                        None => true,
                    }
                }
            }
        }
    }
```

- [ ] **Step 6: Update `evict_stale`**

In `evict_stale` (~line 392-401), add page freshness to the retain condition:

```rust
    fn evict_stale(&mut self, current_tick: u64) {
        self.filters.retain(|_, pf| {
            let content_fresh =
                current_tick.saturating_sub(pf.content_received_tick) <= self.staleness_ticks;
            let flatpack_fresh =
                current_tick.saturating_sub(pf.flatpack_received_tick) <= self.staleness_ticks;
            let memo_fresh =
                current_tick.saturating_sub(pf.memo_received_tick) <= self.staleness_ticks;
            let page_fresh =
                current_tick.saturating_sub(pf.page_received_tick) <= self.staleness_ticks;
            content_fresh || flatpack_fresh || memo_fresh || page_fresh
        });
    }
```

- [ ] **Step 7: Add tests**

In the `tests` module of runtime.rs, add:

```rust
    #[test]
    fn peer_page_filter_upsert_and_query() {
        let mut table = PeerFilterTable::new(100);
        let mut bf = BloomFilter::new(100, 0.01);
        let addr = harmony_athenaeum::PageAddr::new(0xDEADBEEF, harmony_athenaeum::Algorithm::Sha256Msb);
        bf.insert_bytes(&addr.to_bytes());

        table.upsert_page("peer-a".to_string(), bf, 10);

        // Should find the inserted address
        assert!(table.should_query_page("peer-a", &addr, 10));
        // Should NOT find a different address
        let other = harmony_athenaeum::PageAddr::new(0xCAFEBABE, harmony_athenaeum::Algorithm::Sha256Msb);
        assert!(!table.should_query_page("peer-a", &other, 10));
        // Unknown peer → always query
        assert!(table.should_query_page("peer-b", &addr, 10));
    }

    #[test]
    fn peer_page_filter_staleness() {
        let mut table = PeerFilterTable::new(100);
        let mut bf = BloomFilter::new(100, 0.01);
        let addr = harmony_athenaeum::PageAddr::new(0xDEADBEEF, harmony_athenaeum::Algorithm::Sha256Msb);
        bf.insert_bytes(&addr.to_bytes());
        table.upsert_page("peer-a".to_string(), bf, 10);

        // At tick 10 + 101 = 111, the filter is stale → should query even for absent items
        let absent = harmony_athenaeum::PageAddr::new(0x00000000, harmony_athenaeum::Algorithm::Sha256Msb);
        assert!(table.should_query_page("peer-a", &absent, 111));
    }
```

- [ ] **Step 8: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

- [ ] **Step 9: Commit**

```bash
git add crates/harmony-node/src/page_index.rs crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): PeerFilterTable page filter slot + PageIndex accessor

Add page_filter/page_received_tick to PeerFilter, upsert_page,
should_query_page with staleness. addr00_iter() on PageIndex exposes
mode-00 keys for Bloom filter construction.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Build, broadcast, and consume page filter

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

- [ ] **Step 1: Add `pending_page_broadcast` field to `NodeRuntime`**

In the `NodeRuntime` struct fields (~line 473, after `pending_memo_broadcast`), add:

```rust
    pending_page_broadcast: Option<Vec<u8>>,
```

Initialize in the `new()` constructor alongside the others (~line 623):

```rust
            pending_page_broadcast: None,
```

- [ ] **Step 2: Build page filter on timer tick**

In the `tick` method, after the memo filter rebuild block (~line 1132, before the closing brace of `if timer_due`), add:

```rust
                        // Rebuild page Bloom filter from PageIndex mode-00 addresses.
                        if !self.page_index.is_empty() {
                            let mut page_filter = BloomFilter::new(
                                self.filter_broadcast_config.expected_items,
                                self.filter_broadcast_config.fp_rate,
                            );
                            for addr in self.page_index.addr00_iter() {
                                page_filter.insert_bytes(&addr.to_bytes());
                            }
                            self.pending_page_broadcast = Some(page_filter.to_bytes());
                        }
```

- [ ] **Step 3: Flush page filter broadcast**

After the memo broadcast flush (~line 1151, after `pending_memo_broadcast.take()`), add:

```rust
                    if let Some(payload) = self.pending_page_broadcast.take() {
                        let key_expr =
                            harmony_zenoh::namespace::filters::page_key(&self.node_addr);
                        actions.push(RuntimeAction::Publish { key_expr, payload });
                    }
```

- [ ] **Step 4: Consume page filter in `route_subscription`**

In `route_subscription`, after the memo filter consumption block (~line 1978), add:

```rust
        // Check if this is a page filter broadcast.
        if let Some(peer_addr) = key_expr
            .strip_prefix(harmony_zenoh::namespace::filters::PAGE_PREFIX)
            .and_then(|s| s.strip_prefix('/'))
        {
            if peer_addr != self.node_addr {
                match BloomFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters.upsert_page(
                            peer_addr.to_string(),
                            filter,
                            self.tick_count,
                        );
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }
```

Note: The exact line numbers may differ. Find the pattern: the existing memo filter block ends with `return;`, add the page filter block after it, following the same structure.

- [ ] **Step 5: Add test**

In the `tests` module:

```rust
    #[test]
    fn page_filter_broadcast_on_timer() {
        use harmony_athenaeum::{Book, PAGE_SIZE};

        let (mut rt, _startup) = make_runtime();

        // Store a book so the page index has entries
        let data = vec![0xAA; PAGE_SIZE];
        let cid_bytes = [0x11u8; 32];
        let book = Book::from_book(cid_bytes, &data).unwrap();
        let cid = harmony_content::ContentId::from_bytes(cid_bytes);

        // Simulate: push the book data into storage and index it
        rt.page_index.insert_book(cid, &book);

        // Tick with timer due — should produce page filter broadcast
        let prev_tick = rt.ticks_since_filter_broadcast;
        rt.ticks_since_filter_broadcast = rt.filter_broadcast_config.max_interval_ticks;
        let actions = rt.tick(rt.last_now + 1);

        // Find the page filter Publish action
        let page_publishes: Vec<_> = actions
            .iter()
            .filter(|a| matches!(a, RuntimeAction::Publish { key_expr, .. }
                if key_expr.starts_with("harmony/filters/page/")))
            .collect();
        assert_eq!(page_publishes.len(), 1, "expected one page filter broadcast");
    }
```

Note: This test accesses `rt.page_index` directly and `rt.ticks_since_filter_broadcast`. Check that these fields are `pub(crate)` or accessible from tests. If `page_index` is private, either make it `pub(crate)` or feed a book through the normal `AnnounceContent` path. Check `make_runtime()` helper in the test module for the existing pattern — the memo broadcast test (`timer_skipped_when_threshold_broadcast_pending`) shows how to set up the timer trigger.

- [ ] **Step 6: Run tests**

Run: `cargo test -p harmony-node`
Expected: all tests pass

- [ ] **Step 7: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): build, broadcast, and consume page Bloom filter

Build page filter from PageIndex mode-00 addresses on FilterTimerTick.
Broadcast to harmony/filters/page/{node_addr}. Consume in
route_subscription, store in PeerFilterTable.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Full verification

- [ ] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`

- [ ] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-content -p harmony-zenoh -p harmony-node`

- [ ] **Step 3: Run fmt**

Run: `cargo fmt --all -- --check`

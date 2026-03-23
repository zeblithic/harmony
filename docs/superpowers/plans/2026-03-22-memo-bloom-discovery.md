# P2P Memo Discovery via Bloom Filters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire memo Bloom filter broadcasting and receiving into the NodeRuntime so peers can discover which memos their neighbors have before querying.

**Architecture:** Extend PeerFilterTable with memo filter fields, add memo filter broadcast on the timer tick (alongside content/flatpack), route incoming memo filter subscriptions, and add startup subscription. All changes in a single file (runtime.rs).

**Tech Stack:** Rust, harmony-content (BloomFilter), harmony-memo (MemoStore), harmony-zenoh (namespace)

**Spec:** `docs/superpowers/specs/2026-03-22-memo-bloom-discovery-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-node/src/runtime.rs` | All changes: PeerFilter/PeerFilterTable extension, broadcast, receive, startup subscription |

---

### Task 1: Extend PeerFilter and PeerFilterTable with memo fields

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

Add memo filter fields to PeerFilter and methods to PeerFilterTable, following the existing content/flatpack pattern.

- [x] **Step 1: Add memo fields to PeerFilter struct**

In the `PeerFilter` struct (around line 235), add:

```rust
struct PeerFilter {
    content_filter: Option<BloomFilter>,
    flatpack_filter: Option<CuckooFilter>,
    memo_filter: Option<BloomFilter>,       // NEW
    content_received_tick: u64,
    flatpack_received_tick: u64,
    memo_received_tick: u64,                 // NEW
}
```

Update ALL places that construct `PeerFilter { ... }` to include the new fields:
- `upsert_content()` — add `memo_filter: None, memo_received_tick: 0`
- `upsert_flatpack()` — add `memo_filter: None, memo_received_tick: 0`

- [x] **Step 2: Add upsert_memo method**

Following the pattern of `upsert_content()` and `upsert_flatpack()`:

```rust
fn upsert_memo(&mut self, peer_addr: String, filter: BloomFilter, tick: u64) {
    let entry = self.filters.entry(peer_addr).or_insert(PeerFilter {
        content_filter: None,
        flatpack_filter: None,
        memo_filter: None,
        content_received_tick: 0,
        flatpack_received_tick: 0,
        memo_received_tick: tick,
    });
    entry.memo_filter = Some(filter);
    entry.memo_received_tick = tick;
}
```

- [x] **Step 3: Add should_query_memo method**

Following the pattern of `should_query()` and `should_query_flatpack()`:

```rust
/// Returns true if the peer should be queried for memos about `input_cid`.
/// Same logic as content: unknown peer → true, stale → true, fresh → may_contain.
fn should_query_memo(
    &self,
    peer_addr: &str,
    input_cid: &ContentId,
    current_tick: u64,
) -> bool {
    match self.filters.get(peer_addr) {
        None => true,
        Some(pf) => {
            if current_tick.saturating_sub(pf.memo_received_tick) > self.staleness_ticks {
                true
            } else {
                match &pf.memo_filter {
                    Some(bf) => bf.may_contain(input_cid),
                    None => true,
                }
            }
        }
    }
}
```

- [x] **Step 4: Add tests**

Add to the existing PeerFilterTable test section:

```rust
#[test]
fn peer_memo_filter_upsert_and_query() {
    let mut table = PeerFilterTable::new(100);
    let mut filter = BloomFilter::new(100, 0.01);
    let cid_present = ContentId::from_bytes([0x11; 32]);
    let cid_absent = ContentId::from_bytes([0x22; 32]);
    filter.insert(&cid_present);

    table.upsert_memo("peer1".to_string(), filter, 10);

    // Present CID should be queryable
    assert!(table.should_query_memo("peer1", &cid_present, 10));
    // Absent CID should NOT be queryable (filter says no)
    assert!(!table.should_query_memo("peer1", &cid_absent, 10));
    // Unknown peer should be queryable
    assert!(table.should_query_memo("unknown", &cid_absent, 10));
}

#[test]
fn peer_memo_filter_staleness() {
    let mut table = PeerFilterTable::new(10); // stale after 10 ticks
    let filter = BloomFilter::new(100, 0.01); // empty filter
    let cid = ContentId::from_bytes([0x33; 32]);

    table.upsert_memo("peer1".to_string(), filter, 5);

    // Fresh: empty filter says "definitely not" → false
    assert!(!table.should_query_memo("peer1", &cid, 10));
    // Stale: > staleness_ticks since received → true (query anyway)
    assert!(table.should_query_memo("peer1", &cid, 20));
}
```

- [x] **Step 5: Run tests**

Run: `cargo test -p harmony-node -- peer_memo`
Expected: both new tests pass

- [x] **Step 6: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): extend PeerFilterTable with memo filter

Add memo_filter + memo_received_tick to PeerFilter struct.
upsert_memo() and should_query_memo() follow existing content/flatpack
pattern. 2 new tests.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Memo filter broadcast on timer tick

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

Add memo filter rebuild and publish alongside existing content/flatpack filter broadcasts.

- [x] **Step 1: Add memo filter broadcast after content/flatpack broadcasts**

In the timer tick handler, after the existing `pending_cuckoo_broadcast` flush (around line 966), add:

```rust
                    // Memo filter: rebuild from MemoStore input CIDs and broadcast.
                    // Timer-only (no mutation counter) since memos arrive infrequently.
                    if timer_due {
                        let mut memo_bloom = BloomFilter::new(
                            self.filter_config.expected_items,
                            self.filter_config.fp_rate,
                        );
                        for cid in self.memo_store.input_cids() {
                            memo_bloom.insert(cid);
                        }
                        // Only broadcast if we have any memos
                        if !self.memo_store.is_empty() {
                            let key_expr =
                                harmony_zenoh::namespace::filters::memo_key(&self.node_addr);
                            actions.push(RuntimeAction::Publish {
                                key_expr,
                                payload: memo_bloom.to_bytes(),
                            });
                        }
                    }
```

Note: `timer_due` is already computed earlier in the tick handler. The memo broadcast runs in the same `if timer_due` block as the content filter timer rebuild, OR as a separate block right after the content/flatpack flush. Check the exact control flow and place it appropriately — it must only fire when `timer_due` is true.

IMPORTANT: The `filter_config` field may be named differently. Check the actual struct field name (it's likely `self.filter_broadcast_config` or similar). Also check if `BloomFilter` is imported — it should be via `harmony_content::bloom::BloomFilter`.

- [x] **Step 2: Add test**

```rust
#[test]
fn memo_filter_broadcast_on_timer_tick() {
    // Create a NodeRuntime with a MemoStore containing some memos
    // Trigger a timer tick
    // Verify that a Publish action with harmony/filters/memo/ key is emitted
}
```

This test may be complex to set up (NodeRuntime requires full initialization). If constructing a full runtime is too involved, test at the PeerFilterTable + BloomFilter level instead:

```rust
#[test]
fn memo_filter_contains_inserted_inputs() {
    let mut filter = BloomFilter::new(100, 0.01);
    let cid1 = ContentId::from_bytes([0x11; 32]);
    let cid2 = ContentId::from_bytes([0x22; 32]);
    let cid_absent = ContentId::from_bytes([0x33; 32]);

    filter.insert(&cid1);
    filter.insert(&cid2);

    assert!(filter.may_contain(&cid1));
    assert!(filter.may_contain(&cid2));
    assert!(!filter.may_contain(&cid_absent));
}
```

- [x] **Step 3: Run tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [x] **Step 4: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): broadcast memo Bloom filter on timer tick

Rebuild filter from MemoStore.input_cids() on each filter timer tick.
Publish to harmony/filters/memo/{node_addr}. Timer-only, no mutation
counter. Only broadcasts when MemoStore is non-empty.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Route incoming memo filter subscriptions + startup subscription

**Files:**
- Modify: `crates/harmony-node/src/runtime.rs`

Add subscription routing for `harmony/filters/memo/` and startup subscription.

- [x] **Step 1: Add startup subscription**

In the `new()` constructor, where the startup actions are built (around line 494), add alongside the existing CONTENT_SUB and FLATPACK_SUB:

```rust
        RuntimeAction::Subscribe {
            key_expr: harmony_zenoh::namespace::filters::MEMO_SUB.to_string(),
        },
```

- [x] **Step 2: Add memo filter routing in route_subscription**

In the `route_subscription()` method, add a branch for memo filters. Find where content and flatpack filters are routed (look for `strip_prefix` on `CONTENT_PREFIX` and `FLATPACK_PREFIX`). Add a similar branch:

```rust
        // Memo filter broadcast from a peer
        if let Some(rest) = key_expr.strip_prefix(harmony_zenoh::namespace::filters::MEMO_PREFIX) {
            let peer_addr = rest.strip_prefix('/').unwrap_or(rest);
            if peer_addr != self.node_addr {
                match BloomFilter::from_bytes(&payload) {
                    Ok(filter) => {
                        self.peer_filters
                            .upsert_memo(peer_addr.to_string(), filter, self.tick_count);
                    }
                    Err(_) => {
                        self.peer_filters.record_parse_error();
                    }
                }
            }
            return;
        }
```

Place this AFTER the existing content and flatpack filter routing, following the same pattern exactly.

- [x] **Step 3: Add test**

```rust
#[test]
fn memo_filter_subscription_routed() {
    // This test verifies that a subscription message with the memo filter
    // prefix is correctly parsed and stored in PeerFilterTable.
    //
    // Create a PeerFilterTable, build a BloomFilter with a known CID,
    // serialize it, and call the routing logic. Verify should_query_memo
    // returns the correct result.

    let mut table = PeerFilterTable::new(100);
    let mut filter = BloomFilter::new(100, 0.01);
    let known_cid = ContentId::from_bytes([0xAA; 32]);
    filter.insert(&known_cid);

    // Simulate what route_subscription does:
    table.upsert_memo("remote_peer".to_string(), filter, 50);

    assert!(table.should_query_memo("remote_peer", &known_cid, 50));
    assert!(!table.should_query_memo("remote_peer", &ContentId::from_bytes([0xBB; 32]), 50));
}
```

Note: Testing the full `route_subscription` path requires a NodeRuntime, which may be complex. The PeerFilterTable-level test above verifies the key logic. If you can test route_subscription directly, add a test that feeds a `SubscriptionMessage` with `harmony/filters/memo/peer1` key and verifies the filter is stored.

- [x] **Step 4: Add a public accessor for memo queries**

Expose `should_query_memo` through the runtime's public API:

```rust
/// Check if a peer should be queried for memos about the given input CID.
pub fn should_query_memo_peer(&self, peer_addr: &str, input_cid: &ContentId) -> bool {
    self.peer_filters.should_query_memo(peer_addr, input_cid, self.tick_count)
}
```

- [x] **Step 5: Run tests**

Run: `cargo test -p harmony-node -v`
Expected: all tests pass

- [x] **Step 6: Commit**

```bash
git add crates/harmony-node/src/runtime.rs
git commit -m "feat(runtime): route memo filter subscriptions, startup subscribe

Subscribe to harmony/filters/memo/** at startup. Route incoming memo
filters to PeerFilterTable.upsert_memo(). Expose should_query_memo_peer()
for callers to check before querying peers.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Full verification

**Files:** None (verification only)

- [x] **Step 1: Run full workspace tests**

Run: `cargo test --workspace`
Expected: all tests pass

- [x] **Step 2: Run clippy**

Run: `cargo clippy -p harmony-node`
Expected: no new warnings

- [x] **Step 3: Run fmt**

Run: `cargo fmt --all -- --check`
Expected: clean

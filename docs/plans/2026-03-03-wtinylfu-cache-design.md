# W-TinyLFU Cache Design

**Goal:** Add scan-resistant, frequency-aware caching to harmony-content so that sequential access patterns (backups, indexing) don't poison the cache by evicting hot content.

**Architecture:** Three layered modules — a Count-Min Sketch for frequency estimation, a slab-allocated LRU for eviction ordering, and a `ContentStore<S: BlobStore>` wrapper that wires them together with W-TinyLFU admission policy. Single-tier, in-memory, entry-count-based capacity. Sans-I/O, no new dependencies.

---

## Why W-TinyLFU

**LRU fails:** A sequential scan touches thousands of blobs once, evicts all hot content, then those scanned blobs are never accessed again.

**LFU fails:** Exact frequency tracking for millions of CIDs uses enormous memory. Items popular last month accumulate unassailable scores and squat forever.

**W-TinyLFU:** Approximate frequency (Count-Min Sketch, ~400KB) + recency (segmented LRU) + periodic decay (counter halving). New items get a small window to build frequency before facing an admission challenge against the main cache's least-valuable resident.

## Module Structure

| Module | Contents |
|--------|----------|
| `sketch.rs` | Count-Min Sketch with 4-bit pair-packed counters |
| `lru.rs` | Slab-allocated doubly-linked-list LRU |
| `cache.rs` | `ContentStore<S: BlobStore>` — W-TinyLFU admission, pin support |

## Count-Min Sketch (`sketch.rs`)

4 hash functions, each mapping to a row of 4-bit counters pair-packed into a `Vec<u8>`.

```rust
pub struct CountMinSketch {
    counters: Vec<u8>,       // 4-bit counters, 2 per byte
    width: usize,            // counters per row
    total_increments: u64,
    halving_threshold: u64,
}
```

**Operations:**
- `increment(cid)` — Hash CID with 4 seeds, increment each position (saturating at 15). Auto-halve when threshold reached.
- `estimate(cid)` — Minimum of the 4 counter values. Never underestimates.
- `halve()` — Right-shift every counter by 1. Decays stale popularity.

**Hashing:** CID bytes are already SHA-256 — extract two `u64` values from the first 16 bytes, derive 4 row indices via `h0.wrapping_mul(seed) ^ h1.wrapping_mul(seed.rotate_left(32))`. Both halves are mixed per-row for pairwise independence. No additional hash function needed.

**Sizing:** Width = `2 * max_entries` counters per row. Memory = `4 * width / 2` bytes.

## LRU Linked List (`lru.rs`)

Slab-allocated doubly-linked list. Used three times: Window, Probation, Protected.

```rust
pub struct Lru {
    entries: Vec<LruEntry>,
    map: HashMap<ContentId, usize>,  // CID -> slab index
    head: Option<usize>,             // most recently used
    tail: Option<usize>,             // eviction candidate
    free: Vec<usize>,                // recycled slots
    len: usize,
    capacity: usize,
}
```

**Operations:**
- `touch(cid)` — Move to head if present.
- `insert(cid) -> Option<ContentId>` — Insert at head. If over capacity, evict tail and return evicted CID.
- `remove(cid) -> bool` — Unlink and recycle slot.
- `peek_lru() -> Option<ContentId>` — Peek at tail without removing.

**Why slab allocation:** Avoids per-node heap allocations. Pre-allocated to capacity. Removed entries go on free list for reuse. Cache-friendly contiguous memory.

## ContentStore (`cache.rs`)

```rust
pub struct ContentStore<S: BlobStore> {
    store: S,
    sketch: CountMinSketch,
    window: Lru,                  // 1% of capacity
    probation: Lru,               // ~79% of capacity
    protected: Lru,               // ~20% of capacity
    pinned: HashSet<ContentId>,
    capacity: usize,
}
```

**Capacity split:**
- Window: `max(1, N / 100)`
- Protected: `N / 5`
- Probation: `N - window - protected`

### get(cid)

1. Increment CID in sketch.
2. If in protected — touch, return data.
3. If in probation — promote to protected. If protected overflows, demote victim to probation head.
4. If in window — touch, return data.
5. Miss — return `None`.

### put(cid, data)

1. Already cached or pinned — store data, done.
2. Insert into window.
3. If window overflows, admission challenge:
   - `sketch.estimate(window_evictee)` vs `sketch.estimate(probation.peek_lru())`
   - Higher frequency wins admission to probation. Loser is dropped.

### Pin/Unpin with Quota

- `pin(cid) -> bool` — Add to pinned set if quota allows. Returns false when limit reached.
- `unpin(cid)` — Remove from pinned set. CID becomes evictable.
- `pin_limit()` — Returns `capacity / 2`. At most half the cache can be pinned.
- If eviction candidate is pinned, scan to next non-pinned tail entry via `peek_lru_excluding`.
- Pinned candidates (evicted from window) are always admitted to probation — never dropped.

**Storage model:** The pin quota enforces a 50/50 split between shared network
caching (evictable) and user-owned replicated content (pinned). A node
contributing *N* slots of storage gets *N/2* slots of guaranteed replication.
This means: for every 1TB drive plugged into the network, the contributor gets
~500GB of "permanent cloud storage" — data replicated across multiple nodes,
resistant to any single-node failure. The other 500GB serves the network as
freely evictable cache, improving performance for everyone.

**Quota invariant:** Because `pin_limit = capacity / 2` and `probation_cap ≈ 79%
of capacity`, a pinned candidate always finds at least one non-pinned victim in
probation during the admission challenge. This eliminates the need for
force-insert or capacity overflow logic.

### BlobStore implementation

`ContentStore` implements `BlobStore` — drop-in replacement for existing consumers.

## Edge Cases

- **Pin quota:** Max pinned items = `capacity / 2`. `pin()` returns `false` when quota reached.
- **Zero-capacity segments:** `protected_cap = 0` for capacity ≤ 4. The LRU rejects inserts immediately (returns `Some(cid)`), so promotions bounce back to probation. Correct behavior.
- **Zero-capacity cache:** `ContentStore::new` panics if capacity < 1. A zero-item cache is nonsensical.
- **Capacity=1:** Only window segment active (probation=0, protected=0). Pin quota=0, so pinning is disabled. Items cycle through window only.
- **Thread safety:** Not included. Single-threaded like the rest of harmony-content. Caller manages concurrency.
- **Error handling:** No new error variants. Sketch and LRU are infallible. Storage errors delegate to inner BlobStore.

## Testing (24 tests)

**Sketch (5):**
1. Increment and estimate returns at least true count
2. Never underestimates (Count-Min guarantee)
3. Halving decays counters to roughly half
4. Auto-halving triggers at threshold
5. Hash rows are independent (per-row h1 mixing)

**LRU (5):**
6. Eviction order — oldest entry evicted first
7. Touch promotes to head
8. Remove frees slot for reuse
9. `peek_lru_excluding` skips excluded entries
10. Zero-capacity LRU rejects inserts

**ContentStore (11):**
11. Implements BlobStore (get/insert/contains)
12. `get_and_record` updates frequency and promotes
13. Frequency-based admission — low-frequency newcomer loses
14. Probation-to-protected promotion on access
15. Scan resistance — cold scan does not evict hot CID
16. Pin exempts from eviction under pressure
17. Capacity-1 segments don't exceed capacity
18. Capacity-2 segments don't exceed capacity
19. Zero capacity panics
20. Pinned candidate survives admission challenge (regression)
21. Pinned victim skipped in admission (regression)
22. Pin quota rejects over limit
23. Pin quota zero for capacity-1
24. Zero-cap protected bounces to probation

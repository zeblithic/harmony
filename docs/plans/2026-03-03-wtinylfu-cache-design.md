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

**Hashing:** CID bytes are already SHA-256 — extract two `u64` values from the first 16 bytes, derive 4 row indices via multiply-xor with different constants. No additional hash function needed.

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

### Pin/Unpin

- `pin(cid)` — Add to pinned set. Eviction logic skips pinned CIDs.
- `unpin(cid)` — Remove from pinned set. CID becomes evictable.
- If eviction candidate is pinned, skip to next tail entry.

### BlobStore implementation

`ContentStore` implements `BlobStore` — drop-in replacement for existing consumers.

## Edge Cases

- **Pin overflow:** If everything is pinned, cache grows beyond capacity. No pin limit (YAGNI).
- **Thread safety:** Not included. Single-threaded like the rest of harmony-content. Caller manages concurrency.
- **Error handling:** No new error variants. Sketch and LRU are infallible. Storage errors delegate to inner BlobStore.

## Testing (~12 tests)

**Sketch (4):**
1. Increment and estimate returns at least true count
2. Never underestimates (Count-Min guarantee)
3. Halving decays counters to roughly half
4. Auto-halving triggers at threshold

**LRU (3):**
5. Eviction order — oldest entry evicted first
6. Touch promotes to head
7. Remove frees slot for reuse

**ContentStore (5):**
8. Scan resistance — sequential cold scan does not evict hot CID
9. Frequency-based admission — low-frequency newcomer loses to high-frequency resident
10. Probation-to-protected promotion on access
11. Pin exempts from eviction under pressure
12. ContentStore implements BlobStore (get/insert/contains work)

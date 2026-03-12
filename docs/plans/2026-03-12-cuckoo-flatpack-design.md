# Cuckoo Filter & Flatpack Reverse Index Design

> **Goal:** Distributed reverse index mapping `child_cid → [bundle_cids...]` so nodes can answer "which bundles reference this blob?" — enabling GC safety, dedup discovery, and impact analysis. Cuckoo filters broadcast index coverage summaries so peers skip pointless reverse-lookup queries.

**Approach:** Per-node Flatpack index built from admitted bundles, with cuckoo filter broadcasts for query pre-filtering. Request-response queries for on-demand reverse lookups. Local-only GC safety — each node protects blobs referenced by its own bundles.

**Depends on:** Bloom filter infrastructure (harmony-x76) landing first, establishing broadcast patterns, `PeerFilterTable`, and `FilterBroadcastConfig`.

## Problem

When a node needs to know which bundles reference a given blob — for GC safety (don't evict a blob that a pinned bundle needs), dedup discovery (which bundles share this blob?), or impact analysis (if this blob is corrupt, what's affected?) — there's no index to answer this. Bundles store their child CID arrays, but there's no reverse mapping from child → parent.

## Use Cases

| Use Case | Query | Tolerance for False Negatives |
|----------|-------|-------------------------------|
| **GC safety** | Is this blob referenced by any local bundle? | Zero (local) — false negative = data loss |
| **Dedup discovery** | Which bundles share this blob? | High — missed optimization is harmless |
| **Impact analysis** | If this blob is bad, what bundles are affected? | Medium — understated blast radius is recoverable |

GC safety uses the local index only (no network query needed). Dedup and impact analysis may query the network for a broader picture.

## Cuckoo Filter Data Structure

New module: `harmony-content/src/cuckoo.rs`, following the same pattern as `bloom.rs` and `sketch.rs`.

### Geometry

4-entry buckets, 12-bit fingerprints (stored as `u16`), 95% max load factor.

| Items (n) | Bits/item | Total Size | FP Rate |
|-----------|-----------|------------|---------|
| 100,000   | ~9        | ~110 KB    | <0.1%   |
| 10,000    | ~9        | ~11 KB     | <0.1%   |
| 1,000     | ~9        | ~1.1 KB    | <0.1%   |

### API

```rust
pub struct CuckooFilter {
    buckets: Vec<[u16; 4]>,  // 4 entries per bucket, 12-bit fingerprints in u16
    num_buckets: u32,
    count: u32,              // current item count
    max_kicks: u32,          // relocation limit (default: 500)
}

impl CuckooFilter {
    /// Create a filter sized for `capacity` items.
    pub fn new(capacity: u32) -> Self;

    /// Insert a CID. Returns Err(FilterFull) if relocation limit exceeded.
    pub fn insert(&mut self, cid: &ContentId) -> Result<(), FilterFull>;

    /// Delete a CID. Returns true if found and removed.
    pub fn delete(&mut self, cid: &ContentId) -> bool;

    /// Check if a CID might be in the filter.
    /// Returns false only if the CID is definitely absent.
    pub fn may_contain(&self, cid: &ContentId) -> bool;

    /// Current item count.
    pub fn count(&self) -> u32;

    /// Reset all buckets to zero.
    pub fn clear(&mut self);

    /// Serialize for broadcast.
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from broadcast payload.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FilterError>;
}
```

### Hashing

Standard cuckoo filter alternating-index trick:

```
fingerprint = hash(cid)[0..2] & 0x0FFF   // 12 bits
i1 = hash(cid) % num_buckets
i2 = i1 XOR hash(fingerprint) % num_buckets
```

Uses the same SplitMix64 mixing over `ContentId.hash` as bloom.rs and sketch.rs. No new dependencies needed.

## Flatpack Index

New module: `harmony-content/src/flatpack.rs`

### Data Structure

```rust
pub struct FlatpackIndex {
    /// child_cid → set of bundle_cids that reference it
    reverse: HashMap<ContentId, HashSet<ContentId>>,
    /// bundle_cid → list of child_cids (for cleanup on eviction)
    forward: HashMap<ContentId, Vec<ContentId>>,
    /// Cuckoo filter summarizing which child_cids have entries
    filter: CuckooFilter,
    /// Mutations since last filter broadcast
    mutations_since_broadcast: u32,
}
```

### API

```rust
impl FlatpackIndex {
    pub fn new(capacity: u32) -> Self;

    /// Called by StorageTier when a bundle is admitted.
    /// StorageTier parses the bundle and passes structured CID list.
    pub fn on_bundle_admitted(
        &mut self,
        bundle_cid: ContentId,
        child_cids: Vec<ContentId>,
    );

    /// Called by StorageTier when a bundle is evicted.
    /// Removes all reverse entries and deletes child CIDs from cuckoo filter
    /// (only if no other bundle references that child).
    pub fn on_bundle_evicted(&mut self, bundle_cid: &ContentId);

    /// Local reverse lookup: which bundles reference this child?
    pub fn lookup(&self, child_cid: &ContentId) -> Option<&HashSet<ContentId>>;

    /// GC safety check: is this child referenced by any local bundle?
    pub fn is_referenced(&self, child_cid: &ContentId) -> bool;

    /// Check if cuckoo filter should be broadcast.
    /// Returns Some(BroadcastCuckooFilter) if mutation threshold reached.
    pub fn check_broadcast_threshold(
        &mut self,
        config: &FilterBroadcastConfig,
    ) -> Option<BroadcastCuckooFilter>;

    /// Timer-triggered broadcast (even if no mutations).
    pub fn timer_triggered_broadcast(&mut self) -> BroadcastCuckooFilter;

    /// Rebuild cuckoo filter from scratch (all child_cids in reverse map).
    pub fn rebuild_filter(&mut self);
}
```

### Relationship to StorageTier

```
StorageTier
  ├── ContentStore (W-TinyLFU cache)
  ├── BloomFilter (content membership broadcast)
  └── FlatpackIndex (reverse map + cuckoo filter)
```

StorageTier calls into FlatpackIndex on bundle admission/eviction. FlatpackIndex returns actions (`BroadcastCuckooFilter`) that StorageTier surfaces alongside existing actions (`BroadcastFilter`, `AnnounceContent`).

### Bundle Parsing

StorageTier parses the bundle payload on admission to extract the child CID list, then passes `(bundle_cid, Vec<child_cid>)` to `FlatpackIndex::on_bundle_admitted()`. FlatpackIndex never touches raw bundle bytes — this keeps content-format knowledge in one place and makes the index trivially testable.

### Eviction Cascade

When `on_bundle_evicted(bundle_cid)` is called:

1. Look up `forward[bundle_cid]` to get the child CID list
2. For each child CID, remove `bundle_cid` from `reverse[child_cid]`
3. If `reverse[child_cid]` is now empty, remove the entry and call `filter.delete(child_cid)`
4. Remove `forward[bundle_cid]`
5. Increment `mutations_since_broadcast`

The cuckoo filter only deletes a child CID when *no* local bundle references it — this prevents false negatives for other bundles sharing the same child.

## Broadcast Infrastructure

### Zenoh Namespace

Addition to bloom's `harmony/filters/**`:

```
harmony/filters/flatpack/{node_addr}   — per-node cuckoo filter snapshot
harmony/filters/flatpack/**            — subscription pattern
```

### Wire Format

```
[4 bytes] num_buckets    (u32 big-endian)
[4 bytes] item_count     (u32 big-endian)
[remaining] bucket data  (num_buckets × 4 × 2 bytes, each u16 big-endian)
```

### Broadcast Trigger

Reuses `FilterBroadcastConfig` pattern from bloom filters. Mutation counter increments on `on_bundle_admitted` and `on_bundle_evicted`. Same hybrid timer/mutation-threshold logic, potentially with different default thresholds tuned for index churn.

## Query Protocol

### Zenoh Queryable Namespace

```
harmony/flatpack/{child_cid_hex}    — reverse-lookup query
harmony/flatpack/0** through f**    — 16-shard prefix subscription
```

### Query Flow

```
Need reverse lookup for child_cid
  → check local FlatpackIndex.lookup(child_cid)
  → if network query needed:
      → check PeerFilterTable cuckoo filters
      → skip peers whose cuckoo says "definitely no"
      → query remaining peers on harmony/flatpack/{child_cid_hex}
      → collect responses (with timeout)
      → merge with local results
```

Response payload: list of `bundle_cid`s (each 32 bytes).

## GC Integration

**Local-only safety.** Before evicting a blob from cache:

```rust
if flatpack_index.is_referenced(&blob_cid) {
    // Block eviction — a local bundle needs this blob
    // TODO(future): network-confirmed GC could query peers here
    //   to check if any remote pinned bundles also reference this blob.
    //   For now, local references are sufficient since each node
    //   protects its own cache independently.
    return EvictionDecision::Block;
}
```

Each node protects its own cached blobs. If another node pins a bundle referencing the same blob, that node's own index protects its copy.

## PeerFilterTable Extension

The existing `PeerFilterTable` in NodeRuntime (from bloom filter infrastructure) adds a second filter slot per peer:

```rust
pub struct PeerFilter {
    content_filter: Option<BloomFilter>,    // from bloom broadcasts
    flatpack_filter: Option<CuckooFilter>,  // from cuckoo broadcasts
    received_at: u64,
    // ...
}
```

Same staleness expiry logic: missing filter = "query this peer anyway."

## Coexistence with Bloom Filters

| | Bloom | Cuckoo |
|---|---|---|
| **Purpose** | "Do you have this CID cached?" | "Do you have reverse-index entries for this child CID?" |
| **Namespace** | `harmony/filters/content/{addr}` | `harmony/filters/flatpack/{addr}` |
| **Deletion** | Not needed (rebuilt from scratch) | Required (bundle eviction) |
| **Size (100K)** | ~175 KB | ~110 KB |
| **Lives in** | StorageTier (standalone) | FlatpackIndex (encapsulated) |

Both use the same PeerFilterTable, broadcast config pattern, and timer/mutation trigger.

## What Changes Where

| Crate | File | Change |
|-------|------|--------|
| `harmony-content` | `cuckoo.rs` (new) | `CuckooFilter` struct — insert, delete, query, serialize |
| `harmony-content` | `flatpack.rs` (new) | `FlatpackIndex` — reverse map, forward map, cuckoo filter, GC check |
| `harmony-content` | `storage_tier.rs` | Bundle admission/eviction hooks into FlatpackIndex, `BroadcastCuckooFilter` action, timer tick |
| `harmony-zenoh` | `namespace.rs` | `flatpack` module with key patterns, queryable builders |
| `harmony-node` | `runtime.rs` | PeerFilter gets cuckoo slot, subscribe to cuckoo broadcasts, cuckoo pre-check before flatpack queries |
| `harmony-node` | `main.rs` | CLI flags for flatpack config (if any differ from bloom defaults) |

## Non-Goals

- Full index gossip / pub-sub replication (query-only)
- Network-confirmed GC (local-only, extension point marked)
- Compressed cuckoo variants (semi-sorted, Morton — optimize later if needed)
- Cross-node Flatpack merging (each node's index is independent)
- Bundle format parsing in FlatpackIndex (StorageTier passes structured data)
- Cuckoo filter as replacement for bloom (complementary, different purposes)

## Testing

**Unit tests (cuckoo.rs):**
- Insert + `may_contain` round-trip
- `delete` removes item, `may_contain` returns false after
- False positive rate within 0.1% bounds
- Insert until full → `FilterFull` error
- Serialization round-trip (`to_bytes` / `from_bytes`)
- Alternating-index correctness (i2 = i1 XOR hash(fp))

**Unit tests (flatpack.rs):**
- `on_bundle_admitted` creates correct reverse + forward entries
- `on_bundle_evicted` removes reverse entries, deletes from cuckoo only when no other bundle references child
- `is_referenced` returns true/false correctly
- `lookup` returns correct bundle set
- Mutation counter increments, `check_broadcast_threshold` fires at threshold
- Timer-triggered broadcast works with zero mutations
- `rebuild_filter` matches reverse map contents

**Unit tests (storage_tier.rs):**
- Bundle admission triggers FlatpackIndex
- Bundle eviction triggers FlatpackIndex cleanup
- GC blocked when blob is referenced by local bundle
- `BroadcastCuckooFilter` action emitted alongside existing actions

**Integration tests:**
- Full cycle: admit bundle → index built → cuckoo broadcast → deserialize → `may_contain` confirms child CIDs
- GC safety: pin bundle → attempt evict referenced blob → blocked
- Eviction cascade: evict bundle → reverse entries removed → blob eviction unblocked
- Query flow: local miss → cuckoo pre-check → network query → response merge

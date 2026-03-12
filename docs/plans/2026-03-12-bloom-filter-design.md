# Bloom Filter Content Discovery Design

> **Goal:** Nodes periodically broadcast Bloom filters summarizing their cached CID set so that peers can skip content queries to nodes that definitely don't have the content.

**Approach:** Per-node Bloom filter rebuilt from scratch on each broadcast, triggered by a hybrid timer/mutation-threshold policy. Receivers maintain a per-peer filter table and check it before dispatching content queries.

## Problem

Today, when a node needs content, it queries all peers listening on the relevant shard (`harmony/content/{prefix}/**`). Most peers don't have the content, so most queries are wasted — the querier waits for timeouts from peers that have nothing to offer.

## Content Classes & Filter Membership

The filter only contains CIDs that the node would actually serve — it respects `ContentPolicy`:

| Class | In filter? |
|-------|-----------|
| PublicDurable (00) | Always |
| PublicEphemeral (01) | Always |
| EncryptedDurable (10) | Only if `encrypted_durable_persist` |
| EncryptedEphemeral (11) | Never |

This ensures the filter accurately reflects what the node can answer queries for.

## Bloom Filter Data Structure

New module: `harmony-content/src/bloom.rs`, following the same pattern as `sketch.rs`.

### Sizing

Classic Bloom filter formula: `m = -n * ln(p) / (ln2)^2`, `k = (m/n) * ln2`.

| Items (n) | FP Rate (p) | Bits (m) | Bytes | Hashes (k) |
|-----------|-------------|----------|-------|------------|
| 100,000 | 0.1% | 1,437,759 | ~175 KB | 10 |
| 10,000 | 0.1% | 143,776 | ~17.5 KB | 10 |
| 1,000 | 0.1% | 14,378 | ~1.8 KB | 10 |

Filter size scales linearly with expected items. The `expected_items` parameter should match the node's `cache_capacity` config.

### API

```rust
pub struct BloomFilter {
    bits: Vec<u64>,     // bit array packed into u64 words
    num_hashes: u32,    // k
    num_bits: u32,      // m
}

impl BloomFilter {
    /// Create a filter sized for `expected_items` at the given false positive rate.
    pub fn new(expected_items: u32, fp_rate: f64) -> Self;

    /// Insert a CID into the filter.
    pub fn insert(&mut self, cid: &ContentId);

    /// Check if a CID might be in the filter.
    /// Returns false only if the CID is definitely absent.
    pub fn may_contain(&self, cid: &ContentId) -> bool;

    /// Reset all bits to zero.
    pub fn clear(&mut self);

    /// Serialize for broadcast.
    pub fn to_bytes(&self) -> Vec<u8>;

    /// Deserialize from broadcast payload.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, FilterError>;

    /// Approximate item count from bit density (useful for staleness heuristics).
    pub fn estimated_count(&self) -> u32;
}
```

### Hashing

Kirsch-Mitzenmacher optimization: derive k hash indices from 2 base hashes.

```
h_i(x) = h1(x) + i * h2(x)   mod m
```

Uses the same SplitMix64 mixing from `sketch.rs` over the 28-byte `ContentId.hash` field:

```rust
let h0 = u64::from_le_bytes(hash[0..8]);
let h1 = u64::from_le_bytes(hash[8..16]);
// h0 and h1 serve as the two base hashes
// k indices derived via: (h0 + i * h1) % num_bits
```

No new dependencies needed.

## Broadcast Infrastructure

### Zenoh Namespace

New `filters` module in `harmony-zenoh/src/namespace.rs`:

```
harmony/filters/content/{node_addr}    — per-node Bloom filter snapshot
harmony/filters/content/**             — subscription pattern
```

One filter per node (not per-shard). Sharding the filter would create 16× the broadcast messages for no benefit — the total bit count is the same and `may_contain` is O(k) regardless.

### Wire Format

```
[4 bytes] num_bits    (u32 big-endian)
[4 bytes] num_hashes  (u32 big-endian)
[4 bytes] item_count  (u32 big-endian, from estimated_count())
[remaining bytes] bit array (ceil(num_bits / 8) bytes)
```

The `item_count` lets receivers gauge staleness — if a peer's filter claims 50K items but their cache capacity is 1K, something is off.

### Broadcast Trigger (Hybrid Timer/Threshold)

```rust
pub struct FilterBroadcastConfig {
    /// Broadcast after this many cache admissions.
    pub mutation_threshold: u32,       // default: 100
    /// Maximum ticks between broadcasts (even if no mutations).
    pub max_interval_ticks: u32,       // default: 30
}
```

**Mutation-triggered:** StorageTier tracks a `mutations_since_broadcast` counter. On every admission, it increments. When it reaches `mutation_threshold`, StorageTier rebuilds the filter and returns a `BroadcastFilter` action.

**Timer-triggered:** The runtime injects `FilterTimerTick` events at `max_interval_ticks` intervals. StorageTier responds by rebuilding and broadcasting even if no mutations occurred (guarantees freshness ceiling).

**Rebuild process:** Iterate all CIDs via `ContentStore::iter_admitted()` (window + probation + protected segments), insert each into a fresh `BloomFilter`, serialize, emit `BroadcastFilter` action.

For 100K items, iteration + insertion is sub-millisecond (28-byte hash × 10 hash functions × 100K = ~28M simple operations).

## Receiving & Routing

### Per-Peer Filter Table

Lives in `NodeRuntime` (the I/O bridge), not in `StorageTier` (which doesn't know about peers).

```rust
pub struct PeerFilterTable {
    filters: HashMap<String, PeerFilter>,  // node_addr -> filter + metadata
}

pub struct PeerFilter {
    filter: BloomFilter,
    received_at: u64,       // runtime-injected timestamp
    item_count: u32,        // from broadcast header
}
```

**Staleness expiry:** Drop filters not refreshed within `3 × max_interval_ticks` (default 90 ticks). A missing filter means "query this peer anyway" — safe default, no false negatives.

### Query Routing Decision

When a content query produces a cache miss, before the runtime fans out to peers:

```
for each peer on the relevant shard:
    if peer has a filter:
        if filter says "definitely not":
            skip  ← saved a wasted query
        else:
            query ← might have it (could be false positive)
    else:
        query  ← no filter = assume they might have it
```

### Coexistence with Announcements

Individual per-CID announcements (`harmony/announce/{cid_hex}`) remain unchanged. Bloom filters are complementary:

- **Announcements** = real-time push: "I just got CID X"
- **Bloom filters** = periodic snapshot: "here's everything I have"

A peer might receive an announcement for a CID not yet in the sender's latest filter (built before admission). Both mechanisms work independently.

## What Changes Where

| Crate | File | Change |
|-------|------|--------|
| `harmony-content` | `bloom.rs` (new) | `BloomFilter` struct with insert, query, serialize, sizing |
| `harmony-content` | `storage_tier.rs` | `FilterBroadcastConfig`, mutation counter, `BroadcastFilter` action, `FilterTimerTick` event, rebuild logic |
| `harmony-content` | `cache.rs` | `iter_admitted()` on `ContentStore` — iterates CIDs in all three LRU segments |
| `harmony-zenoh` | `namespace.rs` | `filters` module with key patterns and builders |
| `harmony-node` | `runtime.rs` | `PeerFilterTable`, subscribe to filter broadcasts, filter check before peer dispatch |
| `harmony-node` | `main.rs` | CLI flags: `--filter-broadcast-interval`, `--filter-mutation-threshold` |

## Non-Goals

- Cuckoo filters / Flatpack reverse index (separate bead)
- Shard-level filter splitting (one filter per node is sufficient)
- Filter compression (raw bits at 175 KB; zstd later if needed)
- Peer trust / filter authentication (future concern)
- Filter-based routing for transit content (only query routing)

## Testing

**Unit tests (bloom.rs):**
- Insert + `may_contain` round-trip
- Empty filter returns false for everything
- False positive rate within expected bounds (insert n items, check m random non-members)
- Serialization round-trip (`to_bytes` / `from_bytes`)
- `estimated_count` accuracy
- Kirsch-Mitzenmacher hash distribution (no degenerate clustering)

**Unit tests (storage_tier.rs):**
- Mutation counter increments on admission/eviction
- `BroadcastFilter` action emitted at threshold
- `FilterTimerTick` triggers broadcast even with zero mutations
- Filter contains only policy-admitted CIDs (no EncryptedEphemeral)

**Unit tests (cache.rs):**
- `iter_admitted()` returns exactly the CIDs in window + probation + protected

**Integration tests:**
- Full cycle: store content → mutation threshold → filter broadcast action → deserialize → `may_contain` confirms membership
- Policy filtering: EncryptedEphemeral content excluded from filter
- Staleness: expired peer filter results in "query anyway" behavior

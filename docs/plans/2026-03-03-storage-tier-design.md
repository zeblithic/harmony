# StorageTier: Content Cache with Zenoh Integration

Sans-I/O wrapper that integrates `ContentStore` with Zenoh queryable and
subscriber patterns. Lives in `harmony-content`, depends on
`harmony-zenoh::namespace` for canonical key expressions.

Part of the Node Trinity design (see `2026-03-03-node-trinity-design.md`,
sections 2 and 4). Bead: `harmony-k5ed`.

---

## Core Struct

```rust
pub struct StorageTier<B: BlobStore> {
    cache: ContentStore<B>,
    budget: StorageBudget,
    metrics: StorageMetrics,
}
```

## Sans-I/O Interface

### Events (input)

```rust
pub enum StorageTierEvent {
    /// Content query on harmony/content/{prefix}/**
    ContentQuery { query_id: u64, cid: ContentId },
    /// Content transiting through router (harmony/content/transit/**)
    TransitContent { cid: ContentId, data: Vec<u8> },
    /// Explicit publish request (harmony/content/publish/*)
    PublishContent { cid: ContentId, data: Vec<u8> },
    /// Stats query on harmony/content/stats/{node_addr}
    StatsQuery { query_id: u64 },
}
```

### Actions (output)

```rust
pub enum StorageTierAction {
    /// Reply to content query with data
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Announce content availability on harmony/announce/{cid_hex}
    AnnounceContent { key_expr: String, payload: Vec<u8> },
    /// Reply with cache metrics
    SendStatsReply { query_id: u64, payload: Vec<u8> },
    /// Register shard queryables (startup)
    DeclareQueryables { key_exprs: Vec<String> },
    /// Register subscriptions (startup)
    DeclareSubscribers { key_exprs: Vec<String> },
}
```

## Configuration

```rust
pub struct StorageBudget {
    /// Maximum items in the W-TinyLFU cache.
    pub cache_capacity: usize,
    /// Maximum bytes reserved for pinned content.
    pub max_pinned_bytes: u64,
}
```

## Metrics

```rust
pub struct StorageMetrics {
    pub queries_served: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub transit_admitted: u64,
    pub transit_rejected: u64,
    pub publishes_stored: u64,
}
```

Simple counters. Serialized as the payload for `StatsQuery` replies.

## Behavior

### Startup

`StorageTier::new(budget)` returns startup actions:
- `DeclareQueryables`: 16 shard patterns from `namespace::content::all_shard_patterns()` + stats key
- `DeclareSubscribers`: `TRANSIT_SUB` + `PUBLISH_SUB`

### Event handling

| Event | Behavior | Action |
|-------|----------|--------|
| `ContentQuery` | Look up CID in cache via `get_and_record` | `SendReply` on hit, nothing on miss |
| `TransitContent` | W-TinyLFU admission via `ContentStore::insert` | `AnnounceContent` if admitted |
| `PublishContent` | Always store via `ContentStore::store` | `AnnounceContent` |
| `StatsQuery` | Serialize metrics | `SendStatsReply` |

### Transit admission (opportunistic caching)

Transit content passes through the W-TinyLFU admission gate. The cache
decides whether to keep it based on frequency estimation. This is the
"content naturally replicates toward the edges" mechanism.

### Publish (explicit storage)

Always admitted. The caller explicitly wants this content stored. Bypasses
frequency comparison.

## Migration

The existing `zenoh_bridge.rs` functions (`content_queryable_key_exprs`,
`cid_to_key_expr`, `handle_content_query`, `handle_content_stored`) are
absorbed by `StorageTier` methods using `harmony_zenoh::namespace::content::*`.
`zenoh_bridge.rs` is replaced, not extended.

## Scope

**In scope:**
- `StorageTier<B>` struct, event/action types
- `StorageBudget`, `StorageMetrics`
- Startup actions (declare queryables + subscribers)
- All four event handlers
- Tests for each handler + metrics
- Replace `zenoh_bridge.rs` with StorageTier

**Deferred:**
- Zenoh session integration (harmony-0jk6)
- Persistent backends (SQLite, RocksDB)
- Pin budget byte tracking

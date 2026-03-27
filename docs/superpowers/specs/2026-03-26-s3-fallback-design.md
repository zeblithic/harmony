# S3 Fallback Resolver for CAS Content

## Overview

When a content query misses both the in-memory cache and local disk, and the CID is durable (flags 00/10), fall back to S3 via the existing `harmony-s3` library. Fetched books are injected through the existing StorageTier pipeline so caching, disk persistence, and mesh announcement happen automatically.

**Goal:** Durable CAS content that no mesh peer has can be retrieved from S3, hydrating the local node and nearby peers.

**Scope:** Read-side S3 fallback only. The write-side archivist (uploading to S3) already exists.

## Trigger

In `StorageTier::handle_content_query`, after the cache miss + disk miss path:

```
Cache miss → disk_index miss → if s3_enabled && is_durable_class(cid) → emit S3Lookup { cid, query_id }
```

`s3_enabled` is a new bool on StorageTier, set when archivist config is present.

## Data Flow

```
StorageTier emits S3Lookup { cid, query_id }
  → NodeRuntime converts to RuntimeAction::S3Lookup { cid, query_id }
  → Event loop spawns async s3_library.get_book(&cid.to_bytes())
  → On success: send RuntimeEvent::S3ReadComplete { cid, query_id, data }
  → NodeRuntime routes to StorageTier
  → StorageTier: verify CID hash, cache book, emit SendReply
    + PersistToDisk (if disk_enabled) + AnnounceContent (if policy allows)
  → On failure/not-found: send RuntimeEvent::S3ReadFailed { cid, query_id }
  → StorageTier: emit empty SendReply
```

## Design Decisions

### Reuse existing pipeline

S3ReadComplete feeds the book through the same verify → cache → persist → announce path as TransitContent. This gives us:
- W-TinyLFU caching (book stays hot in memory)
- Disk persistence via PersistToDisk (survives restarts, avoids re-fetching from S3)
- Mesh announcement via AnnounceContent (nearby peers learn this node has the book)

All for free from existing StorageTier logic.

### Feature-gated

All S3 fallback code behind `#[cfg(feature = "archivist")]`. When compiled without the feature, `S3Lookup` is never emitted and the event/action variants don't exist. This matches the existing archivist feature gate pattern.

### Durable-only

Only durable CIDs (content flags 00/10) fall back to S3. Ephemeral content (01/11) is never archived and won't exist in S3. The `is_durable_class()` check already exists in StorageTier.

### No timeout in StorageTier

The bead mentions "configurable timeout before fallback (default 5s)." StorageTier is sans-I/O and doesn't have timers. The S3 lookup is emitted immediately on cache+disk miss — the "timeout" is the natural microsecond latency of checking cache and disk. If mesh peers had the content, it would already be cached from a prior transit. No artificial delay is needed.

## New Types

### StorageTierAction

```rust
/// Fall back to S3 for a durable CID not found in cache or on disk.
#[cfg(feature = "archivist")]
S3Lookup { cid: ContentId, query_id: u64 },
```

### StorageTierEvent

```rust
/// S3 read completed — book fetched from remote storage.
#[cfg(feature = "archivist")]
S3ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },

/// S3 read failed — book not found or network error.
#[cfg(feature = "archivist")]
S3ReadFailed { cid: ContentId, query_id: u64 },
```

### RuntimeAction

```rust
/// Fetch a CAS book from S3 (spawned as async task by event loop).
#[cfg(feature = "archivist")]
S3Lookup { cid: ContentId, query_id: u64 },
```

### RuntimeEvent

```rust
/// S3 fetch completed.
#[cfg(feature = "archivist")]
S3ReadComplete { cid: ContentId, query_id: u64, data: Vec<u8> },

/// S3 fetch failed.
#[cfg(feature = "archivist")]
S3ReadFailed { cid: ContentId, query_id: u64 },
```

## StorageTier Changes

### New fields

- `s3_enabled: bool` — default false, set via `enable_s3()` method

### handle_content_query modification

After the existing cache miss + disk miss path (which currently returns empty vec), add:

```rust
// S3 fallback for durable content when archivist is configured.
#[cfg(feature = "archivist")]
if self.s3_enabled && Self::is_durable_class(cid) {
    return vec![StorageTierAction::S3Lookup { cid: *cid, query_id }];
}
vec![]
```

### S3ReadComplete handler

```rust
StorageTierEvent::S3ReadComplete { cid, query_id, data } => {
    // Verify integrity.
    if !Self::verify_cid(&cid, &data) {
        return vec![StorageTierAction::SendReply { query_id, payload: vec![] }];
    }
    // Cache the book (warm frequency to survive admission challenge).
    self.cache.warm_frequency(&cid, 5);
    self.cache.store(cid, data.clone());
    self.mutations_since_broadcast += 1;

    let mut actions = vec![StorageTierAction::SendReply { query_id, payload: data.clone() }];

    // Persist to disk if enabled (same as transit path).
    if self.disk_enabled && Self::is_durable_class(&cid) && !self.disk_index.contains(&cid) {
        self.disk_index.insert(cid);
        actions.push(StorageTierAction::PersistToDisk { cid, data: data.clone() });
    }

    // Announce to mesh if policy allows.
    if self.should_announce(&cid) {
        // ... same announce logic as handle_transit
    }

    actions
}
```

### S3ReadFailed handler

```rust
StorageTierEvent::S3ReadFailed { query_id, .. } => {
    vec![StorageTierAction::SendReply { query_id, payload: vec![] }]
}
```

## NodeRuntime Changes

- Wire `StorageTierAction::S3Lookup` → `RuntimeAction::S3Lookup`
- Wire `RuntimeEvent::S3ReadComplete` → `StorageTierEvent::S3ReadComplete`
- Wire `RuntimeEvent::S3ReadFailed` → `StorageTierEvent::S3ReadFailed`

All behind `#[cfg(feature = "archivist")]`.

## Event Loop Changes

In `dispatch_action`, handle `RuntimeAction::S3Lookup`:

```rust
RuntimeAction::S3Lookup { cid, query_id } => {
    if let Some(ref s3) = s3_library {
        let s3 = s3.clone();
        let tx = s3_tx.clone();
        tokio::spawn(async move {
            match s3.get_book(&cid.to_bytes()).await {
                Ok(Some(data)) => { tx.send(S3Result::Complete { cid, query_id, data }).await; }
                Ok(None) => { tx.send(S3Result::Failed { cid, query_id }).await; }
                Err(e) => {
                    tracing::warn!(cid = %hex::encode(&cid.to_bytes()[..8]), "S3 fetch failed: {e}");
                    tx.send(S3Result::Failed { cid, query_id }).await;
                }
            }
        });
    }
}
```

Add an `mpsc` channel for S3 completions (same pattern as disk I/O channel) and a select! arm to receive results.

The `S3Library` instance already exists in the event loop (created at startup when archivist config is present). It just needs to be shared with `dispatch_action`.

## Testing

### StorageTier unit tests

1. **S3Lookup emitted on durable miss** — s3_enabled + cache miss + disk miss + durable CID → S3Lookup
2. **S3Lookup NOT emitted for ephemeral** — ephemeral CID → empty (no S3Lookup)
3. **S3Lookup NOT emitted when s3 disabled** — s3_enabled=false → empty
4. **S3Lookup NOT emitted on cache hit** — cached CID → SendReply (no S3Lookup)
5. **S3Lookup NOT emitted on disk hit** — disk-indexed CID → DiskLookup (not S3Lookup)
6. **S3ReadComplete caches and replies** — valid data → cache + SendReply + PersistToDisk
7. **S3ReadComplete with bad data** — corrupted → empty SendReply
8. **S3ReadFailed replies empty** — failure → empty SendReply

## Files Modified

| File | Change |
|------|--------|
| `crates/harmony-content/src/storage_tier.rs` | Add s3_enabled, S3Lookup emission, S3ReadComplete/Failed handlers |
| `crates/harmony-node/src/runtime.rs` | Add RuntimeAction/Event variants, wire S3 actions |
| `crates/harmony-node/src/event_loop.rs` | Add S3 completion channel, spawn async get_book, select! arm |

## Dependencies

No new crates. `harmony-s3` is already a workspace dependency, gated behind the `archivist` feature.

## What This Does NOT Include

- Write-side S3 upload — already implemented in `harmony-s3::archivist`
- Configurable timeout before fallback — sans-I/O StorageTier can't do timers; immediate fallback on miss
- S3 retry logic — `harmony-s3::get_book` handles retries internally
- Rate limiting S3 requests — deferred; natural rate limiting from cache hits

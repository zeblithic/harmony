# P2P Memo Discovery via Bloom Filters

**Bead:** harmony-boa
**Date:** 2026-03-22
**Status:** Draft

## Problem

The memo attestation layer (harmony-m5y) stores signed input→output mappings,
but there's no way for peers to discover which memos their neighbors have
without querying each one individually. A node looking for memos about a
computation input has to blindly query every peer.

## Solution

Wire the existing memo Bloom filter infrastructure into the NodeRuntime.
On each filter timer tick, rebuild a Bloom filter from `MemoStore.input_cids()`
and broadcast it to `harmony/filters/memo/{node_addr}`. Peers receive these
filters and use them to decide which neighbors to query for memos about a
given input CID — skipping peers that definitely don't have relevant memos.

This replicates the exact pattern used by content filters in StorageTier,
adapted for the memo use case.

## Design Decisions

### Timer-only broadcast (no mutation counter)

Memos arrive much less frequently than content cache admissions. The mutation
threshold that content filters use (100 mutations before broadcast) would
never fire for memos. Timer-only broadcast on the existing filter interval
(~7.5 seconds at 30 ticks × 250ms) is sufficient and simpler.

### Inline in NodeRuntime (no separate MemoTier)

Content filters have a full `StorageTier` because it manages cache admission,
eviction, Bloom AND Cuckoo filters, and content policy. The memo filter is
just "rebuild Bloom from input_cids(), broadcast." That's ~15 lines of logic —
a separate state machine struct would be over-engineering.

### Extend PeerFilterTable (not a separate table)

PeerFilterTable already tracks per-peer content and flatpack filters with
staleness tracking and eviction. Adding a third filter type is one field +
two methods. A separate MemoFilterTable would duplicate infrastructure.

### Shared FilterBroadcastConfig

The memo filter reuses the same `expected_items` and `fp_rate` from the
existing `FilterBroadcastConfig`. The memo filter will typically be much
smaller, but sharing the config avoids adding new configuration surface.

## Architecture

### Broadcasting (outbound)

On each filter timer tick (same tick that triggers content filter rebuilds),
the runtime rebuilds the memo filter:

1. Create `BloomFilter::new(expected_items, fp_rate)`
2. Iterate `memo_store.input_cids()`, insert each into the filter
3. Emit `RuntimeAction::Publish` to `harmony/filters/memo/{node_addr}`

No mutation counter — timer-only.

### Receiving (inbound)

In `route_subscription()`, add a branch for `harmony/filters/memo/`:

1. Strip the prefix to extract `peer_addr`
2. Skip if `peer_addr == self.node_addr` (own broadcast)
3. Parse `BloomFilter::from_bytes(&payload)`
4. Call `peer_filters.upsert_memo(peer_addr, filter, tick_count)`

### PeerFilterTable extension

Add to the per-peer entry:

```rust
pub memo_filter: Option<BloomFilter>,
pub memo_received_tick: u64,
```

New methods:

- `upsert_memo(peer_addr, filter, tick)` — store/update the memo filter
- `should_query_memo(peer_addr, input_cid, current_tick) -> bool` — same
  logic as content: unknown peer → true, stale filter → true, fresh filter
  → `may_contain(input_cid)`

### Startup subscription

Add `filters::MEMO_SUB` (`harmony/filters/memo/**`) to the runtime's startup
subscription actions, alongside the existing content and flatpack subscriptions.

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/src/runtime.rs` | Memo filter rebuild on timer tick, publish. Route memo filter subscriptions. Add MEMO_SUB to startup. Extend PeerFilterTable with memo fields + methods. |

## What is NOT in Scope

- No fetch orchestration (bead `harmony-5vf`)
- No architecture-aware sharding (bead `harmony-s1i`)
- No new configuration surface (reuses FilterBroadcastConfig)
- No new Zenoh namespace changes (already defined)
- No new crate

## Testing

- `memo_filter_broadcast_on_timer_tick` — insert memos, tick, verify Publish action
- `memo_filter_contains_inserted_inputs` — verify may_contain for stored CIDs
- `peer_memo_filter_upsert_and_query` — receive filter, verify should_query_memo
- `peer_memo_filter_staleness` — stale filter → query anyway
- `memo_filter_subscription_routed` — subscription with memo prefix → stored

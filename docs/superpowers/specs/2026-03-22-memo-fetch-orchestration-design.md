# Memo Fetch Orchestration

**Date:** 2026-03-22
**Status:** Draft
**Scope:** `harmony-node` (runtime.rs)
**Bead:** harmony-5vf

## Problem

The memo attestation layer stores signed input→output mappings locally, and
peers broadcast Bloom filters advertising which input CIDs they have memos
for (harmony-boa). But there's no way to actually fetch memos — no query
handler serves them, and no fetch pipeline retrieves them from peers.

## Solution

Two complementary pieces in `NodeRuntime`:

1. **Memo queryable** — declare `harmony/memo/**`, handle incoming queries by
   looking up the local MemoStore and replying with serialized memos.
2. **Memo fetch orchestration** — on `MemoFetchRequest`, check local store
   first, then issue a Zenoh query. Replies are verified and inserted into
   the local MemoStore silently (no explicit completion callback).

## Design Decisions

### Zenoh queries for both peer and fallback

The Bloom filter tells us which peers probably have relevant memos, but the
actual query goes over Zenoh (`session.get()`), not a custom tunnel message.
This means "query peers" and "Zenoh fallback" use the same transport — the
Bloom filter is an optimization hint, not a routing decision. Peers who
declared `harmony/memo/**` as a queryable respond if they have data.

### Silent insertion (no completion callback)

Fetched memos are verified and inserted into MemoStore. The caller (e.g.,
workflow engine) re-checks the store on its next tick. This is the simplest
approach that works — the sans-I/O pattern already has callers polling on
tick boundaries. Explicit completion signals can be added later if needed.

### In-flight dedup with tick-based timeout

A `HashMap<ContentId, u64>` tracks pending fetches (input CID → started
tick). This prevents re-querying for the same input while a fetch is
in-flight. Entries expire after `MEMO_FETCH_TIMEOUT_TICKS` (20 ticks ≈ 5s).
After expiry, the same input can be fetched again.

### Serve all local memos (no trust filtering)

The queryable handler returns all memos for the requested input, regardless
of signer. Memos are self-certifying — each carries a `Credential` with the
issuer's ML-DSA signature. The receiver decides what to trust.

### Response size limits

Responses are capped at `MAX_MEMO_RESPONSE_COUNT` (256) memos and
`MAX_MEMO_RESPONSE_BYTES` (1 MiB). The serve handler truncates at the cap.
The fetch handler rejects responses exceeding the byte limit before parsing.
This prevents memory exhaustion from malicious or buggy peers.

### Batch response on input-CID key

The queryable is declared on `harmony/memo/**` which covers the full
structured namespace (`harmony/memo/{input}/{output}/sign/{signer}`), but
the query handler treats the key expression as a batch lookup by input CID
only. The output/signer structure in the namespace is used for publishing
individual attestations (future work) — query responses return all memos
for the input as a packed blob.

### All-in-runtime (no separate crate or module)

The fetch logic is ~50 lines and uses MemoStore, PeerFilterTable, and
QueryableRouter — all owned by NodeRuntime. Extracting to a separate struct
would create more interface boilerplate than actual logic. Follows the
pattern established by content queries, page queries, and compute queries.

## Architecture

### Memo Queryable (Serving)

At startup, declare a queryable on `harmony/memo/**`. Store the queryable ID
as `memo_queryable_id`. In `route_query()`, dispatch to `handle_memo_query()`
when the queryable ID matches.

Handler:
1. Strip `harmony/memo/` prefix and trailing `/**` from the query key expression
2. Decode `input_hex` to `ContentId` via `hex::decode` + `ContentId::from_bytes`.
   If the key expression is malformed (bad hex, wrong length), return no reply.
3. Look up `memo_store.get_by_input(&input_cid)`
4. If empty → no reply
5. If found → serialize each memo via `harmony_memo::serialize()`, cap at
   `MAX_MEMO_RESPONSE_COUNT` (256), build response, reply

**Response format:** `[u16 LE count][memo_1_len: u32 LE][memo_1_bytes]...`

Length-prefixed because memos are variable-size. The receiver splits on
length boundaries, deserializes each with `harmony_memo::deserialize()`,
and verifies independently.

### Memo Fetch (Requesting)

When `RuntimeEvent::MemoFetchRequest { input }` arrives:

1. **Local check** — `memo_store.get_by_input(&input)` non-empty → done
2. **Dedup check** — `pending_memo_fetches.contains_key(&input)` → skip
3. **Issue query** — emit `RuntimeAction::QueryMemo { key_expr }` where
   `key_expr = harmony/memo/{input_hex}/**`
4. **Track** — insert `(input, self.tick_count)` into `pending_memo_fetches`

When `RuntimeEvent::MemoFetchResponse { key_expr, payload, unix_now }` arrives:

1. Parse `input_cid` from `key_expr` by stripping `harmony/memo/` prefix
   and trailing `/**`. If malformed, discard the response (debug log).
2. Guard: reject if `payload.len() > MAX_MEMO_RESPONSE_BYTES` (1 MiB).
3. Decode response: read u16 count, then for each memo read u32 length + bytes.
4. Deserialize each via `harmony_memo::deserialize()`. Skip malformed entries.
5. Verify each via `harmony_memo::verify::verify_memo(memo, unix_now, &resolver)`.
   The resolver uses `pubkey_cache` to look up the issuer's ML-DSA public key.
   Memos with unknown issuers (cache miss) are silently discarded.
6. Insert valid memos into `memo_store`.

Note: `pending_memo_fetches` is NOT removed on response arrival. Multiple
peers may respond to the same Zenoh query, and each reply becomes a separate
`MemoFetchResponse` event. All responses are processed regardless of dedup
state. The dedup map only prevents re-issuing `QueryMemo` for the same input.

On each tick: remove expired entries from `pending_memo_fetches` where
`current_tick - started_tick > MEMO_FETCH_TIMEOUT_TICKS`.

## Changes

### New RuntimeEvent variants

```rust
/// Memo fetch: caller requests memos for an input CID.
MemoFetchRequest { input: ContentId },

/// Memo fetch: Zenoh query reply arrived with memo data.
/// `unix_now` is Unix epoch seconds for credential verification.
MemoFetchResponse { key_expr: String, payload: Vec<u8>, unix_now: u64 },
```

### New RuntimeAction variant

```rust
/// Memo fetch: emit a Zenoh session.get() query for memos.
QueryMemo { key_expr: String },
```

### New fields on NodeRuntime

```rust
/// Queryable ID for the memo query handler.
memo_queryable_id: QueryableId,

/// In-flight memo fetches: input CID → tick when fetch was started.
pending_memo_fetches: HashMap<ContentId, u64>,
```

### Constants

```rust
/// Ticks before an in-flight memo fetch expires (20 × 250ms = 5s).
const MEMO_FETCH_TIMEOUT_TICKS: u64 = 20;

/// Maximum number of memos in a single query response.
const MAX_MEMO_RESPONSE_COUNT: usize = 256;

/// Maximum total response payload size (1 MiB).
const MAX_MEMO_RESPONSE_BYTES: usize = 1_048_576;
```

## File Changes

| File | Change |
|------|--------|
| `crates/harmony-node/src/runtime.rs` | Declare memo queryable at startup. Add `handle_memo_query()` to serve incoming queries. Add `handle_memo_fetch_request()` and `handle_memo_fetch_response()` for outbound fetch. Add `pending_memo_fetches` field and tick-based cleanup. New event/action variants. |

## Testing

- `memo_queryable_serves_local_memos` — insert memos, simulate query, verify reply format
- `memo_queryable_empty_no_reply` — query for unknown input, verify no reply
- `memo_fetch_request_local_short_circuit` — insert memos, send fetch request, verify no QueryMemo emitted
- `memo_fetch_request_emits_query` — empty store, send fetch request, verify QueryMemo action
- `memo_fetch_request_dedup` — send same fetch request twice, verify only one QueryMemo
- `memo_fetch_response_inserts_verified_memos` — simulate response, verify memos in store
- `memo_fetch_response_rejects_invalid_memos` — corrupt memo bytes, verify not inserted
- `memo_fetch_timeout_clears_pending` — start fetch, advance ticks past timeout, verify re-fetch possible
- `memo_fetch_multiple_responses_all_inserted` — two responses for same input, verify all valid memos inserted
- `memo_fetch_response_oversized_rejected` — payload exceeds MAX_MEMO_RESPONSE_BYTES, verify discarded
- `memo_queryable_caps_response_count` — store >256 memos, verify response truncated to MAX_MEMO_RESPONSE_COUNT

## What is NOT in Scope

- No trust filtering on serve (memos are self-certifying)
- No explicit completion callbacks (callers poll MemoStore)
- No Bloom filter changes (harmony-boa handles that)
- No new crates or modules

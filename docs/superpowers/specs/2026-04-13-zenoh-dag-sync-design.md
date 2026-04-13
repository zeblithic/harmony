# harmony-db Zenoh DAG Sync Design (ZEB-108)

## Overview

Zenoh-based DAG synchronization for harmony-db. When a peer commits a new root CID, other peers discover the change via pub/sub, compare against local state, and pull only the missing blocks via Zenoh queryable. Leverages Prolly Tree structural sharing — most of the tree is already present locally, so sync cost is O(log n) blocks per mutation.

## Design Decisions

- **Single-writer semantics:** Each database has one authoritative writer. Other peers maintain read replicas. No conflict resolution needed.
- **Pull-based block exchange:** Subscribers walk the tree top-down via `rebuild_from`, fetching missing CIDs via Zenoh queryable. No push bundles or diff manifests.
- **Generic block server:** One queryable serves any CID (commits, tree nodes, blobs) — CAS is type-agnostic.
- **Whole-database sync:** Root CID announcements cover the entire database (all tables). No per-table granularity.
- **No state machine crate:** The `BookStore` trait is the sans-I/O boundary. Sync coordination is glue code in the application layer.
- **Key expression namespace:** Follows existing `harmony-zenoh` conventions with hex-encoded owner addresses.

## Key Expressions

### Root CID Announcements (pub/sub)

```
harmony/db/{owner_addr_hex}/{db_name}/root
```

- Published by the writer after each `commit()`
- Payload: 32-byte raw `ContentId` (the commit manifest CID)
- Subscribers compare against local `db.head()` and trigger sync if different

### Block Fetch (queryable)

```
harmony/db/{owner_addr_hex}/{db_name}/block/{cid_hex}
```

- Registered by the writer to serve local CAS blocks
- Query: key expression with the desired CID
- Reply: raw bytes of the block
- Fetcher verifies CID integrity (BLAKE3 hash of received bytes must match requested CID)

### Subscription Patterns

- Subscribe to a specific peer's database: `harmony/db/{peer_addr_hex}/{db_name}/root`
- Subscribe to all peers for a database name: `harmony/db/*/{db_name}/root` (Zenoh wildcard)

## Components

### 1. Key Expression Helpers (harmony-zenoh)

Add builder functions to `crates/harmony-zenoh/src/keyspace.rs` following existing patterns:

- `db_root_key(owner_addr_hex, db_name) -> OwnedKeyExpr` — publish key for root announcements
- `db_root_sub(owner_addr_hex, db_name) -> OwnedKeyExpr` — subscribe pattern
- `db_block_key(owner_addr_hex, db_name, cid_hex) -> OwnedKeyExpr` — key for a specific block
- `db_block_queryable(owner_addr_hex, db_name) -> OwnedKeyExpr` — wildcard pattern for registering the block server (e.g., `harmony/db/{owner}/mail/block/*`)

All functions validate inputs with `reject_slashes()` and return `Result<OwnedKeyExpr, ZenohError>`, consistent with existing keyspace helpers.

### 2. ZenohBookStore (application layer)

Implements `harmony_content::BookStore` to bridge local CAS with Zenoh network fetches.

#### Construction

Created per-sync-session with:
- `data_dir: &Path` — local CAS directory
- Remote peer address and database name (for constructing query key expressions)
- A Zenoh query callback or handle

#### Method behavior

- **`get(cid)`** — Check local CAS first (`commits/{hex}.bin` and `blobs/{hex}.bin`). On local miss, send Zenoh query to `harmony/db/{owner}/{db_name}/block/{cid_hex}`. Verify BLAKE3 integrity of received bytes. Cache locally on success.
- **`contains(cid)`** — Local CAS only. No network round-trip.
- **`store(cid, data)`** — Write to local CAS.
- **`insert_with_flags(data, flags)`** — Compute CID, write to local CAS.

#### Lifetime

Short-lived: created when a root CID announcement arrives, used for the duration of `rebuild_from`, then dropped.

### 3. Block Server (application layer)

Registers a Zenoh queryable on `harmony/db/{my_addr_hex}/{db_name}/block/*`.

On query:
1. Extract `{cid_hex}` from key expression
2. Look up in local CAS (both `commits/` and `blobs/` directories)
3. Reply with raw bytes if found, empty/error if not

Security: content-addressed — fetcher verifies CID matches data. Malicious servers can withhold but cannot forge.

One queryable per database the peer owns. Read-replicas do not serve blocks (not authoritative).

### 4. Root Announcement and Sync Trigger (application layer)

#### Writer side

After `db.commit()` returns `root_cid`:
1. Publish `root_cid` (32 bytes) to `harmony/db/{my_addr_hex}/{db_name}/root`

#### Reader side

Subscribe to `harmony/db/{peer_addr_hex}/{db_name}/root`. On message:
1. Decode 32-byte `ContentId` from payload
2. Compare against `db.head()`
3. If identical: no-op
4. If different: create `ZenohBookStore` for that peer, call `db.rebuild_from(new_root, Some(&store))`
5. `rebuild_from` walks the tree, fetching missing blocks via `ZenohBookStore`, updates local state

#### Subscription management

Which peers and databases to follow is an application-level concern (contacts, social graph, configuration). Outside the scope of this sync layer — we provide the subscribe/publish primitives.

## Sync Flow

```
Writer (Peer A)                          Reader (Peer B)
─────────────                            ────────────────
1. db.insert(...)
2. db.commit() → root_cid
3. publish root_cid ──pub/sub──────────> receive root_cid
                                         4. compare with db.head()
                                         5. if different:
                                            create ZenohBookStore(peer_a)
                                            db.rebuild_from(root_cid, store)
                                              │
                                              ├─ get(root_cid) ──query──> serve from CAS
                                              ├─ parse manifest
                                              ├─ for each table root:
                                              │   ├─ get(table_root) ──query──> serve
                                              │   ├─ walk branches:
                                              │   │   skip if CID matches local ← structural sharing
                                              │   │   get(child_cid) ──query──> serve
                                              │   └─ walk leaves:
                                              │       get(value_cid) ──query──> serve blob
                                              └─ update local head
```

## Files Changed

### harmony-zenoh

- `crates/harmony-zenoh/src/keyspace.rs` — Add `db_root_key`, `db_root_sub`, `db_block_key`, `db_block_queryable` builder functions with tests

### application layer (harmony-node/harmony-runtime)

- New module (e.g., `sync.rs` or `db_sync.rs`):
  - `ZenohBookStore` struct implementing `BookStore`
  - `serve_blocks(db, zenoh_session)` — register block server queryable
  - `publish_root(root_cid, zenoh_session)` — publish after commit
  - `handle_root_announcement(root_cid, peer, db)` — trigger `rebuild_from`

### harmony-db

No changes. Existing `commit()`, `rebuild_from()`, and `BookStore` interface are sufficient.

## Testing Strategy

### Unit tests (harmony-zenoh keyspace)

- Key expression builders produce correct patterns
- Wildcard subscription patterns match expected keys
- Input validation (reject slashes, empty strings)

### Unit tests (ZenohBookStore)

- `get()` returns local CAS data when present (no network call)
- `get()` falls through to network fetch on local miss (mock Zenoh query)
- `contains()` only checks local, never network
- `store()` writes to local CAS correctly
- CID integrity verification rejects corrupted data

### Integration test (end-to-end sync)

- Two `HarmonyDb` instances with separate data directories
- Writer commits entries, passes root CID directly to reader (simulated pub/sub)
- Reader creates `ZenohBookStore` backed by writer's local CAS (in-process, no actual Zenoh), calls `rebuild_from`
- Verify reader state matches writer (same entries, same head CID)
- Verify incremental sync: writer commits more changes, reader syncs again, structural sharing means only new blocks fetched

### Not in scope for automated tests

- Actual Zenoh pub/sub and queryable wire protocol (requires live Zenoh session — smoke/integration test territory)

## Future Extensions (not in this spec)

- **Multi-writer with CRDT merge:** Fork-aware DAG with application-level conflict resolution
- **Per-table sync:** Subscribe to specific tables within a database
- **Push-based diff bundles:** Publish changed block list alongside root CID for one-round-trip sync
- **Sans-I/O state machine:** Extract sync protocol into a dedicated crate if complexity grows (retry logic, pending fetch tracking, backpressure)
- **Merkle proofs for thin clients:** Serve verifiable range proofs without full tree sync

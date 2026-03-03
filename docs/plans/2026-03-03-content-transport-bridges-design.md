# Content Transport Bridges Design

**Goal:** Wire harmony-content's storage layer to harmony-reticulum and harmony-zenoh's transport layers, enabling content to move across the network. Adds a Queryable primitive to harmony-zenoh for request/reply semantics.

**Scope:** Reticulum resource transfer bridge, Zenoh queryable + content bridge. No Flatpack reverse index, no Bloom/Cuckoo filter broadcasting, no DAG-level transfer coordination.

---

## Module Structure

| Crate | Module | Purpose |
|-------|--------|---------|
| `harmony-zenoh` | `src/queryable.rs` | Queryable state machine — request/reply for content retrieval |
| `harmony-content` | `src/reticulum_bridge.rs` | Maps blobs to Reticulum ResourceSender/Receiver transfers |
| `harmony-content` | `src/zenoh_bridge.rs` | Wires ContentStore to Zenoh queryables + content announcements |

Bridges live in `harmony-content` because they depend on content types (`ContentId`, `BlobStore`). They accept reticulum/zenoh types via generics. New dependencies: `harmony-content` gains `harmony-reticulum` and `harmony-zenoh`.

---

## Zenoh Queryable State Machine

Request/reply primitive for harmony-zenoh. A peer declares "I can answer queries on this key expression" and receives queries to reply to.

### Types

```rust
pub type QueryableId = u64;
pub type QueryId = u64;

pub enum QueryableEvent {
    /// Remote peer sent a query matching our key expression
    QueryReceived { query_id: QueryId, key_expr: String, payload: Vec<u8> },
    /// Remote peer declared a queryable
    QueryableDeclared { key_expr: String },
    /// Remote peer undeclared a queryable
    QueryableUndeclared { key_expr: String },
}

pub enum QueryableAction {
    /// Tell the network we can answer queries on this key expression
    SendQueryableDeclare { key_expr: String },
    /// Remove our queryable declaration
    SendQueryableUndeclare { key_expr: String },
    /// Deliver a query to local handler for processing
    DeliverQuery { queryable_id: QueryableId, query_id: QueryId, key_expr: String, payload: Vec<u8> },
    /// Send a reply back to the querier
    SendReply { query_id: QueryId, payload: Vec<u8> },
    /// Forward to session layer
    Session(SessionAction),
}
```

### QueryableRouter

```rust
pub struct QueryableRouter {
    local_queryables: HashMap<QueryableId, String>,
    remote_queryables: SubscriptionTable,
    next_id: QueryableId,
}
```

**Operations:**
- `declare(key_expr, session)` — Register a queryable, returns `(QueryableId, Vec<QueryableAction>)`
- `undeclare(id, session)` — Remove a queryable declaration
- `query(key_expr, payload)` — Send a query to peers, returns `(QueryId, Vec<QueryableAction>)`
- `reply(query_id, payload)` — Reply to a received query
- `on_event(event)` — Process incoming queryable events

Sans-I/O: caller drives with events, executes returned actions over their transport.

---

## Reticulum Content Bridge

Maps content blobs 1:1 to Reticulum resource transfers. Each blob (up to ~1MB) becomes a single ResourceSender/ResourceReceiver pair.

### Sending

```rust
pub struct ContentSender {
    cid: ContentId,
    sender: ResourceSender,
}

pub enum ContentSendEvent {
    ResourceEvent(ResourceEvent),
}

pub enum ContentSendAction {
    ResourceAction(ResourceAction),
    Complete { cid: ContentId },
    Failed { cid: ContentId, reason: String },
}
```

**Flow:**
1. Look up blob by CID in BlobStore
2. Prepend CID (32 bytes) as header to blob data
3. Create `ResourceSender::new(rng, crypto, cid_prefixed_data, now)`
4. Drive events/actions as normal resource transfer

### Receiving

```rust
pub struct ContentReceiver {
    receiver: ResourceReceiver,
}

pub enum ContentRecvAction {
    ResourceAction(ResourceAction),
    ContentReady { cid: ContentId, data: Vec<u8> },
    ContentCorrupt { reason: String },
}
```

**Flow:**
1. Resource transfer completes → `Completed { data }`
2. Split first 32 bytes as CID, remainder as blob data
3. Recompute `ContentId::for_blob(&data)`, compare against received CID
4. Match → `ContentReady`, caller stores in BlobStore
5. Mismatch → `ContentCorrupt`

**Dual integrity:** ResourceHash verifies encrypted wire payload wasn't corrupted. CID verification confirms plaintext content is what we asked for — protects against a malicious sender encrypting valid-looking garbage.

**Not included:** Multi-blob DAG transfer. Caller walks the DAG with `dag::walk()` and sends each blob individually.

---

## Zenoh Content Bridge

Wires ContentStore to Zenoh queryables for content retrieval and pub/sub for content announcements.

### Types

```rust
pub struct ContentBridge {
    queryable_ids: Vec<QueryableId>,   // 16 prefix-sharded queryables
    publisher_id: Option<PublisherId>, // for content announcements
}

pub enum ContentBridgeEvent {
    /// Remote peer querying for content we might have
    ContentQuery { query_id: QueryId, cid: ContentId },
    /// Remote peer announced they have content
    ContentAnnounced { peer: Identity, cid: ContentId },
    /// Local store ingested new content
    ContentStored { cid: ContentId },
}

pub enum ContentBridgeAction {
    /// Reply to a query with blob data
    SendReply { query_id: QueryId, payload: Vec<u8> },
    /// Announce we have this content
    Publish { key_expr: String, payload: Vec<u8> },
    /// Notify caller that a peer has content we might want
    PeerHasContent { peer: Identity, cid: ContentId },
}
```

### Queryable Registration

16 prefix-sharded declarations on startup:

```
harmony/content/0**  harmony/content/1**  ...  harmony/content/f**
```

### Query Handling

1. Query arrives on `harmony/content/{cid_hex}`
2. Extract CID from key expression
3. Look up in ContentStore via `get_and_record(&cid)` (updates frequency tracking)
4. Found → `SendReply` with blob data
5. Miss → no reply (querier times out, tries next peer)

### Content Announcements

When new content stored locally:

```
Publish to: harmony/announce/{cid_hex}
Payload: minimal metadata (size, type, timestamp)
```

Peers subscribe to `harmony/announce/**` to discover content. Zenoh's interest-based routing means announcements only flow to subscribed peers.

### Not Included

- DAG-aware fetching (caller walks DAG, queries each CID)
- Caching policy (ContentStore/W-TinyLFU handles this)
- Filter broadcasting (deferred to future Flatpack bead)
- Retry logic on query misses (caller's responsibility)

---

## Testing (10 tests)

### Zenoh Queryable (4)

1. **declare_and_receive_query** — Declare queryable, feed matching QueryReceived, verify DeliverQuery emitted
2. **reply_sends_to_querier** — Receive query, call reply(), verify SendReply with correct query_id
3. **query_dispatches_to_matching_queryable** — Two queryables on different key exprs, query routes to correct one
4. **undeclare_stops_delivery** — Undeclare, subsequent queries produce no actions

### Reticulum Content Bridge (3)

5. **send_receive_round_trip** — ContentSender blob → drive to completion → ContentReceiver verifies CID and data
6. **receiver_detects_corrupt_content** — Tampered data after encryption → ContentCorrupt
7. **cid_header_verified** — Swap CID header to wrong value → receiver rejects (hash mismatch)

### Zenoh Content Bridge (3)

8. **query_returns_cached_content** — Store blob, simulate ContentQuery, verify SendReply with correct data
9. **query_miss_produces_no_reply** — Query for absent CID, no actions returned
10. **content_stored_triggers_announcement** — ContentStored event → Publish on `harmony/announce/{cid_hex}`

All tests use MemoryBlobStore and mock crypto/transport. No async, no sockets.

---

## Non-Goals (Future Work)

- Flatpack reverse lookup index
- Bloom/Cuckoo filter content discovery broadcasting
- DAG-level multi-blob transfer coordination
- Retry/timeout logic for query misses
- Sidecar deployment configuration
- Direct QUIC bulk swarming

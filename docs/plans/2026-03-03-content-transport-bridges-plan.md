# Content Transport Bridges Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Wire harmony-content storage to harmony-reticulum and harmony-zenoh transports via sans-I/O bridge state machines, and add Queryable request/reply semantics to harmony-zenoh.

**Architecture:** Three new modules: a QueryableRouter in harmony-zenoh (request/reply state machine), a Reticulum content bridge in harmony-content (maps blobs to resource transfers), and a Zenoh content bridge in harmony-content (wires ContentStore to queryables + content announcements). All sans-I/O. New dependencies: harmony-content depends on harmony-reticulum and harmony-zenoh.

**Tech Stack:** Rust, harmony-content, harmony-reticulum, harmony-zenoh, hex

---

### Task 1: Scaffold module stubs and wire dependencies

**Files:**
- Modify: `Cargo.toml` (workspace root)
- Modify: `crates/harmony-content/Cargo.toml`
- Modify: `crates/harmony-content/src/lib.rs`

**Step 1: Add harmony-reticulum to workspace dependencies**

In the root `Cargo.toml`, add to the `[workspace.dependencies]` section (after the existing `harmony-zenoh` entry):

```toml
harmony-reticulum = { path = "crates/harmony-reticulum" }
```

**Step 2: Add new dependencies to harmony-content**

In `crates/harmony-content/Cargo.toml`, add to `[dependencies]`:

```toml
harmony-identity = { workspace = true }
harmony-reticulum = { workspace = true }
harmony-zenoh = { workspace = true }
hex = { workspace = true }
```

`harmony-identity` is needed for the `Identity` type in the Zenoh bridge. `hex` is needed for CID-to-key-expression encoding.

**Step 3: Add module declarations to harmony-content**

In `crates/harmony-content/src/lib.rs`, add these two lines after the existing module list:

```rust
pub mod reticulum_bridge;
pub mod zenoh_bridge;
```

**Step 4: Create empty module files**

Create `crates/harmony-content/src/reticulum_bridge.rs`:

```rust
//! Reticulum content bridge — maps content blobs to resource transfers.
```

Create `crates/harmony-content/src/zenoh_bridge.rs`:

```rust
//! Zenoh content bridge — wires ContentStore to queryables and content announcements.
```

**Step 5: Verify it compiles**

Run: `cargo build -p harmony-content`
Expected: Compiles with no errors

**Step 6: Commit**

```
feat(content): scaffold reticulum_bridge and zenoh_bridge module stubs
```

---

### Task 2: Implement QueryableRouter in harmony-zenoh

**Files:**
- Modify: `crates/harmony-zenoh/src/error.rs`
- Create: `crates/harmony-zenoh/src/queryable.rs`
- Modify: `crates/harmony-zenoh/src/lib.rs`

**Step 1: Add error variants**

In `crates/harmony-zenoh/src/error.rs`, add two new variants to the `ZenohError` enum (before the closing `}`):

```rust
    #[error("unknown queryable ID: {0}")]
    UnknownQueryableId(u64),

    #[error("unknown query ID: {0}")]
    UnknownQueryId(u64),
```

**Step 2: Write the failing tests**

Create `crates/harmony-zenoh/src/queryable.rs` with types, the `QueryableRouter` struct, constructor, `Default` impl, and tests — but leave the method bodies as `todo!()`.

```rust
//! Sans-I/O queryable router for request/reply content retrieval.
//!
//! [`QueryableRouter`] manages queryable declarations (key expressions this
//! node can answer queries for), incoming query dispatch, and reply routing.
//! Follows the same sans-I/O pattern as [`PubSubRouter`].

use std::collections::HashMap;

use zenoh_keyexpr::key_expr::{keyexpr, OwnedKeyExpr};

use crate::error::ZenohError;
use crate::subscription::{SubscriptionId, SubscriptionTable};

/// Opaque queryable identifier, returned on declare.
pub type QueryableId = u64;

/// Opaque query identifier for correlating queries with replies.
pub type QueryId = u64;

/// Inbound events the caller feeds into the router.
#[derive(Debug, Clone)]
pub enum QueryableEvent {
    /// Remote peer sent a query matching one of our queryables.
    QueryReceived {
        query_id: QueryId,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Remote peer declared a queryable (they can answer queries).
    QueryableDeclared { key_expr: String },
    /// Remote peer undeclared a queryable.
    QueryableUndeclared { key_expr: String },
}

/// Outbound actions the router returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryableAction {
    /// Tell the network we can answer queries on this key expression.
    SendQueryableDeclare { key_expr: String },
    /// Remove our queryable declaration from the network.
    SendQueryableUndeclare { key_expr: String },
    /// Deliver a query to the local handler for the matching queryable.
    DeliverQuery {
        queryable_id: QueryableId,
        query_id: QueryId,
        key_expr: String,
        payload: Vec<u8>,
    },
    /// Send a reply back to the querier.
    SendReply { query_id: QueryId, payload: Vec<u8> },
}

/// A sans-I/O router for queryable registration, query dispatch, and reply routing.
///
/// Decoupled from [`Session`] — the caller is responsible for wiring
/// `SendQueryableDeclare`/`SendQueryableUndeclare` actions to the session's
/// resource declaration mechanism.
pub struct QueryableRouter {
    /// Local queryables: QueryableId → canonical key expression.
    local_queryables: HashMap<QueryableId, OwnedKeyExpr>,
    /// Matching table for incoming queries against local queryables.
    local_table: SubscriptionTable,
    /// Maps QueryableId → SubscriptionId in the local matching table.
    local_sub_ids: HashMap<QueryableId, SubscriptionId>,
    next_queryable_id: QueryableId,
}

impl QueryableRouter {
    /// Create a new empty router.
    pub fn new() -> Self {
        Self {
            local_queryables: HashMap::new(),
            local_table: SubscriptionTable::new(),
            local_sub_ids: HashMap::new(),
            next_queryable_id: 1,
        }
    }

    /// Declare a local queryable on the given key expression.
    pub fn declare(
        &mut self,
        key_expr: &str,
    ) -> Result<(QueryableId, Vec<QueryableAction>), ZenohError> {
        todo!("declare")
    }

    /// Undeclare a local queryable.
    pub fn undeclare(&mut self, id: QueryableId) -> Result<Vec<QueryableAction>, ZenohError> {
        todo!("undeclare")
    }

    /// Reply to a received query.
    pub fn reply(
        &mut self,
        query_id: QueryId,
        payload: Vec<u8>,
    ) -> Vec<QueryableAction> {
        todo!("reply")
    }

    /// Process an inbound queryable event.
    pub fn on_event(&mut self, event: QueryableEvent) -> Vec<QueryableAction> {
        todo!("on_event")
    }
}

impl Default for QueryableRouter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declare_and_receive_query() {
        let mut router = QueryableRouter::new();
        let (id, actions) = router.declare("harmony/content/a**").unwrap();
        assert_eq!(id, 1);
        assert!(actions.iter().any(|a| matches!(
            a,
            QueryableAction::SendQueryableDeclare { key_expr } if key_expr == "harmony/content/a**"
        )));

        // Simulate incoming query.
        let actions = router.on_event(QueryableEvent::QueryReceived {
            query_id: 42,
            key_expr: "harmony/content/abcd1234".to_string(),
            payload: b"request".to_vec(),
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            QueryableAction::DeliverQuery {
                queryable_id,
                query_id,
                key_expr,
                payload,
            } => {
                assert_eq!(*queryable_id, 1);
                assert_eq!(*query_id, 42);
                assert_eq!(key_expr, "harmony/content/abcd1234");
                assert_eq!(payload, b"request");
            }
            other => panic!("expected DeliverQuery, got {other:?}"),
        }
    }

    #[test]
    fn reply_sends_to_querier() {
        let mut router = QueryableRouter::new();
        let actions = router.reply(42, b"blob-data".to_vec());
        assert_eq!(actions.len(), 1);
        assert_eq!(
            actions[0],
            QueryableAction::SendReply {
                query_id: 42,
                payload: b"blob-data".to_vec(),
            }
        );
    }

    #[test]
    fn query_dispatches_to_matching_queryable() {
        let mut router = QueryableRouter::new();
        let (id_a, _) = router.declare("harmony/content/a**").unwrap();
        let (id_b, _) = router.declare("harmony/content/b**").unwrap();

        // Query matching only 'a' prefix.
        let actions = router.on_event(QueryableEvent::QueryReceived {
            query_id: 1,
            key_expr: "harmony/content/abcd".to_string(),
            payload: vec![],
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            QueryableAction::DeliverQuery { queryable_id, .. } => {
                assert_eq!(*queryable_id, id_a);
            }
            other => panic!("expected DeliverQuery for id_a, got {other:?}"),
        }

        // Query matching only 'b' prefix.
        let actions = router.on_event(QueryableEvent::QueryReceived {
            query_id: 2,
            key_expr: "harmony/content/bcde".to_string(),
            payload: vec![],
        });
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            QueryableAction::DeliverQuery { queryable_id, .. } => {
                assert_eq!(*queryable_id, id_b);
            }
            other => panic!("expected DeliverQuery for id_b, got {other:?}"),
        }
    }

    #[test]
    fn undeclare_stops_delivery() {
        let mut router = QueryableRouter::new();
        let (id, _) = router.declare("harmony/content/a**").unwrap();

        // Undeclare.
        let actions = router.undeclare(id).unwrap();
        assert!(actions.iter().any(|a| matches!(
            a,
            QueryableAction::SendQueryableUndeclare { key_expr } if key_expr == "harmony/content/a**"
        )));

        // Subsequent query should produce no actions.
        let actions = router.on_event(QueryableEvent::QueryReceived {
            query_id: 99,
            key_expr: "harmony/content/abcd".to_string(),
            payload: vec![],
        });
        assert!(actions.is_empty());
    }
}
```

**Step 3: Add module declaration and exports to lib.rs**

In `crates/harmony-zenoh/src/lib.rs`, add the module and re-exports:

```rust
pub mod queryable;
```

And add re-exports:

```rust
pub use queryable::{QueryableAction, QueryableEvent, QueryableId, QueryableRouter, QueryId};
```

**Step 4: Run tests to verify they fail**

Run: `cargo test -p harmony-zenoh queryable`
Expected: FAIL (4 tests panic with `todo!()`)

**Step 5: Implement QueryableRouter methods**

Replace the `todo!()` bodies in `queryable.rs`:

**`declare`:**
```rust
    pub fn declare(
        &mut self,
        key_expr: &str,
    ) -> Result<(QueryableId, Vec<QueryableAction>), ZenohError> {
        let ke = OwnedKeyExpr::autocanonize(key_expr.to_string())
            .map_err(|e| ZenohError::InvalidKeyExpr(e.to_string()))?;

        let id = self.next_queryable_id;
        self.next_queryable_id += 1;

        let sub_id = self.local_table.subscribe(&ke);
        self.local_sub_ids.insert(id, sub_id);
        self.local_queryables.insert(id, ke.clone());

        let actions = vec![QueryableAction::SendQueryableDeclare {
            key_expr: ke.to_string(),
        }];

        Ok((id, actions))
    }
```

**`undeclare`:**
```rust
    pub fn undeclare(&mut self, id: QueryableId) -> Result<Vec<QueryableAction>, ZenohError> {
        let ke = self
            .local_queryables
            .remove(&id)
            .ok_or(ZenohError::UnknownQueryableId(id))?;

        if let Some(sub_id) = self.local_sub_ids.remove(&id) {
            let _ = self.local_table.unsubscribe(sub_id);
        }

        Ok(vec![QueryableAction::SendQueryableUndeclare {
            key_expr: ke.to_string(),
        }])
    }
```

**`reply`:**
```rust
    pub fn reply(
        &mut self,
        query_id: QueryId,
        payload: Vec<u8>,
    ) -> Vec<QueryableAction> {
        vec![QueryableAction::SendReply { query_id, payload }]
    }
```

**`on_event`:**
```rust
    pub fn on_event(&mut self, event: QueryableEvent) -> Vec<QueryableAction> {
        match event {
            QueryableEvent::QueryReceived {
                query_id,
                key_expr,
                payload,
            } => self.handle_query_received(query_id, key_expr, payload),
            QueryableEvent::QueryableDeclared { .. }
            | QueryableEvent::QueryableUndeclared { .. } => {
                // Remote queryable tracking is informational only for now.
                vec![]
            }
        }
    }
```

Add the private helper:
```rust
    fn handle_query_received(
        &self,
        query_id: QueryId,
        key_expr: String,
        payload: Vec<u8>,
    ) -> Vec<QueryableAction> {
        let ke = match keyexpr::new(&key_expr) {
            Ok(ke) => ke,
            Err(_) => return vec![],
        };

        let matches = self.local_table.matches(ke);
        let mut actions = Vec::new();

        for sub_id in matches {
            // Find the QueryableId for this SubscriptionId.
            for (&qid, &sid) in &self.local_sub_ids {
                if sid == sub_id {
                    actions.push(QueryableAction::DeliverQuery {
                        queryable_id: qid,
                        query_id,
                        key_expr: key_expr.clone(),
                        payload: payload.clone(),
                    });
                    break;
                }
            }
        }

        actions
    }
```

**Step 6: Run tests**

Run: `cargo test -p harmony-zenoh queryable`
Expected: PASS (all 4 tests)

**Step 7: Commit**

```
feat(zenoh): add QueryableRouter for request/reply content retrieval
```

---

### Task 3: Implement Reticulum content bridge

**Files:**
- Modify: `crates/harmony-content/src/reticulum_bridge.rs`

**Step 1: Write the failing tests and type definitions**

Replace the contents of `crates/harmony-content/src/reticulum_bridge.rs` with the full module including types and tests, but `todo!()` for method bodies:

```rust
//! Reticulum content bridge — maps content blobs to resource transfers.
//!
//! Each blob (up to ~1MB) maps 1:1 to a single Reticulum resource transfer.
//! The CID is prepended as a 32-byte header so the receiver can verify
//! content integrity independently of the transport hash.

use crate::cid::ContentId;
use crate::error::ContentError;

/// CID header length prepended to blob data for transport.
const CID_HEADER_LEN: usize = 32;

/// Pack a content blob for transport: CID header (32 bytes) + blob data.
///
/// The caller feeds the returned bytes into `ResourceSender::new()`.
pub fn pack_for_transport(cid: &ContentId, data: &[u8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity(CID_HEADER_LEN + data.len());
    packed.extend_from_slice(&cid.to_bytes());
    packed.extend_from_slice(data);
    packed
}

/// Unpack received transport data: extract CID header and verify blob integrity.
///
/// Returns `(cid, blob_data)` if the CID matches the data's hash.
/// Returns `ContentError` if the data is too short or the hash doesn't match.
pub fn unpack_from_transport(transport_data: &[u8]) -> Result<(ContentId, Vec<u8>), ContentError> {
    if transport_data.len() < CID_HEADER_LEN {
        return Err(ContentError::InvalidBundleLength {
            len: transport_data.len(),
        });
    }

    let cid_bytes: [u8; 32] = transport_data[..CID_HEADER_LEN]
        .try_into()
        .expect("slice is exactly 32 bytes");
    let received_cid = ContentId::from_bytes(cid_bytes);
    let blob_data = &transport_data[CID_HEADER_LEN..];

    // Recompute CID from the blob data and verify it matches.
    let computed_cid = ContentId::for_blob(blob_data)?;

    if computed_cid != received_cid {
        return Err(ContentError::ChecksumMismatch);
    }

    Ok((received_cid, blob_data.to_vec()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::{BlobStore, MemoryBlobStore};

    #[test]
    fn pack_unpack_round_trip() {
        let mut store = MemoryBlobStore::new();
        let data = b"hello harmony reticulum bridge";
        let cid = store.insert(data).unwrap();

        let packed = pack_for_transport(&cid, data);
        assert_eq!(packed.len(), 32 + data.len());

        let (unpacked_cid, unpacked_data) = unpack_from_transport(&packed).unwrap();
        assert_eq!(unpacked_cid, cid);
        assert_eq!(unpacked_data, data);
    }

    #[test]
    fn unpack_detects_corrupt_content() {
        let mut store = MemoryBlobStore::new();
        let data = b"original data for corruption test";
        let cid = store.insert(data).unwrap();

        let mut packed = pack_for_transport(&cid, data);
        // Corrupt a byte in the blob data (after the 32-byte CID header).
        packed[CID_HEADER_LEN] ^= 0xFF;

        let result = unpack_from_transport(&packed);
        assert!(result.is_err(), "corrupted data should fail verification");
    }

    #[test]
    fn unpack_detects_wrong_cid_header() {
        let mut store = MemoryBlobStore::new();
        let data_a = b"blob A content";
        let data_b = b"blob B different content";
        let cid_a = store.insert(data_a).unwrap();
        let _cid_b = store.insert(data_b).unwrap();

        // Pack blob B's data with blob A's CID header.
        let packed = pack_for_transport(&cid_a, data_b);

        let result = unpack_from_transport(&packed);
        assert!(result.is_err(), "mismatched CID header should fail");
    }
}
```

**Step 2: Run tests to verify they pass**

Run: `cargo test -p harmony-content reticulum_bridge`
Expected: PASS (all 3 tests — these are pure functions, not `todo!()`)

Note: Unlike other tasks, the implementation is inline with the tests here because `pack_for_transport` and `unpack_from_transport` are simple pure functions. The sans-I/O resource transfer driving is the caller's responsibility.

**Step 3: Commit**

```
feat(content): implement Reticulum content bridge with CID-verified transport
```

---

### Task 4: Implement Zenoh content bridge

**Files:**
- Modify: `crates/harmony-content/src/zenoh_bridge.rs`

**Step 1: Write the types, tests (with `todo!()`), then implement**

Replace the contents of `crates/harmony-content/src/zenoh_bridge.rs`:

```rust
//! Zenoh content bridge — wires ContentStore to queryables and content announcements.
//!
//! [`ContentBridge`] manages prefix-sharded queryable registrations for content
//! retrieval and publishes content announcements when new blobs are stored.
//! Sans-I/O: the caller feeds events and executes returned actions.

use crate::blob::BlobStore;
use crate::cid::ContentId;

/// The 16 hex-digit prefixes for sharded queryable registration.
const HEX_PREFIXES: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f',
];

/// Inbound events the caller feeds into the bridge.
#[derive(Debug, Clone)]
pub enum ContentBridgeEvent {
    /// A remote peer is querying for content we might have.
    ContentQuery {
        query_id: u64,
        cid: ContentId,
    },
    /// Local store ingested new content (trigger announcement).
    ContentStored {
        cid: ContentId,
    },
}

/// Outbound actions the bridge returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentBridgeAction {
    /// Reply to a content query with the blob data.
    SendReply { query_id: u64, payload: Vec<u8> },
    /// Announce that we have this content on the network.
    Publish { key_expr: String, payload: Vec<u8> },
}

/// Generate the 16 prefix-sharded key expressions for content queryables.
///
/// Returns key expressions like `harmony/content/0**`, `harmony/content/1**`, etc.
pub fn content_queryable_key_exprs() -> Vec<String> {
    HEX_PREFIXES
        .iter()
        .map(|prefix| format!("harmony/content/{prefix}**"))
        .collect()
}

/// Convert a CID to its content key expression: `harmony/content/{hex_cid}`.
pub fn cid_to_key_expr(cid: &ContentId) -> String {
    format!("harmony/content/{}", hex::encode(cid.to_bytes()))
}

/// Convert a CID to its announce key expression: `harmony/announce/{hex_cid}`.
pub fn cid_to_announce_key_expr(cid: &ContentId) -> String {
    format!("harmony/announce/{}", hex::encode(cid.to_bytes()))
}

/// Handle a content query: look up the CID in the store, return SendReply if found.
pub fn handle_content_query<S: BlobStore>(
    store: &S,
    query_id: u64,
    cid: &ContentId,
) -> Vec<ContentBridgeAction> {
    match store.get(cid) {
        Some(data) => vec![ContentBridgeAction::SendReply {
            query_id,
            payload: data.to_vec(),
        }],
        None => vec![],
    }
}

/// Handle new content being stored: produce an announcement action.
pub fn handle_content_stored(cid: &ContentId, size: u32) -> Vec<ContentBridgeAction> {
    let key_expr = cid_to_announce_key_expr(cid);
    // Minimal announcement payload: just the size as 4 bytes big-endian.
    let payload = size.to_be_bytes().to_vec();
    vec![ContentBridgeAction::Publish { key_expr, payload }]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blob::MemoryBlobStore;

    #[test]
    fn query_returns_cached_content() {
        let mut store = MemoryBlobStore::new();
        let data = b"cached blob for query test";
        let cid = store.insert(data).unwrap();

        let actions = handle_content_query(&store, 42, &cid);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ContentBridgeAction::SendReply { query_id, payload } => {
                assert_eq!(*query_id, 42);
                assert_eq!(payload.as_slice(), data.as_slice());
            }
            other => panic!("expected SendReply, got {other:?}"),
        }
    }

    #[test]
    fn query_miss_produces_no_reply() {
        let store = MemoryBlobStore::new();
        let cid = ContentId::for_blob(b"not in store").unwrap();

        let actions = handle_content_query(&store, 99, &cid);
        assert!(actions.is_empty(), "cache miss should produce no actions");
    }

    #[test]
    fn content_stored_triggers_announcement() {
        let cid = ContentId::for_blob(b"new content to announce").unwrap();
        let size = cid.payload_size();

        let actions = handle_content_stored(&cid, size);
        assert_eq!(actions.len(), 1);
        match &actions[0] {
            ContentBridgeAction::Publish { key_expr, payload } => {
                assert!(
                    key_expr.starts_with("harmony/announce/"),
                    "key_expr should start with harmony/announce/, got: {key_expr}"
                );
                // Payload should be 4 bytes (size as big-endian u32).
                assert_eq!(payload.len(), 4);
                let announced_size = u32::from_be_bytes(payload[..4].try_into().unwrap());
                assert_eq!(announced_size, size);
            }
            other => panic!("expected Publish, got {other:?}"),
        }
    }
}
```

**Step 2: Run tests**

Run: `cargo test -p harmony-content zenoh_bridge`
Expected: PASS (all 3 tests — pure functions, implementation inline)

**Step 3: Commit**

```
feat(content): implement Zenoh content bridge with queryable support and announcements
```

---

### Task 5: Final quality gates

**Step 1: Run all new tests**

Run: `cargo test -p harmony-zenoh queryable && cargo test -p harmony-content reticulum_bridge && cargo test -p harmony-content zenoh_bridge`
Expected: All 10 tests pass (4 + 3 + 3)

**Step 2: Run clippy on affected crates**

Run: `cargo clippy -p harmony-zenoh -p harmony-content`
Expected: No warnings

**Step 3: Run format check**

Run: `cargo fmt --all -- --check`
Expected: No formatting issues

**Step 4: Run full workspace tests (regression check)**

Run: `cargo test --workspace`
Expected: All tests pass (previous 636 + 10 new = 646+)

**Step 5: Commit any cleanups, then the bead is ready for delivery**

Close the bead: `bd close harmony-0p5 --reason "Reticulum content bridge, Zenoh queryable + content bridge implemented"`

# Gateway Merkle Bridge: CAS Mailbox + Zenoh Publication (ZEB-114 Phase 1)

## Overview

Add a Merkle mailbox write path to the harmony-mail gateway so that inbound SMTP messages are stored in the CAS Merkle tree (MailRoot/MailFolder/MailPage) alongside the existing ImapStore. Publish updated root CIDs to Zenoh so harmony-client can receive mail natively without IMAP.

This is Phase 1 of ZEB-114. Later phases add the client receive path (Phase 2) and client send path (Phase 3).

## Scope

- **Dual-write on delivery:** Each inbound SMTP message writes to both ImapStore (for IMAP clients) and the CAS Merkle mailbox (for harmony-client).
- **Per-user Merkle trees:** One MailRoot per registered user, with independent inbox/sent/drafts/trash folders.
- **Zenoh publication:** Publish updated root CID via pub/sub (real-time) and queryable (catch-up).
- **Zenoh session at startup:** First feature requiring the gateway to open a Zenoh session.

## Non-Scope

- Client-side tree walking and mail UI (Phase 2).
- Native Harmony-to-Harmony send from client (Phase 3).
- Modifications to the IMAP store or IMAP protocol handling.
- Zenoh subscriptions in the gateway (gateway is publisher/queryable only).

## Architecture

```
SMTP in → translate → spam check → CAS ingest
                                      │
                              ┌───────┴───────┐
                              ↓               ↓
                        ImapStore         MailboxManager
                      (SQLite, IMAP)    (CAS Merkle tree)
                              │               │
                              ↓               ↓
                     IMAP clients       Zenoh publish
                     (Thunderbird)     (harmony-client)
```

The two write paths are independent. If the Merkle write fails, the IMAP delivery is already done — the message is safe. The Merkle tree is a secondary output, not the critical path.

## Design Decisions

- **Dual-write over derived views** — The IMAP store has mutable state (flags, seen/unseen, UID ordering) that doesn't map cleanly to an immutable Merkle tree. Trying to derive one from the other adds complexity. Two independent writes are simpler and avoid coupling.
- **Per-user from the start** — MailRoot already has `owner_address`, Zenoh topics are already per-user. Avoids the "global INBOX" problem that PR #228 reviewers flagged.
- **Pub/sub + queryable** — Matches the proven pattern from harmony-db DAG sync (ZEB-108). Real-time push for online clients, pull-based catch-up for clients coming online.
- **Head-append, newest-first** — New messages prepend to the first page. Only the head page, folder, and root get new CIDs per message. Matches the existing `MailFolder.page_cids` ordering (documented as newest-first).
- **In-memory manager with CAS durability** — The Merkle tree is self-describing from any root CID. Only the "current root pointer" needs persistence, stored in a small dedicated SQLite table.

## MailboxManager

### State

```rust
pub struct MailboxManager {
    /// Per-user current root CIDs.
    roots: HashMap<[u8; ADDRESS_HASH_LEN], [u8; CID_LEN]>,
    /// CAS storage path for DiskBookStore access.
    content_store_path: PathBuf,
    /// Persistence for root CID pointers.
    db: rusqlite::Connection,
    /// Optional Zenoh session for publishing root CID updates.
    /// None if Zenoh is unavailable — Merkle writes still happen to CAS.
    zenoh: Option<Arc<zenoh::Session>>,
}
```

The `db` is a separate SQLite file (`mailbox_roots.db`) alongside the ImapStore, with one table:

```sql
CREATE TABLE IF NOT EXISTS mailbox_roots (
    address BLOB NOT NULL UNIQUE,
    root_cid BLOB NOT NULL
);
```

### Startup

1. `run()` opens `mailbox_roots.db` at `{store_path_dir}/mailbox_roots.db`
2. For each user in ImapStore, check if they have a root CID in `mailbox_roots`
3. If no root exists, create an empty MailRoot with 4 empty folders (inbox, sent, drafts, trash), each pointing to an empty MailPage. Ingest all blobs into CAS. Store root CID.
4. Load all root CIDs into the in-memory HashMap
5. If Zenoh is configured, register a queryable for each user on `harmony/messages/{addr_hex}/inbox`

### Insert Message

Called from Phase 5 of `DeliverToHarmony`, after the IMAP insert succeeds:

```
insert_message(user_address, message_cid, msg_id, sender_address, timestamp, subject) -> Result<()>
```

Steps:
1. Look up user's current root CID in HashMap
2. Load MailRoot from CAS (deserialize from root CID)
3. Load inbox MailFolder from CAS (first folder CID in MailRoot)
4. Load head MailPage from CAS (first page CID in MailFolder, or create new if folder has no pages)
5. Build `MessageEntry { message_cid, message_id, sender_address, timestamp, subject_snippet, read: false }`
6. If head page has room (< PAGE_CAPACITY=100): prepend entry to page
7. If head page is full: create new page with just this entry, push its CID to front of `page_cids`
8. Write new MailPage to CAS → new page CID
9. Update MailFolder: replace head page CID, increment `message_count` and `unread_count`
10. Write new MailFolder to CAS → new folder CID
11. Update MailRoot: replace inbox folder CID, update `updated_at`
12. Write new MailRoot to CAS → new root CID
13. Update HashMap and `mailbox_roots.db`
14. If Zenoh is available: publish new root CID to `harmony/messages/{addr_hex}/inbox`

### Error Handling

All errors in the Merkle write path are logged and swallowed. The IMAP delivery has already succeeded — the Merkle tree will catch up on the next successful insert (since it always reads current state from the root CID).

## DeliverToHarmony Phase 5

Phase 5 runs after Phase 4 (IMAP insert), using values already computed:

- `message_cid` — from Phase 3 (CAS ingest)
- `msg_id`, `msg_timestamp` — extracted from translated message before Phase 3 closure
- `sender_address` — extracted from translated message before Phase 3 closure (new extraction needed)
- `subject` — extracted from translated message before Phase 3 closure (new extraction needed, truncated to 128 bytes)
- Recipient addresses — from the `recipients` vec

Phase 5 runs inside `spawn_blocking` because CAS reads (loading current page/folder from `DiskBookStore`) and writes (ingesting new blobs) are synchronous.

For each recipient that was successfully delivered to ImapStore (tracked by `delivered_count` in Phase 4, but we need to track *which* recipients succeeded):

```rust
// Phase 5: Update Merkle mailbox (non-critical)
if let Some(ref mailbox_mgr) = mailbox_manager {
    for addr in &successfully_delivered {
        if let Err(e) = mailbox_mgr.lock().unwrap().insert_message(
            addr, &message_cid, &msg_id, &sender_address,
            msg_timestamp, &subject_snippet,
        ) {
            tracing::warn!(
                recipient = hex::encode(addr),
                error = %e,
                "Merkle mailbox update failed, IMAP delivery unaffected"
            );
        }
    }
}
```

Note: Phase 4 currently counts deliveries but doesn't track which recipients succeeded. We need to change `delivered_count` to a `Vec<[u8; ADDRESS_HASH_LEN]>` of successfully-delivered addresses so Phase 5 knows which users to update.

## Zenoh Integration

### Session

The gateway's `run()` function creates an optional Zenoh session at startup:

```rust
let zenoh_session: Option<Arc<zenoh::Session>> = match &config.zenoh {
    Some(cfg) if cfg.enabled => {
        match zenoh::open(cfg.to_zenoh_config()).await {
            Ok(session) => Some(Arc::new(session)),
            Err(e) => {
                tracing::warn!(error = %e, "Zenoh unavailable, Merkle publication disabled");
                None
            }
        }
    }
    _ => None,
};
```

If Zenoh is unavailable, the gateway still works — IMAP delivery and CAS writes proceed, but root CID publication is skipped.

### Pub/Sub (Real-Time)

On each Merkle tree update, publish the 32-byte root CID:

```
Topic: harmony/messages/{addr_hex}/inbox
Payload: [u8; 32] (new root CID)
```

### Queryable (Catch-Up)

At startup, register a queryable for each local user:

```
Key: harmony/messages/{addr_hex}/inbox
Response: [u8; 32] (current root CID from HashMap)
```

When harmony-client comes online, it queries this topic to get the latest root CID, then walks the Merkle tree via standard CAS block fetches.

### Configuration

New optional section in config.toml:

```toml
[zenoh]
enabled = true
# Optional explicit endpoint; if omitted, uses Zenoh's default peer discovery
# endpoint = "tcp/192.168.1.1:7447"
```

## Files Changed

### New

- **`crates/harmony-mail/src/mailbox_manager.rs`** — `MailboxManager` struct: in-memory root tracking, Merkle tree insert/split logic, CAS read/write via `DiskBookStore`, root CID persistence (`mailbox_roots.db`), Zenoh pub/queryable registration.

### Modified

- **`crates/harmony-mail/src/server.rs`** — Create `MailboxManager` and optional Zenoh session in `run()`. Thread `Arc<Mutex<MailboxManager>>` through SMTP handlers and `process_async_actions`. Add Phase 5 in `DeliverToHarmony`. Extract `sender_address` and `subject` before Phase 3 closure. Change Phase 4's `delivered_count: u32` to `delivered_to: Vec<[u8; ADDRESS_HASH_LEN]>`.
- **`crates/harmony-mail/src/config.rs`** — Add optional `ZenohConfig` section with `enabled` flag and optional `endpoint`.
- **`crates/harmony-mail/Cargo.toml`** — Add `zenoh` dependency.

### Unchanged

- **`mailbox.rs`** — Existing MailRoot/MailFolder/MailPage/MessageEntry types used as-is.
- **`imap_store.rs`** — IMAP store untouched. Independent write path.

## Testing

### Unit Tests (mailbox_manager.rs)

- **`insert_into_empty_mailbox`** — Create empty MailRoot, insert one message, verify inbox folder has message_count=1, head page has 1 entry with correct fields, root CID changed from initial.
- **`insert_fills_page`** — Insert 100 messages, verify single page at capacity (100 entries), folder message_count=100.
- **`insert_splits_page`** — Insert 101 messages, verify two pages: head page has 1 entry (newest), second page has 100. Folder has 2 page_cids.
- **`root_cid_persistence`** — Create manager, insert message, drop manager, reopen from same DB path, verify root CID matches and tree is intact via CAS.
- **`per_user_isolation`** — Insert messages for two users, verify each has independent trees with correct message counts.

### Integration Test (server.rs)

- **`smtp_delivers_to_merkle_mailbox`** — Full SMTP transaction, then verify:
  - MailboxManager has a root CID for the recipient
  - Root CID resolves to a valid MailRoot via CAS deserialization
  - Inbox MailFolder has `message_count == 1`
  - Head page entry has correct `message_cid` (matches ImapStore's CID), `sender_address`, and subject snippet
  - Both ImapStore and Merkle tree agree on message content (dual-write consistency)

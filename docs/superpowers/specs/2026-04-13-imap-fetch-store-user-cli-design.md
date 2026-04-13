# IMAP FETCH/STORE Wiring + User Add CLI (ZEB-111 Part 1)

## Overview

Wire IMAP FETCH and STORE to real message content and flags, and implement the user add CLI. The IMAP protocol parsing, state machine, SQLite store, and response renderer are all complete (PR #212). This spec covers the I/O layer wiring that connects these existing components.

## Scope

Items #1, #2 (partial — STORE only), and #3 from ZEB-111:

- **FETCH**: Resolve message CIDs from SQLite, retrieve content from CAS via DAG reassembly, render IMAP responses
- **STORE**: Apply flag operations (set/add/remove) to SQLite, send untagged updates
- **User add CLI**: Interactive password prompt, create user in SQLite store

SEARCH, COPY, STARTTLS, and SMTP integration are out of scope (future work).

## Design Decisions

- **Per-connection DiskBookStore** — `DiskBookStore` uses `RefCell` (not `Sync`), so it cannot be `Arc`-shared across async tasks. Instead, each connection creates its own `DiskBookStore::new(path)` when FETCH needs CAS access. Construction is trivially cheap (stores path + empty cache HashMap). The file system is the shared state — concurrent readers are safe.
- **Content store path in config** — A new `content_store_path` field in the TOML `[imap]` section tells the server where to find CAS data (`commits/` and `blobs/` subdirectories). Defaults to `/var/lib/harmony-mail/content/`.
- **Skip messages without CIDs** — Messages with `message_cid: None` (pre-CAS legacy) are skipped during FETCH. The client receives responses for all messages that have content available.
- **Graceful per-message errors** — If CAS retrieval or deserialization fails for one message, log a warning and skip it rather than failing the entire FETCH command.
- **rpassword for hidden input** — The user add CLI uses `rpassword` for secure password entry with confirmation.
- **Drop --namespace from CLI** — No schema field or consumer exists for namespace. YAGNI.

## FETCH Wiring

### Data Flow

```text
1. Extract selected mailbox from session.state
2. store.get_messages(mailbox_id)                    -> Vec<MessageRow>
3. Build uid_to_seqnum HashMap (same pattern as Expunge)
4. resolve_sequence_set(set, uid_mode, &messages)    -> Vec<(uid, seqnum)>
5. For each resolved (uid, seqnum):
   a. Find MessageRow by uid
   b. If message_cid is None -> skip
   c. ContentId::from_bytes(message_cid)             -> content_id
   d. spawn_blocking: DiskBookStore::new(content_store_path)
   e. spawn_blocking: dag::reassemble(&content_id, &book_store) -> raw_bytes
   f. HarmonyMessage::from_bytes(&raw_bytes)          -> message
   g. store.get_flags(message_row.id)                 -> flags
   h. build_fetch_response(seqnum, uid, &attrs, &msg, &flags, rfc822_size, &domain)
   i. Write response.to_bytes() to client
6. Send tagged OK
```

### Sequence Set Resolution

A helper function resolves `SequenceSet` to `(uid, seqnum)` pairs:

```rust
fn resolve_sequence_set(
    set: &SequenceSet,
    uid_mode: bool,
    messages: &[MessageRow],
) -> Vec<(u32, u32)>
```

- `uid_mode=true`: ranges refer to UIDs directly. `u32::MAX` means the highest UID in the mailbox.
- `uid_mode=false`: ranges refer to 1-based sequence numbers. Map back to UIDs via the sorted message list.
- Returns `(uid, seqnum)` pairs for all matching messages, preserving order.

This helper is shared between FETCH and STORE.

### Async Bridging

`DiskBookStore` does file I/O, so CAS retrieval runs inside `tokio::task::spawn_blocking`. This matches the existing pattern for `ImapAction::Authenticate` (server.rs line 994).

## STORE Wiring

### Data Flow

```text
1. Extract selected mailbox from session.state
2. store.get_messages(mailbox_id)                    -> Vec<MessageRow>
3. Build uid_to_seqnum HashMap
4. resolve_sequence_set(set, uid_mode, &messages)    -> Vec<(uid, seqnum)>
5. For each resolved (uid, seqnum):
   a. Find MessageRow by uid -> message_row.id
   b. Match StoreOperation:
      - Set/SetSilent   -> store.set_flags(id, &flags)
      - Add/AddSilent   -> store.add_flags(id, &flags)
      - Remove/RemoveSilent -> store.remove_flags(id, &flags)
   c. If not silent:
      - store.get_flags(id) -> updated_flags
      - Write untagged: "* {seqnum} FETCH (FLAGS ({flags}) UID {uid})"
6. Send tagged OK
```

No CAS access needed — flags are purely SQLite.

## User Add CLI

### Current State

```rust
UserAction::Add {
    name: String,
    namespace: String,  // removed — no schema field or consumer
    identity: String,
}
```

### New Interface

```rust
UserAction::Add {
    #[arg(long)]
    name: String,
    #[arg(long)]
    identity: String,  // hex-encoded 16-byte harmony address (32 hex chars)
}
```

### Flow

```text
1. Parse --name and --identity from CLI
2. Validate identity: hex::decode -> [u8; 16] (error if not 32 hex chars)
3. Prompt "Password: " via rpassword::prompt_password()
4. Prompt "Confirm password: " via rpassword::prompt_password()
5. If passwords don't match -> print error, exit
6. Load config, open ImapStore::open(config.imap.store_path)
7. store.create_user(name, password, &harmony_address)
8. Print "User '{name}' created successfully"
```

On duplicate username, `create_user` returns a SQLite unique constraint error — surface as `Error: user '{name}' already exists`.

## Config Changes

Add to `config.rs` TOML `ImapConfig`:

```rust
#[serde(default = "default_content_store_path")]
pub content_store_path: String,
```

Default: `/var/lib/harmony-mail/content/`

This path must contain `commits/` and `blobs/` subdirectories (the `DiskBookStore` layout from harmony-db).

## Files Changed

### Modified

- `crates/harmony-mail/Cargo.toml` — add `harmony-content` (workspace, std), `harmony-db` (workspace), `rpassword = "5"` dependencies
- `crates/harmony-mail/src/config.rs` — add `content_store_path` field + default function + `Default` impl update
- `crates/harmony-mail/src/server.rs` — thread `content_store_path` through startup/handler/async-actions. Replace FETCH and STORE stubs. Add `resolve_sequence_set` helper.
- `crates/harmony-mail/src/main.rs` — wire user add: remove `--namespace`, add password prompt, call `create_user()`

### No New Files

All changes are wiring in existing files.

## Testing Strategy

### FETCH

- **Round-trip**: Build a `HarmonyMessage`, serialize via `to_bytes()`, DAG-ingest into `MemoryBookStore`, insert into `ImapStore` with the CID, then exercise the FETCH rendering path. Verify the IMAP response contains correct envelope, flags, and body content.
- **Missing CID**: Insert message with `message_cid: None`, verify FETCH skips it.
- **UID vs sequence mode**: Test `resolve_sequence_set` with `uid_mode=true` and `uid_mode=false`, including wildcard (`*`) ranges and multi-range sets like `1:3,5`.

### STORE

- **Set/Add/Remove**: Exercise each `StoreOperation`, verify flags in the database.
- **Silent mode**: Verify silent operations produce no untagged FETCH responses.

### User Add CLI

- **Invalid identity hex**: Non-hex or wrong length produces error.
- **Duplicate user**: Second create with same name produces error.
- Argument validation tested directly — `create_user()` itself is already tested in `imap_store.rs`.

### Test Dependencies

- `tempfile` (already a dev-dependency) for temp directories
- `harmony-content`'s `MemoryBookStore` for in-memory CAS

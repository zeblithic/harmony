# IMAP FETCH/STORE Wiring + User Add CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire IMAP FETCH and STORE to real message content and flags, and implement the user add CLI for harmony-mail.

**Architecture:** Per-connection `DiskBookStore` retrieves message content from CAS via DAG reassembly. A shared `resolve_sequence_set` helper maps SequenceSet to UIDs for both FETCH and STORE. STORE feeds results back through the sans-I/O state machine's `StoreComplete` event. The user CLI opens the store directly and calls `create_user()`.

**Tech Stack:** Rust, tokio (async), harmony-content (CAS/DAG), harmony-db (DiskBookStore), rusqlite, rpassword

---

## File Structure

| File | Responsibility |
|------|---------------|
| `crates/harmony-mail/Cargo.toml` | Add harmony-content, harmony-db, rpassword dependencies |
| `crates/harmony-mail/src/config.rs` | Add `content_store_path` field to TOML ImapConfig |
| `crates/harmony-mail/src/server.rs` | Thread content_store_path, replace FETCH/STORE stubs, add `resolve_sequence_set` helper |
| `crates/harmony-mail/src/main.rs` | Wire user add CLI: password prompt, hex validation, create_user call |

No new files — all changes are wiring in existing code.

---

### Task 1: Add dependencies and content_store_path config

**Files:**
- Modify: `crates/harmony-mail/Cargo.toml`
- Modify: `crates/harmony-mail/src/config.rs:168-200`

- [ ] **Step 1: Add dependencies to Cargo.toml**

Add these three dependencies to the `[dependencies]` section in `crates/harmony-mail/Cargo.toml`:

```toml
harmony-content = { workspace = true, features = ["std"] }
harmony-db = { workspace = true }
rpassword = "5"
```

- [ ] **Step 2: Add content_store_path field to config.rs**

In `crates/harmony-mail/src/config.rs`, add the `content_store_path` field to the `ImapConfig` struct (after `store_path` at line 180):

```rust
/// IMAP server configuration (v1.1).
///
/// Defaults to disabled. Add `[imap]` section to config.toml to enable.
#[derive(Debug, Deserialize)]
pub struct ImapConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_listen_imap")]
    pub listen_imap: String,
    #[serde(default = "default_listen_imaps")]
    pub listen_imaps: String,
    #[serde(default = "default_imap_store_path")]
    pub store_path: String,
    #[serde(default = "default_content_store_path")]
    pub content_store_path: String,
    /// IDLE timeout in seconds (default: 30 minutes per RFC 9051 recommendation).
    #[serde(default = "default_idle_timeout")]
    pub idle_timeout: u64,
    /// Maximum authentication failures before disconnect.
    #[serde(default = "default_max_auth_failures")]
    pub max_auth_failures: u32,
}
```

- [ ] **Step 3: Add default function and update Default impl**

Add the default function after `default_imap_store_path` (line 209):

```rust
fn default_content_store_path() -> String {
    "/var/lib/harmony-mail/content".to_string()
}
```

Update the `Default` impl to include the new field:

```rust
impl Default for ImapConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            listen_imap: default_listen_imap(),
            listen_imaps: default_listen_imaps(),
            store_path: default_imap_store_path(),
            content_store_path: default_content_store_path(),
            idle_timeout: default_idle_timeout(),
            max_auth_failures: default_max_auth_failures(),
        }
    }
}
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors (warnings about unused imports are fine)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/Cargo.toml crates/harmony-mail/src/config.rs
git commit -m "feat(mail): add CAS dependencies and content_store_path config (ZEB-111)"
```

---

### Task 2: Implement resolve_sequence_set helper with tests

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

This helper is shared by FETCH and STORE. It resolves a `SequenceSet` to `(uid, seqnum)` pairs.

- [ ] **Step 1: Write the tests**

Add a `#[cfg(test)]` module at the bottom of `crates/harmony-mail/src/server.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::resolve_sequence_set;
    use crate::imap_parse::{SequenceRange, SequenceSet};
    use crate::imap_store::MessageRow;
    use crate::message::MESSAGE_ID_LEN;

    fn make_messages(uids: &[u32]) -> Vec<MessageRow> {
        uids.iter()
            .map(|&uid| MessageRow {
                id: uid as i64,
                mailbox_id: 1,
                uid,
                harmony_msg_id: [0u8; MESSAGE_ID_LEN],
                message_cid: None,
                internal_date: 1000,
                rfc822_size: 100,
            })
            .collect()
    }

    fn seq(ranges: Vec<(u32, Option<u32>)>) -> SequenceSet {
        SequenceSet {
            ranges: ranges
                .into_iter()
                .map(|(start, end)| SequenceRange { start, end })
                .collect(),
        }
    }

    #[test]
    fn resolve_uid_mode_single() {
        let msgs = make_messages(&[1, 3, 5, 7]);
        let result = resolve_sequence_set(&seq(vec![(3, None)]), true, &msgs);
        assert_eq!(result, vec![(3, 2)]);
    }

    #[test]
    fn resolve_uid_mode_range() {
        let msgs = make_messages(&[1, 3, 5, 7]);
        let result = resolve_sequence_set(&seq(vec![(3, Some(5))]), true, &msgs);
        assert_eq!(result, vec![(3, 2), (5, 3)]);
    }

    #[test]
    fn resolve_uid_mode_wildcard() {
        let msgs = make_messages(&[1, 3, 5, 7]);
        // 5:* should match UIDs 5 and 7
        let result = resolve_sequence_set(&seq(vec![(5, Some(u32::MAX))]), true, &msgs);
        assert_eq!(result, vec![(5, 3), (7, 4)]);
    }

    #[test]
    fn resolve_seqnum_mode_single() {
        let msgs = make_messages(&[10, 20, 30]);
        // Sequence number 2 -> UID 20
        let result = resolve_sequence_set(&seq(vec![(2, None)]), false, &msgs);
        assert_eq!(result, vec![(20, 2)]);
    }

    #[test]
    fn resolve_seqnum_mode_range() {
        let msgs = make_messages(&[10, 20, 30]);
        // Sequence numbers 1:2 -> UIDs 10, 20
        let result = resolve_sequence_set(&seq(vec![(1, Some(2))]), false, &msgs);
        assert_eq!(result, vec![(10, 1), (20, 2)]);
    }

    #[test]
    fn resolve_seqnum_mode_wildcard() {
        let msgs = make_messages(&[10, 20, 30]);
        // 2:* -> seqnums 2, 3 -> UIDs 20, 30
        let result = resolve_sequence_set(&seq(vec![(2, Some(u32::MAX))]), false, &msgs);
        assert_eq!(result, vec![(20, 2), (30, 3)]);
    }

    #[test]
    fn resolve_multi_range() {
        let msgs = make_messages(&[1, 2, 3, 4, 5]);
        // "1,3:4" in UID mode
        let result = resolve_sequence_set(
            &seq(vec![(1, None), (3, Some(4))]),
            true,
            &msgs,
        );
        assert_eq!(result, vec![(1, 1), (3, 3), (4, 4)]);
    }

    #[test]
    fn resolve_empty_mailbox() {
        let msgs: Vec<MessageRow> = vec![];
        let result = resolve_sequence_set(&seq(vec![(1, Some(u32::MAX))]), true, &msgs);
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_uid_not_found() {
        let msgs = make_messages(&[1, 3, 5]);
        // UID 2 doesn't exist — should be skipped
        let result = resolve_sequence_set(&seq(vec![(2, None)]), true, &msgs);
        assert!(result.is_empty());
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p harmony-mail resolve_sequence_set -- --nocapture`
Expected: FAIL — `resolve_sequence_set` not found

- [ ] **Step 3: Implement resolve_sequence_set**

Add this function in `crates/harmony-mail/src/server.rs`, above the `#[cfg(test)]` module (or near the other helper functions at the bottom of the file):

```rust
/// Resolve an IMAP SequenceSet to (uid, seqnum) pairs.
///
/// `messages` must be sorted by UID (ascending), as returned by `ImapStore::get_messages`.
/// In `uid_mode`, set ranges refer to UIDs. Otherwise, they refer to 1-based sequence numbers.
/// `u32::MAX` represents the wildcard `*` (highest UID or last sequence number).
fn resolve_sequence_set(
    set: &imap_parse::SequenceSet,
    uid_mode: bool,
    messages: &[imap_store::MessageRow],
) -> Vec<(u32, u32)> {
    if messages.is_empty() {
        return Vec::new();
    }

    let max_uid = messages.last().map(|m| m.uid).unwrap_or(0);
    let max_seqnum = messages.len() as u32;
    let mut result = Vec::new();

    for range in &set.ranges {
        let (start, end) = if uid_mode {
            let s = if range.start == u32::MAX { max_uid } else { range.start };
            let e = match range.end {
                None => s,
                Some(u32::MAX) => max_uid,
                Some(e) => e,
            };
            (s, e)
        } else {
            let s = if range.start == u32::MAX { max_seqnum } else { range.start };
            let e = match range.end {
                None => s,
                Some(u32::MAX) => max_seqnum,
                Some(e) => e,
            };
            (s, e)
        };

        let (lo, hi) = if start <= end { (start, end) } else { (end, start) };

        if uid_mode {
            for (idx, msg) in messages.iter().enumerate() {
                if msg.uid >= lo && msg.uid <= hi {
                    result.push((msg.uid, (idx + 1) as u32));
                }
            }
        } else {
            for seqnum in lo..=hi {
                if seqnum >= 1 && seqnum <= max_seqnum {
                    let msg = &messages[(seqnum - 1) as usize];
                    result.push((msg.uid, seqnum));
                }
            }
        }
    }

    result
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p harmony-mail resolve_sequence_set -- --nocapture`
Expected: all 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): add resolve_sequence_set helper for FETCH/STORE (ZEB-111)"
```

---

### Task 3: Thread content_store_path through IMAP connection handlers

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:290-410,832-933,981-991`

This is mechanical parameter threading — no logic changes.

- [ ] **Step 1: Add PathBuf parameter to handle_imap_connection**

In `crates/harmony-mail/src/server.rs`, update the `handle_imap_connection` function signature (line 832) to add the `content_store_path` parameter:

```rust
async fn handle_imap_connection<R, W>(
    reader: R,
    mut writer: W,
    config: ImapConfig,
    store: Arc<ImapStore>,
    idle_timeout: Duration,
    content_store_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
```

Add `use std::path::PathBuf;` to the imports at the top of `server.rs` if not already present.

Save the domain before creating the session (the `config` is moved into `ImapSession::new` and `ImapSession.config` is private, so we need to clone it first):

```rust
    let domain = config.domain.clone();
    let mut session = ImapSession::new(config);
```

Pass both `content_store_path` and `domain` through to `process_imap_async_actions` in the body (line 866):

```rust
                        process_imap_async_actions(
                            &actions,
                            &mut session,
                            &mut writer,
                            &store,
                            &mut framed,
                            &content_store_path,
                            &domain,
                        )
                        .await?;
```

- [ ] **Step 2: Add PathBuf parameter to handle_imap_connection_starttls**

Update `handle_imap_connection_starttls` (line 917) to accept and forward the path:

```rust
async fn handle_imap_connection_starttls<R, W>(
    reader: R,
    writer: W,
    config: ImapConfig,
    store: Arc<ImapStore>,
    idle_timeout: Duration,
    _tls_acceptor: Option<Arc<tokio_rustls::TlsAcceptor>>,
    content_store_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    handle_imap_connection(reader, writer, config, store, idle_timeout, content_store_path).await
}
```

- [ ] **Step 3: Add PathBuf parameter to process_imap_async_actions**

Update `process_imap_async_actions` (line 981) to accept the path and domain:

```rust
async fn process_imap_async_actions<R, W>(
    actions: &[ImapAction],
    session: &mut ImapSession,
    writer: &mut W,
    store: &Arc<ImapStore>,
    framed: &mut FramedRead<R, ImapCodec>,
    content_store_path: &Path,
    domain: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
```

Add `use std::path::Path;` to the imports if not already present.

- [ ] **Step 4: Update call sites in the IMAP setup block**

In the IMAP setup block (around line 290), capture the content store path from config:

```rust
    let content_store_path = PathBuf::from(&config.imap.content_store_path);
```

Add this line after `let imap_max_auth_failures = config.imap.max_auth_failures;` (line 302).

Update the port 993 spawned task (around line 342) — add `content_store_path` cloning and pass it through. Before the `tokio::spawn` inside the per-connection handler, clone the path:

```rust
                                    let csp = content_store_path.clone();
```

And pass it to the call:

```rust
                                                if let Err(e) = handle_imap_connection(reader, writer, imap_cfg, st, imap_idle_timeout, csp).await {
```

Update the port 143 spawned task (around line 398) similarly — clone `content_store_path` and pass it:

```rust
                                let csp = content_store_path.clone();
```

```rust
                                    if let Err(e) = handle_imap_connection_starttls(reader, writer, imap_cfg, st, imap_idle_timeout, acc, csp).await {
```

Note: `content_store_path` must be cloned into the outer `tokio::spawn` closure for each listener (port 993 and port 143), then cloned again per-connection inside the inner `tokio::spawn`.

- [ ] **Step 5: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors (content_store_path unused in process_imap_async_actions body — that's fine for now, the FETCH/STORE tasks will use it)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): thread content_store_path through IMAP handlers (ZEB-111)"
```

---

### Task 4: Wire FETCH handler

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:1248-1257`

Replace the `NO [CANNOT]` stub with real CAS retrieval and response rendering.

- [ ] **Step 1: Write the FETCH round-trip test**

Add to the `#[cfg(test)] mod tests` in `server.rs`:

```rust
    use crate::imap_parse::FetchAttribute;
    use crate::imap_render::build_fetch_response;
    use crate::imap_store::ImapStore;
    use crate::message::{
        HarmonyMessage, MailMessageType, MessageFlags, Recipient, RecipientType, ADDRESS_HASH_LEN,
        CID_LEN, VERSION,
    };
    use harmony_content::book::MemoryBookStore;
    use harmony_content::chunker::ChunkerConfig;
    use harmony_content::cid::ContentId;
    use harmony_content::dag;
    use tempfile::tempdir;

    /// Build a minimal HarmonyMessage for testing.
    fn test_message() -> HarmonyMessage {
        HarmonyMessage {
            version: VERSION,
            message_type: MailMessageType::Email,
            flags: MessageFlags::new(false, false, false),
            timestamp: 1713000000,
            message_id: [1u8; MESSAGE_ID_LEN],
            in_reply_to: None,
            sender_address: [0xAA; ADDRESS_HASH_LEN],
            recipients: vec![Recipient {
                address_hash: [0xBB; ADDRESS_HASH_LEN],
                recipient_type: RecipientType::To,
            }],
            subject: "Test subject".to_string(),
            body: "Hello, world!".to_string(),
            attachments: vec![],
        }
    }

    #[test]
    fn fetch_round_trip_via_cas() {
        let msg = test_message();
        let msg_bytes = msg.to_bytes().unwrap();

        // Ingest message into CAS
        let mut store = MemoryBookStore::new();
        let root_cid = dag::ingest(&msg_bytes, &ChunkerConfig::DEFAULT, &mut store).unwrap();

        // Set up IMAP store and insert message with CID
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        let cid_bytes: [u8; CID_LEN] = root_cid.to_bytes();
        let uid = imap_store
            .insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], Some(&cid_bytes), 1713000000, 500)
            .unwrap();

        // Reassemble from CAS and deserialize
        let reassembled = dag::reassemble(&root_cid, &store).unwrap();
        let restored = HarmonyMessage::from_bytes(&reassembled).unwrap();
        assert_eq!(restored.subject, "Test subject");
        assert_eq!(restored.body, "Hello, world!");

        // Render FETCH response
        let flags = imap_store.get_flags(1).unwrap(); // message db id = 1 (first message)
        let attrs = vec![FetchAttribute::Flags, FetchAttribute::Uid, FetchAttribute::Envelope];
        let response = build_fetch_response(1, uid, &attrs, &restored, &flags, 500, "test.local").unwrap();
        let bytes = response.to_bytes();
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.starts_with("* 1 FETCH ("));
        assert!(text.contains(&format!("UID {uid}")));
        assert!(text.contains("FLAGS ()"));
        assert!(text.contains("ENVELOPE"));
    }

    #[test]
    fn fetch_skips_message_without_cid() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        // Insert message with no CID
        imap_store
            .insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 1713000000, 100)
            .unwrap();

        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let messages = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].message_cid.is_none());
        // The FETCH handler should skip this message — verified by the absence
        // of a panic or error when message_cid is None.
    }
```

- [ ] **Step 2: Run tests to verify they pass (pipeline test, not handler)**

Run: `cargo test -p harmony-mail fetch_round_trip_via_cas -- --nocapture`
Run: `cargo test -p harmony-mail fetch_skips_message_without_cid -- --nocapture`
Expected: both PASS (these test the pipeline components, not the async handler)

- [ ] **Step 3: Replace the FETCH stub in process_imap_async_actions**

In `crates/harmony-mail/src/server.rs`, replace the `ImapAction::FetchMessages` match arm (lines 1248-1257) with:

```rust
            ImapAction::FetchMessages {
                sequence_set,
                attributes,
                uid_mode,
            } => {
                let fetch_result: Result<(Vec<(u32, u32)>, Vec<imap_store::MessageRow>, PathBuf), String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;
                    let resolved = resolve_sequence_set(sequence_set, *uid_mode, &all_msgs);

                    let csp = content_store_path.to_path_buf();
                    Ok((resolved, all_msgs, csp))
                })();

                match fetch_result {
                    Ok((resolved, all_msgs, csp)) => {
                        for (uid, seqnum) in &resolved {
                            let msg_row = match all_msgs.iter().find(|m| m.uid == *uid) {
                                Some(r) => r,
                                None => continue,
                            };
                            let cid_bytes = match msg_row.message_cid {
                                Some(c) => c,
                                None => continue, // skip messages without CAS content
                            };

                            // Retrieve message from CAS via spawn_blocking (file I/O)
                            let csp_clone = csp.clone();
                            let cas_result = tokio::task::spawn_blocking(move || {
                                let book_store = harmony_db::store::DiskBookStore::new(&csp_clone);
                                let content_id = harmony_content::cid::ContentId::from_bytes(cid_bytes);
                                harmony_content::dag::reassemble(&content_id, &book_store)
                            })
                            .await
                            .map_err(|e| format!("CAS task panicked: {e}"));

                            let raw_bytes = match cas_result {
                                Ok(Ok(bytes)) => bytes,
                                Ok(Err(e)) => {
                                    tracing::warn!(uid = uid, error = %e, "FETCH: CAS retrieval failed, skipping message");
                                    continue;
                                }
                                Err(e) => {
                                    tracing::warn!(uid = uid, error = %e, "FETCH: CAS task failed, skipping message");
                                    continue;
                                }
                            };

                            let harmony_msg = match crate::message::HarmonyMessage::from_bytes(&raw_bytes) {
                                Ok(m) => m,
                                Err(e) => {
                                    tracing::warn!(uid = uid, error = %e, "FETCH: message deserialization failed, skipping");
                                    continue;
                                }
                            };

                            let flags = store.get_flags(msg_row.id).unwrap_or_default();
                            match crate::imap_render::build_fetch_response(
                                *seqnum, *uid, attributes, &harmony_msg, &flags,
                                msg_row.rfc822_size, domain,
                            ) {
                                Ok(response) => {
                                    writer.write_all(&response.to_bytes()).await?;
                                }
                                Err(e) => {
                                    tracing::warn!(uid = uid, error = %e, "FETCH: render failed, skipping");
                                    continue;
                                }
                            }
                        }

                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} OK FETCH completed\r\n").as_bytes())
                                .await?;
                        }
                        writer.flush().await?;
                    }
                    Err(reason) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                                .await?;
                        }
                        writer.flush().await?;
                    }
                }
            }
```

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors

- [ ] **Step 5: Run all tests**

Run: `cargo test -p harmony-mail -- --nocapture`
Expected: all tests pass (including the new ones)

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire IMAP FETCH to CAS message retrieval (ZEB-111)"
```

---

### Task 5: Wire STORE handler

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:1237-1246`

Replace the STORE stub. The state machine's `handle_store_complete` method generates the untagged FETCH responses and tagged OK — the I/O layer just needs to do the flag operations and feed results back.

- [ ] **Step 1: Write the STORE test**

Add to the `#[cfg(test)] mod tests` in `server.rs`:

```rust
    #[test]
    fn store_flags_add_and_retrieve() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        let uid = imap_store
            .insert_message("INBOX", &[3u8; MESSAGE_ID_LEN], None, 1713000000, 100)
            .unwrap();
        assert_eq!(uid, 1);

        // Get message DB id
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        let msg_id = msgs[0].id;

        // Add flags
        imap_store.add_flags(msg_id, &["\\Seen"]).unwrap();
        let flags = imap_store.get_flags(msg_id).unwrap();
        assert_eq!(flags, vec!["\\Seen"]);

        // Set flags (replaces all)
        imap_store.set_flags(msg_id, &["\\Flagged", "\\Answered"]).unwrap();
        let flags = imap_store.get_flags(msg_id).unwrap();
        assert_eq!(flags, vec!["\\Answered", "\\Flagged"]); // sorted by flag name

        // Remove flags
        imap_store.remove_flags(msg_id, &["\\Answered"]).unwrap();
        let flags = imap_store.get_flags(msg_id).unwrap();
        assert_eq!(flags, vec!["\\Flagged"]);
    }
```

- [ ] **Step 2: Run test to verify it passes**

Run: `cargo test -p harmony-mail store_flags_add_and_retrieve -- --nocapture`
Expected: PASS (this tests the store layer, which already works)

- [ ] **Step 3: Replace the STORE stub in process_imap_async_actions**

In `crates/harmony-mail/src/server.rs`, replace the `ImapAction::StoreFlags` match arm (lines 1237-1246) with:

```rust
            ImapAction::StoreFlags {
                sequence_set,
                operation,
                flags,
                uid_mode,
                silent,
            } => {
                let store_result: Result<Vec<(u32, Vec<String>)>, String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;
                    let resolved = resolve_sequence_set(sequence_set, *uid_mode, &all_msgs);

                    let flag_refs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();
                    let mut updated = Vec::new();

                    for (uid, seqnum) in &resolved {
                        let msg_row = match all_msgs.iter().find(|m| m.uid == *uid) {
                            Some(r) => r,
                            None => continue,
                        };

                        match operation {
                            StoreOperation::Set | StoreOperation::SetSilent => {
                                store.set_flags(msg_row.id, &flag_refs).map_err(|e| e.to_string())?;
                            }
                            StoreOperation::Add | StoreOperation::AddSilent => {
                                store.add_flags(msg_row.id, &flag_refs).map_err(|e| e.to_string())?;
                            }
                            StoreOperation::Remove | StoreOperation::RemoveSilent => {
                                store.remove_flags(msg_row.id, &flag_refs).map_err(|e| e.to_string())?;
                            }
                        }

                        if !silent {
                            let current_flags = store.get_flags(msg_row.id).map_err(|e| e.to_string())?;
                            updated.push((*seqnum, current_flags));
                        }
                    }

                    Ok(updated)
                })();

                match store_result {
                    Ok(updated) => {
                        let callback = session.handle(ImapEvent::StoreComplete { updated });
                        execute_imap_actions(&callback, writer).await?;
                    }
                    Err(reason) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                                .await?;
                            writer.flush().await?;
                        }
                    }
                }
            }
```

Note: this requires `use crate::imap_parse::StoreOperation;` at the top of the file or in the function scope. Add it to the existing imports.

- [ ] **Step 4: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors

- [ ] **Step 5: Run all tests**

Run: `cargo test -p harmony-mail -- --nocapture`
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire IMAP STORE to flag operations (ZEB-111)"
```

---

### Task 6: Wire user add CLI

**Files:**
- Modify: `crates/harmony-mail/src/main.rs:52-108`

Replace the CLI stub with real user creation using interactive password prompt.

- [ ] **Step 1: Update UserAction::Add to remove namespace**

In `crates/harmony-mail/src/main.rs`, update the `UserAction` enum (line 52):

```rust
#[derive(Subcommand)]
enum UserAction {
    Add {
        /// IMAP login username
        #[arg(long)]
        name: String,
        /// Hex-encoded 16-byte harmony address (32 hex chars)
        #[arg(long)]
        identity: String,
        /// Path to config file (for store_path)
        #[arg(long, default_value = "/etc/harmony-mail/config.toml")]
        config: String,
    },
}
```

- [ ] **Step 2: Implement the user add handler**

Replace the `Commands::User` match arm (lines 100-108) with:

```rust
        Commands::User { action } => match action {
            UserAction::Add {
                name,
                identity,
                config: config_path,
            } => {
                // Validate identity hex
                let addr_bytes = match hex::decode(&identity) {
                    Ok(bytes) if bytes.len() == harmony_mail::message::ADDRESS_HASH_LEN => {
                        let mut arr = [0u8; harmony_mail::message::ADDRESS_HASH_LEN];
                        arr.copy_from_slice(&bytes);
                        arr
                    }
                    Ok(bytes) => {
                        eprintln!(
                            "Error: identity must be {} hex chars ({} bytes), got {} bytes",
                            harmony_mail::message::ADDRESS_HASH_LEN * 2,
                            harmony_mail::message::ADDRESS_HASH_LEN,
                            bytes.len()
                        );
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("Error: invalid hex in --identity: {e}");
                        std::process::exit(1);
                    }
                };

                // Prompt for password
                let password = rpassword::prompt_password("Password: ").unwrap_or_else(|e| {
                    eprintln!("Error reading password: {e}");
                    std::process::exit(1);
                });
                let confirm = rpassword::prompt_password("Confirm password: ").unwrap_or_else(|e| {
                    eprintln!("Error reading password: {e}");
                    std::process::exit(1);
                });
                if password != confirm {
                    eprintln!("Error: passwords do not match");
                    std::process::exit(1);
                }
                if password.is_empty() {
                    eprintln!("Error: password cannot be empty");
                    std::process::exit(1);
                }

                // Load config and open store
                let cfg = harmony_mail::config::Config::from_file(Path::new(&config_path))
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to load config from {config_path}: {e}");
                        std::process::exit(1);
                    });
                let store = harmony_mail::imap_store::ImapStore::open(Path::new(&cfg.imap.store_path))
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to open IMAP store: {e}");
                        std::process::exit(1);
                    });

                match store.create_user(&name, &password, &addr_bytes) {
                    Ok(()) => println!("User '{name}' created successfully"),
                    Err(harmony_mail::imap_store::StoreError::UserExists(_)) => {
                        eprintln!("Error: user '{name}' already exists");
                        std::process::exit(1);
                    }
                    Err(e) => {
                        eprintln!("Error creating user: {e}");
                        std::process::exit(1);
                    }
                }
            }
        },
```

- [ ] **Step 3: Verify it compiles**

Run: `cargo check -p harmony-mail`
Expected: compiles with no errors

- [ ] **Step 4: Test the CLI help output**

Run: `cargo run -p harmony-mail -- user add --help`
Expected: shows `--name`, `--identity`, `--config` flags (no `--namespace`)

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/main.rs
git commit -m "feat(mail): wire user add CLI with password prompt (ZEB-111)"
```

---

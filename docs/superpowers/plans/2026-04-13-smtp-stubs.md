# SMTP Integration Stubs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the three SMTP stubs (CheckSpf, ResolveHarmonyAddress, DeliverToHarmony) to enable end-to-end local delivery: SMTP in -> translate -> IMAP mailbox.

**Architecture:** The I/O layer (`process_async_actions`) gains shared context: `Arc<ImapStore>` for user/message ops, `Arc<MessageAuthenticator>` for SPF, domain/config for address resolution, and a per-connection `SpfResult` for spam scoring. The IMAP store is created unconditionally in `run()` and shared between SMTP and IMAP handlers.

**Tech Stack:** Rust, tokio, mail-auth (SPF), harmony-content (CAS), harmony-db (DiskBookStore)

---

### Task 1: Add `get_user_by_address` to ImapStore

**Files:**
- Modify: `crates/harmony-mail/src/imap_store.rs`

- [ ] **Step 1: Write the test**

Add to the `tests` module in `imap_store.rs`, after the existing `get_user` test:

```rust
    #[test]
    fn get_user_by_address() {
        let store = test_store();
        let addr_a = [0xAA; ADDRESS_HASH_LEN];
        let addr_b = [0xBB; ADDRESS_HASH_LEN];
        store.create_user("alice", "pass", &addr_a).unwrap();
        store.create_user("bob", "pass", &addr_b).unwrap();

        let user = store.get_user_by_address(&addr_a).unwrap().unwrap();
        assert_eq!(user.username, "alice");

        let user = store.get_user_by_address(&addr_b).unwrap().unwrap();
        assert_eq!(user.username, "bob");

        let missing = store.get_user_by_address(&[0xCC; ADDRESS_HASH_LEN]).unwrap();
        assert!(missing.is_none());
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p harmony-mail get_user_by_address -- --nocapture 2>&1`
Expected: FAIL — method does not exist

- [ ] **Step 3: Implement `get_user_by_address`**

Add to `ImapStore` impl block, after the existing `get_user` method:

```rust
    pub fn get_user_by_address(
        &self,
        address: &[u8; ADDRESS_HASH_LEN],
    ) -> Result<Option<UserRow>, StoreError> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, username, harmony_address, created_at FROM users WHERE harmony_address = ?1",
        )?;
        let row = stmt
            .query_row([address.as_slice()], |row| {
                let harmony_blob: Vec<u8> = row.get(2)?;
                let mut harmony_address = [0u8; ADDRESS_HASH_LEN];
                if harmony_blob.len() == ADDRESS_HASH_LEN {
                    harmony_address.copy_from_slice(&harmony_blob);
                }
                Ok(UserRow {
                    id: row.get(0)?,
                    username: row.get(1)?,
                    harmony_address,
                    created_at: row.get(3)?,
                })
            })
            .optional()?;
        Ok(row)
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p harmony-mail get_user_by_address -- --nocapture 2>&1`
Expected: PASS

- [ ] **Step 5: Run full suite and commit**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 382 passed

```bash
git add crates/harmony-mail/src/imap_store.rs
git commit -m "feat(mail): add get_user_by_address for SMTP delivery (ZEB-111)"
```

---

### Task 2: Thread shared context into SMTP handlers

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

This task adds parameters to SMTP handler functions and threads them from `run()`. No behavior changes — the stubs remain stubs until Tasks 3-5.

- [ ] **Step 1: Create ImapStore unconditionally and create authenticator in `run()`**

In `run()`, move the ImapStore creation before the IMAP-enabled check so SMTP handlers can use it. Also create the MessageAuthenticator at startup. Find the `// ── IMAP setup` section (around line 291) and restructure:

```rust
    // ── Shared IMAP store (used by both SMTP delivery and IMAP) ────
    let imap_store = Arc::new(
        ImapStore::open(Path::new(&config.imap.store_path))
            .map_err(|e| format!("failed to open IMAP store: {e}"))?,
    );
    imap_store
        .initialize_default_mailboxes()
        .map_err(|e| format!("failed to init IMAP mailboxes: {e}"))?;

    // ── SPF authenticator (shared across all SMTP connections) ──────
    let mail_authenticator = match crate::auth::create_authenticator() {
        Ok(auth) => Some(Arc::new(auth)),
        Err(e) => {
            tracing::warn!(error = %e, "failed to create mail authenticator, SPF checking disabled");
            None
        }
    };

    let content_store_path = PathBuf::from(&config.imap.content_store_path);
    let reject_threshold = config.spam.reject_threshold;
    let local_domain = config.domain.name.clone();
```

The existing IMAP setup block (`if config.imap.enabled`) should then use the already-created `imap_store` instead of creating its own.

- [ ] **Step 2: Add parameters to `process_async_actions`**

Change the signature from:

```rust
async fn process_async_actions<W: AsyncWrite + Unpin>(
    actions: &[SmtpAction],
    session: &mut SmtpSession,
    writer: &mut W,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>
```

to:

```rust
async fn process_async_actions<W: AsyncWrite + Unpin>(
    actions: &[SmtpAction],
    session: &mut SmtpSession,
    writer: &mut W,
    imap_store: &Arc<ImapStore>,
    authenticator: &Option<Arc<mail_auth::MessageAuthenticator>>,
    local_domain: &str,
    content_store_path: &Path,
    spf_result: &mut crate::spam::SpfResult,
    reject_threshold: i32,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>
```

- [ ] **Step 3: Add parameters to `handle_connection` and `handle_connection_generic`**

Add to both functions: `imap_store: Arc<ImapStore>`, `authenticator: Option<Arc<mail_auth::MessageAuthenticator>>`, `local_domain: String`, `content_store_path: PathBuf`, `reject_threshold: i32`.

In both functions, add a per-connection SPF result variable:

```rust
let mut spf_result = crate::spam::SpfResult::None;
```

Update the `process_async_actions` call sites to pass the new parameters.

In `handle_connection`, when it calls `handle_connection_generic` for post-STARTTLS, pass the parameters through.

- [ ] **Step 4: Update all spawn sites in `run()` to pass the new parameters**

There are 3 SMTP spawn sites:
1. Port 465 implicit TLS (`handle_connection_generic`) — around line 242
2. Port 587 STARTTLS (`handle_connection`) — around line 280
3. Port 25 main loop (`handle_connection`) — around line 454

Each needs clones of `imap_store`, `mail_authenticator`, `local_domain`, `content_store_path`, `reject_threshold`.

- [ ] **Step 5: Update existing SMTP tests**

The two existing SMTP tests (`smtp_handshake_ehlo_quit` and `smtp_full_mail_transaction`) call `handle_connection` directly. Add the new parameters with sensible defaults:

```rust
// Create a minimal store for SMTP tests
let smtp_test_dir = tempfile::tempdir().unwrap();
let smtp_store = Arc::new(
    crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db")).unwrap()
);
smtp_store.initialize_default_mailboxes().unwrap();
```

Pass `smtp_store`, `None` (no authenticator), `"test.example.com"`, `smtp_test_dir.path().to_path_buf()`, `5` (reject threshold).

- [ ] **Step 6: Run tests and commit**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 382 passed

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "refactor(mail): thread shared context into SMTP handlers (ZEB-111)"
```

---

### Task 3: Wire CheckSpf

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Implement CheckSpf handler**

Replace the `SmtpAction::CheckSpf` stub in `process_async_actions`:

```rust
            SmtpAction::CheckSpf { sender_domain, peer_ip } => {
                if let Some(ref auth) = authenticator {
                    match auth.verify_spf()
                        .helo_domain(sender_domain)
                        .ip(*peer_ip)
                        .mail_from(sender_domain)
                        .check_host()
                        .await
                    {
                        Ok(output) => {
                            *spf_result = crate::auth::map_spf_result(&output);
                            tracing::debug!(
                                %sender_domain,
                                result = ?spf_result,
                                "SPF check complete"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                %sender_domain,
                                error = %e,
                                "SPF check failed, defaulting to None"
                            );
                            *spf_result = crate::spam::SpfResult::None;
                        }
                    }
                }
            }
```

Note: The exact `mail-auth` v0.7 API may differ slightly. The implementer should check the actual API. The key pattern is: call the SPF verifier with domain + IP, map the result, store in `spf_result`. If mail-auth uses a different builder pattern, adapt accordingly.

- [ ] **Step 2: Run tests and commit**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 382 passed

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire CheckSpf to mail-auth SPF resolver (ZEB-111)"
```

---

### Task 4: Wire ResolveHarmonyAddress

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Implement ResolveHarmonyAddress handler**

Replace the stub in `process_async_actions`:

```rust
            SmtpAction::ResolveHarmonyAddress { local_part, domain } => {
                use crate::address::{self, LocalPart};
                use crate::message::ADDRESS_HASH_LEN;

                let identity = if domain.eq_ignore_ascii_case(local_domain) {
                    // Local domain: resolve against our user store
                    match address::parse_local_part(local_part) {
                        LocalPart::Hex(hash) => Some(hash),
                        LocalPart::Named { name, .. } => {
                            match imap_store.get_user(&name) {
                                Ok(Some(user)) => Some(user.harmony_address),
                                Ok(None) => None,
                                Err(e) => {
                                    tracing::warn!(name = %name, error = %e, "user lookup failed");
                                    None
                                }
                            }
                        }
                        LocalPart::Alias(alias) => {
                            match imap_store.get_user(&alias) {
                                Ok(Some(user)) => Some(user.harmony_address),
                                Ok(None) => None,
                                Err(e) => {
                                    tracing::warn!(alias = %alias, error = %e, "alias lookup failed");
                                    None
                                }
                            }
                        }
                    }
                } else {
                    // Non-local domain: remote delivery not yet supported (ZEB-113)
                    tracing::debug!(
                        %domain,
                        %local_domain,
                        "non-local domain, rejecting (remote delivery tracked in ZEB-113)"
                    );
                    None
                };

                let callback_actions = session.handle(SmtpEvent::HarmonyResolved {
                    local_part: local_part.clone(),
                    identity,
                });
                execute_actions_generic(&callback_actions, writer).await?;
                for a in &callback_actions {
                    if matches!(
                        a,
                        SmtpAction::ResolveHarmonyAddress { .. }
                            | SmtpAction::DeliverToHarmony { .. }
                            | SmtpAction::CheckSpf { .. }
                    ) {
                        tracing::warn!(
                            "nested async action from HarmonyResolved callback dropped: {a:?}"
                        );
                    }
                }
            }
```

- [ ] **Step 2: Run tests and commit**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 382 passed

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire ResolveHarmonyAddress to local user store (ZEB-111)"
```

---

### Task 5: Wire DeliverToHarmony

**Files:**
- Modify: `crates/harmony-mail/src/server.rs`

- [ ] **Step 1: Implement DeliverToHarmony handler**

Replace the stub in `process_async_actions`:

```rust
            SmtpAction::DeliverToHarmony { recipients, data } => {
                use crate::message::ADDRESS_HASH_LEN;

                // Phase 1: Spam scoring
                let signals = crate::spam::SpamSignals {
                    dnsbl_listed: false,
                    fcrdns_pass: true,
                    spf_result: spf_result.clone(),
                    dkim_result: crate::spam::DkimResult::Missing,
                    dmarc_result: crate::spam::DmarcResult::None,
                    has_executable_attachment: false,
                    url_count: 0,
                    empty_subject: false,
                    known_harmony_sender: false,
                    gateway_trust: None,
                    first_contact: true,
                };
                let verdict = crate::spam::score(&signals, reject_threshold);
                if matches!(verdict.action, crate::spam::SpamAction::Reject) {
                    tracing::info!(
                        score = verdict.score,
                        "message rejected by spam filter"
                    );
                    let callback_actions =
                        session.handle(SmtpEvent::DeliveryResult { success: false });
                    execute_actions_generic(&callback_actions, writer).await?;
                    continue;
                }

                // Phase 2: Translate RFC 5322 -> HarmonyMessage
                let translated = match crate::translate::translate_inbound(data) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!(error = %e, "message translation failed");
                        let callback_actions =
                            session.handle(SmtpEvent::DeliveryResult { success: false });
                        execute_actions_generic(&callback_actions, writer).await?;
                        continue;
                    }
                };

                // Phase 3: Store message in CAS
                let message_cid = match translated.message.to_bytes() {
                    Ok(msg_bytes) => {
                        let csp = content_store_path.to_path_buf();
                        let cid_result = tokio::task::spawn_blocking(move || {
                            let mut book_store = harmony_db::DiskBookStore::new(&csp);
                            harmony_content::dag::ingest(
                                &msg_bytes,
                                &harmony_content::chunker::ChunkerConfig::default(),
                                &mut book_store,
                            )
                        })
                        .await
                        .map_err(|e| format!("CAS task panicked: {e}"))?;
                        match cid_result {
                            Ok(cid) => Some(cid.to_bytes()),
                            Err(e) => {
                                tracing::warn!(error = %e, "CAS storage failed, delivering without CID");
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "message serialization failed, delivering without CID");
                        None
                    }
                };

                // Phase 4: Deliver to local IMAP mailboxes
                let mut delivered_count = 0u32;
                let timestamp = translated.message.timestamp;
                let rfc822_size = data.len() as u32;
                let msg_id = translated.message.message_id;

                for recipient_hash in recipients {
                    match imap_store.get_user_by_address(recipient_hash) {
                        Ok(Some(_user)) => {
                            let cid_ref = message_cid.as_ref();
                            match imap_store.insert_message(
                                "INBOX",
                                &msg_id,
                                cid_ref,
                                timestamp,
                                rfc822_size,
                            ) {
                                Ok(_uid) => {
                                    delivered_count += 1;
                                    tracing::debug!(
                                        recipient = hex::encode(recipient_hash),
                                        "delivered to local INBOX"
                                    );
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        recipient = hex::encode(recipient_hash),
                                        error = %e,
                                        "failed to insert message into INBOX"
                                    );
                                }
                            }
                        }
                        Ok(None) => {
                            tracing::debug!(
                                recipient = hex::encode(recipient_hash),
                                "no local user for recipient (remote delivery not yet supported)"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                recipient = hex::encode(recipient_hash),
                                error = %e,
                                "user lookup failed"
                            );
                        }
                    }
                }

                let success = delivered_count > 0;
                let callback_actions =
                    session.handle(SmtpEvent::DeliveryResult { success });
                execute_actions_generic(&callback_actions, writer).await?;
                for a in &callback_actions {
                    if matches!(
                        a,
                        SmtpAction::ResolveHarmonyAddress { .. }
                            | SmtpAction::DeliverToHarmony { .. }
                            | SmtpAction::CheckSpf { .. }
                    ) {
                        tracing::warn!(
                            "nested async action from DeliveryResult callback dropped: {a:?}"
                        );
                    }
                }
            }
```

- [ ] **Step 2: Add necessary imports**

Ensure `harmony_content` and `harmony_db` are available. Check `Cargo.toml` — `harmony-content` and `harmony-db` should already be dependencies (they were added for IMAP FETCH). If not, add them.

- [ ] **Step 3: Run tests and commit**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 382 passed

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire DeliverToHarmony for local IMAP delivery (ZEB-111)"
```

---

### Task 6: Integration test — SMTP to IMAP delivery

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` (test module)

- [ ] **Step 1: Write the end-to-end delivery test**

Add to the `mod tests` block in server.rs:

```rust
    #[tokio::test]
    async fn smtp_delivers_to_local_imap() {
        // Set up IMAP store with a test user
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let store = crate::imap_store::ImapStore::open(&db_path).unwrap();
        store.initialize_default_mailboxes().unwrap();
        let addr = [0xAAu8; crate::message::ADDRESS_HASH_LEN];
        store.create_user("alice", "pass", &addr).unwrap();
        let store = std::sync::Arc::new(store);

        let config = test_config();
        let max_message_size = parse_message_size(&config.spam.max_message_size);
        let smtp_config = SmtpConfig {
            domain: config.domain.name.clone(),
            mx_host: config.domain.mx_host.clone(),
            max_message_size,
            max_recipients: 100,
            tls_available: false,
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr_listen = listener.local_addr().unwrap();
        let csp = dir.path().join("content");
        let domain = config.domain.name.clone();

        let server_store = Arc::clone(&store);
        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(
                stream,
                peer_addr.ip(),
                smtp_config,
                max_message_size,
                None,
                server_store,
                None, // no authenticator
                domain,
                csp,
                5,
            )
            .await
            .unwrap();
        });

        let stream = TcpStream::connect(addr_listen).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Read 220 greeting
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        assert!(line.starts_with("220 "), "greeting: {line}");

        // EHLO
        line.clear();
        write_half.write_all(b"EHLO test.com\r\n").await.unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(ehlo_resp.contains("250"), "EHLO: {ehlo_resp}");

        // MAIL FROM
        write_half.write_all(b"MAIL FROM:<sender@test.com>\r\n").await.unwrap();
        let mail_resp = read_smtp_response(&mut reader).await;
        assert!(mail_resp.contains("250"), "MAIL FROM: {mail_resp}");

        // RCPT TO — use alice's name at the test domain
        write_half
            .write_all(b"RCPT TO:<alice@test.example.com>\r\n")
            .await
            .unwrap();
        let rcpt_resp = read_smtp_response(&mut reader).await;
        assert!(rcpt_resp.contains("250"), "RCPT TO: {rcpt_resp}");

        // DATA
        write_half.write_all(b"DATA\r\n").await.unwrap();
        let data_resp = read_smtp_response(&mut reader).await;
        assert!(data_resp.starts_with("354"), "DATA: {data_resp}");

        // Send message body
        write_half
            .write_all(b"From: sender@test.com\r\nTo: alice@test.example.com\r\nSubject: Test Delivery\r\n\r\nHello from SMTP!\r\n.\r\n")
            .await
            .unwrap();
        let deliver_resp = read_smtp_response(&mut reader).await;
        assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

        // QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "QUIT: {quit_resp}");

        server_handle.await.unwrap();

        // Verify message landed in alice's INBOX
        let mbox = store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = store.get_messages(mbox.id).unwrap();
        assert_eq!(msgs.len(), 1, "expected 1 message in INBOX, got {}", msgs.len());
        assert!(msgs[0].rfc822_size > 0);
    }
```

Note: The exact `handle_connection` parameters depend on how Task 2 structured them. The implementer should match the actual signature. The key assertions are: RCPT TO succeeds (250), DATA delivery succeeds (250), and the message appears in the IMAP INBOX.

- [ ] **Step 2: Run the test**

Run: `cargo test -p harmony-mail smtp_delivers_to_local_imap -- --nocapture 2>&1`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: 383 passed

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "test(mail): integration test for SMTP-to-IMAP local delivery (ZEB-111)"
```

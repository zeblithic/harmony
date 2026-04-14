# IMAP STARTTLS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire IMAP STARTTLS on port 143 so plaintext connections can upgrade to TLS.

**Architecture:** Signal-and-break pattern mirroring SMTP. `process_imap_async_actions` returns a bool signaling STARTTLS requested. `handle_imap_connection_starttls` owns the pre-TLS loop, breaks on signal, performs TLS handshake, then delegates to the generic `handle_imap_connection` with the TLS stream.

**Tech Stack:** Rust, tokio, tokio-rustls, tokio-util (FramedRead)

---

### Task 1: Change `process_imap_async_actions` return type to signal STARTTLS

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:997-1719`

- [ ] **Step 1: Change the function signature**

Change `process_imap_async_actions` return type from `Result<(), ...>` to `Result<bool, ...>`, and update the `ImapAction::StartTls` arm to return `Ok(true)` immediately. The function must return `Ok(false)` at the end.

In `crates/harmony-mail/src/server.rs`, change the signature (line ~1006):

```rust
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>
```

Change the `ImapAction::StartTls` arm (line ~1707):

```rust
            ImapAction::StartTls => {
                return Ok(true);
            }
```

Change the final return (line ~1718):

```rust
    Ok(false)
```

- [ ] **Step 2: Update the call site in `handle_imap_connection`**

In `handle_imap_connection` (line ~880), the call to `process_imap_async_actions` currently returns `Result<()>`. Update it to discard the bool:

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

This call site stays the same — the `?` still works because `Result<bool, E>` converts fine, and the bool is discarded. No code change needed here, just verify it compiles.

- [ ] **Step 3: Run tests to verify nothing broke**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: `test result: ok. 380 passed`

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "refactor(mail): process_imap_async_actions returns STARTTLS signal (ZEB-111)"
```

---

### Task 2: Enable STARTTLS advertisement on port 143

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:402-407`

- [ ] **Step 1: Change `tls_available` in port 143 listener**

In the port 143 listener setup (line ~402), change:

```rust
                                    let imap_cfg = ImapConfig {
                                        domain: d,
                                        tls_active: false,
                                        tls_available: acc.is_some(),
                                        max_auth_failures: imap_max_auth_failures,
                                    };
```

The previous code had `tls_available: false` with a comment about STARTTLS not being wired. Remove that comment since we're wiring it now.

- [ ] **Step 2: Run tests**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: `test result: ok. 380 passed`

- [ ] **Step 3: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): advertise STARTTLS on port 143 when TLS available (ZEB-111)"
```

---

### Task 3: Implement `handle_imap_connection_starttls`

**Files:**
- Modify: `crates/harmony-mail/src/server.rs:932-950`

- [ ] **Step 1: Replace the stub with the full implementation**

Replace the entire `handle_imap_connection_starttls` function (lines ~932-950) with:

```rust
/// Handle an IMAP connection on port 143 (plaintext with optional STARTTLS).
async fn handle_imap_connection_starttls(
    reader: tokio::net::tcp::OwnedReadHalf,
    mut writer: tokio::net::tcp::OwnedWriteHalf,
    config: ImapConfig,
    store: Arc<ImapStore>,
    idle_timeout: Duration,
    tls_acceptor: Option<Arc<tokio_rustls::TlsAcceptor>>,
    content_store_path: PathBuf,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let domain = config.domain.clone();
    let mut session = ImapSession::new(config);
    let codec = ImapCodec::new();

    // Send greeting
    let actions = session.handle(ImapEvent::Connected);
    execute_imap_actions(&actions, &mut writer).await?;

    let mut framed = FramedRead::new(reader, codec);

    loop {
        let timeout_dur = match session.state {
            ImapState::Idling { .. } => idle_timeout,
            _ => IMAP_COMMAND_TIMEOUT,
        };

        let frame = tokio::time::timeout(timeout_dur, framed.next()).await;

        match frame {
            Ok(Some(Ok(imap_frame))) => match imap_frame {
                ImapFrame::CommandLine(line) => match imap_parse::parse_command(&line) {
                    Ok(cmd) => {
                        let actions = session.handle(ImapEvent::Command(cmd));
                        let should_close = execute_imap_actions(&actions, &mut writer).await?;
                        let needs_starttls = process_imap_async_actions(
                            &actions,
                            &mut session,
                            &mut writer,
                            &store,
                            &mut framed,
                            &content_store_path,
                            &domain,
                        )
                        .await?;

                        if needs_starttls {
                            break;
                        }
                        if should_close || session.state == ImapState::Logout {
                            let _ = writer.shutdown().await;
                            return Ok(());
                        }
                    }
                    Err(e) => {
                        let response = if let Some(tag) = e.tag() {
                            format!("{tag} BAD {e}\r\n")
                        } else {
                            format!("* BAD {e}\r\n")
                        };
                        writer.write_all(response.as_bytes()).await?;
                        writer.flush().await?;
                    }
                },
                ImapFrame::NeedsContinuation { .. } => {
                    writer.write_all(b"+ Ready\r\n").await?;
                    writer.flush().await?;
                    framed.decoder_mut().acknowledge_continuation();
                }
                ImapFrame::Done => {
                    let actions = session.handle(ImapEvent::IdleDone);
                    execute_imap_actions(&actions, &mut writer).await?;
                    framed.decoder_mut().exit_idle_mode();
                }
            },
            Ok(Some(Err(e))) => {
                tracing::debug!(error = %e, "IMAP codec error");
                writer.write_all(b"* BYE protocol error\r\n").await?;
                let _ = writer.shutdown().await;
                return Ok(());
            }
            Ok(None) => {
                let _ = writer.shutdown().await;
                return Ok(());
            }
            Err(_) => {
                writer.write_all(b"* BYE connection timed out\r\n").await?;
                let _ = writer.shutdown().await;
                return Ok(());
            }
        }
    }

    // ── STARTTLS upgrade ──────────────────────────────────────────
    let acceptor = match tls_acceptor {
        Some(ref acc) => acc,
        None => {
            writer.write_all(b"* BYE TLS not available\r\n").await?;
            let _ = writer.shutdown().await;
            return Ok(());
        }
    };

    writer.flush().await?;

    let parts = framed.into_parts();
    let tcp_reader = parts.io;

    if !parts.read_buf.is_empty() {
        tracing::warn!(
            buffered_bytes = parts.read_buf.len(),
            "IMAP STARTTLS: rejecting — client sent data before OK response"
        );
        let mut stream = tcp_reader
            .reunite(writer)
            .map_err(|e| format!("failed to reunite: {e}"))?;
        let _ = stream.write_all(b"* BYE STARTTLS protocol error\r\n").await;
        return Ok(());
    }

    let tcp_stream = tcp_reader
        .reunite(writer)
        .map_err(|e| format!("failed to reunite TCP halves: {e}"))?;

    match tls::starttls_upgrade(tcp_stream, acceptor).await {
        Ok(tls_stream) => {
            let (tls_reader, tls_writer) = tokio::io::split(tls_stream);
            // Feed TlsCompleted to state machine
            let tls_actions = session.handle(ImapEvent::TlsCompleted);
            let mut tls_writer = tls_writer;
            execute_imap_actions(&tls_actions, &mut tls_writer).await?;
            // Continue the session over TLS
            handle_imap_connection(
                tls_reader,
                tls_writer,
                session.config.clone(),
                store,
                idle_timeout,
                content_store_path,
            )
            .await
        }
        Err(e) => {
            tracing::warn!(error = %e, "IMAP STARTTLS handshake failed");
            Ok(())
        }
    }
}
```

- [ ] **Step 2: Update the port 143 listener call site**

The port 143 listener (line ~401) currently calls `handle_imap_connection_starttls(reader, writer, ...)`. Since the function signature changed from generic `R, W` to concrete `OwnedReadHalf, OwnedWriteHalf`, the call site needs to use `stream.into_split()` instead of `tokio::io::split(stream)`.

Change the spawn block (around line ~400):

```rust
                                let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                                let (reader, writer) = stream.into_split();
```

The previous code used `tokio::io::split(stream)`. Change to `stream.into_split()` which returns `(OwnedReadHalf, OwnedWriteHalf)` — the concrete types needed for `reunite()`.

- [ ] **Step 3: Run tests**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: `test result: ok. 380 passed`

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-mail/src/server.rs
git commit -m "feat(mail): wire IMAP STARTTLS upgrade on port 143 (ZEB-111)"
```

---

### Task 4: Integration test for IMAP STARTTLS

**Files:**
- Modify: `crates/harmony-mail/src/server.rs` (test module at end of file)

- [ ] **Step 1: Write the integration test**

Add this test to the `mod tests` block at the end of `server.rs` (after the existing `smtp_full_mail_transaction` test):

```rust
    #[tokio::test]
    async fn imap_starttls_upgrade() {
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

        // Generate self-signed cert
        let rcgen::CertifiedKey { cert, key_pair } =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        let cert_pem = cert.pem();
        let key_pem = key_pair.serialize_pem();

        let mut cert_file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut cert_file, cert_pem.as_bytes()).unwrap();
        std::io::Write::flush(&mut cert_file).unwrap();
        let mut key_file = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut key_file, key_pem.as_bytes()).unwrap();
        std::io::Write::flush(&mut key_file).unwrap();

        let acceptor = tls::load_tls_config(cert_file.path(), key_file.path()).unwrap();
        let acceptor = std::sync::Arc::new(acceptor);

        // Set up IMAP store
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let store = crate::imap_store::ImapStore::open(&db_path).unwrap();
        store.initialize_default_mailboxes().unwrap();
        store.create_user("testuser", "testpass", &[0xAAu8; crate::message::ADDRESS_HASH_LEN]).unwrap();
        let store = std::sync::Arc::new(store);

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let csp = dir.path().join("content");

        let server_acceptor = acceptor.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let (reader, writer) = stream.into_split();
            let config = crate::imap::ImapConfig {
                domain: "localhost".to_string(),
                tls_active: false,
                tls_available: true,
                max_auth_failures: 3,
            };
            let _ = handle_imap_connection_starttls(
                reader,
                writer,
                config,
                store,
                Duration::from_secs(30),
                Some(server_acceptor),
                csp,
            )
            .await;
        });

        // Client: connect plaintext
        let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();
        let (read_half, mut write_half) = tcp.into_split();
        let mut reader = BufReader::new(read_half);

        // Read greeting
        let mut line = String::new();
        reader.read_line(&mut line).await.unwrap();
        assert!(line.contains("OK"), "expected greeting, got: {line}");

        // Send CAPABILITY — should advertise STARTTLS
        line.clear();
        write_half.write_all(b"a1 CAPABILITY\r\n").await.unwrap();
        // Read untagged capability
        reader.read_line(&mut line).await.unwrap();
        assert!(line.contains("STARTTLS"), "expected STARTTLS in capabilities: {line}");
        // Read tagged OK
        let mut ok_line = String::new();
        reader.read_line(&mut ok_line).await.unwrap();
        assert!(ok_line.contains("a1 OK"), "expected tagged OK: {ok_line}");

        // Send STARTTLS
        line.clear();
        write_half.write_all(b"a2 STARTTLS\r\n").await.unwrap();
        reader.read_line(&mut line).await.unwrap();
        assert!(line.contains("a2 OK"), "expected STARTTLS OK: {line}");

        // Perform TLS handshake
        let tcp_stream = reader.into_inner().reunite(write_half).unwrap();
        let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
        let mut root_store = rustls::RootCertStore::empty();
        let certs = tls::load_certs(cert_file.path()).unwrap();
        for c in &certs {
            root_store.add(c.clone()).unwrap();
        }
        let client_config = rustls::ClientConfig::builder()
            .with_root_certificates(root_store)
            .with_no_client_auth();
        let connector = tokio_rustls::TlsConnector::from(std::sync::Arc::new(client_config));
        let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();
        let tls_stream = connector.connect(server_name, tcp_stream).await.unwrap();
        let (tls_read, mut tls_write) = tokio::io::split(tls_stream);
        let mut tls_reader = BufReader::new(tls_read);

        // Post-TLS: LOGIN should work
        line.clear();
        tls_write.write_all(b"a3 LOGIN testuser testpass\r\n").await.unwrap();
        tls_reader.read_line(&mut line).await.unwrap();
        assert!(line.contains("a3 OK"), "expected LOGIN OK: {line}");

        // QUIT
        tls_write.write_all(b"a4 LOGOUT\r\n").await.unwrap();
        line.clear();
        tls_reader.read_line(&mut line).await.unwrap(); // * BYE
        line.clear();
        tls_reader.read_line(&mut line).await.unwrap(); // a4 OK

        server.await.unwrap();
    }
```

- [ ] **Step 2: Make `tls::load_certs` accessible from tests**

The test calls `tls::load_certs` to build the client trust store. Check if `load_certs` is `pub` or `pub(crate)`. If it's private, change its visibility in `tls.rs` to `pub(crate)`:

```rust
pub(crate) fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, TlsError> {
```

- [ ] **Step 3: Run the test**

Run: `cargo test -p harmony-mail imap_starttls_upgrade -- --nocapture 2>&1`
Expected: PASS

- [ ] **Step 4: Run full test suite**

Run: `cargo test -p harmony-mail 2>&1 | grep "test result"`
Expected: `test result: ok. 381 passed`

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-mail/src/server.rs crates/harmony-mail/src/tls.rs
git commit -m "test(mail): integration test for IMAP STARTTLS upgrade (ZEB-111)"
```

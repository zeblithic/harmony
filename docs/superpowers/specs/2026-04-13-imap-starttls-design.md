# IMAP STARTTLS on Port 143 (ZEB-111 Item #4)

## Overview

Wire IMAP STARTTLS on port 143 so plaintext connections can upgrade to TLS. The IMAP state machine already has full STARTTLS support (states, events, actions, capability advertisement). This spec covers the I/O layer wiring that replaces the stub.

## Scope

- STARTTLS on port 143 only. Port 993 stays implicit-TLS (no STARTTLS advertised).
- Mirror the existing SMTP STARTTLS pattern (signal-and-break).

## Design Decisions

- **Signal-and-break pattern** — `process_imap_async_actions` returns a `needs_starttls` bool. The pre-TLS loop breaks, extracts the TCP stream, performs TLS handshake, then delegates to the generic `handle_imap_connection` with the TLS stream. This matches the proven SMTP STARTTLS flow and keeps `handle_imap_connection` untouched.
- **Concrete TCP types for port 143** — `handle_imap_connection_starttls` takes concrete `OwnedReadHalf`/`OwnedWriteHalf` instead of generic `R: AsyncRead` since TCP reunification is required for the TLS handshake. The generic handler continues to accept any `AsyncRead`/`AsyncWrite` for port 993 and post-upgrade streams.
- **Buffered bytes rejection** — Per RFC 3207 §4 (applied to IMAP by principle), if the client sends data before the server's tagged OK for STARTTLS, the connection is rejected. Prevents corrupted TLS handshakes from misbehaving clients.

## Data Flow

```text
Port 143 connection:
1. handle_imap_connection_starttls(reader, writer, config, ..., acceptor)
2. Send greeting, create FramedRead<OwnedReadHalf, ImapCodec>
3. Pre-TLS command loop:
   a. Read frame via FramedRead
   b. Dispatch to session.handle(ImapEvent::Command(cmd))
   c. execute_imap_actions (writes tagged OK for STARTTLS)
   d. process_imap_async_actions → returns needs_starttls=true
   e. Break out of loop
4. TLS upgrade:
   a. writer.flush()
   b. framed.into_parts() → extract OwnedReadHalf + validate read_buf empty
   c. reader.reunite(writer) → TcpStream
   d. tls::starttls_upgrade(tcp_stream, acceptor) → TlsStream
   e. tokio::io::split(tls_stream) → (tls_reader, tls_writer)
   f. session.handle(ImapEvent::TlsCompleted) → sets tls_active=true
   g. execute_imap_actions (no actions expected from TlsCompleted)
5. Delegate: handle_imap_connection(tls_reader, tls_writer, session.config, ...)
```

## Changes

### Modified: `crates/harmony-mail/src/server.rs`

**`process_imap_async_actions`** — Change return type from `Result<(), ...>` to `Result<bool, ...>` where `true` = STARTTLS requested. The `ImapAction::StartTls` arm returns `Ok(true)` instead of logging a debug stub. All other paths return `Ok(false)`.

**`handle_imap_connection_starttls`** — Replace stub with full implementation:
- Concrete signature: `(reader: OwnedReadHalf, writer: OwnedWriteHalf, config: ImapConfig, store: Arc<ImapStore>, idle_timeout: Duration, tls_acceptor: Option<Arc<TlsAcceptor>>, content_store_path: PathBuf)`
- Pre-TLS command loop: greeting → FramedRead → command dispatch (same structure as `handle_imap_connection`)
- STARTTLS break-out: flush, into_parts, validate, reunite, upgrade, split, TlsCompleted, delegate
- Error paths: `* BYE TLS handshake failed` on handshake error, `* BYE STARTTLS protocol error` on buffered bytes

**Port 143 listener setup** — Change `tls_available: false` to `tls_available: acc.is_some()` (where `acc` is the cloned TLS acceptor). This enables STARTTLS advertisement when TLS certs are configured.

**Callers of `process_imap_async_actions`** — `handle_imap_connection` also calls this function. Its call site needs to accept the new `Result<bool>` return type but discards the bool (STARTTLS is impossible in post-upgrade/port-993 contexts since `tls_active=true`). The return value is only acted upon in `handle_imap_connection_starttls`.

### No New Files

All changes are in the existing server.rs file.

## Error Handling

| Condition | Response |
|-----------|----------|
| Client sends STARTTLS but no TLS acceptor | State machine returns `BAD` (tls_available=false) |
| TLS handshake fails | `* BYE TLS handshake failed\r\n` + close |
| Buffered bytes before handshake | `* BYE STARTTLS protocol error\r\n` + close |
| Client sends STARTTLS when already TLS | State machine returns `BAD` (tls_active=true) |

## Testing

### Existing Coverage (no changes needed)

Five state machine tests already validate STARTTLS transitions:
- `starttls_transitions_to_negotiating`
- `starttls_rejected_when_tls_active`
- `starttls_rejected_when_tls_unavailable`
- `tls_completed_returns_to_not_authenticated`
- `commands_during_idle_rejected`

### New: Integration test

`imap_starttls_upgrade` — connect to port 143 plaintext, send CAPABILITY (verify STARTTLS advertised), send STARTTLS command, perform TLS handshake using self-signed cert, send CAPABILITY again (verify STARTTLS no longer advertised, LOGINDISABLED removed), send LOGIN, verify authentication succeeds. Mirrors the existing `tls::tests::starttls_upgrade_with_self_signed` test pattern.

//! SMTP server: TCP listener, connection handling, and state machine I/O driver.
//!
//! Binds to configured SMTP ports, accepts connections, and spawns per-connection
//! async tasks that wire [`SmtpCodec`](crate::io::SmtpCodec) frames to the
//! [`SmtpSession`](crate::smtp::SmtpSession) sans-I/O state machine.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt};
use tokio::net::TcpListener;
use tokio_util::codec::FramedRead;

use futures_util::StreamExt;

use crate::config::{Config, TlsMode};
use crate::imap::{
    ImapAction, ImapConfig, ImapEvent, ImapSession, ImapState, MailboxSnapshot, ResponseStatus,
};
use crate::imap_codec::{ImapCodec, ImapFrame};
use crate::imap_parse;
use crate::imap_parse::StoreOperation;
use crate::imap_store::{self, ImapStore};
use crate::io::{SmtpCodec, SmtpFrame};
use crate::smtp::{SmtpAction, SmtpConfig, SmtpEvent, SmtpSession, SmtpState};
use crate::smtp_parse::parse_command;
use crate::tls;

/// Per-IP connection tracking entry.
struct IpEntry {
    active: usize,
    last_seen: Instant,
}

/// RAII guard that calls `SharedState::disconnect` on drop.
/// Ensures the per-IP counter is decremented even if the connection task panics.
struct ConnectionGuard {
    shared: Arc<SharedState>,
    ip: IpAddr,
}

impl Drop for ConnectionGuard {
    fn drop(&mut self) {
        self.shared.disconnect(self.ip);
    }
}

/// Shared state across all connections.
struct SharedState {
    connections: Mutex<HashMap<IpAddr, IpEntry>>,
    max_connections_per_ip: usize,
}

impl SharedState {
    fn new(max_connections_per_ip: usize) -> Self {
        Self {
            connections: Mutex::new(HashMap::new()),
            max_connections_per_ip,
        }
    }

    /// Try to register a new connection. Returns false if limit exceeded.
    fn try_connect(&self, ip: IpAddr) -> bool {
        let mut conns = self.connections.lock().unwrap();
        let entry = conns.entry(ip).or_insert(IpEntry {
            active: 0,
            last_seen: Instant::now(),
        });
        if entry.active >= self.max_connections_per_ip {
            return false;
        }
        entry.active += 1;
        entry.last_seen = Instant::now();
        true
    }

    /// Unregister a connection.
    fn disconnect(&self, ip: IpAddr) {
        let mut conns = self.connections.lock().unwrap();
        if let Some(entry) = conns.get_mut(&ip) {
            entry.active = entry.active.saturating_sub(1);
            if entry.active == 0 {
                conns.remove(&ip);
            }
        }
    }

    /// Remove stale entries older than the given duration.
    fn cleanup(&self, max_age: Duration) {
        let mut conns = self.connections.lock().unwrap();
        let now = Instant::now();
        conns.retain(|_, entry| entry.active > 0 || now.duration_since(entry.last_seen) < max_age);
    }
}

/// Parse the max_message_size config string (e.g. "25MB") to bytes.
fn parse_message_size(s: &str) -> usize {
    const DEFAULT_SIZE: usize = 25 * 1024 * 1024; // 25 MB

    let s = s.trim();
    let (num_str, multiplier) = if let Some(n) = s.strip_suffix("MB") {
        (n, 1024 * 1024)
    } else if let Some(n) = s.strip_suffix("KB") {
        (n, 1024)
    } else if let Some(n) = s.strip_suffix("GB") {
        (n, 1024 * 1024 * 1024)
    } else {
        (s, 1)
    };
    num_str
        .trim()
        .parse::<usize>()
        .ok()
        .and_then(|n| n.checked_mul(multiplier))
        .unwrap_or(DEFAULT_SIZE)
}

/// Idle timeout for SMTP command reads.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(300); // 5 minutes
/// Timeout for DATA phase reads.
const DATA_TIMEOUT: Duration = Duration::from_secs(600); // 10 minutes
/// Interval for cleaning up stale per-IP entries.
const CLEANUP_INTERVAL: Duration = Duration::from_secs(60);

/// Run the SMTP server with the given configuration.
pub async fn run(config: Config) -> Result<(), Box<dyn std::error::Error>> {
    let max_message_size = parse_message_size(&config.spam.max_message_size);

    let shared = Arc::new(SharedState::new(config.spam.max_connections_per_ip));

    // Load TLS if configured with manual certs
    let tls_acceptor = if config.tls.mode == TlsMode::Manual {
        match (&config.tls.cert, &config.tls.key) {
            (Some(cert), Some(key)) => {
                let acceptor = tls::load_tls_config(Path::new(cert), Path::new(key))?;
                tracing::info!("TLS loaded from cert={cert}, key={key}");
                Some(Arc::new(acceptor))
            }
            _ => {
                tracing::warn!("TLS mode=manual but cert/key not configured");
                None
            }
        }
    } else {
        tracing::info!("TLS mode=acme (not yet implemented), STARTTLS unavailable");
        None
    };

    let smtp_config = SmtpConfig {
        domain: config.domain.name.clone(),
        mx_host: config.domain.mx_host.clone(),
        max_message_size,
        max_recipients: 100,
        tls_available: tls_acceptor.is_some(),
    };

    // Bind port 25 (inbound SMTP, STARTTLS)
    let smtp_listener = TcpListener::bind(&config.gateway.listen_smtp).await?;
    tracing::info!(addr = %config.gateway.listen_smtp, "SMTP listener started (port 25)");

    // Bind port 465 (implicit TLS) if TLS is available
    let submission_listener = if tls_acceptor.is_some() {
        match TcpListener::bind(&config.gateway.listen_submission).await {
            Ok(l) => {
                tracing::info!(addr = %config.gateway.listen_submission, "Submission listener started (port 465, implicit TLS)");
                Some(l)
            }
            Err(e) => {
                tracing::warn!(addr = %config.gateway.listen_submission, error = %e, "failed to bind submission port");
                None
            }
        }
    } else {
        None
    };

    // Bind port 587 (STARTTLS submission) if TLS is available
    let starttls_listener = if tls_acceptor.is_some() {
        match TcpListener::bind(&config.gateway.listen_submission_starttls).await {
            Ok(l) => {
                tracing::info!(addr = %config.gateway.listen_submission_starttls, "STARTTLS submission listener started (port 587)");
                Some(l)
            }
            Err(e) => {
                tracing::warn!(addr = %config.gateway.listen_submission_starttls, error = %e, "failed to bind STARTTLS submission port");
                None
            }
        }
    } else {
        None
    };

    // Shared cancellation token for graceful shutdown of all tasks
    let cancel = tokio_util::sync::CancellationToken::new();

    // Background cleanup task for per-IP tracking
    let cleanup_shared = Arc::clone(&shared);
    let cleanup_cancel = cancel.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(CLEANUP_INTERVAL);
        loop {
            tokio::select! {
                _ = interval.tick() => {
                    cleanup_shared.cleanup(Duration::from_secs(3600));
                }
                _ = cleanup_cancel.cancelled() => break,
            }
        }
    });

    // Spawn implicit TLS listener (port 465)
    if let Some(sub_listener) = submission_listener {
        let smtp_config = smtp_config.clone();
        let shared = Arc::clone(&shared);
        let acceptor = tls_acceptor.clone().unwrap();
        let listener_cancel = cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = sub_listener.accept() => {
                        let (stream, addr) = match result {
                            Ok(sa) => sa,
                            Err(e) => { tracing::warn!(error = %e, "submission accept error"); continue; }
                        };
                        let peer_ip = addr.ip();
                        if !shared.try_connect(peer_ip) {
                            drop(stream);
                            continue;
                        }
                        let cfg = smtp_config.clone();
                        let sh = Arc::clone(&shared);
                        let acc = Arc::clone(&acceptor);
                        tokio::spawn(async move {
                            let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                            match tls::implicit_tls_wrap(stream, &acc).await {
                                Ok(tls_stream) => {
                                    let (reader, writer) = tokio::io::split(tls_stream);
                                    let session = SmtpSession::new(cfg);
                                    if let Err(e) = handle_connection_generic(reader, writer, peer_ip, session, max_message_size, None).await {
                                        tracing::debug!(%peer_ip, error = %e, "implicit TLS connection ended with error");
                                    }
                                }
                                Err(e) => tracing::debug!(%peer_ip, error = %e, "implicit TLS handshake failed"),
                            }
                        });
                    }
                    _ = listener_cancel.cancelled() => break,
                }
            }
        });
    }

    // Spawn STARTTLS submission listener (port 587)
    if let Some(starttls_listener) = starttls_listener {
        let smtp_config = smtp_config.clone();
        let shared = Arc::clone(&shared);
        let acceptor = tls_acceptor.clone().unwrap();
        let listener_cancel = cancel.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = starttls_listener.accept() => {
                        let (stream, addr) = match result {
                            Ok(sa) => sa,
                            Err(e) => { tracing::warn!(error = %e, "STARTTLS submission accept error"); continue; }
                        };
                        let peer_ip = addr.ip();
                        if !shared.try_connect(peer_ip) {
                            drop(stream);
                            continue;
                        }
                        let cfg = smtp_config.clone();
                        let sh = Arc::clone(&shared);
                        let acc = Arc::clone(&acceptor);
                        tokio::spawn(async move {
                            let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                            if let Err(e) = handle_connection(stream, peer_ip, cfg, max_message_size, Some((*acc).clone())).await {
                                tracing::debug!(%peer_ip, error = %e, "STARTTLS submission connection ended with error");
                            }
                        });
                    }
                    _ = listener_cancel.cancelled() => break,
                }
            }
        });
    }

    // ── IMAP setup ───────────────────────────────────────────────────
    if config.imap.enabled {
        let imap_store = Arc::new(
            ImapStore::open(Path::new(&config.imap.store_path))
                .map_err(|e| format!("failed to open IMAP store: {e}"))?,
        );
        imap_store
            .initialize_default_mailboxes()
            .map_err(|e| format!("failed to init IMAP mailboxes: {e}"))?;

        let imap_domain = config.domain.name.clone();
        let imap_idle_timeout = Duration::from_secs(config.imap.idle_timeout);
        let imap_max_auth_failures = config.imap.max_auth_failures;
        let content_store_path = PathBuf::from(&config.imap.content_store_path);
        if !content_store_path.is_dir() {
            tracing::warn!(
                path = %content_store_path.display(),
                "IMAP content store path is not a directory — FETCH will skip all messages"
            );
        }

        // Port 993: IMAP implicit TLS
        if let Some(ref acceptor) = tls_acceptor {
            match TcpListener::bind(&config.imap.listen_imaps).await {
                Ok(imaps_listener) => {
                    tracing::info!(addr = %config.imap.listen_imaps, "IMAPS listener started (port 993, implicit TLS)");
                    let store = Arc::clone(&imap_store);
                    let shared = Arc::clone(&shared);
                    let acc = Arc::clone(acceptor);
                    let domain = imap_domain.clone();
                    let listener_cancel = cancel.clone();
                    let csp_outer = content_store_path.clone();
                    tokio::spawn(async move {
                        loop {
                            tokio::select! {
                                result = imaps_listener.accept() => {
                                    let (stream, addr) = match result {
                                        Ok(sa) => sa,
                                        Err(e) => { tracing::warn!(error = %e, "IMAPS accept error"); continue; }
                                    };
                                    let peer_ip = addr.ip();
                                    if !shared.try_connect(peer_ip) {
                                        drop(stream);
                                        continue;
                                    }
                                    let sh = Arc::clone(&shared);
                                    let a = Arc::clone(&acc);
                                    let st = Arc::clone(&store);
                                    let d = domain.clone();
                                    let csp = csp_outer.clone();
                                    tokio::spawn(async move {
                                        let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                                        match tls::implicit_tls_wrap(stream, &a).await {
                                            Ok(tls_stream) => {
                                                let (reader, writer) = tokio::io::split(tls_stream);
                                                let imap_cfg = ImapConfig {
                                                    domain: d,
                                                    tls_active: true,
                                                    tls_available: true,
                                                    max_auth_failures: imap_max_auth_failures,
                                                };
                                                if let Err(e) = handle_imap_connection(reader, writer, imap_cfg, st, imap_idle_timeout, csp).await {
                                                    tracing::debug!(%peer_ip, error = %e, "IMAPS connection ended with error");
                                                }
                                            }
                                            Err(e) => tracing::debug!(%peer_ip, error = %e, "IMAPS TLS handshake failed"),
                                        }
                                    });
                                }
                                _ = listener_cancel.cancelled() => break,
                            }
                        }
                    });
                }
                Err(e) => {
                    tracing::warn!(addr = %config.imap.listen_imaps, error = %e, "failed to bind IMAPS port");
                }
            }
        }

        // Port 143: IMAP STARTTLS
        match TcpListener::bind(&config.imap.listen_imap).await {
            Ok(imap_listener) => {
                tracing::info!(addr = %config.imap.listen_imap, "IMAP listener started (port 143, STARTTLS)");
                let store = Arc::clone(&imap_store);
                let shared = Arc::clone(&shared);
                let acceptor_for_imap = tls_acceptor.clone();
                let domain = imap_domain.clone();
                let listener_cancel = cancel.clone();
                let csp_outer = content_store_path.clone();
                tokio::spawn(async move {
                    loop {
                        tokio::select! {
                            result = imap_listener.accept() => {
                                let (stream, addr) = match result {
                                    Ok(sa) => sa,
                                    Err(e) => { tracing::warn!(error = %e, "IMAP accept error"); continue; }
                                };
                                let peer_ip = addr.ip();
                                if !shared.try_connect(peer_ip) {
                                    drop(stream);
                                    continue;
                                }
                                let sh = Arc::clone(&shared);
                                let st = Arc::clone(&store);
                                let d = domain.clone();
                                let acc = acceptor_for_imap.clone();
                                let csp = csp_outer.clone();
                                tokio::spawn(async move {
                                    let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                                    let (reader, writer) = tokio::io::split(stream);
                                    let imap_cfg = ImapConfig {
                                        domain: d,
                                        tls_active: false,
                                        // STARTTLS upgrade not yet wired on port 143 — don't
                                        // advertise it until the upgrade path is implemented
                                        tls_available: false,
                                        max_auth_failures: imap_max_auth_failures,
                                    };
                                    if let Err(e) = handle_imap_connection_starttls(reader, writer, imap_cfg, st, imap_idle_timeout, acc, csp).await {
                                        tracing::debug!(%peer_ip, error = %e, "IMAP connection ended with error");
                                    }
                                });
                            }
                            _ = listener_cancel.cancelled() => break,
                        }
                    }
                });
            }
            Err(e) => {
                tracing::warn!(addr = %config.imap.listen_imap, error = %e, "failed to bind IMAP port");
            }
        }
    }

    // Main SMTP accept loop (port 25)
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = smtp_listener.accept() => {
                let (stream, addr) = match result {
                    Ok(sa) => sa,
                    Err(e) => {
                        tracing::error!(error = %e, "SMTP accept error, shutting down");
                        cancel.cancel();
                        return Err(e.into());
                    }
                };
                let peer_ip = addr.ip();

                if !shared.try_connect(peer_ip) {
                    tracing::warn!(%peer_ip, "connection rejected: per-IP limit exceeded");
                    drop(stream);
                    continue;
                }

                let smtp_config = smtp_config.clone();
                let shared = Arc::clone(&shared);
                let acceptor = tls_acceptor.clone();

                tokio::spawn(async move {
                    let _guard = ConnectionGuard { shared: Arc::clone(&shared), ip: peer_ip };
                    let acc = acceptor.as_ref().map(|a| (**a).clone());
                    if let Err(e) = handle_connection(stream, peer_ip, smtp_config, max_message_size, acc).await {
                        tracing::debug!(%peer_ip, error = %e, "connection ended with error");
                    }
                });
            }
            _ = &mut shutdown => {
                tracing::info!("shutting down SMTP server");
                cancel.cancel(); // Signal listener loops to stop accepting
                // Note: in-flight connection tasks are not tracked/awaited here.
                // They will drain naturally as they complete or when the runtime
                // shuts down. A JoinSet-based approach for graceful connection
                // draining is a v1.1 improvement.
                break;
            }
        }
    }

    Ok(())
}

/// Handle a single plaintext SMTP connection with optional STARTTLS support.
async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_ip: IpAddr,
    smtp_config: SmtpConfig,
    max_message_size: usize,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut session = SmtpSession::new(smtp_config);

    // Send initial greeting on plaintext
    let actions = session.handle(SmtpEvent::Connected {
        peer_ip,
        tls: false,
    });

    // Use into_split so we can reunite for STARTTLS upgrade
    let (reader, mut writer) = stream.into_split();
    let mut framed = FramedRead::new(reader, SmtpCodec::new(max_message_size));
    execute_actions_generic(&actions, &mut writer).await?;

    loop {
        let timeout_dur = if session.state == SmtpState::DataReceiving {
            DATA_TIMEOUT
        } else {
            COMMAND_TIMEOUT
        };

        let frame = tokio::time::timeout(timeout_dur, framed.next()).await;

        match frame {
            Ok(Some(Ok(smtp_frame))) => {
                let actions = match smtp_frame {
                    SmtpFrame::Line(line) => match parse_command(line.as_bytes()) {
                        Ok(cmd) => {
                            let actions = session.handle(SmtpEvent::Command(cmd));
                            if session.state == SmtpState::DataReceiving {
                                framed.decoder_mut().enter_data_mode();
                            }
                            actions
                        }
                        Err(_) => {
                            vec![SmtpAction::SendResponse(
                                500,
                                "5.5.1 Command not recognized".to_string(),
                            )]
                        }
                    },
                    SmtpFrame::Data(body) => {
                        framed.decoder_mut().enter_command_mode();
                        session.handle(SmtpEvent::DataComplete(body))
                    }
                };

                let should_close = execute_actions_generic(&actions, &mut writer).await?;
                let needs_starttls =
                    process_async_actions(&actions, &mut session, &mut writer).await?;

                // STARTTLS: upgrade the connection to TLS
                if needs_starttls {
                    if let Some(ref acceptor) = tls_acceptor {
                        // Flush pending writes before TLS handshake
                        writer.flush().await?;
                        // Use into_parts() to preserve any buffered bytes (e.g., an
                        // eagerly-sent TLS ClientHello that arrived with STARTTLS)
                        let parts = framed.into_parts();
                        let tcp_reader = parts.io;
                        if !parts.read_buf.is_empty() {
                            // Per RFC 3207 §4: the client must wait for the 220
                            // response before sending TLS data. Buffered bytes here
                            // mean a misbehaving client; reject rather than risk a
                            // corrupted TLS handshake.
                            tracing::warn!(
                                %peer_ip,
                                buffered_bytes = parts.read_buf.len(),
                                "STARTTLS: rejecting — client sent data before 220 response"
                            );
                            let mut wr = tcp_reader
                                .reunite(writer)
                                .map_err(|e| format!("failed to reunite: {e}"))?;
                            let _ = wr.write_all(b"454 4.7.0 STARTTLS failed\r\n").await;
                            return Ok(());
                        }
                        let tcp_stream = tcp_reader
                            .reunite(writer)
                            .map_err(|e| format!("failed to reunite TCP halves: {e}"))?;
                        // Perform TLS handshake
                        match tls::starttls_upgrade(tcp_stream, acceptor).await {
                            Ok(tls_stream) => {
                                let (tls_reader, tls_writer) = tokio::io::split(tls_stream);
                                // Feed TlsCompleted to state machine
                                let tls_actions = session.handle(SmtpEvent::TlsCompleted);
                                let mut tls_writer = tls_writer;
                                execute_actions_generic(&tls_actions, &mut tls_writer).await?;
                                // Continue the session over TLS
                                return handle_connection_generic(
                                    tls_reader,
                                    tls_writer,
                                    peer_ip,
                                    session,
                                    max_message_size,
                                    None,
                                )
                                .await;
                            }
                            Err(e) => {
                                tracing::warn!(%peer_ip, error = %e, "STARTTLS handshake failed");
                                return Ok(());
                            }
                        }
                    } else {
                        tracing::warn!(%peer_ip, "STARTTLS requested but no TLS configured");
                    }
                }

                if should_close || session.state == SmtpState::Closed {
                    break;
                }
            }
            Ok(Some(Err(e))) => {
                let msg = e.to_string();
                if msg.contains("exceeds maximum size") {
                    // Oversize DATA: permanent rejection per RFC 5321
                    tracing::debug!(%peer_ip, "message too large");
                    let _ = writer.write_all(b"552 5.3.4 Message too large\r\n").await;
                    // Reset codec and session so the connection can continue.
                    // We can't use SmtpCommand::Rset here because the state machine
                    // rejects RSET during DataReceiving. reset_transaction() clears
                    // all transaction state (mail_from, recipients, pending_rcpt).
                    framed.decoder_mut().enter_command_mode();
                    session.reset_transaction();
                } else {
                    tracing::debug!(%peer_ip, error = %e, "codec error");
                    let _ = writer
                        .write_all(b"421 4.7.0 Error processing input\r\n")
                        .await;
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => {
                let _ = writer
                    .write_all(b"421 4.4.2 Connection timed out\r\n")
                    .await;
                break;
            }
        }
    }

    let _ = writer.shutdown().await;
    Ok(())
}

/// Generic connection handler over any AsyncRead + AsyncWrite (used after TLS upgrade
/// and for implicit TLS connections).
async fn handle_connection_generic<R, W>(
    reader: R,
    mut writer: W,
    peer_ip: IpAddr,
    mut session: SmtpSession,
    max_message_size: usize,
    _tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    // For implicit TLS, send the greeting if session is fresh
    if session.state == SmtpState::Connected {
        let actions = session.handle(SmtpEvent::Connected { peer_ip, tls: true });
        execute_actions_generic(&actions, &mut writer).await?;
    }

    let mut framed = FramedRead::new(reader, SmtpCodec::new(max_message_size));

    loop {
        let timeout_dur = if session.state == SmtpState::DataReceiving {
            DATA_TIMEOUT
        } else {
            COMMAND_TIMEOUT
        };

        let frame = tokio::time::timeout(timeout_dur, framed.next()).await;

        match frame {
            Ok(Some(Ok(smtp_frame))) => {
                let actions = match smtp_frame {
                    SmtpFrame::Line(line) => match parse_command(line.as_bytes()) {
                        Ok(cmd) => {
                            let actions = session.handle(SmtpEvent::Command(cmd));
                            if session.state == SmtpState::DataReceiving {
                                framed.decoder_mut().enter_data_mode();
                            }
                            actions
                        }
                        Err(_) => {
                            vec![SmtpAction::SendResponse(
                                500,
                                "5.5.1 Command not recognized".to_string(),
                            )]
                        }
                    },
                    SmtpFrame::Data(body) => {
                        framed.decoder_mut().enter_command_mode();
                        session.handle(SmtpEvent::DataComplete(body))
                    }
                };

                let should_close = execute_actions_generic(&actions, &mut writer).await?;
                let _ = process_async_actions(&actions, &mut session, &mut writer).await?;

                if should_close || session.state == SmtpState::Closed {
                    break;
                }
            }
            Ok(Some(Err(e))) => {
                let msg = e.to_string();
                if msg.contains("exceeds maximum size") {
                    tracing::debug!(%peer_ip, "message too large");
                    let _ = writer.write_all(b"552 5.3.4 Message too large\r\n").await;
                    framed.decoder_mut().enter_command_mode();
                    session.reset_transaction();
                } else {
                    tracing::debug!(%peer_ip, error = %e, "codec error");
                    let _ = writer
                        .write_all(b"421 4.7.0 Error processing input\r\n")
                        .await;
                    break;
                }
            }
            Ok(None) => break,
            Err(_) => {
                let _ = writer
                    .write_all(b"421 4.4.2 Connection timed out\r\n")
                    .await;
                break;
            }
        }
    }

    let _ = writer.shutdown().await;
    Ok(())
}

/// Execute SMTP actions by writing responses to the client.
/// Returns true if a Close action was encountered.
async fn execute_actions_generic<W: AsyncWrite + Unpin>(
    actions: &[SmtpAction],
    writer: &mut W,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let mut should_close = false;

    for action in actions {
        match action {
            SmtpAction::SendResponse(code, msg) => {
                let response = format!("{code} {msg}\r\n");
                writer.write_all(response.as_bytes()).await?;
            }
            SmtpAction::SendEhloResponse {
                greeting,
                capabilities,
            } => {
                if capabilities.is_empty() {
                    writer
                        .write_all(format!("250 {greeting}\r\n").as_bytes())
                        .await?;
                } else {
                    writer
                        .write_all(format!("250-{greeting}\r\n").as_bytes())
                        .await?;
                    for (i, cap) in capabilities.iter().enumerate() {
                        let prefix = if i == capabilities.len() - 1 {
                            "250 "
                        } else {
                            "250-"
                        };
                        writer
                            .write_all(format!("{prefix}{cap}\r\n").as_bytes())
                            .await?;
                    }
                }
                writer.flush().await?;
            }
            SmtpAction::Close => {
                should_close = true;
            }
            // Async actions are handled by the caller after execute_actions returns
            SmtpAction::StartTls
            | SmtpAction::CheckSpf { .. }
            | SmtpAction::ResolveHarmonyAddress { .. }
            | SmtpAction::DeliverToHarmony { .. } => {}
        }
    }

    writer.flush().await?;
    Ok(should_close)
}

/// Process async action callbacks (address resolution, delivery, SPF).
/// Returns true if a StartTls action was encountered.
async fn process_async_actions<W: AsyncWrite + Unpin>(
    actions: &[SmtpAction],
    session: &mut SmtpSession,
    writer: &mut W,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let mut needs_starttls = false;
    for action in actions {
        match action {
            SmtpAction::ResolveHarmonyAddress { local_part, .. } => {
                use crate::message::ADDRESS_HASH_LEN;
                // Stub: all addresses resolve to a dummy identity.
                // Real implementation will query the announce cache.
                let identity = Some([0u8; ADDRESS_HASH_LEN]);
                let callback_actions = session.handle(SmtpEvent::HarmonyResolved {
                    local_part: local_part.clone(),
                    identity,
                });
                execute_actions_generic(&callback_actions, writer).await?;
                // Warn if callbacks produced further async actions (not yet handled)
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
            SmtpAction::DeliverToHarmony { .. } => {
                // Stub: delivery always succeeds.
                // Real implementation will deliver to Harmony network.
                let callback_actions = session.handle(SmtpEvent::DeliveryResult { success: true });
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
            SmtpAction::CheckSpf { .. } => {
                // SPF checking tracked in Linear — requires mail_auth SPF resolver integration
            }
            SmtpAction::StartTls => {
                needs_starttls = true;
            }
            _ => {}
        }
    }
    Ok(needs_starttls)
}

// ── IMAP connection handling ─────────────────────────────────────────

/// IMAP idle timeout for command reads (5 minutes, tighter than SMTP).
const IMAP_COMMAND_TIMEOUT: Duration = Duration::from_secs(300);

/// Handle an IMAP connection over an already-TLS'd stream (generic reader/writer).
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
                        if should_close || session.state == ImapState::Logout {
                            break;
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
                break;
            }
            Ok(None) => break, // EOF
            Err(_) => {
                writer.write_all(b"* BYE connection timed out\r\n").await?;
                break;
            }
        }
    }

    let _ = writer.shutdown().await;
    Ok(())
}

/// Handle an IMAP connection on port 143 (plaintext with optional STARTTLS).
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
    // For v1.1, STARTTLS on IMAP port 143 is not yet wired — use the
    // generic handler. Full STARTTLS upgrade (similar to SMTP's
    // handle_connection) will be added when needed.
    handle_imap_connection(reader, writer, config, store, idle_timeout, content_store_path).await
}

/// Execute IMAP actions by writing responses to the client.
/// Returns true if a Close action was encountered.
async fn execute_imap_actions<W: AsyncWrite + Unpin>(
    actions: &[ImapAction],
    writer: &mut W,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let mut should_close = false;

    for action in actions {
        match action {
            ImapAction::SendUntagged(msg) => {
                writer.write_all(format!("* {msg}\r\n").as_bytes()).await?;
            }
            ImapAction::SendTagged {
                tag,
                status,
                code,
                text,
            } => {
                let status_str = match status {
                    ResponseStatus::Ok => "OK",
                    ResponseStatus::No => "NO",
                    ResponseStatus::Bad => "BAD",
                };
                let response = match code {
                    Some(c) => format!("{tag} {status_str} [{c}] {text}\r\n"),
                    None => format!("{tag} {status_str} {text}\r\n"),
                };
                writer.write_all(response.as_bytes()).await?;
            }
            ImapAction::SendContinuation(msg) => {
                writer.write_all(format!("+ {msg}\r\n").as_bytes()).await?;
            }
            ImapAction::Close => {
                should_close = true;
            }
            // Async actions handled by process_imap_async_actions
            _ => {}
        }
    }

    writer.flush().await?;
    Ok(should_close)
}

/// Process IMAP async actions (authentication, mailbox ops, fetch, etc.).
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
    for action in actions {
        match action {
            ImapAction::Authenticate { username, password } => {
                // Argon2 verification is CPU-intensive — run off the async runtime.
                let u = username.clone();
                let p = password.clone();
                let store_clone = Arc::clone(store);
                let result = tokio::task::spawn_blocking(move || store_clone.authenticate(&u, &p))
                    .await
                    .map_err(|e| format!("auth task panicked: {e}"))?;
                let (success, uname) = match result {
                    Ok(user) => (true, user.username),
                    Err(_) => (false, username.clone()),
                };
                let callback = session.handle(ImapEvent::AuthResult {
                    success,
                    username: uname,
                });
                execute_imap_actions(&callback, writer).await?;
            }
            ImapAction::SelectMailbox { name, read_only } => {
                let select_result: Result<MailboxSnapshot, String> = (|| {
                    let mbox = store
                        .get_mailbox(name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let total = store.count_messages(mbox.id).map_err(|e| e.to_string())?;
                    let recent = store.count_recent(mbox.id).map_err(|e| e.to_string())?;
                    let unseen = store.count_unseen(mbox.id).map_err(|e| e.to_string())?;
                    let first_unseen = store
                        .first_unseen_seqnum(mbox.id)
                        .map_err(|e| e.to_string())?;
                    Ok(MailboxSnapshot {
                        name: mbox.name,
                        uid_validity: mbox.uid_validity,
                        uid_next: mbox.uid_next,
                        total_messages: total,
                        recent_count: recent,
                        unseen_count: unseen,
                        first_unseen,
                        read_only: *read_only,
                        is_examine: *read_only,
                    })
                })();
                match select_result {
                    Ok(snapshot) => {
                        let callback = session.handle(ImapEvent::MailboxLoaded(snapshot));
                        execute_imap_actions(&callback, writer).await?;
                    }
                    Err(reason) => {
                        let tag = session
                            .pending_tag
                            .take()
                            .unwrap_or_else(|| "?".to_string());
                        writer
                            .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                            .await?;
                        writer.flush().await?;
                    }
                }
            }
            ImapAction::ListMailboxes { pattern, .. } => {
                match store.list_mailboxes(pattern) {
                    Ok(mailboxes) => {
                        for mbox in &mailboxes {
                            let quoted = imap_quote_mailbox(&mbox.name);
                            let line = format!("* LIST () NIL {quoted}\r\n");
                            writer.write_all(line.as_bytes()).await?;
                        }
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} OK LIST completed\r\n").as_bytes())
                                .await?;
                        }
                    }
                    Err(e) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} NO {e}\r\n").as_bytes())
                                .await?;
                        }
                    }
                }
                writer.flush().await?;
            }
            ImapAction::GetStatus { mailbox, items } => {
                let status_result = store.get_mailbox(mailbox).ok().flatten().and_then(|mbox| {
                    let total = store.count_messages(mbox.id).ok()?;
                    let recent = store.count_recent(mbox.id).ok()?;
                    let unseen = store.count_unseen(mbox.id).ok()?;
                    Some((mbox, total, recent, unseen))
                });
                if let Some((mbox, total, recent, unseen)) = status_result {
                    let mut status_items = Vec::new();
                    for item in items {
                        match item {
                            crate::imap_parse::StatusItem::Messages => {
                                status_items.push(format!("MESSAGES {total}"))
                            }
                            crate::imap_parse::StatusItem::Recent => {
                                status_items.push(format!("RECENT {recent}"))
                            }
                            crate::imap_parse::StatusItem::UidNext => {
                                status_items.push(format!("UIDNEXT {}", mbox.uid_next))
                            }
                            crate::imap_parse::StatusItem::UidValidity => {
                                status_items.push(format!("UIDVALIDITY {}", mbox.uid_validity))
                            }
                            crate::imap_parse::StatusItem::Unseen => {
                                status_items.push(format!("UNSEEN {unseen}"))
                            }
                        }
                    }
                    let quoted = imap_quote_mailbox(mailbox);
                    writer
                        .write_all(
                            format!("* STATUS {quoted} ({})\r\n", status_items.join(" "))
                                .as_bytes(),
                        )
                        .await?;
                    if let Some(tag) = session.pending_tag.take() {
                        writer
                            .write_all(format!("{tag} OK STATUS completed\r\n").as_bytes())
                            .await?;
                    }
                } else if let Some(tag) = session.pending_tag.take() {
                    writer
                        .write_all(format!("{tag} NO mailbox not found\r\n").as_bytes())
                        .await?;
                }
                writer.flush().await?;
            }
            ImapAction::CreateMailbox { name } => {
                let result = store.create_mailbox(name);
                if let Some(tag) = session.pending_tag.take() {
                    match result {
                        Ok(()) => {
                            writer
                                .write_all(format!("{tag} OK CREATE completed\r\n").as_bytes())
                                .await?
                        }
                        Err(e) => {
                            writer
                                .write_all(format!("{tag} NO {e}\r\n").as_bytes())
                                .await?
                        }
                    }
                }
                writer.flush().await?;
            }
            ImapAction::DeleteMailbox { name } => {
                let result = store.delete_mailbox(name);
                if let Some(tag) = session.pending_tag.take() {
                    match result {
                        Ok(()) => {
                            writer
                                .write_all(format!("{tag} OK DELETE completed\r\n").as_bytes())
                                .await?
                        }
                        Err(e) => {
                            writer
                                .write_all(format!("{tag} NO {e}\r\n").as_bytes())
                                .await?
                        }
                    }
                }
                writer.flush().await?;
            }
            ImapAction::SubscribeMailbox { name } => {
                let result = store.subscribe(name);
                if let Some(tag) = session.pending_tag.take() {
                    match result {
                        Ok(()) => {
                            writer
                                .write_all(format!("{tag} OK SUBSCRIBE completed\r\n").as_bytes())
                                .await?
                        }
                        Err(e) => {
                            writer
                                .write_all(format!("{tag} NO {e}\r\n").as_bytes())
                                .await?
                        }
                    }
                }
                writer.flush().await?;
            }
            ImapAction::UnsubscribeMailbox { name } => {
                let result = store.unsubscribe(name);
                if let Some(tag) = session.pending_tag.take() {
                    match result {
                        Ok(()) => {
                            writer
                                .write_all(format!("{tag} OK UNSUBSCRIBE completed\r\n").as_bytes())
                                .await?
                        }
                        Err(e) => {
                            writer
                                .write_all(format!("{tag} NO {e}\r\n").as_bytes())
                                .await?
                        }
                    }
                }
                writer.flush().await?;
            }
            ImapAction::Expunge => {
                let expunge_result: Result<Vec<u32>, String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;
                    let uid_to_seqnum: std::collections::HashMap<u32, u32> = all_msgs
                        .iter()
                        .enumerate()
                        .map(|(i, m)| (m.uid, (i + 1) as u32))
                        .collect();
                    let expunged_uids = store.expunge(mbox.id).map_err(|e| e.to_string())?;
                    let mut seqnums: Vec<u32> = expunged_uids
                        .iter()
                        .filter_map(|uid| uid_to_seqnum.get(uid).copied())
                        .collect();
                    seqnums.sort_unstable_by(|a, b| b.cmp(a));
                    Ok(seqnums)
                })();
                match expunge_result {
                    Ok(seqnums) => {
                        let callback = session.handle(ImapEvent::ExpungeComplete {
                            expunged_seqnums: seqnums,
                        });
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
                    let rows_by_uid: std::collections::HashMap<u32, &imap_store::MessageRow> =
                        all_msgs.iter().map(|m| (m.uid, m)).collect();

                    let flag_refs: Vec<&str> = flags.iter().map(|s| s.as_str()).collect();
                    let mut updated = Vec::new();

                    for (uid, seqnum) in &resolved {
                        let msg_row = match rows_by_uid.get(uid) {
                            Some(r) => *r,
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
            ImapAction::FetchMessages {
                sequence_set,
                attributes,
                uid_mode,
            } => {
                // RFC 9051 §6.4.8: UID FETCH responses MUST include UID
                let attrs = if *uid_mode && !attributes.contains(&imap_parse::FetchAttribute::Uid) {
                    let mut a = attributes.clone();
                    a.push(imap_parse::FetchAttribute::Uid);
                    a
                } else {
                    attributes.clone()
                };
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
                        let rows_by_uid: std::collections::HashMap<u32, &imap_store::MessageRow> =
                            all_msgs.iter().map(|m| (m.uid, m)).collect();
                        for (uid, seqnum) in &resolved {
                            let msg_row = match rows_by_uid.get(uid) {
                                Some(r) => *r,
                                None => continue,
                            };
                            let cid_bytes = match msg_row.message_cid {
                                Some(c) => c,
                                None => {
                                    // No CAS content — render metadata-only attributes
                                    // (FLAGS, UID, RFC822.SIZE) per RFC 9051 §7.5.2
                                    let mut items = Vec::new();
                                    for attr in &attrs {
                                        match attr {
                                            imap_parse::FetchAttribute::Flags => {
                                                let f = store.get_flags(msg_row.id).unwrap_or_default();
                                                let flags_str = if f.is_empty() {
                                                    "()".to_string()
                                                } else {
                                                    format!("({})", f.join(" "))
                                                };
                                                items.push(format!("FLAGS {flags_str}"));
                                            }
                                            imap_parse::FetchAttribute::Uid => {
                                                items.push(format!("UID {uid}"));
                                            }
                                            imap_parse::FetchAttribute::Rfc822Size => {
                                                items.push(format!("RFC822.SIZE {}", msg_row.rfc822_size));
                                            }
                                            _ => {} // content-requiring attributes skipped
                                        }
                                    }
                                    if !items.is_empty() {
                                        writer
                                            .write_all(
                                                format!("* {seqnum} FETCH ({})\r\n", items.join(" "))
                                                    .as_bytes(),
                                            )
                                            .await?;
                                    }
                                    continue;
                                }
                            };

                            // Retrieve message from CAS via spawn_blocking (file I/O)
                            let csp_clone = csp.clone();
                            let cas_result = tokio::task::spawn_blocking(move || {
                                let book_store = harmony_db::DiskBookStore::new(&csp_clone);
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

                            let flags = match store.get_flags(msg_row.id) {
                                Ok(f) => f,
                                Err(e) => {
                                    tracing::warn!(uid = uid, error = %e, "FETCH: flag retrieval failed, skipping");
                                    continue;
                                }
                            };
                            match crate::imap_render::build_fetch_response(
                                *seqnum, *uid, &attrs, &harmony_msg, &flags,
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
            ImapAction::Search { .. } => {
                // Not yet wired to store search
                if let Some(tag) = session.pending_tag.take() {
                    writer
                        .write_all(
                            format!("{tag} NO [CANNOT] SEARCH not yet implemented\r\n").as_bytes(),
                        )
                        .await?;
                }
                writer.flush().await?;
            }
            ImapAction::CopyMessages { .. } => {
                // Not yet wired to store copy
                if let Some(tag) = session.pending_tag.take() {
                    writer
                        .write_all(
                            format!("{tag} NO [CANNOT] COPY/MOVE not yet implemented\r\n")
                                .as_bytes(),
                        )
                        .await?;
                }
                writer.flush().await?;
            }
            ImapAction::StartIdle { .. } => {
                framed.decoder_mut().enter_idle_mode();
                // IDLE loop: wait for DONE from client (via codec) or timeout
                // The main loop handles IdleDone events from the codec
            }
            ImapAction::StopIdle => {
                framed.decoder_mut().exit_idle_mode();
            }
            ImapAction::CloseExpunge { tag, mailbox } => {
                // RFC 9051 §6.4.2: CLOSE silently expunges — no untagged EXPUNGE responses.
                if let Ok(Some(mbox)) = store.get_mailbox(mailbox) {
                    let _ = store.expunge(mbox.id);
                }
                writer
                    .write_all(format!("{tag} OK CLOSE completed\r\n").as_bytes())
                    .await?;
                writer.flush().await?;
            }
            ImapAction::StartTls => {
                // STARTTLS for IMAP — not yet wired (tls_available=false prevents advertisement)
                tracing::debug!("IMAP STARTTLS requested but not yet implemented");
            }
            // Synchronous actions already handled by execute_imap_actions
            ImapAction::SendUntagged(_)
            | ImapAction::SendTagged { .. }
            | ImapAction::SendContinuation(_)
            | ImapAction::Close => {}
        }
    }
    Ok(())
}

/// Quote a mailbox name for safe inclusion in IMAP responses.
/// Returns the name as-is if it's atom-safe, otherwise wraps in double quotes
/// with proper escaping. Prevents CRLF injection via crafted mailbox names.
fn imap_quote_mailbox(name: &str) -> String {
    // Atom-safe: printable ASCII excluding specials per RFC 9051 §9 formal syntax
    let needs_quoting = name.is_empty()
        || name.bytes().any(|b| {
            b <= 0x20
                || b >= 0x7F
                || matches!(
                    b,
                    b'"' | b'\\' | b'(' | b')' | b'{' | b'}' | b' ' | b'*' | b'%'
                )
        });

    if !needs_quoting {
        name.to_string()
    } else {
        let mut quoted = String::with_capacity(name.len() + 4);
        quoted.push('"');
        for ch in name.chars() {
            if ch == '"' || ch == '\\' {
                quoted.push('\\');
            }
            // Strip control characters including CR/LF to prevent injection
            if ch >= ' ' && ch != '\x7f' {
                quoted.push(ch);
            }
        }
        quoted.push('"');
        quoted
    }
}

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
    let mut seen = std::collections::HashSet::new();

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
                if msg.uid >= lo && msg.uid <= hi && seen.insert(msg.uid) {
                    result.push((msg.uid, (idx + 1) as u32));
                }
            }
        } else {
            // Intersect with valid range [1, max_seqnum]; skip if no overlap
            if lo > max_seqnum || hi == 0 {
                continue;
            }
            let intersect_lo = lo.max(1);
            let intersect_hi = hi.min(max_seqnum);
            for seqnum in intersect_lo..=intersect_hi {
                let msg = &messages[(seqnum - 1) as usize];
                if seen.insert(msg.uid) {
                    result.push((msg.uid, seqnum));
                }
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
    use tokio::net::TcpStream;

    /// Read a complete SMTP response (may be multi-line).
    /// Returns the full response text. Multi-line responses use "NNN-" prefix
    /// for continuation lines and "NNN " for the final line.
    async fn read_smtp_response(reader: &mut BufReader<tokio::net::tcp::OwnedReadHalf>) -> String {
        let mut full_response = String::new();
        loop {
            let mut line = String::new();
            reader.read_line(&mut line).await.unwrap();
            let is_last = line.len() >= 4 && line.as_bytes()[3] == b' ';
            full_response.push_str(&line);
            if is_last || line.is_empty() {
                break;
            }
        }
        full_response
    }

    fn test_config() -> Config {
        let toml_str = r#"
[domain]
name = "test.example.com"
mx_host = "mail.test.example.com"

[gateway]
identity_key = "/tmp/test-key"
listen_smtp = "127.0.0.1:0"

[tls]
mode = "manual"
cert = "/tmp/test-cert.pem"
key = "/tmp/test-key.pem"

[dkim]
key = "/tmp/test-dkim.key"

[spam]
reject_threshold = 5

[outbound]
queue_path = "/tmp/test-queue"

[harmony]
node_config = "/tmp/test-node.toml"
"#;
        Config::from_toml(toml_str).unwrap()
    }

    #[tokio::test]
    async fn smtp_handshake_ehlo_quit() {
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
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size, None)
                .await
                .unwrap();
        });

        let stream = TcpStream::connect(addr).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Read 220 greeting
        let greeting = read_smtp_response(&mut reader).await;
        assert!(
            greeting.starts_with("220 "),
            "expected 220 greeting, got: {greeting}"
        );

        // Send EHLO
        write_half
            .write_all(b"EHLO client.example.com\r\n")
            .await
            .unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(
            ehlo_resp.contains("250"),
            "expected 250 in EHLO response, got: {ehlo_resp}"
        );

        // Send QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(
            quit_resp.starts_with("221 "),
            "expected 221, got: {quit_resp}"
        );

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn smtp_full_mail_transaction() {
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
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size, None)
                .await
                .unwrap();
        });

        let stream = TcpStream::connect(addr).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Read 220 greeting
        let greeting = read_smtp_response(&mut reader).await;
        assert!(greeting.starts_with("220 "), "greeting: {greeting}");

        // EHLO
        write_half.write_all(b"EHLO test.com\r\n").await.unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(ehlo_resp.contains("250"), "EHLO: {ehlo_resp}");

        // MAIL FROM
        write_half
            .write_all(b"MAIL FROM:<sender@test.com>\r\n")
            .await
            .unwrap();
        let mail_resp = read_smtp_response(&mut reader).await;
        assert!(mail_resp.contains("250"), "MAIL FROM: {mail_resp}");

        // RCPT TO (triggers resolve -> stub accepts)
        write_half
            .write_all(b"RCPT TO:<user@test.example.com>\r\n")
            .await
            .unwrap();
        let rcpt_resp = read_smtp_response(&mut reader).await;
        assert!(rcpt_resp.contains("250"), "RCPT TO: {rcpt_resp}");

        // DATA
        write_half.write_all(b"DATA\r\n").await.unwrap();
        let data_resp = read_smtp_response(&mut reader).await;
        assert!(data_resp.starts_with("354"), "DATA: {data_resp}");

        // Send message body + terminator
        write_half
            .write_all(b"Subject: Test\r\n\r\nHello World\r\n.\r\n")
            .await
            .unwrap();
        let deliver_resp = read_smtp_response(&mut reader).await;
        assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

        // QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "QUIT: {quit_resp}");

        server_handle.await.unwrap();
    }

    #[test]
    fn per_ip_rate_limiting() {
        let shared = SharedState::new(2);
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        assert!(shared.try_connect(ip));
        assert!(shared.try_connect(ip));
        assert!(!shared.try_connect(ip)); // third rejected

        shared.disconnect(ip);
        assert!(shared.try_connect(ip)); // slot freed
    }

    #[test]
    fn parse_message_size_variants() {
        assert_eq!(parse_message_size("25MB"), 25 * 1024 * 1024);
        assert_eq!(parse_message_size("10KB"), 10 * 1024);
        assert_eq!(parse_message_size("1GB"), 1024 * 1024 * 1024);
        assert_eq!(parse_message_size("1000"), 1000);
        assert_eq!(parse_message_size("  25MB  "), 25 * 1024 * 1024);
    }
}

#[cfg(test)]
mod sequence_set_tests {
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

    #[test]
    fn resolve_seqnum_out_of_range_returns_empty() {
        let msgs = make_messages(&[10, 20, 30]);
        // Seqnum 999 doesn't exist in a 3-message mailbox — must not resolve to message 3
        let result = resolve_sequence_set(&seq(vec![(999, None)]), false, &msgs);
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_seqnum_range_out_of_range_returns_empty() {
        let msgs = make_messages(&[10, 20, 30]);
        // Seqnums 100:200 don't exist — must return nothing
        let result = resolve_sequence_set(&seq(vec![(100, Some(200))]), false, &msgs);
        assert!(result.is_empty());
    }

    #[test]
    fn resolve_seqnum_partial_overlap() {
        let msgs = make_messages(&[10, 20, 30]);
        // Seqnums 2:10 — only 2 and 3 are valid
        let result = resolve_sequence_set(&seq(vec![(2, Some(10))]), false, &msgs);
        assert_eq!(result, vec![(20, 2), (30, 3)]);
    }

    #[test]
    fn resolve_dedup_overlapping_ranges() {
        let msgs = make_messages(&[1, 2, 3, 4, 5]);
        // "1:3,2" — UID 2 appears in both ranges, should only appear once
        let result = resolve_sequence_set(
            &seq(vec![(1, Some(3)), (2, None)]),
            true,
            &msgs,
        );
        assert_eq!(result, vec![(1, 1), (2, 2), (3, 3)]);
    }

    #[test]
    fn resolve_dedup_repeated_single() {
        let msgs = make_messages(&[1, 2, 3]);
        // "2,2" — same UID twice, should appear once
        let result = resolve_sequence_set(
            &seq(vec![(2, None), (2, None)]),
            true,
            &msgs,
        );
        assert_eq!(result, vec![(2, 2)]);
    }

    use crate::imap_parse::FetchAttribute;
    use crate::imap_render::build_fetch_response;
    use crate::imap_store::ImapStore;
    use crate::message::{
        HarmonyMessage, MailMessageType, MessageFlags, Recipient, RecipientType, ADDRESS_HASH_LEN,
        CID_LEN, VERSION,
    };
    use harmony_content::book::MemoryBookStore;
    use harmony_content::chunker::ChunkerConfig;
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
    fn fetch_metadata_only_without_cid() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        // Insert message with no CID
        let uid = imap_store
            .insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 1713000000, 100)
            .unwrap();

        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let messages = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(messages.len(), 1);
        assert!(messages[0].message_cid.is_none());

        // Add a flag so we can verify it's returned
        imap_store.add_flags(messages[0].id, &["\\Seen"]).unwrap();
        let flags = imap_store.get_flags(messages[0].id).unwrap();
        assert_eq!(flags, vec!["\\Seen"]);

        // Verify metadata is available even without CAS content:
        // UID, FLAGS, and RFC822.SIZE should all be renderable from MessageRow
        assert_eq!(uid, 1);
        assert_eq!(messages[0].rfc822_size, 100);
    }

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
}

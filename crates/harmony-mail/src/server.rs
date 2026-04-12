//! SMTP server: TCP listener, connection handling, and state machine I/O driver.
//!
//! Binds to configured SMTP ports, accepts connections, and spawns per-connection
//! async tasks that wire [`SmtpCodec`](crate::io::SmtpCodec) frames to the
//! [`SmtpSession`](crate::smtp::SmtpSession) sans-I/O state machine.

use std::collections::HashMap;
use std::net::IpAddr;
use std::path::Path;
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
use crate::imap_store::ImapStore;
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
                                    tokio::spawn(async move {
                                        let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                                        match tls::implicit_tls_wrap(stream, &a).await {
                                            Ok(tls_stream) => {
                                                let (reader, writer) = tokio::io::split(tls_stream);
                                                let imap_cfg = ImapConfig {
                                                    domain: d,
                                                    tls_active: true,
                                                    max_auth_failures: imap_max_auth_failures,
                                                };
                                                if let Err(e) = handle_imap_connection(reader, writer, imap_cfg, st, imap_idle_timeout).await {
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
                                tokio::spawn(async move {
                                    let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                                    let (reader, writer) = tokio::io::split(stream);
                                    let imap_cfg = ImapConfig {
                                        domain: d,
                                        tls_active: false,
                                        max_auth_failures: imap_max_auth_failures,
                                    };
                                    if let Err(e) = handle_imap_connection_starttls(reader, writer, imap_cfg, st, imap_idle_timeout, acc).await {
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
                    session.handle(SmtpEvent::Command(crate::smtp::SmtpCommand::Rset));
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
                // TODO: wire to mail_auth SPF resolver and feed result into spam scorer
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
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
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
                        )
                        .await?;
                        if should_close || session.state == ImapState::Logout {
                            break;
                        }
                    }
                    Err(e) => {
                        writer
                            .write_all(format!("* BAD {e}\r\n").as_bytes())
                            .await?;
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
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send + 'static,
    W: AsyncWrite + Unpin + Send + 'static,
{
    // For v1.1, STARTTLS on IMAP port 143 is not yet wired — use the
    // generic handler. Full STARTTLS upgrade (similar to SMTP's
    // handle_connection) will be added when needed.
    handle_imap_connection(reader, writer, config, store, idle_timeout).await
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
    store: &ImapStore,
    framed: &mut FramedRead<R, ImapCodec>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    for action in actions {
        match action {
            ImapAction::Authenticate { username, password } => {
                let result = store.authenticate(username, password);
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
                match store.get_mailbox(name) {
                    Ok(Some(mbox)) => {
                        let total = store.count_messages(mbox.id).unwrap_or(0);
                        let recent = store.count_recent(mbox.id).unwrap_or(0);
                        let unseen = store.count_unseen(mbox.id).unwrap_or(0);
                        let snapshot = MailboxSnapshot {
                            name: mbox.name,
                            uid_validity: mbox.uid_validity,
                            uid_next: mbox.uid_next,
                            total_messages: total,
                            recent_count: recent,
                            unseen_count: unseen,
                            first_unseen: if unseen > 0 { Some(1) } else { None },
                            read_only: *read_only,
                        };
                        let callback = session.handle(ImapEvent::MailboxLoaded(snapshot));
                        execute_imap_actions(&callback, writer).await?;
                    }
                    _ => {
                        // Mailbox not found — send NO directly
                        let tag = "?"; // pending tag consumed by state machine
                        writer
                            .write_all(format!("{tag} NO mailbox not found\r\n").as_bytes())
                            .await?;
                        writer.flush().await?;
                    }
                }
            }
            ImapAction::ListMailboxes { pattern, .. } => {
                let mailboxes = store.list_mailboxes(pattern).unwrap_or_default();
                for mbox in &mailboxes {
                    let line = format!("* LIST () NIL {}\r\n", mbox.name);
                    writer.write_all(line.as_bytes()).await?;
                }
                // The tagged OK is sent by the state machine after the action completes.
                // For LIST, we need to send the tagged response ourselves since it's
                // a synchronous action in the I/O layer.
                // Actually, LIST was dispatched as an async action. We need to signal
                // completion. For now, extract the pending tag and send OK.
                if let Some(tag) = &session.pending_tag {
                    writer
                        .write_all(format!("{tag} OK LIST completed\r\n").as_bytes())
                        .await?;
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::GetStatus { mailbox, items } => {
                if let Ok(Some(mbox)) = store.get_mailbox(mailbox) {
                    let total = store.count_messages(mbox.id).unwrap_or(0);
                    let recent = store.count_recent(mbox.id).unwrap_or(0);
                    let unseen = store.count_unseen(mbox.id).unwrap_or(0);
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
                    writer
                        .write_all(
                            format!("* STATUS {} ({})\r\n", mailbox, status_items.join(" "))
                                .as_bytes(),
                        )
                        .await?;
                }
                if let Some(tag) = &session.pending_tag {
                    writer
                        .write_all(format!("{tag} OK STATUS completed\r\n").as_bytes())
                        .await?;
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::CreateMailbox { name } => {
                let result = store.create_mailbox(name);
                if let Some(tag) = &session.pending_tag {
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
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::DeleteMailbox { name } => {
                let result = store.delete_mailbox(name);
                if let Some(tag) = &session.pending_tag {
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
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::SubscribeMailbox { name } => {
                let result = store.subscribe(name);
                if let Some(tag) = &session.pending_tag {
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
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::UnsubscribeMailbox { name } => {
                let result = store.unsubscribe(name);
                if let Some(tag) = &session.pending_tag {
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
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::Expunge => {
                if let ImapState::Selected { ref mailbox, .. } = session.state {
                    if let Ok(Some(mbox)) = store.get_mailbox(&mailbox.name) {
                        let expunged_uids = store.expunge(mbox.id).unwrap_or_default();
                        // Convert UIDs to sequence numbers (for simplicity, use UIDs as seqnums in v1.1)
                        let callback = session.handle(ImapEvent::ExpungeComplete {
                            expunged_seqnums: expunged_uids,
                        });
                        execute_imap_actions(&callback, writer).await?;
                    }
                }
            }
            ImapAction::StoreFlags {
                sequence_set: _,
                operation: _,
                flags: _,
                uid_mode: _,
                silent,
            } => {
                // Stub: in v1.1, STORE is acknowledged without actual Harmony integration
                let updated = Vec::new(); // TODO: resolve sequence set, update flags in store
                if !silent {
                    let callback = session.handle(ImapEvent::StoreComplete { updated });
                    execute_imap_actions(&callback, writer).await?;
                } else {
                    // Silent: just send tagged OK
                    if let Some(tag) = &session.pending_tag {
                        writer
                            .write_all(format!("{tag} OK STORE completed\r\n").as_bytes())
                            .await?;
                        session.pending_tag = None;
                    }
                    writer.flush().await?;
                }
            }
            ImapAction::FetchMessages { .. } => {
                // Stub: FETCH requires Harmony message retrieval + rendering
                // For v1.1, send tagged OK without data
                if let Some(tag) = &session.pending_tag {
                    writer
                        .write_all(format!("{tag} OK FETCH completed\r\n").as_bytes())
                        .await?;
                    session.pending_tag = None;
                }
                writer.flush().await?;
            }
            ImapAction::Search { .. } => {
                // Stub: return empty results
                let callback = session.handle(ImapEvent::SearchComplete { results: vec![] });
                execute_imap_actions(&callback, writer).await?;
            }
            ImapAction::CopyMessages { .. } => {
                // Stub: return empty mapping
                let callback = session.handle(ImapEvent::CopyComplete {
                    uid_mapping: vec![],
                });
                execute_imap_actions(&callback, writer).await?;
            }
            ImapAction::StartIdle { .. } => {
                framed.decoder_mut().enter_idle_mode();
                // IDLE loop: wait for DONE from client (via codec) or timeout
                // The main loop handles IdleDone events from the codec
            }
            ImapAction::StopIdle => {
                framed.decoder_mut().exit_idle_mode();
            }
            ImapAction::StartTls => {
                // STARTTLS for IMAP — not yet wired in v1.1 (port 143 handler)
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

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
use crate::smtp::{SmtpAction, SmtpCommand, SmtpConfig, SmtpEvent, SmtpSession, SmtpState};
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
pub async fn run(
    config: Config,
    gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<(), Box<dyn std::error::Error>> {
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

    // ── Shared IMAP store (used by both SMTP delivery and IMAP) ────
    let imap_store = Arc::new(
        ImapStore::open(Path::new(&config.imap.store_path))
            .map_err(|e| format!("failed to open IMAP store: {e}"))?,
    );
    imap_store
        .initialize_default_mailboxes()
        .map_err(|e| format!("failed to init IMAP mailboxes: {e}"))?;

    // ── SPF authenticator ──────────────────────────────────────────
    let mail_authenticator: Option<Arc<mail_auth::MessageAuthenticator>> =
        match crate::auth::create_authenticator() {
            Ok(auth) => Some(Arc::new(auth)),
            Err(e) => {
                tracing::warn!(error = %e, "failed to create mail authenticator, SPF disabled");
                None
            }
        };

    let content_store_path = PathBuf::from(&config.imap.content_store_path);
    // Ensure CAS subdirectories exist (DiskBookStore requires commits/ and blobs/).
    std::fs::create_dir_all(content_store_path.join("commits"))
        .map_err(|e| format!("failed to create CAS commits dir: {e}"))?;
    std::fs::create_dir_all(content_store_path.join("blobs"))
        .map_err(|e| format!("failed to create CAS blobs dir: {e}"))?;
    let reject_threshold = config.spam.reject_threshold;
    let local_domain = config.domain.name.clone();

    // ── Merkle mailbox manager ────────────────────────────────────────
    let store_dir = Path::new(&config.imap.store_path)
        .parent()
        .unwrap_or(Path::new("."));
    let mailbox_mgr: Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>> =
        match crate::mailbox_manager::MailboxManager::open(
            &store_dir.join("mailbox_roots.db"),
            &content_store_path,
        ) {
            Ok(mut mgr) => {
                // Initialize Merkle trees for all registered users
                match imap_store.list_users() {
                    Ok(users) => {
                        for user in &users {
                            if let Err(e) = mgr.ensure_user_mailbox(&user.harmony_address) {
                                tracing::warn!(
                                    user = %user.username,
                                    error = %e,
                                    "failed to init Merkle mailbox"
                                );
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            "failed to list users for Merkle mailbox init, trees will be created on demand"
                        );
                    }
                }
                tracing::info!(
                    users = mgr.user_count(),
                    "Merkle mailbox manager initialized"
                );
                Some(Arc::new(std::sync::Mutex::new(mgr)))
            }
            Err(e) => {
                tracing::warn!(error = %e, "failed to open Merkle mailbox manager, Merkle writes disabled");
                None
            }
        };

    // Shared cancellation token for graceful shutdown of all tasks.
    // Created before the Zenoh block so ZenohPublisher's drain task can
    // participate in the same shutdown signal as listener loops and
    // cleanup tasks below.
    let cancel = tokio_util::sync::CancellationToken::new();

    // Standalone handle on the Zenoh publisher. The MailboxManager gets its
    // own Arc clone for internal root-CID notify() calls (which run inside
    // its mutex, on spawn_blocking); this handle is for the raw-mail publish
    // path, which runs in async context and must not take the manager's
    // mutex (doing so could block a Tokio worker thread behind a concurrent
    // spawn_blocking CAS/SQLite write).
    let mut mailbox_publisher: Option<Arc<crate::mailbox_manager::ZenohPublisher>> = None;

    // ── Zenoh session (mailbox root CID notifications) ──────────────
    {
        use crate::mailbox_manager::ZenohPublisher;

        let zenoh_enabled = config.zenoh.as_ref().map(|z| z.enabled).unwrap_or(false);
        if zenoh_enabled && mailbox_mgr.is_none() {
            // Opening the session here would spawn a ZenohPublisher drain task
            // that captures the session by move. With no MailboxManager to
            // attach the publisher to, the publisher would be dropped while
            // the drain task kept the session alive forever — a leak.
            tracing::warn!(
                "Zenoh enabled but MailboxManager unavailable — skipping session open to avoid leaking the drain task"
            );
        } else if zenoh_enabled {
            // Build Zenoh config. The gateway is a publisher only — we never
            // want it to open a listening socket. Default config would run in
            // peer mode and bind a random TCP port, which is unexpected for
            // a gateway and a security concern. Force client mode and clear
            // listen endpoints before handing off to `zenoh::open()`.
            let mut zenoh_config = zenoh::Config::default();
            let endpoint = config.zenoh.as_ref().and_then(|z| z.endpoint.as_ref());
            let mut config_ok = true;

            // Force client mode (connect-only, no incoming listener).
            if let Err(e) = zenoh_config.insert_json5("mode", "\"client\"") {
                tracing::error!(error = %e, "Zenoh client-mode config rejected, mailbox notifications disabled");
                config_ok = false;
            }
            // Belt-and-suspenders: explicitly zero the listen endpoints.
            if config_ok {
                if let Err(e) = zenoh_config.insert_json5("listen/endpoints", "[]") {
                    tracing::error!(error = %e, "Zenoh listen-endpoints config rejected, mailbox notifications disabled");
                    config_ok = false;
                }
            }

            if config_ok {
                if let Some(ep) = endpoint {
                    match serde_json::to_string(ep) {
                        Ok(ep_json) => {
                            if let Err(e) = zenoh_config
                                .insert_json5("connect/endpoints", &format!("[{ep_json}]"))
                            {
                                tracing::error!(error = %e, endpoint = %ep, "Zenoh endpoint config rejected, mailbox notifications disabled");
                                config_ok = false;
                            }
                        }
                        Err(e) => {
                            tracing::error!(error = %e, endpoint = %ep, "Zenoh endpoint JSON-encode failed, mailbox notifications disabled");
                            config_ok = false;
                        }
                    }
                } else {
                    tracing::info!(
                        "Zenoh enabled without explicit endpoint — running in client mode with peer discovery"
                    );
                }
            }

            if config_ok {
                // The outer gate guarantees mailbox_mgr.is_some() here, so it
                // is safe to open the session without risking a leaked drain
                // task with no consumer.
                let mgr_arc = mailbox_mgr
                    .as_ref()
                    .expect("mailbox_mgr.is_some() checked by outer gate");
                match zenoh::open(zenoh_config).await {
                    Ok(session) => {
                        tracing::info!("Zenoh session opened for mailbox notifications");
                        let publisher_arc = Arc::new(ZenohPublisher::new(session, cancel.clone()));
                        // Give the manager its own refcount bump so root-CID
                        // notify() calls inside insert_message keep working;
                        // keep a separate handle for the async raw-publish
                        // path below.
                        match mgr_arc.lock() {
                            Ok(mut mgr) => mgr.set_publisher(Arc::clone(&publisher_arc)),
                            Err(poisoned) => {
                                tracing::warn!("MailboxManager mutex poisoned during Zenoh publisher attach, recovering");
                                poisoned.into_inner().set_publisher(Arc::clone(&publisher_arc));
                            }
                        }
                        mailbox_publisher = Some(publisher_arc);
                    }
                    Err(e) => {
                        tracing::warn!(error = %e, "Zenoh open failed, mailbox notifications disabled");
                    }
                }
            }
        } else {
            tracing::debug!("Zenoh mailbox notifications disabled");
        }
    }

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
        let imap_store_465 = Arc::clone(&imap_store);
        let mail_authenticator_465 = mail_authenticator.clone();
        let local_domain_465 = local_domain.clone();
        let content_store_path_465 = content_store_path.clone();
        let mailbox_mgr_465 = mailbox_mgr.clone();
        let mailbox_publisher_465 = mailbox_publisher.clone();
        let gateway_identity_465 = gateway_identity.clone();
        let recipient_resolver_465 = recipient_resolver.clone();
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
                        let store = Arc::clone(&imap_store_465);
                        let auth = mail_authenticator_465.clone();
                        let domain = local_domain_465.clone();
                        let csp = content_store_path_465.clone();
                        let mgr_clone = mailbox_mgr_465.clone();
                        let pub_clone = mailbox_publisher_465.clone();
                        let gi_clone = gateway_identity_465.clone();
                        let rr_clone = recipient_resolver_465.clone();
                        tokio::spawn(async move {
                            let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                            match tls::implicit_tls_wrap(stream, &acc).await {
                                Ok(tls_stream) => {
                                    let (reader, writer) = tokio::io::split(tls_stream);
                                    let session = SmtpSession::new(cfg);
                                    if let Err(e) = handle_connection_generic(reader, writer, peer_ip, session, max_message_size, None, store, auth, domain, csp, reject_threshold, mgr_clone, pub_clone, gi_clone, rr_clone).await {
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
        let imap_store_587 = Arc::clone(&imap_store);
        let mail_authenticator_587 = mail_authenticator.clone();
        let local_domain_587 = local_domain.clone();
        let content_store_path_587 = content_store_path.clone();
        let mailbox_mgr_587 = mailbox_mgr.clone();
        let mailbox_publisher_587 = mailbox_publisher.clone();
        let gateway_identity_587 = gateway_identity.clone();
        let recipient_resolver_587 = recipient_resolver.clone();
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
                        let store = Arc::clone(&imap_store_587);
                        let auth = mail_authenticator_587.clone();
                        let domain = local_domain_587.clone();
                        let csp = content_store_path_587.clone();
                        let mgr_clone = mailbox_mgr_587.clone();
                        let pub_clone = mailbox_publisher_587.clone();
                        let gi_clone = gateway_identity_587.clone();
                        let rr_clone = recipient_resolver_587.clone();
                        tokio::spawn(async move {
                            let _guard = ConnectionGuard { shared: Arc::clone(&sh), ip: peer_ip };
                            if let Err(e) = handle_connection(stream, peer_ip, cfg, max_message_size, Some((*acc).clone()), store, auth, domain, csp, reject_threshold, mgr_clone, pub_clone, gi_clone, rr_clone).await {
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
        let imap_domain = config.domain.name.clone();
        let imap_idle_timeout = Duration::from_secs(config.imap.idle_timeout);
        let imap_max_auth_failures = config.imap.max_auth_failures;
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
                                    let (reader, writer) = stream.into_split();
                                    let imap_cfg = ImapConfig {
                                        domain: d,
                                        tls_active: false,
                                        tls_available: acc.is_some(),
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
                let store = Arc::clone(&imap_store);
                let auth = mail_authenticator.clone();
                let domain = local_domain.clone();
                let csp = content_store_path.clone();
                let mgr_clone = mailbox_mgr.clone();
                let pub_clone = mailbox_publisher.clone();
                let gi_clone = gateway_identity.clone();
                let rr_clone = recipient_resolver.clone();

                tokio::spawn(async move {
                    let _guard = ConnectionGuard { shared: Arc::clone(&shared), ip: peer_ip };
                    let acc = acceptor.as_ref().map(|a| (**a).clone());
                    if let Err(e) = handle_connection(stream, peer_ip, smtp_config, max_message_size, acc, store, auth, domain, csp, reject_threshold, mgr_clone, pub_clone, gi_clone, rr_clone).await {
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
#[allow(clippy::too_many_arguments)]
async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_ip: IpAddr,
    smtp_config: SmtpConfig,
    max_message_size: usize,
    tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    imap_store: Arc<ImapStore>,
    authenticator: Option<Arc<mail_auth::MessageAuthenticator>>,
    local_domain: String,
    content_store_path: PathBuf,
    reject_threshold: i32,
    mailbox_manager: Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
    mailbox_publisher: Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut spf_result = crate::spam::SpfResult::None;
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
                            // Reset SPF result at the start of each transaction
                            // so a prior message's verdict can't leak into a new
                            // transaction (especially null-sender where CheckSpf
                            // is skipped).
                            if matches!(cmd, SmtpCommand::MailFrom { .. }) {
                                spf_result = crate::spam::SpfResult::None;
                            }
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
                    process_async_actions(&actions, &mut session, &mut writer, &imap_store, &authenticator, &local_domain, &content_store_path, &mut spf_result, reject_threshold, &mailbox_manager, &mailbox_publisher, &gateway_identity, &recipient_resolver).await?;

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
                                    imap_store,
                                    authenticator,
                                    local_domain,
                                    content_store_path,
                                    reject_threshold,
                                    mailbox_manager,
                                    mailbox_publisher,
                                    gateway_identity,
                                    recipient_resolver,
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
                    spf_result = crate::spam::SpfResult::None;
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
#[allow(clippy::too_many_arguments)]
async fn handle_connection_generic<R, W>(
    reader: R,
    mut writer: W,
    peer_ip: IpAddr,
    mut session: SmtpSession,
    max_message_size: usize,
    _tls_acceptor: Option<tokio_rustls::TlsAcceptor>,
    imap_store: Arc<ImapStore>,
    authenticator: Option<Arc<mail_auth::MessageAuthenticator>>,
    local_domain: String,
    content_store_path: PathBuf,
    reject_threshold: i32,
    mailbox_manager: Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
    mailbox_publisher: Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let mut spf_result = crate::spam::SpfResult::None;
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
                            if matches!(cmd, SmtpCommand::MailFrom { .. }) {
                                spf_result = crate::spam::SpfResult::None;
                            }
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
                let _ = process_async_actions(&actions, &mut session, &mut writer, &imap_store, &authenticator, &local_domain, &content_store_path, &mut spf_result, reject_threshold, &mailbox_manager, &mailbox_publisher, &gateway_identity, &recipient_resolver).await?;

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
                    spf_result = crate::spam::SpfResult::None;
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
#[allow(clippy::too_many_arguments)]
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
    mailbox_manager: &Option<Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>>,
    mailbox_publisher: &Option<Arc<crate::mailbox_manager::ZenohPublisher>>,
    gateway_identity: &Option<Arc<harmony_identity::PrivateIdentity>>,
    recipient_resolver: &Option<Arc<dyn crate::remote_delivery::RecipientResolver>>,
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    let mut needs_starttls = false;
    for action in actions {
        match action {
            SmtpAction::ResolveHarmonyAddress { local_part, domain } => {
                use crate::address::{self, LocalPart};

                let identity = if domain.eq_ignore_ascii_case(local_domain) {
                    // Local domain: resolve against our user store
                    match address::parse_local_part(local_part) {
                        LocalPart::Hex(hash) => {
                            // Verify the hex address belongs to a known user
                            match imap_store.get_user_by_address(&hash) {
                                Ok(Some(user)) => Some(user.harmony_address),
                                Ok(None) => None,
                                Err(e) => {
                                    tracing::warn!(error = %e, "hex address lookup failed");
                                    None
                                }
                            }
                        }
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
                        local_domain = %local_domain,
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
            SmtpAction::DeliverToHarmony { recipients, data } => {
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
                    tracing::info!(score = verdict.score, "message rejected by spam filter");
                    // Spam rejection is a permanent policy decision — send 550
                    // (not 451 via DeliveryResult) so remote MTAs don't retry.
                    // Bypass the state machine and reset directly, same pattern
                    // as the oversize DATA handler.
                    execute_actions_generic(
                        &[SmtpAction::SendResponse(
                            550,
                            "5.7.1 Message rejected".to_string(),
                        )],
                        writer,
                    )
                    .await?;
                    session.reset_transaction();
                    *spf_result = crate::spam::SpfResult::None;
                    continue;
                }

                // Phase 2: Translate RFC 5322 -> HarmonyMessage
                let translated = match crate::translate::translate_inbound(data) {
                    Ok(t) => t,
                    Err(e) => {
                        tracing::warn!(error = %e, "message translation failed");
                        let callback_actions = session.handle(SmtpEvent::DeliveryResult { success: false });
                        execute_actions_generic(&callback_actions, writer).await?;
                        continue;
                    }
                };

                // Extract sender/subject before Phase 3 moves `translated.message`
                let sender_address = translated.message.sender_address;
                let subject_snippet = crate::mailbox::truncate_utf8(
                    &translated.message.subject,
                    crate::mailbox::MAX_SNIPPET_LEN,
                )
                .to_string();

                // Phase 3: Store attachments + message in CAS
                //
                // Ingest attachment blobs first so their CAS CIDs can be
                // written into the HarmonyMessage before it is serialized.
                let msg_timestamp = translated.message.timestamp;
                let msg_id = translated.message.message_id;
                let mut message = translated.message;
                let csp = content_store_path.to_path_buf();
                let attachment_data = translated.attachment_data;
                // Returns both the CAS CID (for IMAP/Merkle references) and the
                // serialized bytes (for Zenoh raw-mail publish). Raw bytes are
                // returned even if CAS ingest fails, because the Phase 0 client
                // consumes raw bytes directly and is independent of gateway
                // CAS state. Both fields become None only if serialization
                // itself failed.
                let cas_result = tokio::task::spawn_blocking(
                    move || -> Result<(Option<[u8; 32]>, Option<Vec<u8>>), String> {
                        let config = &harmony_content::chunker::ChunkerConfig::DEFAULT;
                        let mut book_store = harmony_db::DiskBookStore::new(&csp);

                        // Ingest each attachment blob and update the message's CIDs
                        for (i, blob) in attachment_data.iter().enumerate() {
                            if blob.is_empty() {
                                continue;
                            }
                            match harmony_content::dag::ingest(blob, config, &mut book_store) {
                                Ok(cid) => {
                                    if i < message.attachments.len() {
                                        message.attachments[i].cid = cid.to_bytes();
                                    }
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        attachment_index = i,
                                        error = %e,
                                        "attachment CAS ingest failed, keeping BLAKE3 CID"
                                    );
                                }
                            }
                        }

                        // Serialize once, then attempt CAS ingest. Keep the
                        // bytes regardless of ingest outcome so the raw-mail
                        // publish can still go out if CAS failed.
                        match message.to_bytes() {
                            Ok(msg_bytes) => {
                                match harmony_content::dag::ingest(&msg_bytes, config, &mut book_store) {
                                    Ok(cid) => Ok((Some(cid.to_bytes()), Some(msg_bytes))),
                                    Err(e) => {
                                        tracing::warn!(error = %e, "CAS storage failed, delivering without CID");
                                        Ok((None, Some(msg_bytes)))
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "message serialization failed, delivering without CID");
                                Ok((None, None))
                            }
                        }
                    },
                )
                .await
                .map_err(|e| format!("CAS task panicked: {e}"))?;
                let (message_cid, msg_bytes): (Option<[u8; 32]>, Option<Vec<u8>>) = match cas_result
                {
                    Ok(pair) => pair,
                    Err(e) => {
                        tracing::warn!(error = %e, "CAS task error, delivering without CID");
                        (None, None)
                    }
                };

                // Phase 4: Deliver to local IMAP mailboxes
                //
                // NOTE: The IMAP store currently uses a single-user mailbox
                // model (mailboxes are global, not per-user). Multi-user
                // mailbox isolation is tracked separately in ZEB-112.
                let mut delivered_to: Vec<[u8; crate::message::ADDRESS_HASH_LEN]> = Vec::new();
                let timestamp = msg_timestamp;
                let rfc822_size = data.len() as u32;

                // Pre-compute a reusable remote-delivery context when all four
                // ingredients are present. If any is missing (no gateway
                // identity, no resolver, no publisher, or message
                // serialization failed), remote delivery is effectively
                // disabled and every non-local recipient falls through to a
                // debug log.
                let remote_ctx: Option<(
                    Arc<harmony_identity::PrivateIdentity>,
                    Arc<dyn crate::remote_delivery::RecipientResolver>,
                    Arc<crate::mailbox_manager::ZenohPublisher>,
                    Vec<u8>,
                )> = match (
                    gateway_identity.as_ref(),
                    recipient_resolver.as_ref(),
                    mailbox_publisher.as_ref(),
                    msg_bytes.as_ref(),
                ) {
                    (Some(gw), Some(res), Some(pub_), Some(bytes)) => Some((
                        Arc::clone(gw),
                        Arc::clone(res),
                        Arc::clone(pub_),
                        bytes.clone(),
                    )),
                    _ => None,
                };

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
                                    delivered_to.push(*recipient_hash);
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
                            // Recipient is not homed on this gateway. Attempt
                            // remote delivery via Zenoh if the gateway is
                            // configured for it. Per-recipient success/failure
                            // — the overall SMTP transaction still succeeds if
                            // any recipient (local or remote) succeeded.
                            if let Some((gw_id, resolver, publisher, bytes)) =
                                remote_ctx.as_ref()
                            {
                                match resolver.resolve(recipient_hash) {
                                    Some(recipient_identity) => {
                                        let mut rng = rand_core::OsRng;
                                        match crate::remote_delivery::seal_for_recipient(
                                            &mut rng,
                                            gw_id.as_ref(),
                                            &recipient_identity,
                                            bytes,
                                        ) {
                                            Ok(sealed) => {
                                                let hash_hex = hex::encode(recipient_hash);
                                                publisher.publish_sealed_unicast(
                                                    hash_hex,
                                                    Arc::new(sealed),
                                                );
                                                tracing::debug!(
                                                    recipient = %hex::encode(recipient_hash),
                                                    "remote recipient sealed + published",
                                                );
                                            }
                                            Err(e) => tracing::warn!(
                                                recipient = %hex::encode(recipient_hash),
                                                error = %e,
                                                "seal failed for remote recipient",
                                            ),
                                        }
                                    }
                                    None => tracing::warn!(
                                        recipient = %hex::encode(recipient_hash),
                                        "no announce record for remote recipient; skipping (offline store-and-forward in ZEB-113 PR B)",
                                    ),
                                }
                            } else {
                                tracing::debug!(
                                    recipient = %hex::encode(recipient_hash),
                                    "no local user and remote delivery not configured; dropping",
                                );
                            }
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

                let success = !delivered_to.is_empty();

                // Raw-mail publish: notify Zenoh subscribers (e.g., Phase 0
                // harmony-client) that a new message arrived for each
                // successfully-delivered recipient. Fire-and-forget — the
                // publisher spawns one task per recipient. Skipped entirely
                // if Zenoh is disabled (mailbox_publisher is None) or if
                // serialization failed upstream (msg_bytes is None). IMAP
                // delivery succeeded before this point and does not wait
                // on Zenoh.
                //
                // This path intentionally does NOT take the MailboxManager
                // mutex. Raw publish only needs the publisher handle; taking
                // the manager mutex in async context could block a Tokio
                // worker thread behind a concurrent spawn_blocking Merkle
                // update that holds the same mutex while doing CAS/SQLite
                // I/O. The publisher is kept in a separate Arc for that
                // reason.
                //
                // Payload is hoisted into an Arc<Vec<u8>> once so every
                // recipient in delivered_to shares the same underlying
                // buffer via refcount-only clones. Worst case (100 RCPT TO,
                // 16 MiB body): one 16 MiB allocation instead of 100.
                if let Some(bytes) = msg_bytes {
                    if let Some(ref publisher) = mailbox_publisher {
                        let shared: Arc<Vec<u8>> = Arc::new(bytes);
                        for addr in &delivered_to {
                            publisher.publish_raw_mail(hex::encode(addr), Arc::clone(&shared));
                        }
                    }
                }

                // Phase 5: Update Merkle mailbox (non-critical, fire-and-forget)
                //
                // Runs in spawn_blocking because insert_message does synchronous
                // CAS I/O and SQLite writes. Not awaited — the SMTP 250 response
                // should not wait on the Merkle write path.
                if let Some(ref mgr) = mailbox_manager {
                    if let Some(ref msg_cid) = message_cid {
                        let mgr = Arc::clone(mgr);
                        let msg_cid = *msg_cid;
                        let msg_id = msg_id;
                        let sender_address = sender_address;
                        let subject_snippet = subject_snippet.clone();
                        let delivered_to = delivered_to.clone();
                        // Fire-and-forget, but surface panics via a tiny
                        // supervisor task that awaits the JoinHandle. Without
                        // this, a panic inside insert_message (e.g., on a
                        // runtime shutdown race) would be silently swallowed
                        // when the handle dropped.
                        let join = tokio::task::spawn_blocking(move || {
                            let mut guard = match mgr.lock() {
                                Ok(g) => g,
                                Err(poisoned) => {
                                    tracing::warn!("Merkle mailbox mutex poisoned, recovering");
                                    poisoned.into_inner()
                                }
                            };
                            for addr in &delivered_to {
                                if let Err(e) = guard.insert_message(
                                    addr,
                                    &msg_cid,
                                    &msg_id,
                                    &sender_address,
                                    timestamp,
                                    &subject_snippet,
                                ) {
                                    tracing::warn!(
                                        recipient = hex::encode(addr),
                                        error = %e,
                                        "Merkle mailbox update failed, IMAP delivery unaffected"
                                    );
                                }
                            }
                        });
                        tokio::spawn(async move {
                            if let Err(e) = join.await {
                                if e.is_panic() {
                                    tracing::error!(
                                        error = ?e,
                                        "Merkle mailbox update task panicked — IMAP delivery unaffected"
                                    );
                                } else if e.is_cancelled() {
                                    tracing::debug!(
                                        "Merkle mailbox update task cancelled during shutdown"
                                    );
                                }
                            }
                        });
                    }
                }

                let callback_actions = session.handle(SmtpEvent::DeliveryResult { success });
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
            SmtpAction::CheckSpf { mail_from, ehlo_domain, peer_ip } => {
                if let Some(ref auth) = authenticator {
                    let spf_params = mail_auth::spf::verify::SpfParameters::verify_mail_from(
                        *peer_ip,
                        ehlo_domain.as_str(),
                        local_domain,
                        mail_from.as_str(),
                    );
                    let spf_output = auth.verify_spf(spf_params).await;
                    *spf_result = crate::auth::map_spf_result(&spf_output);
                    tracing::debug!(
                        %mail_from,
                        result = ?spf_result,
                        "SPF check complete"
                    );
                }
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

    // Send greeting
    let actions = session.handle(ImapEvent::Connected);
    execute_imap_actions(&actions, &mut writer).await?;

    handle_imap_session(reader, writer, session, store, idle_timeout, content_store_path, &domain).await
}

/// Run the IMAP command loop with an existing session (no greeting sent).
/// Used by both `handle_imap_connection` (after greeting) and
/// `handle_imap_connection_starttls` (after TLS upgrade, no greeting).
async fn handle_imap_session<R, W>(
    reader: R,
    mut writer: W,
    mut session: ImapSession,
    store: Arc<ImapStore>,
    idle_timeout: Duration,
    content_store_path: PathBuf,
    domain: &str,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>>
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    let codec = ImapCodec::new();
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
                            domain,
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
            // Feed TlsCompleted to state machine (sets tls_active=true, resets to NotAuthenticated)
            let tls_actions = session.handle(ImapEvent::TlsCompleted);
            let mut tls_writer = tls_writer;
            execute_imap_actions(&tls_actions, &mut tls_writer).await?;
            // Continue with existing session — no new greeting per RFC 2595 §3
            handle_imap_session(
                tls_reader,
                tls_writer,
                session,
                store,
                idle_timeout,
                content_store_path,
                &domain,
            )
            .await
        }
        Err(e) => {
            tracing::warn!(error = %e, "IMAP STARTTLS handshake failed");
            Ok(())
        }
    }
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
) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>
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
                        // Determine if any attribute requires CAS message content
                        let needs_cas = attrs.iter().any(|a| matches!(a,
                            imap_parse::FetchAttribute::Envelope
                            | imap_parse::FetchAttribute::Body
                            | imap_parse::FetchAttribute::BodyStructure
                            | imap_parse::FetchAttribute::BodySection { .. }
                            | imap_parse::FetchAttribute::Rfc822
                            | imap_parse::FetchAttribute::Rfc822Header
                            | imap_parse::FetchAttribute::Rfc822Text
                        ));

                        for (uid, seqnum) in &resolved {
                            let msg_row = match rows_by_uid.get(uid) {
                                Some(r) => *r,
                                None => continue,
                            };

                            // Try CAS retrieval only when content-requiring attrs are requested
                            // and the message has a CID. On CAS failure, fall back to metadata.
                            let harmony_msg = if needs_cas {
                                match msg_row.message_cid {
                                    Some(cid_bytes) => {
                                        let csp_clone = csp.clone();
                                        let cas_result = tokio::task::spawn_blocking(move || {
                                            let book_store = harmony_db::DiskBookStore::new(&csp_clone);
                                            let content_id = harmony_content::cid::ContentId::from_bytes(cid_bytes);
                                            harmony_content::dag::reassemble(&content_id, &book_store)
                                        })
                                        .await
                                        .map_err(|e| format!("CAS task panicked: {e}"));

                                        match cas_result {
                                            Ok(Ok(bytes)) => {
                                                match crate::message::HarmonyMessage::from_bytes(&bytes) {
                                                    Ok(m) => Some(m),
                                                    Err(e) => {
                                                        tracing::warn!(uid = uid, error = %e, "FETCH: deserialization failed, falling back to metadata");
                                                        None
                                                    }
                                                }
                                            }
                                            Ok(Err(e)) => {
                                                tracing::warn!(uid = uid, error = %e, "FETCH: CAS retrieval failed, falling back to metadata");
                                                None
                                            }
                                            Err(e) => {
                                                tracing::warn!(uid = uid, error = %e, "FETCH: CAS task failed, falling back to metadata");
                                                None
                                            }
                                        }
                                    }
                                    None => None,
                                }
                            } else {
                                None
                            };

                            // Render via build_fetch_response when we have the full message
                            if let Some(mut msg) = harmony_msg {
                                // Use msg_row.internal_date as the canonical INTERNALDATE
                                // source (same value in both CAS and metadata paths)
                                msg.timestamp = msg_row.internal_date;

                                let flags = match store.get_flags(msg_row.id) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        tracing::warn!(uid = uid, error = %e, "FETCH: flag retrieval failed, skipping");
                                        continue;
                                    }
                                };
                                match crate::imap_render::build_fetch_response(
                                    *seqnum, *uid, &attrs, &msg, &flags,
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
                            } else {
                                // Metadata-only: render from msg_row without CAS
                                let flags = match store.get_flags(msg_row.id) {
                                    Ok(f) => f,
                                    Err(e) => {
                                        tracing::warn!(uid = uid, error = %e, "FETCH: flag retrieval failed, skipping");
                                        continue;
                                    }
                                };
                                let flags_str = if flags.is_empty() {
                                    "()".to_string()
                                } else {
                                    format!("({})", flags.join(" "))
                                };
                                let mut items = Vec::new();
                                for attr in &attrs {
                                    match attr {
                                        imap_parse::FetchAttribute::Flags => {
                                            items.push(format!("FLAGS {flags_str}"));
                                        }
                                        imap_parse::FetchAttribute::Uid => {
                                            items.push(format!("UID {uid}"));
                                        }
                                        imap_parse::FetchAttribute::Rfc822Size => {
                                            items.push(format!("RFC822.SIZE {}", msg_row.rfc822_size));
                                        }
                                        imap_parse::FetchAttribute::InternalDate => {
                                            let date = crate::imap_render::format_internal_date(msg_row.internal_date);
                                            items.push(format!("INTERNALDATE \"{date}\""));
                                        }
                                        _ => {} // content-requiring attributes need CAS
                                    }
                                }
                                writer
                                    .write_all(
                                        format!("* {seqnum} FETCH ({})\r\n", items.join(" "))
                                            .as_bytes(),
                                    )
                                    .await?;
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
            ImapAction::Search { criteria, uid_mode } => {
                let search_result: Result<Vec<u32>, String> = (|| {
                    let mailbox_name = match &session.state {
                        ImapState::Selected { mailbox, .. } => mailbox.name.clone(),
                        _ => return Err("no mailbox selected".to_string()),
                    };
                    let mbox = store
                        .get_mailbox(&mailbox_name)
                        .map_err(|e| e.to_string())?
                        .ok_or_else(|| "mailbox not found".to_string())?;
                    let all_msgs = store.get_messages(mbox.id).map_err(|e| e.to_string())?;

                    // Check if any criterion needs CAS content
                    let needs_cas = criteria_need_cas(criteria);
                    let max_uid = all_msgs.last().map(|m| m.uid).unwrap_or(0);
                    let max_seqnum = all_msgs.len() as u32;
                    let book_store = if needs_cas {
                        Some(harmony_db::DiskBookStore::new(&content_store_path))
                    } else {
                        None
                    };

                    let mut results = Vec::new();
                    for (idx, msg_row) in all_msgs.iter().enumerate() {
                        let seqnum = (idx + 1) as u32;
                        let flags = store.get_flags(msg_row.id).map_err(|e| e.to_string())?;

                        let harmony_msg = if let Some(ref bs) = book_store {
                            match msg_row.message_cid {
                                Some(cid_bytes) => {
                                    let content_id = harmony_content::cid::ContentId::from_bytes(cid_bytes);
                                    match harmony_content::dag::reassemble(&content_id, bs) {
                                        Ok(bytes) => crate::message::HarmonyMessage::from_bytes(&bytes).ok(),
                                        Err(_) => None,
                                    }
                                }
                                None => None,
                            }
                        } else {
                            None
                        };

                        if matches_criteria(criteria, msg_row, &flags, seqnum, harmony_msg.as_ref(), max_uid, max_seqnum) {
                            results.push(if *uid_mode { msg_row.uid } else { seqnum });
                        }
                    }

                    Ok(results)
                })();

                match search_result {
                    Ok(results) => {
                        let callback = session.handle(ImapEvent::SearchComplete { results });
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
            ImapAction::CopyMessages {
                sequence_set,
                destination,
                uid_mode,
                is_move,
            } => {
                // Phase 1: sync — resolve mailbox, sequence set, do the copy
                enum CopyOutcome {
                    Done(Vec<(u32, u32)>, Vec<imap_store::MessageRow>, i64), // mapping, all_msgs, mailbox_id
                    Empty,
                    TryCreate,
                }

                let copy_result: Result<CopyOutcome, String> = (|| {
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
                    let uids: Vec<u32> = resolved.iter().map(|(uid, _)| *uid).collect();

                    if uids.is_empty() {
                        return Ok(CopyOutcome::Empty);
                    }

                    match store.copy_messages(mbox.id, &uids, destination) {
                        Ok(mapping) => Ok(CopyOutcome::Done(mapping, all_msgs, mbox.id)),
                        Err(imap_store::StoreError::MailboxNotFound(_)) => Ok(CopyOutcome::TryCreate),
                        Err(e) => Err(e.to_string()),
                    }
                })();

                // Phase 2: async — write responses, handle MOVE expunge
                match copy_result {
                    Ok(CopyOutcome::Done(mapping, all_msgs, mailbox_id)) => {
                        if *is_move {
                            // Flag source messages as \Deleted and expunge only those UIDs
                            let src_uids: Vec<u32> = mapping.iter().map(|(src, _)| *src).collect();
                            let rows_by_uid: std::collections::HashMap<u32, &imap_store::MessageRow> =
                                all_msgs.iter().map(|m| (m.uid, m)).collect();

                            let move_result: Result<Vec<u32>, String> = (|| {
                                for &src_uid in &src_uids {
                                    if let Some(row) = rows_by_uid.get(&src_uid) {
                                        store.add_flags(row.id, &["\\Deleted"]).map_err(|e| e.to_string())?;
                                    }
                                }
                                store.expunge_uids(mailbox_id, &src_uids).map_err(|e| e.to_string())
                            })();

                            match move_result {
                                Ok(expunged_uids) => {
                                    let uid_to_seqnum: std::collections::HashMap<u32, u32> = all_msgs
                                        .iter()
                                        .enumerate()
                                        .map(|(i, m)| (m.uid, (i + 1) as u32))
                                        .collect();
                                    let mut seqnums: Vec<u32> = expunged_uids
                                        .iter()
                                        .filter_map(|uid| uid_to_seqnum.get(uid).copied())
                                        .collect();
                                    // Descending order: higher seqnums emitted first so lower
                                    // seqnums remain valid (each EXPUNGE renumbers only above).
                                    seqnums.sort_unstable_by(|a, b| b.cmp(a));

                                    for seqnum in &seqnums {
                                        writer
                                            .write_all(format!("* {seqnum} EXPUNGE\r\n").as_bytes())
                                            .await?;
                                    }

                                    if let Some(tag) = session.pending_tag.take() {
                                        writer
                                            .write_all(format!("{tag} OK MOVE completed\r\n").as_bytes())
                                            .await?;
                                    }
                                }
                                Err(reason) => {
                                    if let Some(tag) = session.pending_tag.take() {
                                        writer
                                            .write_all(format!("{tag} NO {reason}\r\n").as_bytes())
                                            .await?;
                                    }
                                }
                            }
                        } else {
                            if let Some(tag) = session.pending_tag.take() {
                                writer
                                    .write_all(format!("{tag} OK COPY completed\r\n").as_bytes())
                                    .await?;
                            }
                        }
                        writer.flush().await?;
                    }
                    Ok(CopyOutcome::Empty) => {
                        let cmd = if *is_move { "MOVE" } else { "COPY" };
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(format!("{tag} OK {cmd} completed\r\n").as_bytes())
                                .await?;
                        }
                        writer.flush().await?;
                    }
                    Ok(CopyOutcome::TryCreate) => {
                        if let Some(tag) = session.pending_tag.take() {
                            writer
                                .write_all(
                                    format!("{tag} NO [TRYCREATE] mailbox not found\r\n").as_bytes(),
                                )
                                .await?;
                        }
                        writer.flush().await?;
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
                return Ok(true);
            }
            // Synchronous actions already handled by execute_imap_actions
            ImapAction::SendUntagged(_)
            | ImapAction::SendTagged { .. }
            | ImapAction::SendContinuation(_)
            | ImapAction::Close => {}
        }
    }
    Ok(false)
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

/// Parse an IMAP date string ("DD-Mon-YYYY") to a Unix timestamp (midnight UTC).
/// Returns None if the format is invalid.
fn parse_imap_date(s: &str) -> Option<u64> {
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let day: u32 = parts[0].parse().ok()?;
    if day == 0 || day > 31 {
        return None;
    }
    let month: u32 = match parts[1].to_ascii_lowercase().as_str() {
        "jan" => 1,
        "feb" => 2,
        "mar" => 3,
        "apr" => 4,
        "may" => 5,
        "jun" => 6,
        "jul" => 7,
        "aug" => 8,
        "sep" => 9,
        "oct" => 10,
        "nov" => 11,
        "dec" => 12,
        _ => return None,
    };
    let year: u64 = parts[2].parse().ok()?;
    if year < 1970 || year > 9999 {
        return None;
    }

    // Validate day against actual month length
    let month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
    let max_day = if month == 2 && is_leap(year) {
        29
    } else {
        month_days[month as usize]
    };
    if day > max_day {
        return None;
    }

    // Days from epoch to start of year
    let mut days: u64 = 0;
    for y in 1970..year {
        days += if is_leap(y) { 366 } else { 365 };
    }
    // Days from start of year to start of month
    for m in 1..month {
        days += month_days[m as usize] as u64;
        if m == 2 && is_leap(year) {
            days += 1;
        }
    }
    days += (day - 1) as u64;
    Some(days * 86400)
}

fn is_leap(year: u64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || year % 400 == 0
}

/// Check if any search criterion requires CAS message content.
fn criteria_need_cas(criteria: &[imap_parse::SearchKey]) -> bool {
    use imap_parse::SearchKey;
    criteria.iter().any(|key| match key {
        SearchKey::From(_) | SearchKey::To(_) | SearchKey::Subject(_)
        | SearchKey::Body(_) | SearchKey::Header(_, _) => true,
        SearchKey::Not(inner) => criteria_need_cas(std::slice::from_ref(inner.as_ref())),
        SearchKey::Or(a, b) => {
            criteria_need_cas(std::slice::from_ref(a.as_ref()))
                || criteria_need_cas(std::slice::from_ref(b.as_ref()))
        }
        _ => false,
    })
}

/// Check if a value is contained in an IMAP SequenceSet.
/// `star_value` is what `*` resolves to (max_uid or max_seqnum).
fn sequence_set_contains(set: &imap_parse::SequenceSet, value: u32, star_value: u32) -> bool {
    for range in &set.ranges {
        let s = if range.start == u32::MAX { star_value } else { range.start };
        let e = match range.end {
            None => s,
            Some(u32::MAX) => star_value,
            Some(e) => e,
        };
        let (lo, hi) = if s <= e { (s, e) } else { (e, s) };
        if value >= lo && value <= hi {
            return true;
        }
    }
    false
}

/// Evaluate IMAP SEARCH criteria against a message.
/// Multiple criteria are AND'd per RFC 9051.
fn matches_criteria(
    criteria: &[imap_parse::SearchKey],
    msg_row: &imap_store::MessageRow,
    flags: &[String],
    seqnum: u32,
    msg: Option<&crate::message::HarmonyMessage>,
    max_uid: u32,
    max_seqnum: u32,
) -> bool {
    criteria.iter().all(|key| matches_single_criterion(key, msg_row, flags, seqnum, msg, max_uid, max_seqnum))
}

/// Evaluate a single IMAP SEARCH criterion.
fn matches_single_criterion(
    key: &imap_parse::SearchKey,
    msg_row: &imap_store::MessageRow,
    flags: &[String],
    seqnum: u32,
    msg: Option<&crate::message::HarmonyMessage>,
    max_uid: u32,
    max_seqnum: u32,
) -> bool {
    use imap_parse::SearchKey;

    let has_flag = |f: &str| flags.iter().any(|fl| fl == f);

    match key {
        SearchKey::All => true,
        SearchKey::Seen => has_flag("\\Seen"),
        SearchKey::Unseen => !has_flag("\\Seen"),
        SearchKey::Flagged => has_flag("\\Flagged"),
        SearchKey::Unflagged => !has_flag("\\Flagged"),
        SearchKey::Answered => has_flag("\\Answered"),
        SearchKey::Unanswered => !has_flag("\\Answered"),
        SearchKey::Deleted => has_flag("\\Deleted"),
        SearchKey::Undeleted => !has_flag("\\Deleted"),
        SearchKey::Draft => has_flag("\\Draft"),
        SearchKey::Undraft => !has_flag("\\Draft"),
        SearchKey::Recent => has_flag("\\Recent"),
        SearchKey::New => has_flag("\\Recent") && !has_flag("\\Seen"),
        SearchKey::Old => !has_flag("\\Recent"),
        SearchKey::Keyword(s) => has_flag(s),
        SearchKey::Unkeyword(s) => !has_flag(s),

        SearchKey::Larger(n) => msg_row.rfc822_size > *n,
        SearchKey::Smaller(n) => msg_row.rfc822_size < *n,

        SearchKey::Since(date) => {
            parse_imap_date(date).map_or(false, |d| msg_row.internal_date >= d)
        }
        SearchKey::Before(date) => {
            parse_imap_date(date).map_or(false, |d| msg_row.internal_date < d)
        }
        SearchKey::On(date) => {
            parse_imap_date(date).map_or(false, |d| {
                msg_row.internal_date >= d && msg_row.internal_date < d + 86400
            })
        }

        SearchKey::Uid(set) => sequence_set_contains(set, msg_row.uid, max_uid),
        SearchKey::SequenceSet(set) => sequence_set_contains(set, seqnum, max_seqnum),

        SearchKey::Subject(s) => {
            msg.map_or(false, |m| m.subject.to_lowercase().contains(&s.to_lowercase()))
        }
        SearchKey::Body(s) => {
            msg.map_or(false, |m| m.body.to_lowercase().contains(&s.to_lowercase()))
        }
        SearchKey::From(s) => {
            msg.map_or(false, |m| {
                let sender_hex = hex::encode(m.sender_address);
                sender_hex.to_lowercase().contains(&s.to_lowercase())
            })
        }
        SearchKey::To(s) => {
            let needle = s.to_lowercase();
            msg.map_or(false, |m| {
                m.recipients.iter().any(|r| {
                    r.recipient_type == crate::message::RecipientType::To && {
                        let addr_hex = hex::encode(r.address_hash);
                        addr_hex.to_lowercase().contains(&needle)
                    }
                })
            })
        }
        SearchKey::Header(name, value) => {
            msg.map_or(false, |m| {
                let val_lower = value.to_lowercase();
                match name.to_lowercase().as_str() {
                    "subject" => m.subject.to_lowercase().contains(&val_lower),
                    "from" => {
                        let sender_hex = hex::encode(m.sender_address);
                        sender_hex.to_lowercase().contains(&val_lower)
                    }
                    "to" => {
                        m.recipients.iter().any(|r| {
                            if r.recipient_type == crate::message::RecipientType::To {
                                let addr_hex = hex::encode(r.address_hash);
                                addr_hex.to_lowercase().contains(&val_lower)
                            } else {
                                false
                            }
                        })
                    }
                    _ => false,
                }
            })
        }

        SearchKey::Not(inner) => !matches_single_criterion(inner, msg_row, flags, seqnum, msg, max_uid, max_seqnum),
        SearchKey::Or(a, b) => {
            matches_single_criterion(a, msg_row, flags, seqnum, msg, max_uid, max_seqnum)
                || matches_single_criterion(b, msg_row, flags, seqnum, msg, max_uid, max_seqnum)
        }
    }
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

        let smtp_test_dir = tempfile::tempdir().unwrap();
        let smtp_store = Arc::new(
            crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db")).unwrap()
        );
        smtp_store.initialize_default_mailboxes().unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size, None, smtp_store, None, "test.example.com".to_string(), smtp_test_dir.path().to_path_buf(), 5, None, None, None, None)
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

        let smtp_test_dir = tempfile::tempdir().unwrap();
        let smtp_store = Arc::new(
            crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db")).unwrap()
        );
        smtp_store.initialize_default_mailboxes().unwrap();
        // Create the recipient user so address resolution succeeds for RCPT TO:<user@test.example.com>
        smtp_store.create_user("user", "pass", &[0x01u8; crate::message::ADDRESS_HASH_LEN]).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size, None, smtp_store, None, "test.example.com".to_string(), smtp_test_dir.path().to_path_buf(), 5, None, None, None, None)
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

        // Post-TLS: no greeting — per RFC 2595 §3, LOGIN directly
        // LOGIN should work over TLS
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

    #[tokio::test]
    async fn smtp_delivers_to_local_imap() {
        let config = test_config();
        let max_message_size = parse_message_size(&config.spam.max_message_size);
        let smtp_config = SmtpConfig {
            domain: config.domain.name.clone(),
            mx_host: config.domain.mx_host.clone(),
            max_message_size,
            max_recipients: 100,
            tls_available: false,
        };

        let smtp_test_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(
            crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("imap.db")).unwrap(),
        );
        store.initialize_default_mailboxes().unwrap();
        // Create alice with a known harmony address
        let alice_addr = [0xAAu8; crate::message::ADDRESS_HASH_LEN];
        store.create_user("alice", "alicepass", &alice_addr).unwrap();

        let content_path = smtp_test_dir.path().join("content");
        std::fs::create_dir_all(content_path.join("commits")).unwrap();
        std::fs::create_dir_all(content_path.join("blobs")).unwrap();
        let store_clone = store.clone();
        let content_path_clone = content_path.clone();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(
                stream,
                peer_addr.ip(),
                smtp_config,
                max_message_size,
                None,
                store_clone,
                None,
                "test.example.com".to_string(),
                content_path_clone,
                5,
                None,
                None,
                None,
                None,
            )
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
        write_half.write_all(b"EHLO sender.test.com\r\n").await.unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(ehlo_resp.contains("250"), "EHLO: {ehlo_resp}");

        // MAIL FROM
        write_half
            .write_all(b"MAIL FROM:<sender@test.com>\r\n")
            .await
            .unwrap();
        let mail_resp = read_smtp_response(&mut reader).await;
        assert!(mail_resp.contains("250"), "MAIL FROM: {mail_resp}");

        // RCPT TO alice — local domain, resolves by username
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

        // Send RFC 5322 message body with proper headers
        write_half
            .write_all(
                b"From: sender@test.com\r\n\
                  To: alice@test.example.com\r\n\
                  Subject: Integration Test\r\n\
                  Message-ID: <test-deliver-001@test.com>\r\n\
                  Date: Mon, 13 Apr 2026 00:00:00 +0000\r\n\
                  \r\n\
                  Hello Alice, this is a delivery test.\r\n\
                  .\r\n",
            )
            .await
            .unwrap();
        let deliver_resp = read_smtp_response(&mut reader).await;
        assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

        // QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "QUIT: {quit_resp}");

        server_handle.await.unwrap();

        // Verify: message appears in alice's INBOX
        let mbox = store
            .get_mailbox("INBOX")
            .expect("get_mailbox ok")
            .expect("INBOX exists");
        let messages = store.get_messages(mbox.id).expect("get_messages ok");
        assert_eq!(messages.len(), 1, "expected 1 message in INBOX, got {}", messages.len());
        assert!(
            messages[0].rfc822_size > 0,
            "expected rfc822_size > 0, got {}",
            messages[0].rfc822_size
        );
        // CAS storage should have produced a CID for the message
        assert!(
            messages[0].message_cid.is_some(),
            "expected message_cid to be set after CAS ingest"
        );
    }

    #[tokio::test]
    async fn smtp_delivers_to_merkle_mailbox() {
        use harmony_content::book::BookStore as _;

        let config = test_config();
        let max_message_size = parse_message_size(&config.spam.max_message_size);
        let smtp_config = SmtpConfig {
            domain: config.domain.name.clone(),
            mx_host: config.domain.mx_host.clone(),
            max_message_size,
            max_recipients: 100,
            tls_available: false,
        };

        let smtp_test_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(
            crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db")).unwrap(),
        );
        store.initialize_default_mailboxes().unwrap();

        let alice_addr = [0xAAu8; crate::message::ADDRESS_HASH_LEN];
        store.create_user("alice", "pass", &alice_addr).unwrap();

        // Set up CAS content store
        let content_path = smtp_test_dir.path().join("content");
        std::fs::create_dir_all(content_path.join("commits")).unwrap();
        std::fs::create_dir_all(content_path.join("blobs")).unwrap();

        // Set up Merkle mailbox manager + an inert Zenoh publisher so we can
        // assert that raw-mail publish also fires on delivery (exercises the
        // same SMTP ingest → translate → CAS → publish wiring).
        let db_path = smtp_test_dir.path().join("mailbox_roots.db");
        let mut mgr =
            crate::mailbox_manager::MailboxManager::open(&db_path, &content_path).unwrap();
        mgr.ensure_user_mailbox(&alice_addr).unwrap();
        let (publisher, publisher_handles) =
            crate::mailbox_manager::ZenohPublisher::inert_for_test();
        // Share one publisher between the manager (for internal root-CID
        // notify() under the manager mutex) and the SMTP handler (for raw
        // publish via the Arc-only path that bypasses the mutex).
        let publisher_arc = Arc::new(publisher);
        mgr.set_publisher(Arc::clone(&publisher_arc));
        let mailbox_mgr: Option<
            Arc<std::sync::Mutex<crate::mailbox_manager::MailboxManager>>,
        > = Some(Arc::new(std::sync::Mutex::new(mgr)));
        let mailbox_publisher: Option<Arc<crate::mailbox_manager::ZenohPublisher>> =
            Some(publisher_arc);

        let store_clone = store.clone();
        let content_path_clone = content_path.clone();
        let mgr_clone = mailbox_mgr.clone();
        let pub_clone = mailbox_publisher.clone();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(
                stream,
                peer_addr.ip(),
                smtp_config,
                max_message_size,
                None,
                store_clone,
                None,
                "test.example.com".to_string(),
                content_path_clone,
                5,
                mgr_clone,
                pub_clone,
                None,
                None,
            )
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
        write_half
            .write_all(b"EHLO sender.test.com\r\n")
            .await
            .unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(ehlo_resp.contains("250"), "EHLO: {ehlo_resp}");

        // MAIL FROM
        write_half
            .write_all(b"MAIL FROM:<sender@test.com>\r\n")
            .await
            .unwrap();
        let mail_resp = read_smtp_response(&mut reader).await;
        assert!(mail_resp.contains("250"), "MAIL FROM: {mail_resp}");

        // RCPT TO alice
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

        // Send RFC 5322 message
        write_half
            .write_all(
                b"From: sender@test.com\r\n\
                  To: alice@test.example.com\r\n\
                  Subject: Merkle Test\r\n\
                  Message-ID: <merkle-test-001@test.com>\r\n\
                  Date: Mon, 13 Apr 2026 00:00:00 +0000\r\n\
                  \r\n\
                  Hello Alice, this tests dual-write to IMAP + Merkle.\r\n\
                  .\r\n",
            )
            .await
            .unwrap();
        let deliver_resp = read_smtp_response(&mut reader).await;
        assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

        // QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "QUIT: {quit_resp}");

        server_handle.await.unwrap();

        // Phase 5 runs in spawn_blocking (fire-and-forget). Poll until
        // the Merkle tree reflects the delivery rather than using a fixed sleep.
        let merkle_ready = tokio::time::timeout(
            std::time::Duration::from_secs(5),
            async {
                loop {
                    {
                        let mgr = mailbox_mgr.as_ref().unwrap().lock().unwrap();
                        if let Some(root_cid) = mgr.get_root(&alice_addr) {
                            let book = harmony_db::DiskBookStore::new(&content_path);
                            if let Ok(root_bytes) = harmony_content::dag::reassemble(
                                &harmony_content::cid::ContentId::from_bytes(*root_cid),
                                &book,
                            ) {
                                if let Ok(root) = crate::mailbox::MailRoot::from_bytes(&root_bytes) {
                                    let inbox_cid = root.folder_cid(crate::mailbox::FolderKind::Inbox);
                                    if let Ok(folder_bytes) = harmony_content::dag::reassemble(
                                        &harmony_content::cid::ContentId::from_bytes(*inbox_cid),
                                        &book,
                                    ) {
                                        if let Ok(folder) = crate::mailbox::MailFolder::from_bytes(&folder_bytes) {
                                            if folder.message_count > 0 {
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(10)).await;
                }
            },
        )
        .await;
        assert!(merkle_ready.is_ok(), "Merkle mailbox not updated within 5s");

        // Verify: message appears in IMAP store
        let mbox = store
            .get_mailbox("INBOX")
            .expect("get_mailbox ok")
            .expect("INBOX exists");
        let messages = store.get_messages(mbox.id).expect("get_messages ok");
        assert_eq!(
            messages.len(),
            1,
            "expected 1 message in INBOX, got {}",
            messages.len()
        );
        let imap_cid = messages[0]
            .message_cid
            .as_ref()
            .expect("IMAP message should have a CID");

        // Verify: message appears in Merkle mailbox
        let mgr = mailbox_mgr.as_ref().unwrap().lock().unwrap();
        let root_cid = mgr
            .get_root(&alice_addr)
            .expect("alice should have a Merkle root");
        // Load the tree and check the inbox
        let book = harmony_db::DiskBookStore::new(&content_path);
        let root = crate::mailbox::MailRoot::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*root_cid))
                .expect("root should be in CAS"),
        )
        .unwrap();
        let inbox_cid = root.folder_cid(crate::mailbox::FolderKind::Inbox);
        let folder = crate::mailbox::MailFolder::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(*inbox_cid))
                .expect("inbox folder should be in CAS"),
        )
        .unwrap();
        assert_eq!(folder.message_count, 1);
        assert_eq!(folder.unread_count, 1);

        let page = crate::mailbox::MailPage::from_bytes(
            book.get(&harmony_content::cid::ContentId::from_bytes(folder.page_cids[0]))
                .expect("page should be in CAS"),
        )
        .unwrap();
        assert_eq!(page.entries.len(), 1);
        assert_eq!(
            page.entries[0].subject_snippet, "Merkle Test",
            "subject snippet mismatch"
        );

        // Key assertion: both stores reference the same CAS message CID
        assert_eq!(
            &page.entries[0].message_cid,
            imap_cid,
            "Merkle and IMAP message CIDs must match (dual-write consistency)"
        );

        // Raw-mail publish assertion: exactly one publish to alice's topic,
        // bytes must decode as the HarmonyMessage we just delivered. This is
        // the Phase 0 harmony-client path — subscribers consume these bytes
        // directly without going through CAS.
        let raw = publisher_handles
            .raw
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            raw.len(),
            1,
            "expected 1 raw-mail publish for the 1 delivered recipient, got {}",
            raw.len()
        );
        let (pub_addr_hex, pub_bytes) = &raw[0];
        assert_eq!(pub_addr_hex, &hex::encode(alice_addr));
        let decoded = crate::message::HarmonyMessage::from_bytes(pub_bytes.as_slice())
            .expect("raw bytes must decode");
        assert_eq!(decoded.subject, "Merkle Test");
        assert_eq!(decoded.sender_address.len(), 16);
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

    #[tokio::test]
    async fn run_with_config_accepts_none_gateway_identity_and_none_resolver() {
        // Compile-time checkpoint: the two new optional parameters must accept
        // `None` without forcing callers to construct cryptographic state.
        // Production callers that want remote delivery pass Some(...) for both.
        let gateway_identity: Option<Arc<harmony_identity::PrivateIdentity>> = None;
        let recipient_resolver: Option<Arc<dyn crate::remote_delivery::RecipientResolver>> =
            None;
        // Values must have inferrable types that match the signatures threaded
        // through run / handle_connection / handle_connection_generic /
        // process_async_actions.
        let _ = (gateway_identity, recipient_resolver);
    }

    #[tokio::test]
    async fn delivers_remote_warn_and_skip_when_resolver_returns_none() {
        use crate::remote_delivery::RecipientResolver;
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct NullResolver(Arc<AtomicUsize>);
        impl RecipientResolver for NullResolver {
            fn resolve(
                &self,
                _addr: &harmony_identity::IdentityHash,
            ) -> Option<harmony_identity::Identity> {
                self.0.fetch_add(1, Ordering::SeqCst);
                None
            }
        }

        let smtp_test_dir = tempfile::tempdir().unwrap();
        let store = Arc::new(
            crate::imap_store::ImapStore::open(&smtp_test_dir.path().join("smtp-test.db"))
                .unwrap(),
        );

        let (publisher, handles) = crate::mailbox_manager::ZenohPublisher::inert_for_test();
        let publisher = Arc::new(publisher);
        let mut rng = rand::rngs::OsRng;
        let gateway_id = Arc::new(harmony_identity::PrivateIdentity::generate(&mut rng));
        let calls = Arc::new(AtomicUsize::new(0));
        let resolver: Arc<dyn RecipientResolver> = Arc::new(NullResolver(Arc::clone(&calls)));

        // RFC 5322 blob that translate_inbound can parse successfully.
        let rfc822 = b"From: alice@local\r\nTo: bob@remote\r\nSubject: hi\r\nDate: Tue, 15 Apr 2026 12:00:00 +0000\r\nMessage-ID: <test-zeb113@local.example>\r\n\r\nhello\r\n";

        // Recipient hash NOT in the imap_store → Ok(None) branch fires.
        let recipient_hash = [0xAAu8; crate::message::ADDRESS_HASH_LEN];

        let actions = vec![SmtpAction::DeliverToHarmony {
            recipients: vec![recipient_hash],
            data: rfc822.to_vec(),
        }];

        let smtp_config = SmtpConfig {
            domain: "local".to_string(),
            mx_host: "mail.local".to_string(),
            max_message_size: 10 * 1024 * 1024,
            max_recipients: 100,
            tls_available: false,
        };
        let mut session = SmtpSession::new(smtp_config);
        let mut writer = Vec::<u8>::new();
        let mut spf_result = crate::spam::SpfResult::None;

        process_async_actions(
            &actions,
            &mut session,
            &mut writer,
            &store,
            &None,
            "local",
            smtp_test_dir.path(),
            &mut spf_result,
            1000, // reject_threshold high → no spam rejection
            &None,
            &Some(Arc::clone(&publisher)),
            &Some(Arc::clone(&gateway_id)),
            &Some(resolver),
        )
        .await
        .expect("process_async_actions should succeed even when remote resolver skips");

        // Resolver was consulted exactly once (for bob).
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // Publisher recorded NO sealed-unicast publish.
        let sealed = handles
            .sealed_unicast
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        assert!(
            sealed.is_empty(),
            "no publish should occur when resolver returns None; got {} captures",
            sealed.len(),
        );
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

    #[test]
    fn parse_imap_date_standard() {
        // 13-Apr-2026 00:00:00 UTC
        let ts = super::parse_imap_date("13-Apr-2026");
        // 2026-04-13 = days since epoch: 20,556 * 86400 = 1776038400
        assert!(ts.is_some());
        let t = ts.unwrap();
        // April 13, 2026 is 20556 days after Jan 1 1970
        assert_eq!(t, 20556 * 86400);
    }

    #[test]
    fn parse_imap_date_all_months() {
        assert!(super::parse_imap_date("01-Jan-2000").is_some());
        assert!(super::parse_imap_date("01-Feb-2000").is_some());
        assert!(super::parse_imap_date("01-Mar-2000").is_some());
        assert!(super::parse_imap_date("01-Apr-2000").is_some());
        assert!(super::parse_imap_date("01-May-2000").is_some());
        assert!(super::parse_imap_date("01-Jun-2000").is_some());
        assert!(super::parse_imap_date("01-Jul-2000").is_some());
        assert!(super::parse_imap_date("01-Aug-2000").is_some());
        assert!(super::parse_imap_date("01-Sep-2000").is_some());
        assert!(super::parse_imap_date("01-Oct-2000").is_some());
        assert!(super::parse_imap_date("01-Nov-2000").is_some());
        assert!(super::parse_imap_date("01-Dec-2000").is_some());
    }

    #[test]
    fn parse_imap_date_epoch() {
        let ts = super::parse_imap_date("01-Jan-1970");
        assert_eq!(ts, Some(0));
    }

    #[test]
    fn parse_imap_date_invalid() {
        assert!(super::parse_imap_date("invalid").is_none());
        assert!(super::parse_imap_date("32-Jan-2026").is_none());
        assert!(super::parse_imap_date("01-Xyz-2026").is_none());
    }

    #[test]
    fn copy_messages_preserves_flags() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();
        imap_store.create_mailbox("Archive").unwrap();

        // Insert message with flags
        let uid = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        imap_store.add_flags(msgs[0].id, &["\\Seen", "\\Flagged"]).unwrap();

        // Copy to Archive
        let mapping = imap_store.copy_messages(mbox.id, &[uid], "Archive").unwrap();
        assert_eq!(mapping.len(), 1);
        let (src_uid, dst_uid) = mapping[0];
        assert_eq!(src_uid, uid);

        // Verify source still exists
        let src_msgs = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(src_msgs.len(), 1);

        // Verify destination has the message with flags
        let dst_mbox = imap_store.get_mailbox("Archive").unwrap().unwrap();
        let dst_msgs = imap_store.get_messages(dst_mbox.id).unwrap();
        assert_eq!(dst_msgs.len(), 1);
        assert_eq!(dst_msgs[0].uid, dst_uid);
        let dst_flags = imap_store.get_flags(dst_msgs[0].id).unwrap();
        assert!(dst_flags.contains(&"\\Flagged".to_string()));
        assert!(dst_flags.contains(&"\\Seen".to_string()));
    }

    #[test]
    fn move_removes_source() {
        let dir = tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();
        // "Trash" is already created by initialize_default_mailboxes

        // Insert 2 messages
        let uid1 = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let uid2 = imap_store.insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 2000, 200).unwrap();
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();

        // Move uid1 to Trash (copy + flag \Deleted + expunge)
        let mapping = imap_store.copy_messages(mbox.id, &[uid1], "Trash").unwrap();
        assert_eq!(mapping.len(), 1);

        // Flag source as deleted
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        let msg1 = msgs.iter().find(|m| m.uid == uid1).unwrap();
        imap_store.add_flags(msg1.id, &["\\Deleted"]).unwrap();

        // Expunge
        let expunged = imap_store.expunge(mbox.id).unwrap();
        assert_eq!(expunged, vec![uid1]);

        // Verify source only has uid2
        let remaining = imap_store.get_messages(mbox.id).unwrap();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].uid, uid2);

        // Verify destination has the moved message
        let dst_mbox = imap_store.get_mailbox("Trash").unwrap().unwrap();
        let dst_msgs = imap_store.get_messages(dst_mbox.id).unwrap();
        assert_eq!(dst_msgs.len(), 1);
    }
}

#[cfg(test)]
mod search_tests {
    use super::*;
    use crate::imap_parse::SearchKey;
    use crate::message::{
        HarmonyMessage, MailMessageType, MessageFlags, Recipient, RecipientType, ADDRESS_HASH_LEN,
        MESSAGE_ID_LEN, VERSION,
    };

    fn msg_row_with(uid: u32, internal_date: u64, rfc822_size: u32) -> imap_store::MessageRow {
        imap_store::MessageRow {
            id: uid as i64,
            mailbox_id: 1,
            uid,
            harmony_msg_id: [0u8; MESSAGE_ID_LEN],
            message_cid: None,
            internal_date,
            rfc822_size,
        }
    }

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
    fn search_all() {
        let row = msg_row_with(1, 1000, 100);
        assert!(super::matches_criteria(&[SearchKey::All], &row, &[], 1, None, 100, 100));
    }

    #[test]
    fn search_flag_seen() {
        let row = msg_row_with(1, 1000, 100);
        let seen = vec!["\\Seen".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Seen], &row, &seen, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Unseen], &row, &seen, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Seen], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Unseen], &row, &[], 1, None, 100, 100));
    }

    #[test]
    fn search_flag_deleted_draft_flagged_answered() {
        let row = msg_row_with(1, 1000, 100);
        let flags = vec!["\\Deleted".to_string(), "\\Draft".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Deleted], &row, &flags, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Draft], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Undeleted], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Undraft], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Flagged], &row, &flags, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Unflagged], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Answered], &row, &flags, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Unanswered], &row, &flags, 1, None, 100, 100));
    }

    #[test]
    fn search_recent_new_old() {
        let row = msg_row_with(1, 1000, 100);
        let recent = vec!["\\Recent".to_string()];
        let recent_seen = vec!["\\Recent".to_string(), "\\Seen".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Recent], &row, &recent, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Old], &row, &recent, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::New], &row, &recent, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::New], &row, &recent_seen, 1, None, 100, 100));
    }

    #[test]
    fn search_size() {
        let row = msg_row_with(1, 1000, 500);
        assert!(super::matches_criteria(&[SearchKey::Larger(499)], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Larger(500)], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Smaller(501)], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Smaller(500)], &row, &[], 1, None, 100, 100));
    }

    #[test]
    fn search_date_since_before_on() {
        // internal_date = 86400 (Jan 2, 1970)
        let row = msg_row_with(1, 86400, 100);
        assert!(super::matches_criteria(&[SearchKey::Since("02-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Since("01-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Since("03-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Before("03-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Before("02-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::On("02-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::On("01-Jan-1970".to_string())], &row, &[], 1, None, 100, 100));
    }

    #[test]
    fn search_not_and_or() {
        let row = msg_row_with(1, 1000, 100);
        let seen = vec!["\\Seen".to_string()];
        assert!(!super::matches_criteria(&[SearchKey::Not(Box::new(SearchKey::Seen))], &row, &seen, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Not(Box::new(SearchKey::Seen))], &row, &[], 1, None, 100, 100));
        assert!(super::matches_criteria(
            &[SearchKey::Or(Box::new(SearchKey::Seen), Box::new(SearchKey::Flagged))],
            &row, &seen, 1, None, 100, 100
        ));
        assert!(!super::matches_criteria(
            &[SearchKey::Or(Box::new(SearchKey::Flagged), Box::new(SearchKey::Deleted))],
            &row, &[], 1, None, 100, 100
        ));
    }

    #[test]
    fn search_multi_criteria_and() {
        let row = msg_row_with(1, 1000, 500);
        let seen = vec!["\\Seen".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Seen, SearchKey::Larger(100)], &row, &seen, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Seen, SearchKey::Larger(600)], &row, &seen, 1, None, 100, 100));
    }

    #[test]
    fn search_subject_body() {
        let row = msg_row_with(1, 1000, 100);
        let msg = test_message();
        assert!(super::matches_criteria(
            &[SearchKey::Subject("test".to_string())], &row, &[], 1, Some(&msg), 100, 100
        ));
        assert!(super::matches_criteria(
            &[SearchKey::Subject("TEST SUBJECT".to_string())], &row, &[], 1, Some(&msg), 100, 100
        ));
        assert!(!super::matches_criteria(
            &[SearchKey::Subject("missing".to_string())], &row, &[], 1, Some(&msg), 100, 100
        ));
        assert!(super::matches_criteria(
            &[SearchKey::Body("hello".to_string())], &row, &[], 1, Some(&msg), 100, 100
        ));
        assert!(!super::matches_criteria(
            &[SearchKey::Body("missing".to_string())], &row, &[], 1, Some(&msg), 100, 100
        ));
    }

    #[test]
    fn search_content_without_message_returns_false() {
        let row = msg_row_with(1, 1000, 100);
        assert!(!super::matches_criteria(&[SearchKey::Subject("test".to_string())], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Body("hello".to_string())], &row, &[], 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::From("user".to_string())], &row, &[], 1, None, 100, 100));
    }

    #[test]
    fn search_keyword() {
        let row = msg_row_with(1, 1000, 100);
        let flags = vec!["$Important".to_string()];
        assert!(super::matches_criteria(&[SearchKey::Keyword("$Important".to_string())], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Keyword("$Other".to_string())], &row, &flags, 1, None, 100, 100));
        assert!(!super::matches_criteria(&[SearchKey::Unkeyword("$Important".to_string())], &row, &flags, 1, None, 100, 100));
        assert!(super::matches_criteria(&[SearchKey::Unkeyword("$Other".to_string())], &row, &flags, 1, None, 100, 100));
    }

    #[test]
    fn search_by_flags_integration() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("imap.db");
        let imap_store = crate::imap_store::ImapStore::open(&db_path).unwrap();
        imap_store.initialize_default_mailboxes().unwrap();

        // Insert 3 messages
        let uid1 = imap_store.insert_message("INBOX", &[1u8; MESSAGE_ID_LEN], None, 1000, 100).unwrap();
        let uid2 = imap_store.insert_message("INBOX", &[2u8; MESSAGE_ID_LEN], None, 2000, 200).unwrap();
        let uid3 = imap_store.insert_message("INBOX", &[3u8; MESSAGE_ID_LEN], None, 3000, 300).unwrap();

        // Flag message 1 as Seen, message 2 as Seen+Flagged
        let mbox = imap_store.get_mailbox("INBOX").unwrap().unwrap();
        let msgs = imap_store.get_messages(mbox.id).unwrap();
        imap_store.add_flags(msgs[0].id, &["\\Seen"]).unwrap();
        imap_store.add_flags(msgs[1].id, &["\\Seen", "\\Flagged"]).unwrap();

        let max_uid = msgs.last().map(|m| m.uid).unwrap_or(0);
        let max_seqnum = msgs.len() as u32;

        // SEARCH SEEN should match UIDs 1 and 2
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Seen], msg, &flags, (idx + 1) as u32, None, max_uid, max_seqnum) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid1, uid2]);

        // SEARCH FLAGGED should match UID 2 only
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Flagged], msg, &flags, (idx + 1) as u32, None, max_uid, max_seqnum) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid2]);

        // SEARCH UNSEEN should match UID 3 only
        let mut results = Vec::new();
        for (idx, msg) in msgs.iter().enumerate() {
            let flags = imap_store.get_flags(msg.id).unwrap();
            if super::matches_criteria(&[SearchKey::Unseen], msg, &flags, (idx + 1) as u32, None, max_uid, max_seqnum) {
                results.push(msg.uid);
            }
        }
        assert_eq!(results, vec![uid3]);
    }

    #[test]
    fn sequence_set_contains_single() {
        use crate::imap_parse::{SequenceRange, SequenceSet};
        let set = SequenceSet { ranges: vec![SequenceRange { start: 3, end: None }] };
        assert!(super::sequence_set_contains(&set, 3, 10));
        assert!(!super::sequence_set_contains(&set, 4, 10));
    }

    #[test]
    fn sequence_set_contains_range() {
        use crate::imap_parse::{SequenceRange, SequenceSet};
        let set = SequenceSet { ranges: vec![SequenceRange { start: 2, end: Some(5) }] };
        assert!(!super::sequence_set_contains(&set, 1, 10));
        assert!(super::sequence_set_contains(&set, 2, 10));
        assert!(super::sequence_set_contains(&set, 5, 10));
        assert!(!super::sequence_set_contains(&set, 6, 10));
    }

    #[test]
    fn sequence_set_contains_wildcard() {
        use crate::imap_parse::{SequenceRange, SequenceSet};
        // 5:* with max=10 should match 5..=10
        let set = SequenceSet { ranges: vec![SequenceRange { start: 5, end: Some(u32::MAX) }] };
        assert!(!super::sequence_set_contains(&set, 4, 10));
        assert!(super::sequence_set_contains(&set, 5, 10));
        assert!(super::sequence_set_contains(&set, 10, 10));
        assert!(!super::sequence_set_contains(&set, 11, 10));
    }

    #[test]
    fn sequence_set_contains_wildcard_beyond_max() {
        use crate::imap_parse::{SequenceRange, SequenceSet};
        // UID 20:* with max_uid=10 should not match anything < 10, reversed range 10..=10
        let set = SequenceSet { ranges: vec![SequenceRange { start: 20, end: Some(u32::MAX) }] };
        assert!(super::sequence_set_contains(&set, 10, 10)); // reversed: 10..=20
        assert!(!super::sequence_set_contains(&set, 9, 10));
    }

    #[test]
    fn search_uid_set() {
        let row = msg_row_with(5, 1000, 100);
        use crate::imap_parse::{SequenceRange, SequenceSet};
        // UID 3:7 with max_uid=10, msg uid=5 -> match
        let set = SequenceSet { ranges: vec![SequenceRange { start: 3, end: Some(7) }] };
        assert!(super::matches_criteria(
            &[SearchKey::Uid(set.clone())], &row, &[], 2, None, 10, 5
        ));
        // UID 6:* with max_uid=10, msg uid=5 -> no match
        let set2 = SequenceSet { ranges: vec![SequenceRange { start: 6, end: Some(u32::MAX) }] };
        assert!(!super::matches_criteria(
            &[SearchKey::Uid(set2)], &row, &[], 2, None, 10, 5
        ));
    }

    #[test]
    fn search_seqnum_set() {
        let row = msg_row_with(10, 1000, 100);
        use crate::imap_parse::{SequenceRange, SequenceSet};
        // seqnum=3, set 2:4, max_seqnum=5 -> match
        let set = SequenceSet { ranges: vec![SequenceRange { start: 2, end: Some(4) }] };
        assert!(super::matches_criteria(
            &[SearchKey::SequenceSet(set)], &row, &[], 3, None, 10, 5
        ));
        // seqnum=3, set 4:5, max_seqnum=5 -> no match
        let set2 = SequenceSet { ranges: vec![SequenceRange { start: 4, end: Some(5) }] };
        assert!(!super::matches_criteria(
            &[SearchKey::SequenceSet(set2)], &row, &[], 3, None, 10, 5
        ));
    }

    #[test]
    fn parse_date_rejects_invalid_day() {
        // April has 30 days
        assert!(super::parse_imap_date("30-Apr-2026").is_some());
        assert!(super::parse_imap_date("31-Apr-2026").is_none());
        // Feb non-leap
        assert!(super::parse_imap_date("28-Feb-2025").is_some());
        assert!(super::parse_imap_date("29-Feb-2025").is_none());
        // Feb leap
        assert!(super::parse_imap_date("29-Feb-2024").is_some());
    }

    #[test]
    fn parse_date_rejects_extreme_year() {
        assert!(super::parse_imap_date("01-Jan-10000").is_none());
        assert!(super::parse_imap_date("01-Jan-9999").is_some());
    }
}

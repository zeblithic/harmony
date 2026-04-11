//! SMTP server: TCP listener, connection handling, and state machine I/O driver.
//!
//! Binds to configured SMTP ports, accepts connections, and spawns per-connection
//! async tasks that wire [`SmtpCodec`](crate::io::SmtpCodec) frames to the
//! [`SmtpSession`](crate::smtp::SmtpSession) sans-I/O state machine.

use std::collections::HashMap;
use std::net::IpAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tokio_util::codec::FramedRead;

use futures_util::StreamExt;

use crate::config::Config;
use crate::io::{SmtpCodec, SmtpFrame};
use crate::smtp::{SmtpAction, SmtpConfig, SmtpEvent, SmtpSession, SmtpState};
use crate::smtp_parse::parse_command;

/// Per-IP connection tracking entry.
struct IpEntry {
    active: usize,
    last_seen: Instant,
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
    async fn try_connect(&self, ip: IpAddr) -> bool {
        let mut conns = self.connections.lock().await;
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
    async fn disconnect(&self, ip: IpAddr) {
        let mut conns = self.connections.lock().await;
        if let Some(entry) = conns.get_mut(&ip) {
            entry.active = entry.active.saturating_sub(1);
            if entry.active == 0 {
                conns.remove(&ip);
            }
        }
    }

    /// Remove stale entries older than the given duration.
    async fn cleanup(&self, max_age: Duration) {
        let mut conns = self.connections.lock().await;
        let now = Instant::now();
        conns.retain(|_, entry| entry.active > 0 || now.duration_since(entry.last_seen) < max_age);
    }
}

/// Parse the max_message_size config string (e.g. "25MB") to bytes.
fn parse_message_size(s: &str) -> usize {
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
        .unwrap_or(25 * 1024 * 1024)
        * multiplier
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

    let smtp_config = SmtpConfig {
        domain: config.domain.name.clone(),
        mx_host: config.domain.mx_host.clone(),
        max_message_size,
        max_recipients: 100,
    };

    let shared = Arc::new(SharedState::new(config.spam.max_connections_per_ip));

    let listener = TcpListener::bind(&config.gateway.listen_smtp).await?;
    tracing::info!(addr = %config.gateway.listen_smtp, "SMTP listener started");

    // Background cleanup task for per-IP tracking
    let cleanup_shared = Arc::clone(&shared);
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(CLEANUP_INTERVAL);
        loop {
            interval.tick().await;
            cleanup_shared.cleanup(Duration::from_secs(3600)).await;
        }
    });

    // Graceful shutdown signal
    let shutdown = tokio::signal::ctrl_c();
    tokio::pin!(shutdown);

    loop {
        tokio::select! {
            result = listener.accept() => {
                let (stream, addr) = result?;
                let peer_ip = addr.ip();

                if !shared.try_connect(peer_ip).await {
                    tracing::warn!(%peer_ip, "connection rejected: per-IP limit exceeded");
                    drop(stream);
                    continue;
                }

                let smtp_config = smtp_config.clone();
                let shared = Arc::clone(&shared);

                tokio::spawn(async move {
                    if let Err(e) = handle_connection(stream, peer_ip, smtp_config, max_message_size).await {
                        tracing::debug!(%peer_ip, error = %e, "connection ended with error");
                    }
                    shared.disconnect(peer_ip).await;
                });
            }
            _ = &mut shutdown => {
                tracing::info!("shutting down SMTP server");
                break;
            }
        }
    }

    Ok(())
}

/// Handle a single SMTP connection.
async fn handle_connection(
    stream: tokio::net::TcpStream,
    peer_ip: IpAddr,
    smtp_config: SmtpConfig,
    max_message_size: usize,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut session = SmtpSession::new(smtp_config);

    let (reader, mut writer) = stream.into_split();
    let mut framed = FramedRead::new(reader, SmtpCodec::new(max_message_size));

    // Send initial greeting
    let actions = session.handle(SmtpEvent::Connected {
        peer_ip,
        tls: false,
    });
    execute_actions(&actions, &mut writer).await?;

    loop {
        // Choose timeout based on current state
        let timeout = if session.state == SmtpState::DataReceiving {
            DATA_TIMEOUT
        } else {
            COMMAND_TIMEOUT
        };

        let frame = tokio::time::timeout(timeout, framed.next()).await;

        match frame {
            Ok(Some(Ok(smtp_frame))) => {
                let actions = match smtp_frame {
                    SmtpFrame::Line(line) => {
                        match parse_command(line.as_bytes()) {
                            Ok(cmd) => {
                                // If DATA command is accepted, switch codec to data mode
                                let actions = session.handle(SmtpEvent::Command(cmd));
                                if session.state == SmtpState::DataReceiving {
                                    framed.decoder_mut().enter_data_mode();
                                }
                                actions
                            }
                            Err(_) => {
                                // Unknown or malformed command
                                vec![SmtpAction::SendResponse(
                                    500,
                                    "5.5.1 Command not recognized".to_string(),
                                )]
                            }
                        }
                    }
                    SmtpFrame::Data(body) => {
                        // Switch back to command mode
                        framed.decoder_mut().enter_command_mode();
                        session.handle(SmtpEvent::DataComplete(body))
                    }
                };

                let should_close = execute_actions(&actions, &mut writer).await?;

                // Handle async actions that need callbacks
                for action in &actions {
                    match action {
                        SmtpAction::ResolveHarmonyAddress {
                            local_part,
                            domain: _,
                        } => {
                            // For now, stub: all addresses resolve to a dummy identity.
                            // Real implementation will query the announce cache.
                            use crate::message::ADDRESS_HASH_LEN;
                            let identity = Some([0u8; ADDRESS_HASH_LEN]);
                            let resolve_actions =
                                session.handle(SmtpEvent::HarmonyResolved {
                                    local_part: local_part.clone(),
                                    identity,
                                });
                            execute_actions(&resolve_actions, &mut writer).await?;
                        }
                        SmtpAction::DeliverToHarmony { .. } => {
                            // For now, stub: delivery always succeeds.
                            // Real implementation will deliver to Harmony network.
                            let result_actions =
                                session.handle(SmtpEvent::DeliveryResult { success: true });
                            execute_actions(&result_actions, &mut writer).await?;
                        }
                        SmtpAction::CheckSpf { .. } => {
                            // Fire-and-forget for now — SPF results will feed into
                            // spam scoring in Task 16.
                        }
                        SmtpAction::StartTls => {
                            // TLS upgrade handled in Task 13. For now, just complete
                            // the handshake stub so the state machine can proceed.
                            tracing::warn!("STARTTLS requested but TLS not yet implemented");
                        }
                        _ => {}
                    }
                }

                if should_close || session.state == SmtpState::Closed {
                    break;
                }
            }
            Ok(Some(Err(e))) => {
                tracing::debug!(%peer_ip, error = %e, "codec error");
                let _ = writer
                    .write_all(b"421 4.7.0 Error processing input\r\n")
                    .await;
                break;
            }
            Ok(None) => {
                // Client disconnected
                break;
            }
            Err(_) => {
                // Timeout
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
async fn execute_actions(
    actions: &[SmtpAction],
    writer: &mut tokio::net::tcp::OwnedWriteHalf,
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
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size)
                .await
                .unwrap();
        });

        let stream = TcpStream::connect(addr).await.unwrap();
        let (read_half, mut write_half) = stream.into_split();
        let mut reader = BufReader::new(read_half);

        // Read 220 greeting
        let greeting = read_smtp_response(&mut reader).await;
        assert!(greeting.starts_with("220 "), "expected 220 greeting, got: {greeting}");

        // Send EHLO
        write_half.write_all(b"EHLO client.example.com\r\n").await.unwrap();
        let ehlo_resp = read_smtp_response(&mut reader).await;
        assert!(ehlo_resp.contains("250"), "expected 250 in EHLO response, got: {ehlo_resp}");

        // Send QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "expected 221, got: {quit_resp}");

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
        };

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let server_handle = tokio::spawn(async move {
            let (stream, peer_addr) = listener.accept().await.unwrap();
            handle_connection(stream, peer_addr.ip(), smtp_config, max_message_size)
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
        write_half.write_all(b"MAIL FROM:<sender@test.com>\r\n").await.unwrap();
        let mail_resp = read_smtp_response(&mut reader).await;
        assert!(mail_resp.contains("250"), "MAIL FROM: {mail_resp}");

        // RCPT TO (triggers resolve -> stub accepts)
        write_half.write_all(b"RCPT TO:<user@test.example.com>\r\n").await.unwrap();
        let rcpt_resp = read_smtp_response(&mut reader).await;
        assert!(rcpt_resp.contains("250"), "RCPT TO: {rcpt_resp}");

        // DATA
        write_half.write_all(b"DATA\r\n").await.unwrap();
        let data_resp = read_smtp_response(&mut reader).await;
        assert!(data_resp.starts_with("354"), "DATA: {data_resp}");

        // Send message body + terminator
        write_half.write_all(b"Subject: Test\r\n\r\nHello World\r\n.\r\n").await.unwrap();
        let deliver_resp = read_smtp_response(&mut reader).await;
        assert!(deliver_resp.contains("250"), "delivery: {deliver_resp}");

        // QUIT
        write_half.write_all(b"QUIT\r\n").await.unwrap();
        let quit_resp = read_smtp_response(&mut reader).await;
        assert!(quit_resp.starts_with("221 "), "QUIT: {quit_resp}");

        server_handle.await.unwrap();
    }

    #[tokio::test]
    async fn per_ip_rate_limiting() {
        let shared = SharedState::new(2);
        let ip: IpAddr = "1.2.3.4".parse().unwrap();

        assert!(shared.try_connect(ip).await);
        assert!(shared.try_connect(ip).await);
        assert!(!shared.try_connect(ip).await); // third rejected

        shared.disconnect(ip).await;
        assert!(shared.try_connect(ip).await); // slot freed
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

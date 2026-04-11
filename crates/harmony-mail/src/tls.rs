//! TLS integration for SMTP.
//!
//! Provides certificate loading, `TlsAcceptor` construction, and helpers
//! for STARTTLS upgrade and implicit TLS connections.

use std::io;
use std::path::Path;
use std::sync::Arc;

use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

/// Load a TLS server configuration from PEM certificate and key files.
///
/// Returns a `TlsAcceptor` ready for use with `tokio-rustls`.
pub fn load_tls_config(cert_path: &Path, key_path: &Path) -> Result<TlsAcceptor, TlsError> {
    let certs = load_certs(cert_path)?;
    let key = load_private_key(key_path)?;

    // Ensure a default crypto provider is installed (idempotent if already set)
    let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)
        .map_err(|e| TlsError::Config(e.to_string()))?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}

/// Load PEM-encoded certificates from a file.
fn load_certs(
    path: &Path,
) -> Result<Vec<rustls::pki_types::CertificateDer<'static>>, TlsError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TlsError::Io(format!("failed to open cert file {}: {}", path.display(), e)))?;
    let mut reader = io::BufReader::new(file);

    let certs: Vec<_> = rustls_pemfile::certs(&mut reader)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| TlsError::Io(format!("failed to parse certs from {}: {}", path.display(), e)))?;

    if certs.is_empty() {
        return Err(TlsError::NoCerts(path.display().to_string()));
    }

    Ok(certs)
}

/// Load a PEM-encoded private key from a file.
/// Supports PKCS#8, RSA, and EC private keys.
fn load_private_key(
    path: &Path,
) -> Result<rustls::pki_types::PrivateKeyDer<'static>, TlsError> {
    let file = std::fs::File::open(path)
        .map_err(|e| TlsError::Io(format!("failed to open key file {}: {}", path.display(), e)))?;
    let mut reader = io::BufReader::new(file);

    // Try all key formats
    for item in rustls_pemfile::read_all(&mut reader) {
        match item {
            Ok(rustls_pemfile::Item::Pkcs8Key(key)) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Pkcs8(key));
            }
            Ok(rustls_pemfile::Item::Pkcs1Key(key)) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Pkcs1(key));
            }
            Ok(rustls_pemfile::Item::Sec1Key(key)) => {
                return Ok(rustls::pki_types::PrivateKeyDer::Sec1(key));
            }
            Ok(_) => continue,
            Err(e) => {
                return Err(TlsError::Io(format!(
                    "failed to parse key from {}: {}",
                    path.display(),
                    e
                )));
            }
        }
    }

    Err(TlsError::NoKey(path.display().to_string()))
}

/// Upgrade a plain TCP stream to TLS using STARTTLS.
///
/// Takes ownership of the TCP stream, performs the TLS handshake, and returns
/// the encrypted stream. The caller should rebuild its `FramedRead` over the
/// returned `TlsStream` and feed `SmtpEvent::TlsCompleted` to the state machine.
pub async fn starttls_upgrade(
    stream: tokio::net::TcpStream,
    acceptor: &TlsAcceptor,
) -> Result<tokio_rustls::server::TlsStream<tokio::net::TcpStream>, TlsError> {
    acceptor
        .accept(stream)
        .await
        .map_err(|e| TlsError::Handshake(e.to_string()))
}

/// Wrap a TCP stream in implicit TLS (for port 465).
///
/// Same as `starttls_upgrade` but used for connections that start with TLS
/// from the beginning (no STARTTLS negotiation needed).
pub async fn implicit_tls_wrap(
    stream: tokio::net::TcpStream,
    acceptor: &TlsAcceptor,
) -> Result<tokio_rustls::server::TlsStream<tokio::net::TcpStream>, TlsError> {
    // Same operation — TLS handshake on accept
    starttls_upgrade(stream, acceptor).await
}

/// TLS-related errors.
#[derive(Debug, thiserror::Error)]
pub enum TlsError {
    #[error("I/O error: {0}")]
    Io(String),

    #[error("no certificates found in {0}")]
    NoCerts(String),

    #[error("no private key found in {0}")]
    NoKey(String),

    #[error("TLS configuration error: {0}")]
    Config(String),

    #[error("TLS handshake failed: {0}")]
    Handshake(String),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Generate a self-signed certificate and key for testing.
    /// Returns (cert_pem, key_pem) as byte vectors.
    fn generate_self_signed() -> (Vec<u8>, Vec<u8>) {
        use rcgen::CertifiedKey;
        let CertifiedKey { cert, key_pair } =
            rcgen::generate_simple_self_signed(vec!["localhost".to_string()]).unwrap();
        (cert.pem().into_bytes(), key_pair.serialize_pem().into_bytes())
    }

    fn write_temp_file(data: &[u8]) -> NamedTempFile {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn load_tls_config_with_self_signed() {
        let (cert_pem, key_pem) = generate_self_signed();
        let cert_file = write_temp_file(&cert_pem);
        let key_file = write_temp_file(&key_pem);

        let acceptor = load_tls_config(cert_file.path(), key_file.path());
        assert!(acceptor.is_ok(), "failed to load TLS config: {:?}", acceptor.err());
    }

    #[test]
    fn load_tls_config_missing_cert_file() {
        let (_, key_pem) = generate_self_signed();
        let key_file = write_temp_file(&key_pem);

        let result = load_tls_config(Path::new("/nonexistent/cert.pem"), key_file.path());
        assert!(matches!(result, Err(TlsError::Io(_))));
    }

    #[test]
    fn load_tls_config_empty_cert_file() {
        let (_, key_pem) = generate_self_signed();
        let cert_file = write_temp_file(b"");
        let key_file = write_temp_file(&key_pem);

        let result = load_tls_config(cert_file.path(), key_file.path());
        assert!(matches!(result, Err(TlsError::NoCerts(_))));
    }

    #[test]
    fn load_tls_config_empty_key_file() {
        let (cert_pem, _) = generate_self_signed();
        let cert_file = write_temp_file(&cert_pem);
        let key_file = write_temp_file(b"");

        let result = load_tls_config(cert_file.path(), key_file.path());
        assert!(matches!(result, Err(TlsError::NoKey(_))));
    }

    #[tokio::test]
    async fn starttls_upgrade_with_self_signed() {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        use tokio::net::TcpListener;

        let (cert_pem, key_pem) = generate_self_signed();
        let cert_file = write_temp_file(&cert_pem);
        let key_file = write_temp_file(&key_pem);
        let acceptor = load_tls_config(cert_file.path(), key_file.path()).unwrap();

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        // Server: accept TCP, upgrade to TLS, read a message
        let server_acceptor = acceptor.clone();
        let server = tokio::spawn(async move {
            let (stream, _) = listener.accept().await.unwrap();
            let mut tls_stream = starttls_upgrade(stream, &server_acceptor).await.unwrap();
            let mut buf = vec![0u8; 256];
            let n = tls_stream.read(&mut buf).await.unwrap();
            String::from_utf8_lossy(&buf[..n]).to_string()
        });

        // Client: connect, upgrade to TLS, send a message
        let client = tokio::spawn(async move {
            let tcp = tokio::net::TcpStream::connect(addr).await.unwrap();

            // Build a client TLS config that trusts our self-signed cert
            let _ = rustls::crypto::aws_lc_rs::default_provider().install_default();
            let mut root_store = rustls::RootCertStore::empty();
            let certs = load_certs(cert_file.path()).unwrap();
            for cert in &certs {
                root_store.add(cert.clone()).unwrap();
            }
            let client_config = rustls::ClientConfig::builder()
                .with_root_certificates(root_store)
                .with_no_client_auth();
            let connector = tokio_rustls::TlsConnector::from(Arc::new(client_config));
            let server_name = rustls::pki_types::ServerName::try_from("localhost").unwrap();

            let mut tls_stream = connector.connect(server_name, tcp).await.unwrap();
            tls_stream.write_all(b"EHLO secure.test\r\n").await.unwrap();
            tls_stream.flush().await.unwrap();
        });

        client.await.unwrap();
        let received = server.await.unwrap();
        assert_eq!(received, "EHLO secure.test\r\n");
    }
}

//! Mock pkarr-relay HTTP server for testing.
//!
//! Implements the minimal subset of the BEP44-over-HTTP relay protocol
//! that real pkarr relays expose:
//! - `PUT /{z32_public_key}` — store a BEP44 envelope (signed payload).
//! - `GET /{z32_public_key}` — retrieve the most recent envelope.
//!
//! Storage is in-memory `HashMap<String, Vec<u8>>`. No persistence.
//! No DHT-style propagation. Latest write wins.
//!
//! Spawn with `MockPkarrRelay::start().await` → returns a `MockPkarrRelay`
//! whose `base_url` field points at the listening address.
//! Drop the handle to shut down the server.

#![cfg(any(test, feature = "test-fixtures"))]

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::sync::RwLock;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    routing::put,
    Router,
};

/// In-memory BEP44 store. Key is the z-base-32-encoded public key string
/// (as it appears in the URL path). Value is the raw envelope bytes
/// (whatever the client PUT).
type Store = Arc<RwLock<HashMap<String, Vec<u8>>>>;

/// Handle to a running mock relay. Drop to stop the server.
pub struct MockPkarrRelay {
    /// The `http://host:port` base URL to use in relay-client configuration.
    pub base_url: String,
    _shutdown: tokio::sync::oneshot::Sender<()>,
    _server_task: tokio::task::JoinHandle<()>,
}

impl MockPkarrRelay {
    /// Start a fresh mock relay on a random localhost port.
    pub async fn start() -> Self {
        let store: Store = Arc::new(RwLock::new(HashMap::new()));
        let app = Router::new()
            .route("/{key}", put(put_record).get(get_record))
            .with_state(store);

        let listener = tokio::net::TcpListener::bind(SocketAddr::from(([127, 0, 0, 1], 0)))
            .await
            .expect("bind to 127.0.0.1:0");
        let addr = listener.local_addr().expect("local_addr");
        let base_url = format!("http://{}", addr);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel();
        let server_task = tokio::spawn(async move {
            axum::serve(listener, app)
                .with_graceful_shutdown(async move {
                    let _ = shutdown_rx.await;
                })
                .await
                .expect("axum serve");
        });

        Self {
            base_url,
            _shutdown: shutdown_tx,
            _server_task: server_task,
        }
    }
}

async fn put_record(
    Path(key): Path<String>,
    State(store): State<Store>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    store.write().await.insert(key, body.to_vec());
    StatusCode::OK
}

async fn get_record(Path(key): Path<String>, State(store): State<Store>) -> impl IntoResponse {
    match store.read().await.get(&key) {
        Some(bytes) => (StatusCode::OK, bytes.clone()).into_response(),
        None => StatusCode::NOT_FOUND.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trip_put_then_get() {
        let relay = MockPkarrRelay::start().await;
        let client = reqwest::Client::new();
        let put_resp = client
            .put(format!("{}/abc123", relay.base_url))
            .body(b"hello".to_vec())
            .send()
            .await
            .expect("put request");
        assert_eq!(put_resp.status(), 200);

        let get_resp = client
            .get(format!("{}/abc123", relay.base_url))
            .send()
            .await
            .expect("get request");
        assert_eq!(get_resp.status(), 200);
        assert_eq!(
            get_resp.bytes().await.expect("body bytes").as_ref(),
            b"hello"
        );
    }

    #[tokio::test]
    async fn get_missing_key_returns_404() {
        let relay = MockPkarrRelay::start().await;
        let client = reqwest::Client::new();
        let get_resp = client
            .get(format!("{}/missing", relay.base_url))
            .send()
            .await
            .expect("get request");
        assert_eq!(get_resp.status(), 404);
    }
}

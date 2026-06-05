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

/// In-memory store keyed by the URL path key (z-base-32 in strict mode).
type Store = Arc<RwLock<HashMap<String, Vec<u8>>>>;

#[derive(Clone)]
struct MockState {
    store: Store,
    /// When true, PUT validates the z32 key + BEP44 signature and returns 400
    /// for anything that is not a real pkarr relay payload.
    strict: bool,
}

/// Handle to a running mock relay. Drop to stop the server.
pub struct MockPkarrRelay {
    /// The `http://host:port` base URL to use in relay-client configuration.
    pub base_url: String,
    _shutdown: tokio::sync::oneshot::Sender<()>,
    _server_task: tokio::task::JoinHandle<()>,
}

impl MockPkarrRelay {
    /// Start a lax mock (stores whatever is PUT). For relay-pool/cooldown tests.
    pub async fn start() -> Self {
        Self::start_inner(false).await
    }

    /// Start a strict mock that validates the real pkarr relay format on PUT
    /// (z-base-32 key + `from_relay_payload` signature check). Rejects the old
    /// in-house dialect with 400.
    pub async fn start_strict() -> Self {
        Self::start_inner(true).await
    }

    async fn start_inner(strict: bool) -> Self {
        let state = MockState {
            store: Arc::new(RwLock::new(HashMap::new())),
            strict,
        };
        let app = Router::new()
            .route("/{key}", put(put_record).get(get_record))
            .with_state(state);

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

impl Drop for MockPkarrRelay {
    /// Explicit Drop so the spawned axum task stops promptly when the test
    /// finishes. `_shutdown` (oneshot::Sender) being dropped would already
    /// cause axum's `with_graceful_shutdown` future to resolve via
    /// `RecvError`, but `JoinHandle::abort()` is the belt-and-suspenders
    /// guarantee that no leftover task lingers across test boundaries.
    fn drop(&mut self) {
        self._server_task.abort();
    }
}

async fn put_record(
    Path(key): Path<String>,
    State(state): State<MockState>,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    if state.strict {
        let Ok(pk) = pkarr::PublicKey::try_from(key.as_str()) else {
            return StatusCode::BAD_REQUEST;
        };
        let payload = bytes::Bytes::copy_from_slice(&body);
        if pkarr::SignedPacket::from_relay_payload(&pk, &payload).is_err() {
            return StatusCode::BAD_REQUEST;
        }
    }
    state.store.write().await.insert(key, body.to_vec());
    StatusCode::OK
}

async fn get_record(
    Path(key): Path<String>,
    State(state): State<MockState>,
) -> impl IntoResponse {
    match state.store.read().await.get(&key) {
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

    #[tokio::test]
    async fn strict_relay_rejects_non_pkarr_bytes() {
        let relay = MockPkarrRelay::start_strict().await;
        let client = reqwest::Client::new();
        // Non-z32 key + garbage body → 400.
        let resp = client
            .put(format!("{}/not-a-real-key", relay.base_url))
            .body(b"in-house-bytes".to_vec())
            .send()
            .await
            .expect("put");
        assert_eq!(resp.status(), 400);
    }
}

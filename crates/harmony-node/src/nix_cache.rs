//! HTTP server for the Nix binary cache protocol.
//!
//! Exposes three endpoints:
//!
//! - `GET /nix-cache-info`  — cache metadata (plain text)
//! - `GET /{hash}.narinfo`  — narinfo lookup by 32-char nix-base32 store hash
//! - `GET /nar/{cid}.nar`   — NAR download by 64-hex-char CID
//!
//! All state is read-only. Content is looked up from the in-memory
//! [`MemoryBookStore`] first, falling back to disk under `data_dir`.

#![cfg(feature = "nix-cache")]

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::get,
    Router,
};
use harmony_content::{
    book::{BookStore, MemoryBookStore},
    bundle,
    cid::{ContentFlags, ContentId},
    dag,
};
use harmony_memo::store::MemoStore;

use crate::disk_io;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Shared state passed to every HTTP handler.
pub struct NixCacheState {
    pub book_store: Arc<MemoryBookStore>,
    pub memo_store: Arc<MemoStore>,
    /// Optional directory for disk-backed CAS books.
    /// When `None`, only the in-memory store is consulted.
    pub data_dir: Option<std::path::PathBuf>,
}

// ---------------------------------------------------------------------------
// Router
// ---------------------------------------------------------------------------

/// Build the axum [`Router`] for the Nix binary cache endpoints.
///
/// Axum 0.8 does not support parameters embedded in the middle of a path
/// segment (e.g. `/{hash}.narinfo`).  We capture the full filename as a
/// parameter and strip the suffix inside the handler.
pub fn router(state: Arc<NixCacheState>) -> Router {
    Router::new()
        .route("/nix-cache-info", get(cache_info_handler))
        // Captures "<hash>.narinfo" as the `filename` parameter.
        .route("/{filename}", get(narinfo_handler))
        // Captures "<cid>.nar" as the `filename` parameter.
        .route("/nar/{filename}", get(nar_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /nix-cache-info`
///
/// Returns the static cache metadata expected by Nix clients.
async fn cache_info_handler() -> Response {
    const BODY: &str = "StoreDir: /nix/store\nWantMassQuery: 1\nPriority: 30\n";
    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/x-nix-cache-info")],
        BODY,
    )
        .into_response()
}

/// `GET /{hash}.narinfo`
///
/// Looks up the narinfo for a Nix store hash.
///
/// The path parameter captures the full filename (e.g. `<hash>.narinfo`).
/// The handler strips the `.narinfo` suffix to obtain the 32-character
/// nix-base32 store path hash.
async fn narinfo_handler(
    State(state): State<Arc<NixCacheState>>,
    Path(filename): Path<String>,
) -> Response {
    // Strip the ".narinfo" suffix; reject requests that don't have it.
    let hash = match filename.strip_suffix(".narinfo") {
        Some(h) => h.to_string(),
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    // Validate: Nix store hashes are exactly 32 characters.
    if hash.len() != 32 {
        return (StatusCode::BAD_REQUEST, "invalid store hash length").into_response();
    }

    // The input CID is derived from the store hash bytes (not hex-decoded —
    // the hash string itself is the key material).
    let input_cid = match ContentId::for_book(hash.as_bytes(), ContentFlags::default()) {
        Ok(cid) => cid,
        Err(_) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, "cid derivation failed").into_response()
        }
    };

    // Look up memos for this input.
    let memos = state.memo_store.peek_by_input(&input_cid);
    let narinfo_cid = match memos.first() {
        Some(memo) => memo.output,
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    // Retrieve the narinfo bytes from the in-memory store.
    let narinfo_bytes = match state.book_store.get(&narinfo_cid) {
        Some(data) => data.to_vec(),
        None => {
            // Fall back to disk.
            match &state.data_dir {
                Some(dir) => match disk_io::read_book(dir, &narinfo_cid) {
                    Ok(data) => data,
                    Err(_) => return StatusCode::NOT_FOUND.into_response(),
                },
                None => return StatusCode::NOT_FOUND.into_response(),
            }
        }
    };

    (
        StatusCode::OK,
        [(header::CONTENT_TYPE, "text/x-nix-narinfo")],
        narinfo_bytes,
    )
        .into_response()
}

/// `GET /nar/{cid}.nar`
///
/// Reassembles and streams the NAR archive for a CID.
///
/// The path parameter captures the full filename (e.g. `<cid>.nar`).
/// The handler strips the `.nar` suffix to obtain the 64-character hex CID.
///
/// NOTE: The entire NAR is reassembled into a `Vec<u8>` before the response
/// is sent. For large packages (100+ MB), consider streaming reassembly via
/// `Body::from_stream` in a future iteration.
async fn nar_handler(
    State(state): State<Arc<NixCacheState>>,
    Path(filename): Path<String>,
) -> Response {
    // Strip the ".nar" suffix; reject requests that don't have it.
    let cid_hex = match filename.strip_suffix(".nar") {
        Some(h) => h.to_string(),
        None => return StatusCode::NOT_FOUND.into_response(),
    };

    // Validate: CIDs are 32 bytes = 64 hex chars.
    if cid_hex.len() != 64 {
        return (StatusCode::BAD_REQUEST, "invalid cid length").into_response();
    }

    let cid_bytes: [u8; 32] = match hex::decode(&cid_hex) {
        Ok(bytes) => match bytes.try_into() {
            Ok(arr) => arr,
            Err(_) => return (StatusCode::BAD_REQUEST, "invalid cid hex").into_response(),
        },
        Err(_) => return (StatusCode::BAD_REQUEST, "invalid cid hex").into_response(),
    };
    let root_cid = ContentId::from_bytes(cid_bytes);

    // Try reassembling from the in-memory store first (no disk I/O).
    let nar_bytes = match dag::reassemble(&root_cid, state.book_store.as_ref()) {
        Ok(data) => data,
        Err(_) => {
            // Fall back to loading the DAG from disk. Use spawn_blocking to
            // avoid blocking the async executor (worker_threads = 1).
            let data_dir = match &state.data_dir {
                Some(dir) => dir.clone(),
                None => return StatusCode::NOT_FOUND.into_response(),
            };
            let cid = root_cid;
            let result = tokio::task::spawn_blocking(move || {
                let mut temp_store = MemoryBookStore::new();
                if load_dag_from_disk(&data_dir, &cid, &mut temp_store).is_err() {
                    return None;
                }
                dag::reassemble(&cid, &temp_store).ok()
            })
            .await;
            match result {
                Ok(Some(data)) => data,
                _ => return StatusCode::NOT_FOUND.into_response(),
            }
        }
    };

    let content_length = nar_bytes.len();
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "application/x-nix-nar".to_string()),
            (header::CONTENT_LENGTH, content_length.to_string()),
        ],
        Body::from(nar_bytes),
    )
        .into_response()
}

// ---------------------------------------------------------------------------
// Disk DAG loader
// ---------------------------------------------------------------------------

/// Recursively load all books in a DAG from disk into `store`.
///
/// For bundle CIDs (depth > 0), parses the bundle to discover children and
/// recurses. For leaf books (depth == 0), loads the raw bytes. Sentinel CIDs
/// (inline metadata) are skipped.
fn load_dag_from_disk(
    data_dir: &std::path::Path,
    cid: &ContentId,
    store: &mut MemoryBookStore,
) -> Result<(), std::io::Error> {
    // Skip sentinels — they carry no on-disk data.
    if cid.is_sentinel() {
        return Ok(());
    }

    // Skip if already loaded (handles diamond DAGs and bundles).
    if store.contains(cid) {
        return Ok(());
    }

    let data = disk_io::read_book(data_dir, cid)?;

    if cid.depth() > 0 {
        // Bundle: parse children and recurse before inserting the bundle itself.
        let children: Vec<ContentId> = {
            let children_slice = bundle::parse_bundle(&data).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::InvalidData, "invalid bundle")
            })?;
            children_slice.to_vec()
        };
        store.store(*cid, data);
        for child in &children {
            load_dag_from_disk(data_dir, child, store)?;
        }
    } else {
        // Leaf book: just store it.
        store.store(*cid, data);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::Request;
    use harmony_content::book::BookStore;
    use harmony_content::cid::ContentFlags;
    use harmony_content::ContentId;
    use harmony_memo::store::MemoStore;
    use tower::util::ServiceExt as _;

    /// Build a test state with a pre-populated book and memo.
    ///
    /// Returns `(state, narinfo_cid, store_hash)`.
    fn setup_test_state() -> (Arc<NixCacheState>, ContentId, String) {
        let mut book_store = MemoryBookStore::new();
        let mut memo_store = MemoStore::new();

        let narinfo_text = "StorePath: /nix/store/aaaabbbbccccddddeeeeffffgggghhhh-hello-1.0\n\
            URL: nar/deadbeef01234567deadbeef01234567deadbeef01234567deadbeef01234567.nar\n\
            Compression: none\n\
            FileHash: sha256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\
            FileSize: 0\n\
            NarHash: sha256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU=\n\
            NarSize: 0\n";
        let narinfo_cid = book_store.insert(narinfo_text.as_bytes()).unwrap();

        let store_hash = "aaaabbbbccccddddeeeeffffgggghhhh";
        let input_cid = ContentId::for_book(store_hash.as_bytes(), ContentFlags::default()).unwrap();

        let identity =
            harmony_identity::pq_identity::PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
        let memo = harmony_memo::create::create_memo(
            input_cid,
            narinfo_cid,
            &identity,
            &mut rand::rngs::OsRng,
            1000,
            9_999_999,
        )
        .unwrap();
        memo_store.insert(memo);

        let state = Arc::new(NixCacheState {
            book_store: Arc::new(book_store),
            memo_store: Arc::new(memo_store),
            data_dir: None,
        });
        (state, narinfo_cid, store_hash.to_string())
    }

    #[tokio::test]
    async fn nix_cache_info_returns_expected_fields() {
        let state = Arc::new(NixCacheState {
            book_store: Arc::new(MemoryBookStore::new()),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nix-cache-info")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "text/x-nix-cache-info"
        );
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("StoreDir: /nix/store"), "missing StoreDir");
        assert!(text.contains("WantMassQuery: 1"), "missing WantMassQuery");
        assert!(text.contains("Priority: 30"), "missing Priority");
    }

    #[tokio::test]
    async fn narinfo_returns_200_when_memo_exists() {
        let (state, _narinfo_cid, store_hash) = setup_test_state();
        let app = router(state);

        let uri = format!("/{store_hash}.narinfo");
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "text/x-nix-narinfo"
        );
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let text = std::str::from_utf8(&body).unwrap();
        assert!(text.contains("StorePath:"), "narinfo body missing StorePath");
    }

    #[tokio::test]
    async fn narinfo_returns_404_when_missing() {
        let state = Arc::new(NixCacheState {
            book_store: Arc::new(MemoryBookStore::new()),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        // 32-char hash that has no memo.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz.narinfo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn narinfo_returns_400_for_bad_hash() {
        let state = Arc::new(NixCacheState {
            book_store: Arc::new(MemoryBookStore::new()),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        // Only 5 chars — too short.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/short.narinfo")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn nar_returns_200_for_known_book() {
        let mut book_store = MemoryBookStore::new();
        let nar_data = b"fake-nar-content-for-testing";
        let cid = book_store.insert(nar_data).unwrap();

        let state = Arc::new(NixCacheState {
            book_store: Arc::new(book_store),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        let cid_hex = hex::encode(cid.to_bytes());
        let uri = format!("/nar/{cid_hex}.nar");
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get(header::CONTENT_TYPE)
                .unwrap()
                .to_str()
                .unwrap(),
            "application/x-nix-nar"
        );
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        assert_eq!(body.as_ref(), nar_data);
    }

    #[tokio::test]
    async fn nar_returns_404_for_unknown_cid() {
        let state = Arc::new(NixCacheState {
            book_store: Arc::new(MemoryBookStore::new()),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        // Valid 64-hex-char CID that is not in the store.
        let cid_hex = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef";
        let uri = format!("/nar/{cid_hex}.nar");
        let resp = app
            .oneshot(
                Request::builder()
                    .uri(&uri)
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn nar_returns_400_for_bad_cid() {
        let state = Arc::new(NixCacheState {
            book_store: Arc::new(MemoryBookStore::new()),
            memo_store: Arc::new(MemoStore::new()),
            data_dir: None,
        });
        let app = router(state);

        // Invalid hex (too short).
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/nar/notvalidhex.nar")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }
}

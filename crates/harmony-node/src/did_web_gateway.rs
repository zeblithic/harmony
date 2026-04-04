//! did:web gateway service.
//!
//! Declares a Zenoh queryable on `harmony/identity/web/**` and resolves
//! did:web DIDs by fetching DID Documents over HTTPS. Results are cached
//! with a configurable TTL.
//!
//! Queries are handled concurrently: each incoming query spawns a task that
//! checks the cache, and on a miss, fetches over HTTPS. A semaphore caps
//! the number of concurrent outbound fetches, and a pending-request map
//! deduplicates simultaneous queries for the same DID (thundering-herd
//! protection).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures_util::TryStreamExt as _;
use harmony_identity::did_document::{did_web_to_url, parse_did_document};
use harmony_identity::ResolvedDidDocument;
use tokio::sync::{broadcast, RwLock, Semaphore};

/// Cached DID Document with expiry.
struct CacheEntry {
    document: ResolvedDidDocument,
    expires_at: Instant,
}

/// Maximum DID Document response size (1 MiB). DID Documents are small JSON
/// files; anything larger is malicious or misconfigured.
const MAX_DOC_BYTES: u64 = 1 << 20;

/// Maximum number of cached DID Documents. When exceeded, all expired entries
/// are pruned; if still over capacity, the entire cache is cleared.
const MAX_CACHE_ENTRIES: usize = 1_000;

/// Maximum concurrent outbound HTTPS fetches. Prevents resource exhaustion
/// from a burst of cache misses for distinct DIDs.
const MAX_CONCURRENT_FETCHES: usize = 16;

/// Shared state for the concurrent gateway.
struct GatewayState {
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Tracks in-flight fetches. When a DID is being fetched, subsequent
    /// queries for the same DID subscribe to the broadcast channel instead
    /// of issuing a duplicate HTTP request.
    pending: RwLock<HashMap<String, broadcast::Sender<Option<ResolvedDidDocument>>>>,
    client: reqwest::Client,
    ttl: Duration,
    semaphore: Semaphore,
}

/// Run the did:web gateway loop.
pub async fn run(
    queryable: zenoh::query::Queryable<zenoh::handlers::FifoChannelHandler<zenoh::query::Query>>,
    cache_ttl_secs: u64,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .expect("reqwest client must build (TLS backend unavailable?)");

    let state = Arc::new(GatewayState {
        cache: RwLock::new(HashMap::new()),
        pending: RwLock::new(HashMap::new()),
        client,
        ttl: Duration::from_secs(cache_ttl_secs),
        semaphore: Semaphore::new(MAX_CONCURRENT_FETCHES),
    });

    tracing::info!("did:web gateway started (TTL={cache_ttl_secs}s, max_concurrent={MAX_CONCURRENT_FETCHES})");

    while let Ok(query) = queryable.recv_async().await {
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            handle_query(query, &state).await;
        });
    }

    tracing::info!("did:web gateway stopped");
}

/// Handle a single query, potentially coalescing with other in-flight
/// requests for the same DID.
async fn handle_query(query: zenoh::query::Query, state: &GatewayState) {
    let key_expr = query.key_expr().to_string();

    let did = match key_expr_to_did(&key_expr) {
        Some(d) => d,
        None => {
            tracing::warn!(%key_expr, "malformed did:web gateway query");
            return;
        }
    };

    // Fast path: cache hit (read lock only).
    {
        let cache = state.cache.read().await;
        if let Some(entry) = cache.get(&did) {
            if entry.expires_at > Instant::now() {
                tracing::debug!(%did, "did:web cache hit");
                reply_with_document(&query, &entry.document).await;
                return;
            }
        }
    }

    // Check if another task is already fetching this DID.
    {
        let pending = state.pending.read().await;
        if let Some(tx) = pending.get(&did) {
            // Subscribe before dropping the lock so we don't miss the send.
            let mut rx = tx.subscribe();
            drop(pending);
            tracing::debug!(%did, "did:web coalescing with in-flight fetch");
            match rx.recv().await {
                Ok(Some(doc)) => reply_with_document(&query, &doc).await,
                _ => reply_empty(&query).await,
            }
            return;
        }
    }

    // Register ourselves as the fetcher for this DID.
    let (tx, _) = broadcast::channel::<Option<ResolvedDidDocument>>(1);
    {
        let mut pending = state.pending.write().await;
        // Double-check: another task may have registered between our read and write.
        if let Some(existing_tx) = pending.get(&did) {
            let mut rx = existing_tx.subscribe();
            drop(pending);
            match rx.recv().await {
                Ok(Some(doc)) => reply_with_document(&query, &doc).await,
                _ => reply_empty(&query).await,
            }
            return;
        }
        pending.insert(did.clone(), tx.clone());
    }

    // Fetch with semaphore limiting concurrent outbound requests.
    let result = fetch_and_cache(state, &did).await;

    // Broadcast result to coalesced waiters and clean up.
    let _ = tx.send(result.clone());
    {
        let mut pending = state.pending.write().await;
        pending.remove(&did);
    }

    // Reply to the original query.
    match result {
        Some(doc) => reply_with_document(&query, &doc).await,
        None => reply_empty(&query).await,
    }
}

/// Fetch a DID Document over HTTPS, cache it, and return it.
/// Acquires a semaphore permit to limit concurrent outbound fetches.
async fn fetch_and_cache(state: &GatewayState, did: &str) -> Option<ResolvedDidDocument> {
    let _permit = state.semaphore.acquire().await.ok()?;

    // Re-check cache — another task may have populated it while we waited
    // for the semaphore.
    {
        let cache = state.cache.read().await;
        if let Some(entry) = cache.get(did) {
            if entry.expires_at > Instant::now() {
                tracing::debug!(%did, "did:web cache hit after semaphore wait");
                return Some(entry.document.clone());
            }
        }
    }

    let url = match did_web_to_url(did) {
        Ok(u) => u,
        Err(e) => {
            tracing::warn!(%did, err = ?e, "invalid did:web DID");
            return None;
        }
    };

    if !url.starts_with("https://") {
        tracing::warn!(%did, %url, "did:web requires HTTPS");
        return None;
    }

    // Fetch via HTTPS
    let resp = match state.client.get(&url).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(%did, err = %e, "did:web HTTPS fetch failed");
            return None;
        }
    };

    if !resp.status().is_success() {
        tracing::warn!(%did, status = %resp.status(), "did:web fetch failed");
        return None;
    }

    // Stream response with size cap.
    let mut body = bytes::BytesMut::new();
    let mut stream = resp.bytes_stream();
    loop {
        match stream.try_next().await {
            Ok(Some(chunk)) => {
                body.extend_from_slice(&chunk);
                if body.len() as u64 > MAX_DOC_BYTES {
                    tracing::warn!(%did, "did:web document too large");
                    return None;
                }
            }
            Ok(None) => break,
            Err(e) => {
                tracing::warn!(%did, err = %e, "did:web body stream error");
                return None;
            }
        }
    }

    let bytes = body.freeze();
    match parse_did_document(did, &bytes) {
        Ok(doc) => {
            tracing::info!(%did, keys = doc.verification_methods.len(), "did:web resolved");
            // Insert into cache.
            let now = Instant::now();
            let mut cache = state.cache.write().await;
            if cache.len() >= MAX_CACHE_ENTRIES {
                cache.retain(|_, e| e.expires_at > now);
                if cache.len() >= MAX_CACHE_ENTRIES {
                    tracing::warn!("did:web cache overflow, clearing");
                    cache.clear();
                }
            }
            cache.insert(
                did.to_string(),
                CacheEntry {
                    document: doc.clone(),
                    expires_at: now + state.ttl,
                },
            );
            Some(doc)
        }
        Err(e) => {
            tracing::warn!(%did, err = ?e, "did:web document parse failed");
            None
        }
    }
}

/// Parse a Zenoh key expression into a did:web DID.
fn key_expr_to_did(key_expr: &str) -> Option<String> {
    let rest = key_expr.strip_prefix("harmony/identity/web/")?;
    if rest.is_empty() {
        return None;
    }
    Some(format!("did:web:{}", rest.replace('/', ":")))
}

/// Reply with a postcard-serialized ResolvedDidDocument.
async fn reply_with_document(query: &zenoh::query::Query, doc: &ResolvedDidDocument) {
    match postcard::to_allocvec(doc) {
        Ok(payload) => {
            if let Err(e) = query.reply(query.key_expr(), payload).await {
                tracing::warn!(err = ?e, "did:web gateway reply failed");
            }
        }
        Err(e) => {
            tracing::error!(err = ?e, "postcard serialization failed");
            reply_empty(query).await;
        }
    }
}

/// Reply with empty payload (signal "not found").
async fn reply_empty(query: &zenoh::query::Query) {
    let _ = query.reply(query.key_expr(), Vec::<u8>::new()).await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_expr_to_did_root_domain() {
        assert_eq!(
            key_expr_to_did("harmony/identity/web/example.com"),
            Some("did:web:example.com".to_string())
        );
    }

    #[test]
    fn key_expr_to_did_with_path() {
        assert_eq!(
            key_expr_to_did("harmony/identity/web/example.com/issuers/1"),
            Some("did:web:example.com:issuers:1".to_string())
        );
    }

    #[test]
    fn key_expr_to_did_empty_rest() {
        assert_eq!(key_expr_to_did("harmony/identity/web/"), None);
    }

    #[test]
    fn key_expr_to_did_wrong_prefix() {
        assert_eq!(key_expr_to_did("harmony/identity/announce/abc"), None);
    }
}

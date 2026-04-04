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
use tokio::sync::{broadcast, OwnedSemaphorePermit, RwLock, Semaphore};

/// Pending map uses std::sync::RwLock (not tokio) so it can be locked in
/// Drop for panic-safe cleanup. Operations are fast (HashMap insert/remove)
/// and never held across await points.
type PendingMap = std::sync::RwLock<HashMap<String, broadcast::Sender<Option<ResolvedDidDocument>>>>;

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

/// Maximum number of query-handling tasks. Provides back-pressure when the
/// gateway is flooded with queries.
const MAX_OUTSTANDING_QUERIES: usize = 256;

/// Shared state for the concurrent gateway.
struct GatewayState {
    cache: RwLock<HashMap<String, CacheEntry>>,
    /// Tracks in-flight fetches. Uses std::sync::RwLock (not tokio) so the
    /// PendingGuard can lock it in Drop for guaranteed panic-safe cleanup.
    pending: PendingMap,
    client: reqwest::Client,
    ttl: Duration,
    fetch_semaphore: Semaphore,
}

/// RAII guard that removes a DID from the pending map on drop, even if the
/// owning task panics. Uses std::sync::RwLock::write() which blocks until
/// the lock is acquired — guaranteed cleanup, no silent channel leaks.
struct PendingGuard {
    state: Arc<GatewayState>,
    did: String,
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        // unwrap is safe: pending map operations never panic while holding
        // the lock, so poisoning cannot occur.
        let mut pending = self.state.pending.write().unwrap();
        pending.remove(&self.did);
    }
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

    let task_semaphore = Arc::new(Semaphore::new(MAX_OUTSTANDING_QUERIES));

    let state = Arc::new(GatewayState {
        cache: RwLock::new(HashMap::new()),
        pending: std::sync::RwLock::new(HashMap::new()),
        client,
        ttl: Duration::from_secs(cache_ttl_secs),
        fetch_semaphore: Semaphore::new(MAX_CONCURRENT_FETCHES),
    });

    tracing::info!(
        "did:web gateway started (TTL={cache_ttl_secs}s, max_concurrent={MAX_CONCURRENT_FETCHES}, max_tasks={MAX_OUTSTANDING_QUERIES})"
    );

    while let Ok(query) = queryable.recv_async().await {
        // Back-pressure: wait for a task permit before spawning.
        let permit = match Arc::clone(&task_semaphore).acquire_owned().await {
            Ok(p) => p,
            Err(_) => break, // Semaphore closed — shutting down.
        };
        let state = Arc::clone(&state);
        tokio::spawn(async move {
            handle_query(query, state, permit).await;
        });
    }

    tracing::info!("did:web gateway stopped");
}

/// Handle a single query, potentially coalescing with other in-flight
/// requests for the same DID.
async fn handle_query(
    query: zenoh::query::Query,
    state: Arc<GatewayState>,
    _task_permit: OwnedSemaphorePermit,
) {
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
    // pending uses std::sync::RwLock — extract receiver while holding the
    // lock, then drop the lock before any .await.
    let coalesce_rx = {
        let pending = state.pending.read().unwrap();
        pending.get(&did).map(|tx| tx.subscribe())
    };
    if let Some(mut rx) = coalesce_rx {
        tracing::debug!(%did, "did:web coalescing with in-flight fetch");
        if let Ok(Some(doc)) = rx.recv().await {
            reply_with_document(&query, &doc).await;
            return;
        }
        if let Some(doc) = try_cache_hit(&state, &did).await {
            reply_with_document(&query, &doc).await;
            return;
        }
        reply_empty(&query).await;
        return;
    }

    // Register ourselves as the fetcher for this DID.
    let (tx, _) = broadcast::channel::<Option<ResolvedDidDocument>>(1);
    let double_check_rx = {
        let mut pending = state.pending.write().unwrap();
        if let Some(existing_tx) = pending.get(&did) {
            Some(existing_tx.subscribe())
        } else {
            pending.insert(did.clone(), tx.clone());
            None
        }
    };
    if let Some(mut rx) = double_check_rx {
        if let Ok(Some(doc)) = rx.recv().await {
            reply_with_document(&query, &doc).await;
            return;
        }
        if let Some(doc) = try_cache_hit(&state, &did).await {
            reply_with_document(&query, &doc).await;
            return;
        }
        reply_empty(&query).await;
        return;
    }

    // RAII guard ensures the pending entry is removed even if we panic.
    // On the normal path, broadcast fires before the guard drops; on panic,
    // the guard cleans up the dead Sender so future queries don't hang.
    let _guard = PendingGuard {
        state: Arc::clone(&state),
        did: did.clone(),
    };

    // Fetch with semaphore limiting concurrent outbound requests.
    let result = fetch_and_cache(&state, &did).await;

    // Broadcast result to coalesced waiters. The guard's Drop will remove
    // the pending entry (both normal and panic paths).
    let _ = tx.send(result.clone());

    // Reply to the original query.
    match result {
        Some(doc) => reply_with_document(&query, &doc).await,
        None => reply_empty(&query).await,
    }
}

/// Check cache for a valid entry. Used as fallback when a broadcast is missed.
async fn try_cache_hit(state: &GatewayState, did: &str) -> Option<ResolvedDidDocument> {
    let cache = state.cache.read().await;
    let entry = cache.get(did)?;
    if entry.expires_at > Instant::now() {
        Some(entry.document.clone())
    } else {
        None
    }
}

/// Fetch a DID Document over HTTPS, cache it, and return it.
/// Acquires a semaphore permit to limit concurrent outbound fetches.
async fn fetch_and_cache(state: &GatewayState, did: &str) -> Option<ResolvedDidDocument> {
    let _permit = state.fetch_semaphore.acquire().await.ok()?;

    // Re-check cache — another task may have populated it while we waited
    // for the semaphore.
    if let Some(doc) = try_cache_hit(state, did).await {
        tracing::debug!(%did, "did:web cache hit after semaphore wait");
        return Some(doc);
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

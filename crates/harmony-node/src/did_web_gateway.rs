//! did:web gateway service.
//!
//! Declares a Zenoh queryable on `harmony/identity/web/**` and resolves
//! did:web DIDs by fetching DID Documents over HTTPS. Results are cached
//! with a configurable TTL.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use harmony_identity::did_document::{did_web_to_url, parse_did_document};
use harmony_identity::ResolvedDidDocument;

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

/// Run the did:web gateway loop.
pub async fn run(
    queryable: zenoh::query::Queryable<zenoh::handlers::FifoChannelHandler<zenoh::query::Query>>,
    cache_ttl_secs: u64,
) {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .build()
        .unwrap_or_default();
    let ttl = Duration::from_secs(cache_ttl_secs);
    let mut cache: HashMap<String, CacheEntry> = HashMap::new();

    tracing::info!("did:web gateway started (TTL={cache_ttl_secs}s)");

    while let Ok(query) = queryable.recv_async().await {
        let key_expr = query.key_expr().to_string();

        let did = match key_expr_to_did(&key_expr) {
            Some(d) => d,
            None => {
                tracing::warn!(%key_expr, "malformed did:web gateway query");
                continue;
            }
        };

        // Check cache
        let now = Instant::now();
        if let Some(entry) = cache.get(&did) {
            if entry.expires_at > now {
                tracing::debug!(%did, "did:web cache hit");
                reply_with_document(&query, &entry.document).await;
                continue;
            }
        }

        // Construct URL
        let url = match did_web_to_url(&did) {
            Ok(u) => u,
            Err(e) => {
                tracing::warn!(%did, err = ?e, "invalid did:web DID");
                reply_empty(&query).await;
                continue;
            }
        };

        if !url.starts_with("https://") {
            tracing::warn!(%did, %url, "did:web requires HTTPS");
            reply_empty(&query).await;
            continue;
        }

        // Fetch via HTTPS
        match client.get(&url).send().await {
            Ok(resp) => {
                if !resp.status().is_success() {
                    tracing::warn!(%did, status = %resp.status(), "did:web fetch failed");
                    reply_empty(&query).await;
                    continue;
                }
                if resp.content_length().unwrap_or(0) > MAX_DOC_BYTES {
                    tracing::warn!(%did, "did:web document too large");
                    reply_empty(&query).await;
                    continue;
                }
                match resp.bytes().await {
                    Ok(bytes) => match parse_did_document(&did, &bytes) {
                        Ok(doc) => {
                            tracing::info!(%did, keys = doc.verification_methods.len(), "did:web resolved");
                            reply_with_document(&query, &doc).await;
                            cache.insert(
                                did,
                                CacheEntry {
                                    document: doc,
                                    expires_at: now + ttl,
                                },
                            );
                            if cache.len() > MAX_CACHE_ENTRIES {
                                cache.retain(|_, e| e.expires_at > now);
                                if cache.len() > MAX_CACHE_ENTRIES {
                                    tracing::warn!("did:web cache overflow, clearing");
                                    cache.clear();
                                }
                            }
                        }
                        Err(e) => {
                            tracing::warn!(%did, err = ?e, "did:web document parse failed");
                            reply_empty(&query).await;
                        }
                    },
                    Err(e) => {
                        tracing::warn!(%did, err = %e, "did:web response read failed");
                        reply_empty(&query).await;
                    }
                }
            }
            Err(e) => {
                tracing::warn!(%did, err = %e, "did:web HTTPS fetch failed");
                reply_empty(&query).await;
            }
        }
    }

    tracing::info!("did:web gateway stopped");
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

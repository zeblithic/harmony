//! `PkarrResolver` — relay-pool resolver with parallel epoch-window queries
//! and in-memory LRU cache.
//!
//! Spec Sections 5.4 + 7.1.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ed25519_dalek::VerifyingKey;
use futures::future::join_all;
use lru::LruCache;

use crate::error::PkarrError;
use crate::record::PkarrRoutingRecord;
use crate::relay::RelayClient;

const POSITIVE_CACHE_TTL: Duration = Duration::from_secs(15 * 60);
const NEGATIVE_CACHE_TTL: Duration = Duration::from_secs(60);

#[derive(Clone)]
struct CachedResolution {
    record: Option<PkarrRoutingRecord>,
    fetched_at: Instant,
    ttl: Duration,
}

pub struct PkarrResolver {
    relay: Arc<RelayClient>,
    cache: Arc<Mutex<LruCache<[u8; 32], CachedResolution>>>,
}

impl PkarrResolver {
    pub fn new(relay: Arc<RelayClient>) -> Self {
        Self {
            relay,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(1024).expect("nonzero"),
            ))),
        }
    }

    /// Resolve a single ephemeral public key. Returns `Ok(Some)` if a valid
    /// signed record is found; `Ok(None)` if confirmed-absent; `Err` on
    /// transport failures.
    pub async fn resolve(
        &self,
        pk: &VerifyingKey,
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let pk_bytes = pk.to_bytes();
        if let Some(cached) = self.cache_get(&pk_bytes) {
            return Ok(cached.record.clone());
        }
        let key_z32 = crate::wire::z32_for_verifying_key(pk)?;
        match self.relay.get(&key_z32).await? {
            None => {
                self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                Ok(None)
            }
            Some(envelope) => {
                // RPK1: outer sig failure → silent-drop (cache negative).
                let record = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(
                            key = %key_z32,
                            error = ?e,
                            "pkarr envelope failed outer verification — dropping (RPK1)"
                        );
                        self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                        return Ok(None);
                    }
                };
                // RPK2: inner sig failure → silent-drop (cache negative).
                if let Err(e) = record.verify_inner_sig() {
                    tracing::warn!(
                        key = %key_z32,
                        error = ?e,
                        "pkarr record failed inner sig verification — dropping (RPK2)"
                    );
                    self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                    return Ok(None);
                }
                // RPK4: skew check (±30 min) — silent-drop (cache negative).
                // Don't cache positively: a record outside the skew window
                // is invalid, and could become valid if the local clock
                // corrects, but the publisher's next republish will land
                // a fresh record anyway. Negative-cache so we don't spam
                // the relay during the 60s window.
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock < UNIX epoch is unsupported")
                    .as_millis() as u64;
                if let Err(e) = record.verify_skew(now_ms) {
                    tracing::warn!(
                        key = %key_z32,
                        announced_at_ms = record.announced_at_ms,
                        now_ms,
                        error = ?e,
                        "pkarr record outside ±30min skew window — dropping (RPK4)"
                    );
                    self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                    return Ok(None);
                }
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
            }
        }
    }

    /// Resolve any of `keys` (the 3-key epoch tolerance window) in parallel.
    /// Returns the freshest valid record by `announced_at_ms`, or `None` if
    /// none resolve.
    pub async fn resolve_window(
        &self,
        keys: &[VerifyingKey],
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        let futures = keys.iter().map(|pk| self.resolve(pk));
        let results = join_all(futures).await;

        let mut best: Option<PkarrRoutingRecord> = None;
        let mut any_ok_none = false;
        let mut any_err: Option<PkarrError> = None;
        for r in results {
            match r {
                Ok(Some(rec)) => {
                    if best
                        .as_ref()
                        .is_none_or(|b| rec.announced_at_ms > b.announced_at_ms)
                    {
                        best = Some(rec);
                    }
                }
                Ok(None) => any_ok_none = true,
                Err(e) => any_err = Some(e),
            }
        }

        // Prefer definitive answers over transient network errors:
        //   - A valid record always wins.
        //   - At least one key returned Ok(None) (confirmed absent) → surface
        //     Ok(None) even if sibling keys errored transiently.
        //   - Only return Err when every key errored (no Ok(_) at all).
        match (best, any_ok_none, any_err) {
            (Some(rec), _, _) => Ok(Some(rec)),
            (None, true, _) => Ok(None),
            (None, false, Some(e)) => Err(e),
            (None, false, None) => Ok(None),
        }
    }

    fn cache_get(&self, pk: &[u8; 32]) -> Option<CachedResolution> {
        let mut cache = self.cache.lock().expect("cache poisoned");
        let entry = cache.get(pk)?;
        if entry.fetched_at.elapsed() >= entry.ttl {
            cache.pop(pk);
            return None;
        }
        // RPK4 re-verification on cache hit: a record cached near the +30min
        // skew edge could otherwise be served from the 15min positive cache
        // up to 14min after it falls outside the skew window. Re-check on
        // every lookup; on skew failure, evict + treat as miss (the next
        // resolve will go to the relay and either negative-cache or refetch
        // a fresh record).
        if let Some(rec) = entry.record.as_ref() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock < UNIX epoch is unsupported")
                .as_millis() as u64;
            if rec.verify_skew(now_ms).is_err() {
                cache.pop(pk);
                return None;
            }
        }
        Some(entry.clone())
    }

    fn cache_put(&self, pk: [u8; 32], record: Option<PkarrRoutingRecord>, ttl: Duration) {
        let mut cache = self.cache.lock().expect("cache poisoned");
        cache.put(
            pk,
            CachedResolution {
                record,
                fetched_at: Instant::now(),
                ttl,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::publisher::{PkarrPublisher, RecordBuilder};
    use crate::testing::MockPkarrRelay;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_identity_pubkey(sk: &SigningKey) -> [u8; 64] {
        let mut out = [0u8; 64];
        out[32..].copy_from_slice(&sk.verifying_key().to_bytes());
        out
    }

    #[tokio::test]
    async fn publish_then_resolve_round_trip() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));

        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _ph = Arc::clone(&publisher).spawn();
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let key_builder: crate::EphemeralKeyBuilder = Arc::new(move |_| ephemeral.clone());
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let identity_sk_clone = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(
                b"r-blob".to_vec(),
                identity_pub,
                now_ms,
                &identity_sk_clone,
            )
            .expect("sign")
        });

        publisher
            .register("round-trip".to_string(), key_builder, builder)
            .await;

        // Wait for publish to land + then resolve.
        let mut attempts = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            attempts += 1;
            assert!(attempts < 60, "resolve timed out");
            let r = resolver.resolve(&vk).await.expect("resolve");
            if let Some(rec) = r {
                assert_eq!(rec.routing_blob, b"r-blob");
                assert!(rec.verify_inner_sig().is_ok());
                return;
            }
        }
    }

    #[tokio::test]
    async fn resolve_missing_returns_none() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let absent_key = SigningKey::generate(&mut OsRng).verifying_key();
        let result = resolver.resolve(&absent_key).await.expect("resolve");
        assert!(result.is_none());
    }

    #[tokio::test]
    async fn resolve_caches_positive_result() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let publisher = Arc::new(PkarrPublisher::new(Arc::clone(&client)));
        let _ph = Arc::clone(&publisher).spawn();
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let key_builder: crate::EphemeralKeyBuilder = Arc::new(move |_| ephemeral.clone());
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let identity_sk_clone = identity_sk.clone();
        let builder: RecordBuilder = Arc::new(move |now_ms| {
            PkarrRoutingRecord::sign_new(
                b"cached".to_vec(),
                identity_pub,
                now_ms,
                &identity_sk_clone,
            )
            .expect("sign")
        });
        publisher
            .register("cache-test".to_string(), key_builder, builder)
            .await;

        // Wait for publish.
        let mut attempts = 0;
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            attempts += 1;
            assert!(attempts < 60, "first resolve timed out");
            if resolver.resolve(&vk).await.expect("resolve").is_some() {
                break;
            }
        }

        // Second resolve hits cache.
        let r = resolver.resolve(&vk).await.expect("resolve");
        assert!(r.is_some());
    }

    #[tokio::test]
    async fn resolves_real_pkarr_payload() {
        let relay = MockPkarrRelay::start_strict().await;
        let put_client = Arc::new(crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec![relay.base_url.clone()]),
        ));

        let ephemeral = ed25519_dalek::SigningKey::from_bytes(&[5u8; 32]);
        let id_sk = ed25519_dalek::SigningKey::from_bytes(&[6u8; 32]);
        let mut id_pub = [0u8; 64];
        id_pub[32..].copy_from_slice(&id_sk.verifying_key().to_bytes());
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let rec = crate::record::PkarrRoutingRecord::sign_new(
            b"iroh-routing".to_vec(),
            id_pub,
            now_ms,
            &id_sk,
        )
        .expect("sign");
        let (z32, payload) = crate::wire::build_relay_payload(&ephemeral, &rec).unwrap();
        put_client.put(&z32, &payload).await.expect("publish");

        let resolver = PkarrResolver::new(put_client);
        let got = resolver
            .resolve(&ephemeral.verifying_key())
            .await
            .expect("resolve")
            .expect("present");
        assert_eq!(got.routing_blob, b"iroh-routing".to_vec());
    }
}

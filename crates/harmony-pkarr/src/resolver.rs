//! `PkarrResolver` — relay-pool resolver with parallel epoch-window queries
//! and in-memory LRU cache.
//!
//! Spec Sections 5.4 + 7.1.

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
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
        let key_z32 = hex::encode(pk_bytes);
        match self.relay.get(&key_z32).await? {
            None => {
                self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                Ok(None)
            }
            Some(envelope) => {
                // RPK1: outer sig failure → silent-drop (cache negative).
                let record = match parse_and_verify(&envelope, pk) {
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
                Ok(None) => {}
                Err(e) => any_err = Some(e),
            }
        }

        match (best, any_err) {
            (Some(rec), _) => Ok(Some(rec)),
            (None, Some(e)) => Err(e),
            (None, None) => Ok(None),
        }
    }

    fn cache_get(&self, pk: &[u8; 32]) -> Option<CachedResolution> {
        let mut cache = self.cache.lock().expect("cache poisoned");
        let entry = cache.get(pk)?;
        if entry.fetched_at.elapsed() < entry.ttl {
            Some(entry.clone())
        } else {
            cache.pop(pk);
            None
        }
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

/// Parse the BEP44 envelope, verify the outer Ed25519 sig under `expected_pk`,
/// then parse + return the inner `PkarrRoutingRecord` (does NOT verify the
/// inner sig — caller does that with the expected identity).
///
/// Envelope = 64 sig + 8 seq (LE u64) + payload (matching publisher.rs format).
fn parse_and_verify(
    envelope: &[u8],
    expected_pk: &VerifyingKey,
) -> Result<PkarrRoutingRecord, PkarrError> {
    // Envelope = 64 sig + 8 seq (LE u64) + payload (matching publisher.rs format).
    if envelope.len() < 64 + 8 {
        return Err(PkarrError::RelayResponseInvalid);
    }
    let sig_bytes: [u8; 64] = envelope[..64].try_into().expect("64 == 64");
    let seq = u64::from_le_bytes(envelope[64..72].try_into().expect("8 == 8"));
    let payload = &envelope[72..];

    let mut to_verify = alloc::vec::Vec::new();
    to_verify.extend_from_slice(format!("3:seqi{}e1:v{}:", seq, payload.len()).as_bytes());
    to_verify.extend_from_slice(payload);

    let sig = Signature::from_bytes(&sig_bytes);
    expected_pk
        .verify(&to_verify, &sig)
        .map_err(|_| PkarrError::OuterSignatureInvalid)?;

    PkarrRoutingRecord::from_canonical_cbor(payload)
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
            .register("round-trip".to_string(), ephemeral, builder)
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
            .register("cache-test".to_string(), ephemeral, builder)
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
}

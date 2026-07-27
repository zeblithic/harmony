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
    /// Per-key highest accepted BEP44 `seq`. In-memory anti-rollback: a relay
    /// replaying a strictly-older (but validly-signed, still-within-TTL) record
    /// is rejected once a newer one has been seen this session. Boot-lifetime
    /// only (persisted seq is deferred to the async-DM milestone).
    ///
    /// Bounded LRU (not an unbounded map): ephemeral keys rotate per epoch, so
    /// the keyspace grows without bound over time — an unbounded map would be a
    /// slow memory-growth / DoS vector. LRU eviction of a stale key drops only
    /// its rollback protection, which is already best-effort (it resets on
    /// reboot) and irrelevant once that epoch's key is no longer queried.
    seq_highwater: Arc<Mutex<LruCache<[u8; 32], u64>>>,
}

impl PkarrResolver {
    pub fn new(relay: Arc<RelayClient>) -> Self {
        Self {
            relay,
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(1024).expect("nonzero"),
            ))),
            seq_highwater: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(4096).expect("nonzero"),
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
                // Outer-sig failure (RPK1) OR malformed record (bad TXT/base64/
                // CBOR) → silent-drop (cache negative). The error field carries
                // the specific cause.
                let (record, seq) = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(
                            key = %key_z32,
                            error = ?e,
                            "pkarr relay payload rejected — bad outer signature (RPK1) or malformed record; see error field — dropping"
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
                // RPK4: freshness check (future-strict + signed TTL) —
                // silent-drop (cache negative). Don't cache positively: an
                // expired/forged record is invalid, and the publisher's next
                // republish will land a fresh record anyway. Negative-cache so
                // we don't spam the relay during the 60s window.
                let now_ms = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("system clock < UNIX epoch is unsupported")
                    .as_millis() as u64;
                if let Err(e) = record.verify_freshness(now_ms) {
                    tracing::warn!(
                        key = %key_z32,
                        announced_at_ms = record.announced_at_ms,
                        valid_until_ms = record.valid_until_ms,
                        now_ms,
                        error = ?e,
                        "pkarr record failed freshness (expired or forged-future) — dropping (RPK4)"
                    );
                    self.cache_put(pk_bytes, None, NEGATIVE_CACHE_TTL);
                    return Ok(None);
                }
                // RPK5: in-memory anti-rollback. Reject a strictly-older seq than
                // the highest accepted for this key (a relay replaying a stale,
                // still-within-TTL record). Equal seq = same signed bytes → allow
                // (idempotent re-resolve after cache expiry). Update on accept,
                // only after every other gate passed so a bad record can't poison
                // the highwater.
                {
                    let mut hw = self.seq_highwater.lock().expect("seq_highwater poisoned");
                    if let Some(&seen) = hw.get(&pk_bytes) {
                        if seq < seen {
                            tracing::warn!(
                                key = %key_z32,
                                seq,
                                seen,
                                "pkarr record seq rolled back — dropping (RPK5)"
                            );
                            // Do NOT negative-cache: a rollback means a newer
                            // record exists (highwater > seq), just not on the
                            // relay that answered first-hit here. Caching None
                            // would mask it for the whole negative-cache TTL,
                            // turning one stale relay into a hard miss. Leave the
                            // cache untouched so the next resolve (or a
                            // resolve_freshest cross-relay sweep) can recover it.
                            return Ok(None);
                        }
                    }
                    hw.put(pk_bytes, seq);
                }
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
            }
        }
    }

    /// Test-only: drop any cached resolution for a key, forcing the next
    /// `resolve` to hit the relay (so seq anti-rollback can be exercised).
    #[cfg(test)]
    pub(crate) fn invalidate_for_test(&self, pk: &[u8; 32]) {
        self.cache.lock().expect("cache poisoned").pop(pk);
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

    /// Cache-bypassing resolve that queries ALL relays and returns the freshest
    /// valid record by BEP44 `seq` (ties broken by latest `announced_at` within
    /// TTL). Used on a dial failure to cross-check relays — a single stale relay
    /// cannot pin the resolver to an old-but-within-TTL record. Updates the
    /// positive cache + seq highwater with the winner.
    ///
    /// NOTE (ZEB-817): callers resolving slots whose keys derive from public
    /// inputs (e.g. Case E vines) MUST use the `_with` variant — this fn
    /// accepts the freshest self-certified record without caller verification.
    pub async fn resolve_freshest(
        &self,
        pk: &VerifyingKey,
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        // Delegate: this IS the verified variant with an always-true predicate.
        // Sorted-first == the original reduce's winner (seq desc, announced_at
        // desc, stable ⇒ first-encountered wins an exact tie), and the
        // highwater/cache path is the same code, so behaviour is unchanged —
        // it just stops being a second copy of security-sensitive logic.
        self.resolve_freshest_with(pk, &|_: &PkarrRoutingRecord| true)
            .await
    }

    /// `resolve_freshest`, but the caller supplies the record-authenticity
    /// predicate that the self-certified inner signature cannot provide
    /// (ZEB-817).
    ///
    /// A pkarr slot whose key derives from public inputs (Case E vines) can be
    /// squatted: anyone can publish a record that passes the outer sig, its own
    /// inner sig and freshness, because the inner sig verifies against the
    /// record's *own* embedded `harmony_identity_pub`. `resolve_freshest` would
    /// hand back that squat as the freshest-by-seq winner — and worse, pin the
    /// seq-highwater and the positive cache with it, hiding the genuine record
    /// for the process lifetime.
    ///
    /// Here every surviving candidate is kept and ranked freshest-first; the
    /// first one for which `verify` returns true is the winner. Candidates that
    /// fail `verify` are discarded before the highwater/cache write, so they can
    /// influence neither surface. Anti-rollback still applies to the *verified*
    /// winner: a verified record older than the highwater is rejected as usual.
    ///
    /// Returns `Ok(None)` when no candidate verifies (nothing cached, nothing
    /// pinned), so a later genuine publish still resolves. That case is logged
    /// at debug with per-reason reject counts, which is what distinguishes
    /// "no record published" from "present but unverifiable".
    ///
    /// # Callback contract
    ///
    /// `verify` is invoked once per surviving candidate, in freshest-first
    /// order, and **outside every resolver lock** (no re-entrancy hazard — it
    /// may itself call back into the resolver). It may be called zero times (no
    /// candidate survived the sig/freshness gates) or once per relay that
    /// answered, so it must be cheap and pure — do not perform I/O or mutate
    /// shared state in it, and do not rely on the call count. It runs *after*
    /// outer-sig, inner-sig and freshness, so it can only ever **narrow** the
    /// accepted set, never widen it. Returning `false` skips that candidate
    /// without touching the seq-highwater or the positive cache.
    pub async fn resolve_freshest_with<F>(
        &self,
        pk: &VerifyingKey,
        verify: &F,
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError>
    where
        F: Fn(&PkarrRoutingRecord) -> bool + Sync,
    {
        let pk_bytes = pk.to_bytes();
        let key_z32 = crate::wire::z32_for_verifying_key(pk)?;
        let hits = self.relay.get_all(&key_z32).await?;
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock < UNIX epoch is unsupported")
            .as_millis() as u64;

        // Reject tallies for the one-line miss diagnostic below. Per-candidate
        // logging would be spam on a wide relay fan-out; `resolve` already
        // warns per-RPK-code on the single-hit path.
        let hits_total = hits.len();
        let mut parse_rejected = 0usize;
        let mut inner_sig_rejected = 0usize;
        let mut freshness_rejected = 0usize;

        let mut candidates: Vec<(u64, PkarrRoutingRecord)> = Vec::new();
        for (_relay, envelope) in hits {
            let (record, seq) = match crate::wire::parse_relay_payload(&pk_bytes, &envelope) {
                Ok(v) => v,
                Err(_) => {
                    parse_rejected += 1;
                    continue;
                }
            };
            if record.verify_inner_sig().is_err() {
                inner_sig_rejected += 1;
                continue;
            }
            if record.verify_freshness(now_ms).is_err() {
                freshness_rejected += 1;
                continue;
            }
            candidates.push((seq, record));
        }
        // Freshest-first: seq primary, announced_at_ms tiebreak.
        candidates.sort_by(|(sa, ra), (sb, rb)| {
            sb.cmp(sa).then(rb.announced_at_ms.cmp(&ra.announced_at_ms))
        });
        // First candidate passing FULL caller verification wins. Candidates
        // that fail verification never touch the highwater or the cache
        // (ZEB-817: an unverified record must not pin either surface).
        //
        // If `find` yields None then every one of these failed `verify`, so
        // this count IS the caller-verify reject count in the miss branch.
        let self_certified = candidates.len();
        let best = candidates.into_iter().find(|(_, rec)| verify(rec));

        match best {
            Some((seq, record)) => {
                // Anti-rollback: never accept a winner older than what we've seen.
                {
                    let mut hw = self.seq_highwater.lock().expect("seq_highwater poisoned");
                    if let Some(&seen) = hw.get(&pk_bytes) {
                        if seq < seen {
                            return Ok(None);
                        }
                    }
                    hw.put(pk_bytes, seq);
                }
                self.cache_put(pk_bytes, Some(record.clone()), POSITIVE_CACHE_TTL);
                Ok(Some(record))
            }
            None => {
                // Relays answered but nothing survived: tell the operator which
                // gate ate them. Silent on a genuine all-404 (hits_total == 0),
                // which is an ordinary "not published" and not worth a line.
                if hits_total > 0 {
                    tracing::debug!(
                        key = %key_z32,
                        hits = hits_total,
                        parse_rejected,
                        inner_sig_rejected,
                        freshness_rejected,
                        verify_rejected = self_certified,
                        "pkarr freshest-resolve found no verified record — relays answered but every \
                         candidate was dropped (present-but-unverified, not absent)"
                    );
                }
                Ok(None)
            }
        }
    }

    /// `resolve_freshest` across the epoch-tolerance key window; returns the
    /// freshest valid record found for any key.
    ///
    /// NOTE (ZEB-817): callers resolving slots whose keys derive from public
    /// inputs (e.g. Case E vines) MUST use the `_with` variant — this fn
    /// accepts the freshest self-certified record without caller verification.
    pub async fn resolve_window_freshest(
        &self,
        keys: &[VerifyingKey],
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError> {
        // Delegate — see `resolve_freshest`'s note on why this is behaviour-
        // preserving.
        self.resolve_window_freshest_with(keys, &|_: &PkarrRoutingRecord| true)
            .await
    }

    /// `resolve_window_freshest` with the caller's record-authenticity
    /// predicate applied per key (ZEB-817) — see `resolve_freshest_with`.
    ///
    /// The window winner is chosen only among records that already passed
    /// `verify`, so a squat on one epoch-window key cannot shadow the genuine
    /// record published under a sibling key.
    ///
    /// # Callback contract
    ///
    /// As `resolve_freshest_with`, with one window-specific amplification:
    /// `verify` is called concurrently from the per-key `join_all`, so across a
    /// whole window it may run many times and in no guaranteed order. It must
    /// be cheap, pure, and free of shared-state mutation; it is never invoked
    /// while a resolver lock is held.
    pub async fn resolve_window_freshest_with<F>(
        &self,
        keys: &[VerifyingKey],
        verify: &F,
    ) -> Result<Option<PkarrRoutingRecord>, PkarrError>
    where
        F: Fn(&PkarrRoutingRecord) -> bool + Sync,
    {
        let futures = keys.iter().map(|pk| self.resolve_freshest_with(pk, verify));
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
        // Mirror `resolve_window_freshest`: prefer a definitive answer (a
        // record, or a confirmed-absent Ok(None) from any key) over a transient
        // error on a sibling key. Only surface Err when every key errored.
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
        // RPK4 re-verification on cache hit: a record whose signed TTL expires
        // during the 15min positive-cache window must not keep being served.
        // Re-check freshness on every lookup; on failure, evict + treat as miss
        // (the next resolve will go to the relay and either negative-cache or
        // refetch a fresh record).
        if let Some(rec) = entry.record.as_ref() {
            let now_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock < UNIX epoch is unsupported")
                .as_millis() as u64;
            if rec.verify_freshness(now_ms).is_err() {
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

    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("system clock < UNIX epoch is unsupported")
            .as_millis() as u64
    }

    /// PUT `record` at BEP44 `seq` to exactly ONE relay, via a single-relay
    /// client. Each `MockPkarrRelay` stores one envelope per key (latest write
    /// wins), so competing records need one relay each.
    async fn put_to_relay(
        relay: &MockPkarrRelay,
        key_z32: &str,
        ephemeral: &SigningKey,
        record: &PkarrRoutingRecord,
        seq: u64,
    ) {
        let client = crate::relay::RelayClient::new(crate::relay::RelayPool::new(vec![relay
            .base_url
            .clone()]));
        client
            .put(
                key_z32,
                &crate::wire::build_relay_payload_with_seq(ephemeral, record, seq).unwrap(),
            )
            .await
            .unwrap();
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
                now_ms + 604_800_000,
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
                now_ms + 604_800_000,
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
            now_ms + 604_800_000,
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

    #[tokio::test]
    async fn rejects_rolled_back_seq() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let identity_sk = SigningKey::generate(&mut OsRng);
        let identity_pub = fixture_identity_pubkey(&identity_sk);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let rec = PkarrRoutingRecord::sign_new(
            b"r".to_vec(),
            identity_pub,
            now,
            now + 604_800_000,
            &identity_sk,
        )
        .expect("sign");

        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();

        // Publish with a HIGH seq, resolve → accepted, highwater = 200.
        let hi = crate::wire::build_relay_payload_with_seq(&ephemeral, &rec, 200).unwrap();
        client.put(&key_z32, &hi).await.unwrap();
        assert!(resolver.resolve(&vk).await.unwrap().is_some());

        // A relay now serves an OLDER seq for the same key → rolled back → drop.
        let lo = crate::wire::build_relay_payload_with_seq(&ephemeral, &rec, 100).unwrap();
        client.put(&key_z32, &lo).await.unwrap();
        // Bypass the positive cache to force a fresh GET + seq check.
        resolver.invalidate_for_test(&vk.to_bytes());
        assert!(
            resolver.resolve(&vk).await.unwrap().is_none(),
            "older seq must be rejected as rollback"
        );
    }

    #[tokio::test]
    async fn resolve_freshest_beats_stale_relay() {
        // Two relays: one holds an OLD seq, one holds a NEW seq. resolve_freshest
        // must return the new one regardless of pool order (stale listed first).
        let stale = MockPkarrRelay::start().await;
        let fresh = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![
            stale.base_url.clone(), // stale FIRST — first-hit `get` would pick this
            fresh.base_url.clone(),
        ]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(Arc::clone(&client));

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let id_sk = SigningKey::generate(&mut OsRng);
        let id_pub = fixture_identity_pubkey(&id_sk);
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        let old_rec =
            PkarrRoutingRecord::sign_new(b"old".to_vec(), id_pub, now, now + 604_800_000, &id_sk)
                .unwrap();
        let new_rec =
            PkarrRoutingRecord::sign_new(b"new".to_vec(), id_pub, now, now + 604_800_000, &id_sk)
                .unwrap();

        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        // Put directly to each relay via single-relay clients.
        let stale_c = crate::relay::RelayClient::new(crate::relay::RelayPool::new(vec![stale
            .base_url
            .clone()]));
        let fresh_c = crate::relay::RelayClient::new(crate::relay::RelayPool::new(vec![fresh
            .base_url
            .clone()]));
        stale_c
            .put(
                &key_z32,
                &crate::wire::build_relay_payload_with_seq(&ephemeral, &old_rec, 100).unwrap(),
            )
            .await
            .unwrap();
        fresh_c
            .put(
                &key_z32,
                &crate::wire::build_relay_payload_with_seq(&ephemeral, &new_rec, 200).unwrap(),
            )
            .await
            .unwrap();

        let got = resolver
            .resolve_freshest(&vk)
            .await
            .unwrap()
            .expect("present");
        assert_eq!(got.routing_blob, b"new", "freshest-by-seq must win");
    }

    /// ZEB-817: an attacker record with a higher seq but failing caller
    /// verification must not shadow the genuine record, must not pin the
    /// seq-highwater, and must not enter the positive cache.
    #[tokio::test]
    async fn verified_resolve_defeats_higher_seq_squat() {
        let squat = MockPkarrRelay::start().await;
        let genuine = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![
            squat.base_url.clone(), // squat FIRST — it also holds the higher seq
            genuine.base_url.clone(),
        ]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();

        let genuine_sk = SigningKey::generate(&mut OsRng);
        let genuine_pub = fixture_identity_pubkey(&genuine_sk);
        let attacker_sk = SigningKey::generate(&mut OsRng);
        let attacker_pub = fixture_identity_pubkey(&attacker_sk);

        let now = now_ms();
        let genuine_rec = PkarrRoutingRecord::sign_new(
            b"genuine".to_vec(),
            genuine_pub,
            now,
            now + 604_800_000,
            &genuine_sk,
        )
        .unwrap();
        // Self-certifying: the squat passes outer sig, inner sig and freshness.
        let attacker_rec = PkarrRoutingRecord::sign_new(
            b"squat".to_vec(),
            attacker_pub,
            now + 1,
            now + 604_800_000,
            &attacker_sk,
        )
        .unwrap();

        // Genuine record seq 100 on relay 2; attacker seq 200 on relay 1.
        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        put_to_relay(&genuine, &key_z32, &ephemeral, &genuine_rec, 100).await;
        put_to_relay(&squat, &key_z32, &ephemeral, &attacker_rec, 200).await;

        let verify = |rec: &PkarrRoutingRecord| rec.harmony_identity_pub == genuine_pub;

        // 1. Squat defeated: genuine (lower-seq) record wins.
        let got = resolver
            .resolve_freshest_with(&vk, &verify)
            .await
            .unwrap()
            .expect("genuine record must resolve");
        assert_eq!(got.routing_blob, b"genuine");

        // 2. Positive cache holds the VERIFIED winner, not the squat: a plain
        // cache-first `resolve` (which shares the cache) must not serve it.
        let cached = resolver
            .resolve(&vk)
            .await
            .unwrap()
            .expect("verified winner must be cached");
        assert_eq!(cached.routing_blob, b"genuine");

        // 3. Highwater NOT pinned by the unverified seq-200 record: bypass the
        // positive cache and resolve again — still the genuine record (a pinned
        // highwater would reject seq 100 as rollback and return None).
        resolver.invalidate_for_test(&vk.to_bytes());
        let again = resolver.resolve_freshest_with(&vk, &verify).await.unwrap();
        assert_eq!(
            again
                .expect("genuine record must still resolve")
                .routing_blob,
            b"genuine"
        );
    }

    /// Verified rollback is still rejected: highwater must keep guarding
    /// against replay of an older VERIFIED record.
    #[tokio::test]
    async fn verified_resolve_still_rejects_verified_rollback() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let id_sk = SigningKey::generate(&mut OsRng);
        let id_pub = fixture_identity_pubkey(&id_sk);
        let now = now_ms();
        let rec = PkarrRoutingRecord::sign_new(
            b"genuine".to_vec(),
            id_pub,
            now,
            now + 604_800_000,
            &id_sk,
        )
        .unwrap();
        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        let verify = |r: &PkarrRoutingRecord| r.harmony_identity_pub == id_pub;

        // Accepted → highwater := 200.
        put_to_relay(&relay, &key_z32, &ephemeral, &rec, 200).await;
        assert!(
            resolver
                .resolve_freshest_with(&vk, &verify)
                .await
                .unwrap()
                .is_some(),
            "verified seq-200 record must be accepted"
        );

        // The relay now replays the same verified record at an OLDER seq
        // (latest write wins on the mock, so seq 200 is gone).
        put_to_relay(&relay, &key_z32, &ephemeral, &rec, 100).await;
        resolver.invalidate_for_test(&vk.to_bytes());
        assert!(
            resolver
                .resolve_freshest_with(&vk, &verify)
                .await
                .unwrap()
                .is_none(),
            "verified rollback must still be rejected by the seq highwater"
        );
    }

    /// The UNVERIFIED window fn now delegates to the `_with` variant with an
    /// always-true predicate; pin that the delegation is behaviour-preserving
    /// end-to-end — cross-key freshest-by-`announced_at_ms` still wins, and it
    /// still accepts a self-certified record from an identity nobody vouched
    /// for (exactly the property its ZEB-817 doc note warns callers about).
    #[tokio::test]
    async fn unverified_window_freshest_accepts_any_self_certified() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let key_a = SigningKey::generate(&mut OsRng);
        let key_b = SigningKey::generate(&mut OsRng);
        // Two UNRELATED identities — no caller predicate vouches for either.
        let id_a = SigningKey::generate(&mut OsRng);
        let id_b = SigningKey::generate(&mut OsRng);
        let now = now_ms();
        let older = PkarrRoutingRecord::sign_new(
            b"older".to_vec(),
            fixture_identity_pubkey(&id_a),
            now - 60_000,
            now + 604_800_000,
            &id_a,
        )
        .unwrap();
        let newer = PkarrRoutingRecord::sign_new(
            b"newer".to_vec(),
            fixture_identity_pubkey(&id_b),
            now,
            now + 604_800_000,
            &id_b,
        )
        .unwrap();

        let z32_a = crate::wire::z32_for_verifying_key(&key_a.verifying_key()).unwrap();
        let z32_b = crate::wire::z32_for_verifying_key(&key_b.verifying_key()).unwrap();
        put_to_relay(&relay, &z32_a, &key_a, &older, 100).await;
        put_to_relay(&relay, &z32_b, &key_b, &newer, 100).await;

        let window = [key_a.verifying_key(), key_b.verifying_key()];
        let got = resolver
            .resolve_window_freshest(&window)
            .await
            .unwrap()
            .expect("unverified window resolve must return a record");
        assert_eq!(
            got.routing_blob, b"newer",
            "cross-key winner must be the later announced_at_ms"
        );
    }

    /// Freshest-wins ordering among candidates that ALL pass verification.
    /// Every other test has at most one verifying candidate per key, so `find`
    /// never has to *choose* — this is what pins the comparator direction and
    /// keeps the `_with` variant inheriting `resolve_freshest`'s guarantee that
    /// a single lagging relay cannot serve an old-but-within-TTL record.
    #[tokio::test]
    async fn verified_resolve_picks_freshest_among_multiple_verified() {
        let stale = MockPkarrRelay::start().await;
        let fresh = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![
            stale.base_url.clone(), // stale FIRST — first-hit order and an
            fresh.base_url.clone(), // ascending comparator both pick it
        ]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        // ONE identity signs both records, so both pass `verify` and the sort
        // order is the only thing that can decide the winner.
        let id_sk = SigningKey::generate(&mut OsRng);
        let id_pub = fixture_identity_pubkey(&id_sk);
        let now = now_ms();
        let old_rec =
            PkarrRoutingRecord::sign_new(b"old".to_vec(), id_pub, now, now + 604_800_000, &id_sk)
                .unwrap();
        let new_rec =
            PkarrRoutingRecord::sign_new(b"new".to_vec(), id_pub, now, now + 604_800_000, &id_sk)
                .unwrap();

        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        put_to_relay(&stale, &key_z32, &ephemeral, &old_rec, 100).await;
        put_to_relay(&fresh, &key_z32, &ephemeral, &new_rec, 200).await;

        let verify = |rec: &PkarrRoutingRecord| rec.harmony_identity_pub == id_pub;
        let got = resolver
            .resolve_freshest_with(&vk, &verify)
            .await
            .unwrap()
            .expect("a verified record must resolve");
        assert_eq!(
            got.routing_blob, b"new",
            "highest seq must win among verified candidates"
        );
    }

    /// Equal-seq tiebreak direction, both candidates verifying: the later
    /// `announced_at_ms` wins. The later-announced record sits on the SECOND
    /// relay, so a flipped tiebreak — or a dropped one, leaving the stable sort
    /// holding relay order — returns the earlier record instead.
    #[tokio::test]
    async fn verified_resolve_tiebreaks_equal_seq_by_announced_at() {
        let earlier_relay = MockPkarrRelay::start().await;
        let later_relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![
            earlier_relay.base_url.clone(),
            later_relay.base_url.clone(),
        ]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let id_sk = SigningKey::generate(&mut OsRng);
        let id_pub = fixture_identity_pubkey(&id_sk);
        let now = now_ms();
        let earlier = PkarrRoutingRecord::sign_new(
            b"earlier".to_vec(),
            id_pub,
            now - 60_000,
            now + 604_800_000,
            &id_sk,
        )
        .unwrap();
        let later =
            PkarrRoutingRecord::sign_new(b"later".to_vec(), id_pub, now, now + 604_800_000, &id_sk)
                .unwrap();

        // Identical seq on both relays: only the tiebreak can separate them.
        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();
        put_to_relay(&earlier_relay, &key_z32, &ephemeral, &earlier, 150).await;
        put_to_relay(&later_relay, &key_z32, &ephemeral, &later, 150).await;

        let verify = |rec: &PkarrRoutingRecord| rec.harmony_identity_pub == id_pub;
        let got = resolver
            .resolve_freshest_with(&vk, &verify)
            .await
            .unwrap()
            .expect("a verified record must resolve");
        assert_eq!(
            got.routing_blob, b"later",
            "equal seq must tiebreak on the later announced_at_ms"
        );
    }

    /// All candidates failing verification => Ok(None), and nothing is
    /// cached or pinned: a genuine record published afterwards resolves.
    #[tokio::test]
    async fn all_unverified_resolves_none_without_pinning() {
        let squat = MockPkarrRelay::start().await;
        let genuine = MockPkarrRelay::start().await;
        let pool =
            crate::relay::RelayPool::new(vec![squat.base_url.clone(), genuine.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        let ephemeral = SigningKey::generate(&mut OsRng);
        let vk = ephemeral.verifying_key();
        let key_z32 = crate::wire::z32_for_verifying_key(&vk).unwrap();

        let genuine_sk = SigningKey::generate(&mut OsRng);
        let genuine_pub = fixture_identity_pubkey(&genuine_sk);
        let attacker_sk = SigningKey::generate(&mut OsRng);
        let attacker_pub = fixture_identity_pubkey(&attacker_sk);
        let now = now_ms();
        let genuine_rec = PkarrRoutingRecord::sign_new(
            b"genuine".to_vec(),
            genuine_pub,
            now,
            now + 604_800_000,
            &genuine_sk,
        )
        .unwrap();
        let attacker_rec = PkarrRoutingRecord::sign_new(
            b"squat".to_vec(),
            attacker_pub,
            now,
            now + 604_800_000,
            &attacker_sk,
        )
        .unwrap();

        let verify = |rec: &PkarrRoutingRecord| rec.harmony_identity_pub == genuine_pub;

        // Only the attacker has published — no candidate verifies.
        put_to_relay(&squat, &key_z32, &ephemeral, &attacker_rec, 200).await;
        assert!(
            resolver
                .resolve_freshest_with(&vk, &verify)
                .await
                .unwrap()
                .is_none(),
            "no verified candidate must resolve to Ok(None)"
        );

        // The genuine publisher lands a LOWER seq afterwards: neither a pinned
        // highwater nor a poisoned cache may hide it.
        put_to_relay(&genuine, &key_z32, &ephemeral, &genuine_rec, 100).await;
        resolver.invalidate_for_test(&vk.to_bytes());
        let got = resolver
            .resolve_freshest_with(&vk, &verify)
            .await
            .unwrap()
            .expect("genuine record must resolve after the failed squat");
        assert_eq!(got.routing_blob, b"genuine");
    }

    /// The window wrapper picks its cross-key winner only among VERIFIED
    /// records: a squat on one epoch-window key cannot shadow the genuine
    /// record published under a sibling key.
    #[tokio::test]
    async fn verified_resolve_window_skips_squatted_key() {
        let relay = MockPkarrRelay::start().await;
        let pool = crate::relay::RelayPool::new(vec![relay.base_url.clone()]);
        let client = Arc::new(crate::relay::RelayClient::new(pool));
        let resolver = PkarrResolver::new(client);

        // Two distinct window keys: the squatted one is queried first.
        let squatted_key = SigningKey::generate(&mut OsRng);
        let genuine_key = SigningKey::generate(&mut OsRng);

        let genuine_sk = SigningKey::generate(&mut OsRng);
        let genuine_pub = fixture_identity_pubkey(&genuine_sk);
        let attacker_sk = SigningKey::generate(&mut OsRng);
        let attacker_pub = fixture_identity_pubkey(&attacker_sk);
        let now = now_ms();
        let genuine_rec = PkarrRoutingRecord::sign_new(
            b"genuine".to_vec(),
            genuine_pub,
            now,
            now + 604_800_000,
            &genuine_sk,
        )
        .unwrap();
        // Newer announced_at as well as higher seq: would win the cross-key
        // `announced_at_ms` comparison if it were ever allowed to be a winner.
        let attacker_rec = PkarrRoutingRecord::sign_new(
            b"squat".to_vec(),
            attacker_pub,
            now + 1,
            now + 604_800_000,
            &attacker_sk,
        )
        .unwrap();

        let squatted_z32 =
            crate::wire::z32_for_verifying_key(&squatted_key.verifying_key()).unwrap();
        let genuine_z32 = crate::wire::z32_for_verifying_key(&genuine_key.verifying_key()).unwrap();
        put_to_relay(&relay, &squatted_z32, &squatted_key, &attacker_rec, 200).await;
        put_to_relay(&relay, &genuine_z32, &genuine_key, &genuine_rec, 100).await;

        let verify = |rec: &PkarrRoutingRecord| rec.harmony_identity_pub == genuine_pub;
        let window = [squatted_key.verifying_key(), genuine_key.verifying_key()];
        let got = resolver
            .resolve_window_freshest_with(&window, &verify)
            .await
            .unwrap()
            .expect("genuine record must resolve from the sibling key");
        assert_eq!(got.routing_blob, b"genuine");
    }

    /// harmony-client drives these from `tokio::spawn`-ed tasks, so the
    /// returned futures must be `Send`. That is exactly what the `F: Sync`
    /// bound buys (`&F: Send` iff `F: Sync`), plus holding no `MutexGuard`
    /// across an await — pin both at compile time. The futures are never
    /// polled here, so no relay is contacted.
    #[test]
    fn verified_resolve_futures_are_send() {
        fn assert_send<T: Send>(_: T) {}

        let client = Arc::new(crate::relay::RelayClient::new(
            crate::relay::RelayPool::new(vec!["http://127.0.0.1:1".to_string()]),
        ));
        let resolver = PkarrResolver::new(client);
        let vk = SigningKey::generate(&mut OsRng).verifying_key();
        let verify = |_: &PkarrRoutingRecord| true;

        assert_send(resolver.resolve_freshest_with(&vk, &verify));
        assert_send(resolver.resolve_window_freshest_with(std::slice::from_ref(&vk), &verify));
        // The unverified fns now delegate, holding a temporary always-true
        // closure across the await — their existing callers still spawn them.
        assert_send(resolver.resolve_freshest(&vk));
        assert_send(resolver.resolve_window_freshest(std::slice::from_ref(&vk)));
    }
}

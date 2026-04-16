//! TTL + LRU caches for the resolver (spec §5.2).
//!
//! Four positive caches (claim, signing-key, master-key, revocation),
//! one negative cache, and one "domain last seen" sliding-window cache
//! for the 72h soft-fail. Time is injected via `TimeSource` so tests
//! control expiry with `FakeTimeSource::advance`.

use std::sync::atomic::{AtomicU64, Ordering};

use dashmap::DashMap;

use crate::claim::{DomainRecord, RevocationView, SignedClaim, SigningKeyCert};

/// Clock abstraction. Production uses `SystemTimeSource`; tests use
/// `FakeTimeSource` (see `test_support`).
pub trait TimeSource: Send + Sync + 'static {
    fn now(&self) -> u64;
}

pub struct SystemTimeSource;

impl TimeSource for SystemTimeSource {
    fn now(&self) -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

/// Cheap, monotonic counter used as an LRU-ish recency token. Not a
/// real LRU — we only need "evict oldest N when over bound," which a
/// monotonic counter + sort-and-drop achieves without a doubly-linked
/// list. Spec §7.8 allows coarse-grained eviction since the attack
/// model is "bound memory under adversarial input," not "serve hot
/// items preferentially."
#[derive(Debug)]
pub struct RecencyCounter(AtomicU64);

impl RecencyCounter {
    pub fn new() -> Self {
        Self(AtomicU64::new(1))
    }
    pub fn tick(&self) -> u64 {
        self.0.fetch_add(1, Ordering::Relaxed)
    }
}

impl Default for RecencyCounter {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheEntry<T> {
    pub value: T,
    pub expires_at: u64,
    pub recency: u64,
}

impl<T> CacheEntry<T> {
    pub fn is_expired(&self, now: u64) -> bool {
        now >= self.expires_at
    }
}

// ─── Type aliases ────────────────────────────────────────────────────────────

pub type HashedLocalPart = [u8; 32];
pub type SigningKeyId = [u8; 8];

// ─── Cache limits ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CacheLimits {
    pub claim_max: usize,
    pub signing_key_max: usize,
    pub master_key_max: usize,
    pub revocation_max: usize,
    pub negative_max: usize,
}

impl Default for CacheLimits {
    fn default() -> Self {
        Self {
            claim_max: 10_000,
            signing_key_max: 10_000,
            master_key_max: 10_000,
            revocation_max: 10_000,
            negative_max: 10_000,
        }
    }
}

// ─── ResolverCaches ──────────────────────────────────────────────────────────

/// All resolver caches in one place.
///
/// `claim` stores decoded `SignedClaim` values keyed by (domain, hashed_local_part).
/// The other positive caches store their decoded types.
pub struct ResolverCaches {
    limits: CacheLimits,
    recency: RecencyCounter,
    // Positive caches — value is stored in a CacheEntry that tracks
    // expires_at (set by caller to the correct TTL: claim.expires_at,
    // cert.valid_until, DNS TTL, now+6h respectively).
    claim: DashMap<(String, HashedLocalPart), CacheEntry<SignedClaim>>,
    signing_key: DashMap<(String, SigningKeyId), CacheEntry<SigningKeyCert>>,
    master_key: DashMap<String, CacheEntry<DomainRecord>>,
    revocation: DashMap<String, CacheEntry<RevocationView>>,
    // Sliding window: last time we saw this domain resolve successfully.
    domain_last_seen: DashMap<String, u64>,
    // Negative cache: key -> expires_at.
    negative: DashMap<(String, HashedLocalPart), u64>,
    // Track last successful revocation refresh per domain for 24h safety valve.
    revocation_last_refreshed: DashMap<String, u64>,
}

impl ResolverCaches {
    pub fn new(limits: CacheLimits) -> Self {
        Self {
            limits,
            recency: RecencyCounter::new(),
            claim: DashMap::new(),
            signing_key: DashMap::new(),
            master_key: DashMap::new(),
            revocation: DashMap::new(),
            domain_last_seen: DashMap::new(),
            negative: DashMap::new(),
            revocation_last_refreshed: DashMap::new(),
        }
    }

    // --- Claim cache ---

    pub fn put_claim(
        &self,
        key: (String, HashedLocalPart),
        value: SignedClaim,
        expires_at: u64,
        _recency_hint: u64,
    ) {
        let entry = CacheEntry {
            value,
            expires_at,
            recency: self.recency.tick(),
        };
        self.claim.insert(key, entry);
        self.enforce_bound(&self.claim, self.limits.claim_max);
    }

    pub fn get_claim(&self, key: &(String, HashedLocalPart), now: u64) -> Option<SignedClaim> {
        let entry = self.claim.get(key)?;
        if entry.is_expired(now) {
            return None;
        }
        Some(entry.value.clone())
    }

    // --- Signing-key cache ---

    pub fn put_signing_key(
        &self,
        key: (String, SigningKeyId),
        cert: SigningKeyCert,
        expires_at: u64,
    ) {
        let entry = CacheEntry {
            value: cert,
            expires_at,
            recency: self.recency.tick(),
        };
        self.signing_key.insert(key, entry);
        self.enforce_bound(&self.signing_key, self.limits.signing_key_max);
    }

    pub fn get_signing_key(
        &self,
        key: &(String, SigningKeyId),
        now: u64,
    ) -> Option<SigningKeyCert> {
        let e = self.signing_key.get(key)?;
        if e.is_expired(now) {
            return None;
        }
        Some(e.value.clone())
    }

    // --- Master-key (DomainRecord) cache ---

    pub fn put_master_key(&self, domain: &str, rec: DomainRecord, expires_at: u64) {
        let entry = CacheEntry {
            value: rec,
            expires_at,
            recency: self.recency.tick(),
        };
        self.master_key.insert(domain.to_ascii_lowercase(), entry);
        self.enforce_bound(&self.master_key, self.limits.master_key_max);
    }

    pub fn get_master_key(&self, domain: &str, now: u64) -> Option<DomainRecord> {
        let e = self.master_key.get(&domain.to_ascii_lowercase())?;
        if e.is_expired(now) {
            return None;
        }
        Some(e.value.clone())
    }

    // --- Revocation cache ---

    pub fn put_revocation(
        &self,
        domain: &str,
        view: RevocationView,
        expires_at: u64,
        last_refreshed: u64,
    ) {
        let entry = CacheEntry {
            value: view,
            expires_at,
            recency: self.recency.tick(),
        };
        let d = domain.to_ascii_lowercase();
        self.revocation.insert(d.clone(), entry);
        self.revocation_last_refreshed.insert(d, last_refreshed);
        self.enforce_bound(&self.revocation, self.limits.revocation_max);
    }

    /// Returns the cached revocation view WHETHER OR NOT it's past
    /// expires_at — the resolver uses expires_at to decide when to
    /// refresh, and the 24h safety valve to decide when to stop serving.
    pub fn get_revocation(
        &self,
        domain: &str,
    ) -> Option<(
        RevocationView,
        /* expires_at */ u64,
        /* last_refreshed */ u64,
    )> {
        let d = domain.to_ascii_lowercase();
        let e = self.revocation.get(&d)?;
        let last_refreshed = self
            .revocation_last_refreshed
            .get(&d)
            .map(|v| *v)
            .unwrap_or(0);
        Some((e.value.clone(), e.expires_at, last_refreshed))
    }

    // --- Domain last-seen (72h soft-fail window) ---

    pub fn mark_domain_seen(&self, domain: &str, now: u64) {
        self.domain_last_seen
            .insert(domain.to_ascii_lowercase(), now);
    }

    pub fn was_domain_seen_within(&self, domain: &str, now: u64, window_secs: u64) -> bool {
        self.domain_last_seen
            .get(&domain.to_ascii_lowercase())
            .map(|last| now < last.saturating_add(window_secs))
            .unwrap_or(false)
    }

    // --- Negative cache ---

    pub fn mark_negative(&self, key: (String, HashedLocalPart), expires_at: u64) {
        self.negative.insert(key, expires_at);
        self.enforce_bound_negative();
    }

    pub fn is_negative(&self, key: &(String, HashedLocalPart), now: u64) -> bool {
        self.negative.get(key).map(|e| now < *e).unwrap_or(false)
    }

    pub fn has_claim_key(&self, key: &(String, HashedLocalPart)) -> bool {
        self.claim.contains_key(key)
    }

    pub fn sweep_domain_last_seen(&self, now: u64, window_secs: u64) {
        self.domain_last_seen
            .retain(|_, last| now < last.saturating_add(window_secs));
    }

    // --- Sweep ---

    /// Evict expired entries from claim, signing_key, master_key, and negative
    /// caches. Does NOT touch the revocation cache — the resolver controls
    /// staleness there via the 24h safety valve.
    pub fn sweep_expired(&self, now: u64) {
        self.claim.retain(|_, e| !e.is_expired(now));
        self.signing_key.retain(|_, e| !e.is_expired(now));
        self.master_key.retain(|_, e| !e.is_expired(now));
        // revocation cache is NOT swept here.
        self.negative.retain(|_, expires_at| now < *expires_at);
    }

    // --- Private helpers ---

    fn enforce_bound<K, V>(&self, map: &DashMap<K, CacheEntry<V>>, max: usize)
    where
        K: std::hash::Hash + Eq + Clone,
    {
        if map.len() <= max {
            return;
        }
        let mut pairs: Vec<_> = map
            .iter()
            .map(|e| (e.value().recency, e.key().clone()))
            .collect();
        pairs.sort_by_key(|(r, _)| *r);
        let excess = map.len().saturating_sub(max);
        for (_, k) in pairs.into_iter().take(excess) {
            map.remove(&k);
        }
    }

    /// Returns the domains whose revocation list is due for a proactive
    /// refresh: those where `last_refreshed + refresh_secs < now` (spec §6.3).
    ///
    /// Strict inequality: at exactly the refresh boundary we wait one more
    /// tick so that the cadence matches the spec's "every 6h" exactly.
    pub fn revocation_refresh_candidates(&self, now: u64, refresh_secs: u64) -> Vec<String> {
        self.revocation_last_refreshed
            .iter()
            .filter(|e| now.saturating_sub(*e.value()) > refresh_secs)
            .map(|e| e.key().clone())
            .collect()
    }

    /// LRU-bound eviction for the negative cache whose values are raw `u64`
    /// expiry timestamps (not `CacheEntry<T>`). Evicts entries with the
    /// lowest expires_at (i.e., inserted earliest, since all entries share
    /// the same +60 s TTL).
    fn enforce_bound_negative(&self) {
        let max = self.limits.negative_max;
        if self.negative.len() <= max {
            return;
        }
        let mut pairs: Vec<_> = self
            .negative
            .iter()
            .map(|e| (*e.value(), e.key().clone()))
            .collect();
        pairs.sort_by_key(|(expires, _)| *expires);
        let excess = self.negative.len().saturating_sub(max);
        for (_, k) in pairs.into_iter().take(excess) {
            self.negative.remove(&k);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::{ClaimBuilder, FakeTimeSource, TestDomain};
    use rand_core::OsRng;

    fn build_claim() -> crate::claim::SignedClaim {
        let mut rng = OsRng;
        let d = TestDomain::new(&mut rng, "q8.fyi");
        let sk = d.mint_signing_key(&mut rng, 1000, 10_000_000);
        ClaimBuilder::new(&d, &sk, 1000).build()
    }

    #[test]
    fn cache_entry_is_expired_at_boundary() {
        let entry = CacheEntry {
            value: 42u32,
            expires_at: 100,
            recency: 1,
        };
        assert!(!entry.is_expired(99));
        assert!(entry.is_expired(100));
        assert!(entry.is_expired(101));
    }

    #[test]
    fn fake_time_source_advances() {
        let t = FakeTimeSource::new(1000);
        assert_eq!(t.now(), 1000);
        t.advance(500);
        assert_eq!(t.now(), 1500);
    }

    #[test]
    fn claim_cache_inserts_and_retrieves() {
        let caches = ResolverCaches::new(CacheLimits::default());
        let key = ("q8.fyi".to_string(), [0x11u8; 32]);
        let claim = build_claim();
        caches.put_claim(key.clone(), claim.clone(), 1000, 100);
        let got = caches.get_claim(&key, 500).expect("hit");
        assert_eq!(got, claim);
    }

    #[test]
    fn claim_cache_ttl_expiry_is_honored() {
        let caches = ResolverCaches::new(CacheLimits::default());
        let key = ("q8.fyi".to_string(), [0x11u8; 32]);
        let claim = build_claim();
        caches.put_claim(key.clone(), claim, 1000, 100);
        assert!(caches.get_claim(&key, 999).is_some());
        assert!(
            caches.get_claim(&key, 1000).is_none(),
            "expires_at is exclusive"
        );
        assert!(caches.get_claim(&key, 1500).is_none());
    }

    #[test]
    fn claim_cache_lru_bound_evicts_oldest() {
        let limits = CacheLimits {
            claim_max: 3,
            ..CacheLimits::default()
        };
        let caches = ResolverCaches::new(limits);
        let claim = build_claim();
        for i in 0..5u8 {
            let key = ("q8.fyi".to_string(), [i; 32]);
            caches.put_claim(key, claim.clone(), 10_000, i as u64);
        }
        // After inserting 5, only the 3 most recent (i=2,3,4) should remain.
        assert!(caches
            .get_claim(&("q8.fyi".to_string(), [0; 32]), 100)
            .is_none());
        assert!(caches
            .get_claim(&("q8.fyi".to_string(), [1; 32]), 100)
            .is_none());
        assert!(caches
            .get_claim(&("q8.fyi".to_string(), [2; 32]), 100)
            .is_some());
        assert!(caches
            .get_claim(&("q8.fyi".to_string(), [4; 32]), 100)
            .is_some());
    }

    #[test]
    fn negative_cache_expires_after_60s() {
        let caches = ResolverCaches::new(CacheLimits::default());
        let key = ("q8.fyi".to_string(), [0xff; 32]);
        caches.mark_negative(key.clone(), 1000 + 60);
        assert!(caches.is_negative(&key, 1050));
        assert!(!caches.is_negative(&key, 1060));
    }

    #[test]
    fn domain_last_seen_supports_72h_soft_fail_query() {
        let caches = ResolverCaches::new(CacheLimits::default());
        caches.mark_domain_seen("q8.fyi", 1000);
        assert!(caches.was_domain_seen_within("q8.fyi", 1000 + 3600, 72 * 3600));
        assert!(caches.was_domain_seen_within("q8.fyi", 1000 + 72 * 3600 - 1, 72 * 3600));
        assert!(!caches.was_domain_seen_within("q8.fyi", 1000 + 72 * 3600, 72 * 3600));
    }

    #[test]
    fn revocation_refresh_candidates_returns_stale_domains() {
        let caches = ResolverCaches::new(CacheLimits::default());
        // last_refreshed: fresh at 1000, stale at 100, boundary at 900.
        caches.put_revocation("fresh.example", RevocationView::empty(), 99999, 1000);
        caches.put_revocation("stale.example", RevocationView::empty(), 99999, 100);
        caches.put_revocation("boundary.example", RevocationView::empty(), 99999, 900);
        // now=1500, refresh_secs=600. Ages: fresh=500s, stale=1400s, boundary=600s.
        // Spec §6.3: refresh when last + refresh_secs < now (strict), so
        // boundary.example (exactly 600s old) must NOT be a candidate.
        let refresh_secs = 600;
        let now = 1500;
        let candidates = caches.revocation_refresh_candidates(now, refresh_secs);
        assert!(
            candidates.contains(&"stale.example".to_string()),
            "stale domain must be a candidate"
        );
        assert!(
            !candidates.contains(&"fresh.example".to_string()),
            "fresh domain must not be a candidate"
        );
        assert!(
            !candidates.contains(&"boundary.example".to_string()),
            "boundary domain (exactly refresh_secs old) must wait one more tick"
        );
    }

    #[test]
    fn sweep_evicts_expired_entries_across_caches() {
        let caches = ResolverCaches::new(CacheLimits::default());
        let claim = build_claim();
        caches.put_claim(("a".into(), [1; 32]), claim.clone(), 50, 1);
        caches.put_claim(("b".into(), [2; 32]), claim, 500, 2);
        caches.sweep_expired(100);
        assert!(caches.get_claim(&("a".into(), [1; 32]), 100).is_none());
        assert!(caches.get_claim(&("b".into(), [2; 32]), 100).is_some());
    }
}

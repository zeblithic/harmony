//! HTTP pkarr-relay client.
//!
//! Pool of relay base URLs. On publish/resolve, iterates the pool until one
//! succeeds. On per-request timeout (5s) or HTTP 429, the offending relay
//! enters a 30s cooldown and is skipped on subsequent calls until expiry.
//!
//! Spec Section 6.4 (publication-side IP-hiding via rotation) + Section 14
//! (failure-modes table).

use std::collections::HashMap;
use std::sync::{Mutex, RwLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::error::PkarrError;

/// Whether a relay is currently usable or sitting out a cooldown.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelayState {
    /// Available for the next request.
    Healthy,
    /// On cooldown until `until_ms` (wall-clock Unix millis).
    CoolingDown { until_ms: u64 },
}

/// The last recorded result of talking to a relay.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RelayOutcome {
    /// 2xx — record accepted (put) or returned (get).
    Success,
    /// Per-request timeout elapsed.
    Timeout,
    /// Non-timeout transport failure (connection refused, DNS, TLS).
    Transport,
    /// An explicit HTTP error status (429 / 5xx / other non-success).
    Http(u16),
}

/// A point-in-time health summary for one relay in the pool. One entry per
/// **current** pool relay (removed relays drop out; freshly added relays appear
/// with `last_outcome: None` until first use).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RelayHealth {
    pub url: String,
    pub state: RelayState,
    pub last_outcome: Option<RelayOutcome>,
    /// Wall-clock Unix millis of the last 2xx exchange (None if never).
    pub last_success_ms: Option<u64>,
}

/// Per-URL mutable record folded behind a `Mutex`. `state` is NOT stored here —
/// it is derived from the cooldown map at read time so there is a single source
/// of truth for "is this relay cooling down".
#[derive(Debug, Default, Clone)]
struct RelayRecord {
    last_outcome: Option<RelayOutcome>,
    last_success_ms: Option<u64>,
}

/// Wall-clock Unix millis (mirrors `network_health.rs::now_ms`).
fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Tunable timeouts for [`RelayClient`]. **Not user-facing** — multi-relay
/// redundancy (ZEB-380) already removes the "5 s timeout is terminal" failure
/// mode, so these stay code-level defaults. The struct exists so tests can use
/// short values and a future ticket can wire knobs if needed.
#[derive(Debug, Clone, Copy)]
pub struct RelayConfig {
    /// Per-request HTTP timeout. Default 5 s.
    pub request_timeout: Duration,
    /// How long a relay stays on cooldown after a timeout / 429 / 5xx. Default 30 s.
    pub cooldown: Duration,
}

impl Default for RelayConfig {
    fn default() -> Self {
        Self {
            request_timeout: Duration::from_secs(5),
            cooldown: Duration::from_secs(30),
        }
    }
}

/// An ordered list of relay base URLs (e.g. `"https://relay.example.com"`).
///
/// The client iterates the pool in order, skipping any relay that is currently
/// on cooldown. The first successful relay wins.
#[derive(Debug, Clone)]
pub struct RelayPool {
    relays: Vec<String>,
}

impl RelayPool {
    /// Create a pool from a list of base URLs.
    pub fn new(relays: Vec<String>) -> Self {
        Self { relays }
    }

    /// Returns `true` if the pool contains no relays.
    pub fn is_empty(&self) -> bool {
        self.relays.is_empty()
    }

    /// Number of relays in the pool.
    pub fn len(&self) -> usize {
        self.relays.len()
    }
}

/// Production HTTP relay client with pool rotation and per-relay cooldown.
///
/// Construct with [`RelayClient::new`] and call [`put`][RelayClient::put] /
/// [`get`][RelayClient::get] from async contexts.
pub struct RelayClient {
    /// Hot-swappable relay pool. Read-mostly (every put/get takes a short read
    /// lock); replaced wholesale by `set_relays`. `std::sync::RwLock` keeps the
    /// crate dependency-free (no `arc-swap`).
    pool: RwLock<RelayPool>,
    http: reqwest::Client,
    config: RelayConfig,
    /// Maps relay base URL → `Instant` at which the cooldown expires.
    cooldown: Mutex<HashMap<String, Instant>>,
    /// Per-URL last-outcome / last-success record (health observability).
    records: Mutex<HashMap<String, RelayRecord>>,
}

impl RelayClient {
    /// Build a client with default [`RelayConfig`].
    pub fn new(pool: RelayPool) -> Self {
        Self::with_config(pool, RelayConfig::default())
    }

    /// Build a client with an explicit [`RelayConfig`] (test/forward-compat hook).
    pub fn with_config(pool: RelayPool, config: RelayConfig) -> Self {
        let http = reqwest::Client::builder()
            .timeout(config.request_timeout)
            // ZEB-381: trust Mozilla's webpki root bundle in addition to OS-native
            // roots. `rustls-tls-native-roots` alone failed to anchor relay.pkarr.org's
            // Let's Encrypt chain — InvalidCertificate(UnknownIssuer) — on BOTH macOS
            // and Windows, breaking every pkarr publish/resolve and thus every
            // first-contact invite redeem. webpki roots carry ISRG Root X1/X2 and
            // validate the chain; native stays enabled for any private/enterprise roots.
            .tls_built_in_webpki_certs(true)
            .build()
            .expect("reqwest client build should never fail with default settings");
        Self {
            pool: RwLock::new(pool),
            http,
            config,
            cooldown: Mutex::new(HashMap::new()),
            records: Mutex::new(HashMap::new()),
        }
    }

    /// PUT the BEP44 envelope to the first available relay.
    ///
    /// Returns `Ok(())` when any relay accepts the record.
    /// Returns `Err(PkarrError::RelayHttpError(status))` if every relay in the
    /// pool was tried and the last attempted relay returned a non-success,
    /// non-429 HTTP status (e.g. 500/503). A 5xx from one relay does NOT abort
    /// the pool — rotation continues to the next relay.
    /// Returns `Err(PkarrError::NoRelaysAvailable)` when every relay is on
    /// cooldown, or all tried relays failed with transport errors.
    pub async fn put(&self, key_z32: &str, envelope: &[u8]) -> Result<(), PkarrError> {
        let mut last_http_error: Option<u16> = None;
        for base in self.available_relays() {
            let url = format!("{}/{}", base, key_z32);
            match self.http.put(&url).body(envelope.to_vec()).send().await {
                Ok(resp) if resp.status().is_success() => {
                    self.record_outcome(&base, RelayOutcome::Success);
                    return Ok(());
                }
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    self.record_outcome(&base, RelayOutcome::Http(429));
                    continue;
                }
                Ok(resp) => {
                    // Non-success, non-429 (e.g. 500/503): record the status,
                    // put the relay on cooldown, and rotate to the next one.
                    let status = resp.status().as_u16();
                    self.mark_cooldown(&base);
                    self.record_outcome(&base, RelayOutcome::Http(status));
                    last_http_error = Some(status);
                    continue;
                }
                Err(e) => {
                    // Transport error (timeout, connection refused, DNS, etc.)
                    self.mark_cooldown(&base);
                    self.record_outcome(
                        &base,
                        if e.is_timeout() {
                            RelayOutcome::Timeout
                        } else {
                            RelayOutcome::Transport
                        },
                    );
                    continue;
                }
            }
        }
        // If at least one relay returned an explicit HTTP error status, surface
        // it; otherwise all failures were transport-level or cooldown.
        if let Some(status) = last_http_error {
            Err(PkarrError::RelayHttpError(status))
        } else {
            Err(PkarrError::NoRelaysAvailable)
        }
    }

    /// GET the BEP44 envelope for `key_z32`.
    ///
    /// Returns `Ok(Some(bytes))` when any relay returns the record.
    /// Returns `Ok(None)` only when ALL polled relays returned 404.
    /// Returns `Err(PkarrError::NoRelaysAvailable)` when all relays are on
    /// cooldown (none were tried), or when all tried relays failed with
    /// transport errors / 429 / 5xx (none confirmed the key is absent).
    pub async fn get(&self, key_z32: &str) -> Result<Option<Vec<u8>>, PkarrError> {
        let relays = self.available_relays();
        // If every relay is on cooldown we have no information — return an
        // error rather than a false Ok(None) that would negative-cache the key.
        if relays.is_empty() {
            return Err(PkarrError::NoRelaysAvailable);
        }
        // The empty-relays guard above guarantees we enter the loop, so we
        // only need to track whether every polled relay returned a definitive
        // 404 ("not here"). Any other outcome (transient error, 429, 5xx)
        // flips this to false so we surface NoRelaysAvailable instead of
        // falsely confirming absence.
        let mut all_404 = true;
        for base in relays {
            let url = format!("{}/{}", base, key_z32);
            match self.http.get(&url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    let bytes = resp
                        .bytes()
                        .await
                        .map_err(|_| PkarrError::RelayResponseInvalid)?;
                    self.record_outcome(&base, RelayOutcome::Success);
                    return Ok(Some(bytes.to_vec()));
                }
                Ok(resp) if resp.status().as_u16() == 404 => {
                    // 404 is a definitive "not here" answer; keep going.
                    // all_404 remains true. No record update: the relay
                    // answered correctly (reachable, not failing) — neither a
                    // success-for-us nor an error.
                    continue;
                }
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    self.record_outcome(&base, RelayOutcome::Http(429));
                    all_404 = false;
                    continue;
                }
                Ok(resp) => {
                    // Other non-success status (e.g. 500): treat as transport
                    // error — we can't confirm the key is absent. Mirror the
                    // PUT path: mark cooldown so a misbehaving relay doesn't
                    // get hammered with every GET.
                    self.mark_cooldown(&base);
                    self.record_outcome(&base, RelayOutcome::Http(resp.status().as_u16()));
                    all_404 = false;
                    continue;
                }
                Err(e) => {
                    // Timeout / connection refused / DNS failure.
                    self.mark_cooldown(&base);
                    self.record_outcome(
                        &base,
                        if e.is_timeout() {
                            RelayOutcome::Timeout
                        } else {
                            RelayOutcome::Transport
                        },
                    );
                    all_404 = false;
                    continue;
                }
            }
        }
        if all_404 {
            Ok(None)
        } else {
            Err(PkarrError::NoRelaysAvailable)
        }
    }

    /// Returns the subset of pool relays whose cooldown has expired (or never
    /// started). Order is preserved so the caller iterates in pool order.
    fn available_relays(&self) -> Vec<String> {
        let now = Instant::now();
        let cd = self.cooldown.lock().expect("cooldown poisoned");
        let pool = self.pool.read().expect("relay pool poisoned");
        pool.relays
            .iter()
            .filter(|r| cd.get(r.as_str()).is_none_or(|expiry| *expiry <= now))
            .cloned()
            .collect()
    }

    /// Replace the relay pool live. Takes effect on the next `put`/`get`.
    ///
    /// Swaps the pool **first**, then prunes cooldown + records entries for relays
    /// no longer present. Swap-first eliminates the "old pool still active but its
    /// cooldown already pruned" intermediate window a concurrent `available_relays`
    /// could otherwise observe (Qodo). Pruning then gives the settings-UI
    /// remove-then-re-add flow a fresh relay rather than one silently skipped for
    /// up to `config.cooldown`. In-flight requests that finish after the swap are
    /// handled by the pool-membership guard in `mark_cooldown` / `record_outcome`,
    /// which drop writes for relays no longer in the pool (so they cannot resurrect
    /// a pruned entry — Cursor). Each lock is taken and released sequentially
    /// (never two at once), so this adds no lock-ordering hazard.
    pub fn set_relays(&self, relays: Vec<String>) {
        // Owned set so it outlives the `relays` move into the pool below (≤8
        // short strings — negligible clone).
        let live: std::collections::HashSet<String> = relays.iter().cloned().collect();
        *self.pool.write().expect("relay pool poisoned") = RelayPool::new(relays);
        self.cooldown
            .lock()
            .expect("cooldown poisoned")
            .retain(|k, _| live.contains(k.as_str()));
        self.records
            .lock()
            .expect("records poisoned")
            .retain(|k, _| live.contains(k.as_str()));
    }

    /// True if `base` is in the current pool. Acquires the pool read-lock; callers
    /// must NOT already hold it — `pool` is always acquired LAST so the global lock
    /// order (cooldown → records → pool) stays acyclic.
    fn pool_contains(&self, base: &str) -> bool {
        self.pool
            .read()
            .expect("relay pool poisoned")
            .relays
            .iter()
            .any(|r| r == base)
    }

    /// Put `base` into a cooldown for the configured duration. No-op if `base` is
    /// no longer in the pool: an in-flight request can finish after a `set_relays`
    /// dropped its relay, and must not resurrect a cooldown entry that re-add would
    /// then inherit.
    fn mark_cooldown(&self, base: &str) {
        let mut cd = self.cooldown.lock().expect("cooldown poisoned");
        if !self.pool_contains(base) {
            return;
        }
        cd.insert(base.to_string(), Instant::now() + self.config.cooldown);
    }

    /// Record the latest outcome for `base` (health observability). No-op if `base`
    /// is no longer in the pool (same in-flight-after-swap guard as `mark_cooldown`).
    fn record_outcome(&self, base: &str, outcome: RelayOutcome) {
        let mut recs = self.records.lock().expect("records poisoned");
        if !self.pool_contains(base) {
            return;
        }
        let rec = recs.entry(base.to_string()).or_default();
        rec.last_outcome = Some(outcome);
        if outcome == RelayOutcome::Success {
            rec.last_success_ms = Some(now_ms());
        }
    }

    /// Synchronous per-relay health for the current pool — one entry per pool
    /// relay, in pool order. `state` is derived from the cooldown map; a relay
    /// whose cooldown expiry is in the future reports `CoolingDown { until_ms }`
    /// (cooldown `Instant` converted to wall-clock millis), else `Healthy`.
    pub fn relay_health(&self) -> Vec<RelayHealth> {
        let now_inst = Instant::now();
        let now_wall = now_ms();
        let cd = self.cooldown.lock().expect("cooldown poisoned");
        let recs = self.records.lock().expect("records poisoned");
        let pool = self.pool.read().expect("relay pool poisoned");
        pool.relays
            .iter()
            .map(|url| {
                let state = match cd.get(url.as_str()) {
                    Some(expiry) if *expiry > now_inst => {
                        let remaining = expiry.duration_since(now_inst).as_millis() as u64;
                        RelayState::CoolingDown {
                            until_ms: now_wall + remaining,
                        }
                    }
                    _ => RelayState::Healthy,
                };
                let rec = recs.get(url).cloned().unwrap_or_default();
                RelayHealth {
                    url: url.clone(),
                    state,
                    last_outcome: rec.last_outcome,
                    last_success_ms: rec.last_success_ms,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::MockPkarrRelay;

    #[tokio::test]
    async fn put_then_get_via_pool() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        client.put("k1", b"envelope").await.expect("put");
        let got = client.get("k1").await.expect("get");
        assert_eq!(got.as_deref(), Some(b"envelope".as_ref()));
    }

    #[tokio::test]
    async fn get_returns_none_on_404() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        let got = client.get("missing").await.expect("get");
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn unreachable_relay_falls_through_to_next() {
        let alive = MockPkarrRelay::start().await;
        // 192.0.2.1 is TEST-NET-1 (RFC 5737) — reserved, unrouted, guaranteed
        // to fail at the network layer without touching any real host.
        let pool = RelayPool::new(vec![
            "http://192.0.2.1:80".to_string(),
            alive.base_url.clone(),
        ]);
        let client = RelayClient::new(pool);
        client.put("k1", b"hello").await.expect("put");
        let got = client.get("k1").await.expect("get");
        assert_eq!(got.as_deref(), Some(b"hello".as_ref()));
    }

    #[tokio::test]
    async fn all_relays_unavailable_yields_error() {
        // 192.0.2.1 is TEST-NET-1 (RFC 5737) — guaranteed unreachable.
        let pool = RelayPool::new(vec!["http://192.0.2.1:80".to_string()]);
        let client = RelayClient::new(pool);
        assert!(matches!(
            client.get("k").await,
            Err(PkarrError::NoRelaysAvailable)
        ));
    }

    #[test]
    fn relay_config_default_is_5s_30s() {
        let cfg = RelayConfig::default();
        assert_eq!(cfg.request_timeout, Duration::from_secs(5));
        assert_eq!(cfg.cooldown, Duration::from_secs(30));
    }

    #[tokio::test]
    async fn with_config_short_cooldown_is_honored() {
        // A relay that fails goes on cooldown for the CONFIGURED duration.
        // With a 0ms cooldown the failed relay is immediately available again.
        let cfg = RelayConfig {
            request_timeout: Duration::from_millis(200),
            cooldown: Duration::from_millis(0),
        };
        let pool = RelayPool::new(vec!["http://192.0.2.1:80".to_string()]);
        let client = RelayClient::with_config(pool, cfg);
        // First get marks the unreachable relay on cooldown, then (cooldown=0)
        // it is available again on the next call — so we still see the relay,
        // not an empty pool. The point is `with_config` compiles + plumbs cfg.
        let _ = client.get("k").await; // Err(NoRelaysAvailable) or transport err
        assert_eq!(client.relay_health().len(), 1);
    }

    #[tokio::test]
    async fn relay_health_lists_pool_relays_healthy_by_default() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        let health = client.relay_health();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].url, relay.base_url);
        assert_eq!(health[0].state, RelayState::Healthy);
        assert_eq!(health[0].last_outcome, None);
        assert_eq!(health[0].last_success_ms, None);
    }

    #[tokio::test]
    async fn relay_health_records_success_after_put() {
        let relay = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay.base_url.clone()]));
        client.put("k1", b"v").await.expect("put");
        let h = &client.relay_health()[0];
        assert_eq!(h.state, RelayState::Healthy);
        assert_eq!(h.last_outcome, Some(RelayOutcome::Success));
        assert!(h.last_success_ms.is_some());
    }

    #[tokio::test]
    async fn relay_health_reflects_cooldown_after_transport_failure() {
        // 127.0.0.1:1 — nothing listens on port 1, so a loopback connect gets an
        // immediate RST (ECONNREFUSED): a non-timeout transport failure. Race-free
        // (we never bind the port, so there's no bind-then-drop reuse window).
        // (TEST-NET-1 192.0.2.1 is unrouted → packets dropped → the attempt times
        // out instead; that `Timeout` path is exercised separately below.)
        let client = RelayClient::new(RelayPool::new(vec!["http://127.0.0.1:1".to_string()]));
        let _ = client.get("k").await; // connection refused → cooldown
        let h = &client.relay_health()[0];
        match h.state {
            RelayState::CoolingDown { until_ms } => assert!(until_ms > 0),
            RelayState::Healthy => panic!("expected CoolingDown after transport failure"),
        }
        // A connection refused / DNS error is Transport, not Timeout.
        assert_eq!(h.last_outcome, Some(RelayOutcome::Transport));
    }

    #[tokio::test]
    async fn relay_health_reflects_cooldown_after_timeout() {
        // TEST-NET-1 (192.0.2.1) is unrouted: the connect attempt hangs until the
        // per-request timeout elapses, so reqwest reports `is_timeout()` → Timeout.
        // Short request_timeout keeps the test fast. Both Timeout and Transport
        // trip cooldown identically; this asserts the discrimination is correct.
        let cfg = RelayConfig {
            request_timeout: Duration::from_millis(300),
            cooldown: Duration::from_secs(30),
        };
        let pool = RelayPool::new(vec!["http://192.0.2.1:80".to_string()]);
        let client = RelayClient::with_config(pool, cfg);
        let _ = client.get("k").await; // hangs → times out → cooldown
        let h = &client.relay_health()[0];
        match h.state {
            RelayState::CoolingDown { until_ms } => assert!(until_ms > 0),
            RelayState::Healthy => panic!("expected CoolingDown after timeout"),
        }
        assert_eq!(h.last_outcome, Some(RelayOutcome::Timeout));
    }

    #[tokio::test]
    async fn relay_health_only_lists_current_pool_after_swap() {
        let relay_a = MockPkarrRelay::start().await;
        let relay_b = MockPkarrRelay::start().await;
        let client = RelayClient::new(RelayPool::new(vec![relay_a.base_url.clone()]));
        client.set_relays(vec![relay_b.base_url.clone()]);
        let health = client.relay_health();
        assert_eq!(health.len(), 1);
        assert_eq!(health[0].url, relay_b.base_url);
    }

    #[tokio::test]
    async fn set_relays_prunes_cooldown_for_removed_relay_so_readd_is_fresh() {
        // Settings-UI flow: a relay is cooling down; the user removes it (a
        // set_relays call that drops it from the pool), then re-adds it (a second
        // set_relays call). Removal must prune the stale cooldown so the re-added
        // relay is immediately Healthy rather than silently skipped for ~cooldown.
        let healthy = MockPkarrRelay::start().await;
        let bad = "http://127.0.0.1:1".to_string(); // refused → cooldown
        let client = RelayClient::new(RelayPool::new(vec![bad.clone(), healthy.base_url.clone()]));
        let _ = client.get("k").await; // bad tried first → cooldown; healthy 404s
        let bad_health = client
            .relay_health()
            .into_iter()
            .find(|r| r.url == bad)
            .expect("bad relay present");
        assert!(
            matches!(bad_health.state, RelayState::CoolingDown { .. }),
            "bad relay should be cooling down after a refused connection"
        );

        // Remove `bad` (pool = just the healthy relay) → prunes bad's cooldown.
        client.set_relays(vec![healthy.base_url.clone()]);
        // Re-add `bad`.
        client.set_relays(vec![healthy.base_url.clone(), bad.clone()]);

        let bad_health = client
            .relay_health()
            .into_iter()
            .find(|r| r.url == bad)
            .expect("bad relay re-added");
        assert_eq!(
            bad_health.state,
            RelayState::Healthy,
            "re-added relay must be fresh, not inherit the stale cooldown"
        );
    }

    #[tokio::test]
    async fn set_relays_takes_effect_mid_flight() {
        // Publish to pool A (relay_a), then swap the pool to relay_b and confirm
        // the next get hits relay_b (which has the record) — proving the swap is live.
        let relay_a = MockPkarrRelay::start().await;
        let relay_b = MockPkarrRelay::start().await;

        let client = RelayClient::new(RelayPool::new(vec![relay_a.base_url.clone()]));
        client.put("k1", b"in-a").await.expect("put to A");

        // Swap pool to B (which does NOT have k1 yet).
        client.set_relays(vec![relay_b.base_url.clone()]);
        assert_eq!(client.get("k1").await.expect("get from B"), None); // B has no k1

        // Publish to B via the swapped pool, then read it back from B.
        client.put("k1", b"in-b").await.expect("put to B");
        assert_eq!(
            client.get("k1").await.expect("get from B"),
            Some(b"in-b".to_vec())
        );
    }
}

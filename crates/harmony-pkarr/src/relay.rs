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

use crate::error::PkarrError;

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
                Ok(resp) if resp.status().is_success() => return Ok(()),
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    continue;
                }
                Ok(resp) => {
                    // Non-success, non-429 (e.g. 500/503): record the status,
                    // put the relay on cooldown, and rotate to the next one.
                    let status = resp.status().as_u16();
                    self.mark_cooldown(&base);
                    last_http_error = Some(status);
                    continue;
                }
                Err(_) => {
                    // Transport error (timeout, connection refused, DNS, etc.)
                    self.mark_cooldown(&base);
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
                    return Ok(Some(bytes.to_vec()));
                }
                Ok(resp) if resp.status().as_u16() == 404 => {
                    // 404 is a definitive "not here" answer; keep going.
                    // all_404 remains true.
                    continue;
                }
                Ok(resp) if resp.status().as_u16() == 429 => {
                    self.mark_cooldown(&base);
                    all_404 = false;
                    continue;
                }
                Ok(_) => {
                    // Other non-success status (e.g. 500): treat as transport
                    // error — we can't confirm the key is absent. Mirror the
                    // PUT path: mark cooldown so a misbehaving relay doesn't
                    // get hammered with every GET.
                    self.mark_cooldown(&base);
                    all_404 = false;
                    continue;
                }
                Err(_) => {
                    // Timeout / connection refused / DNS failure.
                    self.mark_cooldown(&base);
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
    /// Cooldown entries keyed on a now-removed relay simply never match a live
    /// relay again and age out — harmless, no explicit pruning needed.
    pub fn set_relays(&self, relays: Vec<String>) {
        *self.pool.write().expect("relay pool poisoned") = RelayPool::new(relays);
    }

    /// Put `base` into a cooldown for the configured duration.
    fn mark_cooldown(&self, base: &str) {
        let mut cd = self.cooldown.lock().expect("cooldown poisoned");
        cd.insert(base.to_string(), Instant::now() + self.config.cooldown);
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

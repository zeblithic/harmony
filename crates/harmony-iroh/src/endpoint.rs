//! Persistent [`iroh::Endpoint`] wrapper.
//!
//! Ported from harmony-client's `iroh_endpoint.rs` (ZEB-321 Phase 1 Task 4),
//! with two app-coupling seams cut for reuse (ZEB-739):
//!
//! - **Seam 1a — injected identity.** The client loaded its Ed25519
//!   [`SecretKey`] from the OS keychain (or an encrypted-file fallback) inside
//!   the constructor. That policy is app-specific, so this crate takes the key
//!   as a constructor argument ([`IrohEndpoint::new_with_secret`]) and never
//!   touches a keychain.
//! - **Seam 1b — caller-supplied ALPNs.** The client hardcoded its
//!   `harmony/*` ALPN wire strings on the builder. This crate advertises
//!   exactly the ALPNs the caller passes via [`AlpnConfig`] — the endpoint
//!   knows no wire strings.
//!
//! Everything else (relay-map tracking + live reconciliation, address
//! snapshots, the `watch_addr` stream, graceful shutdown) is carried over
//! unchanged.
//!
//! ## iroh 1.0 API notes (carried from the source)
//!
//! - `iroh::NodeId` is `iroh::EndpointId` (an alias for `iroh::PublicKey`).
//! - `Endpoint::builder` takes a `Preset`; production uses
//!   `iroh::endpoint::presets::N0`, whose default relay map is n0's stable
//!   production cluster.
//! - The local-id accessor is `.id()`; the address snapshot is `.addr()`;
//!   graceful shutdown is `.close()`.

use std::collections::BTreeSet;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

use iroh::endpoint::{presets, Endpoint, RelayMode};
use iroh::{EndpointId, RelayUrl, SecretKey};

use crate::error::IrohEndpointError;

/// ALPNs an [`IrohEndpoint`] advertises and accepts on inbound connections.
///
/// Seam 1b: the endpoint is app-agnostic and holds no wire strings — the
/// caller (harmony-node, the tunnel bridge, …) supplies the exact ALPN
/// byte-strings its protocols use.
#[derive(Debug, Clone, Default)]
pub struct AlpnConfig {
    /// The ALPNs, in advertisement order. Passed verbatim to iroh's
    /// `Builder::alpns`.
    pub advertised: Vec<Vec<u8>>,
}

impl AlpnConfig {
    /// Convenience constructor from an ALPN list.
    pub fn new(advertised: Vec<Vec<u8>>) -> Self {
        Self { advertised }
    }
}

/// The endpoint's relay-map policy at build time.
///
/// The client's constructor accepted `custom_relays: Option<Vec<RelayUrl>>`
/// where `None`/empty followed the `presets::N0` default relay map and a
/// non-empty list pinned exactly those relays; its hermetic tests reached
/// `RelayMode::Disabled` through the raw builder. This enum makes all three
/// modes first-class so the crate is testable in isolation without a raw
/// builder escape hatch. (Deviation from the brief's single-field struct —
/// a single `Vec` can't distinguish "follow the n0 default map" from
/// "relays disabled"; see the ZEB-739 report.)
#[derive(Debug, Clone, Default)]
pub enum RelayConfig {
    /// Follow the `presets::N0` default relay map (n0's stable production
    /// cluster). The tracked configured set is seeded from that map.
    #[default]
    N0Default,
    /// Pin exactly these relay URLs via `RelayMode::custom`. An empty list is
    /// treated as [`RelayConfig::N0Default`] (matches the client's
    /// empty-`custom_relays` behavior).
    Custom(Vec<RelayUrl>),
    /// Relays disabled (`RelayMode::Disabled`) — hermetic / offline use. The
    /// tracked configured set is empty.
    Disabled,
}

/// Wrapper around [`iroh::Endpoint`] exposing the transport-substrate surface:
/// identity + address snapshots, relay-map tracking/reconciliation, an address
/// watch stream, and graceful shutdown. Keeping the surface small lets us swap
/// iroh versions without churning every call site — callers that genuinely
/// need the raw endpoint use [`IrohEndpoint::inner`].
#[derive(Clone, Debug)]
pub struct IrohEndpoint {
    inner: Endpoint,
    /// ZEB-624: authoritative, in-process view of the endpoint's CONFIGURED
    /// relay URLs. iroh 1.0's `Endpoint` exposes relay-map *mutators*
    /// ([`Endpoint::insert_relay`]/[`Endpoint::remove_relay`]) but no reader of
    /// the full configured map (only `home_relay_status()`, the
    /// negotiated/connected subset), so we track the configured set here: seeded
    /// at build and kept in lock-step by [`Self::apply_relay_urls`] — the ONLY
    /// path that mutates the endpoint's relay map. Shared via `Arc` so every
    /// `Clone` of this wrapper observes the same live set.
    relay_urls: Arc<Mutex<BTreeSet<RelayUrl>>>,
}

impl IrohEndpoint {
    /// Build and bind an endpoint using `secret` as the persistent identity,
    /// advertising `alpn.advertised`, with the relay policy from `relays`.
    ///
    /// Production path: uses `presets::N0` and iroh's default system DNS
    /// resolver. The CONFIGURED relay set is recorded so [`Self::relay_map_urls`]
    /// can report it and [`Self::apply_relay_urls`] can diff against it.
    pub async fn new_with_secret(
        secret: SecretKey,
        relays: RelayConfig,
        alpn: AlpnConfig,
    ) -> Result<Self, IrohEndpointError> {
        Self::new_with_secret_inner(secret, relays, alpn, None).await
    }

    async fn new_with_secret_inner(
        secret: SecretKey,
        relays: RelayConfig,
        alpn: AlpnConfig,
        dns_resolver: Option<iroh::dns::DnsResolver>,
    ) -> Result<Self, IrohEndpointError> {
        let builder = Endpoint::builder(presets::N0)
            .secret_key(secret)
            .alpns(alpn.advertised);
        // Seed the tracked configured-relay set from the SAME source the builder
        // binds with.
        let configured = configured_relay_set(&relays);
        let builder = match relay_mode_for(&relays) {
            Some(mode) => builder.relay_mode(mode),
            None => builder,
        };
        let builder = match dns_resolver {
            Some(resolver) => builder.dns_resolver(resolver),
            None => builder,
        };
        let inner = builder
            .bind()
            .await
            .map_err(|e| IrohEndpointError::Bind(Box::new(e)))?;
        Ok(Self::from_parts(inner, configured))
    }

    /// Wrap an already-bound iroh [`Endpoint`] with the given CONFIGURED relay
    /// set. The single struct-literal constructor so the `relay_urls` tracking
    /// invariant lives in one place.
    fn from_parts(inner: Endpoint, relay_urls: BTreeSet<RelayUrl>) -> Self {
        Self {
            inner,
            relay_urls: Arc::new(Mutex::new(relay_urls)),
        }
    }

    /// ZEB-624: the endpoint's CONFIGURED relay URLs as normalized strings,
    /// sorted. Reads the tracked set — the authoritative view of what the
    /// endpoint's relay map holds, since iroh 1.0's `Endpoint` has no reader for
    /// the full configured map. The trailing slash `RelayUrl`'s `Display` adds
    /// (a relay is a host-only base) is stripped; the strings still round-trip
    /// through `RelayUrl::from_str`.
    pub fn relay_map_urls(&self) -> Vec<String> {
        let mut urls: Vec<String> = {
            let guard = self.relay_urls.lock().unwrap_or_else(|p| p.into_inner());
            guard
                .iter()
                .map(|u| u.to_string().trim_end_matches('/').to_string())
                .collect()
        };
        urls.sort();
        urls
    }

    /// ZEB-624: reconcile the endpoint's relay map to exactly `target` — insert
    /// each target relay not already configured, remove each configured relay
    /// not in `target` — updating the tracked set in lock-step. Returns
    /// `(inserted, removed)` counts; `(0, 0)` when already reconciled
    /// (idempotent). `insert_relay`/`remove_relay` no-op on a closed endpoint,
    /// which the counts still reflect so a caller's log matches the intended
    /// diff.
    pub async fn apply_relay_urls(&self, target: &[RelayUrl]) -> (usize, usize) {
        let current: BTreeSet<RelayUrl> = {
            let guard = self.relay_urls.lock().unwrap_or_else(|p| p.into_inner());
            guard.clone()
        };
        let target_set: BTreeSet<RelayUrl> = target.iter().cloned().collect();
        let mut inserted = 0usize;
        let mut removed = 0usize;
        for url in &target_set {
            if !current.contains(url) {
                self.inner
                    .insert_relay(url.clone(), Arc::new(iroh::RelayConfig::from(url.clone())))
                    .await;
                inserted += 1;
            }
        }
        for url in &current {
            if !target_set.contains(url) {
                self.inner.remove_relay(url).await;
                removed += 1;
            }
        }
        if inserted > 0 || removed > 0 {
            let mut guard = self.relay_urls.lock().unwrap_or_else(|p| p.into_inner());
            *guard = target_set;
        }
        (inserted, removed)
    }

    /// This endpoint's stable identity, derived from the secret key.
    /// `EndpointId` is a type alias for `iroh::PublicKey`.
    pub fn node_id(&self) -> EndpointId {
        self.inner.id()
    }

    /// Snapshot of the current home relay url, if any has been negotiated.
    /// Returns `None` before the relay round-trip completes or when relays are
    /// disabled.
    pub fn home_relay(&self) -> Option<RelayUrl> {
        self.inner.addr().relay_urls().next().cloned()
    }

    /// Snapshot of the direct addresses other peers can dial us at. May be
    /// empty immediately after bind — typically populated once the
    /// address-lookup service has probed interfaces.
    pub fn direct_addresses(&self) -> Vec<SocketAddr> {
        self.inner.addr().ip_addrs().copied().collect()
    }

    /// Local socket addresses the underlying iroh sockets are bound to.
    ///
    /// Unlike [`Self::direct_addresses`] (which routes through the `addr()`
    /// snapshot that depends on the address-lookup service), this returns the
    /// actual `bind()`-result sockets and is populated immediately on bind —
    /// including for hermetic endpoints built without the address-lookup
    /// service.
    pub fn bound_sockets(&self) -> Vec<SocketAddr> {
        self.inner.bound_sockets()
    }

    /// A `'static` stream of [`iroh::EndpointAddr`] updates sourced from iroh's
    /// own `watch_addr` watcher, boxed so a reachability publisher can merge it
    /// into a network-change arm (ZEB-621).
    ///
    /// Uses `stream_updates_only`, which **skips the watcher's current
    /// value** — subscribing at boot does not itself emit an item. The stream
    /// ends when the last [`iroh::Endpoint`] clone drops.
    pub fn watch_addr_stream(&self) -> futures::stream::BoxStream<'static, iroh::EndpointAddr> {
        use futures::StreamExt as _;
        use iroh::Watcher as _;
        self.inner.watch_addr().stream_updates_only().boxed()
    }

    /// Nudge iroh to re-probe the local network (interfaces + relays). Thin
    /// passthrough to [`iroh::Endpoint::network_change`].
    pub async fn network_change(&self) {
        self.inner.network_change().await;
    }

    /// Escape hatch: the raw [`iroh::Endpoint`] for callers that need the full
    /// iroh API — outbound `.connect()`, inbound `.accept()` (see
    /// [`crate::dispatch::spawn_accept`]), or any surface not re-exposed here.
    ///
    /// Cross-crate `pub` by contract (ZEB-739 Risk #3): the link manager, the
    /// tunnel driver, and the accept helper all live in downstream crates and
    /// need the raw endpoint. Prefer adding a method to [`IrohEndpoint`] over
    /// reaching through this when the need is general.
    pub fn inner(&self) -> &Endpoint {
        &self.inner
    }

    /// Gracefully close the endpoint and all open connections. Safe to call
    /// multiple times — `iroh::Endpoint::close` is idempotent.
    pub async fn shutdown(&self) {
        self.inner.close().await;
    }
}

/// The tracked configured-relay set for a [`RelayConfig`] — seeded from the
/// SAME source the builder binds with (the custom list, else the `presets::N0`
/// default relay map, else empty when disabled). Shared by the production
/// constructor and the hermetic test constructor so their tracking can't drift.
fn configured_relay_set(relays: &RelayConfig) -> BTreeSet<RelayUrl> {
    match relays {
        RelayConfig::Custom(urls) if !urls.is_empty() => urls.iter().cloned().collect(),
        RelayConfig::Custom(_) | RelayConfig::N0Default => iroh::endpoint::default_relay_mode()
            .relay_map()
            .urls::<Vec<RelayUrl>>()
            .into_iter()
            .collect(),
        RelayConfig::Disabled => BTreeSet::new(),
    }
}

/// The [`RelayMode`] override a [`RelayConfig`] applies to the builder, or
/// `None` to follow the preset default (n0 stable). Shared by the production
/// and hermetic constructors.
fn relay_mode_for(relays: &RelayConfig) -> Option<RelayMode> {
    match relays {
        RelayConfig::Custom(urls) if !urls.is_empty() => Some(RelayMode::custom(urls.clone())),
        RelayConfig::Custom(_) | RelayConfig::N0Default => None,
        RelayConfig::Disabled => Some(RelayMode::Disabled),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A DNS resolver for hermetic test endpoints that never reads the system
    /// DNS configuration. iroh's `Builder::bind` eagerly constructs the system
    /// resolver when none is supplied, and hickory's macOS `read_system_conf`
    /// blocks in `SCDynamicStoreCreateWithOptions` (~22s/process for unentitled
    /// callers — every test binary). Hermetic tests never resolve a name, so
    /// the nameserver below (loopback port 1) is intentionally unanswering.
    fn hermetic_dns_resolver() -> iroh::dns::DnsResolver {
        iroh::dns::DnsResolver::with_nameserver(SocketAddr::from(([127, 0, 0, 1], 1)))
    }

    /// Hermetic constructor for the endpoint tests: same relay/ALPN mapping as
    /// [`IrohEndpoint::new_with_secret`] but with a non-resolving DNS resolver
    /// injected so bind never pays iroh's eager system-DNS read.
    async fn new_hermetic(
        secret: SecretKey,
        relays: RelayConfig,
        alpn: AlpnConfig,
    ) -> Result<IrohEndpoint, IrohEndpointError> {
        IrohEndpoint::new_with_secret_inner(secret, relays, alpn, Some(hermetic_dns_resolver()))
            .await
    }

    /// ALPNs for tests — the endpoint is agnostic to the exact strings.
    fn test_alpns() -> AlpnConfig {
        AlpnConfig::new(vec![b"harmony-iroh/test/v1".to_vec()])
    }

    /// Parse `relay_map_urls()` output back into a `RelayUrl` set — the
    /// round-trip comparison the ZEB-624 endpoint tests use so they don't depend
    /// on iroh's exact URL string canonicalization (trailing slash / FQDN dot).
    fn relay_url_set(ep: &IrohEndpoint) -> BTreeSet<RelayUrl> {
        ep.relay_map_urls()
            .iter()
            .map(|s| s.parse::<RelayUrl>().expect("relay_map_urls round-trips"))
            .collect()
    }

    /// Lifecycle smoke test against an ephemeral secret with relays disabled —
    /// keeps the test hermetic. Identity must round-trip and the snapshot
    /// accessors must not panic.
    #[tokio::test]
    async fn iroh_endpoint_inits_with_ephemeral_secret() {
        let secret = SecretKey::generate();
        let expected_id = secret.public();

        let ep = new_hermetic(secret, RelayConfig::Disabled, test_alpns())
            .await
            .expect("bind ephemeral endpoint");

        // Identity round-trips through the secret key we generated.
        assert_eq!(ep.node_id(), expected_id);

        // Snapshots must not panic. With relays disabled `home_relay` is `None`;
        // direct addresses may or may not be populated yet (accept either).
        assert!(ep.home_relay().is_none());
        let _direct: Vec<SocketAddr> = ep.direct_addresses();

        ep.shutdown().await;
    }

    /// ZEB-617/619 regression guard: the relay map the production builder gets
    /// (via `default_relay_mode()`, which `presets::N0` applies) must be the
    /// stable production cluster, never n0's decommissioned canary relays.
    #[test]
    fn default_relay_map_is_stable_non_canary() {
        let map = iroh::endpoint::default_relay_mode().relay_map();
        let urls: Vec<String> = map.urls::<Vec<_>>().iter().map(|u| u.to_string()).collect();
        assert!(!urls.is_empty(), "default relay map must not be empty");
        for url in &urls {
            assert!(
                !url.contains("canary"),
                "canary relay leaked into defaults: {url}"
            );
            assert!(
                url.contains(".relay.n0.iroh.link"),
                "unexpected relay host: {url}"
            );
        }
    }

    /// ZEB-624: a custom relay list supplied at build overrides the n0 preset
    /// default map — the configured relay map is EXACTLY the custom list.
    /// Asserts via `RelayUrl` round-trip equality (not a raw string literal) so
    /// the test is agnostic to iroh's URL canonicalization.
    #[tokio::test]
    async fn custom_relay_list_overrides_default_map() {
        let secret = SecretKey::generate();
        let custom: RelayUrl = "https://relay.example.com"
            .parse()
            .expect("parse custom relay url");
        let ep = new_hermetic(
            secret,
            RelayConfig::Custom(vec![custom.clone()]),
            test_alpns(),
        )
        .await
        .expect("bind endpoint with custom relay");
        assert_eq!(relay_url_set(&ep), BTreeSet::from([custom.clone()]));
        ep.shutdown().await;
    }

    /// ZEB-624: `apply_relay_urls` diffs the target against the configured set —
    /// one insert + one remove when swapping [A] → [B], then a no-op when the
    /// same target is re-applied (idempotent). No relay traffic is generated by
    /// merely holding a relay map, so this stays hermetic.
    #[tokio::test]
    async fn apply_relay_urls_diffs_insert_and_remove() {
        let secret = SecretKey::generate();
        let a: RelayUrl = "https://relay-a.example.com".parse().expect("parse A");
        let b: RelayUrl = "https://relay-b.example.com".parse().expect("parse B");
        let ep = new_hermetic(secret, RelayConfig::Custom(vec![a.clone()]), test_alpns())
            .await
            .expect("bind endpoint with [A]");
        assert_eq!(relay_url_set(&ep), BTreeSet::from([a.clone()]));

        // Swap to [B]: B inserted, A removed.
        let (inserted, removed) = ep.apply_relay_urls(std::slice::from_ref(&b)).await;
        assert_eq!((inserted, removed), (1, 1));
        assert_eq!(relay_url_set(&ep), BTreeSet::from([b.clone()]));

        // Re-applying the same target is a no-op.
        let (inserted2, removed2) = ep.apply_relay_urls(std::slice::from_ref(&b)).await;
        assert_eq!((inserted2, removed2), (0, 0));
        assert_eq!(relay_url_set(&ep), BTreeSet::from([b.clone()]));
        ep.shutdown().await;
    }
}

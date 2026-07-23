//! Shared **hermetic** test helpers for the manager + driver unit tests.
//!
//! Core has no vendored-netdev CoreWLAN patch, so a production `presets::N0`
//! bind can stall ~44s on macOS (interface probing) and iroh's eager system-DNS
//! read can stall ~22s. Every endpoint built here therefore uses
//! `presets::Minimal` + `clear_ip_transports()` + `RelayMode::Disabled` + a
//! non-resolving DNS resolver, and dials by explicit loopback socket address —
//! exactly the technique the client's tunnel tests use. No test binds a real
//! n0/production endpoint or touches the network.

#![cfg(test)]

use std::sync::Arc;

use harmony_identity::PqPrivateIdentity;
use harmony_iroh::endpoint::IrohEndpoint;
use tokio::sync::mpsc;

use crate::manager::{CompatSink, HandshakeOutcome, InboundDm, TunnelManager};
use crate::peer::TunnelPeer;

/// A DNS resolver that never reads the system DNS configuration (points at an
/// intentionally-unanswering loopback nameserver). Hermetic tests dial by
/// address and never resolve a name.
pub fn hermetic_dns_resolver() -> iroh::dns::DnsResolver {
    iroh::dns::DnsResolver::with_nameserver(std::net::SocketAddr::from(([127, 0, 0, 1], 1)))
}

/// The tunnel ALPN bind list (both generations), mirroring production.
pub fn tunnel_alpns() -> Vec<Vec<u8>> {
    vec![
        crate::versioning::alpn::HARMONY_TUNNEL_V1.to_vec(),
        crate::versioning::alpn::HARMONY_TUNNEL_V2.to_vec(),
    ]
}

/// Build a loopback-only iroh endpoint (fresh random identity) advertising
/// `alpns`. No relays, no address lookup, no DNS — fully hermetic.
pub async fn loopback_endpoint(alpns: Vec<Vec<u8>>) -> iroh::endpoint::Endpoint {
    use iroh::endpoint::{presets, Endpoint, RelayMode};
    use std::net::Ipv4Addr;
    Endpoint::builder(presets::Minimal)
        .secret_key(iroh::SecretKey::generate())
        .alpns(alpns)
        .relay_mode(RelayMode::Disabled)
        .dns_resolver(hermetic_dns_resolver())
        .clear_ip_transports()
        .bind_addr((Ipv4Addr::LOCALHOST, 0))
        .expect("bind_addr")
        .bind()
        .await
        .expect("bind loopback endpoint")
}

/// A loopback endpoint binding ONLY the `/v1` tunnel ALPN — a
/// one-generation-behind peer. A v2 dialer's `/v2` connect fails ALPN
/// negotiation against it and must fall back to `/v1`.
pub async fn loopback_endpoint_v1_only() -> iroh::endpoint::Endpoint {
    loopback_endpoint(vec![crate::versioning::alpn::HARMONY_TUNNEL_V1.to_vec()]).await
}

/// Pay iroh's per-process first-bind cost (crypto-provider init, netmon setup)
/// ONCE at the top of a heavy endpoint-binding test, OUTSIDE the timeout that
/// guards the behavior under test. A 120s kill-switch still bounds a true bind
/// hang (a hermetic bind is ~0.06s post-warmup).
pub async fn warm_up_iroh_global_init() {
    use iroh::endpoint::{presets, Endpoint, RelayMode};
    let builder = Endpoint::builder(presets::Minimal)
        .relay_mode(RelayMode::Disabled)
        .dns_resolver(hermetic_dns_resolver())
        .clear_ip_transports()
        .bind_addr((std::net::Ipv4Addr::LOCALHOST, 0))
        .expect("warm-up bind_addr loopback");
    let ep = tokio::time::timeout(std::time::Duration::from_secs(120), builder.bind())
        .await
        .expect("warm-up iroh bind exceeded its 120s kill-switch")
        .expect("warm-up iroh bind");
    ep.close().await;
}

/// Wrap a raw hermetic endpoint into the `Arc<IrohEndpoint>` the manager/driver
/// APIs expect (via harmony-iroh's `#[doc(hidden)]` test seam).
pub fn wrap(ep: iroh::endpoint::Endpoint) -> Arc<IrohEndpoint> {
    Arc::new(IrohEndpoint::from_endpoint_for_test(ep))
}

/// A no-op [`CompatSink`] for tests that don't assert on protocol-compat.
pub struct NoopCompatSink;
impl CompatSink for NoopCompatSink {
    fn record_handshake_outcome(&self, _peer: [u8; 32], _outcome: HandshakeOutcome) {}
}

/// A [`CompatSink`] that records the incompatibility reason per peer (mirroring
/// the client's `ProtocolCompatRegistry`), so Seam-B tests can assert on it.
#[derive(Default)]
pub struct RecordingCompatSink {
    inner: std::sync::Mutex<std::collections::HashMap<[u8; 32], String>>,
}

impl RecordingCompatSink {
    /// Pre-seed an incompatibility record directly (simulating a peer flagged on
    /// an earlier connect).
    pub fn note_incompatible_direct(&self, peer: [u8; 32], reason: String) {
        self.inner
            .lock()
            .expect("compat sink poisoned")
            .insert(peer, reason);
    }

    /// The recorded incompatibility reason for `peer`, if any.
    pub fn incompat_reason(&self, peer: &[u8; 32]) -> Option<String> {
        self.inner
            .lock()
            .expect("compat sink poisoned")
            .get(peer)
            .cloned()
    }
}

impl CompatSink for RecordingCompatSink {
    fn record_handshake_outcome(&self, peer: [u8; 32], outcome: HandshakeOutcome) {
        let mut g = self.inner.lock().expect("compat sink poisoned");
        match outcome {
            HandshakeOutcome::Incompatible { reason } => {
                g.insert(peer, reason);
            }
            HandshakeOutcome::Compatible => {
                g.remove(&peer);
            }
        }
    }
}

/// A manager over a fresh hermetic loopback endpoint + no-op compat sink. Never
/// dials on its own (its spawned dials, when triggered, go to unreachable
/// contacts and are dropped when the test ends).
pub async fn test_manager() -> (Arc<TunnelManager>, mpsc::Receiver<InboundDm>) {
    let endpoint = wrap(loopback_endpoint(tunnel_alpns()).await);
    let local_pq = Arc::new(PqPrivateIdentity::generate(&mut rand::rngs::OsRng));
    let (ingest_tx, ingest_rx) = mpsc::channel(16);
    (
        Arc::new(TunnelManager::new(
            endpoint,
            local_pq,
            ingest_tx,
            Arc::new(NoopCompatSink),
        )),
        ingest_rx,
    )
}

/// A manager with `self_node_id` pinned to a known value, so the dial-vs-await
/// election against a sentinel peer is deterministic.
pub async fn test_manager_with_self(
    self_node_id: [u8; 32],
) -> (Arc<TunnelManager>, mpsc::Receiver<InboundDm>) {
    let endpoint = wrap(loopback_endpoint(tunnel_alpns()).await);
    let local_pq = Arc::new(PqPrivateIdentity::generate(&mut rand::rngs::OsRng));
    let (ingest_tx, ingest_rx) = mpsc::channel(16);
    (
        Arc::new(TunnelManager::new_with_self_node_id(
            endpoint,
            local_pq,
            ingest_tx,
            self_node_id,
        )),
        ingest_rx,
    )
}

/// A [`TunnelPeer`] fixture with a real (generated) PQ identity and a fixed
/// bogus iroh `node_id` byte. The manager unit tests never actually dial it (on
/// a `current_thread` runtime the spawned dial can't run before the sync
/// assertion), so the iroh id need not be a valid endpoint key.
pub fn test_peer(iroh_id_byte: u8) -> TunnelPeer {
    let pq = PqPrivateIdentity::generate(&mut rand::rngs::OsRng);
    TunnelPeer {
        node_id: [iroh_id_byte; 32],
        pq_identity: pq.public_identity().clone(),
        home_relay: None,
    }
}

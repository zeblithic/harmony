//! ALPN-keyed inbound-connection dispatch.
//!
//! Generalized from harmony-client (ZEB-739): the client had a
//! `MultiplexHandshakeDispatcher` that hard-coded a friend/invite/PEX fan-out
//! and an accept loop with per-ALPN `if/else` arms. Here that collapses to a
//! data-driven [`AlpnDispatchTable`] — a map from ALPN byte-string to a
//! [`IrohHandshakeDispatcher`] — plus a generic [`spawn_accept`] loop that
//! reads each inbound connection's negotiated ALPN and routes it through the
//! table. Callers register whichever protocols they speak; this crate knows
//! none of them.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;

/// Pluggable handler for an inbound iroh connection whose negotiated ALPN the
/// accept loop has already matched to this dispatcher.
///
/// Lifted from harmony-client's `IrohHandshakeDispatcher`: the accept loop
/// hands the accepted [`iroh::endpoint::Connection`] directly — implementations
/// own it and are responsible for opening any bi-streams and consuming it.
/// Implementations may run synchronously or spawn their own task; errors are
/// not propagated (log-and-return).
#[async_trait]
pub trait IrohHandshakeDispatcher: Send + Sync {
    /// Called once per inbound connection that routed to this dispatcher.
    async fn handle_connection(&self, conn: iroh::endpoint::Connection);
}

/// Routes inbound connections to a [`IrohHandshakeDispatcher`] by their
/// negotiated ALPN. Replaces the client's hard-coded multiplexer with a
/// registry the caller populates.
#[derive(Default)]
pub struct AlpnDispatchTable {
    routes: HashMap<Vec<u8>, Arc<dyn IrohHandshakeDispatcher>>,
}

impl AlpnDispatchTable {
    /// A new, empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `d` as the dispatcher for connections negotiating `alpn`.
    /// A later insert for the same ALPN replaces the earlier dispatcher.
    pub fn insert(&mut self, alpn: Vec<u8>, d: Arc<dyn IrohHandshakeDispatcher>) {
        self.routes.insert(alpn, d);
    }

    /// The dispatcher registered for `alpn`, if any. The returned `Arc` is a
    /// clone, so the caller can move it into a spawned task while the table
    /// stays shared.
    pub fn dispatch_for(&self, alpn: &[u8]) -> Option<Arc<dyn IrohHandshakeDispatcher>> {
        self.routes.get(alpn).cloned()
    }
}

/// Spawn the inbound-accept loop for `endpoint`: for each incoming connection,
/// resolve its negotiated ALPN against `table` and hand the connection to the
/// matched dispatcher on its own task. Connections whose ALPN has no registered
/// dispatcher are dropped (which closes them); iroh already filters inbound
/// connections to the ALPNs the endpoint advertised, so an unmatched ALPN is a
/// defensive tail rather than an expected case.
///
/// The loop ends when `endpoint`'s underlying [`iroh::Endpoint`] is closed
/// (`accept()` then returns `None`) — hold the returned [`tokio::task::JoinHandle`]
/// to drive shutdown. Errors resolving an individual `Incoming` are swallowed
/// per-connection so one bad dial can't tear down the loop.
///
/// This is deliberately minimal: it carries none of the client's
/// connection-cap / boot-window-queue / same-peer-supersession machinery, which
/// was zenoh-transport-specific. A caller needing that layers it inside its
/// [`IrohHandshakeDispatcher`] implementation.
pub fn spawn_accept(
    endpoint: Arc<crate::endpoint::IrohEndpoint>,
    table: Arc<AlpnDispatchTable>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let ep = endpoint.inner().clone();
        while let Some(incoming) = ep.accept().await {
            let table = Arc::clone(&table);
            tokio::spawn(async move {
                let conn = match incoming.await {
                    Ok(c) => c,
                    // Handshake/connect failed before we ever saw an ALPN.
                    Err(_) => return,
                };
                // `alpn()` returns the negotiated peer ALPN; the returned `Arc`
                // no longer borrows `conn`, so `conn` can be moved into the
                // dispatcher.
                if let Some(dispatcher) = table.dispatch_for(conn.alpn()) {
                    dispatcher.handle_connection(conn).await;
                }
                // else: unknown ALPN — drop `conn` (closes it).
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Recorder(std::sync::Arc<std::sync::atomic::AtomicUsize>);

    #[async_trait]
    impl IrohHandshakeDispatcher for Recorder {
        async fn handle_connection(&self, _c: iroh::endpoint::Connection) {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        }
    }

    #[test]
    fn table_routes_by_alpn() {
        let mut t = AlpnDispatchTable::new();
        let r = Arc::new(Recorder(Default::default()));
        t.insert(b"proto/a".to_vec(), r.clone());
        assert!(t.dispatch_for(b"proto/a").is_some());
        assert!(t.dispatch_for(b"proto/unknown").is_none());
    }
}

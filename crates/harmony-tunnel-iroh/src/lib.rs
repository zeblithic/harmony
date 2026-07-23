//! Per-peer iroh QUIC tunnel manager + async driver for Harmony's PQ tunnel.
//!
//! Extracted to core from harmony-client (ZEB-739, iroh-tier task 4). This crate
//! re-hosts the iroh-stream glue around the shared no_std
//! [`harmony_tunnel`](harmony_tunnel) PQ session:
//!
//! * [`TunnelManager`] — the per-peer session map: lazily dials an outbound
//!   tunnel on the first DM, reuses an inbound tunnel a peer opened to us,
//!   buffers DMs sent before a dial completes, and resolves simultaneous-dial
//!   collisions deterministically (lower-NodeId initiator wins).
//! * [`run_tunnel_initiator`] / [`run_tunnel_responder`] — the per-connection
//!   async driver that pumps iroh QUIC bi-streams into a `TunnelSession`,
//!   preserving the 4-byte big-endian length framing byte-for-byte.
//!
//! Two app-coupling seams were cut for reuse:
//!
//! * **Seam A** — [`TunnelPeer`] replaces the client's owner-state
//!   `DeviceTunnelContact`. The app maps its device contact into a `TunnelPeer`
//!   at the [`TunnelManager::send_dm`] boundary.
//! * **Seam B** — the [`CompatSink`] trait replaces the concrete
//!   `ProtocolCompatRegistry`. The driver reports one
//!   [`HandshakeOutcome`] per peer; the app supplies a sink (the client's
//!   network-health registry; harmony-node can use a no-op).
//!
//! This crate takes the workspace's shared `iroh` 1.0 pin — the same version
//! every iroh-touching crate in the workspace now resolves to.

mod driver;
mod framing;
mod manager;
mod peer;
mod versioning;

#[cfg(test)]
mod testsupport;

pub use driver::{run_tunnel_initiator, run_tunnel_responder};
pub use manager::{
    node_id_from_dsa_pubkey, CompatSink, HandshakeOutcome, InboundDm, TunnelCommand, TunnelManager,
};
pub use peer::TunnelPeer;
// ZEB-739 task 5: re-export the ALPN wire strings so a consumer (harmony-node)
// can advertise EXACTLY the ALPNs this crate's driver negotiates on
// (`run_tunnel_initiator` dials `/v2` then `/v1`; `run_tunnel_responder` reads
// the generation from `conn.alpn()`). Single-sourcing them here prevents the
// endpoint's advertised set from drifting from the driver's negotiated set.
pub use versioning::alpn;

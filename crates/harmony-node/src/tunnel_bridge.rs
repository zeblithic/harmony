//! Node-side glue for the shared iroh tunnel stack (harmony-iroh +
//! harmony-tunnel-iroh).
//!
//! ZEB-739 (task 5): harmony-node's forked per-connection tunnel driver (the old
//! `tunnel_task.rs`) was deleted and the node rewired onto the shared
//! `harmony-tunnel-iroh` crate — the single home for the DM-over-iroh tunnel the
//! client also uses. This module holds the two small node-specific adapters the
//! shared crate needs, plus the node's monotonic clock:
//!
//!  * [`NodeCompatSink`] — a no-op [`CompatSink`]. The client surfaces protocol
//!    incompatibility in a network-health view; harmony-node has no such UI, so
//!    it discards the driver's per-peer compatibility reports.
//!  * [`millis_since_start`] — the node's process-global monotonic clock, used
//!    for `RuntimeEvent` timestamps and deferred-dial scheduling. (The shared
//!    driver keeps its OWN private epoch for `TunnelSession` timestamps; the two
//!    never compare across the boundary, so separate epochs are fine.)
//!
//! The node is NOT a DM consumer: inbound DMs the shared driver surfaces on its
//! ingest channel are drained-and-dropped by the event loop, preserving the old
//! `tunnel_task` behavior of ignoring `DmReceived`. The node's Zenoh- and
//! replication-over-tunnel paths were verified dead (all send sites are stubs;
//! the shared DM-only driver drops any inbound non-DM frame) and elided.

use std::time::Instant;

use harmony_tunnel_iroh::{CompatSink, HandshakeOutcome};

/// No-op protocol-compatibility sink for harmony-node.
///
/// The tunnel driver reports one [`HandshakeOutcome`] per peer (compatible /
/// incompatible-hello). harmony-node has no network-health surface to join those
/// on, so it discards them — the DM tunnel is otherwise a no-op on this node.
pub struct NodeCompatSink;

impl CompatSink for NodeCompatSink {
    fn record_handshake_outcome(&self, _peer: [u8; 32], _outcome: HandshakeOutcome) {}
}

/// Monotonic milliseconds since the first call (process-global epoch).
///
/// Used for `RuntimeEvent` timestamps and deferred-dial fire times. Shared epoch
/// across all node callers via `OnceLock`.
pub fn millis_since_start() -> u64 {
    use std::sync::OnceLock;
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_millis() as u64
}

//! Bridge between spawned iroh tunnel tasks and the main event loop.
//!
//! Each active tunnel connection runs in its own spawned task. Events flow
//! back to the select loop via an mpsc channel, matching the pattern used
//! for Zenoh task bridging.

use tokio::sync::mpsc;

/// Events sent from tunnel tasks to the main select loop.
#[derive(Debug)]
pub enum TunnelBridgeEvent {
    /// A tunnel handshake completed — peer is authenticated.
    HandshakeComplete {
        /// Short hex identifier for the tunnel interface name.
        interface_name: String,
        /// The remote peer's NodeId (BLAKE3 of ML-DSA pubkey), for routing.
        peer_node_id: [u8; 32],
        /// The remote peer's ML-DSA public key bytes.
        peer_dsa_pubkey: Vec<u8>,
        /// Monotonic connection ID assigned by the event loop.
        connection_id: u64,
    },
    /// A decrypted Reticulum packet arrived from a tunnel peer.
    ReticulumReceived {
        interface_name: String,
        packet: Vec<u8>,
    },
    /// A decrypted Zenoh message arrived from a tunnel peer.
    ZenohReceived {
        interface_name: String,
        message: Vec<u8>,
    },
    /// A tunnel connection was closed or errored.
    TunnelClosed {
        interface_name: String,
        reason: String,
        /// Monotonic connection ID — only remove the sender if it matches.
        connection_id: u64,
    },
}

/// Handle for sending commands into a tunnel task (held by the event loop).
#[derive(Debug, Clone)]
pub struct TunnelSender {
    tx: mpsc::Sender<TunnelCommand>,
    /// Monotonic connection ID for stale-close detection.
    pub connection_id: u64,
}

/// Commands sent from the event loop to a tunnel task.
#[derive(Debug)]
pub enum TunnelCommand {
    /// Send a Reticulum packet through this tunnel.
    SendReticulum { packet: Vec<u8> },
    /// Send a Zenoh message through this tunnel.
    SendZenoh { message: Vec<u8> },
    /// Close this tunnel.
    Close,
}

/// A QUIC connection that completed its handshake and is ready for tunnel setup.
///
/// Sent from a spawned handshake task back to the event loop via an mpsc channel,
/// so the QUIC round-trip doesn't block the select loop.
pub struct ReadyConnection {
    pub connection: iroh::endpoint::Connection,
    pub connection_id: u64,
    pub interface_name: String,
}

impl TunnelSender {
    pub fn new(tx: mpsc::Sender<TunnelCommand>, connection_id: u64) -> Self {
        Self { tx, connection_id }
    }

    pub fn try_send_reticulum(&self, packet: Vec<u8>) -> Result<(), mpsc::error::TrySendError<TunnelCommand>> {
        self.tx.try_send(TunnelCommand::SendReticulum { packet })
    }
}

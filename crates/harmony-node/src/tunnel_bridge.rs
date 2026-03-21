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
    },
}

/// Handle for sending commands into a tunnel task (held by the event loop).
#[derive(Debug, Clone)]
pub struct TunnelSender {
    tx: mpsc::Sender<TunnelCommand>,
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

impl TunnelSender {
    pub fn new(tx: mpsc::Sender<TunnelCommand>) -> Self {
        Self { tx }
    }

    pub async fn send_reticulum(&self, packet: Vec<u8>) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::SendReticulum { packet }).await
    }

    pub fn try_send_reticulum(&self, packet: Vec<u8>) -> Result<(), mpsc::error::TrySendError<TunnelCommand>> {
        self.tx.try_send(TunnelCommand::SendReticulum { packet })
    }

    pub async fn send_zenoh(&self, message: Vec<u8>) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::SendZenoh { message }).await
    }

    pub async fn close(&self) -> Result<(), mpsc::error::SendError<TunnelCommand>> {
        self.tx.send(TunnelCommand::Close).await
    }
}

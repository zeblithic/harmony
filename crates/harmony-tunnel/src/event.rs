use alloc::vec::Vec;

/// Events fed into the TunnelSession by the caller.
#[derive(Debug)]
pub enum TunnelEvent {
    /// Raw bytes received from the transport (iroh-net connection).
    InboundBytes { data: Vec<u8> },
    /// Send a Reticulum packet through this tunnel.
    SendReticulum { packet: Vec<u8> },
    /// Send a Zenoh message through this tunnel.
    SendZenoh { message: Vec<u8> },
    /// Periodic timer tick for keepalive management.
    Tick { now_ms: u64 },
    /// Request graceful tunnel shutdown.
    Close,
}

/// Actions the TunnelSession asks the caller to perform.
#[derive(Debug)]
pub enum TunnelAction {
    /// Encrypted bytes to write to the transport.
    OutboundBytes { data: Vec<u8> },
    /// A decrypted Reticulum packet received from the tunnel peer.
    ReticulumReceived { packet: Vec<u8> },
    /// A decrypted Zenoh message received from the tunnel peer.
    ZenohReceived { message: Vec<u8> },
    /// Handshake completed — the peer's PQ identity has been authenticated.
    HandshakeComplete {
        peer_dsa_pubkey: Vec<u8>,
        peer_node_id: [u8; 32],
    },
    /// An error occurred in the tunnel.
    Error { reason: alloc::string::String },
    /// The tunnel has been closed.
    Closed,
}

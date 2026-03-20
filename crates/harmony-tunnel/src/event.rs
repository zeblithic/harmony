/// Events fed into the TunnelSession by the caller.
#[derive(Debug)]
pub enum TunnelEvent {
    /// Raw bytes received from the transport (iroh-net connection).
    InboundBytes { data: alloc::vec::Vec<u8> },
    /// Periodic timer tick for keepalive management.
    Tick { now_ms: u64 },
}

/// Actions the TunnelSession asks the caller to perform.
#[derive(Debug)]
pub enum TunnelAction {
    /// Encrypted bytes to write to the transport.
    OutboundBytes { data: alloc::vec::Vec<u8> },
}

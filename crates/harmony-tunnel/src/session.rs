use crate::error::TunnelError;
use crate::event::{TunnelAction, TunnelEvent};

/// Tunnel session states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TunnelState {
    Idle,
    Initiating,
    Active,
    Closed,
}

/// Sans-I/O state machine for a single tunnel connection.
pub struct TunnelSession {
    state: TunnelState,
}

impl TunnelSession {
    /// Returns the current state.
    pub fn state(&self) -> TunnelState {
        self.state
    }
}

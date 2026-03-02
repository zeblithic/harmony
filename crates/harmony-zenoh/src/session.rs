//! Sans-I/O session state machine for peer-to-peer connections.
//!
//! Each `Session` manages a single two-party encrypted connection:
//! identity-based handshake, resource ID mapping, keepalive, and
//! graceful close. The caller drives the session via events and
//! executes the returned actions.

use std::collections::HashMap;

use harmony_identity::{Identity, PrivateIdentity};

use crate::ZenohError;

/// Numeric expression ID for wire-efficient key expression references.
pub type ExprId = u64;

/// Session lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// Awaiting mutual identity handshake.
    Init,
    /// Authenticated, resources can be declared/undeclared.
    Active,
    /// Close message sent, waiting for ack.
    Closing,
    /// Terminal state, no further events accepted.
    Closed,
}

/// Tunable session timing parameters.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Interval between keepalive emissions (ms).
    pub keepalive_interval_ms: u64,
    /// Time without any inbound event before declaring peer stale (ms).
    pub stale_timeout_ms: u64,
    /// Time to wait for CloseAck before force-closing (ms).
    pub close_timeout_ms: u64,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            keepalive_interval_ms: 30_000,
            stale_timeout_ms: 90_000,
            close_timeout_ms: 5_000,
        }
    }
}

/// Inbound events the caller feeds into the session.
#[derive(Debug, Clone)]
pub enum SessionEvent {
    /// Peer's handshake proof received.
    HandshakeReceived { proof: Vec<u8> },
    /// Peer declared a resource (key expression <-> ExprId mapping).
    ResourceDeclared { expr_id: ExprId, key_expr: String },
    /// Peer undeclared a resource.
    ResourceUndeclared { expr_id: ExprId },
    /// Peer sent a keepalive.
    KeepaliveReceived,
    /// Peer initiated close.
    CloseReceived,
    /// Peer acknowledged our close.
    CloseAckReceived,
    /// Timer tick with caller-provided timestamp.
    TimerTick { now_ms: u64 },
}

/// Outbound actions the session returns for the caller to execute.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SessionAction {
    /// Send our handshake proof to the peer.
    SendHandshake { proof: Vec<u8> },
    /// Send a keepalive to the peer.
    SendKeepalive,
    /// Send a close request to the peer.
    SendClose,
    /// Send a close acknowledgment to the peer.
    SendCloseAck,
    /// Send a resource declaration to the peer.
    SendResourceDeclare { expr_id: ExprId, key_expr: String },
    /// Send a resource undeclaration to the peer.
    SendResourceUndeclare { expr_id: ExprId },
    /// A new remote resource is available.
    ResourceAdded { expr_id: ExprId, key_expr: String },
    /// A remote resource was removed.
    ResourceRemoved { expr_id: ExprId },
    /// Peer has gone stale (no inbound events within timeout).
    PeerStale,
    /// Session is now authenticated and active.
    SessionOpened,
    /// Session has been closed (gracefully or forced).
    SessionClosed,
}

/// A sans-I/O session managing a single peer-to-peer connection.
pub struct Session {
    state: SessionState,
    local_identity: PrivateIdentity,
    remote_identity: Identity,
    handshake_sent: bool,
    handshake_verified: bool,
    local_resources: HashMap<ExprId, String>,
    remote_resources: HashMap<ExprId, String>,
    next_expr_id: ExprId,
    last_received_ms: u64,
    last_sent_ms: u64,
    last_tick_ms: u64,
    close_initiated_ms: Option<u64>,
    config: SessionConfig,
}

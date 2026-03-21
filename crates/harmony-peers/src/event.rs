use alloc::string::String;
use harmony_identity::IdentityHash;

/// Events fed into the PeerManager by the caller/runtime.
#[derive(Debug, Clone)]
pub enum PeerEvent {
    ContactChanged {
        identity_hash: IdentityHash,
    },
    ContactRemoved {
        identity_hash: IdentityHash,
    },
    AnnounceReceived {
        identity_hash: IdentityHash,
    },
    LinkEstablished {
        identity_hash: IdentityHash,
        now: u64,
    },
    LinkClosed {
        identity_hash: IdentityHash,
    },
    TunnelEstablished {
        identity_hash: IdentityHash,
        node_id: [u8; 32],
        now: u64,
    },
    TunnelFailed {
        identity_hash: IdentityHash,
    },
    TunnelDropped {
        identity_hash: IdentityHash,
    },
    Tick {
        now: u64,
    },
}

/// Actions the PeerManager asks the caller to perform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PeerAction {
    InitiateLink {
        identity_hash: IdentityHash,
    },
    InitiateTunnel {
        identity_hash: IdentityHash,
        node_id: [u8; 32],
        relay_url: Option<String>,
    },
    SendPathRequest {
        identity_hash: IdentityHash,
    },
    CloseLink {
        identity_hash: IdentityHash,
    },
    UpdateLastSeen {
        identity_hash: IdentityHash,
        timestamp: u64,
    },
}

//! harmony-rawlink: AF_PACKET bridge for Zenoh over raw 802.11s Ethernet.
//!
//! Bypasses the Linux IP stack by sending/receiving Ethernet frames directly
//! via AF_PACKET sockets with TPACKET_V3 ring buffers. Bridges to Zenoh
//! pub/sub via shared memory for zero-copy delivery.

#[cfg(target_os = "linux")]
pub mod af_packet;
pub mod bridge;
pub mod error;
pub mod batch;
pub mod frame;
pub mod padded_socket;
pub mod peer_table;
pub mod socket;

pub use bridge::{Bridge, BridgeConfig};
pub use padded_socket::{PaddedSocket, DEFAULT_PAD_BLOCK};

/// IEEE 802.1 Local Experimental EtherType, shared across all Harmony L2 protocols.
pub const HARMONY_ETHERTYPE: u16 = 0x88B5;

/// Frame type discriminator (first byte after EtherType).
pub mod frame_type {
    /// Raw Reticulum packet (harmony-os, harmony-kuw).
    pub const RETICULUM: u8 = 0x00;
    /// L2 scouting — broadcast presence announcement.
    pub const SCOUT: u8 = 0x01;
    /// Zenoh data — carries key expression + payload.
    pub const DATA: u8 = 0x02;
    /// Batch container — multiple sub-frames in one Ethernet frame.
    pub const BATCH: u8 = 0x03;
}

/// Ethernet header size: 6 dst + 6 src + 2 EtherType.
pub const ETH_HEADER_LEN: usize = 14;

/// Total overhead: Ethernet header + 1 byte frame type tag.
pub const FRAME_OVERHEAD: usize = ETH_HEADER_LEN + 1;

/// Maximum payload after overhead within 1500-byte MTU.
pub const MAX_PAYLOAD: usize = 1500 - FRAME_OVERHEAD;

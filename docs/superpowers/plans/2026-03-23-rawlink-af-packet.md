# harmony-rawlink Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bridge AF_PACKET raw Ethernet sockets to Zenoh via shared memory, bypassing the Linux IP stack for 802.11s mesh networking.

**Architecture:** New `harmony-rawlink` crate with `RawSocket` trait abstraction (real AF_PACKET on Linux, mock elsewhere), TPACKET_V3 ring buffers for zero-copy NIC ingestion, zenoh SHM for zero-copy bridge-to-consumer delivery, and L2 broadcast scouting for peer discovery.

**Tech Stack:** Rust, libc (AF_PACKET/TPACKET_V3/BPF), zenoh (shared-memory feature), tokio, postcard

**Spec:** `docs/superpowers/specs/2026-03-23-rawlink-af-packet-design.md`

---

### Task 1: Crate scaffolding, constants, and error types

Create the `harmony-rawlink` crate with workspace membership, define the EtherType constant, frame type tags, and error types.

**Files:**
- Modify: `Cargo.toml` (workspace root) — add member + dependencies
- Create: `crates/harmony-rawlink/Cargo.toml`
- Create: `crates/harmony-rawlink/src/lib.rs`
- Create: `crates/harmony-rawlink/src/error.rs`

- [ ] **Step 1: Add workspace member and dependencies**

In the root `Cargo.toml`, add `"crates/harmony-rawlink"` to the `[workspace]` members list (after the last member, ~line 32).

Add to `[workspace.dependencies]`:
```toml
libc = "0.2"
```

Update the existing zenoh dependency (line 89) to enable SHM:
```toml
zenoh = { version = "1", features = ["shared-memory"] }
```

- [ ] **Step 2: Create crate Cargo.toml**

Create `crates/harmony-rawlink/Cargo.toml`:
```toml
[package]
name = "harmony-rawlink"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true

[features]
default = ["std"]
std = []
rawlink-integration = []  # Linux-only integration tests

[dependencies]
hex.workspace = true
libc = { workspace = true }
postcard.workspace = true
rand.workspace = true
serde = { workspace = true, default-features = false, features = ["derive", "alloc"] }
tokio = { workspace = true }
tracing.workspace = true
zenoh = { workspace = true }

[dev-dependencies]
tokio = { workspace = true, features = ["rt", "macros", "test-util"] }
```

- [ ] **Step 3: Create lib.rs**

Create `crates/harmony-rawlink/src/lib.rs`:
```rust
//! harmony-rawlink: AF_PACKET bridge for Zenoh over raw 802.11s Ethernet.
//!
//! Bypasses the Linux IP stack by sending/receiving Ethernet frames directly
//! via AF_PACKET sockets with TPACKET_V3 ring buffers. Bridges to Zenoh
//! pub/sub via shared memory for zero-copy delivery.

pub mod error;
pub mod frame;

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
}

/// Ethernet header size: 6 dst + 6 src + 2 EtherType.
pub const ETH_HEADER_LEN: usize = 14;

/// Total overhead: Ethernet header + 1 byte frame type tag.
pub const FRAME_OVERHEAD: usize = ETH_HEADER_LEN + 1;

/// Maximum payload after overhead within 1500-byte MTU.
pub const MAX_PAYLOAD: usize = 1500 - FRAME_OVERHEAD;
```

- [ ] **Step 4: Create error.rs**

Create `crates/harmony-rawlink/src/error.rs`:
```rust
//! Error types for the rawlink crate.

use std::fmt;

/// Errors that can occur in rawlink operations.
#[derive(Debug)]
pub enum RawLinkError {
    /// Socket creation or binding failed.
    SocketError(String),
    /// Ring buffer setup failed.
    RingError(String),
    /// Frame encoding/decoding failed.
    FrameError(String),
    /// I/O error from the underlying socket.
    IoError(std::io::Error),
    /// Capability check failed (CAP_NET_RAW missing).
    PermissionDenied(String),
}

impl fmt::Display for RawLinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SocketError(msg) => write!(f, "socket error: {msg}"),
            Self::RingError(msg) => write!(f, "ring buffer error: {msg}"),
            Self::FrameError(msg) => write!(f, "frame error: {msg}"),
            Self::IoError(e) => write!(f, "I/O error: {e}"),
            Self::PermissionDenied(msg) => write!(f, "permission denied: {msg}"),
        }
    }
}

impl std::error::Error for RawLinkError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::IoError(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RawLinkError {
    fn from(e: std::io::Error) -> Self {
        if e.raw_os_error() == Some(libc::EPERM) || e.raw_os_error() == Some(libc::EACCES) {
            Self::PermissionDenied(
                "AF_PACKET requires CAP_NET_RAW capability (run as root or setcap)".into(),
            )
        } else {
            Self::IoError(e)
        }
    }
}
```

- [ ] **Step 5: Verify compilation**

```bash
cargo check -p harmony-rawlink
```

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/harmony-rawlink/
git commit -m "feat(rawlink): scaffold crate with EtherType, frame types, and errors"
```

---

### Task 2: Frame encoding and decoding

Implement Scout and Data frame serialization/deserialization. Pure logic, fully testable.

**Files:**
- Create: `crates/harmony-rawlink/src/frame.rs`

- [ ] **Step 1: Write failing tests**

Add to `frame.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scout_frame_round_trip() {
        let identity_hash = [0xAA; 16];
        let src_mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let encoded = encode_scout_frame(src_mac, &identity_hash);
        assert_eq!(encoded[12..14], HARMONY_ETHERTYPE.to_be_bytes());
        assert_eq!(encoded[14], frame_type::SCOUT);
        let (parsed_src, parsed_hash) = decode_scout_frame(&encoded).unwrap();
        assert_eq!(parsed_src, src_mac);
        assert_eq!(parsed_hash, identity_hash);
    }

    #[test]
    fn data_frame_round_trip() {
        let src_mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        let dst_mac = [0xFF; 6];
        let key_expr = "harmony/identity/test";
        let payload = b"hello world";
        let encoded = encode_data_frame(src_mac, dst_mac, key_expr, payload);
        assert_eq!(encoded[12..14], HARMONY_ETHERTYPE.to_be_bytes());
        assert_eq!(encoded[14], frame_type::DATA);
        let (parsed_src, parsed_key, parsed_payload) = decode_data_frame(&encoded).unwrap();
        assert_eq!(parsed_src, src_mac);
        assert_eq!(parsed_key, key_expr);
        assert_eq!(parsed_payload, payload);
    }

    #[test]
    fn scout_frame_broadcast_dst() {
        let encoded = encode_scout_frame([0; 6], &[0; 16]);
        assert_eq!(&encoded[0..6], &[0xFF; 6]); // broadcast MAC
    }

    #[test]
    fn data_frame_too_short_rejected() {
        let short = vec![0u8; 10];
        assert!(decode_data_frame(&short).is_err());
    }

    #[test]
    fn wrong_ethertype_rejected() {
        let mut frame = encode_scout_frame([0; 6], &[0; 16]);
        frame[12] = 0x08; frame[13] = 0x00; // IP EtherType
        assert!(decode_scout_frame(&frame).is_err());
    }

    #[test]
    fn wrong_frame_type_rejected() {
        let mut frame = encode_scout_frame([0; 6], &[0; 16]);
        frame[14] = frame_type::DATA; // wrong type for scout decoder
        assert!(decode_scout_frame(&frame).is_err());
    }
}
```

- [ ] **Step 2: Implement frame encoding/decoding**

```rust
//! Ethernet frame encoding and decoding for Harmony L2 protocol.

use crate::error::RawLinkError;
use crate::{frame_type, ETH_HEADER_LEN, FRAME_OVERHEAD, HARMONY_ETHERTYPE};

/// Broadcast MAC address.
pub const BROADCAST_MAC: [u8; 6] = [0xFF; 6];

/// Encode a Scout frame (broadcast, carries identity_hash).
pub fn encode_scout_frame(src_mac: [u8; 6], identity_hash: &[u8; 16]) -> Vec<u8> {
    let mut frame = Vec::with_capacity(FRAME_OVERHEAD + 16);
    frame.extend_from_slice(&BROADCAST_MAC);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&HARMONY_ETHERTYPE.to_be_bytes());
    frame.push(frame_type::SCOUT);
    frame.extend_from_slice(identity_hash);
    frame
}

/// Decode a Scout frame. Returns (src_mac, identity_hash).
pub fn decode_scout_frame(frame: &[u8]) -> Result<([u8; 6], [u8; 16]), RawLinkError> {
    if frame.len() < FRAME_OVERHEAD + 16 {
        return Err(RawLinkError::FrameError("scout frame too short".into()));
    }
    validate_header(frame, frame_type::SCOUT)?;
    let src_mac: [u8; 6] = frame[6..12].try_into().unwrap();
    let identity_hash: [u8; 16] = frame[15..31].try_into().unwrap();
    Ok((src_mac, identity_hash))
}

/// Encode a Data frame (carries zenoh key expression + payload).
pub fn encode_data_frame(
    src_mac: [u8; 6],
    dst_mac: [u8; 6],
    key_expr: &str,
    payload: &[u8],
) -> Vec<u8> {
    let key_bytes = key_expr.as_bytes();
    let key_len = key_bytes.len() as u16;
    let mut frame = Vec::with_capacity(FRAME_OVERHEAD + 2 + key_bytes.len() + payload.len());
    frame.extend_from_slice(&dst_mac);
    frame.extend_from_slice(&src_mac);
    frame.extend_from_slice(&HARMONY_ETHERTYPE.to_be_bytes());
    frame.push(frame_type::DATA);
    frame.extend_from_slice(&key_len.to_be_bytes());
    frame.extend_from_slice(key_bytes);
    frame.extend_from_slice(payload);
    frame
}

/// Decode a Data frame. Returns (src_mac, key_expr, payload).
pub fn decode_data_frame(frame: &[u8]) -> Result<([u8; 6], String, Vec<u8>), RawLinkError> {
    if frame.len() < FRAME_OVERHEAD + 2 {
        return Err(RawLinkError::FrameError("data frame too short".into()));
    }
    validate_header(frame, frame_type::DATA)?;
    let src_mac: [u8; 6] = frame[6..12].try_into().unwrap();
    let key_len = u16::from_be_bytes([frame[15], frame[16]]) as usize;
    if frame.len() < FRAME_OVERHEAD + 2 + key_len {
        return Err(RawLinkError::FrameError("data frame key truncated".into()));
    }
    let key_expr = std::str::from_utf8(&frame[17..17 + key_len])
        .map_err(|e| RawLinkError::FrameError(format!("invalid key expression UTF-8: {e}")))?
        .to_string();
    let payload = frame[17 + key_len..].to_vec();
    Ok((src_mac, key_expr, payload))
}

/// Validate Ethernet header: EtherType and frame type tag.
fn validate_header(frame: &[u8], expected_type: u8) -> Result<(), RawLinkError> {
    if frame.len() < FRAME_OVERHEAD {
        return Err(RawLinkError::FrameError("frame too short for header".into()));
    }
    let ethertype = u16::from_be_bytes([frame[12], frame[13]]);
    if ethertype != HARMONY_ETHERTYPE {
        return Err(RawLinkError::FrameError(format!(
            "wrong EtherType: expected 0x{:04X}, got 0x{ethertype:04X}",
            HARMONY_ETHERTYPE
        )));
    }
    if frame[14] != expected_type {
        return Err(RawLinkError::FrameError(format!(
            "wrong frame type: expected 0x{expected_type:02X}, got 0x{:02X}",
            frame[14]
        )));
    }
    Ok(())
}
```

Note: The spec says LEB128 for key expression length but u16 big-endian is simpler and sufficient (max key expression is 65535 bytes, well beyond any real key). Use u16 BE for consistency with the existing frame encoding patterns in the codebase.

- [ ] **Step 3: Run tests**

```bash
cargo test -p harmony-rawlink
```

- [ ] **Step 4: Commit**

```bash
git add crates/harmony-rawlink/src/frame.rs crates/harmony-rawlink/src/lib.rs
git commit -m "feat(rawlink): add Scout and Data frame encoding/decoding"
```

---

### Task 3: RawSocket trait and MockSocket

Define the platform abstraction trait and a channel-backed mock for testing.

**Files:**
- Create: `crates/harmony-rawlink/src/socket.rs`
- Modify: `crates/harmony-rawlink/src/lib.rs` — add module

- [ ] **Step 1: Write MockSocket tests**

Add to `socket.rs`:
```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_socket_send_recv_round_trip() {
        let (mut sock_a, mut sock_b) = MockSocket::pair([0xAA; 6], [0xBB; 6]);
        let payload = b"hello";
        sock_a.send_frame([0xBB; 6], payload).unwrap();
        let mut received = Vec::new();
        sock_b.recv_frames(&mut |src_mac, data| {
            received.push((src_mac.to_owned(), data.to_vec()));
        }).unwrap();
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].0, [0xAA; 6]);
        assert_eq!(received[0].1, payload);
    }

    #[test]
    fn mock_socket_local_mac() {
        let (sock, _) = MockSocket::pair([0xAA; 6], [0xBB; 6]);
        assert_eq!(sock.local_mac(), [0xAA; 6]);
    }

    #[test]
    fn mock_socket_empty_recv() {
        let (_, mut sock_b) = MockSocket::pair([0xAA; 6], [0xBB; 6]);
        let mut count = 0;
        sock_b.recv_frames(&mut |_, _| { count += 1; }).unwrap();
        assert_eq!(count, 0);
    }
}
```

- [ ] **Step 2: Implement RawSocket trait and MockSocket**

```rust
//! Platform-abstracted raw socket interface.

use crate::error::RawLinkError;

/// Platform-abstracted raw Ethernet socket.
///
/// Implementations provide send/receive of raw Ethernet frame payloads
/// (after the Ethernet header). The Ethernet header is handled by the
/// implementation — callers work with destination MAC + payload.
pub trait RawSocket: Send {
    /// Send a raw frame to the given destination MAC.
    fn send_frame(&mut self, dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError>;

    /// Receive all pending frames, invoking the callback for each.
    /// Callback receives (source_mac, payload_after_header).
    fn recv_frames(
        &mut self,
        callback: &mut dyn FnMut(&[u8; 6], &[u8]),
    ) -> Result<(), RawLinkError>;

    /// This socket's local MAC address.
    fn local_mac(&self) -> [u8; 6];
}

/// Channel-backed mock socket for cross-platform testing.
#[cfg(test)]
pub struct MockSocket {
    mac: [u8; 6],
    tx: std::sync::mpsc::Sender<([u8; 6], Vec<u8>)>,
    rx: std::sync::mpsc::Receiver<([u8; 6], Vec<u8>)>,
}

#[cfg(test)]
impl MockSocket {
    /// Create a connected pair of mock sockets.
    pub fn pair(mac_a: [u8; 6], mac_b: [u8; 6]) -> (Self, Self) {
        let (tx_a, rx_a) = std::sync::mpsc::channel();
        let (tx_b, rx_b) = std::sync::mpsc::channel();
        (
            Self { mac: mac_a, tx: tx_b, rx: rx_a },
            Self { mac: mac_b, tx: tx_a, rx: rx_b },
        )
    }
}

#[cfg(test)]
impl RawSocket for MockSocket {
    fn send_frame(&mut self, _dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError> {
        let _ = self.tx.send((self.mac, payload.to_vec()));
        Ok(())
    }

    fn recv_frames(
        &mut self,
        callback: &mut dyn FnMut(&[u8; 6], &[u8]),
    ) -> Result<(), RawLinkError> {
        while let Ok((src_mac, data)) = self.rx.try_recv() {
            callback(&src_mac, &data);
        }
        Ok(())
    }

    fn local_mac(&self) -> [u8; 6] {
        self.mac
    }
}
```

- [ ] **Step 3: Add module to lib.rs**

Add `pub mod socket;` to lib.rs.

- [ ] **Step 4: Run tests**

```bash
cargo test -p harmony-rawlink
```

- [ ] **Step 5: Commit**

```bash
git add crates/harmony-rawlink/
git commit -m "feat(rawlink): add RawSocket trait and MockSocket"
```

---

### Task 4: Peer table

Identity-to-MAC mapping populated by Scout frames, with expiry.

**Files:**
- Create: `crates/harmony-rawlink/src/peer_table.rs`
- Modify: `crates/harmony-rawlink/src/lib.rs` — add module

- [ ] **Step 1: Write tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn insert_and_lookup() {
        let mut table = PeerTable::new(Duration::from_secs(30));
        let id = [0xAA; 16];
        let mac = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06];
        table.update(id, mac);
        assert_eq!(table.lookup(&id), Some(mac));
    }

    #[test]
    fn unknown_returns_none() {
        let table = PeerTable::new(Duration::from_secs(30));
        assert_eq!(table.lookup(&[0xBB; 16]), None);
    }

    #[test]
    fn expired_entry_returns_none() {
        let mut table = PeerTable::new(Duration::from_millis(1));
        table.update([0xAA; 16], [0x01; 6]);
        std::thread::sleep(Duration::from_millis(10));
        assert_eq!(table.lookup(&[0xAA; 16]), None);
    }

    #[test]
    fn update_refreshes_expiry() {
        let mut table = PeerTable::new(Duration::from_millis(50));
        let id = [0xAA; 16];
        table.update(id, [0x01; 6]);
        std::thread::sleep(Duration::from_millis(30));
        table.update(id, [0x01; 6]); // refresh
        std::thread::sleep(Duration::from_millis(30));
        // 60ms total but refreshed at 30ms — still within TTL
        assert!(table.lookup(&id).is_some());
    }

    #[test]
    fn peer_count() {
        let mut table = PeerTable::new(Duration::from_secs(30));
        assert_eq!(table.peer_count(), 0);
        table.update([0x01; 16], [0x01; 6]);
        table.update([0x02; 16], [0x02; 6]);
        assert_eq!(table.peer_count(), 2);
    }
}
```

- [ ] **Step 2: Implement PeerTable**

```rust
//! Peer table mapping identity hashes to MAC addresses.

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Entry in the peer table.
struct PeerEntry {
    mac: [u8; 6],
    last_seen: Instant,
}

/// Maps identity_hash → MAC address, with TTL-based expiry.
pub struct PeerTable {
    entries: HashMap<[u8; 16], PeerEntry>,
    ttl: Duration,
}

impl PeerTable {
    pub fn new(ttl: Duration) -> Self {
        Self {
            entries: HashMap::new(),
            ttl,
        }
    }

    /// Update or insert a peer's MAC address, refreshing the expiry.
    pub fn update(&mut self, identity_hash: [u8; 16], mac: [u8; 6]) {
        self.entries.insert(
            identity_hash,
            PeerEntry {
                mac,
                last_seen: Instant::now(),
            },
        );
    }

    /// Look up a peer's MAC. Returns None if unknown or expired.
    pub fn lookup(&self, identity_hash: &[u8; 16]) -> Option<[u8; 6]> {
        self.entries.get(identity_hash).and_then(|entry| {
            if entry.last_seen.elapsed() < self.ttl {
                Some(entry.mac)
            } else {
                None
            }
        })
    }

    /// Number of entries (including expired — lazy eviction).
    pub fn peer_count(&self) -> usize {
        self.entries.len()
    }
}
```

- [ ] **Step 3: Add module to lib.rs, run tests, commit**

```bash
cargo test -p harmony-rawlink
git add crates/harmony-rawlink/
git commit -m "feat(rawlink): add PeerTable with TTL-based expiry"
```

---

### Task 5: TPACKET_V3 AfPacketSocket (Linux only)

The real AF_PACKET implementation with memory-mapped ring buffers and BPF filtering. This is the most complex task — it involves unsafe code, kernel APIs, and mmap'd memory.

**Files:**
- Create: `crates/harmony-rawlink/src/af_packet.rs`
- Modify: `crates/harmony-rawlink/src/lib.rs` — add module

**Context:**
- This file is `#[cfg(target_os = "linux")]` — it won't compile on macOS
- Uses `libc` for raw syscalls: `socket()`, `bind()`, `setsockopt()`, `mmap()`, `poll()`, `sendto()`
- TPACKET_V3 structs (`tpacket_req3`, `tpacket_block_desc`, `tpacket3_hdr`) may not be fully defined in the `libc` crate — define them manually if needed
- BPF program is a 3-instruction filter for EtherType 0x88B5

- [ ] **Step 1: Define TPACKET_V3 constants and structs**

At the top of `af_packet.rs`, define:
```rust
//! AF_PACKET socket with TPACKET_V3 ring buffers.
//!
//! Linux-only. Requires CAP_NET_RAW capability.

#![cfg(target_os = "linux")]

use crate::error::RawLinkError;
use crate::socket::RawSocket;
use crate::HARMONY_ETHERTYPE;
use crate::ETH_HEADER_LEN;

use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::ptr;

// ── TPACKET_V3 constants ──
const TPACKET_V3: libc::c_int = 2;
const PACKET_RX_RING: libc::c_int = 5;
const PACKET_TX_RING: libc::c_int = 13;
const PACKET_VERSION: libc::c_int = 10;

// Ring buffer configuration
const BLOCK_SIZE: u32 = 1 << 20;   // 1 MiB per block
const BLOCK_COUNT: u32 = 4;        // 4 blocks = 4 MiB total
const FRAME_SIZE: u32 = 2048;      // Max frame slot size
const BLOCK_TIMEOUT_MS: u32 = 100; // Retire block after 100ms

// TPACKET status flags
const TP_STATUS_KERNEL: u32 = 0;
const TP_STATUS_USER: u32 = 1;
const TP_STATUS_SEND_REQUEST: u32 = 1;
const TP_STATUS_AVAILABLE: u32 = 0;
```

Also define the `tpacket_req3` struct (check if `libc` has it; if not, define manually):
```rust
#[repr(C)]
struct tpacket_req3 {
    tp_block_size: libc::c_uint,
    tp_block_nr: libc::c_uint,
    tp_frame_size: libc::c_uint,
    tp_frame_nr: libc::c_uint,
    tp_retire_blk_tov: libc::c_uint,
    tp_sizeof_priv: libc::c_uint,
    tp_feature_req_word: libc::c_uint,
}
```

And the block descriptor + frame header structs needed for walking the ring.

**Important:** Verify all constant values and struct layouts against the Linux kernel headers (`linux/if_packet.h`). The values above are from kernel 5.x+ but check the `libc` crate first — it may define some of these.

- [ ] **Step 2: Implement AfPacketSocket struct**

```rust
pub struct AfPacketSocket {
    fd: OwnedFd,
    rx_ring: *mut u8,
    tx_ring: *mut u8,
    rx_ring_size: usize,
    tx_ring_size: usize,
    rx_block_idx: usize,
    tx_frame_idx: usize,
    local_mac: [u8; 6],
    if_index: i32,
}

// Safety: The mmap'd pointers are only accessed by the owning thread.
// AfPacketSocket is Send but NOT Sync.
unsafe impl Send for AfPacketSocket {}
```

- [ ] **Step 3: Implement AfPacketSocket::new()**

Constructor that:
1. Creates `socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL))`
2. Looks up interface index and MAC via `ioctl(SIOCGIFINDEX)` and `ioctl(SIOCGIFHWADDR)`
3. Binds to the interface via `bind()` with `sockaddr_ll`
4. Sets `PACKET_VERSION` to `TPACKET_V3`
5. Configures `PACKET_RX_RING` and `PACKET_TX_RING` via `setsockopt()`
6. `mmap()`s both rings
7. Attaches the BPF filter via `setsockopt(SO_ATTACH_FILTER)`

```rust
impl AfPacketSocket {
    pub fn new(interface_name: &str) -> Result<Self, RawLinkError> {
        // ... (implement the 7 steps above)
        // Return RawLinkError::PermissionDenied if socket() returns EPERM
        // Return RawLinkError::SocketError for other failures
        todo!("implement AF_PACKET socket setup")
    }
}
```

The BPF filter program (4 instructions):
```rust
fn bpf_filter_ethertype() -> [libc::sock_filter; 4] {
    [
        libc::sock_filter { code: 0x28, jt: 0, jf: 0, k: 12 },       // ldh [12]
        libc::sock_filter { code: 0x15, jt: 0, jf: 1, k: 0x88B5 },   // jeq #0x88B5, +0, +1
        libc::sock_filter { code: 0x06, jt: 0, jf: 0, k: 65535 },    // ret #65535 (accept)
        libc::sock_filter { code: 0x06, jt: 0, jf: 0, k: 0 },        // ret #0 (reject)
    ]
}
```

- [ ] **Step 4: Implement RawSocket for AfPacketSocket**

Implement `recv_frames()`:
1. `poll()` on the fd with a short timeout (10ms)
2. If ready, walk the current RX block:
   - Check block status header for `TP_STATUS_USER`
   - Iterate through `tpacket3_hdr` entries in the block
   - For each: extract src MAC (offset 6 in frame) and full frame bytes
   - Invoke the callback with `(src_mac, payload_after_eth_header)`
   - After processing all frames, mark block as `TP_STATUS_KERNEL`
   - Advance `rx_block_idx`

Implement `send_frame()`:
1. Build the full Ethernet frame (dst_mac + local_mac + EtherType + frame_type + payload)
2. Find next TX frame slot, write the frame
3. Mark as `TP_STATUS_SEND_REQUEST`
4. `sendto()` to flush

Implement `local_mac()`: return stored MAC.

**Note:** This involves unsafe pointer arithmetic on the mmap'd ring. Be very careful with alignment and bounds. Add comments explaining the safety invariants.

- [ ] **Step 5: Implement Drop for cleanup**

```rust
impl Drop for AfPacketSocket {
    fn drop(&mut self) {
        unsafe {
            if !self.rx_ring.is_null() {
                libc::munmap(self.rx_ring as *mut libc::c_void, self.rx_ring_size);
            }
            if !self.tx_ring.is_null() {
                libc::munmap(self.tx_ring as *mut libc::c_void, self.tx_ring_size);
            }
        }
        // OwnedFd handles close() on drop
    }
}
```

- [ ] **Step 6: Add module to lib.rs**

```rust
#[cfg(target_os = "linux")]
pub mod af_packet;
```

- [ ] **Step 7: Verify compilation on current platform**

```bash
cargo check -p harmony-rawlink
```

On macOS this will compile everything except `af_packet.rs` (gated by cfg). Verify no errors in the rest.

- [ ] **Step 8: Commit**

```bash
git add crates/harmony-rawlink/
git commit -m "feat(rawlink): add AfPacketSocket with TPACKET_V3 ring buffers and BPF filter"
```

---

### Task 6: Bridge (RawSocket ↔ Zenoh with SHM)

The async bridge task that connects a RawSocket to a zenoh session. Handles inbound frame processing, outbound subscription, scouting, and SHM buffer management.

**Files:**
- Create: `crates/harmony-rawlink/src/bridge.rs`
- Modify: `crates/harmony-rawlink/src/lib.rs` — add module, re-export Bridge

**Context:**
- The bridge runs as a tokio task (`bridge.run()` is an async fn)
- Inbound: recv_frames → parse → SHM alloc → zenoh publish
- Outbound: zenoh subscribe → encode → send_frame (broadcast)
- Scouting: periodic broadcast of identity_hash
- Zenoh SHM: `zenoh::shm` module — check the actual API in zenoh v1.7.2. The main types are likely `SharedMemoryProvider` or `ShmProvider`. Read the zenoh SHM documentation or source to find the correct API. If the SHM API is too complex or unstable, fall back to regular `ZBytes` and add SHM in a follow-up.

- [ ] **Step 1: Define Bridge struct**

```rust
//! Bridge between RawSocket and Zenoh session.

use crate::error::RawLinkError;
use crate::frame::{self, BROADCAST_MAC};
use crate::peer_table::PeerTable;
use crate::socket::RawSocket;
use crate::{frame_type, FRAME_OVERHEAD};

use std::time::{Duration, Instant};

/// Configuration for the rawlink bridge.
pub struct BridgeConfig {
    /// Identity hash of the local node (included in Scout frames).
    pub identity_hash: [u8; 16],
    /// Zenoh key expression pattern to subscribe for outbound bridging.
    pub subscribe_pattern: String,
    /// Scout broadcast interval range (uniform jitter).
    pub scout_interval: Duration,
    /// Peer table entry TTL.
    pub peer_ttl: Duration,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            identity_hash: [0; 16],
            subscribe_pattern: "harmony/**".to_string(),
            scout_interval: Duration::from_secs(5),
            peer_ttl: Duration::from_secs(30),
        }
    }
}

/// Bridges a RawSocket to a Zenoh session.
pub struct Bridge<S: RawSocket> {
    socket: S,
    session: zenoh::Session,
    config: BridgeConfig,
    peer_table: PeerTable,
}
```

- [ ] **Step 2: Implement Bridge::new() and Bridge::run()**

```rust
impl<S: RawSocket> Bridge<S> {
    pub fn new(
        socket: S,
        session: zenoh::Session,
        config: BridgeConfig,
    ) -> Self {
        let peer_table = PeerTable::new(config.peer_ttl);
        Self {
            socket,
            session,
            config,
            peer_table,
        }
    }

    /// Run the bridge loop. Returns on error or session close.
    pub async fn run(mut self) -> Result<(), RawLinkError> {
        // Subscribe to outbound zenoh traffic
        let subscriber = self.session
            .declare_subscriber(&self.config.subscribe_pattern)
            .await
            .map_err(|e| RawLinkError::SocketError(format!("zenoh subscribe failed: {e}")))?;

        let mut next_scout = Instant::now();
        let recv_interval = Duration::from_millis(10);

        loop {
            // 1. Scout if due
            if Instant::now() >= next_scout {
                self.send_scout()?;
                // Jitter: 5-10 seconds
                let jitter_ms = 5000 + (rand::random::<u64>() % 5000);
                next_scout = Instant::now() + Duration::from_millis(jitter_ms);
            }

            // 2. Receive inbound frames
            self.process_inbound_frames()?;

            // 3. Process outbound zenoh samples (non-blocking)
            while let Ok(sample) = subscriber.try_recv() {
                self.process_outbound_sample(&sample)?;
            }

            // 4. Yield to tokio
            tokio::time::sleep(recv_interval).await;
        }
    }

    fn send_scout(&mut self) -> Result<(), RawLinkError> {
        let frame = frame::encode_scout_frame(
            self.socket.local_mac(),
            &self.config.identity_hash,
        );
        // Send raw frame (bypass the RawSocket trait's payload-only interface)
        // Actually, the RawSocket trait sends payload only (dst_mac + payload).
        // But Scout needs the full Ethernet frame...
        //
        // DESIGN DECISION: The RawSocket trait should handle Ethernet header
        // construction internally. send_frame takes (dst_mac, payload_after_ethertype).
        // The implementation prepends src_mac + dst_mac + EtherType.
        //
        // So for Scout, send_frame(BROADCAST_MAC, &[SCOUT_TAG, identity_hash...])
        let mut payload = Vec::with_capacity(1 + 16);
        payload.push(frame_type::SCOUT);
        payload.extend_from_slice(&self.config.identity_hash);
        self.socket.send_frame(BROADCAST_MAC, &payload)
    }

    fn process_inbound_frames(&mut self) -> Result<(), RawLinkError> {
        // Destructure self to avoid partial borrow conflict — the closure
        // needs peer_table + session while socket.recv_frames needs &mut socket.
        let Self { socket, peer_table, session, .. } = self;

        socket.recv_frames(&mut |src_mac, payload| {
            if payload.is_empty() { return; }
            match payload[0] {
                frame_type::SCOUT => {
                    if payload.len() >= 17 {
                        let identity_hash: [u8; 16] = payload[1..17].try_into().unwrap();
                        peer_table.update(identity_hash, *src_mac);
                        tracing::debug!(
                            identity = %hex::encode(identity_hash),
                            mac = %format!("{:02X}:{:02X}:{:02X}:{:02X}:{:02X}:{:02X}",
                                src_mac[0], src_mac[1], src_mac[2],
                                src_mac[3], src_mac[4], src_mac[5]),
                            "scout received"
                        );
                    }
                }
                frame_type::DATA => {
                    if payload.len() >= 3 {
                        let key_len = u16::from_be_bytes([payload[1], payload[2]]) as usize;
                        if payload.len() >= 3 + key_len {
                            if let Ok(key_expr) = std::str::from_utf8(&payload[3..3 + key_len]) {
                                let data = &payload[3 + key_len..];
                                // Publish to zenoh (blocking in callback — consider
                                // buffering if this becomes a bottleneck)
                                let _ = session.put(key_expr, data.to_vec())
                                    .wait();
                            }
                        }
                    }
                }
                _ => {
                    // Unknown frame type — ignore (may be Reticulum or future types)
                }
            }
        })
    }

    fn process_outbound_sample(&mut self, sample: &zenoh::sample::Sample) -> Result<(), RawLinkError> {
        let key_expr = sample.key_expr().as_str();
        let payload = sample.payload().to_bytes();
        let key_bytes = key_expr.as_bytes();
        let key_len = (key_bytes.len() as u16).to_be_bytes();

        let mut frame_payload = Vec::with_capacity(1 + 2 + key_bytes.len() + payload.len());
        frame_payload.push(frame_type::DATA);
        frame_payload.extend_from_slice(&key_len);
        frame_payload.extend_from_slice(key_bytes);
        frame_payload.extend_from_slice(&payload);

        // All outbound is broadcast in this version
        self.socket.send_frame(BROADCAST_MAC, &frame_payload)
    }
}
```

**IMPORTANT NOTE on SHM:** The code above uses `session.put(key_expr, data.to_vec())` which does a memcpy. To use zenoh SHM:
1. Check if `zenoh::shm::SharedMemoryProvider` or equivalent exists in zenoh v1.7.2 with the `shared-memory` feature
2. If available, allocate SHM buffers and write frame data into them instead of `data.to_vec()`
3. If the SHM API is not readily available or too complex, use `data.to_vec()` and add SHM in a follow-up

The subagent implementing this task MUST check the zenoh SHM API and use it if feasible. The bridge structure supports either approach — only the `process_inbound_frames` data path changes.

- [ ] **Step 3: Write bridge tests using MockSocket**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::socket::MockSocket;

    #[test]
    fn scout_populates_peer_table() {
        // Create a mock socket pair
        let (mut sender, mut receiver) = MockSocket::pair([0xAA; 6], [0xBB; 6]);

        // Send a scout payload through sender
        let mut scout_payload = vec![frame_type::SCOUT];
        scout_payload.extend_from_slice(&[0xCC; 16]); // identity_hash
        sender.send_frame(BROADCAST_MAC, &scout_payload).unwrap();

        // Process on receiver side with a peer table
        let mut peer_table = PeerTable::new(Duration::from_secs(30));
        receiver.recv_frames(&mut |src_mac, payload| {
            if payload.len() >= 17 && payload[0] == frame_type::SCOUT {
                let hash: [u8; 16] = payload[1..17].try_into().unwrap();
                peer_table.update(hash, *src_mac);
            }
        }).unwrap();

        assert_eq!(peer_table.lookup(&[0xCC; 16]), Some([0xAA; 6]));
    }

    #[test]
    fn data_frame_payload_encoding() {
        let key_expr = "harmony/test";
        let payload = b"hello";
        let key_bytes = key_expr.as_bytes();
        let key_len = (key_bytes.len() as u16).to_be_bytes();

        let mut frame_payload = Vec::new();
        frame_payload.push(frame_type::DATA);
        frame_payload.extend_from_slice(&key_len);
        frame_payload.extend_from_slice(key_bytes);
        frame_payload.extend_from_slice(payload);

        // Decode
        assert_eq!(frame_payload[0], frame_type::DATA);
        let decoded_len = u16::from_be_bytes([frame_payload[1], frame_payload[2]]) as usize;
        let decoded_key = std::str::from_utf8(&frame_payload[3..3 + decoded_len]).unwrap();
        assert_eq!(decoded_key, key_expr);
        assert_eq!(&frame_payload[3 + decoded_len..], payload);
    }
}
```

- [ ] **Step 4: Add module to lib.rs, run tests, commit**

```bash
cargo test -p harmony-rawlink
git add crates/harmony-rawlink/
git commit -m "feat(rawlink): add Bridge connecting RawSocket to Zenoh"
```

---

### Task 7: Event loop integration and config

Wire the bridge into harmony-node with a config option.

**Files:**
- Modify: `crates/harmony-node/Cargo.toml` — add harmony-rawlink dependency
- Modify: `crates/harmony-node/src/config.rs` — add rawlink_interface field
- Modify: `crates/harmony-node/src/event_loop.rs` — spawn bridge at startup
- Modify: `crates/harmony-node/src/main.rs` — thread config

**Context:**
- `ConfigFile` is at config.rs:56-75, uses `#[serde(deny_unknown_fields)]`
- Event loop `run()` at event_loop.rs:187-197 takes many parameters
- Zenoh session created at event_loop.rs:235
- The bridge should only start on Linux when `rawlink_interface` is configured

- [ ] **Step 1: Add harmony-rawlink dependency**

In `crates/harmony-node/Cargo.toml`, add:
```toml
harmony-rawlink = { workspace = true, optional = true }
```

Add feature:
```toml
[features]
rawlink = ["harmony-rawlink"]
```

In root `Cargo.toml`, add to workspace dependencies:
```toml
harmony-rawlink = { path = "crates/harmony-rawlink" }
```

- [ ] **Step 2: Add config field**

In `config.rs`, add to `ConfigFile`:
```rust
pub rawlink_interface: Option<String>,
```

- [ ] **Step 3: Wire up in event_loop.rs**

Add `rawlink_interface: Option<String>` parameter to `run()`.

After the zenoh session is created (~line 235), add:
```rust
// ── Raw L2 bridge (Linux only, optional) ────────────────────────────
#[cfg(all(target_os = "linux", feature = "rawlink"))]
if let Some(ref iface) = rawlink_interface {
    match harmony_rawlink::af_packet::AfPacketSocket::new(iface) {
        Ok(socket) => {
            let bridge_config = harmony_rawlink::bridge::BridgeConfig {
                identity_hash: runtime.local_identity_hash(),
                ..Default::default()
            };
            let bridge = harmony_rawlink::bridge::Bridge::new(
                socket,
                session.clone(),
                bridge_config,
            );
            tokio::spawn(async move {
                if let Err(e) = bridge.run().await {
                    tracing::warn!(err = %e, "rawlink bridge stopped");
                }
            });
            tracing::info!(%iface, "rawlink AF_PACKET bridge started");
        }
        Err(e) => {
            tracing::warn!(interface = %iface, err = %e, "rawlink bridge failed to start — continuing without L2 transport");
        }
    }
}
```

**Note:** Check if `runtime.local_identity_hash()` or similar method exists to get the node's 16-byte identity hash. If not, derive it from the identity available in `TunnelConfig` or the runtime. Search for `identity_hash` usage in the event loop for the pattern.

- [ ] **Step 4: Thread config from main.rs**

Find where config fields are extracted and passed to `event_loop::run()` (~line 602 in main.rs). Add `rawlink_interface`:

```rust
let rawlink_interface = config_file.rawlink_interface.clone();
```

Pass as a new parameter to `event_loop::run()`.

- [ ] **Step 5: Run tests and compile**

```bash
cargo test -p harmony-node
cargo test -p harmony-rawlink
cargo clippy -p harmony-rawlink
```

- [ ] **Step 6: Commit**

```bash
git add crates/harmony-node/ crates/harmony-rawlink/ Cargo.toml
git commit -m "feat(node): integrate rawlink AF_PACKET bridge with config"
```

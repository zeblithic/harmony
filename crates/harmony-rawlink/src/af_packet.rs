//! AF_PACKET socket with TPACKET_V3 memory-mapped ring buffers and BPF filtering.
//!
//! This module provides a high-performance raw Ethernet socket implementation
//! for Linux using `AF_PACKET` with `TPACKET_V3` ring buffers. Incoming frames
//! are delivered to userspace via an `mmap`'d receive ring, avoiding per-packet
//! `recvfrom()` syscall overhead. A BPF filter is attached to accept only our
//! experimental EtherType (0x88B5), so the kernel drops irrelevant traffic
//! before it reaches the ring.
//!
//! # Safety
//!
//! This module contains significant `unsafe` code for:
//! - Direct syscalls via `libc` (socket, bind, ioctl, setsockopt, mmap, sendto, poll)
//! - Pointer arithmetic on the `mmap`'d ring buffer
//! - Manual struct layout assumptions matching Linux kernel headers
//!
//! All unsafe blocks include safety comments explaining the invariants relied upon.

use std::io;
use std::os::fd::OwnedFd;
use std::os::unix::io::{AsRawFd, FromRawFd};

use crate::error::RawLinkError;
use crate::socket::RawSocket;
use crate::{ETH_HEADER_LEN, HARMONY_ETHERTYPE};

// ---------------------------------------------------------------------------
// Ring buffer configuration
// ---------------------------------------------------------------------------

/// Block size for the TPACKET_V3 ring: 1 MiB.
const BLOCK_SIZE: u32 = 1 << 20; // 1 MiB

/// Number of blocks in each ring (RX and TX).
const BLOCK_NR: u32 = 4;

/// Frame (slot) size within a block.
const FRAME_SIZE: u32 = 2048;

/// Frames per block = block_size / frame_size.
const FRAMES_PER_BLOCK: u32 = BLOCK_SIZE / FRAME_SIZE;

/// Total ring size in bytes (for one direction).
const RING_SIZE: usize = (BLOCK_SIZE as usize) * (BLOCK_NR as usize);

/// Block retirement timeout in milliseconds.
/// If the block is not full within this period the kernel retires it anyway.
const BLOCK_RETIRE_TIMEOUT_MS: u32 = 100;

/// `poll()` timeout in milliseconds when waiting for RX data.
const POLL_TIMEOUT_MS: i32 = 10;

// ---------------------------------------------------------------------------
// BPF filter — accept only EtherType 0x88B5
// ---------------------------------------------------------------------------

/// Returns a 4-instruction BPF program that accepts only frames whose
/// EtherType field (bytes 12-13 of the Ethernet header) equals `0x88B5`.
///
/// ```text
/// ldh  [12]          ; load half-word at byte offset 12 (EtherType)
/// jeq  #0x88B5, 0, 1 ; if equal jump +0 (accept), else jump +1 (reject)
/// ret  #65535        ; accept — return max snaplen
/// ret  #0            ; reject — return 0 (drop frame)
/// ```
fn bpf_filter_ethertype() -> [libc::sock_filter; 4] {
    [
        libc::sock_filter {
            code: 0x28,
            jt: 0,
            jf: 0,
            k: 12,
        }, // BPF_LD | BPF_H | BPF_ABS — ldh [12]
        libc::sock_filter {
            code: 0x15,
            jt: 0,
            jf: 1,
            k: HARMONY_ETHERTYPE as u32,
        }, // BPF_JMP | BPF_JEQ | BPF_K — jeq #0x88B5
        libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: 65535,
        }, // BPF_RET | BPF_K — ret #65535
        libc::sock_filter {
            code: 0x06,
            jt: 0,
            jf: 0,
            k: 0,
        }, // BPF_RET | BPF_K — ret #0
    ]
}

// ---------------------------------------------------------------------------
// AfPacketSocket
// ---------------------------------------------------------------------------

/// A raw Ethernet socket using Linux `AF_PACKET` with `TPACKET_V3` ring buffers.
///
/// Frames are received via an `mmap`'d ring buffer and sent via `sendto()`.
/// A BPF filter ensures only EtherType `0x88B5` frames enter the ring.
///
/// # Invariants
///
/// - `rx_ring` points to a valid `mmap`'d region of `rx_ring_size` bytes.
/// - `tx_ring` points to a valid `mmap`'d region of `tx_ring_size` bytes,
///   located immediately after the RX ring in the same mapping.
/// - `fd` is a valid `AF_PACKET` socket bound to interface `if_index`.
/// - `rx_block_idx < BLOCK_NR` at all times.
pub struct AfPacketSocket {
    fd: OwnedFd,
    rx_ring: *mut u8,
    tx_ring: *mut u8,
    rx_ring_size: usize,
    tx_ring_size: usize,
    rx_block_idx: usize,
    #[allow(dead_code)]
    tx_frame_idx: usize,
    local_mac: [u8; 6],
    if_index: i32,
}

// SAFETY: The mmap'd ring pointers are only accessed through &mut self methods,
// so single-threaded access is guaranteed by Rust's borrow checker. The socket
// fd is process-global but safe to move between threads.
unsafe impl Send for AfPacketSocket {}

impl AfPacketSocket {
    /// Creates a new `AfPacketSocket` bound to the named network interface.
    ///
    /// This requires `CAP_NET_RAW` capability (typically root).
    ///
    /// # Steps
    ///
    /// 1. Open an `AF_PACKET / SOCK_RAW` socket
    /// 2. Look up interface index and MAC address via `ioctl`
    /// 3. Bind the socket to the interface
    /// 4. Set `TPACKET_V3` version
    /// 5. Configure RX and TX ring buffers
    /// 6. `mmap` both rings into userspace
    /// 7. Attach BPF filter for EtherType `0x88B5`
    ///
    /// # Errors
    ///
    /// Returns `RawLinkError::PermissionDenied` if the caller lacks `CAP_NET_RAW`.
    /// Returns `RawLinkError::SocketError` for interface lookup failures.
    /// Returns `RawLinkError::RingError` for ring buffer setup or mmap failures.
    pub fn new(interface_name: &str) -> Result<Self, RawLinkError> {
        // --- 1. Create AF_PACKET socket ---
        // SAFETY: socket() is a well-defined syscall. We check the return value.
        let raw_fd = unsafe {
            libc::socket(
                libc::AF_PACKET,
                libc::SOCK_RAW,
                (libc::ETH_P_ALL as u16).to_be() as i32,
            )
        };
        if raw_fd < 0 {
            return Err(io::Error::last_os_error().into());
        }

        // SAFETY: raw_fd is a valid file descriptor returned by socket().
        let fd = unsafe { OwnedFd::from_raw_fd(raw_fd) };

        // --- 2. Look up interface index ---
        let if_index = Self::get_interface_index(fd.as_raw_fd(), interface_name)?;

        // --- 3. Look up interface MAC address ---
        let local_mac = Self::get_interface_mac(fd.as_raw_fd(), interface_name)?;

        // --- 4. Bind to the interface ---
        Self::bind_to_interface(fd.as_raw_fd(), if_index)?;

        // --- 5. Set TPACKET_V3 ---
        Self::set_packet_version(fd.as_raw_fd())?;

        // --- 6. Configure RX ring ---
        let req = Self::make_ring_request();
        Self::set_ring(fd.as_raw_fd(), libc::PACKET_RX_RING, &req)?;

        // --- 7. Configure TX ring ---
        Self::set_ring(fd.as_raw_fd(), libc::PACKET_TX_RING, &req)?;

        // --- 8. mmap both rings ---
        let total_size = RING_SIZE * 2; // RX + TX
        // SAFETY: mmap() with valid fd, correct total_size, and MAP_SHARED | MAP_LOCKED.
        // We request PROT_READ | PROT_WRITE for both rings. The kernel validates all
        // parameters and returns MAP_FAILED on error.
        let map = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                total_size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED | libc::MAP_LOCKED,
                fd.as_raw_fd(),
                0,
            )
        };
        if map == libc::MAP_FAILED {
            return Err(RawLinkError::RingError(format!(
                "mmap failed: {}",
                io::Error::last_os_error()
            )));
        }

        let rx_ring = map as *mut u8;
        // SAFETY: tx_ring starts at offset RING_SIZE within the same contiguous mapping.
        // The total mapping is 2 * RING_SIZE bytes, so rx_ring + RING_SIZE is within bounds.
        let tx_ring = unsafe { rx_ring.add(RING_SIZE) };

        // --- 9. Attach BPF filter ---
        Self::attach_bpf_filter(fd.as_raw_fd())?;

        Ok(AfPacketSocket {
            fd,
            rx_ring,
            tx_ring,
            rx_ring_size: RING_SIZE,
            tx_ring_size: RING_SIZE,
            rx_block_idx: 0,
            tx_frame_idx: 0,
            local_mac,
            if_index,
        })
    }

    /// Copies the interface name into a zero-padded `IFNAMSIZ`-length byte array
    /// suitable for use in `ifreq.ifr_name`.
    fn copy_ifname(interface_name: &str) -> Result<[libc::c_char; libc::IFNAMSIZ], RawLinkError> {
        let name_bytes = interface_name.as_bytes();
        if name_bytes.len() >= libc::IFNAMSIZ {
            return Err(RawLinkError::SocketError(format!(
                "interface name too long: {} (max {})",
                interface_name,
                libc::IFNAMSIZ - 1,
            )));
        }
        let mut ifr_name = [0i8; libc::IFNAMSIZ];
        for (i, &b) in name_bytes.iter().enumerate() {
            ifr_name[i] = b as libc::c_char;
        }
        Ok(ifr_name)
    }

    /// Retrieves the interface index via `ioctl(SIOCGIFINDEX)`.
    fn get_interface_index(fd: i32, interface_name: &str) -> Result<i32, RawLinkError> {
        let ifr_name = Self::copy_ifname(interface_name)?;

        // SAFETY: We construct a zeroed ifreq, populate ifr_name, and call ioctl.
        // The kernel writes ifr_ifru.ifru_ifindex on success. We check the return value.
        unsafe {
            let mut ifr: libc::ifreq = std::mem::zeroed();
            ifr.ifr_name = ifr_name;

            let ret = libc::ioctl(fd, libc::SIOCGIFINDEX as _, &mut ifr);
            if ret < 0 {
                return Err(RawLinkError::SocketError(format!(
                    "SIOCGIFINDEX failed for '{}': {}",
                    interface_name,
                    io::Error::last_os_error(),
                )));
            }

            Ok(ifr.ifr_ifru.ifru_ifindex)
        }
    }

    /// Retrieves the interface MAC address via `ioctl(SIOCGIFHWADDR)`.
    fn get_interface_mac(fd: i32, interface_name: &str) -> Result<[u8; 6], RawLinkError> {
        let ifr_name = Self::copy_ifname(interface_name)?;

        // SAFETY: We construct a zeroed ifreq, populate ifr_name, and call ioctl.
        // The kernel writes ifr_ifru.ifru_hwaddr (a sockaddr) on success. The first
        // 6 bytes of sa_data contain the MAC address. We check the return value.
        unsafe {
            let mut ifr: libc::ifreq = std::mem::zeroed();
            ifr.ifr_name = ifr_name;

            let ret = libc::ioctl(fd, libc::SIOCGIFHWADDR as _, &mut ifr);
            if ret < 0 {
                return Err(RawLinkError::SocketError(format!(
                    "SIOCGIFHWADDR failed for '{}': {}",
                    interface_name,
                    io::Error::last_os_error(),
                )));
            }

            let mut mac = [0u8; 6];
            for (i, byte) in mac.iter_mut().enumerate() {
                *byte = ifr.ifr_ifru.ifru_hwaddr.sa_data[i] as u8;
            }
            Ok(mac)
        }
    }

    /// Binds the socket to a specific interface via `sockaddr_ll`.
    fn bind_to_interface(fd: i32, if_index: i32) -> Result<(), RawLinkError> {
        // SAFETY: We construct a zeroed sockaddr_ll, fill in the required fields,
        // and call bind(). The kernel validates the address family and ifindex.
        unsafe {
            let mut sll: libc::sockaddr_ll = std::mem::zeroed();
            sll.sll_family = libc::AF_PACKET as u16;
            sll.sll_protocol = (libc::ETH_P_ALL as u16).to_be();
            sll.sll_ifindex = if_index;

            let ret = libc::bind(
                fd,
                &sll as *const libc::sockaddr_ll as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
            );
            if ret < 0 {
                return Err(RawLinkError::SocketError(format!(
                    "bind failed: {}",
                    io::Error::last_os_error(),
                )));
            }
        }
        Ok(())
    }

    /// Sets the packet version to `TPACKET_V3` via `setsockopt(SOL_PACKET, PACKET_VERSION)`.
    fn set_packet_version(fd: i32) -> Result<(), RawLinkError> {
        let version: libc::c_int = libc::TPACKET_V3 as libc::c_int;

        // SAFETY: setsockopt with a valid integer option value. The kernel validates
        // the option level and name, returning -1 on error.
        unsafe {
            let ret = libc::setsockopt(
                fd,
                libc::SOL_PACKET,
                libc::PACKET_VERSION,
                &version as *const libc::c_int as *const libc::c_void,
                std::mem::size_of::<libc::c_int>() as libc::socklen_t,
            );
            if ret < 0 {
                return Err(RawLinkError::RingError(format!(
                    "setsockopt PACKET_VERSION failed: {}",
                    io::Error::last_os_error(),
                )));
            }
        }
        Ok(())
    }

    /// Builds a `tpacket_req3` with our ring configuration constants.
    fn make_ring_request() -> libc::tpacket_req3 {
        libc::tpacket_req3 {
            tp_block_size: BLOCK_SIZE,
            tp_block_nr: BLOCK_NR,
            tp_frame_size: FRAME_SIZE,
            tp_frame_nr: FRAMES_PER_BLOCK * BLOCK_NR,
            tp_retire_blk_tov: BLOCK_RETIRE_TIMEOUT_MS,
            tp_sizeof_priv: 0,
            tp_feature_req_word: 0,
        }
    }

    /// Configures a ring buffer (RX or TX) via `setsockopt`.
    fn set_ring(fd: i32, ring_type: libc::c_int, req: &libc::tpacket_req3) -> Result<(), RawLinkError> {
        // SAFETY: setsockopt with a tpacket_req3 struct. The kernel validates the struct
        // fields and returns -1 on error.
        unsafe {
            let ret = libc::setsockopt(
                fd,
                libc::SOL_PACKET,
                ring_type,
                req as *const libc::tpacket_req3 as *const libc::c_void,
                std::mem::size_of::<libc::tpacket_req3>() as libc::socklen_t,
            );
            if ret < 0 {
                let name = if ring_type == libc::PACKET_RX_RING {
                    "PACKET_RX_RING"
                } else {
                    "PACKET_TX_RING"
                };
                return Err(RawLinkError::RingError(format!(
                    "setsockopt {name} failed: {}",
                    io::Error::last_os_error(),
                )));
            }
        }
        Ok(())
    }

    /// Attaches a BPF filter that only accepts frames with EtherType `0x88B5`.
    fn attach_bpf_filter(fd: i32) -> Result<(), RawLinkError> {
        let mut filter = bpf_filter_ethertype();
        let prog = libc::sock_fprog {
            len: filter.len() as libc::c_ushort,
            filter: filter.as_mut_ptr(),
        };

        // SAFETY: setsockopt with SO_ATTACH_FILTER and a sock_fprog struct.
        // The kernel copies the BPF program and validates each instruction.
        // The filter array must outlive this call (it does — it's on the stack).
        unsafe {
            let ret = libc::setsockopt(
                fd,
                libc::SOL_SOCKET,
                libc::SO_ATTACH_FILTER,
                &prog as *const libc::sock_fprog as *const libc::c_void,
                std::mem::size_of::<libc::sock_fprog>() as libc::socklen_t,
            );
            if ret < 0 {
                return Err(RawLinkError::SocketError(format!(
                    "setsockopt SO_ATTACH_FILTER failed: {}",
                    io::Error::last_os_error(),
                )));
            }
        }
        Ok(())
    }

    /// Returns a pointer to the block descriptor at the given block index in the RX ring.
    ///
    /// # Safety
    ///
    /// `block_idx` must be < `BLOCK_NR`. The caller must ensure `self.rx_ring` is valid
    /// and the mmap'd region has not been unmapped.
    unsafe fn rx_block_desc(&self, block_idx: usize) -> *mut libc::tpacket_block_desc {
        debug_assert!(block_idx < BLOCK_NR as usize);
        // Each block starts at block_idx * BLOCK_SIZE within the RX ring.
        self.rx_ring
            .add(block_idx * BLOCK_SIZE as usize) as *mut libc::tpacket_block_desc
    }

    /// Builds a `sockaddr_ll` targeting the bound interface for `sendto()`.
    fn make_send_addr(&self, dst_mac: &[u8; 6]) -> libc::sockaddr_ll {
        // SAFETY: zeroing sockaddr_ll is safe — all fields are integers/arrays.
        let mut sll: libc::sockaddr_ll = unsafe { std::mem::zeroed() };
        sll.sll_family = libc::AF_PACKET as u16;
        sll.sll_protocol = HARMONY_ETHERTYPE.to_be();
        sll.sll_ifindex = self.if_index;
        sll.sll_halen = 6;
        sll.sll_addr[..6].copy_from_slice(dst_mac);
        sll
    }
}

impl RawSocket for AfPacketSocket {
    fn send_frame(&mut self, dst_mac: [u8; 6], payload: &[u8]) -> Result<(), RawLinkError> {
        // Build the full Ethernet frame: dst(6) + src(6) + EtherType(2) + payload
        let frame_len = ETH_HEADER_LEN + payload.len();
        let mut frame = Vec::with_capacity(frame_len);
        frame.extend_from_slice(&dst_mac);
        frame.extend_from_slice(&self.local_mac);
        frame.extend_from_slice(&HARMONY_ETHERTYPE.to_be_bytes());
        frame.extend_from_slice(payload);

        let sll = self.make_send_addr(&dst_mac);

        // SAFETY: sendto() with a valid fd, frame buffer, and sockaddr_ll.
        // The kernel validates all parameters. We check the return value.
        let sent = unsafe {
            libc::sendto(
                self.fd.as_raw_fd(),
                frame.as_ptr() as *const libc::c_void,
                frame.len(),
                0,
                &sll as *const libc::sockaddr_ll as *const libc::sockaddr,
                std::mem::size_of::<libc::sockaddr_ll>() as libc::socklen_t,
            )
        };
        if sent < 0 {
            return Err(io::Error::last_os_error().into());
        }
        if (sent as usize) != frame.len() {
            return Err(RawLinkError::SocketError(format!(
                "sendto short write: {sent} < {}",
                frame.len(),
            )));
        }
        Ok(())
    }

    fn recv_frames(
        &mut self,
        callback: &mut dyn FnMut(&[u8; 6], &[u8]),
    ) -> Result<(), RawLinkError> {
        // --- poll() for readability ---
        let mut pfd = libc::pollfd {
            fd: self.fd.as_raw_fd(),
            events: libc::POLLIN | libc::POLLERR,
            revents: 0,
        };

        // SAFETY: poll() with a valid pollfd struct and timeout. The kernel writes
        // revents on completion. We check the return value for errors.
        let poll_ret = unsafe { libc::poll(&mut pfd, 1, POLL_TIMEOUT_MS) };
        if poll_ret < 0 {
            let err = io::Error::last_os_error();
            // EINTR is not a real error — just retry next time.
            if err.raw_os_error() == Some(libc::EINTR) {
                return Ok(());
            }
            return Err(err.into());
        }
        if poll_ret == 0 {
            // Timeout — no data available.
            return Ok(());
        }

        // --- Walk available RX blocks ---
        loop {
            // SAFETY: rx_block_idx is always < BLOCK_NR (maintained by modular arithmetic
            // below). self.rx_ring is valid because we only enter this path after a
            // successful mmap, and we munmap in Drop.
            let block_desc = unsafe { self.rx_block_desc(self.rx_block_idx) };

            // SAFETY: block_desc points into a valid mmap'd region. Reading bh1 from
            // the tpacket_bd_header_u union is safe because the kernel always fills in
            // the bh1 variant for TPACKET_V3.
            let hdr_v1 = unsafe { &(*block_desc).hdr.bh1 };
            let block_status = unsafe {
                // Use a volatile read to prevent the compiler from caching the status
                // value, since the kernel writes it asynchronously.
                std::ptr::read_volatile(&hdr_v1.block_status)
            };

            if (block_status & libc::TP_STATUS_USER) == 0 {
                // This block is still owned by the kernel — no more data.
                break;
            }

            let num_pkts = hdr_v1.num_pkts;
            let mut pkt_offset = hdr_v1.offset_to_first_pkt as usize;

            for _ in 0..num_pkts {
                // SAFETY: pkt_offset is within the current block (block_desc is at the
                // start of a BLOCK_SIZE region, and the kernel guarantees
                // offset_to_first_pkt and tp_next_offset stay within bounds).
                let pkt_ptr = unsafe {
                    (block_desc as *const u8).add(pkt_offset) as *const libc::tpacket3_hdr
                };

                // SAFETY: pkt_ptr points to a valid tpacket3_hdr within the mmap'd block.
                let tp_hdr = unsafe { &*pkt_ptr };
                let tp_mac = tp_hdr.tp_mac as usize;
                let tp_snaplen = tp_hdr.tp_snaplen as usize;

                // The Ethernet frame starts at pkt_ptr + tp_mac.
                // SAFETY: tp_mac is the offset from the tpacket3_hdr to the Ethernet
                // frame data, as set by the kernel. The frame data (tp_snaplen bytes)
                // is within the mmap'd block.
                let frame_ptr = unsafe { (pkt_ptr as *const u8).add(tp_mac) };
                let frame = unsafe { std::slice::from_raw_parts(frame_ptr, tp_snaplen) };

                // The frame is a full Ethernet frame. Extract src_mac (bytes 6..12)
                // and the payload after the Ethernet header.
                if frame.len() >= ETH_HEADER_LEN {
                    let mut src_mac = [0u8; 6];
                    src_mac.copy_from_slice(&frame[6..12]);
                    let payload = &frame[ETH_HEADER_LEN..];
                    callback(&src_mac, payload);
                }

                // Advance to the next packet in this block.
                let next_offset = tp_hdr.tp_next_offset as usize;
                if next_offset == 0 {
                    // Last packet in the block.
                    break;
                }
                pkt_offset += next_offset;
            }

            // Return this block to the kernel.
            // SAFETY: We're writing to the block_status field of a block_desc that we
            // verified is in TP_STATUS_USER state. The volatile write ensures the kernel
            // sees the update. Setting to TP_STATUS_KERNEL hands the block back.
            unsafe {
                std::ptr::write_volatile(
                    &mut (*block_desc).hdr.bh1.block_status,
                    libc::TP_STATUS_KERNEL,
                );
            }

            // Advance to the next block (wrapping around).
            self.rx_block_idx = (self.rx_block_idx + 1) % (BLOCK_NR as usize);
        }

        Ok(())
    }

    fn local_mac(&self) -> [u8; 6] {
        self.local_mac
    }
}

impl Drop for AfPacketSocket {
    fn drop(&mut self) {
        // SAFETY: rx_ring was returned by a successful mmap() call with total size
        // rx_ring_size + tx_ring_size. The mapping is contiguous (RX then TX), so we
        // munmap the full region starting at rx_ring. OwnedFd handles close().
        unsafe {
            let total_size = self.rx_ring_size + self.tx_ring_size;
            libc::munmap(self.rx_ring as *mut libc::c_void, total_size);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpf_filter_has_four_instructions() {
        let filter = bpf_filter_ethertype();
        assert_eq!(filter.len(), 4);
    }

    #[test]
    fn bpf_filter_checks_correct_offset() {
        let filter = bpf_filter_ethertype();
        // First instruction loads half-word at offset 12 (EtherType field)
        assert_eq!(filter[0].k, 12);
    }

    #[test]
    fn bpf_filter_checks_harmony_ethertype() {
        let filter = bpf_filter_ethertype();
        // Second instruction compares against 0x88B5
        assert_eq!(filter[1].k, 0x88B5);
    }

    #[test]
    fn ring_config_consistency() {
        assert_eq!(FRAMES_PER_BLOCK, BLOCK_SIZE / FRAME_SIZE);
        assert_eq!(RING_SIZE, (BLOCK_SIZE as usize) * (BLOCK_NR as usize));
    }

    #[test]
    fn ring_request_fields() {
        let req = AfPacketSocket::make_ring_request();
        assert_eq!(req.tp_block_size, BLOCK_SIZE);
        assert_eq!(req.tp_block_nr, BLOCK_NR);
        assert_eq!(req.tp_frame_size, FRAME_SIZE);
        assert_eq!(req.tp_frame_nr, FRAMES_PER_BLOCK * BLOCK_NR);
        assert_eq!(req.tp_retire_blk_tov, BLOCK_RETIRE_TIMEOUT_MS);
        assert_eq!(req.tp_sizeof_priv, 0);
        assert_eq!(req.tp_feature_req_word, 0);
    }
}

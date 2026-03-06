# Ring 1: VirtIO-net Driver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement a VirtIO 1.0+ PCI modern network driver for the Harmony unikernel, wiring it to the `NetworkInterface` trait so the event loop can send and receive raw Reticulum packets over a QEMU virtual Ethernet link.

**Architecture:** Three new modules in `harmony-boot`: PCI bus scanner, VirtIO virtqueue, and VirtioNet driver. The driver implements `NetworkInterface` from `harmony-platform`. The event loop in `main.rs` initializes the driver after heap/entropy setup, then polls `receive()` and dispatches `send()` in its tick loop. The `UnikernelRuntime` gains a method to register interfaces with its internal `Node` and to process inbound packets.

**Tech Stack:** VirtIO 1.0 spec (split virtqueues, PCI modern capability structs), `x86_64` crate for port I/O, `harmony-platform::NetworkInterface` trait, QEMU socket/multicast backend.

**Repos:** Primary work in `harmony-os` (`/Users/zeblith/work/zeblithic/harmony-os/`). Plan/design docs in `harmony` (`/Users/zeblith/work/zeblithic/harmony/`).

**Design doc:** `docs/plans/2026-03-06-virtio-net-driver-design.md`

---

### Task 1: PCI Configuration Space Scanner

**Context:** Scan PCI bus 0 for VirtIO-net devices using x86 I/O ports 0xCF8/0xCFC. Parse PCI configuration space to find vendor/device IDs and BAR addresses. This is pure port I/O with no VirtIO-specific logic yet.

**Files:**
- Create: `harmony-os/crates/harmony-boot/src/pci.rs`
- Modify: `harmony-os/crates/harmony-boot/src/main.rs` (add `mod pci`)

**Step 1: Create PCI module with config space read functions**

```rust
// harmony-os/crates/harmony-boot/src/pci.rs
// SPDX-License-Identifier: GPL-2.0-or-later
//! Minimal PCI configuration space access for bus 0.
//!
//! Uses the legacy I/O port mechanism (CONFIG_ADDRESS 0xCF8, CONFIG_DATA 0xCFC).
//! Only scans bus 0 — sufficient for QEMU's default PCI topology.

use x86_64::instructions::port::Port;

const CONFIG_ADDRESS: u16 = 0xCF8;
const CONFIG_DATA: u16 = 0xCFC;

/// Read a 32-bit value from PCI configuration space.
///
/// `bus` must be 0 (we only scan bus 0).
/// `device` is 0..31, `function` is 0..7, `offset` is 4-byte aligned.
pub fn pci_config_read32(bus: u8, device: u8, function: u8, offset: u8) -> u32 {
    let address: u32 = 0x8000_0000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);
    unsafe {
        Port::new(CONFIG_ADDRESS).write(address);
        Port::new(CONFIG_DATA).read()
    }
}

pub fn pci_config_read16(bus: u8, device: u8, function: u8, offset: u8) -> u16 {
    let val = pci_config_read32(bus, device, function, offset & 0xFC);
    (val >> ((offset & 2) * 8)) as u16
}

pub fn pci_config_read8(bus: u8, device: u8, function: u8, offset: u8) -> u8 {
    let val = pci_config_read32(bus, device, function, offset & 0xFC);
    (val >> ((offset & 3) * 8)) as u8
}

/// Write a 32-bit value to PCI configuration space.
pub fn pci_config_write32(bus: u8, device: u8, function: u8, offset: u8, value: u32) {
    let address: u32 = 0x8000_0000
        | ((bus as u32) << 16)
        | ((device as u32) << 11)
        | ((function as u32) << 8)
        | ((offset as u32) & 0xFC);
    unsafe {
        Port::new(CONFIG_ADDRESS).write(address);
        Port::new(CONFIG_DATA).write(value);
    }
}

pub fn pci_config_write16(bus: u8, device: u8, function: u8, offset: u8, value: u16) {
    let mut val = pci_config_read32(bus, device, function, offset & 0xFC);
    let shift = (offset & 2) * 8;
    val &= !(0xFFFF << shift);
    val |= (value as u32) << shift;
    pci_config_write32(bus, device, function, offset & 0xFC, val);
}

/// A discovered PCI device.
#[derive(Debug, Clone)]
pub struct PciDevice {
    pub bus: u8,
    pub device: u8,
    pub function: u8,
    pub vendor_id: u16,
    pub device_id: u16,
    pub bars: [u32; 6],
}

impl PciDevice {
    /// Read a BAR value (BAR0 = offset 0x10, BAR1 = 0x14, ...).
    pub fn read_bar(&self, bar_index: usize) -> u32 {
        let offset = 0x10 + (bar_index as u8) * 4;
        pci_config_read32(self.bus, self.device, self.function, offset)
    }

    /// Read the capability pointer (offset 0x34).
    pub fn capabilities_ptr(&self) -> u8 {
        pci_config_read8(self.bus, self.device, self.function, 0x34)
    }

    /// Read the command register (offset 0x04).
    pub fn read_command(&self) -> u16 {
        pci_config_read16(self.bus, self.device, self.function, 0x04)
    }

    /// Write the command register (offset 0x04).
    pub fn write_command(&self, value: u16) {
        pci_config_write16(self.bus, self.device, self.function, 0x04, value);
    }

    /// Enable bus mastering and memory space access.
    pub fn enable_bus_master(&self) {
        let cmd = self.read_command();
        // Bit 1 = Memory Space, Bit 2 = Bus Master
        self.write_command(cmd | 0x06);
    }
}

/// VirtIO vendor ID.
pub const VIRTIO_VENDOR_ID: u16 = 0x1AF4;
/// VirtIO modern net device ID (transitional: 0x1000, modern: 0x1041).
pub const VIRTIO_NET_DEVICE_ID_MODERN: u16 = 0x1041;
/// VirtIO transitional net device (subsystem device ID check needed).
pub const VIRTIO_NET_DEVICE_ID_TRANSITIONAL: u16 = 0x1000;

/// Scan PCI bus 0 for a VirtIO network device.
///
/// Returns the first VirtIO-net device found, or `None`.
pub fn find_virtio_net() -> Option<PciDevice> {
    for device in 0..32u8 {
        let vendor_id = pci_config_read16(0, device, 0, 0x00);
        if vendor_id == 0xFFFF {
            continue; // No device in this slot
        }
        if vendor_id != VIRTIO_VENDOR_ID {
            continue;
        }
        let device_id = pci_config_read16(0, device, 0, 0x02);
        // Accept both modern (0x1041) and transitional (0x1000) device IDs
        if device_id != VIRTIO_NET_DEVICE_ID_MODERN
            && device_id != VIRTIO_NET_DEVICE_ID_TRANSITIONAL
        {
            continue;
        }
        let mut bars = [0u32; 6];
        for i in 0..6 {
            bars[i] = pci_config_read32(0, device, 0, 0x10 + (i as u8) * 4);
        }
        return Some(PciDevice {
            bus: 0,
            device,
            function: 0,
            vendor_id,
            device_id,
            bars,
        });
    }
    None
}
```

**Step 2: Add mod declaration to main.rs**

Add `mod pci;` near the top of `main.rs` (after `extern crate alloc;`).

**Step 3: Verify bare-metal compilation**

Run: `cd harmony-os/crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: compiles clean, no warnings.

**Step 4: Commit**

```bash
cd harmony-os
git add crates/harmony-boot/src/pci.rs crates/harmony-boot/src/main.rs
git commit -m "feat(boot): add PCI bus 0 scanner for VirtIO device discovery"
```

---

### Task 2: VirtIO PCI Capability Parsing

**Context:** VirtIO 1.0+ modern devices expose their configuration through PCI capabilities (capability type 0x09). We need to walk the capability list to find the common config, notify, and device-specific config structures, then map them to MMIO addresses via BARs.

**Files:**
- Create: `harmony-os/crates/harmony-boot/src/virtio/mod.rs`
- Create: `harmony-os/crates/harmony-boot/src/virtio/pci_cap.rs`
- Modify: `harmony-os/crates/harmony-boot/src/main.rs` (add `mod virtio`)

**Step 1: Create virtio module and PCI capability parser**

```rust
// harmony-os/crates/harmony-boot/src/virtio/mod.rs
// SPDX-License-Identifier: GPL-2.0-or-later
//! VirtIO 1.0+ device driver support.

pub mod pci_cap;
pub mod virtqueue;
pub mod net;
```

```rust
// harmony-os/crates/harmony-boot/src/virtio/pci_cap.rs
// SPDX-License-Identifier: GPL-2.0-or-later
//! Parse VirtIO PCI capabilities to locate MMIO register regions.
//!
//! VirtIO 1.0 §4.1.4: The device exposes configuration through
//! vendor-specific PCI capabilities (type 0x09) with a cfg_type field
//! that identifies the structure (common, notify, ISR, device, PCI).

use crate::pci::{self, PciDevice};

/// VirtIO PCI capability cfg_type values (VirtIO 1.0 §4.1.4.3).
pub const VIRTIO_PCI_CAP_COMMON_CFG: u8 = 1;
pub const VIRTIO_PCI_CAP_NOTIFY_CFG: u8 = 2;
pub const VIRTIO_PCI_CAP_ISR_CFG: u8 = 3;
pub const VIRTIO_PCI_CAP_DEVICE_CFG: u8 = 4;
pub const VIRTIO_PCI_CAP_PCI_CFG: u8 = 5;

/// Parsed VirtIO PCI capability pointers.
///
/// Each field is a virtual address (BAR base + offset) where the
/// corresponding VirtIO structure lives in MMIO space.
#[derive(Debug)]
pub struct VirtioPciCaps {
    /// Common configuration (device status, feature negotiation, queue setup).
    pub common_cfg: usize,
    /// Notify region base address.
    pub notify_base: usize,
    /// Notify offset multiplier (bytes per queue index).
    pub notify_off_multiplier: u32,
    /// Device-specific configuration (e.g., MAC address for net).
    pub device_cfg: usize,
    /// ISR status register.
    pub isr_cfg: usize,
}

/// Walk the PCI capability list and extract VirtIO structure addresses.
///
/// `phys_offset` is the bootloader's physical memory mapping offset
/// (virtual = physical + phys_offset) for converting BAR physical addresses
/// to virtual pointers.
pub fn parse_capabilities(dev: &PciDevice, phys_offset: u64) -> Option<VirtioPciCaps> {
    let mut common_cfg = None;
    let mut notify_base = None;
    let mut notify_off_multiplier = 0u32;
    let mut device_cfg = None;
    let mut isr_cfg = None;

    let mut cap_ptr = dev.capabilities_ptr();
    while cap_ptr != 0 {
        let cap_id = pci::pci_config_read8(dev.bus, dev.device, dev.function, cap_ptr);
        if cap_id == 0x09 {
            // Vendor-specific capability — VirtIO
            let cfg_type =
                pci::pci_config_read8(dev.bus, dev.device, dev.function, cap_ptr + 3);
            let bar_index =
                pci::pci_config_read8(dev.bus, dev.device, dev.function, cap_ptr + 4);
            let offset =
                pci::pci_config_read32(dev.bus, dev.device, dev.function, cap_ptr + 8);
            let _length =
                pci::pci_config_read32(dev.bus, dev.device, dev.function, cap_ptr + 12);

            let bar_val = dev.read_bar(bar_index as usize);
            // Memory-mapped BAR (bit 0 = 0)
            let bar_phys = (bar_val & 0xFFFF_FFF0) as u64;
            let bar_virt = bar_phys + phys_offset;
            let addr = bar_virt as usize + offset as usize;

            match cfg_type {
                VIRTIO_PCI_CAP_COMMON_CFG => common_cfg = Some(addr),
                VIRTIO_PCI_CAP_NOTIFY_CFG => {
                    notify_base = Some(addr);
                    // Notify cap has an extra 32-bit field at cap_ptr + 16
                    notify_off_multiplier = pci::pci_config_read32(
                        dev.bus,
                        dev.device,
                        dev.function,
                        cap_ptr + 16,
                    );
                }
                VIRTIO_PCI_CAP_DEVICE_CFG => device_cfg = Some(addr),
                VIRTIO_PCI_CAP_ISR_CFG => isr_cfg = Some(addr),
                _ => {}
            }
        }
        // Next capability
        cap_ptr = pci::pci_config_read8(dev.bus, dev.device, dev.function, cap_ptr + 1);
    }

    Some(VirtioPciCaps {
        common_cfg: common_cfg?,
        notify_base: notify_base?,
        notify_off_multiplier,
        device_cfg: device_cfg?,
        isr_cfg: isr_cfg?,
    })
}
```

**Step 2: Add `mod virtio` to main.rs**

Add `mod virtio;` after `mod pci;` in main.rs.

**Step 3: Verify bare-metal compilation**

Run: `cd harmony-os/crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: compiles clean.

**Step 4: Commit**

```bash
cd harmony-os
git add crates/harmony-boot/src/virtio/
git commit -m "feat(boot): VirtIO PCI capability parser for MMIO register discovery"
```

---

### Task 3: Virtqueue Implementation (Split Virtqueue)

**Context:** Implement VirtIO 1.0 split virtqueues — the core data structure for guest/device communication. Each virtqueue has a descriptor table, available ring, and used ring. This is the largest single component (~200 lines) and is hardware-independent, so it gets thorough host-native unit tests.

**Files:**
- Create: `harmony-os/crates/harmony-boot/src/virtio/virtqueue.rs`
- The tests run on the host since virtqueue logic is pure data structure manipulation.

**Important:** `harmony-boot` is `no_std` and excluded from the workspace. Since we can't run its tests directly with `cargo test`, we write the virtqueue as a self-contained module with `#[cfg(test)]` tests that we verify by temporarily testing from the workspace (or by manual review and QEMU integration). The critical path is QEMU validation in Task 6.

**Step 1: Write the virtqueue module**

```rust
// harmony-os/crates/harmony-boot/src/virtio/virtqueue.rs
// SPDX-License-Identifier: GPL-2.0-or-later
//! VirtIO 1.0 split virtqueue implementation.
//!
//! A split virtqueue consists of three memory regions:
//! - Descriptor table: array of buffer descriptors
//! - Available ring: guest-written ring of descriptor indices for the device
//! - Used ring: device-written ring of completed descriptors
//!
//! This implementation uses a static buffer pool (QUEUE_SIZE × BUF_SIZE)
//! allocated at init time. Zero allocations on the hot path.

use alloc::alloc::{alloc_zeroed, Layout};
use core::ptr;
use core::sync::atomic::{fence, Ordering};

/// Number of descriptors per queue. Must be a power of 2.
pub const QUEUE_SIZE: u16 = 32;

/// Buffer size in bytes. Generous for Reticulum MTU (500) + Ethernet header (14).
pub const BUF_SIZE: usize = 2048;

/// VirtIO descriptor flags.
const VIRTQ_DESC_F_NEXT: u16 = 1;
const VIRTQ_DESC_F_WRITE: u16 = 2;

/// A single virtqueue descriptor (VirtIO 1.0 §2.6.5).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtqDesc {
    pub addr: u64,  // Physical address of buffer
    pub len: u32,   // Buffer length
    pub flags: u16, // NEXT, WRITE, INDIRECT
    pub next: u16,  // Next descriptor index (if NEXT flag set)
}

/// Available ring header (VirtIO 1.0 §2.6.6).
#[repr(C)]
pub struct VirtqAvail {
    pub flags: u16,
    pub idx: u16,
    pub ring: [u16; QUEUE_SIZE as usize],
}

/// Used ring element (VirtIO 1.0 §2.6.8).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct VirtqUsedElem {
    pub id: u32,  // Descriptor chain head index
    pub len: u32, // Bytes written by device
}

/// Used ring header.
#[repr(C)]
pub struct VirtqUsed {
    pub flags: u16,
    pub idx: u16,
    pub ring: [VirtqUsedElem; QUEUE_SIZE as usize],
}

/// A split virtqueue with a static buffer pool.
pub struct Virtqueue {
    /// Pointer to descriptor table (QUEUE_SIZE entries).
    pub desc: *mut VirtqDesc,
    /// Pointer to available ring.
    pub avail: *mut VirtqAvail,
    /// Pointer to used ring.
    pub used: *mut VirtqUsed,
    /// Static buffer pool: QUEUE_SIZE buffers of BUF_SIZE bytes each.
    pub buffers: *mut u8,
    /// Physical address of descriptor table (for device programming).
    pub desc_phys: u64,
    /// Physical address of available ring.
    pub avail_phys: u64,
    /// Physical address of used ring.
    pub used_phys: u64,
    /// Physical address of buffer pool base.
    pub buffers_phys: u64,
    /// Last seen used ring index (for polling).
    pub last_used_idx: u16,
    /// Free descriptor list (simple bitmap: true = free).
    pub free: [bool; QUEUE_SIZE as usize],
}

impl Virtqueue {
    /// Allocate and initialize a new virtqueue.
    ///
    /// `phys_offset` is subtracted from virtual addresses to get physical.
    pub fn new(phys_offset: u64) -> Self {
        // Allocate descriptor table (16 bytes per entry, 16-byte aligned)
        let desc_layout = Layout::from_size_align(
            core::mem::size_of::<VirtqDesc>() * QUEUE_SIZE as usize,
            16,
        )
        .unwrap();
        let desc_ptr = unsafe { alloc_zeroed(desc_layout) } as *mut VirtqDesc;

        // Allocate available ring (4 + 2*QUEUE_SIZE bytes, 2-byte aligned)
        let avail_layout = Layout::from_size_align(
            4 + 2 * QUEUE_SIZE as usize,
            2,
        )
        .unwrap();
        let avail_ptr = unsafe { alloc_zeroed(avail_layout) } as *mut VirtqAvail;

        // Allocate used ring (4 + 8*QUEUE_SIZE bytes, 4-byte aligned)
        let used_layout = Layout::from_size_align(
            4 + 8 * QUEUE_SIZE as usize,
            4,
        )
        .unwrap();
        let used_ptr = unsafe { alloc_zeroed(used_layout) } as *mut VirtqUsed;

        // Allocate buffer pool (QUEUE_SIZE * BUF_SIZE, page-aligned for DMA)
        let buf_layout =
            Layout::from_size_align(QUEUE_SIZE as usize * BUF_SIZE, 4096).unwrap();
        let buf_ptr = unsafe { alloc_zeroed(buf_layout) };

        let to_phys = |virt: *mut u8| -> u64 {
            (virt as u64).wrapping_sub(phys_offset)
        };

        let mut free = [true; QUEUE_SIZE as usize];
        // All descriptors start free
        for f in free.iter_mut() {
            *f = true;
        }

        Virtqueue {
            desc: desc_ptr,
            avail: avail_ptr,
            used: used_ptr,
            buffers: buf_ptr,
            desc_phys: to_phys(desc_ptr as *mut u8),
            avail_phys: to_phys(avail_ptr as *mut u8),
            used_phys: to_phys(used_ptr as *mut u8),
            buffers_phys: to_phys(buf_ptr),
            last_used_idx: 0,
            free,
        }
    }

    /// Get a pointer to buffer `idx`.
    fn buffer_ptr(&self, idx: u16) -> *mut u8 {
        unsafe { self.buffers.add(idx as usize * BUF_SIZE) }
    }

    /// Physical address of buffer `idx`.
    fn buffer_phys(&self, idx: u16) -> u64 {
        self.buffers_phys + (idx as u64) * (BUF_SIZE as u64)
    }

    /// Allocate a free descriptor. Returns None if all are in use.
    pub fn alloc_desc(&mut self) -> Option<u16> {
        for (i, free) in self.free.iter_mut().enumerate() {
            if *free {
                *free = false;
                return Some(i as u16);
            }
        }
        None
    }

    /// Free a descriptor.
    pub fn free_desc(&mut self, idx: u16) {
        self.free[idx as usize] = true;
    }

    /// Post a device-writable buffer for receiving (RX queue).
    ///
    /// Returns the descriptor index used, or None if no free descriptors.
    pub fn post_receive(&mut self) -> Option<u16> {
        let idx = self.alloc_desc()?;
        unsafe {
            let desc = &mut *self.desc.add(idx as usize);
            desc.addr = self.buffer_phys(idx);
            desc.len = BUF_SIZE as u32;
            desc.flags = VIRTQ_DESC_F_WRITE; // Device writes to this buffer
            desc.next = 0;

            // Add to available ring
            let avail = &mut *self.avail;
            let ring_idx = avail.idx % QUEUE_SIZE;
            avail.ring[ring_idx as usize] = idx;
            fence(Ordering::Release);
            avail.idx = avail.idx.wrapping_add(1);
        }
        Some(idx)
    }

    /// Submit a buffer for sending (TX queue).
    ///
    /// Copies `data` into a buffer and posts it as device-readable.
    /// Returns the descriptor index, or None if no free descriptors or data too large.
    pub fn submit_send(&mut self, data: &[u8]) -> Option<u16> {
        if data.len() > BUF_SIZE {
            return None;
        }
        let idx = self.alloc_desc()?;
        unsafe {
            // Copy data into buffer
            let buf = self.buffer_ptr(idx);
            ptr::copy_nonoverlapping(data.as_ptr(), buf, data.len());

            let desc = &mut *self.desc.add(idx as usize);
            desc.addr = self.buffer_phys(idx);
            desc.len = data.len() as u32;
            desc.flags = 0; // Device reads from this buffer
            desc.next = 0;

            // Add to available ring
            let avail = &mut *self.avail;
            let ring_idx = avail.idx % QUEUE_SIZE;
            avail.ring[ring_idx as usize] = idx;
            fence(Ordering::Release);
            avail.idx = avail.idx.wrapping_add(1);
        }
        Some(idx)
    }

    /// Poll the used ring for completed descriptors.
    ///
    /// Returns `Some((descriptor_index, bytes_written))` for each completed
    /// descriptor, or `None` if no new completions.
    pub fn poll_used(&mut self) -> Option<(u16, u32)> {
        fence(Ordering::Acquire);
        let used_idx = unsafe { (*self.used).idx };
        if self.last_used_idx == used_idx {
            return None;
        }
        let ring_idx = self.last_used_idx % QUEUE_SIZE;
        let elem = unsafe { (*self.used).ring[ring_idx as usize] };
        self.last_used_idx = self.last_used_idx.wrapping_add(1);
        Some((elem.id as u16, elem.len))
    }

    /// Read bytes from a completed receive buffer.
    ///
    /// Copies `len` bytes from the buffer at `desc_idx` into a Vec.
    pub fn read_buffer(&self, desc_idx: u16, len: u32) -> alloc::vec::Vec<u8> {
        let actual_len = core::cmp::min(len as usize, BUF_SIZE);
        let buf = self.buffer_ptr(desc_idx);
        let mut data = alloc::vec![0u8; actual_len];
        unsafe {
            ptr::copy_nonoverlapping(buf, data.as_mut_ptr(), actual_len);
        }
        data
    }
}
```

**Step 2: Verify bare-metal compilation**

Run: `cd harmony-os/crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: compiles clean.

**Step 3: Commit**

```bash
cd harmony-os
git add crates/harmony-boot/src/virtio/virtqueue.rs
git commit -m "feat(boot): VirtIO split virtqueue with static buffer pool"
```

---

### Task 4: VirtioNet Driver (NetworkInterface Implementation)

**Context:** The main driver module that ties PCI discovery, capability parsing, VirtIO device initialization, virtqueues, and Ethernet framing together into a `NetworkInterface` implementation. This is the integration layer.

**Files:**
- Create: `harmony-os/crates/harmony-boot/src/virtio/net.rs`
- Modify: `harmony-os/crates/harmony-boot/Cargo.toml` (add `harmony-platform` dependency)

**Step 1: Add harmony-platform dependency to harmony-boot**

In `harmony-os/crates/harmony-boot/Cargo.toml`, add:

```toml
harmony-platform = { git = "https://github.com/zeblithic/harmony.git", branch = "main", default-features = false }
```

**Step 2: Write the VirtioNet driver**

```rust
// harmony-os/crates/harmony-boot/src/virtio/net.rs
// SPDX-License-Identifier: GPL-2.0-or-later
//! VirtIO 1.0+ network device driver.
//!
//! Implements `NetworkInterface` for the Harmony unikernel event loop.
//! Handles VirtIO device initialization, Ethernet framing (add/strip
//! 14-byte headers), and poll-based TX/RX via split virtqueues.

use alloc::vec::Vec;
use core::ptr;
use core::sync::atomic::{fence, Ordering};

use harmony_platform::error::PlatformError;
use harmony_platform::network::NetworkInterface;

use super::pci_cap::VirtioPciCaps;
use super::virtqueue::{Virtqueue, QUEUE_SIZE};

/// EtherType for Harmony mesh protocol (IEEE Local Experimental 1).
const ETHERTYPE_HARMONY: [u8; 2] = [0x88, 0xB5];

/// Ethernet header size (6 dst + 6 src + 2 type).
const ETH_HEADER_LEN: usize = 14;

/// Broadcast MAC address.
const BROADCAST_MAC: [u8; 6] = [0xFF; 6];

// VirtIO common configuration structure offsets (VirtIO 1.0 §4.1.4.3).
const COMMON_DEVICE_FEATURE_SELECT: usize = 0x00;
const COMMON_DEVICE_FEATURE: usize = 0x04;
const COMMON_DRIVER_FEATURE_SELECT: usize = 0x08;
const COMMON_DRIVER_FEATURE: usize = 0x0C;
const COMMON_DEVICE_STATUS: usize = 0x14;
const COMMON_QUEUE_SELECT: usize = 0x16;
const COMMON_QUEUE_SIZE: usize = 0x18;
const COMMON_QUEUE_ENABLE: usize = 0x1C;
const COMMON_QUEUE_NOTIFY_OFF: usize = 0x1E;
const COMMON_QUEUE_DESC: usize = 0x20;
const COMMON_QUEUE_AVAIL: usize = 0x28;
const COMMON_QUEUE_USED: usize = 0x30;

// VirtIO device status bits (VirtIO 1.0 §2.1).
const STATUS_ACKNOWLEDGE: u8 = 1;
const STATUS_DRIVER: u8 = 2;
const STATUS_FEATURES_OK: u8 = 8;
const STATUS_DRIVER_OK: u8 = 4;

// VirtIO net feature bits (VirtIO 1.0 §5.1.3).
const VIRTIO_NET_F_MAC: u64 = 1 << 5;

/// MMIO read/write helpers.
unsafe fn mmio_read8(addr: usize) -> u8 {
    ptr::read_volatile(addr as *const u8)
}
unsafe fn mmio_write8(addr: usize, val: u8) {
    ptr::write_volatile(addr as *mut u8, val);
}
unsafe fn mmio_read16(addr: usize) -> u16 {
    ptr::read_volatile(addr as *const u16)
}
unsafe fn mmio_write16(addr: usize, val: u16) {
    ptr::write_volatile(addr as *mut u16, val);
}
unsafe fn mmio_read32(addr: usize) -> u32 {
    ptr::read_volatile(addr as *const u32)
}
unsafe fn mmio_write32(addr: usize, val: u32) {
    ptr::write_volatile(addr as *mut u32, val);
}
unsafe fn mmio_write64(addr: usize, val: u64) {
    ptr::write_volatile(addr as *mut u64, val);
}

pub struct VirtioNet {
    caps: VirtioPciCaps,
    rx_queue: Virtqueue,
    tx_queue: Virtqueue,
    mac: [u8; 6],
    /// Notify offset for RX queue.
    rx_notify_addr: usize,
    /// Notify offset for TX queue.
    tx_notify_addr: usize,
}

impl VirtioNet {
    /// Initialize the VirtIO network device.
    ///
    /// Follows the VirtIO 1.0 §3.1 initialization sequence:
    /// Reset -> Acknowledge -> Driver -> Feature negotiation ->
    /// Queue setup -> Driver OK.
    pub fn init(caps: VirtioPciCaps, phys_offset: u64) -> Result<Self, &'static str> {
        let common = caps.common_cfg;

        // 1. Reset device
        unsafe { mmio_write8(common + COMMON_DEVICE_STATUS, 0) };

        // 2. Acknowledge
        unsafe { mmio_write8(common + COMMON_DEVICE_STATUS, STATUS_ACKNOWLEDGE) };

        // 3. Driver
        unsafe {
            let status = mmio_read8(common + COMMON_DEVICE_STATUS);
            mmio_write8(common + COMMON_DEVICE_STATUS, status | STATUS_DRIVER);
        }

        // 4. Feature negotiation — we only need VIRTIO_NET_F_MAC
        let device_features = unsafe {
            mmio_write32(common + COMMON_DEVICE_FEATURE_SELECT, 0);
            mmio_read32(common + COMMON_DEVICE_FEATURE) as u64
        };
        let mut driver_features = 0u64;
        if device_features & VIRTIO_NET_F_MAC != 0 {
            driver_features |= VIRTIO_NET_F_MAC;
        }
        unsafe {
            mmio_write32(common + COMMON_DRIVER_FEATURE_SELECT, 0);
            mmio_write32(common + COMMON_DRIVER_FEATURE, driver_features as u32);
        }

        // 5. Features OK
        unsafe {
            let status = mmio_read8(common + COMMON_DEVICE_STATUS);
            mmio_write8(common + COMMON_DEVICE_STATUS, status | STATUS_FEATURES_OK);
            fence(Ordering::SeqCst);
            let status = mmio_read8(common + COMMON_DEVICE_STATUS);
            if status & STATUS_FEATURES_OK == 0 {
                return Err("VirtIO: device did not accept features");
            }
        }

        // 6. Read MAC address from device config
        let mut mac = [0u8; 6];
        for i in 0..6 {
            mac[i] = unsafe { mmio_read8(caps.device_cfg + i) };
        }

        // 7. Set up RX queue (index 0)
        let rx_queue = Virtqueue::new(phys_offset);
        unsafe {
            mmio_write16(common + COMMON_QUEUE_SELECT, 0);
            let max_size = mmio_read16(common + COMMON_QUEUE_SIZE);
            if max_size == 0 {
                return Err("VirtIO: RX queue size is 0");
            }
            let size = core::cmp::min(max_size, QUEUE_SIZE);
            mmio_write16(common + COMMON_QUEUE_SIZE, size);
            mmio_write64(common + COMMON_QUEUE_DESC, rx_queue.desc_phys);
            mmio_write64(common + COMMON_QUEUE_AVAIL, rx_queue.avail_phys);
            mmio_write64(common + COMMON_QUEUE_USED, rx_queue.used_phys);
            mmio_write16(common + COMMON_QUEUE_ENABLE, 1);
        }
        let rx_notify_off = unsafe {
            mmio_read16(common + COMMON_QUEUE_NOTIFY_OFF)
        };
        let rx_notify_addr =
            caps.notify_base + rx_notify_off as usize * caps.notify_off_multiplier as usize;

        // 8. Set up TX queue (index 1)
        let tx_queue = Virtqueue::new(phys_offset);
        unsafe {
            mmio_write16(common + COMMON_QUEUE_SELECT, 1);
            let max_size = mmio_read16(common + COMMON_QUEUE_SIZE);
            if max_size == 0 {
                return Err("VirtIO: TX queue size is 0");
            }
            let size = core::cmp::min(max_size, QUEUE_SIZE);
            mmio_write16(common + COMMON_QUEUE_SIZE, size);
            mmio_write64(common + COMMON_QUEUE_DESC, tx_queue.desc_phys);
            mmio_write64(common + COMMON_QUEUE_AVAIL, tx_queue.avail_phys);
            mmio_write64(common + COMMON_QUEUE_USED, tx_queue.used_phys);
            mmio_write16(common + COMMON_QUEUE_ENABLE, 1);
        }
        let tx_notify_off = unsafe {
            mmio_read16(common + COMMON_QUEUE_NOTIFY_OFF)
        };
        let tx_notify_addr =
            caps.notify_base + tx_notify_off as usize * caps.notify_off_multiplier as usize;

        // 9. Pre-post all RX buffers
        let mut driver = VirtioNet {
            caps,
            rx_queue,
            tx_queue,
            mac,
            rx_notify_addr,
            tx_notify_addr,
        };
        for _ in 0..QUEUE_SIZE {
            driver.rx_queue.post_receive();
        }
        // Notify device that RX buffers are available
        unsafe { mmio_write16(driver.rx_notify_addr, 0) };

        // 10. Set DRIVER_OK
        unsafe {
            let status = mmio_read8(common + COMMON_DEVICE_STATUS);
            mmio_write8(common + COMMON_DEVICE_STATUS, status | STATUS_DRIVER_OK);
        }

        Ok(driver)
    }

    /// Device MAC address.
    pub fn mac(&self) -> [u8; 6] {
        self.mac
    }

    /// Format MAC as "xx:xx:xx:xx:xx:xx" into a buffer.
    pub fn mac_str(&self, buf: &mut [u8; 17]) {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        for i in 0..6 {
            buf[i * 3] = HEX[(self.mac[i] >> 4) as usize];
            buf[i * 3 + 1] = HEX[(self.mac[i] & 0xf) as usize];
            if i < 5 {
                buf[i * 3 + 2] = b':';
            }
        }
    }

    /// Reclaim completed TX descriptors.
    fn reclaim_tx(&mut self) {
        while let Some((desc_idx, _len)) = self.tx_queue.poll_used() {
            self.tx_queue.free_desc(desc_idx);
        }
    }
}

impl NetworkInterface for VirtioNet {
    fn name(&self) -> &str {
        "virtio0"
    }

    fn mtu(&self) -> usize {
        // Ethernet payload MTU. The Ethernet header is handled internally.
        1500
    }

    fn send(&mut self, data: &[u8]) -> Result<(), PlatformError> {
        // Reclaim any completed TX buffers first
        self.reclaim_tx();

        // Build Ethernet frame: dst(6) + src(6) + type(2) + payload
        let frame_len = ETH_HEADER_LEN + data.len();
        if frame_len > super::virtqueue::BUF_SIZE {
            return Err(PlatformError::SendFailed);
        }

        // Construct frame in a stack buffer
        let mut frame = [0u8; super::virtqueue::BUF_SIZE];
        frame[0..6].copy_from_slice(&BROADCAST_MAC);
        frame[6..12].copy_from_slice(&self.mac);
        frame[12..14].copy_from_slice(&ETHERTYPE_HARMONY);
        frame[14..14 + data.len()].copy_from_slice(data);

        match self.tx_queue.submit_send(&frame[..frame_len]) {
            Some(_idx) => {
                // Notify device
                unsafe { mmio_write16(self.tx_notify_addr, 1) };
                Ok(())
            }
            None => Err(PlatformError::SendFailed),
        }
    }

    fn receive(&mut self) -> Option<Vec<u8>> {
        let (desc_idx, len) = self.rx_queue.poll_used()?;
        let frame = self.rx_queue.read_buffer(desc_idx, len);

        // Free and re-post the descriptor for future receives
        self.rx_queue.free_desc(desc_idx);
        self.rx_queue.post_receive();
        // Notify device of re-posted buffer
        unsafe { mmio_write16(self.rx_notify_addr, 0) };

        // Strip Ethernet header: need at least 14 bytes
        if frame.len() < ETH_HEADER_LEN {
            return None; // Runt frame
        }

        // Only accept our EtherType (ignore ARP, IP, etc.)
        if frame[12..14] != ETHERTYPE_HARMONY {
            return None;
        }

        Some(frame[ETH_HEADER_LEN..].to_vec())
    }
}
```

**Step 3: Verify bare-metal compilation**

Run: `cd harmony-os/crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: compiles clean.

**Step 4: Commit**

```bash
cd harmony-os
git add crates/harmony-boot/Cargo.toml crates/harmony-boot/src/virtio/net.rs
git commit -m "feat(boot): VirtioNet driver implementing NetworkInterface"
```

---

### Task 5: Event Loop Integration

**Context:** Wire the VirtioNet driver into `kernel_main()` and extend `UnikernelRuntime` to accept network interfaces, feed inbound packets, and dispatch outbound actions.

**Files:**
- Modify: `harmony-os/crates/harmony-boot/src/main.rs`
- Modify: `harmony-os/crates/harmony-unikernel/src/event_loop.rs`
- Modify: `harmony-os/crates/harmony-unikernel/src/lib.rs` (re-export)
- Modify: `harmony-os/justfile` (add VirtIO-net to QEMU flags)

**Step 1: Extend UnikernelRuntime with network interface support**

Add methods to `UnikernelRuntime` in `event_loop.rs`:

```rust
// Add these imports at the top of event_loop.rs:
use alloc::string::String;
use harmony_platform::NetworkInterface;
use harmony_reticulum::interface::InterfaceMode;

// Add to UnikernelRuntime impl block:

    /// Register a network interface with the node's routing table.
    pub fn register_interface(&mut self, name: &str) {
        self.node.register_interface(
            String::from(name),
            InterfaceMode::Full,
            None,
        );
    }

    /// Feed an inbound packet from a network interface into the node.
    pub fn handle_packet(&mut self, interface_name: &str, data: Vec<u8>, now: u64) -> Vec<NodeAction> {
        self.node.handle_event(NodeEvent::InboundPacket {
            interface_name: String::from(interface_name),
            raw: data,
            now,
        })
    }
```

**Step 2: Update main.rs event loop**

After the identity generation section (step 5) and before the event loop (step 6), add VirtIO-net initialization:

```rust
    // 5.5 VirtIO-net init
    let mut virtio_net = match pci::find_virtio_net() {
        Some(pci_dev) => {
            pci_dev.enable_bus_master();
            match virtio::pci_cap::parse_capabilities(&pci_dev, phys_offset) {
                Some(caps) => match virtio::net::VirtioNet::init(caps, phys_offset) {
                    Ok(net) => {
                        let mut mac_buf = [0u8; 17];
                        net.mac_str(&mut mac_buf);
                        let mac_str = core::str::from_utf8(&mac_buf)
                            .unwrap_or("??:??:??:??:??:??");
                        serial.log("VIRTIO", mac_str);
                        Some(net)
                    }
                    Err(e) => {
                        serial.log("VIRTIO", e);
                        None
                    }
                },
                None => {
                    serial.log("VIRTIO", "no capabilities found");
                    None
                }
            }
        }
        None => {
            serial.log("VIRTIO", "no device found");
            None
        }
    };

    // Register interface with runtime if available
    if virtio_net.is_some() {
        runtime.register_interface("virtio0");
    }
```

Update the event loop to poll and dispatch:

```rust
    let mut tick: u64 = 0;
    loop {
        // Poll network for inbound packets
        if let Some(ref mut net) = virtio_net {
            while let Some(data) = net.receive() {
                let _actions = runtime.handle_packet("virtio0", data, tick);
                // TODO(harmony-44x): dispatch actions from inbound handling
            }
        }

        let actions = runtime.tick(tick);

        // Dispatch outbound packets
        if let Some(ref mut net) = virtio_net {
            for action in &actions {
                if let NodeAction::SendOnInterface { interface_name, raw } = action {
                    if interface_name.as_ref() == "virtio0" {
                        let _ = net.send(raw);
                    }
                }
            }
        }

        tick += 1;
        x86_64::instructions::hlt();
    }
```

**Step 3: Add NodeAction import to main.rs**

Add `use harmony_reticulum::NodeAction;` to the imports. Since `harmony-boot` doesn't directly depend on `harmony-reticulum`, re-export it from `harmony-unikernel`:

In `harmony-os/crates/harmony-unikernel/src/lib.rs`, add:
```rust
pub use harmony_reticulum::NodeAction;
```

Then in `main.rs`, use:
```rust
use harmony_unikernel::NodeAction;
```

**Step 4: Update justfile QEMU flags**

Update the `run` recipe to include VirtIO-net with multicast backend:

```just
# Run in QEMU interactively with VirtIO-net (multicast LAN)
run: build
    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio \
        -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234

# Run a second peer on the same virtual LAN
run-peer: build
    qemu-system-x86_64 \
        -drive format=raw,file=target/harmony-boot-bios.img \
        -serial stdio \
        -display none \
        -device isa-debug-exit,iobase=0xf4,iosize=0x04 \
        -cpu qemu64,+rdrand \
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234
```

Also update `test-qemu` to include VirtIO-net flags so the QEMU test verifies the full init path:

Add to the `test-qemu` QEMU command:
```
        -device virtio-net-pci,netdev=n0 \
        -netdev socket,id=n0,mcast=230.0.0.1:1234 \
```

**Step 5: Verify bare-metal compilation**

Run: `cd harmony-os/crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: compiles clean.

**Step 6: Commit**

```bash
cd harmony-os
git add crates/harmony-unikernel/src/event_loop.rs crates/harmony-unikernel/src/lib.rs \
        crates/harmony-boot/src/main.rs justfile
git commit -m "feat(boot): wire VirtioNet into event loop with packet dispatch"
```

---

### Task 6: QEMU Boot Test

**Context:** Verify the full VirtIO initialization works on real (emulated) hardware. The QEMU test should boot, scan PCI, find the VirtIO device, initialize it, log the MAC address, and exit cleanly.

**Files:**
- Modify: `harmony-os/justfile` (update test-qemu to check `[VIRTIO]` line)

**Step 1: Update test-qemu to verify VirtIO init**

In the `test-qemu` recipe, add a check for the `[VIRTIO]` line alongside the existing `[IDENTITY]` check:

```bash
    if ! grep -q '\[VIRTIO\]' "$LOG"; then
        echo "QEMU boot test: FAILED (no [VIRTIO] line — VirtIO init failed)"
        exit 1
    fi
```

Add this check after the `[IDENTITY]` check.

**Step 2: Build and run the QEMU test**

Run: `cd harmony-os && just test-qemu`
Expected: QEMU boots, serial output includes:
```
[BOOT] Harmony unikernel v0.1.0
[HEAP] ...
[ENTROPY] RDRAND available
[IDENTITY] <hex>
[VIRTIO] <mac:address>
[READY] entering event loop
QEMU boot test: PASSED
```

**Step 3: Verify host tests still pass**

Run: `cd harmony-os && cargo test --workspace`
Expected: all tests pass.

Run: `cd harmony-os && cargo clippy --workspace && cd crates/harmony-boot && cargo clippy --target x86_64-unknown-none`
Expected: zero warnings.

**Step 4: Commit**

```bash
cd harmony-os
git add justfile
git commit -m "test: verify VirtIO-net initialization in QEMU boot test"
```

---

### Task 7: Quality Gate and Delivery Prep

**Context:** Run full quality gates across both repos, ensure everything is clean for delivery.

**Step 1: Run harmony workspace tests**

Run: `cd /Users/zeblith/work/zeblithic/harmony && cargo test --workspace`
Expected: all 365+ tests pass.

Run: `cd /Users/zeblith/work/zeblithic/harmony && cargo clippy --workspace`
Expected: zero warnings.

**Step 2: Run harmony-os quality gate**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && just check`
Expected: all checks pass.

**Step 3: Run QEMU boot test**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && just test-qemu`
Expected: PASSED.

**Step 4: Verify formatting**

Run: `cd /Users/zeblith/work/zeblithic/harmony-os && cargo fmt --all -- --check`
Expected: no formatting issues.

**Step 5: Delivery**

Use `/delivertask` to push both repos, create PRs, and trigger reviews. Remember:
- harmony PR: docs only (design + plan)
- harmony-os PR: all driver code + justfile changes
- harmony PR merges first (as usual for docs-then-code ordering)
